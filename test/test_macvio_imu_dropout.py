"""Ride a 1.06 s IMU dropout through the node's frame path into a real MACVIO IMU chain."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pypose as pp
import pytest
import rclpy
from sensor_msgs.msg import Image, Imu
import torch

from mac_ros2.macvio_node import (
    MACVIO,
    UNKNOWN_TWIST_VARIANCE,
    MacvioNode,
    load_macvio_config,
)
from Src.Module import IInitializer, IPreintegrator
from Src.Scripts.AutoTest.imu_dropout_fixtures import push_motion
from Src.Utility.Enum import BodyNodeStatus, VIOMode


GRAVITY = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)
ACCELERATION = torch.tensor([0.4, -0.1, 0.05], dtype=torch.float64)
VELOCITY_0 = torch.tensor([2.0, 0.5, 0.0], dtype=torch.float64)
START_NS = 1_000_000_000
IMU_DT_NS = 10_000_000
CAMERA_DT_NS = 100_000_000
GAP_START_NS = START_NS + 2_005_000_000
GAP_NS = 1_060_000_000
FRAMES = 50
INIT_BODIES = 5


def position(time_ns: int) -> torch.Tensor:
    t = time_ns * 1e-9
    return VELOCITY_0 * t + 0.5 * ACCELERATION * t * t


def velocity(time_ns: int) -> torch.Tensor:
    return VELOCITY_0 + ACCELERATION * time_ns * 1e-9


def body_pose(time_ns: int) -> torch.Tensor:
    """World <- body; the body never rotates."""
    return torch.cat([position(time_ns), torch.tensor([0.0, 0.0, 0.0, 1.0])]).double()


class GroundTruthTracker:
    """Stand-in for the network tracker: one motion edge per new frame."""

    @staticmethod
    def init_local(_config):
        return None

    def get_data(self, state, _local):
        graph = state[0]
        return graph, len(graph.body) >= 2

    def run(self, graph):
        return graph

    def set_data(self, state, _local, graph):
        push_motion(graph, len(graph.body) - 2, len(graph.body) - 1)


class GroundTruthOptimizer:
    """Stand-in for the visual solve: each edge's destination body gets its true pose."""

    @staticmethod
    def init_local(_config):
        return SimpleNamespace(next_optimize_id=0)

    def get_data(self, state, local):
        graph = state[0]
        if local.next_optimize_id >= len(graph.motion):
            return None, False
        local.next_optimize_id += 1
        return (graph, local.next_optimize_id - 1), True

    def run(self, problem):
        return problem

    def set_data(self, _state, _local, problem):
        graph, motion = problem
        body = graph.motion2body1.neighbors(torch.tensor([motion]))
        graph.body.data["pose"][body] = body_pose(int(graph.body.data["time_ns"][body].item()))
        graph.body.data["status"][body] |= BodyNodeStatus.Optimized


class GroundTruthInitializer:
    """Initialize from the true state once INIT_BODIES chained bodies exist."""

    def __init__(self, imu_t_bs: pp.LieTensor, camera_t_bs: pp.LieTensor) -> None:
        self.imu_t_bs = pp.SE3(imu_t_bs.tensor()[0])
        self.camera_t_bs = pp.SE3(camera_t_bs.tensor()[0])

    def initialize(self, graph, _frame, *, from_body=0):
        count = len(graph.body) - from_body
        if count < INIT_BODIES:
            return None
        times = graph.body.data["time_ns"][from_body:].flatten().tolist()
        T_ic = self.imu_t_bs.Inv() @ self.camera_t_bs
        return IInitializer.Output(
            T_bi=self.imu_t_bs,
            T_ic=T_ic,
            T_bc=self.imu_t_bs @ T_ic,
            T_wb=pp.SE3(torch.stack([body_pose(t) for t in times])),
            T_wb_cov=torch.zeros(count, 6, 6, dtype=torch.float64),
            b_vel_Tws=torch.stack([velocity(t) for t in times]),
            b_vel_Tws_cov=torch.eye(3, dtype=torch.float64).expand(count, 3, 3) * 1e-4,
            gravity_Tw=GRAVITY.clone(),
            gyro_bias=torch.zeros(1, 3, dtype=torch.float64),
            acc_bias=torch.zeros(1, 3, dtype=torch.float64),
        )


def imu_message(time_ns: int) -> Imu:
    message = Imu()
    message.header.stamp.sec = time_ns // 1_000_000_000
    message.header.stamp.nanosec = time_ns % 1_000_000_000
    message.header.frame_id = "novatel/imu_frame"
    (
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    ) = (ACCELERATION - GRAVITY).tolist()
    return message


@pytest.fixture
def dropout_node(monkeypatch):
    config = load_macvio_config("thermal")[0]
    ros = config.ROS

    def factory(cfg):
        odometry = cfg.Odometry
        return MACVIO(
            imu_preintegrate=IPreintegrator.instantiate(
                odometry.imu_preintegrate.type, odometry.imu_preintegrate.args
            ),
            imu_initializer=GroundTruthInitializer(
                MacvioNode._se3_from_config(ros.extrinsics.imu_t_bs),
                MacvioNode._se3_from_config(ros.extrinsics.camera_t_bs),
            ),
            feature_tracker=(GroundTruthTracker(), None),
            track_optimizer=(GroundTruthOptimizer(), None),
            imu_health=odometry.imu_health,
        )

    monkeypatch.setattr(MACVIO, "from_config", Mock(side_effect=factory))
    rclpy.init(args=["--ros-args", "-p", "modality:=thermal"])
    node = MacvioNode()
    node.status_publisher = Mock()
    node.odom_publisher = Mock()
    node.publish_tf = False
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_node_rides_through_imu_dropout(dropout_node):
    node = dropout_node
    height, width = node.camera_mask.shape[-2:]
    image = Image(height=height, width=width, encoding="mono8", step=width,
                  data=np.zeros((height, width), dtype=np.uint8).tobytes())
    imu_times = iter(range(START_NS - 200_000_000, START_NS + FRAMES * CAMERA_DT_NS, IMU_DT_NS))
    next_imu = next(imu_times)
    states, modes, published = [], [], []

    for index in range(FRAMES):
        frame_ns = START_NS + index * CAMERA_DT_NS
        while next_imu <= frame_ns:
            if not GAP_START_NS <= next_imu < GAP_START_NS + GAP_NS:
                node._receive_imu(imu_message(next_imu))
            next_imu = next(imu_times)
        image.header.stamp.sec = frame_ns // 1_000_000_000
        image.header.stamp.nanosec = frame_ns % 1_000_000_000
        node.odom_publisher.publish.reset_mock()

        node._process_stereo(image, image)

        status = json.loads(node.status_publisher.publish.call_args.args[0].data)
        states.append(status["state"])
        modes.append(node.system.vio_mode)
        if node.odom_publisher.publish.called:
            published.append((status["state"], node.odom_publisher.publish.call_args.args[0]))

    def collapse(values):
        return [value for k, value in enumerate(values) if k == 0 or values[k - 1] != value]

    assert collapse(states) == ["INITIALIZING", "TRACKING", "DEGRADED", "TRACKING"]
    assert collapse(modes) == [
        VIOMode.PRE_INIT, VIOMode.VIO, VIOMode.VO_FALLBACK, VIOMode.RECOVERING, VIOMode.VIO
    ]
    assert 10 <= states.count("DEGRADED") <= 16
    assert status["reset_count"] == 0 and status["last_error"] == ""
    assert status["n_chain_breaks"] == 1 and status["n_dropped_samples"] == 0
    assert status["imu_gap_count"] == 1
    assert status["largest_imu_gap_sec"] == pytest.approx((GAP_NS + IMU_DT_NS) / 1e9, abs=0.011)
    assert status["frames_processed"] == FRAMES
    assert not node._stop_event.is_set()

    assert len(published) == states.count("TRACKING") + states.count("DEGRADED")
    for state, message in published:
        twist = np.asarray(message.twist.covariance).reshape(6, 6)
        assert np.isfinite(twist).all()
        if state == "TRACKING":
            assert twist[0, 0] < UNKNOWN_TWIST_VARIANCE
        assert message.pose.pose.orientation.w == pytest.approx(1.0)
    degraded_twists = [
        np.asarray(message.twist.covariance).reshape(6, 6)
        for state, message in published if state == "DEGRADED"
    ]
    assert any(twist[0, 0] == UNKNOWN_TWIST_VARIANCE for twist in degraded_twists)
    final = published[-1][1]
    expected_velocity = velocity(START_NS + (FRAMES - 1) * CAMERA_DT_NS)
    np.testing.assert_allclose(
        [final.twist.twist.linear.x, final.twist.twist.linear.y, final.twist.twist.linear.z],
        expected_velocity.numpy(), atol=1e-6,
    )
