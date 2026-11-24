"""Core modules around a real MACVIO IMU chain, with ground-truth tracker, optimizer and initializer.

The loop mirrors what the runner does: IMU samples feed the buffer and the state machine, each
received pair builds a window and runs MACVIO, and the status precedes the odometry into the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pypose as pp
import torch

from mac_ros2._macslam import MACSLAM_PATH
from mac_ros2.imu_buffer import IMUBuffer
from mac_ros2.localization_gate import LocalizationGate
from mac_ros2.odometry_state import OdometryState
from mac_ros2.output import frame_result, odometry_from_map
from mac_ros2.vio_state import VioState, VioStateMachine
from Src.DataLoader import SIFrameData, StereoData
from Src.Module import IInitializer, IPreintegrator
from Src.Odometry.MACVIO import MACVIO
from Src.Scripts.AutoTest.imu_dropout_fixtures import push_motion
from Src.Utility.Config import load_config
from Src.Utility.Enum import BodyNodeStatus, VIOMode

MS = 1_000_000
GRAVITY = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)
ACCELERATION = torch.tensor([0.4, -0.1, 0.05], dtype=torch.float64)
VELOCITY_0 = torch.tensor([2.0, 0.5, 0.0], dtype=torch.float64)
START_NS = 1_000_000_000
IMU_DT_NS = 10 * MS
CAMERA_DT_NS = 100 * MS
FRAMES = 50
INIT_BODIES = 5
IMU_GAP = (START_NS + 2_005 * MS, START_NS + 3_065 * MS)
CAMERA_GAP = (START_NS + 3_000 * MS, START_NS + 3_500 * MS)
T_BODY_BASE = pp.SE3(torch.tensor([0.508, 0.05, 0.152, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
T_BODY_CAMERA = pp.SE3(torch.tensor([[0.4, 0.2, -0.1, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float64))
EXPERIMENT = MACSLAM_PATH / "Config/Experiment/MACVIO/Offroad/VIO_MACI2_FR_Yamaha_Thermal.yaml"


def position(time_ns: int) -> torch.Tensor:
    t = time_ns * 1e-9
    return VELOCITY_0 * t + 0.5 * ACCELERATION * t * t


def velocity(time_ns: int) -> torch.Tensor:
    return VELOCITY_0 + ACCELERATION * time_ns * 1e-9


def body_pose(time_ns: int) -> torch.Tensor:
    """World <- body; the body never rotates."""
    return torch.cat([position(time_ns), torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)])


class GroundTruthTracker:
    """One motion edge from the previous body to each new one."""

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
    """Each edge's destination body gets its true pose and is marked Optimized."""

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
    """Initialize from the true state once INIT_BODIES bodies exist."""

    def initialize(self, graph, _frame, *, from_body=0):
        count = len(graph.body) - from_body
        if count < INIT_BODIES:
            return None
        times = graph.body.data["time_ns"][from_body:].flatten().tolist()
        T_bi = pp.identity_SE3(dtype=torch.float64)
        T_bc = pp.SE3(T_BODY_CAMERA.tensor()[0])
        return IInitializer.Output(
            T_bi=T_bi,
            T_ic=T_bc,
            T_bc=T_bc,
            T_wb=pp.SE3(torch.stack([body_pose(t) for t in times])),
            T_wb_cov=torch.zeros(count, 6, 6, dtype=torch.float64),
            b_vel_Tws=torch.stack([velocity(t) for t in times]),
            b_vel_Tws_cov=torch.eye(3, dtype=torch.float64).expand(count, 3, 3) * 1e-4,
            gravity_Tw=GRAVITY.clone(),
            gyro_bias=torch.zeros(1, 3, dtype=torch.float64),
            acc_bias=torch.zeros(1, 3, dtype=torch.float64),
            body_start=from_body,
        )


@dataclass(frozen=True)
class NovatelSpec:
    acc_noise_density: float = 8.0e-2
    gyro_noise_density: float = 4.0e-3
    acc_random_walk: float = 3.0e-3
    gyro_random_walk: float = 1.9e-5
    gravity: torch.Tensor = field(default_factory=lambda: GRAVITY.view(1, 3))
    max_bracket_ns: int = 50 * MS


def make_system() -> MACVIO:
    odometry = load_config(EXPERIMENT)[0].Odometry
    return MACVIO(
        imu_preintegrate=IPreintegrator.instantiate(odometry.imu_preintegrate.type, odometry.imu_preintegrate.args),
        imu_initializer=GroundTruthInitializer(),
        feature_tracker=(GroundTruthTracker(), None),
        track_optimizer=(GroundTruthOptimizer(), None),
        imu_health=odometry.imu_health,
    )


def make_frame(index: int, time_ns: int, imu) -> SIFrameData:
    image = torch.zeros(1, 3, 8, 8, dtype=torch.float32)
    return SIFrameData(
        idx=torch.tensor([index], dtype=torch.long),
        time_ns=[time_ns],
        stereo=StereoData(
            data_ids=None,
            T_BS=T_BODY_CAMERA,
            K=torch.eye(3, dtype=torch.float32).unsqueeze(0),
            baseline=torch.tensor([0.3], dtype=torch.float32),
            time_ns=[time_ns],
            height=8,
            width=8,
            imageL=image,
            imageR=image.clone(),
        ),
        imu=imu,
    )


def reference_odometry(time_ns: int) -> OdometryState:
    pose = (pp.SE3(body_pose(time_ns)) @ T_BODY_BASE).tensor().numpy()
    return OdometryState(stamp_ns=time_ns, pose=pose, pose_cov=np.eye(6), twist=np.zeros(6), twist_cov=np.eye(6))


@dataclass
class Trace:
    events: list[tuple[int, VioState]] = field(default_factory=list)     # (stamp, state) after every event
    frame_states: list[VioState] = field(default_factory=list)
    modes: list[VIOMode] = field(default_factory=list)
    published: list[tuple[VioState, OdometryState]] = field(default_factory=list)
    passed: list[OdometryState] = field(default_factory=list)
    hold_at: dict[int, bool] = field(default_factory=dict)              # gate hold at IMU stamps


def run(imu_gap: tuple[int, int] | None = None, camera_gap: tuple[int, int] | None = None):
    system = make_system()
    buffer = IMUBuffer(NovatelSpec(), capacity=4096, wait_timeout_s=0.0)
    machine = VioStateMachine(frame_rate_hz=1e9 / CAMERA_DT_NS, imu_silence_ns=50 * MS)
    gate = LocalizationGate()
    trace = Trace()
    a_meas = tuple((ACCELERATION - GRAVITY).tolist())
    next_imu, t_prev, processed = START_NS - 200 * MS, None, 0

    def publish_status(now: int) -> None:
        gate.on_status(machine.status(now), now)

    for index in range(FRAMES):
        frame_ns = START_NS + index * CAMERA_DT_NS
        while next_imu <= frame_ns + IMU_DT_NS:
            if imu_gap is None or not imu_gap[0] <= next_imu < imu_gap[1]:
                buffer.add(next_imu, a_meas, (0.0, 0.0, 0.0))
                previous = machine.state
                trace.events.append((next_imu, machine.on_imu(next_imu, next_imu)))
                if machine.state is not previous:
                    publish_status(next_imu)
                trace.hold_at[next_imu] = gate.hold(next_imu)
            next_imu += IMU_DT_NS
        now = frame_ns + IMU_DT_NS
        if camera_gap is not None and camera_gap[0] <= frame_ns < camera_gap[1]:
            continue
        trace.events.append((frame_ns, machine.on_pair(frame_ns, now)))
        imu = buffer.build_window(t_prev, frame_ns)
        assert imu is not None
        system.run(make_frame(processed, frame_ns, imu))
        t_prev, processed = frame_ns, processed + 1

        result = frame_result(system)
        state = machine.on_frame(result, now)
        trace.events.append((frame_ns, state))
        trace.frame_states.append(state)
        trace.modes.append(system.vio_mode)
        gate.on_reference(reference_odometry(frame_ns))
        publish_status(now)
        if result.initialized:
            odometry = odometry_from_map(system.Map, T_BODY_BASE, fd_twist_variance=1.0)
            trace.published.append((state, odometry))
            passed = gate.on_odometry(odometry, now)
            if passed is not None:
                trace.passed.append(passed)
    return trace, machine.status(START_NS + FRAMES * CAMERA_DT_NS), system


def collapse(values: list) -> list:
    return [v for k, v in enumerate(values) if k == 0 or values[k - 1] != v]


def test_rides_through_imu_dropout():
    trace, status, _ = run(imu_gap=IMU_GAP)

    assert collapse(trace.frame_states) == [
        VioState.INITIALIZING, VioState.TRACKING, VioState.DEGRADED, VioState.TRACKING
    ]
    assert collapse(trace.modes) == [VIOMode.PRE_INIT, VIOMode.VIO, VIOMode.VO_FALLBACK, VIOMode.RECOVERING, VIOMode.VIO]
    assert 10 <= trace.frame_states.count(VioState.DEGRADED) <= 16
    vio = status["vio_status"]
    assert vio["n_chain_breaks"] == 1 and vio["n_dropped_samples"] == 0
    assert status["last_error"] == "" and status["frames_processed"] == FRAMES

    assert len(trace.published) == FRAMES - trace.frame_states.count(VioState.INITIALIZING)
    assert len(trace.passed) == len(trace.published)
    for state, odometry in trace.published:
        assert np.isfinite(odometry.twist).all() and np.isfinite(odometry.twist_cov).all()
        if state is VioState.TRACKING:
            assert not odometry.twist_from_poses
    assert any(odometry.twist_from_poses for _, odometry in trace.published)

    final = trace.published[-1][1]
    final_ns = START_NS + (FRAMES - 1) * CAMERA_DT_NS
    np.testing.assert_allclose(final.twist[:3], velocity(final_ns).numpy(), atol=1e-6)
    np.testing.assert_allclose(trace.passed[-1].pose, reference_odometry(final_ns).pose, atol=1e-6)


def test_camera_dropout_holds_until_first_optimized_frame():
    trace, status, _ = run(camera_gap=CAMERA_GAP)
    initialized_at = next(k for k, (_, s) in enumerate(trace.events) if s is VioState.TRACKING)
    after_init = [state for _, state in trace.events[initialized_at:]]

    assert collapse(after_init) == [
        VioState.TRACKING, VioState.CAMERA_DROPOUT, VioState.SEGMENT_PENDING, VioState.TRACKING
    ]
    last_pair = CAMERA_GAP[0] - CAMERA_DT_NS
    detected = next(t for t, s in trace.events if s is VioState.CAMERA_DROPOUT)
    assert detected == last_pair + 300 * MS + IMU_DT_NS
    assert all(trace.hold_at[t] for t in trace.hold_at if detected <= t <= CAMERA_GAP[1])
    assert not trace.hold_at[max(trace.hold_at)]

    assert status["n_camera_dropouts"] == 1 and status["frames_received"] == FRAMES - 5
    assert len(trace.passed) == len(trace.published)
    assert [o.stamp_ns for o in trace.passed if CAMERA_GAP[0] <= o.stamp_ns < CAMERA_GAP[1]] == []
    np.testing.assert_allclose(trace.passed[-1].pose[:3], reference_odometry(trace.passed[-1].stamp_ns).pose[:3],
                               atol=1e-6)


def test_long_camera_gap_recovers():
    trace, status, system = run(camera_gap=(START_NS + 3_000 * MS, START_NS + 4_500 * MS))
    assert status["last_error"] == "" and not system.lost_track
    assert trace.frame_states[-1] is VioState.TRACKING
