"""Check ROS camera selection against the upstream off-road VIO profiles."""

import json
import time
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu
import torch
import yaml

from mac_ros2.macvio_node import (
    ImuSample,
    MACSLAM_PATH,
    MACVIO,
    MacvioNode,
    TwoframeTracker_v2,
    build_macvio,
    create_frame_transform,
    load_macvio_config,
    validate_imu_window,
)
from mac_ros2.localization_adapter_node import (
    initial_alignment,
    pose_matrix,
    transform_odometry,
)
from Src.DataLoader import StereoData, StereoFrameData
from Src.Module.IMU.IMUChain import VIOStatus
from Src.DataLoader.Dataset.Offroad import (
    RGB_T_BS_BODY_CAM,
    THERMAL_T_BS_BODY_CAM,
)
from Src.Utility.Config import load_config
from Src.Utility.Enum import IMUWindowStatus, VIOMode


def idle_vio_status():
    return VIOStatus(
        mode=VIOMode.PRE_INIT,
        frames_in_mode=0,
        last_gap_s=0.0,
        imu_age_s=float("inf"),
        window_status=IMUWindowStatus.Gap,
        n_chain_breaks=0,
        n_dropped_samples=0,
    )


def mock_system():
    return Mock(
        imu_initcond=None,
        lost_track=False,
        vio_mode=VIOMode.PRE_INIT,
        vio_status=idle_vio_status(),
    )


@pytest.mark.parametrize("modality,label", [("thermal", "Thermal"), ("rgb", "RGB")])
def test_profiles_load_upstream_estimator(modality, label):
    config, _, experiment_path = load_macvio_config(modality)
    expected_path = (
        MACSLAM_PATH / f"Config/Experiment/MACVIO/Offroad/VIO_MACI2_FR_Yamaha_{label}.yaml"
    )
    upstream, _ = load_config(expected_path)
    assert experiment_path == expected_path
    assert config.Common == upstream.Common
    assert config.Odometry == upstream.Odometry
    MACVIO.is_valid_config(config.Odometry)
    sequence = getattr(upstream.Preprocess, f"Offroad_{label}_VIO")
    geometric = [
        step for step in sequence if step.type in {"CenterCropFrame", "SmartResizeFrame"}
    ]
    if geometric:
        assert config.ROS.preprocess == geometric[0]
    else:
        mask = MacvioNode._load_camera_mask(config.Camera.mask)
        assert config.ROS.preprocess.type == "CenterCropFrame"
        args = config.ROS.preprocess.args
        assert (args.height, args.width) == tuple(mask.shape[-2:])


def test_explicit_config_overrides_modality():
    _, rgb_path, _ = load_macvio_config("rgb")
    config, actual_path, _ = load_macvio_config("thermal", str(rgb_path))
    assert actual_path == rgb_path
    assert config.ROS.modality == "rgb"


def test_invalid_modality_fails():
    with pytest.raises(ValueError, match="use thermal or rgb"):
        load_macvio_config("unknown")


def test_legacy_custom_config(tmp_path):
    _, profile_path, experiment_path = load_macvio_config("thermal")
    _, profile = load_config(profile_path)
    _, experiment = load_config(experiment_path)
    profile.pop("macslam_config")
    profile.update({key: experiment[key] for key in ("Common", "Odometry")})
    profile["ROS"].pop("modality")
    profile["ROS"]["preprocess"] = {"crop_height": 400, "crop_width": 640}
    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(yaml.safe_dump(profile))
    config, _, source = load_macvio_config("thermal", str(custom_path))
    assert source is None
    assert config.ROS.modality == "thermal"
    assert create_frame_transform(config.ROS.preprocess).config.height == 400
    MACVIO.is_valid_config(config.Odometry)


@pytest.mark.parametrize("modality,height,width", [("thermal", 512, 640), ("rgb", 272, 512)])
def test_preprocessing_preserves_camera_geometry(modality, height, width):
    config, _, _ = load_macvio_config(modality)
    mask = MacvioNode._load_camera_mask(config.Camera.mask)
    input_height, input_width = mask.shape[-2:]
    camera = config.Camera
    intrinsic = torch.tensor([[
        [camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]
    ]], dtype=torch.float32)
    stereo = StereoData(
        data_ids=None,
        T_BS=MacvioNode._se3_from_config(config.ROS.extrinsics.camera_t_bs),
        K=intrinsic.clone(),
        baseline=torch.tensor([camera.bl]),
        time_ns=[1],
        height=input_height,
        width=input_width,
        imageL=torch.zeros((1, 3, input_height, input_width)),
        imageR=torch.zeros((1, 3, input_height, input_width)),
        camera_mask=mask.clone(),
    )
    frame = StereoFrameData(idx=torch.tensor([0]), time_ns=[1], stereo=stereo)
    result = create_frame_transform(config.ROS.preprocess)(frame).stereo
    assert result.imageL.shape == result.imageR.shape == (1, 3, height, width)
    assert result.camera_mask.shape == (1, 1, height, width)
    assert result.camera_mask.dtype == torch.bool
    assert result.camera_mask.any() and not result.camera_mask.all()
    expected = intrinsic.clone()
    if modality == "rgb":
        expected[:, :2] /= 2
    else:
        torch.testing.assert_close(result.camera_mask, mask)
    torch.testing.assert_close(result.K, expected)
    assert float(result.baseline[0]) == pytest.approx(camera.bl)


@pytest.mark.parametrize("modality,upstream_pose", [
    ("rgb", RGB_T_BS_BODY_CAM), ("thermal", THERMAL_T_BS_BODY_CAM)
])
def test_camera_extrinsic_matches_upstream_in_imu_frame(modality, upstream_pose):
    config, _, _ = load_macvio_config(modality)
    extrinsics = config.ROS.extrinsics
    vehicle_from_camera = MacvioNode._se3_from_config(extrinsics.camera_t_bs)
    vehicle_from_imu = MacvioNode._se3_from_config(extrinsics.imu_t_bs)
    imu_from_camera = vehicle_from_imu.Inv() @ vehicle_from_camera
    torch.testing.assert_close(
        imu_from_camera.tensor()[0], torch.tensor(upstream_pose, dtype=torch.float64),
        rtol=1e-6, atol=1e-7,
    )


@pytest.mark.parametrize("encoding,pixels,expected", [
    ("rgb8", [255, 128, 0], [1, 128 / 255, 0]),
    ("bgr8", [0, 128, 255], [1, 128 / 255, 0]),
    ("rgba8", [255, 128, 0, 64], [1, 128 / 255, 0]),
    ("bgra8", [0, 128, 255, 64], [1, 128 / 255, 0]),
    ("mono8", [128], [128 / 255] * 3),
])
def test_image_color_order(encoding, pixels, expected):
    message = Image(height=1, width=1, encoding=encoding, step=len(pixels), data=bytes(pixels))
    actual = MacvioNode._image_to_tensor(message)
    torch.testing.assert_close(actual[:, 0, 0], torch.tensor(expected, dtype=torch.float32))


def make_imu_sample(time_ns):
    return ImuSample(
        time_ns,
        (0, 0, 9.80665),
        (0, 0, 0),
        (0,) * 9,
        (0,) * 9,
        "novatel/imu_frame",
    )


def make_odometry(x, y, z, quaternion=(0.0, 0.0, 0.0, 1.0)):
    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.position.z = z
    (
        message.pose.pose.orientation.x,
        message.pose.pose.orientation.y,
        message.pose.pose.orientation.z,
        message.pose.pose.orientation.w,
    ) = quaternion
    return message


def test_localization_alignment_anchors_macvio_to_reference():
    yaw_90 = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    reference = make_odometry(10.0, 20.0, 1.0, yaw_90)
    macvio_anchor = make_odometry(2.0, 3.0, -1.0)
    alignment = initial_alignment(reference, macvio_anchor)

    output = transform_odometry(
        macvio_anchor, alignment, "sensor_init", "vehicle"
    )

    np.testing.assert_allclose(
        pose_matrix(output), pose_matrix(reference), atol=1e-8
    )
    assert output.header.frame_id == "sensor_init"
    assert output.child_frame_id == "vehicle"


def test_localization_alignment_preserves_macvio_relative_motion():
    yaw_90 = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))
    alignment = initial_alignment(
        make_odometry(10.0, 20.0, 0.0, yaw_90),
        make_odometry(0.0, 0.0, 0.0),
    )
    current = make_odometry(2.0, 0.0, 0.0)
    current.twist.twist.linear.x = 3.0

    output = transform_odometry(current, alignment, "sensor_init", "vehicle")

    np.testing.assert_allclose(
        [output.pose.pose.position.x, output.pose.pose.position.y],
        [10.0, 22.0],
        atol=1e-8,
    )
    assert output.twist.twist.linear.x == pytest.approx(3.0)


def test_imu_window_reports_largest_gap_including_previous():
    samples = [make_imu_sample(value) for value in (1_070_000_000, 1_080_000_000)]
    assert validate_imu_window(samples, 10_000_000) == pytest.approx(1.06)
    assert validate_imu_window(samples, None) == pytest.approx(0.01)


def test_imu_window_accepts_empty_and_single_sample():
    assert validate_imu_window([], 10_000_000) == 0.0
    assert validate_imu_window([make_imu_sample(20_000_000)], None) == 0.0


def test_imu_window_rejects_non_increasing_timestamps():
    samples = [make_imu_sample(value) for value in (10_000_000, 10_000_000)]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_imu_window(samples, None)


@pytest.mark.parametrize("modality", ["thermal", "rgb"])
def test_profiles_take_imu_jitter_from_core(modality):
    config = load_macvio_config(modality)[0]
    assert config.Odometry.imu_health.jitter_max_ms == 50.0
    assert not hasattr(config.ROS.sync, "max_imu_sample_gap_sec")
    assert not hasattr(config.ROS.sync, "max_imu_camera_offset_sec")
    assert config.ROS.output.degraded_pose_covariance_scale == 10.0


@pytest.mark.parametrize("modality", ["thermal", "rgb"])
def test_profiles_define_health_limits(modality):
    health = load_macvio_config(modality)[0].ROS.health
    assert (health.blind_warn_s, health.max_blind_s) == (1.0, 3.0)
    assert health.reinit_timeout_s == 10.0
    assert health.watchdog_rate_hz == 10.0


@pytest.mark.parametrize("modality,topic_prefix", [("thermal", "/thermal_"), ("rgb", "/multisense/")])
def test_ros_node_selects_camera_and_builds_frame(monkeypatch, modality, topic_prefix):
    system = mock_system()
    factory = Mock(return_value=system)
    monkeypatch.setattr(MACVIO, "from_config", factory)
    rclpy.init(args=["--ros-args", "-p", f"modality:={modality}"])
    node = None
    try:
        node = MacvioNode()
        assert node.modality == modality
        assert sum(subscription.topic_name.startswith(topic_prefix) for subscription in node.subscriptions) == 2
        assert any(subscription.topic_name == "/novatel/imu/data" for subscription in node.subscriptions)
        sample = ImuSample(1, (0, 0, 9.80665), (0, 0, 0), (0,) * 9, (0,) * 9, "novatel/imu_frame")
        monkeypatch.setattr(node, "_imu_samples_for_frame", lambda _: ([sample], 0.0))
        height, width = node.camera_mask.shape[-2:]
        message = Image(height=height, width=width, encoding="mono8", step=width,
                        data=np.zeros((height, width), dtype=np.uint8).tobytes())
        message.header.stamp.nanosec = 1
        node._process_stereo(message, message)
        frame = system.run.call_args.args[0]
        assert frame.stereo.imageL.shape[-2:] == ((512, 640) if modality == "thermal" else (272, 512))
        assert frame.imu.acc_ws.shape == (1, 1, 3)
        assert node._frames_processed == 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.fixture
def mocked_node(monkeypatch):
    systems = []

    def factory(_config):
        systems.append(mock_system())
        return systems[-1]

    monkeypatch.setattr(MACVIO, "from_config", Mock(side_effect=factory))
    rclpy.init(args=["--ros-args", "-p", "modality:=thermal"])
    node = MacvioNode()
    node.status_publisher = Mock()
    try:
        yield node, systems
    finally:
        node.destroy_node()
        rclpy.shutdown()


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def last_status(node):
    call = node.status_publisher.publish.call_args
    return {} if call is None else json.loads(call.args[0].data)


def test_imu_windows_chain_without_repeats(mocked_node):
    node, _ = mocked_node
    node._imu_buffer.extend(make_imu_sample(value) for value in range(0, 60, 10))

    def window(frame_ns):
        samples, gap = node._imu_samples_for_frame(frame_ns)
        if samples:
            node._advance_imu_boundary(samples[-1], resend=node._last_frame_ns is None)
        node._last_frame_ns = frame_ns
        return [sample.time_ns for sample in samples], gap

    assert window(15) == ([10], 0.0)
    assert window(35) == ([10, 20, 30], pytest.approx(1e-8))
    assert window(45) == ([40], pytest.approx(1e-8))
    assert window(48) == ([], 0.0)
    node._imu_buffer.append(make_imu_sample(100))
    assert window(105) == ([50, 100], pytest.approx(5e-8))


@pytest.mark.parametrize("count", [0, 1, 3])
def test_imu_data_supports_short_windows(mocked_node, count):
    node, _ = mocked_node
    samples = [make_imu_sample(value * 10_000_000) for value in range(count)]
    imu = node._make_imu_data(samples)
    assert imu.time_ns.shape == (1, count, 1)
    assert imu.acc_ws.shape == imu.gyro_ws.shape == (1, count, 3)
    assert imu.acc_cov.shape == imu.gyro_cov.shape == (1, count, 3, 3)
    assert imu.num_samples == count


def test_stale_imu_does_not_delay_frames(mocked_node):
    node, _ = mocked_node
    node._imu_buffer.append(make_imu_sample(10))
    node._last_imu_arrival = time.monotonic() - 1.0
    started = time.monotonic()
    assert node._imu_samples_for_frame(1_000_000_000)[0][0].time_ns == 10
    assert time.monotonic() - started < 0.02


def test_fresh_imu_wait_stops_once_imu_goes_stale(mocked_node):
    node, _ = mocked_node
    node._imu_buffer.append(make_imu_sample(10))
    node._last_imu_arrival = time.monotonic()
    started = time.monotonic()
    node._imu_samples_for_frame(1_000_000_000)
    elapsed = time.monotonic() - started
    assert node.imu_jitter_sec * 0.8 < elapsed < node.imu_wait_timeout


def test_non_finite_imu_samples_are_rejected(mocked_node):
    node, _ = mocked_node
    message = Imu()
    message.header.stamp.nanosec = 10
    message.linear_acceleration.x = float("nan")
    node._receive_imu(message)
    message.linear_acceleration.x = 0.0
    node._receive_imu(message)
    node._receive_imu(message)
    assert [sample.time_ns for sample in node._imu_buffer] == [10]
    assert node._imu_samples_rejected == 2


def test_unexpected_error_stops_with_error_state(mocked_node, monkeypatch):
    node, systems = mocked_node

    def fail(*_):
        raise RuntimeError("bug")

    monkeypatch.setattr(node, "_process_stereo", fail)
    node._receive_stereo(Image(), Image())
    wait_until(node._stop_event.is_set)
    wait_until(lambda: last_status(node).get("state") == "ERROR")

    status = last_status(node)
    assert status["last_error"] == "RuntimeError: bug"
    assert status["reset_count"] == 0
    assert len(systems) == 1
    node._receive_stereo(Image(), Image())
    assert node._frames_ignored_after_stop == 1


def test_status_reports_liveness_fields(mocked_node):
    node, _ = mocked_node
    node._publish_status()
    status = last_status(node)
    for key in (
        "state", "reset_count", "last_reset_reason", "last_reset_stamp",
        "imu_age_ms", "cam_age_ms", "frames_received", "frames_processed",
        "frames_dropped", "imu_gap_count", "initialized", "lost_track",
        "vio_mode", "frames_in_mode", "last_gap_s", "n_chain_breaks",
        "n_dropped_samples", "imu_samples_rejected",
    ):
        assert key in status
    assert status["state"] == "INITIALIZING"
    assert status["vio_mode"] == "PRE_INIT"


def test_rebuild_reuses_tracker_networks():
    config, _, _ = load_macvio_config("thermal")
    tracker = object.__new__(TwoframeTracker_v2)
    tracker.Frontend = object()
    tracker.prev = object()
    tracker.last_depth = object()
    previous = Mock(FeatureTracker=tracker)

    system = build_macvio(config, previous)

    assert isinstance(system, MACVIO) and system is not previous
    assert system.FeatureTracker is not tracker
    assert system.FeatureTracker.Frontend is tracker.Frontend
    assert system.FeatureTracker.prev is None
    assert system.FeatureTracker.last_depth is None
    assert tracker.prev is not None
    assert system.imu_initcond is None and len(system.Map.body) == 0
    assert system.vio_status == idle_vio_status()
    assert system.IMUChain.preintegrator is system.IMUPreintegrator

