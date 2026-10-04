"""Check ROS camera selection against the upstream off-road VIO profiles."""

from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
import torch
import yaml

from mac_ros2.macvio_node import (
    ImuContinuityError,
    ImuSample,
    MACSLAM_PATH,
    MACVIO,
    MacvioNode,
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
from Src.DataLoader.Dataset.Offroad import (
    RGB_T_BS_BODY_CAM,
    THERMAL_T_BS_BODY_CAM,
)
from Src.Utility.Config import load_config


@pytest.mark.parametrize("modality,label", [("thermal", "Thermal"), ("rgb", "RGB")])
def test_profiles_load_upstream_estimator(modality, label):
    config, _, experiment_path = load_macvio_config(modality)
    expected_path = MACSLAM_PATH / f"Config/Experiment/MACVIO/VIO_MACI2_FR_Offroad_{label}.yaml"
    upstream, _ = load_config(expected_path)
    assert experiment_path == expected_path
    assert config.Common == upstream.Common
    assert config.Odometry == upstream.Odometry
    MACVIO.is_valid_config(config.Odometry)
    sequence = getattr(upstream.Preprocess, f"Offroad_{label}_VIO")
    assert config.ROS.preprocess == sequence[0]


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


@pytest.mark.parametrize("modality,height,width", [("thermal", 400, 640), ("rgb", 272, 512)])
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
        expected[:, 1, 2] -= 56
        torch.testing.assert_close(result.camera_mask, mask[:, :, 56:456])
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


def test_imu_window_accepts_continuous_coverage():
    samples = [make_imu_sample(value) for value in (0, 10_000_000, 20_000_000)]
    largest_gap = validate_imu_window(samples, 25_000_000, 0.03, 0.03)
    assert largest_gap == pytest.approx(0.01)


def test_imu_window_rejects_internal_gap():
    samples = [make_imu_sample(value) for value in (0, 10_000_000, 1_390_000_000)]
    with pytest.raises(ImuContinuityError, match="sample gap is too large"):
        validate_imu_window(samples, 1_400_000_000, 0.03, 0.03)


def test_imu_window_rejects_stale_camera_coverage():
    samples = [make_imu_sample(value) for value in (0, 10_000_000, 20_000_000)]
    with pytest.raises(ImuContinuityError, match="does not cover camera frame"):
        validate_imu_window(samples, 100_000_000, 0.03, 0.03)


@pytest.mark.parametrize("modality,topic_prefix", [("thermal", "/thermal_"), ("rgb", "/multisense/")])
def test_ros_node_selects_camera_and_builds_frame(monkeypatch, modality, topic_prefix):
    system = Mock(imu_initcond=None, lost_track=False)
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
        monkeypatch.setattr(node, "_imu_samples_for_frame", lambda _: [sample])
        height, width = node.camera_mask.shape[-2:]
        message = Image(height=height, width=width, encoding="mono8", step=width,
                        data=np.zeros((height, width), dtype=np.uint8).tobytes())
        message.header.stamp.nanosec = 1
        node._process_stereo(message, message)
        frame = system.run.call_args.args[0]
        assert frame.stereo.imageL.shape[-2:] == ((400, 640) if modality == "thermal" else (272, 512))
        assert frame.imu.acc_ws.shape == (1, 1, 3)
        assert node._frames_processed == 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
