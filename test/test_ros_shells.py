"""ROS 2 message conversions, the adapter shell and the launch file (needs rclpy)."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from diagnostic_msgs.msg import DiagnosticStatus  # noqa: E402
from launch import LaunchContext  # noqa: E402
from launch_ros.actions import Node as NodeAction, PushRosNamespace  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from mac_ros2.localization_adapter_node import LocalizationAdapter  # noqa: E402
from mac_ros2.localization_gate import LocalizationGate  # noqa: E402
from mac_ros2.odometry_state import MAP_POINT_FIELDS, MapPoints, OdometryState  # noqa: E402
from mac_ros2.ros_messages import (  # noqa: E402
    camera_info_message,
    compressed_array,
    diagnostic_status,
    image_array,
    imu_sample,
    mono8_message,
    ns_to_stamp,
    odometry_message,
    odometry_state,
    parse_status,
    point_cloud_message,
    stamp_to_ns,
    static_rotation_message,
    status_message,
    transform_message,
)

LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "macvio.launch.py"
RNG = np.random.default_rng(1)


def make_odometry(stamp_ns: int, x: float = 1.0) -> OdometryState:
    return OdometryState(
        stamp_ns=stamp_ns,
        pose=np.array([x, 2.0, 3.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]),
        pose_cov=np.arange(36, dtype=np.float64).reshape(6, 6),
        twist=np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.2]),
        twist_cov=np.eye(6) * 0.3,
    )


@pytest.mark.parametrize("time_ns", [0, 1, 1_700_000_000_123_456_789])
def test_stamp_round_trip(time_ns):
    assert stamp_to_ns(ns_to_stamp(time_ns)) == time_ns


def test_imu_sample():
    message = Imu()
    message.header.stamp = ns_to_stamp(5_000_000_123)
    message.linear_acceleration.x, message.linear_acceleration.z = 0.1, 9.8
    message.angular_velocity.y = -0.2
    assert imu_sample(message) == (5_000_000_123, (0.1, 0.0, 9.8), (0.0, -0.2, 0.0))


@pytest.mark.parametrize("encoding,image", [
    ("mono16", RNG.integers(0, 65535, size=(4, 6), dtype=np.uint16)),
    ("mono8", RNG.integers(0, 255, size=(4, 6), dtype=np.uint8)),
    ("bgr8", RNG.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)),
])
def test_image_message_to_array(encoding, image):
    message = Image(height=4, width=6, encoding=encoding, step=image.strides[0], data=image.tobytes())
    np.testing.assert_array_equal(image_array(message), image)


def test_compressed_message_to_array():
    image = RNG.integers(0, 65535, size=(4, 6), dtype=np.uint16)
    message = CompressedImage(format="mono16; png compressed ", data=cv2.imencode(".png", image)[1].tobytes())
    decoded = compressed_array(message)
    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, image)


def test_odometry_round_trip_and_transform():
    odometry = make_odometry(1_234_567_890_123)
    message = odometry_message(odometry, "macvio_odom", "vehicle")
    assert (message.header.frame_id, message.child_frame_id) == ("macvio_odom", "vehicle")
    back = odometry_state(message)
    assert back.stamp_ns == odometry.stamp_ns
    for field in ("pose", "pose_cov", "twist", "twist_cov"):
        np.testing.assert_array_equal(getattr(back, field), getattr(odometry, field))

    transform = transform_message(message, "macvio_vehicle")
    assert transform.header.frame_id == "macvio_odom" and transform.child_frame_id == "macvio_vehicle"
    assert transform.transform.translation.x == 1.0 and transform.transform.rotation.w == pytest.approx(np.sqrt(0.5))


def test_point_cloud_message_carries_every_map_point_field():
    values = RNG.normal(size=(5, 15)).astype(np.float32)
    message = point_cloud_message(MapPoints(stamp_ns=1_500_000_000, values=values), "crl_rzr/base_link")
    assert stamp_to_ns(message.header.stamp) == 1_500_000_000 and message.header.frame_id == "crl_rzr/base_link"
    assert [f.name for f in message.fields] == list(MAP_POINT_FIELDS)
    assert (message.height, message.width, message.point_step, message.row_step) == (1, 5, 60, 300)
    dtype = np.dtype({"names": [f.name for f in message.fields], "formats": ["<f4"] * 15,
                      "offsets": [f.offset for f in message.fields]})
    decoded = np.frombuffer(bytes(message.data), dtype=dtype)
    np.testing.assert_array_equal(np.stack([decoded[name] for name in MAP_POINT_FIELDS], axis=1), values)
    empty = point_cloud_message(MapPoints(stamp_ns=1, values=np.zeros((0, 15), np.float32)), "vehicle")
    assert empty.width == 0 and len(empty.data) == 0


def test_processed_image_and_camera_info_messages():
    image = RNG.integers(0, 255, size=(4, 6), dtype=np.uint8)
    message = mono8_message(7, "thermal_left/optical_frame_rect", image)
    assert (message.encoding, message.step, message.header.frame_id) == ("mono8", 6, "thermal_left/optical_frame_rect")
    np.testing.assert_array_equal(image_array(message), image)

    K = np.array([[300.0, 0.0, 320.0], [0.0, 301.0, 256.0], [0.0, 0.0, 1.0]])
    info = camera_info_message(7, "thermal_left/optical_frame_rect", 640, 512, K)
    assert (info.width, info.height, stamp_to_ns(info.header.stamp)) == (640, 512, 7)
    np.testing.assert_array_equal(np.reshape(info.k, (3, 3)), K)
    np.testing.assert_array_equal(np.reshape(info.p, (3, 4))[:, :3], K)
    assert not any(info.d) and not np.reshape(info.p, (3, 4))[:, 3].any()
    np.testing.assert_array_equal(np.reshape(info.r, (3, 3)), np.eye(3))


def test_static_rotation_message():
    transform = static_rotation_message("thermal_left/optical_frame", "thermal_left/optical_frame_rect",
                                        np.array([0.1, 0.2, 0.3, 0.9]))
    assert (transform.header.frame_id, transform.child_frame_id) == (
        "thermal_left/optical_frame", "thermal_left/optical_frame_rect")
    r, t = transform.transform.rotation, transform.transform.translation
    assert (r.x, r.y, r.z, r.w) == (0.1, 0.2, 0.3, 0.9) and (t.x, t.y, t.z) == (0.0, 0.0, 0.0)


def test_status_json_round_trip():
    status = {"state": "TRACKING", "last_frame_stamp_ns": 7, "vio_status": {"vio_mode": "VIO"}}
    assert parse_status(status_message(status)) == status
    assert parse_status(String(data="{not json")) is None
    assert parse_status(String(data="[1, 2]")) is None


def test_diagnostic_status_from_gate():
    gate = LocalizationGate()
    gate.on_status({"state": "DEGRADED", "last_frame_stamp_ns": 1, "vio_status": {"vio_mode": "VO_FALLBACK"}}, 0)
    message = diagnostic_status(gate.diagnostic(0))
    assert message.level == DiagnosticStatus.ERROR and message.message == "ALIGNING"
    values = {kv.key: kv.value for kv in message.values}
    assert values["vio_status.vio_mode"] == "VO_FALLBACK" and values["aligned"] == "false"


@pytest.fixture
def adapter(request) -> Any:
    rclpy.init(args=["--ros-args", "-p", f"pose_source:={request.param}"])
    node = LocalizationAdapter()
    sent: dict[str, list[Any]] = {"odometry": [], "hold": [], "status": [], "diagnostics": []}
    for key in sent:
        getattr(node, f"{key}_pub").publish = sent[key].append
    node.tf_broadcaster.sendTransform = lambda transform: sent.setdefault("tf", []).append(transform)
    yield node, sent
    node.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize("adapter", ["macvio"], indirect=True)
def test_adapter_gates_aligns_and_publishes(adapter):
    node, sent = adapter
    assert "/macvio/odometry_raw" in {s.topic_name for s in node.subscriptions}
    now = node.get_clock().now().nanoseconds
    reference = odometry_message(make_odometry(now, x=10.0), "sensor_init", "vehicle")
    node._on_reference(reference)
    macvio = odometry_message(make_odometry(now, x=1.0), "macvio_odom", "vehicle")
    node._on_macvio(macvio)
    assert sent["odometry"] == [] and sent["hold"][-1].data is True

    node._on_status(String(data=json.dumps({"state": "TRACKING", "last_frame_stamp_ns": now})))
    node._on_macvio(macvio)
    out = sent["odometry"][-1]
    assert (out.header.frame_id, out.child_frame_id) == ("sensor_init", "vehicle")
    assert out.pose.pose.position.x == pytest.approx(10.0)
    assert sent["tf"][-1].child_frame_id == "vehicle"
    assert sent["hold"][-1].data is False
    assert sent["status"][-1].message == "TRACKING" and sent["diagnostics"][-1].status[0].message == "TRACKING"

    node._on_status(String(data=json.dumps({"state": "ERROR", "last_frame_stamp_ns": now})))
    node._on_macvio(macvio)
    assert len(sent["odometry"]) == 1 and sent["hold"][-1].data is True and sent["status"][-1].message == "ERROR"


@pytest.mark.parametrize("adapter", ["super_odometry"], indirect=True)
def test_adapter_passes_the_reference_through(adapter):
    node, sent = adapter
    reference = odometry_message(make_odometry(5, x=4.0), "sensor_init", "vehicle")
    node._on_reference(reference)
    node._on_macvio(odometry_message(make_odometry(5), "macvio_odom", "vehicle"))
    node._tick()
    assert len(sent["odometry"]) == 1 and sent["odometry"][0].pose.pose.position.x == 4.0
    assert sent["hold"] == []


def launch_module() -> Any:
    spec = importlib.util.spec_from_file_location("macvio_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEFAULTS = {"platform": "yamaha", "modality": "thermal", "config_file": "", "pose_source": "macvio",
            "use_sim_time": "false", "reference_topic": "", "output_topic": "", "world_frame": "", "base_frame": ""}


def launch_actions(**arguments: str) -> list[Any]:
    module = launch_module()
    module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update({**DEFAULTS, **arguments})
    (group,) = module._launch_nodes(context)
    return [a for a in group.get_sub_entities() if isinstance(a, (PushRosNamespace, NodeAction))]


def test_launch_defaults_to_macvio():
    description = launch_module().generate_launch_description()
    defaults = {a.name: a.default_value for a in description.entities if hasattr(a, "default_value")}
    assert defaults["pose_source"][0].text == "macvio"


def test_launch_yamaha_keeps_the_deploy_arguments():
    namespace, macvio, adapter = launch_module().resolve({**DEFAULTS, "modality": "rgb", "pose_source": "macvio"})
    assert namespace == "" and macvio == {"profile": "yamaha_rgb", "use_sim_time": False}
    assert adapter == {
        "reference_topic": "/localization/super_odometry", "output_topic": "/superodometry/integrated_to_init",
        "world_frame": "sensor_init", "base_frame": "vehicle", "pose_source": "macvio", "use_sim_time": False,
    }
    actions = launch_actions(modality="rgb")
    assert not any(isinstance(a, PushRosNamespace) for a in actions)
    assert [a.node_executable for a in actions] == ["MACVIO", "localization_adapter"]


def test_launch_stallion_pushes_its_namespace_and_maps_config_file():
    namespace, _, adapter = launch_module().resolve({**DEFAULTS, "platform": "stallion"})
    assert namespace == "crl_rzr" and adapter["output_topic"] == "macvio/odometry"
    arguments = {**DEFAULTS, "platform": "stallion", "config_file": "/tmp/custom.yaml", "output_topic": "odom_out",
                 "use_sim_time": "true"}
    namespace, macvio, adapter = launch_module().resolve(arguments)
    assert namespace == "crl_rzr" and macvio == {"profile": "/tmp/custom.yaml", "use_sim_time": True}
    assert adapter["output_topic"] == "odom_out" and adapter["reference_topic"] == "liopard/fused/odometry"
    assert adapter["world_frame"] == "crl_rzr/odom" and adapter["base_frame"] == "crl_rzr/base_link"
    actions = launch_actions(platform="stallion")
    assert isinstance(actions[0], PushRosNamespace)
    assert [a.node_executable for a in actions[1:]] == ["MACVIO", "localization_adapter"]


RECT_ROTATION = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])


class FakeRunner:
    """Stands in for `MacvioRunner`, whose mac_slam imports this test avoids."""

    def __init__(self, profile: Any, *, now_ns: Any, on_odometry: Any, on_status: Any, on_map_points: Any,
                 on_processed_image: Any, log: Any) -> None:
        self.profile, self.on_odometry, self.on_status = profile, on_odometry, on_status
        self.on_map_points, self.on_processed_image = on_map_points, on_processed_image
        self.rect_rotation = RECT_ROTATION if profile.frames.rect_tf_parent is not None else None
        self.calls: list[tuple[str, Any]] = []
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.stopped = True

    def on_imu(self, *args: Any) -> None:
        self.calls.append(("imu", args))

    def on_stereo(self, *args: Any) -> None:
        self.calls.append(("stereo", args))

    def on_bad_input(self, problem: str) -> None:
        self.calls.append(("bad", problem))

    def tick(self, now_ns: int) -> None:
        self.calls.append(("tick", now_ns))


class FakeStaticBroadcaster:
    sent: list[Any] = []

    def __init__(self, _node: Any) -> None:
        FakeStaticBroadcaster.sent = []

    def sendTransform(self, transform: Any) -> None:
        FakeStaticBroadcaster.sent.append(transform)


@pytest.fixture
def macvio_node(monkeypatch, request) -> Any:
    fake = types.ModuleType("mac_ros2.macvio_runner")
    fake.MacvioRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "mac_ros2.macvio_runner", fake)
    monkeypatch.delitem(sys.modules, "mac_ros2.macvio_node", raising=False)
    module = importlib.import_module("mac_ros2.macvio_node")
    monkeypatch.setattr(module, "StaticTransformBroadcaster", FakeStaticBroadcaster)
    FakeStaticBroadcaster.sent = []
    rclpy.init(args=["--ros-args", "-p", f"profile:={request.param}"])
    node = module.MacvioNode()
    sent: dict[str, list[Any]] = {"odometry": [], "status": [], "map_points": [], "image": [], "camera_info": [],
                                  "static_tf": FakeStaticBroadcaster.sent}
    for key in ("odometry", "status", "map_points", "image", "camera_info"):
        publisher = getattr(node, f"{key}_pub")
        if publisher is not None:
            publisher.publish = sent[key].append
    yield node, sent
    node.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize("macvio_node", ["stallion_thermal"], indirect=True)
def test_macvio_node_shell_wiring(macvio_node):
    node, sent = macvio_node
    runner = node.runner
    assert runner.started and runner.profile.name == "stallion_thermal"
    topics = {s.topic_name for s in node.subscriptions}
    assert {"/microstrain/nav/filtered_imu/data", "/multisense_thermal_front/left/image_mono/compressed",
            "/multisense_thermal_front/right/image_mono/compressed"} <= topics

    imu = Imu()
    imu.header.stamp = ns_to_stamp(42)
    node._on_imu(imu)
    image = RNG.integers(0, 65535, size=(4, 6), dtype=np.uint16)
    left = CompressedImage(data=cv2.imencode(".png", image)[1].tobytes())
    left.header.stamp = ns_to_stamp(1_000)
    node._on_stereo(left, left)
    node._on_stereo(CompressedImage(data=b"junk"), left)
    node._tick()
    kinds = [kind for kind, _ in runner.calls]
    assert kinds == ["imu", "stereo", "bad", "tick"]
    stamp, left_raw, right_raw = runner.calls[1][1]
    assert stamp == 1_000 and left_raw.dtype == np.uint16
    np.testing.assert_array_equal(right_raw, image)

    runner.on_status({"state": "TRACKING", "last_frame_stamp_ns": 1_000})
    runner.on_odometry(make_odometry(1_000))
    assert parse_status(sent["status"][-1])["state"] == "TRACKING"
    odometry = sent["odometry"][-1]
    assert (odometry.header.frame_id, odometry.child_frame_id) == ("crl_rzr/macvio_odom", "crl_rzr/base_link")
    assert not hasattr(node, "tf_broadcaster")

    runner.on_map_points(MapPoints(stamp_ns=1_000, values=np.zeros((3, 15), np.float32)))
    cloud = sent["map_points"][-1]
    assert cloud.header.frame_id == "crl_rzr/base_link" and cloud.width == 3 and stamp_to_ns(cloud.header.stamp) == 1_000

    published = {name for name, _ in node.get_publisher_names_and_types_by_node("macvio_node", "/")}
    assert {"/macvio/odometry_raw", "/macvio/status", "/macvio/map_points"} <= published


def publish_processed_image(runner: FakeRunner, sent: dict[str, list[Any]]) -> tuple[Image, CameraInfo]:
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 256.0], [0.0, 0.0, 1.0]])
    runner.on_processed_image(1_000, np.zeros((512, 640), np.uint8), K)
    image, info = sent["image"][-1], sent["camera_info"][-1]
    assert image.header.frame_id == info.header.frame_id and (info.width, info.height) == (640, 512)
    assert stamp_to_ns(image.header.stamp) == stamp_to_ns(info.header.stamp) == 1_000
    np.testing.assert_allclose(np.reshape(info.p, (3, 4)), np.hstack([K, np.zeros((3, 1))]))
    return image, info


@pytest.mark.parametrize("macvio_node", ["stallion_thermal"], indirect=True)
def test_macvio_node_stallion_stamps_the_driver_optical_frame_without_a_tf(macvio_node):
    node, sent = macvio_node
    image, _ = publish_processed_image(node.runner, sent)
    assert image.header.frame_id == "crl_rzr/multisense_thermal_front/left_camera_optical_frame"
    assert sent["static_tf"] == [] and node.static_tf_broadcaster is None
    published = {name for name, _ in node.get_publisher_names_and_types_by_node("macvio_node", "/")}
    assert {"/multisense_thermal_front/left/image_processed",
            "/multisense_thermal_front/left/camera_info_processed"} <= published
    assert not any(name.endswith("/camera_info") for name in published)


@pytest.mark.parametrize("macvio_node", ["yamaha_thermal"], indirect=True)
def test_macvio_node_yamaha_publishes_the_rect_tf(macvio_node):
    node, sent = macvio_node
    image, _ = publish_processed_image(node.runner, sent)
    assert image.header.frame_id == "thermal_left/optical_frame_rect"
    (static,) = sent["static_tf"]
    assert (static.header.frame_id, static.child_frame_id) == ("thermal_left/optical_frame",
                                                               "thermal_left/optical_frame_rect")
    assert static.transform.rotation.w == pytest.approx(RECT_ROTATION[3])
    published = {name for name, _ in node.get_publisher_names_and_types_by_node("macvio_node", "/")}
    assert {"/thermal_left/image_processed", "/thermal_left/camera_info_processed"} <= published
    assert not any(name.endswith("/camera_info") for name in published)


@pytest.mark.parametrize("macvio_node", ["yamaha_rgb"], indirect=True)
def test_macvio_node_rgb_publishes_no_processed_image(macvio_node):
    node, sent = macvio_node
    assert node.image_pub is None and node.camera_info_pub is None and sent["static_tf"] == []
    published = {name for name, _ in node.get_publisher_names_and_types_by_node("macvio_node", "/")}
    assert "/macvio/map_points" in published and not any("image_processed" in name for name in published)
