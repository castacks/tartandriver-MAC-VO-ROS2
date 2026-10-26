"""Expose a selected localization source through the autonomy frame contract."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
import math

import numpy as np

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .vio_health import MacvioSourceGate


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("pose contains an invalid quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def pose_matrix(message: Odometry) -> np.ndarray:
    """Read the parent-from-child transform carried by an odometry message."""
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_matrix(
        np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
    )
    transform[:3, 3] = [position.x, position.y, position.z]
    return transform


def initial_alignment(reference: Odometry, macvio: Odometry) -> np.ndarray:
    """Return reference-world from MACVIO-world at one synchronized pose."""
    return pose_matrix(reference) @ np.linalg.inv(pose_matrix(macvio))


def transform_odometry(
    source: Odometry,
    world_from_source_world: np.ndarray,
    world_frame: str,
    base_frame: str,
) -> Odometry:
    """Apply a fixed world-frame alignment while preserving body-frame twist."""
    output = deepcopy(source)
    transformed = world_from_source_world @ pose_matrix(source)
    quaternion = matrix_quaternion(transformed[:3, :3])
    output.header.frame_id = world_frame
    output.child_frame_id = base_frame
    output.pose.pose.position.x = float(transformed[0, 3])
    output.pose.pose.position.y = float(transformed[1, 3])
    output.pose.pose.position.z = float(transformed[2, 3])
    output.pose.pose.orientation.x = float(quaternion[0])
    output.pose.pose.orientation.y = float(quaternion[1])
    output.pose.pose.orientation.z = float(quaternion[2])
    output.pose.pose.orientation.w = float(quaternion[3])

    covariance = np.asarray(source.pose.covariance, dtype=np.float64).reshape(6, 6)
    rotation = world_from_source_world[:3, :3]
    covariance_rotation = np.zeros((6, 6), dtype=np.float64)
    covariance_rotation[:3, :3] = rotation
    covariance_rotation[3:, 3:] = rotation
    output.pose.covariance = (
        covariance_rotation @ covariance @ covariance_rotation.T
    ).reshape(-1).tolist()
    return output


class LocalizationAdapter(Node):
    """Select Super Odometry or aligned MACVIO for the autonomy stack."""

    def __init__(self) -> None:
        super().__init__("localization_adapter")
        self.declare_parameter("pose_source", "super_odometry")
        self.declare_parameter(
            "super_odometry_topic", "/localization/super_odometry"
        )
        self.declare_parameter("macvio_topic", "/macvio/odometry")
        self.declare_parameter("macvio_status_topic", "/macvio/status")
        self.declare_parameter(
            "output_topic", "/superodometry/integrated_to_init"
        )
        self.declare_parameter("status_topic", "/localization/status")
        self.declare_parameter("world_frame", "sensor_init")
        self.declare_parameter("base_frame", "vehicle")
        self.declare_parameter("alignment_tolerance_sec", 0.05)
        self.declare_parameter("publish_tf", True)

        self.pose_source = str(self.get_parameter("pose_source").value)
        if self.pose_source not in {"super_odometry", "macvio"}:
            raise ValueError(
                "pose_source must be 'super_odometry' or 'macvio', got "
                f"{self.pose_source!r}"
            )
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.alignment_tolerance_ns = int(
            float(self.get_parameter("alignment_tolerance_sec").value) * 1e9
        )
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        output_topic = str(self.get_parameter("output_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        super_topic = str(self.get_parameter("super_odometry_topic").value)
        macvio_topic = str(self.get_parameter("macvio_topic").value)
        macvio_status_topic = str(
            self.get_parameter("macvio_status_topic").value
        )
        self.odometry_publisher = self.create_publisher(Odometry, output_topic, 10)
        self.status_publisher = self.create_publisher(String, status_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.reference_buffer: deque[Odometry] = deque(maxlen=2000)
        self.world_from_macvio_world: np.ndarray | None = None
        self.reference_messages = 0
        self.macvio_messages = 0
        self.output_messages = 0
        self.alignment_stamp_ns = 0
        self.macvio_gate = MacvioSourceGate()
        self.state = (
            "ready" if self.pose_source == "super_odometry" else "aligning"
        )

        self.create_subscription(
            Odometry,
            super_topic,
            self._receive_super_odometry,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            macvio_topic,
            self._receive_macvio,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            macvio_status_topic,
            self._receive_macvio_status,
            10,
        )
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "Localization pose source: %s; output %s (%s -> %s)"
            % (self.pose_source, output_topic, self.world_frame, self.base_frame)
        )

    @staticmethod
    def _stamp_ns(message: Odometry) -> int:
        return (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def _receive_super_odometry(self, message: Odometry) -> None:
        self.reference_messages += 1
        self.reference_buffer.append(message)
        if self.pose_source == "super_odometry":
            self._publish_selected(message, np.eye(4, dtype=np.float64))

    def _receive_macvio(self, message: Odometry) -> None:
        self.macvio_messages += 1
        if self.pose_source != "macvio":
            return
        if not self.macvio_gate.accept(self._stamp_ns(message)):
            return
        if self.world_from_macvio_world is None or self.macvio_gate.needs_alignment:
            reference = self._nearest_reference(message)
            if reference is None:
                self.state = "waiting_for_reference_alignment"
                return
            self.world_from_macvio_world = initial_alignment(reference, message)
            self.alignment_stamp_ns = self._stamp_ns(message)
            self.macvio_gate.needs_alignment = False
            self.state = "ready"
            self.get_logger().info(
                "Aligned macvio_odom to sensor_init at stamp %d (MACVIO reset %s)"
                % (self.alignment_stamp_ns, self.macvio_gate.reset_count)
            )
        self._publish_selected(message, self.world_from_macvio_world)

    def _nearest_reference(self, message: Odometry) -> Odometry | None:
        if not self.reference_buffer:
            return None
        stamp_ns = self._stamp_ns(message)
        reference = min(
            self.reference_buffer,
            key=lambda candidate: abs(self._stamp_ns(candidate) - stamp_ns),
        )
        difference_ns = abs(self._stamp_ns(reference) - stamp_ns)
        if difference_ns > self.alignment_tolerance_ns:
            return None
        return reference

    def _receive_macvio_status(self, message: String) -> None:
        try:
            newly_failed, reset = self.macvio_gate.update(json.loads(message.data))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            self.get_logger().error(
                "Ignoring malformed MACVIO status: %r" % error,
                throttle_duration_sec=5.0,
            )
            return
        if self.pose_source != "macvio":
            return
        if newly_failed:
            self.get_logger().error("Selected MACVIO localization stopped: ERROR")
        if reset:
            self.world_from_macvio_world = None
            self.get_logger().warning(
                "MACVIO reset %d; re-aligning to %s"
                % (self.macvio_gate.reset_count, self.world_frame)
            )
        if self.macvio_gate.failed:
            self.state = "source_error"
        elif not self.macvio_gate.available:
            self.state = f"source_{self.macvio_gate.state.lower()}"
        elif reset or self.state.startswith("source_"):
            self.state = (
                "aligning" if self.world_from_macvio_world is None else "ready"
            )

    def _publish_selected(
        self, message: Odometry, world_from_source_world: np.ndarray
    ) -> None:
        output = transform_odometry(
            message,
            world_from_source_world,
            self.world_frame,
            self.base_frame,
        )
        self.odometry_publisher.publish(output)
        self.output_messages += 1
        if self.publish_tf:
            transform = TransformStamped()
            transform.header = output.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = output.pose.pose.position.x
            transform.transform.translation.y = output.pose.pose.position.y
            transform.transform.translation.z = output.pose.pose.position.z
            transform.transform.rotation = output.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def _publish_status(self) -> None:
        message = String()
        message.data = json.dumps(
            {
                "pose_source": self.pose_source,
                "state": self.state,
                "reference_messages": self.reference_messages,
                "macvio_messages": self.macvio_messages,
                "output_messages": self.output_messages,
                "alignment_stamp_ns": self.alignment_stamp_ns,
                "macvio_state": self.macvio_gate.state,
                "macvio_reset_count": self.macvio_gate.reset_count,
                "world_frame": self.world_frame,
                "base_frame": self.base_frame,
            },
            sort_keys=True,
        )
        self.status_publisher.publish(message)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LocalizationAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
