"""ROS 2 shell around `LocalizationGate`: the single localization publisher for the autonomy stack."""

from __future__ import annotations

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from .localization_gate import LocalizationGate
from .odometry_state import OdometryState
from .ros_messages import diagnostic_status, odometry_message, odometry_state, parse_status, transform_message

HOLD_RATE_HZ = 10.0
STATUS_EVERY_TICKS = 10


class LocalizationAdapter(Node):
    """`pose_source` macvio: gated, aligned MACVIO. super_odometry: the reference passed through."""

    def __init__(self) -> None:
        super().__init__("localization_adapter")
        self.pose_source = str(self.declare_parameter("pose_source", "super_odometry").value)
        if self.pose_source not in ("super_odometry", "macvio"):
            raise ValueError(f"pose_source must be 'super_odometry' or 'macvio', got {self.pose_source!r}")
        reference_topic = str(self.declare_parameter("reference_topic", "/localization/super_odometry").value)
        macvio_topic = str(self.declare_parameter("macvio_topic", "macvio/odometry").value)
        macvio_status_topic = str(self.declare_parameter("macvio_status_topic", "macvio/status").value)
        output_topic = str(self.declare_parameter("output_topic", "/superodometry/integrated_to_init").value)
        self.world_frame = str(self.declare_parameter("world_frame", "sensor_init").value)
        self.base_frame = str(self.declare_parameter("base_frame", "vehicle").value)
        self.publish_tf = bool(self.declare_parameter("publish_tf", True).value)
        self.gate = LocalizationGate(
            stale_after_s=float(self.declare_parameter("stale_after_s", 0.5).value),
            alignment_tolerance_s=float(self.declare_parameter("alignment_tolerance_s", 0.05).value),
        )

        self.odometry_pub = self.create_publisher(Odometry, output_topic, 10)
        self.hold_pub = self.create_publisher(Bool, "localization/perception_hold", 10)
        self.status_pub = self.create_publisher(DiagnosticStatus, "localization/status", 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._ticks = 0
        self._last_state: str | None = None
        self.create_subscription(Odometry, reference_topic, self._on_reference, qos_profile_sensor_data)
        self.create_subscription(Odometry, macvio_topic, self._on_macvio, qos_profile_sensor_data)
        self.create_subscription(String, macvio_status_topic, self._on_status, 10)
        self.create_timer(1.0 / HOLD_RATE_HZ, self._tick)
        self.get_logger().info(
            f"Localization source {self.pose_source}: {output_topic} ({self.world_frame} -> {self.base_frame})"
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_reference(self, message: Odometry) -> None:
        reference = odometry_state(message)
        self.gate.on_reference(reference)
        if self.pose_source == "super_odometry":
            self._publish_odometry(reference)

    def _on_status(self, message: String) -> None:
        status = parse_status(message)
        if status is not None:
            self.gate.on_status(status, self._now_ns())
            self._publish_state(heartbeat=False)

    def _on_macvio(self, message: Odometry) -> None:
        odometry = self.gate.on_odometry(odometry_state(message), self._now_ns())
        if odometry is not None and self.pose_source == "macvio":
            self._publish_odometry(odometry)
        self._publish_state(heartbeat=False)

    def _tick(self) -> None:
        self._ticks += 1
        self._publish_state(heartbeat=True)

    def _publish_odometry(self, odometry: OdometryState) -> None:
        message = odometry_message(odometry, self.world_frame, self.base_frame)
        self.odometry_pub.publish(message)
        if self.publish_tf:
            self.tf_broadcaster.sendTransform(transform_message(message, self.base_frame))

    def _publish_state(self, heartbeat: bool) -> None:
        """Hold at 10 Hz and status at 1 Hz, both also on a state change."""
        now = self._now_ns()
        diagnostic = self.gate.diagnostic(now)
        changed = diagnostic["message"] != self._last_state
        if changed:
            self.get_logger().info(f"Localization state {self._last_state} -> {diagnostic['message']}")
            self._last_state = diagnostic["message"]
        if self.pose_source == "macvio" and (heartbeat or changed):
            self.hold_pub.publish(Bool(data=self.gate.hold(now)))
        if changed or (heartbeat and self._ticks % STATUS_EVERY_TICKS == 0):
            status = diagnostic_status(diagnostic)
            self.status_pub.publish(status)
            array = DiagnosticArray(status=[status])
            array.header.stamp = self.get_clock().now().to_msg()
            self.diagnostics_pub.publish(array)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LocalizationAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
