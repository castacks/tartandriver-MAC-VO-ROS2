"""ROS 2 shell around `MacvioRunner`: stereo + IMU in, raw odometry and status out."""

from __future__ import annotations

from typing import Any

import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import String

from .macvio_profile import load_profile
from .macvio_runner import MacvioRunner
from .odometry_state import OdometryState
from .ros_messages import (
    compressed_array,
    image_array,
    imu_sample,
    odometry_message,
    stamp_to_ns,
    status_message,
)

STATUS_RATE_HZ = 10.0


class MacvioNode(Node):
    def __init__(self) -> None:
        super().__init__("macvio_node")
        self.profile = load_profile(str(self.declare_parameter("profile", "yamaha_thermal").value))
        topics, frames = self.profile.topics, self.profile.frames

        self.odometry_pub = self.create_publisher(Odometry, topics.odometry, 10)
        self.status_pub = self.create_publisher(String, topics.status, 10)
        self.runner = MacvioRunner(
            self.profile,
            now_ns=self._now_ns,
            on_odometry=self._publish_odometry,
            on_status=self._publish_status,
            log=self.get_logger().warning,
        )

        compressed = self.profile.image_transport == "compressed"
        image_type = CompressedImage if compressed else Image
        self._to_array = compressed_array if compressed else image_array
        self.create_subscription(Imu, topics.imu, self._on_imu, qos_profile_sensor_data)
        self.stereo_sync = ApproximateTimeSynchronizer(
            [
                Subscriber(self, image_type, topics.image_left, qos_profile=qos_profile_sensor_data),
                Subscriber(self, image_type, topics.image_right, qos_profile=qos_profile_sensor_data),
            ],
            queue_size=self.profile.stereo_queue_size,
            slop=self.profile.stereo_slop_s,
        )
        self.stereo_sync.registerCallback(self._on_stereo)
        self.create_timer(1.0 / STATUS_RATE_HZ, self._tick)
        self.runner.start()
        self.get_logger().info(
            f"MACVIO profile {self.profile.name}: {topics.image_left} + {topics.image_right} + {topics.imu} "
            f"-> {topics.odometry} ({frames.odom} -> {frames.base}); estimator {self.profile.macslam_config}"
        )

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_imu(self, message: Imu) -> None:
        self.runner.on_imu(*imu_sample(message))

    def _on_stereo(self, left: Any, right: Any) -> None:
        try:
            left_raw, right_raw = self._to_array(left), self._to_array(right)
        except ValueError as error:
            self.runner.on_bad_input(str(error))
            return
        self.runner.on_stereo(stamp_to_ns(left.header.stamp), left_raw, right_raw)

    def _tick(self) -> None:
        self.runner.tick(self._now_ns())

    def _publish_status(self, status: dict[str, Any]) -> None:
        self.status_pub.publish(status_message(status))

    def _publish_odometry(self, odometry: OdometryState) -> None:
        self.odometry_pub.publish(odometry_message(odometry, self.profile.frames.odom, self.profile.frames.base))

    def destroy_node(self) -> None:
        self.runner.shutdown()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MacvioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
