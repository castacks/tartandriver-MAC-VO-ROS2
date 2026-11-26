"""ROS 2 shell around `MacvioRunner`: stereo + IMU in; raw odometry, map points, processed image and status out."""

from __future__ import annotations

from typing import Any

import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster

from .macvio_profile import load_profile
from .macvio_runner import MacvioRunner
from .odometry_state import MapPoints, OdometryState
from .ros_messages import (
    camera_info_message,
    compressed_array,
    image_array,
    imu_sample,
    mono8_message,
    odometry_message,
    point_cloud_message,
    stamp_to_ns,
    static_rotation_message,
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
        self.map_points_pub = self.create_publisher(PointCloud2, topics.map_points, 10)
        self.image_pub = self.camera_info_pub = None
        if self.profile.publish_processed_image:
            self.image_pub = self.create_publisher(Image, topics.processed_image, 10)
            self.camera_info_pub = self.create_publisher(CameraInfo, topics.processed_camera_info, 10)
        self.runner = MacvioRunner(
            self.profile,
            now_ns=self._now_ns,
            on_odometry=self._publish_odometry,
            on_status=self._publish_status,
            on_map_points=self._publish_map_points,
            on_processed_image=self._publish_processed_image,
            log=self.get_logger().warning,
        )
        self.static_tf_broadcaster: StaticTransformBroadcaster | None = None
        if frames.rect_tf_parent is not None and frames.processed_image is not None:
            assert self.runner.rect_rotation is not None
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            self.static_tf_broadcaster.sendTransform(
                static_rotation_message(frames.rect_tf_parent, frames.processed_image, self.runner.rect_rotation)
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
        outputs = f"{topics.odometry} ({frames.odom} -> {frames.base}), {topics.map_points}"
        if self.image_pub is not None:
            outputs += f", {topics.processed_image} + {topics.processed_camera_info} ({frames.processed_image})"
        self.get_logger().info(
            f"MACVIO profile {self.profile.name}: {topics.image_left} + {topics.image_right} + {topics.imu} "
            f"-> {outputs}; estimator {self.profile.macslam_config}"
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

    def _publish_map_points(self, points: MapPoints) -> None:
        self.map_points_pub.publish(point_cloud_message(points, self.profile.frames.base))

    def _publish_processed_image(self, stamp_ns: int, image: np.ndarray, K: np.ndarray) -> None:
        frame = self.profile.frames.processed_image
        assert self.image_pub is not None and self.camera_info_pub is not None and frame is not None
        height, width = image.shape
        self.image_pub.publish(mono8_message(stamp_ns, frame, image))
        self.camera_info_pub.publish(camera_info_message(stamp_ns, frame, width, height, K))

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
