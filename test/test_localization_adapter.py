"""Check that the localization adapter survives MACVIO resets and blind faults."""

import json
from unittest.mock import Mock

import pytest
import rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from mac_ros2.localization_adapter_node import LocalizationAdapter


def stamped_odometry(stamp_ns, x):
    message = Odometry()
    message.pose.pose.position.x = x
    message.pose.pose.orientation.w = 1.0
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    return message


def macvio_status(state, reset_count, frame_stamp_ns):
    return String(data=json.dumps({
        "state": state, "reset_count": reset_count,
        "last_reset_frame_stamp_ns": frame_stamp_ns,
    }))


def test_adapter_realigns_after_macvio_reset_and_drops_stale_odometry():
    rclpy.init(args=["--ros-args", "-p", "pose_source:=macvio", "-p", "publish_tf:=false"])
    adapter = None
    try:
        adapter = LocalizationAdapter()
        adapter.odometry_publisher = Mock()
        published = adapter.odometry_publisher.publish
        second = 1_000_000_000
        adapter._receive_macvio_status(macvio_status("TRACKING", 0, 0))
        adapter._receive_super_odometry(stamped_odometry(1 * second, 10.0))
        adapter._receive_macvio(stamped_odometry(1 * second, 0.0))
        assert published.call_args.args[0].pose.pose.position.x == pytest.approx(10.0)

        adapter._receive_macvio_status(macvio_status("BLIND_FAULT", 0, 0))
        adapter._receive_macvio(stamped_odometry(2 * second, 1.0))
        assert published.call_count == 1 and adapter.state == "source_blind_fault"

        adapter._receive_macvio_status(macvio_status("REINITIALIZING", 1, 2 * second))
        adapter._receive_macvio_status(macvio_status("TRACKING", 1, 2 * second))
        adapter._receive_macvio(stamped_odometry(2 * second, 1.0))
        assert published.call_count == 1 and adapter.state == "aligning"

        adapter._receive_super_odometry(stamped_odometry(8 * second, 50.0))
        adapter._receive_macvio(stamped_odometry(8 * second, 0.0))
        adapter._receive_macvio(stamped_odometry(9 * second, 2.0))
        assert published.call_count == 3 and adapter.state == "ready"
        assert published.call_args.args[0].pose.pose.position.x == pytest.approx(52.0)

        adapter._receive_macvio_status(macvio_status("ERROR", 1, 2 * second))
        adapter._receive_macvio_status(macvio_status("TRACKING", 1, 2 * second))
        adapter._receive_macvio(stamped_odometry(10 * second, 3.0))
        assert published.call_count == 3 and adapter.state == "source_error"
    finally:
        if adapter is not None:
            adapter.destroy_node()
        rclpy.shutdown()
