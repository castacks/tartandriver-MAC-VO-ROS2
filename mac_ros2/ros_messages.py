"""Conversions between ROS messages and the MACVIO core's plain types."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import String

from .image_decode import decode_compressed, decode_image
from .odometry_state import OdometryState


def stamp_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_stamp(time_ns: int) -> Time:
    return Time(sec=time_ns // 1_000_000_000, nanosec=time_ns % 1_000_000_000)


def imu_sample(message: Imu) -> tuple[int, tuple[float, float, float], tuple[float, float, float]]:
    """(stamp ns, acceleration, angular velocity) in the IMU message frame."""
    a, w = message.linear_acceleration, message.angular_velocity
    return stamp_to_ns(message.header.stamp), (a.x, a.y, a.z), (w.x, w.y, w.z)


def image_array(message: Image) -> np.ndarray:
    return decode_image(message.encoding, message.height, message.width, message.step,
                        bytes(message.data), bool(message.is_bigendian))


def compressed_array(message: CompressedImage) -> np.ndarray:
    return decode_compressed(bytes(message.data))


def odometry_message(odometry: OdometryState, frame_id: str, child_frame_id: str) -> Odometry:
    message = Odometry()
    message.header.stamp = ns_to_stamp(odometry.stamp_ns)
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id
    pose, twist = message.pose.pose, message.twist.twist
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in odometry.pose[:3])
    q = pose.orientation
    q.x, q.y, q.z, q.w = (float(v) for v in odometry.pose[3:7])
    twist.linear.x, twist.linear.y, twist.linear.z = (float(v) for v in odometry.twist[:3])
    twist.angular.x, twist.angular.y, twist.angular.z = (float(v) for v in odometry.twist[3:])
    message.pose.covariance = [float(v) for v in np.asarray(odometry.pose_cov).reshape(36)]
    message.twist.covariance = [float(v) for v in np.asarray(odometry.twist_cov).reshape(36)]
    return message


def odometry_state(message: Odometry) -> OdometryState:
    p, q = message.pose.pose.position, message.pose.pose.orientation
    v, w = message.twist.twist.linear, message.twist.twist.angular
    return OdometryState(
        stamp_ns=stamp_to_ns(message.header.stamp),
        pose=np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64),
        pose_cov=np.asarray(message.pose.covariance, dtype=np.float64).reshape(6, 6),
        twist=np.array([v.x, v.y, v.z, w.x, w.y, w.z], dtype=np.float64),
        twist_cov=np.asarray(message.twist.covariance, dtype=np.float64).reshape(6, 6),
    )


def transform_message(odometry: Odometry, child_frame_id: str) -> TransformStamped:
    """The odometry pose as a header.frame_id <- child_frame_id transform."""
    transform = TransformStamped()
    transform.header = odometry.header
    transform.child_frame_id = child_frame_id
    position = odometry.pose.pose.position
    transform.transform.translation.x = position.x
    transform.transform.translation.y = position.y
    transform.transform.translation.z = position.z
    transform.transform.rotation = odometry.pose.pose.orientation
    return transform


def status_message(status: Mapping[str, Any]) -> String:
    return String(data=json.dumps(status, sort_keys=True))


def parse_status(message: String) -> dict[str, Any] | None:
    try:
        status = json.loads(message.data)
    except json.JSONDecodeError:
        return None
    return status if isinstance(status, dict) else None


def diagnostic_status(diagnostic: Mapping[str, Any]) -> DiagnosticStatus:
    """`LocalizationGate.diagnostic` dict -> message."""
    return DiagnosticStatus(
        level=bytes([int(diagnostic["level"])]),
        name=str(diagnostic["name"]),
        message=str(diagnostic["message"]),
        hardware_id=str(diagnostic["hardware_id"]),
        values=[KeyValue(key=k, value=v) for k, v in diagnostic["values"].items()],
    )
