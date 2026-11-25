"""Conversions between ROS messages and the MACVIO core's plain types."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointField, PointCloud2
from std_msgs.msg import String

from .image_decode import decode_compressed, decode_image
from .odometry_state import MAP_POINT_FIELDS, MapPoints, OdometryState


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


def static_rotation_message(parent: str, child: str, rotation_xyzw: np.ndarray) -> TransformStamped:
    """A pure-rotation parent <- child transform, stamped at time zero as static TFs are."""
    transform = TransformStamped()
    transform.header.frame_id = parent
    transform.child_frame_id = child
    q = transform.transform.rotation
    q.x, q.y, q.z, q.w = (float(v) for v in rotation_xyzw)
    return transform


def point_cloud_message(points: MapPoints, frame_id: str) -> PointCloud2:
    """Map points -> PointCloud2 with one FLOAT32 field per `MAP_POINT_FIELDS` column."""
    data = np.ascontiguousarray(points.values, dtype="<f4")
    assert data.ndim == 2 and data.shape[1] == len(MAP_POINT_FIELDS), f"map points are {data.shape}"
    message = PointCloud2(
        height=1,
        width=data.shape[0],
        fields=[PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, name in enumerate(MAP_POINT_FIELDS)],
        is_bigendian=False,
        point_step=data.itemsize * data.shape[1],
        row_step=data.nbytes,
        data=data.tobytes(),
        is_dense=True,
    )
    message.header.stamp = ns_to_stamp(points.stamp_ns)
    message.header.frame_id = frame_id
    return message


def mono8_message(stamp_ns: int, frame_id: str, image: np.ndarray) -> Image:
    assert image.dtype == np.uint8 and image.ndim == 2, f"mono8 needs an HxW uint8 image, got {image.dtype} {image.shape}"
    message = Image(height=image.shape[0], width=image.shape[1], encoding="mono8", is_bigendian=0,
                    step=image.shape[1], data=np.ascontiguousarray(image).tobytes())
    message.header.stamp = ns_to_stamp(stamp_ns)
    message.header.frame_id = frame_id
    return message


def camera_info_message(stamp_ns: int, frame_id: str, width: int, height: int, K: np.ndarray) -> CameraInfo:
    """Pinhole camera_info of a rectified, undistorted image."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    message = CameraInfo(
        width=width,
        height=height,
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        k=[float(v) for v in K.reshape(9)],
        r=[float(v) for v in np.eye(3).reshape(9)],
        p=[float(v) for v in np.hstack([K, np.zeros((3, 1))]).reshape(12)],
    )
    message.header.stamp = ns_to_stamp(stamp_ns)
    message.header.frame_id = frame_id
    return message


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
