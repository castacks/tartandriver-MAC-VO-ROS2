"""Standalone ROS 2 bridge for running MACSLAM MACVIO in shadow mode."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Condition, Event, Lock, Thread
import time
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pypose as pp
import torch

import rclpy
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from .MessageFactory import from_image


MACSLAM_PATH = Path(__file__).resolve().parent / "mac_slam"
sys.path.insert(0, str(MACSLAM_PATH))

from Src.DataLoader import (  # noqa: E402
    CenterCropFrame,
    IMUData,
    SIFrameData,
    SmartResizeFrame,
    StereoData,
)
from Src.Odometry.MACVIO import MACVIO  # noqa: E402
from Src.Utility.Config import load_config  # noqa: E402
from Src.Utility.Math.IMUMath import (  # noqa: E402
    noise_density_to_sample_covariance,
)


def load_macvio_config(
    modality: str, config_file: str = ""
) -> tuple[SimpleNamespace, Path, Path | None]:
    """Load ROS calibration and the selected upstream estimator profile."""
    if modality not in {"thermal", "rgb"}:
        raise ValueError(
            f"Unsupported MACVIO modality {modality!r}; use thermal or rgb"
        )
    config_path = (
        Path(config_file).expanduser()
        if config_file
        else Path(__file__).resolve().parent / "config" / f"offroad_{modality}_vio.yaml"
    ).resolve()
    cfg, _ = load_config(config_path)
    experiment_path = None
    if hasattr(cfg, "macslam_config"):
        experiment_path = (config_path.parent / cfg.macslam_config).resolve()
        experiment, _ = load_config(experiment_path)
        cfg.Common = experiment.Common
        cfg.Odometry = experiment.Odometry
    # Explicit config_file overrides the modality default. Older custom configs
    # with inline Odometry and crop_height/crop_width remain supported.
    if not hasattr(cfg.ROS, "modality"):
        cfg.ROS.modality = modality
    if cfg.ROS.modality not in {"thermal", "rgb"}:
        raise ValueError(f"Unsupported config ROS.modality: {cfg.ROS.modality!r}")
    return cfg, config_path, experiment_path


def create_frame_transform(
    config: SimpleNamespace,
) -> CenterCropFrame | SmartResizeFrame:
    """Transform images, intrinsics, and occlusion mask together."""
    if not hasattr(config, "type"):
        return CenterCropFrame(
            {"height": int(config.crop_height), "width": int(config.crop_width)}
        )
    transforms = {
        "CenterCropFrame": CenterCropFrame,
        "SmartResizeFrame": SmartResizeFrame,
    }
    if config.type not in transforms:
        raise ValueError(f"Unsupported MACVIO image transform: {config.type!r}")
    transform = transforms[config.type]
    transform.is_valid_config(config.args)
    return transform(config.args)


@dataclass(frozen=True)
class ImuSample:
    time_ns: int
    acceleration: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    acceleration_covariance: tuple[float, ...]
    angular_velocity_covariance: tuple[float, ...]
    frame_id: str


class ImuContinuityError(RuntimeError):
    """The IMU stream cannot support preintegration to a camera frame."""

    def __init__(self, message: str, gap_sec: float) -> None:
        super().__init__(message)
        self.gap_sec = gap_sec


def validate_imu_window(
    samples: list[ImuSample],
    frame_ns: int,
    max_sample_gap_sec: float,
    max_camera_offset_sec: float,
) -> float:
    """Validate temporal coverage and return the largest observed gap."""
    if not samples:
        raise ImuContinuityError("IMU window is empty", float("inf"))

    max_sample_gap_ns = int(max_sample_gap_sec * 1e9)
    max_camera_offset_ns = int(max_camera_offset_sec * 1e9)
    largest_gap_ns = 0
    for previous, current in zip(samples, samples[1:]):
        gap_ns = current.time_ns - previous.time_ns
        if gap_ns <= 0:
            raise ImuContinuityError(
                "IMU timestamps are not strictly increasing: "
                f"{previous.time_ns} -> {current.time_ns}",
                gap_ns / 1e9,
            )
        largest_gap_ns = max(largest_gap_ns, gap_ns)
        if gap_ns > max_sample_gap_ns:
            raise ImuContinuityError(
                "IMU sample gap is too large: "
                f"{gap_ns / 1e9:.6f} s between {previous.time_ns} and "
                f"{current.time_ns} (limit {max_sample_gap_sec:.6f} s)",
                gap_ns / 1e9,
            )

    camera_offset_ns = frame_ns - samples[-1].time_ns
    if camera_offset_ns < 0:
        raise ImuContinuityError(
            f"IMU window ends after camera frame {frame_ns}",
            abs(camera_offset_ns) / 1e9,
        )
    if camera_offset_ns > max_camera_offset_ns:
        raise ImuContinuityError(
            "IMU does not cover camera frame: latest sample "
            f"{samples[-1].time_ns}, camera {frame_ns}, offset "
            f"{camera_offset_ns / 1e9:.6f} s "
            f"(limit {max_camera_offset_sec:.6f} s)",
            camera_offset_ns / 1e9,
        )
    return max(largest_gap_ns, camera_offset_ns) / 1e9


class MacvioNode(Node):
    """Run MACVIO from synchronized stereo and buffered high-rate IMU data."""

    def __init__(self) -> None:
        super().__init__("macvio_node")

        self.declare_parameter("modality", "thermal")
        self.declare_parameter("config_file", "")
        cfg, config_file, experiment_file = load_macvio_config(
            str(self.get_parameter("modality").value),
            str(self.get_parameter("config_file").value),
        )
        MACVIO.is_valid_config(cfg.Odometry)
        ros_cfg = cfg.ROS
        self.modality = str(ros_cfg.modality)
        self.config_file = str(config_file)
        self.experiment_file = str(experiment_file) if experiment_file else ""
        self.get_logger().info(
            f"MACVIO {self.modality}: ROS config={config_file}; "
            f"estimator config={experiment_file or config_file}"
        )

        self.odom_frame = str(ros_cfg.frames.odom)
        self.base_frame = str(ros_cfg.frames.base)
        self.expected_imu_frame = str(ros_cfg.frames.imu)
        self.publish_tf = bool(ros_cfg.output.publish_tf)
        self.publish_before_initialized = bool(
            ros_cfg.output.publish_before_initialized
        )
        self.imu_wait_timeout = float(ros_cfg.sync.imu_wait_timeout_sec)
        self.nominal_imu_dt = 1.0 / float(ros_cfg.imu.nominal_rate_hz)
        self.max_imu_sample_gap_sec = float(
            getattr(
                ros_cfg.sync,
                "max_imu_sample_gap_sec",
                3.0 * self.nominal_imu_dt,
            )
        )
        self.max_imu_camera_offset_sec = float(
            getattr(
                ros_cfg.sync,
                "max_imu_camera_offset_sec",
                3.0 * self.nominal_imu_dt,
            )
        )
        if (
            self.max_imu_sample_gap_sec <= 0.0
            or self.max_imu_camera_offset_sec <= 0.0
        ):
            raise ValueError("IMU continuity limits must be positive")
        self.use_message_covariance = bool(ros_cfg.imu.use_message_covariance)
        self.acc_noise_density = torch.tensor(
            ros_cfg.imu.acc_noise_density, dtype=torch.float64
        )
        self.gyro_noise_density = torch.tensor(
            ros_cfg.imu.gyro_noise_density, dtype=torch.float64
        )
        self.acc_random_walk = float(ros_cfg.imu.acc_random_walk)
        self.gyro_random_walk = float(ros_cfg.imu.gyro_random_walk)
        self.gravity = torch.tensor(
            [ros_cfg.imu.gravity], dtype=torch.float64
        )

        self.camera = cfg.Camera
        self.camera_t_bs = self._se3_from_config(
            ros_cfg.extrinsics.camera_t_bs
        )
        self.imu_t_bs = self._se3_from_config(ros_cfg.extrinsics.imu_t_bs)
        self.frame_transform = create_frame_transform(ros_cfg.preprocess)
        self.camera_mask = self._load_camera_mask(str(self.camera.mask))

        original_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            self.system = MACVIO.from_config(cfg)
        finally:
            os.chdir(original_cwd)

        self.odom_publisher = self.create_publisher(
            Odometry, str(ros_cfg.topics.odometry), 10
        )
        self.status_publisher = self.create_publisher(
            String, str(ros_cfg.topics.status), 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self._imu_lock = Lock()
        self._imu_condition = Condition(self._imu_lock)
        self._imu_buffer: deque[ImuSample] = deque(
            maxlen=int(ros_cfg.sync.imu_buffer_size)
        )
        self._imu_boundary: ImuSample | None = None
        self._last_frame_ns: int | None = None
        self._warned_imu_frame = False

        self._stereo_queue: Queue[tuple[Image, Image] | None] = Queue(
            maxsize=int(ros_cfg.sync.stereo_work_queue_size)
        )
        self._stop_event = Event()
        self._state_lock = Lock()
        self._state = "waiting_for_sensors"
        self._last_error = ""
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._frames_ignored_after_stop = 0
        self._last_stamp_ns = 0
        self._imu_gap_count = 0
        self._largest_imu_gap_sec = 0.0

        self.imu_subscription = self.create_subscription(
            Imu,
            str(ros_cfg.topics.imu),
            self._receive_imu,
            qos_profile_sensor_data,
        )
        left_subscriber = Subscriber(
            self,
            Image,
            str(ros_cfg.topics.image_left),
            qos_profile=qos_profile_sensor_data,
        )
        right_subscriber = Subscriber(
            self,
            Image,
            str(ros_cfg.topics.image_right),
            qos_profile=qos_profile_sensor_data,
        )
        self.stereo_sync = ApproximateTimeSynchronizer(
            [left_subscriber, right_subscriber],
            queue_size=int(ros_cfg.sync.stereo_queue_size),
            slop=float(ros_cfg.sync.stereo_slop_sec),
        )
        self.stereo_sync.registerCallback(self._receive_stereo)

        self.status_timer = self.create_timer(1.0, self._publish_status)
        self._worker = Thread(target=self._worker_loop, name="macvio-worker")
        self._worker.start()

        self.get_logger().info(
            "MACVIO shadow node: %s + %s + %s -> %s (%s -> %s)"
            % (
                ros_cfg.topics.image_left,
                ros_cfg.topics.image_right,
                ros_cfg.topics.imu,
                ros_cfg.topics.odometry,
                self.odom_frame,
                self.base_frame,
            )
        )

    @staticmethod
    def _se3_from_config(config: Any) -> pp.LieTensor:
        values = [*config.translation, *config.quaternion_xyzw]
        return pp.SE3(torch.tensor([values], dtype=torch.float64))

    @staticmethod
    def _stamp_to_ns(stamp: Any) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _load_camera_mask(relative_path: str) -> torch.Tensor:
        mask_path = MACSLAM_PATH / relative_path
        image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"unable to read camera mask {mask_path}")
        return torch.from_numpy(image == 0).unsqueeze(0).unsqueeze(0)

    def _receive_imu(self, message: Imu) -> None:
        sample = ImuSample(
            time_ns=self._stamp_to_ns(message.header.stamp),
            acceleration=(
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ),
            angular_velocity=(
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ),
            acceleration_covariance=tuple(
                message.linear_acceleration_covariance
            ),
            angular_velocity_covariance=tuple(
                message.angular_velocity_covariance
            ),
            frame_id=message.header.frame_id,
        )
        with self._imu_condition:
            if (
                self._imu_buffer
                and sample.time_ns <= self._imu_buffer[-1].time_ns
            ):
                return
            self._imu_buffer.append(sample)
            self._imu_condition.notify_all()

        if (
            sample.frame_id
            and sample.frame_id != self.expected_imu_frame
            and not self._warned_imu_frame
        ):
            self.get_logger().warning(
                "Expected IMU frame '%s', received '%s'; verify imu_t_bs"
                % (self.expected_imu_frame, sample.frame_id)
            )
            self._warned_imu_frame = True

    def _receive_stereo(self, left: Image, right: Image) -> None:
        with self._state_lock:
            if self._stop_event.is_set():
                self._frames_ignored_after_stop += 1
                return
            self._frames_received += 1
        pair = (left, right)
        try:
            self._stereo_queue.put_nowait(pair)
        except Full:
            try:
                self._stereo_queue.get_nowait()
            except Empty:
                pass
            self._stereo_queue.put_nowait(pair)
            with self._state_lock:
                self._frames_dropped += 1

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                pair = self._stereo_queue.get(timeout=0.1)
            except Empty:
                continue
            if pair is None:
                break
            # Keep ROS alive so the error remains visible on the status topic.
            try:
                self._process_stereo(*pair)
            except Exception as error:
                with self._state_lock:
                    self._state = "error"
                    self._last_error = f"{type(error).__name__}: {error}"
                    self._frames_dropped += 1
                    if isinstance(error, ImuContinuityError):
                        self._imu_gap_count += 1
                        self._largest_imu_gap_sec = max(
                            self._largest_imu_gap_sec, error.gap_sec
                        )
                self.get_logger().error("MACVIO frame failed: %s" % error)
                self._stop_event.set()

    def _process_stereo(self, left: Image, right: Image) -> None:
        frame_ns = self._stamp_to_ns(left.header.stamp)
        if self._last_frame_ns is not None and frame_ns <= self._last_frame_ns:
            raise ValueError("stereo timestamps must increase monotonically")

        samples = self._imu_samples_for_frame(frame_ns)
        if samples is None:
            with self._state_lock:
                self._state = "waiting_for_imu"
                self._last_error = ""
                self._frames_dropped += 1
            return

        left_image = self._image_to_tensor(left)
        right_image = self._image_to_tensor(right)
        if left_image.shape != right_image.shape:
            raise ValueError(
                "stereo image shapes differ: "
                f"{left_image.shape} vs {right_image.shape}"
            )
        if tuple(self.camera_mask.shape[-2:]) != tuple(left_image.shape[-2:]):
            raise ValueError(
                "camera mask/calibration and input image shapes differ: "
                f"{tuple(self.camera_mask.shape[-2:])} vs "
                f"{tuple(left_image.shape[-2:])}"
            )

        frame = self.frame_transform(
            SIFrameData(
                idx=torch.tensor([self._frames_processed], dtype=torch.long),
                time_ns=[frame_ns],
                stereo=StereoData(
                    data_ids=None,
                    T_BS=self.camera_t_bs.clone(),
                    K=torch.tensor(
                        [[
                            [self.camera.fx, 0.0, self.camera.cx],
                            [0.0, self.camera.fy, self.camera.cy],
                            [0.0, 0.0, 1.0],
                        ]],
                        dtype=torch.float32,
                    ),
                    baseline=torch.tensor(
                        [self.camera.bl], dtype=torch.float32
                    ),
                    time_ns=[frame_ns],
                    height=left_image.shape[1],
                    width=left_image.shape[2],
                    imageL=left_image.unsqueeze(0),
                    imageR=right_image.unsqueeze(0),
                    camera_mask=self.camera_mask.clone(),
                ),
                imu=self._make_imu_data(samples),
            )
        )

        self.system.run(frame)
        self._last_frame_ns = frame_ns
        self._advance_imu_boundary(samples[-1])
        with self._state_lock:
            self._frames_processed += 1
            self._last_stamp_ns = frame_ns
            self._last_error = ""
            if self.system.lost_track:
                self._state = "lost_track"
            elif self.system.imu_initcond is not None:
                self._state = "ready"
            else:
                self._state = "initializing"

        if (
            self.system.imu_initcond is not None
            or self.publish_before_initialized
        ):
            self._publish_odometry(left.header.stamp)
        self._publish_status()

    def _imu_samples_for_frame(self, frame_ns: int) -> list[ImuSample] | None:
        deadline = time.monotonic() + self.imu_wait_timeout
        with self._imu_condition:
            while (
                not self._stop_event.is_set()
                and (
                    not self._imu_buffer
                    or self._imu_buffer[-1].time_ns < frame_ns
                )
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._imu_condition.wait(timeout=remaining)

            available = [
                sample
                for sample in self._imu_buffer
                if sample.time_ns <= frame_ns
            ]
            if not available:
                return None

            if self._imu_boundary is None:
                return [available[-1]]

            if available[0].time_ns > self._imu_boundary.time_ns:
                raise BufferError(
                    "IMU buffer lost the previous camera boundary sample"
                )
            interval = [
                sample
                for sample in available
                if sample.time_ns >= self._imu_boundary.time_ns
            ]
            if len(interval) < 2:
                return None
            validate_imu_window(
                interval,
                frame_ns,
                self.max_imu_sample_gap_sec,
                self.max_imu_camera_offset_sec,
            )
            return interval

    def _advance_imu_boundary(self, boundary: ImuSample) -> None:
        with self._imu_condition:
            self._imu_boundary = boundary
            while (
                len(self._imu_buffer) > 1
                and self._imu_buffer[1].time_ns <= boundary.time_ns
            ):
                self._imu_buffer.popleft()

    def _make_imu_data(self, samples: list[ImuSample]) -> IMUData:
        timestamps = torch.tensor(
            [[[sample.time_ns] for sample in samples]], dtype=torch.long
        )
        acceleration = torch.tensor(
            [[sample.acceleration for sample in samples]], dtype=torch.float64
        )
        angular_velocity = torch.tensor(
            [[sample.angular_velocity for sample in samples]],
            dtype=torch.float64,
        )

        if len(samples) > 1:
            delta_t = (
                timestamps[0, 1:, 0] - timestamps[0, :-1, 0]
            ).double() / 1e9
            delta_t = torch.cat([delta_t, delta_t[-1:]])
        else:
            delta_t = torch.tensor([self.nominal_imu_dt], dtype=torch.float64)

        acceleration_covariance = self._sample_covariances(
            samples,
            "acceleration_covariance",
            self.acc_noise_density,
            delta_t,
        )
        angular_velocity_covariance = self._sample_covariances(
            samples,
            "angular_velocity_covariance",
            self.gyro_noise_density,
            delta_t,
        )

        return IMUData(
            T_BS=self.imu_t_bs.clone(),
            time_ns=timestamps,
            gravity=self.gravity.clone(),
            acc_ws=acceleration,
            gyro_ws=angular_velocity,
            acc_cov=acceleration_covariance.unsqueeze(0),
            gyro_cov=angular_velocity_covariance.unsqueeze(0),
            acc_bias=torch.zeros((1, 3), dtype=torch.float64),
            gyro_bias=torch.zeros((1, 3), dtype=torch.float64),
            acc_random_walk=torch.tensor(
                [self.acc_random_walk], dtype=torch.float64
            ),
            gyro_random_walk=torch.tensor(
                [self.gyro_random_walk], dtype=torch.float64
            ),
        )

    def _sample_covariances(
        self,
        samples: list[ImuSample],
        field: str,
        noise_density: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        fallback = noise_density_to_sample_covariance(
            noise_density, delta_t, dim=3
        )
        if not self.use_message_covariance:
            return fallback

        result = fallback.clone()
        for index, sample in enumerate(samples):
            values = np.asarray(getattr(sample, field), dtype=np.float64)
            if (
                values.shape == (9,)
                and values[0] >= 0.0
                and np.isfinite(values).all()
                and np.any(np.diag(values.reshape(3, 3)) > 0.0)
            ):
                result[index] = torch.from_numpy(values.reshape(3, 3))
        return result

    @staticmethod
    def _image_to_tensor(message: Image) -> torch.Tensor:
        if message.encoding.lower().startswith("bayer_"):
            raise ValueError("MACVIO requires rectified images, not raw Bayer images")
        image = from_image(message)
        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
        else:
            image = image.astype(np.float32)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] < 3:
            raise ValueError(f"unsupported image shape {image.shape}")
        else:
            image = image[:, :, :3]
            if message.encoding.lower().startswith("bgr"):
                image = image[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)

    def _publish_odometry(self, stamp: Any) -> None:
        map_data = self.system.Map
        pose = map_data.body.data["pose"][-1].detach().cpu().double()
        pose_covariance = (
            map_data.body.data["Tw_cov_pose"][-1].detach().cpu().double()
        )
        velocity_sensor = (
            map_data.imu.data["v_Tbs"][-1].detach().cpu().double()
        )
        angular_velocity = (
            map_data.imu.data["\u03c9_Tbs"][-1].detach().cpu().double()
        )
        imu_t_bs = pp.SE3(
            map_data.imu.data["T_BS"][-1].detach().cpu().double()
        )
        velocity_body = velocity_sensor - torch.cross(
            angular_velocity, imu_t_bs.translation(), dim=0
        )

        if not all(
            torch.isfinite(value).all()
            for value in (
                pose,
                pose_covariance,
                velocity_body,
                angular_velocity,
            )
        ):
            raise ValueError("MACVIO produced non-finite odometry")

        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = float(pose[0])
        message.pose.pose.position.y = float(pose[1])
        message.pose.pose.position.z = float(pose[2])
        message.pose.pose.orientation.x = float(pose[3])
        message.pose.pose.orientation.y = float(pose[4])
        message.pose.pose.orientation.z = float(pose[5])
        message.pose.pose.orientation.w = float(pose[6])
        message.pose.covariance = pose_covariance.reshape(-1).tolist()
        message.twist.twist.linear.x = float(velocity_body[0])
        message.twist.twist.linear.y = float(velocity_body[1])
        message.twist.twist.linear.z = float(velocity_body[2])
        message.twist.twist.angular.x = float(angular_velocity[0])
        message.twist.twist.angular.y = float(angular_velocity[1])
        message.twist.twist.angular.z = float(angular_velocity[2])

        twist_covariance = torch.zeros((6, 6), dtype=torch.float64)
        twist_covariance[:3, :3] = (
            map_data.imu.data["v_covTb"][-1].detach().cpu().double()
        )
        message.twist.covariance = twist_covariance.reshape(-1).tolist()
        self.odom_publisher.publish(message)

        if self.publish_tf:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = message.pose.pose.position.x
            transform.transform.translation.y = message.pose.pose.position.y
            transform.transform.translation.z = message.pose.pose.position.z
            transform.transform.rotation = message.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def _publish_status(self) -> None:
        with self._state_lock:
            payload = {
                "modality": self.modality,
                "config_file": self.config_file,
                "experiment_file": self.experiment_file,
                "state": self._state,
                "initialized": bool(self.system.imu_initcond is not None),
                "lost_track": bool(self.system.lost_track),
                "frames_received": self._frames_received,
                "frames_processed": self._frames_processed,
                "frames_dropped": self._frames_dropped,
                "frames_ignored_after_stop": self._frames_ignored_after_stop,
                "imu_gap_count": self._imu_gap_count,
                "largest_imu_gap_sec": self._largest_imu_gap_sec,
                "last_stamp_ns": self._last_stamp_ns,
                "last_error": self._last_error,
            }
        with self._imu_lock:
            payload["imu_buffered"] = len(self._imu_buffer)
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.status_publisher.publish(message)

    def destroy_node(self) -> bool:
        self._stop_event.set()
        with self._imu_condition:
            self._imu_condition.notify_all()
        try:
            self._stereo_queue.put_nowait(None)
        except Full:
            try:
                self._stereo_queue.get_nowait()
            except Empty:
                pass
            self._stereo_queue.put_nowait(None)
        self._worker.join(timeout=5.0)
        self.system.terminate()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MacvioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
