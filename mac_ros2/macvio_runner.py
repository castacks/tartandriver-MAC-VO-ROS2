"""ROS-free MACVIO runner: stereo pairs and IMU samples in; odometry, map points, processed images and status out."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Condition, Event, Thread
from types import SimpleNamespace
import time
from typing import Any, Callable, Generic, Sequence, TypeVar

import numpy as np
import torch

from . import _macslam  # noqa: F401
from Src.DataLoader import IDataTransform, IMUData, SIFrameData, StereoData, StereoFrameData
from Src.DataLoader.Dataset.OffroadRig import StereoRig, get_rig
from Src.Odometry.MACVIO import MACVIO
from Src.Utility.Config import load_config

from .imu_buffer import IMUBuffer
from .macvio_profile import MacvioProfile
from .odometry_state import MapPoints, OdometryState
from .output import frame_result, map_points_from_map, odometry_from_map, processed_image, rect_rotation
from .vio_state import VioState, VioStateMachine

PACKAGE_DIR = Path(__file__).resolve().parent
_T = TypeVar("_T")


def load_experiment(profile: MacvioProfile) -> SimpleNamespace:
    """The mac_slam experiment config named by the profile (Odometry and Preprocess)."""
    config, _ = load_config(profile.macslam_config)
    MACVIO.is_valid_config(config.Odometry)
    return config


def create_system(experiment: SimpleNamespace) -> MACVIO:
    original_cwd = os.getcwd()
    try:
        os.chdir(PACKAGE_DIR)
        return MACVIO.from_config(experiment)
    finally:
        os.chdir(original_cwd)


IMAGE_TRANSFORMS = frozenset({"SmartResizeFrame", "CastDevice", "ThermalPreprocess"})
IMU_TRANSFORMS = frozenset({"IMUSpecOverrideTransform"})


def split_preprocess(preprocess: Sequence[SimpleNamespace]) -> tuple[list[IDataTransform], list[IDataTransform]]:
    """A `Preprocess.yaml` chain -> (image transforms, IMU transforms), each in chain order."""
    image: list[IDataTransform] = []
    imu: list[IDataTransform] = []
    for t in preprocess:
        if t.type in IMAGE_TRANSFORMS:
            image.append(IDataTransform.instantiate(t.type, t.args))
        elif t.type in IMU_TRANSFORMS:
            imu.append(IDataTransform.instantiate(t.type, t.args))
        else:
            raise ValueError(f"preprocess transform {t.type} is neither a known image nor IMU transform")
    return image, imu


class FrameBuilder:
    """Builds `SIFrameData` as the offline VIO sequences do, in two stages: stereo first, IMU later.

    The chain's image and IMU transforms touch disjoint fields, so splitting them leaves the frame unchanged.
    """

    def __init__(self, rig: StereoRig, preprocess: Sequence[SimpleNamespace]) -> None:
        self.rig = rig
        self.K = rig.K.to(torch.float32).unsqueeze(0)
        self.camera_mask = rig.load_mask()
        self.image_transforms, self.imu_transforms = split_preprocess(preprocess)

    def preprocess(self, time_ns: int, left_raw: np.ndarray, right_raw: np.ndarray) -> StereoData:
        """Decoded pair -> rectified stereo through the image transforms."""
        imageL, imageR = self.rig.to_stereo(left_raw, right_raw)
        frame = StereoFrameData(
            idx=torch.tensor([0], dtype=torch.long),  # set per processed frame in `attach_imu`
            time_ns=[time_ns],
            gt_pose=None,
            stereo=StereoData(
                data_ids=None,
                T_BS=self.rig.T_BS,
                K=self.K,
                baseline=torch.tensor([self.rig.baseline]),
                width=self.rig.width,
                height=self.rig.height,
                time_ns=[time_ns],
                imageL=imageL,
                imageR=imageR,
                camera_mask=self.camera_mask,
            ),
        )
        for transform in self.image_transforms:
            frame = transform(frame)
        return frame.stereo

    def attach_imu(self, index: int, stereo: StereoData, imu: IMUData) -> SIFrameData:
        """Preprocessed stereo + IMU window -> the frame `MACVIO.run` takes, through the IMU transforms."""
        frame = SIFrameData(idx=torch.tensor([index], dtype=torch.long), time_ns=list(stereo.time_ns), gt_pose=None,
                            stereo=stereo, imu=imu, gt_attitude=None)
        for transform in self.imu_transforms:
            frame = transform(frame)
        return frame


@dataclass(frozen=True)
class _Pair:
    time_ns: int
    left: np.ndarray
    right: np.ndarray


class _DropOldestQueue(Generic[_T]):
    """Bounded FIFO that drops its oldest item when full."""

    def __init__(self, size: int) -> None:
        self._items: deque[_T] = deque()
        self._size = size
        self._cond = Condition()

    def put(self, item: _T) -> bool:
        """Enqueue `item`; True when the oldest item was dropped to make room."""
        with self._cond:
            dropped = len(self._items) >= self._size
            if dropped:
                self._items.popleft()
            self._items.append(item)
            self._cond.notify()
            return dropped

    def get(self, timeout_s: float) -> _T | None:
        with self._cond:
            if not self._items:
                self._cond.wait(timeout_s)
            return self._items.popleft() if self._items else None

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


class MacvioRunner:
    """Two worker stages: the front stage preprocesses every received pair and reports the processed
    left image; the VIO stage runs MACVIO on the newest preprocessed frames.

    `now_ns` is the ROS clock. Each processed frame emits its status, then, once initialized, its odometry
    and its map points.
    """

    def __init__(
        self,
        profile: MacvioProfile,
        *,
        now_ns: Callable[[], int],
        on_odometry: Callable[[OdometryState], None],
        on_status: Callable[[dict[str, Any]], None],
        on_map_points: Callable[[MapPoints], None],
        on_processed_image: Callable[[int, np.ndarray, np.ndarray], None],
        log: Callable[[str], None] = print,
        system: MACVIO | None = None,
    ) -> None:
        self.profile = profile
        self.rig = get_rig(profile.platform, profile.modality, profile.rect_mode)
        # (x, y, z, w) of the raw <- rectified optical frame, for the processed image's static TF
        self.rect_rotation = rect_rotation(self.rig) if profile.frames.rect_tf_parent is not None else None
        experiment = load_experiment(profile)
        self.frames = FrameBuilder(self.rig, getattr(experiment.Preprocess, self.rig.preprocess_key))
        self.imu = IMUBuffer(self.rig.imu, profile.imu_buffer_capacity, profile.imu_wait_timeout_s)
        self.state = VioStateMachine(self.rig.frame_rate_hz, int(profile.clock_silence_s * 1e9))
        self.system = system if system is not None else create_system(experiment)
        self._now_ns = now_ns
        self._on_odometry = on_odometry
        self._on_status = on_status
        self._on_map_points = on_map_points
        self._on_processed_image = on_processed_image
        self._log = log
        self._expected_dtype = np.uint8 if self.rig.norm == 255.0 else np.uint16

        self._pairs: _DropOldestQueue[_Pair] = _DropOldestQueue(profile.front_queue_size)
        self._preprocessed: _DropOldestQueue[StereoData] = _DropOldestQueue(profile.work_queue_size)
        self._stop = Event()
        self._threads: list[Thread] = []
        self._last_pair_ns: int | None = None
        self._t_prev_ns: int | None = None
        self._bad_streak = 0
        self.n_bad_inputs = 0
        self.n_before_imu = 0
        self.n_frames_run = 0
        self.n_preprocessed = 0
        self.n_front_dropped = 0
        self.front_latency_s: float | None = None
        self._front_latency_sum_s = 0.0

    def start(self) -> None:
        self._threads = [
            Thread(target=self._loop, args=(self.front_step,), name="macvio-front", daemon=True),
            Thread(target=self._loop, args=(self.step,), name="macvio-vio", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        self.imu.close()
        self._pairs.wake()
        self._preprocessed.wake()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self.system.terminate()

    def on_imu(self, time_ns: int, acc: Sequence[float], gyro: Sequence[float]) -> None:
        if self.imu.add(time_ns, acc, gyro):
            self.state.on_imu(time_ns, self._now_ns())

    def on_stereo(self, time_ns: int, left_raw: np.ndarray, right_raw: np.ndarray) -> None:
        problem = self._check_pair(time_ns, left_raw, right_raw)
        if problem is not None:
            self.on_bad_input(problem)
            return
        self._bad_streak = 0
        self._last_pair_ns = time_ns
        self.state.on_pair(time_ns, self._now_ns())
        if self.state.state is VioState.ERROR:
            return
        if self._pairs.put(_Pair(time_ns, left_raw, right_raw)):
            self.n_front_dropped += 1

    def on_bad_input(self, problem: str) -> None:
        """Count an unusable stereo pair; persistent ones are fatal."""
        self.n_bad_inputs += 1
        self._bad_streak += 1
        if self._bad_streak == 1:
            self._log(f"MACVIO dropped a stereo pair: {problem}")
        if self._bad_streak == self.profile.max_consecutive_bad_inputs:
            self._fail(f"{self._bad_streak} consecutive bad stereo pairs, last: {problem}")

    def tick(self, ros_now_ns: int) -> None:
        self._on_status(self.status(ros_now_ns))

    def status(self, ros_now_ns: int) -> dict[str, Any]:
        status = self.state.status(ros_now_ns)
        n_preprocessed = self.n_preprocessed
        status.update(
            n_bad_inputs=self.n_bad_inputs,
            n_before_imu=self.n_before_imu,
            n_preprocessed=n_preprocessed,
            n_front_dropped=self.n_front_dropped,
            front_latency_s=self.front_latency_s,
            front_latency_mean_s=self._front_latency_sum_s / n_preprocessed if n_preprocessed else None,
            imu_rejected=self.imu.n_rejected,
            imu_history_lost=self.imu.n_history_lost,
            imu_wait_timeouts=self.imu.n_wait_timeouts,
        )
        return status

    def front_step(self, timeout_s: float) -> bool:
        """Preprocess the oldest received pair; False when none arrived within `timeout_s`."""
        pair = self._pairs.get(timeout_s)
        if pair is None:
            return False
        if self.state.state is not VioState.ERROR:
            self._preprocess(pair)
        return True

    def step(self, timeout_s: float) -> bool:
        """Run MACVIO on the oldest preprocessed frame; False when none arrived within `timeout_s`."""
        stereo = self._preprocessed.get(timeout_s)
        if stereo is None:
            return False
        if self.state.state is not VioState.ERROR:
            self._process(stereo)
        return True

    def _loop(self, step: Callable[[float], bool]) -> None:
        while not self._stop.is_set():
            step(0.1)

    def _preprocess(self, pair: _Pair) -> None:
        start = time.monotonic()
        try:
            stereo = self.frames.preprocess(pair.time_ns, pair.left, pair.right)
            image = processed_image(stereo.imageL) if self.profile.publish_processed_image else None
        except Exception as error:
            self._fail(f"preprocessing: {type(error).__name__}: {error}")
            return
        if image is not None:
            self._on_processed_image(pair.time_ns, image, stereo.K[0].detach().cpu().double().numpy())
        latency_s = time.monotonic() - start
        self.front_latency_s = latency_s
        self._front_latency_sum_s += latency_s
        self.n_preprocessed += 1
        if self._preprocessed.put(stereo):
            self.state.on_frame_dropped()

    def _process(self, stereo: StereoData) -> None:
        time_ns = stereo.time_ns[0]
        try:
            imu = self.imu.build_window(self._t_prev_ns, time_ns)
            if imu is None:
                self.n_before_imu += 1
                return
            self.system.run(self.frames.attach_imu(self.n_frames_run, stereo, imu))
            self._t_prev_ns = time_ns
            self.n_frames_run += 1
            result = frame_result(self.system)
            odometry = map_points = None
            if result.initialized:
                odometry = odometry_from_map(self.system.Map, self.rig.T_body_base, self.profile.fd_twist_variance)
                map_points = map_points_from_map(self.system.Map, self.rig.T_body_base)
            now = self._now_ns()
            self.state.on_frame(result, now)
        except Exception as error:
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._on_status(self.status(now))
        if odometry is not None and map_points is not None:
            self._on_odometry(odometry)
            self._on_map_points(map_points)

    def _check_pair(self, time_ns: int, left: np.ndarray, right: np.ndarray) -> str | None:
        if self._last_pair_ns is not None and time_ns <= self._last_pair_ns:
            return f"stereo stamp {time_ns} does not follow {self._last_pair_ns}"
        expected = (self.rig.height, self.rig.width)
        for side, image in (("left", left), ("right", right)):
            if image.shape[:2] != expected or image.ndim not in (2, 3):
                return f"{side} image is {image.shape}, expected {expected}"
            if image.dtype != self._expected_dtype:
                return f"{side} image is {image.dtype}, expected {np.dtype(self._expected_dtype)}"
        if left.shape != right.shape:
            return f"stereo shapes differ: {left.shape} vs {right.shape}"
        return None

    def _fail(self, error: str) -> None:
        now = self._now_ns()
        self.state.fail(error, now)
        self._log(f"MACVIO failed: {error}")
        self._on_status(self.status(now))
