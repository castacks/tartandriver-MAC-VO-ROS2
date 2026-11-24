"""ROS-free MACVIO runner: stereo pairs and IMU samples in, odometry and node status out."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Condition, Event, Thread
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import _macslam  # noqa: F401
from Src.DataLoader import IDataTransform, IMUData, SIFrameData, StereoData, StereoRig, get_rig  # noqa: E402
from Src.Odometry.MACVIO import MACVIO  # noqa: E402
from Src.Utility.Config import load_config  # noqa: E402

from .imu_buffer import IMUBuffer
from .macvio_profile import MacvioProfile
from .odometry_state import OdometryState
from .output import frame_result, odometry_from_map
from .vio_state import VioState, VioStateMachine

PACKAGE_DIR = Path(__file__).resolve().parent


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


class FrameBuilder:
    """Decoded stereo pair + IMU window -> preprocessed `SIFrameData`, as the offline VIO sequences build it."""

    def __init__(self, rig: StereoRig, preprocess: Sequence[SimpleNamespace]) -> None:
        self.rig = rig
        self.K = rig.K.to(torch.float32).unsqueeze(0)
        self.camera_mask = rig.load_mask()
        self.transforms = [IDataTransform.instantiate(t.type, t.args) for t in preprocess]

    def build(self, index: int, time_ns: int, left_raw: np.ndarray, right_raw: np.ndarray, imu: IMUData) -> SIFrameData:
        imageL, imageR = self.rig.to_stereo(left_raw, right_raw)
        frame = SIFrameData(
            idx=torch.tensor([index], dtype=torch.long),
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
            imu=imu,
            gt_attitude=None,
        )
        for transform in self.transforms:
            frame = transform(frame)
        return frame


@dataclass(frozen=True)
class _Pair:
    time_ns: int
    left: np.ndarray
    right: np.ndarray


class MacvioRunner:
    """Feeds MACVIO from a bounded pair queue on a worker thread and reports through callbacks.

    `now_ns` is the ROS clock. Each processed frame emits its status before its odometry.
    """

    def __init__(
        self,
        profile: MacvioProfile,
        *,
        now_ns: Callable[[], int],
        on_odometry: Callable[[OdometryState], None],
        on_status: Callable[[dict[str, Any]], None],
        log: Callable[[str], None] = print,
        system: MACVIO | None = None,
    ) -> None:
        self.profile = profile
        self.rig = get_rig(profile.platform, profile.modality, profile.rect_mode)
        experiment = load_experiment(profile)
        self.frames = FrameBuilder(self.rig, getattr(experiment.Preprocess, self.rig.preprocess_key))
        self.imu = IMUBuffer(self.rig.imu, profile.imu_buffer_capacity, profile.imu_wait_timeout_s)
        self.state = VioStateMachine(self.rig.frame_rate_hz, int(profile.clock_silence_s * 1e9))
        self.system = system if system is not None else create_system(experiment)
        self._now_ns = now_ns
        self._on_odometry = on_odometry
        self._on_status = on_status
        self._log = log
        self._expected_dtype = np.uint8 if self.rig.norm == 255.0 else np.uint16

        self._queue: deque[_Pair] = deque()
        self._queue_cond = Condition()
        self._stop = Event()
        self._worker: Thread | None = None
        self._last_pair_ns: int | None = None
        self._t_prev_ns: int | None = None
        self._bad_streak = 0
        self.n_bad_inputs = 0
        self.n_before_imu = 0
        self.n_frames_run = 0

    def start(self) -> None:
        self._worker = Thread(target=self._work, name="macvio-worker", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        self._stop.set()
        self.imu.close()
        with self._queue_cond:
            self._queue_cond.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
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
        with self._queue_cond:
            if len(self._queue) >= self.profile.work_queue_size:
                self._queue.popleft()
                self.state.on_frame_dropped()
            self._queue.append(_Pair(time_ns, left_raw, right_raw))
            self._queue_cond.notify()

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
        status.update(
            n_bad_inputs=self.n_bad_inputs,
            n_before_imu=self.n_before_imu,
            imu_rejected=self.imu.n_rejected,
            imu_history_lost=self.imu.n_history_lost,
            imu_wait_timeouts=self.imu.n_wait_timeouts,
        )
        return status

    def step(self, timeout_s: float) -> bool:
        """Process the oldest queued pair; False when none arrived within `timeout_s`."""
        with self._queue_cond:
            if not self._queue:
                self._queue_cond.wait(timeout_s)
            if not self._queue:
                return False
            pair = self._queue.popleft()
        if self.state.state is not VioState.ERROR:
            self._process(pair)
        return True

    def _work(self) -> None:
        while not self._stop.is_set():
            self.step(0.1)

    def _process(self, pair: _Pair) -> None:
        try:
            imu = self.imu.build_window(self._t_prev_ns, pair.time_ns)
            if imu is None:
                self.n_before_imu += 1
                return
            frame = self.frames.build(self.n_frames_run, pair.time_ns, pair.left, pair.right, imu)
            self.system.run(frame)
            self._t_prev_ns = pair.time_ns
            self.n_frames_run += 1
            result = frame_result(self.system)
            odometry = None
            if result.initialized:
                odometry = odometry_from_map(self.system.Map, self.rig.T_body_base, self.profile.fd_twist_variance)
            now = self._now_ns()
            self.state.on_frame(result, now)
        except Exception as error:
            self._fail(f"{type(error).__name__}: {error}")
            return
        self._on_status(self.status(now))
        if odometry is not None:
            self._on_odometry(odometry)

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
