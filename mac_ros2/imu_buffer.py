"""Thread-safe IMU sample buffer that builds per-frame windows exactly as the offline loaders do."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from threading import Condition
from typing import Protocol, Sequence

import pypose as pp
import torch

from . import _macslam  # noqa: F401
from Src.DataLoader import IMUData  # noqa: E402
from Src.DataLoader.IMUWindow import imu_window  # noqa: E402
from Src.Utility.Math.IMUMath import noise_density_to_sample_covariance  # noqa: E402

class IMUSpecLike(Protocol):
    """IMU noise model and gap bracket of one rig; the body frame is the IMU frame."""

    @property
    def acc_noise_density(self) -> float: ...   # m/s^2/sqrt(Hz)
    @property
    def gyro_noise_density(self) -> float: ...  # rad/s/sqrt(Hz)
    @property
    def acc_random_walk(self) -> float: ...     # m/s^3/sqrt(Hz)
    @property
    def gyro_random_walk(self) -> float: ...    # rad/s^2/sqrt(Hz)
    @property
    def gravity(self) -> torch.Tensor | Sequence[float]: ...  # world frame, signed
    @property
    def max_bracket_ns(self) -> int: ...        # wider sample spacing around a frame time is an IMU gap


@dataclass(frozen=True)
class IMUSample:
    time_ns: int
    acc: tuple[float, ...]
    gyro: tuple[float, ...]


class IMUBuffer:
    """Bounded IMU history; `build_window` applies the live gap rule on top of `imu_window`."""

    def __init__(self, spec: IMUSpecLike, capacity: int, wait_timeout_s: float) -> None:
        self.spec = spec
        self.wait_timeout_s = wait_timeout_s
        self._samples: deque[IMUSample] = deque(maxlen=capacity)
        self._cond = Condition()
        self._closed = False
        self._T_BS = pp.identity_SE3(1, dtype=torch.float64)
        self._gravity = torch.as_tensor(spec.gravity, dtype=torch.float64).reshape(1, 3).clone()
        self.n_rejected = 0
        self.n_history_lost = 0
        self.n_wait_timeouts = 0

    def __len__(self) -> int:
        with self._cond:
            return len(self._samples)

    @property
    def newest_ns(self) -> int | None:
        with self._cond:
            return self._samples[-1].time_ns if self._samples else None

    def add(self, time_ns: int, acc: Sequence[float], gyro: Sequence[float]) -> bool:
        """Append one sample; non-finite or non-increasing samples are rejected and counted."""
        finite = all(math.isfinite(v) for v in (*acc, *gyro))
        with self._cond:
            accepted = finite and (not self._samples or time_ns > self._samples[-1].time_ns)
            if accepted:
                self._samples.append(IMUSample(time_ns, tuple(acc), tuple(gyro)))
                self._cond.notify_all()
            else:
                self.n_rejected += 1
        return accepted

    def close(self) -> None:
        """Release a pending wait; later windows no longer wait."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def build_window(self, t_prev_ns: int | None, t_cur_ns: int) -> IMUData | None:
        """IMU window of the frame at `t_cur_ns` after the processed frame at `t_prev_ns`.

        Returns None when no sample is at or before `t_cur_ns` (the frame precedes the IMU).
        """
        if t_prev_ns is not None and t_prev_ns >= t_cur_ns:
            raise ValueError(f"previous frame {t_prev_ns} must precede current frame {t_cur_ns}")
        with self._cond:
            if t_prev_ns is not None:
                self._trim(t_prev_ns)
            if not self._samples or self._samples[0].time_ns > t_cur_ns:
                return None
            # A stale newest sample means an IMU gap around t_cur: don't wait for it to close
            if t_cur_ns - self._samples[-1].time_ns <= self.spec.max_bracket_ns:
                arrived = self._cond.wait_for(
                    lambda: self._closed or self._samples[-1].time_ns > t_cur_ns, self.wait_timeout_s
                )
                if not arrived:
                    self.n_wait_timeouts += 1
            samples = list(self._samples)
            if t_prev_ns is not None and t_prev_ns < samples[0].time_ns:
                self.n_history_lost += 1
                t_prev_ns = None
        return self._make_imu(samples, t_prev_ns, t_cur_ns)

    def _trim(self, t_keep_ns: int) -> None:
        """Drop samples before the last one at or before `t_keep_ns`."""
        while len(self._samples) >= 2 and self._samples[1].time_ns <= t_keep_ns:
            self._samples.popleft()

    def _make_imu(self, samples: list[IMUSample], t_prev_ns: int | None, t_cur_ns: int) -> IMUData:
        t = torch.tensor([s.time_ns for s in samples], dtype=torch.int64)
        acc = torch.tensor([[s.acc for s in samples]], dtype=torch.float64)
        gyro = torch.tensor([[s.gyro for s in samples]], dtype=torch.float64)

        # Frames closer than the IMU period: integrate only the sample interpolated at t_cur
        if t_prev_ns is not None and not bool(((t > t_prev_ns) & (t <= t_cur_ns)).any()):
            t_prev_ns = None

        if t.numel() >= 2 and t_cur_ns <= int(t[-1]):
            window = imu_window(t, t_prev_ns, t_cur_ns, self.spec.max_bracket_ns)
            time_ns, dt_s = window.time_ns, window.dt_s
            acc_ws, gyro_ws = window.gather(acc), window.gather(gyro)
        else:
            time_ns, dt_s, raw = _gap_window(t, t_prev_ns, t_cur_ns)
            acc_ws, gyro_ws = acc[:, raw], gyro[:, raw]

        return IMUData(
            T_BS=self._T_BS,
            time_ns=time_ns,
            gravity=self._gravity.clone(),
            acc_ws=acc_ws,
            gyro_ws=gyro_ws,
            acc_cov=noise_density_to_sample_covariance(self.spec.acc_noise_density, dt_s[0], 3).unsqueeze(0),
            gyro_cov=noise_density_to_sample_covariance(self.spec.gyro_noise_density, dt_s[0], 3).unsqueeze(0),
            dt_s=dt_s,
            acc_bias=torch.zeros(1, 3, dtype=torch.float64),
            gyro_bias=torch.zeros(1, 3, dtype=torch.float64),
            acc_random_walk=torch.tensor([self.spec.acc_random_walk], dtype=torch.float64),
            gyro_random_walk=torch.tensor([self.spec.gyro_random_walk], dtype=torch.float64),
        )


def _gap_window(
    t: torch.Tensor, t_prev_ns: int | None, t_cur_ns: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw samples in (t_prev, t_cur) with the last one governing up to t_cur, as `imu_window` in a gap."""
    if t_prev_ns is None:
        raw = torch.empty(0, dtype=torch.int64)
    else:
        raw = torch.nonzero((t > t_prev_ns) & (t < t_cur_ns)).flatten()
    time_ns = t[raw]
    bounds = torch.cat([time_ns, torch.tensor([t_cur_ns], dtype=torch.int64)])
    return time_ns.view(1, -1, 1), (bounds.diff().double() / 1e9).unsqueeze(0), raw
