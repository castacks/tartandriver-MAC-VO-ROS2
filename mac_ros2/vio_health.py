"""ROS-free liveness logic shared by the MACVIO node and the localization adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class VioState(str, Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    BLIND = "BLIND"
    BLIND_FAULT = "BLIND_FAULT"
    REINITIALIZING = "REINITIALIZING"
    ERROR = "ERROR"


class WatchdogEvent(Enum):
    NONE = "none"
    BLIND_WARN = "blind_warn"
    BLIND_FAULT = "blind_fault"


@dataclass(frozen=True)
class HealthConfig:
    watchdog_rate_hz: float
    blind_warn_s: float
    max_blind_s: float
    reinit_timeout_s: float

    @classmethod
    def from_config(cls, config: Any) -> HealthConfig:
        """Read ROS.health; missing fields raise AttributeError."""
        result = cls(
            watchdog_rate_hz=float(config.watchdog_rate_hz),
            blind_warn_s=float(config.blind_warn_s),
            max_blind_s=float(config.max_blind_s),
            reinit_timeout_s=float(config.reinit_timeout_s),
        )
        if result.watchdog_rate_hz <= 0.0 or result.reinit_timeout_s <= 0.0:
            raise ValueError("watchdog_rate_hz and reinit_timeout_s must be positive")
        if not 0.0 < result.blind_warn_s < result.max_blind_s:
            raise ValueError("health limits require 0 < blind_warn_s < max_blind_s")
        return result


class HealthMonitor:
    """Track sensor ages, the blind latch, resets and the published VIO state.

    All times are node-clock nanoseconds. Not thread-safe; the caller holds a lock.
    """

    def __init__(self, config: HealthConfig, now_ns: int) -> None:
        self.config = config
        self._start_ns = now_ns
        self._last_imu_ns: int | None = None
        self._last_cam_ns: int | None = None
        self._base = VioState.INITIALIZING
        self._fault = False
        self._error = False
        self._instance_start_ns = now_ns
        self._instance_frames = 0
        self.reset_count = 0
        self.last_reset_reason = ""
        self.last_reset_stamp_ns = 0
        self.last_reset_frame_stamp_ns = 0
        self.last_reset_latency_ms = 0.0

    @property
    def blind_fault(self) -> bool:
        return self._fault

    @property
    def failed(self) -> bool:
        return self._error

    def record_imu(self, now_ns: int) -> None:
        self._last_imu_ns = now_ns

    def record_camera(self, now_ns: int) -> None:
        self._last_cam_ns = now_ns

    def ages_s(self, now_ns: int) -> tuple[float | None, float | None]:
        """Seconds since the last IMU and stereo message; None if never received."""
        return (
            None if self._last_imu_ns is None else (now_ns - self._last_imu_ns) / 1e9,
            None if self._last_cam_ns is None else (now_ns - self._last_cam_ns) / 1e9,
        )

    def silence_s(self, now_ns: int) -> float:
        """Seconds both sensors have been silent; never-received counts from start."""
        stamps = [
            self._start_ns if stamp is None else stamp
            for stamp in (self._last_imu_ns, self._last_cam_ns)
        ]
        return (now_ns - max(stamps)) / 1e9

    def state(self, now_ns: int) -> VioState:
        if self._error:
            return VioState.ERROR
        if self._fault:
            return VioState.BLIND_FAULT
        if self._base == VioState.REINITIALIZING:
            return self._base
        if self.silence_s(now_ns) > self.config.blind_warn_s:
            return VioState.BLIND
        return self._base

    def publishing_allowed(self) -> bool:
        return not (
            self._error or self._fault or self._base == VioState.REINITIALIZING
        )

    def update_watchdog(self, now_ns: int) -> WatchdogEvent:
        """Latch BLIND_FAULT once; it arms only after this instance consumed a frame."""
        if self._error or self._fault:
            return WatchdogEvent.NONE
        silence = self.silence_s(now_ns)
        if silence > self.config.max_blind_s and self._instance_frames > 0:
            self._fault = True
            return WatchdogEvent.BLIND_FAULT
        if silence > self.config.blind_warn_s:
            return WatchdogEvent.BLIND_WARN
        return WatchdogEvent.NONE

    def pending_reset(self, now_ns: int) -> str | None:
        """Reason a new MACVIO instance is required now, if any."""
        if self._error or self._base == VioState.REINITIALIZING:
            return None
        if self._fault:
            imu_age, cam_age = self.ages_s(now_ns)
            if (
                imu_age is not None
                and cam_age is not None
                and max(imu_age, cam_age) <= self.config.blind_warn_s
            ):
                return "blind_fault_recovered"
            return None
        if (
            self.reset_count > 0
            and self._base == VioState.INITIALIZING
            and self._instance_frames > 0
            and (now_ns - self._instance_start_ns) / 1e9 > self.config.reinit_timeout_s
        ):
            return "reinit_timeout"
        return None

    def begin_reset(self, reason: str, now_ns: int, last_frame_stamp_ns: int) -> None:
        """Retire the current instance; last_frame_stamp_ns is its newest frame."""
        self.reset_count += 1
        self.last_reset_reason = reason
        self.last_reset_stamp_ns = now_ns
        self.last_reset_frame_stamp_ns = last_frame_stamp_ns
        self._base = VioState.REINITIALIZING
        self._fault = False

    def finish_reset(self, now_ns: int, latency_s: float) -> None:
        self.last_reset_latency_ms = latency_s * 1e3
        self._base = VioState.INITIALIZING
        self._instance_start_ns = now_ns
        self._instance_frames = 0

    def record_frame(self, initialized: bool, degraded: bool) -> None:
        self._instance_frames += 1
        if not initialized:
            self._base = VioState.INITIALIZING
        elif degraded:
            self._base = VioState.DEGRADED
        else:
            self._base = VioState.TRACKING

    def fail(self) -> None:
        self._error = True

    def status_fields(self, now_ns: int) -> dict[str, Any]:
        imu_age, cam_age = self.ages_s(now_ns)
        return {
            "state": self.state(now_ns).value,
            "reset_count": self.reset_count,
            "last_reset_reason": self.last_reset_reason,
            "last_reset_stamp": self.last_reset_stamp_ns,
            "last_reset_frame_stamp_ns": self.last_reset_frame_stamp_ns,
            "last_reset_latency_ms": round(self.last_reset_latency_ms, 1),
            "imu_age_ms": None if imu_age is None else round(imu_age * 1e3, 1),
            "cam_age_ms": None if cam_age is None else round(cam_age * 1e3, 1),
        }


class MacvioSourceGate:
    """Decide from /macvio/status whether MACVIO odometry may be fused."""

    UNAVAILABLE = frozenset({VioState.BLIND_FAULT.value, VioState.REINITIALIZING.value})

    def __init__(self) -> None:
        self.state = "unknown"
        self.reset_count: int | None = None
        self.min_stamp_ns = 0
        self.failed = False
        self.needs_alignment = True

    @property
    def available(self) -> bool:
        return not self.failed and self.state not in self.UNAVAILABLE

    def update(self, status: dict[str, Any]) -> tuple[bool, bool]:
        """Apply one status message; return (newly failed, reset observed)."""
        self.state = str(status["state"])
        newly_failed = self.state == VioState.ERROR.value and not self.failed
        self.failed = self.failed or newly_failed
        reset_count = int(status["reset_count"])
        reset = self.reset_count is not None and reset_count != self.reset_count
        if self.reset_count is None or reset:
            self.min_stamp_ns = max(
                self.min_stamp_ns, int(status["last_reset_frame_stamp_ns"])
            )
        if reset:
            self.needs_alignment = True
        self.reset_count = reset_count
        return newly_failed, reset

    def accept(self, stamp_ns: int) -> bool:
        """Reject odometry while unavailable or stamped by a retired instance."""
        return self.available and stamp_ns > self.min_stamp_ns
