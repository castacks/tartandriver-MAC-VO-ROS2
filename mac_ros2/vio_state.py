"""MACVIO node state, perception hold and camera-dropout detection, with time passed in."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from threading import Lock
from typing import Any


class VioState(str, Enum):
    ERROR = "ERROR"
    INITIALIZING = "INITIALIZING"
    CAMERA_DROPOUT = "CAMERA_DROPOUT"
    SEGMENT_PENDING = "SEGMENT_PENDING"
    LOST_TRACK = "LOST_TRACK"
    IMU_DEAD = "IMU_DEAD"
    DEGRADED = "DEGRADED"
    TRACKING = "TRACKING"
    STALE = "STALE"  # set by the adapter only


RELEASED_STATES = frozenset({VioState.DEGRADED, VioState.TRACKING})


def holds(state: VioState) -> bool:
    """Whether `state` raises the perception hold."""
    return state not in RELEASED_STATES


@dataclass(frozen=True, kw_only=True)
class FrameResult:
    """Estimator state after one processed frame."""

    stamp_ns: int
    initialized: bool
    optimized: bool
    lost_track: bool
    vio_mode: str
    frames_in_mode: int
    last_gap_s: float
    n_chain_breaks: int
    n_dropped_samples: int
    segment_pending: bool
    n_segments: int
    last_frame_gap_s: float


class SensorClock:
    """Newest sensor stamp; advances with the ROS clock only while every stream is silent."""

    def __init__(self, silence_ns: int) -> None:
        self.silence_ns = silence_ns
        self.newest_imu_ns: int | None = None
        self.newest_pair_ns: int | None = None
        self._newest_ns: int | None = None
        self._newest_ros_ns = 0

    def on_imu(self, stamp_ns: int, ros_now_ns: int) -> None:
        self.newest_imu_ns = stamp_ns if self.newest_imu_ns is None else max(self.newest_imu_ns, stamp_ns)
        self._advance(stamp_ns, ros_now_ns)

    def on_pair(self, stamp_ns: int, ros_now_ns: int) -> None:
        self.newest_pair_ns = stamp_ns if self.newest_pair_ns is None else max(self.newest_pair_ns, stamp_ns)
        self._advance(stamp_ns, ros_now_ns)

    def now(self, ros_now_ns: int) -> int | None:
        if self._newest_ns is None:
            return None
        silent_for = ros_now_ns - self._newest_ros_ns
        return self._newest_ns + silent_for if silent_for > self.silence_ns else self._newest_ns

    def _advance(self, stamp_ns: int, ros_now_ns: int) -> None:
        if self._newest_ns is None or stamp_ns > self._newest_ns:
            self._newest_ns = stamp_ns
            self._newest_ros_ns = ros_now_ns


class CameraDropoutDetector:
    """Flags received stereo gaps longer than three frame periods, live and retroactively."""

    def __init__(self, frame_rate_hz: float) -> None:
        self.threshold_ns = int(3e9 / frame_rate_hz)
        self.last_pair_ns: int | None = None
        self.dropout = False
        self.awaiting_optimized = False
        self.gap_end_ns: int | None = None  # first pair received after the latest gap
        self.n_dropouts = 0

    def on_pair(self, stamp_ns: int) -> None:
        if self.last_pair_ns is not None and stamp_ns <= self.last_pair_ns:
            return
        long_gap = self.last_pair_ns is not None and stamp_ns - self.last_pair_ns > self.threshold_ns
        if long_gap and not self.dropout:
            self._start_dropout()
        if long_gap or self.dropout:
            self.gap_end_ns = stamp_ns
        self.last_pair_ns = stamp_ns
        self.dropout = False

    def update(self, sensor_now_ns: int | None) -> None:
        """Live check against the sensor clock."""
        if self.last_pair_ns is None or sensor_now_ns is None or self.dropout:
            return
        if sensor_now_ns - self.last_pair_ns > self.threshold_ns:
            self._start_dropout()
            self.dropout = True

    def on_processed(self, stamp_ns: int, optimized: bool) -> None:
        if self.awaiting_optimized and optimized and self.gap_end_ns is not None and stamp_ns >= self.gap_end_ns:
            self.awaiting_optimized = False

    def _start_dropout(self) -> None:
        self.n_dropouts += 1
        self.awaiting_optimized = True
        self.gap_end_ns = None


class VioStateMachine:
    """One node state per contract precedence; every event re-evaluates it. Thread-safe."""

    def __init__(self, frame_rate_hz: float, imu_silence_ns: int) -> None:
        self._lock = Lock()
        self._clock = SensorClock(imu_silence_ns)
        self._camera = CameraDropoutDetector(frame_rate_hz)
        self._frame: FrameResult | None = None
        self._error = ""
        self._failed = False
        self._state = VioState.INITIALIZING
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0

    @property
    def state(self) -> VioState:
        """State as of the latest event or `evaluate` call."""
        with self._lock:
            return self._state

    @property
    def hold(self) -> bool:
        return holds(self.state)

    def on_imu(self, stamp_ns: int, ros_now_ns: int) -> VioState:
        with self._lock:
            self._clock.on_imu(stamp_ns, ros_now_ns)
            return self._evaluate(ros_now_ns)

    def on_pair(self, stamp_ns: int, ros_now_ns: int) -> VioState:
        """A stereo pair arrived at the node, before any queueing."""
        with self._lock:
            self.frames_received += 1
            self._clock.on_pair(stamp_ns, ros_now_ns)
            self._camera.on_pair(stamp_ns)
            return self._evaluate(ros_now_ns)

    def on_frame(self, result: FrameResult, ros_now_ns: int) -> VioState:
        """The estimator finished one frame."""
        with self._lock:
            self.frames_processed += 1
            self._frame = result
            self._camera.on_processed(result.stamp_ns, result.optimized)
            return self._evaluate(ros_now_ns)

    def on_frame_dropped(self) -> None:
        with self._lock:
            self.frames_dropped += 1

    def fail(self, error: str, ros_now_ns: int) -> VioState:
        """Latch ERROR."""
        with self._lock:
            self._failed = True
            self._error = error
            return self._evaluate(ros_now_ns)

    def evaluate(self, ros_now_ns: int) -> VioState:
        with self._lock:
            return self._evaluate(ros_now_ns)

    def status(self, ros_now_ns: int) -> dict[str, Any]:
        """Node status payload: the state plus the fields that explain it."""
        with self._lock:
            state = self._evaluate(ros_now_ns)
            sensor_now = self._clock.now(ros_now_ns)
            frame = self._frame
            return {
                "state": state.value,
                "hold": holds(state),
                "camera_dropout": self._camera.dropout,
                "awaiting_optimized": self._camera.awaiting_optimized,
                "n_camera_dropouts": self._camera.n_dropouts,
                "camera_age_s": _age_s(sensor_now, self._clock.newest_pair_ns),
                "imu_age_s": _age_s(sensor_now, self._clock.newest_imu_ns),
                "initialized": frame is not None and frame.initialized,
                "lost_track": frame is not None and frame.lost_track,
                "last_frame_stamp_ns": None if frame is None else frame.stamp_ns,
                "vio_status": None if frame is None else _vio_status(frame),
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "last_error": self._error,
            }

    def _evaluate(self, ros_now_ns: int) -> VioState:
        self._camera.update(self._clock.now(ros_now_ns))
        self._state = self._resolve()
        return self._state

    def _resolve(self) -> VioState:
        frame = self._frame
        if self._failed:
            return VioState.ERROR
        if frame is None or not frame.initialized or frame.vio_mode == "PRE_INIT":
            return VioState.INITIALIZING
        if self._camera.dropout:
            return VioState.CAMERA_DROPOUT
        if self._camera.awaiting_optimized or frame.segment_pending:
            return VioState.SEGMENT_PENDING
        if frame.lost_track:
            return VioState.LOST_TRACK
        if frame.vio_mode == "IMU_DEAD":
            return VioState.IMU_DEAD
        if frame.vio_mode in ("VO_FALLBACK", "RECOVERING"):
            return VioState.DEGRADED
        return VioState.TRACKING


def _age_s(now_ns: int | None, stamp_ns: int | None) -> float | None:
    return None if now_ns is None or stamp_ns is None else max(now_ns - stamp_ns, 0) / 1e9


def _vio_status(frame: FrameResult) -> dict[str, Any]:
    fields = asdict(frame)
    for key in ("stamp_ns", "initialized", "optimized", "lost_track"):
        del fields[key]
    return fields
