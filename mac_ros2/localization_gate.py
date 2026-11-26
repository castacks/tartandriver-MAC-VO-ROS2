"""Localization adapter logic for MACVIO: alignment, perception hold, staleness and diagnostics.

Time is passed in as ROS-clock nanoseconds; messages are plain `OdometryState` and status dicts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from enum import IntEnum
import math
from threading import Lock
from typing import Any, Mapping

import numpy as np

from .odometry_state import OdometryState
from .vio_state import VioState, holds

ALIGNING = "ALIGNING"


class DiagLevel(IntEnum):
    """`diagnostic_msgs/DiagnosticStatus` levels."""

    OK = 0
    WARN = 1
    ERROR = 2
    STALE = 3


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("pose contains an invalid quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> normalized xyzw quaternion."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    quaternion = np.asarray(q)
    return quaternion / np.linalg.norm(quaternion)


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """(x y z qx qy qz qw) -> 4x4 parent <- child transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def initial_alignment(reference: OdometryState, source: OdometryState) -> np.ndarray:
    """Reference world <- source world from one synchronized pose pair."""
    return pose_matrix(reference.pose) @ np.linalg.inv(pose_matrix(source.pose))


def transform_odometry(source: OdometryState, world_from_source_world: np.ndarray) -> OdometryState:
    """Re-express the pose in the aligned world; the child-frame twist is unchanged."""
    transformed = world_from_source_world @ pose_matrix(source.pose)
    rotation = np.zeros((6, 6), dtype=np.float64)
    rotation[:3, :3] = world_from_source_world[:3, :3]
    rotation[3:, 3:] = world_from_source_world[:3, :3]
    return replace(
        source,
        pose=np.concatenate([transformed[:3, 3], matrix_quaternion(transformed[:3, :3])]),
        pose_cov=rotation @ source.pose_cov @ rotation.T,
    )


class LocalizationGate:
    """Pass aligned MACVIO odometry only while nothing holds perception. Thread-safe.

    Odometry passes only once a node status covering its stamp has arrived, so the node publishes
    each frame's status before its odometry.
    """

    def __init__(
        self, stale_after_s: float = 0.5, alignment_tolerance_s: float = 0.05, reference_buffer_size: int = 2000
    ) -> None:
        self.stale_after_ns = int(stale_after_s * 1e9)
        self.alignment_tolerance_ns = int(alignment_tolerance_s * 1e9)
        self._lock = Lock()
        self._references: deque[OdometryState] = deque(maxlen=reference_buffer_size)
        self._node_status: dict[str, Any] = {}
        self._node_state = VioState.INITIALIZING.value
        self._covered_ns: int | None = None
        self._last_heard_ns: int | None = None
        self._latched = False
        self.world_from_source_world: np.ndarray | None = None
        self.alignment_stamp_ns: int | None = None
        self.odometry_received = 0
        self.odometry_passed = 0

    @property
    def aligned(self) -> bool:
        return self.world_from_source_world is not None

    def on_reference(self, odometry: OdometryState) -> None:
        with self._lock:
            self._references.append(odometry)

    def on_status(self, status: Mapping[str, Any], ros_now_ns: int) -> None:
        with self._lock:
            self._last_heard_ns = ros_now_ns
            self._node_status = dict(status)
            self._node_state = str(status.get("state", ""))
            self._covered_ns = status.get("last_frame_stamp_ns")
            if self._node_state == VioState.ERROR.value:
                self._latched = True

    def on_odometry(self, odometry: OdometryState, ros_now_ns: int) -> OdometryState | None:
        """Aligned odometry to publish, or None while held."""
        with self._lock:
            self._last_heard_ns = ros_now_ns
            self.odometry_received += 1
            covered = self._covered_ns is not None and odometry.stamp_ns <= self._covered_ns
            if self._latched or not covered or _node_holds(self._node_state):
                return None
            transform = self.world_from_source_world
            if transform is None:
                transform = self._align(odometry)
            if transform is None:
                return None
            self.odometry_passed += 1
            return transform_odometry(odometry, transform)

    def state(self, ros_now_ns: int) -> str:
        """Effective state: latched ERROR > STALE > a holding node state > ALIGNING > node state."""
        with self._lock:
            return self._state(ros_now_ns)

    def hold(self, ros_now_ns: int) -> bool:
        return self.state(ros_now_ns) not in (VioState.TRACKING.value, VioState.DEGRADED.value)

    def diagnostic(self, ros_now_ns: int) -> dict[str, Any]:
        """`DiagnosticStatus`-shaped dict: level, name, message, hardware_id, values (str -> str)."""
        with self._lock:
            state = self._state(ros_now_ns)
            values = _flatten(self._node_status)
            values.update(
                {
                    "gate_state": state,
                    "aligned": str(self.aligned).lower(),
                    "alignment_stamp_ns": str(self.alignment_stamp_ns),
                    "latched": str(self._latched).lower(),
                    "odometry_received": str(self.odometry_received),
                    "odometry_passed": str(self.odometry_passed),
                }
            )
        return {"level": int(_level(state)), "name": "localization", "message": state, "hardware_id": "", "values": values}

    def _state(self, ros_now_ns: int) -> str:
        if self._latched:
            return VioState.ERROR.value
        if self._last_heard_ns is None or ros_now_ns - self._last_heard_ns > self.stale_after_ns:
            return VioState.STALE.value
        if _node_holds(self._node_state):
            return self._node_state
        if not self.aligned:
            return ALIGNING
        return self._node_state

    def _align(self, odometry: OdometryState) -> np.ndarray | None:
        """Fix the alignment from the nearest reference within tolerance."""
        if not self._references:
            return None
        reference = min(self._references, key=lambda r: abs(r.stamp_ns - odometry.stamp_ns))
        if abs(reference.stamp_ns - odometry.stamp_ns) > self.alignment_tolerance_ns:
            return None
        self.world_from_source_world = initial_alignment(reference, odometry)
        self.alignment_stamp_ns = odometry.stamp_ns
        return self.world_from_source_world


def _node_holds(state: str) -> bool:
    try:
        return holds(VioState(state))
    except ValueError:
        return True


def _level(state: str) -> DiagLevel:
    if state == VioState.TRACKING.value:
        return DiagLevel.OK
    if state == VioState.DEGRADED.value:
        return DiagLevel.WARN
    if state == VioState.STALE.value:
        return DiagLevel.STALE
    return DiagLevel.ERROR


def _flatten(fields: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in fields.items():
        if isinstance(value, Mapping):
            out.update(_flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = str(value)
    return out
