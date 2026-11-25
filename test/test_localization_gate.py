"""Adapter gate: alignment, hold, staleness, ERROR latch and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from mac_ros2.localization_gate import (
    ALIGNING,
    DiagLevel,
    LocalizationGate,
    matrix_quaternion,
    pose_matrix,
    quaternion_matrix,
)
from mac_ros2.odometry_state import OdometryState

MS = 1_000_000
S = 1_000_000_000
T0 = 100 * S
YAW90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])


def odom(stamp_ns: int, position=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)) -> OdometryState:
    return OdometryState(
        stamp_ns=stamp_ns,
        pose=np.array([*position, *quaternion], dtype=np.float64),
        pose_cov=np.diag([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        twist=np.array([1.0, 0.5, 0.0, 0.0, 0.0, 0.1]),
        twist_cov=np.eye(6) * 0.01,
    )


def status(state: str, stamp_ns: int, **extra) -> dict:
    return {"state": state, "last_frame_stamp_ns": stamp_ns, **extra}


def aligned_gate() -> LocalizationGate:
    """Gate aligned at T0: reference at (100, 50, 0) yawed 90 deg; MACVIO at its origin."""
    gate = LocalizationGate()
    gate.on_reference(odom(T0 + 10 * MS, (100.0, 50.0, 0.0), YAW90))
    gate.on_status(status("TRACKING", T0), T0)
    assert gate.on_odometry(odom(T0), T0) is not None
    return gate


def test_holds_until_aligned_and_aligns_to_reference():
    gate = LocalizationGate()
    gate.on_status(status("TRACKING", T0), T0)
    assert gate.on_odometry(odom(T0), T0) is None
    assert gate.state(T0) == ALIGNING and gate.hold(T0)

    gate.on_reference(odom(T0 + 10 * MS, (100.0, 50.0, 0.0), YAW90))
    out = gate.on_odometry(odom(T0), T0)
    assert out is not None and gate.aligned and gate.alignment_stamp_ns == T0
    np.testing.assert_allclose(out.pose, [100.0, 50.0, 0.0, *YAW90], atol=1e-12)
    assert not gate.hold(T0) and gate.state(T0) == "TRACKING"

    later = gate.on_odometry(odom(T0, (1.0, 0.0, 0.0)), T0)
    np.testing.assert_allclose(later.pose[:3], [100.0, 51.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.diag(later.pose_cov), [2.0, 1.0, 3.0, 0.2, 0.1, 0.3], atol=1e-12)
    np.testing.assert_array_equal(later.twist, odom(T0).twist)


def test_reference_outside_tolerance_does_not_align():
    gate = LocalizationGate(alignment_tolerance_s=0.05)
    gate.on_reference(odom(T0 + 60 * MS, (100.0, 50.0, 0.0)))
    gate.on_status(status("TRACKING", T0), T0)
    assert gate.on_odometry(odom(T0), T0) is None and not gate.aligned


def test_no_alignment_while_the_node_holds():
    gate = LocalizationGate()
    gate.on_reference(odom(T0))
    gate.on_status(status("AWAITING_OPTIMIZED", T0), T0)
    assert gate.on_odometry(odom(T0), T0) is None and not gate.aligned


def test_never_realigns_across_holds():
    gate = aligned_gate()
    transform = gate.world_from_source_world.copy()
    gate.on_reference(odom(T0 + 1 * S, (-7.0, 3.0, 1.0)))
    gate.on_status(status("LOST_TRACK", T0 + 1 * S), T0 + 1 * S)
    assert gate.on_odometry(odom(T0 + 1 * S), T0 + 1 * S) is None
    gate.on_status(status("TRACKING", T0 + 2 * S), T0 + 2 * S)
    out = gate.on_odometry(odom(T0 + 2 * S), T0 + 2 * S)
    np.testing.assert_array_equal(gate.world_from_source_world, transform)
    np.testing.assert_allclose(out.pose[:3], [100.0, 50.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("state", ["CAMERA_DROPOUT", "AWAITING_OPTIMIZED", "LOST_TRACK", "IMU_DEAD", "INITIALIZING"])
def test_hold_states_pause_and_resume(state):
    gate = aligned_gate()
    gate.on_status(status(state, T0 + 100 * MS), T0 + 100 * MS)
    assert gate.on_odometry(odom(T0 + 100 * MS), T0 + 100 * MS) is None
    assert gate.hold(T0 + 100 * MS) and gate.state(T0 + 100 * MS) == state
    gate.on_status(status("TRACKING", T0 + 200 * MS), T0 + 200 * MS)
    assert gate.on_odometry(odom(T0 + 200 * MS), T0 + 200 * MS) is not None


def test_degraded_passes():
    gate = aligned_gate()
    gate.on_status(status("DEGRADED", T0 + 100 * MS), T0 + 100 * MS)
    assert gate.on_odometry(odom(T0 + 100 * MS), T0 + 100 * MS) is not None
    assert not gate.hold(T0 + 100 * MS)


def test_error_latches():
    gate = aligned_gate()
    gate.on_status(status("ERROR", T0 + 100 * MS), T0 + 100 * MS)
    gate.on_status(status("TRACKING", T0 + 200 * MS), T0 + 200 * MS)
    assert gate.on_odometry(odom(T0 + 200 * MS), T0 + 200 * MS) is None
    assert gate.state(T0 + 200 * MS) == "ERROR" and gate.hold(T0 + 200 * MS)
    assert gate.diagnostic(T0 + 200 * MS)["level"] == DiagLevel.ERROR


def test_odometry_ahead_of_its_status_is_held():
    gate = aligned_gate()
    assert gate.on_odometry(odom(T0 + 100 * MS), T0 + 100 * MS) is None
    gate.on_status(status("TRACKING", T0 + 100 * MS), T0 + 100 * MS)
    assert gate.on_odometry(odom(T0 + 100 * MS), T0 + 100 * MS) is not None


def test_staleness():
    gate = LocalizationGate(stale_after_s=0.5)
    assert gate.state(T0) == "STALE" and gate.hold(T0)
    gate = aligned_gate()
    assert gate.state(T0 + 500 * MS) == "TRACKING"
    assert gate.state(T0 + 501 * MS) == "STALE" and gate.hold(T0 + 501 * MS)
    gate.on_odometry(odom(T0), T0 + 900 * MS)
    assert gate.state(T0 + 1 * S) == "TRACKING"


def test_unknown_state_holds():
    gate = aligned_gate()
    gate.on_status(status("ready", T0 + 100 * MS), T0 + 100 * MS)
    assert gate.on_odometry(odom(T0 + 100 * MS), T0 + 100 * MS) is None and gate.hold(T0 + 100 * MS)


@pytest.mark.parametrize(
    "state,level",
    [("TRACKING", DiagLevel.OK), ("DEGRADED", DiagLevel.WARN), ("AWAITING_OPTIMIZED", DiagLevel.ERROR),
     ("CAMERA_DROPOUT", DiagLevel.ERROR)],
)
def test_diagnostic_levels(state, level):
    gate = aligned_gate()
    gate.on_status(status(state, T0, vio_status={"vio_mode": "VIO", "n_chain_breaks": 0}), T0)
    diag = gate.diagnostic(T0)
    assert diag["level"] == level and diag["message"] == state and diag["name"] == "localization"
    assert diag["values"]["vio_status.vio_mode"] == "VIO" and diag["values"]["aligned"] == "true"
    assert all(isinstance(v, str) for v in diag["values"].values())


def test_diagnostic_stale_and_aligning():
    gate = LocalizationGate()
    assert gate.diagnostic(T0)["level"] == DiagLevel.STALE
    gate.on_status(status("TRACKING", T0), T0)
    diag = gate.diagnostic(T0)
    assert diag["message"] == ALIGNING and diag["level"] == DiagLevel.ERROR


ROUND_TRIP = [np.eye(4)[i] for i in range(4)] + [np.random.default_rng(s).normal(size=4) for s in range(8)]


@pytest.mark.parametrize("q", ROUND_TRIP)
def test_quaternion_round_trip(q):
    q = q / np.linalg.norm(q)
    back = matrix_quaternion(quaternion_matrix(q))
    np.testing.assert_allclose(back * np.sign(back @ q), q, atol=1e-12)
    np.testing.assert_allclose(pose_matrix(np.array([1.0, 2.0, 3.0, *q]))[:3, 3], [1.0, 2.0, 3.0])
