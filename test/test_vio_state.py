"""Node state table, hold, and the received-pair camera-dropout detector."""

from __future__ import annotations

import pytest

from mac_ros2.vio_state import (
    CameraDropoutDetector,
    FrameResult,
    SensorClock,
    VioState,
    VioStateMachine,
    holds,
)

MS = 1_000_000
S = 1_000_000_000
T0 = 10 * S
FPS = 10.0
SILENCE = 50 * MS


def frame(stamp_ns: int = T0, **overrides) -> FrameResult:
    fields = dict(
        stamp_ns=stamp_ns, initialized=True, optimized=True, lost_track=False, vio_mode="VIO",
        frames_in_mode=1, last_gap_s=0.0, n_chain_breaks=0, n_dropped_samples=0,
        segment_pending=False, n_segments=0, last_frame_gap_s=0.1,
    )
    fields.update(overrides)
    return FrameResult(**fields)


def machine(
    result: FrameResult | None = None, *, error: bool = False, dropout: bool = False, awaiting: bool = False
) -> VioStateMachine:
    """Machine with a frame processed at T0, optionally a live dropout, a finished gap, or an error."""
    m = VioStateMachine(FPS, SILENCE)
    m.on_imu(T0, T0)
    m.on_pair(T0, T0)
    if result is not None:
        m.on_frame(result, T0)
    if dropout or awaiting:
        m.on_imu(T0 + 400 * MS, T0 + 400 * MS)
    if awaiting and not dropout:
        m.on_pair(T0 + 450 * MS, T0 + 450 * MS)
    if error:
        m.fail("RuntimeError: boom", T0 + 500 * MS)
    return m


HOLD_TABLE = {
    VioState.ERROR: True,
    VioState.INITIALIZING: True,
    VioState.CAMERA_DROPOUT: True,
    VioState.SEGMENT_PENDING: True,
    VioState.LOST_TRACK: True,
    VioState.IMU_DEAD: True,
    VioState.DEGRADED: False,
    VioState.TRACKING: False,
    VioState.STALE: True,
}


@pytest.mark.parametrize("state,hold", HOLD_TABLE.items())
def test_hold_column(state, hold):
    assert holds(state) is hold


# Each row also sets every lower-precedence condition it can, so the row must win
PRECEDENCE_ROWS = [
    ("error", lambda: machine(frame(initialized=False, vio_mode="PRE_INIT"), error=True, dropout=True),
     VioState.ERROR),
    ("initializing", lambda: machine(frame(initialized=False, vio_mode="PRE_INIT", lost_track=True), dropout=True),
     VioState.INITIALIZING),
    ("camera_dropout", lambda: machine(frame(segment_pending=True, lost_track=True, vio_mode="IMU_DEAD"),
                                       dropout=True), VioState.CAMERA_DROPOUT),
    ("awaiting_optimized", lambda: machine(frame(lost_track=True, vio_mode="IMU_DEAD"), awaiting=True),
     VioState.SEGMENT_PENDING),
    ("core_segment_pending", lambda: machine(frame(segment_pending=True, lost_track=True, vio_mode="IMU_DEAD")),
     VioState.SEGMENT_PENDING),
    ("lost_track", lambda: machine(frame(lost_track=True, vio_mode="IMU_DEAD")), VioState.LOST_TRACK),
    ("imu_dead", lambda: machine(frame(vio_mode="IMU_DEAD")), VioState.IMU_DEAD),
    ("vo_fallback", lambda: machine(frame(vio_mode="VO_FALLBACK")), VioState.DEGRADED),
    ("recovering", lambda: machine(frame(vio_mode="RECOVERING")), VioState.DEGRADED),
    ("vio", lambda: machine(frame()), VioState.TRACKING),
]


@pytest.mark.parametrize("name,build,expected", PRECEDENCE_ROWS, ids=[row[0] for row in PRECEDENCE_ROWS])
def test_state_precedence(name, build, expected):
    m = build()
    assert m.state is expected
    assert m.hold is HOLD_TABLE[expected]


def test_initializing_before_any_frame():
    assert VioStateMachine(FPS, SILENCE).evaluate(T0) is VioState.INITIALIZING


def test_error_latches():
    m = machine(frame(), error=True)
    assert m.on_frame(frame(T0 + 600 * MS), T0 + 600 * MS) is VioState.ERROR
    assert m.status(T0 + 600 * MS)["last_error"] == "RuntimeError: boom"


def test_node_frame_drops_are_not_a_dropout():
    """Pairs arrive every 100 ms; the node processes every fourth one (processed gap 400 ms)."""
    m = VioStateMachine(FPS, SILENCE)
    states = []
    for k in range(40):
        t = T0 + k * 100 * MS
        for imu in range(10):
            states.append(m.on_imu(t + imu * 10 * MS, t + imu * 10 * MS))
        states.append(m.on_pair(t, t))
        if k % 4 == 0:
            states.append(m.on_frame(frame(t), t))
        else:
            m.on_frame_dropped()
    assert set(states[states.index(VioState.TRACKING):]) == {VioState.TRACKING}
    assert m.status(T0 + 4 * S)["frames_dropped"] == 30


def test_live_dropout_on_sensor_clock():
    m = machine(frame())
    assert m.on_imu(T0 + 300 * MS, T0) is VioState.TRACKING
    assert m.on_imu(T0 + 301 * MS, T0) is VioState.CAMERA_DROPOUT
    status = m.status(T0)
    assert status["camera_dropout"] and status["awaiting_optimized"] and status["hold"]
    assert status["camera_age_s"] == pytest.approx(0.301) and status["imu_age_s"] == 0.0


def test_dropout_clears_on_first_optimized_frame_after_the_gap():
    m = machine(frame())
    m.on_imu(T0 + 800 * MS, T0 + 800 * MS)
    assert m.on_pair(T0 + 850 * MS, T0 + 850 * MS) is VioState.SEGMENT_PENDING
    # A backlog frame from before the gap does not release the hold
    assert m.on_frame(frame(T0 + 100 * MS), T0 + 860 * MS) is VioState.SEGMENT_PENDING
    assert m.on_frame(frame(T0 + 850 * MS, optimized=False, lost_track=True), T0 + 900 * MS) \
        is VioState.SEGMENT_PENDING
    assert m.on_pair(T0 + 950 * MS, T0 + 950 * MS) is VioState.SEGMENT_PENDING
    assert m.on_frame(frame(T0 + 950 * MS), T0 + 960 * MS) is VioState.TRACKING


def test_retroactive_dropout_when_a_pair_arrives_after_a_long_gap():
    m = machine(frame())
    assert m.on_pair(T0 + 500 * MS, T0 + 1 * MS) is VioState.SEGMENT_PENDING
    status = m.status(T0 + 1 * MS)
    assert not status["camera_dropout"] and status["awaiting_optimized"] and status["n_camera_dropouts"] == 1
    assert m.on_frame(frame(T0 + 500 * MS), T0 + 2 * MS) is VioState.TRACKING


def test_imu_silent_falls_back_to_ros_clock():
    m = machine(frame())
    assert m.evaluate(T0 + 300 * MS) is VioState.TRACKING
    assert m.evaluate(T0 + 301 * MS) is VioState.CAMERA_DROPOUT
    status = m.status(T0 + 400 * MS)
    assert status["camera_age_s"] == pytest.approx(0.4) and status["imu_age_s"] == pytest.approx(0.4)


def test_imu_gap_with_camera_flowing_ages_on_pair_stamps():
    m = machine(frame())
    for k in range(1, 6):
        m.on_pair(T0 + k * 100 * MS, T0 + k * 100 * MS)
    status = m.status(T0 + 500 * MS)
    assert status["state"] == "TRACKING"
    assert status["camera_age_s"] == 0.0 and status["imu_age_s"] == pytest.approx(0.5)


def test_status_fields():
    m = machine(frame(n_chain_breaks=2, n_dropped_samples=0, segment_pending=False))
    status = m.status(T0)
    assert status["state"] == "TRACKING" and status["hold"] is False
    assert status["last_frame_stamp_ns"] == T0 and status["initialized"] and not status["lost_track"]
    assert status["vio_status"] == {
        "vio_mode": "VIO", "frames_in_mode": 1, "last_gap_s": 0.0, "n_chain_breaks": 2,
        "n_dropped_samples": 0, "segment_pending": False, "n_segments": 0, "last_frame_gap_s": 0.1,
    }
    assert (status["frames_received"], status["frames_processed"], status["frames_dropped"]) == (1, 1, 0)


@pytest.mark.parametrize("rate_hz,threshold_ns", [(10.0, 300_000_000), (14.0, 214_285_714)])
def test_threshold_is_three_frame_periods(rate_hz, threshold_ns):
    assert CameraDropoutDetector(rate_hz).threshold_ns == threshold_ns


def test_detector_ignores_out_of_order_pairs():
    detector = CameraDropoutDetector(FPS)
    detector.on_pair(T0)
    detector.on_pair(T0 - 500 * MS)
    assert detector.last_pair_ns == T0 and not detector.awaiting_optimized


def test_sensor_clock_tracks_newest_stamp_until_all_streams_are_silent():
    clock = SensorClock(SILENCE)
    assert clock.now(T0) is None
    clock.on_imu(T0, 5 * S)
    clock.on_pair(T0 - 20 * MS, 5 * S)
    assert clock.now(5 * S + SILENCE) == T0
    assert clock.now(5 * S + 200 * MS) == T0 + 200 * MS
