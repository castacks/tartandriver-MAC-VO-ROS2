"""Check the ROS-free MACVIO liveness state machine and the adapter source gate."""

from types import SimpleNamespace

import pytest

from mac_ros2.vio_health import (
    HealthConfig,
    HealthMonitor,
    MacvioSourceGate,
    VioState,
    WatchdogEvent,
)


S = 1_000_000_000
CONFIG = HealthConfig(
    watchdog_rate_hz=10.0, blind_warn_s=1.0, max_blind_s=3.0, reinit_timeout_s=10.0
)


def tracking_monitor(now_ns=0):
    monitor = HealthMonitor(CONFIG, now_ns)
    monitor.record_imu(now_ns)
    monitor.record_camera(now_ns)
    monitor.record_frame(initialized=True, degraded=False)
    return monitor


def test_config_requires_ordered_limits():
    config = SimpleNamespace(
        watchdog_rate_hz=10.0, blind_warn_s=3.0, max_blind_s=1.0, reinit_timeout_s=10.0
    )
    with pytest.raises(ValueError, match="blind_warn_s < max_blind_s"):
        HealthConfig.from_config(config)
    with pytest.raises(AttributeError):
        HealthConfig.from_config(SimpleNamespace(watchdog_rate_hz=10.0))


def test_initial_state_and_unknown_ages():
    monitor = HealthMonitor(CONFIG, 0)
    fields = monitor.status_fields(S // 2)
    assert fields["state"] == "INITIALIZING"
    assert fields["imu_age_ms"] is None and fields["cam_age_ms"] is None
    assert fields["reset_count"] == 0


def test_frame_updates_tracking_and_degraded():
    monitor = tracking_monitor()
    assert monitor.state(0) == VioState.TRACKING
    monitor.record_frame(initialized=True, degraded=True)
    assert monitor.state(0) == VioState.DEGRADED


def test_one_silent_sensor_is_not_blind():
    monitor = tracking_monitor()
    monitor.record_imu(5 * S)
    assert monitor.update_watchdog(5 * S) == WatchdogEvent.NONE
    assert monitor.state(5 * S) == VioState.TRACKING
    assert monitor.status_fields(5 * S)["cam_age_ms"] == pytest.approx(5000.0)


def test_blind_warn_then_fault_latches_and_stops_publishing():
    monitor = tracking_monitor()
    assert monitor.update_watchdog(int(0.9 * S)) == WatchdogEvent.NONE
    assert monitor.update_watchdog(int(1.1 * S)) == WatchdogEvent.BLIND_WARN
    assert monitor.state(int(1.1 * S)) == VioState.BLIND
    assert monitor.publishing_allowed()
    assert monitor.update_watchdog(int(3.1 * S)) == WatchdogEvent.BLIND_FAULT
    assert monitor.update_watchdog(int(3.2 * S)) == WatchdogEvent.NONE
    assert monitor.state(int(3.2 * S)) == VioState.BLIND_FAULT
    assert not monitor.publishing_allowed()


def test_blind_clears_when_a_sensor_returns_before_fault():
    monitor = tracking_monitor()
    monitor.update_watchdog(2 * S)
    monitor.record_camera(2 * S)
    assert monitor.state(2 * S) == VioState.TRACKING


def test_fault_requires_a_consumed_frame():
    monitor = HealthMonitor(CONFIG, 0)
    assert monitor.update_watchdog(10 * S) == WatchdogEvent.BLIND_WARN
    assert monitor.state(10 * S) == VioState.BLIND
    assert not monitor.blind_fault


def test_reset_after_both_sensors_return_from_fault():
    monitor = tracking_monitor()
    monitor.update_watchdog(4 * S)
    monitor.record_camera(5 * S)
    assert monitor.pending_reset(5 * S) is None
    assert monitor.state(5 * S) == VioState.BLIND_FAULT
    monitor.record_imu(5 * S)
    assert monitor.pending_reset(5 * S) == "blind_fault_recovered"

    monitor.begin_reset("blind_fault_recovered", 5 * S, last_frame_stamp_ns=123)
    assert monitor.state(5 * S) == VioState.REINITIALIZING
    assert not monitor.publishing_allowed()
    assert monitor.pending_reset(5 * S) is None
    monitor.finish_reset(6 * S, latency_s=0.5)
    fields = monitor.status_fields(6 * S)
    assert fields["state"] == "INITIALIZING"
    assert fields["reset_count"] == 1
    assert fields["last_reset_reason"] == "blind_fault_recovered"
    assert fields["last_reset_stamp"] == 5 * S
    assert fields["last_reset_frame_stamp_ns"] == 123
    assert fields["last_reset_latency_ms"] == pytest.approx(500.0)
    monitor.record_frame(initialized=True, degraded=False)
    assert monitor.state(6 * S) == VioState.TRACKING


def test_reinit_timeout_reinstantiates():
    monitor = tracking_monitor()
    monitor.begin_reset("imu_gap", 0, 0)
    monitor.finish_reset(0, 1.0)
    assert monitor.pending_reset(11 * S) is None
    monitor.record_frame(initialized=False, degraded=False)
    assert monitor.pending_reset(9 * S) is None
    assert monitor.pending_reset(11 * S) == "reinit_timeout"
    monitor.record_frame(initialized=True, degraded=False)
    assert monitor.pending_reset(11 * S) is None


def test_no_reinit_timeout_before_first_reset():
    monitor = HealthMonitor(CONFIG, 0)
    monitor.record_frame(initialized=False, degraded=False)
    assert monitor.pending_reset(60 * S) is None


def test_error_is_terminal():
    monitor = tracking_monitor()
    monitor.fail()
    assert monitor.state(10 * S) == VioState.ERROR
    assert monitor.update_watchdog(10 * S) == WatchdogEvent.NONE
    assert monitor.pending_reset(10 * S) is None
    assert not monitor.publishing_allowed()


def status(state, reset_count=0, frame_stamp=0):
    return {
        "state": state,
        "reset_count": reset_count,
        "last_reset_frame_stamp_ns": frame_stamp,
    }


def test_gate_latches_only_on_error():
    gate = MacvioSourceGate()
    assert gate.update(status("TRACKING")) == (False, False)
    gate.update(status("BLIND"))
    assert gate.accept(1)
    assert gate.update(status("ERROR")) == (True, False)
    assert gate.update(status("ERROR")) == (False, False)
    gate.update(status("TRACKING"))
    assert not gate.accept(10)


def test_gate_blind_fault_is_temporary():
    gate = MacvioSourceGate()
    gate.update(status("BLIND_FAULT"))
    assert not gate.accept(10)
    gate.update(status("REINITIALIZING", reset_count=1, frame_stamp=5))
    assert not gate.accept(10)
    gate.update(status("TRACKING", reset_count=1, frame_stamp=5))
    assert gate.accept(10)


def test_gate_realigns_and_rejects_stale_odometry_after_reset():
    gate = MacvioSourceGate()
    gate.update(status("TRACKING"))
    gate.needs_alignment = False
    assert gate.update(status("REINITIALIZING", 1, frame_stamp=100)) == (False, True)
    assert gate.needs_alignment
    gate.update(status("TRACKING", 1, frame_stamp=100))
    assert not gate.accept(100)
    assert gate.accept(101)


def test_gate_started_after_a_reset_does_not_realign_twice():
    gate = MacvioSourceGate()
    assert gate.update(status("TRACKING", 3, frame_stamp=50)) == (False, False)
    assert not gate.accept(50)
    assert gate.accept(51)
