"""MacvioRunner wiring and error policy, with a MACVIO whose tracker, optimizer and initializer are ground truth."""

from __future__ import annotations

from dataclasses import replace
from threading import Event
import time
from typing import Any

import numpy as np
import pypose as pp
import pytest
import torch

from mac_ros2.macvio_profile import load_profile
from mac_ros2.macvio_runner import MacvioRunner
from mac_ros2.odometry_state import OdometryState
from mac_ros2.vio_state import VioState
from test_macvio_dropout import ACCELERATION, CAMERA_DT_NS, GRAVITY, IMU_DT_NS, MS, START_NS, body_pose, make_system

A_MEAS = tuple((ACCELERATION - GRAVITY).tolist())
HEIGHT, WIDTH = 544, 1024  # Yamaha RGB, driver-rectified mono8
IMAGE = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


class Recorder:
    """Callback sink and a settable ROS clock."""

    def __init__(self) -> None:
        self.now = START_NS
        self.events: list[tuple[str, Any]] = []
        self.logs: list[str] = []
        self.odometry_seen = Event()

    def on_status(self, status: dict[str, Any]) -> None:
        self.events.append(("status", status))

    def on_odometry(self, odometry: OdometryState) -> None:
        self.events.append(("odometry", odometry))
        self.odometry_seen.set()

    def statuses(self) -> list[dict[str, Any]]:
        return [e for kind, e in self.events if kind == "status"]

    def odometry(self) -> list[OdometryState]:
        return [e for kind, e in self.events if kind == "odometry"]


class StubSystem:
    """Records frames; `run` raises when told to."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.frames: list[Any] = []
        self.terminated = False

    def run(self, frame: Any) -> None:
        self.frames.append(frame)
        if self.error is not None:
            raise self.error

    def terminate(self) -> None:
        self.terminated = True


def make_runner(system: Any, recorder: Recorder, **overrides: Any) -> MacvioRunner:
    profile = replace(load_profile("yamaha_rgb"), imu_wait_timeout_s=0.0, **overrides)
    return MacvioRunner(
        profile, now_ns=lambda: recorder.now, on_odometry=recorder.on_odometry, on_status=recorder.on_status,
        log=recorder.logs.append, system=system,
    )


def feed_imu(runner: MacvioRunner, recorder: Recorder, until_ns: int, start_ns: int = START_NS - 200 * MS) -> None:
    newest = runner.imu.newest_ns
    t = start_ns if newest is None else newest + IMU_DT_NS
    while t <= until_ns:
        recorder.now = t
        runner.on_imu(t, A_MEAS, (0.0, 0.0, 0.0))
        t += IMU_DT_NS


def run_frames(runner: MacvioRunner, recorder: Recorder, count: int, first: int = 0) -> None:
    for k in range(first, first + count):
        t = START_NS + k * CAMERA_DT_NS
        feed_imu(runner, recorder, t + IMU_DT_NS)
        runner.on_stereo(t, IMAGE, IMAGE)
        assert runner.step(0.0)


def test_frames_flow_to_odometry_with_status_first():
    recorder = Recorder()
    system = make_system()
    frames: list[Any] = []
    run = system.run
    system.run = lambda frame: (frames.append(frame), run(frame))[1]
    runner = make_runner(system, recorder)
    run_frames(runner, recorder, 20)

    assert runner.n_frames_run == 20 and len(frames) == 20
    assert frames[0].stereo.imageL.shape == (1, 3, 272, 512)
    assert [int(f.idx) for f in frames] == list(range(20))
    for k, frame in enumerate(frames):
        assert frame.time_ns == [START_NS + k * CAMERA_DT_NS]
        assert int(frame.imu.time_ns[0, -1, 0]) == frame.time_ns[0]

    for index, (kind, odometry) in enumerate(recorder.events):
        if kind == "odometry":
            previous_kind, previous = recorder.events[index - 1]
            assert previous_kind == "status" and previous["last_frame_stamp_ns"] == odometry.stamp_ns

    states = [s["state"] for s in recorder.statuses()]
    assert states[0] == "INITIALIZING" and states[-1] == "TRACKING"
    odometry = recorder.odometry()
    assert len(odometry) == states.count("TRACKING")
    final_ns = START_NS + 19 * CAMERA_DT_NS
    expected = (pp.SE3(body_pose(final_ns)) @ runner.rig.T_body_base).tensor().numpy()[0]
    np.testing.assert_allclose(odometry[-1].pose, expected, atol=1e-6)
    assert np.isfinite(odometry[-1].twist).all()

    runner.tick(recorder.now)
    status = recorder.statuses()[-1]
    assert status["frames_received"] == status["frames_processed"] == 20
    assert status["n_bad_inputs"] == 0 and status["last_error"] == ""


def test_worker_thread_processes_and_shuts_down():
    recorder = Recorder()
    system = make_system()
    terminated = []
    system.terminate = lambda: terminated.append(True)
    runner = make_runner(system, recorder)
    runner.start()
    for k in range(12):
        t = START_NS + k * CAMERA_DT_NS
        feed_imu(runner, recorder, t + IMU_DT_NS)
        runner.on_stereo(t, IMAGE, IMAGE)
        deadline = time.monotonic() + 10.0
        while runner.state.frames_processed < k + 1 and time.monotonic() < deadline:
            time.sleep(0.005)
    assert recorder.odometry_seen.is_set()
    runner.shutdown()
    assert terminated == [True]
    assert runner._worker is not None and not runner._worker.is_alive()


def test_queue_overflow_drops_oldest_without_a_dropout():
    recorder = Recorder()
    system = StubSystem()
    runner = make_runner(system, recorder, work_queue_size=2)
    feed_imu(runner, recorder, START_NS + 3 * CAMERA_DT_NS)
    for k in range(4):
        runner.on_stereo(START_NS + k * CAMERA_DT_NS, IMAGE, IMAGE)
    assert runner.state.frames_dropped == 2
    status = runner.status(recorder.now)
    assert status["frames_received"] == 4 and status["n_camera_dropouts"] == 0


def test_frame_before_imu_is_skipped():
    recorder = Recorder()
    system = StubSystem()
    runner = make_runner(system, recorder)
    feed_imu(runner, recorder, START_NS + 500 * MS, start_ns=START_NS + 300 * MS)
    runner.on_stereo(START_NS, IMAGE, IMAGE)
    assert runner.step(0.0)
    assert runner.n_before_imu == 1 and system.frames == []
    assert runner.state.state is VioState.INITIALIZING


@pytest.mark.parametrize("left,right,stamp_offset,match", [
    (IMAGE, IMAGE, -CAMERA_DT_NS, "does not follow"),
    (IMAGE, IMAGE, 0, "does not follow"),
    (IMAGE[:, :-1], IMAGE[:, :-1], CAMERA_DT_NS, "expected"),
    (IMAGE, IMAGE.astype(np.uint16), CAMERA_DT_NS, "uint16"),
    (IMAGE, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), CAMERA_DT_NS, "differ"),
])
def test_bad_pair_is_dropped_and_counted(left, right, stamp_offset, match):
    recorder = Recorder()
    system = StubSystem()
    runner = make_runner(system, recorder)
    feed_imu(runner, recorder, START_NS + 500 * MS)
    runner.on_stereo(START_NS, IMAGE, IMAGE)
    runner.on_stereo(START_NS + stamp_offset, left, right)
    assert runner.n_bad_inputs == 1 and runner.state.frames_received == 1
    assert len(recorder.logs) == 1 and match in recorder.logs[0]
    assert runner.state.state is not VioState.ERROR


def test_persistent_bad_input_is_fatal_and_a_good_pair_resets_the_streak():
    recorder = Recorder()
    runner = make_runner(StubSystem(), recorder, max_consecutive_bad_inputs=3)
    bad = IMAGE[:-1]
    feed_imu(runner, recorder, START_NS + 2_000 * MS)
    runner.on_stereo(START_NS, bad, bad)
    runner.on_stereo(START_NS + CAMERA_DT_NS, bad, bad)
    runner.on_stereo(START_NS + 2 * CAMERA_DT_NS, IMAGE, IMAGE)
    runner.on_stereo(START_NS + 3 * CAMERA_DT_NS, bad, bad)
    runner.on_bad_input("undecodable image")
    assert runner.state.state is not VioState.ERROR
    runner.on_stereo(START_NS + 4 * CAMERA_DT_NS, bad, bad)
    assert runner.state.state is VioState.ERROR
    assert "3 consecutive bad stereo pairs" in recorder.statuses()[-1]["last_error"]
    assert runner.n_bad_inputs == 5


def test_estimator_exception_latches_error():
    recorder = Recorder()
    system = StubSystem(error=RuntimeError("boom"))
    runner = make_runner(system, recorder)
    feed_imu(runner, recorder, START_NS + 1_000 * MS)
    runner.on_stereo(START_NS, IMAGE, IMAGE)
    assert runner.step(0.0)
    assert runner.state.state is VioState.ERROR
    assert recorder.statuses()[-1]["last_error"] == "RuntimeError: boom"

    system.error = None
    runner.on_stereo(START_NS + CAMERA_DT_NS, IMAGE, IMAGE)
    assert not runner.step(0.0)
    assert len(system.frames) == 1 and recorder.odometry() == []
    runner.tick(recorder.now)
    assert recorder.statuses()[-1]["state"] == "ERROR"


def test_non_finite_pose_latches_error():
    recorder = Recorder()
    system = make_system()
    runner = make_runner(system, recorder)
    run_frames(runner, recorder, 10)
    assert recorder.odometry()
    run = system.run

    def run_then_corrupt(frame: Any) -> None:
        run(frame)
        system.Map.body.data["pose"][-1] = torch.full((7,), float("nan"), dtype=torch.float64)

    system.run = run_then_corrupt
    published = len(recorder.odometry())
    run_frames(runner, recorder, 1, first=10)
    assert runner.state.state is VioState.ERROR
    assert "non-finite pose" in recorder.statuses()[-1]["last_error"]
    assert len(recorder.odometry()) == published


def test_non_finite_imu_is_rejected():
    recorder = Recorder()
    runner = make_runner(StubSystem(), recorder)
    runner.on_imu(START_NS, (float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0))
    assert runner.status(START_NS)["imu_rejected"] == 1 and runner.imu.newest_ns is None
