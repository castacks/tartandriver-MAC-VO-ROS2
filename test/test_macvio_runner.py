"""MacvioRunner stages, wiring and error policy, with a MACVIO whose tracker, optimizer and initializer are ground truth."""

from __future__ import annotations

from dataclasses import replace
from threading import Event
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pypose as pp
import pytest
import torch

from mac_ros2.macvio_profile import load_profile
from mac_ros2.macvio_runner import MacvioRunner, split_preprocess
from mac_ros2.odometry_state import MapPoints, OdometryState
from mac_ros2.output import processed_image
from mac_ros2.vio_state import VioState
from Src.Module.Map import PointNode
from test_macvio_dropout import (
    ACCELERATION, CAMERA_DT_NS, GRAVITY, IMU_DT_NS, MS, START_NS, GroundTruthTracker, body_pose, make_system,
)

A_MEAS = tuple((ACCELERATION - GRAVITY).tolist())
HEIGHT, WIDTH = 544, 1024  # Yamaha RGB, driver-rectified mono8
IMAGE = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


class Recorder:
    """Callback sink and a settable ROS clock."""

    def __init__(self) -> None:
        self.now = START_NS
        self.events: list[tuple[str, Any]] = []
        self.images: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.logs: list[str] = []
        self.odometry_seen = Event()

    def on_status(self, status: dict[str, Any]) -> None:
        self.events.append(("status", status))

    def on_odometry(self, odometry: OdometryState) -> None:
        self.events.append(("odometry", odometry))
        self.odometry_seen.set()

    def on_map_points(self, points: MapPoints) -> None:
        self.events.append(("map_points", points))

    def on_processed_image(self, stamp_ns: int, image: np.ndarray, K: np.ndarray) -> None:
        self.images.append((stamp_ns, image, K))

    def of_kind(self, kind: str) -> list[Any]:
        return [e for k, e in self.events if k == kind]

    def statuses(self) -> list[dict[str, Any]]:
        return self.of_kind("status")

    def odometry(self) -> list[OdometryState]:
        return self.of_kind("odometry")


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


class MappingTracker(GroundTruthTracker):
    """Ground-truth motion plus `MAP_POINTS` map points on the newest stereo node."""

    def set_data(self, state, local, graph):
        super().set_data(state, local, graph)
        stereo = len(graph.stereo) - 1
        points = graph.map.push(PointNode.init({
            "pos_Tc": torch.tensor([[5.0 + stereo, 0.0, 0.0]] * MAP_POINTS, dtype=torch.float32),
            "cov_Tc": torch.eye(3, dtype=torch.float64).expand(MAP_POINTS, 3, 3) * 0.01,
            "color": torch.full((MAP_POINTS, 3), 128, dtype=torch.uint8),
            "age": torch.ones(MAP_POINTS, 1, dtype=torch.uint8),
        }))
        graph.stereo2map.add(stereo, points[:1], points.size(0))
        graph.map2stereo.set(points, stereo)


MAP_POINTS = 4


def make_mapping_system() -> Any:
    system = make_system()
    system.FeatureTracker = MappingTracker()
    return system


def make_runner(system: Any, recorder: Recorder, profile: str = "yamaha_rgb", **overrides: Any) -> MacvioRunner:
    profile_ = replace(load_profile(profile), imu_wait_timeout_s=0.0, **overrides)
    return MacvioRunner(
        profile_, now_ns=lambda: recorder.now, on_odometry=recorder.on_odometry, on_status=recorder.on_status,
        on_map_points=recorder.on_map_points, on_processed_image=recorder.on_processed_image,
        log=recorder.logs.append, system=system,
    )


def publishing(runner: MacvioRunner) -> MacvioRunner:
    """Publish processed images from an RGB runner, whose CPU-only chain keeps these tests GPU-free."""
    runner.profile = replace(runner.profile, publish_processed_image=True)
    return runner


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
        assert runner.front_step(0.0) and runner.step(0.0)


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
            assert recorder.events[index + 1][0] == "map_points"
    map_points = recorder.of_kind("map_points")
    assert [p.stamp_ns for p in map_points] == [START_NS + k * CAMERA_DT_NS for k in range(20)]
    assert all(p.values.shape == (0, 15) for p in map_points)
    assert recorder.images == []

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
    assert status["frames_received"] == status["frames_processed"] == status["n_preprocessed"] == 20
    assert status["n_bad_inputs"] == 0 and status["n_front_dropped"] == 0 and status["last_error"] == ""
    assert 0.0 < status["front_latency_s"] and 0.0 < status["front_latency_mean_s"]


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
    assert len(runner._threads) == 2 and not any(t.is_alive() for t in runner._threads)


def test_queue_overflows_drop_oldest_without_a_dropout():
    recorder = Recorder()
    system = StubSystem()
    runner = make_runner(system, recorder, front_queue_size=2, work_queue_size=2)
    for k in range(4):
        feed_imu(runner, recorder, START_NS + k * CAMERA_DT_NS)
        runner.on_stereo(START_NS + k * CAMERA_DT_NS, IMAGE, IMAGE)
    assert runner.n_front_dropped == 2 and runner.state.frames_dropped == 0
    for k in range(4, 7):
        feed_imu(runner, recorder, START_NS + k * CAMERA_DT_NS)
        runner.on_stereo(START_NS + k * CAMERA_DT_NS, IMAGE, IMAGE)
        assert runner.front_step(0.0)
    assert runner.front_step(0.0) and not runner.front_step(0.0)
    status = runner.status(recorder.now)
    assert status["n_front_dropped"] == 3 and status["n_preprocessed"] == 4 and status["frames_dropped"] == 2
    assert status["frames_received"] == 7 and status["n_camera_dropouts"] == 0
    assert runner.step(0.0)
    assert system.frames[0].time_ns == [START_NS + 5 * CAMERA_DT_NS]


def test_slow_vio_still_publishes_every_received_pair():
    recorder = Recorder()
    system = make_system()
    run = system.run

    def slow_run(frame: Any) -> None:
        time.sleep(0.1)
        run(frame)

    system.run = slow_run
    runner = publishing(make_runner(system, recorder))
    feed_imu(runner, recorder, START_NS + 30 * CAMERA_DT_NS)
    runner.start()
    stamps = [START_NS + k * CAMERA_DT_NS for k in range(30)]
    for t in stamps:
        runner.on_stereo(t, IMAGE, IMAGE)
        time.sleep(0.02)
    deadline = time.monotonic() + 10.0
    while len(recorder.images) < len(stamps) and time.monotonic() < deadline:
        time.sleep(0.005)
    runner.shutdown()

    assert [stamp for stamp, _, _ in recorder.images] == stamps
    assert runner.n_front_dropped == 0 and runner.n_preprocessed == len(stamps)
    assert runner.state.frames_dropped > 0 and runner.n_frames_run < len(stamps)


def test_vio_receives_the_front_stage_frames():
    recorder = Recorder()
    system = make_system()
    frames: list[Any] = []
    run = system.run
    system.run = lambda frame: (frames.append(frame), run(frame))[1]
    runner = publishing(make_runner(system, recorder))
    preprocessed: list[Any] = []
    preprocess = runner.frames.preprocess
    runner.frames.preprocess = lambda *args: preprocessed.append(preprocess(*args)) or preprocessed[-1]
    run_frames(runner, recorder, 3)

    assert len(frames) == len(preprocessed) == len(recorder.images) == 3
    for frame, stereo, (stamp, image, K) in zip(frames, preprocessed, recorder.images):
        assert frame.stereo is stereo and frame.time_ns == [stamp] == stereo.time_ns
        np.testing.assert_array_equal(image, processed_image(stereo.imageL))
        np.testing.assert_array_equal(K, stereo.K[0].double().numpy())
    assert [int(f.idx) for f in frames] == [0, 1, 2]


def test_split_preprocess_keeps_order_and_rejects_unknown_transforms():
    chain = [SimpleNamespace(type=t, args=a) for t, a in [
        ("SmartResizeFrame", {"height": 4, "width": 4, "interp": "nearest"}),
        ("IMUSpecOverrideTransform", {"acc_noise_density": None, "gyro_noise_density": None,
                                      "acc_random_walk": None, "gyro_random_walk": None}),
        ("CastDevice", {"device": "cpu"}),
    ]]
    image, imu = split_preprocess(chain)
    assert [type(t).__name__ for t in image] == ["SmartResizeFrame", "CastDevice"]
    assert [type(t).__name__ for t in imu] == ["IMUSpecOverrideTransform"]
    with pytest.raises(ValueError, match="ScaleFrame"):
        split_preprocess([*chain, SimpleNamespace(type="ScaleFrame", args={"scale_u": 2, "scale_v": 2, "interp": "nearest"})])


def test_processed_image_and_map_points_share_the_frame_stamp():
    recorder = Recorder()
    runner = publishing(make_runner(make_mapping_system(), recorder))
    run_frames(runner, recorder, 8)
    image_stamps = [stamp for stamp, _, _ in recorder.images]
    map_points = recorder.of_kind("map_points")
    assert [p.stamp_ns for p in map_points] == image_stamps == [START_NS + k * CAMERA_DT_NS for k in range(8)]
    assert map_points[0].values.shape == (0, 15)
    for k, points in enumerate(map_points[1:], start=1):
        assert points.values.shape == (MAP_POINTS, 15)
        T_base_camera = runner.rig.T_body_base.Inv() @ runner.rig.T_BS
        expected = T_base_camera.Act(torch.tensor([[5.0 + k, 0.0, 0.0]], dtype=torch.float64)).numpy()
        np.testing.assert_allclose(points.values[:, :3], np.repeat(expected, MAP_POINTS, 0), atol=1e-5)
        np.testing.assert_array_equal(points.values[:, 3:6], 128.0)


def test_processed_image_is_resized_with_its_k():
    recorder = Recorder()
    runner = publishing(make_runner(StubSystem(), recorder))
    image = np.random.default_rng(0).integers(0, 255, size=(HEIGHT, WIDTH), dtype=np.uint8)
    runner.on_stereo(START_NS, image, image)
    assert runner.front_step(0.0)
    (stamp, out, K), = recorder.images
    assert stamp == START_NS and out.shape == (HEIGHT // 2, WIDTH // 2) and out.dtype == np.uint8
    # Nearest-exact sampling at half scale; float [0, 1] -> uint8 truncation may lose one level
    np.testing.assert_allclose(out, image[1::2, 1::2], atol=1)
    np.testing.assert_allclose(K, runner.rig.K.numpy() * np.array([[0.5], [0.5], [1.0]]), rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="thermal preprocessing runs on CUDA")
def test_stallion_thermal_processed_image_is_mono8_at_half_scale():
    recorder = Recorder()
    runner = make_runner(StubSystem(), recorder, profile="stallion_thermal")
    rng = np.random.default_rng(0)
    raw = rng.integers(20_000, 30_000, size=(1024, 1280), dtype=np.uint16)
    feed_imu(runner, recorder, START_NS + 100 * MS)
    runner.on_stereo(START_NS, raw, raw)
    assert runner.front_step(0.0)
    (stamp, out, K), = recorder.images
    assert out.shape == (512, 640) and out.dtype == np.uint8 and out.max() > 200 and out.min() < 50
    np.testing.assert_allclose(K, runner.rig.K.numpy() * np.array([[0.5], [0.5], [1.0]]), rtol=1e-6)
    assert runner.rect_rotation is not None


def test_frame_before_imu_is_skipped():
    recorder = Recorder()
    system = StubSystem()
    runner = make_runner(system, recorder)
    feed_imu(runner, recorder, START_NS + 500 * MS, start_ns=START_NS + 300 * MS)
    runner.on_stereo(START_NS, IMAGE, IMAGE)
    assert runner.front_step(0.0) and runner.step(0.0)
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
    assert runner.front_step(0.0) and runner.step(0.0)
    assert runner.state.state is VioState.ERROR
    assert recorder.statuses()[-1]["last_error"] == "RuntimeError: boom"

    system.error = None
    runner.on_stereo(START_NS + CAMERA_DT_NS, IMAGE, IMAGE)
    assert not runner.front_step(0.0) and not runner.step(0.0)
    assert len(system.frames) == 1 and recorder.odometry() == []
    runner.tick(recorder.now)
    assert recorder.statuses()[-1]["state"] == "ERROR"


def test_preprocessing_exception_latches_error():
    recorder = Recorder()
    runner = make_runner(StubSystem(), recorder)

    def broken(*_args: Any) -> Any:
        raise RuntimeError("cuda gone")

    runner.frames.preprocess = broken
    runner.on_stereo(START_NS, IMAGE, IMAGE)
    assert runner.front_step(0.0) and not runner.step(0.0)
    assert runner.state.state is VioState.ERROR
    assert recorder.statuses()[-1]["last_error"] == "preprocessing: RuntimeError: cuda gone"


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
