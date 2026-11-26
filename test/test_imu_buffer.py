"""IMU windows built live by `IMUBuffer` against the offline `imu_window` loaders."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np
import pytest
import torch

from mac_ros2.imu_buffer import IMUBuffer
from Src.DataLoader import IMUData
from Src.DataLoader.Dataset.Offroad import OffroadIMULoader
from Src.DataLoader.IMUWindow import imu_window
from Src.Module.IMU.IMUChain import IMUChain
from Src.Module.IMU.IMUHealth import assess_window
from Src.Scripts.AutoTest.imu_dropout_fixtures import (
    A_W,
    G_TW,
    gt_velocity,
    health_config,
    mark_vo_optimized,
    new_map,
    push_body,
)
from Src.Utility.Enum import IMUWindowStatus, VIOMode

MS = 1_000_000
S = 1_000_000_000
BRACKET = 50 * MS
ZERO3 = torch.zeros(1, 3, dtype=torch.float64)



@dataclass(frozen=True)
class Spec:
    acc_noise_density: float
    gyro_noise_density: float
    acc_random_walk: float
    gyro_random_walk: float
    gravity: torch.Tensor
    max_bracket_ns: int = BRACKET


FIXTURE = Spec(8e-2, 4e-3, 1e-3, 1e-4, G_TW.view(1, 3))


def linear_value(t_ns: int) -> tuple[float, float, float]:
    t = t_ns * 1e-9
    return (1.0 + t, -2.0 * t, 0.5)


class Stream:
    """IMU samples fed to a buffer in stamp order, as they would arrive live."""

    def __init__(self, times_ns: list[int], acc: np.ndarray | None = None, gyro: np.ndarray | None = None):
        self.times = list(times_ns)
        self.acc = acc if acc is not None else np.asarray([linear_value(t) for t in self.times])
        self.gyro = gyro if gyro is not None else 0.1 * self.acc
        self.next = 0

    def feed_until(self, buffer: IMUBuffer, t_ns: int) -> None:
        while self.next < len(self.times) and self.times[self.next] <= t_ns:
            k = self.next
            buffer.add(self.times[k], tuple(self.acc[k]), tuple(self.gyro[k]))
            self.next += 1

    def feed_past(self, buffer: IMUBuffer, t_ns: int) -> None:
        """Feed through the first sample after `t_ns`."""
        self.feed_until(buffer, t_ns)
        if self.next < len(self.times):
            self.feed_until(buffer, self.times[self.next])

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.times, dtype=torch.int64),
            torch.from_numpy(self.acc).unsqueeze(0).double(),
            torch.from_numpy(self.gyro).unsqueeze(0).double(),
        )


def offline(stream: Stream, t_prev: int | None, t_cur: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(time_ns, dt_s, acc) of `imu_window` on the full stream."""
    t, acc, _ = stream.tensors()
    window = imu_window(t, t_prev, t_cur, BRACKET)
    return window.time_ns, window.dt_s, window.gather(acc)


def assert_matches_offline(imu: IMUData, stream: Stream, t_prev: int | None, t_cur: int) -> None:
    time_ns, dt_s, acc = offline(stream, t_prev, t_cur)
    assert torch.equal(imu.time_ns, time_ns)
    assert torch.equal(imu.dt_s, dt_s)
    assert torch.equal(imu.acc_ws, acc)


def uniform(t0: int, t1: int, step: int = 10 * MS, drop: tuple[int, int] | None = None) -> list[int]:
    times = range(t0, t1, step)
    return [t for t in times if drop is None or not drop[0] <= t < drop[1]]


def run_chain(windows: list[tuple[int, IMUData]], init_at: int | None = None):
    """Feed windows through a real `IMUChain` with VO-optimized GT bodies; returns (chain, modes, statuses)."""
    m = new_map()
    chain = IMUChain(health_config(), m.pre_integrator)
    modes, statuses = [], []
    for f, (t, imu) in enumerate(windows):
        b = push_body(m, t)
        _, health = chain.update_graph(m, imu, t, f, b, ZERO3, ZERO3)
        statuses.append(health.status)
        mark_vo_optimized(m, int(b.item()))
        if f == init_at:
            m.gravity_Tw = G_TW.clone()
            for i in range(len(m.imu)):
                m.imu.data["v_Tbs"][i] = gt_velocity(int(m.imu.data["time_ns"][i].item())).view(1, 3)
            chain.on_initialized(f)
        chain.after_tracking(m, f)
        modes.append(chain.mode)
    return chain, modes, statuses


def collapse(values: list) -> list:
    return [v for k, v in enumerate(values) if k == 0 or values[k - 1] != v]


def live_windows(
    stream: Stream, buffer: IMUBuffer, frames: list[int], lead_ns: int | None = None
) -> list[tuple[int, IMUData]]:
    """Windows of consecutive processed frames; the IMU leads each frame by `lead_ns`, else by one sample."""
    out, t_prev = [], None
    for t_cur in frames:
        if lead_ns is None:
            stream.feed_past(buffer, t_cur)
        else:
            stream.feed_until(buffer, t_cur + lead_ns)
        imu = buffer.build_window(t_prev, t_cur)
        assert imu is not None
        out.append((t_cur, imu))
        t_prev = t_cur
    return out


# 1. Clean 100 Hz stream, 10 Hz frames: bit-identical to the Offroad loader
def test_clean_stream_is_bit_identical_to_offroad_loader(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    ts_sec = 1761011590.0 + np.arange(n) * 0.01
    data = np.concatenate([np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), rng.normal(size=(n, 6))], axis=1)
    np.savetxt(tmp_path / "interp_data.txt", data, fmt="%.17g")
    np.savetxt(tmp_path / "interp_timestamps.txt", ts_sec, fmt="%.17g")
    first_ns = int(ts_sec[0] * 1e9)
    cam_ns = np.asarray([first_ns + 25 * MS + k * 100 * MS for k in range(25)], dtype=np.int64)
    loader = OffroadIMULoader(tmp_path, cam_ns)
    first, _ = loader.frameRangeQuery(None, 0)
    novatel = Spec(
        loader.acc_noise_density,
        loader.gyro_noise_density,
        float(loader.acc_random_walk),
        float(loader.gyro_random_walk),
        first.gravity,
    )

    stream = Stream(
        loader.timestamp[0, :, 0].tolist(), loader.acc[0].numpy().copy(), loader.gyro[0].numpy().copy()
    )
    buffer = IMUBuffer(novatel, capacity=4096, wait_timeout_s=0.0)
    windows = live_windows(stream, buffer, cam_ns.tolist())
    for k, (t_cur, imu) in enumerate(windows):
        expected, _ = loader.frameRangeQuery(None if k == 0 else k - 1, k)
        for field in ("time_ns", "dt_s", "acc_ws", "gyro_ws", "acc_cov", "gyro_cov", "gravity",
                      "acc_bias", "gyro_bias", "acc_random_walk", "gyro_random_walk"):
            assert torch.equal(getattr(imu, field), getattr(expected, field)), (k, field)
        assert torch.equal(imu.T_BS.tensor(), expected.T_BS.tensor())
        assert int(imu.time_ns[0, -1, 0]) == t_cur
    assert buffer.n_wait_timeouts == 0

    chain, _, statuses = run_chain(windows)
    assert all(s is IMUWindowStatus.Healthy for s in statuses[1:])
    assert chain.status.n_dropped_samples == 0


# 2. First frame: one sample interpolated at t_cur
def test_first_frame_is_one_interpolated_sample():
    stream = Stream(uniform(0, 1 * S))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    stream.feed_until(buffer, 1 * S)
    imu = buffer.build_window(None, 105 * MS)
    assert imu is not None and imu.num_samples == 1
    assert int(imu.time_ns[0, 0, 0]) == 105 * MS
    torch.testing.assert_close(imu.acc_ws[0, 0], torch.tensor(linear_value(105 * MS), dtype=torch.float64))
    assert float(imu.dt_s[0, 0]) == pytest.approx(0.005)


# 3. t_cur on a sample: no duplicate, end sample equals the raw one
def test_frame_on_sample_has_no_duplicate():
    stream = Stream(uniform(0, 1 * S))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    stream.feed_until(buffer, 1 * S)
    imu = buffer.build_window(100 * MS, 200 * MS)
    assert imu is not None
    times = imu.time_ns[0, :, 0].tolist()
    assert times == list(range(110 * MS, 201 * MS, 10 * MS))
    assert torch.equal(imu.acc_ws[0, -1], torch.from_numpy(stream.acc[20]))


# 4. NovAtel jitter: Healthy windows, finite covariances
def test_novatel_jitter_stays_healthy():
    steps = [int(12.6 * MS), int(37.9 * MS), int(0.4 * MS)]
    times, t = [], 0
    while t < 3 * S:
        times.append(t)
        t += steps[len(times) % 3]
    stream = Stream(times)
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    windows = live_windows(stream, buffer, [105 * MS + k * 100 * MS for k in range(25)])
    for t_cur, imu in windows:
        assert torch.isfinite(imu.acc_cov).all() and torch.isfinite(imu.gyro_cov).all()
    _, _, statuses = run_chain(windows)
    assert all(s is IMUWindowStatus.Healthy for s in statuses[1:])


GAP = (2 * S, 3 * S + 60 * MS)  # last sample before 1.99 s, next 3.06 s
T_A = 2 * S - 10 * MS


# 5. Gap entry within the bracket: raw only up to t_cur, Healthy; buffered or live gives the same window
@pytest.mark.parametrize("live", [False, True])
def test_gap_entry_is_raw_only_and_healthy(live):
    stream = Stream(uniform(0, 5 * S, drop=GAP))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    t_prev, t_cur = 1905 * MS, 2005 * MS
    stream.feed_until(buffer, t_cur if live else 4 * S)
    imu = buffer.build_window(t_prev, t_cur)
    assert imu is not None
    assert_matches_offline(imu, stream, t_prev, t_cur)
    assert int(imu.time_ns[0, -1, 0]) == T_A
    assert float(imu.dt_s[0, -1]) == pytest.approx((t_cur - T_A) / 1e9)
    health = assess_window(imu, t_cur, t_prev, t_prev, True, BRACKET)
    assert health.status is IMUWindowStatus.Healthy and health.n_dropped == 0


# 6. Inside the gap: an empty but valid window, classified Gap
def test_inside_gap_is_empty_window():
    stream = Stream(uniform(0, 5 * S, drop=GAP))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    stream.feed_until(buffer, 2405 * MS)
    imu = buffer.build_window(2305 * MS, 2405 * MS)
    assert imu is not None and imu.num_samples == 0
    assert imu.dt_s.shape == (1, 0) and imu.acc_cov.shape == (1, 0, 3, 3)
    assert assess_window(imu, 2405 * MS, T_A, None, False, BRACKET).status is IMUWindowStatus.Gap


# 7. Live gap past the bracket: no wait, same window as offline on the full stream
def test_live_gap_does_not_wait():
    stream = Stream(uniform(0, 5 * S, drop=GAP))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=5.0)
    stream.feed_until(buffer, 2055 * MS)
    start = time.monotonic()
    imu = buffer.build_window(1955 * MS, 2055 * MS)
    assert time.monotonic() - start < 0.5
    assert imu is not None and imu.num_samples == 4
    assert_matches_offline(imu, stream, 1955 * MS, 2055 * MS)
    assert buffer.n_wait_timeouts == 0


# 8. Gap exit: post-gap samples plus the interpolated end, classified Restart
def test_gap_exit_restarts_chain():
    stream = Stream(uniform(0, 5 * S, drop=GAP))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    stream.feed_until(buffer, 3115 * MS)
    imu = buffer.build_window(3005 * MS, 3105 * MS)
    assert imu is not None
    assert_matches_offline(imu, stream, 3005 * MS, 3105 * MS)
    assert int(imu.time_ns[0, 0, 0]) == GAP[1] and int(imu.time_ns[0, -1, 0]) == 3105 * MS
    health = assess_window(imu, 3105 * MS, T_A, None, False, BRACKET)
    assert health.status is IMUWindowStatus.Restart
    assert health.max_gap_ns == GAP[1] - T_A


# 9. Evicted history: no raise, single-sample restart
def test_evicted_history_restarts():
    stream = Stream(uniform(0, 5 * S))
    buffer = IMUBuffer(FIXTURE, capacity=20, wait_timeout_s=0.0)
    stream.feed_until(buffer, 1015 * MS)
    imu = buffer.build_window(505 * MS, 1005 * MS)
    assert imu is not None and imu.num_samples == 1
    assert int(imu.time_ns[0, 0, 0]) == 1005 * MS
    assert buffer.n_history_lost == 1


# 10. IMU starts after the camera: no window
def test_frame_before_imu_has_no_window():
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    assert buffer.build_window(None, 500 * MS) is None
    Stream(uniform(1 * S, 2 * S)).feed_until(buffer, 2 * S)
    assert buffer.build_window(None, 500 * MS) is None


# 11. IMUChain through a 1.07 s gap with live windows
def test_chain_rides_gap_without_dropped_samples():
    times = uniform(0, 6 * S, drop=GAP)
    a_meas = (A_W - G_TW).numpy()
    stream = Stream(times, np.tile(a_meas, (len(times), 1)), np.zeros((len(times), 3)))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    frames = [5 * MS + k * 100 * MS for k in range(55)]
    windows = live_windows(stream, buffer, frames, lead_ns=10 * MS)
    for k, (t_cur, imu) in enumerate(windows):
        time_ns, dt_s, _ = offline(stream, None if k == 0 else frames[k - 1], t_cur)
        assert torch.equal(imu.time_ns, time_ns) and torch.equal(imu.dt_s, dt_s)

    chain, modes, statuses = run_chain(windows, init_at=10)
    assert collapse(modes) == [VIOMode.PRE_INIT, VIOMode.VIO, VIOMode.VO_FALLBACK, VIOMode.RECOVERING, VIOMode.VIO]
    assert statuses[31] is IMUWindowStatus.Restart
    assert chain.status.n_chain_breaks == 1
    assert chain.status.n_dropped_samples == 0
    assert buffer.n_wait_timeouts == 1


def test_closely_spaced_frames_use_the_interpolated_end():
    stream = Stream(uniform(0, 1 * S, step=40 * MS))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    stream.feed_until(buffer, 1 * S)
    imu = buffer.build_window(405 * MS, 435 * MS)
    assert imu is not None and imu.time_ns[0, :, 0].tolist() == [435 * MS]


def test_wait_returns_when_the_next_sample_arrives():
    stream = Stream(uniform(0, 1 * S))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=2.0)
    stream.feed_until(buffer, 500 * MS)

    def late_sample() -> None:
        time.sleep(0.05)
        stream.feed_until(buffer, 510 * MS)

    worker = threading.Thread(target=late_sample)
    worker.start()
    imu = buffer.build_window(405 * MS, 505 * MS)
    worker.join()
    assert imu is not None
    assert_matches_offline(imu, stream, 405 * MS, 505 * MS)
    assert buffer.n_wait_timeouts == 0


def test_wait_timeout_treats_frame_as_gap():
    stream = Stream(uniform(0, 1 * S))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.02)
    stream.feed_until(buffer, 500 * MS)
    imu = buffer.build_window(405 * MS, 505 * MS)
    assert imu is not None and buffer.n_wait_timeouts == 1
    assert imu.time_ns[0, :, 0].tolist() == list(range(410 * MS, 501 * MS, 10 * MS))
    assert float(imu.dt_s[0, -1]) == pytest.approx(0.005)


def test_close_releases_a_waiting_window():
    stream = Stream(uniform(0, 1 * S))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=10.0)
    stream.feed_until(buffer, 500 * MS)
    threading.Timer(0.05, buffer.close).start()
    start = time.monotonic()
    assert buffer.build_window(405 * MS, 505 * MS) is not None
    assert time.monotonic() - start < 2.0


def test_rejects_non_finite_and_non_increasing_samples():
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    assert buffer.add(10, (0.0, 0.0, 9.8), (0.0, 0.0, 0.0))
    assert not buffer.add(10, (0.0, 0.0, 9.8), (0.0, 0.0, 0.0))
    assert not buffer.add(20, (float("nan"), 0.0, 9.8), (0.0, 0.0, 0.0))
    assert buffer.n_rejected == 2 and len(buffer) == 1 and buffer.newest_ns == 10


def test_buffer_is_trimmed_to_previous_frame():
    stream = Stream(uniform(0, 2 * S))
    buffer = IMUBuffer(FIXTURE, capacity=4096, wait_timeout_s=0.0)
    stream.feed_until(buffer, 1 * S)
    buffer.build_window(905 * MS, 955 * MS)
    assert len(buffer) == 1 + (1 * S - 900 * MS) // (10 * MS)
