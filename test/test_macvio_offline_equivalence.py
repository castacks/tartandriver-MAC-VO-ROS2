"""The node's frames equal the offline VIO sequence's frames, fed the same raw files and IMU samples."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import socket
from typing import Any, Callable

import cv2
import numpy as np
import pytest
import torch
import yaml

from mac_ros2._macslam import MACSLAM_PATH
from mac_ros2.image_decode import decode_compressed, decode_image
from mac_ros2.imu_buffer import IMUBuffer
from mac_ros2.macvio_profile import MacvioProfile, load_profile
from mac_ros2.macvio_runner import FrameBuilder, load_experiment
from Src.DataLoader import SIFrameData, SequenceBase, get_rig, smart_transform

FRAMES = 6
IMU_LOOKAHEAD_NS = 1_000_000_000


def data_root() -> Path:
    """DATA_ROOT of this host from mac_slam's root config; MACVIO_DATA_ROOT overrides it."""
    if "MACVIO_DATA_ROOT" in os.environ:
        return Path(os.environ["MACVIO_DATA_ROOT"])
    config = MACSLAM_PATH / "Config/_Root.yaml"
    if not config.exists():
        config = MACSLAM_PATH / "Config/Root.yaml"
    host = socket.gethostname()
    for entry in yaml.safe_load(config.read_text()):
        if re.match(entry["pattern"], host):
            return Path(entry["env"]["DATA_ROOT"])
    return Path("/data")


def as_ros_image(image: np.ndarray) -> np.ndarray:
    encoding = {np.dtype(np.uint16): "mono16", np.dtype(np.uint8): "mono8"}[image.dtype]
    return decode_image(encoding, image.shape[0], image.shape[1], image.strides[0], image.tobytes(), False)


def as_ros_compressed(image: np.ndarray) -> np.ndarray:
    ok, payload = cv2.imencode(".png", image)
    assert ok
    return decode_compressed(payload.tobytes())


@dataclass(frozen=True)
class Case:
    seq: str
    type: str
    args: dict[str, str]
    imu_dir: str


CASES = {
    "yamaha_thermal": Case(
        seq="yamaha/27_micah_back2pavillion", type="Offroad_Thermal_VIO",
        args=dict(gt_pose="none", thermal_mode="raw", rect_mode="stereo_rect"), imu_dir="novatel_imu",
    ),
    "stallion_thermal": Case(
        seq="stallion/20260730-230242U_Ch_sfvd2+anza-forest-trail-avoid-open+dlio-updates-manual-drive",
        type="Stallion_Thermal_VIO", args=dict(gt_pose="none", rect_mode="stereo_rect"), imu_dir="microstrain_imu",
    ),
}
ROS_DECODE: dict[str, Callable[[np.ndarray], np.ndarray]] = {"raw": as_ros_image, "compressed": as_ros_compressed}


def offline_sequence(profile: MacvioProfile, case: Case) -> SequenceBase[SIFrameData]:
    seq = Path(case.seq)
    config = dict(name=seq.name, root=str(data_root() / seq.parent), seq=seq.name, **case.args)
    sequence = SequenceBase.instantiate(case.type, config)
    return smart_transform(sequence, load_experiment(profile).Preprocess)


def camera_rows(sequence: Any) -> list[tuple[int, Path, Path]]:
    """(stamp ns, left file, right file) of each logical VIO frame."""
    vio = sequence.original_seq
    stereo = vio.stereo_seq
    if hasattr(vio, "offset"):
        cams = [vio.offset + vio.get_index(k) for k in range(FRAMES)]
    else:
        cams = [stereo.get_index(vio.get_index(k)) for k in range(FRAMES)]
    return [(int(stereo.timestamps[c] * 1e9), stereo.Camera_L.files[c], stereo.Camera_R.files[c]) for c in cams]


def imu_rows(imu_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(imu_dir / "interp_data.txt", dtype=np.float64)
    stamps = (np.loadtxt(imu_dir / "interp_timestamps.txt", dtype=np.float64) * 1e9).astype(np.int64)
    return stamps, raw


def assert_tensor_equal(name: str, online: Any, offline: Any) -> None:
    if offline is None:
        assert online is None, name
        return
    online_t = online.tensor() if hasattr(online, "tensor") else online
    offline_t = offline.tensor() if hasattr(offline, "tensor") else offline
    assert online_t.dtype == offline_t.dtype and online_t.shape == offline_t.shape, name
    assert torch.equal(online_t.cpu(), offline_t.cpu()), name


@pytest.mark.local
@pytest.mark.parametrize("name", list(CASES))
def test_node_frames_equal_offline_frames(name):
    case = CASES[name]
    seq_dir = data_root() / case.seq
    if not seq_dir.is_dir():
        pytest.skip(f"{seq_dir} not available")
    profile = load_profile(name)
    rig = get_rig(profile.platform, profile.modality, profile.rect_mode)
    offline = offline_sequence(profile, case)
    builder = FrameBuilder(rig, getattr(load_experiment(profile).Preprocess, rig.preprocess_key))
    buffer = IMUBuffer(rig.imu, profile.imu_buffer_capacity, wait_timeout_s=0.0)
    to_ros = ROS_DECODE[profile.image_transport]

    imu_dir = seq_dir / "sensors" / case.imu_dir
    imu_stamps, imu = imu_rows(imu_dir if imu_dir.is_dir() else seq_dir / case.imu_dir)
    fed, t_prev = 0, None
    for k, (t_cur, left_path, right_path) in enumerate(camera_rows(offline)):
        while fed < len(imu_stamps) and imu_stamps[fed] <= t_cur + IMU_LOOKAHEAD_NS:
            buffer.add(int(imu_stamps[fed]), imu[fed, 7:10].tolist(), imu[fed, 4:7].tolist())
            fed += 1
        left = to_ros(cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED))
        right = to_ros(cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED))
        window = buffer.build_window(t_prev, t_cur)
        assert window is not None
        online = builder.attach_imu(k, builder.preprocess(t_cur, left, right), window)
        reference = offline[k]
        t_prev = t_cur

        assert online.time_ns == reference.time_ns and int(online.idx) == int(reference.idx)
        a, b = online.stereo, reference.stereo
        assert (a.width, a.height, a.time_ns) == (b.width, b.height, b.time_ns)
        for field in ("imageL", "imageR", "K", "baseline", "T_BS", "camera_mask"):
            assert_tensor_equal(f"frame {k} stereo.{field}", getattr(a, field), getattr(b, field))
        for field in ("T_BS", "time_ns", "gravity", "acc_ws", "gyro_ws", "acc_cov", "gyro_cov", "dt_s",
                      "acc_bias", "gyro_bias", "acc_random_walk", "gyro_random_walk"):
            assert_tensor_equal(f"frame {k} imu.{field}", getattr(online.imu, field), getattr(reference.imu, field))
