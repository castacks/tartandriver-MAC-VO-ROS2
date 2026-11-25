"""Profiles: load, validate, and agree with the mac_slam rig and experiment configs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from mac_ros2.macvio_profile import PROFILE_DIR, load_profile, parse_profile
from mac_ros2.macvio_runner import load_experiment, split_preprocess
from Src.DataLoader.Dataset.OffroadRig import get_rig

PROFILES = ("yamaha_thermal", "yamaha_rgb", "stallion_thermal", "stallion_rgb")
EXPERIMENTS = {
    "yamaha_thermal": "VIO_MACI2_FR_Yamaha_Thermal.yaml",
    "yamaha_rgb": "VIO_MACI2_FR_Yamaha_RGB.yaml",
    "stallion_thermal": "VIO_MACI2_FR_Yamaha_RGB.yaml",
    "stallion_rgb": "VIO_MACI2_FR_Yamaha_RGB.yaml",
}


def raw_profile(name: str) -> dict:
    return yaml.safe_load((PROFILE_DIR / f"{name}.yaml").read_text())


@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads_its_experiment_and_preprocess_chain(name):
    profile = load_profile(name)
    assert profile.name == name and profile.macslam_config.name == EXPERIMENTS[name]
    rig = get_rig(profile.platform, profile.modality, profile.rect_mode)
    experiment = load_experiment(profile)
    image, imu = split_preprocess(getattr(experiment.Preprocess, rig.preprocess_key))
    assert image and [type(t).__name__ for t in imu] == ["IMUSpecOverrideTransform"]


def test_profile_by_path_equals_profile_by_name():
    assert load_profile(PROFILE_DIR / "yamaha_rgb.yaml") == load_profile("yamaha_rgb")


@pytest.mark.parametrize("name,base,transport", [
    ("yamaha_thermal", "vehicle", "raw"),
    ("yamaha_rgb", "vehicle", "raw"),
    ("stallion_thermal", "crl_rzr/base_link", "compressed"),
    ("stallion_rgb", "crl_rzr/base_link", "compressed"),
])
def test_platform_frames_and_transport(name, base, transport):
    profile = load_profile(name)
    assert profile.frames.base == base and profile.image_transport == transport


def test_topics_are_relative_and_match_the_bags():
    topics = {name: load_profile(name).topics for name in PROFILES}
    assert topics["yamaha_thermal"].image_left == "thermal_left/image"
    assert topics["yamaha_rgb"].image_right == "multisense/right/image_rect"
    assert topics["stallion_thermal"].image_left == "multisense_thermal_front/left/image_mono/compressed"
    assert topics["stallion_rgb"].image_right == "multisense_front/right/image_rect/compressed"
    assert {t.imu for t in topics.values()} == {"novatel/imu/data", "microstrain/nav/filtered_imu/data"}
    for t in topics.values():
        assert (t.odometry, t.status, t.map_points) == ("macvio/odometry_raw", "macvio/status", "macvio/map_points")


@pytest.mark.parametrize("name,image,camera_info,frame,tf_parent", [
    ("yamaha_thermal", "thermal_left/image_processed", "thermal_left/camera_info_processed",
     "thermal_left/optical_frame_rect", "thermal_left/optical_frame"),
    ("stallion_thermal", "multisense_thermal_front/left/image_processed",
     "multisense_thermal_front/left/camera_info_processed",
     "crl_rzr/multisense_thermal_front/left_camera_optical_frame", None),
    ("yamaha_rgb", None, None, None, None),
    ("stallion_rgb", None, None, None, None),
])
def test_processed_image_outputs(name, image, camera_info, frame, tf_parent):
    profile = load_profile(name)
    assert profile.publish_processed_image == (image is not None)
    assert (profile.topics.processed_image, profile.topics.processed_camera_info) == (image, camera_info)
    assert (profile.frames.processed_image, profile.frames.rect_tf_parent) == (frame, tf_parent)


def test_stallion_imu_buffer_covers_its_500hz_imu():
    for name in ("stallion_thermal", "stallion_rgb"):
        assert load_profile(name).imu_buffer_capacity >= 16384


def test_thermal_rect_mode():
    assert load_profile("yamaha_thermal").rect_mode == "stereo_rect"
    assert load_profile("stallion_thermal").rect_mode == "stereo_rect"
    assert load_profile("yamaha_rgb").rect_mode is None


@pytest.mark.parametrize("edit,match", [
    (lambda r: r.update(platform="husky"), "platform"),
    (lambda r: r.update(modality="lidar"), "modality"),
    (lambda r: r.update(rect_mode=None), "rect_mode"),
    (lambda r: r.update(rect_mode="stereo_rect", modality="rgb"), "rect_mode must be null"),
    (lambda r: r.update(macslam_config="Config/missing.yaml"), "does not exist"),
    (lambda r: r.update(image_transport="theora"), "image_transport"),
    (lambda r: r["topics"].update(imu="/novatel/imu/data"), "relative"),
    (lambda r: r["imu"].update(buffer_capacity=0), "positive"),
    (lambda r: r["imu"].update(buffer_capacity=4096.0), "int"),
    (lambda r: r["imu"].update(wait_timeout_s=-1), "non-negative"),
    (lambda r: r["sync"].pop("stereo_slop_s"), "missing"),
    (lambda r: r.update(Camera={"fx": 1.0}), "unknown"),
    (lambda r: r["frames"].update(tf_base="macvio_vehicle"), "unknown"),
    (lambda r: r.update(publish_processed_image="yes"), "bool"),
    (lambda r: r["topics"].update(processed_image=None), "relative"),
    (lambda r: r["frames"].update(processed_image=None), "frames.processed_image"),
    (lambda r: r.update(publish_processed_image=False), "rect_tf_parent"),
    (lambda r: r["sync"].update(front_queue_size=0), "positive"),
])
def test_invalid_profiles_are_rejected(edit, match):
    raw = raw_profile("yamaha_thermal")
    edit(raw)
    with pytest.raises(ValueError, match=match):
        parse_profile(raw, "bad")


def test_profiles_hold_no_calibration():
    for name in PROFILES:
        text = (PROFILE_DIR / f"{name}.yaml").read_text()
        for key in ("fx", "baseline", "T_BS", "extrinsics", "mask", "noise_density"):
            assert key not in text, f"{name} holds {key}"


def test_old_profiles_are_gone():
    assert not any(Path(PROFILE_DIR).glob("offroad_*_vio.yaml"))


def test_profile_is_immutable_but_replaceable():
    profile = load_profile("yamaha_rgb")
    assert replace(profile, imu_wait_timeout_s=0.0).imu_wait_timeout_s == 0.0
