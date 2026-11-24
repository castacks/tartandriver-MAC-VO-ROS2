"""MACVIO node profiles: estimator config, rig selection, topics, frames and runtime limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, get_args

import yaml

from ._macslam import MACSLAM_PATH

Platform = Literal["yamaha", "stallion"]
Modality = Literal["rgb", "thermal"]
RectMode = Literal["mono_rect", "stereo_rect"]
ImageTransport = Literal["raw", "compressed"]

PROFILE_DIR = Path(__file__).resolve().parent / "config"


@dataclass(frozen=True, kw_only=True)
class ProfileTopics:
    """Relative topic names, resolved under the node namespace."""

    image_left: str
    image_right: str
    imu: str
    odometry: str
    status: str


@dataclass(frozen=True, kw_only=True)
class ProfileFrames:
    odom: str
    base: str
    tf_base: str  # TF child of `odom`; empty disables the node's TF


@dataclass(frozen=True, kw_only=True)
class MacvioProfile:
    name: str
    platform: Platform
    modality: Modality
    rect_mode: RectMode | None
    macslam_config: Path  # absolute experiment yaml
    image_transport: ImageTransport
    topics: ProfileTopics
    frames: ProfileFrames
    stereo_queue_size: int
    stereo_slop_s: float
    work_queue_size: int
    imu_buffer_capacity: int
    imu_wait_timeout_s: float
    clock_silence_s: float
    max_consecutive_bad_inputs: int
    fd_twist_variance: float


def load_profile(spec: str | Path) -> MacvioProfile:
    """Profile by name (a yaml in the package config dir) or by path."""
    path = Path(spec).expanduser()
    if path.suffix not in (".yaml", ".yml"):
        path = PROFILE_DIR / f"{spec}.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    return parse_profile(raw, name=path.stem)


def parse_profile(raw: Mapping[str, Any], name: str) -> MacvioProfile:
    """Validate a profile mapping; raises ValueError on any unknown, missing or invalid field."""
    _keys(raw, {"platform", "modality", "rect_mode", "macslam_config", "image_transport",
                "topics", "frames", "sync", "imu", "errors", "output"}, name)
    platform = _choice(raw["platform"], Platform, "platform")
    modality = _choice(raw["modality"], Modality, "modality")
    rect_mode = raw["rect_mode"]
    if modality == "thermal":
        rect_mode = _choice(rect_mode, RectMode, "rect_mode")
    elif rect_mode is not None:
        raise ValueError(f"{name}: RGB images are driver-rectified, rect_mode must be null")

    macslam_config = MACSLAM_PATH / str(raw["macslam_config"])
    if not macslam_config.is_file():
        raise ValueError(f"{name}: macslam_config {macslam_config} does not exist")

    topics = _section(raw, "topics", {"image_left", "image_right", "imu", "odometry", "status"}, name)
    for key, topic in topics.items():
        if not isinstance(topic, str) or not topic or topic.startswith("/"):
            raise ValueError(f"{name}: topics.{key} must be a non-empty relative name, got {topic!r}")
    frames = _section(raw, "frames", {"odom", "base", "tf_base"}, name)
    sync = _section(raw, "sync", {"stereo_queue_size", "stereo_slop_s", "work_queue_size"}, name)
    imu = _section(raw, "imu", {"buffer_capacity", "wait_timeout_s", "clock_silence_s"}, name)
    errors = _section(raw, "errors", {"max_consecutive_bad_inputs"}, name)
    output = _section(raw, "output", {"fd_twist_variance"}, name)

    return MacvioProfile(
        name=name,
        platform=platform,
        modality=modality,
        rect_mode=rect_mode,
        macslam_config=macslam_config,
        image_transport=_choice(raw["image_transport"], ImageTransport, "image_transport"),
        topics=ProfileTopics(**{k: str(v) for k, v in topics.items()}),
        frames=ProfileFrames(**{k: str(v or "") for k, v in frames.items()}),
        stereo_queue_size=_positive(sync["stereo_queue_size"], int, "sync.stereo_queue_size"),
        stereo_slop_s=_positive(sync["stereo_slop_s"], float, "sync.stereo_slop_s"),
        work_queue_size=_positive(sync["work_queue_size"], int, "sync.work_queue_size"),
        imu_buffer_capacity=_positive(imu["buffer_capacity"], int, "imu.buffer_capacity"),
        imu_wait_timeout_s=_non_negative(imu["wait_timeout_s"], "imu.wait_timeout_s"),
        clock_silence_s=_positive(imu["clock_silence_s"], float, "imu.clock_silence_s"),
        max_consecutive_bad_inputs=_positive(errors["max_consecutive_bad_inputs"], int,
                                             "errors.max_consecutive_bad_inputs"),
        fd_twist_variance=_positive(output["fd_twist_variance"], float, "output.fd_twist_variance"),
    )


def _keys(raw: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping")
    missing, unknown = expected - raw.keys(), raw.keys() - expected
    if missing or unknown:
        raise ValueError(f"{where}: missing {sorted(missing)}, unknown {sorted(unknown)}")


def _section(raw: Mapping[str, Any], key: str, expected: set[str], name: str) -> Mapping[str, Any]:
    section = raw[key]
    _keys(section, expected, f"{name}.{key}")
    return section


def _choice(value: Any, options: Any, field: str) -> Any:
    if value not in get_args(options):
        raise ValueError(f"{field} must be one of {get_args(options)}, got {value!r}")
    return value


def _positive(value: Any, kind: type[int] | type[float], field: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (kind is int and not isinstance(value, int)):
        raise ValueError(f"{field} must be {kind.__name__}, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value!r}")
    return kind(value)


def _non_negative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field} must be a non-negative number, got {value!r}")
    return float(value)
