import torch
from pathlib import Path
import typing as T
import os, sys
import pypose as pp

MACSLAM_PATH = Path(__file__).resolve().parent / "mac_slam"
sys.path.insert(0, str(MACSLAM_PATH))


from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

from Src.DataLoader import SmartResizeFrame, StereoData, StereoFrameData  # noqa: E402
from Src.Module.Map import BodyNode, StereoNode  # noqa: E402
from Src.Odometry.MACVO import MACVO  # noqa: E402
from Src.Utility.Config import asNamespace, load_config  # noqa: E402

PointCloudData = dict[str, torch.Tensor]


def empty_mapping(device: torch.device) -> PointCloudData:
    """Return a valid empty point cloud for frames without a dense map."""
    return {
        "pos_Tc": torch.empty((0, 3), dtype=torch.float32, device=device),
        "cov_Tc": torch.empty((0, 3, 3), dtype=torch.float32, device=device),
        "color": torch.empty((0, 3), dtype=torch.float32, device=device),
    }


def extract_current_camera_mapping(
    system: T.Any,
    device: torch.device,
) -> PointCloudData:
    """Extract the latest dense map and express it in the current camera frame.

    MACSLAM creates the dense map for the source image of the latest stereo
    pair. The coordinator stamps its output with the current image, so points
    and covariance must be transformed from that source camera into the current
    camera before being returned.
    """
    assert system.prev is not None, "MACSLAM must process a frame before publishing"
    current_stereo_id = int(system.prev.index)
    if current_stereo_id == 0:
        return empty_mapping(device)

    source_stereo_id = current_stereo_id - 1
    source_stereo_idx = torch.tensor([source_stereo_id], dtype=torch.long)
    current_stereo_idx = torch.tensor([current_stereo_id], dtype=torch.long)
    map_point_idx = system.Map.stereo2map.neighbors(source_stereo_idx)
    if map_point_idx.numel() == 0:
        return empty_mapping(device)

    source_body_idx = system.Map.stereo2body.neighbors(source_stereo_idx)
    current_body_idx = system.Map.stereo2body.neighbors(current_stereo_idx)

    # T_WS = T_WB @ T_BS. Therefore T_S1S0 maps source-camera points to
    # the current camera while preserving the NED camera-axis convention.
    T_ws_source = (
        pp.SE3(system.Map.body.data["pose"][source_body_idx])
        @ pp.SE3(system.Map.stereo.data["T_BS"][source_stereo_idx])
    )
    T_ws_current = (
        pp.SE3(system.Map.body.data["pose"][current_body_idx])
        @ pp.SE3(system.Map.stereo.data["T_BS"][current_stereo_idx])
    )
    T_current_source = T_ws_current.Inv() @ T_ws_source

    source_points = system.Map.map.data["pos_Tc"][map_point_idx]
    source_covariances = system.Map.map.data["cov_Tc"][map_point_idx]
    rotation = T_current_source.rotation().matrix()[0]

    current_points = T_current_source.Act(
        source_points.to(
            device=T_current_source.device,
            dtype=T_current_source.dtype,
        )
    )
    current_covariances = (
        rotation.unsqueeze(0)
        @ source_covariances.to(device=rotation.device, dtype=rotation.dtype)
        @ rotation.T.unsqueeze(0)
    )

    return {
        "pos_Tc": current_points.detach().clone().to(device=device, dtype=torch.float32),
        "cov_Tc": current_covariances.detach()
        .clone()
        .to(device=device, dtype=torch.float32),
        "color": system.Map.map.data["color"][map_point_idx]
        .detach()
        .clone()
        .to(device=device, dtype=torch.float32),
    }


def reset_online_map(system: T.Any) -> None:
    """Bound Map v3 storage while retaining the latest frame and frontend cache."""
    assert system.prev is not None, "MACSLAM context is required before resetting"
    assert system.FeatureTracker.prev is not None, "Frontend context is required before resetting"

    latest_body = BodyNode.init({
        key: value[-1:].detach().clone()
        for key, value in system.Map.body.data.items()
    })
    latest_stereo = StereoNode.init({
        key: value[-1:].detach().clone()
        for key, value in system.Map.stereo.data.items()
    })

    system.Map.clear()
    body_idx = system.Map.body.push(latest_body)
    stereo_idx = system.Map.stereo.push(latest_stereo)
    system.Map.body2stereo.set(body_idx, stereo_idx)
    system.Map.stereo2body.set(stereo_idx, body_idx)

    system.prev.index = 0
    system.prev.ba_id = 0
    system.FeatureTracker.prev.prev_stereo_id = 0

class MACVONode:

    def __init__(self, config_fp, device) -> None:

        if isinstance(config_fp, dict):
            cfg = asNamespace(config_fp)
        else:
            cfg, _ = load_config(Path(config_fp))

        MACVO.is_valid_config(cfg.Odometry)
        assert isinstance(cfg.Adapter.map_reset_interval, int), (
            "Adapter.map_reset_interval must be an integer"
        )
        assert cfg.Adapter.map_reset_interval >= 0, (
            "Adapter.map_reset_interval must be a non-negative integer"
        )

        self.frame_id = 0

        self.camera = cfg.Camera
        self.device = torch.device(device)
        self.map_reset_interval = cfg.Adapter.map_reset_interval
        self._mapped_frames_since_reset = 0
        self._last_published_frame_id = -1
        self._cached_output: tuple[
            torch.Tensor,
            dict[str, torch.Tensor],
            int,
            FeatureKeyList,
        ] | None = None

        if cfg.useRR:
            raise ValueError(
                "The embedded torch_coordinator adapter does not support Rerun; "
                "set useRR: false"
            )

        original_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            self.odometry = MACVO.from_config(cfg)
        finally:
            os.chdir(original_cwd)

        self.frame_fn = SmartResizeFrame({
            "height": cfg.Adapter.resize.height,
            "width": cfg.Adapter.resize.width,
            "interp": cfg.Adapter.resize.interp,
        })

    def publish_data(self, system: MACVO):
        """Return the latest pose and dense map in the current camera frame."""
        published_frame_id = self.frame_id - 1
        if self._last_published_frame_id == published_frame_id:
            assert self._cached_output is not None
            return self._cached_output

        assert system.prev is not None, "MACVO must receive a frame before publishing"
        pose_value = system.Map.body.data["pose"][-1]
        if isinstance(pose_value, pp.LieTensor):
            pose_value = pose_value.tensor()
        pose = pose_value.detach().clone().to(self.device, dtype=torch.float32)
        time_ns = int(system.Map.body.data["time_ns"][-1].item())
        points = extract_current_camera_mapping(system, self.device)

        output = (pose, points, time_ns, self.output_feature_keys)
        self._cached_output = output
        self._last_published_frame_id = published_frame_id

        if int(system.prev.index) > 0 and self.map_reset_interval > 0:
            self._mapped_frames_since_reset += 1
            if self._mapped_frames_since_reset >= self.map_reset_interval:
                reset_online_map(system)
                self._mapped_frames_since_reset = 0

        return output

    def receive_stereo(self, imageL, imageR, imageLColor, timestamp) -> None:
        """Convert a synchronized ROS tensor pair and advance MACVO once."""
        del imageLColor  # Retained in the API for coordinator compatibility.
        time_ns = int(timestamp * 1e9)
        image_left = imageL.image.to(dtype=torch.float32)
        image_right = imageR.image.to(dtype=torch.float32)

        stereo_frame = self.frame_fn(StereoFrameData(
            idx=torch.tensor([self.frame_id], dtype=torch.long),
            time_ns=[time_ns],
            stereo=StereoData(
                data_ids=None,
                T_BS=pp.identity_SE3(1, dtype=torch.float64),
                K=torch.tensor([[
                    [self.camera.fx, 0.0, self.camera.cx],
                    [0.0, self.camera.fy, self.camera.cy],
                    [0.0, 0.0, 1.0],
                ]], dtype=torch.float32),
                baseline=torch.tensor([self.camera.bl], dtype=torch.float32),
                time_ns=[time_ns],
                height=image_left.shape[0],
                width=image_left.shape[1],
                imageL=image_left.permute(2, 0, 1).unsqueeze(0),
                imageR=image_right.permute(2, 0, 1).unsqueeze(0),
            ),
        ))

        self.odometry.run(stereo_frame)
        self.frame_id += 1

    @property
    def output_feature_keys(self):
        """Describe RGB and covariance channels consumed downstream."""
        labels = ["r", "g", "b"] + [f"cov_{i}" for i in range(1, 10)]
        metainfo = ["raw"] * 3 + ["macvo"] * 9

        return FeatureKeyList(
            label=labels,
            metainfo=metainfo
        )
