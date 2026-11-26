"""Read the newest MACVIO estimate: base-frame odometry, map points, processed image and the frame summary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pypose as pp
import torch

from . import _macslam  # noqa: F401
from Src.Utility.Enum import BodyNodeStatus

from .odometry_state import MapPoints, OdometryState
from .vio_state import FrameResult

if TYPE_CHECKING:
    from Src.DataLoader.Dataset.OffroadRig import StereoRig
    from Src.Module.IMU.IMUChain import VIOStatus
    from Src.Module.Map import StereoVisualInertialMap
    from Src.Odometry.MACVIO import MACVIO


def map_points_from_map(map: StereoVisualInertialMap, T_body_base: pp.LieTensor) -> MapPoints:
    """Map points of the newest stereo node, moved from its camera (NED) frame to the base frame."""
    stereo_idx = torch.tensor([len(map.stereo) - 1])
    point_idx = map.stereo2map.neighbors(stereo_idx)
    T_body_camera = pp.SE3(map.stereo.data["T_BS"][stereo_idx].reshape(7).double())
    T_base_camera = pp.SE3(T_body_base.tensor().reshape(7).double()).Inv() @ T_body_camera
    R = T_base_camera.rotation().matrix()

    points = map.map.data["pos_Tc"][point_idx].detach().cpu().double()
    covariances = map.map.data["cov_Tc"][point_idx].detach().cpu().double()
    colors = map.map.data["color"][point_idx].detach().cpu().double()
    values = torch.cat([
        points @ R.mT + T_base_camera.translation(),
        colors,
        (R @ covariances @ R.mT).reshape(-1, 9),
    ], dim=1)
    return MapPoints(stamp_ns=int(map.stereo.data["time_ns"][stereo_idx].item()),
                     values=values.to(torch.float32).numpy())


def processed_image(image: torch.Tensor) -> np.ndarray:
    """Channel 0 of a preprocessed 1x3xHxW [0, 1] image as HxW mono8, quantized like map point colors."""
    return (image[0, 0].clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()


def rect_rotation(rig: StereoRig) -> np.ndarray:
    """Raw left optical <- rectified left optical rotation as (x, y, z, w)."""
    if rig.undistort is None:
        raise ValueError(f"the {rig.platform} {rig.modality} rig takes rectified input and has no rectification")
    # OpenCV R1 is rectified <- raw optical
    R_raw_rect = torch.tensor(rig.undistort[0].R, dtype=torch.float64).mT
    return pp.mat2SO3(R_raw_rect).tensor().numpy()


def odometry_from_map(
    map: StereoVisualInertialMap, T_body_base: pp.LieTensor, fd_twist_variance: float
) -> OdometryState:
    """Newest body state as World <- base pose and base-frame twist.

    `T_body_base` is Body (IMU) <- base. When the IMU velocity is unknown (NaN), the twist is a
    finite difference of the last two body poses with body-frame covariance `fd_twist_variance * I`.
    """
    T_body_base = pp.SE3(T_body_base.tensor().reshape(7).double())
    pose = map.body.data["pose"][-1].detach().cpu().double()
    pose_cov = map.body.data["Tw_cov_pose"][-1].detach().cpu().double()
    if not (torch.isfinite(pose).all() and torch.isfinite(pose_cov).all()):
        raise ValueError("MACVIO produced a non-finite pose")
    # A left-perturbation covariance is unchanged by a constant right factor
    T_world_base = pp.SE3(pose) @ T_body_base

    imu_idx = map.body2imu.neighbors(torch.tensor([len(map.body) - 1]))
    v_sensor = map.imu.data["v_Tbs"][imu_idx].reshape(3).detach().cpu().double()
    omega = map.imu.data["ω_Tbs"][imu_idx].reshape(3).detach().cpu().double()
    from_poses = not bool(torch.isfinite(v_sensor).all() and torch.isfinite(omega).all())
    if from_poses:
        v_body, omega = _finite_difference_twist(map)
        twist_cov_body = fd_twist_variance * torch.eye(6, dtype=torch.float64)
    else:
        t_body_sensor = pp.SE3(map.imu.data["T_BS"][imu_idx].reshape(7).double()).translation()
        v_body = v_sensor - torch.linalg.cross(omega, t_body_sensor)
        twist_cov_body = torch.zeros(6, 6, dtype=torch.float64)
        twist_cov_body[:3, :3] = map.imu.data["v_covTb"][imu_idx].reshape(3, 3).detach().cpu().double()

    twist, twist_cov = _twist_to_base(v_body, omega, twist_cov_body, T_body_base)
    return OdometryState(
        stamp_ns=int(map.body.data["time_ns"][-1].item()),
        pose=T_world_base.tensor().numpy(),
        pose_cov=_left_se3_to_ros_cov(pose_cov, T_world_base.translation()).numpy(),
        twist=twist.numpy(),
        twist_cov=twist_cov.numpy(),
        twist_from_poses=from_poses,
    )


def frame_result(system: MACVIO) -> FrameResult:
    """Estimator state after the frame it just ran."""
    body = system.Map.body
    status = system.vio_status
    n_tracking_restarts, last_frame_gap_s = _camera_gap_fields(status)
    return FrameResult(
        stamp_ns=int(body.data["time_ns"][-1].item()),
        initialized=system.imu_initcond is not None,
        optimized=bool(int(body.data["status"][-1].item()) & BodyNodeStatus.Optimized),
        lost_track=bool(system.lost_track),
        vio_mode=status.mode.name,
        frames_in_mode=status.frames_in_mode,
        last_gap_s=status.last_gap_s,
        n_chain_breaks=status.n_chain_breaks,
        n_dropped_samples=status.n_dropped_samples,
        n_tracking_restarts=n_tracking_restarts,
        last_frame_gap_s=last_frame_gap_s,
    )


def _camera_gap_fields(status: VIOStatus) -> tuple[int, float]:
    return status.n_tracking_restarts, status.last_frame_gap_s


def _finite_difference_twist(map: StereoVisualInertialMap) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant body twist (linear, angular) carrying the second-newest body pose to the newest."""
    if len(map.body) < 2:
        raise ValueError("finite-difference twist needs two body poses")
    poses = map.body.data["pose"][-2:].detach().cpu().double()
    times = map.body.data["time_ns"][-2:].flatten()
    dt_s = int(times[1] - times[0]) / 1e9
    if dt_s <= 0.0:
        raise ValueError("finite-difference twist needs increasing body stamps")
    xi = (pp.SE3(poses[0]).Inv() @ pp.SE3(poses[1])).Log().tensor() / dt_s
    return xi[:3], xi[3:]


def _twist_to_base(
    v_body: torch.Tensor, omega_body: torch.Tensor, cov_body: torch.Tensor, T_body_base: pp.LieTensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Body-origin twist in Body -> base-origin twist in base, with its covariance."""
    R_base_body = T_body_base.rotation().matrix().mT
    t = T_body_base.translation()
    twist = torch.cat([R_base_body @ (v_body + torch.linalg.cross(omega_body, t)), R_base_body @ omega_body])
    J = torch.zeros(6, 6, dtype=torch.float64)
    J[:3, :3] = R_base_body
    J[:3, 3:] = -R_base_body @ pp.vec2skew(t)
    J[3:, 3:] = R_base_body
    return twist, J @ cov_body @ J.mT


def _left_se3_to_ros_cov(cov: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Left-perturbation se3 covariance [rho, phi] -> ROS (world position, world fixed-axis rotation).

    With T = Exp([rho, phi]) T_bar, the position error is rho - [p]x phi.
    """
    J = torch.eye(6, dtype=torch.float64)
    J[:3, 3:] = -pp.vec2skew(position)
    return J @ cov @ J.mT
