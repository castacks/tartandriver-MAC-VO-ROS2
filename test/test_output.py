"""Body (IMU) state in the MACVIO map -> base-frame odometry."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from mac_ros2.output import _camera_gap_fields, odometry_from_map
from Src.Module.Map import IMUNode
from Src.Scripts.AutoTest.imu_dropout_fixtures import new_map, push_body

NAN3 = (float("nan"),) * 3
YAMAHA_T_BODY_BASE = pp.SE3(torch.tensor([0.508, 0.05, 0.152, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
# 180 deg about y, as the Microstrain is mounted on the Stallion
ROTATED_T_BODY_BASE = pp.SE3(torch.tensor([2.5, 0.01, 1.2, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
T_WB = pp.SE3(torch.tensor([10.0, 5.0, 1.0, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476], dtype=torch.float64))


def push_state(m, t_ns: int, pose: pp.LieTensor, v=(1.0, 0.2, 0.0), w=(0.0, 0.0, 0.3), cov=None) -> None:
    body = push_body(m, t_ns, pp.SE3(pose.tensor().view(1, 7)))
    if cov is not None:
        m.body.data["Tw_cov_pose"][body] = cov.view(1, 6, 6)
    imu = m.imu.push(IMUNode.init({
        "time_ns": torch.tensor([t_ns], dtype=torch.long),
        "T_BS": pp.identity_SE3(1, dtype=torch.float64).tensor(),
        "v_Tbs": torch.tensor([v], dtype=torch.float64),
        "v_covTb": torch.diag(torch.tensor([0.01, 0.02, 0.03], dtype=torch.float64)).unsqueeze(0),
        "ω_Tbs": torch.tensor([w], dtype=torch.float64),
        "gyro_Ts": torch.zeros(1, 3, dtype=torch.float64),
        "bias_acc": torch.zeros(1, 3, dtype=torch.float64),
        "bias_gyro": torch.zeros(1, 3, dtype=torch.float64),
        "bias_acc_stability": torch.tensor([1e-3], dtype=torch.float64),
        "bias_gyro_stability": torch.tensor([1e-4], dtype=torch.float64),
    }))
    m.body2imu.set(body, imu)
    m.imu2body.set(imu, body)


def rigid_base_twist(T_body_base: pp.LieTensor, v_body: np.ndarray, w_body: np.ndarray) -> np.ndarray:
    """Base-origin velocity and rate in the base frame, from the rigid-body velocity field."""
    R = T_body_base.rotation().matrix().numpy()
    t = T_body_base.translation().numpy()
    return np.concatenate([R.T @ (v_body + np.cross(w_body, t)), R.T @ w_body])


@pytest.mark.parametrize("T_body_base", [YAMAHA_T_BODY_BASE, ROTATED_T_BODY_BASE], ids=["yamaha", "rotated"])
def test_pose_and_twist_in_base_frame(T_body_base):
    m = new_map()
    push_state(m, 1_000, T_WB, v=(1.0, 0.2, -0.1), w=(0.05, -0.02, 0.3))
    odom = odometry_from_map(m, T_body_base, fd_twist_variance=1.0)

    expected_pose = (T_WB @ T_body_base).matrix().numpy()
    got = pp.SE3(torch.from_numpy(odom.pose)).matrix().numpy()
    np.testing.assert_allclose(got, expected_pose, atol=1e-12)
    np.testing.assert_allclose(
        odom.twist, rigid_base_twist(T_body_base, np.array([1.0, 0.2, -0.1]), np.array([0.05, -0.02, 0.3])),
        atol=1e-12,
    )
    R = T_body_base.rotation().matrix().numpy()
    np.testing.assert_allclose(odom.twist_cov[:3, :3], R.T @ np.diag([0.01, 0.02, 0.03]) @ R, atol=1e-15)
    assert odom.stamp_ns == 1_000 and not odom.twist_from_poses


def test_pose_covariance_is_world_position_then_fixed_axis_rotation():
    """Numerically differentiate T = Exp(xi) T_bar into (position error, world rotation error)."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    cov = torch.from_numpy(A @ A.T * 1e-3)
    m = new_map()
    push_state(m, 1_000, T_WB, cov=cov)
    odom = odometry_from_map(m, YAMAHA_T_BODY_BASE, fd_twist_variance=1.0)

    T_bar = T_WB @ YAMAHA_T_BODY_BASE
    eps = 1e-7
    J = np.zeros((6, 6))
    for i in range(6):
        xi = torch.zeros(6, dtype=torch.float64)
        xi[i] = eps
        T = pp.se3(xi).Exp() @ T_bar
        J[:3, i] = ((T.translation() - T_bar.translation()) / eps).numpy()
        J[3:, i] = ((T.rotation() @ T_bar.rotation().Inv()).Log().tensor() / eps).numpy()
    np.testing.assert_allclose(odom.pose_cov, J @ cov.numpy() @ J.T, atol=1e-8)


@pytest.mark.parametrize("v,w", [(NAN3, (0.0, 0.0, 0.3)), (NAN3, NAN3), ((1.0, 0.0, 0.0), NAN3)])
def test_unknown_imu_velocity_uses_pose_finite_difference(v, w):
    xi = torch.tensor([2.0, 0.3, -0.1, 0.02, -0.01, 0.25], dtype=torch.float64)
    dt_ns = 100_000_000
    T_next = T_WB @ pp.se3(xi * dt_ns / 1e9).Exp()
    m = new_map()
    push_state(m, 1_000_000_000, T_WB)
    push_state(m, 1_000_000_000 + dt_ns, T_next, v=v, w=w)
    odom = odometry_from_map(m, ROTATED_T_BODY_BASE, fd_twist_variance=0.5)

    assert odom.twist_from_poses
    np.testing.assert_allclose(odom.twist, rigid_base_twist(ROTATED_T_BODY_BASE, xi[:3].numpy(), xi[3:].numpy()),
                               atol=1e-9)
    assert np.isfinite(odom.twist_cov).all()
    np.testing.assert_allclose(odom.twist_cov[3:, 3:], 0.5 * np.eye(3), atol=1e-15)
    assert np.all(np.diag(odom.twist_cov)[:3] >= 0.5)


def test_finite_difference_needs_two_bodies():
    m = new_map()
    push_state(m, 1_000, T_WB, v=NAN3, w=NAN3)
    with pytest.raises(ValueError, match="two body poses"):
        odometry_from_map(m, YAMAHA_T_BODY_BASE, fd_twist_variance=1.0)


@pytest.mark.parametrize("field", ["pose", "Tw_cov_pose"])
def test_non_finite_pose_raises(field):
    m = new_map()
    push_state(m, 1_000, T_WB)
    m.body.data[field][0] = torch.full_like(m.body.data[field][0], float("nan"))
    with pytest.raises(ValueError, match="non-finite pose"):
        odometry_from_map(m, YAMAHA_T_BODY_BASE, fd_twist_variance=1.0)


def test_camera_gap_fields_default_when_missing():
    assert _camera_gap_fields(SimpleNamespace()) == (False, 0, 0.0)
    status = SimpleNamespace(segment_pending=True, n_segments=2, last_frame_gap_s=1.5)
    assert _camera_gap_fields(status) == (True, 2, 1.5)
