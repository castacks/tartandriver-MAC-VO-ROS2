"""MACVIO map -> base-frame odometry and map points; processed image and rectification rotation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pypose as pp
import pytest
import torch

from mac_ros2.odometry_state import MAP_POINT_FIELDS
from mac_ros2.output import _camera_gap_fields, map_points_from_map, odometry_from_map, processed_image, rect_rotation
from Src.DataLoader import get_rig
from Src.DataLoader.Rig import R_OPTICAL_NED
from Src.Module.Map import IMUNode, PointNode, StereoNode
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


def push_stereo_points(m, t_ns: int, T_BS: pp.LieTensor, positions, covariances, colors) -> None:
    """A body + stereo node at `t_ns` whose new map points are given in its camera NED frame."""
    body = push_body(m, t_ns, T_WB.unsqueeze(0))
    stereo = m.stereo.push(StereoNode.init({
        "K": torch.eye(3, dtype=torch.float32).unsqueeze(0),
        "baseline": torch.tensor([0.3], dtype=torch.float32),
        "T_BS": T_BS.tensor().view(1, 7),
        "time_ns": torch.tensor([t_ns], dtype=torch.long),
    }))
    m.body2stereo.set(body, stereo)
    m.stereo2body.set(stereo, body)
    if len(positions):
        points = m.map.push(PointNode.init({
            "pos_Tc": torch.tensor(positions, dtype=torch.float32),
            "cov_Tc": torch.tensor(covariances, dtype=torch.float64),
            "color": torch.tensor(colors, dtype=torch.uint8),
            "age": torch.ones(len(positions), 1, dtype=torch.uint8),
        }))
        m.stereo2map.add(stereo, points[:1], points.size(0))
        m.map2stereo.set(points, stereo)


SIGMA_C = [[4.0, 0.3, 0.2], [0.3, 0.5, 0.1], [0.2, 0.1, 0.25]]


def test_map_point_fields_match_the_coordinator_feature_keys():
    assert MAP_POINT_FIELDS == ("x", "y", "z", "r", "g", "b", *(f"cov_{i}" for i in range(1, 10)))


def test_map_points_in_base_frame_by_hand():
    """Camera looking forward from a Stallion-like mount: NED (fwd, right, down) -> base FLU."""
    m = new_map()
    T_BS = pp.SE3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64))  # 180 deg about z
    push_stereo_points(m, 7_000, T_BS, [[10.0, 1.0, 2.0]], [SIGMA_C], [[10, 20, 30]])
    points = map_points_from_map(m, ROTATED_T_BODY_BASE)

    assert points.stamp_ns == 7_000 and points.values.dtype == np.float32 and points.values.shape == (1, 15)
    x, rgb, cov = points.values[0, :3], points.values[0, 3:6], points.values[0, 6:].reshape(3, 3)
    # 10 m ahead, 1 m right, 2 m below a camera at base (2.5, -0.01, 1.2)
    np.testing.assert_allclose(x, [12.5, -1.01, -0.8], atol=1e-5)
    np.testing.assert_array_equal(rgb, [10.0, 20.0, 30.0])
    flip = np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(cov, flip @ np.array(SIGMA_C) @ flip, atol=1e-6)
    assert cov[0, 1] == pytest.approx(-0.3) and cov[1, 2] == pytest.approx(0.1)


@pytest.mark.parametrize("platform,modality,rect_mode", [
    ("yamaha", "thermal", "stereo_rect"), ("stallion", "thermal", "stereo_rect"), ("stallion", "rgb", None),
])
def test_map_points_match_the_world_round_trip(platform, modality, rect_mode):
    """p_base = (T_wb T_body_base)^-1 T_wb T_BS p_c with the rig extrinsics; the body pose cancels."""
    rig = get_rig(platform, modality, rect_mode)
    rng = np.random.default_rng(3)
    positions = rng.normal(scale=5.0, size=(6, 3))
    A = rng.normal(size=(6, 3, 3))
    covariances = A @ A.transpose(0, 2, 1)
    m = new_map()
    push_stereo_points(m, 1_000, rig.T_BS, positions.tolist(), covariances.tolist(), [[1, 2, 3]] * 6)
    points = map_points_from_map(m, rig.T_body_base)

    T_world_base = T_WB @ pp.SE3(rig.T_body_base.tensor().reshape(7))
    T_world_camera = T_WB @ pp.SE3(rig.T_BS.tensor().reshape(7))
    p_world = T_world_camera.matrix().numpy() @ np.vstack([positions.T, np.ones(6)])
    expected = (np.linalg.inv(T_world_base.matrix().numpy()) @ p_world)[:3].T
    np.testing.assert_allclose(points.values[:, :3], expected, atol=1e-4)
    R = (T_world_base.Inv() @ T_world_camera).rotation().matrix().numpy()
    np.testing.assert_allclose(points.values[:, 6:].reshape(-1, 3, 3), R @ covariances @ R.T, rtol=1e-5, atol=1e-5)


def test_only_the_newest_stereo_node_points_are_returned():
    m = new_map()
    T_BS = pp.SE3(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    push_stereo_points(m, 1_000, T_BS, [[1.0, 0.0, 0.0]] * 3, [SIGMA_C] * 3, [[0, 0, 0]] * 3)
    push_stereo_points(m, 2_000, T_BS, [[2.0, 0.0, 0.0]] * 2, [SIGMA_C] * 2, [[0, 0, 0]] * 2)
    points = map_points_from_map(m, YAMAHA_T_BODY_BASE)
    assert points.stamp_ns == 2_000 and points.values.shape == (2, 15)
    np.testing.assert_allclose(points.values[:, 0], 2.0 - 0.508, atol=1e-6)

    push_stereo_points(m, 3_000, T_BS, [], [], [])
    empty = map_points_from_map(m, YAMAHA_T_BODY_BASE)
    assert empty.stamp_ns == 3_000 and empty.values.shape == (0, 15)


def test_processed_image_is_channel_zero_quantized():
    image = torch.zeros(1, 3, 2, 3)
    image[0, 0] = torch.tensor([[0.0, 0.5, 1.0], [-0.1, 1.2, 0.999]])
    image[0, 1:] = 0.7
    out = processed_image(image)
    assert out.dtype == np.uint8 and out.shape == (2, 3)
    np.testing.assert_array_equal(out, [[0, 127, 255], [0, 255, 254]])


def quaternion_matrix(xyzw: np.ndarray) -> np.ndarray:
    return pp.SO3(torch.tensor(xyzw, dtype=torch.float64)).matrix().numpy()


@pytest.mark.parametrize("platform", ["yamaha", "stallion"])
def test_rect_rotation_is_raw_from_rectified(platform):
    rig = get_rig(platform, "thermal", "stereo_rect")
    R_raw_rect = quaternion_matrix(rect_rotation(rig))
    R1 = rig.undistort[0].R
    np.testing.assert_allclose(R_raw_rect, R1.T, atol=1e-12)
    # A ray along the rectified optical axis, seen from the raw camera
    np.testing.assert_allclose(R_raw_rect @ [0.0, 0.0, 1.0], R1[2], atol=1e-12)


def test_rect_rotation_reproduces_the_yamaha_rectified_extrinsic():
    """Body <- rect NED = Body <- raw NED @ (raw <- rect, moved from optical to NED axes)."""
    rig = get_rig("yamaha", "thermal", "stereo_rect")
    R_raw_rect = R_OPTICAL_NED.T @ quaternion_matrix(rect_rotation(rig)) @ R_OPTICAL_NED
    expected = rig.T_BS_ref.rotation().matrix().numpy()[0] @ R_raw_rect
    np.testing.assert_allclose(rig.T_BS.rotation().matrix().numpy()[0], expected, atol=1e-9)


def test_stallion_rect_frame_is_the_driver_optical_frame():
    """The bag's left_camera_optical_frame -> left_camera_frame (raw) static TF is the inverse of ours."""
    rig = get_rig("stallion", "thermal", "stereo_rect")
    q_bag = np.array([0.00044, 0.00065, -0.00128, 1.0])
    R_optical_raw = quaternion_matrix(q_bag / np.linalg.norm(q_bag))
    np.testing.assert_allclose(quaternion_matrix(rect_rotation(rig)) @ R_optical_raw, np.eye(3), atol=2e-5)


def test_rect_rotation_needs_a_rectifying_rig():
    with pytest.raises(ValueError, match="no rectification"):
        rect_rotation(get_rig("yamaha", "rgb"))
