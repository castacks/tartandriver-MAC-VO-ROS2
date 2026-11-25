"""Plain records shared by the MACVIO core, the localization gate and the ROS shells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class OdometryState:
    """Parent <- child pose and child-frame twist, laid out as in ROS `nav_msgs/Odometry`.

    `pose_cov` orders (x, y, z, rot_x, rot_y, rot_z): position error and fixed-axis rotation error,
    both in the parent frame. `twist_cov` orders (vx, vy, vz, wx, wy, wz) in the child frame.
    """

    stamp_ns: int
    pose: np.ndarray       # (7,) x y z qx qy qz qw
    pose_cov: np.ndarray   # (6, 6)
    twist: np.ndarray      # (6,) vx vy vz wx wy wz
    twist_cov: np.ndarray  # (6, 6)
    twist_from_poses: bool = False  # twist is a finite difference of body poses


MAP_POINT_FIELDS = ("x", "y", "z", "r", "g", "b", *(f"cov_{i}" for i in range(1, 10)))


@dataclass(frozen=True, kw_only=True)
class MapPoints:
    """One frame's new map points in the base frame; `values` columns follow `MAP_POINT_FIELDS`.

    Colors are 0-255; `cov_1..cov_9` is the row-major 3x3 position covariance [m^2].
    """

    stamp_ns: int
    values: np.ndarray  # (N, 15) float32
