"""Plain odometry record shared by the MACVIO core, the localization gate and the ROS shells."""

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
