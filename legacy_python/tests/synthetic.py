"""Synthetic two-view geometry for testing the epipolar pipeline.

Generating correspondences from a known camera motion is the only way to test
essential-matrix estimation with a ground-truth answer, so sign conventions and
degenerate-case handling can be verified exactly rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from monocular_slam.geometry.transforms import invert_se3, se3_from_rt

#: KITTI sequence 00 intrinsics, so tests exercise realistic focal lengths.
KITTI_K = np.array(
    [[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]], dtype=np.float64
)


def project(points_cam: np.ndarray, K: np.ndarray = KITTI_K) -> np.ndarray:
    """Pinhole projection of ``(N, 3)`` camera-frame points to ``(N, 2)`` pixels."""
    homogeneous = points_cam @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def make_scene(
    n_points: int = 400,
    seed: int = 0,
    depth_range: tuple[float, float] = (5.0, 60.0),
    lateral: float = 15.0,
) -> np.ndarray:
    """Sample ``(N, 3)`` points in front of the camera (``+z`` forward)."""
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(-lateral, lateral, n_points),
            rng.uniform(-lateral / 3.0, lateral / 3.0, n_points),
            rng.uniform(*depth_range, n_points),
        ]
    )


def two_view_correspondences(
    T_c1_c2: np.ndarray,
    points_world: np.ndarray | None = None,
    K: np.ndarray = KITTI_K,
    noise_px: float = 0.0,
    seed: int = 0,
    image_size: tuple[int, int] | None = (1241, 376),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a scene into two views related by ``T_c1_c2``.

    Parameters
    ----------
    T_c1_c2:
        Pose of camera 2 expressed in camera 1's frame — the same convention
        :func:`~monocular_slam.geometry.epipolar.recover_relative_pose` returns.
    image_size:
        When given, correspondences falling outside the image in either view
        are dropped, mimicking a real detector.

    Returns
    -------
    (points_a, points_b, points_cam1)
        Pixel correspondences plus the surviving 3D points in camera 1's frame.
    """
    points_cam1 = make_scene(seed=seed) if points_world is None else np.asarray(points_world)

    T_c2_c1 = invert_se3(T_c1_c2)
    points_cam2 = points_cam1 @ T_c2_c1[:3, :3].T + T_c2_c1[:3, 3]

    in_front = (points_cam1[:, 2] > 0.5) & (points_cam2[:, 2] > 0.5)
    points_cam1 = points_cam1[in_front]
    points_cam2 = points_cam2[in_front]

    points_a = project(points_cam1, K)
    points_b = project(points_cam2, K)

    if image_size is not None:
        width, height = image_size
        inside = (
            (points_a[:, 0] >= 0)
            & (points_a[:, 0] < width)
            & (points_a[:, 1] >= 0)
            & (points_a[:, 1] < height)
            & (points_b[:, 0] >= 0)
            & (points_b[:, 0] < width)
            & (points_b[:, 1] >= 0)
            & (points_b[:, 1] < height)
        )
        points_a, points_b, points_cam1 = points_a[inside], points_b[inside], points_cam1[inside]

    if noise_px > 0:
        rng = np.random.default_rng(seed + 977)
        points_a = points_a + rng.normal(scale=noise_px, size=points_a.shape)
        points_b = points_b + rng.normal(scale=noise_px, size=points_b.shape)

    return points_a, points_b, points_cam1


def forward_motion(
    distance: float = 0.9, yaw_deg: float = 0.0, lateral: float = 0.0
) -> np.ndarray:
    """A KITTI-like step: drive forward along ``+z`` with an optional yaw."""
    R = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    return se3_from_rt(R, np.array([lateral, 0.0, distance]))


def pure_rotation(yaw_deg: float = 3.0) -> np.ndarray:
    """A degenerate motion: rotation with no translation."""
    return se3_from_rt(Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix(), np.zeros(3))


def straight_drive(n_frames: int = 40, step: float = 0.9, yaw_rate_deg: float = 0.8) -> np.ndarray:
    """``(N, 4, 4)`` ground-truth camera-to-world poses for a curving drive."""
    poses = [np.eye(4)]
    for _ in range(n_frames - 1):
        poses.append(poses[-1] @ forward_motion(step, yaw_rate_deg))
    return np.stack(poses)
