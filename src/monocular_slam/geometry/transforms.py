"""SE(3) and SO(3) transform utilities.

Conventions used throughout the project
---------------------------------------
* A pose is a ``4x4`` homogeneous matrix ``T``.
* ``T_w_c`` maps points from the camera frame into the world frame:
  ``p_w = T_w_c @ p_c``. This matches the KITTI Odometry ground-truth format,
  where each row is the camera-to-world transform of the left camera.
* Camera axes follow the OpenCV/KITTI convention: ``+x`` right, ``+y`` down,
  ``+z`` forward. The ground plane is therefore ``x``-``z``, which is why
  top-down plots use X against Z.
* ``recoverPose`` returns ``(R, t)`` describing the *world-to-camera* motion of
  the second view relative to the first, i.e. ``T_c1_c2 = inv([R|t])``. That
  inversion is handled once, in :mod:`monocular_slam.odometry.visual_odometry`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

#: Numerical tolerance for orthonormality / determinant checks.
SO3_TOLERANCE = 1e-6


def se3_from_rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a ``4x4`` SE(3) matrix from a rotation and a translation.

    ``t`` may be shape ``(3,)``, ``(3, 1)`` or ``(1, 3)``.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def split_se3(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``4x4`` pose into ``(R, t)`` with ``t`` shaped ``(3,)``."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    return T[:3, :3].copy(), T[:3, 3].copy()


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) matrix analytically (``R^T``, ``-R^T t``).

    Cheaper and numerically better behaved than ``np.linalg.inv`` because it
    exploits orthonormality instead of solving a general 4x4 system.
    """
    R, t = split_se3(T)
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right composition of SE(3) matrices.

    ``compose(A, B, C)`` returns ``A @ B @ C``. With no arguments it returns
    the identity.
    """
    result = np.eye(4, dtype=np.float64)
    for T in transforms:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Each transform must be 4x4, got {T.shape}")
        result = result @ T
    return result


def relative_pose(T_w_a: np.ndarray, T_w_b: np.ndarray) -> np.ndarray:
    """Return ``T_a_b``: the pose of frame ``b`` expressed in frame ``a``."""
    return invert_se3(T_w_a) @ np.asarray(T_w_b, dtype=np.float64)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to an ``(N, 3)`` array of points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    R, t = split_se3(T)
    return points @ R.T + t


# --------------------------------------------------------------------------- #
# Validation and projection
# --------------------------------------------------------------------------- #


def is_rotation_matrix(R: np.ndarray, tol: float = SO3_TOLERANCE) -> bool:
    """Check that ``R`` is orthonormal with ``det(R) == +1``.

    Rejects reflections (``det == -1``), which is exactly the failure mode of a
    badly conditioned essential-matrix decomposition.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.all(np.isfinite(R)):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return abs(np.linalg.det(R) - 1.0) <= max(tol, 1e-9)


def is_valid_se3(T: np.ndarray, tol: float = SO3_TOLERANCE) -> bool:
    """Check shape, finiteness, rotation validity and the bottom row."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        return False
    return is_rotation_matrix(T[:3, :3], tol=tol)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Return the closest rotation matrix to ``R`` in the Frobenius sense.

    Standard orthogonal Procrustes: ``R = U V^T`` from the SVD, with the sign
    of the last singular vector flipped if that would otherwise produce a
    reflection. Used to clean up drift-accumulated rotations before handing
    them to GTSAM, which requires exact SO(3) members.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be 3x3, got {R.shape}")
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj


def project_to_se3(T: np.ndarray) -> np.ndarray:
    """Re-orthonormalise the rotation block of a pose and fix its bottom row."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = project_to_so3(T[:3, :3])
    out[:3, 3] = T[:3, 3]
    return out


# --------------------------------------------------------------------------- #
# Angles, logs and exponentials
# --------------------------------------------------------------------------- #


def rotation_angle_rad(R: np.ndarray) -> float:
    """Geodesic rotation angle of ``R`` in radians, in ``[0, pi]``.

    Computed from the trace with clipping, which is stable for the small
    rotations dominating frame-to-frame motion.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 or 4x4 matrix, got {R.shape}")
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def rotation_angle_deg(R: np.ndarray) -> float:
    """Geodesic rotation angle of ``R`` in degrees."""
    return float(np.degrees(rotation_angle_rad(R)))


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map SO(3) -> ``R^3`` (rotation vector, axis * angle)."""
    return Rotation.from_matrix(project_to_so3(R)).as_rotvec()


def so3_exp(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map ``R^3`` -> SO(3)."""
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map SE(3) -> ``R^6`` ordered ``[rho (3), phi (3)]``.

    ``phi`` is the rotation vector and ``rho`` the translation coordinate in
    the Lie algebra, i.e. ``t = V(phi) @ rho``. Provided for completeness and
    for uncertainty-weighted comparisons; the pipeline mostly needs separate
    translation and rotation errors, which are easier to interpret.
    """
    R, t = split_se3(T)
    phi = so3_log(R)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        V_inv = np.eye(3) - 0.5 * _skew(phi)
    else:
        a = phi / theta
        A = _skew(a)
        half = theta / 2.0
        cot_half = 1.0 / np.tan(half)
        V_inv = (
            half * cot_half * np.eye(3)
            + (1.0 - half * cot_half) * np.outer(a, a)
            - half * A
        )
    rho = V_inv @ t
    return np.concatenate([rho, phi])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map ``R^6`` -> SE(3), inverse of :func:`se3_log`."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho, phi = xi[:3], xi[3:]
    theta = float(np.linalg.norm(phi))
    R = so3_exp(phi)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * _skew(phi)
    else:
        A = _skew(phi / theta)
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta) * A
            + ((theta - np.sin(theta)) / theta) * (A @ A)
        )
    return se3_from_rt(R, V @ rho)


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``_skew(v) @ w == np.cross(v, w)``."""
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# KITTI serialisation
# --------------------------------------------------------------------------- #


def se3_to_kitti_row(T: np.ndarray) -> np.ndarray:
    """Flatten a pose to KITTI's 12-value row-major ``[R|t]`` layout."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be 4x4, got {T.shape}")
    return T[:3, :4].reshape(12).copy()


def kitti_row_to_se3(row: np.ndarray) -> np.ndarray:
    """Expand a KITTI 12-value pose row into a ``4x4`` matrix."""
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    if row.size != 12:
        raise ValueError(f"KITTI pose rows must hold 12 values, got {row.size}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = row.reshape(3, 4)
    return T
