"""Two-view epipolar geometry: essential matrix and relative pose.

Given calibrated correspondences between two views, the essential matrix ``E``
satisfies ``x2^T E x1 = 0`` for normalized image coordinates. Decomposing ``E``
yields four candidate ``(R, t)`` poses, of which exactly one places the
triangulated points in front of both cameras; OpenCV's ``recoverPose`` performs
that cheirality test.

Two things this module refuses to take on trust from OpenCV:

* **The returned matrices may be garbage.** ``findEssentialMat`` can return
  ``None``, a stack of several candidate matrices, or a matrix containing
  ``NaN``. ``recoverPose`` can return a rotation with ``det = -1``. Every
  output is validated before it reaches the trajectory.
* **Degenerate geometry is silently accepted.** Under pure rotation, or when a
  car is stopped at a traffic light, the translation direction is
  unobservable and ``E`` is ill-conditioned. Median parallax is measured and
  reported so the caller can reject those transitions instead of integrating
  noise.

Sign convention
---------------
``cv2.recoverPose`` returns ``(R, t)`` such that a point expressed in the first
camera frame maps into the second as ``x2 = R x1 + t`` — that is, ``[R|t]`` is
``T_c2_c1``. The camera *motion* needed to walk the trajectory forward is its
inverse, ``T_c1_c2``, and that is what :func:`recover_relative_pose` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from monocular_slam.geometry.transforms import (
    is_rotation_matrix,
    project_to_so3,
    rotation_angle_deg,
    se3_from_rt,
)
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EssentialMatrixResult:
    """Outcome of RANSAC essential-matrix estimation."""

    E: np.ndarray | None
    inlier_mask: np.ndarray
    n_points: int
    ok: bool
    reason: str = ""

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        return float(self.n_inliers / self.n_points) if self.n_points else 0.0

    def __repr__(self) -> str:
        status = "ok" if self.ok else f"rejected({self.reason})"
        return f"EssentialMatrixResult({status}, inliers={self.n_inliers}/{self.n_points})"


@dataclass
class RelativePoseResult:
    """Outcome of decomposing an essential matrix into a relative pose.

    Attributes
    ----------
    T:
        ``T_c1_c2``: the pose of the second camera in the first camera's frame,
        with **unit-norm translation**. Monocular geometry cannot determine the
        magnitude; see :mod:`monocular_slam.odometry.scale`.
    n_cheirality_inliers:
        Correspondences that triangulate in front of both cameras.
    median_parallax_deg:
        Median angle between corresponding calibrated rays. Near zero means the
        motion is (close to) pure rotation and the translation direction is
        not observable.
    """

    T: np.ndarray | None
    R: np.ndarray | None
    t_unit: np.ndarray | None
    inlier_mask: np.ndarray
    n_cheirality_inliers: int
    median_parallax_deg: float
    rotation_deg: float
    ok: bool
    reason: str = ""

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    def __repr__(self) -> str:
        status = "ok" if self.ok else f"rejected({self.reason})"
        return (
            f"RelativePoseResult({status}, inliers={self.n_inliers}, "
            f"rot={self.rotation_deg:.2f} deg, parallax={self.median_parallax_deg:.3f} deg)"
        )


def estimate_essential_matrix(
    points_a: np.ndarray,
    points_b: np.ndarray,
    K: np.ndarray,
    *,
    threshold_px: float = 1.0,
    confidence: float = 0.999,
    max_iters: int = 2000,
    min_points: int = 8,
) -> EssentialMatrixResult:
    """Estimate the essential matrix with RANSAC.

    The threshold is expressed in **pixels** and passed alongside ``K`` so
    OpenCV performs the normalization internally — this keeps the geometric
    meaning of the threshold intact regardless of focal length, which a
    hand-normalized threshold quietly loses.

    Returns a result whose ``ok`` flag is ``False`` (with a ``reason``) rather
    than raising, so a single bad frame pair cannot abort a long run.
    """
    points_a = np.ascontiguousarray(points_a, dtype=np.float64)
    points_b = np.ascontiguousarray(points_b, dtype=np.float64)
    n = len(points_a)
    empty_mask = np.zeros(n, dtype=bool)

    if points_a.shape != points_b.shape or points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError(
            f"points must be matching (N, 2) arrays, got {points_a.shape} and {points_b.shape}"
        )
    if n < min_points:
        return EssentialMatrixResult(None, empty_mask, n, False, f"too_few_points({n})")

    E, mask = cv2.findEssentialMat(
        points_a,
        points_b,
        cameraMatrix=np.asarray(K, dtype=np.float64),
        method=cv2.RANSAC,
        prob=float(confidence),
        threshold=float(threshold_px),
        maxIters=int(max_iters),
    )

    if E is None or E.size == 0:
        return EssentialMatrixResult(None, empty_mask, n, False, "no_solution")

    # The 5-point solver can return several stacked candidates (3k x 3) when
    # the configuration is degenerate. OpenCV orders them best-first.
    if E.shape[0] > 3:
        E = E[:3]
    if E.shape != (3, 3):
        return EssentialMatrixResult(None, empty_mask, n, False, f"bad_shape{E.shape}")
    if not np.all(np.isfinite(E)):
        return EssentialMatrixResult(None, empty_mask, n, False, "non_finite")

    inlier_mask = (
        empty_mask if mask is None else np.asarray(mask, dtype=np.int32).ravel().astype(bool)
    )
    if len(inlier_mask) != n:  # pragma: no cover - OpenCV always matches lengths
        inlier_mask = empty_mask

    return EssentialMatrixResult(E.astype(np.float64), inlier_mask, n, True, "")


def recover_relative_pose(
    E: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    K: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    *,
    min_inliers: int = 30,
    min_inlier_ratio: float = 0.3,
    max_rotation_deg: float = 30.0,
    min_parallax_deg: float = 0.0,
) -> RelativePoseResult:
    """Decompose ``E`` into ``T_c1_c2`` and validate the result.

    Validation performed, in order:

    * ``R`` must be a finite, right-handed rotation (``det = +1``).
    * ``t`` must be finite and non-degenerate before normalization.
    * The cheirality test must retain ``min_inliers`` correspondences and at
      least ``min_inlier_ratio`` of the input.
    * The rotation must be under ``max_rotation_deg`` — at KITTI's 10 Hz a
      larger inter-frame rotation is a decomposition failure, not real motion.
    * Median parallax must exceed ``min_parallax_deg`` when that guard is on.
    """
    K = np.asarray(K, dtype=np.float64)
    points_a = np.ascontiguousarray(points_a, dtype=np.float64)
    points_b = np.ascontiguousarray(points_b, dtype=np.float64)
    n = len(points_a)

    def failure(reason: str, mask: np.ndarray | None = None) -> RelativePoseResult:
        return RelativePoseResult(
            T=None,
            R=None,
            t_unit=None,
            inlier_mask=np.zeros(n, dtype=bool) if mask is None else mask,
            n_cheirality_inliers=0,
            median_parallax_deg=float("nan"),
            rotation_deg=float("nan"),
            ok=False,
            reason=reason,
        )

    if E is None:
        return failure("no_essential_matrix")

    # recoverPose mutates the mask in place, turning RANSAC inliers into
    # cheirality inliers, so a writable int32 copy is required.
    if inlier_mask is None:
        pose_mask = np.ones((n, 1), dtype=np.uint8)
    else:
        pose_mask = np.asarray(inlier_mask).astype(np.uint8).reshape(n, 1).copy()

    n_good, R, t, pose_mask = cv2.recoverPose(
        np.asarray(E, dtype=np.float64), points_a, points_b, K, mask=pose_mask
    )
    cheirality_mask = np.asarray(pose_mask).ravel().astype(bool)

    if R is None or t is None or not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
        return failure("non_finite_pose")

    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    if not is_rotation_matrix(R, tol=1e-4):
        return failure("invalid_rotation")
    # Snap tiny numerical drift away so the pose is an exact SO(3) member.
    R = project_to_so3(R)

    t_norm = float(np.linalg.norm(t))
    if t_norm < 1e-9:
        return failure("degenerate_translation")
    t_unit = t / t_norm

    n_cheirality = int(np.count_nonzero(cheirality_mask))
    ratio = n_cheirality / n if n else 0.0
    rotation_deg = rotation_angle_deg(R)

    # T_c2_c1 = [R|t]; the forward camera motion is its inverse.
    T_c1_c2 = se3_from_rt(R.T, -R.T @ t_unit)
    parallax = median_parallax_deg(points_a, points_b, K, T_c1_c2[:3, :3], cheirality_mask)

    result = RelativePoseResult(
        T=T_c1_c2,
        R=R,
        t_unit=t_unit,
        inlier_mask=cheirality_mask,
        n_cheirality_inliers=n_cheirality,
        median_parallax_deg=parallax,
        rotation_deg=rotation_deg,
        ok=True,
    )

    if n_cheirality < min_inliers:
        result.ok, result.reason = False, f"too_few_inliers({n_cheirality})"
    elif ratio < min_inlier_ratio:
        result.ok, result.reason = False, f"low_inlier_ratio({ratio:.2f})"
    elif rotation_deg > max_rotation_deg:
        result.ok, result.reason = False, f"implausible_rotation({rotation_deg:.1f}deg)"
    elif min_parallax_deg > 0.0 and parallax < min_parallax_deg:
        result.ok, result.reason = False, f"low_parallax({parallax:.3f}deg)"

    del n_good  # recoverPose's own count; cheirality_mask is the authority
    return result


def median_parallax_deg(
    points_a: np.ndarray,
    points_b: np.ndarray,
    K: np.ndarray,
    R_c1_c2: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """Median parallax angle between corresponding viewing rays, in degrees.

    Parallax is the angular disagreement between the two rays *after the
    camera rotation has been removed*: each ray from the second view is
    rotated into the first camera's frame via ``R_c1_c2`` before the
    comparison. Skipping that step measures rotation rather than parallax, and
    would report a large value for a purely rotating camera — exactly the case
    this metric exists to catch.

    With rotation compensated, near-zero parallax means the two views share an
    optical centre (stationary or pure-rotation motion), so the translation
    direction is unobservable and the essential matrix is ill-conditioned.

    ``R_c1_c2`` may be omitted when the rotation is known to be negligible.
    """
    if mask is not None:
        points_a = points_a[mask]
        points_b = points_b[mask]
    if len(points_a) == 0:
        return float("nan")

    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    rays_a = _to_rays(points_a, K_inv)
    rays_b = _to_rays(points_b, K_inv)
    if R_c1_c2 is not None:
        rays_b = rays_b @ np.asarray(R_c1_c2, dtype=np.float64).T
    cosines = np.clip(np.sum(rays_a * rays_b, axis=1), -1.0, 1.0)
    return float(np.degrees(np.median(np.arccos(cosines))))


def _to_rays(points: np.ndarray, K_inv: np.ndarray) -> np.ndarray:
    """Back-project ``(N, 2)`` pixels to unit-norm camera rays."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    rays = homogeneous @ K_inv.T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


@dataclass
class ModelSelection:
    """Homography-vs-essential comparison used to detect degenerate two-view geometry.

    Two configurations break monocular essential-matrix estimation:

    * **Pure rotation / stationary camera.** With no translation the essential
      matrix ``E = [t]_x R`` collapses to zero and RANSAC fits noise. The
      recovered rotation is then wrong too, which is why rotation-compensated
      parallax cannot detect this case — it de-rotates by a bogus rotation.
    * **Planar scene.** When all correspondences lie on one plane (a road
      surface filling the frame) the essential matrix is under-constrained.

    In both cases a homography explains the correspondences at least as well as
    the epipolar model. Following the model-selection heuristic popularised by
    ORB-SLAM, both models are scored by the same robust criterion and compared
    via ``ratio_h = S_H / (S_H + S_E)``. A high ratio means "a homography is
    sufficient", i.e. the two-view translation is not reliably observable.
    """

    score_h: float
    score_e: float
    ratio_h: float
    is_degenerate: bool
    H: np.ndarray | None = field(default=None, repr=False)


#: Chi-squared 95% thresholds. A homography residual has 2 degrees of freedom,
#: an epipolar (point-to-line) residual has 1.
_CHI2_H = 5.991
_CHI2_E = 3.841


def select_two_view_model(
    points_a: np.ndarray,
    points_b: np.ndarray,
    K: np.ndarray,
    E: np.ndarray | None,
    *,
    threshold_px: float = 1.0,
    ratio_threshold: float = 0.45,
) -> ModelSelection:
    """Score a homography against the essential matrix for the same matches."""
    points_a = np.ascontiguousarray(points_a, dtype=np.float64)
    points_b = np.ascontiguousarray(points_b, dtype=np.float64)
    if len(points_a) < 4 or E is None:
        return ModelSelection(0.0, 0.0, 0.0, False, None)

    sigma_sq = float(threshold_px) ** 2

    H, _ = cv2.findHomography(points_a, points_b, method=cv2.RANSAC, ransacReprojThreshold=threshold_px)
    score_h = 0.0 if H is None else _score_homography(H, points_a, points_b, sigma_sq)

    K = np.asarray(K, dtype=np.float64)
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ np.asarray(E, dtype=np.float64) @ K_inv
    score_e = _score_fundamental(F, points_a, points_b, sigma_sq)

    total = score_h + score_e
    ratio_h = float(score_h / total) if total > 0 else 0.0
    return ModelSelection(
        score_h=float(score_h),
        score_e=float(score_e),
        ratio_h=ratio_h,
        is_degenerate=ratio_h > float(ratio_threshold),
        H=H,
    )


def _score_homography(
    H: np.ndarray, points_a: np.ndarray, points_b: np.ndarray, sigma_sq: float
) -> float:
    """Robust score of a homography via symmetric transfer error."""
    forward = _apply_homography(H, points_a)
    backward = _apply_homography(np.linalg.inv(H), points_b)
    chi2_forward = np.sum((points_b - forward) ** 2, axis=1) / sigma_sq
    chi2_backward = np.sum((points_a - backward) ** 2, axis=1) / sigma_sq
    return _robust_score(chi2_forward, _CHI2_H) + _robust_score(chi2_backward, _CHI2_H)


def _score_fundamental(
    F: np.ndarray, points_a: np.ndarray, points_b: np.ndarray, sigma_sq: float
) -> float:
    """Robust score of a fundamental matrix via symmetric epipolar distance."""
    ha = np.column_stack([points_a, np.ones(len(points_a))])
    hb = np.column_stack([points_b, np.ones(len(points_b))])

    lines_b = ha @ F.T  # epipolar lines in image b
    lines_a = hb @ F  # epipolar lines in image a
    residual = np.sum(hb * lines_b, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_b = residual**2 / (np.sum(lines_b[:, :2] ** 2, axis=1) * sigma_sq)
        chi2_a = residual**2 / (np.sum(lines_a[:, :2] ** 2, axis=1) * sigma_sq)
    chi2_a = np.nan_to_num(chi2_a, nan=np.inf, posinf=np.inf)
    chi2_b = np.nan_to_num(chi2_b, nan=np.inf, posinf=np.inf)

    # Both models are scored against the same ceiling so the totals compare.
    return _robust_score(chi2_a, _CHI2_E) + _robust_score(chi2_b, _CHI2_E)


def _robust_score(chi2: np.ndarray, threshold: float) -> float:
    """Sum of ``(_CHI2_H - chi2)`` over inliers; outliers contribute nothing."""
    inliers = chi2 < threshold
    return float(np.sum(_CHI2_H - chi2[inliers]))


def _apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(H).T
    w = homogeneous[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(homogeneous[:, :2] / w, nan=1e9, posinf=1e9, neginf=-1e9)


def median_flow_px(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Median optical-flow magnitude in pixels.

    The cheapest reliable stationary-camera detector: when a KITTI vehicle is
    stopped at a light, flow collapses to sub-pixel noise and any recovered
    translation direction is meaningless.
    """
    if len(points_a) == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(np.asarray(points_b) - np.asarray(points_a), axis=1)))


def triangulate_points(
    T_c1_c2: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Triangulate correspondences into the first camera's frame.

    ``T_c1_c2`` is the relative pose returned by :func:`recover_relative_pose`.
    Returns ``(N, 3)`` points; those that fail to triangulate cleanly (points
    at infinity) come back as ``NaN`` instead of enormous finite values.
    """
    K = np.asarray(K, dtype=np.float64)
    P1 = K @ np.eye(3, 4)
    # Projection of camera 2 needs world-to-camera, i.e. the inverse of T_c1_c2.
    R = T_c1_c2[:3, :3]
    t = T_c1_c2[:3, 3]
    P2 = K @ np.hstack([R.T, (-R.T @ t).reshape(3, 1)])

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        np.ascontiguousarray(points_a, dtype=np.float64).T,
        np.ascontiguousarray(points_b, dtype=np.float64).T,
    )
    w = points_4d[3]
    with np.errstate(divide="ignore", invalid="ignore"):
        points_3d = (points_4d[:3] / w).T
    points_3d[np.abs(w) < 1e-9] = np.nan
    return points_3d
