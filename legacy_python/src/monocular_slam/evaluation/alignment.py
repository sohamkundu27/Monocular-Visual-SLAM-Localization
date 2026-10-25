"""Umeyama trajectory alignment.

Why alignment is required
-------------------------
An estimated trajectory lives in its own coordinate frame: it starts at the
identity with an arbitrary orientation, and — for monocular VO — an arbitrary
scale. Comparing raw coordinates against ground truth would therefore measure
the frame mismatch, not the quality of the estimate.

The standard fix is Umeyama's closed-form solution for the least-squares
similarity transform between two point sets: find ``s``, ``R``, ``t``
minimising ``sum_i || s R x_i + t - y_i ||^2``. Two variants matter here:

``sim3`` (similarity, 7 DoF)
    Also fits a global scale factor. This is the **correct** choice for a pure
    monocular trajectory, whose scale is unobservable by construction. Metrics
    computed after sim3 alignment measure trajectory *shape*.

``se3`` (rigid, 6 DoF)
    Fixes scale at 1. Appropriate when the trajectory is already metric —
    for instance when ground-truth scale was supplied per frame — because it
    then keeps any residual scale drift visible in the error rather than
    absorbing it.

Reference: S. Umeyama, "Least-squares estimation of transformation parameters
between two point patterns", IEEE TPAMI 13(4), 1991.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import se3_from_rt
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)

ALIGNMENT_MODES = ("sim3", "se3", "none")


@dataclass
class AlignmentResult:
    """The similarity transform mapping an estimate onto ground truth."""

    #: Rotation applied to the estimate.
    R: np.ndarray
    #: Translation applied after rotation and scaling.
    t: np.ndarray
    #: Global scale factor (exactly 1.0 for rigid alignment).
    scale: float
    #: Which mode produced this result.
    mode: str

    @property
    def T(self) -> np.ndarray:
        """The rotation and translation as a ``4x4`` SE(3) matrix.

        Scale is *not* folded in: it is applied separately so that rotations
        stay orthonormal. :meth:`Trajectory.transformed` takes both.
        """
        return se3_from_rt(self.R, self.t)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` points through ``s R x + t``."""
        points = np.asarray(points, dtype=np.float64)
        return (self.scale * (self.R @ points.T)).T + self.t

    def apply_to_trajectory(self, trajectory: Trajectory) -> Trajectory:
        return trajectory.transformed(self.T, scale=self.scale)

    @classmethod
    def identity(cls, mode: str = "none") -> AlignmentResult:
        return cls(R=np.eye(3), t=np.zeros(3), scale=1.0, mode=mode)

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "scale": round(float(self.scale), 6),
            "translation_m": [round(float(v), 4) for v in self.t],
        }


def umeyama_alignment(
    source: np.ndarray, target: np.ndarray, with_scale: bool = True
) -> AlignmentResult:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    Parameters
    ----------
    source, target:
        ``(N, 3)`` point sets in correspondence (row ``i`` of one matches row
        ``i`` of the other).
    with_scale:
        Fit a global scale factor (sim3) as well as rotation and translation.

    Returns
    -------
    AlignmentResult
        Such that ``result.apply(source)`` best matches ``target``.

    Notes
    -----
    The rotation comes from the SVD of the cross-covariance matrix. Umeyama's
    key correction over naive Procrustes is the sign handling: when the
    covariance is degenerate or the point sets are mirrored, ``U V^T`` can be a
    reflection, and the fix is to flip the sign of the smallest singular
    direction — which is what the ``S`` matrix below does.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch: source {source.shape} vs target {target.shape}")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Points must be (N, 3), got {source.shape}")
    n = len(source)
    if n < 3:
        raise ValueError(f"Umeyama alignment needs at least 3 points, got {n}")

    mean_source = source.mean(axis=0)
    mean_target = target.mean(axis=0)
    centered_source = source - mean_source
    centered_target = target - mean_target

    # Cross-covariance of the centred sets.
    covariance = (centered_target.T @ centered_source) / n
    U, singular_values, Vt = np.linalg.svd(covariance)

    # Guard against producing a reflection instead of a rotation.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        variance_source = float((centered_source**2).sum() / n)
        if variance_source < 1e-12:
            # A stationary source has no extent to scale.
            scale = 1.0
        else:
            scale = float(np.trace(np.diag(singular_values) @ S) / variance_source)
    else:
        scale = 1.0

    t = mean_target - scale * (R @ mean_source)
    return AlignmentResult(R=R, t=t, scale=scale, mode="sim3" if with_scale else "se3")


def align_trajectory(
    estimated: Trajectory, reference: Trajectory, mode: str = "sim3"
) -> tuple[Trajectory, AlignmentResult]:
    """Align ``estimated`` onto ``reference`` and return both.

    ``mode="none"`` returns the estimate untouched with an identity transform,
    which is how unaligned metrics are produced.
    """
    mode = str(mode).lower().strip()
    if mode not in ALIGNMENT_MODES:
        raise ValueError(f"Unknown alignment mode '{mode}'. Choose from {list(ALIGNMENT_MODES)}")
    if len(estimated) != len(reference):
        raise ValueError(
            f"Trajectories must have equal length to align: "
            f"{len(estimated)} vs {len(reference)}"
        )

    if mode == "none":
        return estimated.copy(), AlignmentResult.identity()

    result = umeyama_alignment(
        estimated.positions, reference.positions, with_scale=(mode == "sim3")
    )
    logger.info(
        "Umeyama %s alignment: scale=%.4f, |t|=%.3f m",
        mode,
        result.scale,
        float(np.linalg.norm(result.t)),
    )
    return result.apply_to_trajectory(estimated), result
