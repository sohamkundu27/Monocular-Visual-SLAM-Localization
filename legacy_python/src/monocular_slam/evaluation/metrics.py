"""Trajectory error metrics.

Three complementary families are implemented, because each answers a different
question and none is sufficient alone:

**Absolute Trajectory Error (ATE)** — global consistency.
    After aligning the estimate to ground truth, ATE is the Euclidean distance
    between corresponding camera positions. It answers "how far from the truth
    does the trajectory end up?" and is the metric loop closure most visibly
    improves. It is dominated by low-frequency drift: one early heading error
    inflates ATE for the rest of the sequence.

**Relative Pose Error (RPE)** — local accuracy.
    For a fixed frame gap ``delta``, RPE compares the estimated relative motion
    ``T_i_{i+delta}`` against the ground-truth relative motion. It is invariant
    to global drift, so it isolates front-end quality. Reported separately for
    translation (metres) and rotation (degrees).

**KITTI drift** — the odometry benchmark's own metric.
    Errors are evaluated over sub-trajectories of fixed *path length*
    (100 m ... 800 m) and normalised by that length, giving translation error
    as a percentage and rotation error in degrees per metre. Normalising by
    distance rather than by frame count makes results comparable across
    sequences of different speed and length. This is the number quoted on the
    KITTI odometry leaderboard.

Reference for ATE/RPE: Sturm et al., "A Benchmark for the Evaluation of RGB-D
SLAM Systems", IROS 2012. Reference for the drift metric: Geiger et al., "Are
we ready for Autonomous Driving? The KITTI Vision Benchmark Suite", CVPR 2012.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from monocular_slam.evaluation.alignment import AlignmentResult, align_trajectory
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import invert_se3, rotation_angle_deg
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Absolute Trajectory Error
# --------------------------------------------------------------------------- #


@dataclass
class ATEResult:
    """Absolute trajectory error statistics, all in metres."""

    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int
    alignment: str = "none"
    alignment_scale: float = 1.0
    errors: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)

    def to_dict(self) -> dict[str, float | str | int]:
        data = {k: v for k, v in asdict(self).items() if k != "errors"}
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in data.items()}


def absolute_trajectory_error(
    estimated: Trajectory,
    reference: Trajectory,
    alignment: str = "sim3",
) -> ATEResult:
    """Compute ATE between two equal-length trajectories.

    ``alignment`` selects the Umeyama mode applied first; ``"none"`` yields the
    unaligned error, which is only meaningful when both trajectories are
    already in the same metric frame.

    Note on ``std``: this is the standard deviation of the error *magnitudes*,
    not a residual about the mean of a signed quantity, so ``rmse`` is not
    equal to ``sqrt(mean^2 + std^2)`` by accident — it is exactly that
    identity, and both are reported because different papers quote different
    ones.
    """
    if len(estimated) != len(reference):
        raise ValueError(
            f"Trajectories must be the same length: {len(estimated)} vs {len(reference)}"
        )
    if len(estimated) == 0:
        raise ValueError("Cannot compute ATE on an empty trajectory")

    aligned, alignment_result = align_trajectory(estimated, reference, mode=alignment)
    errors = np.linalg.norm(aligned.positions - reference.positions, axis=1)

    return ATEResult(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        min=float(np.min(errors)),
        max=float(np.max(errors)),
        n=int(len(errors)),
        alignment=alignment_result.mode,
        alignment_scale=float(alignment_result.scale),
        errors=errors,
    )


# --------------------------------------------------------------------------- #
# Relative Pose Error
# --------------------------------------------------------------------------- #


@dataclass
class RPEResult:
    """Relative pose error at a fixed frame gap."""

    delta: int
    n: int
    #: Translation error in metres.
    trans_rmse: float
    trans_mean: float
    trans_median: float
    trans_max: float
    #: Rotation error in degrees.
    rot_rmse_deg: float
    rot_mean_deg: float
    rot_median_deg: float
    rot_max_deg: float

    def to_dict(self) -> dict[str, float | int]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def relative_pose_error(
    estimated: Trajectory,
    reference: Trajectory,
    delta: int = 1,
    scale: float = 1.0,
) -> RPEResult:
    """Compute RPE at a frame gap of ``delta``.

    For each ``i``, the error transform is

    ``E_i = (T_gt_i^-1 T_gt_{i+d})^-1 (T_est_i^-1 T_est_{i+d})``

    whose translation norm and rotation angle are the reported errors. Because
    only *relative* motions are compared, no global alignment is needed —
    which is exactly what makes RPE insensitive to accumulated drift.

    ``scale`` multiplies the estimated translations first, for evaluating a
    scale-free (unit-scale) trajectory against metric ground truth. Pass the
    factor from a sim3 alignment.
    """
    if len(estimated) != len(reference):
        raise ValueError(
            f"Trajectories must be the same length: {len(estimated)} vs {len(reference)}"
        )
    if delta < 1:
        raise ValueError(f"delta must be >= 1, got {delta}")

    n_pairs = len(estimated) - delta
    if n_pairs <= 0:
        raise ValueError(
            f"delta={delta} exceeds the trajectory length ({len(estimated)} poses)"
        )

    trans_errors = np.zeros(n_pairs)
    rot_errors = np.zeros(n_pairs)

    for i in range(n_pairs):
        gt_rel = invert_se3(reference[i]) @ reference[i + delta]
        est_rel = invert_se3(estimated[i]) @ estimated[i + delta]
        if scale != 1.0:
            est_rel = est_rel.copy()
            est_rel[:3, 3] *= scale
        error = invert_se3(gt_rel) @ est_rel
        trans_errors[i] = np.linalg.norm(error[:3, 3])
        rot_errors[i] = rotation_angle_deg(error[:3, :3])

    return RPEResult(
        delta=int(delta),
        n=int(n_pairs),
        trans_rmse=float(np.sqrt(np.mean(trans_errors**2))),
        trans_mean=float(np.mean(trans_errors)),
        trans_median=float(np.median(trans_errors)),
        trans_max=float(np.max(trans_errors)),
        rot_rmse_deg=float(np.sqrt(np.mean(rot_errors**2))),
        rot_mean_deg=float(np.mean(rot_errors)),
        rot_median_deg=float(np.median(rot_errors)),
        rot_max_deg=float(np.max(rot_errors)),
    )


# --------------------------------------------------------------------------- #
# KITTI drift
# --------------------------------------------------------------------------- #


@dataclass
class DriftResult:
    """KITTI-style drift, averaged over fixed-length sub-trajectories."""

    #: Translation error as a percentage of sub-trajectory length.
    translation_pct: float
    #: Rotation error in degrees per metre.
    rotation_deg_per_m: float
    #: The same rotation figure per 100 m, which is the more readable form.
    rotation_deg_per_100m: float
    #: Number of sub-trajectories evaluated.
    n_segments: int
    #: Per-length breakdown, ``{length_m: (trans_pct, rot_deg_per_m)}``.
    per_length: dict[float, tuple[float, float]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "translation_pct": round(self.translation_pct, 4),
            "rotation_deg_per_m": round(self.rotation_deg_per_m, 8),
            "rotation_deg_per_100m": round(self.rotation_deg_per_100m, 5),
            "n_segments": self.n_segments,
            "per_length": {
                str(int(length)): [round(t, 4), round(r, 8)]
                for length, (t, r) in sorted(self.per_length.items())
            },
        }


DEFAULT_SEGMENT_LENGTHS = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)


def kitti_drift(
    estimated: Trajectory,
    reference: Trajectory,
    segment_lengths: tuple[float, ...] = DEFAULT_SEGMENT_LENGTHS,
    step: int = 10,
) -> DriftResult:
    """KITTI odometry drift metric.

    For every start frame (every ``step``-th) and every requested sub-trajectory
    length, find the end frame at which ground truth has travelled that far,
    then compare the estimated and true relative pose over that stretch. The
    translation error is divided by the sub-trajectory length, so the result is
    a dimensionless percentage that does not depend on how long the sequence is.

    Sub-trajectories that run off the end of the sequence are skipped rather
    than truncated, which is what keeps the normalisation honest.
    """
    if len(estimated) != len(reference):
        raise ValueError(
            f"Trajectories must be the same length: {len(estimated)} vs {len(reference)}"
        )
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    distances = reference.cumulative_distance()
    n = len(reference)

    errors_by_length: dict[float, list[tuple[float, float]]] = {
        float(length): [] for length in segment_lengths
    }

    for start in range(0, n, step):
        for length in segment_lengths:
            end = _find_segment_end(distances, start, float(length))
            if end is None:
                continue
            gt_rel = invert_se3(reference[start]) @ reference[end]
            est_rel = invert_se3(estimated[start]) @ estimated[end]
            error = invert_se3(gt_rel) @ est_rel
            trans_error = float(np.linalg.norm(error[:3, 3]))
            rot_error = rotation_angle_deg(error[:3, :3])
            errors_by_length[float(length)].append((trans_error / length, rot_error / length))

    per_length: dict[float, tuple[float, float]] = {}
    all_trans: list[float] = []
    all_rot: list[float] = []
    for length, records in errors_by_length.items():
        if not records:
            continue
        trans = [r[0] for r in records]
        rot = [r[1] for r in records]
        per_length[length] = (100.0 * float(np.mean(trans)), float(np.mean(rot)))
        all_trans.extend(trans)
        all_rot.extend(rot)

    if not all_trans:
        logger.warning(
            "Trajectory is only %.1f m long; no sub-trajectory reaches the shortest "
            "requested length (%.0f m), so drift is undefined",
            reference.path_length(),
            min(segment_lengths),
        )
        return DriftResult(float("nan"), float("nan"), float("nan"), 0, {})

    mean_trans_pct = 100.0 * float(np.mean(all_trans))
    mean_rot_per_m = float(np.mean(all_rot))
    return DriftResult(
        translation_pct=mean_trans_pct,
        rotation_deg_per_m=mean_rot_per_m,
        rotation_deg_per_100m=mean_rot_per_m * 100.0,
        n_segments=len(all_trans),
        per_length=per_length,
    )


def _find_segment_end(distances: np.ndarray, start: int, length: float) -> int | None:
    """First index at or after ``start`` where travelled distance reaches ``length``."""
    target = distances[start] + length
    if distances[-1] < target:
        return None
    # distances is non-decreasing, so a binary search is exact and cheap.
    end = int(np.searchsorted(distances, target, side="left"))
    return end if end < len(distances) else None


# --------------------------------------------------------------------------- #
# Combined report
# --------------------------------------------------------------------------- #


@dataclass
class TrajectoryEvaluation:
    """Full evaluation of one trajectory against ground truth."""

    label: str
    ate_aligned: ATEResult
    ate_unaligned: ATEResult
    rpe: dict[int, RPEResult]
    drift: DriftResult
    path_length_m: float
    reference_path_length_m: float
    alignment: AlignmentResult | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "path_length_m": round(self.path_length_m, 3),
            "reference_path_length_m": round(self.reference_path_length_m, 3),
            "ate_aligned": self.ate_aligned.to_dict(),
            "ate_unaligned": self.ate_unaligned.to_dict(),
            "rpe": {str(delta): result.to_dict() for delta, result in sorted(self.rpe.items())},
            "drift": self.drift.to_dict(),
        }

    def summary_line(self) -> str:
        return (
            f"{self.label}: ATE RMSE {self.ate_aligned.rmse:.3f} m "
            f"({self.ate_aligned.alignment}-aligned), "
            f"unaligned {self.ate_unaligned.rmse:.3f} m, "
            f"drift {self.drift.translation_pct:.3f}% / "
            f"{self.drift.rotation_deg_per_100m:.4f} deg per 100 m"
        )


def evaluate_trajectory(
    estimated: Trajectory,
    reference: Trajectory,
    label: str = "trajectory",
    alignment: str = "sim3",
    rpe_deltas: tuple[int, ...] = (1, 10, 100),
    segment_lengths: tuple[float, ...] = DEFAULT_SEGMENT_LENGTHS,
) -> TrajectoryEvaluation:
    """Run every metric and bundle the results.

    Both aligned and unaligned ATE are always reported. The distinction is not
    cosmetic for a monocular system: the aligned figure measures trajectory
    shape (having absorbed the unobservable global scale and frame), while the
    unaligned figure additionally penalises being in the wrong frame at all.
    Drift and RPE are computed on the aligned trajectory so that a global frame
    offset is not double-counted as a local error.
    """
    ate_aligned = absolute_trajectory_error(estimated, reference, alignment=alignment)
    ate_unaligned = absolute_trajectory_error(estimated, reference, alignment="none")
    aligned, alignment_result = align_trajectory(estimated, reference, mode=alignment)

    rpe: dict[int, RPEResult] = {}
    for delta in rpe_deltas:
        if delta >= len(estimated):
            logger.debug("Skipping RPE delta=%d: trajectory has %d poses", delta, len(estimated))
            continue
        rpe[int(delta)] = relative_pose_error(aligned, reference, delta=delta)

    drift = kitti_drift(aligned, reference, segment_lengths=segment_lengths)

    evaluation = TrajectoryEvaluation(
        label=label,
        ate_aligned=ate_aligned,
        ate_unaligned=ate_unaligned,
        rpe=rpe,
        drift=drift,
        path_length_m=estimated.path_length(),
        reference_path_length_m=reference.path_length(),
        alignment=alignment_result,
    )
    logger.info(evaluation.summary_line())
    return evaluation


def ate_reduction_pct(before: float, after: float) -> float:
    """Percentage reduction in ATE, positive when ``after`` is better.

    Returns NaN when ``before`` is zero or non-finite, since the reduction is
    undefined there rather than infinite.
    """
    if not np.isfinite(before) or before <= 0:
        return float("nan")
    return float((before - after) / before * 100.0)
