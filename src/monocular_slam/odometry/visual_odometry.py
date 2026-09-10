"""Monocular visual odometry front end.

Per frame transition ``i -> i+1``:

1. Detect ORB keypoints and descriptors in frame ``i+1`` (frame ``i`` is
   cached from the previous iteration, so each image is described once).
2. Match descriptors against the previous frame and filter the correspondences.
3. Estimate the essential matrix with RANSAC and decompose it into a
   validated relative pose ``T_i_{i+1}`` with **unit-norm translation**.
4. Ask the scale strategy for the translation magnitude and apply it.
5. Compose onto the global pose: ``T_w_{i+1} = T_w_i @ T_i_{i+1}``.

Failure handling
----------------
A 4541-frame sequence will contain transitions that cannot be solved: too few
matches under motion blur, a degenerate essential matrix while stopped at a
light, a decomposition that yields an implausible rotation. Aborting the run is
the wrong answer, and so is pretending the estimate succeeded. Each transition
therefore resolves to one of three outcomes, all recorded in the diagnostics:

``ok``
    A validated relative pose was integrated.
``stationary``
    Median optical flow was below threshold, so the vehicle is treated as
    stopped and the previous pose is held. This is the *correct* estimate, not
    a fallback, and it counts as a success.
``failed``
    Estimation failed or was rejected. The last successful relative motion is
    re-applied as a constant-velocity prediction — at 10 Hz a car's motion
    changes little between frames, so this bridges the gap far better than
    holding still — and the transition is counted as a failure in the
    successful-pose-rate metric.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from monocular_slam.datasets.kitti import KittiOdometryDataset
from monocular_slam.features.detector import FeatureDetector, Frame
from monocular_slam.features.matcher import FeatureMatcher, MatchResult
from monocular_slam.geometry.epipolar import (
    estimate_essential_matrix,
    median_flow_px,
    recover_relative_pose,
    select_two_view_model,
)
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import project_to_se3, se3_from_rt
from monocular_slam.odometry.scale import ScaleEstimator, build_scale_estimator
from monocular_slam.utils.logging import get_logger
from monocular_slam.utils.timing import StageTimer

logger = get_logger(__name__)

#: Transition outcomes.
STATUS_OK = "ok"
STATUS_STATIONARY = "stationary"
STATUS_FAILED = "failed"


@dataclass
class FrameDiagnostics:
    """Per-transition record, exported to ``diagnostics.csv``."""

    index: int
    frame_id: int
    status: str
    reason: str = ""
    n_keypoints: int = 0
    n_raw_matches: int = 0
    n_matches: int = 0
    n_ransac_inliers: int = 0
    n_cheirality_inliers: int = 0
    #: Epipolar inliers as a fraction of filtered matches -- the number
    #: conventionally reported as "RANSAC inlier ratio".
    ransac_inlier_ratio: float = 0.0
    #: Cheirality survivors as a fraction of epipolar inliers.
    inlier_ratio: float = 0.0
    rotation_deg: float = float("nan")
    parallax_deg: float = float("nan")
    flow_px: float = float("nan")
    homography_ratio: float = float("nan")
    degenerate: bool = False
    scale_m: float = float("nan")
    detect_ms: float = 0.0
    match_ms: float = 0.0
    pose_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Stationary transitions are correct estimates, so they count as successes."""
        return self.status in (STATUS_OK, STATUS_STATIONARY)

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class OdometryResult:
    """Everything a VO run produces."""

    trajectory: Trajectory
    diagnostics: list[FrameDiagnostics]
    relative_poses: list[np.ndarray] = field(default_factory=list)
    timer: StageTimer | None = None
    scale_strategy: dict[str, object] = field(default_factory=dict)

    # ----------------------------------------------------------------- #
    # Aggregate statistics (all measured, none assumed)
    # ----------------------------------------------------------------- #

    @property
    def n_transitions(self) -> int:
        return len(self.diagnostics)

    @property
    def n_successful(self) -> int:
        return sum(1 for d in self.diagnostics if d.succeeded)

    @property
    def successful_pose_rate_pct(self) -> float:
        """Percentage of attempted transitions that produced a usable pose."""
        return 100.0 * self.n_successful / self.n_transitions if self.n_transitions else 0.0

    def _mean(self, attribute: str, only_successful: bool = False) -> float:
        source = (
            [d for d in self.diagnostics if d.succeeded] if only_successful else self.diagnostics
        )
        values = [getattr(d, attribute) for d in source]
        values = [v for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def avg_features_per_frame(self) -> float:
        return self._mean("n_keypoints")

    @property
    def avg_matches_per_pair(self) -> float:
        return self._mean("n_matches")

    @property
    def avg_ransac_inliers(self) -> float:
        return self._mean("n_ransac_inliers")

    @property
    def avg_inlier_ratio_pct(self) -> float:
        """Mean RANSAC inlier ratio over successful transitions, as a percentage."""
        return 100.0 * self._mean("ransac_inlier_ratio", only_successful=True)

    @property
    def avg_cheirality_ratio_pct(self) -> float:
        return 100.0 * self._mean("inlier_ratio", only_successful=True)

    @property
    def n_degenerate(self) -> int:
        return sum(1 for d in self.diagnostics if d.degenerate)

    def feature_statistics(self) -> dict[str, float]:
        return {
            "avg_features_per_frame": round(self.avg_features_per_frame, 1),
            "avg_matches_per_pair": round(self.avg_matches_per_pair, 1),
            "avg_ransac_inliers": round(self.avg_ransac_inliers, 1),
            "avg_inlier_ratio_pct": round(self.avg_inlier_ratio_pct, 2),
            "avg_cheirality_ratio_pct": round(self.avg_cheirality_ratio_pct, 2),
        }

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.diagnostics:
            counts[d.status] = counts.get(d.status, 0) + 1
        return counts


class VisualOdometry:
    """Frame-to-frame monocular visual odometry."""

    def __init__(
        self,
        K: np.ndarray,
        detector: FeatureDetector,
        matcher: FeatureMatcher,
        scale_estimator: ScaleEstimator,
        *,
        ransac_method: str = "magsac",
        ransac_threshold_px: float = 0.5,
        ransac_confidence: float = 0.999,
        ransac_max_iters: int = 2000,
        cheirality_distance: float = 200.0,
        min_matches: int = 20,
        min_inliers: int = 30,
        min_inlier_ratio: float = 0.3,
        max_rotation_deg: float = 30.0,
        min_parallax_deg: float = 0.0,
        min_flow_px: float = 0.7,
        homography_ratio_threshold: float = 0.45,
        reject_degenerate: bool = False,
        timer: StageTimer | None = None,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.detector = detector
        self.matcher = matcher
        self.scale_estimator = scale_estimator
        self.ransac_method = str(ransac_method)
        self.ransac_threshold_px = float(ransac_threshold_px)
        self.ransac_confidence = float(ransac_confidence)
        self.ransac_max_iters = int(ransac_max_iters)
        self.cheirality_distance = float(cheirality_distance)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_rotation_deg = float(max_rotation_deg)
        self.min_parallax_deg = float(min_parallax_deg)
        self.min_flow_px = float(min_flow_px)
        self.homography_ratio_threshold = float(homography_ratio_threshold)
        self.reject_degenerate = bool(reject_degenerate)
        self.timer = timer if timer is not None else StageTimer()

    @classmethod
    def from_config(cls, config, dataset: KittiOdometryDataset, timer: StageTimer | None = None):  # noqa: ANN001
        """Assemble a VO front end from configuration plus a dataset."""
        detector = FeatureDetector.from_config(config)
        matcher = FeatureMatcher.from_config(config, norm=detector.descriptor_norm)
        o = config.odometry
        return cls(
            K=dataset.K,
            detector=detector,
            matcher=matcher,
            scale_estimator=build_scale_estimator(config, dataset),
            ransac_method=o.ransac_method,
            ransac_threshold_px=o.ransac_threshold_px,
            ransac_confidence=o.ransac_confidence,
            ransac_max_iters=o.ransac_max_iters,
            cheirality_distance=o.cheirality_distance,
            min_matches=config.matcher.min_matches,
            min_inliers=o.min_inliers,
            min_inlier_ratio=o.min_inlier_ratio,
            max_rotation_deg=o.max_rotation_deg,
            min_parallax_deg=o.min_parallax_deg,
            min_flow_px=o.min_flow_px,
            homography_ratio_threshold=o.homography_ratio_threshold,
            reject_degenerate=o.reject_degenerate,
            timer=timer,
        )

    # ----------------------------------------------------------------- #
    # Two-view estimation
    # ----------------------------------------------------------------- #

    def estimate_relative_pose(
        self, matches: MatchResult, diagnostics: FrameDiagnostics
    ) -> np.ndarray | None:
        """Estimate ``T_i_{i+1}`` with unit translation, or ``None`` on failure.

        ``diagnostics`` is filled in as a side effect so every transition —
        successful or not — leaves an auditable trail.
        """
        diagnostics.n_raw_matches = matches.n_raw
        diagnostics.n_matches = len(matches)

        if len(matches) < self.min_matches:
            diagnostics.status = STATUS_FAILED
            diagnostics.reason = f"too_few_matches({len(matches)})"
            return None

        flow = median_flow_px(matches.points_a, matches.points_b)
        diagnostics.flow_px = flow
        if np.isfinite(flow) and flow < self.min_flow_px:
            # The camera has not moved: identity is the right answer, and
            # attempting to decompose a degenerate E would inject pure noise.
            diagnostics.status = STATUS_STATIONARY
            diagnostics.reason = f"low_flow({flow:.2f}px)"
            return np.eye(4)

        essential = estimate_essential_matrix(
            matches.points_a,
            matches.points_b,
            self.K,
            threshold_px=self.ransac_threshold_px,
            confidence=self.ransac_confidence,
            max_iters=self.ransac_max_iters,
            method=self.ransac_method,
        )
        diagnostics.n_ransac_inliers = essential.n_inliers
        diagnostics.ransac_inlier_ratio = essential.inlier_ratio
        if not essential.ok:
            diagnostics.status = STATUS_FAILED
            diagnostics.reason = essential.reason
            return None

        selection = select_two_view_model(
            matches.points_a,
            matches.points_b,
            self.K,
            essential.E,
            threshold_px=self.ransac_threshold_px,
            ratio_threshold=self.homography_ratio_threshold,
        )
        diagnostics.homography_ratio = selection.ratio_h
        diagnostics.degenerate = selection.is_degenerate
        if selection.is_degenerate and self.reject_degenerate:
            diagnostics.status = STATUS_FAILED
            diagnostics.reason = f"degenerate_geometry(ratio_h={selection.ratio_h:.2f})"
            return None

        pose = recover_relative_pose(
            essential.E,
            matches.points_a,
            matches.points_b,
            self.K,
            essential.inlier_mask,
            min_inliers=self.min_inliers,
            min_inlier_ratio=self.min_inlier_ratio,
            max_rotation_deg=self.max_rotation_deg,
            min_parallax_deg=self.min_parallax_deg,
            cheirality_distance=self.cheirality_distance,
        )
        diagnostics.n_cheirality_inliers = pose.n_cheirality_inliers
        diagnostics.inlier_ratio = (
            pose.n_cheirality_inliers / essential.n_inliers if essential.n_inliers else 0.0
        )
        diagnostics.rotation_deg = pose.rotation_deg
        diagnostics.parallax_deg = pose.median_parallax_deg

        if not pose.ok:
            diagnostics.status = STATUS_FAILED
            diagnostics.reason = pose.reason
            return None

        diagnostics.status = STATUS_OK
        return pose.T

    # ----------------------------------------------------------------- #
    # Full sequence
    # ----------------------------------------------------------------- #

    def run(
        self,
        dataset: KittiOdometryDataset,
        on_frame: Callable[[int, Frame, np.ndarray, np.ndarray], None] | None = None,
        progress_interval: int = 250,
    ) -> OdometryResult:
        """Process an entire sequence and accumulate the global trajectory.

        ``on_frame(index, frame, image, pose)`` is invoked for every processed
        image after its global pose is known, which lets the SLAM pipeline
        harvest keyframes without a second pass over the data or a second
        feature-detection cost.
        """
        n = len(dataset)
        if n < 2:
            raise ValueError(f"Need at least 2 frames for odometry, got {n}")

        self.timer.reset_wall_clock()

        poses: list[np.ndarray] = [np.eye(4)]
        relative_poses: list[np.ndarray] = []
        diagnostics: list[FrameDiagnostics] = []

        with self.timer.time("detect"):
            previous = self.detector.detect(dataset[0], index=0, frame_id=int(dataset.frame_ids[0]))
        if on_frame is not None:
            on_frame(0, previous, dataset[0], poses[0])

        last_relative_direction: np.ndarray | None = None

        for i in range(1, n):
            image = dataset[i]
            frame_id = int(dataset.frame_ids[i])
            diag = FrameDiagnostics(index=i, frame_id=frame_id, status=STATUS_FAILED)

            t_detect = self.timer.stats["detect"].total_s
            with self.timer.time("detect"):
                current = self.detector.detect(image, index=i, frame_id=frame_id)
            diag.detect_ms = (self.timer.stats["detect"].total_s - t_detect) * 1000.0
            diag.n_keypoints = len(current)

            t_match = self.timer.stats["match"].total_s
            with self.timer.time("match"):
                matches = self.matcher.match(previous, current)
            diag.match_ms = (self.timer.stats["match"].total_s - t_match) * 1000.0

            t_pose = self.timer.stats["pose"].total_s
            with self.timer.time("pose"):
                relative = self.estimate_relative_pose(matches, diag)
            diag.pose_ms = (self.timer.stats["pose"].total_s - t_pose) * 1000.0

            scale = self.scale_estimator.scale_for(i - 1)
            diag.scale_m = float("nan") if scale is None else float(scale)

            step = self._resolve_step(relative, scale, diag, last_relative_direction)
            if diag.status == STATUS_OK and relative is not None:
                last_relative_direction = relative

            poses.append(project_to_se3(poses[-1] @ step))
            relative_poses.append(step)
            diagnostics.append(diag)
            previous = current

            if on_frame is not None:
                on_frame(i, current, image, poses[-1])

            if progress_interval and i % progress_interval == 0:
                ok = sum(1 for d in diagnostics if d.succeeded)
                logger.info(
                    "frame %d/%d | success %.1f%% | matches %d | inliers %d | %.1f fps",
                    i,
                    n - 1,
                    100.0 * ok / len(diagnostics),
                    diag.n_matches,
                    diag.n_cheirality_inliers,
                    self.timer.fps(i + 1),
                )

        trajectory = Trajectory(
            np.stack(poses), frame_ids=dataset.frame_ids, timestamps=dataset.timestamps
        )
        result = OdometryResult(
            trajectory=trajectory,
            diagnostics=diagnostics,
            relative_poses=relative_poses,
            timer=self.timer,
            scale_strategy=self.scale_estimator.describe(),
        )
        logger.info(
            "Visual odometry complete: %d transitions, %.1f%% successful, %s",
            result.n_transitions,
            result.successful_pose_rate_pct,
            result.status_counts(),
        )
        return result

    def _resolve_step(
        self,
        relative: np.ndarray | None,
        scale: float | None,
        diagnostics: FrameDiagnostics,
        last_direction: np.ndarray | None,
    ) -> np.ndarray:
        """Turn a unit-translation pose plus a scale into the metric step.

        Handles the three outcomes described in the module docstring.
        """
        if diagnostics.status == STATUS_STATIONARY:
            return np.eye(4)

        if relative is not None and scale is not None:
            R = relative[:3, :3]
            t = relative[:3, 3] * float(scale)
            return se3_from_rt(R, t)

        if relative is not None and scale is None:
            # Geometry succeeded but the scale source has nothing to say.
            diagnostics.status = STATUS_FAILED
            diagnostics.reason = (diagnostics.reason + "|no_scale").lstrip("|")

        # Constant-velocity prediction from the last successful motion.
        if last_direction is not None:
            R = last_direction[:3, :3]
            magnitude = float(scale) if scale is not None else 0.0
            return se3_from_rt(R, last_direction[:3, 3] * magnitude)
        return np.eye(4)
