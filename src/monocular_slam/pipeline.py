"""End-to-end monocular SLAM pipeline.

Stages
------
1. **Front end** — visual odometry over the sequence, harvesting keyframes as it
   goes so descriptors are computed exactly once.
2. **Vocabulary** — cluster a descriptor sample from the harvested keyframes.
3. **Loop closure** — appearance retrieval plus geometric verification.
4. **Back end** — build the keyframe pose graph and optimize it with GTSAM.
5. **Interpolation** — propagate the keyframe corrections back to every frame,
   so the optimized trajectory can be evaluated against dense ground truth.
6. **Evaluation** — metrics for the raw and optimized trajectories, plots,
   diagnostics and a machine-readable summary.

Every number written to ``metrics.json`` comes from this run. When ground truth
is unavailable the metric keys are emitted as ``null`` rather than filled with
plausible-looking values.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from monocular_slam.config import Config
from monocular_slam.datasets.kitti import KittiOdometryDataset
from monocular_slam.evaluation.metrics import (
    TrajectoryEvaluation,
    absolute_trajectory_error,
    ate_reduction_pct,
    evaluate_trajectory,
)
from monocular_slam.evaluation.plotting import (
    plot_diagnostics,
    plot_error_over_time,
    plot_loop_closures,
    plot_trajectory,
    plot_trajectory_3d,
    plot_trajectory_comparison,
)
from monocular_slam.features.matcher import FeatureMatcher
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import invert_se3, project_to_se3
from monocular_slam.loop_closure.database import (
    Keyframe,
    KeyframeDatabase,
    KeyframeSelector,
    VisualVocabulary,
)
from monocular_slam.loop_closure.detector import LoopClosure, LoopClosureDetector
from monocular_slam.odometry.visual_odometry import OdometryResult, VisualOdometry
from monocular_slam.optimization.gtsam_backend import (
    OptimizationResult,
    gtsam_available,
    optimize_pose_graph,
)
from monocular_slam.optimization.pose_graph import build_pose_graph
from monocular_slam.utils.logging import get_logger, setup_logging
from monocular_slam.utils.seeding import set_global_seed
from monocular_slam.utils.timing import StageTimer

logger = get_logger(__name__)


@dataclass
class SlamResult:
    """Everything one pipeline run produces."""

    config: Config
    dataset_info: dict[str, object]
    odometry: OdometryResult
    raw_trajectory: Trajectory
    optimized_trajectory: Trajectory | None
    ground_truth: Trajectory | None
    loop_closures: list[LoopClosure]
    optimization: OptimizationResult | None
    timer: StageTimer
    evaluation_raw: TrajectoryEvaluation | None = None
    evaluation_optimized: TrajectoryEvaluation | None = None
    keyframe_frame_indices: list[int] = field(default_factory=list)
    loop_stats: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None

    def diagnostics_available(self) -> bool:
        return bool(self.odometry.diagnostics)


class SlamPipeline:
    """Runs the full SLAM pipeline for one sequence."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.timer = StageTimer()

    # ----------------------------------------------------------------- #
    # Main entry point
    # ----------------------------------------------------------------- #

    def run(self, dataset: KittiOdometryDataset | None = None) -> SlamResult:
        """Execute the pipeline end to end."""
        config = self.config
        set_global_seed(config.runtime.seed)

        if dataset is None:
            dataset = KittiOdometryDataset.from_config(config)

        self.timer.reset_wall_clock()

        # -- Stage 1: visual odometry with keyframe harvesting ---------- #
        vo = VisualOdometry.from_config(config, dataset, timer=self.timer)
        database = KeyframeDatabase()
        selector = KeyframeSelector.from_config(config)
        harvest = _KeyframeHarvester(database, selector, enabled=config.loop_closure.enabled)

        odometry = vo.run(
            dataset,
            on_frame=harvest,
            progress_interval=config.runtime.progress_interval,
        )
        raw_trajectory = odometry.trajectory
        ground_truth = dataset.ground_truth

        # -- Stages 2-3: loop closure ---------------------------------- #
        loop_closures: list[LoopClosure] = []
        loop_stats: dict[str, object] = {}
        if config.loop_closure.enabled and len(database) > config.loop_closure.min_keyframe_separation:
            loop_closures, loop_stats = self._detect_loops(config, dataset, vo, database)
        elif config.loop_closure.enabled:
            logger.warning(
                "Only %d keyframes harvested; need more than min_keyframe_separation=%d "
                "for loop closure",
                len(database),
                config.loop_closure.min_keyframe_separation,
            )

        # -- Stage 4-5: pose graph optimization ------------------------ #
        optimization: OptimizationResult | None = None
        optimized_trajectory: Trajectory | None = None
        if loop_closures and gtsam_available():
            optimization, optimized_trajectory = self._optimize(
                config, database, raw_trajectory, loop_closures
            )
        elif loop_closures:  # pragma: no cover - environment dependent
            logger.error("Loop closures were found but GTSAM is unavailable; skipping optimization")
        else:
            logger.info("No verified loop closures; the optimized trajectory equals raw odometry")

        # -- Stage 6: evaluation --------------------------------------- #
        evaluation_raw = None
        evaluation_optimized = None
        if ground_truth is not None:
            evaluation_raw = evaluate_trajectory(
                raw_trajectory,
                ground_truth,
                label="raw_vo",
                alignment=config.evaluation.alignment,
                rpe_deltas=config.evaluation.rpe_deltas,
                segment_lengths=config.evaluation.drift_segment_lengths_m,
            )
            if optimized_trajectory is not None:
                evaluation_optimized = evaluate_trajectory(
                    optimized_trajectory,
                    ground_truth,
                    label="optimized_slam",
                    alignment=config.evaluation.alignment,
                    rpe_deltas=config.evaluation.rpe_deltas,
                    segment_lengths=config.evaluation.drift_segment_lengths_m,
                )
        else:
            logger.warning(
                "Sequence %s has no ground truth: trajectory metrics will be reported as null",
                config.dataset.sequence,
            )

        return SlamResult(
            config=config,
            dataset_info=dataset.describe(),
            odometry=odometry,
            raw_trajectory=raw_trajectory,
            optimized_trajectory=optimized_trajectory,
            ground_truth=ground_truth,
            loop_closures=loop_closures,
            optimization=optimization,
            timer=self.timer,
            evaluation_raw=evaluation_raw,
            evaluation_optimized=evaluation_optimized,
            keyframe_frame_indices=[kf.frame_index for kf in database],
            loop_stats=loop_stats,
        )

    # ----------------------------------------------------------------- #
    # Stage helpers
    # ----------------------------------------------------------------- #

    def _detect_loops(
        self,
        config: Config,
        dataset: KittiOdometryDataset,
        vo: VisualOdometry,
        database: KeyframeDatabase,
    ) -> tuple[list[LoopClosure], dict[str, object]]:
        """Train the vocabulary and run loop detection."""
        with self.timer.time("vocabulary"):
            descriptors = np.concatenate([kf.descriptors for kf in database])
            budget = config.loop_closure.vocabulary_train_descriptors
            if len(descriptors) > budget:
                rng = np.random.default_rng(config.runtime.seed)
                descriptors = descriptors[rng.choice(len(descriptors), size=budget, replace=False)]
            vocabulary = VisualVocabulary.train(
                descriptors,
                vocabulary_size=config.loop_closure.vocabulary_size,
                seed=config.runtime.seed,
            )
        logger.info(
            "Trained vocabulary: %d words from %d descriptors in %.1f s",
            vocabulary.size,
            len(descriptors),
            self.timer.total_s("vocabulary"),
        )
        database.set_vocabulary(vocabulary)

        matcher = FeatureMatcher.from_config(config, norm=vo.detector.descriptor_norm)
        detector = LoopClosureDetector.from_config(config, dataset.K, matcher, timer=self.timer)
        closures = detector.detect(database)
        return closures, detector.stats.to_dict()

    def _optimize(
        self,
        config: Config,
        database: KeyframeDatabase,
        raw_trajectory: Trajectory,
        loop_closures: list[LoopClosure],
    ) -> tuple[OptimizationResult, Trajectory]:
        """Build and solve the keyframe pose graph, then densify the result."""
        keyframe_indices = [kf.frame_index for kf in database]
        keyframe_trajectory = Trajectory(
            np.stack([kf.pose for kf in database]),
            frame_ids=np.array([kf.frame_id for kf in database], dtype=np.int64),
        )
        pg = config.pose_graph
        graph = build_pose_graph(
            keyframe_trajectory,
            loop_closures,
            node_frame_indices=keyframe_indices,
            prior_sigma_rot=pg.prior_sigma_rot,
            prior_sigma_trans=pg.prior_sigma_trans,
            odom_sigma_rot=pg.odom_sigma_rot,
            odom_sigma_trans=pg.odom_sigma_trans,
            loop_sigma_rot=pg.loop_sigma_rot,
            loop_sigma_trans=pg.loop_sigma_trans,
            robust_loops=pg.robust_loop_kernel,
        )
        optimization = optimize_pose_graph(
            graph,
            optimizer=pg.optimizer,
            max_iterations=pg.max_iterations,
            relative_error_tol=pg.relative_error_tol,
            absolute_error_tol=pg.absolute_error_tol,
            robust_loop_kernel=pg.robust_loop_kernel,
            huber_k=pg.huber_k,
            verbose=pg.verbose,
            timer=self.timer,
        )
        dense = propagate_keyframe_corrections(
            raw_trajectory, keyframe_indices, optimization.trajectory
        )
        return optimization, dense


class _KeyframeHarvester:
    """Callback that turns VO frames into keyframes as the front end runs.

    Tracking travelled distance here (rather than recomputing it later) keeps
    the path-separation loop-closure filter exact even when frames are dropped.
    """

    def __init__(self, database: KeyframeDatabase, selector: KeyframeSelector, enabled: bool = True):
        self.database = database
        self.selector = selector
        self.enabled = enabled
        self.travelled = 0.0
        self._last_position: np.ndarray | None = None

    def __call__(self, index: int, frame, image, pose: np.ndarray) -> None:
        position = pose[:3, 3]
        if self._last_position is not None:
            self.travelled += float(np.linalg.norm(position - self._last_position))
        self._last_position = position.copy()

        if not self.enabled or not frame.is_usable:
            return
        if not self.selector.should_select(index, pose):
            return
        self.selector.accept(index, pose)
        self.database.add(
            Keyframe(
                keyframe_id=-1,
                frame_index=index,
                frame_id=frame.frame_id,
                points=frame.points.copy(),
                descriptors=frame.descriptors.copy(),
                pose=pose.copy(),
                path_distance_m=self.travelled,
            )
        )


def propagate_keyframe_corrections(
    dense: Trajectory, keyframe_indices: list[int], optimized_keyframes: Trajectory
) -> Trajectory:
    """Push keyframe-level corrections back onto every frame.

    The pose graph only optimizes keyframes, but evaluation compares against
    ground truth at every frame. For a frame between keyframes ``a`` and ``b``,
    the correction ``C_a = T_opt_a @ inv(T_raw_a)`` from the preceding keyframe
    is applied.

    Applying the *preceding* keyframe's correction (rather than blending
    between the two) keeps the odometry between keyframes exactly intact — the
    front end's local estimate is the best information available there, and
    interpolating would distort it. The trade-off is a small discontinuity at
    each keyframe boundary, which is bounded by how much the optimizer moved
    that keyframe: metres over a full sequence, centimetres between adjacent
    keyframes.
    """
    if len(keyframe_indices) == 0 or len(optimized_keyframes) == 0:
        return dense.copy()
    if len(keyframe_indices) != len(optimized_keyframes):
        raise ValueError(
            f"keyframe count mismatch: {len(keyframe_indices)} indices vs "
            f"{len(optimized_keyframes)} optimized poses"
        )

    corrections = [
        optimized_keyframes[i] @ invert_se3(dense[frame_index])
        for i, frame_index in enumerate(keyframe_indices)
    ]

    indices = np.asarray(keyframe_indices, dtype=np.int64)
    out = np.empty_like(dense.poses)
    for i in range(len(dense)):
        # searchsorted gives the preceding keyframe for every frame.
        slot = int(np.searchsorted(indices, i, side="right")) - 1
        slot = max(slot, 0)
        out[i] = project_to_se3(corrections[slot] @ dense[i])

    return Trajectory(out, frame_ids=dense.frame_ids.copy(), timestamps=dense.timestamps)


# --------------------------------------------------------------------------- #
# Output serialisation
# --------------------------------------------------------------------------- #


def _round(value, digits: int = 4):
    """Round for JSON output, mapping non-finite values to ``None``."""
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


def build_metrics(result: SlamResult) -> dict[str, object]:
    """Assemble the machine-readable metrics summary.

    Keys whose value cannot be measured on this run (no ground truth, no loop
    closures) are emitted as ``null``. Nothing here is estimated or defaulted.
    """
    config = result.config
    odometry = result.odometry
    n_frames = len(result.raw_trajectory)
    runtime_s = result.timer.elapsed_s

    ate_raw = result.evaluation_raw.ate_aligned.rmse if result.evaluation_raw else None
    ate_optimized = (
        result.evaluation_optimized.ate_aligned.rmse if result.evaluation_optimized else None
    )
    drift_raw = (
        result.evaluation_raw.drift.translation_pct if result.evaluation_raw else None
    )
    drift_optimized = (
        result.evaluation_optimized.drift.translation_pct if result.evaluation_optimized else None
    )
    rotation_drift = (
        result.evaluation_raw.drift.rotation_deg_per_100m if result.evaluation_raw else None
    )
    rotation_drift_optimized = (
        result.evaluation_optimized.drift.rotation_deg_per_100m
        if result.evaluation_optimized
        else None
    )
    distance_m = (
        result.ground_truth.path_length()
        if result.ground_truth is not None
        else result.raw_trajectory.path_length()
    )

    reduction = (
        ate_reduction_pct(ate_raw, ate_optimized)
        if ate_raw is not None and ate_optimized is not None
        else None
    )

    feature_stats = odometry.feature_statistics()

    metrics: dict[str, object] = {
        # --- headline figures, directly usable on a resume --------------- #
        "sequence": config.dataset.sequence,
        "frames": n_frames,
        "distance_km": _round(distance_m / 1000.0, 4),
        "successful_pose_rate_pct": _round(odometry.successful_pose_rate_pct, 2),
        "ate_rmse_raw_m": _round(ate_raw, 4),
        "ate_rmse_optimized_m": _round(ate_optimized, 4),
        "ate_reduction_pct": _round(reduction, 2),
        "translational_drift_raw_pct": _round(drift_raw, 4),
        "translational_drift_optimized_pct": _round(drift_optimized, 4),
        "rotational_drift_deg_per_100m": _round(rotation_drift, 5),
        "rotational_drift_optimized_deg_per_100m": _round(rotation_drift_optimized, 5),
        "loop_closures_detected": len(result.loop_closures),
        "avg_features_per_frame": _round(feature_stats["avg_features_per_frame"], 1),
        "avg_matches_per_pair": _round(feature_stats["avg_matches_per_pair"], 1),
        "avg_inlier_ratio_pct": _round(feature_stats["avg_inlier_ratio_pct"], 2),
        "runtime_fps": _round(n_frames / runtime_s if runtime_s > 0 else None, 2),
        # --- provenance and caveats ------------------------------------- #
        "ground_truth_available": result.has_ground_truth,
        "scale_source": config.odometry.scale_source,
        "scale_uses_ground_truth": bool(odometry.scale_strategy.get("uses_ground_truth", False)),
        "alignment": config.evaluation.alignment,
        # --- detail ------------------------------------------------------ #
        "dataset": result.dataset_info,
        "odometry": {
            "transitions": odometry.n_transitions,
            "successful": odometry.n_successful,
            "status_counts": odometry.status_counts(),
            "degenerate_transitions": odometry.n_degenerate,
            "avg_ransac_inliers": feature_stats["avg_ransac_inliers"],
            "avg_cheirality_ratio_pct": feature_stats["avg_cheirality_ratio_pct"],
            "scale_strategy": odometry.scale_strategy,
        },
        "loop_closure": {
            "enabled": config.loop_closure.enabled,
            "n_keyframes": len(result.keyframe_frame_indices),
            "accepted": len(result.loop_closures),
            "stats": result.loop_stats,
            "closures": [lc.to_dict() for lc in result.loop_closures],
        },
        "optimization": result.optimization.to_dict() if result.optimization else None,
        "evaluation_raw": result.evaluation_raw.to_dict() if result.evaluation_raw else None,
        "evaluation_optimized": (
            result.evaluation_optimized.to_dict() if result.evaluation_optimized else None
        ),
        "timing": {
            "total_runtime_s": _round(runtime_s, 3),
            "avg_frame_latency_ms": _round(runtime_s / n_frames * 1000.0 if n_frames else None, 3),
            "fps": _round(n_frames / runtime_s if runtime_s > 0 else None, 2),
            "stages": result.timer.summary(),
        },
        "environment": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": config.runtime.seed,
        },
    }
    return metrics


def write_outputs(result: SlamResult, output_dir: Path | str | None = None) -> dict[str, str]:
    """Write every artefact for a run and return ``{name: path}``."""
    config = result.config
    out_dir = Path(output_dir) if output_dir is not None else config.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    metrics = build_metrics(result)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    artifacts["metrics"] = str(metrics_path)

    config_path = config.to_yaml(out_dir / "config.resolved.yaml")
    artifacts["config"] = str(config_path)

    if config.output.save_diagnostics:
        artifacts["diagnostics"] = str(write_diagnostics_csv(result, out_dir / "diagnostics.csv"))

    artifacts["trajectory_raw_txt"] = str(
        result.raw_trajectory.save_kitti(out_dir / "trajectory_raw.txt")
    )
    if result.optimized_trajectory is not None:
        artifacts["trajectory_optimized_txt"] = str(
            result.optimized_trajectory.save_kitti(out_dir / "trajectory_optimized.txt")
        )

    if config.output.save_plots:
        artifacts.update(_write_plots(result, out_dir))

    logger.info("Wrote %d artefacts to %s", len(artifacts), out_dir)
    result.artifacts = artifacts
    return artifacts


def _write_plots(result: SlamResult, out_dir: Path) -> dict[str, str]:
    """Render every figure for a run."""
    config = result.config
    dpi = config.output.dpi
    extension = config.output.plot_format
    plots: dict[str, str] = {}
    gt = result.ground_truth

    plots["trajectory_raw"] = str(
        plot_trajectory(
            result.raw_trajectory,
            out_dir / f"trajectory_raw.{extension}",
            reference=gt,
            label="Raw visual odometry",
            title=f"KITTI {config.dataset.sequence} - raw monocular VO",
            dpi=dpi,
        )
    )

    if result.optimized_trajectory is not None:
        plots["trajectory_optimized"] = str(
            plot_trajectory(
                result.optimized_trajectory,
                out_dir / f"trajectory_optimized.{extension}",
                reference=gt,
                label="Optimized SLAM",
                title=f"KITTI {config.dataset.sequence} - after pose-graph optimization",
                dpi=dpi,
            )
        )
        comparison = {
            "Raw VO": result.raw_trajectory,
            "Optimized SLAM": result.optimized_trajectory,
        }
    else:
        comparison = {"Raw VO": result.raw_trajectory}

    plots["trajectory_comparison"] = str(
        plot_trajectory_comparison(
            comparison,
            out_dir / f"trajectory_comparison.{extension}",
            reference=gt,
            title=f"KITTI {config.dataset.sequence} - trajectory comparison",
            dpi=dpi,
        )
    )

    if config.output.plot_3d:
        plots["trajectory_3d"] = str(
            plot_trajectory_3d(
                comparison,
                out_dir / f"trajectory_3d.{extension}",
                reference=gt,
                title=f"KITTI {config.dataset.sequence} - 3D trajectory",
                dpi=dpi,
            )
        )

    loop_pairs = [
        (lc.match_frame_index, lc.query_frame_index) for lc in result.loop_closures
    ]
    plots["loop_closures"] = str(
        plot_loop_closures(
            result.raw_trajectory,
            loop_pairs,
            out_dir / f"loop_closures.{extension}",
            reference=gt,
            title=f"KITTI {config.dataset.sequence} - loop closures",
            dpi=dpi,
        )
    )

    if result.diagnostics_available():
        plots["diagnostics"] = str(
            plot_diagnostics(
                result.odometry.diagnostics,
                out_dir / f"diagnostics.{extension}",
                title=f"KITTI {config.dataset.sequence} - front-end diagnostics",
                dpi=dpi,
            )
        )

    if gt is not None:
        errors = {
            "Raw VO": absolute_trajectory_error(
                result.raw_trajectory, gt, alignment=config.evaluation.alignment
            ).errors
        }
        if result.optimized_trajectory is not None:
            errors["Optimized SLAM"] = absolute_trajectory_error(
                result.optimized_trajectory, gt, alignment=config.evaluation.alignment
            ).errors
        plots["error_over_time"] = str(
            plot_error_over_time(
                errors,
                out_dir / f"error_over_time.{extension}",
                distances=gt.cumulative_distance(),
                title=f"KITTI {config.dataset.sequence} - position error",
                dpi=dpi,
            )
        )

    return plots


def write_diagnostics_csv(result: SlamResult, path: Path | str) -> Path:
    """Write the per-frame diagnostics table."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [d.to_row() for d in result.odometry.diagnostics]
    if not rows:
        out.write_text("", encoding="utf-8")
        return out
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def run_pipeline(
    config: Config,
    output_dir: Path | str | None = None,
    write_artifacts: bool = True,
) -> SlamResult:
    """Convenience wrapper: configure logging, run, and write outputs."""
    out_dir = Path(output_dir) if output_dir is not None else config.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(config.runtime.log_level, log_file=out_dir / "run.log")

    logger.info("=" * 78)
    logger.info("Monocular SLAM | sequence %s | %s", config.dataset.sequence, out_dir)
    logger.info("=" * 78)

    result = SlamPipeline(config).run()
    if write_artifacts:
        write_outputs(result, out_dir)

    logger.info("Stage timings:\n%s", result.timer.report())
    return result
