"""Integration tests for the SLAM pipeline.

These run against the synthetic KITTI tree from ``kitti_fixture``, so the whole
suite works without the 20 GB dataset download. They verify wiring and output
contracts, not accuracy — accuracy is measured by the real benchmark runs
recorded in the README.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from kitti_fixture import write_kitti_sequence
from monocular_slam.config import Config
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import invert_se3, se3_from_rt
from monocular_slam.pipeline import (
    SlamPipeline,
    build_metrics,
    propagate_keyframe_corrections,
    run_pipeline,
    write_outputs,
)
from synthetic import straight_drive


@pytest.fixture
def kitti_root(tmp_path):
    """A 40-frame synthetic sequence: enough for a full pipeline pass."""
    return write_kitti_sequence(tmp_path / "dataset", "00", n_frames=40)


@pytest.fixture
def config(kitti_root, tmp_path):
    return Config().with_overrides(
        {
            "dataset.path": str(kitti_root),
            "dataset.sequence": "00",
            "output.root": str(tmp_path / "outputs"),
            "runtime.log_level": "ERROR",
            "runtime.progress_interval": 0,
            "features.max_features": 400,
            "matcher.min_matches": 8,
            "odometry.min_inliers": 6,
            "odometry.min_inlier_ratio": 0.05,
            "loop_closure.enabled": False,
            "output.plot_3d": False,
            "evaluation.rpe_deltas": [1, 5],
        }
    )


class TestPipelineRun:
    def test_produces_a_trajectory_of_the_right_length(self, config):
        result = SlamPipeline(config).run()
        assert len(result.raw_trajectory) == 40
        assert result.raw_trajectory.is_valid()

    def test_records_one_diagnostic_per_transition(self, config):
        result = SlamPipeline(config).run()
        assert len(result.odometry.diagnostics) == 39
        assert all(d.index == i + 1 for i, d in enumerate(result.odometry.diagnostics))

    def test_ground_truth_is_loaded_and_evaluated(self, config):
        result = SlamPipeline(config).run()
        assert result.has_ground_truth
        assert result.evaluation_raw is not None
        assert np.isfinite(result.evaluation_raw.ate_aligned.rmse)

    def test_runs_without_ground_truth(self, tmp_path):
        root = write_kitti_sequence(
            tmp_path / "d", "00", n_frames=30, with_ground_truth=False
        )
        config = Config().with_overrides(
            {
                "dataset.path": str(root),
                "output.root": str(tmp_path / "out"),
                "runtime.log_level": "ERROR",
                "runtime.progress_interval": 0,
                "features.max_features": 400,
                "matcher.min_matches": 8,
                "odometry.min_inliers": 6,
                "odometry.min_inlier_ratio": 0.05,
                "loop_closure.enabled": False,
            }
        )
        result = SlamPipeline(config).run()
        assert not result.has_ground_truth
        assert result.evaluation_raw is None
        # Ground-truth scale must fall back rather than fabricate magnitudes.
        assert result.odometry.scale_strategy["uses_ground_truth"] is False

    def test_timing_is_recorded_for_each_stage(self, config):
        result = SlamPipeline(config).run()
        summary = result.timer.summary()
        assert {"detect", "match", "pose"} <= set(summary)
        assert summary["detect"]["calls"] == 40

    def test_scale_source_none_is_honoured(self, config):
        result = SlamPipeline(config.with_overrides({"odometry.scale_source": "none"})).run()
        assert result.odometry.scale_strategy["strategy"] == "unit"
        assert result.odometry.scale_strategy["uses_ground_truth"] is False

    def test_keyframes_are_harvested_when_loop_closure_is_on(self, config):
        result = SlamPipeline(
            config.with_overrides({"loop_closure.enabled": True, "keyframes.every_n_frames": 3})
        ).run()
        assert len(result.keyframe_frame_indices) > 5
        assert result.keyframe_frame_indices == sorted(result.keyframe_frame_indices)

    def test_no_keyframes_harvested_when_loop_closure_is_off(self, config):
        assert SlamPipeline(config).run().keyframe_frame_indices == []

    def test_too_few_frames_raises_clearly(self, tmp_path):
        root = write_kitti_sequence(tmp_path / "d", "00", n_frames=1)
        config = Config().with_overrides(
            {"dataset.path": str(root), "runtime.log_level": "ERROR"}
        )
        with pytest.raises(ValueError, match="at least 2 frames"):
            SlamPipeline(config).run()


class TestKeyframeCorrectionPropagation:
    def test_identity_correction_leaves_the_trajectory_alone(self):
        dense = Trajectory(straight_drive(30))
        indices = [0, 10, 20]
        keyframes = Trajectory(np.stack([dense[i] for i in indices]))
        result = propagate_keyframe_corrections(dense, indices, keyframes)
        np.testing.assert_allclose(result.poses, dense.poses, atol=1e-9)

    def test_correction_is_applied_exactly_at_keyframes(self):
        dense = Trajectory(straight_drive(30))
        indices = [0, 10, 20]
        shift = se3_from_rt(np.eye(3), np.array([5.0, 0.0, 0.0]))
        keyframes = Trajectory(np.stack([shift @ dense[i] for i in indices]))
        result = propagate_keyframe_corrections(dense, indices, keyframes)
        for i in indices:
            np.testing.assert_allclose(result[i], shift @ dense[i], atol=1e-9)

    def test_relative_motion_between_keyframes_is_preserved(self):
        """The front end's local estimate must survive propagation untouched."""
        dense = Trajectory(straight_drive(30))
        indices = [0, 10, 20]
        rng = np.random.default_rng(0)
        keyframes = Trajectory(
            np.stack(
                [se3_from_rt(np.eye(3), rng.normal(size=3)) @ dense[i] for i in indices]
            )
        )
        result = propagate_keyframe_corrections(dense, indices, keyframes)
        for i in range(11, 20):  # strictly between two keyframes
            expected = invert_se3(dense[i - 1]) @ dense[i]
            actual = invert_se3(result[i - 1]) @ result[i]
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_frames_before_the_first_keyframe_use_the_first_correction(self):
        dense = Trajectory(straight_drive(20))
        indices = [5, 15]
        shift = se3_from_rt(np.eye(3), np.array([2.0, 0.0, 0.0]))
        keyframes = Trajectory(np.stack([shift @ dense[i] for i in indices]))
        result = propagate_keyframe_corrections(dense, indices, keyframes)
        np.testing.assert_allclose(result[0], shift @ dense[0], atol=1e-9)

    def test_output_stays_valid_se3(self):
        dense = Trajectory(straight_drive(25))
        indices = [0, 12, 24]
        rng = np.random.default_rng(1)
        keyframes = Trajectory(
            np.stack([se3_from_rt(np.eye(3), rng.normal(size=3)) @ dense[i] for i in indices])
        )
        assert propagate_keyframe_corrections(dense, indices, keyframes).is_valid()

    def test_frame_ids_are_preserved(self):
        dense = Trajectory(straight_drive(20), frame_ids=np.arange(20) * 2)
        indices = [0, 10]
        keyframes = Trajectory(np.stack([dense[i] for i in indices]))
        result = propagate_keyframe_corrections(dense, indices, keyframes)
        np.testing.assert_array_equal(result.frame_ids, np.arange(20) * 2)

    def test_no_keyframes_returns_a_copy(self):
        dense = Trajectory(straight_drive(10))
        result = propagate_keyframe_corrections(dense, [], Trajectory(np.zeros((0, 4, 4))))
        np.testing.assert_allclose(result.poses, dense.poses)

    def test_count_mismatch_raises(self):
        dense = Trajectory(straight_drive(10))
        with pytest.raises(ValueError, match="keyframe count mismatch"):
            propagate_keyframe_corrections(dense, [0, 5], Trajectory(dense[0]))


class TestMetrics:
    def test_metrics_are_json_serialisable(self, config):
        metrics = build_metrics(SlamPipeline(config).run())
        json.loads(json.dumps(metrics))

    def test_resume_keys_are_all_present(self, config):
        """The headline schema must be stable, even when a value is null."""
        metrics = build_metrics(SlamPipeline(config).run())
        required = {
            "sequence", "frames", "distance_km", "successful_pose_rate_pct",
            "ate_rmse_raw_m", "ate_rmse_optimized_m", "ate_reduction_pct",
            "translational_drift_raw_pct", "translational_drift_optimized_pct",
            "rotational_drift_deg_per_100m", "loop_closures_detected",
            "avg_features_per_frame", "avg_matches_per_pair", "avg_inlier_ratio_pct",
            "runtime_fps",
        }
        assert required <= set(metrics)

    def test_unmeasurable_values_are_null_not_invented(self, config):
        """No loop closures means no optimized trajectory, so those keys are null."""
        metrics = build_metrics(SlamPipeline(config).run())
        assert metrics["loop_closures_detected"] == 0
        assert metrics["ate_rmse_optimized_m"] is None
        assert metrics["ate_reduction_pct"] is None
        assert metrics["translational_drift_optimized_pct"] is None

    def test_ground_truth_scale_use_is_flagged(self, config):
        metrics = build_metrics(SlamPipeline(config).run())
        assert metrics["scale_source"] == "ground_truth"
        assert metrics["scale_uses_ground_truth"] is True

    def test_scale_free_run_is_flagged_as_such(self, config):
        result = SlamPipeline(config.with_overrides({"odometry.scale_source": "none"})).run()
        assert build_metrics(result)["scale_uses_ground_truth"] is False

    def test_frame_and_distance_counts_are_real(self, config):
        result = SlamPipeline(config).run()
        metrics = build_metrics(result)
        assert metrics["frames"] == 40
        assert metrics["distance_km"] == pytest.approx(
            result.ground_truth.path_length() / 1000.0, rel=1e-3
        )

    def test_timing_block_is_populated(self, config):
        timing = build_metrics(SlamPipeline(config).run())["timing"]
        assert timing["total_runtime_s"] > 0
        assert timing["fps"] > 0
        assert "detect" in timing["stages"]


class TestOutputs:
    def test_writes_the_expected_artifacts(self, config, tmp_path):
        result = SlamPipeline(config).run()
        out_dir = tmp_path / "run"
        artifacts = write_outputs(result, out_dir)

        assert (out_dir / "metrics.json").is_file()
        assert (out_dir / "diagnostics.csv").is_file()
        assert (out_dir / "config.resolved.yaml").is_file()
        assert (out_dir / "trajectory_raw.txt").is_file()
        assert (out_dir / "trajectory_raw.png").is_file()
        assert (out_dir / "trajectory_comparison.png").is_file()
        assert (out_dir / "loop_closures.png").is_file()
        assert "metrics" in artifacts

    def test_diagnostics_csv_has_one_row_per_transition(self, config, tmp_path):
        result = SlamPipeline(config).run()
        write_outputs(result, tmp_path / "run")
        with (tmp_path / "run" / "diagnostics.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 39
        assert {"index", "frame_id", "status", "n_matches", "n_ransac_inliers"} <= set(rows[0])

    def test_saved_trajectory_round_trips(self, config, tmp_path):
        result = SlamPipeline(config).run()
        write_outputs(result, tmp_path / "run")
        loaded = Trajectory.load_kitti(tmp_path / "run" / "trajectory_raw.txt")
        np.testing.assert_allclose(loaded.positions, result.raw_trajectory.positions, atol=1e-6)

    def test_resolved_config_can_be_reloaded(self, config, tmp_path):
        result = SlamPipeline(config).run()
        write_outputs(result, tmp_path / "run")
        reloaded = Config.from_yaml(tmp_path / "run" / "config.resolved.yaml")
        assert reloaded.dataset.sequence == config.dataset.sequence
        assert reloaded.odometry.min_inliers == config.odometry.min_inliers

    def test_plots_can_be_disabled(self, config, tmp_path):
        result = SlamPipeline(config.with_overrides({"output.save_plots": False})).run()
        write_outputs(result, tmp_path / "run")
        assert not (tmp_path / "run" / "trajectory_raw.png").exists()
        assert (tmp_path / "run" / "metrics.json").is_file()

    def test_run_pipeline_writes_a_log(self, config, tmp_path):
        out_dir = tmp_path / "run"
        run_pipeline(config, output_dir=out_dir)
        assert (out_dir / "run.log").is_file()
        assert (out_dir / "run.log").stat().st_size > 0


class TestLoopClosureIntegration:
    def test_pipeline_survives_a_sequence_with_no_loops(self, config):
        """Loop closure enabled but nothing to find must not break the run."""
        result = SlamPipeline(
            config.with_overrides(
                {
                    "loop_closure.enabled": True,
                    "keyframes.every_n_frames": 2,
                    "loop_closure.min_keyframe_separation": 5,
                    "loop_closure.vocabulary_size": 32,
                }
            )
        ).run()
        assert result.loop_closures == []
        assert result.optimized_trajectory is None
        assert result.evaluation_optimized is None

    def test_short_sequence_skips_loop_closure_gracefully(self, config):
        result = SlamPipeline(
            config.with_overrides(
                {"loop_closure.enabled": True, "loop_closure.min_keyframe_separation": 10_000}
            )
        ).run()
        assert result.loop_closures == []
