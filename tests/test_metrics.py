"""Tests for trajectory error metrics.

Errors are injected analytically so the expected metric value is known in
closed form, which is the only way to catch a formula that is merely plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from synthetic import straight_drive

from monocular_slam.evaluation.metrics import (
    absolute_trajectory_error,
    ate_reduction_pct,
    evaluate_trajectory,
    kitti_drift,
    relative_pose_error,
)
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import se3_from_rt


@pytest.fixture
def reference():
    """A 40-frame curving drive, ~35 m of travel."""
    return Trajectory(straight_drive(40, step=0.9, yaw_rate_deg=0.8))


@pytest.fixture
def long_reference():
    """Long enough (~450 m) that the 100-400 m drift segments exist."""
    return Trajectory(straight_drive(500, step=0.9, yaw_rate_deg=0.2))


def offset_trajectory(trajectory: Trajectory, offset: np.ndarray) -> Trajectory:
    """Shift every camera centre by a constant vector."""
    poses = trajectory.poses.copy()
    poses[:, :3, 3] += offset
    return Trajectory(poses, trajectory.frame_ids.copy())


class TestATE:
    def test_identical_trajectories_have_zero_error(self, reference):
        result = absolute_trajectory_error(reference, reference, alignment="none")
        assert result.rmse == pytest.approx(0.0, abs=1e-12)
        assert result.max == pytest.approx(0.0, abs=1e-12)
        assert result.n == len(reference)

    def test_constant_offset_gives_exactly_that_error(self, reference):
        """A rigid 3-4-0 offset must produce exactly 5 m of unaligned error."""
        shifted = offset_trajectory(reference, np.array([3.0, 4.0, 0.0]))
        result = absolute_trajectory_error(shifted, reference, alignment="none")
        assert result.rmse == pytest.approx(5.0)
        assert result.mean == pytest.approx(5.0)
        assert result.median == pytest.approx(5.0)
        assert result.std == pytest.approx(0.0, abs=1e-9)

    def test_alignment_removes_a_rigid_offset(self, reference):
        shifted = offset_trajectory(reference, np.array([3.0, 4.0, 0.0]))
        assert absolute_trajectory_error(shifted, reference, alignment="se3").rmse < 1e-9

    def test_sim3_alignment_removes_a_scale_error(self, reference):
        scaled = reference.transformed(np.eye(4), scale=0.4)
        assert absolute_trajectory_error(scaled, reference, alignment="sim3").rmse < 1e-9
        # Rigid alignment cannot: the shape is right but the size is wrong.
        assert absolute_trajectory_error(scaled, reference, alignment="se3").rmse > 1.0

    def test_alignment_scale_is_reported(self, reference):
        scaled = reference.transformed(np.eye(4), scale=0.4)
        result = absolute_trajectory_error(scaled, reference, alignment="sim3")
        assert result.alignment == "sim3"
        assert result.alignment_scale == pytest.approx(2.5, rel=1e-6)

    def test_rmse_identity_holds(self, reference):
        """RMSE^2 == mean^2 + std^2 for the error magnitudes."""
        rng = np.random.default_rng(3)
        noisy = reference.poses.copy()
        noisy[:, :3, 3] += rng.normal(scale=0.5, size=(len(reference), 3))
        result = absolute_trajectory_error(Trajectory(noisy), reference, alignment="none")
        assert result.rmse**2 == pytest.approx(result.mean**2 + result.std**2, rel=1e-9)

    def test_statistics_are_ordered_sensibly(self, reference):
        rng = np.random.default_rng(11)
        noisy = reference.poses.copy()
        noisy[:, :3, 3] += rng.normal(scale=0.5, size=(len(reference), 3))
        result = absolute_trajectory_error(Trajectory(noisy), reference, alignment="none")
        assert result.min <= result.median <= result.max
        assert result.mean <= result.rmse  # by Jensen's inequality

    def test_length_mismatch_raises(self, reference):
        with pytest.raises(ValueError, match="same length"):
            absolute_trajectory_error(reference[:10], reference)

    def test_empty_trajectory_raises(self):
        empty = Trajectory(np.zeros((0, 4, 4)))
        with pytest.raises(ValueError):
            absolute_trajectory_error(empty, empty)

    def test_to_dict_excludes_raw_errors(self, reference):
        data = absolute_trajectory_error(reference, reference, alignment="none").to_dict()
        assert "errors" not in data
        assert set(data) >= {"rmse", "mean", "median", "std", "min", "max", "n"}


class TestRPE:
    def test_identical_trajectories_have_zero_error(self, reference):
        result = relative_pose_error(reference, reference, delta=1)
        assert result.trans_rmse == pytest.approx(0.0, abs=1e-12)
        assert result.rot_rmse_deg == pytest.approx(0.0, abs=1e-9)
        assert result.n == len(reference) - 1

    def test_rpe_ignores_a_global_offset(self, reference):
        """This invariance is the whole point of RPE."""
        shifted = offset_trajectory(reference, np.array([50.0, -20.0, 7.0]))
        result = relative_pose_error(shifted, reference, delta=1)
        assert result.trans_rmse == pytest.approx(0.0, abs=1e-9)

    def test_rpe_ignores_a_global_rotation(self, reference):
        T = se3_from_rt(Rotation.from_euler("y", 63, degrees=True).as_matrix(), np.zeros(3))
        rotated = reference.transformed(T)
        result = relative_pose_error(rotated, reference, delta=1)
        assert result.trans_rmse == pytest.approx(0.0, abs=1e-9)
        assert result.rot_rmse_deg == pytest.approx(0.0, abs=1e-8)

    def test_scale_error_produces_known_translation_error(self):
        """Halving each 1 m step must give exactly 0.5 m of RPE at delta=1."""
        poses = np.repeat(np.eye(4)[None], 11, axis=0)
        poses[:, 2, 3] = np.arange(11) * 1.0
        reference = Trajectory(poses)
        halved = reference.transformed(np.eye(4), scale=0.5)
        result = relative_pose_error(halved, reference, delta=1)
        assert result.trans_rmse == pytest.approx(0.5, rel=1e-9)

    def test_scale_argument_compensates(self):
        poses = np.repeat(np.eye(4)[None], 11, axis=0)
        poses[:, 2, 3] = np.arange(11) * 1.0
        reference = Trajectory(poses)
        halved = reference.transformed(np.eye(4), scale=0.5)
        result = relative_pose_error(halved, reference, delta=1, scale=2.0)
        assert result.trans_rmse == pytest.approx(0.0, abs=1e-9)

    def test_rotation_error_is_measured_in_degrees(self):
        """A 5 deg per-step yaw error must show up as exactly 5 deg of RPE."""
        n = 12
        reference = Trajectory(np.repeat(np.eye(4)[None], n, axis=0))
        poses = [np.eye(4)]
        R_err = Rotation.from_euler("y", 5.0, degrees=True).as_matrix()
        for _ in range(n - 1):
            poses.append(poses[-1] @ se3_from_rt(R_err, np.zeros(3)))
        result = relative_pose_error(Trajectory(np.stack(poses)), reference, delta=1)
        assert result.rot_rmse_deg == pytest.approx(5.0, abs=1e-6)
        assert result.rot_median_deg == pytest.approx(5.0, abs=1e-6)

    def test_larger_delta_reduces_pair_count(self, reference):
        assert relative_pose_error(reference, reference, delta=5).n == len(reference) - 5

    def test_invalid_delta_raises(self, reference):
        with pytest.raises(ValueError, match="delta must be"):
            relative_pose_error(reference, reference, delta=0)
        with pytest.raises(ValueError, match="exceeds"):
            relative_pose_error(reference, reference, delta=len(reference))


class TestKittiDrift:
    def test_perfect_trajectory_has_zero_drift(self, long_reference):
        result = kitti_drift(long_reference, long_reference)
        assert result.n_segments > 0
        assert result.translation_pct == pytest.approx(0.0, abs=1e-9)
        assert result.rotation_deg_per_m == pytest.approx(0.0, abs=1e-9)

    def test_scale_error_gives_predictable_drift(self):
        """A uniform 10% scale shortfall on a straight run gives ~10% drift."""
        poses = np.repeat(np.eye(4)[None], 600, axis=0)
        poses[:, 2, 3] = np.arange(600) * 1.0
        reference = Trajectory(poses)
        shrunk = reference.transformed(np.eye(4), scale=0.9)
        result = kitti_drift(shrunk, reference, segment_lengths=(100.0, 200.0))
        assert result.translation_pct == pytest.approx(10.0, rel=0.02)

    def test_rotation_units_are_consistent(self, long_reference):
        result = kitti_drift(long_reference, long_reference)
        assert result.rotation_deg_per_100m == pytest.approx(result.rotation_deg_per_m * 100.0)

    def test_per_length_breakdown_is_populated(self, long_reference):
        result = kitti_drift(long_reference, long_reference, segment_lengths=(100.0, 200.0))
        assert set(result.per_length) == {100.0, 200.0}

    def test_short_trajectory_yields_nan_not_a_crash(self, reference):
        """35 m of travel cannot support a 100 m sub-trajectory."""
        result = kitti_drift(reference, reference, segment_lengths=(100.0,))
        assert result.n_segments == 0
        assert np.isnan(result.translation_pct)

    def test_segments_longer_than_the_path_are_skipped(self, long_reference):
        result = kitti_drift(long_reference, long_reference, segment_lengths=(100.0, 10_000.0))
        assert 10_000.0 not in result.per_length
        assert 100.0 in result.per_length

    def test_invalid_step_raises(self, long_reference):
        with pytest.raises(ValueError, match="step must be"):
            kitti_drift(long_reference, long_reference, step=0)


class TestCombinedEvaluation:
    def test_reports_every_metric(self, long_reference):
        evaluation = evaluate_trajectory(
            long_reference, long_reference, label="perfect", rpe_deltas=(1, 10)
        )
        assert evaluation.label == "perfect"
        assert evaluation.ate_aligned.rmse == pytest.approx(0.0, abs=1e-8)
        assert set(evaluation.rpe) == {1, 10}
        assert evaluation.drift.n_segments > 0
        assert evaluation.path_length_m == pytest.approx(evaluation.reference_path_length_m)

    def test_oversized_rpe_delta_is_skipped_not_fatal(self, reference):
        evaluation = evaluate_trajectory(reference, reference, rpe_deltas=(1, 10_000))
        assert set(evaluation.rpe) == {1}

    def test_to_dict_is_json_serialisable(self, long_reference):
        import json

        data = evaluate_trajectory(long_reference, long_reference, rpe_deltas=(1,)).to_dict()
        json.loads(json.dumps(data))  # raises if anything is not serialisable
        assert "ate_aligned" in data and "drift" in data

    def test_summary_line_mentions_the_numbers(self, long_reference):
        line = evaluate_trajectory(long_reference, long_reference, rpe_deltas=(1,)).summary_line()
        assert "ATE RMSE" in line and "drift" in line


class TestATEReduction:
    def test_improvement_is_positive(self):
        assert ate_reduction_pct(10.0, 4.0) == pytest.approx(60.0)

    def test_regression_is_negative(self):
        assert ate_reduction_pct(4.0, 5.0) == pytest.approx(-25.0)

    def test_no_change_is_zero(self):
        assert ate_reduction_pct(7.0, 7.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("before", [0.0, -1.0, float("nan"), float("inf")])
    def test_undefined_cases_return_nan(self, before):
        assert np.isnan(ate_reduction_pct(before, 1.0))
