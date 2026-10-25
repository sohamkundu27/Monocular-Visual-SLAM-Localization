"""Tests for Umeyama trajectory alignment."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import random_se3
from monocular_slam.evaluation.alignment import (
    AlignmentResult,
    align_trajectory,
    umeyama_alignment,
)
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import rotation_angle_deg, se3_from_rt
from synthetic import straight_drive


class TestUmeyamaCore:
    def test_identical_sets_give_identity(self, rng):
        points = rng.normal(size=(30, 3)) * 10
        result = umeyama_alignment(points, points)
        np.testing.assert_allclose(result.R, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(result.t, np.zeros(3), atol=1e-9)
        assert result.scale == pytest.approx(1.0, abs=1e-9)

    def test_recovers_known_similarity_transform(self, rng):
        points = rng.normal(size=(50, 3)) * 5
        R_true = Rotation.from_euler("xyz", [20, -35, 60], degrees=True).as_matrix()
        t_true = np.array([3.0, -7.0, 12.0])
        s_true = 2.7
        transformed = (s_true * (R_true @ points.T)).T + t_true

        result = umeyama_alignment(points, transformed)
        assert result.scale == pytest.approx(s_true, rel=1e-9)
        np.testing.assert_allclose(result.R, R_true, atol=1e-9)
        np.testing.assert_allclose(result.t, t_true, atol=1e-8)
        np.testing.assert_allclose(result.apply(points), transformed, atol=1e-8)

    def test_rigid_mode_ignores_scale(self, rng):
        points = rng.normal(size=(40, 3)) * 5
        scaled = points * 3.0
        result = umeyama_alignment(points, scaled, with_scale=False)
        assert result.scale == 1.0
        # Without a scale degree of freedom the fit cannot match the target.
        assert np.linalg.norm(result.apply(points) - scaled) > 1.0

    def test_rigid_mode_recovers_rigid_transform(self, rng):
        points = rng.normal(size=(40, 3)) * 5
        T = random_se3(rng)
        moved = (T[:3, :3] @ points.T).T + T[:3, 3]
        result = umeyama_alignment(points, moved, with_scale=False)
        np.testing.assert_allclose(result.R, T[:3, :3], atol=1e-9)
        np.testing.assert_allclose(result.apply(points), moved, atol=1e-8)

    def test_result_is_always_a_rotation_not_a_reflection(self, rng):
        """A mirrored target must not produce det(R) == -1."""
        points = rng.normal(size=(30, 3))
        mirrored = points * np.array([1.0, 1.0, -1.0])
        result = umeyama_alignment(points, mirrored)
        assert np.linalg.det(result.R) == pytest.approx(1.0, abs=1e-9)
        np.testing.assert_allclose(result.R.T @ result.R, np.eye(3), atol=1e-9)

    def test_collinear_points_still_yield_a_rotation(self):
        """Degenerate (rank-deficient) geometry must not produce a reflection."""
        points = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
        target = np.column_stack([np.zeros(10), np.arange(10.0), np.zeros(10)])
        result = umeyama_alignment(points, target)
        assert np.linalg.det(result.R) == pytest.approx(1.0, abs=1e-6)

    def test_noise_gives_approximate_recovery(self, rng):
        points = rng.normal(size=(200, 3)) * 5
        R_true = Rotation.from_euler("y", 30, degrees=True).as_matrix()
        target = (1.5 * (R_true @ points.T)).T + np.array([1.0, 2.0, 3.0])
        target = target + rng.normal(scale=0.02, size=target.shape)
        result = umeyama_alignment(points, target)
        assert result.scale == pytest.approx(1.5, rel=0.02)
        assert rotation_angle_deg(result.R.T @ R_true) < 0.5

    def test_stationary_source_does_not_divide_by_zero(self):
        points = np.zeros((5, 3))
        result = umeyama_alignment(points, np.ones((5, 3)))
        assert result.scale == 1.0
        assert np.all(np.isfinite(result.t))

    @pytest.mark.parametrize(
        "source,target",
        [
            (np.zeros((5, 3)), np.zeros((4, 3))),
            (np.zeros((5, 2)), np.zeros((5, 2))),
            (np.zeros((2, 3)), np.zeros((2, 3))),
        ],
    )
    def test_invalid_inputs_raise(self, source, target):
        with pytest.raises(ValueError):
            umeyama_alignment(source, target)


class TestAlignmentResult:
    def test_T_holds_rotation_and_translation_without_scale(self):
        result = AlignmentResult(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]), scale=5.0, mode="sim3")
        np.testing.assert_allclose(result.T[:3, :3], np.eye(3))
        np.testing.assert_allclose(result.T[:3, 3], [1, 2, 3])

    def test_identity_result(self):
        result = AlignmentResult.identity()
        np.testing.assert_allclose(result.apply(np.ones((3, 3))), np.ones((3, 3)))

    def test_describe_is_serialisable(self):
        info = umeyama_alignment(np.eye(3), np.eye(3) * 2).describe()
        assert info["mode"] == "sim3"
        assert isinstance(info["scale"], float)


class TestTrajectoryAlignment:
    @pytest.fixture
    def reference(self):
        return Trajectory(straight_drive(40))

    def test_sim3_recovers_a_scaled_rotated_trajectory(self, reference):
        """The classic monocular case: right shape, wrong scale and frame."""
        T = se3_from_rt(
            Rotation.from_euler("y", 47, degrees=True).as_matrix(),
            np.array([5.0, 0.0, -3.0]),
        )
        distorted = reference.transformed(T, scale=0.35)
        aligned, result = align_trajectory(distorted, reference, mode="sim3")
        assert result.scale == pytest.approx(1 / 0.35, rel=1e-6)
        np.testing.assert_allclose(aligned.positions, reference.positions, atol=1e-8)

    def test_se3_cannot_undo_a_scale_error(self, reference):
        distorted = reference.transformed(np.eye(4), scale=0.5)
        aligned, result = align_trajectory(distorted, reference, mode="se3")
        assert result.scale == 1.0
        assert np.linalg.norm(aligned.positions - reference.positions, axis=1).max() > 1.0

    def test_none_mode_is_a_no_op(self, reference):
        distorted = reference.transformed(np.eye(4), scale=0.5)
        aligned, result = align_trajectory(distorted, reference, mode="none")
        np.testing.assert_allclose(aligned.positions, distorted.positions)
        assert result.scale == 1.0
        assert result.mode == "none"

    def test_alignment_preserves_frame_ids(self, reference):
        distorted = reference.transformed(np.eye(4), scale=2.0)
        aligned, _ = align_trajectory(distorted, reference)
        np.testing.assert_array_equal(aligned.frame_ids, distorted.frame_ids)

    def test_aligned_poses_remain_valid_se3(self, reference):
        distorted = reference.transformed(np.eye(4), scale=0.25)
        aligned, _ = align_trajectory(distorted, reference)
        assert aligned.is_valid()

    def test_length_mismatch_raises(self, reference):
        with pytest.raises(ValueError, match="equal length"):
            align_trajectory(reference[:10], reference)

    def test_unknown_mode_raises(self, reference):
        with pytest.raises(ValueError, match="Unknown alignment mode"):
            align_trajectory(reference, reference, mode="procrustes")
