"""Tests for SE(3)/SO(3) transform utilities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import random_se3, small_se3
from monocular_slam.geometry.transforms import (
    compose,
    invert_se3,
    is_rotation_matrix,
    is_valid_se3,
    kitti_row_to_se3,
    project_to_se3,
    project_to_so3,
    relative_pose,
    rotation_angle_deg,
    rotation_angle_rad,
    se3_exp,
    se3_from_rt,
    se3_log,
    se3_to_kitti_row,
    split_se3,
    transform_points,
)


class TestConstruction:
    def test_se3_from_rt_places_blocks_correctly(self):
        R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        t = np.array([1.0, 2.0, 3.0])
        T = se3_from_rt(R, t)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[:3, :3], R)
        np.testing.assert_allclose(T[:3, 3], t)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1])

    @pytest.mark.parametrize("shape", [(3,), (3, 1), (1, 3)])
    def test_translation_shapes_accepted(self, shape):
        T = se3_from_rt(np.eye(3), np.arange(3.0).reshape(shape))
        np.testing.assert_allclose(T[:3, 3], [0, 1, 2])

    def test_split_round_trips(self, rng):
        T = random_se3(rng)
        R, t = split_se3(T)
        np.testing.assert_allclose(se3_from_rt(R, t), T)

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            se3_from_rt(np.eye(2), np.zeros(3))
        with pytest.raises(ValueError):
            split_se3(np.eye(3))


class TestInversionAndComposition:
    def test_inverse_is_exact(self, rng):
        for _ in range(20):
            T = random_se3(rng)
            np.testing.assert_allclose(T @ invert_se3(T), np.eye(4), atol=1e-12)
            np.testing.assert_allclose(invert_se3(T) @ T, np.eye(4), atol=1e-12)

    def test_inverse_matches_dense_inverse(self, rng):
        T = random_se3(rng)
        np.testing.assert_allclose(invert_se3(T), np.linalg.inv(T), atol=1e-10)

    def test_compose_is_left_to_right(self, rng):
        A, B, C = (random_se3(rng) for _ in range(3))
        np.testing.assert_allclose(compose(A, B, C), A @ B @ C, atol=1e-12)

    def test_compose_of_nothing_is_identity(self):
        np.testing.assert_allclose(compose(), np.eye(4))

    def test_relative_pose_definition(self, rng):
        """T_a_b must satisfy T_w_a @ T_a_b == T_w_b."""
        T_w_a, T_w_b = random_se3(rng), random_se3(rng)
        T_a_b = relative_pose(T_w_a, T_w_b)
        np.testing.assert_allclose(T_w_a @ T_a_b, T_w_b, atol=1e-10)

    def test_relative_pose_of_self_is_identity(self, rng):
        T = random_se3(rng)
        np.testing.assert_allclose(relative_pose(T, T), np.eye(4), atol=1e-12)

    def test_chained_relative_poses_reconstruct_trajectory(self, rng):
        """Accumulating relative motions must reproduce the absolute poses.

        This is the exact operation the VO trajectory builder performs.
        """
        poses = [np.eye(4)]
        for _ in range(30):
            poses.append(poses[-1] @ small_se3(rng))
        rebuilt = [np.eye(4)]
        for i in range(1, len(poses)):
            rebuilt.append(rebuilt[-1] @ relative_pose(poses[i - 1], poses[i]))
        for expected, actual in zip(poses, rebuilt):
            np.testing.assert_allclose(actual, expected, atol=1e-9)


class TestPointTransforms:
    def test_transform_points_matches_manual(self, rng):
        T = random_se3(rng)
        pts = rng.normal(size=(50, 3))
        R, t = split_se3(T)
        np.testing.assert_allclose(transform_points(T, pts), (R @ pts.T).T + t, atol=1e-12)

    def test_transform_then_inverse_recovers_points(self, rng):
        T = random_se3(rng)
        pts = rng.normal(size=(25, 3))
        moved = transform_points(T, pts)
        np.testing.assert_allclose(transform_points(invert_se3(T), moved), pts, atol=1e-10)

    def test_wrong_point_shape_raises(self):
        with pytest.raises(ValueError):
            transform_points(np.eye(4), np.zeros((5, 2)))


class TestValidation:
    def test_identity_is_valid(self):
        assert is_rotation_matrix(np.eye(3))
        assert is_valid_se3(np.eye(4))

    def test_random_rotations_are_valid(self, rng):
        for _ in range(20):
            assert is_rotation_matrix(random_se3(rng)[:3, :3])

    def test_reflection_rejected(self):
        """det == -1 is the classic degenerate essential-matrix decomposition."""
        R = np.diag([1.0, 1.0, -1.0])
        assert np.allclose(R.T @ R, np.eye(3))
        assert not is_rotation_matrix(R)

    def test_scaled_rotation_rejected(self):
        assert not is_rotation_matrix(1.5 * np.eye(3))

    def test_nan_rejected(self):
        R = np.eye(3).copy()
        R[0, 0] = np.nan
        assert not is_rotation_matrix(R)
        T = np.eye(4)
        T[0, 3] = np.inf
        assert not is_valid_se3(T)

    def test_broken_bottom_row_rejected(self):
        T = np.eye(4)
        T[3, 0] = 0.5
        assert not is_valid_se3(T)


class TestProjection:
    def test_project_to_so3_is_identity_on_rotations(self, rng):
        R = random_se3(rng)[:3, :3]
        np.testing.assert_allclose(project_to_so3(R), R, atol=1e-12)

    def test_project_to_so3_repairs_perturbation(self, rng):
        R = random_se3(rng)[:3, :3]
        noisy = R + rng.normal(scale=1e-3, size=(3, 3))
        assert not is_rotation_matrix(noisy)
        fixed = project_to_so3(noisy)
        assert is_rotation_matrix(fixed)
        assert rotation_angle_deg(R.T @ fixed) < 0.5

    def test_project_never_returns_reflection(self):
        fixed = project_to_so3(np.diag([1.0, 1.0, -1.0]))
        assert is_rotation_matrix(fixed)
        assert np.linalg.det(fixed) > 0

    def test_project_to_se3_preserves_translation(self, rng):
        T = random_se3(rng)
        T[:3, :3] += rng.normal(scale=1e-4, size=(3, 3))
        fixed = project_to_se3(T)
        assert is_valid_se3(fixed)
        np.testing.assert_allclose(fixed[:3, 3], T[:3, 3])


class TestAngles:
    @pytest.mark.parametrize("deg", [0.0, 1.0, 30.0, 90.0, 179.0])
    def test_rotation_angle_recovers_input(self, deg):
        R = Rotation.from_euler("y", deg, degrees=True).as_matrix()
        assert rotation_angle_deg(R) == pytest.approx(deg, abs=1e-6)

    def test_rotation_angle_accepts_4x4(self):
        T = se3_from_rt(Rotation.from_euler("x", 45, degrees=True).as_matrix(), np.zeros(3))
        assert rotation_angle_deg(T) == pytest.approx(45.0, abs=1e-6)

    def test_rotation_angle_is_symmetric(self, rng):
        R = random_se3(rng)[:3, :3]
        assert rotation_angle_rad(R) == pytest.approx(rotation_angle_rad(R.T), abs=1e-10)

    def test_angle_is_bounded(self, rng):
        for _ in range(50):
            assert 0.0 <= rotation_angle_rad(random_se3(rng)[:3, :3]) <= np.pi + 1e-12


class TestLieMaps:
    def test_se3_exp_log_round_trip(self, rng):
        for _ in range(30):
            T = small_se3(rng, angle_deg=40.0, trans=3.0)
            np.testing.assert_allclose(se3_exp(se3_log(T)), T, atol=1e-9)

    def test_se3_log_exp_round_trip(self, rng):
        for _ in range(30):
            xi = rng.normal(size=6) * 0.4
            np.testing.assert_allclose(se3_log(se3_exp(xi)), xi, atol=1e-9)

    def test_log_of_identity_is_zero(self):
        np.testing.assert_allclose(se3_log(np.eye(4)), np.zeros(6), atol=1e-12)

    def test_tiny_rotation_uses_stable_branch(self):
        """The small-angle branch must not blow up as theta -> 0."""
        xi = np.array([0.1, -0.2, 0.3, 1e-14, 0.0, 0.0])
        T = se3_exp(xi)
        assert is_valid_se3(T)
        np.testing.assert_allclose(T[:3, 3], xi[:3], atol=1e-10)

    def test_pure_translation(self):
        T = se3_exp(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], [1, 2, 3], atol=1e-12)


class TestKittiSerialisation:
    def test_row_round_trip(self, rng):
        T = random_se3(rng)
        np.testing.assert_allclose(kitti_row_to_se3(se3_to_kitti_row(T)), T, atol=1e-15)

    def test_row_is_row_major(self):
        T = np.arange(16.0).reshape(4, 4)
        T[3] = [0, 0, 0, 1]
        np.testing.assert_allclose(se3_to_kitti_row(T), np.arange(12.0))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            kitti_row_to_se3(np.zeros(11))
