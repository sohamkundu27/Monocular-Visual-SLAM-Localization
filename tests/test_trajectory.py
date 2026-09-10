"""Tests for the Trajectory container."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_se3, small_se3
from scipy.spatial.transform import Rotation

from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import (
    relative_pose,
    rotation_angle_deg,
    se3_from_rt,
)


def straight_line_trajectory(n: int = 11, step: float = 2.0) -> Trajectory:
    """Camera driving forward along +z, the KITTI forward axis."""
    poses = np.repeat(np.eye(4)[None, ...], n, axis=0)
    poses[:, 2, 3] = np.arange(n) * step
    return Trajectory(poses)


class TestConstruction:
    def test_defaults_frame_ids(self):
        traj = straight_line_trajectory(5)
        np.testing.assert_array_equal(traj.frame_ids, np.arange(5))

    def test_single_pose_promoted(self):
        traj = Trajectory(np.eye(4))
        assert len(traj) == 1

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            Trajectory(np.zeros((5, 3, 3)))

    def test_frame_id_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Trajectory(np.repeat(np.eye(4)[None], 3, axis=0), frame_ids=[0, 1])

    def test_timestamp_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Trajectory(np.repeat(np.eye(4)[None], 3, axis=0), timestamps=[0.0, 0.1])


class TestDerivedQuantities:
    def test_positions_and_xz(self):
        traj = straight_line_trajectory(4, step=1.5)
        np.testing.assert_allclose(traj.positions[:, 2], [0, 1.5, 3.0, 4.5])
        np.testing.assert_allclose(traj.xz[:, 0], 0.0)
        np.testing.assert_allclose(traj.xz[:, 1], [0, 1.5, 3.0, 4.5])

    def test_path_length_of_straight_line(self):
        assert straight_line_trajectory(11, step=2.0).path_length() == pytest.approx(20.0)

    def test_cumulative_distance_is_monotonic(self, rng):
        poses = [np.eye(4)]
        for _ in range(20):
            poses.append(poses[-1] @ small_se3(rng))
        traj = Trajectory(np.stack(poses))
        cumulative = traj.cumulative_distance()
        assert len(cumulative) == len(traj)
        assert cumulative[0] == 0.0
        assert np.all(np.diff(cumulative) >= -1e-12)
        assert cumulative[-1] == pytest.approx(traj.path_length())

    def test_empty_trajectory_step_lengths(self):
        traj = Trajectory(np.zeros((0, 4, 4)))
        assert len(traj) == 0
        assert traj.step_lengths().size == 0
        assert traj.path_length() == 0.0

    def test_relative_poses_reconstruct(self, rng):
        poses = [np.eye(4)]
        for _ in range(15):
            poses.append(poses[-1] @ small_se3(rng))
        traj = Trajectory(np.stack(poses))
        rel = traj.relative_poses(delta=1)
        assert rel.shape == (15, 4, 4)
        accumulated = np.eye(4)
        for r in rel:
            accumulated = accumulated @ r
        np.testing.assert_allclose(accumulated, poses[-1], atol=1e-9)

    def test_relative_poses_with_larger_delta(self, rng):
        traj = Trajectory(np.stack([random_se3(rng) for _ in range(10)]))
        rel = traj.relative_poses(delta=3)
        assert rel.shape == (7, 4, 4)
        np.testing.assert_allclose(rel[0], relative_pose(traj[0], traj[3]), atol=1e-12)

    def test_delta_larger_than_trajectory_is_empty(self):
        assert straight_line_trajectory(3).relative_poses(delta=5).shape == (0, 4, 4)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            straight_line_trajectory(3).relative_poses(delta=0)


class TestTransformations:
    def test_transformed_scales_positions(self):
        traj = straight_line_trajectory(5, step=2.0)
        scaled = traj.transformed(np.eye(4), scale=3.0)
        np.testing.assert_allclose(scaled.positions, traj.positions * 3.0)
        assert scaled.path_length() == pytest.approx(traj.path_length() * 3.0)

    def test_transformed_rotates_positions_and_orientations(self):
        R = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        T = se3_from_rt(R, np.array([1.0, 0.0, 0.0]))
        traj = straight_line_trajectory(3, step=1.0)
        moved = traj.transformed(T)
        # (0,0,1) rotated 90 deg about +y becomes (1,0,0), then translated by (1,0,0).
        np.testing.assert_allclose(moved.positions[1], [2.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(moved.rotations[0], R, atol=1e-12)

    def test_transform_preserves_shape_and_validity(self, rng):
        traj = Trajectory(np.stack([random_se3(rng) for _ in range(8)]))
        moved = traj.transformed(random_se3(rng), scale=2.5)
        assert moved.is_valid()
        # A similarity transform scales all distances uniformly.
        np.testing.assert_allclose(moved.step_lengths(), traj.step_lengths() * 2.5, atol=1e-9)

    def test_relative_to_first_anchors_at_identity(self, rng):
        traj = Trajectory(np.stack([random_se3(rng) for _ in range(6)]))
        anchored = traj.relative_to_first()
        np.testing.assert_allclose(anchored[0], np.eye(4), atol=1e-10)
        # Relative geometry between poses must be untouched.
        np.testing.assert_allclose(
            relative_pose(anchored[1], anchored[4]),
            relative_pose(traj[1], traj[4]),
            atol=1e-9,
        )

    def test_orthonormalized_repairs_drifted_rotations(self, rng):
        traj = Trajectory(np.stack([random_se3(rng) for _ in range(5)]))
        original = traj.rotations.copy()
        traj.poses[:, :3, :3] += rng.normal(scale=1e-3, size=(5, 3, 3))
        assert not traj.is_valid()
        fixed = traj.orthonormalized()
        assert fixed.is_valid()
        # Repair must snap back to (essentially) the rotation we started from.
        for a, b in zip(original, fixed.rotations):
            assert rotation_angle_deg(a.T @ b) < 0.5

    def test_copy_is_deep(self):
        traj = straight_line_trajectory(3)
        clone = traj.copy()
        clone.poses[0, 0, 3] = 99.0
        assert traj.poses[0, 0, 3] == 0.0


class TestSelection:
    def test_slicing_returns_sub_trajectory(self):
        traj = straight_line_trajectory(10)
        sub = traj[2:5]
        assert isinstance(sub, Trajectory)
        assert len(sub) == 3
        np.testing.assert_array_equal(sub.frame_ids, [2, 3, 4])

    def test_integer_index_returns_pose(self):
        assert straight_line_trajectory(4)[1].shape == (4, 4)

    def test_select_frames_matches_ids(self):
        traj = Trajectory(np.repeat(np.eye(4)[None], 6, axis=0), frame_ids=[0, 3, 6, 9, 12, 15])
        picked = traj.select_frames([3, 12])
        np.testing.assert_array_equal(picked.frame_ids, [3, 12])

    def test_select_missing_frame_raises(self):
        traj = Trajectory(np.repeat(np.eye(4)[None], 3, axis=0), frame_ids=[0, 2, 4])
        with pytest.raises(KeyError):
            traj.select_frames([1])

    def test_timestamps_follow_selection(self):
        traj = Trajectory(
            np.repeat(np.eye(4)[None], 4, axis=0),
            frame_ids=[0, 1, 2, 3],
            timestamps=[0.0, 0.1, 0.2, 0.3],
        )
        sub = traj[1:3]
        np.testing.assert_allclose(sub.timestamps, [0.1, 0.2])


class TestSerialisation:
    def test_kitti_row_round_trip(self, rng):
        traj = Trajectory(np.stack([random_se3(rng) for _ in range(7)]))
        rows = traj.to_kitti_rows()
        assert rows.shape == (7, 12)
        np.testing.assert_allclose(Trajectory.from_kitti_rows(rows).poses, traj.poses, atol=1e-15)

    def test_file_round_trip(self, tmp_path, rng):
        traj = Trajectory(np.stack([random_se3(rng) for _ in range(5)]))
        path = traj.save_kitti(tmp_path / "poses.txt")
        loaded = Trajectory.load_kitti(path)
        np.testing.assert_allclose(loaded.poses, traj.poses, atol=1e-10)

    def test_bad_row_shape_raises(self):
        with pytest.raises(ValueError):
            Trajectory.from_kitti_rows(np.zeros((3, 11)))

    def test_empty_to_rows(self):
        assert Trajectory(np.zeros((0, 4, 4))).to_kitti_rows().shape == (0, 12)

    def test_repr_mentions_length(self):
        assert "n=5" in repr(straight_line_trajectory(5))
        assert "empty" in repr(Trajectory(np.zeros((0, 4, 4))))
