"""Smoke tests for the plotting helpers.

Rendered pixels are not asserted; what matters is that every figure writes a
non-trivial file, handles edge cases (empty loop lists, single-pose
trajectories) without raising, and always closes its figure so a long run does
not exhaust matplotlib's figure pool.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from synthetic import straight_drive

from monocular_slam.evaluation.plotting import (
    plot_diagnostics,
    plot_error_over_time,
    plot_loop_closures,
    plot_trajectory,
    plot_trajectory_3d,
    plot_trajectory_comparison,
)
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.odometry.visual_odometry import FrameDiagnostics


@pytest.fixture
def trajectory():
    return Trajectory(straight_drive(60))


@pytest.fixture
def noisy(trajectory):
    poses = trajectory.poses.copy()
    poses[:, :3, 3] += np.random.default_rng(0).normal(scale=0.4, size=(len(trajectory), 3))
    return Trajectory(poses)


@pytest.fixture
def diagnostics():
    rng = np.random.default_rng(1)
    return [
        FrameDiagnostics(
            index=i,
            frame_id=i,
            status="ok" if i % 17 else "failed",
            n_keypoints=int(rng.integers(1500, 3000)),
            n_matches=int(rng.integers(400, 1200)),
            n_ransac_inliers=int(rng.integers(300, 900)),
            ransac_inlier_ratio=float(rng.uniform(0.5, 0.95)),
            inlier_ratio=float(rng.uniform(0.5, 0.95)),
            rotation_deg=float(rng.uniform(0, 2)),
        )
        for i in range(120)
    ]


def assert_wrote_image(path):
    assert path.exists()
    assert path.stat().st_size > 1000


class TestTrajectoryPlots:
    def test_single_trajectory(self, tmp_path, trajectory):
        assert_wrote_image(plot_trajectory(trajectory, tmp_path / "t.png"))

    def test_with_reference(self, tmp_path, trajectory, noisy):
        assert_wrote_image(plot_trajectory(noisy, tmp_path / "t.png", reference=trajectory))

    def test_comparison_of_several(self, tmp_path, trajectory, noisy):
        path = plot_trajectory_comparison(
            {"Raw VO": noisy, "Optimized": trajectory}, tmp_path / "c.png", reference=trajectory
        )
        assert_wrote_image(path)

    def test_comparison_without_reference(self, tmp_path, trajectory):
        assert_wrote_image(plot_trajectory_comparison({"a": trajectory}, tmp_path / "c.png"))

    def test_3d_plot(self, tmp_path, trajectory, noisy):
        path = plot_trajectory_3d({"Raw VO": noisy}, tmp_path / "t3.png", reference=trajectory)
        assert_wrote_image(path)

    def test_creates_missing_directories(self, tmp_path, trajectory):
        assert_wrote_image(plot_trajectory(trajectory, tmp_path / "deep" / "nested" / "t.png"))

    def test_single_pose_trajectory_does_not_crash(self, tmp_path):
        assert_wrote_image(plot_trajectory(Trajectory(np.eye(4)), tmp_path / "one.png"))


class TestLoopClosurePlot:
    def test_draws_edges(self, tmp_path, trajectory):
        loops = [(5, 40), (10, 50)]
        assert_wrote_image(plot_loop_closures(trajectory, loops, tmp_path / "l.png"))

    def test_empty_loop_list_is_fine(self, tmp_path, trajectory):
        assert_wrote_image(plot_loop_closures(trajectory, [], tmp_path / "l.png"))

    def test_out_of_range_indices_are_skipped(self, tmp_path, trajectory):
        assert_wrote_image(plot_loop_closures(trajectory, [(0, 9999)], tmp_path / "l.png"))


class TestDiagnosticPlots:
    def test_error_over_time(self, tmp_path):
        errors = {"Raw": np.linspace(0, 5, 100), "Optimized": np.linspace(0, 2, 100)}
        assert_wrote_image(plot_error_over_time(errors, tmp_path / "e.png"))

    def test_error_against_distance(self, tmp_path):
        errors = {"Raw": np.linspace(0, 5, 100)}
        distances = np.linspace(0, 90, 100)
        assert_wrote_image(plot_error_over_time(errors, tmp_path / "e.png", distances=distances))

    def test_front_end_diagnostics(self, tmp_path, diagnostics):
        assert_wrote_image(plot_diagnostics(diagnostics, tmp_path / "d.png"))

    def test_empty_diagnostics_raise(self, tmp_path):
        with pytest.raises(ValueError, match="No diagnostics"):
            plot_diagnostics([], tmp_path / "d.png")


class TestResourceHygiene:
    def test_figures_are_closed(self, tmp_path, trajectory, diagnostics):
        """A 4000-frame run must not leak a figure per plot call."""
        plt.close("all")
        before = len(plt.get_fignums())
        plot_trajectory(trajectory, tmp_path / "a.png")
        plot_trajectory_comparison({"x": trajectory}, tmp_path / "b.png")
        plot_trajectory_3d({"x": trajectory}, tmp_path / "c.png")
        plot_loop_closures(trajectory, [(1, 2)], tmp_path / "d.png")
        plot_error_over_time({"x": np.arange(10.0)}, tmp_path / "e.png")
        plot_diagnostics(diagnostics, tmp_path / "f.png")
        assert len(plt.get_fignums()) == before
