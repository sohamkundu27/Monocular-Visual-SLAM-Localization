"""Tests for pose-graph construction and GTSAM optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import random_se3
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import (
    invert_se3,
    is_valid_se3,
    rotation_angle_deg,
    se3_from_rt,
)
from monocular_slam.optimization.gtsam_backend import (
    gtsam_available,
    numpy_to_pose3,
    optimize_pose_graph,
    pose3_to_numpy,
)
from monocular_slam.optimization.pose_graph import (
    OdometryEdge,
    PoseGraph,
    build_pose_graph,
)

requires_gtsam = pytest.mark.skipif(not gtsam_available(), reason="GTSAM is not installed")


@dataclass
class FakeLoopClosure:
    """Stand-in for a verified LoopClosure, matching the fields build_pose_graph reads."""

    match_id: int
    query_id: int
    T_match_query: np.ndarray
    n_inliers: int = 200
    similarity: float = 0.9


def circular_trajectory(n: int = 120, step: float = 1.0) -> Trajectory:
    """A closed circuit that returns exactly to its starting pose."""
    poses = [np.eye(4)]
    yaw = 360.0 / n
    for _ in range(n - 1):
        R = Rotation.from_euler("y", yaw, degrees=True).as_matrix()
        poses.append(poses[-1] @ se3_from_rt(R, np.array([0.0, 0.0, step])))
    return Trajectory(np.stack(poses))


def add_yaw_drift(reference: Trajectory, bias_deg: float) -> Trajectory:
    """Re-integrate a trajectory with a constant per-step yaw bias."""
    drifted = [np.eye(4)]
    R_bias = Rotation.from_euler("y", bias_deg, degrees=True).as_matrix()
    for i in range(len(reference) - 1):
        relative = invert_se3(reference[i]) @ reference[i + 1]
        step = se3_from_rt(R_bias @ relative[:3, :3], relative[:3, 3])
        drifted.append(drifted[-1] @ step)
    return Trajectory(np.stack(drifted))


class TestPoseGraphConstruction:
    def test_nodes_get_sequential_ids(self, rng):
        graph = PoseGraph()
        for _ in range(5):
            graph.add_node(random_se3(rng))
        assert [n.node_id for n in graph.nodes] == [0, 1, 2, 3, 4]
        assert len(graph) == 5

    def test_node_poses_are_orthonormalised(self, rng):
        graph = PoseGraph()
        pose = random_se3(rng)
        pose[:3, :3] += rng.normal(scale=1e-3, size=(3, 3))
        assert is_valid_se3(graph.add_node(pose).pose, tol=1e-6)

    def test_bad_pose_shape_raises(self):
        with pytest.raises(ValueError, match="4x4"):
            PoseGraph().add_node(np.eye(3))

    def test_edge_ids_are_validated(self):
        graph = PoseGraph()
        graph.add_node(np.eye(4))
        with pytest.raises(IndexError):
            graph.add_odometry_edge(0, 5, np.eye(4))

    def test_build_from_trajectory_creates_a_chain(self):
        reference = circular_trajectory(30)
        graph = build_pose_graph(reference, [])
        assert len(graph.nodes) == 30
        assert len(graph.odometry_edges) == 29
        assert len(graph.loop_edges) == 0
        assert graph.is_connected()

    def test_odometry_edges_reproduce_the_trajectory(self):
        """Chaining the edges must reconstruct the input poses exactly."""
        reference = circular_trajectory(20)
        graph = build_pose_graph(reference, [])
        accumulated = np.eye(4)
        for edge in graph.odometry_edges:
            accumulated = accumulated @ edge.T
        np.testing.assert_allclose(accumulated, reference[-1], atol=1e-9)

    def test_loop_edges_are_added(self):
        reference = circular_trajectory(30)
        loop = FakeLoopClosure(0, 29, invert_se3(reference[0]) @ reference[29])
        graph = build_pose_graph(reference, [loop])
        assert len(graph.loop_edges) == 1
        assert graph.loop_edges[0].n_inliers == 200

    def test_loop_edges_use_looser_translation_noise(self):
        reference = circular_trajectory(30)
        loop = FakeLoopClosure(0, 29, np.eye(4))
        graph = build_pose_graph(reference, [loop])
        assert graph.loop_edges[0].sigma_trans > graph.odometry_edges[0].sigma_trans

    def test_frame_ids_are_carried_through(self):
        reference = Trajectory(circular_trajectory(10).poses, frame_ids=np.arange(0, 50, 5))
        graph = build_pose_graph(reference, [])
        assert [n.frame_id for n in graph.nodes] == list(range(0, 50, 5))


class TestGraphValidation:
    def test_a_healthy_graph_reports_no_problems(self):
        assert build_pose_graph(circular_trajectory(20), []).validate() == []

    def test_empty_graph_is_flagged(self):
        assert "graph has no nodes" in PoseGraph().validate()

    def test_disconnected_graph_is_detected(self):
        graph = PoseGraph()
        for _ in range(4):
            graph.add_node(np.eye(4))
        graph.add_odometry_edge(0, 1, np.eye(4))
        graph.add_odometry_edge(2, 3, np.eye(4))  # separate component
        assert not graph.is_connected()
        assert any("disconnected" in p for p in graph.validate())

    def test_self_loop_is_flagged(self):
        graph = PoseGraph()
        graph.add_node(np.eye(4))
        graph.odometry_edges.append(OdometryEdge(0, 0, np.eye(4)))
        assert any("self-loop" in p for p in graph.validate())

    def test_non_positive_sigma_is_flagged(self):
        graph = build_pose_graph(circular_trajectory(5), [])
        graph.odometry_edges[0].sigma_trans = 0.0
        assert any("non-positive sigma" in p for p in graph.validate())

    def test_bad_prior_node_is_flagged(self):
        graph = build_pose_graph(circular_trajectory(5), [])
        graph.prior_node_id = 99
        assert any("prior_node_id" in p for p in graph.validate())

    def test_single_node_graph_is_connected(self):
        graph = PoseGraph()
        graph.add_node(np.eye(4))
        assert graph.is_connected()


class TestEdgeResiduals:
    def test_consistent_graph_has_zero_residuals(self):
        graph = build_pose_graph(circular_trajectory(30), [])
        residuals = graph.edge_residuals()
        assert residuals["odometry_translation_m"].max() < 1e-9
        assert residuals["odometry_rotation_deg"].max() < 1e-9

    def test_loop_residual_measures_accumulated_drift(self):
        """A drifted estimate must show its end-point gap on the loop edge."""
        reference = circular_trajectory(120)
        drifted = add_yaw_drift(reference, 0.2)
        loop = FakeLoopClosure(0, 119, invert_se3(reference[0]) @ reference[119])
        graph = build_pose_graph(drifted, [loop])
        residual = graph.edge_residuals()["loop_translation_m"][0]
        expected = float(np.linalg.norm(drifted[119][:3, 3] - reference[119][:3, 3]))
        assert residual == pytest.approx(expected, rel=0.05)

    def test_describe_summarises_the_graph(self):
        reference = circular_trajectory(30)
        loop = FakeLoopClosure(0, 29, invert_se3(reference[0]) @ reference[29])
        info = build_pose_graph(reference, [loop]).describe()
        assert info["n_nodes"] == 30
        assert info["n_loop_edges"] == 1
        assert info["connected"] is True


@requires_gtsam
class TestPoseConversion:
    def test_round_trip_is_exact(self, rng):
        for _ in range(20):
            T = random_se3(rng)
            np.testing.assert_allclose(pose3_to_numpy(numpy_to_pose3(T)), T, atol=1e-12)

    def test_identity_round_trips(self):
        np.testing.assert_allclose(pose3_to_numpy(numpy_to_pose3(np.eye(4))), np.eye(4), atol=1e-15)

    def test_drifted_rotation_is_repaired_not_rejected(self, rng):
        """GTSAM's Rot3 rejects non-orthonormal input; conversion must fix it."""
        T = random_se3(rng)
        T[:3, :3] += rng.normal(scale=1e-4, size=(3, 3))
        assert not is_valid_se3(T, tol=1e-6)
        converted = pose3_to_numpy(numpy_to_pose3(T))
        assert is_valid_se3(converted, tol=1e-9)
        assert rotation_angle_deg(converted[:3, :3].T @ T[:3, :3]) < 0.05

    def test_translation_survives_conversion(self):
        T = se3_from_rt(np.eye(3), np.array([1.5, -2.25, 3.75]))
        np.testing.assert_allclose(pose3_to_numpy(numpy_to_pose3(T))[:3, 3], [1.5, -2.25, 3.75])


@requires_gtsam
class TestOptimization:
    def test_chain_without_loops_is_already_optimal(self):
        """With no cycles, the odometry chain is the exact solution."""
        reference = circular_trajectory(40)
        result = optimize_pose_graph(build_pose_graph(reference, []))
        np.testing.assert_allclose(result.trajectory.positions, reference.positions, atol=1e-6)
        assert result.final_error < 1e-8

    def test_loop_closure_corrects_accumulated_drift(self):
        reference = circular_trajectory(150)
        drifted = add_yaw_drift(reference, 0.2)
        error_before = np.linalg.norm(drifted.positions - reference.positions, axis=1)
        assert error_before.max() > 5.0

        loop = FakeLoopClosure(0, 149, invert_se3(reference[0]) @ reference[149])
        result = optimize_pose_graph(build_pose_graph(drifted, [loop]))

        error_after = np.linalg.norm(result.trajectory.positions - reference.positions, axis=1)
        assert error_after.max() < error_before.max() / 3.0
        assert result.error_reduction_pct > 50.0

    def test_loop_residual_is_driven_down(self):
        reference = circular_trajectory(150)
        drifted = add_yaw_drift(reference, 0.2)
        loop = FakeLoopClosure(0, 149, invert_se3(reference[0]) @ reference[149])
        result = optimize_pose_graph(build_pose_graph(drifted, [loop]))
        before = result.residuals_before["loop_translation_m"][0]
        after = result.residuals_after["loop_translation_m"][0]
        assert after < before / 100.0

    def test_optimized_poses_remain_valid_se3(self):
        reference = circular_trajectory(80)
        drifted = add_yaw_drift(reference, 0.3)
        loop = FakeLoopClosure(0, 79, invert_se3(reference[0]) @ reference[79])
        assert optimize_pose_graph(build_pose_graph(drifted, [loop])).trajectory.is_valid()

    def test_prior_anchors_the_first_pose(self):
        reference = circular_trajectory(60)
        drifted = add_yaw_drift(reference, 0.3)
        loop = FakeLoopClosure(0, 59, invert_se3(reference[0]) @ reference[59])
        result = optimize_pose_graph(build_pose_graph(drifted, [loop]))
        np.testing.assert_allclose(result.trajectory[0], np.eye(4), atol=1e-3)

    def test_robust_kernel_limits_a_false_positive_loop(self):
        """A bogus loop must not be allowed to wreck the whole trajectory."""
        reference = circular_trajectory(120)
        bogus = FakeLoopClosure(
            10, 60, se3_from_rt(np.eye(3), np.array([500.0, 0.0, 0.0])), n_inliers=50
        )
        robust = optimize_pose_graph(build_pose_graph(reference, [bogus]), robust_loop_kernel=True)
        naive = optimize_pose_graph(build_pose_graph(reference, [bogus]), robust_loop_kernel=False)

        robust_damage = np.linalg.norm(
            robust.trajectory.positions - reference.positions, axis=1
        ).max()
        naive_damage = np.linalg.norm(
            naive.trajectory.positions - reference.positions, axis=1
        ).max()
        assert robust_damage < naive_damage

    def test_gauss_newton_backend_runs(self):
        reference = circular_trajectory(60)
        drifted = add_yaw_drift(reference, 0.15)
        loop = FakeLoopClosure(0, 59, invert_se3(reference[0]) @ reference[59])
        result = optimize_pose_graph(
            build_pose_graph(drifted, [loop]), optimizer="gauss_newton"
        )
        assert result.optimizer == "gauss_newton"
        assert result.final_error < result.initial_error

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            optimize_pose_graph(build_pose_graph(circular_trajectory(10), []), optimizer="adam")

    def test_invalid_graph_raises_before_gtsam_sees_it(self):
        graph = PoseGraph()
        for _ in range(4):
            graph.add_node(np.eye(4))
        graph.add_odometry_edge(0, 1, np.eye(4))
        with pytest.raises(ValueError, match="not optimizable"):
            optimize_pose_graph(graph)

    def test_result_serialises(self):
        reference = circular_trajectory(40)
        loop = FakeLoopClosure(0, 39, invert_se3(reference[0]) @ reference[39])
        data = optimize_pose_graph(build_pose_graph(reference, [loop])).to_dict()
        import json

        json.loads(json.dumps(data))
        assert data["n_loop_edges"] == 1

    def test_frame_ids_survive_optimization(self):
        reference = Trajectory(circular_trajectory(20).poses, frame_ids=np.arange(0, 100, 5))
        result = optimize_pose_graph(build_pose_graph(reference, []))
        np.testing.assert_array_equal(result.trajectory.frame_ids, np.arange(0, 100, 5))
