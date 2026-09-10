"""Tests for essential-matrix estimation and relative pose recovery.

Correspondences come from a synthetic scene with a known camera motion, which
makes it possible to assert the *exact* sign convention rather than only that
"something plausible" came back.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import (
    KITTI_K,
    forward_motion,
    make_scene,
    pure_rotation,
    two_view_correspondences,
)

from monocular_slam.geometry.epipolar import (
    EssentialMatrixResult,
    estimate_essential_matrix,
    median_flow_px,
    median_parallax_deg,
    recover_relative_pose,
    select_two_view_model,
    triangulate_points,
)
from monocular_slam.geometry.transforms import (
    invert_se3,
    is_valid_se3,
    rotation_angle_deg,
)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def sampson_distance(E: np.ndarray, pa: np.ndarray, pb: np.ndarray, K: np.ndarray) -> np.ndarray:
    """First-order geometric epipolar error in pixels, per correspondence."""
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    ha = np.column_stack([pa, np.ones(len(pa))])
    hb = np.column_stack([pb, np.ones(len(pb))])
    numerator = np.sum(hb * (ha @ F.T), axis=1) ** 2
    lines_b = ha @ F.T
    lines_a = hb @ F
    denominator = lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2 + lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2
    return np.sqrt(numerator / denominator)


def estimate(points_a, points_b, **kwargs):
    """Run the full estimate-then-recover pipeline."""
    em = estimate_essential_matrix(points_a, points_b, KITTI_K)
    return em, recover_relative_pose(em.E, points_a, points_b, KITTI_K, em.inlier_mask, **kwargs)


class TestEssentialMatrixEstimation:
    def test_clean_correspondences_yield_high_inlier_ratio(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.5))
        result = estimate_essential_matrix(pa, pb, KITTI_K)
        assert result.ok
        assert result.E.shape == (3, 3)
        assert result.inlier_ratio > 0.95

    @pytest.mark.parametrize("method", ["magsac", "ransac", "usac_accurate"])
    def test_essential_matrix_satisfies_epipolar_constraint(self, method):
        """Every backend must fit E to well under a pixel of Sampson error.

        Sampson distance (the first-order geometric point-to-epipolar-curve
        distance, in pixels) is the meaningful residual here. The raw algebraic
        form x2^T F x1 is unnormalised and its magnitude depends on the image
        coordinates, so it is not comparable across backends.
        """
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        E = estimate_essential_matrix(pa, pb, KITTI_K, method=method).E
        assert np.median(sampson_distance(E, pa, pb, KITTI_K)) < 0.05

    def test_unknown_method_raises(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9))
        with pytest.raises(ValueError, match="Unknown ransac_method"):
            estimate_essential_matrix(pa, pb, KITTI_K, method="bogus")

    def test_essential_matrix_has_rank_two(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        singular = np.linalg.svd(estimate_essential_matrix(pa, pb, KITTI_K).E, compute_uv=False)
        assert singular[2] / singular[0] < 1e-6
        assert singular[1] / singular[0] == pytest.approx(1.0, abs=1e-6)

    def test_survives_pixel_noise(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0), noise_px=0.5)
        result = estimate_essential_matrix(pa, pb, KITTI_K, threshold_px=1.5)
        assert result.ok
        assert result.inlier_ratio > 0.7

    def test_outliers_are_rejected(self):
        """Half the correspondences are shuffled; RANSAC must isolate them."""
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        rng = np.random.default_rng(7)
        corrupt = pb.copy()
        n_bad = len(pb) // 2
        corrupt[:n_bad] = rng.uniform([0, 0], [1241, 376], size=(n_bad, 2))
        result = estimate_essential_matrix(pa, corrupt, KITTI_K, threshold_px=1.0)
        assert result.ok
        # Nearly all clean matches kept, nearly all corrupted ones dropped.
        assert result.inlier_mask[n_bad:].mean() > 0.9
        assert result.inlier_mask[:n_bad].mean() < 0.1

    def test_too_few_points_is_reported_not_raised(self):
        pts = np.zeros((4, 2))
        result = estimate_essential_matrix(pts, pts, KITTI_K)
        assert not result.ok
        assert "too_few_points" in result.reason
        assert result.n_inliers == 0

    def test_mismatched_input_shapes_raise(self):
        with pytest.raises(ValueError, match="matching"):
            estimate_essential_matrix(np.zeros((10, 2)), np.zeros((9, 2)), KITTI_K)

    def test_result_ratio_arithmetic(self):
        result = EssentialMatrixResult(
            E=np.eye(3), inlier_mask=np.array([True, False, True, True]), n_points=4, ok=True
        )
        assert result.n_inliers == 3
        assert result.inlier_ratio == pytest.approx(0.75)


class TestRelativePoseRecovery:
    @pytest.mark.parametrize(
        "motion",
        [
            forward_motion(0.9),
            forward_motion(0.9, yaw_deg=2.0),
            forward_motion(0.9, yaw_deg=-3.0, lateral=0.1),
            forward_motion(2.5, yaw_deg=5.0),
        ],
    )
    def test_recovers_true_motion_direction_and_rotation(self, motion):
        """The recovered T must be T_c1_c2 with unit translation."""
        pa, pb, _ = two_view_correspondences(motion)
        _, pose = estimate(pa, pb)
        assert pose.ok, pose.reason
        assert is_valid_se3(pose.T)
        # Rotation matches exactly.
        assert rotation_angle_deg(pose.T[:3, :3].T @ motion[:3, :3]) < 0.1
        # Translation direction matches; magnitude is by definition unity.
        assert angle_between(pose.T[:3, 3], motion[:3, 3]) < 0.5
        assert np.linalg.norm(pose.T[:3, 3]) == pytest.approx(1.0)

    def test_translation_is_unit_norm_not_metric(self):
        """Monocular geometry cannot see scale: 0.9 m and 9 m look identical."""
        _, near = estimate(*two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))[:2])
        _, far = estimate(*two_view_correspondences(forward_motion(9.0, yaw_deg=1.0))[:2])
        assert np.linalg.norm(near.T[:3, 3]) == pytest.approx(1.0)
        assert np.linalg.norm(far.T[:3, 3]) == pytest.approx(1.0)

    def test_reversing_the_view_order_inverts_the_pose(self):
        """Swapping the views must return T_c2_c1 == inv(T_c1_c2)."""
        motion = forward_motion(0.9, yaw_deg=2.0)
        pa, pb, _ = two_view_correspondences(motion)
        _, forward = estimate(pa, pb)
        _, backward = estimate(pb, pa)
        expected = invert_se3(forward.T)
        assert rotation_angle_deg(backward.T[:3, :3].T @ expected[:3, :3]) < 0.5
        assert angle_between(backward.T[:3, 3], expected[:3, 3]) < 1.0

    def test_reported_rotation_angle_is_correct(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=4.0))
        _, pose = estimate(pa, pb)
        assert pose.rotation_deg == pytest.approx(4.0, abs=0.2)

    def test_noise_degrades_gracefully(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=2.0), noise_px=0.7)
        em = estimate_essential_matrix(pa, pb, KITTI_K, threshold_px=1.5)
        pose = recover_relative_pose(em.E, pa, pb, KITTI_K, em.inlier_mask)
        assert pose.ok, pose.reason
        assert angle_between(pose.T[:3, 3], np.array([0.0, 0.0, 1.0])) < 8.0


class TestValidationGuards:
    def test_missing_essential_matrix_is_rejected(self):
        pose = recover_relative_pose(None, np.zeros((10, 2)), np.zeros((10, 2)), KITTI_K)
        assert not pose.ok
        assert pose.reason == "no_essential_matrix"
        assert pose.T is None

    def test_min_inliers_guard_fires(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        _, pose = estimate(pa, pb, min_inliers=10_000)
        assert not pose.ok
        assert "too_few_inliers" in pose.reason

    def test_inlier_ratio_guard_fires(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        _, pose = estimate(pa, pb, min_inlier_ratio=1.01)
        assert not pose.ok
        assert "low_inlier_ratio" in pose.reason

    def test_rotation_guard_fires(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=10.0))
        _, pose = estimate(pa, pb, max_rotation_deg=2.0)
        assert not pose.ok
        assert "implausible_rotation" in pose.reason

    def test_parallax_guard_fires_when_configured(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9))
        _, pose = estimate(pa, pb, min_parallax_deg=90.0)
        assert not pose.ok
        assert "low_parallax" in pose.reason

    def test_translating_motion_has_measurable_parallax(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9))
        _, pose = estimate(pa, pb)
        assert pose.median_parallax_deg > 0.1

    def test_pure_rotation_is_rejected(self):
        """Pure rotation must never yield an accepted pose.

        The essential matrix is degenerate there, so the recovered rotation is
        wrong and rotation-compensated parallax becomes meaningless — which is
        precisely why the cheirality and model-selection guards exist rather
        than a parallax threshold alone.
        """
        pa, pb, _ = two_view_correspondences(pure_rotation(3.0))
        em, pose = estimate(pa, pb)
        assert not pose.ok
        assert select_two_view_model(pa, pb, KITTI_K, em.E).is_degenerate

    def test_rejected_pose_still_reports_diagnostics(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        _, pose = estimate(pa, pb, min_inliers=10_000)
        assert pose.T is not None  # geometry succeeded; the guard rejected it
        assert pose.n_cheirality_inliers > 0


class TestParallax:
    def test_identical_points_have_zero_parallax(self):
        pts = np.array([[100.0, 100.0], [400.0, 200.0]])
        assert median_parallax_deg(pts, pts, KITTI_K) == pytest.approx(0.0, abs=1e-5)

    def test_parallax_grows_with_displacement(self):
        pts = np.array([[600.0, 180.0], [620.0, 200.0]])
        small = median_parallax_deg(pts, pts + 2.0, KITTI_K)
        large = median_parallax_deg(pts, pts + 20.0, KITTI_K)
        assert 0 < small < large

    def test_empty_input_is_nan(self):
        assert np.isnan(median_parallax_deg(np.zeros((0, 2)), np.zeros((0, 2)), KITTI_K))

    def test_mask_restricts_the_median(self):
        pts = np.array([[600.0, 180.0], [620.0, 200.0], [640.0, 210.0]])
        moved = pts.copy()
        moved[0] += 50.0
        mask = np.array([False, True, True])
        assert median_parallax_deg(pts, moved, KITTI_K, mask=mask) == pytest.approx(0.0, abs=1e-5)

    def test_rotation_compensation_cancels_pure_rotation(self):
        """De-rotating must drive parallax to zero for a rotation-only motion."""
        motion = pure_rotation(4.0)
        pa, pb, _ = two_view_correspondences(motion)
        uncompensated = median_parallax_deg(pa, pb, KITTI_K)
        compensated = median_parallax_deg(pa, pb, KITTI_K, motion[:3, :3])
        assert uncompensated > 3.0
        assert compensated < 1e-4


class TestModelSelection:
    def measure(self, motion, **kwargs):
        pa, pb, _ = two_view_correspondences(motion, **kwargs)
        em = estimate_essential_matrix(pa, pb, KITTI_K)
        return select_two_view_model(pa, pb, KITTI_K, em.E)

    def test_general_motion_prefers_the_epipolar_model(self):
        selection = self.measure(forward_motion(0.9, yaw_deg=2.0))
        assert selection.score_e > selection.score_h
        assert selection.ratio_h < 0.45
        assert not selection.is_degenerate

    def test_pure_rotation_is_flagged_degenerate(self):
        assert self.measure(pure_rotation(3.0)).is_degenerate

    def test_small_pure_rotation_is_flagged_degenerate(self):
        assert self.measure(pure_rotation(0.5)).is_degenerate

    def test_stationary_camera_is_flagged_degenerate(self):
        assert self.measure(forward_motion(0.0)).is_degenerate

    def test_planar_scene_is_flagged_degenerate(self):
        """A road-plane-only scene leaves the essential matrix under-constrained."""
        points = make_scene(600, seed=2)
        points[:, 2] = 20.0 + 0.5 * points[:, 0]
        assert self.measure(forward_motion(0.9, yaw_deg=1.0), points_world=points).is_degenerate

    def test_noise_does_not_trigger_a_false_positive(self):
        assert not self.measure(forward_motion(0.9, yaw_deg=1.0), noise_px=0.5).is_degenerate

    def test_threshold_is_configurable(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=2.0))
        em = estimate_essential_matrix(pa, pb, KITTI_K)
        strict = select_two_view_model(pa, pb, KITTI_K, em.E, ratio_threshold=0.1)
        assert strict.is_degenerate

    def test_missing_inputs_are_handled(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9))
        assert select_two_view_model(pa, pb, KITTI_K, None).ratio_h == 0.0
        tiny = np.zeros((3, 2))
        assert not select_two_view_model(tiny, tiny, KITTI_K, np.eye(3)).is_degenerate


class TestMedianFlow:
    def test_zero_for_identical_points(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert median_flow_px(pts, pts) == 0.0

    def test_measures_known_displacement(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        assert median_flow_px(pts, pts + np.array([3.0, 4.0])) == pytest.approx(5.0)

    def test_stationary_camera_has_near_zero_flow(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.0))
        assert median_flow_px(pa, pb) < 1e-9

    def test_driving_camera_has_clear_flow(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9))
        assert median_flow_px(pa, pb) > 1.0

    def test_empty_is_nan(self):
        assert np.isnan(median_flow_px(np.zeros((0, 2)), np.zeros((0, 2))))


class TestTriangulation:
    def test_recovers_scene_structure_up_to_scale(self):
        motion = forward_motion(0.9, yaw_deg=1.5)
        pa, pb, points_cam1 = two_view_correspondences(motion)
        _, pose = estimate(pa, pb)
        triangulated = triangulate_points(pose.T, pa, pb, KITTI_K)
        # Unit-norm translation means depths come back scaled by 1 / |t_true|.
        scale = np.linalg.norm(motion[:3, 3])
        np.testing.assert_allclose(triangulated * scale, points_cam1, rtol=2e-3, atol=1e-2)

    def test_triangulated_points_lie_in_front_of_the_camera(self):
        pa, pb, _ = two_view_correspondences(forward_motion(0.9, yaw_deg=1.0))
        _, pose = estimate(pa, pb)
        triangulated = triangulate_points(pose.T, pa, pb, KITTI_K)
        finite = triangulated[np.all(np.isfinite(triangulated), axis=1)]
        assert (finite[:, 2] > 0).mean() > 0.99
