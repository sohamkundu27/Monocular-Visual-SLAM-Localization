"""Tests for feature detection and descriptor matching."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from kitti_fixture import make_textured_image

from monocular_slam.config import Config
from monocular_slam.features.detector import FeatureDetector, Frame, draw_keypoints
from monocular_slam.features.matcher import FeatureMatcher, MatchResult, draw_matches


@pytest.fixture
def image():
    return make_textured_image(0, width=480, height=200)


@pytest.fixture
def shifted_image():
    """The same scene translated by a known amount, for match verification."""
    base = make_textured_image(0, width=480, height=200)
    shifted = np.zeros_like(base)
    shifted[:, :-10] = base[:, 10:]
    return shifted


@pytest.fixture
def orb():
    return FeatureDetector("orb", max_features=500)


class TestDetectorConstruction:
    def test_unknown_detector_raises(self):
        with pytest.raises(ValueError, match="Unsupported detector"):
            FeatureDetector("surf")

    def test_non_positive_budget_raises(self):
        with pytest.raises(ValueError, match="max_features"):
            FeatureDetector("orb", max_features=0)

    def test_orb_uses_hamming_norm(self, orb):
        assert orb.descriptor_norm == cv2.NORM_HAMMING

    def test_sift_uses_l2_norm(self):
        assert FeatureDetector("sift", max_features=100).descriptor_norm == cv2.NORM_L2

    def test_from_config(self):
        config = Config().with_overrides({"features.max_features": 123})
        assert FeatureDetector.from_config(config).max_features == 123


class TestDetection:
    def test_detects_keypoints_with_descriptors(self, orb, image):
        frame = orb.detect(image, index=3, frame_id=17)
        assert len(frame) > 50
        assert frame.index == 3 and frame.frame_id == 17
        assert frame.descriptors.shape == (len(frame), 32)
        assert frame.descriptors.dtype == np.uint8
        assert frame.is_usable

    def test_points_match_keypoints(self, orb, image):
        frame = orb.detect(image)
        assert frame.points.shape == (len(frame), 2)
        np.testing.assert_allclose(frame.points[0], frame.keypoints[0].pt)

    def test_respects_feature_budget(self, image):
        frame = FeatureDetector("orb", max_features=40).detect(image)
        assert len(frame) <= 40

    def test_sift_produces_float_descriptors(self, image):
        frame = FeatureDetector("sift", max_features=200).detect(image)
        assert frame.descriptors.dtype == np.float32
        assert frame.descriptors.shape[1] == 128

    def test_blank_image_returns_empty_frame(self, orb):
        frame = orb.detect(np.zeros((120, 160), dtype=np.uint8), index=5, frame_id=5)
        assert len(frame) == 0
        assert frame.descriptors is None
        assert not frame.is_usable
        assert frame.points.shape == (0, 2)

    def test_colour_input_is_converted(self, orb, image):
        colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        assert len(orb.detect(colour)) == len(orb.detect(image))

    def test_clahe_variant_still_detects(self, image):
        frame = FeatureDetector("orb", max_features=300, clahe=True).detect(image)
        assert len(frame) > 20

    def test_detection_is_deterministic(self, orb, image):
        a, b = orb.detect(image), orb.detect(image)
        np.testing.assert_allclose(a.points, b.points)
        np.testing.assert_array_equal(a.descriptors, b.descriptors)

    def test_mask_restricts_detection_region(self, orb, image):
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[:, :100] = 255
        frame = orb.detect(image, mask=mask)
        assert len(frame) > 0
        assert frame.points[:, 0].max() < 120  # allow for the ORB patch border

    def test_frame_rejects_mismatched_descriptors(self):
        kp = (cv2.KeyPoint(1.0, 2.0, 5.0),)
        with pytest.raises(ValueError, match="descriptor count"):
            Frame(index=0, frame_id=0, keypoints=kp, descriptors=np.zeros((3, 32), np.uint8))


class TestKeypointVisualisation:
    def test_returns_colour_canvas(self, orb, image):
        vis = draw_keypoints(image, orb.detect(image))
        assert vis.shape == (image.shape[0], image.shape[1], 3)

    def test_does_not_mutate_input(self, orb, image):
        original = image.copy()
        draw_keypoints(image, orb.detect(image))
        np.testing.assert_array_equal(image, original)


class TestMatching:
    def test_matches_a_shifted_scene(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        result = FeatureMatcher(norm=orb.descriptor_norm).match(f0, f1)
        assert len(result) > 20
        # The scene moved 10 px to the left, so matched points should too.
        dx = (result.points_b - result.points_a)[:, 0]
        assert np.median(dx) == pytest.approx(-10.0, abs=1.5)

    def test_identical_frames_match_at_zero_distance(self, orb, image):
        frame = orb.detect(image)
        result = FeatureMatcher(norm=orb.descriptor_norm, max_distance=None).match(frame, frame)
        assert len(result) > 20
        assert result.mean_distance == pytest.approx(0.0)
        np.testing.assert_array_equal(result.indices_a, result.indices_b)

    def test_indices_address_the_source_frames(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        result = FeatureMatcher(norm=orb.descriptor_norm).match(f0, f1)
        np.testing.assert_allclose(result.points_a, f0.points[result.indices_a])
        np.testing.assert_allclose(result.points_b, f1.points[result.indices_b])

    def test_ratio_test_tightens_the_match_set(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        loose = FeatureMatcher(norm=orb.descriptor_norm, use_ratio_test=False, cross_check=False)
        strict = FeatureMatcher(
            norm=orb.descriptor_norm, use_ratio_test=True, ratio=0.6, cross_check=False
        )
        assert len(strict.match(f0, f1)) < len(loose.match(f0, f1))

    def test_cross_check_tightens_the_match_set(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        without = FeatureMatcher(norm=orb.descriptor_norm, use_ratio_test=False, cross_check=False)
        with_cc = FeatureMatcher(norm=orb.descriptor_norm, use_ratio_test=False, cross_check=True)
        assert len(with_cc.match(f0, f1)) < len(without.match(f0, f1))

    def test_cross_check_keeps_only_mutual_neighbours(self, orb, image, shifted_image):
        """Every surviving match must also be the reverse nearest neighbour."""
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        result = FeatureMatcher(
            norm=orb.descriptor_norm, use_ratio_test=False, cross_check=True, max_distance=None
        ).match(f0, f1)
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
        reverse = {m.queryIdx: m.trainIdx for m in bf.match(f1.descriptors, f0.descriptors)}
        for ia, ib in zip(result.indices_a, result.indices_b):
            assert reverse[int(ib)] == int(ia)

    def test_distance_ceiling_is_enforced(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        result = FeatureMatcher(norm=orb.descriptor_norm, max_distance=30.0).match(f0, f1)
        assert np.all(result.distances <= 30.0)

    def test_empty_frame_yields_no_matches(self, orb, image):
        blank = orb.detect(np.zeros((100, 100), dtype=np.uint8))
        result = FeatureMatcher(norm=orb.descriptor_norm).match(orb.detect(image), blank)
        assert len(result) == 0
        assert result.n_raw == 0

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError, match="ratio must be"):
            FeatureMatcher(ratio=0.0)

    def test_unsupported_backend_raises(self):
        config = Config().with_overrides({"matcher.matcher": "flann"})
        with pytest.raises(ValueError, match="Unsupported matcher"):
            FeatureMatcher.from_config(config)

    def test_min_match_threshold(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        matcher = FeatureMatcher(norm=orb.descriptor_norm, min_matches=10)
        assert matcher.is_sufficient(matcher.match(f0, f1))
        assert not matcher.is_sufficient(MatchResult.empty())

    def test_float_descriptors_are_coerced_for_l2(self):
        """SIFT descriptors arrive as float32; the matcher must not choke."""
        rng = np.random.default_rng(0)
        desc = rng.random((30, 128)).astype(np.float64)
        pts = rng.random((30, 2))
        matcher = FeatureMatcher(norm=cv2.NORM_L2, use_ratio_test=False, max_distance=None)
        result = matcher.match_descriptors(desc, desc, pts, pts)
        assert len(result) == 30


class TestMatchResult:
    @pytest.fixture
    def result(self):
        return MatchResult(
            points_a=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            points_b=np.array([[3.0, 0.0], [4.0, 1.0], [5.0, 2.0]]),
            indices_a=np.array([0, 1, 2]),
            indices_b=np.array([5, 6, 7]),
            distances=np.array([10.0, 20.0, 30.0]),
            n_raw=6,
        )

    def test_length_and_survival(self, result):
        assert len(result) == 3
        assert result.survival_ratio == pytest.approx(0.5)

    def test_mean_statistics(self, result):
        assert result.mean_distance == pytest.approx(20.0)
        assert result.mean_pixel_displacement == pytest.approx(3.0)

    def test_select_boolean_mask(self, result):
        subset = result.select(np.array([True, False, True]))
        assert len(subset) == 2
        np.testing.assert_array_equal(subset.indices_b, [5, 7])
        assert subset.n_raw == 6

    def test_select_index_array(self, result):
        np.testing.assert_array_equal(result.select(np.array([1]))
                                      .indices_a, [1])

    def test_select_rejects_wrong_length_mask(self, result):
        with pytest.raises(ValueError, match="mask length"):
            result.select(np.array([True, False]))

    def test_empty_result_statistics(self):
        empty = MatchResult.empty(n_raw=5)
        assert len(empty) == 0
        assert empty.survival_ratio == 0.0
        assert np.isnan(empty.mean_distance)
        assert np.isnan(empty.mean_pixel_displacement)


class TestMatchVisualisation:
    def test_canvas_is_side_by_side(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        result = FeatureMatcher(norm=orb.descriptor_norm).match(f0, f1)
        canvas = draw_matches(image, f0, shifted_image, f1, result)
        assert canvas.shape == (image.shape[0], image.shape[1] * 2, 3)

    def test_inlier_mask_is_accepted(self, orb, image, shifted_image):
        f0, f1 = orb.detect(image, 0, 0), orb.detect(shifted_image, 1, 1)
        result = FeatureMatcher(norm=orb.descriptor_norm).match(f0, f1)
        mask = np.zeros(len(result), dtype=bool)
        mask[::2] = True
        canvas = draw_matches(image, f0, shifted_image, f1, result, inlier_mask=mask)
        assert canvas.ndim == 3

    def test_no_matches_still_renders(self, orb, image):
        frame = orb.detect(image)
        canvas = draw_matches(image, frame, image, frame, MatchResult.empty())
        assert canvas.shape[2] == 3
