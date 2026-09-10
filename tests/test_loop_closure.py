"""Tests for keyframe selection, BoW retrieval and loop verification."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from kitti_fixture import make_textured_image
from scipy.spatial.transform import Rotation
from synthetic import KITTI_K, forward_motion, two_view_correspondences

from monocular_slam.config import Config
from monocular_slam.features.detector import FeatureDetector
from monocular_slam.features.matcher import FeatureMatcher
from monocular_slam.geometry.transforms import (
    invert_se3,
    rotation_angle_deg,
    se3_from_rt,
)
from monocular_slam.loop_closure.database import (
    Keyframe,
    KeyframeDatabase,
    KeyframeSelector,
    VisualVocabulary,
)
from monocular_slam.loop_closure.detector import (
    LoopCandidate,
    LoopClosureDetector,
)


def make_keyframe(kf_id: int, descriptors, points=None, pose=None, distance=0.0) -> Keyframe:
    n = len(descriptors)
    if points is None:
        points = np.random.default_rng(kf_id).uniform(0, 500, size=(n, 2))
    return Keyframe(
        keyframe_id=kf_id,
        frame_index=kf_id * 5,
        frame_id=kf_id * 5,
        points=np.asarray(points, dtype=np.float64),
        descriptors=np.asarray(descriptors),
        pose=np.eye(4) if pose is None else pose,
        path_distance_m=distance,
    )


def clustered_descriptors(n_clusters: int, per_cluster: int, seed: int = 0) -> np.ndarray:
    """Binary descriptors drawn from ``n_clusters`` well-separated prototypes."""
    rng = np.random.default_rng(seed)
    prototypes = rng.integers(0, 256, size=(n_clusters, 32), dtype=np.uint8)
    out = []
    for prototype in prototypes:
        block = np.repeat(prototype[None, :], per_cluster, axis=0)
        # Flip a handful of bits so members differ slightly but stay clustered.
        flips = rng.integers(0, 2, size=block.shape, dtype=np.uint8) & rng.integers(
            0, 2, size=block.shape, dtype=np.uint8
        ) & 0x03
        out.append(block ^ flips)
    return np.concatenate(out)


class TestKeyframeSelector:
    def test_first_frame_is_always_selected(self):
        assert KeyframeSelector().should_select(0, np.eye(4))

    def test_frame_interval_triggers(self):
        selector = KeyframeSelector(every_n_frames=5, min_translation_m=1e9, min_rotation_deg=1e9)
        selector.accept(0, np.eye(4))
        assert not selector.should_select(4, np.eye(4))
        assert selector.should_select(5, np.eye(4))

    def test_translation_triggers_before_the_interval(self):
        selector = KeyframeSelector(every_n_frames=1000, min_translation_m=2.0, min_rotation_deg=1e9)
        selector.accept(0, np.eye(4))
        near = se3_from_rt(np.eye(3), np.array([0.0, 0.0, 1.5]))
        far = se3_from_rt(np.eye(3), np.array([0.0, 0.0, 2.5]))
        assert not selector.should_select(1, near)
        assert selector.should_select(2, far)

    def test_rotation_triggers_before_the_interval(self):
        selector = KeyframeSelector(every_n_frames=1000, min_translation_m=1e9, min_rotation_deg=10.0)
        selector.accept(0, np.eye(4))
        small = se3_from_rt(Rotation.from_euler("y", 5, degrees=True).as_matrix(), np.zeros(3))
        large = se3_from_rt(Rotation.from_euler("y", 15, degrees=True).as_matrix(), np.zeros(3))
        assert not selector.should_select(1, small)
        assert selector.should_select(2, large)

    def test_accept_resets_the_triggers(self):
        selector = KeyframeSelector(every_n_frames=5)
        selector.accept(0, np.eye(4))
        assert selector.should_select(5, np.eye(4))
        selector.accept(5, np.eye(4))
        assert not selector.should_select(6, np.eye(4))

    def test_reset_restores_initial_state(self):
        selector = KeyframeSelector(every_n_frames=100)
        selector.accept(0, np.eye(4))
        assert not selector.should_select(1, np.eye(4))
        selector.reset()
        assert selector.should_select(1, np.eye(4))

    def test_from_config(self):
        config = Config().with_overrides({"keyframes.every_n_frames": 9})
        assert KeyframeSelector.from_config(config).every_n_frames == 9


class TestVisualVocabulary:
    def test_binary_vocabulary_recovers_cluster_structure(self):
        descriptors = clustered_descriptors(n_clusters=8, per_cluster=50, seed=1)
        vocab = VisualVocabulary.train(descriptors, vocabulary_size=8, seed=0)
        assert vocab.binary
        assert vocab.size == 8
        words = vocab.quantize(descriptors)
        # Every member of a synthetic cluster should land on the same word.
        for cluster in range(8):
            block = words[cluster * 50 : (cluster + 1) * 50]
            assert len(np.unique(block)) == 1

    def test_word_assignment_is_deterministic(self):
        descriptors = clustered_descriptors(4, 30, seed=2)
        vocab = VisualVocabulary.train(descriptors, vocabulary_size=4, seed=0)
        np.testing.assert_array_equal(vocab.quantize(descriptors), vocab.quantize(descriptors))

    def test_histogram_totals_match_descriptor_count(self):
        descriptors = clustered_descriptors(4, 25, seed=3)
        vocab = VisualVocabulary.train(descriptors, vocabulary_size=6, seed=0)
        histogram = vocab.histogram(descriptors)
        assert histogram.shape == (6,)
        assert histogram.sum() == len(descriptors)

    def test_float_descriptors_use_euclidean_kmeans(self):
        rng = np.random.default_rng(0)
        descriptors = np.concatenate(
            [rng.normal(loc=c, scale=0.05, size=(40, 8)) for c in (-5.0, 0.0, 5.0)]
        ).astype(np.float32)
        vocab = VisualVocabulary.train(descriptors, vocabulary_size=3, seed=0)
        assert not vocab.binary
        assert len(np.unique(vocab.quantize(descriptors[:40]))) == 1

    def test_vocabulary_is_capped_by_descriptor_count(self):
        descriptors = clustered_descriptors(2, 5, seed=4)
        assert VisualVocabulary.train(descriptors, vocabulary_size=1000, seed=0).size == 10

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="zero descriptors"):
            VisualVocabulary.train(np.zeros((0, 32), dtype=np.uint8))

    def test_single_descriptor_raises(self):
        with pytest.raises(ValueError, match="at least 2 words"):
            VisualVocabulary.train(np.zeros((1, 32), dtype=np.uint8))

    def test_quantizing_nothing_returns_empty(self):
        vocab = VisualVocabulary.train(clustered_descriptors(3, 10, seed=5), 3, seed=0)
        assert vocab.quantize(np.zeros((0, 32), dtype=np.uint8)).shape == (0,)


class TestKeyframeDatabase:
    @pytest.fixture
    def database(self):
        """Six keyframes: 0/1 and 4/5 look alike, forming a synthetic 'loop'."""
        scene_a = clustered_descriptors(6, 40, seed=10)
        scene_b = clustered_descriptors(6, 40, seed=20)
        scene_c = clustered_descriptors(6, 40, seed=30)
        descriptor_sets = [scene_a, scene_a, scene_b, scene_c, scene_a, scene_a]

        vocab = VisualVocabulary.train(
            np.concatenate(descriptor_sets), vocabulary_size=24, seed=0
        )
        db = KeyframeDatabase(vocab)
        for i, descriptors in enumerate(descriptor_sets):
            db.add(make_keyframe(i, descriptors, distance=float(i) * 50.0))
        db.build_index()
        return db

    def test_ids_are_assigned_in_order(self, database):
        assert [kf.keyframe_id for kf in database] == [0, 1, 2, 3, 4, 5]

    def test_bow_vectors_are_unit_norm(self, database):
        for kf in database:
            assert np.linalg.norm(kf.bow) == pytest.approx(1.0, abs=1e-9)

    def test_self_similarity_is_one(self, database):
        for i in range(len(database)):
            assert database.similarity_scores(i)[i] == pytest.approx(1.0, abs=1e-9)

    def test_similar_scenes_score_higher_than_different_ones(self, database):
        scores = database.similarity_scores(0)
        assert scores[4] > scores[2]
        assert scores[4] > scores[3]

    def test_query_finds_the_revisited_keyframe(self, database):
        results = database.query(
            5, min_keyframe_separation=2, min_path_separation_m=50.0, top_k=3, min_similarity=0.0
        )
        assert results
        assert results[0][0] in (0, 1)

    def test_temporal_exclusion_blocks_adjacent_keyframes(self, database):
        results = database.query(
            5, min_keyframe_separation=3, min_path_separation_m=0.0, top_k=5, min_similarity=0.0
        )
        assert all(kf_id <= 2 for kf_id, _ in results)

    def test_path_exclusion_blocks_a_stationary_vehicle(self):
        """Many keyframes at the same place must not match each other."""
        descriptors = clustered_descriptors(6, 40, seed=7)
        vocab = VisualVocabulary.train(descriptors, vocabulary_size=16, seed=0)
        db = KeyframeDatabase(vocab)
        for i in range(60):
            db.add(make_keyframe(i, descriptors, distance=0.05 * i))  # barely moving
        db.build_index()
        assert db.query(59, min_keyframe_separation=30, min_path_separation_m=30.0) == []

    def test_min_similarity_filters_results(self, database):
        assert database.query(5, min_keyframe_separation=2, min_path_separation_m=0.0,
                              min_similarity=1.01) == []

    def test_top_k_limits_results(self, database):
        results = database.query(
            5, min_keyframe_separation=1, min_path_separation_m=0.0, top_k=2, min_similarity=0.0
        )
        assert len(results) <= 2

    def test_results_are_sorted_by_descending_score(self, database):
        results = database.query(
            5, min_keyframe_separation=1, min_path_separation_m=0.0, top_k=5, min_similarity=0.0
        )
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_neighbour_score_reflects_adjacent_similarity(self, database):
        assert 0.0 < database.neighbour_score(1, window=1) <= 1.0

    def test_empty_database_query_returns_nothing(self):
        assert KeyframeDatabase(None).query(0) == []

    def test_index_without_vocabulary_raises(self):
        db = KeyframeDatabase()
        db.add(make_keyframe(0, clustered_descriptors(2, 5, seed=1)))
        with pytest.raises(RuntimeError, match="vocabulary must be set"):
            db.build_index()

    def test_out_of_range_query_raises(self, database):
        with pytest.raises(IndexError):
            database.similarity_scores(99)

    def test_describe_reports_configuration(self, database):
        info = database.describe()
        assert info["n_keyframes"] == 6
        assert info["binary_vocabulary"] is True


class TestGeometricVerification:
    """The verification stage, exercised with real ORB features."""

    @pytest.fixture
    def detector_and_matcher(self):
        detector = FeatureDetector("orb", max_features=1000)
        matcher = FeatureMatcher(norm=cv2.NORM_HAMMING, min_matches=10)
        return detector, matcher

    @pytest.fixture
    def loop_detector(self, detector_and_matcher):
        _, matcher = detector_and_matcher
        return LoopClosureDetector(
            K=KITTI_K,
            matcher=matcher,
            min_matches=20,
            min_inliers=15,
            min_inlier_ratio=0.3,
        )

    def synthetic_pair(self, motion):
        """Two keyframes whose correspondences come from a known camera motion."""
        pa, pb, _ = two_view_correspondences(motion, noise_px=0.2)
        rng = np.random.default_rng(0)
        # Identical descriptors on both sides means matching is trivial; the
        # test isolates the geometric stage.
        descriptors = rng.integers(0, 256, size=(len(pa), 32), dtype=np.uint8)
        kf_a = make_keyframe(0, descriptors, points=pa, pose=np.eye(4), distance=0.0)
        kf_b = make_keyframe(1, descriptors, points=pb, pose=motion, distance=100.0)
        db = KeyframeDatabase()
        db.add(kf_a)
        db.add(kf_b)
        return db

    def test_verifies_a_true_pair_and_recovers_the_motion(self, loop_detector):
        motion = forward_motion(3.0, yaw_deg=4.0)
        db = self.synthetic_pair(motion)
        closure = loop_detector.verify(db, LoopCandidate(query_id=1, match_id=0, similarity=0.9))
        assert closure is not None
        assert closure.n_inliers >= 15
        assert rotation_angle_deg(closure.T_match_query[:3, :3].T @ motion[:3, :3]) < 2.0

    def test_translation_magnitude_comes_from_odometry(self, loop_detector):
        motion = forward_motion(3.0, yaw_deg=2.0)
        db = self.synthetic_pair(motion)
        closure = loop_detector.verify(db, LoopCandidate(query_id=1, match_id=0, similarity=0.9))
        odometry_gap = invert_se3(db[0].pose) @ db[1].pose
        expected = float(np.linalg.norm(odometry_gap[:3, 3]))
        assert closure.scale_m == pytest.approx(expected, rel=1e-9)
        assert np.linalg.norm(closure.T_match_query[:3, 3]) == pytest.approx(expected, rel=1e-6)

    def test_random_correspondences_are_rejected(self, loop_detector):
        """A false positive from perceptual aliasing must not survive geometry."""
        rng = np.random.default_rng(5)
        n = 300
        descriptors = rng.integers(0, 256, size=(n, 32), dtype=np.uint8)
        db = KeyframeDatabase()
        db.add(make_keyframe(0, descriptors, points=rng.uniform([0, 0], [1241, 376], (n, 2))))
        db.add(make_keyframe(1, descriptors, points=rng.uniform([0, 0], [1241, 376], (n, 2)),
                             distance=100.0))
        closure = loop_detector.verify(db, LoopCandidate(query_id=1, match_id=0, similarity=0.9))
        assert closure is None
        assert loop_detector.stats.rejected_geometry >= 1

    def test_too_few_matches_is_rejected(self, loop_detector):
        rng = np.random.default_rng(6)
        db = KeyframeDatabase()
        for i in range(2):
            descriptors = rng.integers(0, 256, size=(8, 32), dtype=np.uint8)
            db.add(make_keyframe(i, descriptors, distance=100.0 * i))
        assert loop_detector.verify(db, LoopCandidate(1, 0, 0.9)) is None
        assert loop_detector.stats.rejected_too_few_matches == 1

    def test_inlier_ratio_gate_is_enforced(self, detector_and_matcher):
        _, matcher = detector_and_matcher
        strict = LoopClosureDetector(
            K=KITTI_K, matcher=matcher, min_matches=20, min_inliers=15, min_inlier_ratio=1.01
        )
        db = self.synthetic_pair(forward_motion(3.0, yaw_deg=2.0))
        assert strict.verify(db, LoopCandidate(1, 0, 0.9)) is None

    def test_closure_serialises(self, loop_detector):
        db = self.synthetic_pair(forward_motion(3.0, yaw_deg=2.0))
        data = loop_detector.verify(db, LoopCandidate(1, 0, 0.9)).to_dict()
        assert data["query_keyframe"] == 1 and data["match_keyframe"] == 0
        assert data["n_inliers"] > 0


class TestEndToEndDetection:
    """Detection over a database built from real rendered images."""

    def test_detects_a_planted_revisit(self):
        detector = FeatureDetector("orb", max_features=800)
        matcher = FeatureMatcher(norm=cv2.NORM_HAMMING, min_matches=10)

        # Frames 0..3 revisited at 20..23. Each middle frame gets its own seed:
        # rendering one scene at successive shifts would make the middle
        # self-similar, and the detector would (correctly) close loops there.
        images = (
            [make_textured_image(i, seed=1) for i in range(4)]
            + [make_textured_image(0, seed=200 + i) for i in range(16)]
            + [make_textured_image(i, seed=1) for i in range(4)]
        )
        db = KeyframeDatabase()
        for i, image in enumerate(images):
            frame = detector.detect(image, i, i)
            db.add(make_keyframe(i, frame.descriptors, points=frame.points,
                                 pose=se3_from_rt(np.eye(3), np.array([0.0, 0.0, 5.0 * i])),
                                 distance=5.0 * i))

        sample = np.concatenate([kf.descriptors for kf in db])
        db.set_vocabulary(VisualVocabulary.train(sample, vocabulary_size=64, seed=0))

        loop_detector = LoopClosureDetector(
            K=KITTI_K,
            matcher=matcher,
            min_keyframe_separation=10,
            min_path_separation_m=30.0,
            min_similarity=0.1,
            similarity_ratio=0.3,
            min_matches=20,
            min_inliers=10,
            min_inlier_ratio=0.2,
            cooldown_keyframes=1,
        )
        closures = loop_detector.detect(db)

        assert closures, "expected the planted revisit to be detected"
        for closure in closures:
            # Every accepted loop must link the repeated region to the original.
            assert closure.query_id >= 20
            assert closure.match_id <= 3
        assert loop_detector.stats.accepted == len(closures)

    def test_no_loops_in_a_non_repeating_sequence(self):
        detector = FeatureDetector("orb", max_features=800)
        matcher = FeatureMatcher(norm=cv2.NORM_HAMMING, min_matches=10)
        db = KeyframeDatabase()
        for i in range(40):
            frame = detector.detect(make_textured_image(0, seed=100 + i), i, i)
            db.add(make_keyframe(i, frame.descriptors, points=frame.points,
                                 pose=se3_from_rt(np.eye(3), np.array([0.0, 0.0, 5.0 * i])),
                                 distance=5.0 * i))
        sample = np.concatenate([kf.descriptors for kf in db])
        db.set_vocabulary(VisualVocabulary.train(sample, vocabulary_size=64, seed=0))

        loop_detector = LoopClosureDetector(
            K=KITTI_K, matcher=matcher, min_keyframe_separation=10, min_path_separation_m=30.0
        )
        assert loop_detector.detect(db) == []

    def test_empty_database_detects_nothing(self):
        loop_detector = LoopClosureDetector(
            K=KITTI_K, matcher=FeatureMatcher(norm=cv2.NORM_HAMMING)
        )
        assert loop_detector.detect(KeyframeDatabase()) == []

    def test_from_config(self):
        config = Config().with_overrides({"loop_closure.min_inliers": 77})
        loop_detector = LoopClosureDetector.from_config(
            config, KITTI_K, FeatureMatcher(norm=cv2.NORM_HAMMING)
        )
        assert loop_detector.min_inliers == 77
