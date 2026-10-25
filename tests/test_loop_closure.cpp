// Tests for keyframe selection, BoW retrieval and loop verification (mirrors
// tests/test_loop_closure.py), plus golden checks with Python-trained
// vocabularies so retrieval and verification can be compared exactly.
#include <gtest/gtest.h>

#include <cmath>
#include <set>

#include <opencv2/core/version.hpp>

#include "mvslam/config.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/features/detector.hpp"
#include "mvslam/features/matcher.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/loop_closure/database.hpp"
#include "mvslam/loop_closure/detector.hpp"
#include "support/golden.hpp"
#include "support/kitti_fixture.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using synthetic::forward_motion;
using synthetic::kitti_K;

namespace {

constexpr const char* kModule = "test_loop_closure";

/// make_keyframe(kf_id, descriptors, points=None, pose=None, distance=0.0):
/// default points are uniform draws from default_rng(kf_id) (recorded).
Keyframe make_keyframe(golden::Stream* stream, std::int64_t kf_id, const cv::Mat& descriptors,
                       const std::optional<Points2>& points = std::nullopt,
                       const Eigen::Matrix4d& pose = Eigen::Matrix4d::Identity(), double distance = 0.0) {
    Keyframe kf;
    kf.keyframe_id = kf_id;
    kf.frame_index = kf_id * 5;
    kf.frame_id = kf_id * 5;
    if (points) {
        kf.points = *points;
    } else if (stream != nullptr) {
        kf.points = golden::points2(stream->next("uniform"), &stream->shared());
    } else {
        throw std::runtime_error("make_keyframe needs a stream for its default points");
    }
    kf.descriptors = descriptors;
    kf.pose = pose;
    kf.path_distance_m = distance;
    return kf;
}

/// Binary descriptors drawn from well-separated prototypes (recorded).
cv::Mat clustered_descriptors(golden::Stream& stream) {
    return golden::cvmat(stream.next("clustered_descriptors"), &stream.shared());
}

cv::Mat vconcat(const std::vector<cv::Mat>& blocks) {
    cv::Mat out;
    cv::vconcat(blocks, out);
    return out;
}

LoopCandidate candidate(std::int64_t query, std::int64_t match, double similarity = 0.9) {
    return {query, match, similarity, 0.0};
}

}  // namespace

// --------------------------------------------------------------------------- //
// TestKeyframeSelector
// --------------------------------------------------------------------------- //

// py: test_loop_closure.py::TestKeyframeSelector::test_first_frame_is_always_selected
TEST(TestKeyframeSelector, FirstFrameIsAlwaysSelected) {
    EXPECT_TRUE(KeyframeSelector().should_select(0, Eigen::Matrix4d::Identity()));
}

// py: test_loop_closure.py::TestKeyframeSelector::test_frame_interval_triggers
TEST(TestKeyframeSelector, FrameIntervalTriggers) {
    KeyframeSelector selector(5, 1e9, 1e9);
    selector.accept(0, Eigen::Matrix4d::Identity());
    EXPECT_FALSE(selector.should_select(4, Eigen::Matrix4d::Identity()));
    EXPECT_TRUE(selector.should_select(5, Eigen::Matrix4d::Identity()));
}

// py: test_loop_closure.py::TestKeyframeSelector::test_translation_triggers_before_the_interval
TEST(TestKeyframeSelector, TranslationTriggersBeforeTheInterval) {
    KeyframeSelector selector(1000, 2.0, 1e9);
    selector.accept(0, Eigen::Matrix4d::Identity());
    EXPECT_FALSE(selector.should_select(1, se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 1.5))));
    EXPECT_TRUE(selector.should_select(2, se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 2.5))));
}

// py: test_loop_closure.py::TestKeyframeSelector::test_rotation_triggers_before_the_interval
TEST(TestKeyframeSelector, RotationTriggersBeforeTheInterval) {
    KeyframeSelector selector(1000, 1e9, 10.0);
    selector.accept(0, Eigen::Matrix4d::Identity());
    EXPECT_FALSE(selector.should_select(1, se3_from_rt(rotation::from_euler("y", 5.0), Eigen::Vector3d::Zero())));
    EXPECT_TRUE(selector.should_select(2, se3_from_rt(rotation::from_euler("y", 15.0), Eigen::Vector3d::Zero())));
}

// py: test_loop_closure.py::TestKeyframeSelector::test_accept_resets_the_triggers
TEST(TestKeyframeSelector, AcceptResetsTheTriggers) {
    KeyframeSelector selector(5);
    selector.accept(0, Eigen::Matrix4d::Identity());
    EXPECT_TRUE(selector.should_select(5, Eigen::Matrix4d::Identity()));
    selector.accept(5, Eigen::Matrix4d::Identity());
    EXPECT_FALSE(selector.should_select(6, Eigen::Matrix4d::Identity()));
}

// py: test_loop_closure.py::TestKeyframeSelector::test_reset_restores_initial_state
TEST(TestKeyframeSelector, ResetRestoresInitialState) {
    KeyframeSelector selector(100);
    selector.accept(0, Eigen::Matrix4d::Identity());
    EXPECT_FALSE(selector.should_select(1, Eigen::Matrix4d::Identity()));
    selector.reset();
    EXPECT_TRUE(selector.should_select(1, Eigen::Matrix4d::Identity()));
}

// py: test_loop_closure.py::TestKeyframeSelector::test_from_config
TEST(TestKeyframeSelector, FromConfig) {
    EXPECT_EQ(KeyframeSelector::from_config(Config().with_override("keyframes.every_n_frames", 9)).every_n_frames(), 9);
}

// --------------------------------------------------------------------------- //
// TestVisualVocabulary
// --------------------------------------------------------------------------- //

// py: test_loop_closure.py::TestVisualVocabulary::test_binary_vocabulary_recovers_cluster_structure
TEST(TestVisualVocabulary, BinaryVocabularyRecoversClusterStructure) {
    golden::Stream s(kModule, "TestVisualVocabulary::test_binary_vocabulary_recovers_cluster_structure");
    const cv::Mat descriptors = clustered_descriptors(s);  // 8 clusters x 50
    const auto vocab = VisualVocabulary::train(descriptors, 8, 12, 0);
    EXPECT_TRUE(vocab.binary());
    EXPECT_EQ(vocab.size(), 8u);
    const auto words = vocab.quantize(descriptors);
    // Every member of a synthetic cluster should land on the same word.
    for (int cluster = 0; cluster < 8; ++cluster) {
        std::set<std::int64_t> unique(words.begin() + cluster * 50, words.begin() + (cluster + 1) * 50);
        EXPECT_EQ(unique.size(), 1u) << "cluster " << cluster;
    }
}

// py: test_loop_closure.py::TestVisualVocabulary::test_word_assignment_is_deterministic
TEST(TestVisualVocabulary, WordAssignmentIsDeterministic) {
    golden::Stream s(kModule, "TestVisualVocabulary::test_word_assignment_is_deterministic");
    const cv::Mat descriptors = clustered_descriptors(s);
    const auto vocab = VisualVocabulary::train(descriptors, 4, 12, 0);
    EXPECT_EQ(vocab.quantize(descriptors), vocab.quantize(descriptors));
}

// py: test_loop_closure.py::TestVisualVocabulary::test_histogram_totals_match_descriptor_count
TEST(TestVisualVocabulary, HistogramTotalsMatchDescriptorCount) {
    golden::Stream s(kModule, "TestVisualVocabulary::test_histogram_totals_match_descriptor_count");
    const cv::Mat descriptors = clustered_descriptors(s);
    const auto vocab = VisualVocabulary::train(descriptors, 6, 12, 0);
    const Eigen::VectorXd histogram = vocab.histogram(descriptors);
    EXPECT_EQ(histogram.size(), 6);
    EXPECT_EQ(histogram.sum(), descriptors.rows);
}

// py: test_loop_closure.py::TestVisualVocabulary::test_float_descriptors_use_euclidean_kmeans
TEST(TestVisualVocabulary, FloatDescriptorsUseEuclideanKmeans) {
    golden::Stream s(kModule, "TestVisualVocabulary::test_float_descriptors_use_euclidean_kmeans");
    std::vector<cv::Mat> blocks;
    for (int i = 0; i < 3; ++i) {
        cv::Mat block = s.next_cvmat("normal");
        block.convertTo(block, CV_32F);
        blocks.push_back(block);
    }
    const cv::Mat descriptors = vconcat(blocks);
    const auto vocab = VisualVocabulary::train(descriptors, 3, 12, 0);
    EXPECT_FALSE(vocab.binary());
    const auto words = vocab.quantize(descriptors.rowRange(0, 40));
    EXPECT_EQ(std::set<std::int64_t>(words.begin(), words.end()).size(), 1u);
}

// py: test_loop_closure.py::TestVisualVocabulary::test_vocabulary_is_capped_by_descriptor_count
TEST(TestVisualVocabulary, VocabularyIsCappedByDescriptorCount) {
    golden::Stream s(kModule, "TestVisualVocabulary::test_vocabulary_is_capped_by_descriptor_count");
    EXPECT_EQ(VisualVocabulary::train(clustered_descriptors(s), 1000, 12, 0).size(), 10u);
}

// py: test_loop_closure.py::TestVisualVocabulary::test_empty_input_raises
TEST(TestVisualVocabulary, EmptyInputRaises) {
    try {
        VisualVocabulary::train(cv::Mat(0, 32, CV_8U));
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("zero descriptors"), std::string::npos);
    }
}

// py: test_loop_closure.py::TestVisualVocabulary::test_single_descriptor_raises
TEST(TestVisualVocabulary, SingleDescriptorRaises) {
    try {
        VisualVocabulary::train(cv::Mat::zeros(1, 32, CV_8U));
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("at least 2 words"), std::string::npos);
    }
}

// py: test_loop_closure.py::TestVisualVocabulary::test_quantizing_nothing_returns_empty
TEST(TestVisualVocabulary, QuantizingNothingReturnsEmpty) {
    golden::Stream s(kModule, "TestVisualVocabulary::test_quantizing_nothing_returns_empty");
    const auto vocab = VisualVocabulary::train(clustered_descriptors(s), 3, 12, 0);
    EXPECT_TRUE(vocab.quantize(cv::Mat(0, 32, CV_8U)).empty());
}

// --------------------------------------------------------------------------- //
// TestKeyframeDatabase
// --------------------------------------------------------------------------- //

namespace {
/// Six keyframes: 0/1 and 4/5 look alike, forming a synthetic 'loop'.
KeyframeDatabase database_fixture(golden::Stream& s) {
    const cv::Mat scene_a = clustered_descriptors(s);
    const cv::Mat scene_b = clustered_descriptors(s);
    const cv::Mat scene_c = clustered_descriptors(s);
    const std::vector<cv::Mat> sets = {scene_a, scene_a, scene_b, scene_c, scene_a, scene_a};
    KeyframeDatabase db(VisualVocabulary::train(vconcat(sets), 24, 12, 0));
    for (std::size_t i = 0; i < sets.size(); ++i) {
        db.add(make_keyframe(&s, static_cast<std::int64_t>(i), sets[i], std::nullopt,
                             Eigen::Matrix4d::Identity(), static_cast<double>(i) * 50.0));
    }
    db.build_index();
    return db;
}

class TestKeyframeDatabase : public ::testing::Test {
protected:
    KeyframeDatabase db(const std::string& test) {
        stream_ = std::make_unique<golden::Stream>(kModule, "TestKeyframeDatabase::" + test);
        return database_fixture(*stream_);
    }
    std::unique_ptr<golden::Stream> stream_;
};
}  // namespace

// py: test_loop_closure.py::TestKeyframeDatabase::test_ids_are_assigned_in_order
TEST_F(TestKeyframeDatabase, IdsAreAssignedInOrder) {
    const auto database = db("test_ids_are_assigned_in_order");
    std::vector<std::int64_t> ids;
    for (const auto& kf : database) {
        ids.push_back(kf.keyframe_id);
    }
    EXPECT_EQ(ids, (std::vector<std::int64_t>{0, 1, 2, 3, 4, 5}));
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_bow_vectors_are_unit_norm
TEST_F(TestKeyframeDatabase, BowVectorsAreUnitNorm) {
    const auto database = db("test_bow_vectors_are_unit_norm");
    for (const auto& kf : database) {
        EXPECT_APPROX(kf.bow->norm(), 1.0, std::nullopt, 1e-9);
    }
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_self_similarity_is_one
TEST_F(TestKeyframeDatabase, SelfSimilarityIsOne) {
    auto database = db("test_self_similarity_is_one");
    for (std::size_t i = 0; i < database.size(); ++i) {
        EXPECT_APPROX(database.similarity_scores(static_cast<std::int64_t>(i))[i], 1.0, std::nullopt, 1e-9);
    }
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_similar_scenes_score_higher_than_different_ones
TEST_F(TestKeyframeDatabase, SimilarScenesScoreHigherThanDifferentOnes) {
    auto database = db("test_similar_scenes_score_higher_than_different_ones");
    const auto scores = database.similarity_scores(0);
    EXPECT_GT(scores[4], scores[2]);
    EXPECT_GT(scores[4], scores[3]);
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_query_finds_the_revisited_keyframe
TEST_F(TestKeyframeDatabase, QueryFindsTheRevisitedKeyframe) {
    auto database = db("test_query_finds_the_revisited_keyframe");
    const auto results = database.query(5, 2, 50.0, 3, 0.0);
    ASSERT_FALSE(results.empty());
    EXPECT_TRUE(results[0].first == 0 || results[0].first == 1);
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_temporal_exclusion_blocks_adjacent_keyframes
TEST_F(TestKeyframeDatabase, TemporalExclusionBlocksAdjacentKeyframes) {
    auto database = db("test_temporal_exclusion_blocks_adjacent_keyframes");
    for (const auto& [kf_id, score] : database.query(5, 3, 0.0, 5, 0.0)) {
        EXPECT_LE(kf_id, 2);
    }
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_path_exclusion_blocks_a_stationary_vehicle
TEST_F(TestKeyframeDatabase, PathExclusionBlocksAStationaryVehicle) {
    // Many keyframes at the same place must not match each other.
    golden::Stream s(kModule, "TestKeyframeDatabase::test_path_exclusion_blocks_a_stationary_vehicle");
    const cv::Mat descriptors = clustered_descriptors(s);
    KeyframeDatabase database(VisualVocabulary::train(descriptors, 16, 12, 0));
    for (int i = 0; i < 60; ++i) {
        database.add(make_keyframe(&s, i, descriptors, std::nullopt, Eigen::Matrix4d::Identity(), 0.05 * i));
    }
    database.build_index();
    EXPECT_TRUE(database.query(59, 30, 30.0).empty());
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_min_similarity_filters_results
TEST_F(TestKeyframeDatabase, MinSimilarityFiltersResults) {
    auto database = db("test_min_similarity_filters_results");
    EXPECT_TRUE(database.query(5, 2, 0.0, 5, 1.01).empty());
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_top_k_limits_results
TEST_F(TestKeyframeDatabase, TopKLimitsResults) {
    auto database = db("test_top_k_limits_results");
    EXPECT_LE(database.query(5, 1, 0.0, 2, 0.0).size(), 2u);
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_results_are_sorted_by_descending_score
TEST_F(TestKeyframeDatabase, ResultsAreSortedByDescendingScore) {
    auto database = db("test_results_are_sorted_by_descending_score");
    const auto results = database.query(5, 1, 0.0, 5, 0.0);
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].second, results[i].second);
    }
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_neighbour_score_reflects_adjacent_similarity
TEST_F(TestKeyframeDatabase, NeighbourScoreReflectsAdjacentSimilarity) {
    auto database = db("test_neighbour_score_reflects_adjacent_similarity");
    const double score = database.neighbour_score(1, 1);
    EXPECT_GT(score, 0.0);
    EXPECT_LE(score, 1.0);
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_empty_database_query_returns_nothing
TEST_F(TestKeyframeDatabase, EmptyDatabaseQueryReturnsNothing) {
    EXPECT_TRUE(KeyframeDatabase().query(0).empty());
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_index_without_vocabulary_raises
TEST_F(TestKeyframeDatabase, IndexWithoutVocabularyRaises) {
    golden::Stream s(kModule, "TestKeyframeDatabase::test_index_without_vocabulary_raises");
    KeyframeDatabase database;
    database.add(make_keyframe(&s, 0, clustered_descriptors(s)));
    try {
        database.build_index();
        ADD_FAILURE() << "expected RuntimeError";
    } catch (const RuntimeError& exc) {
        EXPECT_NE(std::string(exc.what()).find("vocabulary must be set"), std::string::npos);
    }
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_out_of_range_query_raises
TEST_F(TestKeyframeDatabase, OutOfRangeQueryRaises) {
    auto database = db("test_out_of_range_query_raises");
    EXPECT_THROW(database.similarity_scores(99), IndexError);
}

// py: test_loop_closure.py::TestKeyframeDatabase::test_describe_reports_configuration
TEST_F(TestKeyframeDatabase, DescribeReportsConfiguration) {
    const auto database = db("test_describe_reports_configuration");
    const Json info = database.describe();
    EXPECT_EQ(info["n_keyframes"], 6);
    EXPECT_EQ(info["binary_vocabulary"], true);
}

// --------------------------------------------------------------------------- //
// TestGeometricVerification
// --------------------------------------------------------------------------- //

namespace {

FeatureMatcher matcher_fixture() {
    return FeatureMatcher(cv::NORM_HAMMING, true, 0.75, true, 64.0, 10);
}

LoopClosureDetector loop_detector_fixture() {
    LoopClosureParams params;
    params.min_matches = 20;
    params.min_inliers = 15;
    params.min_inlier_ratio = 0.3;
    return LoopClosureDetector(kitti_K(), matcher_fixture(), params);
}

/// Two keyframes whose correspondences come from a known camera motion.
/// Identical descriptors on both sides means matching is trivial; the test
/// isolates the geometric stage.
KeyframeDatabase synthetic_pair(golden::Stream& s, const Eigen::Matrix4d& motion) {
    const auto c = synthetic::two_view_correspondences(s, motion, std::nullopt, 0.2);
    const cv::Mat descriptors = s.next_cvmat("integers");
    KeyframeDatabase db;
    db.add(make_keyframe(nullptr, 0, descriptors, c.points_a, Eigen::Matrix4d::Identity(), 0.0));
    db.add(make_keyframe(nullptr, 1, descriptors, c.points_b, motion, 100.0));
    return db;
}

/// A loop pair plus a metric neighbour, with drift in the stored poses.
///
/// Layout in the *true* world frame: ``match`` keyframe 0 at the origin,
/// ``neighbour`` keyframe 1 one odometry step behind the query, ``query``
/// keyframe 2 ``true_separation_m`` from ``match``. The stored poses of the
/// neighbour and query carry ``drift_m`` of accumulated error, exactly as a
/// drifted trajectory would. Their *relative* pose stays metric and correct,
/// which is the information odometry actually provides locally.
KeyframeDatabase three_view_database(golden::Stream& s, double true_separation_m, double drift_m) {
    const auto xs = s.next_vec("uniform");
    const auto ys = s.next_vec("uniform");
    const auto zs = s.next_vec("uniform");
    const cv::Mat descriptors = s.next_cvmat("integers");
    const auto n = static_cast<Eigen::Index>(xs.size());
    Points3 points_world(n, 3);
    for (Eigen::Index i = 0; i < n; ++i) {
        points_world.row(i) << xs[static_cast<std::size_t>(i)], ys[static_cast<std::size_t>(i)],
            zs[static_cast<std::size_t>(i)];
    }
    const Eigen::Matrix4d T_match = Eigen::Matrix4d::Identity();
    const Eigen::Matrix4d T_query = se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(true_separation_m, 0, 0));
    const Eigen::Matrix4d T_neighbour = T_query * se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, -3.0));

    auto observe = [&](const Eigen::Matrix4d& T_world_cam, Points2& uv, Mask& ok) {
        const Eigen::Matrix4d cam = invert_se3(T_world_cam);
        Points3 points_cam(n, 3);
        for (Eigen::Index i = 0; i < n; ++i) {
            points_cam.row(i) = (cam.topLeftCorner<3, 3>() * points_world.row(i).transpose() +
                                 cam.topRightCorner<3, 1>()).transpose();
        }
        uv = synthetic::project(points_cam);
        ok.resize(static_cast<std::size_t>(n));
        for (Eigen::Index i = 0; i < n; ++i) {
            ok[static_cast<std::size_t>(i)] = points_cam(i, 2) > 1.0 ? 1 : 0;
        }
    };
    Points2 uv_match, uv_query, uv_neighbour;
    Mask ok_m, ok_q, ok_n;
    observe(T_match, uv_match, ok_m);
    observe(T_query, uv_query, ok_q);
    observe(T_neighbour, uv_neighbour, ok_n);
    Mask keep(static_cast<std::size_t>(n));
    std::vector<int> rows;
    for (std::size_t i = 0; i < keep.size(); ++i) {
        keep[i] = ok_m[i] && ok_q[i] && ok_n[i];
        if (keep[i]) {
            rows.push_back(static_cast<int>(i));
        }
    }
    cv::Mat kept_desc(static_cast<int>(rows.size()), descriptors.cols, descriptors.type());
    for (std::size_t k = 0; k < rows.size(); ++k) {
        descriptors.row(rows[k]).copyTo(kept_desc.row(static_cast<int>(k)));
    }
    const Eigen::Matrix4d drift = se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, drift_m));
    KeyframeDatabase db;
    db.add(make_keyframe(nullptr, 0, kept_desc, select_rows(uv_match, keep), T_match));
    db.add(make_keyframe(nullptr, 1, kept_desc, select_rows(uv_neighbour, keep), drift * T_neighbour, 100.0));
    db.add(make_keyframe(nullptr, 2, kept_desc, select_rows(uv_query, keep), drift * T_query, 103.0));
    return db;
}

}  // namespace

// py: test_loop_closure.py::TestGeometricVerification::test_verifies_a_true_pair_and_recovers_the_motion
TEST(TestGeometricVerification, VerifiesATruePairAndRecoversTheMotion) {
    golden::Stream s(kModule, "TestGeometricVerification::test_verifies_a_true_pair_and_recovers_the_motion");
    const Eigen::Matrix4d motion = forward_motion(3.0, 4.0);
    auto db = synthetic_pair(s, motion);
    auto detector = loop_detector_fixture();
    const auto closure = detector.verify(db, candidate(1, 0));
    ASSERT_TRUE(closure.has_value());
    EXPECT_GE(closure->n_inliers, 15);
    EXPECT_LT(rotation_angle_deg(Eigen::Matrix3d(closure->T_match_query.topLeftCorner<3, 3>().transpose() *
                                                 motion.topLeftCorner<3, 3>())),
              2.0);
}

// py: test_loop_closure.py::TestGeometricVerification::test_scale_is_recovered_from_structure_not_the_odometry_gap
TEST(TestGeometricVerification, ScaleIsRecoveredFromStructureNotTheOdometryGap) {
    // Regression test for the bug that made optimization worse than raw VO:
    // using the odometry gap between the two loop keyframes as the loop
    // translation magnitude bakes the accumulated drift into the constraint.
    // The magnitude must instead come from triangulated structure.
    golden::Stream s(kModule, "TestGeometricVerification::test_scale_is_recovered_from_structure_not_the_odometry_gap");
    auto db = three_view_database(s, 2.0, 15.0);
    auto detector = loop_detector_fixture();
    const auto closure = detector.verify(db, candidate(2, 0));
    ASSERT_TRUE(closure.has_value());
    EXPECT_EQ(closure->scale_method, "structure");
    EXPECT_GE(closure->n_scale_points, 15);
    const double odometry_gap = translation_of(invert_se3(db[0].pose) * db[2].pose).norm();
    EXPECT_GT(odometry_gap, 14.0);  // the drift really is in there
    EXPECT_APPROX(closure->scale_m, 2.0, 0.15);
    EXPECT_GT(std::fabs(closure->scale_m - odometry_gap), 10.0);
}

// py: test_loop_closure.py::TestGeometricVerification::test_recovered_translation_matches_the_true_displacement
TEST(TestGeometricVerification, RecoveredTranslationMatchesTheTrueDisplacement) {
    golden::Stream s(kModule, "TestGeometricVerification::test_recovered_translation_matches_the_true_displacement");
    auto db = three_view_database(s, 3.0, 20.0);
    auto detector = loop_detector_fixture();
    const auto closure = detector.verify(db, candidate(2, 0));
    ASSERT_TRUE(closure.has_value());
    const Eigen::Vector3d translation = translation_of(closure->T_match_query);
    EXPECT_APPROX(translation.norm(), 3.0, 0.15);
    // True displacement is along +x in the match keyframe's frame.
    EXPECT_APPROX(translation(0) / translation.norm(), 1.0, std::nullopt, 0.1);
}

// py: test_loop_closure.py::TestGeometricVerification::test_unmeasurable_scale_is_rejected_by_default
TEST(TestGeometricVerification, UnmeasurableScaleIsRejectedByDefault) {
    // A pair that cannot be confirmed co-located must not become a constraint.
    golden::Stream s(kModule, "TestGeometricVerification::test_unmeasurable_scale_is_rejected_by_default");
    auto db = synthetic_pair(s, forward_motion(3.0, 2.0));
    auto detector = loop_detector_fixture();
    detector.params().min_scale_points = 10'000;  // force estimation to fail
    EXPECT_FALSE(detector.verify(db, candidate(1, 0)).has_value());
    EXPECT_EQ(detector.stats().scale_estimation_failed, 1);
}

// py: test_loop_closure.py::TestGeometricVerification::test_unmeasurable_scale_can_be_kept_as_orientation_only
TEST(TestGeometricVerification, UnmeasurableScaleCanBeKeptAsOrientationOnly) {
    // Opting out keeps the loop, but with no asserted magnitude.
    golden::Stream s(kModule, "TestGeometricVerification::test_unmeasurable_scale_can_be_kept_as_orientation_only");
    auto db = synthetic_pair(s, forward_motion(3.0, 2.0));
    auto detector = loop_detector_fixture();
    detector.params().min_scale_points = 10'000;
    detector.params().require_measured_scale = false;
    const auto closure = detector.verify(db, candidate(1, 0));
    ASSERT_TRUE(closure.has_value());
    EXPECT_EQ(closure->scale_method, "unscaled");
    EXPECT_EQ(closure->scale_m, 0.0);
    EXPECT_FALSE(closure->scale_is_measured());
}

// py: test_loop_closure.py::TestGeometricVerification::test_implausibly_large_scale_is_rejected
TEST(TestGeometricVerification, ImplausiblyLargeScaleIsRejected) {
    // The highway-aliasing case: a confident match between distant places.
    // Regression test for KITTI sequence 01, where nine detections passed
    // appearance and geometric verification despite a true separation of
    // 73-108 m, because forward motion along a repetitive corridor is a
    // perfectly consistent camera motion.
    golden::Stream s(kModule, "TestGeometricVerification::test_implausibly_large_scale_is_rejected");
    auto db = three_view_database(s, 6.0, 5.0);
    auto detector = loop_detector_fixture();
    detector.params().max_loop_scale_m = 2.0;  // below the true separation
    EXPECT_FALSE(detector.verify(db, candidate(2, 0)).has_value());
    EXPECT_EQ(detector.stats().rejected_large_scale, 1);
}

// py: test_loop_closure.py::TestGeometricVerification::test_plausible_scale_is_accepted
TEST(TestGeometricVerification, PlausibleScaleIsAccepted) {
    golden::Stream s(kModule, "TestGeometricVerification::test_plausible_scale_is_accepted");
    auto db = three_view_database(s, 2.0, 15.0);
    auto detector = loop_detector_fixture();
    detector.params().max_loop_scale_m = 10.0;
    EXPECT_TRUE(detector.verify(db, candidate(2, 0)).has_value());
    EXPECT_EQ(detector.stats().rejected_large_scale, 0);
}

// py: test_loop_closure.py::TestGeometricVerification::test_random_correspondences_are_rejected
TEST(TestGeometricVerification, RandomCorrespondencesAreRejected) {
    // A false positive from perceptual aliasing must not survive geometry.
    golden::Stream s(kModule, "TestGeometricVerification::test_random_correspondences_are_rejected");
    const cv::Mat descriptors = s.next_cvmat("integers");
    KeyframeDatabase db;
    db.add(make_keyframe(nullptr, 0, descriptors, golden::points2(s.next("uniform"), &s.shared())));
    db.add(make_keyframe(nullptr, 1, descriptors, golden::points2(s.next("uniform"), &s.shared()),
                         Eigen::Matrix4d::Identity(), 100.0));
    auto detector = loop_detector_fixture();
    EXPECT_FALSE(detector.verify(db, candidate(1, 0)).has_value());
    EXPECT_GE(detector.stats().rejected_geometry, 1);
}

// py: test_loop_closure.py::TestGeometricVerification::test_too_few_matches_is_rejected
TEST(TestGeometricVerification, TooFewMatchesIsRejected) {
    golden::Stream s(kModule, "TestGeometricVerification::test_too_few_matches_is_rejected");
    KeyframeDatabase db;
    for (int i = 0; i < 2; ++i) {
        const cv::Mat descriptors = s.next_cvmat("integers");
        db.add(make_keyframe(&s, i, descriptors, std::nullopt, Eigen::Matrix4d::Identity(), 100.0 * i));
    }
    auto detector = loop_detector_fixture();
    EXPECT_FALSE(detector.verify(db, candidate(1, 0)).has_value());
    EXPECT_EQ(detector.stats().rejected_too_few_matches, 1);
}

// py: test_loop_closure.py::TestGeometricVerification::test_inlier_ratio_gate_is_enforced
TEST(TestGeometricVerification, InlierRatioGateIsEnforced) {
    golden::Stream s(kModule, "TestGeometricVerification::test_inlier_ratio_gate_is_enforced");
    LoopClosureParams params;
    params.min_matches = 20;
    params.min_inliers = 15;
    params.min_inlier_ratio = 1.01;
    LoopClosureDetector strict(kitti_K(), matcher_fixture(), params);
    auto db = synthetic_pair(s, forward_motion(3.0, 2.0));
    EXPECT_FALSE(strict.verify(db, candidate(1, 0)).has_value());
}

// py: test_loop_closure.py::TestGeometricVerification::test_closure_serialises
TEST(TestGeometricVerification, ClosureSerialises) {
    golden::Stream s(kModule, "TestGeometricVerification::test_closure_serialises");
    auto db = synthetic_pair(s, forward_motion(3.0, 2.0));
    auto detector = loop_detector_fixture();
    const Json data = detector.verify(db, candidate(1, 0))->to_dict();
    EXPECT_EQ(data["query_keyframe"], 1);
    EXPECT_EQ(data["match_keyframe"], 0);
    EXPECT_GT(data["n_inliers"].get<std::int64_t>(), 0);
}

// --------------------------------------------------------------------------- //
// TestEndToEndDetection
// --------------------------------------------------------------------------- //

namespace {
KeyframeDatabase image_database(const std::vector<cv::Mat>& images) {
    const FeatureDetector detector("orb", 800);
    KeyframeDatabase db;
    for (std::size_t i = 0; i < images.size(); ++i) {
        const auto idx = static_cast<std::int64_t>(i);
        const Frame frame = detector.detect(images[i], idx, idx);
        db.add(make_keyframe(nullptr, idx, frame.descriptors, frame.points,
                             se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 5.0 * idx)),
                             5.0 * idx));
    }
    return db;
}

cv::Mat all_descriptors(const KeyframeDatabase& db) {
    std::vector<cv::Mat> blocks;
    for (const auto& kf : db) {
        blocks.push_back(kf.descriptors);
    }
    return vconcat(blocks);
}

std::vector<cv::Mat> planted_revisit_images() {
    // Frames 0..3 revisited at 20..23. Each middle frame gets its own seed:
    // rendering one scene at successive shifts would make the middle
    // self-similar, and the detector would (correctly) close loops there.
    std::vector<cv::Mat> images;
    for (int i = 0; i < 4; ++i) {
        images.push_back(kitti_fixture::make_textured_image(i, 320, 120, 1));
    }
    for (int i = 0; i < 16; ++i) {
        images.push_back(kitti_fixture::make_textured_image(0, 320, 120, 200 + i));
    }
    for (int i = 0; i < 4; ++i) {
        images.push_back(kitti_fixture::make_textured_image(i, 320, 120, 1));
    }
    return images;
}

LoopClosureDetector planted_detector() {
    LoopClosureParams params;
    params.min_keyframe_separation = 10;
    params.min_path_separation_m = 30.0;
    params.min_similarity = 0.1;
    params.similarity_ratio = 0.3;
    params.min_matches = 20;
    params.min_inliers = 10;
    params.min_inlier_ratio = 0.2;
    params.cooldown_keyframes = 1;
    return LoopClosureDetector(kitti_K(), matcher_fixture(), params);
}
}  // namespace

// py: test_loop_closure.py::TestEndToEndDetection::test_detects_a_planted_revisit
TEST(TestEndToEndDetection, DetectsAPlantedRevisit) {
    KeyframeDatabase db = image_database(planted_revisit_images());
    db.set_vocabulary(VisualVocabulary::train(all_descriptors(db), 64, 12, 0));
    auto detector = planted_detector();
    const auto closures = detector.detect(db);
    ASSERT_FALSE(closures.empty()) << "expected the planted revisit to be detected";
    for (const auto& closure : closures) {
        // Every accepted loop must link the repeated region to the original.
        EXPECT_GE(closure.query_id, 20);
        EXPECT_LE(closure.match_id, 3);
    }
    EXPECT_EQ(detector.stats().accepted, static_cast<std::int64_t>(closures.size()));
}

// py: test_loop_closure.py::TestEndToEndDetection::test_no_loops_in_a_non_repeating_sequence
TEST(TestEndToEndDetection, NoLoopsInANonRepeatingSequence) {
    std::vector<cv::Mat> images;
    for (int i = 0; i < 40; ++i) {
        images.push_back(kitti_fixture::make_textured_image(0, 320, 120, 100 + i));
    }
    KeyframeDatabase db = image_database(images);
    db.set_vocabulary(VisualVocabulary::train(all_descriptors(db), 64, 12, 0));
    LoopClosureParams params;
    params.min_keyframe_separation = 10;
    params.min_path_separation_m = 30.0;
    LoopClosureDetector detector(kitti_K(), matcher_fixture(), params);
    EXPECT_TRUE(detector.detect(db).empty());
}

// py: test_loop_closure.py::TestEndToEndDetection::test_empty_database_detects_nothing
TEST(TestEndToEndDetection, EmptyDatabaseDetectsNothing) {
    KeyframeDatabase db;
    LoopClosureDetector detector(kitti_K(), FeatureMatcher(cv::NORM_HAMMING));
    EXPECT_TRUE(detector.detect(db).empty());
}

// py: test_loop_closure.py::TestEndToEndDetection::test_from_config
TEST(TestEndToEndDetection, FromConfig) {
    const Config config = Config().with_override("loop_closure.min_inliers", 77);
    EXPECT_EQ(LoopClosureDetector::from_config(config, kitti_K(), FeatureMatcher(cv::NORM_HAMMING)).params().min_inliers,
              77);
}

// --------------------------------------------------------------------------- //
// Golden parity with Python-trained vocabularies
// --------------------------------------------------------------------------- //

namespace {
bool same_opencv_as_reference() {
    static const std::string name = "metadata.json";
    const std::string reference = golden::load(name).at("cv2").get<std::string>();
    return reference.rfind(std::to_string(CV_VERSION_MAJOR) + "." + std::to_string(CV_VERSION_MINOR) + ".", 0) == 0;
}
}  // namespace

TEST(LoopClosureGolden, BowIndexAndRetrievalMatchPython) {
    static const std::string name = "database.json";
    const Json& g = golden::load(name);
    KeyframeDatabase db(VisualVocabulary(golden::cvmat(g.at("centroids")), true));
    const auto& sets = g.at("descriptor_sets");
    for (std::size_t i = 0; i < sets.size(); ++i) {
        db.add(make_keyframe(nullptr, static_cast<std::int64_t>(i), golden::cvmat(sets[i]),
                             golden::points2(g.at("points")[i]), Eigen::Matrix4d::Identity(),
                             static_cast<double>(i) * 50.0));
    }
    db.build_index();
    for (std::size_t i = 0; i < sets.size(); ++i) {
        EXPECT_EQ(db.vocabulary()->quantize(golden::cvmat(sets[i])), golden::ivec(g.at("words")[i]));
        const Eigen::VectorXd histogram = db.vocabulary()->histogram(golden::cvmat(sets[i]));
        EXPECT_ALLCLOSE(std::vector<double>(histogram.data(), histogram.data() + histogram.size()),
                        golden::vec(g.at("histograms")[i]), 0.0, 0.0);
        EXPECT_ALLCLOSE(db.similarity_scores(static_cast<std::int64_t>(i)), golden::vec(g.at("similarity")[i]),
                        1e-14, 1e-15);
    }
    EXPECT_ALLCLOSE(Eigen::MatrixXd(db.bow_matrix()), golden::mat(g.at("bow")), 1e-14, 1e-16);
    for (const auto& item : g.at("neighbour_score")) {
        EXPECT_NEAR(db.neighbour_score(item.at(0).get<std::int64_t>(), item.at(1).get<std::int64_t>()),
                    golden::from_hex(item.at(2)), 1e-14);
    }
    for (const auto& item : g.at("queries")) {
        const auto& args = item.at("args");
        const auto result = db.query(item.at("query").get<std::int64_t>(), args.at(0).get<std::int64_t>(),
                                     args.at(1).get<double>(), args.at(2).get<std::int64_t>(),
                                     args.at(3).get<double>());
        const auto& expected = item.at("result");
        ASSERT_EQ(result.size(), expected.size()) << py::dumps(item);
        for (std::size_t k = 0; k < result.size(); ++k) {
            EXPECT_EQ(result[k].first, expected[k].at(0).get<std::int64_t>()) << py::dumps(item);
            EXPECT_NEAR(result[k].second, golden::from_hex(expected[k].at(1)), 1e-14);
        }
    }
    // Float vocabulary: Python-trained centroids assign the same words.
    const VisualVocabulary float_vocab(golden::cvmat(g.at("float_centroids")), false);
    EXPECT_EQ(float_vocab.quantize(golden::cvmat(g.at("float_descriptors"))), golden::ivec(g.at("float_words")));
}

TEST(LoopClosureGolden, VerificationMatchesPython) {
    if (!same_opencv_as_reference()) {
        GTEST_SKIP() << "OpenCV " << CV_VERSION << " differs from the reference cv2 version";
    }
    static const std::string name = "loop_detector.json";
    const Json& g = golden::load(name);
    for (const auto& item : g.at("verify")) {
        SCOPED_TRACE(item.at("name").get<std::string>());
        KeyframeDatabase db;
        std::int64_t id = 0;
        for (const auto& kf : item.at("database")) {
            Keyframe k = make_keyframe(nullptr, id++, golden::cvmat(kf.at("descriptors")),
                                       golden::points2(kf.at("points")), golden::mat4(kf.at("pose")),
                                       golden::from_hex(kf.at("distance")));
            k.frame_index = kf.at("frame_index").get<std::int64_t>();
            db.add(k);
        }
        auto detector = loop_detector_fixture();
        const Json& attributes = item.at("attributes");
        if (attributes.contains("max_loop_scale_m")) detector.params().max_loop_scale_m = attributes.at("max_loop_scale_m").get<double>();
        if (attributes.contains("min_scale_points")) detector.params().min_scale_points = attributes.at("min_scale_points").get<std::int64_t>();
        if (attributes.contains("require_measured_scale")) detector.params().require_measured_scale = attributes.at("require_measured_scale").get<bool>();
        cv::setRNGSeed(0);
        const auto closure = detector.verify(db, candidate(item.at("query").get<std::int64_t>(),
                                                           item.at("match").get<std::int64_t>()));
        EXPECT_EQ(detector.stats().to_dict(), item.at("stats"));
        if (item.at("closure").is_null()) {
            EXPECT_FALSE(closure.has_value());
            continue;
        }
        ASSERT_TRUE(closure.has_value());
        EXPECT_EQ(closure->to_dict(), item.at("closure").at("dict"));
        EXPECT_ALLCLOSE(Eigen::MatrixXd(closure->T_match_query), golden::mat(item.at("closure").at("T")), 1e-9, 1e-10);
        EXPECT_NEAR(closure->scale_m, golden::from_hex(item.at("closure").at("scale_m")), 1e-9);
    }
}

TEST(LoopClosureGolden, PlantedRevisitDetectionMatchesPythonWithTheSameVocabulary) {
    if (!same_opencv_as_reference()) {
        GTEST_SKIP() << "OpenCV " << CV_VERSION << " differs from the reference cv2 version";
    }
    static const std::string name = "loop_detector.json";
    const Json& planted = golden::load(name).at("planted");
    KeyframeDatabase db = image_database(planted_revisit_images());
    db.set_vocabulary(VisualVocabulary(golden::cvmat(planted.at("centroids")), true));
    auto detector = planted_detector();
    cv::setRNGSeed(0);
    const auto closures = detector.detect(db);
    EXPECT_ALLCLOSE(Eigen::MatrixXd(db.bow_matrix()), golden::mat(planted.at("bow")), 1e-14, 1e-16);
    EXPECT_EQ(detector.stats().to_dict(), planted.at("stats"));
    ASSERT_EQ(closures.size(), planted.at("closures").size());
    for (std::size_t i = 0; i < closures.size(); ++i) {
        EXPECT_EQ(closures[i].to_dict(), planted.at("closures")[i].at("dict"));
        EXPECT_ALLCLOSE(Eigen::MatrixXd(closures[i].T_match_query),
                        golden::mat(planted.at("closures")[i].at("T")), 1e-9, 1e-10);
    }
}
