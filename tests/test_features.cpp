// Tests for feature detection and descriptor matching (mirrors
// tests/test_features.py), plus golden checks that the C++ front end produces
// exactly what cv2 produced for the Python implementation.
#include <gtest/gtest.h>

#include <cmath>
#include <map>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc.hpp>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/config.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/features/detector.hpp"
#include "mvslam/features/matcher.hpp"
#include "support/golden.hpp"
#include "support/kitti_fixture.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;

namespace {

Eigen::MatrixXd M(const Eigen::MatrixXd& m) { return m; }

cv::Mat image() { return kitti_fixture::make_textured_image(0, 480, 200); }

/// The same scene translated by a known amount, for match verification.
cv::Mat shifted_image() {
    const cv::Mat base = image();
    cv::Mat shifted = cv::Mat::zeros(base.size(), base.type());
    base(cv::Rect(10, 0, base.cols - 10, base.rows)).copyTo(shifted(cv::Rect(0, 0, base.cols - 10, base.rows)));
    return shifted;
}

FeatureDetector orb() { return FeatureDetector("orb", 500); }

template <typename Exc, typename Fn>
void expect_raises(Fn&& fn, const std::string& fragment) {
    try {
        fn();
        ADD_FAILURE() << "expected exception matching '" << fragment << "'";
    } catch (const Exc& exc) {
        EXPECT_NE(std::string(exc.what()).find(fragment), std::string::npos) << exc.what();
    }
}

/// Bit-level front-end parity is only meaningful against the same OpenCV.
bool same_opencv_as_reference() {
    const std::string reference = golden::load("metadata.json").at("cv2").get<std::string>();
    return reference.rfind(std::to_string(CV_VERSION_MAJOR) + "." + std::to_string(CV_VERSION_MINOR) + ".", 0) == 0;
}

}  // namespace

// --------------------------------------------------------------------------- //
// TestDetectorConstruction
// --------------------------------------------------------------------------- //

// py: test_features.py::TestDetectorConstruction::test_unknown_detector_raises
TEST(TestDetectorConstruction, UnknownDetectorRaises) {
    expect_raises<ValueError>([] { FeatureDetector("surf"); }, "Unsupported detector");
}

// py: test_features.py::TestDetectorConstruction::test_non_positive_budget_raises
TEST(TestDetectorConstruction, NonPositiveBudgetRaises) {
    expect_raises<ValueError>([] { FeatureDetector("orb", 0); }, "max_features");
}

// py: test_features.py::TestDetectorConstruction::test_orb_uses_hamming_norm
TEST(TestDetectorConstruction, OrbUsesHammingNorm) { EXPECT_EQ(orb().descriptor_norm(), cv::NORM_HAMMING); }

// py: test_features.py::TestDetectorConstruction::test_sift_uses_l2_norm
TEST(TestDetectorConstruction, SiftUsesL2Norm) {
    EXPECT_EQ(FeatureDetector("sift", 100).descriptor_norm(), cv::NORM_L2);
}

// py: test_features.py::TestDetectorConstruction::test_from_config
TEST(TestDetectorConstruction, FromConfig) {
    const Config config = Config().with_override("features.max_features", 123);
    EXPECT_EQ(FeatureDetector::from_config(config).max_features(), 123);
}

// --------------------------------------------------------------------------- //
// TestDetection
// --------------------------------------------------------------------------- //

// py: test_features.py::TestDetection::test_detects_keypoints_with_descriptors
TEST(TestDetection, DetectsKeypointsWithDescriptors) {
    const Frame frame = orb().detect(image(), 3, 17);
    EXPECT_GT(frame.size(), 50u);
    EXPECT_EQ(frame.index, 3);
    EXPECT_EQ(frame.frame_id, 17);
    EXPECT_EQ(frame.descriptors.rows, static_cast<int>(frame.size()));
    EXPECT_EQ(frame.descriptors.cols, 32);
    EXPECT_EQ(frame.descriptors.depth(), CV_8U);
    EXPECT_TRUE(frame.is_usable());
}

// py: test_features.py::TestDetection::test_points_match_keypoints
TEST(TestDetection, PointsMatchKeypoints) {
    const Frame frame = orb().detect(image());
    EXPECT_EQ(frame.points.rows(), static_cast<Eigen::Index>(frame.size()));
    EXPECT_EQ(frame.points(0, 0), frame.keypoints[0].pt.x);
    EXPECT_EQ(frame.points(0, 1), frame.keypoints[0].pt.y);
}

// py: test_features.py::TestDetection::test_respects_feature_budget
TEST(TestDetection, RespectsFeatureBudget) {
    EXPECT_LE(FeatureDetector("orb", 40).detect(image()).size(), 40u);
}

// py: test_features.py::TestDetection::test_sift_produces_float_descriptors
TEST(TestDetection, SiftProducesFloatDescriptors) {
    const Frame frame = FeatureDetector("sift", 200).detect(image());
    EXPECT_EQ(frame.descriptors.depth(), CV_32F);
    EXPECT_EQ(frame.descriptors.cols, 128);
}

// py: test_features.py::TestDetection::test_blank_image_returns_empty_frame
TEST(TestDetection, BlankImageReturnsEmptyFrame) {
    const Frame frame = orb().detect(cv::Mat::zeros(120, 160, CV_8U), 5, 5);
    EXPECT_EQ(frame.size(), 0u);
    EXPECT_TRUE(frame.descriptors.empty());
    EXPECT_FALSE(frame.is_usable());
    EXPECT_EQ(frame.points.rows(), 0);
    EXPECT_EQ(frame.points.cols(), 2);
}

// py: test_features.py::TestDetection::test_colour_input_is_converted
TEST(TestDetection, ColourInputIsConverted) {
    cv::Mat colour;
    cv::cvtColor(image(), colour, cv::COLOR_GRAY2BGR);
    EXPECT_EQ(orb().detect(colour).size(), orb().detect(image()).size());
}

// py: test_features.py::TestDetection::test_clahe_variant_still_detects
TEST(TestDetection, ClaheVariantStillDetects) {
    EXPECT_GT(FeatureDetector("orb", 300, 20, 1.2, 8, 31, 0, true).detect(image()).size(), 20u);
}

// py: test_features.py::TestDetection::test_detection_is_deterministic
TEST(TestDetection, DetectionIsDeterministic) {
    const Frame a = orb().detect(image());
    const Frame b = orb().detect(image());
    EXPECT_ALLCLOSE(M(a.points), M(b.points));
    EXPECT_EQ(cv::norm(a.descriptors, b.descriptors, cv::NORM_HAMMING), 0.0);
}

// py: test_features.py::TestDetection::test_mask_restricts_detection_region
TEST(TestDetection, MaskRestrictsDetectionRegion) {
    const cv::Mat img = image();
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
    mask(cv::Rect(0, 0, 100, img.rows)).setTo(255);
    const Frame frame = orb().detect(img, 0, std::nullopt, mask);
    ASSERT_GT(frame.size(), 0u);
    EXPECT_LT(frame.points.col(0).maxCoeff(), 120);  // allow for the ORB patch border
}

// py: test_features.py::TestDetection::test_frame_rejects_mismatched_descriptors
TEST(TestDetection, FrameRejectsMismatchedDescriptors) {
    expect_raises<ValueError>(
        [] { Frame(0, 0, {cv::KeyPoint(1.0f, 2.0f, 5.0f)}, cv::Mat::zeros(3, 32, CV_8U)); },
        "descriptor count");
}

// --------------------------------------------------------------------------- //
// TestKeypointVisualisation
// --------------------------------------------------------------------------- //

// py: test_features.py::TestKeypointVisualisation::test_returns_colour_canvas
TEST(TestKeypointVisualisation, ReturnsColourCanvas) {
    const cv::Mat img = image();
    const cv::Mat vis = draw_keypoints(img, orb().detect(img));
    EXPECT_EQ(vis.rows, img.rows);
    EXPECT_EQ(vis.cols, img.cols);
    EXPECT_EQ(vis.channels(), 3);
}

// py: test_features.py::TestKeypointVisualisation::test_does_not_mutate_input
TEST(TestKeypointVisualisation, DoesNotMutateInput) {
    const cv::Mat img = image();
    const cv::Mat original = img.clone();
    (void)draw_keypoints(img, orb().detect(img));
    EXPECT_EQ(cv::norm(img, original, cv::NORM_INF), 0.0);
}

// --------------------------------------------------------------------------- //
// TestMatching
// --------------------------------------------------------------------------- //

namespace {
struct Pair {
    Frame f0;
    Frame f1;
};
Pair shifted_pair() { return {orb().detect(image(), 0, 0), orb().detect(shifted_image(), 1, 1)}; }
}  // namespace

// py: test_features.py::TestMatching::test_matches_a_shifted_scene
TEST(TestMatching, MatchesAShiftedScene) {
    const Pair p = shifted_pair();
    const MatchResult result = FeatureMatcher(cv::NORM_HAMMING).match(p.f0, p.f1);
    EXPECT_GT(result.size(), 20u);
    // The scene moved 10 px to the left, so matched points should too.
    std::vector<double> dx(result.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        dx[i] = result.points_b(static_cast<Eigen::Index>(i), 0) - result.points_a(static_cast<Eigen::Index>(i), 0);
    }
    EXPECT_APPROX(np::median(dx), -10.0, std::nullopt, 1.5);
}

// py: test_features.py::TestMatching::test_identical_frames_match_at_zero_distance
TEST(TestMatching, IdenticalFramesMatchAtZeroDistance) {
    const Frame frame = orb().detect(image());
    const MatchResult result =
        FeatureMatcher(cv::NORM_HAMMING, true, 0.75, true, std::nullopt).match(frame, frame);
    EXPECT_GT(result.size(), 20u);
    EXPECT_APPROX(result.mean_distance(), 0.0);
    EXPECT_EQ(result.indices_a, result.indices_b);
}

// py: test_features.py::TestMatching::test_indices_address_the_source_frames
TEST(TestMatching, IndicesAddressTheSourceFrames) {
    const Pair p = shifted_pair();
    const MatchResult result = FeatureMatcher(cv::NORM_HAMMING).match(p.f0, p.f1);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto r = static_cast<Eigen::Index>(i);
        EXPECT_EQ(result.points_a.row(r), p.f0.points.row(result.indices_a[i]));
        EXPECT_EQ(result.points_b.row(r), p.f1.points.row(result.indices_b[i]));
    }
}

// py: test_features.py::TestMatching::test_ratio_test_tightens_the_match_set
TEST(TestMatching, RatioTestTightensTheMatchSet) {
    const Pair p = shifted_pair();
    const FeatureMatcher loose(cv::NORM_HAMMING, false, 0.75, false);
    const FeatureMatcher strict(cv::NORM_HAMMING, true, 0.6, false);
    EXPECT_LT(strict.match(p.f0, p.f1).size(), loose.match(p.f0, p.f1).size());
}

// py: test_features.py::TestMatching::test_cross_check_tightens_the_match_set
TEST(TestMatching, CrossCheckTightensTheMatchSet) {
    const Pair p = shifted_pair();
    const FeatureMatcher without(cv::NORM_HAMMING, false, 0.75, false);
    const FeatureMatcher with_cc(cv::NORM_HAMMING, false, 0.75, true);
    EXPECT_LT(with_cc.match(p.f0, p.f1).size(), without.match(p.f0, p.f1).size());
}

// py: test_features.py::TestMatching::test_cross_check_keeps_only_mutual_neighbours
TEST(TestMatching, CrossCheckKeepsOnlyMutualNeighbours) {
    // Every surviving match must also be the reverse nearest neighbour.
    const Pair p = shifted_pair();
    const MatchResult result =
        FeatureMatcher(cv::NORM_HAMMING, false, 0.75, true, std::nullopt).match(p.f0, p.f1);
    auto bf = cv::BFMatcher::create(cv::NORM_HAMMING, false);
    std::vector<cv::DMatch> reverse_matches;
    bf->match(p.f1.descriptors, p.f0.descriptors, reverse_matches);
    std::map<int, int> reverse;
    for (const auto& m : reverse_matches) {
        reverse[m.queryIdx] = m.trainIdx;
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_EQ(reverse.at(static_cast<int>(result.indices_b[i])), result.indices_a[i]);
    }
}

// py: test_features.py::TestMatching::test_distance_ceiling_is_enforced
TEST(TestMatching, DistanceCeilingIsEnforced) {
    const Pair p = shifted_pair();
    const MatchResult result = FeatureMatcher(cv::NORM_HAMMING, true, 0.75, true, 30.0).match(p.f0, p.f1);
    for (double d : result.distances) {
        EXPECT_LE(d, 30.0);
    }
}

// py: test_features.py::TestMatching::test_empty_frame_yields_no_matches
TEST(TestMatching, EmptyFrameYieldsNoMatches) {
    const Frame blank = orb().detect(cv::Mat::zeros(100, 100, CV_8U));
    const MatchResult result = FeatureMatcher(cv::NORM_HAMMING).match(orb().detect(image()), blank);
    EXPECT_EQ(result.size(), 0u);
    EXPECT_EQ(result.n_raw, 0);
}

// py: test_features.py::TestMatching::test_invalid_ratio_raises
TEST(TestMatching, InvalidRatioRaises) {
    expect_raises<ValueError>([] { FeatureMatcher(cv::NORM_HAMMING, true, 0.0); }, "ratio must be");
}

// py: test_features.py::TestMatching::test_unsupported_backend_raises
TEST(TestMatching, UnsupportedBackendRaises) {
    const Config config = Config().with_override("matcher.matcher", "flann");
    expect_raises<ValueError>([&] { FeatureMatcher::from_config(config); }, "Unsupported matcher");
}

// py: test_features.py::TestMatching::test_min_match_threshold
TEST(TestMatching, MinMatchThreshold) {
    const Pair p = shifted_pair();
    const FeatureMatcher matcher(cv::NORM_HAMMING, true, 0.75, true, 64.0, 10);
    EXPECT_TRUE(matcher.is_sufficient(matcher.match(p.f0, p.f1)));
    EXPECT_FALSE(matcher.is_sufficient(MatchResult::empty()));
}

// py: test_features.py::TestMatching::test_float_descriptors_are_coerced_for_l2
TEST(TestMatching, FloatDescriptorsAreCoercedForL2) {
    // SIFT descriptors arrive as float32; the matcher must not choke on float64.
    golden::Stream rng("test_features", "TestMatching::test_float_descriptors_are_coerced_for_l2");
    const cv::Mat desc = rng.next_cvmat("random");  // (30, 128) float64
    const Points2 pts = golden::points2(rng.next("random"), &rng.shared());
    ASSERT_EQ(desc.depth(), CV_64F);
    const FeatureMatcher matcher(cv::NORM_L2, false, 0.75, true, std::nullopt);
    EXPECT_EQ(matcher.match_descriptors(desc, desc, pts, pts).size(), 30u);
}

// --------------------------------------------------------------------------- //
// TestMatchResult
// --------------------------------------------------------------------------- //

namespace {
MatchResult result_fixture() {
    MatchResult r;
    r.points_a.resize(3, 2);
    r.points_a << 0.0, 0.0, 1.0, 1.0, 2.0, 2.0;
    r.points_b.resize(3, 2);
    r.points_b << 3.0, 0.0, 4.0, 1.0, 5.0, 2.0;
    r.indices_a = {0, 1, 2};
    r.indices_b = {5, 6, 7};
    r.distances = {10.0, 20.0, 30.0};
    r.n_raw = 6;
    return r;
}
}  // namespace

// py: test_features.py::TestMatchResult::test_length_and_survival
TEST(TestMatchResult, LengthAndSurvival) {
    const MatchResult result = result_fixture();
    EXPECT_EQ(result.size(), 3u);
    EXPECT_APPROX(result.survival_ratio(), 0.5);
}

// py: test_features.py::TestMatchResult::test_mean_statistics
TEST(TestMatchResult, MeanStatistics) {
    const MatchResult result = result_fixture();
    EXPECT_APPROX(result.mean_distance(), 20.0);
    EXPECT_APPROX(result.mean_pixel_displacement(), 3.0);
}

// py: test_features.py::TestMatchResult::test_select_boolean_mask
TEST(TestMatchResult, SelectBooleanMask) {
    const MatchResult subset = result_fixture().select(Mask{1, 0, 1});
    EXPECT_EQ(subset.size(), 2u);
    EXPECT_EQ(subset.indices_b, (std::vector<std::int64_t>{5, 7}));
    EXPECT_EQ(subset.n_raw, 6);
}

// py: test_features.py::TestMatchResult::test_select_index_array
TEST(TestMatchResult, SelectIndexArray) {
    EXPECT_EQ(result_fixture().select_indices({1}).indices_a, (std::vector<std::int64_t>{1}));
}

// py: test_features.py::TestMatchResult::test_select_rejects_wrong_length_mask
TEST(TestMatchResult, SelectRejectsWrongLengthMask) {
    expect_raises<ValueError>([] { result_fixture().select(Mask{1, 0}); }, "mask length");
}

// py: test_features.py::TestMatchResult::test_empty_result_statistics
TEST(TestMatchResult, EmptyResultStatistics) {
    const MatchResult empty = MatchResult::empty(5);
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_EQ(empty.survival_ratio(), 0.0);
    EXPECT_TRUE(std::isnan(empty.mean_distance()));
    EXPECT_TRUE(std::isnan(empty.mean_pixel_displacement()));
}

// --------------------------------------------------------------------------- //
// TestMatchVisualisation
// --------------------------------------------------------------------------- //

// py: test_features.py::TestMatchVisualisation::test_canvas_is_side_by_side
TEST(TestMatchVisualisation, CanvasIsSideBySide) {
    const Pair p = shifted_pair();
    const MatchResult result = FeatureMatcher(cv::NORM_HAMMING).match(p.f0, p.f1);
    const cv::Mat canvas = draw_matches(image(), p.f0, shifted_image(), p.f1, result);
    EXPECT_EQ(canvas.rows, image().rows);
    EXPECT_EQ(canvas.cols, image().cols * 2);
    EXPECT_EQ(canvas.channels(), 3);
}

// py: test_features.py::TestMatchVisualisation::test_inlier_mask_is_accepted
TEST(TestMatchVisualisation, InlierMaskIsAccepted) {
    const Pair p = shifted_pair();
    const MatchResult result = FeatureMatcher(cv::NORM_HAMMING).match(p.f0, p.f1);
    Mask mask(result.size(), 0);
    for (std::size_t i = 0; i < mask.size(); i += 2) {
        mask[i] = 1;
    }
    const cv::Mat canvas = draw_matches(image(), p.f0, shifted_image(), p.f1, result, 150, mask);
    EXPECT_EQ(canvas.channels(), 3);
}

// py: test_features.py::TestMatchVisualisation::test_no_matches_still_renders
TEST(TestMatchVisualisation, NoMatchesStillRenders) {
    const Frame frame = orb().detect(image());
    const cv::Mat canvas = draw_matches(image(), frame, image(), frame, MatchResult::empty());
    EXPECT_EQ(canvas.channels(), 3);
}

// --------------------------------------------------------------------------- //
// Golden: bit-level parity with cv2 (tests/golden/features.json)
// --------------------------------------------------------------------------- //

namespace {

const Json& features_golden() {
    static const std::string name = "features.json";
    return golden::load(name);
}

void expect_same_detection(const Frame& frame, const Json& expected) {
    const golden::Array kps = golden::decode(expected.at("keypoints"));
    const auto values = kps.as_doubles();
    ASSERT_EQ(static_cast<std::int64_t>(frame.size()), kps.shape[0]);
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const cv::KeyPoint& kp = frame.keypoints[i];
        const double* row = values.data() + i * 7;
        ASSERT_EQ(static_cast<double>(kp.pt.x), row[0]) << "keypoint " << i;
        ASSERT_EQ(static_cast<double>(kp.pt.y), row[1]) << "keypoint " << i;
        EXPECT_EQ(static_cast<double>(kp.size), row[2]);
        EXPECT_EQ(static_cast<double>(kp.angle), row[3]);
        EXPECT_EQ(static_cast<double>(kp.response), row[4]);
        EXPECT_EQ(static_cast<double>(kp.octave), row[5]);
    }
    if (expected.at("descriptors").is_null()) {
        EXPECT_TRUE(frame.descriptors.empty());
        return;
    }
    const cv::Mat descriptors = golden::cvmat(expected.at("descriptors"));
    ASSERT_EQ(descriptors.size(), frame.descriptors.size());
    EXPECT_EQ(cv::norm(descriptors, frame.descriptors, cv::NORM_INF), 0.0);
}

}  // namespace

TEST(FeaturesGolden, FixtureImageMatchesPython) {
    const cv::Mat img = image();
    EXPECT_EQ(kitti_fixture::md5_hex(img.data, img.total()),
              features_golden().at("image_md5").get<std::string>());
}

TEST(FeaturesGolden, DetectionsAreBitIdenticalToCv2) {
    if (!same_opencv_as_reference()) {
        GTEST_SKIP() << "OpenCV " << CV_VERSION << " differs from the reference cv2 version";
    }
    const std::map<std::string, FeatureDetector> detectors = {
        {"orb500", FeatureDetector("orb", 500)},
        {"orb40", FeatureDetector("orb", 40)},
        {"orb300_clahe", FeatureDetector("orb", 300, 20, 1.2, 8, 31, 0, true)},
        {"orb_fast_score", FeatureDetector("orb", 500, 20, 1.2, 8, 31, 1)},
        {"sift200", FeatureDetector("sift", 200)},
    };
    for (const auto& [key, expected] : features_golden().at("detections").items()) {
        const auto slash = key.find('/');
        const std::string name = key.substr(0, slash);
        const cv::Mat img = key.substr(slash + 1) == "image" ? image() : shifted_image();
        SCOPED_TRACE(key);
        expect_same_detection(detectors.at(name).detect(img), expected);
    }
    // Detection under a mask.
    cv::Mat mask = cv::Mat::zeros(image().size(), CV_8U);
    mask(cv::Rect(0, 0, 100, mask.rows)).setTo(255);
    const Frame masked = orb().detect(image(), 0, std::nullopt, mask);
    Json wrapper = Json::object();
    wrapper["keypoints"] = features_golden().at("masked");
    wrapper["descriptors"] = nullptr;
    const golden::Array kps = golden::decode(wrapper["keypoints"]);
    EXPECT_EQ(static_cast<std::int64_t>(masked.size()), kps.shape[0]);
}

TEST(FeaturesGolden, MatcherOutputsAreIdenticalToPython) {
    if (!same_opencv_as_reference()) {
        GTEST_SKIP() << "OpenCV " << CV_VERSION << " differs from the reference cv2 version";
    }
    const Pair p = shifted_pair();
    const std::map<std::string, FeatureMatcher> matchers = {
        {"default", FeatureMatcher()},
        {"no_ceiling", FeatureMatcher(cv::NORM_HAMMING, true, 0.75, true, std::nullopt)},
        {"ratio0.6_nocc", FeatureMatcher(cv::NORM_HAMMING, true, 0.6, false)},
        {"loose", FeatureMatcher(cv::NORM_HAMMING, false, 0.75, false)},
        {"cc_only", FeatureMatcher(cv::NORM_HAMMING, false, 0.75, true)},
        {"ceiling30", FeatureMatcher(cv::NORM_HAMMING, true, 0.75, true, 30.0)},
    };
    for (const auto& [name, expected] : features_golden().at("matches").items()) {
        SCOPED_TRACE(name);
        const MatchResult result = matchers.at(name).match(p.f0, p.f1);
        EXPECT_EQ(result.indices_a, golden::ivec(expected.at("indices_a")));
        EXPECT_EQ(result.indices_b, golden::ivec(expected.at("indices_b")));
        EXPECT_EQ(result.distances, golden::vec(expected.at("distances")));
        EXPECT_EQ(result.n_raw, expected.at("n_raw").get<std::int64_t>());
    }
}
