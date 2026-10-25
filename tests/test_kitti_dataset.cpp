// Tests for KITTI calibration parsing and the odometry dataset loader
// (mirrors tests/test_kitti_dataset.py). All tests run against the synthetic
// tree built by kitti_fixture so no dataset download is required.
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/datasets/calibration.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/errors.hpp"
#include "support/golden.hpp"
#include "support/kitti_fixture.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using testing_utils::TempDir;

namespace {

Eigen::MatrixXd M(const Eigen::MatrixXd& m) { return m; }

template <typename Exc, typename Fn>
void expect_raises(Fn&& fn, const std::string& fragment) {
    try {
        fn();
        ADD_FAILURE() << "expected exception matching '" << fragment << "'";
    } catch (const Exc& exc) {
        EXPECT_NE(std::string(exc.what()).find(fragment), std::string::npos)
            << "message was: " << exc.what();
    }
}

/// The ``kitti_root`` fixture: a 12-frame synthetic sequence 00 with ground truth.
class KittiRoot : public ::testing::Test {
protected:
    void SetUp() override { root = kitti_fixture::write_kitti_sequence(tmp / "dataset", "00", 12); }
    TempDir tmp;
    std::filesystem::path root;
};

/// The ``calib_file`` fixture.
class CalibFile : public ::testing::Test {
protected:
    void SetUp() override {
        path = tmp / "calib.txt";
        testing_utils::write_text(path, kitti_fixture::kCalibText);
    }
    TempDir tmp;
    std::filesystem::path path;
};

}  // namespace

// --------------------------------------------------------------------------- //
// TestCalibParsing
// --------------------------------------------------------------------------- //

using TestCalibParsing = CalibFile;

// py: test_kitti_dataset.py::TestCalibParsing::test_parses_all_projection_matrices
TEST_F(TestCalibParsing, ParsesAllProjectionMatrices) {
    const auto calib = parse_kitti_calib(path);
    std::vector<std::string> keys;
    for (const auto& [k, _] : calib) {
        keys.push_back(k);
    }
    EXPECT_EQ(keys, (std::vector<std::string>{"P0", "P1", "P2", "P3", "Tr"}));
    for (const char* key : {"P0", "P1", "P2", "P3"}) {
        EXPECT_EQ(calib.at(key).rows(), 3);
        EXPECT_EQ(calib.at(key).cols(), 4);
    }
}

// py: test_kitti_dataset.py::TestCalibParsing::test_tr_is_promoted_to_4x4
TEST_F(TestCalibParsing, TrIsPromotedTo4x4) {
    const Eigen::MatrixXd Tr = parse_kitti_calib(path).at("Tr");
    EXPECT_EQ(Tr.rows(), 4);
    EXPECT_EQ(Tr.cols(), 4);
    EXPECT_ALLCLOSE(M(Tr.row(3)), M(Eigen::RowVector4d(0, 0, 0, 1)));
}

// py: test_kitti_dataset.py::TestCalibParsing::test_intrinsics_match_kitti_sequence_00
TEST_F(TestCalibParsing, IntrinsicsMatchKittiSequence00) {
    const auto cam = calibration_for_camera(parse_kitti_calib(path), "image_0");
    EXPECT_APPROX(cam.fx(), kitti_fixture::kKitti00Fx);
    EXPECT_APPROX(cam.fy(), kitti_fixture::kKitti00Fx);
    EXPECT_APPROX(cam.cx(), kitti_fixture::kKitti00Cx);
    EXPECT_APPROX(cam.cy(), kitti_fixture::kKitti00Cy);
}

// py: test_kitti_dataset.py::TestCalibParsing::test_K_is_upper_triangular_with_unit_scale
TEST_F(TestCalibParsing, KIsUpperTriangularWithUnitScale) {
    const Eigen::Matrix3d K = calibration_for_camera(parse_kitti_calib(path), "image_0").K();
    EXPECT_EQ(K(1, 0), 0.0);
    EXPECT_EQ(K(2, 0), 0.0);
    EXPECT_EQ(K(2, 1), 0.0);
    EXPECT_APPROX(K(2, 2), 1.0);
}

// py: test_kitti_dataset.py::TestCalibParsing::test_stereo_baseline_matches_kitti
TEST_F(TestCalibParsing, StereoBaselineMatchesKitti) {
    // P1's fourth column encodes -fx * baseline; KITTI's is ~0.54 m.
    const auto calib = parse_kitti_calib(path);
    EXPECT_APPROX(calibration_for_camera(calib, "image_0").baseline_m(), 0.0);
    EXPECT_APPROX(calibration_for_camera(calib, "image_1").baseline_m(), 0.537, std::nullopt, 1e-3);
}

// py: test_kitti_dataset.py::TestCalibParsing::test_missing_file_raises
TEST_F(TestCalibParsing, MissingFileRaises) {
    expect_raises<CalibrationError>([&] { parse_kitti_calib(tmp / "nope.txt"); }, "not found");
}

// py: test_kitti_dataset.py::TestCalibParsing::test_wrong_value_count_raises
TEST_F(TestCalibParsing, WrongValueCountRaises) {
    testing_utils::write_text(path, "P0: 1 2 3\n");
    expect_raises<CalibrationError>([&] { parse_kitti_calib(path); }, "expected 12");
}

// py: test_kitti_dataset.py::TestCalibParsing::test_missing_colon_raises
TEST_F(TestCalibParsing, MissingColonRaises) {
    testing_utils::write_text(path, "P0 1 2 3\n");
    expect_raises<CalibrationError>([&] { parse_kitti_calib(path); }, "expected 'key: values'");
}

// py: test_kitti_dataset.py::TestCalibParsing::test_non_finite_values_raise
TEST_F(TestCalibParsing, NonFiniteValuesRaise) {
    std::string line = "P0:";
    for (int i = 0; i < 12; ++i) {
        line += " nan";
    }
    testing_utils::write_text(path, line + "\n");
    expect_raises<CalibrationError>([&] { parse_kitti_calib(path); }, "non-finite");
}

// py: test_kitti_dataset.py::TestCalibParsing::test_blank_lines_ignored
TEST_F(TestCalibParsing, BlankLinesIgnored) {
    testing_utils::write_text(path, std::string("\n# comment\n") + kitti_fixture::kCalibText + "\n");
    EXPECT_TRUE(parse_kitti_calib(path).count("P0"));
}

// py: test_kitti_dataset.py::TestCalibParsing::test_unknown_camera_raises
TEST_F(TestCalibParsing, UnknownCameraRaises) {
    expect_raises<CalibrationError>(
        [&] { calibration_for_camera(parse_kitti_calib(path), "image_9"); }, "Unknown camera");
}

// --------------------------------------------------------------------------- //
// TestCameraModel
// --------------------------------------------------------------------------- //

namespace {
CameraCalibration test_camera() {
    Eigen::Matrix3d K;
    K << 500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0;
    Eigen::Matrix<double, 3, 4> P = Eigen::Matrix<double, 3, 4>::Zero();
    P.leftCols<3>() = K;
    return CameraCalibration(K, P);
}
}  // namespace

// py: test_kitti_dataset.py::TestCameraModel::test_normalize_points_centres_principal_point
TEST(TestCameraModel, NormalizePointsCentresPrincipalPoint) {
    Eigen::MatrixXd pts(2, 2);
    pts << 320.0, 240.0, 820.0, 240.0;
    const Points2 normalized = test_camera().normalize_points(pts);
    EXPECT_ALLCLOSE(M(normalized.row(0)), M(Eigen::RowVector2d(0.0, 0.0)));
    EXPECT_ALLCLOSE(M(normalized.row(1)), M(Eigen::RowVector2d(1.0, 0.0)));
}

// py: test_kitti_dataset.py::TestCameraModel::test_project_round_trips_with_normalize
TEST(TestCameraModel, ProjectRoundTripsWithNormalize) {
    golden::Stream rng("test_kitti_dataset", "TestCameraModel::test_project_round_trips_with_normalize");
    const auto x = rng.next_vec("normal");
    const auto y = rng.next_vec("normal");
    const auto z = rng.next_vec("uniform");
    Eigen::MatrixXd pts_cam(20, 3);
    for (int i = 0; i < 20; ++i) {
        pts_cam.row(i) << x[static_cast<std::size_t>(i)], y[static_cast<std::size_t>(i)],
            z[static_cast<std::size_t>(i)];
    }
    const CameraCalibration cam = test_camera();
    const Points2 normalized = cam.normalize_points(cam.project(pts_cam));
    Eigen::MatrixXd expected(20, 2);
    for (int i = 0; i < 20; ++i) {
        expected.row(i) << pts_cam(i, 0) / pts_cam(i, 2), pts_cam(i, 1) / pts_cam(i, 2);
    }
    EXPECT_ALLCLOSE(M(normalized), expected, 1e-7, 1e-9);
}

// py: test_kitti_dataset.py::TestCameraModel::test_points_behind_camera_become_nan
TEST(TestCameraModel, PointsBehindCameraBecomeNan) {
    Eigen::MatrixXd pts(2, 3);
    pts << 1.0, 1.0, -5.0, 1.0, 1.0, 5.0;
    const Points2 uv = test_camera().project(pts);
    EXPECT_TRUE(std::isnan(uv(0, 0)) && std::isnan(uv(0, 1)));
    EXPECT_TRUE(std::isfinite(uv(1, 0)) && std::isfinite(uv(1, 1)));
}

// py: test_kitti_dataset.py::TestCameraModel::test_invalid_shapes_rejected
TEST(TestCameraModel, InvalidShapesRejected) {
    EXPECT_THROW(CameraCalibration::from_matrices(Eigen::MatrixXd::Identity(2, 2),
                                                  Eigen::MatrixXd::Zero(3, 4)),
                 CalibrationError);
    expect_raises<CalibrationError>(
        [] {
            CameraCalibration::from_matrices(Eigen::Vector3d(-1.0, 1.0, 1.0).asDiagonal().toDenseMatrix(),
                                             Eigen::MatrixXd::Zero(3, 4));
        },
        "Focal lengths");
    EXPECT_THROW(test_camera().project(Eigen::MatrixXd::Zero(4, 2)), ValueError);
}

// --------------------------------------------------------------------------- //
// TestFrameSelection
// --------------------------------------------------------------------------- //

using Ids = std::vector<std::int64_t>;

// py: test_kitti_dataset.py::TestFrameSelection::test_default_selects_everything
TEST(TestFrameSelection, DefaultSelectsEverything) {
    EXPECT_EQ(FrameSelection().apply(5), (Ids{0, 1, 2, 3, 4}));
}

// py: test_kitti_dataset.py::TestFrameSelection::test_step_subsamples
TEST(TestFrameSelection, StepSubsamples) {
    EXPECT_EQ((FrameSelection{0, 3, std::nullopt}.apply(10)), (Ids{0, 3, 6, 9}));
}

// py: test_kitti_dataset.py::TestFrameSelection::test_start_offsets
TEST(TestFrameSelection, StartOffsets) {
    EXPECT_EQ((FrameSelection{2, 2, std::nullopt}.apply(8)), (Ids{2, 4, 6}));
}

// py: test_kitti_dataset.py::TestFrameSelection::test_max_frames_truncates
TEST(TestFrameSelection, MaxFramesTruncates) {
    EXPECT_EQ((FrameSelection{0, 1, 4}.apply(100)).size(), 4u);
}

// py: test_kitti_dataset.py::TestFrameSelection::test_max_frames_applies_after_step
TEST(TestFrameSelection, MaxFramesAppliesAfterStep) {
    EXPECT_EQ((FrameSelection{0, 5, 3}.apply(100)), (Ids{0, 5, 10}));
}

// py: test_kitti_dataset.py::TestFrameSelection::test_invalid_selection_raises[kwargs0]
// py: test_kitti_dataset.py::TestFrameSelection::test_invalid_selection_raises[kwargs1]
// py: test_kitti_dataset.py::TestFrameSelection::test_invalid_selection_raises[kwargs2]
TEST(TestFrameSelection, InvalidSelectionRaises) {
    EXPECT_THROW((FrameSelection{0, 0, std::nullopt}.apply(10)), ValueError);   // step=0
    EXPECT_THROW((FrameSelection{-1, 1, std::nullopt}.apply(10)), ValueError);  // start=-1
    EXPECT_THROW((FrameSelection{99, 1, std::nullopt}.apply(10)), ValueError);  // start=99
}

// --------------------------------------------------------------------------- //
// TestDatasetLoader
// --------------------------------------------------------------------------- //

using TestDatasetLoader = KittiRoot;

// py: test_kitti_dataset.py::TestDatasetLoader::test_loads_all_frames
TEST_F(TestDatasetLoader, LoadsAllFrames) {
    const KittiOdometryDataset ds(root, "00");
    EXPECT_EQ(ds.size(), 12u);
    EXPECT_EQ(ds.frame_ids(), (Ids{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_sequence_id_is_normalised
TEST_F(TestDatasetLoader, SequenceIdIsNormalised) {
    EXPECT_EQ(KittiOdometryDataset(root, 0).sequence(), "00");
    EXPECT_EQ(KittiOdometryDataset(root, "0").sequence(), "00");
    EXPECT_EQ(KittiOdometryDataset(root, "00").sequence(), "00");
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_images_are_grayscale_uint8
TEST_F(TestDatasetLoader, ImagesAreGrayscaleUint8) {
    const cv::Mat image = KittiOdometryDataset(root, "00").load_image(0);
    EXPECT_EQ(image.channels(), 1);
    EXPECT_EQ(image.depth(), CV_8U);
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_colour_mode_returns_three_channels
TEST_F(TestDatasetLoader, ColourModeReturnsThreeChannels) {
    const cv::Mat image = KittiOdometryDataset(root, "00", "image_0", false).load_image(0);
    EXPECT_EQ(image.channels(), 3);
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_image_size_property
TEST_F(TestDatasetLoader, ImageSizeProperty) {
    EXPECT_EQ(KittiOdometryDataset(root, "00").image_size(), std::make_pair(320, 120));
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_intrinsics_exposed
TEST_F(TestDatasetLoader, IntrinsicsExposed) {
    const KittiOdometryDataset ds(root, "00");
    EXPECT_EQ(ds.K().rows(), 3);
    EXPECT_APPROX(ds.calibration().fx(), kitti_fixture::kKitti00Fx);
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_timestamps_align_with_frames
TEST_F(TestDatasetLoader, TimestampsAlignWithFrames) {
    const KittiOdometryDataset ds(root, "00", "image_0", true, 0, 4);
    ASSERT_TRUE(ds.timestamps().has_value());
    EXPECT_EQ(ds.timestamps()->size(), ds.size());
    EXPECT_NEAR((*ds.timestamps())[1] - (*ds.timestamps())[0], 4 * 0.1037, 1e-6);
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_missing_times_file_is_tolerated
TEST_F(TestDatasetLoader, MissingTimesFileIsTolerated) {
    const auto d = kitti_fixture::write_kitti_sequence(tmp / "d", "00", 5, true, false);
    EXPECT_FALSE(KittiOdometryDataset(d, "00").timestamps().has_value());
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_subsampling_selects_matching_frames
TEST_F(TestDatasetLoader, SubsamplingSelectsMatchingFrames) {
    const KittiOdometryDataset ds(root, "00", "image_0", true, 2, 3, 3);
    EXPECT_EQ(ds.frame_ids(), (Ids{2, 5, 8}));
    EXPECT_EQ(ds.image_paths().size(), 3u);
    EXPECT_EQ(ds.image_paths()[0].filename().string(), "000002.png");
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_iteration_yields_every_selected_frame
TEST_F(TestDatasetLoader, IterationYieldsEverySelectedFrame) {
    const KittiOdometryDataset ds(root, "00", "image_0", true, 0, 1, 4);
    std::size_t count = 0;
    for (std::size_t i = 0; i < ds.size(); ++i) {
        count += ds.load_image(i).empty() ? 0 : 1;
    }
    EXPECT_EQ(count, 4u);
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_missing_sequence_raises_with_hint
TEST_F(TestDatasetLoader, MissingSequenceRaisesWithHint) {
    expect_raises<DatasetError>([&] { KittiOdometryDataset(root, "07"); }, "Available sequences");
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_missing_camera_raises
TEST_F(TestDatasetLoader, MissingCameraRaises) {
    expect_raises<DatasetError>([&] { KittiOdometryDataset(root, "00", "image_3"); },
                                "Image directory not found");
}

// py: test_kitti_dataset.py::TestDatasetLoader::test_describe_reports_real_values
TEST_F(TestDatasetLoader, DescribeReportsRealValues) {
    const Json info = KittiOdometryDataset(root, "00").describe();
    EXPECT_EQ(info["sequence"], "00");
    EXPECT_EQ(info["frames_selected"], 12);
    EXPECT_EQ(info["has_ground_truth"], true);
    EXPECT_GT(info["ground_truth_path_length_m"].get<double>(), 0);
}

// --------------------------------------------------------------------------- //
// TestGroundTruth
// --------------------------------------------------------------------------- //

using TestGroundTruth = KittiRoot;

// py: test_kitti_dataset.py::TestGroundTruth::test_ground_truth_matches_frame_count
TEST_F(TestGroundTruth, GroundTruthMatchesFrameCount) {
    const KittiOdometryDataset ds(root, "00");
    const auto& gt = ds.ground_truth();
    ASSERT_TRUE(gt.has_value());
    EXPECT_EQ(gt->size(), 12u);
    EXPECT_TRUE(gt->is_valid());
}

// py: test_kitti_dataset.py::TestGroundTruth::test_ground_truth_starts_at_identity
TEST_F(TestGroundTruth, GroundTruthStartsAtIdentity) {
    const KittiOdometryDataset ds(root, "00");
    EXPECT_ALLCLOSE(M((*ds.ground_truth())[0]), M(Eigen::Matrix4d::Identity()), 1e-7, 1e-9);
}

// py: test_kitti_dataset.py::TestGroundTruth::test_ground_truth_is_subsampled_with_frames
TEST_F(TestGroundTruth, GroundTruthIsSubsampledWithFrames) {
    const KittiOdometryDataset ds(root, "00", "image_0", true, 0, 2);
    const auto& gt = ds.ground_truth();
    EXPECT_EQ(gt->size(), ds.size());
    EXPECT_EQ(gt->frame_ids(), ds.frame_ids());
}

// py: test_kitti_dataset.py::TestGroundTruth::test_scales_are_positive_and_match_step
TEST_F(TestGroundTruth, ScalesArePositiveAndMatchStep) {
    const auto scales = KittiOdometryDataset(root, "00").ground_truth_scales();
    ASSERT_TRUE(scales.has_value());
    EXPECT_EQ(scales->size(), 11u);
    EXPECT_ALLCLOSE(*scales, std::vector<double>(11, 0.9), 1e-7, 1e-6);
}

// py: test_kitti_dataset.py::TestGroundTruth::test_subsampled_scales_grow_with_step
TEST_F(TestGroundTruth, SubsampledScalesGrowWithStep) {
    // Skipping frames means each transition covers more ground.
    const auto scales = KittiOdometryDataset(root, "00", "image_0", true, 0, 3).ground_truth_scales();
    for (double s : *scales) {
        EXPECT_GT(s, 2.0);
    }
}

// py: test_kitti_dataset.py::TestGroundTruth::test_missing_ground_truth_returns_none
TEST_F(TestGroundTruth, MissingGroundTruthReturnsNone) {
    const auto d = kitti_fixture::write_kitti_sequence(tmp / "d", "00", 5, false);
    const KittiOdometryDataset ds(d, "00");
    EXPECT_FALSE(ds.has_ground_truth());
    EXPECT_FALSE(ds.ground_truth().has_value());
    EXPECT_FALSE(ds.ground_truth_scales().has_value());
}

// py: test_kitti_dataset.py::TestGroundTruth::test_truncated_pose_file_raises
TEST_F(TestGroundTruth, TruncatedPoseFileRaises) {
    std::string text;
    const double zeros[12] = {};
    for (int i = 0; i < 3; ++i) {
        text += py::savetxt_row(zeros, 12, "%.6e");
    }
    testing_utils::write_text(root / "poses" / "00.txt", text);
    expect_raises<DatasetError>([&] { (void)KittiOdometryDataset(root, "00").ground_truth(); },
                                "poses but the sequence has");
}

// py: test_kitti_dataset.py::TestGroundTruth::test_non_se3_pose_file_raises
TEST_F(TestGroundTruth, NonSe3PoseFileRaises) {
    std::string text;
    double row[12];
    for (int i = 0; i < 12; ++i) {
        row[i] = i;  // rotation block is not orthonormal
    }
    for (int i = 0; i < 12; ++i) {
        text += py::savetxt_row(row, 12, "%.6e");
    }
    testing_utils::write_text(root / "poses" / "00.txt", text);
    expect_raises<DatasetError>([&] { (void)KittiOdometryDataset(root, "00").ground_truth(); },
                                "not valid SE");
}

// --------------------------------------------------------------------------- //
// TestConfigIntegration
// --------------------------------------------------------------------------- //

using TestConfigIntegration = KittiRoot;

// py: test_kitti_dataset.py::TestConfigIntegration::test_from_config_round_trip
TEST_F(TestConfigIntegration, FromConfigRoundTrip) {
    Json overrides = Json::object();
    overrides["dataset.path"] = root.string();
    overrides["dataset.sequence"] = "00";
    overrides["dataset.max_frames"] = 5;
    const KittiOdometryDataset ds = KittiOdometryDataset::from_config(Config().with_overrides(overrides));
    EXPECT_EQ(ds.size(), 5u);
    EXPECT_EQ(ds.sequence(), "00");
}

// --------------------------------------------------------------------------- //
// Golden: the fixture renders pixel-identical images to the Python helper
// --------------------------------------------------------------------------- //

TEST(KittiFixtureGolden, TexturedImagesArePixelIdentical) {
    const std::string fixture_name = "textured_images.json";
    const Json& fixture = golden::load(fixture_name);
    const Json& renders = fixture.at("renders");
    int checked = 0;
    for (const auto& item : renders) {
        const cv::Mat image = kitti_fixture::make_textured_image(
            item.at(0).get<int>(), item.at(1).get<int>(), item.at(2).get<int>(), item.at(3).get<int>());
        ASSERT_TRUE(image.isContinuous());
        EXPECT_EQ(kitti_fixture::md5_hex(image.data, image.total()), item.at(4).get<std::string>())
            << "index=" << item.at(0) << " size=" << item.at(1) << "x" << item.at(2)
            << " seed=" << item.at(3);
        ++checked;
    }
    EXPECT_GT(checked, 40);
}

TEST(KittiFixtureGolden, Md5OfKnownInput) {
    const std::string text = "The quick brown fox jumps over the lazy dog";
    EXPECT_EQ(kitti_fixture::md5_hex(reinterpret_cast<const unsigned char*>(text.data()), text.size()),
              "9e107d9d372bb6826bd81d3542a419d6");
}
