// Tests for trajectory error metrics (mirrors tests/test_metrics.py), plus
// golden checks against the NumPy implementation on noisy trajectories.
//
// Errors are injected analytically so the expected metric value is known in
// closed form, which is the only way to catch a formula that is merely
// plausible.
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/evaluation/metrics.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "support/golden.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;

namespace {

constexpr const char* kModule = "test_metrics";

/// A 40-frame curving drive, ~35 m of travel.
Trajectory reference_fixture() { return Trajectory(synthetic::straight_drive(40, 0.9, 0.8)); }

/// Long enough (~450 m) that the 100-400 m drift segments exist.
Trajectory long_reference_fixture() { return Trajectory(synthetic::straight_drive(500, 0.9, 0.2)); }

/// Shift every camera centre by a constant vector.
Trajectory offset_trajectory(const Trajectory& trajectory, const Eigen::Vector3d& offset) {
    PoseList poses = trajectory.poses();
    for (auto& T : poses) {
        T.topRightCorner<3, 1>() += offset;
    }
    return Trajectory(poses, trajectory.frame_ids());
}

/// A trajectory moving 1 m per step along +z.
Trajectory unit_steps(int n) {
    PoseList poses(static_cast<std::size_t>(n), Eigen::Matrix4d::Identity());
    for (int i = 0; i < n; ++i) {
        poses[static_cast<std::size_t>(i)](2, 3) = i * 1.0;
    }
    return Trajectory(poses);
}

Trajectory with_position_noise(const Trajectory& reference, const Eigen::MatrixXd& noise) {
    PoseList poses = reference.poses();
    for (std::size_t i = 0; i < poses.size(); ++i) {
        poses[i].topRightCorner<3, 1>() += noise.row(static_cast<Eigen::Index>(i)).transpose();
    }
    return Trajectory(poses);
}

template <typename Fn>
void expect_value_error(Fn&& fn, const std::string& fragment = "") {
    try {
        fn();
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find(fragment), std::string::npos) << exc.what();
    }
}

}  // namespace

// --------------------------------------------------------------------------- //
// TestATE
// --------------------------------------------------------------------------- //

// py: test_metrics.py::TestATE::test_identical_trajectories_have_zero_error
TEST(TestATE, IdenticalTrajectoriesHaveZeroError) {
    const Trajectory reference = reference_fixture();
    const auto result = absolute_trajectory_error(reference, reference, "none");
    EXPECT_APPROX(result.rmse, 0.0, std::nullopt, 1e-12);
    EXPECT_APPROX(result.max, 0.0, std::nullopt, 1e-12);
    EXPECT_EQ(result.n, static_cast<std::int64_t>(reference.size()));
}

// py: test_metrics.py::TestATE::test_constant_offset_gives_exactly_that_error
TEST(TestATE, ConstantOffsetGivesExactlyThatError) {
    // A rigid 3-4-0 offset must produce exactly 5 m of unaligned error.
    const Trajectory reference = reference_fixture();
    const auto result = absolute_trajectory_error(
        offset_trajectory(reference, Eigen::Vector3d(3.0, 4.0, 0.0)), reference, "none");
    EXPECT_APPROX(result.rmse, 5.0);
    EXPECT_APPROX(result.mean, 5.0);
    EXPECT_APPROX(result.median, 5.0);
    EXPECT_APPROX(result.std, 0.0, std::nullopt, 1e-9);
}

// py: test_metrics.py::TestATE::test_alignment_removes_a_rigid_offset
TEST(TestATE, AlignmentRemovesARigidOffset) {
    const Trajectory reference = reference_fixture();
    const Trajectory shifted = offset_trajectory(reference, Eigen::Vector3d(3.0, 4.0, 0.0));
    EXPECT_LT(absolute_trajectory_error(shifted, reference, "se3").rmse, 1e-9);
}

// py: test_metrics.py::TestATE::test_sim3_alignment_removes_a_scale_error
TEST(TestATE, Sim3AlignmentRemovesAScaleError) {
    const Trajectory reference = reference_fixture();
    const Trajectory scaled = reference.transformed(Eigen::Matrix4d::Identity(), 0.4);
    EXPECT_LT(absolute_trajectory_error(scaled, reference, "sim3").rmse, 1e-9);
    // Rigid alignment cannot: the shape is right but the size is wrong.
    EXPECT_GT(absolute_trajectory_error(scaled, reference, "se3").rmse, 1.0);
}

// py: test_metrics.py::TestATE::test_alignment_scale_is_reported
TEST(TestATE, AlignmentScaleIsReported) {
    const Trajectory reference = reference_fixture();
    const Trajectory scaled = reference.transformed(Eigen::Matrix4d::Identity(), 0.4);
    const auto result = absolute_trajectory_error(scaled, reference, "sim3");
    EXPECT_EQ(result.alignment, "sim3");
    EXPECT_APPROX(result.alignment_scale, 2.5, 1e-6);
}

// py: test_metrics.py::TestATE::test_rmse_identity_holds
TEST(TestATE, RmseIdentityHolds) {
    // RMSE^2 == mean^2 + std^2 for the error magnitudes.
    golden::Stream rng(kModule, "TestATE::test_rmse_identity_holds");
    const Trajectory reference = reference_fixture();
    const auto result = absolute_trajectory_error(with_position_noise(reference, rng.next_mat("normal")),
                                                  reference, "none");
    EXPECT_APPROX(result.rmse * result.rmse, result.mean * result.mean + result.std * result.std, 1e-9);
}

// py: test_metrics.py::TestATE::test_statistics_are_ordered_sensibly
TEST(TestATE, StatisticsAreOrderedSensibly) {
    golden::Stream rng(kModule, "TestATE::test_statistics_are_ordered_sensibly");
    const Trajectory reference = reference_fixture();
    const auto result = absolute_trajectory_error(with_position_noise(reference, rng.next_mat("normal")),
                                                  reference, "none");
    EXPECT_LE(result.min, result.median);
    EXPECT_LE(result.median, result.max);
    EXPECT_LE(result.mean, result.rmse);  // by Jensen's inequality
}

// py: test_metrics.py::TestATE::test_length_mismatch_raises
TEST(TestATE, LengthMismatchRaises) {
    const Trajectory reference = reference_fixture();
    expect_value_error([&] { absolute_trajectory_error(reference.slice(0, 10), reference); }, "same length");
}

// py: test_metrics.py::TestATE::test_empty_trajectory_raises
TEST(TestATE, EmptyTrajectoryRaises) {
    const Trajectory empty{PoseList{}};
    expect_value_error([&] { absolute_trajectory_error(empty, empty); });
}

// py: test_metrics.py::TestATE::test_to_dict_excludes_raw_errors
TEST(TestATE, ToDictExcludesRawErrors) {
    const Trajectory reference = reference_fixture();
    const Json data = absolute_trajectory_error(reference, reference, "none").to_dict();
    EXPECT_FALSE(data.contains("errors"));
    for (const char* key : {"rmse", "mean", "median", "std", "min", "max", "n"}) {
        EXPECT_TRUE(data.contains(key)) << key;
    }
}

// --------------------------------------------------------------------------- //
// TestRPE
// --------------------------------------------------------------------------- //

// py: test_metrics.py::TestRPE::test_identical_trajectories_have_zero_error
TEST(TestRPE, IdenticalTrajectoriesHaveZeroError) {
    const Trajectory reference = reference_fixture();
    const auto result = relative_pose_error(reference, reference, 1);
    EXPECT_APPROX(result.trans_rmse, 0.0, std::nullopt, 1e-12);
    EXPECT_APPROX(result.rot_rmse_deg, 0.0, std::nullopt, 1e-9);
    EXPECT_EQ(result.n, static_cast<std::int64_t>(reference.size()) - 1);
}

// py: test_metrics.py::TestRPE::test_rpe_ignores_a_global_offset
TEST(TestRPE, RpeIgnoresAGlobalOffset) {
    // This invariance is the whole point of RPE.
    const Trajectory reference = reference_fixture();
    const auto result = relative_pose_error(
        offset_trajectory(reference, Eigen::Vector3d(50.0, -20.0, 7.0)), reference, 1);
    EXPECT_APPROX(result.trans_rmse, 0.0, std::nullopt, 1e-9);
}

// py: test_metrics.py::TestRPE::test_rpe_ignores_a_global_rotation
TEST(TestRPE, RpeIgnoresAGlobalRotation) {
    const Trajectory reference = reference_fixture();
    const Eigen::Matrix4d T = se3_from_rt(rotation::from_euler("y", 63.0), Eigen::Vector3d::Zero());
    const auto result = relative_pose_error(reference.transformed(T), reference, 1);
    EXPECT_APPROX(result.trans_rmse, 0.0, std::nullopt, 1e-9);
    EXPECT_APPROX(result.rot_rmse_deg, 0.0, std::nullopt, 1e-8);
}

// py: test_metrics.py::TestRPE::test_scale_error_produces_known_translation_error
TEST(TestRPE, ScaleErrorProducesKnownTranslationError) {
    // Halving each 1 m step must give exactly 0.5 m of RPE at delta=1.
    const Trajectory reference = unit_steps(11);
    const Trajectory halved = reference.transformed(Eigen::Matrix4d::Identity(), 0.5);
    EXPECT_APPROX(relative_pose_error(halved, reference, 1).trans_rmse, 0.5, 1e-9);
}

// py: test_metrics.py::TestRPE::test_scale_argument_compensates
TEST(TestRPE, ScaleArgumentCompensates) {
    const Trajectory reference = unit_steps(11);
    const Trajectory halved = reference.transformed(Eigen::Matrix4d::Identity(), 0.5);
    EXPECT_APPROX(relative_pose_error(halved, reference, 1, 2.0).trans_rmse, 0.0, std::nullopt, 1e-9);
}

// py: test_metrics.py::TestRPE::test_rotation_error_is_measured_in_degrees
TEST(TestRPE, RotationErrorIsMeasuredInDegrees) {
    // A 5 deg per-step yaw error must show up as exactly 5 deg of RPE.
    const int n = 12;
    const Trajectory reference(PoseList(n, Eigen::Matrix4d::Identity()));
    PoseList poses{Eigen::Matrix4d::Identity()};
    const Eigen::Matrix4d step = se3_from_rt(rotation::from_euler("y", 5.0), Eigen::Vector3d::Zero());
    for (int i = 0; i < n - 1; ++i) {
        poses.push_back(poses.back() * step);
    }
    const auto result = relative_pose_error(Trajectory(poses), reference, 1);
    EXPECT_APPROX(result.rot_rmse_deg, 5.0, std::nullopt, 1e-6);
    EXPECT_APPROX(result.rot_median_deg, 5.0, std::nullopt, 1e-6);
}

// py: test_metrics.py::TestRPE::test_larger_delta_reduces_pair_count
TEST(TestRPE, LargerDeltaReducesPairCount) {
    const Trajectory reference = reference_fixture();
    EXPECT_EQ(relative_pose_error(reference, reference, 5).n, static_cast<std::int64_t>(reference.size()) - 5);
}

// py: test_metrics.py::TestRPE::test_invalid_delta_raises
TEST(TestRPE, InvalidDeltaRaises) {
    const Trajectory reference = reference_fixture();
    expect_value_error([&] { relative_pose_error(reference, reference, 0); }, "delta must be");
    expect_value_error(
        [&] { relative_pose_error(reference, reference, static_cast<std::int64_t>(reference.size())); },
        "exceeds");
}

// --------------------------------------------------------------------------- //
// TestKittiDrift
// --------------------------------------------------------------------------- //

// py: test_metrics.py::TestKittiDrift::test_perfect_trajectory_has_zero_drift
TEST(TestKittiDrift, PerfectTrajectoryHasZeroDrift) {
    const Trajectory long_reference = long_reference_fixture();
    const auto result = kitti_drift(long_reference, long_reference);
    EXPECT_GT(result.n_segments, 0);
    EXPECT_APPROX(result.translation_pct, 0.0, std::nullopt, 1e-9);
    EXPECT_APPROX(result.rotation_deg_per_m, 0.0, std::nullopt, 1e-9);
}

// py: test_metrics.py::TestKittiDrift::test_scale_error_gives_predictable_drift
TEST(TestKittiDrift, ScaleErrorGivesPredictableDrift) {
    // A uniform 10% scale shortfall on a straight run gives ~10% drift.
    const Trajectory reference = unit_steps(600);
    const Trajectory shrunk = reference.transformed(Eigen::Matrix4d::Identity(), 0.9);
    EXPECT_APPROX(kitti_drift(shrunk, reference, {100.0, 200.0}).translation_pct, 10.0, 0.02);
}

// py: test_metrics.py::TestKittiDrift::test_rotation_units_are_consistent
TEST(TestKittiDrift, RotationUnitsAreConsistent) {
    const Trajectory long_reference = long_reference_fixture();
    const auto result = kitti_drift(long_reference, long_reference);
    EXPECT_APPROX(result.rotation_deg_per_100m, result.rotation_deg_per_m * 100.0);
}

// py: test_metrics.py::TestKittiDrift::test_per_length_breakdown_is_populated
TEST(TestKittiDrift, PerLengthBreakdownIsPopulated) {
    const Trajectory long_reference = long_reference_fixture();
    const auto result = kitti_drift(long_reference, long_reference, {100.0, 200.0});
    EXPECT_EQ(result.per_length.size(), 2u);
    EXPECT_TRUE(result.per_length.count(100.0) && result.per_length.count(200.0));
}

// py: test_metrics.py::TestKittiDrift::test_short_trajectory_yields_nan_not_a_crash
TEST(TestKittiDrift, ShortTrajectoryYieldsNanNotACrash) {
    // 35 m of travel cannot support a 100 m sub-trajectory.
    const Trajectory reference = reference_fixture();
    const auto result = kitti_drift(reference, reference, {100.0});
    EXPECT_EQ(result.n_segments, 0);
    EXPECT_TRUE(std::isnan(result.translation_pct));
}

// py: test_metrics.py::TestKittiDrift::test_segments_longer_than_the_path_are_skipped
TEST(TestKittiDrift, SegmentsLongerThanThePathAreSkipped) {
    const Trajectory long_reference = long_reference_fixture();
    const auto result = kitti_drift(long_reference, long_reference, {100.0, 10'000.0});
    EXPECT_FALSE(result.per_length.count(10'000.0));
    EXPECT_TRUE(result.per_length.count(100.0));
}

// py: test_metrics.py::TestKittiDrift::test_invalid_step_raises
TEST(TestKittiDrift, InvalidStepRaises) {
    const Trajectory long_reference = long_reference_fixture();
    expect_value_error([&] { kitti_drift(long_reference, long_reference, default_segment_lengths(), 0); },
                       "step must be");
}

// --------------------------------------------------------------------------- //
// TestCombinedEvaluation
// --------------------------------------------------------------------------- //

// py: test_metrics.py::TestCombinedEvaluation::test_reports_every_metric
TEST(TestCombinedEvaluation, ReportsEveryMetric) {
    const Trajectory long_reference = long_reference_fixture();
    const auto evaluation = evaluate_trajectory(long_reference, long_reference, "perfect", "sim3", {1, 10});
    EXPECT_EQ(evaluation.label, "perfect");
    EXPECT_APPROX(evaluation.ate_aligned.rmse, 0.0, std::nullopt, 1e-8);
    EXPECT_EQ(evaluation.rpe.size(), 2u);
    EXPECT_TRUE(evaluation.rpe.count(1) && evaluation.rpe.count(10));
    EXPECT_GT(evaluation.drift.n_segments, 0);
    EXPECT_APPROX(evaluation.path_length_m, evaluation.reference_path_length_m);
}

// py: test_metrics.py::TestCombinedEvaluation::test_oversized_rpe_delta_is_skipped_not_fatal
TEST(TestCombinedEvaluation, OversizedRpeDeltaIsSkippedNotFatal) {
    const Trajectory reference = reference_fixture();
    const auto evaluation = evaluate_trajectory(reference, reference, "trajectory", "sim3", {1, 10'000});
    EXPECT_EQ(evaluation.rpe.size(), 1u);
    EXPECT_TRUE(evaluation.rpe.count(1));
}

// py: test_metrics.py::TestCombinedEvaluation::test_to_dict_is_json_serialisable
TEST(TestCombinedEvaluation, ToDictIsJsonSerialisable) {
    const Trajectory long_reference = long_reference_fixture();
    const Json data = evaluate_trajectory(long_reference, long_reference, "trajectory", "sim3", {1}).to_dict();
    EXPECT_NO_THROW(py::loads(py::dumps(data)));
    EXPECT_TRUE(data.contains("ate_aligned") && data.contains("drift"));
}

// py: test_metrics.py::TestCombinedEvaluation::test_summary_line_mentions_the_numbers
TEST(TestCombinedEvaluation, SummaryLineMentionsTheNumbers) {
    const Trajectory long_reference = long_reference_fixture();
    const std::string line =
        evaluate_trajectory(long_reference, long_reference, "trajectory", "sim3", {1}).summary_line();
    EXPECT_NE(line.find("ATE RMSE"), std::string::npos);
    EXPECT_NE(line.find("drift"), std::string::npos);
}

// --------------------------------------------------------------------------- //
// TestATEReduction
// --------------------------------------------------------------------------- //

// py: test_metrics.py::TestATEReduction::test_improvement_is_positive
TEST(TestATEReduction, ImprovementIsPositive) { EXPECT_APPROX(ate_reduction_pct(10.0, 4.0), 60.0); }

// py: test_metrics.py::TestATEReduction::test_regression_is_negative
TEST(TestATEReduction, RegressionIsNegative) { EXPECT_APPROX(ate_reduction_pct(4.0, 5.0), -25.0); }

// py: test_metrics.py::TestATEReduction::test_no_change_is_zero
TEST(TestATEReduction, NoChangeIsZero) { EXPECT_APPROX(ate_reduction_pct(7.0, 7.0), 0.0); }

// py: test_metrics.py::TestATEReduction::test_undefined_cases_return_nan[0.0]
// py: test_metrics.py::TestATEReduction::test_undefined_cases_return_nan[-1.0]
// py: test_metrics.py::TestATEReduction::test_undefined_cases_return_nan[nan]
// py: test_metrics.py::TestATEReduction::test_undefined_cases_return_nan[inf]
TEST(TestATEReduction, UndefinedCasesReturnNan) {
    for (double before : {0.0, -1.0, static_cast<double>(NAN), static_cast<double>(INFINITY)}) {
        EXPECT_TRUE(std::isnan(ate_reduction_pct(before, 1.0))) << before;
    }
}

// --------------------------------------------------------------------------- //
// Golden parity with NumPy (tests/golden/metrics.json)
// --------------------------------------------------------------------------- //

TEST(MetricsGolden, AlignmentAteRpeAndDriftMatchNumpy) {
    static const std::string name = "metrics.json";
    const Json& g = golden::load(name);
    for (const auto& item : g.at("cases")) {
        const std::string case_name = item.at("name").get<std::string>();
        SCOPED_TRACE(case_name);
        const Trajectory est(golden::poses(item.at("estimated")));
        const Trajectory ref(golden::poses(item.at("reference")));
        for (const std::string mode : {"sim3", "se3", "none"}) {
            const Json& expected = item.at("align_" + mode);
            const auto [aligned, result] = align_trajectory(est, ref, mode);
            EXPECT_TRUE(testing_utils::allclose(result.R, golden::mat(expected.at("R")), 1e-12, 1e-13)) << mode;
            EXPECT_TRUE(testing_utils::allclose(golden::vec(expected.at("t")),
                                                std::vector<double>(result.t.data(), result.t.data() + 3),
                                                1e-11, 1e-11))
                << mode;
            EXPECT_NEAR(result.scale, golden::from_hex(expected.at("scale")), 1e-12) << mode;
            EXPECT_EQ(result.describe(), expected.at("describe")) << mode;
            const auto ate = absolute_trajectory_error(est, ref, mode);
            EXPECT_NEAR(ate.rmse, golden::from_hex(expected.at("ate_rmse")), 1e-10) << mode;
            EXPECT_TRUE(testing_utils::allclose(ate.errors, golden::vec(expected.at("ate_errors")), 1e-9, 1e-11))
                << mode;
        }
        for (const std::string delta : {"1", "10", "100"}) {
            if (!item.contains("rpe_" + delta)) {
                continue;
            }
            const Json& expected = item.at("rpe_" + delta);
            const auto rpe = relative_pose_error(est, ref, std::stoll(delta));
            EXPECT_NEAR(rpe.trans_rmse, golden::from_hex(expected.at("trans_rmse")), 1e-11);
            EXPECT_NEAR(rpe.rot_rmse_deg, golden::from_hex(expected.at("rot_rmse_deg")), 1e-9);
            EXPECT_NEAR(rpe.trans_median, golden::from_hex(expected.at("trans_median")), 1e-11);
            EXPECT_NEAR(rpe.rot_max_deg, golden::from_hex(expected.at("rot_max_deg")), 1e-9);
        }
        const auto drift = kitti_drift(est, ref);
        const Json& expected_drift = item.at("drift");
        EXPECT_EQ(drift.n_segments, expected_drift.at("n_segments").get<std::int64_t>());
        if (drift.n_segments > 0) {
            EXPECT_NEAR(drift.translation_pct, golden::from_hex(expected_drift.at("translation_pct")), 1e-10);
            EXPECT_NEAR(drift.rotation_deg_per_m, golden::from_hex(expected_drift.at("rotation_deg_per_m")), 1e-12);
        }
        // The serialized evaluation (rounded to 3-8 digits, NaN for an
        // undefined drift) is byte-identical to the Python json.dumps text.
        const auto evaluation = evaluate_trajectory(est, ref, case_name, "sim3", {1, 10, 100});
        EXPECT_EQ(py::dumps(evaluation.to_dict(), 2), item.at("evaluation_json").get<std::string>());
        EXPECT_EQ(evaluation.summary_line(), item.at("summary_line").get<std::string>());
    }
    for (const auto& item : g.at("ate_reduction")) {
        EXPECT_NEAR(ate_reduction_pct(golden::from_hex(item.at(0)), golden::from_hex(item.at(1))),
                    golden::from_hex(item.at(2)), 1e-12);
    }
}
