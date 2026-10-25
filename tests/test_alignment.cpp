// Tests for Umeyama trajectory alignment (mirrors tests/test_alignment.py).
#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "mvslam/errors.hpp"
#include "mvslam/evaluation/alignment.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "support/golden.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;

namespace {

constexpr const char* kModule = "test_alignment";

Eigen::MatrixXd M(const Eigen::MatrixXd& m) { return m; }

/// (scale * (R @ points.T)).T + t
Eigen::MatrixXd similarity(const Eigen::MatrixXd& points, double s, const Eigen::Matrix3d& R,
                           const Eigen::Vector3d& t) {
    Eigen::MatrixXd out = (s * (R * points.transpose())).transpose();
    out.rowwise() += t.transpose();
    return out;
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
// TestUmeyamaCore
// --------------------------------------------------------------------------- //

// py: test_alignment.py::TestUmeyamaCore::test_identical_sets_give_identity
TEST(TestUmeyamaCore, IdenticalSetsGiveIdentity) {
    golden::Stream rng(kModule, "TestUmeyamaCore::test_identical_sets_give_identity");
    const Eigen::MatrixXd points = rng.next_mat("normal") * 10;
    const auto result = umeyama_alignment(points, points);
    EXPECT_ALLCLOSE(M(result.R), M(Eigen::Matrix3d::Identity()), 1e-7, 1e-9);
    EXPECT_ALLCLOSE(M(result.t), M(Eigen::Vector3d::Zero()), 1e-7, 1e-9);
    EXPECT_APPROX(result.scale, 1.0, std::nullopt, 1e-9);
}

// py: test_alignment.py::TestUmeyamaCore::test_recovers_known_similarity_transform
TEST(TestUmeyamaCore, RecoversKnownSimilarityTransform) {
    golden::Stream rng(kModule, "TestUmeyamaCore::test_recovers_known_similarity_transform");
    const Eigen::MatrixXd points = rng.next_mat("normal") * 5;
    const Eigen::Matrix3d R_true = rotation::from_euler("xyz", {20, -35, 60});
    const Eigen::Vector3d t_true(3.0, -7.0, 12.0);
    const double s_true = 2.7;
    const Eigen::MatrixXd transformed = similarity(points, s_true, R_true, t_true);
    const auto result = umeyama_alignment(points, transformed);
    EXPECT_APPROX(result.scale, s_true, 1e-9);
    EXPECT_ALLCLOSE(M(result.R), M(R_true), 1e-7, 1e-9);
    EXPECT_ALLCLOSE(M(result.t), M(t_true), 1e-7, 1e-8);
    EXPECT_ALLCLOSE(M(result.apply(points)), transformed, 1e-7, 1e-8);
}

// py: test_alignment.py::TestUmeyamaCore::test_rigid_mode_ignores_scale
TEST(TestUmeyamaCore, RigidModeIgnoresScale) {
    golden::Stream rng(kModule, "TestUmeyamaCore::test_rigid_mode_ignores_scale");
    const Eigen::MatrixXd points = rng.next_mat("normal") * 5;
    const Eigen::MatrixXd scaled = points * 3.0;
    const auto result = umeyama_alignment(points, scaled, false);
    EXPECT_EQ(result.scale, 1.0);
    // Without a scale degree of freedom the fit cannot match the target.
    EXPECT_GT((M(result.apply(points)) - scaled).norm(), 1.0);
}

// py: test_alignment.py::TestUmeyamaCore::test_rigid_mode_recovers_rigid_transform
TEST(TestUmeyamaCore, RigidModeRecoversRigidTransform) {
    golden::Stream rng(kModule, "TestUmeyamaCore::test_rigid_mode_recovers_rigid_transform");
    const Eigen::MatrixXd points = rng.next_mat("normal") * 5;
    const Eigen::Matrix4d T = synthetic::random_se3(rng);
    const Eigen::MatrixXd moved = similarity(points, 1.0, rotation_of(T), translation_of(T));
    const auto result = umeyama_alignment(points, moved, false);
    EXPECT_ALLCLOSE(M(result.R), M(rotation_of(T)), 1e-7, 1e-9);
    EXPECT_ALLCLOSE(M(result.apply(points)), moved, 1e-7, 1e-8);
}

// py: test_alignment.py::TestUmeyamaCore::test_result_is_always_a_rotation_not_a_reflection
TEST(TestUmeyamaCore, ResultIsAlwaysARotationNotAReflection) {
    // A mirrored target must not produce det(R) == -1.
    golden::Stream rng(kModule, "TestUmeyamaCore::test_result_is_always_a_rotation_not_a_reflection");
    const Eigen::MatrixXd points = rng.next_mat("normal");
    Eigen::MatrixXd mirrored = points;
    mirrored.col(2) *= -1.0;
    const auto result = umeyama_alignment(points, mirrored);
    EXPECT_APPROX(result.R.determinant(), 1.0, std::nullopt, 1e-9);
    EXPECT_ALLCLOSE(M(result.R.transpose() * result.R), M(Eigen::Matrix3d::Identity()), 1e-7, 1e-9);
}

// py: test_alignment.py::TestUmeyamaCore::test_collinear_points_still_yield_a_rotation
TEST(TestUmeyamaCore, CollinearPointsStillYieldARotation) {
    // Degenerate (rank-deficient) geometry must not produce a reflection.
    Eigen::MatrixXd points = Eigen::MatrixXd::Zero(10, 3);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(10, 3);
    for (int i = 0; i < 10; ++i) {
        points(i, 0) = i;
        target(i, 1) = i;
    }
    EXPECT_APPROX(umeyama_alignment(points, target).R.determinant(), 1.0, std::nullopt, 1e-6);
}

// py: test_alignment.py::TestUmeyamaCore::test_noise_gives_approximate_recovery
TEST(TestUmeyamaCore, NoiseGivesApproximateRecovery) {
    golden::Stream rng(kModule, "TestUmeyamaCore::test_noise_gives_approximate_recovery");
    const Eigen::MatrixXd points = rng.next_mat("normal") * 5;
    const Eigen::Matrix3d R_true = rotation::from_euler("y", 30.0);
    Eigen::MatrixXd target = similarity(points, 1.5, R_true, Eigen::Vector3d(1.0, 2.0, 3.0));
    target += rng.next_mat("normal");
    const auto result = umeyama_alignment(points, target);
    EXPECT_APPROX(result.scale, 1.5, 0.02);
    EXPECT_LT(rotation_angle_deg(Eigen::Matrix3d(result.R.transpose() * R_true)), 0.5);
}

// py: test_alignment.py::TestUmeyamaCore::test_stationary_source_does_not_divide_by_zero
TEST(TestUmeyamaCore, StationarySourceDoesNotDivideByZero) {
    const auto result = umeyama_alignment(Eigen::MatrixXd::Zero(5, 3), Eigen::MatrixXd::Ones(5, 3));
    EXPECT_EQ(result.scale, 1.0);
    EXPECT_TRUE(result.t.allFinite());
}

// py: test_alignment.py::TestUmeyamaCore::test_invalid_inputs_raise[source0-target0]
// py: test_alignment.py::TestUmeyamaCore::test_invalid_inputs_raise[source1-target1]
// py: test_alignment.py::TestUmeyamaCore::test_invalid_inputs_raise[source2-target2]
TEST(TestUmeyamaCore, InvalidInputsRaise) {
    expect_value_error([] { umeyama_alignment(Eigen::MatrixXd::Zero(5, 3), Eigen::MatrixXd::Zero(4, 3)); });
    expect_value_error([] { umeyama_alignment(Eigen::MatrixXd::Zero(5, 2), Eigen::MatrixXd::Zero(5, 2)); });
    expect_value_error([] { umeyama_alignment(Eigen::MatrixXd::Zero(2, 3), Eigen::MatrixXd::Zero(2, 3)); });
}

// --------------------------------------------------------------------------- //
// TestAlignmentResult
// --------------------------------------------------------------------------- //

// py: test_alignment.py::TestAlignmentResult::test_T_holds_rotation_and_translation_without_scale
TEST(TestAlignmentResult, THoldsRotationAndTranslationWithoutScale) {
    AlignmentResult result;
    result.t = Eigen::Vector3d(1.0, 2.0, 3.0);
    result.scale = 5.0;
    result.mode = "sim3";
    EXPECT_ALLCLOSE(M(rotation_of(result.T())), M(Eigen::Matrix3d::Identity()));
    EXPECT_ALLCLOSE(M(translation_of(result.T())), M(Eigen::Vector3d(1, 2, 3)));
}

// py: test_alignment.py::TestAlignmentResult::test_identity_result
TEST(TestAlignmentResult, IdentityResult) {
    const Points3 ones = Points3::Ones(3, 3);
    EXPECT_ALLCLOSE(M(AlignmentResult::identity().apply(ones)), M(ones));
}

// py: test_alignment.py::TestAlignmentResult::test_describe_is_serialisable
TEST(TestAlignmentResult, DescribeIsSerialisable) {
    const Json info = umeyama_alignment(Eigen::MatrixXd::Identity(3, 3),
                                        Eigen::MatrixXd::Identity(3, 3) * 2).describe();
    EXPECT_EQ(info["mode"], "sim3");
    EXPECT_TRUE(info["scale"].is_number_float());
}

// --------------------------------------------------------------------------- //
// TestTrajectoryAlignment
// --------------------------------------------------------------------------- //

namespace {
Trajectory reference_fixture() { return Trajectory(synthetic::straight_drive(40)); }
}  // namespace

// py: test_alignment.py::TestTrajectoryAlignment::test_sim3_recovers_a_scaled_rotated_trajectory
TEST(TestTrajectoryAlignment, Sim3RecoversAScaledRotatedTrajectory) {
    // The classic monocular case: right shape, wrong scale and frame.
    const Trajectory reference = reference_fixture();
    const Eigen::Matrix4d T = se3_from_rt(rotation::from_euler("y", 47.0), Eigen::Vector3d(5.0, 0.0, -3.0));
    const Trajectory distorted = reference.transformed(T, 0.35);
    const auto [aligned, result] = align_trajectory(distorted, reference, "sim3");
    EXPECT_APPROX(result.scale, 1 / 0.35, 1e-6);
    EXPECT_ALLCLOSE(M(aligned.positions()), M(reference.positions()), 1e-7, 1e-8);
}

// py: test_alignment.py::TestTrajectoryAlignment::test_se3_cannot_undo_a_scale_error
TEST(TestTrajectoryAlignment, Se3CannotUndoAScaleError) {
    const Trajectory reference = reference_fixture();
    const Trajectory distorted = reference.transformed(Eigen::Matrix4d::Identity(), 0.5);
    const auto [aligned, result] = align_trajectory(distorted, reference, "se3");
    EXPECT_EQ(result.scale, 1.0);
    const Eigen::MatrixXd diff = M(aligned.positions()) - M(reference.positions());
    EXPECT_GT(diff.rowwise().norm().maxCoeff(), 1.0);
}

// py: test_alignment.py::TestTrajectoryAlignment::test_none_mode_is_a_no_op
TEST(TestTrajectoryAlignment, NoneModeIsANoOp) {
    const Trajectory reference = reference_fixture();
    const Trajectory distorted = reference.transformed(Eigen::Matrix4d::Identity(), 0.5);
    const auto [aligned, result] = align_trajectory(distorted, reference, "none");
    EXPECT_ALLCLOSE(M(aligned.positions()), M(distorted.positions()));
    EXPECT_EQ(result.scale, 1.0);
    EXPECT_EQ(result.mode, "none");
}

// py: test_alignment.py::TestTrajectoryAlignment::test_alignment_preserves_frame_ids
TEST(TestTrajectoryAlignment, AlignmentPreservesFrameIds) {
    const Trajectory reference = reference_fixture();
    const Trajectory distorted = reference.transformed(Eigen::Matrix4d::Identity(), 2.0);
    EXPECT_EQ(align_trajectory(distorted, reference).first.frame_ids(), distorted.frame_ids());
}

// py: test_alignment.py::TestTrajectoryAlignment::test_aligned_poses_remain_valid_se3
TEST(TestTrajectoryAlignment, AlignedPosesRemainValidSe3) {
    const Trajectory reference = reference_fixture();
    const Trajectory distorted = reference.transformed(Eigen::Matrix4d::Identity(), 0.25);
    EXPECT_TRUE(align_trajectory(distorted, reference).first.is_valid());
}

// py: test_alignment.py::TestTrajectoryAlignment::test_length_mismatch_raises
TEST(TestTrajectoryAlignment, LengthMismatchRaises) {
    const Trajectory reference = reference_fixture();
    expect_value_error([&] { align_trajectory(reference.slice(0, 10), reference); }, "equal length");
}

// py: test_alignment.py::TestTrajectoryAlignment::test_unknown_mode_raises
TEST(TestTrajectoryAlignment, UnknownModeRaises) {
    const Trajectory reference = reference_fixture();
    expect_value_error([&] { align_trajectory(reference, reference, "procrustes"); },
                       "Unknown alignment mode");
}
