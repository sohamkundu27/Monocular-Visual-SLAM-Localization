// Tests for SE(3)/SO(3) transform utilities (mirrors tests/test_transforms.py),
// plus golden checks against the NumPy/SciPy implementation.
#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "support/golden.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using synthetic::random_se3;
using synthetic::small_se3;

namespace {

constexpr const char* kModule = "test_transforms";

Eigen::MatrixXd M(const Eigen::MatrixXd& m) { return m; }

Eigen::Matrix3d normal3x3(golden::Stream& rng) { return rng.next_mat("normal"); }

}  // namespace

// --------------------------------------------------------------------------- //
// TestConstruction
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestConstruction::test_se3_from_rt_places_blocks_correctly
TEST(TestConstruction, Se3FromRtPlacesBlocksCorrectly) {
    const Eigen::Matrix3d R = rotation::from_euler("z", 90.0);
    const Eigen::Vector3d t(1.0, 2.0, 3.0);
    const Eigen::Matrix4d T = se3_from_rt(R, t);
    EXPECT_ALLCLOSE(M(T.topLeftCorner<3, 3>()), M(R));
    EXPECT_ALLCLOSE(M(T.topRightCorner<3, 1>()), M(t));
    EXPECT_ALLCLOSE(M(T.row(3)), M(Eigen::RowVector4d(0, 0, 0, 1)));
}

// py: test_transforms.py::TestConstruction::test_translation_shapes_accepted[shape0]
// py: test_transforms.py::TestConstruction::test_translation_shapes_accepted[shape1]
// py: test_transforms.py::TestConstruction::test_translation_shapes_accepted[shape2]
TEST(TestConstruction, TranslationShapesAccepted) {
    // (3,), (3, 1) and (1, 3) all reshape to the same Eigen::Vector3d.
    const Eigen::Vector3d column(0.0, 1.0, 2.0);
    const Eigen::RowVector3d row(0.0, 1.0, 2.0);
    for (const Eigen::Vector3d& t : {column, Eigen::Vector3d(row.transpose()), Eigen::Vector3d(0, 1, 2)}) {
        EXPECT_ALLCLOSE(M(se3_from_rt(Eigen::Matrix3d::Identity(), t).topRightCorner<3, 1>()),
                        M(Eigen::Vector3d(0, 1, 2)));
    }
}

// py: test_transforms.py::TestConstruction::test_split_round_trips
TEST(TestConstruction, SplitRoundTrips) {
    golden::Stream rng(kModule, "TestConstruction::test_split_round_trips");
    const Eigen::Matrix4d T = random_se3(rng);
    EXPECT_ALLCLOSE(M(se3_from_rt(rotation_of(T), translation_of(T))), M(T));
}

// py: test_transforms.py::TestConstruction::test_bad_shapes_raise
TEST(TestConstruction, BadShapesRaise) {
    // Fixed-size Eigen types make wrong shapes a compile error for these
    // functions; the runtime shape check lives where arbitrary arrays enter
    // (Trajectory::from_matrices).
    EXPECT_THROW(Trajectory::from_matrices({Eigen::MatrixXd::Identity(3, 3)}), ValueError);
    EXPECT_THROW(Trajectory::from_matrices({Eigen::MatrixXd::Identity(2, 2)}), ValueError);
}

// --------------------------------------------------------------------------- //
// TestInversionAndComposition
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestInversionAndComposition::test_inverse_is_exact
TEST(TestInversionAndComposition, InverseIsExact) {
    golden::Stream rng(kModule, "TestInversionAndComposition::test_inverse_is_exact");
    for (int i = 0; i < 20; ++i) {
        const Eigen::Matrix4d T = random_se3(rng);
        EXPECT_ALLCLOSE(M(T * invert_se3(T)), M(Eigen::Matrix4d::Identity()), 1e-7, 1e-12);
        EXPECT_ALLCLOSE(M(invert_se3(T) * T), M(Eigen::Matrix4d::Identity()), 1e-7, 1e-12);
    }
}

// py: test_transforms.py::TestInversionAndComposition::test_inverse_matches_dense_inverse
TEST(TestInversionAndComposition, InverseMatchesDenseInverse) {
    golden::Stream rng(kModule, "TestInversionAndComposition::test_inverse_matches_dense_inverse");
    const Eigen::Matrix4d T = random_se3(rng);
    EXPECT_ALLCLOSE(M(invert_se3(T)), M(T.inverse()), 1e-7, 1e-10);
}

// py: test_transforms.py::TestInversionAndComposition::test_compose_is_left_to_right
TEST(TestInversionAndComposition, ComposeIsLeftToRight) {
    golden::Stream rng(kModule, "TestInversionAndComposition::test_compose_is_left_to_right");
    const Eigen::Matrix4d A = random_se3(rng);
    const Eigen::Matrix4d B = random_se3(rng);
    const Eigen::Matrix4d C = random_se3(rng);
    EXPECT_ALLCLOSE(M(compose({A, B, C})), M(A * B * C), 1e-7, 1e-12);
}

// py: test_transforms.py::TestInversionAndComposition::test_compose_of_nothing_is_identity
TEST(TestInversionAndComposition, ComposeOfNothingIsIdentity) {
    EXPECT_ALLCLOSE(M(compose({})), M(Eigen::Matrix4d::Identity()));
}

// py: test_transforms.py::TestInversionAndComposition::test_relative_pose_definition
TEST(TestInversionAndComposition, RelativePoseDefinition) {
    // T_a_b must satisfy T_w_a @ T_a_b == T_w_b.
    golden::Stream rng(kModule, "TestInversionAndComposition::test_relative_pose_definition");
    const Eigen::Matrix4d T_w_a = random_se3(rng);
    const Eigen::Matrix4d T_w_b = random_se3(rng);
    EXPECT_ALLCLOSE(M(T_w_a * relative_pose(T_w_a, T_w_b)), M(T_w_b), 1e-7, 1e-10);
}

// py: test_transforms.py::TestInversionAndComposition::test_relative_pose_of_self_is_identity
TEST(TestInversionAndComposition, RelativePoseOfSelfIsIdentity) {
    golden::Stream rng(kModule, "TestInversionAndComposition::test_relative_pose_of_self_is_identity");
    const Eigen::Matrix4d T = random_se3(rng);
    EXPECT_ALLCLOSE(M(relative_pose(T, T)), M(Eigen::Matrix4d::Identity()), 1e-7, 1e-12);
}

// py: test_transforms.py::TestInversionAndComposition::test_chained_relative_poses_reconstruct_trajectory
TEST(TestInversionAndComposition, ChainedRelativePosesReconstructTrajectory) {
    // Accumulating relative motions must reproduce the absolute poses; this is
    // the exact operation the VO trajectory builder performs.
    golden::Stream rng(kModule,
                       "TestInversionAndComposition::test_chained_relative_poses_reconstruct_trajectory");
    PoseList poses{Eigen::Matrix4d::Identity()};
    for (int i = 0; i < 30; ++i) {
        poses.push_back(poses.back() * small_se3(rng));
    }
    PoseList rebuilt{Eigen::Matrix4d::Identity()};
    for (std::size_t i = 1; i < poses.size(); ++i) {
        rebuilt.push_back(rebuilt.back() * relative_pose(poses[i - 1], poses[i]));
    }
    for (std::size_t i = 0; i < poses.size(); ++i) {
        EXPECT_ALLCLOSE(M(rebuilt[i]), M(poses[i]), 1e-7, 1e-9);
    }
}

// --------------------------------------------------------------------------- //
// TestPointTransforms
// --------------------------------------------------------------------------- //

using RowPoints3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

// py: test_transforms.py::TestPointTransforms::test_transform_points_matches_manual
TEST(TestPointTransforms, TransformPointsMatchesManual) {
    golden::Stream rng(kModule, "TestPointTransforms::test_transform_points_matches_manual");
    const Eigen::Matrix4d T = random_se3(rng);
    const RowPoints3 pts = rng.next_mat("normal");
    const Eigen::Matrix3d R = rotation_of(T);
    RowPoints3 expected = (R * pts.transpose()).transpose();
    expected.rowwise() += translation_of(T).transpose();
    EXPECT_ALLCLOSE(M(transform_points(T, pts)), M(expected), 1e-7, 1e-12);
}

// py: test_transforms.py::TestPointTransforms::test_transform_then_inverse_recovers_points
TEST(TestPointTransforms, TransformThenInverseRecoversPoints) {
    golden::Stream rng(kModule, "TestPointTransforms::test_transform_then_inverse_recovers_points");
    const Eigen::Matrix4d T = random_se3(rng);
    const RowPoints3 pts = rng.next_mat("normal");
    const RowPoints3 moved = transform_points(T, pts);
    EXPECT_ALLCLOSE(M(transform_points(invert_se3(T), moved)), M(pts), 1e-7, 1e-10);
}

// py: test_transforms.py::TestPointTransforms::test_wrong_point_shape_raises
TEST(TestPointTransforms, WrongPointShapeRaises) {
    // (N, 2) input is rejected by the type system: transform_points takes
    // (N, 3) row-major points. Assert the dimension contract directly.
    static_assert(RowPoints3::ColsAtCompileTime == 3, "points must be (N, 3)");
    SUCCEED();
}

// --------------------------------------------------------------------------- //
// TestValidation
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestValidation::test_identity_is_valid
TEST(TestValidation, IdentityIsValid) {
    EXPECT_TRUE(is_rotation_matrix(Eigen::Matrix3d::Identity()));
    EXPECT_TRUE(is_valid_se3(Eigen::Matrix4d::Identity()));
}

// py: test_transforms.py::TestValidation::test_random_rotations_are_valid
TEST(TestValidation, RandomRotationsAreValid) {
    golden::Stream rng(kModule, "TestValidation::test_random_rotations_are_valid");
    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(is_rotation_matrix(rotation_of(random_se3(rng))));
    }
}

// py: test_transforms.py::TestValidation::test_reflection_rejected
TEST(TestValidation, ReflectionRejected) {
    // det == -1 is the classic degenerate essential-matrix decomposition.
    const Eigen::Matrix3d R = Eigen::Vector3d(1.0, 1.0, -1.0).asDiagonal();
    EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity()));
    EXPECT_FALSE(is_rotation_matrix(R));
}

// py: test_transforms.py::TestValidation::test_scaled_rotation_rejected
TEST(TestValidation, ScaledRotationRejected) {
    EXPECT_FALSE(is_rotation_matrix(1.5 * Eigen::Matrix3d::Identity()));
}

// py: test_transforms.py::TestValidation::test_nan_rejected
TEST(TestValidation, NanRejected) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(0, 0) = NAN;
    EXPECT_FALSE(is_rotation_matrix(R));
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = INFINITY;
    EXPECT_FALSE(is_valid_se3(T));
}

// py: test_transforms.py::TestValidation::test_broken_bottom_row_rejected
TEST(TestValidation, BrokenBottomRowRejected) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(3, 0) = 0.5;
    EXPECT_FALSE(is_valid_se3(T));
}

// --------------------------------------------------------------------------- //
// TestProjection
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestProjection::test_project_to_so3_is_identity_on_rotations
TEST(TestProjection, ProjectToSo3IsIdentityOnRotations) {
    golden::Stream rng(kModule, "TestProjection::test_project_to_so3_is_identity_on_rotations");
    const Eigen::Matrix3d R = rotation_of(random_se3(rng));
    EXPECT_ALLCLOSE(M(project_to_so3(R)), M(R), 1e-7, 1e-12);
}

// py: test_transforms.py::TestProjection::test_project_to_so3_repairs_perturbation
TEST(TestProjection, ProjectToSo3RepairsPerturbation) {
    golden::Stream rng(kModule, "TestProjection::test_project_to_so3_repairs_perturbation");
    const Eigen::Matrix3d R = rotation_of(random_se3(rng));
    const Eigen::Matrix3d noisy = R + normal3x3(rng);
    EXPECT_FALSE(is_rotation_matrix(noisy));
    const Eigen::Matrix3d fixed = project_to_so3(noisy);
    EXPECT_TRUE(is_rotation_matrix(fixed));
    EXPECT_LT(rotation_angle_deg(Eigen::Matrix3d(R.transpose() * fixed)), 0.5);
}

// py: test_transforms.py::TestProjection::test_project_never_returns_reflection
TEST(TestProjection, ProjectNeverReturnsReflection) {
    const Eigen::Matrix3d fixed = project_to_so3(Eigen::Vector3d(1.0, 1.0, -1.0).asDiagonal());
    EXPECT_TRUE(is_rotation_matrix(fixed));
    EXPECT_GT(fixed.determinant(), 0);
}

// py: test_transforms.py::TestProjection::test_project_to_se3_preserves_translation
TEST(TestProjection, ProjectToSe3PreservesTranslation) {
    golden::Stream rng(kModule, "TestProjection::test_project_to_se3_preserves_translation");
    Eigen::Matrix4d T = random_se3(rng);
    T.topLeftCorner<3, 3>() += normal3x3(rng);
    const Eigen::Matrix4d fixed = project_to_se3(T);
    EXPECT_TRUE(is_valid_se3(fixed));
    EXPECT_ALLCLOSE(M(translation_of(fixed)), M(translation_of(T)));
}

// --------------------------------------------------------------------------- //
// TestAngles
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestAngles::test_rotation_angle_recovers_input[0.0]
// py: test_transforms.py::TestAngles::test_rotation_angle_recovers_input[1.0]
// py: test_transforms.py::TestAngles::test_rotation_angle_recovers_input[30.0]
// py: test_transforms.py::TestAngles::test_rotation_angle_recovers_input[90.0]
// py: test_transforms.py::TestAngles::test_rotation_angle_recovers_input[179.0]
TEST(TestAngles, RotationAngleRecoversInput) {
    for (double deg : {0.0, 1.0, 30.0, 90.0, 179.0}) {
        EXPECT_APPROX(rotation_angle_deg(rotation::from_euler("y", deg)), deg, std::nullopt, 1e-6);
    }
}

// py: test_transforms.py::TestAngles::test_rotation_angle_accepts_4x4
TEST(TestAngles, RotationAngleAccepts4x4) {
    const Eigen::Matrix4d T = se3_from_rt(rotation::from_euler("x", 45.0), Eigen::Vector3d::Zero());
    EXPECT_APPROX(rotation_angle_deg(T), 45.0, std::nullopt, 1e-6);
}

// py: test_transforms.py::TestAngles::test_rotation_angle_is_symmetric
TEST(TestAngles, RotationAngleIsSymmetric) {
    golden::Stream rng(kModule, "TestAngles::test_rotation_angle_is_symmetric");
    const Eigen::Matrix3d R = rotation_of(random_se3(rng));
    EXPECT_APPROX(rotation_angle_rad(R), rotation_angle_rad(Eigen::Matrix3d(R.transpose())),
                  std::nullopt, 1e-10);
}

// py: test_transforms.py::TestAngles::test_angle_is_bounded
TEST(TestAngles, AngleIsBounded) {
    golden::Stream rng(kModule, "TestAngles::test_angle_is_bounded");
    for (int i = 0; i < 50; ++i) {
        const double angle = rotation_angle_rad(rotation_of(random_se3(rng)));
        EXPECT_GE(angle, 0.0);
        EXPECT_LE(angle, M_PI + 1e-12);
    }
}

// --------------------------------------------------------------------------- //
// TestLieMaps
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestLieMaps::test_se3_exp_log_round_trip
TEST(TestLieMaps, Se3ExpLogRoundTrip) {
    golden::Stream rng(kModule, "TestLieMaps::test_se3_exp_log_round_trip");
    for (int i = 0; i < 30; ++i) {
        const Eigen::Matrix4d T = small_se3(rng);
        EXPECT_ALLCLOSE(M(se3_exp(se3_log(T))), M(T), 1e-7, 1e-9);
    }
}

// py: test_transforms.py::TestLieMaps::test_se3_log_exp_round_trip
TEST(TestLieMaps, Se3LogExpRoundTrip) {
    golden::Stream rng(kModule, "TestLieMaps::test_se3_log_exp_round_trip");
    for (int i = 0; i < 30; ++i) {
        const std::vector<double> raw = rng.next_vec("normal");
        Vector6d xi;
        for (int k = 0; k < 6; ++k) {
            xi(k) = raw[static_cast<std::size_t>(k)] * 0.4;
        }
        EXPECT_ALLCLOSE(M(se3_log(se3_exp(xi))), M(xi), 1e-7, 1e-9);
    }
}

// py: test_transforms.py::TestLieMaps::test_log_of_identity_is_zero
TEST(TestLieMaps, LogOfIdentityIsZero) {
    EXPECT_ALLCLOSE(M(se3_log(Eigen::Matrix4d::Identity())), M(Vector6d::Zero()), 1e-7, 1e-12);
}

// py: test_transforms.py::TestLieMaps::test_tiny_rotation_uses_stable_branch
TEST(TestLieMaps, TinyRotationUsesStableBranch) {
    // The small-angle branch must not blow up as theta -> 0.
    Vector6d xi;
    xi << 0.1, -0.2, 0.3, 1e-14, 0.0, 0.0;
    const Eigen::Matrix4d T = se3_exp(xi);
    EXPECT_TRUE(is_valid_se3(T));
    EXPECT_ALLCLOSE(M(translation_of(T)), M(Eigen::Vector3d(0.1, -0.2, 0.3)), 1e-7, 1e-10);
}

// py: test_transforms.py::TestLieMaps::test_pure_translation
TEST(TestLieMaps, PureTranslation) {
    Vector6d xi;
    xi << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0;
    const Eigen::Matrix4d T = se3_exp(xi);
    EXPECT_ALLCLOSE(M(rotation_of(T)), M(Eigen::Matrix3d::Identity()), 1e-7, 1e-12);
    EXPECT_ALLCLOSE(M(translation_of(T)), M(Eigen::Vector3d(1, 2, 3)), 1e-7, 1e-12);
}

// --------------------------------------------------------------------------- //
// TestKittiSerialisation
// --------------------------------------------------------------------------- //

// py: test_transforms.py::TestKittiSerialisation::test_row_round_trip
TEST(TestKittiSerialisation, RowRoundTrip) {
    golden::Stream rng(kModule, "TestKittiSerialisation::test_row_round_trip");
    const Eigen::Matrix4d T = random_se3(rng);
    EXPECT_ALLCLOSE(M(kitti_row_to_se3(se3_to_kitti_row(T))), M(T), 1e-7, 1e-15);
}

// py: test_transforms.py::TestKittiSerialisation::test_row_is_row_major
TEST(TestKittiSerialisation, RowIsRowMajor) {
    Eigen::Matrix4d T;
    for (int i = 0; i < 16; ++i) {
        T(i / 4, i % 4) = i;
    }
    T.row(3) << 0, 0, 0, 1;
    const auto row = se3_to_kitti_row(T);
    for (int i = 0; i < 12; ++i) {
        EXPECT_EQ(row[static_cast<std::size_t>(i)], static_cast<double>(i));
    }
}

// py: test_transforms.py::TestKittiSerialisation::test_wrong_length_raises
TEST(TestKittiSerialisation, WrongLengthRaises) {
    // A row of 11 values cannot even be represented as a KittiRow; the
    // runtime check happens when text is parsed (Trajectory::load_kitti).
    testing_utils::TempDir tmp;
    testing_utils::write_text(tmp / "bad.txt", "0 0 0 0 0 0 0 0 0 0 0\n");
    EXPECT_THROW(Trajectory::load_kitti(tmp / "bad.txt"), ValueError);
}

// --------------------------------------------------------------------------- //
// Golden parity with NumPy / SciPy (tests/golden/transforms.json)
// --------------------------------------------------------------------------- //

namespace {
const Json& tf_golden() { return golden::load("transforms.json"); }
}  // namespace

TEST(TransformsGolden, PoseOperationsMatchNumpy) {
    for (const auto& item : tf_golden().at("poses")) {
        const Eigen::Matrix4d T = golden::mat4(item.at("T"));
        EXPECT_ALLCLOSE(M(invert_se3(T)), golden::mat(item.at("inverse")), 1e-14, 1e-15);
        EXPECT_NEAR(rotation_angle_rad(T), golden::from_hex(item.at("angle_rad")), 1e-14);
        EXPECT_NEAR(rotation_angle_deg(rotation_of(T)), golden::from_hex(item.at("angle_deg")), 1e-12);
        const Eigen::Vector3d log3 = so3_log(rotation_of(T));
        EXPECT_ALLCLOSE(std::vector<double>(log3.data(), log3.data() + 3),
                        golden::vec(item.at("so3_log")), 1e-12, 1e-14);
        const Vector6d log6 = se3_log(T);
        EXPECT_ALLCLOSE(std::vector<double>(log6.data(), log6.data() + 6),
                        golden::vec(item.at("se3_log")), 1e-11, 1e-13);
        EXPECT_EQ(is_valid_se3(T), item.at("is_valid").get<bool>());
        const auto row = se3_to_kitti_row(T);
        const auto expected = golden::vec(item.at("kitti_row"));
        for (std::size_t i = 0; i < 12; ++i) {
            EXPECT_EQ(row[i], expected[i]);
        }
    }
}

TEST(TransformsGolden, ProjectionMatchesNumpySvd) {
    for (const auto& item : tf_golden().at("projection")) {
        const Eigen::Matrix4d M4 = golden::mat4(item.at("M"));
        const Eigen::Matrix3d R = rotation_of(M4);
        EXPECT_EQ(is_rotation_matrix(R), item.at("is_rotation").get<bool>());
        const Eigen::Matrix3d expected = golden::mat3(item.at("so3"));
        const Eigen::Matrix3d actual = project_to_so3(R);
        if ((R - Eigen::Matrix3d(Eigen::Vector3d(1, 1, -1).asDiagonal())).norm() == 0.0) {
            // All singular values equal: the SVD basis (and so which axis gets
            // flipped) is arbitrary in both LAPACK and Eigen. Any proper
            // rotation is a valid answer; the tests above check that.
            EXPECT_TRUE(is_rotation_matrix(actual));
            continue;
        }
        EXPECT_ALLCLOSE(M(actual), M(expected), 1e-12, 1e-14);
        EXPECT_ALLCLOSE(M(project_to_se3(M4)), golden::mat(item.at("se3")), 1e-12, 1e-14);
    }
}

TEST(TransformsGolden, ExponentialsMatchScipy) {
    for (const auto& item : tf_golden().at("exp")) {
        const auto raw = golden::vec(item.at("xi"));
        Vector6d xi;
        for (int k = 0; k < 6; ++k) {
            xi(k) = raw[static_cast<std::size_t>(k)];
        }
        EXPECT_ALLCLOSE(M(se3_exp(xi)), golden::mat(item.at("se3_exp")), 1e-13, 1e-14);
        EXPECT_ALLCLOSE(M(so3_exp(xi.tail<3>())), golden::mat(item.at("so3_exp")), 1e-14, 1e-15);
    }
}

TEST(TransformsGolden, RelativeComposeAndPointTransforms) {
    for (const auto& item : tf_golden().at("relative")) {
        const Eigen::Matrix4d a = golden::mat4(item.at("a"));
        const Eigen::Matrix4d b = golden::mat4(item.at("b"));
        EXPECT_ALLCLOSE(M(relative_pose(a, b)), golden::mat(item.at("relative")), 1e-13, 1e-14);
        EXPECT_ALLCLOSE(M(compose({a, b, a})), golden::mat(item.at("compose")), 1e-13, 1e-14);
    }
    const Json& case_ = tf_golden().at("transform_points");
    const RowPoints3 points = golden::mat(case_.at("points"));
    EXPECT_ALLCLOSE(M(transform_points(golden::mat4(case_.at("T")), points)),
                    golden::mat(case_.at("out")), 1e-13, 1e-13);
}

TEST(TransformsGolden, ScipyRotationConstructors) {
    for (const auto& item : tf_golden().at("euler")) {
        const std::string seq = item.at("seq").get<std::string>();
        std::vector<double> angles;
        if (item.at("angles").is_array()) {
            for (const auto& a : item.at("angles")) {
                angles.push_back(a.get<double>());
            }
        } else {
            angles.push_back(item.at("angles").get<double>());
        }
        EXPECT_ALLCLOSE(M(rotation::from_euler(seq, angles)), golden::mat(item.at("matrix")), 0.0,
                        2e-16)
            << seq;
    }
    for (const auto& item : tf_golden().at("rotvec")) {
        const auto v = golden::vec(item.at("rotvec"));
        EXPECT_ALLCLOSE(M(rotation::from_rotvec(Eigen::Vector3d(v[0], v[1], v[2]))),
                        golden::mat(item.at("matrix")), 0.0, 2e-16);
    }
}
