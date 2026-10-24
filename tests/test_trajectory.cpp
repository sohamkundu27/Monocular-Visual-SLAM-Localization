// Tests for the Trajectory container (mirrors tests/test_trajectory.py),
// plus golden checks against the NumPy implementation.
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/errors.hpp"
#include "mvslam/geometry/pose.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "support/golden.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using synthetic::random_se3;
using synthetic::small_se3;
using testing_utils::TempDir;

namespace {

constexpr const char* kModule = "test_trajectory";

Eigen::MatrixXd M(const Eigen::MatrixXd& m) { return m; }

/// Camera driving forward along +z, the KITTI forward axis.
Trajectory straight_line_trajectory(int n = 11, double step = 2.0) {
    PoseList poses(static_cast<std::size_t>(n), Eigen::Matrix4d::Identity());
    for (int i = 0; i < n; ++i) {
        poses[static_cast<std::size_t>(i)](2, 3) = i * step;
    }
    return Trajectory(poses);
}

PoseList identities(std::size_t n) { return PoseList(n, Eigen::Matrix4d::Identity()); }

}  // namespace

// --------------------------------------------------------------------------- //
// TestConstruction
// --------------------------------------------------------------------------- //

// py: test_trajectory.py::TestConstruction::test_defaults_frame_ids
TEST(TestConstruction, DefaultsFrameIds) {
    EXPECT_EQ(straight_line_trajectory(5).frame_ids(), (std::vector<std::int64_t>{0, 1, 2, 3, 4}));
}

// py: test_trajectory.py::TestConstruction::test_single_pose_promoted
TEST(TestConstruction, SinglePosePromoted) {
    EXPECT_EQ(Trajectory(Eigen::Matrix4d::Identity()).size(), 1u);
}

// py: test_trajectory.py::TestConstruction::test_bad_shape_raises
TEST(TestConstruction, BadShapeRaises) {
    EXPECT_THROW(Trajectory::from_matrices(std::vector<Eigen::MatrixXd>(5, Eigen::MatrixXd::Zero(3, 3))),
                 ValueError);
}

// py: test_trajectory.py::TestConstruction::test_frame_id_length_mismatch_raises
TEST(TestConstruction, FrameIdLengthMismatchRaises) {
    EXPECT_THROW(Trajectory(identities(3), std::vector<std::int64_t>{0, 1}), ValueError);
}

// py: test_trajectory.py::TestConstruction::test_timestamp_length_mismatch_raises
TEST(TestConstruction, TimestampLengthMismatchRaises) {
    EXPECT_THROW(Trajectory(identities(3), std::nullopt, std::vector<double>{0.0, 0.1}), ValueError);
}

// --------------------------------------------------------------------------- //
// TestDerivedQuantities
// --------------------------------------------------------------------------- //

// py: test_trajectory.py::TestDerivedQuantities::test_positions_and_xz
TEST(TestDerivedQuantities, PositionsAndXz) {
    const Trajectory traj = straight_line_trajectory(4, 1.5);
    EXPECT_ALLCLOSE(M(traj.positions().col(2)), M(Eigen::Vector4d(0, 1.5, 3.0, 4.5)));
    EXPECT_ALLCLOSE(M(traj.xz().col(0)), M(Eigen::Vector4d::Zero()));
    EXPECT_ALLCLOSE(M(traj.xz().col(1)), M(Eigen::Vector4d(0, 1.5, 3.0, 4.5)));
}

// py: test_trajectory.py::TestDerivedQuantities::test_path_length_of_straight_line
TEST(TestDerivedQuantities, PathLengthOfStraightLine) {
    EXPECT_APPROX(straight_line_trajectory(11, 2.0).path_length(), 20.0);
}

// py: test_trajectory.py::TestDerivedQuantities::test_cumulative_distance_is_monotonic
TEST(TestDerivedQuantities, CumulativeDistanceIsMonotonic) {
    golden::Stream rng(kModule, "TestDerivedQuantities::test_cumulative_distance_is_monotonic");
    PoseList poses{Eigen::Matrix4d::Identity()};
    for (int i = 0; i < 20; ++i) {
        poses.push_back(poses.back() * small_se3(rng));
    }
    const Trajectory traj(poses);
    const auto cumulative = traj.cumulative_distance();
    EXPECT_EQ(cumulative.size(), traj.size());
    EXPECT_EQ(cumulative[0], 0.0);
    for (std::size_t i = 1; i < cumulative.size(); ++i) {
        EXPECT_GE(cumulative[i] - cumulative[i - 1], -1e-12);
    }
    EXPECT_APPROX(cumulative.back(), traj.path_length());
}

// py: test_trajectory.py::TestDerivedQuantities::test_empty_trajectory_step_lengths
TEST(TestDerivedQuantities, EmptyTrajectoryStepLengths) {
    const Trajectory traj{PoseList{}};
    EXPECT_EQ(traj.size(), 0u);
    EXPECT_TRUE(traj.step_lengths().empty());
    EXPECT_EQ(traj.path_length(), 0.0);
}

// py: test_trajectory.py::TestDerivedQuantities::test_relative_poses_reconstruct
TEST(TestDerivedQuantities, RelativePosesReconstruct) {
    golden::Stream rng(kModule, "TestDerivedQuantities::test_relative_poses_reconstruct");
    PoseList poses{Eigen::Matrix4d::Identity()};
    for (int i = 0; i < 15; ++i) {
        poses.push_back(poses.back() * small_se3(rng));
    }
    const PoseList rel = Trajectory(poses).relative_poses(1);
    EXPECT_EQ(rel.size(), 15u);
    Eigen::Matrix4d accumulated = Eigen::Matrix4d::Identity();
    for (const auto& r : rel) {
        accumulated = accumulated * r;
    }
    EXPECT_ALLCLOSE(M(accumulated), M(poses.back()), 1e-7, 1e-9);
}

// py: test_trajectory.py::TestDerivedQuantities::test_relative_poses_with_larger_delta
TEST(TestDerivedQuantities, RelativePosesWithLargerDelta) {
    golden::Stream rng(kModule, "TestDerivedQuantities::test_relative_poses_with_larger_delta");
    PoseList poses;
    for (int i = 0; i < 10; ++i) {
        poses.push_back(random_se3(rng));
    }
    const Trajectory traj(poses);
    const PoseList rel = traj.relative_poses(3);
    EXPECT_EQ(rel.size(), 7u);
    EXPECT_ALLCLOSE(M(rel[0]), M(relative_pose(traj[0], traj[3])), 1e-7, 1e-12);
}

// py: test_trajectory.py::TestDerivedQuantities::test_delta_larger_than_trajectory_is_empty
TEST(TestDerivedQuantities, DeltaLargerThanTrajectoryIsEmpty) {
    EXPECT_TRUE(straight_line_trajectory(3).relative_poses(5).empty());
}

// py: test_trajectory.py::TestDerivedQuantities::test_invalid_delta_raises
TEST(TestDerivedQuantities, InvalidDeltaRaises) {
    EXPECT_THROW(straight_line_trajectory(3).relative_poses(0), ValueError);
}

// --------------------------------------------------------------------------- //
// TestTransformations
// --------------------------------------------------------------------------- //

// py: test_trajectory.py::TestTransformations::test_transformed_scales_positions
TEST(TestTransformations, TransformedScalesPositions) {
    const Trajectory traj = straight_line_trajectory(5, 2.0);
    const Trajectory scaled = traj.transformed(Eigen::Matrix4d::Identity(), 3.0);
    EXPECT_ALLCLOSE(M(scaled.positions()), M(traj.positions() * 3.0));
    EXPECT_APPROX(scaled.path_length(), traj.path_length() * 3.0);
}

// py: test_trajectory.py::TestTransformations::test_transformed_rotates_positions_and_orientations
TEST(TestTransformations, TransformedRotatesPositionsAndOrientations) {
    const Eigen::Matrix3d R = rotation::from_euler("y", 90.0);
    const Eigen::Matrix4d T = se3_from_rt(R, Eigen::Vector3d(1.0, 0.0, 0.0));
    const Trajectory moved = straight_line_trajectory(3, 1.0).transformed(T);
    // (0,0,1) rotated 90 deg about +y becomes (1,0,0), then translated by (1,0,0).
    EXPECT_ALLCLOSE(M(moved.positions().row(1)), M(Eigen::RowVector3d(2.0, 0.0, 0.0)), 1e-7, 1e-9);
    EXPECT_ALLCLOSE(M(moved.rotations()[0]), M(R), 1e-7, 1e-12);
}

// py: test_trajectory.py::TestTransformations::test_transform_preserves_shape_and_validity
TEST(TestTransformations, TransformPreservesShapeAndValidity) {
    golden::Stream rng(kModule, "TestTransformations::test_transform_preserves_shape_and_validity");
    PoseList poses;
    for (int i = 0; i < 8; ++i) {
        poses.push_back(random_se3(rng));
    }
    const Trajectory traj(poses);
    const Trajectory moved = traj.transformed(random_se3(rng), 2.5);
    EXPECT_TRUE(moved.is_valid());
    // A similarity transform scales all distances uniformly.
    std::vector<double> expected = traj.step_lengths();
    for (double& v : expected) {
        v *= 2.5;
    }
    EXPECT_ALLCLOSE(moved.step_lengths(), expected, 1e-7, 1e-9);
}

// py: test_trajectory.py::TestTransformations::test_relative_to_first_anchors_at_identity
TEST(TestTransformations, RelativeToFirstAnchorsAtIdentity) {
    golden::Stream rng(kModule, "TestTransformations::test_relative_to_first_anchors_at_identity");
    PoseList poses;
    for (int i = 0; i < 6; ++i) {
        poses.push_back(random_se3(rng));
    }
    const Trajectory traj(poses);
    const Trajectory anchored = traj.relative_to_first();
    EXPECT_ALLCLOSE(M(anchored[0]), M(Eigen::Matrix4d::Identity()), 1e-7, 1e-10);
    // Relative geometry between poses must be untouched.
    EXPECT_ALLCLOSE(M(relative_pose(anchored[1], anchored[4])), M(relative_pose(traj[1], traj[4])),
                    1e-7, 1e-9);
}

// py: test_trajectory.py::TestTransformations::test_orthonormalized_repairs_drifted_rotations
TEST(TestTransformations, OrthonormalizedRepairsDriftedRotations) {
    golden::Stream rng(kModule, "TestTransformations::test_orthonormalized_repairs_drifted_rotations");
    PoseList poses;
    for (int i = 0; i < 5; ++i) {
        poses.push_back(random_se3(rng));
    }
    Trajectory traj(poses);
    const auto original = traj.rotations();
    const golden::Array noise = rng.next_array("normal");  // (5, 3, 3)
    const auto values = noise.as_doubles();
    for (std::size_t n = 0; n < 5; ++n) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                traj[n](r, c) += values[n * 9 + static_cast<std::size_t>(r * 3 + c)];
            }
        }
    }
    EXPECT_FALSE(traj.is_valid());
    const Trajectory fixed = traj.orthonormalized();
    EXPECT_TRUE(fixed.is_valid());
    // Repair must snap back to (essentially) the rotation we started from.
    const auto repaired = fixed.rotations();
    for (std::size_t i = 0; i < original.size(); ++i) {
        EXPECT_LT(rotation_angle_deg(Eigen::Matrix3d(original[i].transpose() * repaired[i])), 0.5);
    }
}

// py: test_trajectory.py::TestTransformations::test_copy_is_deep
TEST(TestTransformations, CopyIsDeep) {
    const Trajectory traj = straight_line_trajectory(3);
    Trajectory clone = traj.copy();
    clone[0](0, 3) = 99.0;
    EXPECT_EQ(traj[0](0, 3), 0.0);
}

// --------------------------------------------------------------------------- //
// TestSelection
// --------------------------------------------------------------------------- //

// py: test_trajectory.py::TestSelection::test_slicing_returns_sub_trajectory
TEST(TestSelection, SlicingReturnsSubTrajectory) {
    const Trajectory sub = straight_line_trajectory(10).slice(2, 5);
    EXPECT_EQ(sub.size(), 3u);
    EXPECT_EQ(sub.frame_ids(), (std::vector<std::int64_t>{2, 3, 4}));
}

// py: test_trajectory.py::TestSelection::test_integer_index_returns_pose
TEST(TestSelection, IntegerIndexReturnsPose) {
    const Eigen::Matrix4d pose = straight_line_trajectory(4)[1];
    EXPECT_EQ(pose.rows(), 4);
    EXPECT_EQ(pose.cols(), 4);
}

// py: test_trajectory.py::TestSelection::test_select_frames_matches_ids
TEST(TestSelection, SelectFramesMatchesIds) {
    const Trajectory traj(identities(6), std::vector<std::int64_t>{0, 3, 6, 9, 12, 15});
    EXPECT_EQ(traj.select_frames({3, 12}).frame_ids(), (std::vector<std::int64_t>{3, 12}));
}

// py: test_trajectory.py::TestSelection::test_select_missing_frame_raises
TEST(TestSelection, SelectMissingFrameRaises) {
    const Trajectory traj(identities(3), std::vector<std::int64_t>{0, 2, 4});
    EXPECT_THROW(traj.select_frames({1}), KeyError);
}

// py: test_trajectory.py::TestSelection::test_timestamps_follow_selection
TEST(TestSelection, TimestampsFollowSelection) {
    const Trajectory traj(identities(4), std::vector<std::int64_t>{0, 1, 2, 3},
                          std::vector<double>{0.0, 0.1, 0.2, 0.3});
    const Trajectory sub = traj.slice(1, 3);
    ASSERT_TRUE(sub.timestamps().has_value());
    EXPECT_ALLCLOSE(*sub.timestamps(), (std::vector<double>{0.1, 0.2}));
}

// --------------------------------------------------------------------------- //
// TestSerialisation
// --------------------------------------------------------------------------- //

// py: test_trajectory.py::TestSerialisation::test_kitti_row_round_trip
TEST(TestSerialisation, KittiRowRoundTrip) {
    golden::Stream rng(kModule, "TestSerialisation::test_kitti_row_round_trip");
    PoseList poses;
    for (int i = 0; i < 7; ++i) {
        poses.push_back(random_se3(rng));
    }
    const Trajectory traj(poses);
    const auto rows = traj.to_kitti_rows();
    EXPECT_EQ(rows.size(), 7u);
    const Trajectory back = Trajectory::from_kitti_rows(rows);
    for (std::size_t i = 0; i < traj.size(); ++i) {
        EXPECT_ALLCLOSE(M(back[i]), M(traj[i]), 1e-7, 1e-15);
    }
}

// py: test_trajectory.py::TestSerialisation::test_file_round_trip
TEST(TestSerialisation, FileRoundTrip) {
    golden::Stream rng(kModule, "TestSerialisation::test_file_round_trip");
    TempDir tmp;
    PoseList poses;
    for (int i = 0; i < 5; ++i) {
        poses.push_back(random_se3(rng));
    }
    const Trajectory traj(poses);
    const Trajectory loaded = Trajectory::load_kitti(traj.save_kitti(tmp / "poses.txt"));
    for (std::size_t i = 0; i < traj.size(); ++i) {
        EXPECT_ALLCLOSE(M(loaded[i]), M(traj[i]), 1e-7, 1e-10);
    }
}

// py: test_trajectory.py::TestSerialisation::test_bad_row_shape_raises
TEST(TestSerialisation, BadRowShapeRaises) {
    TempDir tmp;
    testing_utils::write_text(tmp / "bad.txt", std::string(3, '\n') + "0 0 0 0 0 0 0 0 0 0 0\n"
                                                                       "0 0 0 0 0 0 0 0 0 0 0\n"
                                                                       "0 0 0 0 0 0 0 0 0 0 0\n");
    EXPECT_THROW(Trajectory::load_kitti(tmp / "bad.txt"), ValueError);
}

// py: test_trajectory.py::TestSerialisation::test_empty_to_rows
TEST(TestSerialisation, EmptyToRows) {
    EXPECT_TRUE(Trajectory(PoseList{}).to_kitti_rows().empty());
}

// py: test_trajectory.py::TestSerialisation::test_repr_mentions_length
TEST(TestSerialisation, ReprMentionsLength) {
    EXPECT_NE(straight_line_trajectory(5).repr().find("n=5"), std::string::npos);
    EXPECT_NE(Trajectory(PoseList{}).repr().find("empty"), std::string::npos);
}

// --------------------------------------------------------------------------- //
// Golden parity with NumPy (tests/golden/trajectory.json)
// --------------------------------------------------------------------------- //

namespace {
const Json& traj_golden() { return golden::load("trajectory.json"); }

void expect_poses_close(const PoseList& actual, const PoseList& expected, double rtol, double atol) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_ALLCLOSE(M(actual[i]), M(expected[i]), rtol, atol) << "pose " << i;
    }
}
}  // namespace

TEST(TrajectoryGolden, DerivedQuantitiesMatchNumpy) {
    const Json& g = traj_golden();
    const Trajectory short_traj(golden::poses(g.at("short")));
    const Trajectory long_traj(golden::poses(g.at("long")));
    EXPECT_ALLCLOSE(short_traj.step_lengths(), golden::vec(g.at("step_lengths")), 1e-15, 0.0);
    EXPECT_ALLCLOSE(short_traj.cumulative_distance(), golden::vec(g.at("cumulative")), 1e-15, 0.0);
    EXPECT_NEAR(short_traj.path_length(), golden::from_hex(g.at("path_length")), 1e-12);
    EXPECT_NEAR(long_traj.path_length(), golden::from_hex(g.at("long_path_length")), 1e-10);
    expect_poses_close(short_traj.relative_poses(1), golden::poses(g.at("relative_1")), 1e-12, 1e-14);
    expect_poses_close(short_traj.relative_poses(3), golden::poses(g.at("relative_3")), 1e-12, 1e-14);
}

TEST(TrajectoryGolden, TransformationsMatchNumpy) {
    const Json& g = traj_golden();
    const Trajectory short_traj(golden::poses(g.at("short")));
    const Eigen::Matrix4d T = golden::mat4(g.at("transform_T"));
    expect_poses_close(short_traj.transformed(T, 2.5).poses(), golden::poses(g.at("transformed")),
                       1e-12, 1e-13);
    expect_poses_close(short_traj.transformed(T).relative_to_first().poses(),
                       golden::poses(g.at("relative_to_first")), 1e-11, 1e-13);
    const Trajectory perturbed(golden::poses(g.at("perturbed")));
    expect_poses_close(perturbed.orthonormalized().poses(), golden::poses(g.at("orthonormalized")),
                       1e-12, 1e-14);
}

TEST(TrajectoryGolden, KittiTextIsByteIdenticalToSavetxt) {
    const Json& g = traj_golden();
    TempDir tmp;
    const Trajectory short_traj(golden::poses(g.at("short")));
    short_traj.save_kitti(tmp / "t.txt");
    EXPECT_EQ(testing_utils::read_text(tmp / "t.txt"), g.at("kitti_text").get<std::string>());
}
