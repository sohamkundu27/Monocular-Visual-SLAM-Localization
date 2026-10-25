// Tests for pose-graph construction and GTSAM optimization (mirrors
// tests/test_pose_graph.py), plus golden checks against the Python GTSAM run.
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyjson.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/optimization/gtsam_backend.hpp"
#include "mvslam/optimization/pose_graph.hpp"
#include "support/golden.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;

namespace {

constexpr const char* kModule = "test_pose_graph";

Eigen::MatrixXd M(const Eigen::MatrixXd& m) { return m; }

/// Stand-in for a verified LoopClosure, with the fields build_pose_graph reads.
LoopClosure fake_loop(std::int64_t match_id, std::int64_t query_id, const Eigen::Matrix4d& T,
                      std::int64_t n_inliers = 200, double similarity = 0.9) {
    LoopClosure lc;
    lc.match_id = match_id;
    lc.query_id = query_id;
    lc.T_match_query = T;
    lc.n_inliers = n_inliers;
    lc.similarity = similarity;
    return lc;
}

/// A closed circuit that returns exactly to its starting pose.
Trajectory circular_trajectory(int n = 120, double step = 1.0) {
    PoseList poses{Eigen::Matrix4d::Identity()};
    const double yaw = 360.0 / n;
    for (int i = 0; i < n - 1; ++i) {
        const Eigen::Matrix3d R = rotation::from_euler("y", yaw);
        poses.push_back(poses.back() * se3_from_rt(R, Eigen::Vector3d(0.0, 0.0, step)));
    }
    return Trajectory(poses);
}

/// Re-integrate a trajectory with a constant per-step yaw bias.
Trajectory add_yaw_drift(const Trajectory& reference, double bias_deg) {
    PoseList drifted{Eigen::Matrix4d::Identity()};
    const Eigen::Matrix3d R_bias = rotation::from_euler("y", bias_deg);
    for (std::size_t i = 0; i + 1 < reference.size(); ++i) {
        const Eigen::Matrix4d relative = invert_se3(reference[i]) * reference[i + 1];
        const Eigen::Matrix4d step = se3_from_rt(R_bias * rotation_of(relative), translation_of(relative));
        drifted.push_back(drifted.back() * step);
    }
    return Trajectory(drifted);
}

double max_position_error(const Trajectory& a, const Trajectory& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, (translation_of(a[i]) - translation_of(b[i])).norm());
    }
    return worst;
}

OptimizerOptions with_robust(bool robust) {
    OptimizerOptions options;
    options.robust_loop_kernel = robust;
    return options;
}

bool contains(const std::vector<std::string>& problems, const std::string& fragment) {
    for (const auto& p : problems) {
        if (p.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace

// --------------------------------------------------------------------------- //
// TestPoseGraphConstruction
// --------------------------------------------------------------------------- //

// py: test_pose_graph.py::TestPoseGraphConstruction::test_nodes_get_sequential_ids
TEST(TestPoseGraphConstruction, NodesGetSequentialIds) {
    golden::Stream rng(kModule, "TestPoseGraphConstruction::test_nodes_get_sequential_ids");
    PoseGraph graph;
    for (int i = 0; i < 5; ++i) {
        graph.add_node(synthetic::random_se3(rng));
    }
    std::vector<std::int64_t> ids;
    for (const auto& n : graph.nodes) {
        ids.push_back(n.node_id);
    }
    EXPECT_EQ(ids, (std::vector<std::int64_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(graph.size(), 5u);
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_node_poses_are_orthonormalised
TEST(TestPoseGraphConstruction, NodePosesAreOrthonormalised) {
    golden::Stream rng(kModule, "TestPoseGraphConstruction::test_node_poses_are_orthonormalised");
    PoseGraph graph;
    Eigen::Matrix4d pose = synthetic::random_se3(rng);
    pose.topLeftCorner<3, 3>() += rng.next_mat("normal");
    EXPECT_TRUE(is_valid_se3(graph.add_node(pose).pose, 1e-6));
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_bad_pose_shape_raises
TEST(TestPoseGraphConstruction, BadPoseShapeRaises) {
    try {
        PoseGraph().add_node_checked(Eigen::MatrixXd::Identity(3, 3));
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("4x4"), std::string::npos);
    }
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_edge_ids_are_validated
TEST(TestPoseGraphConstruction, EdgeIdsAreValidated) {
    PoseGraph graph;
    graph.add_node(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    EXPECT_THROW(graph.add_odometry_edge(0, 5, Eigen::Matrix4d::Identity()), IndexError);
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_build_from_trajectory_creates_a_chain
TEST(TestPoseGraphConstruction, BuildFromTrajectoryCreatesAChain) {
    const PoseGraph graph = build_pose_graph(circular_trajectory(30), {});
    EXPECT_EQ(graph.nodes.size(), 30u);
    EXPECT_EQ(graph.odometry_edges.size(), 29u);
    EXPECT_EQ(graph.loop_edges.size(), 0u);
    EXPECT_TRUE(graph.is_connected());
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_odometry_edges_reproduce_the_trajectory
TEST(TestPoseGraphConstruction, OdometryEdgesReproduceTheTrajectory) {
    // Chaining the edges must reconstruct the input poses exactly.
    const Trajectory reference = circular_trajectory(20);
    const PoseGraph graph = build_pose_graph(reference, {});
    Eigen::Matrix4d accumulated = Eigen::Matrix4d::Identity();
    for (const auto& edge : graph.odometry_edges) {
        accumulated = accumulated * edge.T;
    }
    EXPECT_ALLCLOSE(M(accumulated), M(reference[reference.size() - 1]), 1e-7, 1e-9);
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_loop_edges_are_added
TEST(TestPoseGraphConstruction, LoopEdgesAreAdded) {
    const Trajectory reference = circular_trajectory(30);
    const PoseGraph graph = build_pose_graph(reference, {fake_loop(0, 29, invert_se3(reference[0]) * reference[29])});
    EXPECT_EQ(graph.loop_edges.size(), 1u);
    EXPECT_EQ(graph.loop_edges[0].n_inliers, 200);
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_loop_edges_use_looser_translation_noise
TEST(TestPoseGraphConstruction, LoopEdgesUseLooserTranslationNoise) {
    const PoseGraph graph = build_pose_graph(circular_trajectory(30), {fake_loop(0, 29, Eigen::Matrix4d::Identity())});
    EXPECT_GT(graph.loop_edges[0].sigma_trans, graph.odometry_edges[0].sigma_trans);
}

// py: test_pose_graph.py::TestPoseGraphConstruction::test_frame_ids_are_carried_through
TEST(TestPoseGraphConstruction, FrameIdsAreCarriedThrough) {
    std::vector<std::int64_t> ids;
    for (int i = 0; i < 50; i += 5) {
        ids.push_back(i);
    }
    const Trajectory reference(circular_trajectory(10).poses(), ids);
    const PoseGraph graph = build_pose_graph(reference, {});
    std::vector<std::int64_t> frame_ids;
    for (const auto& n : graph.nodes) {
        frame_ids.push_back(n.frame_id);
    }
    EXPECT_EQ(frame_ids, ids);
}

// --------------------------------------------------------------------------- //
// TestGraphValidation
// --------------------------------------------------------------------------- //

// py: test_pose_graph.py::TestGraphValidation::test_a_healthy_graph_reports_no_problems
TEST(TestGraphValidation, AHealthyGraphReportsNoProblems) {
    EXPECT_TRUE(build_pose_graph(circular_trajectory(20), {}).validate().empty());
}

// py: test_pose_graph.py::TestGraphValidation::test_empty_graph_is_flagged
TEST(TestGraphValidation, EmptyGraphIsFlagged) {
    EXPECT_TRUE(contains(PoseGraph().validate(), "graph has no nodes"));
}

// py: test_pose_graph.py::TestGraphValidation::test_disconnected_graph_is_detected
TEST(TestGraphValidation, DisconnectedGraphIsDetected) {
    PoseGraph graph;
    for (int i = 0; i < 4; ++i) {
        graph.add_node(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    }
    graph.add_odometry_edge(0, 1, Eigen::Matrix4d::Identity());
    graph.add_odometry_edge(2, 3, Eigen::Matrix4d::Identity());  // separate component
    EXPECT_FALSE(graph.is_connected());
    EXPECT_TRUE(contains(graph.validate(), "disconnected"));
}

// py: test_pose_graph.py::TestGraphValidation::test_self_loop_is_flagged
TEST(TestGraphValidation, SelfLoopIsFlagged) {
    PoseGraph graph;
    graph.add_node(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    graph.odometry_edges.push_back({0, 0, Eigen::Matrix4d::Identity(), 0.02, 0.10});
    EXPECT_TRUE(contains(graph.validate(), "self-loop"));
}

// py: test_pose_graph.py::TestGraphValidation::test_non_positive_sigma_is_flagged
TEST(TestGraphValidation, NonPositiveSigmaIsFlagged) {
    PoseGraph graph = build_pose_graph(circular_trajectory(5), {});
    graph.odometry_edges[0].sigma_trans = 0.0;
    EXPECT_TRUE(contains(graph.validate(), "non-positive sigma"));
}

// py: test_pose_graph.py::TestGraphValidation::test_bad_prior_node_is_flagged
TEST(TestGraphValidation, BadPriorNodeIsFlagged) {
    PoseGraph graph = build_pose_graph(circular_trajectory(5), {});
    graph.prior_node_id = 99;
    EXPECT_TRUE(contains(graph.validate(), "prior_node_id"));
}

// py: test_pose_graph.py::TestGraphValidation::test_single_node_graph_is_connected
TEST(TestGraphValidation, SingleNodeGraphIsConnected) {
    PoseGraph graph;
    graph.add_node(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    EXPECT_TRUE(graph.is_connected());
}

// --------------------------------------------------------------------------- //
// TestEdgeResiduals
// --------------------------------------------------------------------------- //

// py: test_pose_graph.py::TestEdgeResiduals::test_consistent_graph_has_zero_residuals
TEST(TestEdgeResiduals, ConsistentGraphHasZeroResiduals) {
    const auto residuals = build_pose_graph(circular_trajectory(30), {}).edge_residuals();
    EXPECT_LT(np::max(residual(residuals, "odometry_translation_m")), 1e-9);
    EXPECT_LT(np::max(residual(residuals, "odometry_rotation_deg")), 1e-9);
}

// py: test_pose_graph.py::TestEdgeResiduals::test_loop_residual_measures_accumulated_drift
TEST(TestEdgeResiduals, LoopResidualMeasuresAccumulatedDrift) {
    // A drifted estimate must show its end-point gap on the loop edge.
    const Trajectory reference = circular_trajectory(120);
    const Trajectory drifted = add_yaw_drift(reference, 0.2);
    const PoseGraph graph = build_pose_graph(drifted, {fake_loop(0, 119, invert_se3(reference[0]) * reference[119])});
    const double r = residual(graph.edge_residuals(), "loop_translation_m")[0];
    const double expected = (translation_of(drifted[119]) - translation_of(reference[119])).norm();
    EXPECT_APPROX(r, expected, 0.05);
}

// py: test_pose_graph.py::TestEdgeResiduals::test_describe_summarises_the_graph
TEST(TestEdgeResiduals, DescribeSummarisesTheGraph) {
    const Trajectory reference = circular_trajectory(30);
    const Json info =
        build_pose_graph(reference, {fake_loop(0, 29, invert_se3(reference[0]) * reference[29])}).describe();
    EXPECT_EQ(info["n_nodes"], 30);
    EXPECT_EQ(info["n_loop_edges"], 1);
    EXPECT_EQ(info["connected"], true);
}

// --------------------------------------------------------------------------- //
// TestPoseConversion
// --------------------------------------------------------------------------- //

// py: test_pose_graph.py::TestPoseConversion::test_round_trip_is_exact
TEST(TestPoseConversion, RoundTripIsExact) {
    golden::Stream rng(kModule, "TestPoseConversion::test_round_trip_is_exact");
    for (int i = 0; i < 20; ++i) {
        const Eigen::Matrix4d T = synthetic::random_se3(rng);
        EXPECT_ALLCLOSE(M(pose3_to_numpy(numpy_to_pose3(T))), M(T), 1e-7, 1e-12);
    }
}

// py: test_pose_graph.py::TestPoseConversion::test_identity_round_trips
TEST(TestPoseConversion, IdentityRoundTrips) {
    EXPECT_ALLCLOSE(M(pose3_to_numpy(numpy_to_pose3(Eigen::Matrix4d::Identity()))),
                    M(Eigen::Matrix4d::Identity()), 1e-7, 1e-15);
}

// py: test_pose_graph.py::TestPoseConversion::test_drifted_rotation_is_repaired_not_rejected
TEST(TestPoseConversion, DriftedRotationIsRepairedNotRejected) {
    // GTSAM's Rot3 rejects non-orthonormal input; conversion must fix it.
    golden::Stream rng(kModule, "TestPoseConversion::test_drifted_rotation_is_repaired_not_rejected");
    Eigen::Matrix4d T = synthetic::random_se3(rng);
    T.topLeftCorner<3, 3>() += rng.next_mat("normal");
    EXPECT_FALSE(is_valid_se3(T, 1e-6));
    const Eigen::Matrix4d converted = pose3_to_numpy(numpy_to_pose3(T));
    EXPECT_TRUE(is_valid_se3(converted, 1e-9));
    EXPECT_LT(rotation_angle_deg(Eigen::Matrix3d(rotation_of(converted).transpose() * rotation_of(T))), 0.05);
}

// py: test_pose_graph.py::TestPoseConversion::test_translation_survives_conversion
TEST(TestPoseConversion, TranslationSurvivesConversion) {
    const Eigen::Matrix4d T = se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.5, -2.25, 3.75));
    EXPECT_ALLCLOSE(M(translation_of(pose3_to_numpy(numpy_to_pose3(T)))), M(Eigen::Vector3d(1.5, -2.25, 3.75)));
}

// --------------------------------------------------------------------------- //
// TestOptimization
// --------------------------------------------------------------------------- //

// py: test_pose_graph.py::TestOptimization::test_chain_without_loops_is_already_optimal
TEST(TestOptimization, ChainWithoutLoopsIsAlreadyOptimal) {
    // With no cycles, the odometry chain is the exact solution.
    const Trajectory reference = circular_trajectory(40);
    const auto result = optimize_pose_graph(build_pose_graph(reference, {}));
    EXPECT_ALLCLOSE(M(result.trajectory.positions()), M(reference.positions()), 1e-7, 1e-6);
    EXPECT_LT(result.final_error, 1e-8);
}

// py: test_pose_graph.py::TestOptimization::test_loop_closure_corrects_accumulated_drift
TEST(TestOptimization, LoopClosureCorrectsAccumulatedDrift) {
    const Trajectory reference = circular_trajectory(150);
    const Trajectory drifted = add_yaw_drift(reference, 0.2);
    const double before = max_position_error(drifted, reference);
    EXPECT_GT(before, 5.0);
    const auto result = optimize_pose_graph(
        build_pose_graph(drifted, {fake_loop(0, 149, invert_se3(reference[0]) * reference[149])}));
    EXPECT_LT(max_position_error(result.trajectory, reference), before / 3.0);
    EXPECT_GT(result.error_reduction_pct(), 50.0);
}

// py: test_pose_graph.py::TestOptimization::test_loop_residual_is_driven_down
TEST(TestOptimization, LoopResidualIsDrivenDown) {
    const Trajectory reference = circular_trajectory(150);
    const Trajectory drifted = add_yaw_drift(reference, 0.2);
    const auto result = optimize_pose_graph(
        build_pose_graph(drifted, {fake_loop(0, 149, invert_se3(reference[0]) * reference[149])}));
    const double before = residual(result.residuals_before, "loop_translation_m")[0];
    const double after = residual(result.residuals_after, "loop_translation_m")[0];
    EXPECT_LT(after, before / 100.0);
}

// py: test_pose_graph.py::TestOptimization::test_optimized_poses_remain_valid_se3
TEST(TestOptimization, OptimizedPosesRemainValidSe3) {
    const Trajectory reference = circular_trajectory(80);
    const Trajectory drifted = add_yaw_drift(reference, 0.3);
    EXPECT_TRUE(optimize_pose_graph(build_pose_graph(drifted, {fake_loop(0, 79, invert_se3(reference[0]) * reference[79])}))
                    .trajectory.is_valid());
}

// py: test_pose_graph.py::TestOptimization::test_prior_anchors_the_first_pose
TEST(TestOptimization, PriorAnchorsTheFirstPose) {
    const Trajectory reference = circular_trajectory(60);
    const Trajectory drifted = add_yaw_drift(reference, 0.3);
    const auto result = optimize_pose_graph(
        build_pose_graph(drifted, {fake_loop(0, 59, invert_se3(reference[0]) * reference[59])}));
    EXPECT_ALLCLOSE(M(result.trajectory[0]), M(Eigen::Matrix4d::Identity()), 1e-7, 1e-3);
}

// py: test_pose_graph.py::TestOptimization::test_robust_kernel_limits_a_false_positive_loop
TEST(TestOptimization, RobustKernelLimitsAFalsePositiveLoop) {
    // A bogus loop must not be allowed to wreck the whole trajectory.
    const Trajectory reference = circular_trajectory(120);
    const LoopClosure bogus = fake_loop(10, 60, se3_from_rt(Eigen::Matrix3d::Identity(), Eigen::Vector3d(500.0, 0, 0)), 50);
    const auto robust = optimize_pose_graph(build_pose_graph(reference, {bogus}), with_robust(true));
    const auto naive = optimize_pose_graph(build_pose_graph(reference, {bogus}), with_robust(false));
    EXPECT_LT(max_position_error(robust.trajectory, reference), max_position_error(naive.trajectory, reference));
}

// py: test_pose_graph.py::TestOptimization::test_gauss_newton_backend_runs
TEST(TestOptimization, GaussNewtonBackendRuns) {
    const Trajectory reference = circular_trajectory(60);
    const Trajectory drifted = add_yaw_drift(reference, 0.15);
    OptimizerOptions options;
    options.optimizer = "gauss_newton";
    const auto result = optimize_pose_graph(
        build_pose_graph(drifted, {fake_loop(0, 59, invert_se3(reference[0]) * reference[59])}), options);
    EXPECT_EQ(result.optimizer, "gauss_newton");
    EXPECT_LT(result.final_error, result.initial_error);
}

// py: test_pose_graph.py::TestOptimization::test_unknown_optimizer_raises
TEST(TestOptimization, UnknownOptimizerRaises) {
    OptimizerOptions options;
    options.optimizer = "adam";
    try {
        optimize_pose_graph(build_pose_graph(circular_trajectory(10), {}), options);
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("Unknown optimizer"), std::string::npos);
    }
}

// py: test_pose_graph.py::TestOptimization::test_invalid_graph_raises_before_gtsam_sees_it
TEST(TestOptimization, InvalidGraphRaisesBeforeGtsamSeesIt) {
    PoseGraph graph;
    for (int i = 0; i < 4; ++i) {
        graph.add_node(Eigen::Matrix4d(Eigen::Matrix4d::Identity()));
    }
    graph.add_odometry_edge(0, 1, Eigen::Matrix4d::Identity());
    try {
        optimize_pose_graph(graph);
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("not optimizable"), std::string::npos);
    }
}

// py: test_pose_graph.py::TestOptimization::test_result_serialises
TEST(TestOptimization, ResultSerialises) {
    const Trajectory reference = circular_trajectory(40);
    const Json data = optimize_pose_graph(
        build_pose_graph(reference, {fake_loop(0, 39, invert_se3(reference[0]) * reference[39])})).to_dict();
    EXPECT_NO_THROW(py::loads(py::dumps(data)));
    EXPECT_EQ(data["n_loop_edges"], 1);
}

// py: test_pose_graph.py::TestOptimization::test_frame_ids_survive_optimization
TEST(TestOptimization, FrameIdsSurviveOptimization) {
    std::vector<std::int64_t> ids;
    for (int i = 0; i < 100; i += 5) {
        ids.push_back(i);
    }
    const Trajectory reference(circular_trajectory(20).poses(), ids);
    EXPECT_EQ(optimize_pose_graph(build_pose_graph(reference, {})).trajectory.frame_ids(), ids);
}

// --------------------------------------------------------------------------- //
// Golden parity with the Python GTSAM run (tests/golden/pose_graph.json)
// --------------------------------------------------------------------------- //

TEST(PoseGraphGolden, ResidualsAndGtsamSolutionsMatchPython) {
    static const std::string name = "pose_graph.json";
    const Json& g = golden::load(name);
    for (const auto& item : g.at("cases")) {
        const std::string case_name = item.at("name").get<std::string>();
        SCOPED_TRACE(case_name);
        const Json& input = g.at("inputs").at(item.at("trajectory").get<std::string>());
        const Trajectory trajectory(golden::poses(input.at("poses")), golden::ivec(input.at("frame_ids")));
        std::vector<LoopClosure> loops;
        for (const auto& lc : item.at("loops")) {
            loops.push_back(fake_loop(lc.at("match_id").get<std::int64_t>(), lc.at("query_id").get<std::int64_t>(),
                                      golden::mat4(lc.at("T"))));
        }
        const PoseGraph graph = build_pose_graph(trajectory, loops);
        EXPECT_EQ(graph.describe(), item.at("describe"));
        const EdgeResiduals before = graph.edge_residuals();
        for (const auto& [key, values] : before) {
            EXPECT_ALLCLOSE(values, golden::vec(item.at("residuals_before").at(key)), 1e-9, 1e-11) << key;
        }

        OptimizerOptions options;
        const Json& kwargs = item.at("kwargs");
        if (kwargs.contains("robust_loop_kernel")) options.robust_loop_kernel = kwargs.at("robust_loop_kernel").get<bool>();
        if (kwargs.contains("optimizer")) options.optimizer = kwargs.at("optimizer").get<std::string>();
        if (kwargs.contains("max_iterations")) options.max_iterations = kwargs.at("max_iterations").get<std::int64_t>();
        const auto result = optimize_pose_graph(graph, options);

        // Same GTSAM version and Levenberg-Marquardt path: the same number of
        // iterations and the same solution to well below any metric's
        // rounding. GTSAM here is built against Eigen 3.4.0 and the wheel
        // against its bundled Eigen, so the last bits may differ.
        EXPECT_EQ(result.iterations, item.at("iterations").get<std::int64_t>());
        const double initial = golden::from_hex(item.at("initial_error"));
        const double final_error = golden::from_hex(item.at("final_error"));
        EXPECT_NEAR(result.initial_error, initial, 1e-9 * std::max(1.0, initial));
        EXPECT_NEAR(result.final_error, final_error, 1e-7 * std::max(1e-3, final_error));
        const PoseList expected = golden::poses(item.at("optimized"));
        ASSERT_EQ(expected.size(), result.trajectory.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            worst = std::max(worst, (result.trajectory[i] - expected[i]).cwiseAbs().maxCoeff());
        }
        EXPECT_LT(worst, 1e-8) << "max |pose difference|";
        for (const auto& [key, values] : result.residuals_after) {
            EXPECT_ALLCLOSE(values, golden::vec(item.at("residuals_after").at(key)), 1e-6, 1e-9) << key;
        }
        if (initial > 1e-12) {
            EXPECT_EQ(py::dumps(result.to_dict(), 2), item.at("to_dict_json").get<std::string>());
        } else {
            // A loop-free chain is already optimal: both errors are ~1e-25
            // round-off, so error_reduction_pct is a ratio of numerical noise
            // (it differs between any two builds). Everything else must match.
            Json actual = result.to_dict();
            Json expected = py::loads(item.at("to_dict_json").get<std::string>());
            actual.erase("error_reduction_pct");
            expected.erase("error_reduction_pct");
            EXPECT_EQ(py::dumps(actual, 2), py::dumps(expected, 2));
        }
    }
}
