// GTSAM pose-graph optimization backend.
//
// Maps the framework-agnostic PoseGraph onto GTSAM's factor graph and solves
// it.
//
// What the optimizer actually does
// --------------------------------
// It minimises the sum of squared, noise-weighted residuals of every edge:
//
//     argmin_X  sum_edges || log( Z_ij^-1 (X_i^-1 X_j) ) ||^2_Sigma_ij
//
// Each residual is the SE(3) discrepancy between the measured relative pose
// ``Z_ij`` and the one implied by the current pose estimates, expressed in the
// tangent space and weighted by that edge's covariance. Levenberg-Marquardt
// solves it iteratively, damping the Gauss-Newton step so a poor
// initialisation cannot cause divergence.
//
// With only odometry edges the graph is a chain and the input trajectory is
// already the exact optimum, so nothing moves. Loop edges make it
// over-determined: the cycle they create cannot be satisfied simultaneously
// with all the odometry edges, and the solver distributes that disagreement
// over the whole cycle in proportion to each edge's uncertainty. That is
// precisely the "rubber sheet" correction that pulls a drifted trajectory
// back onto itself.
//
// Conventions
// -----------
// GTSAM's Pose3 tangent-space ordering is **rotation first**
// (``[rx, ry, rz, tx, ty, tz]``), so noise sigma vectors are built in that
// order. Rotations must be exact SO(3) members or Rot3 rejects them, which is
// why every pose passes through project_to_se3().
#pragma once

#include <cstdint>
#include <string>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/geometry/pose.hpp"
#include "mvslam/optimization/pose_graph.hpp"
#include "mvslam/utils/timing.hpp"

namespace mvslam {

/// GTSAM is a hard dependency of the C++ build, so this is always true; it is
/// kept so the pipeline reads like the Python one.
bool gtsam_available();

/// Outcome of a pose-graph optimization.
struct OptimizationResult {
    Trajectory trajectory;
    double initial_error = 0.0;
    double final_error = 0.0;
    std::int64_t iterations = 0;
    std::int64_t n_nodes = 0;
    std::int64_t n_odometry_edges = 0;
    std::int64_t n_loop_edges = 0;
    bool converged = false;
    std::string optimizer = "levenberg_marquardt";
    EdgeResiduals residuals_before;
    EdgeResiduals residuals_after;

    /// Percentage drop in the total weighted squared error.
    double error_reduction_pct() const;
    Json to_dict() const;
};

/// Convert a 4x4 transform to a GTSAM Pose3.
///
/// The rotation is re-projected onto SO(3) first: after thousands of matrix
/// products a VO pose can drift far enough from orthonormal that Rot3 rejects
/// it outright.
gtsam::Pose3 numpy_to_pose3(const Eigen::Matrix4d& T);

/// Convert a GTSAM Pose3 back to a 4x4 transform.
Eigen::Matrix4d pose3_to_numpy(const gtsam::Pose3& pose);

/// A diagonal (optionally Huber-robust) 6-dof noise model.
///
/// Sigma ordering is rotation-then-translation to match Pose3's tangent
/// space; getting this backwards silently swaps the rotational and
/// translational confidence, which produces a plausible-looking but wrong
/// solution.
gtsam::SharedNoiseModel make_noise_model(double sigma_rot, double sigma_trans, bool robust = false,
                                         double huber_k = 1.345);

struct OptimizerOptions {
    std::string optimizer = "levenberg_marquardt";
    std::int64_t max_iterations = 100;
    double relative_error_tol = 1e-5;
    double absolute_error_tol = 1e-5;
    bool robust_loop_kernel = true;
    double huber_k = 1.345;
    bool verbose = false;
};

/// Optimize ``graph`` with GTSAM and return the corrected trajectory.
///
/// Throws ValueError if the graph is structurally unsound (disconnected,
/// invalid poses). Failing here produces a clear message instead of GTSAM's
/// IndeterminantLinearSystemException deep inside the solver.
OptimizationResult optimize_pose_graph(const PoseGraph& graph, const OptimizerOptions& options = {},
                                       StageTimer* timer = nullptr);

}  // namespace mvslam
