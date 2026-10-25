// SE(3) pose-graph representation.
//
// A pose graph is the standard back-end abstraction for SLAM: it discards the
// image measurements and keeps only the *relative pose constraints* they
// implied.
//
// * **Nodes** are keyframe poses ``T_w_kf`` -- the quantities being solved for.
// * **Odometry edges** connect consecutive keyframes with the relative motion
//   the front end integrated between them.
// * **Loop edges** connect a revisited keyframe pair with the relative pose
//   that geometric verification recovered.
//
// Without a loop edge the graph is a chain, and a chain has exactly one
// solution: the odometry itself. Loop edges are what make it a *graph* --
// they introduce cycles whose constraints are mutually inconsistent because
// of accumulated drift, and optimization redistributes that inconsistency
// over the whole trajectory instead of leaving it concentrated at the seam.
//
// Noise model assumptions
// -----------------------
// Every edge carries a diagonal Gaussian noise model over the 6-dof tangent
// space, ordered ``[rx, ry, rz, tx, ty, tz]`` to match GTSAM's Pose3
// convention (rotation first).
//
// * **Prior on the first pose.** The graph is only determined up to a global
//   rigid transform, so one node must be anchored. A very tight prior fixes
//   the gauge freedom without meaningfully constraining the solution.
// * **Odometry edges** get moderate sigmas. Frame-to-frame VO is locally
//   accurate but its errors are correlated over time; the constant diagonal
//   here is a deliberate simplification, since propagating a full covariance
//   would require uncertainty estimates the essential-matrix decomposition
//   does not provide.
// * **Loop edges** get looser translation sigmas than odometry. Their
//   rotation is well determined by wide-baseline matching, but their
//   translation *magnitude* is recovered indirectly, from a triangulated depth
//   ratio against neighbouring structure, so it deserves less confidence than
//   the rotation. A loop whose magnitude could not be recovered at all gets a
//   much larger translation sigma still, making it an orientation-only
//   constraint.
// * **Robust kernel.** Loop factors are optionally wrapped in a Huber kernel.
//   A single false-positive loop asserts that two unrelated places are the
//   same, and under a pure least-squares cost that one enormous residual can
//   drag the entire trajectory with it. Huber bounds its influence, which is
//   cheap insurance for a detector that cannot be perfect.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/geometry/pose.hpp"
#include "mvslam/loop_closure/detector.hpp"

namespace mvslam {

/// Translation sigma multiplier for a loop edge whose metric magnitude could
/// not be recovered. Large enough that the translation residual is
/// effectively unconstrained while the rotation residual still acts.
constexpr double kUnscaledLoopSigmaFactor = 100.0;

/// A pose variable in the graph.
struct PoseGraphNode {
    /// Sequential node id (also the GTSAM variable key).
    std::int64_t node_id = 0;
    /// Initial estimate of the camera-to-world pose.
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    /// Index into the processed frame sequence this node corresponds to.
    std::int64_t frame_index = 0;
    /// Original KITTI frame number.
    std::int64_t frame_id = 0;
};

/// A sequential relative-motion constraint between two nodes.
struct OdometryEdge {
    std::int64_t from_id = 0;
    std::int64_t to_id = 0;
    /// ``T_from_to``: pose of ``to`` expressed in ``from``'s frame.
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    /// Rotation sigma in radians and translation sigma in metres.
    double sigma_rot = 0.02;
    double sigma_trans = 0.10;
};

/// A loop-closure constraint, carrying its verification evidence.
struct LoopEdge : OdometryEdge {
    std::int64_t n_inliers = 0;
    double similarity = 0.0;
    bool robust = true;
};

/// Per-edge residual arrays, keyed in the Python order
/// (odometry_translation_m, odometry_rotation_deg, loop_translation_m,
/// loop_rotation_deg).
using EdgeResiduals = std::vector<std::pair<std::string, std::vector<double>>>;

/// The residual array for ``key`` (empty when absent).
const std::vector<double>& residual(const EdgeResiduals& residuals, std::string_view key);

/// Nodes plus odometry and loop edges, ready for optimization.
class PoseGraph {
public:
    std::vector<PoseGraphNode> nodes;
    std::vector<OdometryEdge> odometry_edges;
    std::vector<LoopEdge> loop_edges;
    /// Node anchored by a prior. The gauge freedom has to be fixed somewhere.
    std::int64_t prior_node_id = 0;
    double prior_sigma_rot = 1e-4;
    double prior_sigma_trans = 1e-4;

    /// Append a node (its pose re-projected onto SE(3)) and return it.
    PoseGraphNode& add_node(const Eigen::Matrix4d& pose, std::int64_t frame_index = 0,
                            std::int64_t frame_id = 0);
    /// Shape-checked variant for arbitrarily sized input (ValueError unless 4x4).
    PoseGraphNode& add_node_checked(const Eigen::MatrixXd& pose, std::int64_t frame_index = 0,
                                    std::int64_t frame_id = 0);

    OdometryEdge& add_odometry_edge(std::int64_t from_id, std::int64_t to_id, const Eigen::Matrix4d& T,
                                    double sigma_rot = 0.02, double sigma_trans = 0.10);

    LoopEdge& add_loop_edge(std::int64_t from_id, std::int64_t to_id, const Eigen::Matrix4d& T,
                            double sigma_rot = 0.05, double sigma_trans = 0.30,
                            std::int64_t n_inliers = 0, double similarity = 0.0, bool robust = true);

    std::size_t size() const { return nodes.size(); }

    /// Odometry edges followed by loop edges.
    std::vector<const OdometryEdge*> edges() const;

    /// True when every node is reachable from the prior node.
    ///
    /// A disconnected component has no path to the anchor, so its absolute
    /// pose is unconstrained and the optimizer's information matrix becomes
    /// singular. Checking up front turns a cryptic GTSAM
    /// IndeterminantLinearSystemException into an actionable message.
    bool is_connected() const;

    /// Structural problems; empty means the graph is sound.
    std::vector<std::string> validate() const;

    /// Per-edge translation (m) and rotation (deg) disagreement.
    ///
    /// Measures how far each constraint is from being satisfied by the
    /// current estimate. Loop-edge residuals before optimization are
    /// effectively a direct readout of accumulated drift around each cycle.
    EdgeResiduals edge_residuals(const std::optional<Trajectory>& trajectory = std::nullopt) const;

    Json describe() const;
    std::string repr() const;

private:
    void check_ids(std::int64_t from_id, std::int64_t to_id) const;
};

/// Sigma parameters for build_pose_graph().
struct PoseGraphSigmas {
    double prior_sigma_rot = 1e-4;
    double prior_sigma_trans = 1e-4;
    double odom_sigma_rot = 0.02;
    double odom_sigma_trans = 0.10;
    double loop_sigma_rot = 0.05;
    double loop_sigma_trans = 0.30;
    bool robust_loops = true;
};

/// Assemble a pose graph from a keyframe trajectory and verified loops.
///
/// ``trajectory`` holds one pose per graph node (keyframes, not every frame).
/// Odometry edges are derived from the trajectory itself: the relative pose
/// between consecutive nodes *is* what the front end estimated for that span.
/// ``loop_closures`` index into the same node ordering via match_id/query_id.
PoseGraph build_pose_graph(const Trajectory& trajectory, const std::vector<LoopClosure>& loop_closures,
                           const std::vector<std::int64_t>& node_frame_indices = {},
                           const PoseGraphSigmas& sigmas = {});

}  // namespace mvslam
