#include "mvslam/optimization/pose_graph.hpp"

#include <cmath>
#include <set>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.optimization.pose_graph");
    return instance;
}

std::string edge_name(const OdometryEdge& edge) {
    return "edge " + std::to_string(edge.from_id) + "->" + std::to_string(edge.to_id);
}

}  // namespace

const std::vector<double>& residual(const EdgeResiduals& residuals, std::string_view key) {
    static const std::vector<double> empty;
    for (const auto& [name, values] : residuals) {
        if (name == key) {
            return values;
        }
    }
    return empty;
}

PoseGraphNode& PoseGraph::add_node(const Eigen::Matrix4d& pose, std::int64_t frame_index,
                                   std::int64_t frame_id) {
    PoseGraphNode node;
    node.node_id = static_cast<std::int64_t>(nodes.size());
    node.pose = project_to_se3(pose);
    node.frame_index = frame_index;
    node.frame_id = frame_id;
    nodes.push_back(node);
    return nodes.back();
}

PoseGraphNode& PoseGraph::add_node_checked(const Eigen::MatrixXd& pose, std::int64_t frame_index,
                                           std::int64_t frame_id) {
    if (pose.rows() != 4 || pose.cols() != 4) {
        throw ValueError("Node pose must be 4x4, got (" + std::to_string(pose.rows()) + ", " +
                         std::to_string(pose.cols()) + ")");
    }
    return add_node(Eigen::Matrix4d(pose), frame_index, frame_id);
}

void PoseGraph::check_ids(std::int64_t from_id, std::int64_t to_id) const {
    for (std::int64_t node_id : {from_id, to_id}) {
        if (!(0 <= node_id && node_id < static_cast<std::int64_t>(nodes.size()))) {
            throw IndexError("Node id " + std::to_string(node_id) + " out of range (0.." +
                             std::to_string(static_cast<std::int64_t>(nodes.size()) - 1) + ")");
        }
    }
}

OdometryEdge& PoseGraph::add_odometry_edge(std::int64_t from_id, std::int64_t to_id,
                                           const Eigen::Matrix4d& T, double sigma_rot,
                                           double sigma_trans) {
    check_ids(from_id, to_id);
    odometry_edges.push_back({from_id, to_id, project_to_se3(T), sigma_rot, sigma_trans});
    return odometry_edges.back();
}

LoopEdge& PoseGraph::add_loop_edge(std::int64_t from_id, std::int64_t to_id, const Eigen::Matrix4d& T,
                                   double sigma_rot, double sigma_trans, std::int64_t n_inliers,
                                   double similarity, bool robust) {
    check_ids(from_id, to_id);
    LoopEdge edge;
    edge.from_id = from_id;
    edge.to_id = to_id;
    edge.T = project_to_se3(T);
    edge.sigma_rot = sigma_rot;
    edge.sigma_trans = sigma_trans;
    edge.n_inliers = n_inliers;
    edge.similarity = similarity;
    edge.robust = robust;
    loop_edges.push_back(edge);
    return loop_edges.back();
}

std::vector<const OdometryEdge*> PoseGraph::edges() const {
    std::vector<const OdometryEdge*> out;
    for (const auto& e : odometry_edges) {
        out.push_back(&e);
    }
    for (const auto& e : loop_edges) {
        out.push_back(&e);
    }
    return out;
}

bool PoseGraph::is_connected() const {
    if (nodes.empty()) {
        return true;
    }
    if (!(0 <= prior_node_id && prior_node_id < static_cast<std::int64_t>(nodes.size()))) {
        // Nothing is reachable from an anchor that does not exist.
        return false;
    }
    std::vector<std::vector<std::int64_t>> adjacency(nodes.size());
    for (const OdometryEdge* edge : edges()) {
        adjacency.at(static_cast<std::size_t>(edge->from_id)).push_back(edge->to_id);
        adjacency.at(static_cast<std::size_t>(edge->to_id)).push_back(edge->from_id);
    }
    std::set<std::int64_t> seen = {prior_node_id};
    std::vector<std::int64_t> stack = {prior_node_id};
    while (!stack.empty()) {
        const std::int64_t current = stack.back();
        stack.pop_back();
        for (std::int64_t neighbour : adjacency[static_cast<std::size_t>(current)]) {
            if (seen.insert(neighbour).second) {
                stack.push_back(neighbour);
            }
        }
    }
    return seen.size() == nodes.size();
}

std::vector<std::string> PoseGraph::validate() const {
    std::vector<std::string> problems;
    if (nodes.empty()) {
        problems.emplace_back("graph has no nodes");
        return problems;
    }
    if (!(0 <= prior_node_id && prior_node_id < static_cast<std::int64_t>(nodes.size()))) {
        problems.push_back("prior_node_id " + std::to_string(prior_node_id) + " is not a valid node");
    }
    for (const auto& node : nodes) {
        if (!is_valid_se3(node.pose, 1e-4)) {
            problems.push_back("node " + std::to_string(node.node_id) +
                               " pose is not a valid SE(3) member");
        }
    }
    for (const OdometryEdge* edge : edges()) {
        if (!is_valid_se3(edge->T, 1e-4)) {
            problems.push_back(edge_name(*edge) + " transform is not valid SE(3)");
        }
        if (edge->from_id == edge->to_id) {
            problems.push_back(edge_name(*edge) + " is a self-loop");
        }
        if (edge->sigma_rot <= 0 || edge->sigma_trans <= 0) {
            problems.push_back(edge_name(*edge) + " has a non-positive sigma");
        }
    }
    if (!is_connected()) {
        problems.emplace_back("graph is disconnected; some poses are unconstrained");
    }
    return problems;
}

EdgeResiduals PoseGraph::edge_residuals(const std::optional<Trajectory>& trajectory) const {
    PoseList poses;
    if (trajectory) {
        poses = trajectory->poses();
    } else {
        for (const auto& node : nodes) {
            poses.push_back(node.pose);
        }
    }
    EdgeResiduals result;
    auto compute = [&](const std::string& name, const auto& edge_list) {
        std::vector<double> translations;
        std::vector<double> rotations;
        for (const auto& edge : edge_list) {
            const Eigen::Matrix4d predicted =
                invert_se3(poses.at(static_cast<std::size_t>(edge.from_id))) *
                poses.at(static_cast<std::size_t>(edge.to_id));
            const Eigen::Matrix4d error = invert_se3(edge.T) * predicted;
            const double x = error(0, 3);
            const double y = error(1, 3);
            const double z = error(2, 3);
            translations.push_back(std::sqrt(x * x + y * y + z * z));
            rotations.push_back(rotation_angle_deg(Eigen::Matrix3d(error.topLeftCorner<3, 3>())));
        }
        result.emplace_back(name + "_translation_m", translations);
        result.emplace_back(name + "_rotation_deg", rotations);
    };
    compute("odometry", odometry_edges);
    compute("loop", loop_edges);
    return result;
}

Json PoseGraph::describe() const {
    const EdgeResiduals residuals = edge_residuals();
    const std::vector<double>& loop_translation = residual(residuals, "loop_translation_m");
    Json out = Json::object();
    out["n_nodes"] = static_cast<std::int64_t>(nodes.size());
    out["n_odometry_edges"] = static_cast<std::int64_t>(odometry_edges.size());
    out["n_loop_edges"] = static_cast<std::int64_t>(loop_edges.size());
    out["connected"] = is_connected();
    out["mean_loop_residual_m"] =
        loop_translation.empty() ? Json(nullptr) : py::make_float(py::round(np::mean(loop_translation), 4));
    out["max_loop_residual_m"] =
        loop_translation.empty() ? Json(nullptr) : py::make_float(py::round(np::max(loop_translation), 4));
    return out;
}

std::string PoseGraph::repr() const {
    return "PoseGraph(nodes=" + std::to_string(nodes.size()) + ", odometry=" +
           std::to_string(odometry_edges.size()) + ", loops=" + std::to_string(loop_edges.size()) + ")";
}

PoseGraph build_pose_graph(const Trajectory& trajectory, const std::vector<LoopClosure>& loop_closures,
                           const std::vector<std::int64_t>& node_frame_indices,
                           const PoseGraphSigmas& sigmas) {
    PoseGraph graph;
    graph.prior_sigma_rot = sigmas.prior_sigma_rot;
    graph.prior_sigma_trans = sigmas.prior_sigma_trans;

    for (std::size_t i = 0; i < trajectory.size(); ++i) {
        const std::int64_t frame_index =
            i < node_frame_indices.size() ? node_frame_indices[i] : static_cast<std::int64_t>(i);
        graph.add_node(trajectory[i], frame_index, trajectory.frame_ids()[i]);
    }
    for (std::size_t i = 0; i + 1 < trajectory.size(); ++i) {
        const Eigen::Matrix4d relative = invert_se3(trajectory[i]) * trajectory[i + 1];
        graph.add_odometry_edge(static_cast<std::int64_t>(i), static_cast<std::int64_t>(i + 1), relative,
                                sigmas.odom_sigma_rot, sigmas.odom_sigma_trans);
    }

    std::int64_t n_unscaled = 0;
    for (const LoopClosure& closure : loop_closures) {
        // A loop whose metric magnitude could not be recovered still carries
        // a trustworthy *rotation*. Rather than discard it, or assert a
        // translation that was never measured, its translation sigma is
        // inflated so the factor acts as an orientation constraint. Heading
        // drift is the dominant error term, so this is worth keeping.
        const bool scale_measured = closure.scale_is_measured();
        const double sigma_trans = scale_measured ? sigmas.loop_sigma_trans
                                                  : sigmas.loop_sigma_trans * kUnscaledLoopSigmaFactor;
        n_unscaled += scale_measured ? 0 : 1;
        graph.add_loop_edge(closure.match_id, closure.query_id, closure.T_match_query,
                            sigmas.loop_sigma_rot, sigma_trans, closure.n_inliers, closure.similarity,
                            sigmas.robust_loops);
    }
    if (n_unscaled) {
        logger().info(py::sprintf("%lld of %zu loop edges have no measured translation magnitude and were "
                                  "added as orientation-only constraints",
                                  static_cast<long long>(n_unscaled), loop_closures.size()));
    }
    logger().info(py::sprintf("Built pose graph: %zu nodes, %zu odometry edges, %zu loop edges",
                              graph.nodes.size(), graph.odometry_edges.size(), graph.loop_edges.size()));
    return graph;
}

}  // namespace mvslam
