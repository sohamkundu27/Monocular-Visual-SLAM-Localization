#include "mvslam/optimization/gtsam_backend.hpp"

#include <cmath>
#include <limits>
#include <memory>

#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.optimization.gtsam_backend");
    return instance;
}

Json summarize(const EdgeResiduals& residuals) {
    Json out = Json::object();
    for (const auto& [key, values] : residuals) {
        out["mean_" + key] = values.empty() ? Json(nullptr) : py::make_float(py::round(np::mean(values), 5));
        out["max_" + key] = values.empty() ? Json(nullptr) : py::make_float(py::round(np::max(values), 5));
    }
    return out;
}

}  // namespace

bool gtsam_available() { return true; }

double OptimizationResult::error_reduction_pct() const {
    if (!std::isfinite(initial_error) || initial_error <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (initial_error - final_error) / initial_error * 100.0;
}

Json OptimizationResult::to_dict() const {
    Json out = Json::object();
    out["optimizer"] = optimizer;
    out["n_nodes"] = n_nodes;
    out["n_odometry_edges"] = n_odometry_edges;
    out["n_loop_edges"] = n_loop_edges;
    out["iterations"] = iterations;
    out["converged"] = converged;
    out["initial_error"] = py::make_float(py::round(initial_error, 6));
    out["final_error"] = py::make_float(py::round(final_error, 6));
    out["error_reduction_pct"] = py::make_float(py::round(error_reduction_pct(), 3));
    out["residuals_before"] = summarize(residuals_before);
    out["residuals_after"] = summarize(residuals_after);
    return out;
}

gtsam::Pose3 numpy_to_pose3(const Eigen::Matrix4d& T) {
    const Eigen::Matrix4d P = project_to_se3(T);
    return gtsam::Pose3(gtsam::Rot3(Eigen::Matrix3d(P.topLeftCorner<3, 3>())),
                        gtsam::Point3(P(0, 3), P(1, 3), P(2, 3)));
}

Eigen::Matrix4d pose3_to_numpy(const gtsam::Pose3& pose) { return pose.matrix(); }

gtsam::SharedNoiseModel make_noise_model(double sigma_rot, double sigma_trans, bool robust,
                                         double huber_k) {
    gtsam::Vector6 sigmas;
    sigmas << sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans;
    gtsam::SharedNoiseModel model = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    if (robust) {
        model = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(huber_k), model);
    }
    return model;
}

OptimizationResult optimize_pose_graph(const PoseGraph& graph, const OptimizerOptions& options,
                                       StageTimer* timer_in) {
    const std::vector<std::string> problems = graph.validate();
    if (!problems.empty()) {
        std::string message = "Pose graph is not optimizable:";
        for (const auto& p : problems) {
            message += "\n  - " + p;
        }
        throw ValueError(message);
    }
    StageTimer local_timer;
    StageTimer& timer = timer_in != nullptr ? *timer_in : local_timer;
    const EdgeResiduals residuals_before = graph.edge_residuals();

    gtsam::NonlinearFactorGraph factors;
    gtsam::Values initial;

    // Anchor the gauge freedom: without a prior the whole graph can translate
    // and rotate freely, leaving the linear system rank deficient.
    const auto prior_key = static_cast<gtsam::Key>(graph.prior_node_id);
    factors.add(gtsam::PriorFactor<gtsam::Pose3>(
        prior_key, numpy_to_pose3(graph.nodes[static_cast<std::size_t>(graph.prior_node_id)].pose),
        make_noise_model(graph.prior_sigma_rot, graph.prior_sigma_trans)));

    for (const auto& node : graph.nodes) {
        initial.insert(static_cast<gtsam::Key>(node.node_id), numpy_to_pose3(node.pose));
    }
    for (const auto& edge : graph.odometry_edges) {
        factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
            static_cast<gtsam::Key>(edge.from_id), static_cast<gtsam::Key>(edge.to_id),
            numpy_to_pose3(edge.T), make_noise_model(edge.sigma_rot, edge.sigma_trans)));
    }
    for (const auto& edge : graph.loop_edges) {
        const bool use_robust = options.robust_loop_kernel && edge.robust;
        factors.add(gtsam::BetweenFactor<gtsam::Pose3>(
            static_cast<gtsam::Key>(edge.from_id), static_cast<gtsam::Key>(edge.to_id),
            numpy_to_pose3(edge.T),
            make_noise_model(edge.sigma_rot, edge.sigma_trans, use_robust, options.huber_k)));
    }

    const double initial_error = factors.error(initial);

    std::string optimizer_name;
    for (unsigned char c : options.optimizer) {
        optimizer_name.push_back(static_cast<char>(std::tolower(c)));
    }
    const auto begin = optimizer_name.find_first_not_of(" \t\r\n\f\v");
    const auto end = optimizer_name.find_last_not_of(" \t\r\n\f\v");
    optimizer_name = begin == std::string::npos ? "" : optimizer_name.substr(begin, end - begin + 1);

    gtsam::Values result_values;
    std::int64_t iterations = 0;
    if (optimizer_name == "levenberg_marquardt" || optimizer_name == "lm") {
        gtsam::LevenbergMarquardtParams params;
        params.setMaxIterations(static_cast<int>(options.max_iterations));
        params.setRelativeErrorTol(options.relative_error_tol);
        params.setAbsoluteErrorTol(options.absolute_error_tol);
        if (options.verbose) {
            params.setVerbosityLM("SUMMARY");
        }
        gtsam::LevenbergMarquardtOptimizer solver(factors, initial, params);
        {
            auto scope = timer.time("optimize");
            result_values = solver.optimize();
        }
        iterations = static_cast<std::int64_t>(solver.iterations());
    } else if (optimizer_name == "gauss_newton" || optimizer_name == "gn") {
        gtsam::GaussNewtonParams params;
        params.setMaxIterations(static_cast<int>(options.max_iterations));
        params.setRelativeErrorTol(options.relative_error_tol);
        params.setAbsoluteErrorTol(options.absolute_error_tol);
        if (options.verbose) {
            params.setVerbosity("SUMMARY");
        }
        gtsam::GaussNewtonOptimizer solver(factors, initial, params);
        {
            auto scope = timer.time("optimize");
            result_values = solver.optimize();
        }
        iterations = static_cast<std::int64_t>(solver.iterations());
    } else {
        throw ValueError("Unknown optimizer '" + options.optimizer +
                         "'. Choose 'levenberg_marquardt' or 'gauss_newton'");
    }

    const double final_error = factors.error(result_values);

    PoseList poses;
    std::vector<std::int64_t> frame_ids;
    for (const auto& node : graph.nodes) {
        poses.push_back(pose3_to_numpy(result_values.at<gtsam::Pose3>(static_cast<gtsam::Key>(node.node_id))));
        frame_ids.push_back(node.frame_id);
    }

    OptimizationResult result;
    result.trajectory = Trajectory(std::move(poses), std::move(frame_ids)).orthonormalized();
    result.residuals_after = graph.edge_residuals(result.trajectory);
    result.initial_error = initial_error;
    result.final_error = final_error;
    result.iterations = iterations;
    result.n_nodes = static_cast<std::int64_t>(graph.nodes.size());
    result.n_odometry_edges = static_cast<std::int64_t>(graph.odometry_edges.size());
    result.n_loop_edges = static_cast<std::int64_t>(graph.loop_edges.size());
    result.converged = iterations < options.max_iterations;
    result.optimizer = optimizer_name;
    result.residuals_before = residuals_before;

    logger().info(py::sprintf(
        "Pose graph optimized: %lld nodes, %lld loop edges, %lld iterations, error %.4g -> %.4g (%.1f%% reduction)",
        static_cast<long long>(result.n_nodes), static_cast<long long>(result.n_loop_edges),
        static_cast<long long>(result.iterations), result.initial_error, result.final_error,
        result.error_reduction_pct()));
    const auto& loop_before = residual(residuals_before, "loop_translation_m");
    const auto& loop_after = residual(result.residuals_after, "loop_translation_m");
    if (!loop_before.empty()) {
        logger().info(py::sprintf("Loop edge residual: %.3f m -> %.3f m (mean)", np::mean(loop_before),
                                  np::mean(loop_after)));
    }
    return result;
}

}  // namespace mvslam
