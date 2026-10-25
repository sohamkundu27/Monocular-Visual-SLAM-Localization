#include "mvslam/evaluation/alignment.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.evaluation.alignment");
    return instance;
}

std::string shape(const Eigen::MatrixXd& m) {
    return "(" + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + ")";
}

// Row-major copy of an (N, 3) matrix, so NumPy-order reductions apply.
std::vector<double> row_major(const Eigen::MatrixXd& m) {
    std::vector<double> out(static_cast<std::size_t>(m.size()));
    for (Eigen::Index r = 0; r < m.rows(); ++r) {
        for (Eigen::Index c = 0; c < m.cols(); ++c) {
            out[static_cast<std::size_t>(r * m.cols() + c)] = m(r, c);
        }
    }
    return out;
}

}  // namespace

const std::vector<std::string>& alignment_modes() {
    static const std::vector<std::string> modes = {"sim3", "se3", "none"};
    return modes;
}

Eigen::Matrix4d AlignmentResult::T() const { return se3_from_rt(R, t); }

Points3 AlignmentResult::apply(const Points3& points) const {
    Points3 out(points.rows(), 3);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        const Eigen::Vector3d p = points.row(i).transpose();
        out.row(i) = (scale * (R * p) + t).transpose();
    }
    return out;
}

Trajectory AlignmentResult::apply_to_trajectory(const Trajectory& trajectory) const {
    return trajectory.transformed(T(), scale);
}

AlignmentResult AlignmentResult::identity(const std::string& mode) {
    AlignmentResult out;
    out.mode = mode;
    return out;
}

Json AlignmentResult::describe() const {
    Json out = Json::object();
    out["mode"] = mode;
    out["scale"] = py::make_float(py::round(scale, 6));
    Json translation = Json::array();
    for (int i = 0; i < 3; ++i) {
        translation.push_back(py::make_float(py::round(t(i), 4)));
    }
    out["translation_m"] = translation;
    return out;
}

AlignmentResult umeyama_alignment(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
                                  bool with_scale) {
    if (source.rows() != target.rows() || source.cols() != target.cols()) {
        throw ValueError("Shape mismatch: source " + shape(source) + " vs target " + shape(target));
    }
    if (source.cols() != 3) {
        throw ValueError("Points must be (N, 3), got " + shape(source));
    }
    const auto n = static_cast<std::size_t>(source.rows());
    if (n < 3) {
        throw ValueError("Umeyama alignment needs at least 3 points, got " + std::to_string(n));
    }

    const std::vector<double> src = row_major(source);
    const std::vector<double> tgt = row_major(target);
    const std::vector<double> ms = np::mean_axis0(src.data(), n, 3);
    const std::vector<double> mt = np::mean_axis0(tgt.data(), n, 3);
    const Eigen::Vector3d mean_source(ms[0], ms[1], ms[2]);
    const Eigen::Vector3d mean_target(mt[0], mt[1], mt[2]);

    Eigen::MatrixXd centered_source = source;
    Eigen::MatrixXd centered_target = target;
    centered_source.rowwise() -= mean_source.transpose();
    centered_target.rowwise() -= mean_target.transpose();

    // Cross-covariance of the centred sets.
    const Eigen::Matrix3d covariance =
        (centered_target.transpose() * centered_source) / static_cast<double>(n);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d Vt = svd.matrixV().transpose();
    const Eigen::Vector3d singular_values = svd.singularValues();

    // Guard against producing a reflection instead of a rotation.
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    if (U.determinant() * Vt.determinant() < 0) {
        S(2, 2) = -1.0;
    }
    const Eigen::Matrix3d R = U * S * Vt;

    double scale = 1.0;
    if (with_scale) {
        std::vector<double> squared = row_major(centered_source);
        for (double& v : squared) {
            v = v * v;
        }
        const double variance_source = np::sum(squared) / static_cast<double>(n);
        if (variance_source < 1e-12) {
            // A stationary source has no extent to scale.
            scale = 1.0;
        } else {
            const double trace = singular_values(0) * S(0, 0) + singular_values(1) * S(1, 1) +
                                 singular_values(2) * S(2, 2);
            scale = trace / variance_source;
        }
    }

    AlignmentResult result;
    result.R = R;
    result.t = mean_target - scale * (R * mean_source);
    result.scale = scale;
    result.mode = with_scale ? "sim3" : "se3";
    return result;
}

std::pair<Trajectory, AlignmentResult> align_trajectory(const Trajectory& estimated,
                                                        const Trajectory& reference,
                                                        const std::string& mode_in) {
    std::string mode;
    for (unsigned char c : mode_in) {
        mode.push_back(static_cast<char>(std::tolower(c)));
    }
    const auto begin = mode.find_first_not_of(" \t\r\n\f\v");
    const auto end = mode.find_last_not_of(" \t\r\n\f\v");
    mode = begin == std::string::npos ? "" : mode.substr(begin, end - begin + 1);
    const auto& modes = alignment_modes();
    if (std::find(modes.begin(), modes.end(), mode) == modes.end()) {
        throw ValueError("Unknown alignment mode '" + mode + "'. Choose from ['sim3', 'se3', 'none']");
    }
    if (estimated.size() != reference.size()) {
        throw ValueError("Trajectories must have equal length to align: " +
                         std::to_string(estimated.size()) + " vs " + std::to_string(reference.size()));
    }
    if (mode == "none") {
        return {estimated.copy(), AlignmentResult::identity()};
    }
    const AlignmentResult result =
        umeyama_alignment(estimated.positions(), reference.positions(), mode == "sim3");
    logger().info(py::sprintf("Umeyama %s alignment: scale=%.4f, |t|=%.3f m", mode.c_str(),
                              result.scale, result.t.norm()));
    return {result.apply_to_trajectory(estimated), result};
}

}  // namespace mvslam
