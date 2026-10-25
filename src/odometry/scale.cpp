#include "mvslam/odometry/scale.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.odometry.scale");
    return instance;
}

}  // namespace

Json ScaleEstimator::describe() const {
    Json out = Json::object();
    out["strategy"] = name();
    out["uses_ground_truth"] = uses_ground_truth();
    return out;
}

ConstantScale::ConstantScale(double value) : value_(value) {
    if (value <= 0) {
        throw ValueError("constant scale must be positive, got " + py::repr(value));
    }
}

Json ConstantScale::describe() const {
    Json out = ScaleEstimator::describe();
    out["value_m"] = py::make_float(value_);
    return out;
}

GroundTruthScale::GroundTruthScale(std::vector<double> scales, double min_scale, double max_scale)
    : scales_(std::move(scales)), min_scale_(min_scale), max_scale_(max_scale) {
    for (double s : scales_) {
        if (!std::isfinite(s)) {
            throw ValueError("Ground-truth scales contain non-finite values");
        }
    }
    const auto n_clipped = std::count_if(scales_.begin(), scales_.end(),
                                         [this](double s) { return s > max_scale_; });
    if (n_clipped) {
        logger().warning(py::sprintf("%ld ground-truth steps exceed max_scale=%.2f m and will be clipped",
                                     static_cast<long>(n_clipped), max_scale_));
    }
}

std::optional<double> GroundTruthScale::scale_for(std::int64_t transition_index) const {
    if (!(0 <= transition_index && transition_index < static_cast<std::int64_t>(scales_.size()))) {
        return std::nullopt;
    }
    const double value = scales_[static_cast<std::size_t>(transition_index)];
    if (value < min_scale_) {
        // The vehicle is effectively stationary; there is no motion to scale.
        return 0.0;
    }
    return std::min(std::max(value, min_scale_), max_scale_);
}

Json GroundTruthScale::describe() const {
    Json out = ScaleEstimator::describe();
    out["n_transitions"] = static_cast<std::int64_t>(scales_.size());
    out["median_step_m"] = scales_.empty() ? Json(nullptr) : py::make_float(np::median(scales_));
    return out;
}

std::unique_ptr<ScaleEstimator> build_scale_estimator(const Config& config,
                                                      const KittiOdometryDataset* dataset) {
    std::string source = config.odometry.scale_source;
    std::transform(source.begin(), source.end(), source.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto begin = source.find_first_not_of(" \t\r\n\f\v");
    const auto end = source.find_last_not_of(" \t\r\n\f\v");
    source = begin == std::string::npos ? "" : source.substr(begin, end - begin + 1);

    if (source == "none") {
        return std::make_unique<UnitScale>();
    }
    if (source == "constant") {
        return std::make_unique<ConstantScale>(config.odometry.constant_scale);
    }
    if (source == "ground_truth") {
        std::optional<std::vector<double>> scales;
        if (dataset != nullptr) {
            scales = dataset->ground_truth_scales();
        }
        if (!scales) {
            logger().warning("scale_source='ground_truth' requested but sequence " +
                             config.dataset.sequence +
                             " has no ground truth; falling back to unit scale (evaluate with sim3 "
                             "alignment)");
            return std::make_unique<UnitScale>();
        }
        logger().info(
            "Scale strategy: ground truth (EVALUATION AID -- absolute scale is not recovered from "
            "the images; see README 'Monocular scale ambiguity')");
        return std::make_unique<GroundTruthScale>(std::move(*scales), config.odometry.min_scale,
                                                  config.odometry.max_scale);
    }
    throw ValueError("Unknown scale_source '" + config.odometry.scale_source +
                     "'. Expected one of: 'ground_truth', 'constant', 'none'");
}

}  // namespace mvslam
