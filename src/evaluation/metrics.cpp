#include "mvslam/evaluation/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.evaluation.metrics");
    return instance;
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Json f(double v, int digits) { return py::make_float(py::round(v, digits)); }

void require_same_length(const Trajectory& a, const Trajectory& b) {
    if (a.size() != b.size()) {
        throw ValueError("Trajectories must be the same length: " + std::to_string(a.size()) +
                         " vs " + std::to_string(b.size()));
    }
}

double translation_norm(const Eigen::Matrix4d& T) {
    const double x = T(0, 3);
    const double y = T(1, 3);
    const double z = T(2, 3);
    return std::sqrt(x * x + y * y + z * z);
}

/// First index at or after ``start`` where travelled distance reaches ``length``.
std::optional<std::size_t> find_segment_end(const std::vector<double>& distances, std::size_t start,
                                            double length) {
    const double target = distances[start] + length;
    if (distances.back() < target) {
        return std::nullopt;
    }
    // distances is non-decreasing, so a binary search is exact and cheap
    // (np.searchsorted(side="left")).
    const auto it = std::lower_bound(distances.begin(), distances.end(), target);
    const auto end = static_cast<std::size_t>(it - distances.begin());
    if (end < distances.size()) {
        return end;
    }
    return std::nullopt;
}

}  // namespace

// --------------------------------------------------------------------------- //
// ATE
// --------------------------------------------------------------------------- //

Json ATEResult::to_dict() const {
    Json out = Json::object();
    out["rmse"] = f(rmse, 6);
    out["mean"] = f(mean, 6);
    out["median"] = f(median, 6);
    out["std"] = f(std, 6);
    out["min"] = f(min, 6);
    out["max"] = f(max, 6);
    out["n"] = n;
    out["alignment"] = alignment;
    out["alignment_scale"] = f(alignment_scale, 6);
    return out;
}

ATEResult absolute_trajectory_error(const Trajectory& estimated, const Trajectory& reference,
                                    const std::string& alignment) {
    require_same_length(estimated, reference);
    if (estimated.empty()) {
        throw ValueError("Cannot compute ATE on an empty trajectory");
    }
    const auto [aligned, alignment_result] = align_trajectory(estimated, reference, alignment);
    std::vector<double> errors(estimated.size());
    for (std::size_t i = 0; i < errors.size(); ++i) {
        const double dx = aligned[i](0, 3) - reference[i](0, 3);
        const double dy = aligned[i](1, 3) - reference[i](1, 3);
        const double dz = aligned[i](2, 3) - reference[i](2, 3);
        errors[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    ATEResult result;
    result.rmse = np::rms(errors);
    result.mean = np::mean(errors);
    result.median = np::median(errors);
    result.std = np::std_dev(errors);
    result.min = np::min(errors);
    result.max = np::max(errors);
    result.n = static_cast<std::int64_t>(errors.size());
    result.alignment = alignment_result.mode;
    result.alignment_scale = alignment_result.scale;
    result.errors = std::move(errors);
    return result;
}

// --------------------------------------------------------------------------- //
// RPE
// --------------------------------------------------------------------------- //

Json RPEResult::to_dict() const {
    Json out = Json::object();
    out["delta"] = delta;
    out["n"] = n;
    out["trans_rmse"] = f(trans_rmse, 6);
    out["trans_mean"] = f(trans_mean, 6);
    out["trans_median"] = f(trans_median, 6);
    out["trans_max"] = f(trans_max, 6);
    out["rot_rmse_deg"] = f(rot_rmse_deg, 6);
    out["rot_mean_deg"] = f(rot_mean_deg, 6);
    out["rot_median_deg"] = f(rot_median_deg, 6);
    out["rot_max_deg"] = f(rot_max_deg, 6);
    return out;
}

RPEResult relative_pose_error(const Trajectory& estimated, const Trajectory& reference,
                              std::int64_t delta, double scale) {
    require_same_length(estimated, reference);
    if (delta < 1) {
        throw ValueError("delta must be >= 1, got " + std::to_string(delta));
    }
    const auto n_pairs = static_cast<std::int64_t>(estimated.size()) - delta;
    if (n_pairs <= 0) {
        throw ValueError("delta=" + std::to_string(delta) + " exceeds the trajectory length (" +
                         std::to_string(estimated.size()) + " poses)");
    }
    std::vector<double> trans_errors(static_cast<std::size_t>(n_pairs));
    std::vector<double> rot_errors(static_cast<std::size_t>(n_pairs));
    for (std::int64_t i = 0; i < n_pairs; ++i) {
        const auto a = static_cast<std::size_t>(i);
        const auto b = static_cast<std::size_t>(i + delta);
        const Eigen::Matrix4d gt_rel = invert_se3(reference[a]) * reference[b];
        Eigen::Matrix4d est_rel = invert_se3(estimated[a]) * estimated[b];
        if (scale != 1.0) {
            est_rel.topRightCorner<3, 1>() *= scale;
        }
        const Eigen::Matrix4d error = invert_se3(gt_rel) * est_rel;
        trans_errors[a] = translation_norm(error);
        rot_errors[a] = rotation_angle_deg(Eigen::Matrix3d(error.topLeftCorner<3, 3>()));
    }
    RPEResult result;
    result.delta = delta;
    result.n = n_pairs;
    result.trans_rmse = np::rms(trans_errors);
    result.trans_mean = np::mean(trans_errors);
    result.trans_median = np::median(trans_errors);
    result.trans_max = np::max(trans_errors);
    result.rot_rmse_deg = np::rms(rot_errors);
    result.rot_mean_deg = np::mean(rot_errors);
    result.rot_median_deg = np::median(rot_errors);
    result.rot_max_deg = np::max(rot_errors);
    return result;
}

// --------------------------------------------------------------------------- //
// KITTI drift
// --------------------------------------------------------------------------- //

const std::vector<double>& default_segment_lengths() {
    static const std::vector<double> lengths = {100.0, 200.0, 300.0, 400.0,
                                                500.0, 600.0, 700.0, 800.0};
    return lengths;
}

Json DriftResult::to_dict() const {
    Json out = Json::object();
    out["translation_pct"] = f(translation_pct, 4);
    out["rotation_deg_per_m"] = f(rotation_deg_per_m, 8);
    out["rotation_deg_per_100m"] = f(rotation_deg_per_100m, 5);
    out["n_segments"] = n_segments;
    Json lengths = Json::object();
    for (const auto& [length, values] : per_length) {  // std::map: sorted by length
        lengths[std::to_string(static_cast<std::int64_t>(length))] =
            Json::array({f(values.first, 4), f(values.second, 8)});
    }
    out["per_length"] = lengths;
    return out;
}

DriftResult kitti_drift(const Trajectory& estimated, const Trajectory& reference,
                        const std::vector<double>& segment_lengths, std::int64_t step) {
    require_same_length(estimated, reference);
    if (step < 1) {
        throw ValueError("step must be >= 1, got " + std::to_string(step));
    }
    const std::vector<double> distances = reference.cumulative_distance();
    const std::size_t n = reference.size();

    // Insertion-ordered like the Python dict (duplicate lengths share a key).
    std::vector<std::pair<double, std::vector<std::pair<double, double>>>> errors_by_length;
    auto bucket = [&](double length) -> std::vector<std::pair<double, double>>& {
        for (auto& [key, records] : errors_by_length) {
            if (key == length) {
                return records;
            }
        }
        errors_by_length.emplace_back(length, std::vector<std::pair<double, double>>{});
        return errors_by_length.back().second;
    };
    for (double length : segment_lengths) {
        (void)bucket(length);
    }

    for (std::size_t start = 0; start < n; start += static_cast<std::size_t>(step)) {
        for (double length : segment_lengths) {
            const auto end = find_segment_end(distances, start, length);
            if (!end) {
                continue;
            }
            const Eigen::Matrix4d gt_rel = invert_se3(reference[start]) * reference[*end];
            const Eigen::Matrix4d est_rel = invert_se3(estimated[start]) * estimated[*end];
            const Eigen::Matrix4d error = invert_se3(gt_rel) * est_rel;
            const double trans_error = translation_norm(error);
            const double rot_error = rotation_angle_deg(Eigen::Matrix3d(error.topLeftCorner<3, 3>()));
            bucket(length).emplace_back(trans_error / length, rot_error / length);
        }
    }

    DriftResult result;
    std::vector<double> all_trans;
    std::vector<double> all_rot;
    for (const auto& [length, records] : errors_by_length) {
        if (records.empty()) {
            continue;
        }
        std::vector<double> trans;
        std::vector<double> rot;
        for (const auto& [t, r] : records) {
            trans.push_back(t);
            rot.push_back(r);
        }
        result.per_length[length] = {100.0 * np::mean(trans), np::mean(rot)};
        all_trans.insert(all_trans.end(), trans.begin(), trans.end());
        all_rot.insert(all_rot.end(), rot.begin(), rot.end());
    }

    if (all_trans.empty()) {
        logger().warning(py::sprintf(
            "Trajectory is only %.1f m long; no sub-trajectory reaches the shortest requested "
            "length (%.0f m), so drift is undefined",
            reference.path_length(), *std::min_element(segment_lengths.begin(), segment_lengths.end())));
        DriftResult undefined;
        undefined.translation_pct = kNaN;
        undefined.rotation_deg_per_m = kNaN;
        undefined.rotation_deg_per_100m = kNaN;
        undefined.n_segments = 0;
        return undefined;
    }
    const double mean_trans_pct = 100.0 * np::mean(all_trans);
    const double mean_rot_per_m = np::mean(all_rot);
    result.translation_pct = mean_trans_pct;
    result.rotation_deg_per_m = mean_rot_per_m;
    result.rotation_deg_per_100m = mean_rot_per_m * 100.0;
    result.n_segments = static_cast<std::int64_t>(all_trans.size());
    return result;
}

// --------------------------------------------------------------------------- //
// Combined report
// --------------------------------------------------------------------------- //

Json TrajectoryEvaluation::to_dict() const {
    Json out = Json::object();
    out["label"] = label;
    out["path_length_m"] = f(path_length_m, 3);
    out["reference_path_length_m"] = f(reference_path_length_m, 3);
    out["ate_aligned"] = ate_aligned.to_dict();
    out["ate_unaligned"] = ate_unaligned.to_dict();
    Json rpe_dict = Json::object();
    for (const auto& [delta, result] : rpe) {  // std::map: sorted by delta
        rpe_dict[std::to_string(delta)] = result.to_dict();
    }
    out["rpe"] = rpe_dict;
    out["drift"] = drift.to_dict();
    return out;
}

std::string TrajectoryEvaluation::summary_line() const {
    return label + ": ATE RMSE " + py::format(ate_aligned.rmse, ".3f") + " m (" + ate_aligned.alignment +
           "-aligned), unaligned " + py::format(ate_unaligned.rmse, ".3f") + " m, drift " +
           py::format(drift.translation_pct, ".3f") + "% / " +
           py::format(drift.rotation_deg_per_100m, ".4f") + " deg per 100 m";
}

TrajectoryEvaluation evaluate_trajectory(const Trajectory& estimated, const Trajectory& reference,
                                         const std::string& label, const std::string& alignment,
                                         const std::vector<std::int64_t>& rpe_deltas,
                                         const std::vector<double>& segment_lengths) {
    TrajectoryEvaluation evaluation;
    evaluation.label = label;
    evaluation.ate_aligned = absolute_trajectory_error(estimated, reference, alignment);
    evaluation.ate_unaligned = absolute_trajectory_error(estimated, reference, "none");
    const auto [aligned, alignment_result] = align_trajectory(estimated, reference, alignment);

    for (std::int64_t delta : rpe_deltas) {
        if (delta >= static_cast<std::int64_t>(estimated.size())) {
            logger().debug("Skipping RPE delta=" + std::to_string(delta) + ": trajectory has " +
                           std::to_string(estimated.size()) + " poses");
            continue;
        }
        evaluation.rpe[delta] = relative_pose_error(aligned, reference, delta);
    }

    evaluation.drift = kitti_drift(aligned, reference, segment_lengths);
    evaluation.path_length_m = estimated.path_length();
    evaluation.reference_path_length_m = reference.path_length();
    evaluation.alignment = alignment_result;
    logger().info(evaluation.summary_line());
    return evaluation;
}

double ate_reduction_pct(double before, double after) {
    if (!std::isfinite(before) || before <= 0) {
        return kNaN;
    }
    return (before - after) / before * 100.0;
}

}  // namespace mvslam
