#include "mvslam/odometry/visual_odometry.hpp"

#include <cmath>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/epipolar.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.odometry.visual_odometry");
    return instance;
}

/// OdometryResult._mean: mean over finite values, NaN when there are none.
template <typename Getter>
double finite_mean(const std::vector<FrameDiagnostics>& diagnostics, bool only_successful,
                   Getter get) {
    std::vector<double> values;
    for (const auto& d : diagnostics) {
        if (only_successful && !d.succeeded()) {
            continue;
        }
        const double v = get(d);
        if (std::isfinite(v)) {
            values.push_back(v);
        }
    }
    return values.empty() ? std::numeric_limits<double>::quiet_NaN() : np::mean(values);
}

}  // namespace

std::vector<std::pair<std::string, std::string>> FrameDiagnostics::to_row() const {
    return {
        {"index", std::to_string(index)},
        {"frame_id", std::to_string(frame_id)},
        {"status", status},
        {"reason", reason},
        {"n_keypoints", std::to_string(n_keypoints)},
        {"n_raw_matches", std::to_string(n_raw_matches)},
        {"n_matches", std::to_string(n_matches)},
        {"n_ransac_inliers", std::to_string(n_ransac_inliers)},
        {"n_cheirality_inliers", std::to_string(n_cheirality_inliers)},
        {"ransac_inlier_ratio", py::repr(ransac_inlier_ratio)},
        {"inlier_ratio", py::repr(inlier_ratio)},
        {"rotation_deg", py::repr(rotation_deg)},
        {"parallax_deg", py::repr(parallax_deg)},
        {"flow_px", py::repr(flow_px)},
        {"homography_ratio", py::repr(homography_ratio)},
        {"degenerate", py::repr(degenerate)},
        {"scale_m", py::repr(scale_m)},
        {"detect_ms", py::repr(detect_ms)},
        {"match_ms", py::repr(match_ms)},
        {"pose_ms", py::repr(pose_ms)},
    };
}

// --------------------------------------------------------------------------- //
// OdometryResult
// --------------------------------------------------------------------------- //

std::int64_t OdometryResult::n_successful() const {
    std::int64_t count = 0;
    for (const auto& d : diagnostics) {
        count += d.succeeded() ? 1 : 0;
    }
    return count;
}

double OdometryResult::successful_pose_rate_pct() const {
    return n_transitions() ? 100.0 * static_cast<double>(n_successful()) /
                                 static_cast<double>(n_transitions())
                           : 0.0;
}

double OdometryResult::avg_features_per_frame() const {
    return finite_mean(diagnostics, false,
                       [](const FrameDiagnostics& d) { return static_cast<double>(d.n_keypoints); });
}

double OdometryResult::avg_matches_per_pair() const {
    return finite_mean(diagnostics, false,
                       [](const FrameDiagnostics& d) { return static_cast<double>(d.n_matches); });
}

double OdometryResult::avg_ransac_inliers() const {
    return finite_mean(diagnostics, false, [](const FrameDiagnostics& d) {
        return static_cast<double>(d.n_ransac_inliers);
    });
}

double OdometryResult::avg_inlier_ratio_pct() const {
    return 100.0 * finite_mean(diagnostics, true,
                               [](const FrameDiagnostics& d) { return d.ransac_inlier_ratio; });
}

double OdometryResult::avg_cheirality_ratio_pct() const {
    return 100.0 *
           finite_mean(diagnostics, true, [](const FrameDiagnostics& d) { return d.inlier_ratio; });
}

std::int64_t OdometryResult::n_degenerate() const {
    std::int64_t count = 0;
    for (const auto& d : diagnostics) {
        count += d.degenerate ? 1 : 0;
    }
    return count;
}

Json OdometryResult::feature_statistics() const {
    Json out = Json::object();
    out["avg_features_per_frame"] = py::make_float(py::round(avg_features_per_frame(), 1));
    out["avg_matches_per_pair"] = py::make_float(py::round(avg_matches_per_pair(), 1));
    out["avg_ransac_inliers"] = py::make_float(py::round(avg_ransac_inliers(), 1));
    out["avg_inlier_ratio_pct"] = py::make_float(py::round(avg_inlier_ratio_pct(), 2));
    out["avg_cheirality_ratio_pct"] = py::make_float(py::round(avg_cheirality_ratio_pct(), 2));
    return out;
}

Json OdometryResult::status_counts() const {
    Json counts = Json::object();
    for (const auto& d : diagnostics) {
        if (counts.contains(d.status)) {
            counts[d.status] = counts[d.status].get<std::int64_t>() + 1;
        } else {
            counts[d.status] = 1;
        }
    }
    return counts;
}

// --------------------------------------------------------------------------- //
// VisualOdometry
// --------------------------------------------------------------------------- //

VisualOdometry::VisualOdometry(const Eigen::Matrix3d& K, FeatureDetector detector,
                               FeatureMatcher matcher,
                               std::unique_ptr<ScaleEstimator> scale_estimator,
                               const VisualOdometryParams& params, StageTimer* timer)
    : K_(K),
      detector_(std::move(detector)),
      matcher_(std::move(matcher)),
      scale_estimator_(std::move(scale_estimator)),
      params_(params) {
    if (timer == nullptr) {
        owned_timer_ = std::make_unique<StageTimer>();
        timer_ = owned_timer_.get();
    } else {
        timer_ = timer;
    }
}

VisualOdometry VisualOdometry::from_config(const Config& config, const KittiOdometryDataset& dataset,
                                           StageTimer* timer) {
    FeatureDetector detector = FeatureDetector::from_config(config);
    FeatureMatcher matcher = FeatureMatcher::from_config(config, detector.descriptor_norm());
    const auto& o = config.odometry;
    // Keyword arguments are evaluated in order in the Python factory: K (which
    // loads and logs the calibration) before the scale estimator (which loads
    // and logs ground truth).
    const Eigen::Matrix3d K = dataset.K();
    std::unique_ptr<ScaleEstimator> scale = build_scale_estimator(config, &dataset);
    VisualOdometryParams params;
    params.ransac_method = o.ransac_method;
    params.ransac_threshold_px = o.ransac_threshold_px;
    params.ransac_confidence = o.ransac_confidence;
    params.ransac_max_iters = o.ransac_max_iters;
    params.cheirality_distance = o.cheirality_distance;
    params.min_matches = config.matcher.min_matches;
    params.min_inliers = o.min_inliers;
    params.min_inlier_ratio = o.min_inlier_ratio;
    params.max_rotation_deg = o.max_rotation_deg;
    params.min_parallax_deg = o.min_parallax_deg;
    params.min_flow_px = o.min_flow_px;
    params.homography_ratio_threshold = o.homography_ratio_threshold;
    params.reject_degenerate = o.reject_degenerate;
    return VisualOdometry(K, std::move(detector), std::move(matcher), std::move(scale), params, timer);
}

std::optional<Eigen::Matrix4d> VisualOdometry::estimate_relative_pose(
    const MatchResult& matches, FrameDiagnostics& diagnostics) const {
    diagnostics.n_raw_matches = matches.n_raw;
    diagnostics.n_matches = static_cast<std::int64_t>(matches.size());

    if (static_cast<std::int64_t>(matches.size()) < params_.min_matches) {
        diagnostics.status = kStatusFailed;
        diagnostics.reason = "too_few_matches(" + std::to_string(matches.size()) + ")";
        return std::nullopt;
    }

    const double flow = median_flow_px(matches.points_a, matches.points_b);
    diagnostics.flow_px = flow;
    if (std::isfinite(flow) && flow < params_.min_flow_px) {
        // The camera has not moved: identity is the right answer, and
        // attempting to decompose a degenerate E would inject pure noise.
        diagnostics.status = kStatusStationary;
        diagnostics.reason = "low_flow(" + py::format(flow, ".2f") + "px)";
        return Eigen::Matrix4d::Identity();
    }

    const EssentialMatrixResult essential = estimate_essential_matrix(
        matches.points_a, matches.points_b, K_, params_.ransac_threshold_px,
        params_.ransac_confidence, static_cast<int>(params_.ransac_max_iters), 8,
        params_.ransac_method);
    diagnostics.n_ransac_inliers = essential.n_inliers();
    diagnostics.ransac_inlier_ratio = essential.inlier_ratio();
    if (!essential.ok) {
        diagnostics.status = kStatusFailed;
        diagnostics.reason = essential.reason;
        return std::nullopt;
    }

    const ModelSelection selection =
        select_two_view_model(matches.points_a, matches.points_b, K_, essential.E,
                              params_.ransac_threshold_px, params_.homography_ratio_threshold);
    diagnostics.homography_ratio = selection.ratio_h;
    diagnostics.degenerate = selection.is_degenerate;
    if (selection.is_degenerate && params_.reject_degenerate) {
        diagnostics.status = kStatusFailed;
        diagnostics.reason = "degenerate_geometry(ratio_h=" + py::format(selection.ratio_h, ".2f") + ")";
        return std::nullopt;
    }

    PoseGates gates;
    gates.min_inliers = params_.min_inliers;
    gates.min_inlier_ratio = params_.min_inlier_ratio;
    gates.max_rotation_deg = params_.max_rotation_deg;
    gates.min_parallax_deg = params_.min_parallax_deg;
    gates.cheirality_distance = params_.cheirality_distance;
    const RelativePoseResult pose = recover_relative_pose(essential.E, matches.points_a,
                                                          matches.points_b, K_, essential.inlier_mask,
                                                          gates);
    diagnostics.n_cheirality_inliers = pose.n_cheirality_inliers;
    diagnostics.inlier_ratio = essential.n_inliers()
                                   ? static_cast<double>(pose.n_cheirality_inliers) /
                                         static_cast<double>(essential.n_inliers())
                                   : 0.0;
    diagnostics.rotation_deg = pose.rotation_deg;
    diagnostics.parallax_deg = pose.median_parallax_deg;

    if (!pose.ok) {
        diagnostics.status = kStatusFailed;
        diagnostics.reason = pose.reason;
        return std::nullopt;
    }
    diagnostics.status = kStatusOk;
    return pose.T;
}

OdometryResult VisualOdometry::run(const KittiOdometryDataset& dataset, const FrameCallback& on_frame,
                                   std::int64_t progress_interval) {
    const auto n = static_cast<std::int64_t>(dataset.size());
    if (n < 2) {
        throw ValueError("Need at least 2 frames for odometry, got " + std::to_string(n));
    }

    timer_->reset_wall_clock();

    PoseList poses{Eigen::Matrix4d::Identity()};
    PoseList relative_poses;
    std::vector<FrameDiagnostics> diagnostics;
    poses.reserve(static_cast<std::size_t>(n));
    relative_poses.reserve(static_cast<std::size_t>(n));
    diagnostics.reserve(static_cast<std::size_t>(n));

    std::optional<Frame> previous;
    {
        cv::Mat first = dataset.load_image(0);
        {
            auto scope = timer_->time("detect");
            previous = detector_.detect(first, 0, dataset.frame_ids()[0]);
        }
        if (on_frame) {
            on_frame(0, *previous, first, poses[0]);
        }
    }

    std::optional<Eigen::Matrix4d> last_relative_direction;

    for (std::int64_t i = 1; i < n; ++i) {
        const cv::Mat image = dataset.load_image(static_cast<std::size_t>(i));
        const std::int64_t frame_id = dataset.frame_ids()[static_cast<std::size_t>(i)];
        FrameDiagnostics diag;
        diag.index = i;
        diag.frame_id = frame_id;
        diag.status = kStatusFailed;

        const double t_detect = timer_->stats("detect").total_s;
        std::optional<Frame> current;
        {
            auto scope = timer_->time("detect");
            current = detector_.detect(image, i, frame_id);
        }
        diag.detect_ms = (timer_->stats("detect").total_s - t_detect) * 1000.0;
        diag.n_keypoints = static_cast<std::int64_t>(current->size());

        const double t_match = timer_->stats("match").total_s;
        MatchResult matches;
        {
            auto scope = timer_->time("match");
            matches = matcher_.match(*previous, *current);
        }
        diag.match_ms = (timer_->stats("match").total_s - t_match) * 1000.0;

        const double t_pose = timer_->stats("pose").total_s;
        std::optional<Eigen::Matrix4d> relative;
        {
            auto scope = timer_->time("pose");
            relative = estimate_relative_pose(matches, diag);
        }
        diag.pose_ms = (timer_->stats("pose").total_s - t_pose) * 1000.0;

        const std::optional<double> scale = scale_estimator_->scale_for(i - 1);
        diag.scale_m = scale ? *scale : std::numeric_limits<double>::quiet_NaN();

        const Eigen::Matrix4d step = resolve_step(relative, scale, diag, last_relative_direction);
        if (diag.status == kStatusOk && relative) {
            last_relative_direction = relative;
        }

        poses.push_back(project_to_se3(poses.back() * step));
        relative_poses.push_back(step);
        diagnostics.push_back(diag);
        previous = std::move(current);

        if (on_frame) {
            on_frame(i, *previous, image, poses.back());
        }

        if (progress_interval && i % progress_interval == 0) {
            std::int64_t ok = 0;
            for (const auto& d : diagnostics) {
                ok += d.succeeded() ? 1 : 0;
            }
            logger().info(py::sprintf(
                "frame %lld/%lld | success %.1f%% | matches %lld | inliers %lld | %.1f fps",
                static_cast<long long>(i), static_cast<long long>(n - 1),
                100.0 * static_cast<double>(ok) / static_cast<double>(diagnostics.size()),
                static_cast<long long>(diag.n_matches), static_cast<long long>(diag.n_cheirality_inliers),
                timer_->fps(i + 1)));
        }
    }

    OdometryResult result;
    result.trajectory = Trajectory(std::move(poses), dataset.frame_ids(), dataset.timestamps());
    result.diagnostics = std::move(diagnostics);
    result.relative_poses = std::move(relative_poses);
    result.timer = timer_;
    result.scale_strategy = scale_estimator_->describe();
    logger().info(py::sprintf("Visual odometry complete: %lld transitions, %.1f%% successful, %s",
                              static_cast<long long>(result.n_transitions()),
                              result.successful_pose_rate_pct(),
                              py::repr(result.status_counts()).c_str()));
    return result;
}

Eigen::Matrix4d VisualOdometry::resolve_step(const std::optional<Eigen::Matrix4d>& relative,
                                             const std::optional<double>& scale,
                                             FrameDiagnostics& diagnostics,
                                             const std::optional<Eigen::Matrix4d>& last_direction) const {
    if (diagnostics.status == kStatusStationary) {
        return Eigen::Matrix4d::Identity();
    }
    if (relative && scale) {
        const Eigen::Matrix3d R = relative->topLeftCorner<3, 3>();
        const Eigen::Vector3d t = relative->topRightCorner<3, 1>() * *scale;
        return se3_from_rt(R, t);
    }
    if (relative && !scale) {
        // Geometry succeeded but the scale source has nothing to say.
        diagnostics.status = kStatusFailed;
        std::string reason = diagnostics.reason + "|no_scale";
        const auto first = reason.find_first_not_of('|');
        diagnostics.reason = first == std::string::npos ? "" : reason.substr(first);
    }
    // Constant-velocity prediction from the last successful motion.
    if (last_direction) {
        const Eigen::Matrix3d R = last_direction->topLeftCorner<3, 3>();
        const double magnitude = scale ? *scale : 0.0;
        return se3_from_rt(R, last_direction->topRightCorner<3, 1>() * magnitude);
    }
    return Eigen::Matrix4d::Identity();
}

}  // namespace mvslam
