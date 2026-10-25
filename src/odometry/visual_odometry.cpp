#include "mvslam/odometry/visual_odometry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <thread>

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

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

/// Front-end results for one frame: everything that depends only on the
/// images, and not on the trajectory accumulated so far.
struct FrontEndWork {
    cv::Mat image;
    std::optional<Frame> frame;
    MatchResult matches;
    std::optional<Eigen::Matrix4d> relative;
    FrameDiagnostics diag;
    double detect_s = 0.0;
    double match_s = 0.0;
    double pose_s = 0.0;
    // A failure is held back and rethrown when the sequential pass reaches
    // this frame, so earlier frames are still delivered first, as in the
    // one-frame-at-a-time loop.
    std::exception_ptr error;
};

/// Worker threads for the front end: MVSLAM_NUM_THREADS when set (1 gives
/// the plain sequential loop), otherwise the hardware concurrency.
std::size_t front_end_threads() {
    if (const char* env = std::getenv("MVSLAM_NUM_THREADS")) {
        const std::optional<std::int64_t> parsed = py::parse_int(env);
        if (parsed && *parsed > 0) {
            return static_cast<std::size_t>(*parsed);
        }
    }
    return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

/// Run ``fn(k)`` for k in [0, n) on up to ``threads`` threads. ``fn`` must
/// not throw (FrontEndWork captures failures itself).
template <typename Fn>
void parallel_for_index(std::size_t n, std::size_t threads, const Fn& fn) {
    const std::size_t workers = std::min(threads, n);
    if (workers <= 1) {
        for (std::size_t k = 0; k < n; ++k) {
            fn(k);
        }
        return;
    }
    std::atomic<std::size_t> next{0};
    auto drain = [&]() {
        for (std::size_t k = next++; k < n; k = next++) {
            fn(k);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (std::size_t w = 1; w < workers; ++w) {
        pool.emplace_back(drain);
    }
    drain();
    for (auto& t : pool) {
        t.join();
    }
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

    // Detection, matching and two-view pose estimation for a transition
    // depend only on the two images, never on the trajectory, so they run
    // for a batch of frames on worker threads. Each OpenCV call seeds its own
    // RNG (USAC state 0, RANSAC's fixed seed, ORB's fixed seeds), so the
    // results are identical to the sequential loop for any thread count.
    // Everything order-dependent -- scale lookup, constant-velocity fallback,
    // trajectory accumulation, the on_frame callback (keyframes and loop
    // closure) and the stage timers -- stays in the sequential pass below,
    // in frame order.
    const std::size_t threads = front_end_threads();
    const auto batch_size = static_cast<std::int64_t>(4 * threads);
    std::vector<FrontEndWork> batch;
    batch.reserve(static_cast<std::size_t>(batch_size));

    for (std::int64_t batch_start = 1; batch_start < n; batch_start += batch_size) {
        const std::int64_t batch_end = std::min(n, batch_start + batch_size);
        const auto m = static_cast<std::size_t>(batch_end - batch_start);
        batch.clear();
        batch.resize(m);

        parallel_for_index(m, threads, [&](std::size_t k) {
            FrontEndWork& w = batch[k];
            const std::int64_t i = batch_start + static_cast<std::int64_t>(k);
            try {
                w.image = dataset.load_image(static_cast<std::size_t>(i));
                const std::int64_t frame_id = dataset.frame_ids()[static_cast<std::size_t>(i)];
                w.diag.index = i;
                w.diag.frame_id = frame_id;
                w.diag.status = kStatusFailed;
                const auto started = Clock::now();
                w.frame = detector_.detect(w.image, i, frame_id);
                w.detect_s = seconds_since(started);
                w.diag.detect_ms = w.detect_s * 1000.0;
                w.diag.n_keypoints = static_cast<std::int64_t>(w.frame->size());
            } catch (...) {
                w.error = std::current_exception();
            }
        });

        parallel_for_index(m, threads, [&](std::size_t k) {
            FrontEndWork& w = batch[k];
            const std::optional<Frame>& prev = k == 0 ? previous : batch[k - 1].frame;
            if (w.error || !prev || !w.frame) {
                return;  // the earlier failure is rethrown in order below
            }
            try {
                auto started = Clock::now();
                w.matches = matcher_.match(*prev, *w.frame);
                w.match_s = seconds_since(started);
                w.diag.match_ms = w.match_s * 1000.0;

                started = Clock::now();
                w.relative = estimate_relative_pose(w.matches, w.diag);
                w.pose_s = seconds_since(started);
                w.diag.pose_ms = w.pose_s * 1000.0;
            } catch (...) {
                w.error = std::current_exception();
            }
        });

        for (std::size_t k = 0; k < m; ++k) {
            FrontEndWork& w = batch[k];
            if (w.error) {
                std::rethrow_exception(w.error);
            }
            const std::int64_t i = batch_start + static_cast<std::int64_t>(k);
            FrameDiagnostics& diag = w.diag;
            timer_->record("detect", w.detect_s);
            timer_->record("match", w.match_s);
            timer_->record("pose", w.pose_s);

            const std::optional<double> scale = scale_estimator_->scale_for(i - 1);
            diag.scale_m = scale ? *scale : std::numeric_limits<double>::quiet_NaN();

            const Eigen::Matrix4d step = resolve_step(w.relative, scale, diag, last_relative_direction);
            if (diag.status == kStatusOk && w.relative) {
                last_relative_direction = w.relative;
            }

            poses.push_back(project_to_se3(poses.back() * step));
            relative_poses.push_back(step);
            diagnostics.push_back(diag);

            if (on_frame) {
                on_frame(i, *w.frame, w.image, poses.back());
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
                    static_cast<long long>(diag.n_matches),
                    static_cast<long long>(diag.n_cheirality_inliers), timer_->fps(i + 1)));
            }
        }
        // The last frame of this batch is the first match partner of the next.
        previous = std::move(batch.back().frame);
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
