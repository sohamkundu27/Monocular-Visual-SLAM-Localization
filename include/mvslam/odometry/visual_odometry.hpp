// Monocular visual odometry front end.
//
// Per frame transition ``i -> i+1``:
//
// 1. Detect ORB keypoints and descriptors in frame ``i+1`` (frame ``i`` is
//    cached from the previous iteration, so each image is described once).
// 2. Match descriptors against the previous frame and filter the
//    correspondences.
// 3. Estimate the essential matrix with RANSAC and decompose it into a
//    validated relative pose ``T_i_{i+1}`` with **unit-norm translation**.
// 4. Ask the scale strategy for the translation magnitude and apply it.
// 5. Compose onto the global pose: ``T_w_{i+1} = T_w_i * T_i_{i+1}``.
//
// Failure handling
// ----------------
// A 4541-frame sequence will contain transitions that cannot be solved: too
// few matches under motion blur, a degenerate essential matrix while stopped
// at a light, a decomposition that yields an implausible rotation. Aborting
// the run is the wrong answer, and so is pretending the estimate succeeded.
// Each transition therefore resolves to one of three outcomes, all recorded in
// the diagnostics:
//
// ok
//     A validated relative pose was integrated.
// stationary
//     Median optical flow was below threshold, so the vehicle is treated as
//     stopped and the previous pose is held. This is the *correct* estimate,
//     not a fallback, and it counts as a success.
// failed
//     Estimation failed or was rejected. The last successful relative motion
//     is re-applied as a constant-velocity prediction -- at 10 Hz a car's
//     motion changes little between frames, so this bridges the gap far
//     better than holding still -- and the transition is counted as a
//     failure in the successful-pose-rate metric.
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/features/detector.hpp"
#include "mvslam/features/matcher.hpp"
#include "mvslam/geometry/pose.hpp"
#include "mvslam/odometry/scale.hpp"
#include "mvslam/utils/timing.hpp"

namespace mvslam {

struct Config;
class KittiOdometryDataset;

/// Transition outcomes.
inline constexpr const char* kStatusOk = "ok";
inline constexpr const char* kStatusStationary = "stationary";
inline constexpr const char* kStatusFailed = "failed";

/// Per-transition record, exported to diagnostics.csv.
struct FrameDiagnostics {
    std::int64_t index = 0;
    std::int64_t frame_id = 0;
    std::string status = kStatusFailed;
    std::string reason;
    std::int64_t n_keypoints = 0;
    std::int64_t n_raw_matches = 0;
    std::int64_t n_matches = 0;
    std::int64_t n_ransac_inliers = 0;
    std::int64_t n_cheirality_inliers = 0;
    /// Epipolar inliers as a fraction of filtered matches -- the number
    /// conventionally reported as "RANSAC inlier ratio".
    double ransac_inlier_ratio = 0.0;
    /// Cheirality survivors as a fraction of epipolar inliers.
    double inlier_ratio = 0.0;
    double rotation_deg = std::numeric_limits<double>::quiet_NaN();
    double parallax_deg = std::numeric_limits<double>::quiet_NaN();
    double flow_px = std::numeric_limits<double>::quiet_NaN();
    double homography_ratio = std::numeric_limits<double>::quiet_NaN();
    bool degenerate = false;
    double scale_m = std::numeric_limits<double>::quiet_NaN();
    double detect_ms = 0.0;
    double match_ms = 0.0;
    double pose_ms = 0.0;

    /// Stationary transitions are correct estimates, so they count as successes.
    bool succeeded() const { return status == kStatusOk || status == kStatusStationary; }

    /// ``dataclasses.asdict(self)`` order, values as ``csv`` writes them
    /// (``str()`` of each field: float repr, ``True``/``False``).
    std::vector<std::pair<std::string, std::string>> to_row() const;
};

/// Everything a VO run produces.
struct OdometryResult {
    Trajectory trajectory;
    std::vector<FrameDiagnostics> diagnostics;
    PoseList relative_poses;
    StageTimer* timer = nullptr;
    Json scale_strategy = Json::object();

    // Aggregate statistics (all measured, none assumed).
    std::int64_t n_transitions() const { return static_cast<std::int64_t>(diagnostics.size()); }
    std::int64_t n_successful() const;
    /// Percentage of attempted transitions that produced a usable pose.
    double successful_pose_rate_pct() const;
    double avg_features_per_frame() const;
    double avg_matches_per_pair() const;
    double avg_ransac_inliers() const;
    /// Mean RANSAC inlier ratio over successful transitions, as a percentage.
    double avg_inlier_ratio_pct() const;
    double avg_cheirality_ratio_pct() const;
    std::int64_t n_degenerate() const;
    /// Rounded statistics, keys in the Python order.
    Json feature_statistics() const;
    /// ``{status: count}`` in order of first appearance.
    Json status_counts() const;
};

/// ``on_frame(index, frame, image, pose)`` callback.
using FrameCallback =
    std::function<void(std::int64_t, const Frame&, const cv::Mat&, const Eigen::Matrix4d&)>;

struct VisualOdometryParams {
    std::string ransac_method = "magsac";
    double ransac_threshold_px = 0.5;
    double ransac_confidence = 0.999;
    std::int64_t ransac_max_iters = 2000;
    double cheirality_distance = 200.0;
    std::int64_t min_matches = 20;
    std::int64_t min_inliers = 30;
    double min_inlier_ratio = 0.3;
    double max_rotation_deg = 30.0;
    double min_parallax_deg = 0.0;
    double min_flow_px = 0.7;
    double homography_ratio_threshold = 0.45;
    bool reject_degenerate = false;
};

/// Frame-to-frame monocular visual odometry.
class VisualOdometry {
public:
    VisualOdometry(const Eigen::Matrix3d& K, FeatureDetector detector, FeatureMatcher matcher,
                   std::unique_ptr<ScaleEstimator> scale_estimator,
                   const VisualOdometryParams& params = {}, StageTimer* timer = nullptr);

    /// Assemble a VO front end from configuration plus a dataset. Objects are
    /// built (and log) in the same order as in the Python factory.
    static VisualOdometry from_config(const Config& config, const KittiOdometryDataset& dataset,
                                      StageTimer* timer = nullptr);

    /// Estimate ``T_i_{i+1}`` with unit translation, or nullopt on failure.
    ///
    /// ``diagnostics`` is filled in as a side effect so every transition --
    /// successful or not -- leaves an auditable trail.
    std::optional<Eigen::Matrix4d> estimate_relative_pose(const MatchResult& matches,
                                                          FrameDiagnostics& diagnostics) const;

    /// Process an entire sequence and accumulate the global trajectory.
    ///
    /// ``on_frame`` is invoked for every processed image after its global pose
    /// is known, which lets the SLAM pipeline harvest keyframes without a
    /// second pass over the data or a second feature-detection cost.
    OdometryResult run(const KittiOdometryDataset& dataset, const FrameCallback& on_frame = {},
                       std::int64_t progress_interval = 250);

    const FeatureDetector& detector() const { return detector_; }
    const FeatureMatcher& matcher() const { return matcher_; }
    const ScaleEstimator& scale_estimator() const { return *scale_estimator_; }
    const Eigen::Matrix3d& K() const { return K_; }
    const VisualOdometryParams& params() const { return params_; }
    StageTimer& timer() { return *timer_; }

private:
    /// Turn a unit-translation pose plus a scale into the metric step.
    Eigen::Matrix4d resolve_step(const std::optional<Eigen::Matrix4d>& relative,
                                 const std::optional<double>& scale, FrameDiagnostics& diagnostics,
                                 const std::optional<Eigen::Matrix4d>& last_direction) const;

    Eigen::Matrix3d K_;
    FeatureDetector detector_;
    FeatureMatcher matcher_;
    std::unique_ptr<ScaleEstimator> scale_estimator_;
    VisualOdometryParams params_;
    std::unique_ptr<StageTimer> owned_timer_;
    StageTimer* timer_;
};

}  // namespace mvslam
