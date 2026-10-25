// Trajectory error metrics.
//
// Three complementary families are implemented, because each answers a
// different question and none is sufficient alone:
//
// **Absolute Trajectory Error (ATE)** -- global consistency.
//     After aligning the estimate to ground truth, ATE is the Euclidean
//     distance between corresponding camera positions. It answers "how far
//     from the truth does the trajectory end up?" and is the metric loop
//     closure most visibly improves. It is dominated by low-frequency drift:
//     one early heading error inflates ATE for the rest of the sequence.
//
// **Relative Pose Error (RPE)** -- local accuracy.
//     For a fixed frame gap ``delta``, RPE compares the estimated relative
//     motion ``T_i_{i+delta}`` against the ground-truth relative motion. It
//     is invariant to global drift, so it isolates front-end quality.
//     Reported separately for translation (metres) and rotation (degrees).
//
// **KITTI drift** -- the odometry benchmark's own metric.
//     Errors are evaluated over sub-trajectories of fixed *path length*
//     (100 m ... 800 m) and normalised by that length, giving translation
//     error as a percentage and rotation error in degrees per metre.
//     Normalising by distance rather than by frame count makes results
//     comparable across sequences of different speed and length. This is the
//     number quoted on the KITTI odometry leaderboard.
//
// Reference for ATE/RPE: Sturm et al., "A Benchmark for the Evaluation of
// RGB-D SLAM Systems", IROS 2012. Reference for the drift metric: Geiger et
// al., "Are we ready for Autonomous Driving? The KITTI Vision Benchmark
// Suite", CVPR 2012.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/evaluation/alignment.hpp"
#include "mvslam/geometry/pose.hpp"

namespace mvslam {

// -- Absolute Trajectory Error ---------------------------------------------- //

/// Absolute trajectory error statistics, all in metres.
struct ATEResult {
    double rmse = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double std = 0.0;
    double min = 0.0;
    double max = 0.0;
    std::int64_t n = 0;
    std::string alignment = "none";
    double alignment_scale = 1.0;
    std::vector<double> errors;

    /// Every field except ``errors``; floats rounded to 6 digits.
    Json to_dict() const;
};

/// ATE between two equal-length trajectories.
///
/// ``alignment`` selects the Umeyama mode applied first; ``"none"`` yields the
/// unaligned error, which is only meaningful when both trajectories are
/// already in the same metric frame.
///
/// Note on ``std``: this is the standard deviation of the error *magnitudes*,
/// not a residual about the mean of a signed quantity, so ``rmse`` is not
/// equal to ``sqrt(mean^2 + std^2)`` by accident -- it is exactly that
/// identity, and both are reported because different papers quote different
/// ones.
ATEResult absolute_trajectory_error(const Trajectory& estimated, const Trajectory& reference,
                                    const std::string& alignment = "sim3");

// -- Relative Pose Error ---------------------------------------------------- //

/// Relative pose error at a fixed frame gap.
struct RPEResult {
    std::int64_t delta = 1;
    std::int64_t n = 0;
    /// Translation error in metres.
    double trans_rmse = 0.0;
    double trans_mean = 0.0;
    double trans_median = 0.0;
    double trans_max = 0.0;
    /// Rotation error in degrees.
    double rot_rmse_deg = 0.0;
    double rot_mean_deg = 0.0;
    double rot_median_deg = 0.0;
    double rot_max_deg = 0.0;

    Json to_dict() const;
};

/// RPE at a frame gap of ``delta``.
///
/// For each ``i``, the error transform is
///
///     E_i = (T_gt_i^-1 T_gt_{i+d})^-1 (T_est_i^-1 T_est_{i+d})
///
/// whose translation norm and rotation angle are the reported errors.
/// Because only *relative* motions are compared, no global alignment is
/// needed -- which is exactly what makes RPE insensitive to accumulated drift.
///
/// ``scale`` multiplies the estimated translations first, for evaluating a
/// scale-free (unit-scale) trajectory against metric ground truth. Pass the
/// factor from a sim3 alignment.
RPEResult relative_pose_error(const Trajectory& estimated, const Trajectory& reference,
                              std::int64_t delta = 1, double scale = 1.0);

// -- KITTI drift ------------------------------------------------------------ //

/// KITTI-style drift, averaged over fixed-length sub-trajectories.
struct DriftResult {
    /// Translation error as a percentage of sub-trajectory length.
    double translation_pct = 0.0;
    /// Rotation error in degrees per metre.
    double rotation_deg_per_m = 0.0;
    /// The same rotation figure per 100 m, which is the more readable form.
    double rotation_deg_per_100m = 0.0;
    /// Number of sub-trajectories evaluated.
    std::int64_t n_segments = 0;
    /// Per-length breakdown, ``{length_m: (trans_pct, rot_deg_per_m)}``.
    std::map<double, std::pair<double, double>> per_length;

    Json to_dict() const;
};

const std::vector<double>& default_segment_lengths();

/// KITTI odometry drift metric.
///
/// For every start frame (every ``step``-th) and every requested
/// sub-trajectory length, find the end frame at which ground truth has
/// travelled that far, then compare the estimated and true relative pose over
/// that stretch. The translation error is divided by the sub-trajectory
/// length, so the result is a dimensionless percentage that does not depend
/// on how long the sequence is.
///
/// Sub-trajectories that run off the end of the sequence are skipped rather
/// than truncated, which is what keeps the normalisation honest.
DriftResult kitti_drift(const Trajectory& estimated, const Trajectory& reference,
                        const std::vector<double>& segment_lengths = default_segment_lengths(),
                        std::int64_t step = 10);

// -- Combined report -------------------------------------------------------- //

/// Full evaluation of one trajectory against ground truth.
struct TrajectoryEvaluation {
    std::string label;
    ATEResult ate_aligned;
    ATEResult ate_unaligned;
    std::map<std::int64_t, RPEResult> rpe;
    DriftResult drift;
    double path_length_m = 0.0;
    double reference_path_length_m = 0.0;
    std::optional<AlignmentResult> alignment;

    Json to_dict() const;
    std::string summary_line() const;
};

/// Run every metric and bundle the results.
///
/// Both aligned and unaligned ATE are always reported. The distinction is not
/// cosmetic for a monocular system: the aligned figure measures trajectory
/// shape (having absorbed the unobservable global scale and frame), while the
/// unaligned figure additionally penalises being in the wrong frame at all.
/// Drift and RPE are computed on the aligned trajectory so that a global
/// frame offset is not double-counted as a local error.
TrajectoryEvaluation evaluate_trajectory(
    const Trajectory& estimated, const Trajectory& reference, const std::string& label = "trajectory",
    const std::string& alignment = "sim3",
    const std::vector<std::int64_t>& rpe_deltas = {1, 10, 100},
    const std::vector<double>& segment_lengths = default_segment_lengths());

/// Percentage reduction in ATE, positive when ``after`` is better.
///
/// NaN when ``before`` is zero or non-finite, since the reduction is
/// undefined there rather than infinite.
double ate_reduction_pct(double before, double after);

}  // namespace mvslam
