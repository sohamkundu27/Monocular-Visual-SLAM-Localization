// Loop candidate retrieval, filtering and geometric verification.
//
// A loop closure is accepted only after passing every stage of this cascade:
//
// 1. **Appearance retrieval** -- BoW cosine similarity against the keyframe
//    database, excluding temporally and spatially adjacent keyframes.
// 2. **Normalised similarity** -- the candidate must score at least
//    ``similarity_ratio`` of the best score among the query's own temporal
//    neighbours. This adapts to scene texture instead of trusting one
//    absolute threshold everywhere.
// 3. **Descriptor matching** -- the two keyframes are re-matched with the
//    same filter cascade the front end uses. Appearance similarity alone is
//    famous for confusing structurally repetitive scenes (rows of identical
//    parked cars, repeating building facades); requiring a large number of
//    individually consistent correspondences is a much stronger test.
// 4. **Geometric verification** -- a robust essential matrix must fit those
//    correspondences with enough inliers and a high enough inlier ratio. A
//    false positive from perceptual aliasing will match descriptors but will
//    not admit a single consistent camera motion.
// 5. **Metric plausibility** -- the recovered translation magnitude must be
//    small enough for the two keyframes to actually be the same place. See
//    below; this is what catches false positives on repetitive corridors,
//    where appearance and geometry both agree but the places are 80 m apart.
// 6. **Cooldown** -- after an accepted loop, detection pauses briefly so one
//    revisit does not generate a burst of near-duplicate constraints.
//
// Recovering the loop translation magnitude
// -----------------------------------------
// The constraint produced is a relative pose ``T_a_b`` between the two
// keyframes. Its rotation is fully determined by the essential matrix; its
// **translation is unit-norm only**, for the same monocular reason the
// odometry front end cannot recover scale.
//
// The obvious shortcut -- take the magnitude from the odometry-estimated gap
// between the two keyframes -- is *wrong*, and wrong in the most damaging
// possible way. When the trajectory has drifted, that gap **is** the drift:
// on KITTI sequence 00 the odometry gap across accepted loops averages 10.9 m
// while the true separation averages 2.6 m. Feeding that in bakes the drift
// into the very constraint meant to remove it, and measurably makes the
// optimized trajectory worse than the raw one.
//
// The magnitude is therefore recovered from **shared scene structure**
// (LoopClosureDetector::estimate_loop_scale):
//
// 1. Triangulate the loop pair's correspondences using the unit-norm
//    relative pose. This yields structure that is correct up to the unknown
//    factor ``s``, the true translation magnitude.
// 2. Triangulate the query keyframe against its odometry neighbour, whose
//    relative pose *is* metric. This yields the same scene at true scale.
// 3. For features seen in all three views, the ratio of metric to unit depth
//    is ``s``. A robust median over those ratios is the estimate.
//
// Why the recovered magnitude is also the best false-positive filter
// -----------------------------------------------------------------
// Geometric verification alone cannot catch every false positive. On a
// repetitive corridor -- a highway with guardrails, lane markings and uniform
// vegetation -- two points 80 m apart look alike *and* admit a perfectly
// consistent camera motion between them, because "straight road ahead" is a
// valid relative pose. Appearance and geometry both say yes; on KITTI
// sequence 01 that produced nine confident detections whose true separation
// was 73-108 m.
//
// The recovered translation magnitude breaks the tie, because a genuine
// revisit puts the camera within a few metres of where it was before.
// Measured across sequences 00 and 05, every true loop recovered a magnitude
// of 8.6 m or less (median 1.2 m), tracking the ground-truth separation
// closely; every sequence-01 false positive recovered 12.9 m or more.
// ``max_loop_scale_m`` therefore acts as a physical plausibility gate, not
// just a numerical sanity bound.
//
// By the same logic, a loop whose magnitude cannot be measured at all is one
// that cannot be confirmed as a genuine revisit, so ``require_measured_scale``
// rejects it by default. Setting that flag to false keeps such loops as
// orientation-only constraints (with a heavily inflated translation sigma),
// which is defensible when the front end is known to be rotation-limited, but
// on repetitive scenes it is exactly the case that lets false positives
// through.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/features/matcher.hpp"
#include "mvslam/loop_closure/database.hpp"
#include "mvslam/utils/timing.hpp"

namespace mvslam {

struct Config;

/// A keyframe pair proposed by appearance retrieval.
struct LoopCandidate {
    std::int64_t query_id = 0;
    std::int64_t match_id = 0;
    double similarity = 0.0;
    double normalized_similarity = 0.0;
};

/// A geometrically verified loop closure constraint.
struct LoopClosure {
    std::int64_t query_id = 0;
    std::int64_t match_id = 0;
    double similarity = 0.0;
    double normalized_similarity = 0.0;
    /// ``T_match_query``: pose of the query keyframe in the matched keyframe's
    /// frame, with a metric translation recovered from shared structure.
    Eigen::Matrix4d T_match_query = Eigen::Matrix4d::Identity();
    std::int64_t n_matches = 0;
    std::int64_t n_inliers = 0;
    double inlier_ratio = 0.0;
    /// Translation magnitude applied to the unit direction, in metres.
    double scale_m = 0.0;
    /// How that magnitude was obtained: "structure" (triangulated against
    /// metrically-scaled neighbouring structure) or "unscaled" (estimation
    /// failed; the constraint is treated as orientation-only).
    std::string scale_method = "structure";
    /// Number of tri-view features that supported the scale estimate.
    std::int64_t n_scale_points = 0;
    /// Frame indices, for plotting against the trajectory.
    std::int64_t query_frame_index = 0;
    std::int64_t match_frame_index = 0;

    /// False when no metric magnitude could be recovered for this loop.
    bool scale_is_measured() const { return scale_method == "structure"; }

    Json to_dict() const;
};

/// Counts of how candidates fared at each stage of the cascade.
struct LoopClosureStats {
    std::int64_t n_queries = 0;
    std::int64_t n_candidates = 0;
    std::int64_t rejected_normalized_similarity = 0;
    std::int64_t rejected_too_few_matches = 0;
    std::int64_t rejected_geometry = 0;
    std::int64_t rejected_cooldown = 0;
    /// Loops accepted geometrically whose metric magnitude could not be
    /// recovered; these become orientation-only constraints.
    std::int64_t scale_estimation_failed = 0;
    /// Loops whose recovered magnitude was too large to be a revisit.
    std::int64_t rejected_large_scale = 0;
    std::int64_t accepted = 0;
    std::vector<double> match_counts;
    std::vector<double> inlier_counts;

    Json to_dict() const;
};

struct LoopClosureParams {
    std::int64_t min_keyframe_separation = 30;
    double min_path_separation_m = 30.0;
    std::int64_t top_k = 5;
    double min_similarity = 0.2;
    double similarity_ratio = 0.85;
    std::int64_t min_matches = 60;
    std::int64_t min_inliers = 40;
    double min_inlier_ratio = 0.35;
    std::int64_t cooldown_keyframes = 10;
    std::int64_t max_loops = 0;
    std::string ransac_method = "magsac";
    /// Loop pairs are wide-baseline, so a slightly looser epipolar threshold
    /// than the frame-to-frame front end is appropriate here.
    double ransac_threshold_px = 1.0;
    double cheirality_distance = 200.0;
    std::int64_t min_scale_points = 15;
    double min_scale_baseline_m = 0.5;
    std::int64_t scale_neighbour_search = 5;
    double min_loop_scale_m = 0.05;
    double max_loop_scale_m = 10.0;
    bool require_measured_scale = true;
};

/// Retrieves, verifies and accepts loop closures over a keyframe database.
class LoopClosureDetector {
public:
    LoopClosureDetector(const Eigen::Matrix3d& K, FeatureMatcher matcher,
                        const LoopClosureParams& params = {}, StageTimer* timer = nullptr);

    static LoopClosureDetector from_config(const Config& config, const Eigen::Matrix3d& K,
                                           FeatureMatcher matcher, StageTimer* timer = nullptr);

    /// Appearance retrieval plus the normalised-similarity filter.
    std::vector<LoopCandidate> retrieve_candidates(KeyframeDatabase& database, std::int64_t query_id);

    /// Re-match and geometrically verify a candidate pair. Returns the
    /// accepted LoopClosure, or nullopt with the reason recorded in stats().
    std::optional<LoopClosure> verify(KeyframeDatabase& database, const LoopCandidate& candidate);

    /// Recover the metric magnitude of a loop translation, or nullopt.
    ///
    /// Compares two triangulations of the same scene points:
    ///
    /// * the loop pair ``(match, query)``, triangulated with the **unit-norm**
    ///   relative pose, giving depths that are correct up to the unknown
    ///   factor ``s``;
    /// * the query keyframe against an **odometry neighbour**, whose relative
    ///   pose carries the front end's metric scale.
    ///
    /// For points seen in all three views, ``s = depth_metric / depth_unit``.
    /// The median over those ratios is returned, which resists the outliers
    /// that individual bad triangulations produce.
    ///
    /// Returns ``(scale, n_supporting_points)``; ``scale`` is nullopt when
    /// there was not enough shared, well-conditioned structure.
    std::pair<std::optional<double>, std::int64_t> estimate_loop_scale(
        const KeyframeDatabase& database, std::int64_t match_id, std::int64_t query_id,
        const Eigen::Matrix4d& T_match_query_unit, const MatchResult& loop_matches);

    /// Run detection across the whole keyframe database.
    ///
    /// Queries run in keyframe order so the cooldown behaves causally, exactly
    /// as it would in an online system.
    std::vector<LoopClosure> detect(KeyframeDatabase& database);

    const LoopClosureStats& stats() const { return stats_; }
    LoopClosureParams& params() { return params_; }
    const LoopClosureParams& params() const { return params_; }

private:
    /// Find a nearby keyframe with enough metric baseline to triangulate.
    ///
    /// Walks backwards from the query keyframe until the odometry translation
    /// exceeds a minimum baseline; a stationary vehicle produces adjacent
    /// keyframes with no parallax, from which nothing can be triangulated.
    std::optional<std::pair<std::int64_t, Eigen::Matrix4d>> metric_neighbour(
        const KeyframeDatabase& database, std::int64_t query_id) const;

    Eigen::Matrix3d K_;
    FeatureMatcher matcher_;
    LoopClosureParams params_;
    std::unique_ptr<StageTimer> owned_timer_;
    StageTimer* timer_;
    LoopClosureStats stats_;
};

}  // namespace mvslam
