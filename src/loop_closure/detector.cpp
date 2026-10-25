#include "mvslam/loop_closure/detector.hpp"

#include <cmath>
#include <map>
#include <memory>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/geometry/epipolar.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.loop_closure.detector");
    return instance;
}

Json f(double v, int digits) { return py::make_float(py::round(v, digits)); }

}  // namespace

Json LoopClosure::to_dict() const {
    Json out = Json::object();
    out["query_keyframe"] = query_id;
    out["match_keyframe"] = match_id;
    out["query_frame_index"] = query_frame_index;
    out["match_frame_index"] = match_frame_index;
    out["similarity"] = f(similarity, 4);
    out["normalized_similarity"] = f(normalized_similarity, 4);
    out["n_matches"] = n_matches;
    out["n_inliers"] = n_inliers;
    out["inlier_ratio"] = f(inlier_ratio, 4);
    out["scale_m"] = f(scale_m, 4);
    out["scale_method"] = scale_method;
    out["n_scale_points"] = n_scale_points;
    return out;
}

Json LoopClosureStats::to_dict() const {
    Json out = Json::object();
    out["queries"] = n_queries;
    out["candidates_retrieved"] = n_candidates;
    out["rejected_normalized_similarity"] = rejected_normalized_similarity;
    out["rejected_too_few_matches"] = rejected_too_few_matches;
    out["rejected_geometry"] = rejected_geometry;
    out["rejected_cooldown"] = rejected_cooldown;
    out["scale_estimation_failed"] = scale_estimation_failed;
    out["rejected_large_scale"] = rejected_large_scale;
    out["accepted"] = accepted;
    out["mean_matches_per_candidate"] =
        match_counts.empty() ? Json(nullptr) : f(np::mean(match_counts), 1);
    out["mean_inliers_per_accepted"] =
        inlier_counts.empty() ? Json(nullptr) : f(np::mean(inlier_counts), 1);
    return out;
}

LoopClosureDetector::LoopClosureDetector(const Eigen::Matrix3d& K, FeatureMatcher matcher,
                                         const LoopClosureParams& params, StageTimer* timer)
    : K_(K), matcher_(std::move(matcher)), params_(params) {
    if (timer == nullptr) {
        owned_timer_ = std::make_unique<StageTimer>();
        timer_ = owned_timer_.get();
    } else {
        timer_ = timer;
    }
}

LoopClosureDetector LoopClosureDetector::from_config(const Config& config, const Eigen::Matrix3d& K,
                                                     FeatureMatcher matcher, StageTimer* timer) {
    const auto& lc = config.loop_closure;
    LoopClosureParams params;
    params.min_keyframe_separation = lc.min_keyframe_separation;
    params.min_path_separation_m = lc.min_path_separation_m;
    params.top_k = lc.top_k;
    params.min_similarity = lc.min_similarity;
    params.similarity_ratio = lc.similarity_ratio;
    params.min_matches = lc.min_matches;
    params.min_inliers = lc.min_inliers;
    params.min_inlier_ratio = lc.min_inlier_ratio;
    params.cooldown_keyframes = lc.cooldown_keyframes;
    params.max_loops = lc.max_loops;
    params.ransac_method = config.odometry.ransac_method;
    params.ransac_threshold_px = std::max(config.odometry.ransac_threshold_px, 1.0);
    params.cheirality_distance = config.odometry.cheirality_distance;
    params.min_scale_points = lc.min_scale_points;
    params.min_scale_baseline_m = lc.min_scale_baseline_m;
    params.scale_neighbour_search = lc.scale_neighbour_search;
    params.min_loop_scale_m = lc.min_loop_scale_m;
    params.max_loop_scale_m = lc.max_loop_scale_m;
    params.require_measured_scale = lc.require_measured_scale;
    return LoopClosureDetector(K, std::move(matcher), params, timer);
}

std::vector<LoopCandidate> LoopClosureDetector::retrieve_candidates(KeyframeDatabase& database,
                                                                   std::int64_t query_id) {
    const auto raw = database.query(query_id, params_.min_keyframe_separation,
                                    params_.min_path_separation_m, params_.top_k,
                                    params_.min_similarity);
    stats_.n_queries += 1;
    stats_.n_candidates += static_cast<std::int64_t>(raw.size());
    if (raw.empty()) {
        return {};
    }
    const double reference = database.neighbour_score(query_id);
    std::vector<LoopCandidate> candidates;
    for (const auto& [match_id, score] : raw) {
        const double normalized = reference > 1e-9 ? score / reference : 0.0;
        if (normalized < params_.similarity_ratio) {
            stats_.rejected_normalized_similarity += 1;
            continue;
        }
        candidates.push_back({query_id, match_id, score, normalized});
    }
    return candidates;
}

std::optional<LoopClosure> LoopClosureDetector::verify(KeyframeDatabase& database,
                                                       const LoopCandidate& candidate) {
    const Keyframe& query = database[static_cast<std::size_t>(candidate.query_id)];
    const Keyframe& match = database[static_cast<std::size_t>(candidate.match_id)];

    const MatchResult matches =
        matcher_.match_descriptors(match.descriptors, query.descriptors, match.points, query.points);
    stats_.match_counts.push_back(static_cast<double>(matches.size()));
    if (static_cast<std::int64_t>(matches.size()) < params_.min_matches) {
        stats_.rejected_too_few_matches += 1;
        return std::nullopt;
    }

    const EssentialMatrixResult essential = estimate_essential_matrix(
        matches.points_a, matches.points_b, K_, params_.ransac_threshold_px, 0.999, 2000, 8,
        params_.ransac_method);
    if (!essential.ok) {
        stats_.rejected_geometry += 1;
        return std::nullopt;
    }

    PoseGates gates;
    gates.min_inliers = params_.min_inliers;
    gates.min_inlier_ratio = 0.0;    // applied below against the match count
    gates.max_rotation_deg = 180.0;  // a loop may be traversed in any direction
    gates.cheirality_distance = params_.cheirality_distance;
    const RelativePoseResult pose = recover_relative_pose(essential.E, matches.points_a,
                                                          matches.points_b, K_, essential.inlier_mask,
                                                          gates);
    if (!pose.ok || !pose.T) {
        stats_.rejected_geometry += 1;
        return std::nullopt;
    }

    const double inlier_ratio =
        static_cast<double>(pose.n_cheirality_inliers) / static_cast<double>(matches.size());
    if (pose.n_cheirality_inliers < params_.min_inliers || inlier_ratio < params_.min_inlier_ratio) {
        stats_.rejected_geometry += 1;
        return std::nullopt;
    }

    // Monocular translation is unit-norm; recover its metric magnitude from
    // structure the query keyframe shares with its metrically-scaled
    // odometry neighbour. See the header comment for why the odometry gap
    // between the two loop keyframes must NOT be used here.
    auto [scale, n_scale_points] =
        estimate_loop_scale(database, candidate.match_id, candidate.query_id, *pose.T, matches);
    std::string scale_method = "structure";
    if (!scale) {
        stats_.scale_estimation_failed += 1;
        if (params_.require_measured_scale) {
            // An unmeasurable magnitude means this pair cannot be confirmed
            // as a genuine revisit -- the exact situation in which a
            // look-alike slips past appearance and geometry alike.
            return std::nullopt;
        }
        scale_method = "unscaled";
        // Keep the direction but give it no asserted magnitude; the pose
        // graph will down-weight the translation for this edge.
        scale = 0.0;
    }

    LoopClosure closure;
    closure.query_id = candidate.query_id;
    closure.match_id = candidate.match_id;
    closure.similarity = candidate.similarity;
    closure.normalized_similarity = candidate.normalized_similarity;
    closure.T_match_query = se3_from_rt(pose.T->topLeftCorner<3, 3>(),
                                        pose.T->topRightCorner<3, 1>() * *scale);
    closure.n_matches = static_cast<std::int64_t>(matches.size());
    closure.n_inliers = pose.n_cheirality_inliers;
    closure.inlier_ratio = inlier_ratio;
    closure.scale_m = *scale;
    closure.scale_method = scale_method;
    closure.n_scale_points = n_scale_points;
    closure.query_frame_index = query.frame_index;
    closure.match_frame_index = match.frame_index;
    stats_.inlier_counts.push_back(static_cast<double>(pose.n_cheirality_inliers));
    return closure;
}

std::pair<std::optional<double>, std::int64_t> LoopClosureDetector::estimate_loop_scale(
    const KeyframeDatabase& database, std::int64_t /*match_id*/, std::int64_t query_id,
    const Eigen::Matrix4d& T_match_query_unit, const MatchResult& loop_matches) {
    const Keyframe& query = database[static_cast<std::size_t>(query_id)];

    const auto neighbour_info = metric_neighbour(database, query_id);
    if (!neighbour_info) {
        return {std::nullopt, 0};
    }
    const Keyframe& neighbour = database[static_cast<std::size_t>(neighbour_info->first)];
    const Eigen::Matrix4d& T_query_neighbour = neighbour_info->second;

    // Structure from the loop pair, expressed in the query keyframe's frame.
    const Points3 loop_points_match_frame =
        triangulate_points(T_match_query_unit, loop_matches.points_a, loop_matches.points_b, K_);
    const Eigen::Matrix4d T_query_match_unit = invert_se3(T_match_query_unit);
    const Eigen::Matrix3d R = T_query_match_unit.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T_query_match_unit.topRightCorner<3, 1>();

    // Metric structure from the query keyframe and its odometry neighbour.
    const MatchResult local_matches = matcher_.match_descriptors(
        query.descriptors, neighbour.descriptors, query.points, neighbour.points);
    if (static_cast<std::int64_t>(local_matches.size()) < params_.min_scale_points) {
        return {std::nullopt, 0};
    }
    const Points3 local_points =
        triangulate_points(T_query_neighbour, local_matches.points_a, local_matches.points_b, K_);

    // Features observed in all three views, keyed by query keypoint index (a
    // repeated index keeps its last slot, like the Python dict).
    std::map<std::int64_t, std::size_t> loop_slot;
    for (std::size_t i = 0; i < loop_matches.indices_b.size(); ++i) {
        loop_slot[loop_matches.indices_b[i]] = i;
    }
    std::vector<std::pair<std::size_t, std::size_t>> shared;
    for (std::size_t i = 0; i < local_matches.indices_a.size(); ++i) {
        auto it = loop_slot.find(local_matches.indices_a[i]);
        if (it != loop_slot.end()) {
            shared.emplace_back(it->second, i);
        }
    }
    if (static_cast<std::int64_t>(shared.size()) < params_.min_scale_points) {
        return {std::nullopt, static_cast<std::int64_t>(shared.size())};
    }

    std::vector<double> ratios;
    for (const auto& [a, b] : shared) {
        // loop_points_query_frame = loop_points_match_frame @ R.T + t, z only.
        const auto ra = static_cast<Eigen::Index>(a);
        const double unit_depth = loop_points_match_frame(ra, 0) * R(2, 0) +
                                  loop_points_match_frame(ra, 1) * R(2, 1) +
                                  loop_points_match_frame(ra, 2) * R(2, 2) + t(2);
        const double metric_depth = local_points(static_cast<Eigen::Index>(b), 2);
        if (std::isfinite(unit_depth) && std::isfinite(metric_depth) && unit_depth > 1e-3 &&
            metric_depth > 1e-3) {
            ratios.push_back(metric_depth / unit_depth);
        }
    }
    const auto n_valid = static_cast<std::int64_t>(ratios.size());
    if (n_valid < params_.min_scale_points) {
        return {std::nullopt, n_valid};
    }

    // Trim the tails before taking the median: a handful of near-degenerate
    // triangulations can otherwise sit at absurd ratios.
    const double low = np::percentile(ratios, 20);
    const double high = np::percentile(ratios, 80);
    std::vector<double> trimmed;
    for (double r : ratios) {
        if (r >= low && r <= high) {
            trimmed.push_back(r);
        }
    }
    const double scale = np::median(trimmed.empty() ? ratios : trimmed);

    if (!std::isfinite(scale) || scale < params_.min_loop_scale_m) {
        return {std::nullopt, n_valid};
    }
    if (scale > params_.max_loop_scale_m) {
        // Physically implausible for a revisit: appearance matched a
        // look-alike further along the same corridor. Counted separately
        // because it is a different failure from "could not measure".
        stats_.rejected_large_scale += 1;
        return {std::nullopt, n_valid};
    }
    return {scale, n_valid};
}

std::optional<std::pair<std::int64_t, Eigen::Matrix4d>> LoopClosureDetector::metric_neighbour(
    const KeyframeDatabase& database, std::int64_t query_id) const {
    const Keyframe& query = database[static_cast<std::size_t>(query_id)];
    for (std::int64_t offset = 1; offset <= params_.scale_neighbour_search; ++offset) {
        const std::int64_t candidate_id = query_id - offset;
        if (candidate_id < 0) {
            break;
        }
        const Keyframe& neighbour = database[static_cast<std::size_t>(candidate_id)];
        const Eigen::Matrix4d T_query_neighbour = invert_se3(query.pose) * neighbour.pose;
        const double x = T_query_neighbour(0, 3);
        const double y = T_query_neighbour(1, 3);
        const double z = T_query_neighbour(2, 3);
        if (std::sqrt(x * x + y * y + z * z) >= params_.min_scale_baseline_m) {
            return std::make_pair(candidate_id, T_query_neighbour);
        }
    }
    return std::nullopt;
}

std::vector<LoopClosure> LoopClosureDetector::detect(KeyframeDatabase& database) {
    if (database.size() == 0) {
        return {};
    }
    stats_ = LoopClosureStats();
    {
        auto scope = timer_->time("loop_index");
        database.build_index();
    }

    std::vector<LoopClosure> closures;
    std::int64_t last_accepted = -1'000'000'000;
    {
        auto scope = timer_->time("loop_detect");
        for (std::int64_t query_id = 0; query_id < static_cast<std::int64_t>(database.size()); ++query_id) {
            if (query_id < params_.min_keyframe_separation) {
                continue;
            }
            if (query_id - last_accepted < params_.cooldown_keyframes) {
                stats_.rejected_cooldown += 1;
                continue;
            }
            if (params_.max_loops && static_cast<std::int64_t>(closures.size()) >= params_.max_loops) {
                break;
            }
            for (const LoopCandidate& candidate : retrieve_candidates(database, query_id)) {
                std::optional<LoopClosure> closure = verify(database, candidate);
                if (!closure) {
                    continue;
                }
                closures.push_back(*closure);
                stats_.accepted += 1;
                last_accepted = query_id;
                logger().info(py::sprintf(
                    "Loop closure: kf %lld <-> kf %lld (frames %lld <-> %lld) | sim %.3f (norm %.2f) | "
                    "%lld/%lld inliers (%.0f%%)",
                    static_cast<long long>(closure->match_id), static_cast<long long>(closure->query_id),
                    static_cast<long long>(closure->match_frame_index),
                    static_cast<long long>(closure->query_frame_index), closure->similarity,
                    closure->normalized_similarity, static_cast<long long>(closure->n_inliers),
                    static_cast<long long>(closure->n_matches), 100.0 * closure->inlier_ratio));
                break;  // one constraint per query keyframe is enough
            }
        }
    }
    logger().info(py::sprintf("Loop closure detection: %lld accepted from %lld candidates over %lld queries",
                              static_cast<long long>(stats_.accepted),
                              static_cast<long long>(stats_.n_candidates),
                              static_cast<long long>(stats_.n_queries)));
    return closures;
}

}  // namespace mvslam
