#include "mvslam/pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/evaluation/plotting.hpp"
#include "mvslam/features/matcher.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/loop_closure/database.hpp"
#include "mvslam/optimization/pose_graph.hpp"
#include "mvslam/utils/logging.hpp"
#include "mvslam/utils/seeding.hpp"
#include "mvslam/version.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.pipeline");
    return instance;
}

/// Callback that turns VO frames into keyframes as the front end runs.
///
/// Tracking travelled distance here (rather than recomputing it later) keeps
/// the path-separation loop-closure filter exact even when frames are dropped.
class KeyframeHarvester {
public:
    KeyframeHarvester(KeyframeDatabase& database, KeyframeSelector& selector, bool enabled)
        : database_(database), selector_(selector), enabled_(enabled) {}

    void operator()(std::int64_t index, const Frame& frame, const cv::Mat& /*image*/,
                    const Eigen::Matrix4d& pose) {
        const Eigen::Vector3d position = pose.topRightCorner<3, 1>();
        if (last_position_) {
            const Eigen::Vector3d d = position - *last_position_;
            travelled_ += std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2));
        }
        last_position_ = position;

        if (!enabled_ || !frame.is_usable()) {
            return;
        }
        if (!selector_.should_select(index, pose)) {
            return;
        }
        selector_.accept(index, pose);
        Keyframe keyframe;
        keyframe.keyframe_id = -1;
        keyframe.frame_index = index;
        keyframe.frame_id = frame.frame_id;
        keyframe.points = frame.points;
        keyframe.descriptors = frame.descriptors.clone();
        keyframe.pose = pose;
        keyframe.path_distance_m = travelled_;
        database_.add(std::move(keyframe));
    }

private:
    KeyframeDatabase& database_;
    KeyframeSelector& selector_;
    bool enabled_;
    double travelled_ = 0.0;
    std::optional<Eigen::Vector3d> last_position_;
};

std::string utc_now_iso() {
    // datetime.now(timezone.utc).isoformat(timespec="seconds")
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S+00:00", &utc);
    return buffer;
}

}  // namespace

// --------------------------------------------------------------------------- //
// SlamPipeline
// --------------------------------------------------------------------------- //

SlamPipeline::SlamPipeline(Config config)
    : config_(std::move(config)), timer_(std::make_shared<StageTimer>()) {}

SlamResult SlamPipeline::run(const KittiOdometryDataset* dataset_in) {
    const Config& config = config_;
    set_global_seed(config.runtime.seed);

    std::optional<KittiOdometryDataset> owned;
    if (dataset_in == nullptr) {
        owned.emplace(KittiOdometryDataset::from_config(config));
    }
    const KittiOdometryDataset& dataset = dataset_in != nullptr ? *dataset_in : *owned;

    timer_->reset_wall_clock();

    // -- Stage 1: visual odometry with keyframe harvesting ----------------- //
    VisualOdometry vo = VisualOdometry::from_config(config, dataset, timer_.get());
    KeyframeDatabase database;
    KeyframeSelector selector = KeyframeSelector::from_config(config);
    KeyframeHarvester harvest(database, selector, config.loop_closure.enabled);

    OdometryResult odometry =
        vo.run(dataset, [&harvest](std::int64_t i, const Frame& f, const cv::Mat& im,
                                   const Eigen::Matrix4d& p) { harvest(i, f, im, p); },
               config.runtime.progress_interval);
    const Trajectory raw_trajectory = odometry.trajectory;
    const std::optional<Trajectory> ground_truth = dataset.ground_truth();

    // -- Stages 2-3: loop closure ------------------------------------------ //
    std::vector<LoopClosure> loop_closures;
    Json loop_stats = Json::object();
    const std::int64_t min_separation = config.loop_closure.min_keyframe_separation;
    if (config.loop_closure.enabled && static_cast<std::int64_t>(database.size()) > min_separation) {
        std::tie(loop_closures, loop_stats) = detect_loops(dataset, vo, database);
    } else if (config.loop_closure.enabled) {
        logger().warning(py::sprintf("Only %zu keyframes harvested; need more than "
                                     "min_keyframe_separation=%lld for loop closure",
                                     database.size(), static_cast<long long>(min_separation)));
    }

    // -- Stages 4-5: pose graph optimization ------------------------------- //
    std::optional<OptimizationResult> optimization;
    std::optional<Trajectory> optimized_trajectory;
    if (!loop_closures.empty() && gtsam_available()) {
        auto [opt, dense] = optimize(database, raw_trajectory, loop_closures);
        optimization = std::move(opt);
        optimized_trajectory = std::move(dense);
    } else if (!loop_closures.empty()) {
        logger().error("Loop closures were found but GTSAM is unavailable; skipping optimization");
    } else {
        logger().info("No verified loop closures; the optimized trajectory equals raw odometry");
    }

    // -- Stage 6: evaluation ----------------------------------------------- //
    std::optional<TrajectoryEvaluation> evaluation_raw;
    std::optional<TrajectoryEvaluation> evaluation_optimized;
    if (ground_truth) {
        evaluation_raw = evaluate_trajectory(raw_trajectory, *ground_truth, "raw_vo",
                                             config.evaluation.alignment, config.evaluation.rpe_deltas,
                                             config.evaluation.drift_segment_lengths_m);
        if (optimized_trajectory) {
            evaluation_optimized = evaluate_trajectory(
                *optimized_trajectory, *ground_truth, "optimized_slam", config.evaluation.alignment,
                config.evaluation.rpe_deltas, config.evaluation.drift_segment_lengths_m);
        }
    } else {
        logger().warning("Sequence " + config.dataset.sequence +
                         " has no ground truth: trajectory metrics will be reported as null");
    }

    SlamResult result;
    result.config = config;
    result.dataset_info = dataset.describe();
    result.odometry = std::move(odometry);
    result.raw_trajectory = raw_trajectory;
    result.optimized_trajectory = std::move(optimized_trajectory);
    result.ground_truth = ground_truth;
    result.loop_closures = std::move(loop_closures);
    result.optimization = std::move(optimization);
    result.timer = timer_;
    result.evaluation_raw = std::move(evaluation_raw);
    result.evaluation_optimized = std::move(evaluation_optimized);
    for (const auto& kf : database) {
        result.keyframe_frame_indices.push_back(kf.frame_index);
    }
    result.loop_stats = loop_stats;
    return result;
}

std::pair<std::vector<LoopClosure>, Json> SlamPipeline::detect_loops(const KittiOdometryDataset& dataset,
                                                                     const VisualOdometry& vo,
                                                                     KeyframeDatabase& database) {
    const Config& config = config_;
    std::optional<VisualVocabulary> vocabulary;
    std::size_t n_training = 0;
    {
        auto scope = timer_->time("vocabulary");
        std::vector<cv::Mat> blocks;
        for (const auto& kf : database) {
            blocks.push_back(kf.descriptors);
        }
        cv::Mat descriptors;
        cv::vconcat(blocks, descriptors);
        const auto budget = static_cast<std::size_t>(config.loop_closure.vocabulary_train_descriptors);
        if (static_cast<std::size_t>(descriptors.rows) > budget) {
            // The Python run samples with np.random.default_rng(seed); the port
            // samples with its own seeded generator (statistically equivalent).
            Sampler rng(static_cast<std::uint64_t>(config.runtime.seed));
            const auto picks = rng.sample_without_replacement(static_cast<std::size_t>(descriptors.rows), budget);
            cv::Mat sample(static_cast<int>(picks.size()), descriptors.cols, descriptors.type());
            for (std::size_t k = 0; k < picks.size(); ++k) {
                descriptors.row(static_cast<int>(picks[k])).copyTo(sample.row(static_cast<int>(k)));
            }
            descriptors = sample;
        }
        n_training = static_cast<std::size_t>(descriptors.rows);
        vocabulary = VisualVocabulary::train(descriptors, config.loop_closure.vocabulary_size, 12,
                                             config.runtime.seed);
    }
    logger().info(py::sprintf("Trained vocabulary: %zu words from %zu descriptors in %.1f s",
                              vocabulary->size(), n_training, timer_->total_s("vocabulary")));
    database.set_vocabulary(*vocabulary);

    FeatureMatcher matcher = FeatureMatcher::from_config(config, vo.detector().descriptor_norm());
    LoopClosureDetector detector = LoopClosureDetector::from_config(config, dataset.K(), matcher, timer_.get());
    std::vector<LoopClosure> closures = detector.detect(database);
    return {closures, detector.stats().to_dict()};
}

std::pair<OptimizationResult, Trajectory> SlamPipeline::optimize(const KeyframeDatabase& database,
                                                                 const Trajectory& raw_trajectory,
                                                                 const std::vector<LoopClosure>& loop_closures) {
    const Config& config = config_;
    std::vector<std::int64_t> keyframe_indices;
    PoseList keyframe_poses;
    std::vector<std::int64_t> keyframe_ids;
    for (const auto& kf : database) {
        keyframe_indices.push_back(kf.frame_index);
        keyframe_poses.push_back(kf.pose);
        keyframe_ids.push_back(kf.frame_id);
    }
    const Trajectory keyframe_trajectory(keyframe_poses, keyframe_ids);
    const auto& pg = config.pose_graph;
    PoseGraphSigmas sigmas;
    sigmas.prior_sigma_rot = pg.prior_sigma_rot;
    sigmas.prior_sigma_trans = pg.prior_sigma_trans;
    sigmas.odom_sigma_rot = pg.odom_sigma_rot;
    sigmas.odom_sigma_trans = pg.odom_sigma_trans;
    sigmas.loop_sigma_rot = pg.loop_sigma_rot;
    sigmas.loop_sigma_trans = pg.loop_sigma_trans;
    sigmas.robust_loops = pg.robust_loop_kernel;
    const PoseGraph graph = build_pose_graph(keyframe_trajectory, loop_closures, keyframe_indices, sigmas);

    OptimizerOptions options;
    options.optimizer = pg.optimizer;
    options.max_iterations = pg.max_iterations;
    options.relative_error_tol = pg.relative_error_tol;
    options.absolute_error_tol = pg.absolute_error_tol;
    options.robust_loop_kernel = pg.robust_loop_kernel;
    options.huber_k = pg.huber_k;
    options.verbose = pg.verbose;
    OptimizationResult optimization = optimize_pose_graph(graph, options, timer_.get());
    Trajectory dense = propagate_keyframe_corrections(raw_trajectory, keyframe_indices, optimization.trajectory);
    return {std::move(optimization), std::move(dense)};
}

// --------------------------------------------------------------------------- //
// Correction propagation
// --------------------------------------------------------------------------- //

Trajectory propagate_keyframe_corrections(const Trajectory& dense,
                                          const std::vector<std::int64_t>& keyframe_indices,
                                          const Trajectory& optimized_keyframes) {
    if (keyframe_indices.empty() || optimized_keyframes.empty()) {
        return dense.copy();
    }
    if (keyframe_indices.size() != optimized_keyframes.size()) {
        throw ValueError("keyframe count mismatch: " + std::to_string(keyframe_indices.size()) +
                         " indices vs " + std::to_string(optimized_keyframes.size()) + " optimized poses");
    }
    std::vector<Eigen::Matrix4d> corrections;
    corrections.reserve(keyframe_indices.size());
    for (std::size_t i = 0; i < keyframe_indices.size(); ++i) {
        corrections.push_back(optimized_keyframes[i] *
                              invert_se3(dense[static_cast<std::size_t>(keyframe_indices[i])]));
    }
    PoseList out(dense.size());
    for (std::size_t i = 0; i < dense.size(); ++i) {
        // searchsorted(side="right") - 1 gives the preceding keyframe.
        const auto it = std::upper_bound(keyframe_indices.begin(), keyframe_indices.end(),
                                         static_cast<std::int64_t>(i));
        const std::ptrdiff_t slot = std::max<std::ptrdiff_t>(0, (it - keyframe_indices.begin()) - 1);
        out[i] = project_to_se3(corrections[static_cast<std::size_t>(slot)] * dense[i]);
    }
    return Trajectory(std::move(out), dense.frame_ids(), dense.timestamps());
}

// --------------------------------------------------------------------------- //
// Output serialisation
// --------------------------------------------------------------------------- //

Json round_metric(const std::optional<double>& value, int digits) {
    if (!value || !std::isfinite(*value)) {
        return Json(nullptr);
    }
    return py::make_float(py::round(*value, digits));
}

Json build_metrics(const SlamResult& result) {
    const Config& config = result.config;
    const OdometryResult& odometry = result.odometry;
    const auto n_frames = static_cast<std::int64_t>(result.raw_trajectory.size());
    const double runtime_s = result.timer ? result.timer->elapsed_s() : 0.0;

    auto opt = [](bool present, double value) { return present ? std::optional<double>(value) : std::nullopt; };
    const auto& er = result.evaluation_raw;
    const auto& eo = result.evaluation_optimized;
    const std::optional<double> ate_raw = opt(er.has_value(), er ? er->ate_aligned.rmse : 0.0);
    const std::optional<double> ate_optimized = opt(eo.has_value(), eo ? eo->ate_aligned.rmse : 0.0);
    const std::optional<double> drift_raw = opt(er.has_value(), er ? er->drift.translation_pct : 0.0);
    const std::optional<double> drift_optimized = opt(eo.has_value(), eo ? eo->drift.translation_pct : 0.0);
    const std::optional<double> rotation_drift = opt(er.has_value(), er ? er->drift.rotation_deg_per_100m : 0.0);
    const std::optional<double> rotation_drift_optimized =
        opt(eo.has_value(), eo ? eo->drift.rotation_deg_per_100m : 0.0);
    const double distance_m =
        result.ground_truth ? result.ground_truth->path_length() : result.raw_trajectory.path_length();
    const std::optional<double> reduction =
        (ate_raw && ate_optimized) ? std::optional<double>(ate_reduction_pct(*ate_raw, *ate_optimized))
                                   : std::nullopt;
    const Json feature_stats = odometry.feature_statistics();
    auto stat = [&](const char* key) { return py::as_double(feature_stats.at(key)); };
    const std::optional<double> fps =
        runtime_s > 0 ? std::optional<double>(static_cast<double>(n_frames) / runtime_s) : std::nullopt;

    Json metrics = Json::object();
    // --- headline figures, directly usable on a resume ------------------- //
    metrics["sequence"] = config.dataset.sequence;
    metrics["frames"] = n_frames;
    metrics["distance_km"] = round_metric(distance_m / 1000.0, 4);
    metrics["successful_pose_rate_pct"] = round_metric(odometry.successful_pose_rate_pct(), 2);
    metrics["ate_rmse_raw_m"] = round_metric(ate_raw, 4);
    metrics["ate_rmse_optimized_m"] = round_metric(ate_optimized, 4);
    metrics["ate_reduction_pct"] = round_metric(reduction, 2);
    metrics["translational_drift_raw_pct"] = round_metric(drift_raw, 4);
    metrics["translational_drift_optimized_pct"] = round_metric(drift_optimized, 4);
    metrics["rotational_drift_deg_per_100m"] = round_metric(rotation_drift, 5);
    metrics["rotational_drift_optimized_deg_per_100m"] = round_metric(rotation_drift_optimized, 5);
    metrics["loop_closures_detected"] = static_cast<std::int64_t>(result.loop_closures.size());
    metrics["avg_features_per_frame"] = round_metric(stat("avg_features_per_frame"), 1);
    metrics["avg_matches_per_pair"] = round_metric(stat("avg_matches_per_pair"), 1);
    metrics["avg_inlier_ratio_pct"] = round_metric(stat("avg_inlier_ratio_pct"), 2);
    metrics["runtime_fps"] = round_metric(fps, 2);
    // --- provenance and caveats ----------------------------------------- //
    metrics["ground_truth_available"] = result.has_ground_truth();
    metrics["scale_source"] = config.odometry.scale_source;
    metrics["scale_uses_ground_truth"] = odometry.scale_strategy.contains("uses_ground_truth") &&
                                         odometry.scale_strategy.at("uses_ground_truth").get<bool>();
    metrics["alignment"] = config.evaluation.alignment;
    // --- detail ---------------------------------------------------------- //
    metrics["dataset"] = result.dataset_info;
    Json odo = Json::object();
    odo["transitions"] = odometry.n_transitions();
    odo["successful"] = odometry.n_successful();
    odo["status_counts"] = odometry.status_counts();
    odo["degenerate_transitions"] = odometry.n_degenerate();
    odo["avg_ransac_inliers"] = feature_stats.at("avg_ransac_inliers");
    odo["avg_cheirality_ratio_pct"] = feature_stats.at("avg_cheirality_ratio_pct");
    odo["scale_strategy"] = odometry.scale_strategy;
    metrics["odometry"] = odo;
    Json lc = Json::object();
    lc["enabled"] = config.loop_closure.enabled;
    lc["n_keyframes"] = static_cast<std::int64_t>(result.keyframe_frame_indices.size());
    lc["accepted"] = static_cast<std::int64_t>(result.loop_closures.size());
    lc["stats"] = result.loop_stats;
    Json closures = Json::array();
    for (const auto& c : result.loop_closures) {
        closures.push_back(c.to_dict());
    }
    lc["closures"] = closures;
    metrics["loop_closure"] = lc;
    metrics["optimization"] = result.optimization ? result.optimization->to_dict() : Json(nullptr);
    metrics["evaluation_raw"] = er ? er->to_dict() : Json(nullptr);
    metrics["evaluation_optimized"] = eo ? eo->to_dict() : Json(nullptr);
    Json timing = Json::object();
    timing["total_runtime_s"] = round_metric(runtime_s, 3);
    timing["avg_frame_latency_ms"] = round_metric(
        n_frames ? std::optional<double>(runtime_s / static_cast<double>(n_frames) * 1000.0) : std::nullopt, 3);
    timing["fps"] = round_metric(fps, 2);
    timing["stages"] = result.timer ? result.timer->summary() : Json::object();
    metrics["timing"] = timing;
    // The Python implementation reports its interpreter here; the port
    // reports the toolchain and libraries that actually produced the run.
    Json environment = Json::object();
    environment["generated_utc"] = utc_now_iso();
    environment["implementation"] = "mvslam C++ " + version();
    environment["cxx_compiler"] = compiler_version();
    environment["opencv"] = opencv_version();
    environment["gtsam"] = gtsam_version();
    environment["eigen"] = eigen_version();
    environment["platform"] = platform_string();
    environment["seed"] = config.runtime.seed;
    metrics["environment"] = environment;
    return metrics;
}

std::filesystem::path write_diagnostics_csv(const SlamResult& result, const std::filesystem::path& path) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const auto& diagnostics = result.odometry.diagnostics;
    if (diagnostics.empty()) {
        return path;
    }
    std::vector<std::string> header;
    for (const auto& [name, _] : diagnostics.front().to_row()) {
        header.push_back(name);
    }
    out << py::csv_row(header);
    for (const auto& d : diagnostics) {
        std::vector<std::string> fields;
        for (const auto& [_, value] : d.to_row()) {
            fields.push_back(value);
        }
        out << py::csv_row(fields);
    }
    return path;
}

namespace {

Json write_plots(const SlamResult& result, const std::filesystem::path& out_dir) {
    const Config& config = result.config;
    const int dpi = static_cast<int>(config.output.dpi);
    const std::string extension = config.output.plot_format;
    const std::string& sequence = config.dataset.sequence;
    const std::optional<Trajectory>& gt = result.ground_truth;
    Json plots = Json::object();
    auto record = [&](const std::string& name, const std::filesystem::path& image) {
        plots[name] = image.string();
        plots[name + "_data"] = plot::data_path_for(image).string();
    };

    record("trajectory_raw",
           plot::plot_trajectory(result.raw_trajectory, out_dir / ("trajectory_raw." + extension), gt,
                                 "Raw visual odometry", "KITTI " + sequence + " - raw monocular VO", dpi));

    std::vector<std::pair<std::string, Trajectory>> comparison;
    comparison.emplace_back("Raw VO", result.raw_trajectory);
    if (result.optimized_trajectory) {
        record("trajectory_optimized",
               plot::plot_trajectory(*result.optimized_trajectory, out_dir / ("trajectory_optimized." + extension),
                                     gt, "Optimized SLAM",
                                     "KITTI " + sequence + " - after pose-graph optimization", dpi));
        comparison.emplace_back("Optimized SLAM", *result.optimized_trajectory);
    }

    record("trajectory_comparison",
           plot::plot_trajectory_comparison(comparison, out_dir / ("trajectory_comparison." + extension), gt,
                                            "KITTI " + sequence + " - trajectory comparison", dpi));

    if (config.output.plot_3d) {
        record("trajectory_3d", plot::plot_trajectory_3d(comparison, out_dir / ("trajectory_3d." + extension), gt,
                                                         "KITTI " + sequence + " - 3D trajectory", dpi));
    }

    std::vector<std::pair<std::int64_t, std::int64_t>> loop_pairs;
    for (const auto& lc : result.loop_closures) {
        loop_pairs.emplace_back(lc.match_frame_index, lc.query_frame_index);
    }
    record("loop_closures",
           plot::plot_loop_closures(result.raw_trajectory, loop_pairs, out_dir / ("loop_closures." + extension), gt,
                                    "KITTI " + sequence + " - loop closures", dpi));

    if (result.diagnostics_available()) {
        record("diagnostics",
               plot::plot_diagnostics(result.odometry.diagnostics, out_dir / ("diagnostics." + extension),
                                      "KITTI " + sequence + " - front-end diagnostics", dpi));
    }

    if (gt) {
        std::vector<std::pair<std::string, std::vector<double>>> errors;
        errors.emplace_back("Raw VO",
                            absolute_trajectory_error(result.raw_trajectory, *gt, config.evaluation.alignment).errors);
        if (result.optimized_trajectory) {
            errors.emplace_back("Optimized SLAM", absolute_trajectory_error(*result.optimized_trajectory, *gt,
                                                                            config.evaluation.alignment)
                                                      .errors);
        }
        record("error_over_time",
               plot::plot_error_over_time(errors, out_dir / ("error_over_time." + extension), gt->cumulative_distance(),
                                          "KITTI " + sequence + " - position error", dpi));
    }
    return plots;
}

}  // namespace

Json write_outputs(SlamResult& result, const std::optional<std::filesystem::path>& output_dir) {
    const Config& config = result.config;
    const std::filesystem::path out_dir = output_dir ? *output_dir : config.run_dir();
    std::filesystem::create_directories(out_dir);
    Json artifacts = Json::object();

    const Json metrics = build_metrics(result);
    const auto metrics_path = out_dir / "metrics.json";
    {
        std::ofstream out(metrics_path, std::ios::binary | std::ios::trunc);
        out << py::dumps(metrics, 2);  // json.dump(..., indent=2): no trailing newline
    }
    artifacts["metrics"] = metrics_path.string();
    artifacts["config"] = config.to_yaml(out_dir / "config.resolved.yaml").string();
    if (config.output.save_diagnostics) {
        artifacts["diagnostics"] = write_diagnostics_csv(result, out_dir / "diagnostics.csv").string();
    }
    artifacts["trajectory_raw_txt"] = result.raw_trajectory.save_kitti(out_dir / "trajectory_raw.txt").string();
    if (result.optimized_trajectory) {
        artifacts["trajectory_optimized_txt"] =
            result.optimized_trajectory->save_kitti(out_dir / "trajectory_optimized.txt").string();
    }
    if (config.output.save_plots) {
        const Json plots = write_plots(result, out_dir);
        for (const auto& [name, path] : plots.items()) {
            artifacts[name] = path;
        }
    }
    logger().info(py::sprintf("Wrote %zu artefacts to %s", artifacts.size(), out_dir.c_str()));
    result.artifacts = artifacts;
    return artifacts;
}

SlamResult run_pipeline(const Config& config, const std::optional<std::filesystem::path>& output_dir,
                        bool write_artifacts) {
    const std::filesystem::path out_dir = output_dir ? *output_dir : config.run_dir();
    std::filesystem::create_directories(out_dir);
    setup_logging(parse_log_level(config.runtime.log_level), out_dir / "run.log");

    logger().info(std::string(78, '='));
    logger().info("Monocular SLAM | sequence " + config.dataset.sequence + " | " + out_dir.string());
    logger().info(std::string(78, '='));

    SlamPipeline pipeline(config);
    SlamResult result = pipeline.run();
    if (write_artifacts) {
        write_outputs(result, out_dir);
    }
    logger().info("Stage timings:\n" + result.timer->report());
    return result;
}

}  // namespace mvslam
