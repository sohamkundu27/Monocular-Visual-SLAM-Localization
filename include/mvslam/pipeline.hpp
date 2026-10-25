// End-to-end monocular SLAM pipeline.
//
// Stages
// ------
// 1. **Front end** -- visual odometry over the sequence, harvesting keyframes
//    as it goes so descriptors are computed exactly once.
// 2. **Vocabulary** -- cluster a descriptor sample from the harvested
//    keyframes.
// 3. **Loop closure** -- appearance retrieval plus geometric verification.
// 4. **Back end** -- build the keyframe pose graph and optimize it with GTSAM.
// 5. **Interpolation** -- propagate the keyframe corrections back to every
//    frame, so the optimized trajectory can be evaluated against dense ground
//    truth.
// 6. **Evaluation** -- metrics for the raw and optimized trajectories, plots,
//    diagnostics and a machine-readable summary.
//
// Every number written to metrics.json comes from this run. When ground truth
// is unavailable the metric keys are emitted as null rather than filled with
// plausible-looking values.
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/config.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/evaluation/metrics.hpp"
#include "mvslam/geometry/pose.hpp"
#include "mvslam/loop_closure/detector.hpp"
#include "mvslam/odometry/visual_odometry.hpp"
#include "mvslam/optimization/gtsam_backend.hpp"
#include "mvslam/utils/timing.hpp"

namespace mvslam {

/// Everything one pipeline run produces.
struct SlamResult {
    Config config;
    Json dataset_info = Json::object();
    OdometryResult odometry;
    Trajectory raw_trajectory;
    std::optional<Trajectory> optimized_trajectory;
    std::optional<Trajectory> ground_truth;
    std::vector<LoopClosure> loop_closures;
    std::optional<OptimizationResult> optimization;
    std::shared_ptr<StageTimer> timer;
    std::optional<TrajectoryEvaluation> evaluation_raw;
    std::optional<TrajectoryEvaluation> evaluation_optimized;
    std::vector<std::int64_t> keyframe_frame_indices;
    Json loop_stats = Json::object();
    /// ``{name: path}`` of the artefacts written, in insertion order.
    Json artifacts = Json::object();

    bool has_ground_truth() const { return ground_truth.has_value(); }
    bool diagnostics_available() const { return !odometry.diagnostics.empty(); }
};

/// Runs the full SLAM pipeline for one sequence.
class SlamPipeline {
public:
    explicit SlamPipeline(Config config);

    /// Execute the pipeline end to end (the dataset is built from the config
    /// when not given).
    SlamResult run(const KittiOdometryDataset* dataset = nullptr);

    const Config& config() const { return config_; }
    StageTimer& timer() { return *timer_; }

private:
    std::pair<std::vector<LoopClosure>, Json> detect_loops(const KittiOdometryDataset& dataset,
                                                           const VisualOdometry& vo,
                                                           KeyframeDatabase& database);
    std::pair<OptimizationResult, Trajectory> optimize(const KeyframeDatabase& database,
                                                       const Trajectory& raw_trajectory,
                                                       const std::vector<LoopClosure>& loop_closures);

    Config config_;
    std::shared_ptr<StageTimer> timer_;
};

/// Push keyframe-level corrections back onto every frame.
///
/// The pose graph only optimizes keyframes, but evaluation compares against
/// ground truth at every frame. For a frame between keyframes ``a`` and
/// ``b``, the correction ``C_a = T_opt_a * inv(T_raw_a)`` from the preceding
/// keyframe is applied.
///
/// Applying the *preceding* keyframe's correction (rather than blending
/// between the two) keeps the odometry between keyframes exactly intact -- the
/// front end's local estimate is the best information available there, and
/// interpolating would distort it. The trade-off is a small discontinuity at
/// each keyframe boundary, which is bounded by how much the optimizer moved
/// that keyframe: metres over a full sequence, centimetres between adjacent
/// keyframes.
Trajectory propagate_keyframe_corrections(const Trajectory& dense,
                                          const std::vector<std::int64_t>& keyframe_indices,
                                          const Trajectory& optimized_keyframes);

/// Round for JSON output, mapping missing and non-finite values to null.
Json round_metric(const std::optional<double>& value, int digits = 4);

/// Assemble the machine-readable metrics summary.
///
/// Keys whose value cannot be measured on this run (no ground truth, no loop
/// closures) are emitted as null. Nothing here is estimated or defaulted.
Json build_metrics(const SlamResult& result);

/// Write every artefact for a run and return ``{name: path}``.
Json write_outputs(SlamResult& result, const std::optional<std::filesystem::path>& output_dir = std::nullopt);

/// Write the per-frame diagnostics table.
std::filesystem::path write_diagnostics_csv(const SlamResult& result, const std::filesystem::path& path);

/// Convenience wrapper: configure logging, run, and write outputs.
SlamResult run_pipeline(const Config& config,
                        const std::optional<std::filesystem::path>& output_dir = std::nullopt,
                        bool write_artifacts = true);

}  // namespace mvslam
