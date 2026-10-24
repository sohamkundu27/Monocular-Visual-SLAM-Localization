// Typed configuration system.
//
// Configuration is expressed as nested structs so every tunable has a
// documented default and a static type. YAML files supply overrides, and the
// CLI can layer further overrides on top using dotted keys
// (``--set odometry.min_inliers=40``).
//
// Unknown keys are rejected rather than silently ignored: a typo in a YAML
// file that quietly reverts a threshold to its default is the kind of bug that
// is very hard to spot in an experiment log.
//
// Values are coerced exactly as the Python dataclass loader coerces them
// (``_coerce``): bools accept true/yes/1/on strings, ints accept numeric
// strings and truncate floats but reject bools, floats accept anything
// ``float()`` does, strs take ``str()`` of the value, tuples need a list.
// YAML scalars are typed by PyYAML's resolver first (see compat/pyyaml.hpp).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/errors.hpp"

namespace mvslam {

/// KITTI Odometry dataset location and frame selection.
struct DatasetConfig {
    /// Root of the KITTI odometry download (the directory holding sequences/).
    std::string path = "data/kitti/dataset";
    /// Two-character sequence id, e.g. "00".
    std::string sequence = "00";
    /// image_0 is the left grayscale camera used for monocular SLAM.
    std::string camera = "image_0";
    /// Process at most this many frames (0 or null means the whole sequence).
    std::optional<std::int64_t> max_frames;
    /// Take every Nth frame; 1 uses every frame.
    std::int64_t frame_step = 1;
    /// Skip this many frames at the start of the sequence.
    std::int64_t start_frame = 0;
    /// Load images as grayscale (monocular front end never needs colour).
    bool grayscale = true;
};

/// Keypoint detector and descriptor settings.
struct FeatureConfig {
    /// "orb" (binary, default) or "sift" (float, requires OpenCV SIFT).
    std::string detector = "orb";
    /// Upper bound on retained keypoints per frame.
    std::int64_t max_features = 3000;
    /// ORB FAST corner threshold; lower detects more (weaker) corners.
    std::int64_t fast_threshold = 20;
    /// ORB pyramid scale factor between levels.
    double scale_factor = 1.2;
    /// Number of pyramid levels.
    std::int64_t n_levels = 8;
    /// ORB patch/edge border size in pixels.
    std::int64_t edge_threshold = 31;
    /// 0 selects the HARRIS_SCORE ranking, 1 selects FAST_SCORE.
    std::int64_t score_type = 0;
    /// Apply CLAHE contrast equalisation before detection.
    bool clahe = false;
};

/// Descriptor matching and correspondence filtering.
struct MatcherConfig {
    /// "bf" brute force (exact) -- the only backend needed at KITTI scale.
    std::string matcher = "bf";
    /// Use k-NN matching plus Lowe's ratio test (recommended).
    bool use_ratio_test = true;
    /// Lowe ratio threshold; lower is stricter.
    double ratio = 0.75;
    /// Require mutual best matches in both directions.
    bool cross_check = true;
    /// Drop matches whose descriptor distance exceeds this (null disables).
    /// Hamming units for ORB, L2 units for SIFT.
    std::optional<double> max_distance = 64.0;
    /// Frame pairs with fewer surviving matches are treated as failures.
    std::int64_t min_matches = 20;
};

/// Two-view relative pose estimation.
struct OdometryConfig {
    /// Robust estimator for the essential matrix: "magsac" (default),
    /// "usac_accurate", "ransac" or "lmeds". MAGSAC++ measurably outperforms
    /// plain RANSAC here; see geometry/epipolar.
    std::string ransac_method = "magsac";
    /// Inlier threshold in pixels. Tighter than the usual 1.0 because MAGSAC
    /// tolerates it and per-frame rotation error drops by roughly 3x.
    double ransac_threshold_px = 0.5;
    /// RANSAC target confidence.
    double ransac_confidence = 0.999;
    /// Maximum RANSAC iterations.
    std::int64_t ransac_max_iters = 2000;
    /// Maximum triangulated-point distance for the cheirality test, in units
    /// of the unit translation. OpenCV's default of 50 is too aggressive for
    /// road scenes.
    double cheirality_distance = 200.0;
    /// Reject a relative pose supported by fewer than this many inliers.
    std::int64_t min_inliers = 30;
    /// Reject a relative pose when this fraction of the epipolar inliers fails
    /// the cheirality test.
    double min_inlier_ratio = 0.3;
    /// Reject rotations larger than this between consecutive frames (degrees).
    double max_rotation_deg = 30.0;
    /// Reject transitions whose rotation-compensated parallax is below this
    /// (degrees). 0 disables. Note that parallax cannot detect pure rotation
    /// on its own -- see homography_ratio_threshold.
    double min_parallax_deg = 0.0;
    /// Treat a transition with less median optical flow than this as a
    /// stationary camera and hold the previous pose instead of integrating
    /// noise. KITTI vehicles stop at traffic lights in several sequences.
    double min_flow_px = 0.7;
    /// Homography-vs-essential model selection threshold. Above this, a
    /// homography explains the matches as well as the epipolar model, meaning
    /// the scene is planar or the motion is rotation-only.
    double homography_ratio_threshold = 0.45;
    /// Reject transitions flagged degenerate by model selection. Off by
    /// default: on KITTI the road plane can dominate legitimately, and the
    /// flag is more useful as a diagnostic than as a hard gate.
    bool reject_degenerate = false;
    /// Scale strategy: "ground_truth", "constant" or "none". See
    /// odometry/scale and the README section on monocular scale ambiguity.
    std::string scale_source = "ground_truth";
    /// Metres per frame when scale_source == "constant".
    double constant_scale = 0.85;
    /// Discard scales outside this range (metres); guards against GT gaps.
    double min_scale = 1e-3;
    double max_scale = 10.0;
};

/// Keyframe selection for the loop-closure database and pose graph.
struct KeyframeConfig {
    /// Insert a keyframe at least every N frames.
    std::int64_t every_n_frames = 5;
    /// ...or sooner if the camera has translated this far since the last one (m).
    double min_translation_m = 2.0;
    /// ...or rotated this much since the last one (degrees).
    double min_rotation_deg = 10.0;
};

/// Appearance-based loop detection and geometric verification.
struct LoopClosureConfig {
    bool enabled = true;
    /// Size of the visual vocabulary clustered from ORB descriptors. Measured
    /// on KITTI 00, top-5 retrieval recall of true loops rises from 40% at 256
    /// words to 48% at 1024; 2048 adds only ~1 point for twice the clustering
    /// cost, so 1024 is the operating point.
    std::int64_t vocabulary_size = 1024;
    /// Descriptors sampled to train the vocabulary.
    std::int64_t vocabulary_train_descriptors = 80000;
    /// Candidate keyframes must be at least this many keyframes in the past.
    std::int64_t min_keyframe_separation = 30;
    /// ...and at least this many metres of travelled path away, which prevents
    /// a slow/stopped vehicle from matching against itself.
    double min_path_separation_m = 30.0;
    /// Number of top-scoring candidates retained per query.
    std::int64_t top_k = 5;
    /// Minimum BoW cosine similarity for a candidate to be verified.
    double min_similarity = 0.20;
    /// Reject candidates scoring below this fraction of the best neighbouring
    /// (temporally adjacent) score -- the standard DBoW normalisation.
    double similarity_ratio = 0.85;
    /// Geometric verification thresholds.
    std::int64_t min_matches = 60;
    std::int64_t min_inliers = 40;
    double min_inlier_ratio = 0.35;
    /// Suppress further detections for this many keyframes after acceptance.
    std::int64_t cooldown_keyframes = 10;
    /// Cap on accepted loops (0 = unlimited); a safety valve for pathological runs.
    std::int64_t max_loops = 0;

    // -- Loop translation scale recovery ---------------------------------- //
    // A loop's translation direction comes from the essential matrix; its
    // magnitude is recovered by triangulating structure shared with a
    // metrically-scaled odometry neighbour. See loop_closure/detector.
    /// Minimum tri-view features required to trust a scale estimate.
    std::int64_t min_scale_points = 15;
    /// Minimum odometry baseline (m) to the neighbour keyframe used for
    /// metric triangulation; too small and there is no usable parallax.
    double min_scale_baseline_m = 0.5;
    /// How many keyframes back to search for that baseline.
    std::int64_t scale_neighbour_search = 5;
    /// Reject recovered loop magnitudes outside this range (metres). This is a
    /// physical plausibility gate, not just numerical hygiene: measured across
    /// KITTI 00 and 05 every true loop recovered <= 8.6 m, while the
    /// sequence-01 highway false positives recovered 12.9-27.4 m.
    double min_loop_scale_m = 0.05;
    double max_loop_scale_m = 10.0;
    /// Reject loops whose magnitude could not be measured at all. A pair that
    /// cannot be confirmed co-located is exactly where a look-alike slips past
    /// appearance and geometry. Set false to keep them as orientation-only
    /// constraints instead.
    bool require_measured_scale = true;
};

/// SE(3) pose-graph construction and optimization.
struct PoseGraphConfig {
    /// Prior on the first pose (rotation rad, translation m). Small = strong anchor.
    double prior_sigma_rot = 1e-4;
    double prior_sigma_trans = 1e-4;
    /// Odometry edge noise (rotation rad, translation m).
    double odom_sigma_rot = 0.02;
    double odom_sigma_trans = 0.10;
    /// Loop closure edge noise (rotation rad, translation m). Looser than
    /// odometry because loop constraints come from wide-baseline matches.
    double loop_sigma_rot = 0.05;
    double loop_sigma_trans = 0.30;
    /// Wrap loop factors in a robust kernel so a single false positive cannot
    /// destroy the solution.
    bool robust_loop_kernel = true;
    /// Huber threshold used by that kernel.
    double huber_k = 1.345;
    /// "levenberg_marquardt" or "gauss_newton".
    std::string optimizer = "levenberg_marquardt";
    std::int64_t max_iterations = 100;
    double relative_error_tol = 1e-5;
    double absolute_error_tol = 1e-5;
    /// Print GTSAM's per-iteration error.
    bool verbose = false;
};

/// Trajectory error metrics.
struct EvaluationConfig {
    /// Umeyama alignment mode: "sim3" (similarity, monocular-appropriate),
    /// "se3" (rigid) or "none".
    std::string alignment = "sim3";
    /// Frame gaps (in frames) used for relative pose error.
    std::vector<std::int64_t> rpe_deltas = {1, 10, 100};
    /// Sub-trajectory lengths in metres for the KITTI-style drift metric.
    std::vector<double> drift_segment_lengths_m = {100.0, 200.0, 300.0, 400.0,
                                                   500.0, 600.0, 700.0, 800.0};
};

/// Where artefacts land and how they are rendered.
struct OutputConfig {
    std::string root = "outputs";
    /// Per-run directory name; {sequence} is substituted.
    std::string run_name = "sequence_{sequence}";
    bool save_plots = true;
    bool save_diagnostics = true;
    std::int64_t dpi = 150;
    std::string plot_format = "png";
    /// Also render a 3D trajectory plot (slower, mostly for presentation).
    bool plot_3d = true;
};

/// Process-level behaviour.
struct RuntimeConfig {
    std::int64_t seed = 0;
    std::string log_level = "INFO";
    /// Emit a progress line every N frames.
    std::int64_t progress_interval = 250;
};

/// Root configuration object.
struct Config {
    DatasetConfig dataset;
    FeatureConfig features;
    MatcherConfig matcher;
    OdometryConfig odometry;
    KeyframeConfig keyframes;
    LoopClosureConfig loop_closure;
    PoseGraphConfig pose_graph;
    EvaluationConfig evaluation;
    OutputConfig output;
    RuntimeConfig runtime;

    // -- Construction ------------------------------------------------------ //
    /// Build a config from a nested mapping, validating every key. A null
    /// value yields the defaults.
    static Config from_dict(const Json& data);
    /// Load a config from a YAML file.
    static Config from_yaml(const std::filesystem::path& path);
    /// Load ``path`` (or defaults) and apply dotted-key ``overrides``.
    static Config load(const std::optional<std::filesystem::path>& path,
                       const Json& overrides = Json::object());

    // -- Mutation / serialisation ----------------------------------------- //
    /// A copy with dotted-key overrides applied (an ordered mapping
    /// ``{"odometry.min_inliers": 40}``). Null values are skipped.
    Config with_overrides(const Json& overrides) const;
    Config with_override(const std::string& dotted_key, const Json& value) const;
    /// Serialise to plain values (JSON/YAML friendly), in declaration order.
    Json to_dict() const;
    /// Write the resolved config to ``path`` for run reproducibility.
    std::filesystem::path to_yaml(const std::filesystem::path& path) const;

    /// Output directory for this configuration's sequence.
    std::filesystem::path run_dir() const;
};

/// ``str.format(sequence=...)`` for the run-name template: ``{sequence}`` is
/// substituted and ``{{``/``}}`` escape braces. Any other field raises KeyError.
std::string format_run_name(const std::string& pattern, const std::string& sequence);

}  // namespace mvslam
