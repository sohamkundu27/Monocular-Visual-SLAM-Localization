// Resume-facing summary generation.
//
// Reads the metrics.json files that real runs produced and renders
// outputs/resume_metrics.md: a results table plus a few suggested resume
// bullets.
//
// The hard rule this module enforces is that **nothing is written that was
// not measured**. A metric that came back null (no ground truth, no loop
// closures, too short a sequence for the drift segments) is rendered as
// ``n/a`` and is never used to generate a claim. Every bullet is assembled
// from values read out of the JSON, with the ground-truth-scale caveat
// attached whenever the run used it.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mvslam/compat/pyjson.hpp"

namespace mvslam {

/// The headline numbers from one run's metrics.json. Values keep their JSON
/// type (null / int / float) because the rendering depends on it, exactly as
/// it does on the Python types.
struct RunSummary {
    std::string sequence;
    Json frames;
    Json distance_km;
    Json successful_pose_rate_pct;
    Json ate_rmse_raw_m;
    Json ate_rmse_optimized_m;
    Json ate_reduction_pct;
    Json translational_drift_raw_pct;
    Json translational_drift_optimized_pct;
    Json rotational_drift_deg_per_100m;
    Json loop_closures_detected;
    Json avg_features_per_frame;
    Json avg_matches_per_pair;
    Json avg_inlier_ratio_pct;
    Json runtime_fps;
    bool scale_uses_ground_truth = false;
    std::string alignment = "sim3";
    Json n_keyframes;

    static RunSummary from_metrics(const Json& metrics);
    static RunSummary from_file(const std::filesystem::path& path);

    /// Field access by the Python attribute name (for _best()).
    const Json& field(const std::string& name) const;
};

/// Load every ``*/metrics.json`` under ``output_root``, sorted by sequence.
/// Unreadable files are skipped with a warning.
std::vector<RunSummary> load_run_summaries(const std::filesystem::path& output_root = "outputs");

/// Markdown results table built only from measured values.
std::string results_table(const std::vector<RunSummary>& summaries);

/// Markdown table of front-end feature statistics.
std::string feature_table(const std::vector<RunSummary>& summaries);

/// Bullet list of the strongest measured results. Every entry is derived
/// from a value present in a metrics.json; nothing is inferred, rounded up,
/// or filled in.
std::vector<std::string> measured_highlights(const std::vector<RunSummary>& summaries);

/// Three concise resume bullets, assembled strictly from measured values.
std::vector<std::string> resume_bullets(const std::vector<RunSummary>& summaries);

/// Render the full resume_metrics.md document.
std::string render_resume_metrics(const std::vector<RunSummary>& summaries);

/// Generate resume_metrics.md from the runs under ``output_root``.
std::filesystem::path write_resume_metrics(const std::filesystem::path& output_root = "outputs",
                                           const std::optional<std::filesystem::path>& path = std::nullopt);

/// Markers delimiting the auto-generated results block in the README.
inline constexpr const char* kReadmeStart = "<!-- RESULTS:START -->";
inline constexpr const char* kReadmeEnd = "<!-- RESULTS:END -->";

/// Figures copied out of a run directory for display in the README, in the
/// order they should appear.
const std::vector<std::pair<std::string, std::string>>& published_figures();

/// Where published figures live (``docs/results`` by default; tests override
/// it, as the Python tests monkeypatch FIGURE_DIR).
std::filesystem::path& figure_dir();

/// Render the README results block from measured runs.
std::string render_readme_results(const std::vector<RunSummary>& summaries);

/// Copy the README-facing figures out of run directories into ``figure_dir``.
///
/// Run outputs are git-ignored (they are regenerated, and include large
/// intermediates), but a handful of result images belong in version control
/// so the README renders on GitHub. Returns the paths written.
std::vector<std::filesystem::path> publish_figures(const std::filesystem::path& output_root = "outputs",
                                                   const std::optional<std::filesystem::path>& figure_dir = std::nullopt);

/// Replace the README's results block with measured results.
///
/// Returns true when the file was rewritten. The markers must already be
/// present; this never appends a block to an arbitrary position in the file.
bool update_readme_results(const std::filesystem::path& readme_path = "README.md",
                           const std::filesystem::path& output_root = "outputs");

}  // namespace mvslam
