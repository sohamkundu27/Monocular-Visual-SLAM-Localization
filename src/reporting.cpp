#include "mvslam/reporting.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.reporting");
    return instance;
}

Json get(const Json& metrics, const char* key) {
    return metrics.is_object() && metrics.contains(key) ? metrics.at(key) : Json(nullptr);
}

/// Python truthiness of a JSON value.
bool truthy(const Json& v) {
    if (v.is_null()) {
        return false;
    }
    if (v.is_boolean()) {
        return v.get<bool>();
    }
    if (v.is_number()) {
        return py::as_double(v) != 0.0;
    }
    if (v.is_string() || v.is_array() || v.is_object()) {
        return !v.empty();
    }
    return true;
}

double num(const Json& v) { return py::as_double(v); }

/// Render a value, or ``n/a`` when it was not measured.
std::string fmt(const Json& value, int digits = 2, const std::string& suffix = "") {
    if (value.is_null()) {
        return "n/a";
    }
    if (value.is_number_integer() || value.is_number_unsigned()) {
        return std::to_string(value.get<std::int64_t>()) + suffix;
    }
    return py::format(num(value), "." + std::to_string(digits) + "f") + suffix;
}

std::string f2(double v, int digits) { return py::format(v, "." + std::to_string(digits) + "f"); }

/// ``f"{n:,}"`` for an int or float total.
std::string grouped(const Json& total) {
    if (total.is_number_float()) {
        return py::format(num(total), ",.1f");
    }
    return py::format(total.get<std::int64_t>(), ",");
}

/// sum(...) over the truthy values of a field, keeping int when all are ints.
Json total_of(const std::vector<RunSummary>& summaries, const std::string& field) {
    bool all_int = true;
    std::int64_t int_total = 0;
    double float_total = 0.0;
    for (const auto& s : summaries) {
        const Json& v = s.field(field);
        if (!truthy(v)) {
            continue;
        }
        if (v.is_number_integer()) {
            int_total += v.get<std::int64_t>();
        } else {
            all_int = false;
        }
        float_total += num(v);
    }
    return all_int ? Json(int_total) : py::make_float(float_total);
}

/// The run with the best measured value of ``attribute`` (first on ties).
const RunSummary* best(const std::vector<RunSummary>& summaries, const std::string& attribute, bool largest) {
    const RunSummary* chosen = nullptr;
    for (const auto& s : summaries) {
        const Json& v = s.field(attribute);
        if (v.is_null()) {
            continue;
        }
        if (chosen == nullptr) {
            chosen = &s;
            continue;
        }
        const double current = num(chosen->field(attribute));
        if (largest ? num(v) > current : num(v) < current) {
            chosen = &s;
        }
    }
    return chosen;
}

/// [v for v in values if v] -> min/max; raises like min([]) when empty.
std::pair<double, double> truthy_min_max(const std::vector<RunSummary>& summaries, const std::string& field) {
    std::vector<double> values;
    for (const auto& s : summaries) {
        if (truthy(s.field(field))) {
            values.push_back(num(s.field(field)));
        }
    }
    if (values.empty()) {
        throw ValueError("min() arg is an empty sequence");
    }
    return {*std::min_element(values.begin(), values.end()), *std::max_element(values.begin(), values.end())};
}

std::string join(const std::vector<std::string>& parts, const std::string& sep) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        out += (i ? sep : "") + parts[i];
    }
    return out;
}

/// Call out any sequence where optimization did not help.
///
/// Reporting only the sequences that improved would be selective quotation.
/// A reader comparing against the results table would spot the gap anyway,
/// and a stated limitation is more credible than a silent one.
std::vector<std::string> regression_note(const std::vector<RunSummary>& summaries) {
    std::vector<const RunSummary*> regressed;
    for (const auto& s : summaries) {
        if (!s.ate_reduction_pct.is_null() && num(s.ate_reduction_pct) < 0) {
            regressed.push_back(&s);
        }
    }
    if (regressed.empty()) {
        return {};
    }
    std::vector<std::string> lines = {"## Where it does not help", ""};
    for (const RunSummary* s : regressed) {
        const std::int64_t n = truthy(s->loop_closures_detected) ? s->loop_closures_detected.get<std::int64_t>() : 0;
        const std::string plural = n == 1 ? "loop closure" : "loop closures";
        lines.push_back("- **Sequence " + s->sequence + "**: optimization made ATE *worse* (" +
                        f2(num(s->ate_rmse_raw_m), 2) + " m -> " + f2(num(s->ate_rmse_optimized_m), 2) + " m, " +
                        f2(num(s->ate_reduction_pct), 1) + "%) from " + std::to_string(n) + " accepted " + plural + ".");
    }
    const std::vector<std::string> tail = {
        "",
        "  KITTI sequence 01 is a fast highway drive that contains **no true loops**, so the",
        "  correct number of closures is zero. Its repetitive corridor (guardrails, lane",
        "  markings, uniform vegetation) defeats appearance matching *and* geometric",
        "  verification, because forward motion along a straight road is a consistent camera",
        "  motion between any two points on it. The metric-plausibility gate removes most such",
        "  detections; the residual ones remain a known limitation, and switchable constraints",
        "  or GNC would be the proper fix.",
        "",
    };
    lines.insert(lines.end(), tail.begin(), tail.end());
    return lines;
}

/// Embed published figures, falling back to a pointer when absent.
std::vector<std::string> figure_section(const std::vector<RunSummary>& summaries) {
    std::vector<std::string> lines = {"### Trajectories", ""};
    bool any_published = false;
    for (const auto& s : summaries) {
        std::vector<std::pair<std::string, std::string>> published;
        for (const auto& [name, caption] : published_figures()) {
            if (std::filesystem::is_regular_file(figure_dir() / ("sequence_" + s.sequence) / name)) {
                published.emplace_back(name, caption);
            }
        }
        if (published.empty()) {
            continue;
        }
        any_published = true;
        lines.push_back("**Sequence " + s.sequence + "**");
        lines.push_back("");
        for (const auto& [name, caption] : published) {
            const auto path = figure_dir() / ("sequence_" + s.sequence) / name;
            lines.push_back("![" + caption + " — KITTI sequence " + s.sequence + "](" + path.generic_string() + ")");
            lines.push_back("");
            lines.push_back("*" + caption + ".*");
            lines.push_back("");
        }
    }
    if (!any_published) {
        lines.push_back(
            "_Figures are written to `outputs/sequence_XX/` by each run. Publish them into "
            "`docs/results/` with `build/report --publish-figures --update-readme`._");
        lines.push_back("");
    }
    return lines;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw OSError("[Errno 2] No such file or directory: " + py::repr_str(path.string()));
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

}  // namespace

// --------------------------------------------------------------------------- //
// RunSummary
// --------------------------------------------------------------------------- //

RunSummary RunSummary::from_metrics(const Json& metrics) {
    RunSummary s;
    s.sequence = metrics.contains("sequence") ? py::str(metrics.at("sequence")) : "?";
    s.frames = get(metrics, "frames");
    s.distance_km = get(metrics, "distance_km");
    s.successful_pose_rate_pct = get(metrics, "successful_pose_rate_pct");
    s.ate_rmse_raw_m = get(metrics, "ate_rmse_raw_m");
    s.ate_rmse_optimized_m = get(metrics, "ate_rmse_optimized_m");
    s.ate_reduction_pct = get(metrics, "ate_reduction_pct");
    s.translational_drift_raw_pct = get(metrics, "translational_drift_raw_pct");
    s.translational_drift_optimized_pct = get(metrics, "translational_drift_optimized_pct");
    s.rotational_drift_deg_per_100m = get(metrics, "rotational_drift_deg_per_100m");
    s.loop_closures_detected = get(metrics, "loop_closures_detected");
    s.avg_features_per_frame = get(metrics, "avg_features_per_frame");
    s.avg_matches_per_pair = get(metrics, "avg_matches_per_pair");
    s.avg_inlier_ratio_pct = get(metrics, "avg_inlier_ratio_pct");
    s.runtime_fps = get(metrics, "runtime_fps");
    s.scale_uses_ground_truth = truthy(get(metrics, "scale_uses_ground_truth"));
    s.alignment = metrics.contains("alignment") ? py::str(metrics.at("alignment")) : "sim3";
    const Json loop_closure = get(metrics, "loop_closure");
    s.n_keyframes = truthy(loop_closure) ? get(loop_closure, "n_keyframes") : Json(nullptr);
    return s;
}

RunSummary RunSummary::from_file(const std::filesystem::path& path) {
    return from_metrics(py::loads(read_file(path)));
}

const Json& RunSummary::field(const std::string& name) const {
    static const Json null_value;
    if (name == "frames") return frames;
    if (name == "distance_km") return distance_km;
    if (name == "successful_pose_rate_pct") return successful_pose_rate_pct;
    if (name == "ate_rmse_raw_m") return ate_rmse_raw_m;
    if (name == "ate_rmse_optimized_m") return ate_rmse_optimized_m;
    if (name == "ate_reduction_pct") return ate_reduction_pct;
    if (name == "translational_drift_raw_pct") return translational_drift_raw_pct;
    if (name == "translational_drift_optimized_pct") return translational_drift_optimized_pct;
    if (name == "rotational_drift_deg_per_100m") return rotational_drift_deg_per_100m;
    if (name == "loop_closures_detected") return loop_closures_detected;
    if (name == "avg_features_per_frame") return avg_features_per_frame;
    if (name == "avg_matches_per_pair") return avg_matches_per_pair;
    if (name == "avg_inlier_ratio_pct") return avg_inlier_ratio_pct;
    if (name == "runtime_fps") return runtime_fps;
    if (name == "n_keyframes") return n_keyframes;
    return null_value;
}

std::vector<RunSummary> load_run_summaries(const std::filesystem::path& output_root) {
    std::vector<std::filesystem::path> paths;
    if (std::filesystem::is_directory(output_root)) {
        for (const auto& entry : std::filesystem::directory_iterator(output_root)) {
            const auto candidate = entry.path() / "metrics.json";
            if (entry.is_directory() && std::filesystem::exists(candidate)) {
                paths.push_back(candidate);
            }
        }
    }
    std::sort(paths.begin(), paths.end());
    std::vector<RunSummary> summaries;
    for (const auto& path : paths) {
        try {
            summaries.push_back(RunSummary::from_file(path));
        } catch (const OSError& exc) {
            logger().warning("Skipping unreadable metrics file " + path.string() + ": " + exc.what());
        } catch (const JsonDecodeError& exc) {
            logger().warning("Skipping unreadable metrics file " + path.string() + ": " + exc.what());
        }
    }
    std::stable_sort(summaries.begin(), summaries.end(),
                     [](const RunSummary& a, const RunSummary& b) { return a.sequence < b.sequence; });
    return summaries;
}

// --------------------------------------------------------------------------- //
// Tables, highlights, bullets
// --------------------------------------------------------------------------- //

std::string results_table(const std::vector<RunSummary>& summaries) {
    if (summaries.empty()) {
        return "_No benchmark runs found. Run `build/run_slam --sequence 00 "
               "--dataset-path /path/to/KITTI/dataset` to populate this table._";
    }
    std::string out =
        "| Sequence | Frames | Distance | Pose rate | Loops | "
        "ATE RMSE (raw) | ATE RMSE (opt.) | ATE reduction | "
        "Drift (raw) | Drift (opt.) | Rot. drift | FPS |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    std::vector<std::string> rows;
    for (const auto& s : summaries) {
        rows.push_back("| " + s.sequence + " | " + fmt(s.frames) + " | " + fmt(s.distance_km, 2, " km") + " | " +
                       fmt(s.successful_pose_rate_pct, 1, "%") + " | " + fmt(s.loop_closures_detected) + " | " +
                       fmt(s.ate_rmse_raw_m, 2, " m") + " | " + fmt(s.ate_rmse_optimized_m, 2, " m") + " | " +
                       fmt(s.ate_reduction_pct, 1, "%") + " | " + fmt(s.translational_drift_raw_pct, 2, "%") +
                       " | " + fmt(s.translational_drift_optimized_pct, 2, "%") + " | " +
                       fmt(s.rotational_drift_deg_per_100m, 3) + " | " + fmt(s.runtime_fps, 1) + " |");
    }
    return out + join(rows, "\n");
}

std::string feature_table(const std::vector<RunSummary>& summaries) {
    if (summaries.empty()) {
        return "_No benchmark runs found._";
    }
    std::string out =
        "| Sequence | Keyframes | Avg. features/frame | Avg. matches/pair | "
        "Avg. RANSAC inlier ratio |\n"
        "|---:|---:|---:|---:|---:|\n";
    std::vector<std::string> rows;
    for (const auto& s : summaries) {
        rows.push_back("| " + s.sequence + " | " + fmt(s.n_keyframes) + " | " + fmt(s.avg_features_per_frame, 0) +
                       " | " + fmt(s.avg_matches_per_pair, 0) + " | " + fmt(s.avg_inlier_ratio_pct, 1, "%") + " |");
    }
    return out + join(rows, "\n");
}

std::vector<std::string> measured_highlights(const std::vector<RunSummary>& summaries) {
    if (summaries.empty()) {
        return {};
    }
    std::vector<std::string> highlights;
    const Json total_frames = total_of(summaries, "frames");
    const Json total_km = total_of(summaries, "distance_km");
    std::vector<std::string> names;
    for (const auto& s : summaries) {
        names.push_back(s.sequence);
    }
    if (truthy(total_frames)) {
        highlights.push_back("Processed **" + grouped(total_frames) + " KITTI frames** across **" +
                             f2(num(total_km), 2) + " km** of driving (sequences " + join(names, ", ") + ").");
    }
    if (best(summaries, "successful_pose_rate_pct", true) != nullptr) {
        const auto [lo, hi] = truthy_min_max(summaries, "successful_pose_rate_pct");
        highlights.push_back("Achieved a **" + f2(lo, 1) + "-" + f2(hi, 1) +
                             "% successful relative-pose rate** across all evaluated sequences.");
    }
    if (const RunSummary* ate = best(summaries, "ate_rmse_optimized_m", false)) {
        highlights.push_back("Achieved **" + f2(num(ate->ate_rmse_optimized_m), 2) +
                             " m optimized ATE RMSE** on sequence " + ate->sequence + " (" +
                             f2(num(ate->distance_km), 2) + " km, " + ate->alignment + " alignment).");
    }
    if (const RunSummary* drift = best(summaries, "translational_drift_optimized_pct", false)) {
        highlights.push_back("Achieved **" + f2(num(drift->translational_drift_optimized_pct), 2) +
                             "% translational drift** on sequence " + drift->sequence +
                             " (KITTI 100-800 m sub-trajectory metric).");
    }
    const RunSummary* reduction = best(summaries, "ate_reduction_pct", true);
    if (reduction != nullptr && num(reduction->ate_reduction_pct) > 0) {
        highlights.push_back("Reduced ATE RMSE by **" + f2(num(reduction->ate_reduction_pct), 1) + "%** on sequence " +
                             reduction->sequence + " (" + f2(num(reduction->ate_rmse_raw_m), 2) + " m -> " +
                             f2(num(reduction->ate_rmse_optimized_m), 2) +
                             " m) via loop closure and pose-graph optimization.");
    }
    const RunSummary* loops = best(summaries, "loop_closures_detected", true);
    if (loops != nullptr && truthy(loops->loop_closures_detected)) {
        std::int64_t total_loops = 0;
        for (const auto& s : summaries) {
            total_loops += truthy(s.loop_closures_detected) ? s.loop_closures_detected.get<std::int64_t>() : 0;
        }
        highlights.push_back("Detected and geometrically verified **" + std::to_string(total_loops) +
                             " loop closures** (" + py::str(loops->loop_closures_detected) + " on sequence " +
                             loops->sequence + ").");
    }
    if (best(summaries, "runtime_fps", true) != nullptr) {
        const auto [lo, hi] = truthy_min_max(summaries, "runtime_fps");
        highlights.push_back("Ran the full pipeline at **" + f2(lo, 1) + "-" + f2(hi, 1) +
                             " FPS** on CPU, end to end including loop closure and optimization.");
    }
    return highlights;
}

std::vector<std::string> resume_bullets(const std::vector<RunSummary>& summaries) {
    if (summaries.empty()) {
        return {};
    }
    std::vector<std::string> bullets;
    const Json total_frames = total_of(summaries, "frames");
    const Json total_km = total_of(summaries, "distance_km");
    std::vector<std::string> names;
    for (const auto& s : summaries) {
        names.push_back(s.sequence);
    }

    // Bullet 1: the system and its front-end scale.
    std::vector<std::string> parts = {
        "Built a monocular visual SLAM system in C++17 (OpenCV, GTSAM) implementing ORB "
        "feature tracking, MAGSAC++ essential-matrix pose estimation, bag-of-words loop "
        "closure and SE(3) pose-graph optimization"};
    if (truthy(total_frames)) {
        parts.push_back("processing " + grouped(total_frames) + " KITTI Odometry frames over " + f2(num(total_km), 1) +
                        " km (sequences " + join(names, "/") + ")");
    }
    std::vector<double> rates;
    for (const auto& s : summaries) {
        if (truthy(s.successful_pose_rate_pct)) {
            rates.push_back(num(s.successful_pose_rate_pct));
        }
    }
    if (!rates.empty()) {
        parts.push_back("at a " + f2(*std::min_element(rates.begin(), rates.end()), 1) +
                        "%+ successful relative-pose rate");
    }
    bullets.push_back(join(parts, ", ") + ".");

    // Bullet 2: accuracy, with the scale caveat attached where it applies.
    const RunSummary* ate = best(summaries, "ate_rmse_optimized_m", false);
    const RunSummary* drift = best(summaries, "translational_drift_optimized_pct", false);
    if (ate != nullptr) {
        std::string accuracy = "Benchmarked against KITTI ground truth with Umeyama " + ate->alignment +
                               " alignment, achieving " + f2(num(ate->ate_rmse_optimized_m), 2) +
                               " m absolute trajectory error (RMSE) on sequence " + ate->sequence + " over " +
                               f2(num(ate->distance_km), 2) + " km";
        if (drift != nullptr) {
            accuracy += " and " + f2(num(drift->translational_drift_optimized_pct), 2) +
                        "% translational drift on the KITTI 100-800 m sub-trajectory metric";
        }
        if (std::any_of(summaries.begin(), summaries.end(), [](const RunSummary& s) { return s.scale_uses_ground_truth; })) {
            accuracy +=
                "; monocular scale ambiguity handled by an explicitly documented "
                "ground-truth-scaled evaluation protocol";
        }
        bullets.push_back(accuracy + ".");
    }

    // Bullet 3: the back end's measured contribution.
    const RunSummary* reduction = best(summaries, "ate_reduction_pct", true);
    std::int64_t total_loops = 0;
    for (const auto& s : summaries) {
        total_loops += truthy(s.loop_closures_detected) ? s.loop_closures_detected.get<std::int64_t>() : 0;
    }
    if (reduction != nullptr && num(reduction->ate_reduction_pct) > 0) {
        bullets.push_back("Cut absolute trajectory error by " + f2(num(reduction->ate_reduction_pct), 1) + "% (" +
                          f2(num(reduction->ate_rmse_raw_m), 2) + " m -> " +
                          f2(num(reduction->ate_rmse_optimized_m), 2) + " m on sequence " + reduction->sequence +
                          ") by detecting " + std::to_string(total_loops) +
                          " geometrically verified loop closures and optimizing the keyframe pose graph with GTSAM "
                          "Levenberg-Marquardt under robust Huber kernels.");
    } else if (total_loops) {
        bullets.push_back("Detected " + std::to_string(total_loops) +
                          " geometrically verified loop closures via TF-IDF bag-of-visual-words retrieval with "
                          "essential-matrix verification, feeding an SE(3) pose graph optimized with GTSAM.");
    }

    std::vector<double> fps;
    for (const auto& s : summaries) {
        if (truthy(s.runtime_fps)) {
            fps.push_back(num(s.runtime_fps));
        }
    }
    if (bullets.size() < 3 && !fps.empty()) {
        bullets.push_back("Instrumented the pipeline end to end, sustaining " +
                          f2(*std::min_element(fps.begin(), fps.end()), 1) + "-" +
                          f2(*std::max_element(fps.begin(), fps.end()), 1) +
                          " FPS on CPU with per-stage timing for feature extraction, matching, pose estimation, "
                          "loop closure and optimization.");
    }
    if (bullets.size() > 3) {
        bullets.resize(3);
    }
    return bullets;
}

std::string render_resume_metrics(const std::vector<RunSummary>& summaries) {
    std::vector<std::string> lines = {
        "# Resume metrics",
        "",
        "Every number below was produced by an actual run of this repository against the",
        "KITTI Odometry dataset and read back from the corresponding",
        "`outputs/sequence_XX/metrics.json`. Nothing here is estimated, and values that",
        "were not measurable on a given run are shown as `n/a` rather than filled in.",
        "",
    };
    if (summaries.empty()) {
        const std::vector<std::string> none = {
            "## No runs found",
            "",
            "No `outputs/sequence_*/metrics.json` files are present, so there are no",
            "measured results to report. Run the pipeline first:",
            "",
            "```bash",
            "build/run_slam --sequence 00 --dataset-path /path/to/KITTI/dataset",
            "```",
            "",
        };
        lines.insert(lines.end(), none.begin(), none.end());
        return join(lines, "\n");
    }

    if (std::any_of(summaries.begin(), summaries.end(), [](const RunSummary& s) { return s.scale_uses_ground_truth; })) {
        const std::vector<std::string> caveat = {
            "> **Scale caveat.** These runs used `odometry.scale_source: ground_truth`, which",
            "> takes the per-frame translation *magnitude* from KITTI ground truth. Monocular",
            "> vision cannot recover absolute scale; the estimator recovers the translation",
            "> *direction* and the full rotation. These figures therefore measure trajectory",
            "> shape, heading drift and loop-closure benefit — not metric-scale localization.",
            "> Any use of these numbers must carry the same caveat.",
            "",
        };
        lines.insert(lines.end(), caveat.begin(), caveat.end());
    }
    for (const std::string& line : {std::string("## Results"), std::string(""), results_table(summaries), std::string(""),
                                    std::string("## Front-end statistics"), std::string(""), feature_table(summaries),
                                    std::string("")}) {
        lines.push_back(line);
    }
    const auto highlights = measured_highlights(summaries);
    if (!highlights.empty()) {
        lines.push_back("## Measured highlights");
        lines.push_back("");
        for (const auto& h : highlights) {
            lines.push_back("- " + h);
        }
        lines.push_back("");
    }
    for (const auto& line : regression_note(summaries)) {
        lines.push_back(line);
    }
    const auto bullets = resume_bullets(summaries);
    if (!bullets.empty()) {
        lines.push_back("## Suggested resume bullets");
        lines.push_back("");
        lines.push_back("Assembled entirely from the measured values above.");
        lines.push_back("");
        for (std::size_t i = 0; i < bullets.size(); ++i) {
            lines.push_back(std::to_string(i + 1) + ". " + bullets[i]);
            lines.push_back("");
        }
    }
    const std::vector<std::string> definitions = {
        "## Metric definitions",
        "",
        "- **ATE RMSE** — root-mean-square Euclidean distance between estimated and",
        "  ground-truth camera positions after Umeyama alignment.",
        "- **ATE reduction** — `(ATE_raw - ATE_optimized) / ATE_raw * 100`.",
        "- **Translational drift** — KITTI odometry metric: translation error over",
        "  100-800 m sub-trajectories, normalised by sub-trajectory length.",
        "- **Rotational drift** — the same sub-trajectory protocol, reporting rotation",
        "  error in degrees per 100 m.",
        "- **Successful pose rate** — validated relative poses divided by attempted frame",
        "  transitions. A correctly detected stationary frame counts as a success.",
        "- **FPS** — total frames divided by end-to-end wall-clock runtime, including",
        "  loop closure and pose-graph optimization.",
        "",
    };
    lines.insert(lines.end(), definitions.begin(), definitions.end());
    return join(lines, "\n");
}

std::filesystem::path write_resume_metrics(const std::filesystem::path& output_root,
                                           const std::optional<std::filesystem::path>& path) {
    const auto summaries = load_run_summaries(output_root);
    const std::filesystem::path out = path ? *path : output_root / "resume_metrics.md";
    if (out.has_parent_path()) {
        std::filesystem::create_directories(out.parent_path());
    }
    std::ofstream file(out, std::ios::binary | std::ios::trunc);
    file << render_resume_metrics(summaries);
    logger().info("Wrote " + out.string() + " from " + std::to_string(summaries.size()) + " run(s)");
    return out;
}

const std::vector<std::pair<std::string, std::string>>& published_figures() {
    static const std::vector<std::pair<std::string, std::string>> figures = {
        {"trajectory_comparison.png", "Ground truth vs raw VO vs optimized SLAM"},
        {"loop_closures.png", "Accepted loop-closure edges"},
        {"error_over_time.png", "Position error against distance travelled"},
    };
    return figures;
}

std::filesystem::path& figure_dir() {
    static std::filesystem::path dir = "docs/results";
    return dir;
}

std::string render_readme_results(const std::vector<RunSummary>& summaries) {
    if (summaries.empty()) {
        return "_No benchmark runs recorded yet. Run the pipeline (see "
               "[Usage](#usage)) and then `build/report --update-readme`._";
    }
    std::vector<std::string> lines;
    if (std::any_of(summaries.begin(), summaries.end(), [](const RunSummary& s) { return s.scale_uses_ground_truth; })) {
        lines.push_back(
            "> **All figures below are from real runs of this repository on KITTI Odometry.** "
            "They use `odometry.scale_source: ground_truth`, which takes the per-frame "
            "translation *magnitude* from ground truth. Monocular vision cannot recover "
            "absolute scale; the estimator recovers translation *direction* and full rotation. "
            "These numbers therefore measure trajectory shape, heading drift and loop-closure "
            "benefit — **not** metric-scale localization. See "
            "[Monocular scale ambiguity](#4-monocular-scale-ambiguity).");
        lines.push_back("");
    } else {
        lines.push_back("> **All figures below are from real runs of this repository on KITTI Odometry.**");
        lines.push_back("");
    }
    lines.push_back(results_table(summaries));
    lines.push_back("");
    lines.push_back("### Front-end statistics");
    lines.push_back("");
    lines.push_back(feature_table(summaries));
    lines.push_back("");
    const auto highlights = measured_highlights(summaries);
    if (!highlights.empty()) {
        lines.push_back("### Highlights");
        lines.push_back("");
        for (const auto& h : highlights) {
            lines.push_back("- " + h);
        }
        lines.push_back("");
    }
    for (const auto& line : regression_note(summaries)) {
        lines.push_back(line == "## Where it does not help" ? "### Where it does not help" : line);
    }
    for (const auto& line : figure_section(summaries)) {
        lines.push_back(line);
    }
    lines.push_back("_Regenerate with `build/report --update-readme`._");
    return join(lines, "\n");
}

std::vector<std::filesystem::path> publish_figures(const std::filesystem::path& output_root,
                                                   const std::optional<std::filesystem::path>& figure_dir_in) {
    const std::filesystem::path destination_root = figure_dir_in ? *figure_dir_in : figure_dir();
    std::vector<std::filesystem::path> written;
    std::vector<std::filesystem::path> run_dirs;
    if (std::filesystem::is_directory(output_root)) {
        for (const auto& entry : std::filesystem::directory_iterator(output_root)) {
            if (entry.path().filename().string().rfind("sequence_", 0) == 0) {
                run_dirs.push_back(entry.path());
            }
        }
    }
    std::sort(run_dirs.begin(), run_dirs.end());
    for (const auto& run_dir : run_dirs) {
        if (!std::filesystem::is_regular_file(run_dir / "metrics.json")) {
            continue;
        }
        const auto destination = destination_root / run_dir.filename();
        std::filesystem::create_directories(destination);
        for (const auto& [name, caption] : published_figures()) {
            const auto source = run_dir / name;
            if (std::filesystem::is_regular_file(source)) {
                std::filesystem::copy_file(source, destination / name,
                                           std::filesystem::copy_options::overwrite_existing);
                written.push_back(destination / name);
            }
        }
    }
    logger().info("Published " + std::to_string(written.size()) + " figures to " + destination_root.string());
    return written;
}

bool update_readme_results(const std::filesystem::path& readme_path, const std::filesystem::path& output_root) {
    if (!std::filesystem::is_regular_file(readme_path)) {
        logger().warning("README not found at " + readme_path.string() + "; skipping results injection");
        return false;
    }
    const std::string text = read_file(readme_path);
    const auto start = text.find(kReadmeStart);
    const auto end_marker = text.find(kReadmeEnd);
    if (start == std::string::npos || end_marker == std::string::npos) {
        logger().warning(std::string("README is missing the ") + kReadmeStart + " / " + kReadmeEnd +
                         " markers; skipping results injection");
        return false;
    }
    // str.partition semantics: the first START, then the first END after it.
    const std::string before = text.substr(0, start);
    const std::string rest = text.substr(start + std::string(kReadmeStart).size());
    const auto end = rest.find(kReadmeEnd);
    const std::string after = end == std::string::npos ? "" : rest.substr(end + std::string(kReadmeEnd).size());
    const std::string block = render_readme_results(load_run_summaries(output_root));
    std::ofstream out(readme_path, std::ios::binary | std::ios::trunc);
    out << before << kReadmeStart << "\n" << block << "\n" << kReadmeEnd << after;
    logger().info("Updated results block in " + readme_path.string());
    return true;
}

}  // namespace mvslam
