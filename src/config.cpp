#include "mvslam/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <sstream>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/compat/pyyaml.hpp"

namespace mvslam {

namespace {

// --------------------------------------------------------------------------- //
// Python conversions used by _coerce
// --------------------------------------------------------------------------- //

std::string strip(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\n\r\f\v");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(begin, end - begin + 1);
}

[[noreturn]] void fail(const std::string& message) { throw ConfigError(message); }

std::string coerce_str(const Json& value, const std::string& path) {
    if (value.is_null()) {
        fail("Config key '" + path + "' does not accept null");
    }
    return py::str(value);
}

bool coerce_bool(const Json& value, const std::string& path) {
    if (value.is_null()) {
        fail("Config key '" + path + "' does not accept null");
    }
    if (value.is_boolean()) {
        return value.get<bool>();
    }
    if (value.is_string()) {
        std::string lowered = strip(value.get<std::string>());
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lowered == "true" || lowered == "yes" || lowered == "1" || lowered == "on") {
            return true;
        }
        if (lowered == "false" || lowered == "no" || lowered == "0" || lowered == "off") {
            return false;
        }
    }
    fail("Config key '" + path + "' expects a boolean, got " + py::repr(value));
}

std::int64_t coerce_int(const Json& value, const std::string& path) {
    if (value.is_null()) {
        fail("Config key '" + path + "' does not accept null");
    }
    const std::string error = "Config key '" + path + "' expects an int, got " + py::repr(value);
    if (value.is_boolean()) {
        fail(error);
    }
    if (value.is_number_integer() || value.is_number_unsigned()) {
        return value.get<std::int64_t>();
    }
    if (value.is_number_float()) {
        const double v = value.get<double>();
        if (std::isnan(v)) {
            fail(error);  // int(nan) raises ValueError
        }
        if (std::isinf(v)) {
            // int(inf) raises OverflowError, which the Python loader does not
            // translate into a ConfigError.
            throw Error("cannot convert float infinity to integer");
        }
        if (std::fabs(v) >= 9.2e18) {
            fail(error);
        }
        return static_cast<std::int64_t>(std::trunc(v));
    }
    if (value.is_string()) {
        if (auto parsed = py::parse_int(value.get<std::string>())) {
            return *parsed;
        }
    }
    fail(error);
}

double coerce_float(const Json& value, const std::string& path) {
    if (value.is_null()) {
        fail("Config key '" + path + "' does not accept null");
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    if (value.is_number()) {
        return py::as_double(value);
    }
    if (value.is_string()) {
        if (auto parsed = py::parse_float(value.get<std::string>())) {
            return *parsed;
        }
    }
    fail("Config key '" + path + "' expects a float, got " + py::repr(value));
}

std::optional<std::int64_t> coerce_opt_int(const Json& value, const std::string& path) {
    if (value.is_null()) {
        return std::nullopt;
    }
    return coerce_int(value, path);
}

std::optional<double> coerce_opt_float(const Json& value, const std::string& path) {
    if (value.is_null()) {
        return std::nullopt;
    }
    return coerce_float(value, path);
}

void require_list(const Json& value, const std::string& path) {
    if (value.is_null()) {
        fail("Config key '" + path + "' does not accept null");
    }
    if (!value.is_array()) {
        fail("Config key '" + path + "' expects a list, got " + py::type_name(value));
    }
}

std::vector<std::int64_t> coerce_int_list(const Json& value, const std::string& path) {
    require_list(value, path);
    std::vector<std::int64_t> out;
    for (const auto& item : value) {
        out.push_back(coerce_int(item, path));
    }
    return out;
}

std::vector<double> coerce_float_list(const Json& value, const std::string& path) {
    require_list(value, path);
    std::vector<double> out;
    for (const auto& item : value) {
        out.push_back(coerce_float(item, path));
    }
    return out;
}

// --------------------------------------------------------------------------- //
// Plain-value serialisation
// --------------------------------------------------------------------------- //

Json to_json(const std::string& v) { return Json(v); }
Json to_json(std::int64_t v) { return Json(v); }
Json to_json(bool v) { return Json(v); }
Json to_json(double v) { return py::make_float(v); }
Json to_json(const std::optional<std::int64_t>& v) { return v ? Json(*v) : Json(nullptr); }
Json to_json(const std::optional<double>& v) { return v ? py::make_float(*v) : Json(nullptr); }
Json to_json(const std::vector<std::int64_t>& v) {
    Json out = Json::array();
    for (auto x : v) {
        out.push_back(x);
    }
    return out;
}
Json to_json(const std::vector<double>& v) {
    Json out = Json::array();
    for (auto x : v) {
        out.push_back(py::make_float(x));
    }
    return out;
}

void assign(std::string& field, const Json& v, const std::string& path) { field = coerce_str(v, path); }
void assign(std::int64_t& field, const Json& v, const std::string& path) { field = coerce_int(v, path); }
void assign(bool& field, const Json& v, const std::string& path) { field = coerce_bool(v, path); }
void assign(double& field, const Json& v, const std::string& path) { field = coerce_float(v, path); }
void assign(std::optional<std::int64_t>& field, const Json& v, const std::string& path) {
    field = coerce_opt_int(v, path);
}
void assign(std::optional<double>& field, const Json& v, const std::string& path) {
    field = coerce_opt_float(v, path);
}
void assign(std::vector<std::int64_t>& field, const Json& v, const std::string& path) {
    field = coerce_int_list(v, path);
}
void assign(std::vector<double>& field, const Json& v, const std::string& path) {
    field = coerce_float_list(v, path);
}

// --------------------------------------------------------------------------- //
// Schema: sections and fields in dataclass declaration order
// --------------------------------------------------------------------------- //

struct Field {
    std::string name;
    std::function<Json(const Config&)> get;
    std::function<void(Config&, const Json&, const std::string&)> set;
};

struct Section {
    std::string name;
    std::vector<Field> fields;
};

#define MVSLAM_FIELD(section, member)                                                       \
    Field {                                                                                 \
        #member, [](const Config& c) { return to_json(c.section.member); },                 \
            [](Config& c, const Json& v, const std::string& p) { assign(c.section.member, v, p); } \
    }

const std::vector<Section>& schema() {
    static const std::vector<Section> sections = {
        {"dataset",
         {MVSLAM_FIELD(dataset, path), MVSLAM_FIELD(dataset, sequence),
          MVSLAM_FIELD(dataset, camera), MVSLAM_FIELD(dataset, max_frames),
          MVSLAM_FIELD(dataset, frame_step), MVSLAM_FIELD(dataset, start_frame),
          MVSLAM_FIELD(dataset, grayscale)}},
        {"features",
         {MVSLAM_FIELD(features, detector), MVSLAM_FIELD(features, max_features),
          MVSLAM_FIELD(features, fast_threshold), MVSLAM_FIELD(features, scale_factor),
          MVSLAM_FIELD(features, n_levels), MVSLAM_FIELD(features, edge_threshold),
          MVSLAM_FIELD(features, score_type), MVSLAM_FIELD(features, clahe)}},
        {"matcher",
         {MVSLAM_FIELD(matcher, matcher), MVSLAM_FIELD(matcher, use_ratio_test),
          MVSLAM_FIELD(matcher, ratio), MVSLAM_FIELD(matcher, cross_check),
          MVSLAM_FIELD(matcher, max_distance), MVSLAM_FIELD(matcher, min_matches)}},
        {"odometry",
         {MVSLAM_FIELD(odometry, ransac_method), MVSLAM_FIELD(odometry, ransac_threshold_px),
          MVSLAM_FIELD(odometry, ransac_confidence), MVSLAM_FIELD(odometry, ransac_max_iters),
          MVSLAM_FIELD(odometry, cheirality_distance), MVSLAM_FIELD(odometry, min_inliers),
          MVSLAM_FIELD(odometry, min_inlier_ratio), MVSLAM_FIELD(odometry, max_rotation_deg),
          MVSLAM_FIELD(odometry, min_parallax_deg), MVSLAM_FIELD(odometry, min_flow_px),
          MVSLAM_FIELD(odometry, homography_ratio_threshold),
          MVSLAM_FIELD(odometry, reject_degenerate), MVSLAM_FIELD(odometry, scale_source),
          MVSLAM_FIELD(odometry, constant_scale), MVSLAM_FIELD(odometry, min_scale),
          MVSLAM_FIELD(odometry, max_scale)}},
        {"keyframes",
         {MVSLAM_FIELD(keyframes, every_n_frames), MVSLAM_FIELD(keyframes, min_translation_m),
          MVSLAM_FIELD(keyframes, min_rotation_deg)}},
        {"loop_closure",
         {MVSLAM_FIELD(loop_closure, enabled), MVSLAM_FIELD(loop_closure, vocabulary_size),
          MVSLAM_FIELD(loop_closure, vocabulary_train_descriptors),
          MVSLAM_FIELD(loop_closure, min_keyframe_separation),
          MVSLAM_FIELD(loop_closure, min_path_separation_m), MVSLAM_FIELD(loop_closure, top_k),
          MVSLAM_FIELD(loop_closure, min_similarity), MVSLAM_FIELD(loop_closure, similarity_ratio),
          MVSLAM_FIELD(loop_closure, min_matches), MVSLAM_FIELD(loop_closure, min_inliers),
          MVSLAM_FIELD(loop_closure, min_inlier_ratio),
          MVSLAM_FIELD(loop_closure, cooldown_keyframes), MVSLAM_FIELD(loop_closure, max_loops),
          MVSLAM_FIELD(loop_closure, min_scale_points),
          MVSLAM_FIELD(loop_closure, min_scale_baseline_m),
          MVSLAM_FIELD(loop_closure, scale_neighbour_search),
          MVSLAM_FIELD(loop_closure, min_loop_scale_m),
          MVSLAM_FIELD(loop_closure, max_loop_scale_m),
          MVSLAM_FIELD(loop_closure, require_measured_scale)}},
        {"pose_graph",
         {MVSLAM_FIELD(pose_graph, prior_sigma_rot), MVSLAM_FIELD(pose_graph, prior_sigma_trans),
          MVSLAM_FIELD(pose_graph, odom_sigma_rot), MVSLAM_FIELD(pose_graph, odom_sigma_trans),
          MVSLAM_FIELD(pose_graph, loop_sigma_rot), MVSLAM_FIELD(pose_graph, loop_sigma_trans),
          MVSLAM_FIELD(pose_graph, robust_loop_kernel), MVSLAM_FIELD(pose_graph, huber_k),
          MVSLAM_FIELD(pose_graph, optimizer), MVSLAM_FIELD(pose_graph, max_iterations),
          MVSLAM_FIELD(pose_graph, relative_error_tol),
          MVSLAM_FIELD(pose_graph, absolute_error_tol), MVSLAM_FIELD(pose_graph, verbose)}},
        {"evaluation",
         {MVSLAM_FIELD(evaluation, alignment), MVSLAM_FIELD(evaluation, rpe_deltas),
          MVSLAM_FIELD(evaluation, drift_segment_lengths_m)}},
        {"output",
         {MVSLAM_FIELD(output, root), MVSLAM_FIELD(output, run_name),
          MVSLAM_FIELD(output, save_plots), MVSLAM_FIELD(output, save_diagnostics),
          MVSLAM_FIELD(output, dpi), MVSLAM_FIELD(output, plot_format),
          MVSLAM_FIELD(output, plot_3d)}},
        {"runtime",
         {MVSLAM_FIELD(runtime, seed), MVSLAM_FIELD(runtime, log_level),
          MVSLAM_FIELD(runtime, progress_interval)}},
    };
    return sections;
}

#undef MVSLAM_FIELD

std::string sorted_repr(std::vector<std::string> names) {
    std::sort(names.begin(), names.end());
    Json list = Json::array();
    for (const auto& n : names) {
        list.push_back(n);
    }
    return py::repr(list);
}

void check_unknown(const Json& data, const std::vector<std::string>& valid,
                   const std::string& path) {
    std::vector<std::string> unknown;
    for (const auto& [key, _] : data.items()) {
        if (std::find(valid.begin(), valid.end(), key) == valid.end()) {
            unknown.push_back(key);
        }
    }
    if (!unknown.empty()) {
        const std::string location = path.empty() ? "root" : path;
        fail("Unknown config key(s) at '" + location + "': " + sorted_repr(unknown) +
             ". Valid keys: " + sorted_repr(valid));
    }
}

void require_mapping(const Json& data, const std::string& path) {
    if (!data.is_object()) {
        fail("Expected a mapping at '" + (path.empty() ? std::string("root") : path) + "', got " +
             py::type_name(data));
    }
}

// Python truthiness of a value standing in for ``data or {}``.
bool truthy(const Json& v) {
    switch (v.type()) {
        case Json::value_t::null:
            return false;
        case Json::value_t::boolean:
            return v.get<bool>();
        case Json::value_t::number_integer:
        case Json::value_t::number_unsigned:
            return v.get<std::int64_t>() != 0;
        case Json::value_t::number_float:
            return v.get<double>() != 0.0;
        case Json::value_t::string:
        case Json::value_t::array:
        case Json::value_t::object:
            return !v.empty();
        default:
            return true;
    }
}

}  // namespace

Config Config::from_dict(const Json& data_in) {
    const Json data = truthy(data_in) ? data_in : Json::object();
    require_mapping(data, "");
    std::vector<std::string> section_names;
    for (const auto& section : schema()) {
        section_names.push_back(section.name);
    }
    check_unknown(data, section_names, "");

    Config config;
    for (const auto& section : schema()) {
        if (!data.contains(section.name)) {
            continue;
        }
        const Json& value = data.at(section.name);
        require_mapping(value, section.name);
        std::vector<std::string> field_names;
        for (const auto& field : section.fields) {
            field_names.push_back(field.name);
        }
        check_unknown(value, field_names, section.name);
        // Fields are coerced in declaration order, so the first invalid field
        // in that order is the one reported, as in the dataclass loader.
        for (const auto& field : section.fields) {
            if (value.contains(field.name)) {
                field.set(config, value.at(field.name), section.name + "." + field.name);
            }
        }
    }
    return config;
}

Config Config::from_yaml(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        fail("Config file not found: " + path.string());
    }
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const Json data = py::yaml_safe_load(buffer.str());
    if (!data.is_null() && !data.is_object()) {
        fail("Config root must be a mapping, got " + py::type_name(data));
    }
    return from_dict(data);
}

Config Config::load(const std::optional<std::filesystem::path>& path, const Json& overrides) {
    Config config = path ? from_yaml(*path) : Config();
    if (truthy(overrides)) {
        config = config.with_overrides(overrides);
    }
    return config;
}

Config Config::with_overrides(const Json& overrides) const {
    Json data = to_dict();
    for (const auto& [dotted_key, value] : overrides.items()) {
        if (value.is_null()) {
            continue;
        }
        std::vector<std::string> parts;
        std::size_t start = 0;
        while (true) {
            const auto dot = dotted_key.find('.', start);
            parts.push_back(dotted_key.substr(start, dot - start));
            if (dot == std::string::npos) {
                break;
            }
            start = dot + 1;
        }
        Json* node = &data;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            if (!node->contains(parts[i]) || !(*node)[parts[i]].is_object()) {
                fail("Unknown config section in override: '" + dotted_key + "'");
            }
            node = &(*node)[parts[i]];
        }
        if (!node->contains(parts.back())) {
            fail("Unknown config key in override: '" + dotted_key + "'");
        }
        (*node)[parts.back()] = value;
    }
    return from_dict(data);
}

Config Config::with_override(const std::string& dotted_key, const Json& value) const {
    Json overrides = Json::object();
    overrides[dotted_key] = value;
    return with_overrides(overrides);
}

Json Config::to_dict() const {
    Json out = Json::object();
    for (const auto& section : schema()) {
        Json values = Json::object();
        for (const auto& field : section.fields) {
            values[field.name] = field.get(*this);
        }
        out[section.name] = values;
    }
    return out;
}

std::filesystem::path Config::to_yaml(const std::filesystem::path& path) const {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << py::yaml_safe_dump(to_dict());
    return path;
}

std::filesystem::path Config::run_dir() const {
    return std::filesystem::path(output.root) / format_run_name(output.run_name, dataset.sequence);
}

std::string format_run_name(const std::string& pattern, const std::string& sequence) {
    std::string out;
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const char c = pattern[i];
        if (c == '{') {
            if (i + 1 < pattern.size() && pattern[i + 1] == '{') {
                out.push_back('{');
                ++i;
                continue;
            }
            const auto close = pattern.find('}', i);
            if (close == std::string::npos) {
                throw ValueError("Single '{' encountered in format string");
            }
            const std::string field = pattern.substr(i + 1, close - i - 1);
            if (field == "sequence") {
                out += sequence;
            } else if (field.empty()) {
                throw IndexError("Replacement index 0 out of range for positional args tuple");
            } else {
                throw KeyError(py::repr_str(field));
            }
            i = close;
        } else if (c == '}') {
            if (i + 1 < pattern.size() && pattern[i + 1] == '}') {
                out.push_back('}');
                ++i;
                continue;
            }
            throw ValueError("Single '}' encountered in format string");
        } else {
            out.push_back(c);
        }
    }
    return out;
}

}  // namespace mvslam
