#include "mvslam/compat/pyyaml.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <regex>

#include "mvslam/compat/pyformat.hpp"

namespace mvslam::py {

namespace {

// PyYAML resolver.py (YAML 1.1 implicit resolvers), in registration order.
const std::regex& bool_re() {
    static const std::regex re(
        "^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$");
    return re;
}

const std::regex& float_re() {
    static const std::regex re(
        "^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+][0-9]+)?"
        "|\\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?"
        "|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*"
        "|[-+]?\\.(?:inf|Inf|INF)"
        "|\\.(?:nan|NaN|NAN))$");
    return re;
}

const std::regex& int_re() {
    static const std::regex re(
        "^(?:[-+]?0b[0-1_]+"
        "|[-+]?0[0-7_]+"
        "|[-+]?(?:0|[1-9][0-9_]*)"
        "|[-+]?0x[0-9a-fA-F_]+"
        "|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$");
    return re;
}

const std::regex& null_re() {
    static const std::regex re("^(?:~|null|Null|NULL|)$");
    return re;
}

const std::regex& timestamp_re() {
    static const std::regex re(
        "^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]"
        "|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?"
        "(?:[Tt]|[ \\t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\\.[0-9]*)?"
        "(?:[ \\t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$");
    return re;
}

std::string remove_underscores(std::string s) {
    s.erase(std::remove(s.begin(), s.end(), '_'), s.end());
    return s;
}

// constructor.py: SafeConstructor.construct_yaml_int
Json construct_int(const std::string& scalar) {
    std::string value = remove_underscores(scalar);
    std::int64_t sign = 1;
    if (!value.empty() && value[0] == '-') {
        sign = -1;
    }
    if (!value.empty() && (value[0] == '+' || value[0] == '-')) {
        value = value.substr(1);
    }
    if (value == "0") {
        return Json(static_cast<std::int64_t>(0));
    }
    if (value.rfind("0b", 0) == 0) {
        return Json(sign * std::strtoll(value.c_str() + 2, nullptr, 2));
    }
    if (value.rfind("0x", 0) == 0) {
        return Json(sign * std::strtoll(value.c_str() + 2, nullptr, 16));
    }
    if (value[0] == '0') {
        return Json(sign * std::strtoll(value.c_str(), nullptr, 8));
    }
    if (value.find(':') != std::string::npos) {
        std::int64_t total = 0;
        std::int64_t base = 1;
        std::vector<std::string> parts;
        std::size_t start = 0;
        while (true) {
            const auto colon = value.find(':', start);
            parts.push_back(value.substr(start, colon - start));
            if (colon == std::string::npos) {
                break;
            }
            start = colon + 1;
        }
        for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
            total += std::strtoll(it->c_str(), nullptr, 10) * base;
            base *= 60;
        }
        return Json(sign * total);
    }
    return Json(sign * std::strtoll(value.c_str(), nullptr, 10));
}

// constructor.py: SafeConstructor.construct_yaml_float
Json construct_float(const std::string& scalar) {
    std::string value = remove_underscores(scalar);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    double sign = 1.0;
    if (!value.empty() && value[0] == '-') {
        sign = -1.0;
    }
    if (!value.empty() && (value[0] == '+' || value[0] == '-')) {
        value = value.substr(1);
    }
    if (value == ".inf") {
        return make_float(sign * std::numeric_limits<double>::infinity());
    }
    if (value == ".nan") {
        return make_float(std::numeric_limits<double>::quiet_NaN());
    }
    if (value.find(':') != std::string::npos) {
        std::vector<std::string> parts;
        std::size_t start = 0;
        while (true) {
            const auto colon = value.find(':', start);
            parts.push_back(value.substr(start, colon - start));
            if (colon == std::string::npos) {
                break;
            }
            start = colon + 1;
        }
        double total = 0.0;
        double base = 1.0;
        for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
            total += std::strtod(it->c_str(), nullptr) * base;
            base *= 60.0;
        }
        return make_float(sign * total);
    }
    return make_float(sign * std::strtod(value.c_str(), nullptr));
}

Json convert(const YAML::Node& node);

Json convert_scalar(const YAML::Node& node) {
    const std::string& text = node.Scalar();
    const std::string& tag = node.Tag();
    if (tag == "!") {
        return Json(text);  // quoted: always a string
    }
    if (tag == "tag:yaml.org,2002:str") {
        return Json(text);
    }
    if (tag == "tag:yaml.org,2002:int") {
        return construct_int(text);
    }
    if (tag == "tag:yaml.org,2002:float") {
        return construct_float(text);
    }
    if (tag == "tag:yaml.org,2002:bool") {
        std::string lowered = text;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return Json(lowered == "yes" || lowered == "true" || lowered == "on");
    }
    if (tag == "tag:yaml.org,2002:null") {
        return Json(nullptr);
    }
    return yaml_resolve_plain(text);
}

Json convert(const YAML::Node& node) {
    switch (node.Type()) {
        case YAML::NodeType::Null:
            if (node.Tag() == "!") {
                return Json("");
            }
            return Json(nullptr);
        case YAML::NodeType::Scalar:
            return convert_scalar(node);
        case YAML::NodeType::Sequence: {
            Json out = Json::array();
            for (const auto& item : node) {
                out.push_back(convert(item));
            }
            return out;
        }
        case YAML::NodeType::Map: {
            Json out = Json::object();
            for (const auto& item : node) {
                // Keys are typed like any scalar; the configuration only has
                // string keys, so non-string keys are kept by their str().
                const Json key = convert(item.first);
                out[str(key)] = convert(item.second);
            }
            return out;
        }
        default:
            return Json(nullptr);
    }
}

// ---------------------------------------------------------------------- //
// Emitter (emitter.py: analyze_scalar / choose_scalar_style)
// ---------------------------------------------------------------------- //

struct ScalarAnalysis {
    bool empty = false;
    bool multiline = false;
    bool allow_block_plain = true;
    bool allow_single_quoted = true;
};

bool is_space_or_break(char c) {
    return c == '\0' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

ScalarAnalysis analyze_scalar(const std::string& scalar) {
    ScalarAnalysis analysis;
    if (scalar.empty()) {
        analysis.empty = true;
        analysis.allow_block_plain = true;
        analysis.allow_single_quoted = true;
        return analysis;
    }
    bool block_indicators = false;
    bool line_breaks = false;
    bool special_characters = false;
    bool leading_space = false;
    bool leading_break = false;
    bool trailing_space = false;
    bool trailing_break = false;
    bool break_space = false;
    bool space_break = false;

    if (scalar.rfind("---", 0) == 0 || scalar.rfind("...", 0) == 0) {
        block_indicators = true;
    }
    bool preceded_by_whitespace = true;
    bool followed_by_whitespace = scalar.size() == 1 || is_space_or_break(scalar[1]);
    bool previous_space = false;
    bool previous_break = false;

    for (std::size_t index = 0; index < scalar.size(); ++index) {
        const char ch = scalar[index];
        if (index == 0) {
            if (std::string("#,[]{}&*!|>'\"%@`").find(ch) != std::string::npos) {
                block_indicators = true;
            }
            if ((ch == '?' || ch == ':') && followed_by_whitespace) {
                block_indicators = true;
            }
            if (ch == '-' && followed_by_whitespace) {
                block_indicators = true;
            }
        } else {
            if (ch == ':' && followed_by_whitespace) {
                block_indicators = true;
            }
            if (ch == '#' && preceded_by_whitespace) {
                block_indicators = true;
            }
        }
        if (ch == '\n') {
            line_breaks = true;
        }
        const auto uch = static_cast<unsigned char>(ch);
        if (!(ch == '\n' || (uch >= 0x20 && uch <= 0x7E))) {
            // allow_unicode is False for safe_dump, so anything outside
            // printable ASCII forces double quotes.
            special_characters = true;
        }
        if (ch == ' ') {
            if (index == 0) {
                leading_space = true;
            }
            if (index == scalar.size() - 1) {
                trailing_space = true;
            }
            if (previous_break) {
                break_space = true;
            }
            previous_space = true;
            previous_break = false;
        } else if (ch == '\n') {
            if (index == 0) {
                leading_break = true;
            }
            if (index == scalar.size() - 1) {
                trailing_break = true;
            }
            if (previous_space) {
                space_break = true;
            }
            previous_space = false;
            previous_break = true;
        } else {
            previous_space = false;
            previous_break = false;
        }
        preceded_by_whitespace = is_space_or_break(ch);
        followed_by_whitespace =
            index + 2 >= scalar.size() || is_space_or_break(scalar[index + 2]);
    }

    if (leading_space || leading_break || trailing_space || trailing_break) {
        analysis.allow_block_plain = false;
    }
    if (break_space) {
        analysis.allow_block_plain = false;
        analysis.allow_single_quoted = false;
    }
    if (space_break || special_characters) {
        analysis.allow_block_plain = false;
        analysis.allow_single_quoted = false;
    }
    if (line_breaks) {
        analysis.allow_block_plain = false;
    }
    if (block_indicators) {
        analysis.allow_block_plain = false;
    }
    analysis.multiline = line_breaks;
    return analysis;
}

std::string double_quoted(const std::string& text) {
    // emitter.py ESCAPE_REPLACEMENTS plus \x / \u escapes.
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
            case '\0':
                out += "\\0";
                break;
            case 0x07:
                out += "\\a";
                break;
            case 0x08:
                out += "\\b";
                break;
            case 0x09:
                out += "\\t";
                break;
            case 0x0A:
                out += "\\n";
                break;
            case 0x0B:
                out += "\\v";
                break;
            case 0x0C:
                out += "\\f";
                break;
            case 0x0D:
                out += "\\r";
                break;
            case 0x1B:
                out += "\\e";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            default:
                if (c < 0x20 || c > 0x7E) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\x%02X", c);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out + "\"";
}

std::string single_quoted(const std::string& text) {
    std::string out = "'";
    for (char c : text) {
        if (c == '\'') {
            out += "''";
        } else {
            out.push_back(c);
        }
    }
    return out + "'";
}

/// representer.py: SafeRepresenter.represent_str + emitter style choice, for
/// a scalar in block context. ``simple_key`` is true for mapping keys.
std::string emit_string(const std::string& text, bool simple_key) {
    const ScalarAnalysis analysis = analyze_scalar(text);
    // implicit[0]: the plain text would resolve back to a str.
    const bool implicit_plain = yaml_resolve_plain(text).is_string();
    if (implicit_plain && !(simple_key && (analysis.empty || analysis.multiline)) &&
        analysis.allow_block_plain) {
        return text;
    }
    if (analysis.allow_single_quoted && !(simple_key && analysis.multiline)) {
        return single_quoted(text);
    }
    return double_quoted(text);
}

std::string emit_float(double value) {
    // representer.py: SafeRepresenter.represent_float
    if (std::isnan(value)) {
        return ".nan";
    }
    if (std::isinf(value)) {
        return value > 0 ? ".inf" : "-.inf";
    }
    std::string text = repr(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (text.find('.') == std::string::npos && text.find('e') != std::string::npos) {
        text.replace(text.find('e'), 1, ".0e");
    }
    return text;
}

std::string emit_scalar(const Json& value) {
    switch (value.type()) {
        case Json::value_t::null:
            return "null";
        case Json::value_t::boolean:
            return value.get<bool>() ? "true" : "false";
        case Json::value_t::number_integer:
            return std::to_string(value.get<std::int64_t>());
        case Json::value_t::number_unsigned:
            return std::to_string(value.get<std::uint64_t>());
        case Json::value_t::number_float:
            return emit_float(value.get<double>());
        case Json::value_t::string:
            return emit_string(value.get<std::string>(), false);
        default:
            throw ValueError("not a scalar");
    }
}

void emit_node(const Json& value, int indent, std::string& out);

void emit_mapping(const Json& value, int indent, std::string& out) {
    for (const auto& [key, item] : value.items()) {
        out.append(static_cast<std::size_t>(indent), ' ');
        out += emit_string(key, true);
        out += ":";
        if (item.is_object() && !item.empty()) {
            out += "\n";
            emit_mapping(item, indent + 2, out);
        } else if (item.is_array() && !item.empty()) {
            // PyYAML writes block sequences inside a mapping without extra
            // indentation ("indentless" sequences).
            out += "\n";
            for (const auto& element : item) {
                out.append(static_cast<std::size_t>(indent), ' ');
                out += "-";
                emit_node(element, indent + 2, out);
            }
        } else {
            emit_node(item, indent + 2, out);
        }
    }
}

void emit_node(const Json& value, int indent, std::string& out) {
    if (value.is_object()) {
        if (value.empty()) {
            out += " {}\n";
            return;
        }
        out += "\n";
        emit_mapping(value, indent, out);
        return;
    }
    if (value.is_array()) {
        if (value.empty()) {
            out += " []\n";
            return;
        }
        out += "\n";
        for (const auto& element : value) {
            out.append(static_cast<std::size_t>(indent), ' ');
            out += "-";
            emit_node(element, indent + 2, out);
        }
        return;
    }
    out += " " + emit_scalar(value) + "\n";
}

}  // namespace

Json yaml_resolve_plain(const std::string& scalar) {
    if (std::regex_match(scalar, bool_re())) {
        std::string lowered = scalar;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return Json(lowered == "yes" || lowered == "true" || lowered == "on");
    }
    if (std::regex_match(scalar, float_re())) {
        return construct_float(scalar);
    }
    if (std::regex_match(scalar, int_re())) {
        return construct_int(scalar);
    }
    if (std::regex_match(scalar, null_re())) {
        return Json(nullptr);
    }
    // Timestamps would construct datetime objects; the only consumer is a
    // str-typed config field, whose str() of a plain date is the same text.
    (void)timestamp_re;
    return Json(scalar);
}

Json yaml_safe_load(const std::string& text) {
    try {
        const YAML::Node root = YAML::Load(text);
        return convert(root);
    } catch (const YAML::Exception& exc) {
        throw YamlError(exc.what());
    }
}

std::string yaml_safe_dump(const Json& value) {
    std::string out;
    if (value.is_object()) {
        if (value.empty()) {
            return "{}\n";
        }
        emit_mapping(value, 0, out);
        return out;
    }
    if (value.is_array()) {
        if (value.empty()) {
            return "[]\n";
        }
        for (const auto& element : value) {
            out += "-";
            emit_node(element, 2, out);
        }
        return out;
    }
    // A bare top-level scalar is followed by a document-end marker.
    return emit_scalar(value) + "\n...\n";
}

}  // namespace mvslam::py
