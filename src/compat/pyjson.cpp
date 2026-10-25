#include "mvslam/compat/pyjson.hpp"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"

namespace mvslam::py {

namespace {

// ---------------------------------------------------------------------- //
// Encoding
// ---------------------------------------------------------------------- //

// Decode one UTF-8 code point starting at s[i]; invalid bytes map to U+FFFD.
std::uint32_t next_code_point(const std::string& s, std::size_t& i) {
    const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = byte(i);
    if (c < 0x80) {
        ++i;
        return c;
    }
    int extra = 0;
    std::uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) {
        extra = 1;
        cp = c & 0x1F;
    } else if ((c & 0xF0) == 0xE0) {
        extra = 2;
        cp = c & 0x0F;
    } else if ((c & 0xF8) == 0xF0) {
        extra = 3;
        cp = c & 0x07;
    } else {
        ++i;
        return 0xFFFD;
    }
    if (i + static_cast<std::size_t>(extra) >= s.size()) {
        ++i;  // truncated sequence
        return 0xFFFD;
    }
    for (int k = 1; k <= extra; ++k) {
        const unsigned char cc = byte(i + static_cast<std::size_t>(k));
        if ((cc & 0xC0) != 0x80) {
            ++i;
            return 0xFFFD;
        }
        cp = (cp << 6) | (cc & 0x3F);
    }
    i += static_cast<std::size_t>(extra) + 1;
    return cp;
}

void append_u_escape(std::string& out, std::uint32_t unit) {
    char buffer[8];
    std::snprintf(buffer, sizeof(buffer), "\\u%04x", unit);
    out += buffer;
}

std::string encode_string(const std::string& s) {
    std::string out = "\"";
    std::size_t i = 0;
    while (i < s.size()) {
        const std::uint32_t cp = next_code_point(s, i);
        switch (cp) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default:
                if (cp < 0x20 || cp > 0x7E) {
                    // ensure_ascii=True: everything outside printable ASCII is
                    // \uXXXX, astral code points as a UTF-16 surrogate pair.
                    if (cp >= 0x10000) {
                        const std::uint32_t v = cp - 0x10000;
                        append_u_escape(out, 0xD800 | (v >> 10));
                        append_u_escape(out, 0xDC00 | (v & 0x3FF));
                    } else {
                        append_u_escape(out, cp);
                    }
                } else {
                    out.push_back(static_cast<char>(cp));
                }
        }
    }
    out.push_back('"');
    return out;
}

std::string float_text(double v) {
    if (std::isnan(v)) {
        return "NaN";
    }
    if (std::isinf(v)) {
        return v > 0 ? "Infinity" : "-Infinity";
    }
    return repr(v);
}

void dump(const Json& value, std::optional<int> indent, int level, std::string& out) {
    const std::string item_separator = indent ? "," : ", ";
    auto newline = [&](int depth) {
        if (indent) {
            out.push_back('\n');
            out.append(static_cast<std::size_t>(*indent * depth), ' ');
        }
    };
    switch (value.type()) {
        case Json::value_t::null:
            out += "null";
            break;
        case Json::value_t::boolean:
            out += value.get<bool>() ? "true" : "false";
            break;
        case Json::value_t::number_integer:
            out += std::to_string(value.get<std::int64_t>());
            break;
        case Json::value_t::number_unsigned:
            out += std::to_string(value.get<std::uint64_t>());
            break;
        case Json::value_t::number_float:
            out += float_text(value.get<double>());
            break;
        case Json::value_t::string:
            out += encode_string(value.get_ref<const std::string&>());
            break;
        case Json::value_t::array: {
            if (value.empty()) {
                out += "[]";
                break;
            }
            out.push_back('[');
            bool first = true;
            for (const auto& item : value) {
                if (!first) {
                    out += item_separator;
                }
                first = false;
                newline(level + 1);
                dump(item, indent, level + 1, out);
            }
            newline(level);
            out.push_back(']');
            break;
        }
        case Json::value_t::object: {
            if (value.empty()) {
                out += "{}";
                break;
            }
            out.push_back('{');
            bool first = true;
            for (const auto& [key, item] : value.items()) {
                if (!first) {
                    out += item_separator;
                }
                first = false;
                newline(level + 1);
                out += encode_string(key);
                out += ": ";
                dump(item, indent, level + 1, out);
            }
            newline(level);
            out.push_back('}');
            break;
        }
        default:
            throw ValueError("cannot serialise this JSON value");
    }
}

// ---------------------------------------------------------------------- //
// Decoding
// ---------------------------------------------------------------------- //

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Json parse_document() {
        skip_ws();
        Json value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            fail("Extra data");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        // Same shape as json.JSONDecodeError's message.
        std::size_t line = 1;
        std::size_t column = 1;
        for (std::size_t k = 0; k < pos_ && k < text_.size(); ++k) {
            if (text_[k] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw JsonDecodeError(what + ": line " + std::to_string(line) + " column " +
                              std::to_string(column) + " (char " + std::to_string(pos_) + ")");
    }

    void skip_ws() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
                                       text_[pos_] == '\n' || text_[pos_] == '\r')) {
            ++pos_;
        }
    }

    bool consume(const char* literal) {
        const std::size_t n = std::char_traits<char>::length(literal);
        if (text_.compare(pos_, n, literal) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    Json parse_value() {
        if (pos_ >= text_.size()) {
            fail("Expecting value");
        }
        const char c = text_[pos_];
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '"') {
            return Json(parse_string());
        }
        if (consume("null")) {
            return Json(nullptr);
        }
        if (consume("true")) {
            return Json(true);
        }
        if (consume("false")) {
            return Json(false);
        }
        if (consume("NaN")) {
            return make_float(std::numeric_limits<double>::quiet_NaN());
        }
        if (consume("Infinity")) {
            return make_float(std::numeric_limits<double>::infinity());
        }
        if (consume("-Infinity")) {
            return make_float(-std::numeric_limits<double>::infinity());
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            return parse_number();
        }
        fail("Expecting value");
    }

    Json parse_number() {
        const std::size_t start = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            pos_ = start;
            fail("Expecting value");
        }
        if (text_[pos_] == '0') {
            ++pos_;
        } else {
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        bool is_float = false;
        if (pos_ < text_.size() && text_[pos_] == '.' && pos_ + 1 < text_.size() &&
            std::isdigit(static_cast<unsigned char>(text_[pos_ + 1]))) {
            is_float = true;
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            std::size_t save = pos_;
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                is_float = true;
                while (pos_ < text_.size() &&
                       std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                    ++pos_;
                }
            } else {
                pos_ = save;
            }
        }
        const std::string token = text_.substr(start, pos_ - start);
        if (is_float) {
            return make_float(std::strtod(token.c_str(), nullptr));
        }
        errno = 0;
        const long long v = std::strtoll(token.c_str(), nullptr, 10);
        if (errno == ERANGE) {
            // Python ints are unbounded; beyond int64 keep the magnitude.
            return make_float(std::strtod(token.c_str(), nullptr));
        }
        return Json(static_cast<std::int64_t>(v));
    }

    static void append_utf8(std::string& out, std::uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    std::uint32_t parse_hex4() {
        if (pos_ + 4 > text_.size()) {
            fail("Invalid \\uXXXX escape");
        }
        std::uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = text_[pos_++];
            v <<= 4;
            if (h >= '0' && h <= '9') {
                v |= static_cast<std::uint32_t>(h - '0');
            } else if (h >= 'a' && h <= 'f') {
                v |= static_cast<std::uint32_t>(h - 'a' + 10);
            } else if (h >= 'A' && h <= 'F') {
                v |= static_cast<std::uint32_t>(h - 'A' + 10);
            } else {
                fail("Invalid \\uXXXX escape");
            }
        }
        return v;
    }

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) {
                fail("Unterminated string starting at");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c == '\\') {
                if (pos_ >= text_.size()) {
                    fail("Unterminated string starting at");
                }
                const char e = text_[pos_++];
                switch (e) {
                    case '"':
                        out.push_back('"');
                        break;
                    case '\\':
                        out.push_back('\\');
                        break;
                    case '/':
                        out.push_back('/');
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u': {
                        std::uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF && text_.compare(pos_, 2, "\\u") == 0) {
                            const std::size_t save = pos_;
                            pos_ += 2;
                            const std::uint32_t low = parse_hex4();
                            if (low >= 0xDC00 && low <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            } else {
                                pos_ = save;
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        fail("Invalid \\escape");
                }
            } else if (static_cast<unsigned char>(c) < 0x20) {
                fail("Invalid control character at");
            } else {
                out.push_back(c);
            }
        }
    }

    Json parse_array() {
        ++pos_;
        Json out = Json::array();
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            return out;
        }
        while (true) {
            skip_ws();
            out.push_back(parse_value());
            skip_ws();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < text_.size() && text_[pos_] == ']') {
                ++pos_;
                return out;
            }
            fail("Expecting ',' delimiter");
        }
    }

    Json parse_object() {
        ++pos_;
        Json out = Json::object();
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            return out;
        }
        while (true) {
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != '"') {
                fail("Expecting property name enclosed in double quotes");
            }
            std::string key = parse_string();
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != ':') {
                fail("Expecting ':' delimiter");
            }
            ++pos_;
            skip_ws();
            // A repeated key keeps its first position but takes the last
            // value, as a Python dict does.
            out[key] = parse_value();
            skip_ws();
            if (pos_ < text_.size() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < text_.size() && text_[pos_] == '}') {
                ++pos_;
                return out;
            }
            fail("Expecting ',' delimiter");
        }
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

}  // namespace

std::string dumps(const Json& value, std::optional<int> indent) {
    std::string out;
    dump(value, indent, 0, out);
    return out;
}

Json loads(const std::string& text) { return Parser(text).parse_document(); }

std::string repr(const Json& value) {
    switch (value.type()) {
        case Json::value_t::null:
            return "None";
        case Json::value_t::boolean:
            return value.get<bool>() ? "True" : "False";
        case Json::value_t::number_integer:
            return std::to_string(value.get<std::int64_t>());
        case Json::value_t::number_unsigned:
            return std::to_string(value.get<std::uint64_t>());
        case Json::value_t::number_float:
            return repr(value.get<double>());
        case Json::value_t::string:
            return repr_str(value.get_ref<const std::string&>());
        case Json::value_t::array: {
            std::string out = "[";
            bool first = true;
            for (const auto& item : value) {
                if (!first) {
                    out += ", ";
                }
                first = false;
                out += repr(item);
            }
            return out + "]";
        }
        case Json::value_t::object: {
            std::string out = "{";
            bool first = true;
            for (const auto& [key, item] : value.items()) {
                if (!first) {
                    out += ", ";
                }
                first = false;
                out += repr_str(key) + ": " + repr(item);
            }
            return out + "}";
        }
        default:
            return "<unknown>";
    }
}

std::string str(const Json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return repr(value);
}

std::string type_name(const Json& value) {
    switch (value.type()) {
        case Json::value_t::null:
            return "NoneType";
        case Json::value_t::boolean:
            return "bool";
        case Json::value_t::number_integer:
        case Json::value_t::number_unsigned:
            return "int";
        case Json::value_t::number_float:
            return "float";
        case Json::value_t::string:
            return "str";
        case Json::value_t::array:
            return "list";
        case Json::value_t::object:
            return "dict";
        default:
            return "object";
    }
}

double as_double(const Json& value) {
    if (value.is_null()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (value.is_number_float()) {
        return value.get<double>();
    }
    if (value.is_number_unsigned()) {
        return static_cast<double>(value.get<std::uint64_t>());
    }
    if (value.is_number_integer()) {
        return static_cast<double>(value.get<std::int64_t>());
    }
    if (value.is_boolean()) {
        return value.get<bool>() ? 1.0 : 0.0;
    }
    throw ValueError("expected a number, got " + type_name(value));
}

Json make_float(double value) {
    Json out;
    out = Json::number_float_t(value);
    return out;
}

}  // namespace mvslam::py
