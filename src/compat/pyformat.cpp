#include "mvslam/compat/pyformat.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <regex>
#include <stdexcept>

#include "mvslam/errors.hpp"

namespace mvslam::py {

namespace {

// Shortest round-trip digits of |x| and the decimal exponent of the first
// digit, e.g. 0.00125 -> ("125", -3).
void shortest_digits(double x, std::string& digits, int& exponent) {
    char buffer[64];
    auto result = std::to_chars(buffer, buffer + sizeof(buffer), std::fabs(x),
                                std::chars_format::scientific);
    std::string text(buffer, result.ptr);
    const auto e_pos = text.find('e');
    std::string mantissa = text.substr(0, e_pos);
    exponent = std::stoi(text.substr(e_pos + 1));
    digits.clear();
    for (char c : mantissa) {
        if (c != '.') {
            digits.push_back(c);
        }
    }
    while (digits.size() > 1 && digits.back() == '0') {
        digits.pop_back();
    }
}

std::string vformat(const char* fmt, va_list args) {
    va_list copy;
    va_copy(copy, args);
    const int size = std::vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    if (size < 0) {
        throw ValueError("invalid format string");
    }
    std::string out(static_cast<std::size_t>(size), '\0');
    std::vsnprintf(out.data(), out.size() + 1, fmt, args);
    return out;
}

// glibc prints a negative NaN as "-nan"; Python never signs NaN.
std::string strip_nan_sign(std::string text) {
    std::size_t pos = 0;
    while ((pos = text.find("-nan", pos)) != std::string::npos) {
        text.replace(pos, 4, "nan");
        pos += 3;
    }
    pos = 0;
    while ((pos = text.find("-NAN", pos)) != std::string::npos) {
        text.replace(pos, 4, "NAN");
        pos += 3;
    }
    return text;
}

std::string group_thousands(const std::string& digits) {
    std::string out;
    const std::size_t n = digits.size();
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(digits[i]);
        const std::size_t remaining = n - i - 1;
        if (remaining > 0 && remaining % 3 == 0) {
            out.push_back(',');
        }
    }
    return out;
}

}  // namespace

std::string repr(double x) {
    if (std::isnan(x)) {
        return "nan";
    }
    if (std::isinf(x)) {
        return x > 0 ? "inf" : "-inf";
    }
    std::string sign = std::signbit(x) ? "-" : "";
    if (x == 0.0) {
        return sign + "0.0";
    }
    std::string digits;
    int exponent = 0;
    shortest_digits(x, digits, exponent);
    // Python (pystrtod.c, 'r' mode): decpt is the position of the decimal
    // point relative to the digit string; exponential when decpt <= -4 or > 16.
    const int decpt = exponent + 1;
    if (decpt <= -4 || decpt > 16) {
        std::string out = sign + digits.substr(0, 1);
        if (digits.size() > 1) {
            out += "." + digits.substr(1);
        }
        const int exp = decpt - 1;
        char exp_buffer[16];
        std::snprintf(exp_buffer, sizeof(exp_buffer), "e%c%02d", exp < 0 ? '-' : '+',
                      std::abs(exp));
        return out + exp_buffer;
    }
    if (decpt <= 0) {
        return sign + "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    }
    const auto point = static_cast<std::size_t>(decpt);
    if (point >= digits.size()) {
        return sign + digits + std::string(point - digits.size(), '0') + ".0";
    }
    return sign + digits.substr(0, point) + "." + digits.substr(point);
}

std::string repr(bool value) { return value ? "True" : "False"; }

std::string repr_str(const std::string& s) {
    // Python picks single quotes unless the text contains a single quote and
    // no double quote.
    const bool has_single = s.find('\'') != std::string::npos;
    const bool has_double = s.find('"') != std::string::npos;
    const char quote = (has_single && !has_double) ? '"' : '\'';
    std::string out(1, quote);
    for (unsigned char c : s) {
        if (c == static_cast<unsigned char>(quote) || c == '\\') {
            out.push_back('\\');
            out.push_back(static_cast<char>(c));
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 0x20 || c == 0x7f) {
            char buffer[8];
            std::snprintf(buffer, sizeof(buffer), "\\x%02x", c);
            out += buffer;
        } else {
            // Bytes >= 0x80 belong to UTF-8 sequences, which Python prints
            // verbatim for printable characters.
            out.push_back(static_cast<char>(c));
        }
    }
    out.push_back(quote);
    return out;
}

namespace {
std::string strip_ws(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\n\r\f\v");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(begin, end - begin + 1);
}
}  // namespace

// int(text): optional sign, decimal digits with single underscores between
// digits, surrounding whitespace allowed.
std::optional<std::int64_t> parse_int(const std::string& text) {
    static const std::regex re("^[+-]?[0-9](?:_?[0-9])*$");
    const std::string s = strip_ws(text);
    if (!std::regex_match(s, re)) {
        return std::nullopt;
    }
    std::string digits;
    for (char c : s) {
        if (c != '_') {
            digits.push_back(c);
        }
    }
    errno = 0;
    const long long v = std::strtoll(digits.c_str(), nullptr, 10);
    if (errno == ERANGE) {
        return std::nullopt;
    }
    return static_cast<std::int64_t>(v);
}

// float(text): decimal / exponent forms with digit-separating underscores,
// inf/infinity/nan in any case, surrounding whitespace allowed.
std::optional<double> parse_float(const std::string& text) {
    static const std::regex number(
        "^[+-]?(?:(?:[0-9](?:_?[0-9])*)?\\.[0-9](?:_?[0-9])*|[0-9](?:_?[0-9])*\\.?)"
        "(?:[eE][+-]?[0-9](?:_?[0-9])*)?$");
    static const std::regex special("^[+-]?(?:inf|infinity|nan)$", std::regex::icase);
    const std::string s = strip_ws(text);
    if (std::regex_match(s, special)) {
        std::string lowered = s;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const bool negative = lowered[0] == '-';
        if (lowered.find("nan") != std::string::npos) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return negative ? -std::numeric_limits<double>::infinity()
                        : std::numeric_limits<double>::infinity();
    }
    if (!std::regex_match(s, number)) {
        return std::nullopt;
    }
    std::string cleaned;
    for (char c : s) {
        if (c != '_') {
            cleaned.push_back(c);
        }
    }
    return std::strtod(cleaned.c_str(), nullptr);
}

double round(double x, int ndigits) {
    if (!std::isfinite(x)) {
        return x;
    }
    // glibc's printf is correctly rounded on the exact binary value with
    // ties-to-even, which is precisely CPython's float.__round__ semantics.
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer), "%.*f", ndigits, x);
    return std::strtod(buffer, nullptr);
}

std::string format(double x, const std::string& spec) {
    if (spec.empty()) {
        return repr(x);
    }
    // [width][,][.precision]type
    std::size_t i = 0;
    std::string width;
    while (i < spec.size() && std::isdigit(static_cast<unsigned char>(spec[i]))) {
        width.push_back(spec[i++]);
    }
    bool grouping = false;
    if (i < spec.size() && spec[i] == ',') {
        grouping = true;
        ++i;
    }
    std::string precision;
    if (i < spec.size() && spec[i] == '.') {
        ++i;
        while (i < spec.size() && std::isdigit(static_cast<unsigned char>(spec[i]))) {
            precision.push_back(spec[i++]);
        }
    }
    if (i + 1 != spec.size()) {
        throw ValueError("unsupported format spec '" + spec + "'");
    }
    const char type = spec[i];
    if (type != 'f' && type != 'e' && type != 'g') {
        throw ValueError("unsupported format spec '" + spec + "'");
    }
    std::string text;
    if (std::isnan(x)) {
        text = "nan";
    } else if (std::isinf(x)) {
        text = x > 0 ? "inf" : "-inf";
    } else {
        const std::string printf_spec = "%" + (precision.empty() ? "" : "." + precision) + type;
        char buffer[512];
        std::snprintf(buffer, sizeof(buffer), printf_spec.c_str(), x);
        text = buffer;
        if (grouping && type == 'f') {
            const bool negative = !text.empty() && text[0] == '-';
            std::string body = negative ? text.substr(1) : text;
            const auto dot = body.find('.');
            std::string integer = body.substr(0, dot);
            std::string fraction = dot == std::string::npos ? "" : body.substr(dot);
            text = (negative ? "-" : "") + group_thousands(integer) + fraction;
        }
    }
    if (!width.empty()) {
        text = rjust(text, static_cast<std::size_t>(std::stoul(width)));
    }
    return text;
}

std::string format(std::int64_t n, const std::string& spec) {
    std::string body = std::to_string(n < 0 ? -n : n);
    std::string s = spec;
    if (!s.empty() && s.back() == 'd') {
        s.pop_back();
    }
    bool grouping = false;
    if (!s.empty() && s.back() == ',') {
        grouping = true;
        s.pop_back();
    }
    if (grouping) {
        body = group_thousands(body);
    }
    std::string text = (n < 0 ? "-" : "") + body;
    if (!s.empty()) {
        text = rjust(text, static_cast<std::size_t>(std::stoul(s)));
    }
    return text;
}

std::string sprintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::string out = vformat(fmt, args);
    va_end(args);
    return strip_nan_sign(out);
}

std::string ljust(const std::string& s, std::size_t width) {
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

std::string rjust(const std::string& s, std::size_t width) {
    return s.size() >= width ? s : std::string(width - s.size(), ' ') + s;
}

std::string csv_row(const std::vector<std::string>& fields) {
    std::string out;
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i) {
            out.push_back(',');
        }
        const std::string& f = fields[i];
        // QUOTE_MINIMAL: quote fields containing the delimiter, the quote
        // character or any line-terminator character; double embedded quotes.
        const bool needs_quotes = f.find_first_of(",\"\r\n") != std::string::npos;
        if (needs_quotes) {
            out.push_back('"');
            for (char c : f) {
                if (c == '"') {
                    out.push_back('"');
                }
                out.push_back(c);
            }
            out.push_back('"');
        } else {
            out += f;
        }
    }
    out += "\r\n";
    return out;
}

std::string savetxt_row(const double* values, std::size_t n, const char* fmt) {
    std::string out;
    char buffer[64];
    for (std::size_t i = 0; i < n; ++i) {
        if (i) {
            out.push_back(' ');
        }
        std::snprintf(buffer, sizeof(buffer), fmt, values[i]);
        out += strip_nan_sign(buffer);
    }
    out.push_back('\n');
    return out;
}

}  // namespace mvslam::py
