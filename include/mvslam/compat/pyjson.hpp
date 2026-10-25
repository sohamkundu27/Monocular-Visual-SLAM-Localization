// Python-compatible JSON.
//
// ``Json`` (an insertion-ordered nlohmann value) stands in for the plain
// Python objects the original builds with dicts and lists: key order is
// preserved exactly as Python dicts preserve it, and integers stay distinct
// from floats.
//
// ``dumps`` reproduces ``json.dumps`` byte for byte (separators, indentation,
// ``ensure_ascii`` escaping, float ``repr``, and the non-standard ``NaN`` /
// ``Infinity`` tokens Python writes for non-finite floats). ``loads`` accepts
// those tokens too, exactly like ``json.loads``.
#pragma once

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace mvslam {

using Json = nlohmann::ordered_json;

namespace py {

/// ``json.dumps(value, indent=indent)``; ``std::nullopt`` is the compact form
/// with ``(", ", ": ")`` separators.
std::string dumps(const Json& value, std::optional<int> indent = std::nullopt);

/// ``json.loads(text)``. Throws JsonDecodeError on malformed input.
Json loads(const std::string& text);

/// ``repr()`` of the Python object a Json value stands for
/// (``None``, ``True``, ``1.5``, ``'text'``, ``[1, 2]``, ``{'a': 1}``).
std::string repr(const Json& value);

/// ``str()`` of that object: like repr, except a str is returned verbatim.
std::string str(const Json& value);

/// ``type(value).__name__``: NoneType, bool, int, float, str, list, dict.
std::string type_name(const Json& value);

/// Float from a Json number (int or float), NaN for null.
double as_double(const Json& value);

/// A float Json value; non-finite values are kept (they dump as NaN/Infinity).
Json make_float(double value);

}  // namespace py
}  // namespace mvslam
