// PyYAML-compatible loading and dumping.
//
// yaml-cpp parses the document structure, but scalar *typing* follows
// PyYAML's YAML 1.1 implicit resolvers, because that is what decided the
// types the Python config saw. The differences from YAML 1.2 matter here:
// ``00`` is the integer 0 (which is why configs/kitti.yaml quotes its
// sequence ids), ``yes``/``off`` are booleans, ``1e-4`` without a dot is a
// *string*, ``1_000`` is an int and ``1:30`` is sexagesimal 90.
//
// ``safe_dump`` reproduces ``yaml.safe_dump(data, sort_keys=False,
// default_flow_style=False)`` for the plain mappings, lists and scalars the
// configuration consists of, including PyYAML's choice between plain,
// single-quoted and double-quoted scalars.
#pragma once

#include <string>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/errors.hpp"

namespace mvslam {

/// Raised on malformed YAML (PyYAML's ``yaml.YAMLError``). Deliberately not a
/// ValueError: the Python CLI does not catch YAMLError either.
class YamlError : public Error {
public:
    using Error::Error;
};

namespace py {

/// ``yaml.safe_load(text)``.
Json yaml_safe_load(const std::string& text);

/// Type a plain (unquoted, untagged) scalar the way PyYAML's resolver does.
Json yaml_resolve_plain(const std::string& scalar);

/// ``yaml.safe_dump(value, sort_keys=False, default_flow_style=False)``.
std::string yaml_safe_dump(const Json& value);

}  // namespace py
}  // namespace mvslam
