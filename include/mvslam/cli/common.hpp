// Shared CLI argument handling.
//
// Configuration is resolved in three layers, each overriding the last:
//
// 1. struct defaults in config.hpp
// 2. the YAML file given by --config
// 3. explicit command-line flags, plus arbitrary ``--set key=value`` overrides
//
// The flags, defaults, choices and exit codes are those of the Python
// argparse entry points. Parsing uses CLI11; help goes to stdout with exit
// code 0, and a usage error prints ``<prog>: error: ...`` to stderr with exit
// code 2, as argparse does. Unambiguous long-option prefixes (e.g. --seq)
// are expanded before CLI11 parsing to preserve argparse's behaviour.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/config.hpp"

namespace mvslam::cli {

inline constexpr const char* kDefaultConfig = "configs/kitti.yaml";

/// Values of the flags every pipeline entry point shares.
struct CommonArgs {
    std::optional<std::string> sequence;
    std::optional<std::string> dataset_path;
    std::optional<std::int64_t> max_frames;
    std::optional<std::int64_t> start_frame;
    std::optional<std::int64_t> frame_step;
    std::string config = kDefaultConfig;
    std::optional<std::string> output_dir;
    std::optional<std::int64_t> seed;
    std::optional<std::string> log_level;
    std::vector<std::string> overrides;
    // Entry-point specific switches (run_slam has both, run_vo only --no-plots).
    bool no_loop_closure = false;
    bool no_plots = false;
};

/// Outcome of argument parsing: either proceed, or exit with ``exit_code``
/// (help was printed, or a usage error was reported).
struct ParseOutcome {
    bool proceed = true;
    int exit_code = 0;
};

/// Parse run_slam's arguments (``argv`` excludes the program name).
ParseOutcome parse_run_slam_args(const std::vector<std::string>& argv, CommonArgs& args);
/// Parse run_vo's arguments.
ParseOutcome parse_run_vo_args(const std::vector<std::string>& argv, CommonArgs& args);

/// Build the resolved Config from parsed arguments. Throws ConfigError.
Config config_from_args(const CommonArgs& args);

/// Convert a ``--set`` value string to bool/int/float/None where sensible.
///
/// Zero-padded values are deliberately left as strings: KITTI sequence ids are
/// written ``00`` and ``05``, and turning those into ``0`` and ``5`` would make
/// ``--set dataset.sequence=00`` silently look for the wrong directory.
Json parse_scalar(const std::string& text);

/// Directory for this run's artefacts.
std::filesystem::path resolve_output_dir(const CommonArgs& args, const Config& config);

}  // namespace mvslam::cli
