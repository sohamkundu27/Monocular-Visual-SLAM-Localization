// Logging configuration.
//
// Every run writes a human readable console stream and, when an output
// directory is supplied, a verbatim copy to ``<output_dir>/run.log`` so
// results are auditable after the fact.
//
// A small hand-rolled logger rather than spdlog: run.log must keep the exact
// Python format
//
//     %(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s   (asctime %H:%M:%S)
//
// and the same handler semantics: a console handler and an optional file
// handler with independent thresholds, re-configuration that replaces rather
// than duplicates handlers, and -- before setup_logging() is called -- only
// WARNING and above reaching stderr as the bare message (Python's
// ``logging.lastResort``).
#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace mvslam {

enum class LogLevel : int { Debug = 10, Info = 20, Warning = 30, Error = 40 };

/// Parse "DEBUG"/"info"/... like ``getattr(logging, level.upper(), default)``.
LogLevel parse_log_level(const std::string& level, LogLevel fallback = LogLevel::Info);

class Logger {
public:
    explicit Logger(std::string name) : name_(std::move(name)) {}

    const std::string& name() const { return name_; }

    void debug(const std::string& message) const { log(LogLevel::Debug, message); }
    void info(const std::string& message) const { log(LogLevel::Info, message); }
    void warning(const std::string& message) const { log(LogLevel::Warning, message); }
    void error(const std::string& message) const { log(LogLevel::Error, message); }
    void log(LogLevel level, const std::string& message) const;

    /// True when a message at ``level`` would reach any handler.
    bool enabled_for(LogLevel level) const;

private:
    std::string name_;
};

/// Configure the package logger.
///
/// ``level`` is the console threshold; ``file_level`` the threshold of the
/// optional ``log_file`` (parent directories are created). ``file_level``
/// defaults to INFO so run.log stays a complete record of the run even when
/// the console is turned down -- an empty log next to a set of result files is
/// worse than useless when you come back to it later. ``quiet`` suppresses the
/// console handler only.
void setup_logging(LogLevel level = LogLevel::Info,
                   const std::optional<std::filesystem::path>& log_file = std::nullopt,
                   bool quiet = false, LogLevel file_level = LogLevel::Info);

/// Return a child logger under the package root ``monocular_slam``. ``name``
/// is the dotted Python module name (e.g. "monocular_slam.features.detector");
/// a name outside the package is nested under it.
Logger get_logger(const std::string& name);

/// Number of installed handlers (console + file), for tests.
int logging_handler_count();

/// True when the only handler is a file handler (``quiet=True``), for tests.
bool logging_has_only_file_handler();

/// Flush the file handler, for tests.
void flush_logging();

/// Remove all handlers and return to the unconfigured state.
void reset_logging();

}  // namespace mvslam
