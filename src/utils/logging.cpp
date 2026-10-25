#include "mvslam/utils/logging.hpp"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>

#include "mvslam/compat/pyformat.hpp"

namespace mvslam {

namespace {

constexpr const char* kRootLoggerName = "monocular_slam";

struct LoggingState {
    std::mutex mutex;
    bool configured = false;
    bool console = false;
    LogLevel console_level = LogLevel::Info;
    std::unique_ptr<std::ofstream> file;
    LogLevel file_level = LogLevel::Info;
};

LoggingState& state() {
    static LoggingState instance;
    return instance;
}

const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "DEBUG";
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warning:
            return "WARNING";
        case LogLevel::Error:
            return "ERROR";
    }
    return "INFO";
}

std::string format_record(LogLevel level, const std::string& name, const std::string& message) {
    std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char clock[16];
    std::strftime(clock, sizeof(clock), "%H:%M:%S", &local);
    return std::string(clock) + " | " + py::ljust(level_name(level), 7) + " | " +
           py::ljust(name, 28) + " | " + message;
}

}  // namespace

LogLevel parse_log_level(const std::string& level, LogLevel fallback) {
    std::string upper = level;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (upper == "DEBUG") {
        return LogLevel::Debug;
    }
    if (upper == "INFO") {
        return LogLevel::Info;
    }
    if (upper == "WARNING" || upper == "WARN") {
        return LogLevel::Warning;
    }
    if (upper == "ERROR") {
        return LogLevel::Error;
    }
    if (upper == "CRITICAL" || upper == "FATAL") {
        return LogLevel::Error;
    }
    return fallback;
}

bool Logger::enabled_for(LogLevel level) const {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured) {
        return level >= LogLevel::Warning;
    }
    return (s.console && level >= s.console_level) || (s.file && level >= s.file_level);
}

void Logger::log(LogLevel level, const std::string& message) const {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.configured) {
        // logging.lastResort: WARNING and above, message only, on stderr.
        if (level >= LogLevel::Warning) {
            std::cerr << message << '\n';
        }
        return;
    }
    const std::string record = format_record(level, name_, message);
    if (s.console && level >= s.console_level) {
        std::cout << record << '\n' << std::flush;
    }
    if (s.file && level >= s.file_level) {
        *s.file << record << '\n';
        s.file->flush();
    }
}

void setup_logging(LogLevel level, const std::optional<std::filesystem::path>& log_file,
                   bool quiet, LogLevel file_level) {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    // Re-configuring (e.g. across sequences in one process) must not
    // duplicate every message, so previously installed handlers go first.
    s.file.reset();
    s.configured = true;
    s.console = !quiet;
    s.console_level = level;
    s.file_level = file_level;
    if (log_file) {
        if (log_file->has_parent_path()) {
            std::filesystem::create_directories(log_file->parent_path());
        }
        s.file = std::make_unique<std::ofstream>(*log_file, std::ios::out | std::ios::trunc);
    }
}

Logger get_logger(const std::string& name) {
    const std::string prefix = std::string(kRootLoggerName) + ".";
    if (name == kRootLoggerName || name.rfind(prefix, 0) == 0) {
        return Logger(name);
    }
    return Logger(prefix + name);
}

int logging_handler_count() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return (s.console ? 1 : 0) + (s.file ? 1 : 0);
}

bool logging_has_only_file_handler() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return !s.console && s.file != nullptr;
}

void flush_logging() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.file) {
        s.file->flush();
    }
    std::cout.flush();
}

void reset_logging() {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.file.reset();
    s.console = false;
    s.configured = false;
}

}  // namespace mvslam
