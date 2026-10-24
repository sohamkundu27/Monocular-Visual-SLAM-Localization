#include "mvslam/utils/timing.hpp"

#include <algorithm>

#include "mvslam/compat/pyformat.hpp"

namespace mvslam {

StageTimer::Scope::~Scope() {
    if (timer_ != nullptr) {
        const std::chrono::duration<double> elapsed = Clock::now() - started_;
        timer_->record(stage_, elapsed.count());
    }
}

StageTimer::StageTimer() : start_(Clock::now()) {}

void StageTimer::record(const std::string& stage, double seconds) {
    StageStats& entry = stats(stage);
    entry.total_s += seconds;
    entry.count += 1;
    entry.max_s = std::max(entry.max_s, seconds);
}

void StageTimer::reset_wall_clock() { start_ = Clock::now(); }

double StageTimer::elapsed_s() const {
    const std::chrono::duration<double> elapsed = Clock::now() - start_;
    return elapsed.count();
}

StageStats& StageTimer::stats(const std::string& stage) {
    for (auto& [name, entry] : stats_) {
        if (name == stage) {
            return entry;
        }
    }
    stats_.emplace_back(stage, StageStats{});
    return stats_.back().second;
}

bool StageTimer::has_stage(const std::string& stage) const {
    return std::any_of(stats_.begin(), stats_.end(),
                       [&](const auto& item) { return item.first == stage; });
}

double StageTimer::total_s(const std::string& stage) const {
    for (const auto& [name, entry] : stats_) {
        if (name == stage) {
            return entry.total_s;
        }
    }
    return 0.0;
}

double StageTimer::mean_ms(const std::string& stage) const {
    for (const auto& [name, entry] : stats_) {
        if (name == stage) {
            return entry.mean_ms();
        }
    }
    return 0.0;
}

double StageTimer::fps(long n_frames) const {
    const double elapsed = elapsed_s();
    return elapsed > 0 ? static_cast<double>(n_frames) / elapsed : 0.0;
}

Json StageTimer::summary() const {
    std::vector<std::pair<std::string, StageStats>> sorted = stats_;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    Json out = Json::object();
    for (const auto& [stage, entry] : sorted) {
        Json item = Json::object();
        item["total_s"] = py::make_float(py::round(entry.total_s, 4));
        item["mean_ms"] = py::make_float(py::round(entry.mean_ms(), 3));
        item["max_ms"] = py::make_float(py::round(entry.max_ms(), 3));
        item["calls"] = entry.count;
        out[stage] = item;
    }
    return out;
}

std::string StageTimer::report() const {
    if (stats_.empty()) {
        return "no timing data";
    }
    std::size_t width = 0;
    for (const auto& item : stats_) {
        width = std::max(width, item.first.size());
    }
    std::vector<std::pair<std::string, StageStats>> sorted = stats_;
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return -a.second.total_s < -b.second.total_s;
    });
    std::string out = py::ljust("stage", width) + "   total_s    mean_ms     max_ms   calls";
    for (const auto& [stage, entry] : sorted) {
        out += "\n" + py::ljust(stage, width) +
               py::sprintf(" %9.3f %10.3f %10.3f %7ld", entry.total_s, entry.mean_ms(),
                           entry.max_ms(), entry.count);
    }
    return out;
}

}  // namespace mvslam
