// Lightweight runtime instrumentation.
//
// Stage timings are accumulated with a monotonic clock (``time.perf_counter``
// in the Python original) around the four front-end stages (detection,
// matching, pose estimation, loop closure) plus optimization. The overhead is
// a pair of clock reads per call, which is negligible next to ORB detection
// at tens of ms per frame, so instrumentation is always on rather than behind
// a flag.
#pragma once

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "mvslam/compat/pyjson.hpp"

namespace mvslam {

/// Accumulated timings for one named stage.
struct StageStats {
    double total_s = 0.0;
    long count = 0;
    double max_s = 0.0;

    double mean_ms() const { return count ? total_s / static_cast<double>(count) * 1000.0 : 0.0; }
    double max_ms() const { return max_s * 1000.0; }
};

/// Accumulates wall-clock time per named stage.
///
///     StageTimer timer;
///     { auto scope = timer.time("detect"); ... }
///     timer.stats("detect").count == 1
class StageTimer {
public:
    using Clock = std::chrono::steady_clock;

    /// RAII scope that records the elapsed time into a stage when destroyed.
    class Scope {
    public:
        Scope(StageTimer& timer, std::string stage)
            : timer_(&timer), stage_(std::move(stage)), started_(Clock::now()) {}
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept
            : timer_(other.timer_), stage_(std::move(other.stage_)), started_(other.started_) {
            other.timer_ = nullptr;
        }
        Scope& operator=(Scope&&) = delete;
        ~Scope();

    private:
        StageTimer* timer_;
        std::string stage_;
        Clock::time_point started_;
    };

    StageTimer();

    /// Time the enclosing block and add it to ``stage``.
    [[nodiscard]] Scope time(const std::string& stage) { return Scope(*this, stage); }

    void record(const std::string& stage, double seconds);

    /// Restart the total-runtime clock (call when real work begins).
    void reset_wall_clock();

    /// Wall-clock seconds since construction or the last reset.
    double elapsed_s() const;

    /// Stats for ``stage``, creating an empty entry when absent (the Python
    /// timer is a defaultdict, and the VO loop relies on that).
    StageStats& stats(const std::string& stage);
    bool has_stage(const std::string& stage) const;
    const std::vector<std::pair<std::string, StageStats>>& all_stats() const { return stats_; }

    double total_s(const std::string& stage) const;
    double mean_ms(const std::string& stage) const;

    /// Frames per second over the wall-clock window.
    double fps(long n_frames) const;

    /// Per-stage totals in a JSON-serialisable form, sorted by stage name.
    Json summary() const;

    /// Human-readable table for the run log, slowest stage first.
    std::string report() const;

private:
    // Insertion-ordered, like the Python dict, so ties sort identically.
    std::vector<std::pair<std::string, StageStats>> stats_;
    Clock::time_point start_;
};

}  // namespace mvslam
