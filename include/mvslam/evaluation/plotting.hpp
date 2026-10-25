// Trajectory and diagnostic plots.
//
// The Python implementation renders these with matplotlib. There is no
// matplotlib in C++, and embedding Python is deliberately avoided, so the same
// figure types are drawn with OpenCV primitives: axes, nice ticks, a dotted
// grid, legends, titles and axis labels. The result is functional rather than
// pretty. Every plot also writes its underlying data next to the image as
// ``<stem>_data.csv`` (columns ``series,x,y[,z]``) so it can be re-plotted
// externally with any tool.
//
// Top-down plots use **X against Z**, not X against Y: KITTI camera axes are
// +x right, +y down, +z forward, so the ground plane is x-z. Figure sizes
// follow the matplotlib ``figsize`` (inches) times ``dpi``.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mvslam/geometry/pose.hpp"
#include "mvslam/odometry/visual_odometry.hpp"

namespace mvslam::plot {

/// Consistent colours across every figure (``#rrggbb``), as in the Python module.
inline constexpr const char* kColorGt = "#111111";
inline constexpr const char* kColorRaw = "#d1495b";
inline constexpr const char* kColorOptimized = "#2a9d8f";
inline constexpr const char* kColorLoop = "#3d5a80";
inline constexpr const char* kColorAccent = "#e9c46a";

using NamedTrajectories = std::vector<std::pair<std::string, Trajectory>>;

/// ``<dir>/<stem>_data.csv`` for an image path.
std::filesystem::path data_path_for(const std::filesystem::path& image_path);

/// Top-down x-z plot of one trajectory, optionally against ground truth.
std::filesystem::path plot_trajectory(const Trajectory& trajectory, const std::filesystem::path& path,
                                      const std::optional<Trajectory>& reference = std::nullopt,
                                      const std::string& label = "Estimated",
                                      const std::string& title = "Trajectory (top-down)", int dpi = 150,
                                      const std::string& color = kColorRaw);

/// Overlay several trajectories (raw VO, optimized SLAM, ground truth).
std::filesystem::path plot_trajectory_comparison(const NamedTrajectories& trajectories,
                                                 const std::filesystem::path& path,
                                                 const std::optional<Trajectory>& reference = std::nullopt,
                                                 const std::string& title = "Trajectory comparison",
                                                 int dpi = 150);

/// 3D view, which is where vertical drift becomes visible.
///
/// KITTI's y axis points *down*, so it is negated to make the plot read the
/// conventional way up. Axes are equalised: without that the vertical axis
/// would be stretched enormously, making a few metres of height drift look
/// like a catastrophic failure.
std::filesystem::path plot_trajectory_3d(const NamedTrajectories& trajectories,
                                         const std::filesystem::path& path,
                                         const std::optional<Trajectory>& reference = std::nullopt,
                                         const std::string& title = "Trajectory (3D)", int dpi = 150);

/// Draw accepted loop-closure edges on the estimated trajectory. ``loops``
/// holds ``(index_a, index_b)`` pairs indexing into ``trajectory``;
/// out-of-range pairs are skipped.
std::filesystem::path plot_loop_closures(const Trajectory& trajectory,
                                         const std::vector<std::pair<std::int64_t, std::int64_t>>& loops,
                                         const std::filesystem::path& path,
                                         const std::optional<Trajectory>& reference = std::nullopt,
                                         const std::string& title = "Detected loop closures", int dpi = 150);

/// Per-frame position error against frame index or travelled distance.
std::filesystem::path plot_error_over_time(const std::vector<std::pair<std::string, std::vector<double>>>& errors,
                                           const std::filesystem::path& path,
                                           const std::optional<std::vector<double>>& distances = std::nullopt,
                                           const std::string& title = "Absolute position error", int dpi = 150);

/// Four-panel summary of feature, match and inlier behaviour over the run.
/// Throws ValueError for an empty list.
std::filesystem::path plot_diagnostics(const std::vector<FrameDiagnostics>& diagnostics,
                                       const std::filesystem::path& path,
                                       const std::string& title = "Front-end diagnostics", int dpi = 150);

}  // namespace mvslam::plot
