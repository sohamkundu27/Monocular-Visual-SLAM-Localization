// Trajectory container.
//
// A Trajectory is an ordered stack of ``T_w_c`` camera-to-world poses plus
// the frame indices they belong to. Keeping frame indices explicit matters
// because monocular VO is allowed to drop frames, and evaluation must compare
// like-for-like against ground truth rather than assuming dense alignment.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "mvslam/utils/cv_eigen.hpp"

namespace mvslam {

using PoseList = std::vector<Eigen::Matrix4d>;
using KittiRow = std::array<double, 12>;

/// An ordered sequence of camera-to-world poses.
class Trajectory {
public:
    Trajectory() = default;

    /// ``frame_ids`` defaults to 0..N-1; ``timestamps`` (seconds since the
    /// sequence start) are optional. Lengths must match the pose count.
    explicit Trajectory(PoseList poses, std::optional<std::vector<std::int64_t>> frame_ids = {},
                        std::optional<std::vector<double>> timestamps = {});

    /// A single pose is promoted to a one-element trajectory.
    explicit Trajectory(const Eigen::Matrix4d& pose);

    /// Construct from generic (rows x cols) matrices, validating that each is
    /// 4x4 -- the C++ counterpart of passing an arbitrary NumPy array.
    static Trajectory from_matrices(const std::vector<Eigen::MatrixXd>& matrices);

    // -- Sequence protocol -------------------------------------------------- //
    std::size_t size() const { return poses_.size(); }
    bool empty() const { return poses_.empty(); }
    const Eigen::Matrix4d& operator[](std::size_t index) const { return poses_.at(index); }
    Eigen::Matrix4d& operator[](std::size_t index) { return poses_.at(index); }
    const PoseList& poses() const { return poses_; }
    PoseList& poses() { return poses_; }
    const std::vector<std::int64_t>& frame_ids() const { return frame_ids_; }
    const std::optional<std::vector<double>>& timestamps() const { return timestamps_; }

    /// Python slice ``traj[start:stop]`` (negative indices count from the end).
    Trajectory slice(std::int64_t start, std::optional<std::int64_t> stop = {}) const;
    /// Fancy indexing ``traj[indices]``.
    Trajectory take(const std::vector<std::int64_t>& indices) const;

    // -- Derived quantities ------------------------------------------------- //
    /// (N, 3) camera centres in world coordinates.
    Points3 positions() const;
    /// Camera-to-world rotations.
    std::vector<Eigen::Matrix3d> rotations() const;
    /// (N, 2) ground-plane coordinates for top-down plots.
    ///
    /// KITTI camera axes are +x right, +y down, +z forward, so the ground
    /// plane is x-z rather than the more familiar x-y.
    Points2 xz() const;
    /// (N-1) Euclidean distance between consecutive camera centres.
    std::vector<double> step_lengths() const;
    /// (N) distance travelled along the path up to each pose.
    std::vector<double> cumulative_distance() const;
    /// Total travelled path length in metres.
    double path_length() const;
    /// (N-delta) relative transforms ``T_i_{i+delta}``.
    PoseList relative_poses(int delta = 1) const;

    // -- Transformations ---------------------------------------------------- //
    /// Apply a global similarity transform ``(scale, T)`` to every pose.
    ///
    /// Positions map as ``p -> s * R p + t`` and orientations as
    /// ``R_i -> R R_i``. This is exactly the form produced by Umeyama
    /// alignment, so it is how an estimated trajectory is brought into the
    /// ground-truth frame before computing absolute trajectory error.
    Trajectory transformed(const Eigen::Matrix4d& T, double scale = 1.0) const;

    /// Re-express every pose relative to the first, so pose 0 is identity.
    Trajectory relative_to_first() const;

    /// Re-project every rotation onto SO(3).
    ///
    /// Long chains of matrix products accumulate small orthonormality errors;
    /// GTSAM rejects rotations that have drifted, so poses are cleaned before
    /// they enter the pose graph.
    Trajectory orthonormalized() const;

    Trajectory copy() const { return *this; }

    // -- Validation --------------------------------------------------------- //
    /// True when every pose is a finite, right-handed SE(3) member.
    bool is_valid() const;

    // -- Selection ---------------------------------------------------------- //
    /// The sub-trajectory whose frame ids match ``frame_ids``.
    ///
    /// Used to line an estimated trajectory up against dense ground truth when
    /// frames were skipped or sub-sampled. Throws KeyError for missing frames.
    Trajectory select_frames(const std::vector<std::int64_t>& frame_ids) const;

    // -- Serialisation (KITTI 12-value rows) -------------------------------- //
    static Trajectory from_kitti_rows(const std::vector<KittiRow>& rows,
                                      std::optional<std::vector<std::int64_t>> frame_ids = {});
    std::vector<KittiRow> to_kitti_rows() const;

    /// Write KITTI-format poses (``np.savetxt(fmt="%.12e")``) so external
    /// tools (e.g. evo) can read them.
    std::filesystem::path save_kitti(const std::filesystem::path& path) const;

    /// Read a KITTI-format pose file (``np.loadtxt`` semantics). Throws
    /// FileNotFoundError when missing and ValueError when malformed.
    static Trajectory load_kitti(const std::filesystem::path& path);

    std::string repr() const;

private:
    PoseList poses_;
    std::vector<std::int64_t> frame_ids_;
    std::optional<std::vector<double>> timestamps_;
};

/// Parse whitespace-separated float rows like ``np.loadtxt`` (``#`` comments,
/// blank lines skipped). All rows must have the same number of values.
std::vector<std::vector<double>> loadtxt(const std::filesystem::path& path);

}  // namespace mvslam
