// KITTI Odometry dataset loader.
//
// Expected layout (as produced by unzipping the official odometry downloads):
//
//     <root>/
//       sequences/
//         00/
//           calib.txt
//           times.txt
//           image_0/   000000.png 000001.png ...
//           image_1/   ...
//         01/ ...
//       poses/
//         00.txt  01.txt ...   # ground truth, sequences 00-10 only
//
// Ground-truth rows are the 3x4 camera-to-world transform of the **left
// camera** (image_0), which is exactly the camera the monocular pipeline runs
// on, so no extrinsic conversion is needed.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/datasets/calibration.hpp"
#include "mvslam/geometry/pose.hpp"

namespace mvslam {

struct Config;

/// Sequences 11-21 are the held-out test split and ship without ground truth.
const std::set<std::string>& sequences_with_ground_truth();

/// Which frames a run will actually consume.
struct FrameSelection {
    std::int64_t start = 0;
    std::int64_t step = 1;
    std::optional<std::int64_t> max_frames;

    /// The selected frame indices within ``[0, total)``. Throws ValueError for
    /// an invalid selection.
    std::vector<std::int64_t> apply(std::int64_t total) const;
};

/// Random-access reader for one KITTI Odometry sequence.
///
/// Images are read lazily so a 4541-frame sequence never has to fit in RAM.
/// Indexing is over the *selected* frames: ``dataset.load_image(0)`` is the
/// first frame a run will process, and ``dataset.frame_ids()[0]`` gives its
/// original index in the sequence.
class KittiOdometryDataset {
public:
    KittiOdometryDataset(const std::filesystem::path& root, const std::string& sequence = "00",
                         const std::string& camera = "image_0", bool grayscale = true,
                         std::int64_t start_frame = 0, std::int64_t frame_step = 1,
                         std::optional<std::int64_t> max_frames = std::nullopt);
    /// Integer sequence ids are accepted too (``0`` -> "00").
    KittiOdometryDataset(const std::filesystem::path& root, int sequence,
                         const std::string& camera = "image_0", bool grayscale = true);

    /// Build a dataset from a Config.
    static KittiOdometryDataset from_config(const Config& config);

    // -- Sequence protocol ------------------------------------------------- //
    std::size_t size() const { return image_paths_.size(); }

    /// Read the ``index``-th selected frame. Throws DatasetError when the file
    /// exists in the listing but cannot be decoded.
    cv::Mat load_image(std::size_t index) const;

    const std::vector<std::int64_t>& frame_ids() const { return frame_ids_; }
    const std::vector<std::filesystem::path>& image_paths() const { return image_paths_; }

    // -- Calibration ------------------------------------------------------- //
    /// All projection matrices from this sequence's calib.txt.
    const CalibEntries& calib_raw() const;
    /// Intrinsics of the configured camera.
    const CameraCalibration& calibration() const;
    /// 3x3 intrinsic matrix of the configured camera.
    Eigen::Matrix3d K() const { return calibration().K(); }
    /// (width, height) of the first frame.
    std::pair<int, int> image_size() const;

    // -- Timestamps -------------------------------------------------------- //
    /// Seconds since sequence start for the selected frames, if available.
    const std::optional<std::vector<double>>& timestamps() const;

    // -- Ground truth ------------------------------------------------------ //
    std::filesystem::path poses_path() const;
    bool has_ground_truth() const;

    /// Ground-truth trajectory restricted to the selected frames.
    ///
    /// Empty for the test-split sequences (11-21), which ship without poses.
    /// Metrics that require ground truth are skipped rather than fabricated in
    /// that case.
    const std::optional<Trajectory>& ground_truth() const;

    /// Per-transition ground-truth translation magnitudes, in metres.
    ///
    /// ``scales[i]`` is the distance between selected frames ``i`` and
    /// ``i + 1``. This is the quantity used by the ground-truth scale strategy
    /// described in odometry/scale.
    std::optional<std::vector<double>> ground_truth_scales() const;

    /// Small summary used for run metadata and logging.
    Json describe() const;

    std::string repr() const;

    const std::filesystem::path& root() const { return root_; }
    const std::string& sequence() const { return sequence_; }
    const std::string& camera() const { return camera_; }
    bool grayscale() const { return grayscale_; }
    const std::filesystem::path& sequence_dir() const { return sequence_dir_; }
    const std::filesystem::path& image_dir() const { return image_dir_; }
    const FrameSelection& selection() const { return selection_; }
    std::int64_t total_frames() const { return total_frames_; }

private:
    std::filesystem::path root_;
    std::string sequence_;
    std::string camera_;
    bool grayscale_ = true;
    std::filesystem::path sequence_dir_;
    std::filesystem::path image_dir_;
    FrameSelection selection_;
    std::vector<std::int64_t> frame_ids_;
    std::vector<std::filesystem::path> image_paths_;
    std::int64_t total_frames_ = 0;

    // functools.cached_property equivalents.
    mutable std::optional<CalibEntries> calib_raw_;
    mutable std::optional<CameraCalibration> calibration_;
    mutable std::optional<std::pair<int, int>> image_size_;
    mutable bool timestamps_loaded_ = false;
    mutable std::optional<std::vector<double>> timestamps_;
    mutable bool ground_truth_loaded_ = false;
    mutable std::optional<Trajectory> ground_truth_;
};

/// Accept ``"0"`` or ``"00"`` (or any digit string) and return the canonical
/// two-digit id; other names pass through unchanged.
std::string normalize_sequence(const std::string& sequence);

/// ``Path.expanduser()``.
std::filesystem::path expanduser(const std::filesystem::path& path);

}  // namespace mvslam
