// Builds a miniature, on-disk KITTI Odometry tree for tests (port of
// tests/kitti_fixture.py).
//
// Everything the loader touches -- calib.txt, times.txt, image_0/ and
// poses/xx.txt -- is synthesised, so the whole test suite runs without the
// 20 GB official download. Real numbers from sequence 00 are used for the
// intrinsics so the parser is exercised against genuine formatting.
//
// make_textured_image() draws the same circles and rectangles as the Python
// helper. Its random layout came from NumPy, so the drawing parameters per
// (seed, size) are read from tests/golden/textured_images.json and rendered
// with the same OpenCV primitives, giving pixel-identical images.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "mvslam/geometry/pose.hpp"

namespace kitti_fixture {

/// Verbatim from KITTI sequence 00 calib.txt.
extern const char* const kCalibText;

constexpr double kKitti00Fx = 718.856;
constexpr double kKitti00Cx = 607.1928;
constexpr double kKitti00Cy = 185.2157;

/// Render a deterministic, feature-rich grayscale image.
///
/// Random noise alone produces unstable ORB keypoints, so the image is built
/// from blobs and edges that shift smoothly with ``index`` -- enough
/// structure for the detector and matcher to behave like they do on real
/// frames.
cv::Mat make_textured_image(int index, int width = 320, int height = 120, int seed = 0);

/// Write a synthetic KITTI sequence under ``root`` and return ``root``.
///
/// The ground truth is a gentle right-hand curve travelling along +z,
/// matching KITTI's camera convention (+x right, +y down, +z forward).
std::filesystem::path write_kitti_sequence(const std::filesystem::path& root,
                                           const std::string& sequence = "00", int n_frames = 12,
                                           bool with_ground_truth = true, bool with_times = true,
                                           const std::string& camera = "image_0",
                                           double step_m = 0.9);

/// (N) KITTI pose rows for a constant-curvature drive.
std::vector<mvslam::KittiRow> ground_truth_rows(int n_frames, double step_m = 0.9,
                                                double yaw_rate_deg = 1.5);

/// Lower-case hex MD5 of a byte buffer (for the golden render checksums).
std::string md5_hex(const unsigned char* data, std::size_t size);

}  // namespace kitti_fixture
