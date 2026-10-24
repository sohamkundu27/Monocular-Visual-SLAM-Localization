// C++ counterparts of tests/conftest.py and tests/synthetic.py.
//
// Helpers that draw random numbers in Python (random_se3, small_se3,
// make_scene, the noise in two_view_correspondences) replay the recorded
// values from a golden::Stream instead; deterministic helpers are ported
// directly.
#pragma once

#include <optional>
#include <utility>

#include <Eigen/Core>

#include "mvslam/geometry/pose.hpp"
#include "mvslam/utils/cv_eigen.hpp"
#include "support/golden.hpp"

namespace synthetic {

/// KITTI sequence 00 intrinsics, so tests exercise realistic focal lengths.
Eigen::Matrix3d kitti_K();

// -- conftest.py ------------------------------------------------------------ //
/// ``random_se3(rng)``: a random valid SE(3) pose (recorded).
Eigen::Matrix4d random_se3(golden::Stream& rng);
/// ``small_se3(rng, ...)``: a small motion (recorded).
Eigen::Matrix4d small_se3(golden::Stream& rng);

// -- synthetic.py ----------------------------------------------------------- //
/// Pinhole projection of (N, 3) camera-frame points to (N, 2) pixels.
mvslam::Points2 project(const mvslam::Points3& points_cam, const Eigen::Matrix3d& K = kitti_K());

/// ``make_scene(...)``: (N, 3) points in front of the camera (recorded).
mvslam::Points3 make_scene(golden::Stream& stream);

struct Correspondences {
    mvslam::Points2 points_a;
    mvslam::Points2 points_b;
    mvslam::Points3 points_cam1;
};

/// ``two_view_correspondences(T_c1_c2, points_world, K, noise_px, ...)``.
///
/// The scene comes from the stream (make_scene) unless ``points_world`` is
/// given; noise, when ``noise_px > 0``, is the recorded normal draws. The
/// projection and image-bounds filtering are ported directly.
Correspondences two_view_correspondences(golden::Stream& stream, const Eigen::Matrix4d& T_c1_c2,
                                         const std::optional<mvslam::Points3>& points_world = {},
                                         double noise_px = 0.0,
                                         std::optional<std::pair<int, int>> image_size =
                                             std::make_pair(1241, 376));

/// A KITTI-like step: drive forward along +z with an optional yaw.
Eigen::Matrix4d forward_motion(double distance = 0.9, double yaw_deg = 0.0, double lateral = 0.0);

/// A degenerate motion: rotation with no translation.
Eigen::Matrix4d pure_rotation(double yaw_deg = 3.0);

/// (N) ground-truth camera-to-world poses for a curving drive.
mvslam::PoseList straight_drive(int n_frames = 40, double step = 0.9, double yaw_rate_deg = 0.8);

}  // namespace synthetic
