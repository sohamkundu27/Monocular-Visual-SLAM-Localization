// cv::Mat <-> Eigen conversions.
//
// Layout rules, made explicit at every boundary:
//
// * cv::Mat is row-major. Eigen's default is column-major. Point arrays are
//   therefore declared as *row-major* Eigen matrices (``Points2``,
//   ``Points3``), so an (N, 2) block of doubles has the same memory layout in
//   both libraries and can be handed to OpenCV without a copy -- the same
//   layout a C-contiguous (N, 2) float64 NumPy array has, which is what cv2
//   receives in the Python original.
// * Fixed-size matrices (Matrix3d/Matrix4d) are column-major and are always
//   converted element by element (``at<double>(r, c)``), never by reinterpreting
//   memory.
// * Geometry is double precision throughout; descriptors stay uint8 (ORB)
//   or float32 (SIFT) cv::Mat rows, as in the Python arrays.
#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace mvslam {

/// (N, 2) pixel coordinates, row-major so the data is OpenCV-compatible.
using Points2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
/// (N, 3) points, row-major.
using Points3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
/// Boolean mask as 0/1 bytes (OpenCV's mask convention; NumPy bool arrays).
using Mask = std::vector<std::uint8_t>;

/// Zero-copy (N x 2, CV_64FC1) view of ``points``. The view aliases the
/// Eigen storage and must not outlive it. OpenCV only reads it.
cv::Mat as_cv_view(const Points2& points);

/// Copy a 3x3 Eigen matrix into a new row-major CV_64F cv::Mat.
cv::Mat to_cv(const Eigen::Matrix3d& m);
/// Copy a 3x4 Eigen matrix into a new row-major CV_64F cv::Mat.
cv::Mat to_cv(const Eigen::Matrix<double, 3, 4>& m);

/// Element-wise copy of a 3x3 cv::Mat (any floating depth) into Eigen.
Eigen::Matrix3d to_matrix3d(const cv::Mat& m);
/// Element-wise copy of a 3-element cv::Mat (3x1 or 1x3) into Eigen.
Eigen::Vector3d to_vector3d(const cv::Mat& m);

/// Mask from a cv::Mat of N bytes (any of N x 1 / 1 x N), nonzero -> 1.
Mask to_mask(const cv::Mat& m, std::size_t n);

/// Number of nonzero entries.
std::size_t count_nonzero(const Mask& mask);

/// Rows of ``points`` where ``mask`` is set.
Points2 select_rows(const Points2& points, const Mask& mask);

}  // namespace mvslam
