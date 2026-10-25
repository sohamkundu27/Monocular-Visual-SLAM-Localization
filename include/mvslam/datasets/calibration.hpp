// Pinhole camera calibration and KITTI ``calib.txt`` parsing.
//
// KITTI Odometry ships one ``calib.txt`` per sequence containing the 3x4
// projection matrices of the four rectified cameras plus the
// velodyne-to-camera extrinsic:
//
//     P0: fx 0 cx 0  0 fy cy 0  0 0 1 0        # left  grayscale (reference)
//     P1: ...                                   # right grayscale
//     P2: ...                                   # left  colour
//     P3: ...                                   # right colour
//     Tr: r11 r12 r13 tx  r21 ...               # velodyne -> camera 0
//
// Because the images are already rectified, the intrinsic matrix is simply
// the left 3x3 block of ``P`` and there is no lens distortion to undo. The
// fourth column encodes the stereo baseline as ``-fx * b``; the monocular
// pipeline never uses it, but it is parsed so the baseline is available for
// reference and for any future stereo-based scale estimator.
#pragma once

#include <filesystem>
#include <map>
#include <string>

#include <Eigen/Core>

#include "mvslam/utils/cv_eigen.hpp"

namespace mvslam {

/// Intrinsics of a single rectified pinhole camera.
class CameraCalibration {
public:
    /// ``K`` is ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``; ``P`` the 3x4
    /// projection matrix as stored by KITTI; ``name`` which row it came from.
    /// Throws CalibrationError for non-finite values or non-positive focals.
    CameraCalibration(const Eigen::Matrix3d& K, const Eigen::Matrix<double, 3, 4>& P,
                      std::string name = "P0");

    /// Validating constructor for arbitrarily shaped inputs (the C++
    /// counterpart of passing NumPy arrays of any shape).
    static CameraCalibration from_matrices(const Eigen::MatrixXd& K, const Eigen::MatrixXd& P,
                                           const std::string& name = "P0");

    const Eigen::Matrix3d& K() const { return K_; }
    const Eigen::Matrix<double, 3, 4>& P() const { return P_; }
    const std::string& name() const { return name_; }

    double fx() const { return K_(0, 0); }
    double fy() const { return K_(1, 1); }
    double cx() const { return K_(0, 2); }
    double cy() const { return K_(1, 2); }

    /// Stereo baseline relative to the reference camera, in metres.
    ///
    /// KITTI encodes it as ``P[0, 3] == -fx * baseline``, so a camera whose
    /// projection matrix has a zero fourth column (the reference camera)
    /// reports 0.0.
    double baseline_m() const;

    /// Convert (N, 2) pixel coordinates to normalized image coordinates.
    ///
    /// ``x_n = (u - cx) / fx``, ``y_n = (v - cy) / fy``. Useful when a routine
    /// needs calibrated rays rather than handing ``K`` to OpenCV.
    Points2 normalize_points(const Eigen::MatrixXd& points) const;

    /// Project (N, 3) camera-frame points to (N, 2) pixels.
    ///
    /// Points at or behind the image plane (``z <= 0``) produce NaN rather
    /// than silently wrapping around to a bogus pixel.
    Points2 project(const Eigen::MatrixXd& points_cam) const;

    std::string repr() const;

private:
    Eigen::Matrix3d K_;
    Eigen::Matrix<double, 3, 4> P_;
    std::string name_;
};

/// ``{key: matrix}``: P0-P3 as 3x4 matrices, Tr (when present) as a 4x4 SE(3)
/// transform with the implicit bottom row filled in.
using CalibEntries = std::map<std::string, Eigen::MatrixXd>;

/// Parse a KITTI ``calib.txt``. Throws CalibrationError on malformed input.
CalibEntries parse_kitti_calib(const std::filesystem::path& path);

/// The image directory name -> projection-matrix key.
const std::map<std::string, std::string>& camera_to_projection();

/// Extract the CameraCalibration for a KITTI image directory.
CameraCalibration calibration_for_camera(const CalibEntries& calib,
                                         const std::string& camera = "image_0");

}  // namespace mvslam
