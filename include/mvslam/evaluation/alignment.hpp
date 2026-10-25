// Umeyama trajectory alignment.
//
// Why alignment is required
// -------------------------
// An estimated trajectory lives in its own coordinate frame: it starts at the
// identity with an arbitrary orientation, and -- for monocular VO -- an
// arbitrary scale. Comparing raw coordinates against ground truth would
// therefore measure the frame mismatch, not the quality of the estimate.
//
// The standard fix is Umeyama's closed-form solution for the least-squares
// similarity transform between two point sets: find ``s``, ``R``, ``t``
// minimising ``sum_i || s R x_i + t - y_i ||^2``. Two variants matter here:
//
// sim3 (similarity, 7 DoF)
//     Also fits a global scale factor. This is the **correct** choice for a
//     pure monocular trajectory, whose scale is unobservable by construction.
//     Metrics computed after sim3 alignment measure trajectory *shape*.
//
// se3 (rigid, 6 DoF)
//     Fixes scale at 1. Appropriate when the trajectory is already metric --
//     for instance when ground-truth scale was supplied per frame -- because
//     it then keeps any residual scale drift visible in the error rather than
//     absorbing it.
//
// Reference: S. Umeyama, "Least-squares estimation of transformation
// parameters between two point patterns", IEEE TPAMI 13(4), 1991.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/geometry/pose.hpp"

namespace mvslam {

const std::vector<std::string>& alignment_modes();

/// The similarity transform mapping an estimate onto ground truth.
struct AlignmentResult {
    /// Rotation applied to the estimate.
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    /// Translation applied after rotation and scaling.
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    /// Global scale factor (exactly 1.0 for rigid alignment).
    double scale = 1.0;
    /// Which mode produced this result.
    std::string mode = "none";

    /// The rotation and translation as a 4x4 SE(3) matrix.
    ///
    /// Scale is *not* folded in: it is applied separately so that rotations
    /// stay orthonormal. Trajectory::transformed() takes both.
    Eigen::Matrix4d T() const;

    /// Map (N, 3) points through ``s R x + t``.
    Points3 apply(const Points3& points) const;

    Trajectory apply_to_trajectory(const Trajectory& trajectory) const;

    static AlignmentResult identity(const std::string& mode = "none");

    Json describe() const;
};

/// Least-squares similarity transform mapping ``source`` onto ``target``.
///
/// ``source`` and ``target`` are (N, 3) point sets in correspondence (row
/// ``i`` of one matches row ``i`` of the other). ``with_scale`` also fits a
/// global scale factor (sim3).
///
/// The rotation comes from the SVD of the cross-covariance matrix. Umeyama's
/// key correction over naive Procrustes is the sign handling: when the
/// covariance is degenerate or the point sets are mirrored, ``U V^T`` can be
/// a reflection, and the fix is to flip the sign of the smallest singular
/// direction -- which is what the ``S`` matrix does.
AlignmentResult umeyama_alignment(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
                                  bool with_scale = true);

/// Align ``estimated`` onto ``reference`` and return both.
///
/// ``mode == "none"`` returns the estimate untouched with an identity
/// transform, which is how unaligned metrics are produced.
std::pair<Trajectory, AlignmentResult> align_trajectory(const Trajectory& estimated,
                                                        const Trajectory& reference,
                                                        const std::string& mode = "sim3");

}  // namespace mvslam
