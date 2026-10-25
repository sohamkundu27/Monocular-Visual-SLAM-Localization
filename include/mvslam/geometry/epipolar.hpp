// Two-view epipolar geometry: essential matrix and relative pose.
//
// Given calibrated correspondences between two views, the essential matrix
// ``E`` satisfies ``x2^T E x1 = 0`` for normalized image coordinates.
// Decomposing ``E`` yields four candidate ``(R, t)`` poses, of which exactly
// one places the triangulated points in front of both cameras; OpenCV's
// ``recoverPose`` performs that cheirality test.
//
// Two things this module refuses to take on trust from OpenCV:
//
// * **The returned matrices may be garbage.** ``findEssentialMat`` can return
//   nothing, a stack of several candidate matrices, or a matrix containing
//   NaN. ``recoverPose`` can return a rotation with ``det = -1``. Every output
//   is validated before it reaches the trajectory.
// * **Degenerate geometry is silently accepted.** Under pure rotation, or
//   when a car is stopped at a traffic light, the translation direction is
//   unobservable and ``E`` is ill-conditioned. Median parallax is measured
//   and reported so the caller can reject those transitions instead of
//   integrating noise.
//
// Sign convention
// ---------------
// ``cv::recoverPose`` returns ``(R, t)`` such that a point expressed in the
// first camera frame maps into the second as ``x2 = R x1 + t`` -- that is,
// ``[R|t]`` is ``T_c2_c1``. The camera *motion* needed to walk the trajectory
// forward is its inverse, ``T_c1_c2``, and that is what recover_relative_pose()
// returns.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "mvslam/utils/cv_eigen.hpp"

namespace mvslam {

/// Outcome of RANSAC essential-matrix estimation.
struct EssentialMatrixResult {
    std::optional<Eigen::Matrix3d> E;
    Mask inlier_mask;
    std::int64_t n_points = 0;
    bool ok = false;
    std::string reason;

    std::int64_t n_inliers() const { return static_cast<std::int64_t>(count_nonzero(inlier_mask)); }
    double inlier_ratio() const {
        return n_points ? static_cast<double>(n_inliers()) / static_cast<double>(n_points) : 0.0;
    }
    std::string repr() const;
};

/// Outcome of decomposing an essential matrix into a relative pose.
struct RelativePoseResult {
    /// ``T_c1_c2``: the pose of the second camera in the first camera's
    /// frame, with **unit-norm translation**. Monocular geometry cannot
    /// determine the magnitude; see odometry/scale.
    std::optional<Eigen::Matrix4d> T;
    std::optional<Eigen::Matrix3d> R;
    std::optional<Eigen::Vector3d> t_unit;
    Mask inlier_mask;
    /// Correspondences that triangulate in front of both cameras.
    std::int64_t n_cheirality_inliers = 0;
    /// Median angle between corresponding calibrated rays. Near zero means the
    /// motion is (close to) pure rotation and the translation direction is
    /// not observable.
    double median_parallax_deg = 0.0;
    double rotation_deg = 0.0;
    bool ok = false;
    std::string reason;

    std::int64_t n_inliers() const { return static_cast<std::int64_t>(count_nonzero(inlier_mask)); }
    std::string repr() const;
};

/// Map a config string to an OpenCV estimator flag.
///
/// MAGSAC++ is the default: on KITTI sequence 00 it cuts mean per-frame
/// rotation error from 0.27 deg to 0.07 deg versus plain RANSAC at the same
/// threshold, because it marginalises over the inlier noise scale instead of
/// committing to one hard threshold.
int resolve_ransac_method(const std::string& name);

/// Estimate the essential matrix with a robust estimator.
///
/// The threshold is expressed in **pixels** and passed alongside ``K`` so
/// OpenCV performs the normalization internally -- this keeps the geometric
/// meaning of the threshold intact regardless of focal length, which a
/// hand-normalized threshold quietly loses.
///
/// Returns a result whose ``ok`` flag is false (with a ``reason``) rather
/// than throwing, so a single bad frame pair cannot abort a long run.
EssentialMatrixResult estimate_essential_matrix(const Points2& points_a, const Points2& points_b,
                                                const Eigen::Matrix3d& K, double threshold_px = 0.5,
                                                double confidence = 0.999, int max_iters = 2000,
                                                int min_points = 8,
                                                const std::string& method = "magsac");

/// Gates applied by recover_relative_pose().
struct PoseGates {
    std::int64_t min_inliers = 30;
    double min_inlier_ratio = 0.3;
    double max_rotation_deg = 30.0;
    double min_parallax_deg = 0.0;
    double cheirality_distance = 200.0;
};

/// Decompose ``E`` into ``T_c1_c2`` and validate the result.
///
/// Validation performed, in order:
///
/// * ``R`` must be a finite, right-handed rotation (``det = +1``).
/// * ``t`` must be finite and non-degenerate before normalization.
/// * The cheirality test must retain ``min_inliers`` correspondences, and
///   they must be at least ``min_inlier_ratio`` of the epipolar inliers
///   passed in.
/// * The rotation must be under ``max_rotation_deg`` -- at KITTI's 10 Hz a
///   larger inter-frame rotation is a decomposition failure, not real motion.
/// * Median parallax must exceed ``min_parallax_deg`` when that guard is on.
///
/// ``cheirality_distance`` bounds how far a triangulated point may lie, in
/// units of the (unit) translation. OpenCV's default of 50 discards anything
/// beyond ~50 baselines, which on KITTI means everything past roughly 40 m --
/// a large fraction of a road scene -- and depresses the inlier count for
/// entirely healthy frames. 200 keeps distant structure while still rejecting
/// points at infinity.
RelativePoseResult recover_relative_pose(const std::optional<Eigen::Matrix3d>& E,
                                         const Points2& points_a, const Points2& points_b,
                                         const Eigen::Matrix3d& K,
                                         const std::optional<Mask>& inlier_mask = std::nullopt,
                                         const PoseGates& gates = {});

/// Median parallax angle between corresponding viewing rays, in degrees.
///
/// Parallax is the angular disagreement between the two rays *after the
/// camera rotation has been removed*: each ray from the second view is
/// rotated into the first camera's frame via ``R_c1_c2`` before the
/// comparison. Skipping that step measures rotation rather than parallax, and
/// would report a large value for a purely rotating camera -- exactly the
/// case this metric exists to catch.
///
/// With rotation compensated, near-zero parallax means the two views share an
/// optical centre (stationary or pure-rotation motion), so the translation
/// direction is unobservable and the essential matrix is ill-conditioned.
///
/// ``R_c1_c2`` may be omitted when the rotation is known to be negligible.
double median_parallax_deg(const Points2& points_a, const Points2& points_b, const Eigen::Matrix3d& K,
                           const std::optional<Eigen::Matrix3d>& R_c1_c2 = std::nullopt,
                           const std::optional<Mask>& mask = std::nullopt);

/// Homography-vs-essential comparison used to detect degenerate two-view
/// geometry.
///
/// Two configurations break monocular essential-matrix estimation:
///
/// * **Pure rotation / stationary camera.** With no translation the
///   essential matrix ``E = [t]_x R`` collapses to zero and RANSAC fits noise.
///   The recovered rotation is then wrong too, which is why
///   rotation-compensated parallax cannot detect this case -- it de-rotates by
///   a bogus rotation.
/// * **Planar scene.** When all correspondences lie on one plane (a road
///   surface filling the frame) the essential matrix is under-constrained.
///
/// In both cases a homography explains the correspondences at least as well
/// as the epipolar model. Following the model-selection heuristic popularised
/// by ORB-SLAM, both models are scored by the same robust criterion and
/// compared via ``ratio_h = S_H / (S_H + S_E)``. A high ratio means "a
/// homography is sufficient", i.e. the two-view translation is not reliably
/// observable.
struct ModelSelection {
    double score_h = 0.0;
    double score_e = 0.0;
    double ratio_h = 0.0;
    bool is_degenerate = false;
    std::optional<Eigen::Matrix3d> H;
};

/// Chi-squared 95% thresholds. A homography residual has 2 degrees of
/// freedom, an epipolar (point-to-line) residual has 1.
constexpr double kChi2H = 5.991;
constexpr double kChi2E = 3.841;

/// Score a homography against the essential matrix for the same matches.
/// ``findHomography`` draws from ``cv::theRNG()``, exactly as cv2's does.
ModelSelection select_two_view_model(const Points2& points_a, const Points2& points_b,
                                     const Eigen::Matrix3d& K,
                                     const std::optional<Eigen::Matrix3d>& E,
                                     double threshold_px = 1.0, double ratio_threshold = 0.45);

/// Median optical-flow magnitude in pixels.
///
/// The cheapest reliable stationary-camera detector: when a KITTI vehicle is
/// stopped at a light, flow collapses to sub-pixel noise and any recovered
/// translation direction is meaningless.
double median_flow_px(const Points2& points_a, const Points2& points_b);

/// Triangulate correspondences into the first camera's frame.
///
/// ``T_c1_c2`` is the relative pose returned by recover_relative_pose().
/// Returns (N, 3) points; those that fail to triangulate cleanly (points at
/// infinity) come back as NaN instead of enormous finite values.
Points3 triangulate_points(const Eigen::Matrix4d& T_c1_c2, const Points2& points_a,
                           const Points2& points_b, const Eigen::Matrix3d& K);

}  // namespace mvslam
