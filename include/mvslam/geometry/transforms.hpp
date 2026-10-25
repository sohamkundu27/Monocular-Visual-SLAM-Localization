// SE(3) and SO(3) transform utilities.
//
// Conventions used throughout the project
// ---------------------------------------
// * A pose is a 4x4 homogeneous matrix ``T``.
// * ``T_w_c`` maps points from the camera frame into the world frame:
//   ``p_w = T_w_c * p_c``. This matches the KITTI Odometry ground-truth format,
//   where each row is the camera-to-world transform of the left camera.
// * Camera axes follow the OpenCV/KITTI convention: ``+x`` right, ``+y`` down,
//   ``+z`` forward. The ground plane is therefore x-z, which is why top-down
//   plots use X against Z.
// * ``recoverPose`` returns ``(R, t)`` describing the *world-to-camera* motion
//   of the second view relative to the first, i.e. ``T_c1_c2 = inv([R|t])``.
//   That inversion is handled once, in odometry/visual_odometry.
#pragma once

#include <array>
#include <string>

#include <Eigen/Core>

namespace mvslam {

using Vector6d = Eigen::Matrix<double, 6, 1>;

/// Numerical tolerance for orthonormality / determinant checks.
constexpr double kSO3Tolerance = 1e-6;

/// Assemble a 4x4 SE(3) matrix from a rotation and a translation.
Eigen::Matrix4d se3_from_rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

/// The rotation block of a pose (a copy).
Eigen::Matrix3d rotation_of(const Eigen::Matrix4d& T);
/// The translation column of a pose (a copy).
Eigen::Vector3d translation_of(const Eigen::Matrix4d& T);

/// Invert an SE(3) matrix analytically (``R^T``, ``-R^T t``).
///
/// Cheaper and numerically better behaved than a general 4x4 inverse because
/// it exploits orthonormality instead of solving a general system.
Eigen::Matrix4d invert_se3(const Eigen::Matrix4d& T);

/// Left-to-right composition: ``compose({A, B, C}) == A * B * C``. With no
/// arguments it returns the identity.
Eigen::Matrix4d compose(std::initializer_list<Eigen::Matrix4d> transforms);

/// ``T_a_b``: the pose of frame ``b`` expressed in frame ``a``.
Eigen::Matrix4d relative_pose(const Eigen::Matrix4d& T_w_a, const Eigen::Matrix4d& T_w_b);

/// Apply an SE(3) transform to (N, 3) row-major points.
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> transform_points(
    const Eigen::Matrix4d& T, const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points);

// -- Validation and projection --------------------------------------------- //

/// Check that ``R`` is orthonormal with ``det(R) == +1``.
///
/// Rejects reflections (``det == -1``), which is exactly the failure mode of a
/// badly conditioned essential-matrix decomposition. Orthonormality is tested
/// like ``np.allclose(R.T @ R, I, atol=tol)`` (rtol 1e-5 included).
bool is_rotation_matrix(const Eigen::Matrix3d& R, double tol = kSO3Tolerance);

/// Check finiteness, rotation validity and the bottom row.
bool is_valid_se3(const Eigen::Matrix4d& T, double tol = kSO3Tolerance);

/// The closest rotation matrix to ``R`` in the Frobenius sense.
///
/// Standard orthogonal Procrustes: ``R = U V^T`` from the SVD, with the sign
/// of the last singular vector flipped if that would otherwise produce a
/// reflection. Used to clean up drift-accumulated rotations before handing
/// them to GTSAM, which requires exact SO(3) members.
Eigen::Matrix3d project_to_so3(const Eigen::Matrix3d& R);

/// Re-orthonormalise the rotation block of a pose and fix its bottom row.
Eigen::Matrix4d project_to_se3(const Eigen::Matrix4d& T);

// -- Angles, logs and exponentials ----------------------------------------- //

/// Geodesic rotation angle of ``R`` in radians, in ``[0, pi]``.
///
/// Uses ``atan2(sin(theta), cos(theta))`` rather than ``acos`` of the trace.
/// The acos form is catastrophically ill-conditioned near identity: its
/// derivative is unbounded there, so the ~1e-16 rounding error in the trace
/// inflates to ~1e-8 rad. That matters directly, because relative pose error
/// between two nearly identical trajectories is exactly the near-identity
/// regime, and the acos form puts a spurious ~1e-5 degree floor under it.
///
/// ``sin(theta)`` comes from the antisymmetric part of ``R`` and
/// ``cos(theta)`` from its trace; ``atan2`` of the pair is exact at zero and
/// well behaved across the full range.
double rotation_angle_rad(const Eigen::Matrix3d& R);
/// Overload taking the rotation block of a pose.
double rotation_angle_rad(const Eigen::Matrix4d& T);

/// Geodesic rotation angle in degrees.
double rotation_angle_deg(const Eigen::Matrix3d& R);
double rotation_angle_deg(const Eigen::Matrix4d& T);

/// Logarithm map SO(3) -> R^3 (rotation vector, axis * angle).
Eigen::Vector3d so3_log(const Eigen::Matrix3d& R);

/// Exponential map R^3 -> SO(3).
Eigen::Matrix3d so3_exp(const Eigen::Vector3d& rotvec);

/// Logarithm map SE(3) -> R^6 ordered ``[rho (3), phi (3)]``.
///
/// ``phi`` is the rotation vector and ``rho`` the translation coordinate in
/// the Lie algebra, i.e. ``t = V(phi) * rho``. Provided for completeness and
/// for uncertainty-weighted comparisons; the pipeline mostly needs separate
/// translation and rotation errors, which are easier to interpret.
Vector6d se3_log(const Eigen::Matrix4d& T);

/// Exponential map R^6 -> SE(3), inverse of se3_log().
Eigen::Matrix4d se3_exp(const Vector6d& xi);

/// Skew-symmetric matrix such that ``skew(v) * w == v.cross(w)``.
Eigen::Matrix3d skew(const Eigen::Vector3d& v);

// -- KITTI serialisation --------------------------------------------------- //

/// Flatten a pose to KITTI's 12-value row-major ``[R|t]`` layout.
std::array<double, 12> se3_to_kitti_row(const Eigen::Matrix4d& T);

/// Expand a KITTI 12-value pose row into a 4x4 matrix.
Eigen::Matrix4d kitti_row_to_se3(const std::array<double, 12>& row);

// -- scipy.spatial.transform.Rotation equivalents -------------------------- //
//
// so3_log/so3_exp go through scipy in the Python original, so the same
// quaternion-based algorithms are reproduced here (Shepperd-style matrix ->
// quaternion, Taylor branches below 1e-3 rad), rather than a closed-form
// Rodrigues map that would differ in the last bits.
namespace rotation {

/// ``Rotation.from_matrix(R).as_quat()`` (x, y, z, w), R assumed orthogonal.
Eigen::Vector4d quat_from_matrix(const Eigen::Matrix3d& R);
/// ``Rotation.from_quat(q).as_matrix()``.
Eigen::Matrix3d matrix_from_quat(const Eigen::Vector4d& q);
/// ``Rotation.from_rotvec(v).as_matrix()``.
Eigen::Matrix3d from_rotvec(const Eigen::Vector3d& rotvec);
/// ``Rotation.from_matrix(R).as_rotvec()``.
Eigen::Vector3d as_rotvec(const Eigen::Matrix3d& R);
/// ``Rotation.from_euler(seq, angles, degrees=degrees).as_matrix()`` for 1-3
/// extrinsic (lower-case) or intrinsic (upper-case) axes.
Eigen::Matrix3d from_euler(const std::string& seq, const std::vector<double>& angles,
                           bool degrees = true);
Eigen::Matrix3d from_euler(const std::string& seq, double angle, bool degrees = true);

}  // namespace rotation

}  // namespace mvslam
