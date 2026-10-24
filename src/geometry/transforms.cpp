#include "mvslam/geometry/transforms.hpp"

#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/errors.hpp"

namespace mvslam {

namespace {

// np.allclose(a, b, rtol=1e-5, atol=atol): |a - b| <= atol + rtol * |b|.
template <typename A, typename B>
bool allclose(const A& a, const B& b, double atol, double rtol = 1e-5) {
    for (Eigen::Index r = 0; r < a.rows(); ++r) {
        for (Eigen::Index c = 0; c < a.cols(); ++c) {
            const double x = a(r, c);
            const double y = b(r, c);
            if (std::isnan(x) || std::isnan(y) || !(std::fabs(x - y) <= atol + rtol * std::fabs(y))) {
                return false;
            }
        }
    }
    return true;
}

double norm3(const Eigen::Vector3d& v) { return std::sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2)); }

}  // namespace

Eigen::Matrix4d se3_from_rt(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = t;
    return T;
}

Eigen::Matrix3d rotation_of(const Eigen::Matrix4d& T) { return T.topLeftCorner<3, 3>(); }

Eigen::Vector3d translation_of(const Eigen::Matrix4d& T) { return T.topRightCorner<3, 1>(); }

Eigen::Matrix4d invert_se3(const Eigen::Matrix4d& T) {
    const Eigen::Matrix3d Rt = T.topLeftCorner<3, 3>().transpose();
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    out.topLeftCorner<3, 3>() = Rt;
    out.topRightCorner<3, 1>() = -Rt * T.topRightCorner<3, 1>();
    return out;
}

Eigen::Matrix4d compose(std::initializer_list<Eigen::Matrix4d> transforms) {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    for (const auto& T : transforms) {
        result = (result * T).eval();
    }
    return result;
}

Eigen::Matrix4d relative_pose(const Eigen::Matrix4d& T_w_a, const Eigen::Matrix4d& T_w_b) {
    return invert_se3(T_w_a) * T_w_b;
}

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> transform_points(
    const Eigen::Matrix4d& T,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points) {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::RowVector3d t = T.topRightCorner<3, 1>().transpose();
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> out = points * R.transpose();
    out.rowwise() += t;
    return out;
}

bool is_rotation_matrix(const Eigen::Matrix3d& R, double tol) {
    if (!R.allFinite()) {
        return false;
    }
    if (!allclose(R.transpose() * R, Eigen::Matrix3d::Identity(), tol)) {
        return false;
    }
    return std::fabs(R.determinant() - 1.0) <= std::max(tol, 1e-9);
}

bool is_valid_se3(const Eigen::Matrix4d& T, double tol) {
    if (!T.allFinite()) {
        return false;
    }
    if (!allclose(T.row(3), Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), 1e-8)) {
        return false;
    }
    return is_rotation_matrix(T.topLeftCorner<3, 3>(), tol);
}

Eigen::Matrix3d project_to_so3(const Eigen::Matrix3d& R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d Vt = svd.matrixV().transpose();
    Eigen::Matrix3d R_proj = U * Vt;
    if (R_proj.determinant() < 0) {
        U.col(2) *= -1.0;
        R_proj = U * Vt;
    }
    return R_proj;
}

Eigen::Matrix4d project_to_se3(const Eigen::Matrix4d& T) {
    Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
    out.topLeftCorner<3, 3>() = project_to_so3(T.topLeftCorner<3, 3>());
    out.topRightCorner<3, 1>() = T.topRightCorner<3, 1>();
    return out;
}

double rotation_angle_rad(const Eigen::Matrix3d& R) {
    const Eigen::Vector3d axis(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    const double sin_theta = norm3(axis) / 2.0;
    const double cos_theta = ((R(0, 0) + R(1, 1) + R(2, 2)) - 1.0) / 2.0;
    return std::atan2(sin_theta, cos_theta);
}

double rotation_angle_rad(const Eigen::Matrix4d& T) {
    return rotation_angle_rad(Eigen::Matrix3d(T.topLeftCorner<3, 3>()));
}

double rotation_angle_deg(const Eigen::Matrix3d& R) { return np::degrees(rotation_angle_rad(R)); }

double rotation_angle_deg(const Eigen::Matrix4d& T) { return np::degrees(rotation_angle_rad(T)); }

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) { return rotation::as_rotvec(project_to_so3(R)); }

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& rotvec) { return rotation::from_rotvec(rotvec); }

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
    return S;
}

Vector6d se3_log(const Eigen::Matrix4d& T) {
    const Eigen::Vector3d phi = so3_log(T.topLeftCorner<3, 3>());
    const Eigen::Vector3d t = T.topRightCorner<3, 1>();
    const double theta = norm3(phi);
    Eigen::Matrix3d V_inv;
    if (theta < 1e-12) {
        V_inv = Eigen::Matrix3d::Identity() - 0.5 * skew(phi);
    } else {
        const Eigen::Vector3d a = phi / theta;
        const Eigen::Matrix3d A = skew(a);
        const double half = theta / 2.0;
        const double cot_half = 1.0 / std::tan(half);
        V_inv = half * cot_half * Eigen::Matrix3d::Identity() +
                (1.0 - half * cot_half) * (a * a.transpose()) - half * A;
    }
    Vector6d out;
    out.head<3>() = V_inv * t;
    out.tail<3>() = phi;
    return out;
}

Eigen::Matrix4d se3_exp(const Vector6d& xi) {
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    const double theta = norm3(phi);
    const Eigen::Matrix3d R = so3_exp(phi);
    Eigen::Matrix3d V;
    if (theta < 1e-12) {
        V = Eigen::Matrix3d::Identity() + 0.5 * skew(phi);
    } else {
        const Eigen::Matrix3d A = skew(phi / theta);
        V = Eigen::Matrix3d::Identity() + ((1.0 - std::cos(theta)) / theta) * A +
            ((theta - std::sin(theta)) / theta) * (A * A);
    }
    return se3_from_rt(R, V * rho);
}

std::array<double, 12> se3_to_kitti_row(const Eigen::Matrix4d& T) {
    std::array<double, 12> row{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            row[static_cast<std::size_t>(r * 4 + c)] = T(r, c);
        }
    }
    return row;
}

Eigen::Matrix4d kitti_row_to_se3(const std::array<double, 12>& row) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            T(r, c) = row[static_cast<std::size_t>(r * 4 + c)];
        }
    }
    return T;
}

// --------------------------------------------------------------------------- //
// scipy.spatial.transform.Rotation (scipy 1.17 algorithms)
// --------------------------------------------------------------------------- //

namespace rotation {

namespace {

Eigen::Vector4d normalize(const Eigen::Vector4d& q) {
    const double n = std::sqrt(q(0) * q(0) + q(1) * q(1) + q(2) * q(2) + q(3) * q(3));
    return q / n;
}

// Rotation._compose_quat(p, q): the quaternion of applying q then p.
Eigen::Vector4d compose_quat(const Eigen::Vector4d& p, const Eigen::Vector4d& q) {
    const Eigen::Vector3d cross = p.head<3>().cross(q.head<3>());
    Eigen::Vector4d out;
    out(0) = p(3) * q(0) + q(3) * p(0) + cross(0);
    out(1) = p(3) * q(1) + q(3) * p(1) + cross(1);
    out(2) = p(3) * q(2) + q(3) * p(2) + cross(2);
    out(3) = p(3) * q(3) - p(0) * q(0) - p(1) * q(1) - p(2) * q(2);
    return out;
}

int axis_index(char axis) {
    switch (axis) {
        case 'x':
        case 'X':
            return 0;
        case 'y':
        case 'Y':
            return 1;
        case 'z':
        case 'Z':
            return 2;
        default:
            throw ValueError(std::string("Expected axis specification to be from 'xyz' or 'XYZ', got ") +
                             axis);
    }
}

}  // namespace

Eigen::Vector4d quat_from_matrix(const Eigen::Matrix3d& m) {
    const double trace = m(0, 0) + m(1, 1) + m(2, 2);
    const double decision[4] = {m(0, 0), m(1, 1), m(2, 2), trace};
    int choice = 0;
    for (int i = 1; i < 4; ++i) {
        if (decision[i] > decision[choice]) {
            choice = i;
        }
    }
    Eigen::Vector4d q;
    if (choice != 3) {
        const int i = choice;
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        q(i) = 1.0 - trace + 2.0 * m(i, i);
        q(j) = m(j, i) + m(i, j);
        q(k) = m(k, i) + m(i, k);
        q(3) = m(k, j) - m(j, k);
    } else {
        q(0) = m(2, 1) - m(1, 2);
        q(1) = m(0, 2) - m(2, 0);
        q(2) = m(1, 0) - m(0, 1);
        q(3) = 1.0 + trace;
    }
    return normalize(q);
}

Eigen::Matrix3d matrix_from_quat(const Eigen::Vector4d& q) {
    const double x = q(0);
    const double y = q(1);
    const double z = q(2);
    const double w = q(3);
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double w2 = w * w;
    const double xy = x * y;
    const double zw = z * w;
    const double xz = x * z;
    const double yw = y * w;
    const double yz = y * z;
    const double xw = x * w;
    Eigen::Matrix3d m;
    m << x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw),  //
        2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw),  //
        2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2;
    return m;
}

Eigen::Matrix3d from_rotvec(const Eigen::Vector3d& rotvec) {
    const double angle = std::sqrt(rotvec(0) * rotvec(0) + rotvec(1) * rotvec(1) + rotvec(2) * rotvec(2));
    double scale = 0.0;
    if (angle <= 1e-3) {
        const double angle2 = angle * angle;
        scale = 0.5 - angle2 / 48 + angle2 * angle2 / 3840;
    } else {
        scale = std::sin(angle / 2) / angle;
    }
    Eigen::Vector4d q;
    q << rotvec(0) * scale, rotvec(1) * scale, rotvec(2) * scale, std::cos(angle / 2);
    return matrix_from_quat(q);
}

Eigen::Vector3d as_rotvec(const Eigen::Matrix3d& R) {
    Eigen::Vector4d q = quat_from_matrix(R);
    // _quat_canonical: w >= 0, ties broken by the first nonzero component.
    bool flip = q(3) < 0;
    if (q(3) == 0) {
        for (int i = 0; i < 3; ++i) {
            if (q(i) != 0) {
                flip = q(i) < 0;
                break;
            }
        }
    }
    if (flip) {
        q = -q;
    }
    const double ax_norm = std::sqrt(q(0) * q(0) + q(1) * q(1) + q(2) * q(2));
    const double angle = 2 * std::atan2(ax_norm, q(3));
    double scale = 0.0;
    if (angle <= 1e-3) {
        const double angle2 = angle * angle;
        scale = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880;
    } else {
        scale = angle / std::sin(angle / 2.0);
    }
    return Eigen::Vector3d(scale * q(0), scale * q(1), scale * q(2));
}

Eigen::Matrix3d from_euler(const std::string& seq, const std::vector<double>& angles,
                           bool degrees) {
    if (seq.empty() || seq.size() > 3 || seq.size() != angles.size()) {
        throw ValueError("Expected axis specification to be a non-empty string of up to 3 "
                         "characters, got " + seq);
    }
    const bool intrinsic = std::isupper(static_cast<unsigned char>(seq[0])) != 0;
    std::vector<double> radians = angles;
    if (degrees) {
        for (double& a : radians) {
            a = np::radians(a);
        }
    }
    auto elementary = [](int axis, double angle) {
        Eigen::Vector4d q = Eigen::Vector4d::Zero();
        q(3) = std::cos(angle / 2.0);
        q(axis) = std::sin(angle / 2.0);
        return q;
    };
    Eigen::Vector4d q = elementary(axis_index(seq[0]), radians[0]);
    for (std::size_t i = 1; i < seq.size(); ++i) {
        const Eigen::Vector4d axis_quat = elementary(axis_index(seq[i]), radians[i]);
        q = intrinsic ? compose_quat(q, axis_quat) : compose_quat(axis_quat, q);
    }
    // Like scipy, the composed quaternion is used as is (not re-normalised).
    return matrix_from_quat(q);
}

Eigen::Matrix3d from_euler(const std::string& seq, double angle, bool degrees) {
    return from_euler(seq, std::vector<double>{angle}, degrees);
}

}  // namespace rotation

}  // namespace mvslam
