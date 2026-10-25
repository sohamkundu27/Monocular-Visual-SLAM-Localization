#include "mvslam/geometry/epipolar.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"

namespace mvslam {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const std::map<std::string, int>& ransac_methods() {
    static const std::map<std::string, int> methods = {
        {"lmeds", cv::LMEDS},
        {"magsac", cv::USAC_MAGSAC},
        {"ransac", cv::RANSAC},
        {"usac_accurate", cv::USAC_ACCURATE},
    };
    return methods;
}

std::string lower_strip(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    const auto begin = out.find_first_not_of(" \t\r\n\f\v");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = out.find_last_not_of(" \t\r\n\f\v");
    return out.substr(begin, end - begin + 1);
}

// ``[x, y, 1] @ M.T`` for one row, accumulated left to right like a dot.
Eigen::Vector3d row_times_transpose(double x, double y, const Eigen::Matrix3d& M) {
    return Eigen::Vector3d(x * M(0, 0) + y * M(0, 1) + M(0, 2), x * M(1, 0) + y * M(1, 1) + M(1, 2),
                           x * M(2, 0) + y * M(2, 1) + M(2, 2));
}

/// _to_rays: back-project pixels to unit-norm camera rays.
Points3 to_rays(const Points2& points, const Eigen::Matrix3d& K_inv) {
    Points3 rays(points.rows(), 3);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        const Eigen::Vector3d r = row_times_transpose(points(i, 0), points(i, 1), K_inv);
        const double n = std::sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(2));
        rays.row(i) = (r / n).transpose();
    }
    return rays;
}

/// _robust_score: sum of (_CHI2_H - chi2) over inliers; outliers contribute nothing.
double robust_score(const std::vector<double>& chi2, double threshold) {
    std::vector<double> contributions;
    for (double c : chi2) {
        if (c < threshold) {
            contributions.push_back(kChi2H - c);
        }
    }
    return np::sum(contributions);
}

/// _apply_homography, with NaN/inf mapped to +-1e9 like np.nan_to_num.
Points2 apply_homography(const Eigen::Matrix3d& H, const Points2& points) {
    Points2 out(points.rows(), 2);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        const Eigen::Vector3d h = row_times_transpose(points(i, 0), points(i, 1), H);
        for (int k = 0; k < 2; ++k) {
            double v = h(k) / h(2);
            if (std::isnan(v)) {
                v = 1e9;
            } else if (std::isinf(v)) {
                v = v > 0 ? 1e9 : -1e9;
            }
            out(i, k) = v;
        }
    }
    return out;
}

double score_homography(const Eigen::Matrix3d& H, const Points2& points_a, const Points2& points_b,
                        double sigma_sq) {
    const Points2 forward = apply_homography(H, points_a);
    const Points2 backward = apply_homography(H.inverse(), points_b);
    std::vector<double> chi2_forward(static_cast<std::size_t>(points_a.rows()));
    std::vector<double> chi2_backward(static_cast<std::size_t>(points_a.rows()));
    for (Eigen::Index i = 0; i < points_a.rows(); ++i) {
        const double fx = points_b(i, 0) - forward(i, 0);
        const double fy = points_b(i, 1) - forward(i, 1);
        const double bx = points_a(i, 0) - backward(i, 0);
        const double by = points_a(i, 1) - backward(i, 1);
        chi2_forward[static_cast<std::size_t>(i)] = (fx * fx + fy * fy) / sigma_sq;
        chi2_backward[static_cast<std::size_t>(i)] = (bx * bx + by * by) / sigma_sq;
    }
    return robust_score(chi2_forward, kChi2H) + robust_score(chi2_backward, kChi2H);
}

double score_fundamental(const Eigen::Matrix3d& F, const Points2& points_a, const Points2& points_b,
                         double sigma_sq) {
    const auto n = static_cast<std::size_t>(points_a.rows());
    std::vector<double> chi2_a(n);
    std::vector<double> chi2_b(n);
    const Eigen::Matrix3d Ft = F.transpose();
    for (std::size_t k = 0; k < n; ++k) {
        const auto i = static_cast<Eigen::Index>(k);
        const Eigen::Vector3d lines_b = row_times_transpose(points_a(i, 0), points_a(i, 1), F);
        const Eigen::Vector3d lines_a = row_times_transpose(points_b(i, 0), points_b(i, 1), Ft);
        const double residual = points_b(i, 0) * lines_b(0) + points_b(i, 1) * lines_b(1) + lines_b(2);
        const double rr = residual * residual;
        double cb = rr / ((lines_b(0) * lines_b(0) + lines_b(1) * lines_b(1)) * sigma_sq);
        double ca = rr / ((lines_a(0) * lines_a(0) + lines_a(1) * lines_a(1)) * sigma_sq);
        // np.nan_to_num(nan=inf, posinf=inf)
        if (std::isnan(cb) || (std::isinf(cb) && cb > 0)) {
            cb = std::numeric_limits<double>::infinity();
        }
        if (std::isnan(ca) || (std::isinf(ca) && ca > 0)) {
            ca = std::numeric_limits<double>::infinity();
        }
        chi2_a[k] = ca;
        chi2_b[k] = cb;
    }
    // Both models are scored against the same ceiling so the totals compare.
    return robust_score(chi2_a, kChi2E) + robust_score(chi2_b, kChi2E);
}

cv::Mat transposed_points(const Points2& points) {
    // (2, N): the layout of ``np.ascontiguousarray(points).T`` handed to cv2.
    cv::Mat out(2, static_cast<int>(points.rows()), CV_64F);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        out.at<double>(0, static_cast<int>(i)) = points(i, 0);
        out.at<double>(1, static_cast<int>(i)) = points(i, 1);
    }
    return out;
}

}  // namespace

std::string EssentialMatrixResult::repr() const {
    const std::string status = ok ? "ok" : "rejected(" + reason + ")";
    return "EssentialMatrixResult(" + status + ", inliers=" + std::to_string(n_inliers()) + "/" +
           std::to_string(n_points) + ")";
}

std::string RelativePoseResult::repr() const {
    const std::string status = ok ? "ok" : "rejected(" + reason + ")";
    return "RelativePoseResult(" + status + ", inliers=" + std::to_string(n_inliers()) +
           ", rot=" + py::format(rotation_deg, ".2f") + " deg, parallax=" +
           py::format(median_parallax_deg, ".3f") + " deg)";
}

int resolve_ransac_method(const std::string& name) {
    const auto& methods = ransac_methods();
    auto it = methods.find(lower_strip(name));
    if (it == methods.end()) {
        throw ValueError("Unknown ransac_method '" + name +
                         "'. Choose from ['lmeds', 'magsac', 'ransac', 'usac_accurate']");
    }
    return it->second;
}

EssentialMatrixResult estimate_essential_matrix(const Points2& points_a, const Points2& points_b,
                                                const Eigen::Matrix3d& K, double threshold_px,
                                                double confidence, int max_iters, int min_points,
                                                const std::string& method) {
    const auto n = static_cast<std::int64_t>(points_a.rows());
    if (points_a.rows() != points_b.rows()) {
        throw ValueError("points must be matching (N, 2) arrays, got (" +
                         std::to_string(points_a.rows()) + ", 2) and (" +
                         std::to_string(points_b.rows()) + ", 2)");
    }
    const Mask empty_mask(static_cast<std::size_t>(n), 0);
    if (n < min_points) {
        return {std::nullopt, empty_mask, n, false, "too_few_points(" + std::to_string(n) + ")"};
    }
    const int flag = resolve_ransac_method(method);
    cv::Mat mask;
    const cv::Mat E = cv::findEssentialMat(as_cv_view(points_a), as_cv_view(points_b), to_cv(K), flag,
                                           confidence, threshold_px, max_iters, mask);
    if (E.empty()) {
        return {std::nullopt, empty_mask, n, false, "no_solution"};
    }
    // The 5-point solver can return several stacked candidates (3k x 3) when
    // the configuration is degenerate. OpenCV orders them best-first.
    cv::Mat E3 = E.rows > 3 ? E.rowRange(0, 3) : E;
    if (E3.rows != 3 || E3.cols != 3) {
        return {std::nullopt, empty_mask, n, false,
                "bad_shape(" + std::to_string(E3.rows) + ", " + std::to_string(E3.cols) + ")"};
    }
    const Eigen::Matrix3d E_eigen = to_matrix3d(E3);
    if (!E_eigen.allFinite()) {
        return {std::nullopt, empty_mask, n, false, "non_finite"};
    }
    Mask inlier_mask = mask.empty() ? empty_mask : to_mask(mask, static_cast<std::size_t>(n));
    if (!mask.empty() && static_cast<std::int64_t>(mask.total()) != n) {
        inlier_mask = empty_mask;  // OpenCV always matches lengths
    }
    return {E_eigen, inlier_mask, n, true, ""};
}

RelativePoseResult recover_relative_pose(const std::optional<Eigen::Matrix3d>& E,
                                         const Points2& points_a, const Points2& points_b,
                                         const Eigen::Matrix3d& K,
                                         const std::optional<Mask>& inlier_mask,
                                         const PoseGates& gates) {
    const auto n = static_cast<std::size_t>(points_a.rows());
    auto failure = [n](const std::string& reason) {
        RelativePoseResult r;
        r.inlier_mask = Mask(n, 0);
        r.n_cheirality_inliers = 0;
        r.median_parallax_deg = kNaN;
        r.rotation_deg = kNaN;
        r.ok = false;
        r.reason = reason;
        return r;
    };
    if (!E) {
        return failure("no_essential_matrix");
    }

    // recoverPose mutates the mask in place, turning epipolar inliers into
    // cheirality inliers, so a writable copy is required.
    cv::Mat pose_mask(static_cast<int>(n), 1, CV_8U);
    std::int64_t n_epipolar = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t v = inlier_mask ? static_cast<std::uint8_t>((*inlier_mask)[i] ? 1 : 0) : 1;
        pose_mask.at<std::uint8_t>(static_cast<int>(i)) = v;
        n_epipolar += v ? 1 : 0;
    }

    // The overload with distanceThresh also returns the triangulated points,
    // which are not needed here (requested anyway, as the cv2 call does).
    cv::Mat R_cv;
    cv::Mat t_cv;
    cv::Mat points4d;
    cv::recoverPose(to_cv(*E), as_cv_view(points_a), as_cv_view(points_b), to_cv(K), R_cv, t_cv,
                    gates.cheirality_distance, pose_mask, points4d);
    const Mask cheirality_mask = to_mask(pose_mask, n);

    if (R_cv.empty() || t_cv.empty()) {
        return failure("non_finite_pose");
    }
    Eigen::Matrix3d R = to_matrix3d(R_cv);
    const Eigen::Vector3d t = to_vector3d(t_cv);
    if (!R.allFinite() || !t.allFinite()) {
        return failure("non_finite_pose");
    }
    if (!is_rotation_matrix(R, 1e-4)) {
        return failure("invalid_rotation");
    }
    // Snap tiny numerical drift away so the pose is an exact SO(3) member.
    R = project_to_so3(R);

    const double t_norm = std::sqrt(t(0) * t(0) + t(1) * t(1) + t(2) * t(2));
    if (t_norm < 1e-9) {
        return failure("degenerate_translation");
    }
    const Eigen::Vector3d t_unit = t / t_norm;

    const auto n_cheirality = static_cast<std::int64_t>(count_nonzero(cheirality_mask));
    // Measured against the epipolar inliers, not all matches: this asks "how
    // much of the geometrically consistent set survives cheirality", which is
    // the question that indicates a bad decomposition. Dividing by all
    // matches would instead conflate it with descriptor-matching quality.
    const double ratio =
        n_epipolar ? static_cast<double>(n_cheirality) / static_cast<double>(n_epipolar) : 0.0;
    const double rotation_deg = rotation_angle_deg(R);

    // T_c2_c1 = [R|t]; the forward camera motion is its inverse.
    const Eigen::Matrix3d Rt = R.transpose();
    const Eigen::Matrix4d T_c1_c2 = se3_from_rt(Rt, -Rt * t_unit);
    const double parallax = median_parallax_deg(points_a, points_b, K,
                                                Eigen::Matrix3d(T_c1_c2.topLeftCorner<3, 3>()),
                                                cheirality_mask);

    RelativePoseResult result;
    result.T = T_c1_c2;
    result.R = R;
    result.t_unit = t_unit;
    result.inlier_mask = cheirality_mask;
    result.n_cheirality_inliers = n_cheirality;
    result.median_parallax_deg = parallax;
    result.rotation_deg = rotation_deg;
    result.ok = true;

    if (n_cheirality < gates.min_inliers) {
        result.ok = false;
        result.reason = "too_few_inliers(" + std::to_string(n_cheirality) + ")";
    } else if (ratio < gates.min_inlier_ratio) {
        result.ok = false;
        result.reason = "low_inlier_ratio(" + py::format(ratio, ".2f") + ")";
    } else if (rotation_deg > gates.max_rotation_deg) {
        result.ok = false;
        result.reason = "implausible_rotation(" + py::format(rotation_deg, ".1f") + "deg)";
    } else if (gates.min_parallax_deg > 0.0 && parallax < gates.min_parallax_deg) {
        result.ok = false;
        result.reason = "low_parallax(" + py::format(parallax, ".3f") + "deg)";
    }
    return result;
}

double median_parallax_deg(const Points2& points_a_in, const Points2& points_b_in,
                           const Eigen::Matrix3d& K, const std::optional<Eigen::Matrix3d>& R_c1_c2,
                           const std::optional<Mask>& mask) {
    const Points2 points_a = mask ? select_rows(points_a_in, *mask) : points_a_in;
    const Points2 points_b = mask ? select_rows(points_b_in, *mask) : points_b_in;
    if (points_a.rows() == 0) {
        return kNaN;
    }
    const Eigen::Matrix3d K_inv = K.inverse();
    const Points3 rays_a = to_rays(points_a, K_inv);
    Points3 rays_b = to_rays(points_b, K_inv);
    if (R_c1_c2) {
        const Eigen::Matrix3d& R = *R_c1_c2;
        for (Eigen::Index i = 0; i < rays_b.rows(); ++i) {
            const double x = rays_b(i, 0);
            const double y = rays_b(i, 1);
            const double z = rays_b(i, 2);
            rays_b(i, 0) = x * R(0, 0) + y * R(0, 1) + z * R(0, 2);
            rays_b(i, 1) = x * R(1, 0) + y * R(1, 1) + z * R(1, 2);
            rays_b(i, 2) = x * R(2, 0) + y * R(2, 1) + z * R(2, 2);
        }
    }
    std::vector<double> angles(static_cast<std::size_t>(rays_a.rows()));
    for (Eigen::Index i = 0; i < rays_a.rows(); ++i) {
        double c = rays_a(i, 0) * rays_b(i, 0) + rays_a(i, 1) * rays_b(i, 1) + rays_a(i, 2) * rays_b(i, 2);
        c = std::min(1.0, std::max(-1.0, c));
        angles[static_cast<std::size_t>(i)] = std::acos(c);
    }
    return np::degrees(np::median(angles));
}

ModelSelection select_two_view_model(const Points2& points_a, const Points2& points_b,
                                     const Eigen::Matrix3d& K,
                                     const std::optional<Eigen::Matrix3d>& E, double threshold_px,
                                     double ratio_threshold) {
    if (points_a.rows() < 4 || !E) {
        return {};
    }
    const double sigma_sq = threshold_px * threshold_px;

    cv::Mat inliers;
    const cv::Mat H_cv =
        cv::findHomography(as_cv_view(points_a), as_cv_view(points_b), cv::RANSAC, threshold_px, inliers);
    std::optional<Eigen::Matrix3d> H;
    double score_h = 0.0;
    if (!H_cv.empty()) {
        H = to_matrix3d(H_cv);
        score_h = score_homography(*H, points_a, points_b, sigma_sq);
    }

    const Eigen::Matrix3d K_inv = K.inverse();
    const Eigen::Matrix3d F = K_inv.transpose() * (*E) * K_inv;
    const double score_e = score_fundamental(F, points_a, points_b, sigma_sq);

    const double total = score_h + score_e;
    const double ratio_h = total > 0 ? score_h / total : 0.0;
    return {score_h, score_e, ratio_h, ratio_h > ratio_threshold, H};
}

double median_flow_px(const Points2& points_a, const Points2& points_b) {
    if (points_a.rows() == 0) {
        return kNaN;
    }
    std::vector<double> flow(static_cast<std::size_t>(points_a.rows()));
    for (Eigen::Index i = 0; i < points_a.rows(); ++i) {
        const double dx = points_b(i, 0) - points_a(i, 0);
        const double dy = points_b(i, 1) - points_a(i, 1);
        flow[static_cast<std::size_t>(i)] = std::sqrt(dx * dx + dy * dy);
    }
    return np::median(flow);
}

Points3 triangulate_points(const Eigen::Matrix4d& T_c1_c2, const Points2& points_a,
                           const Points2& points_b, const Eigen::Matrix3d& K) {
    Eigen::Matrix<double, 3, 4> eye34 = Eigen::Matrix<double, 3, 4>::Zero();
    eye34.leftCols<3>() = Eigen::Matrix3d::Identity();
    const Eigen::Matrix<double, 3, 4> P1 = K * eye34;
    // Projection of camera 2 needs world-to-camera, i.e. the inverse of T_c1_c2.
    const Eigen::Matrix3d Rt = T_c1_c2.topLeftCorner<3, 3>().transpose();
    const Eigen::Vector3d t = T_c1_c2.topRightCorner<3, 1>();
    Eigen::Matrix<double, 3, 4> Rt_t;
    Rt_t.leftCols<3>() = Rt;
    Rt_t.col(3) = -Rt * t;
    const Eigen::Matrix<double, 3, 4> P2 = K * Rt_t;

    cv::Mat points4d;
    cv::triangulatePoints(to_cv(P1), to_cv(P2), transposed_points(points_a), transposed_points(points_b),
                          points4d);
    cv::Mat p4;
    points4d.convertTo(p4, CV_64F);
    Points3 out(points_a.rows(), 3);
    for (Eigen::Index i = 0; i < points_a.rows(); ++i) {
        const int c = static_cast<int>(i);
        const double w = p4.at<double>(3, c);
        if (std::fabs(w) < 1e-9) {
            out.row(i).setConstant(kNaN);
            continue;
        }
        out(i, 0) = p4.at<double>(0, c) / w;
        out(i, 1) = p4.at<double>(1, c) / w;
        out(i, 2) = p4.at<double>(2, c) / w;
    }
    return out;
}

}  // namespace mvslam
