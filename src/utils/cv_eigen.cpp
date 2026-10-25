#include "mvslam/utils/cv_eigen.hpp"

#include <algorithm>

#include "mvslam/errors.hpp"

namespace mvslam {

cv::Mat as_cv_view(const Points2& points) {
    // Row-major (N, 2): element (i, j) lives at data[2 * i + j], exactly the
    // cv::Mat(N, 2, CV_64F) layout. const_cast is safe: OpenCV only reads.
    return cv::Mat(static_cast<int>(points.rows()), 2, CV_64F,
                   const_cast<double*>(points.data()));
}

cv::Mat to_cv(const Eigen::Matrix3d& m) {
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out.at<double>(r, c) = m(r, c);
        }
    }
    return out;
}

cv::Mat to_cv(const Eigen::Matrix<double, 3, 4>& m) {
    cv::Mat out(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.at<double>(r, c) = m(r, c);
        }
    }
    return out;
}

Eigen::Matrix3d to_matrix3d(const cv::Mat& m) {
    if (m.rows != 3 || m.cols != 3 || m.channels() != 1) {
        throw ValueError("expected a 3x3 single-channel matrix");
    }
    cv::Mat as_double;
    m.convertTo(as_double, CV_64F);
    Eigen::Matrix3d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = as_double.at<double>(r, c);
        }
    }
    return out;
}

Eigen::Vector3d to_vector3d(const cv::Mat& m) {
    if (m.total() != 3 || m.channels() != 1) {
        throw ValueError("expected a 3-element single-channel matrix");
    }
    cv::Mat as_double;
    m.convertTo(as_double, CV_64F);
    const cv::Mat flat = as_double.reshape(1, 3);
    return Eigen::Vector3d(flat.at<double>(0), flat.at<double>(1), flat.at<double>(2));
}

Mask to_mask(const cv::Mat& m, std::size_t n) {
    Mask out(n, 0);
    if (m.empty()) {
        return out;
    }
    cv::Mat bytes;
    m.convertTo(bytes, CV_8U);
    const cv::Mat flat = bytes.reshape(1, static_cast<int>(bytes.total()));
    const std::size_t count = std::min(n, static_cast<std::size_t>(flat.total()));
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = flat.at<std::uint8_t>(static_cast<int>(i)) != 0 ? 1 : 0;
    }
    return out;
}

std::size_t count_nonzero(const Mask& mask) {
    return static_cast<std::size_t>(std::count_if(mask.begin(), mask.end(),
                                                  [](std::uint8_t v) { return v != 0; }));
}

Points2 select_rows(const Points2& points, const Mask& mask) {
    Points2 out(static_cast<Eigen::Index>(count_nonzero(mask)), 2);
    Eigen::Index k = 0;
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        if (mask[static_cast<std::size_t>(i)]) {
            out.row(k++) = points.row(i);
        }
    }
    return out;
}

}  // namespace mvslam
