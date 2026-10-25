#include "mvslam/features/matcher.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

#include <opencv2/imgproc.hpp>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.features.matcher");
    return instance;
}

Points2 gather(const Points2& points, const std::vector<std::int64_t>& indices) {
    Points2 out(static_cast<Eigen::Index>(indices.size()), 2);
    for (std::size_t k = 0; k < indices.size(); ++k) {
        out.row(static_cast<Eigen::Index>(k)) = points.row(static_cast<Eigen::Index>(indices[k]));
    }
    return out;
}

template <typename T>
std::vector<T> keep(const std::vector<T>& values, const Mask& mask) {
    std::vector<T> out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (mask[i]) {
            out.push_back(values[i]);
        }
    }
    return out;
}

}  // namespace

// --------------------------------------------------------------------------- //
// MatchResult
// --------------------------------------------------------------------------- //

double MatchResult::survival_ratio() const {
    return n_raw ? static_cast<double>(size()) / static_cast<double>(n_raw) : 0.0;
}

double MatchResult::mean_distance() const {
    return size() ? np::mean(distances) : std::numeric_limits<double>::quiet_NaN();
}

double MatchResult::mean_pixel_displacement() const {
    if (size() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> norms(size());
    for (std::size_t i = 0; i < size(); ++i) {
        const double dx = points_b(static_cast<Eigen::Index>(i), 0) - points_a(static_cast<Eigen::Index>(i), 0);
        const double dy = points_b(static_cast<Eigen::Index>(i), 1) - points_a(static_cast<Eigen::Index>(i), 1);
        norms[i] = std::sqrt(dx * dx + dy * dy);
    }
    return np::mean(norms);
}

MatchResult MatchResult::select(const Mask& mask) const {
    if (mask.size() != size()) {
        throw ValueError("mask length " + std::to_string(mask.size()) + " != match count " +
                         std::to_string(size()));
    }
    std::vector<std::int64_t> indices;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            indices.push_back(static_cast<std::int64_t>(i));
        }
    }
    return select_indices(indices);
}

MatchResult MatchResult::select_indices(const std::vector<std::int64_t>& indices) const {
    MatchResult out;
    out.points_a = gather(points_a, indices);
    out.points_b = gather(points_b, indices);
    for (std::int64_t i : indices) {
        const auto k = static_cast<std::size_t>(i);
        out.indices_a.push_back(indices_a.at(k));
        out.indices_b.push_back(indices_b.at(k));
        out.distances.push_back(distances.at(k));
    }
    out.n_raw = n_raw;
    return out;
}

MatchResult MatchResult::empty(std::int64_t n_raw) {
    MatchResult out;
    out.points_a.resize(0, 2);
    out.points_b.resize(0, 2);
    out.n_raw = n_raw;
    return out;
}

std::string MatchResult::repr() const {
    return "MatchResult(matches=" + std::to_string(size()) + ", raw=" + std::to_string(n_raw) +
           ", mean_dist=" + py::format(mean_distance(), ".1f") + ")";
}

// --------------------------------------------------------------------------- //
// FeatureMatcher
// --------------------------------------------------------------------------- //

FeatureMatcher::FeatureMatcher(int norm, bool use_ratio_test, double ratio, bool cross_check,
                               std::optional<double> max_distance, std::int64_t min_matches)
    : norm_(norm),
      use_ratio_test_(use_ratio_test),
      ratio_(ratio),
      cross_check_(cross_check),
      max_distance_(max_distance),
      min_matches_(min_matches) {
    if (!(0.0 < ratio && ratio <= 1.0)) {
        throw ValueError("ratio must be in (0, 1], got " + py::repr(ratio));
    }
    matcher_ = cv::BFMatcher::create(norm, false);
    logger().info(py::sprintf("Feature matcher: norm=%s, ratio_test=%s (%.2f), cross_check=%s, max_distance=%s",
                              norm == cv::NORM_HAMMING ? "HAMMING" : "L2",
                              py::repr(use_ratio_test_).c_str(), ratio_,
                              py::repr(cross_check_).c_str(),
                              max_distance_ ? py::repr(*max_distance_).c_str() : "None"));
}

FeatureMatcher FeatureMatcher::from_config(const Config& config, int norm) {
    const auto& m = config.matcher;
    std::string backend = m.matcher;
    std::transform(backend.begin(), backend.end(), backend.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (backend != "bf") {
        throw ValueError("Unsupported matcher backend '" + m.matcher + "'; only 'bf' is available");
    }
    return FeatureMatcher(norm, m.use_ratio_test, m.ratio, m.cross_check, m.max_distance,
                          m.min_matches);
}

MatchResult FeatureMatcher::match(const Frame& frame_a, const Frame& frame_b) const {
    if (!frame_a.is_usable() || !frame_b.is_usable()) {
        return MatchResult::empty();
    }
    return match_descriptors(frame_a.descriptors, frame_b.descriptors, frame_a.points, frame_b.points);
}

MatchResult FeatureMatcher::match_descriptors(const cv::Mat& desc_a_in, const cv::Mat& desc_b_in,
                                              const Points2& points_a,
                                              const Points2& points_b) const {
    const cv::Mat desc_a = as_matcher_dtype(desc_a_in, norm_);
    const cv::Mat desc_b = as_matcher_dtype(desc_b_in, norm_);
    if (desc_a.rows == 0 || desc_b.rows == 0) {
        return MatchResult::empty();
    }

    Forward forward = forward_matches(desc_a, desc_b);

    if (!forward.idx_a.empty() && cross_check_) {
        const Mask mask = cross_check_mask(desc_a, desc_b, forward.idx_a, forward.idx_b);
        forward.idx_a = keep(forward.idx_a, mask);
        forward.idx_b = keep(forward.idx_b, mask);
        forward.dist = keep(forward.dist, mask);
    }

    if (!forward.idx_a.empty() && max_distance_) {
        Mask mask(forward.dist.size());
        for (std::size_t i = 0; i < forward.dist.size(); ++i) {
            mask[i] = forward.dist[i] <= *max_distance_ ? 1 : 0;
        }
        forward.idx_a = keep(forward.idx_a, mask);
        forward.idx_b = keep(forward.idx_b, mask);
        forward.dist = keep(forward.dist, mask);
    }

    MatchResult result;
    result.points_a = gather(points_a, forward.idx_a);
    result.points_b = gather(points_b, forward.idx_b);
    result.indices_a = std::move(forward.idx_a);
    result.indices_b = std::move(forward.idx_b);
    result.distances = std::move(forward.dist);
    result.n_raw = forward.n_raw;
    return result;
}

FeatureMatcher::Forward FeatureMatcher::forward_matches(const cv::Mat& desc_a,
                                                        const cv::Mat& desc_b) const {
    Forward out;
    if (use_ratio_test_ && desc_b.rows >= 2) {
        std::vector<std::vector<cv::DMatch>> knn;
        matcher_->knnMatch(desc_a, desc_b, knn, 2);
        out.n_raw = static_cast<std::int64_t>(knn.size());
        for (const auto& pair : knn) {
            if (pair.size() < 2) {
                // Only one neighbour exists, so the ratio test cannot be
                // evaluated; accept it and let later filters decide.
                if (!pair.empty()) {
                    out.idx_a.push_back(pair[0].queryIdx);
                    out.idx_b.push_back(pair[0].trainIdx);
                    out.dist.push_back(static_cast<double>(pair[0].distance));
                }
                continue;
            }
            const cv::DMatch& best = pair[0];
            const cv::DMatch& second = pair[1];
            // DMatch distances are float32; compare in double like Python does
            // after converting them to Python floats.
            const double d1 = static_cast<double>(best.distance);
            const double d2 = static_cast<double>(second.distance);
            if (d2 > 0 && d1 < ratio_ * d2) {
                out.idx_a.push_back(best.queryIdx);
                out.idx_b.push_back(best.trainIdx);
                out.dist.push_back(d1);
            }
        }
        return out;
    }
    std::vector<cv::DMatch> matches;
    matcher_->match(desc_a, desc_b, matches);
    out.n_raw = static_cast<std::int64_t>(matches.size());
    for (const auto& m : matches) {
        out.idx_a.push_back(m.queryIdx);
        out.idx_b.push_back(m.trainIdx);
        out.dist.push_back(static_cast<double>(m.distance));
    }
    return out;
}

Mask FeatureMatcher::cross_check_mask(const cv::Mat& desc_a, const cv::Mat& desc_b,
                                      const std::vector<std::int64_t>& idx_a,
                                      const std::vector<std::int64_t>& idx_b) const {
    // The reverse search runs once over all of b and is then consulted as a
    // lookup table, which is far cheaper than re-querying per match.
    std::vector<cv::DMatch> reverse;
    matcher_->match(desc_b, desc_a, reverse);
    std::vector<std::int64_t> best_from_b(static_cast<std::size_t>(desc_b.rows), -1);
    for (const auto& m : reverse) {
        best_from_b[static_cast<std::size_t>(m.queryIdx)] = m.trainIdx;
    }
    Mask mask(idx_a.size());
    for (std::size_t i = 0; i < idx_a.size(); ++i) {
        mask[i] = best_from_b[static_cast<std::size_t>(idx_b[i])] == idx_a[i] ? 1 : 0;
    }
    return mask;
}

cv::Mat as_matcher_dtype(const cv::Mat& descriptors, int norm) {
    if (descriptors.empty()) {
        return descriptors;
    }
    const int wanted = norm == cv::NORM_HAMMING ? CV_8U : CV_32F;
    if (descriptors.depth() == wanted) {
        return descriptors;
    }
    cv::Mat out;
    descriptors.convertTo(out, wanted);
    return out;
}

cv::Mat draw_matches(const cv::Mat& image_a, const Frame& /*frame_a*/, const cv::Mat& image_b,
                     const Frame& /*frame_b*/, const MatchResult& result, int max_draw,
                     const std::optional<Mask>& inlier_mask) {
    cv::Mat canvas_a;
    cv::Mat canvas_b;
    if (image_a.channels() == 1) {
        cv::cvtColor(image_a, canvas_a, cv::COLOR_GRAY2BGR);
    } else {
        canvas_a = image_a.clone();
    }
    if (image_b.channels() == 1) {
        cv::cvtColor(image_b, canvas_b, cv::COLOR_GRAY2BGR);
    } else {
        canvas_b = image_b.clone();
    }
    const int height = std::max(canvas_a.rows, canvas_b.rows);
    const int width = canvas_a.cols + canvas_b.cols;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    canvas_a.copyTo(canvas(cv::Rect(0, 0, canvas_a.cols, canvas_a.rows)));
    canvas_b.copyTo(canvas(cv::Rect(canvas_a.cols, 0, canvas_b.cols, canvas_b.rows)));
    const double offset = canvas_a.cols;

    const std::size_t n = result.size();
    if (n == 0) {
        return canvas;
    }
    std::vector<std::int64_t> order = np::argsort(result.distances);
    order.resize(std::min(order.size(), static_cast<std::size_t>(std::max(0, max_draw))));

    for (std::int64_t raw : order) {
        const auto i = static_cast<Eigen::Index>(raw);
        // np.round is round-half-to-even, which std::nearbyint also uses.
        const cv::Point pa(static_cast<int>(std::nearbyint(result.points_a(i, 0))),
                           static_cast<int>(std::nearbyint(result.points_a(i, 1))));
        const cv::Point pb(static_cast<int>(std::nearbyint(result.points_b(i, 0) + offset)),
                           static_cast<int>(std::nearbyint(result.points_b(i, 1))));
        cv::Scalar color(0, 220, 0);
        if (inlier_mask && !(*inlier_mask)[static_cast<std::size_t>(i)]) {
            color = cv::Scalar(0, 0, 220);
        }
        cv::line(canvas, pa, pb, color, 1, cv::LINE_AA);
        cv::circle(canvas, pa, 3, color, 1, cv::LINE_AA);
        cv::circle(canvas, pb, 3, color, 1, cv::LINE_AA);
    }

    std::string label = std::to_string(n) + " matches";
    if (inlier_mask) {
        label += " | " + std::to_string(count_nonzero(*inlier_mask)) + " inliers";
    }
    cv::putText(canvas, label, cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return canvas;
}

}  // namespace mvslam
