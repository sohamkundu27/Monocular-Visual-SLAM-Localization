// Descriptor matching and correspondence filtering.
//
// The filtering cascade is deliberately explicit, because match quality is
// what determines whether the essential-matrix estimate is usable:
//
// 1. **k-NN search** (k=2) with a brute-force matcher. Exact rather than
//    approximate: at 3000 descriptors per frame the exhaustive search is not
//    the bottleneck, and approximate search adds a second source of error.
// 2. **Lowe's ratio test** -- keep a match only when the best candidate is
//    clearly better than the runner-up (``d1 < ratio * d2``). This is the
//    single most effective filter against repetitive structure such as road
//    markings, railings and building facades, all of which are everywhere in
//    KITTI.
// 3. **Mutual consistency (cross-check)** -- keep a match only when the two
//    descriptors are each other's nearest neighbour. Ratio-test and
//    cross-check are combined manually rather than via
//    ``BFMatcher(crossCheck=true)``, because OpenCV forbids crossCheck with
//    knnMatch.
// 4. **Absolute distance ceiling** -- reject matches that survive the
//    relative tests but are simply poor in absolute terms.
//
// Everything downstream consumes the resulting MatchResult, which carries the
// aligned point arrays rather than raw DMatch objects.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "mvslam/features/detector.hpp"
#include "mvslam/utils/cv_eigen.hpp"

namespace mvslam {

struct Config;

/// Filtered correspondences between two frames.
struct MatchResult {
    /// (M, 2) pixel coordinates of matched keypoints, index-aligned.
    Points2 points_a;
    Points2 points_b;
    /// (M) keypoint indices into the source frames, so a caller can trace a
    /// correspondence back to its descriptor.
    std::vector<std::int64_t> indices_a;
    std::vector<std::int64_t> indices_b;
    /// (M) descriptor distances of the accepted matches.
    std::vector<double> distances;
    /// Matches before filtering, for diagnostics.
    std::int64_t n_raw = 0;

    std::size_t size() const { return indices_a.size(); }

    /// Fraction of raw matches that survived filtering.
    double survival_ratio() const;
    double mean_distance() const;
    /// Average optical-flow magnitude, a cheap sanity check on motion.
    double mean_pixel_displacement() const;

    /// The subset selected by a boolean mask (ValueError on a length mismatch).
    MatchResult select(const Mask& mask) const;
    /// The subset selected by an index array.
    MatchResult select_indices(const std::vector<std::int64_t>& indices) const;

    static MatchResult empty(std::int64_t n_raw = 0);

    std::string repr() const;
};

/// Brute-force descriptor matcher with a configurable filter cascade.
class FeatureMatcher {
public:
    explicit FeatureMatcher(int norm = cv::NORM_HAMMING, bool use_ratio_test = true,
                            double ratio = 0.75, bool cross_check = true,
                            std::optional<double> max_distance = 64.0,
                            std::int64_t min_matches = 20);

    static FeatureMatcher from_config(const Config& config, int norm = cv::NORM_HAMMING);

    /// Match ``frame_a`` against ``frame_b`` and apply the filter cascade.
    MatchResult match(const Frame& frame_a, const Frame& frame_b) const;

    /// Core matching routine working directly on descriptor arrays.
    MatchResult match_descriptors(const cv::Mat& desc_a, const cv::Mat& desc_b,
                                  const Points2& points_a, const Points2& points_b) const;

    /// True when enough matches survived to attempt pose estimation.
    bool is_sufficient(const MatchResult& result) const {
        return static_cast<std::int64_t>(result.size()) >= min_matches_;
    }

    int norm() const { return norm_; }
    bool use_ratio_test() const { return use_ratio_test_; }
    double ratio() const { return ratio_; }
    bool cross_check() const { return cross_check_; }
    const std::optional<double>& max_distance() const { return max_distance_; }
    std::int64_t min_matches() const { return min_matches_; }

private:
    struct Forward {
        std::vector<std::int64_t> idx_a;
        std::vector<std::int64_t> idx_b;
        std::vector<double> dist;
        std::int64_t n_raw = 0;
    };
    /// Nearest-neighbour search from a to b, with the ratio test.
    Forward forward_matches(const cv::Mat& desc_a, const cv::Mat& desc_b) const;
    /// Keep matches that are mutual nearest neighbours.
    Mask cross_check_mask(const cv::Mat& desc_a, const cv::Mat& desc_b,
                          const std::vector<std::int64_t>& idx_a,
                          const std::vector<std::int64_t>& idx_b) const;

    int norm_;
    bool use_ratio_test_;
    double ratio_;
    bool cross_check_;
    std::optional<double> max_distance_;
    std::int64_t min_matches_;
    // crossCheck stays false here: it is incompatible with knnMatch, so the
    // mutual-consistency test is applied explicitly in cross_check_mask().
    cv::Ptr<cv::BFMatcher> matcher_;
};

/// Coerce descriptors to the type OpenCV's BFMatcher expects: Hamming
/// matching requires uint8, L2 matching requires float32. Passing the wrong
/// type raises a confusing OpenCV error deep in the C++ layer, so it is
/// normalised here instead.
cv::Mat as_matcher_dtype(const cv::Mat& descriptors, int norm);

/// Render accepted correspondences side by side.
///
/// When ``inlier_mask`` is supplied (typically the RANSAC mask from essential
/// matrix estimation), inliers are drawn green and outliers red -- the
/// fastest way to see whether a bad pose came from bad matching or bad
/// geometry.
cv::Mat draw_matches(const cv::Mat& image_a, const Frame& frame_a, const cv::Mat& image_b,
                     const Frame& frame_b, const MatchResult& result, int max_draw = 150,
                     const std::optional<Mask>& inlier_mask = std::nullopt);

}  // namespace mvslam
