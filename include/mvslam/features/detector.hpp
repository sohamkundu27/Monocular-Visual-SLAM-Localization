// Keypoint detection and description.
//
// ORB is the default: it is fast enough to run the full 4541-frame KITTI
// sequence 00 in a few minutes on a laptop CPU, and its 256-bit binary
// descriptors make the loop-closure database cheap to store and compare.
//
// SIFT is offered as an alternative because its float descriptors are more
// discriminative for wide-baseline loop closure, at roughly an order of
// magnitude more compute. The rest of the pipeline adapts automatically: the
// matcher picks the Hamming or L2 norm from the descriptor type, so nothing
// downstream needs to know which detector ran.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "mvslam/utils/cv_eigen.hpp"

namespace mvslam {

struct Config;

/// Detectors this module knows how to build.
const std::vector<std::string>& supported_detectors();

/// Features extracted from a single image.
struct Frame {
    /// Position within the processed sequence (0-based).
    std::int64_t index = 0;
    /// Original index in the KITTI sequence (differs when frames are skipped).
    std::int64_t frame_id = 0;
    /// OpenCV keypoints, kept because drawing utilities and sub-pixel
    /// refinement need the full structure.
    std::vector<cv::KeyPoint> keypoints;
    /// (N, D) descriptor matrix: CV_8U for ORB, CV_32F for SIFT. Empty when
    /// the detector found nothing (``None`` in the Python original).
    cv::Mat descriptors;
    /// (N, 2) double keypoint pixel coordinates, cached because every
    /// geometric routine wants them in this form.
    Points2 points;

    /// Throws ValueError when descriptor and keypoint counts differ.
    Frame(std::int64_t index, std::int64_t frame_id, std::vector<cv::KeyPoint> keypoints,
          cv::Mat descriptors);

    std::size_t size() const { return keypoints.size(); }

    /// True when the frame carries descriptors that can be matched.
    bool is_usable() const { return !descriptors.empty() && !keypoints.empty(); }

    std::string repr() const;
};

/// Wraps an OpenCV detector behind a stable, configurable interface.
class FeatureDetector {
public:
    explicit FeatureDetector(const std::string& detector = "orb", std::int64_t max_features = 3000,
                             std::int64_t fast_threshold = 20, double scale_factor = 1.2,
                             std::int64_t n_levels = 8, std::int64_t edge_threshold = 31,
                             std::int64_t score_type = 0, bool clahe = false);

    static FeatureDetector from_config(const Config& config);

    /// Convert to grayscale and optionally equalise contrast.
    ///
    /// CLAHE helps on the strongly backlit frames in KITTI sequence 01, at a
    /// small runtime cost, so it is off by default.
    cv::Mat preprocess(const cv::Mat& image) const;

    /// Detect keypoints and compute descriptors for one image.
    ///
    /// A frame with no detectable structure yields a Frame with zero
    /// keypoints rather than raising, so the caller can record the failure
    /// and carry on instead of aborting a 4000-frame run.
    Frame detect(const cv::Mat& image, std::int64_t index = 0,
                 std::optional<std::int64_t> frame_id = std::nullopt,
                 const cv::Mat& mask = cv::Mat()) const;

    /// The OpenCV distance norm that matches this detector's descriptors.
    int descriptor_norm() const;

    const std::string& name() const { return name_; }
    std::int64_t max_features() const { return max_features_; }

    std::string repr() const;

private:
    std::string name_;
    std::int64_t max_features_;
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Ptr<cv::Feature2D> detector_;
};

/// Render keypoints for debugging. ``rich=true`` draws size and orientation,
/// which is the quickest way to spot a detector configured with the wrong
/// pyramid or patch size.
cv::Mat draw_keypoints(const cv::Mat& image, const Frame& frame,
                       const cv::Scalar& color = cv::Scalar(0, 255, 0), bool rich = true);

}  // namespace mvslam
