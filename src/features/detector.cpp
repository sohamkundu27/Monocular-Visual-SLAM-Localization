#include "mvslam/features/detector.hpp"

#include <algorithm>
#include <cctype>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.features.detector");
    return instance;
}

std::string normalized_name(const std::string& detector) {
    std::string out = detector;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto begin = out.find_first_not_of(" \t\r\n\f\v");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = out.find_last_not_of(" \t\r\n\f\v");
    return out.substr(begin, end - begin + 1);
}

/// Retain the ``limit`` highest-response keypoints, descriptors in step.
///
/// The order is np.argsort(-responses) (NumPy's non-stable quicksort), so ties
/// at the budget boundary resolve exactly as in the Python run.
void keep_strongest(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::size_t limit) {
    std::vector<double> negated(keypoints.size());
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        negated[i] = -static_cast<double>(keypoints[i].response);
    }
    std::vector<std::int64_t> order = np::argsort(negated);
    order.resize(std::min(limit, order.size()));
    std::vector<cv::KeyPoint> kept;
    kept.reserve(order.size());
    cv::Mat kept_desc;
    if (!descriptors.empty()) {
        kept_desc.create(static_cast<int>(order.size()), descriptors.cols, descriptors.type());
    }
    for (std::size_t k = 0; k < order.size(); ++k) {
        const auto i = static_cast<std::size_t>(order[k]);
        kept.push_back(keypoints[i]);
        if (!descriptors.empty()) {
            descriptors.row(static_cast<int>(i)).copyTo(kept_desc.row(static_cast<int>(k)));
        }
    }
    keypoints = std::move(kept);
    descriptors = kept_desc;
}

}  // namespace

const std::vector<std::string>& supported_detectors() {
    static const std::vector<std::string> names = {"orb", "sift"};
    return names;
}

Frame::Frame(std::int64_t index_, std::int64_t frame_id_, std::vector<cv::KeyPoint> keypoints_,
             cv::Mat descriptors_)
    : index(index_), frame_id(frame_id_), keypoints(std::move(keypoints_)),
      descriptors(std::move(descriptors_)) {
    points.resize(static_cast<Eigen::Index>(keypoints.size()), 2);
    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        // kp.pt is float32; the conversion to double is exact, as in NumPy.
        points(static_cast<Eigen::Index>(i), 0) = static_cast<double>(keypoints[i].pt.x);
        points(static_cast<Eigen::Index>(i), 1) = static_cast<double>(keypoints[i].pt.y);
    }
    if (!descriptors.empty() && static_cast<std::size_t>(descriptors.rows) != keypoints.size()) {
        throw ValueError("descriptor count " + std::to_string(descriptors.rows) +
                         " != keypoint count " + std::to_string(keypoints.size()));
    }
}

std::string Frame::repr() const {
    return "Frame(index=" + std::to_string(index) + ", frame_id=" + std::to_string(frame_id) +
           ", keypoints=" + std::to_string(size()) + ")";
}

FeatureDetector::FeatureDetector(const std::string& detector, std::int64_t max_features,
                                 std::int64_t fast_threshold, double scale_factor,
                                 std::int64_t n_levels, std::int64_t edge_threshold,
                                 std::int64_t score_type, bool clahe)
    : name_(normalized_name(detector)), max_features_(max_features) {
    const auto& supported = supported_detectors();
    if (std::find(supported.begin(), supported.end(), name_) == supported.end()) {
        throw ValueError("Unsupported detector '" + detector + "'. Choose from ['orb', 'sift']");
    }
    if (max_features_ <= 0) {
        throw ValueError("max_features must be positive, got " + std::to_string(max_features));
    }
    if (clahe) {
        clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    }
    if (name_ == "orb") {
        detector_ = cv::ORB::create(
            static_cast<int>(max_features_), static_cast<float>(scale_factor),
            static_cast<int>(n_levels), static_cast<int>(edge_threshold), /*firstLevel=*/0,
            /*WTA_K=*/2, score_type == 0 ? cv::ORB::HARRIS_SCORE : cv::ORB::FAST_SCORE,
            /*patchSize=*/static_cast<int>(edge_threshold), static_cast<int>(fast_threshold));
    } else {
        detector_ = cv::SIFT::create(static_cast<int>(max_features_), /*nOctaveLayers=*/3,
                                     /*contrastThreshold=*/0.04, /*edgeThreshold=*/10,
                                     /*sigma=*/1.6);
    }
    logger().info("Feature detector: " + name_ + " (max_features=" + std::to_string(max_features_) +
                  ", clahe=" + py::repr(clahe) + ")");
}

FeatureDetector FeatureDetector::from_config(const Config& config) {
    const auto& f = config.features;
    return FeatureDetector(f.detector, f.max_features, f.fast_threshold, f.scale_factor, f.n_levels,
                           f.edge_threshold, f.score_type, f.clahe);
}

cv::Mat FeatureDetector::preprocess(const cv::Mat& image) const {
    cv::Mat out = image;
    if (out.channels() == 3) {
        cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
    }
    if (out.depth() != CV_8U) {
        // cv2.normalize(..., NORM_MINMAX) keeps the input type, then NumPy's
        // astype(np.uint8) truncates rather than rounds.
        cv::Mat normalized;
        cv::normalize(out, normalized, 0, 255, cv::NORM_MINMAX);
        cv::Mat as_double;
        normalized.convertTo(as_double, CV_64F);
        cv::Mat truncated(as_double.size(), CV_8U);
        for (int r = 0; r < as_double.rows; ++r) {
            for (int c = 0; c < as_double.cols; ++c) {
                truncated.at<std::uint8_t>(r, c) =
                    static_cast<std::uint8_t>(static_cast<int>(as_double.at<double>(r, c)));
            }
        }
        out = truncated;
    }
    if (clahe_) {
        // CLAHE keeps scratch buffers in the object, so a per-call instance
        // with the same settings keeps detect() safe to call from the VO
        // worker threads (the output does not depend on those buffers).
        cv::Mat equalised;
        cv::createCLAHE(clahe_->getClipLimit(), clahe_->getTilesGridSize())->apply(out, equalised);
        out = equalised;
    }
    return out;
}

Frame FeatureDetector::detect(const cv::Mat& image, std::int64_t index,
                              std::optional<std::int64_t> frame_id, const cv::Mat& mask) const {
    const cv::Mat prepared = preprocess(image);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    if (mask.empty()) {
        detector_->detectAndCompute(prepared, cv::noArray(), keypoints, descriptors);
    } else {
        detector_->detectAndCompute(prepared, mask, keypoints, descriptors);
    }
    const std::int64_t id = frame_id.value_or(index);
    if (keypoints.empty()) {
        logger().warning("No keypoints detected in frame " + std::to_string(id));
        return Frame(index, id, {}, cv::Mat());
    }
    // OpenCV's ORB usually respects nfeatures, but SIFT's cap is applied
    // before its final filtering pass, so enforce the budget here too.
    if (static_cast<std::int64_t>(keypoints.size()) > max_features_) {
        keep_strongest(keypoints, descriptors, static_cast<std::size_t>(max_features_));
    }
    return Frame(index, id, std::move(keypoints), descriptors);
}

int FeatureDetector::descriptor_norm() const {
    return name_ == "orb" ? cv::NORM_HAMMING : cv::NORM_L2;
}

std::string FeatureDetector::repr() const {
    return "FeatureDetector(" + name_ + ", max_features=" + std::to_string(max_features_) + ")";
}

cv::Mat draw_keypoints(const cv::Mat& image, const Frame& frame, const cv::Scalar& color, bool rich) {
    cv::Mat canvas;
    if (image.channels() == 1) {
        cv::cvtColor(image, canvas, cv::COLOR_GRAY2BGR);
    } else {
        canvas = image.clone();
    }
    cv::Mat out;
    cv::drawKeypoints(canvas, frame.keypoints, out, color,
                      rich ? cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS : cv::DrawMatchesFlags::DEFAULT);
    return out;
}

}  // namespace mvslam
