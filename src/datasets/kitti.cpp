#include "mvslam/datasets/kitti.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include <opencv2/imgcodecs.hpp>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.datasets.kitti");
    return instance;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string names_repr(std::vector<std::string> names) {
    std::sort(names.begin(), names.end());
    Json list = Json::array();
    for (const auto& n : names) {
        list.push_back(n);
    }
    return py::repr(list);
}

std::vector<std::string> available_sequences(const std::filesystem::path& root) {
    const auto sequences_dir = root / "sequences";
    std::vector<std::string> out;
    if (!std::filesystem::is_directory(sequences_dir)) {
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(sequences_dir)) {
        if (entry.is_directory()) {
            out.push_back(entry.path().filename().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

const std::set<std::string>& sequences_with_ground_truth() {
    static const std::set<std::string> ids = {"00", "01", "02", "03", "04", "05",
                                              "06", "07", "08", "09", "10"};
    return ids;
}

std::vector<std::int64_t> FrameSelection::apply(std::int64_t total) const {
    if (step < 1) {
        throw ValueError("frame_step must be >= 1, got " + std::to_string(step));
    }
    if (start < 0) {
        throw ValueError("start_frame must be >= 0, got " + std::to_string(start));
    }
    if (start >= total) {
        throw ValueError("start_frame " + std::to_string(start) +
                         " is past the end of the sequence (" + std::to_string(total) + ")");
    }
    std::vector<std::int64_t> indices;
    for (std::int64_t i = start; i < total; i += step) {
        indices.push_back(i);
    }
    if (max_frames && *max_frames != 0) {
        // indices[:max_frames], including Python's negative-stop semantics.
        const auto n = static_cast<std::int64_t>(indices.size());
        std::int64_t stop = *max_frames >= 0 ? std::min(*max_frames, n)
                                             : std::max<std::int64_t>(0, n + *max_frames);
        indices.resize(static_cast<std::size_t>(stop));
    }
    return indices;
}

std::string normalize_sequence(const std::string& sequence) {
    const auto begin = sequence.find_first_not_of(" \t\r\n\f\v");
    if (begin == std::string::npos) {
        throw DatasetError("Sequence identifier must not be empty");
    }
    const auto end = sequence.find_last_not_of(" \t\r\n\f\v");
    const std::string text = sequence.substr(begin, end - begin + 1);
    const bool digits = std::all_of(text.begin(), text.end(),
                                    [](unsigned char c) { return std::isdigit(c) != 0; });
    if (digits) {
        const long long value = std::stoll(text);
        return py::sprintf("%02lld", value);
    }
    return text;
}

std::filesystem::path expanduser(const std::filesystem::path& path) {
    const std::string text = path.string();
    if (text == "~" || text.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (home != nullptr) {
            return std::filesystem::path(std::string(home) + text.substr(1));
        }
    }
    return path;
}

KittiOdometryDataset::KittiOdometryDataset(const std::filesystem::path& root,
                                           const std::string& sequence, const std::string& camera,
                                           bool grayscale, std::int64_t start_frame,
                                           std::int64_t frame_step,
                                           std::optional<std::int64_t> max_frames)
    : root_(expanduser(root)),
      sequence_(normalize_sequence(sequence)),
      camera_(camera),
      grayscale_(grayscale) {
    sequence_dir_ = root_ / "sequences" / sequence_;
    if (!std::filesystem::is_directory(sequence_dir_)) {
        throw DatasetError("Sequence directory not found: " + sequence_dir_.string() +
                           "\nAvailable sequences under " + (root_ / "sequences").string() + ": " +
                           names_repr(available_sequences(root_)));
    }

    image_dir_ = sequence_dir_ / camera_;
    if (!std::filesystem::is_directory(image_dir_)) {
        std::vector<std::string> present;
        for (const auto& entry : std::filesystem::directory_iterator(sequence_dir_)) {
            present.push_back(entry.path().filename().string());
        }
        throw DatasetError("Image directory not found: " + image_dir_.string() +
                           ". Present in this sequence: " + names_repr(present));
    }

    std::vector<std::filesystem::path> all_paths;
    for (const auto& entry : std::filesystem::directory_iterator(image_dir_)) {
        const std::string suffix = lower(entry.path().extension().string());
        if (suffix == ".png" || suffix == ".jpg" || suffix == ".jpeg") {
            all_paths.push_back(entry.path());
        }
    }
    std::sort(all_paths.begin(), all_paths.end());
    if (all_paths.empty()) {
        throw DatasetError("No images found in " + image_dir_.string());
    }

    selection_ = FrameSelection{start_frame, frame_step, max_frames};
    frame_ids_ = selection_.apply(static_cast<std::int64_t>(all_paths.size()));
    for (std::int64_t i : frame_ids_) {
        image_paths_.push_back(all_paths[static_cast<std::size_t>(i)]);
    }
    total_frames_ = static_cast<std::int64_t>(all_paths.size());

    logger().info(py::sprintf("KITTI sequence %s: %zu/%lld frames selected from %s (start=%lld, step=%lld)",
                              sequence_.c_str(), frame_ids_.size(),
                              static_cast<long long>(total_frames_), image_dir_.c_str(),
                              static_cast<long long>(start_frame),
                              static_cast<long long>(frame_step)));
}

KittiOdometryDataset::KittiOdometryDataset(const std::filesystem::path& root, int sequence,
                                           const std::string& camera, bool grayscale)
    : KittiOdometryDataset(root, std::to_string(sequence), camera, grayscale) {}

KittiOdometryDataset KittiOdometryDataset::from_config(const Config& config) {
    const auto& d = config.dataset;
    return KittiOdometryDataset(d.path, d.sequence, d.camera, d.grayscale, d.start_frame,
                                d.frame_step, d.max_frames);
}

cv::Mat KittiOdometryDataset::load_image(std::size_t index) const {
    const auto& path = image_paths_.at(index);
    const int flag = grayscale_ ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat image = cv::imread(path.string(), flag);
    if (image.empty()) {
        throw DatasetError("Failed to decode image: " + path.string());
    }
    return image;
}

const CalibEntries& KittiOdometryDataset::calib_raw() const {
    if (!calib_raw_) {
        calib_raw_ = parse_kitti_calib(sequence_dir_ / "calib.txt");
    }
    return *calib_raw_;
}

const CameraCalibration& KittiOdometryDataset::calibration() const {
    if (!calibration_) {
        calibration_ = calibration_for_camera(calib_raw(), camera_);
        logger().info("Loaded calibration " + calibration_->repr());
    }
    return *calibration_;
}

std::pair<int, int> KittiOdometryDataset::image_size() const {
    if (!image_size_) {
        const cv::Mat image = load_image(0);
        image_size_ = std::make_pair(image.cols, image.rows);
    }
    return *image_size_;
}

const std::optional<std::vector<double>>& KittiOdometryDataset::timestamps() const {
    if (timestamps_loaded_) {
        return timestamps_;
    }
    const auto times_path = sequence_dir_ / "times.txt";
    if (!std::filesystem::is_regular_file(times_path)) {
        logger().warning("No times.txt in " + sequence_dir_.string() + "; timestamps unavailable");
        timestamps_loaded_ = true;
        return timestamps_;
    }
    std::vector<double> times;
    for (const auto& row : loadtxt(times_path)) {
        times.insert(times.end(), row.begin(), row.end());  // .reshape(-1)
    }
    if (static_cast<std::int64_t>(times.size()) < total_frames_) {
        throw DatasetError(times_path.string() + " has " + std::to_string(times.size()) +
                           " timestamps but the sequence has " + std::to_string(total_frames_) +
                           " images");
    }
    std::vector<double> selected;
    selected.reserve(frame_ids_.size());
    for (std::int64_t i : frame_ids_) {
        selected.push_back(times[static_cast<std::size_t>(i)]);
    }
    timestamps_ = std::move(selected);
    timestamps_loaded_ = true;
    return timestamps_;
}

std::filesystem::path KittiOdometryDataset::poses_path() const {
    return root_ / "poses" / (sequence_ + ".txt");
}

bool KittiOdometryDataset::has_ground_truth() const {
    return std::filesystem::is_regular_file(poses_path());
}

const std::optional<Trajectory>& KittiOdometryDataset::ground_truth() const {
    if (ground_truth_loaded_) {
        return ground_truth_;
    }
    if (!has_ground_truth()) {
        if (sequences_with_ground_truth().count(sequence_)) {
            logger().warning("Sequence " + sequence_ + " should have ground truth but " +
                             poses_path().string() + " is missing");
        } else {
            logger().info("Sequence " + sequence_ + " is part of the test split (no ground truth)");
        }
        ground_truth_loaded_ = true;
        return ground_truth_;
    }

    const auto rows = loadtxt(poses_path());
    const std::size_t columns = rows.empty() ? 0 : rows.front().size();
    if (columns != 12) {
        throw DatasetError(poses_path().string() + ": expected 12 values per row, got " +
                           std::to_string(columns));
    }
    if (static_cast<std::int64_t>(rows.size()) < total_frames_) {
        throw DatasetError(poses_path().string() + " has " + std::to_string(rows.size()) +
                           " poses but the sequence has " + std::to_string(total_frames_) +
                           " images");
    }
    std::vector<KittiRow> selected;
    selected.reserve(frame_ids_.size());
    for (std::int64_t i : frame_ids_) {
        KittiRow row{};
        const auto& source = rows[static_cast<std::size_t>(i)];
        std::copy(source.begin(), source.end(), row.begin());
        selected.push_back(row);
    }
    Trajectory trajectory = Trajectory::from_kitti_rows(selected, frame_ids_);
    if (!trajectory.is_valid()) {
        throw DatasetError(poses_path().string() + " contains poses that are not valid SE(3)");
    }
    logger().info(py::sprintf("Loaded ground truth: %zu poses, %.1f m travelled", trajectory.size(),
                              trajectory.path_length()));
    ground_truth_ = std::move(trajectory);
    ground_truth_loaded_ = true;
    return ground_truth_;
}

std::optional<std::vector<double>> KittiOdometryDataset::ground_truth_scales() const {
    const auto& gt = ground_truth();
    if (!gt) {
        return std::nullopt;
    }
    return gt->step_lengths();
}

Json KittiOdometryDataset::describe() const {
    const auto& gt = ground_truth();
    Json out = Json::object();
    out["sequence"] = sequence_;
    out["camera"] = camera_;
    out["root"] = root_.string();
    out["frames_available"] = total_frames_;
    out["frames_selected"] = static_cast<std::int64_t>(size());
    out["start_frame"] = selection_.start;
    out["frame_step"] = selection_.step;
    const auto [width, height] = image_size();
    out["image_size"] = Json::array({width, height});
    out["fx"] = py::make_float(calibration().fx());
    out["fy"] = py::make_float(calibration().fy());
    out["cx"] = py::make_float(calibration().cx());
    out["cy"] = py::make_float(calibration().cy());
    out["has_ground_truth"] = gt.has_value();
    out["ground_truth_path_length_m"] =
        gt ? py::make_float(py::round(gt->path_length(), 3)) : Json(nullptr);
    return out;
}

std::string KittiOdometryDataset::repr() const {
    return "KittiOdometryDataset(sequence=" + py::repr_str(sequence_) +
           ", camera=" + py::repr_str(camera_) + ", frames=" + std::to_string(size()) + ")";
}

}  // namespace mvslam
