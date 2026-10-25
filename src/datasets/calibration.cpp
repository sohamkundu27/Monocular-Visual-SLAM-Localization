#include "mvslam/datasets/calibration.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include "mvslam/compat/pyformat.hpp"
#include "mvslam/compat/pyjson.hpp"
#include "mvslam/errors.hpp"

namespace mvslam {

namespace {

std::string shape_text(Eigen::Index rows, Eigen::Index cols) {
    return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

std::string strip(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n\f\v");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n\f\v");
    return s.substr(begin, end - begin + 1);
}

std::string sorted_keys_repr(const std::vector<std::string>& keys) {
    Json list = Json::array();
    for (const auto& k : keys) {  // std::map keys are already sorted
        list.push_back(k);
    }
    return py::repr(list);
}

// np.fromstring(values, sep=" "): whitespace-separated floats, stopping at
// the first token that does not parse (NumPy warns and returns what it read).
std::vector<double> parse_numbers(const std::string& text) {
    std::vector<double> out;
    std::istringstream tokens(text);
    std::string token;
    while (tokens >> token) {
        char* end = nullptr;
        const double v = std::strtod(token.c_str(), &end);
        if (end == token.c_str() || *end != '\0') {
            break;
        }
        out.push_back(v);
    }
    return out;
}

}  // namespace

CameraCalibration::CameraCalibration(const Eigen::Matrix3d& K, const Eigen::Matrix<double, 3, 4>& P,
                                     std::string name)
    : K_(K), P_(P), name_(std::move(name)) {
    if (!K_.allFinite() || !P_.allFinite()) {
        throw CalibrationError("Calibration contains non-finite values");
    }
    if (K_(0, 0) <= 0 || K_(1, 1) <= 0) {
        throw CalibrationError("Focal lengths must be positive, got " + py::repr(K_(0, 0)) + ", " +
                               py::repr(K_(1, 1)));
    }
}

CameraCalibration CameraCalibration::from_matrices(const Eigen::MatrixXd& K, const Eigen::MatrixXd& P,
                                                   const std::string& name) {
    if (K.rows() != 3 || K.cols() != 3) {
        throw CalibrationError("K must be 3x3, got " + shape_text(K.rows(), K.cols()));
    }
    if (P.rows() != 3 || P.cols() != 4) {
        throw CalibrationError("P must be 3x4, got " + shape_text(P.rows(), P.cols()));
    }
    return CameraCalibration(K, P, name);
}

double CameraCalibration::baseline_m() const { return -P_(0, 3) / fx(); }

Points2 CameraCalibration::normalize_points(const Eigen::MatrixXd& points) const {
    if (points.cols() != 2) {
        throw ValueError("points must be (N, 2), got " + shape_text(points.rows(), points.cols()));
    }
    Points2 out(points.rows(), 2);
    for (Eigen::Index i = 0; i < points.rows(); ++i) {
        out(i, 0) = (points(i, 0) - cx()) / fx();
        out(i, 1) = (points(i, 1) - cy()) / fy();
    }
    return out;
}

Points2 CameraCalibration::project(const Eigen::MatrixXd& points_cam) const {
    if (points_cam.cols() != 3) {
        throw ValueError("points_cam must be (N, 3), got " +
                         shape_text(points_cam.rows(), points_cam.cols()));
    }
    Points2 uv(points_cam.rows(), 2);
    for (Eigen::Index i = 0; i < points_cam.rows(); ++i) {
        const double z = points_cam(i, 2);
        if (z <= 0) {
            uv(i, 0) = uv(i, 1) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        uv(i, 0) = fx() * points_cam(i, 0) / z + cx();
        uv(i, 1) = fy() * points_cam(i, 1) / z + cy();
    }
    return uv;
}

std::string CameraCalibration::repr() const {
    return "CameraCalibration(" + name_ + ", fx=" + py::format(fx(), ".2f") +
           ", fy=" + py::format(fy(), ".2f") + ", cx=" + py::format(cx(), ".2f") +
           ", cy=" + py::format(cy(), ".2f") + ")";
}

CalibEntries parse_kitti_calib(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw CalibrationError("Calibration file not found: " + path.string());
    }
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    CalibEntries entries;
    std::size_t lineno = 0;
    std::size_t start = 0;
    while (start <= text.size()) {
        auto newline = text.find('\n', start);
        if (newline == std::string::npos) {
            if (start == text.size()) {
                break;  // str.splitlines() yields no trailing empty line
            }
            newline = text.size();
        }
        std::string raw = text.substr(start, newline - start);
        if (!raw.empty() && raw.back() == '\r') {
            raw.pop_back();
        }
        start = newline + 1;
        ++lineno;
        const std::string line = strip(raw);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            throw CalibrationError(path.string() + ":" + std::to_string(lineno) +
                                   ": expected 'key: values', got " + py::repr_str(raw));
        }
        const std::string key = strip(line.substr(0, colon));
        const std::vector<double> numbers = parse_numbers(line.substr(colon + 1));
        if (numbers.size() != 12) {
            throw CalibrationError(path.string() + ":" + std::to_string(lineno) + ": key '" + key +
                                   "' has " + std::to_string(numbers.size()) +
                                   " values, expected 12");
        }
        for (double v : numbers) {
            if (!std::isfinite(v)) {
                throw CalibrationError(path.string() + ":" + std::to_string(lineno) + ": key '" +
                                       key + "' has non-finite values");
            }
        }
        Eigen::MatrixXd M;
        if (key == "Tr") {
            M = Eigen::MatrixXd::Identity(4, 4);
        } else {
            M = Eigen::MatrixXd::Zero(3, 4);
        }
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                M(r, c) = numbers[static_cast<std::size_t>(r * 4 + c)];
            }
        }
        entries[key] = M;
    }
    if (entries.empty()) {
        throw CalibrationError(path.string() + ": no calibration entries found");
    }
    return entries;
}

const std::map<std::string, std::string>& camera_to_projection() {
    static const std::map<std::string, std::string> mapping = {
        {"image_0", "P0"},  // left grayscale (the monocular SLAM camera)
        {"image_1", "P1"},  // right grayscale
        {"image_2", "P2"},  // left colour
        {"image_3", "P3"},  // right colour
    };
    return mapping;
}

CameraCalibration calibration_for_camera(const CalibEntries& calib, const std::string& camera) {
    const auto& mapping = camera_to_projection();
    auto it = mapping.find(camera);
    if (it == mapping.end()) {
        std::vector<std::string> names;
        for (const auto& [name, _] : mapping) {
            names.push_back(name);
        }
        throw CalibrationError("Unknown camera '" + camera + "'. Expected one of " +
                               sorted_keys_repr(names));
    }
    const std::string& key = it->second;
    auto entry = calib.find(key);
    if (entry == calib.end()) {
        std::vector<std::string> names;
        for (const auto& [name, _] : calib) {
            names.push_back(name);
        }
        throw CalibrationError("Calibration is missing '" + key + "' (available: " +
                               sorted_keys_repr(names) + ")");
    }
    const Eigen::MatrixXd& P = entry->second;
    return CameraCalibration::from_matrices(P.topLeftCorner(3, 3), P, key);
}

}  // namespace mvslam
