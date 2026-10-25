#include "mvslam/geometry/pose.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"

namespace mvslam {

Trajectory::Trajectory(PoseList poses, std::optional<std::vector<std::int64_t>> frame_ids,
                       std::optional<std::vector<double>> timestamps)
    : poses_(std::move(poses)), timestamps_(std::move(timestamps)) {
    if (frame_ids) {
        if (frame_ids->size() != poses_.size()) {
            throw ValueError("frame_ids length " + std::to_string(frame_ids->size()) +
                             " != poses length " + std::to_string(poses_.size()));
        }
        frame_ids_ = std::move(*frame_ids);
    } else {
        frame_ids_.resize(poses_.size());
        for (std::size_t i = 0; i < poses_.size(); ++i) {
            frame_ids_[i] = static_cast<std::int64_t>(i);
        }
    }
    if (timestamps_ && timestamps_->size() != poses_.size()) {
        throw ValueError("timestamps length " + std::to_string(timestamps_->size()) +
                         " != poses length " + std::to_string(poses_.size()));
    }
}

Trajectory::Trajectory(const Eigen::Matrix4d& pose) : Trajectory(PoseList{pose}) {}

Trajectory Trajectory::from_matrices(const std::vector<Eigen::MatrixXd>& matrices) {
    PoseList poses;
    poses.reserve(matrices.size());
    for (const auto& m : matrices) {
        if (m.rows() != 4 || m.cols() != 4) {
            throw ValueError("poses must be (N, 4, 4), got (" + std::to_string(matrices.size()) +
                             ", " + std::to_string(m.rows()) + ", " + std::to_string(m.cols()) +
                             ")");
        }
        poses.emplace_back(m);
    }
    return Trajectory(std::move(poses));
}

Trajectory Trajectory::slice(std::int64_t start, std::optional<std::int64_t> stop) const {
    const auto n = static_cast<std::int64_t>(size());
    auto clamp = [n](std::int64_t i) {
        if (i < 0) {
            i += n;
        }
        return std::max<std::int64_t>(0, std::min(i, n));
    };
    const std::int64_t begin = clamp(start);
    const std::int64_t end = stop ? clamp(*stop) : n;
    std::vector<std::int64_t> indices;
    for (std::int64_t i = begin; i < end; ++i) {
        indices.push_back(i);
    }
    return take(indices);
}

Trajectory Trajectory::take(const std::vector<std::int64_t>& indices) const {
    PoseList poses;
    std::vector<std::int64_t> ids;
    std::optional<std::vector<double>> times;
    if (timestamps_) {
        times.emplace();
    }
    for (std::int64_t raw : indices) {
        std::int64_t i = raw < 0 ? raw + static_cast<std::int64_t>(size()) : raw;
        if (i < 0 || i >= static_cast<std::int64_t>(size())) {
            throw IndexError("index " + std::to_string(raw) + " is out of bounds for axis 0 with size " +
                             std::to_string(size()));
        }
        const auto k = static_cast<std::size_t>(i);
        poses.push_back(poses_[k]);
        ids.push_back(frame_ids_[k]);
        if (times) {
            times->push_back((*timestamps_)[k]);
        }
    }
    return Trajectory(std::move(poses), std::move(ids), std::move(times));
}

Points3 Trajectory::positions() const {
    Points3 out(static_cast<Eigen::Index>(size()), 3);
    for (std::size_t i = 0; i < size(); ++i) {
        out.row(static_cast<Eigen::Index>(i)) = poses_[i].topRightCorner<3, 1>().transpose();
    }
    return out;
}

std::vector<Eigen::Matrix3d> Trajectory::rotations() const {
    std::vector<Eigen::Matrix3d> out;
    out.reserve(size());
    for (const auto& T : poses_) {
        out.emplace_back(T.topLeftCorner<3, 3>());
    }
    return out;
}

Points2 Trajectory::xz() const {
    Points2 out(static_cast<Eigen::Index>(size()), 2);
    for (std::size_t i = 0; i < size(); ++i) {
        out(static_cast<Eigen::Index>(i), 0) = poses_[i](0, 3);
        out(static_cast<Eigen::Index>(i), 1) = poses_[i](2, 3);
    }
    return out;
}

std::vector<double> Trajectory::step_lengths() const {
    std::vector<double> out;
    if (size() < 2) {
        return out;
    }
    out.reserve(size() - 1);
    for (std::size_t i = 0; i + 1 < size(); ++i) {
        const double dx = poses_[i + 1](0, 3) - poses_[i](0, 3);
        const double dy = poses_[i + 1](1, 3) - poses_[i](1, 3);
        const double dz = poses_[i + 1](2, 3) - poses_[i](2, 3);
        out.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return out;
}

std::vector<double> Trajectory::cumulative_distance() const {
    const std::vector<double> steps = step_lengths();
    std::vector<double> out(size(), 0.0);
    double running = 0.0;
    for (std::size_t i = 0; i < steps.size(); ++i) {
        running += steps[i];
        out[i + 1] = running;
    }
    return out;
}

double Trajectory::path_length() const { return np::sum(step_lengths()); }

PoseList Trajectory::relative_poses(int delta) const {
    if (delta < 1) {
        throw ValueError("delta must be >= 1");
    }
    PoseList out;
    const auto n = static_cast<std::int64_t>(size()) - delta;
    for (std::int64_t i = 0; i < n; ++i) {
        out.push_back(relative_pose(poses_[static_cast<std::size_t>(i)],
                                    poses_[static_cast<std::size_t>(i + delta)]));
    }
    return out;
}

Trajectory Trajectory::transformed(const Eigen::Matrix4d& T, double scale) const {
    const Eigen::Matrix3d R = T.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T.topRightCorner<3, 1>();
    PoseList out(size(), Eigen::Matrix4d::Identity());
    for (std::size_t i = 0; i < size(); ++i) {
        out[i].topLeftCorner<3, 3>() = R * poses_[i].topLeftCorner<3, 3>();
        const Eigen::Vector3d p = poses_[i].topRightCorner<3, 1>();
        out[i].topRightCorner<3, 1>() = (scale * (R * p)) + t;
    }
    return Trajectory(std::move(out), frame_ids_, timestamps_);
}

Trajectory Trajectory::relative_to_first() const {
    if (empty()) {
        return copy();
    }
    const Eigen::Matrix4d T0_inv = invert_se3(poses_[0]);
    PoseList out;
    out.reserve(size());
    for (const auto& T : poses_) {
        out.push_back(T0_inv * T);
    }
    return Trajectory(std::move(out), frame_ids_, timestamps_);
}

Trajectory Trajectory::orthonormalized() const {
    PoseList out;
    out.reserve(size());
    for (const auto& T : poses_) {
        out.push_back(project_to_se3(T));
    }
    return Trajectory(std::move(out), frame_ids_, timestamps_);
}

bool Trajectory::is_valid() const {
    for (const auto& T : poses_) {
        if (!is_valid_se3(T, 1e-4)) {
            return false;
        }
    }
    return true;
}

Trajectory Trajectory::select_frames(const std::vector<std::int64_t>& frame_ids) const {
    std::unordered_map<std::int64_t, std::int64_t> lookup;
    for (std::size_t i = 0; i < frame_ids_.size(); ++i) {
        lookup[frame_ids_[i]] = static_cast<std::int64_t>(i);
    }
    std::vector<std::int64_t> missing;
    std::vector<std::int64_t> indices;
    for (std::int64_t f : frame_ids) {
        auto it = lookup.find(f);
        if (it == lookup.end()) {
            missing.push_back(f);
        } else {
            indices.push_back(it->second);
        }
    }
    if (!missing.empty()) {
        std::string sample = "[";
        for (std::size_t i = 0; i < std::min<std::size_t>(5, missing.size()); ++i) {
            sample += (i ? ", " : "") + std::to_string(missing[i]);
        }
        sample += "]";
        throw KeyError("Trajectory is missing " + std::to_string(missing.size()) +
                       " requested frames, e.g. " + sample);
    }
    return take(indices);
}

Trajectory Trajectory::from_kitti_rows(const std::vector<KittiRow>& rows,
                                       std::optional<std::vector<std::int64_t>> frame_ids) {
    PoseList poses;
    poses.reserve(rows.size());
    for (const auto& row : rows) {
        poses.push_back(kitti_row_to_se3(row));
    }
    return Trajectory(std::move(poses), std::move(frame_ids));
}

std::vector<KittiRow> Trajectory::to_kitti_rows() const {
    std::vector<KittiRow> rows;
    rows.reserve(size());
    for (const auto& T : poses_) {
        rows.push_back(se3_to_kitti_row(T));
    }
    return rows;
}

std::filesystem::path Trajectory::save_kitti(const std::filesystem::path& path) const {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::string text;
    text.reserve(size() * 12 * 20);
    for (const auto& row : to_kitti_rows()) {
        text += py::savetxt_row(row.data(), row.size(), "%.12e");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return path;
}

std::vector<std::vector<double>> loadtxt(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw FileNotFoundError(path.string() + " not found.");
    }
    std::vector<std::vector<double>> rows;
    std::string line;
    std::size_t lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash);
        }
        std::istringstream tokens(line);
        std::vector<double> row;
        std::string token;
        while (tokens >> token) {
            char* end = nullptr;
            const double value = std::strtod(token.c_str(), &end);
            if (end == token.c_str() || *end != '\0') {
                throw ValueError("could not convert string '" + token + "' to float64 at row " +
                                 std::to_string(lineno - 1) + ", column 1.");
            }
            row.push_back(value);
        }
        if (row.empty()) {
            continue;
        }
        if (!rows.empty() && row.size() != rows.front().size()) {
            throw ValueError("the number of columns changed from " +
                             std::to_string(rows.front().size()) + " to " +
                             std::to_string(row.size()) + " at row " + std::to_string(lineno) +
                             "; use `usecols` to select a subset and avoid this error");
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

Trajectory Trajectory::load_kitti(const std::filesystem::path& path) {
    const auto rows = loadtxt(path);
    std::vector<KittiRow> kitti;
    kitti.reserve(rows.size());
    for (const auto& row : rows) {
        if (row.size() != 12) {
            // np.loadtxt of a single-row file is 1-D and gets promoted, so the
            // reported shape is (N, columns) either way.
            throw ValueError("KITTI pose rows must be (N, 12), got (" + std::to_string(rows.size()) +
                             ", " + std::to_string(row.size()) + ")");
        }
        KittiRow r{};
        std::copy(row.begin(), row.end(), r.begin());
        kitti.push_back(r);
    }
    if (kitti.empty()) {
        throw ValueError("KITTI pose rows must be (N, 12), got (1, 0)");
    }
    return from_kitti_rows(kitti);
}

std::string Trajectory::repr() const {
    if (empty()) {
        return "Trajectory(empty)";
    }
    return "Trajectory(n=" + std::to_string(size()) + ", frames=" +
           std::to_string(frame_ids_.front()) + ".." + std::to_string(frame_ids_.back()) +
           ", path_length=" + py::format(path_length(), ".1f") + " m)";
}

}  // namespace mvslam
