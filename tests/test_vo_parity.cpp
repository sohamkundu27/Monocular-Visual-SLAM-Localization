// Front-end parity with the Python implementation (no Python counterpart).
//
// Python's cv2 calls the same OpenCV code as this port, so on identical
// images every per-frame count (keypoints, matches, RANSAC and cheirality
// inliers), every status/reason and the degeneracy flags must agree exactly.
// Floating-point outputs that pass through NumPy/LAPACK (SVD re-projection of
// R, matrix products, matrix inverses) can differ in the last bits, so those
// are compared at tight relative tolerances, as is the integrated trajectory.
//
// The KITTI cases need the dataset: set MVSLAM_KITTI_DATASET (or configure
// with -DMVSLAM_KITTI_DATASET=...) to the directory holding sequences/ and
// poses/. They skip otherwise.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <set>
#include <sstream>

#include <opencv2/core/version.hpp>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/config.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/odometry/visual_odometry.hpp"
#include "mvslam/utils/seeding.hpp"
#include "support/golden.hpp"
#include "support/kitti_fixture.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;

namespace {

using Row = std::vector<std::pair<std::string, std::string>>;

/// Minimal csv.reader for the diagnostics files (QUOTE_MINIMAL, \r\n rows).
std::vector<std::vector<std::string>> parse_csv(const std::string& text) {
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"' && i + 1 < text.size() && text[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                field.push_back(c);
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            row.push_back(field);
            field.clear();
        } else if (c == '\n') {
            row.push_back(field);
            field.clear();
            rows.push_back(row);
            row.clear();
        } else if (c != '\r') {
            field.push_back(c);
        }
    }
    if (!field.empty() || !row.empty()) {
        row.push_back(field);
        rows.push_back(row);
    }
    return rows;
}

const std::set<std::string> kExact = {"index", "frame_id", "status", "reason", "n_keypoints",
                                      "n_raw_matches", "n_matches", "n_ransac_inliers",
                                      "n_cheirality_inliers", "degenerate", "ransac_inlier_ratio",
                                      "inlier_ratio", "flow_px"};
const std::set<std::string> kVolatile = {"detect_ms", "match_ms", "pose_ms"};

struct ParityStats {
    int rows = 0;
    int exact_mismatches = 0;
    double worst_relative = 0.0;
    std::string worst_field;
};

/// Compare C++ diagnostics with the Python CSV text. Returns statistics and
/// records failures through gtest.
ParityStats compare_diagnostics(const std::vector<FrameDiagnostics>& diagnostics,
                                const std::string& python_csv, double rtol) {
    const auto rows = parse_csv(python_csv);
    ParityStats stats;
    EXPECT_EQ(rows.size(), diagnostics.size() + 1) << "row count";
    if (rows.size() != diagnostics.size() + 1) {
        return stats;
    }
    const auto& header = rows[0];
    for (std::size_t r = 0; r < diagnostics.size(); ++r) {
        const Row actual = diagnostics[r].to_row();
        const auto& expected = rows[r + 1];
        ++stats.rows;
        for (std::size_t c = 0; c < header.size(); ++c) {
            const std::string& name = header[c];
            EXPECT_EQ(actual[c].first, name);
            if (kVolatile.count(name)) {
                continue;
            }
            if (kExact.count(name)) {
                if (actual[c].second != expected[c]) {
                    ++stats.exact_mismatches;
                    ADD_FAILURE() << "row " << r + 1 << " (" << diagnostics[r].index << ") " << name
                                  << ": C++ " << actual[c].second << " vs Python " << expected[c];
                }
                continue;
            }
            const double a = std::strtod(actual[c].second.c_str(), nullptr);
            const double e = std::strtod(expected[c].c_str(), nullptr);
            if (std::isnan(a) || std::isnan(e)) {
                EXPECT_TRUE(std::isnan(a) && std::isnan(e))
                    << "row " << r + 1 << " " << name << ": " << a << " vs " << e;
                continue;
            }
            const double rel = std::fabs(a - e) / std::max(1.0, std::fabs(e));
            if (rel > stats.worst_relative) {
                stats.worst_relative = rel;
                stats.worst_field = name;
            }
            EXPECT_LE(rel, rtol) << "row " << r + 1 << " " << name << ": " << a << " vs " << e;
        }
    }
    return stats;
}

bool same_opencv_as_reference() {
    static const std::string name = "metadata.json";
    const std::string reference = golden::load(name).at("cv2").get<std::string>();
    return reference.rfind(std::to_string(CV_VERSION_MAJOR) + "." + std::to_string(CV_VERSION_MINOR) + ".", 0) == 0;
}

std::optional<std::filesystem::path> kitti_dataset() {
    const char* env = std::getenv("MVSLAM_KITTI_DATASET");
    std::string path = env != nullptr ? env : MVSLAM_KITTI_DATASET_DEFAULT;
    if (path.empty() || !std::filesystem::is_directory(std::filesystem::path(path) / "sequences")) {
        return std::nullopt;
    }
    return std::filesystem::path(path);
}

}  // namespace

TEST(VoParity, SyntheticSequenceMatchesPythonFrameByFrame) {
    if (!same_opencv_as_reference()) {
        GTEST_SKIP() << "OpenCV " << CV_VERSION << " differs from the reference cv2 version";
    }
    static const std::string name = "pipeline_synthetic.json";
    const Json& cases = golden::load(name);
    for (const auto& [case_name, item] : cases.items()) {
        SCOPED_TRACE(case_name);
        testing_utils::TempDir tmp;
        const auto root = kitti_fixture::write_kitti_sequence(tmp / "dataset", "00", 40);
        Json overrides = item.at("overrides");
        overrides["dataset.path"] = root.string();
        overrides["output.root"] = (tmp / "outputs").string();
        const Config config = Config().with_overrides(overrides);
        set_global_seed(config.runtime.seed);
        const KittiOdometryDataset dataset = KittiOdometryDataset::from_config(config);
        VisualOdometry vo = VisualOdometry::from_config(config, dataset);
        const OdometryResult result = vo.run(dataset, {}, config.runtime.progress_interval);

        const ParityStats stats =
            compare_diagnostics(result.diagnostics, item.at("diagnostics_csv").get<std::string>(), 1e-9);
        EXPECT_EQ(stats.exact_mismatches, 0);
        const PoseList expected = golden::poses(item.at("trajectory"));
        ASSERT_EQ(expected.size(), result.trajectory.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_TRUE(testing_utils::allclose(result.trajectory[i], expected[i], 1e-9, 1e-9))
                << "pose " << i;
        }
    }
}

class KittiVoParity : public ::testing::TestWithParam<std::string> {};

TEST_P(KittiVoParity, First300FramesMatchPython) {
    const auto root = kitti_dataset();
    if (!root) {
        GTEST_SKIP() << "KITTI not available (set MVSLAM_KITTI_DATASET)";
    }
    if (!same_opencv_as_reference()) {
        GTEST_SKIP() << "OpenCV " << CV_VERSION << " differs from the reference cv2 version";
    }
    const std::string run = GetParam();
    const auto golden_dir = std::filesystem::path(golden::dir()) / "kitti" / run;
    Config config = Config::from_yaml(golden_dir / "config.resolved.yaml");
    config = config.with_override("dataset.path", root->string());
    set_global_seed(config.runtime.seed);
    const KittiOdometryDataset dataset = KittiOdometryDataset::from_config(config);
    VisualOdometry vo = VisualOdometry::from_config(config, dataset);
    const OdometryResult result = vo.run(dataset, {}, 0);

    const ParityStats stats = compare_diagnostics(
        result.diagnostics, testing_utils::read_text(golden_dir / "diagnostics.csv"), 1e-6);
    EXPECT_EQ(stats.exact_mismatches, 0);
    std::cout << "[parity] " << run << ": " << stats.rows << " transitions, "
              << stats.exact_mismatches << " exact-field mismatches, worst relative difference "
              << stats.worst_relative << " (" << stats.worst_field << ")\n";

    const Trajectory expected = Trajectory::load_kitti(golden_dir / "trajectory_raw.txt");
    ASSERT_EQ(expected.size(), result.trajectory.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        worst = std::max(worst, (result.trajectory[i].topRightCorner<3, 1>() -
                                 expected[i].topRightCorner<3, 1>()).norm());
    }
    std::cout << "[parity] " << run << ": max position difference " << worst << " m\n";
    EXPECT_LT(worst, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(Kitti, KittiVoParity, ::testing::Values("sequence_00_300", "sequence_05_300"));
