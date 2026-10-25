#include "support/test_utils.hpp"

#include <unistd.h>

#include <atomic>
#include <fstream>
#include <iostream>
#include <random>

namespace testing_utils {

bool approx(double actual, double expected, std::optional<double> rel, std::optional<double> abs) {
    if (std::isnan(expected) || std::isnan(actual)) {
        return false;  // pytest.approx never equals NaN unless nan_ok
    }
    if (actual == expected) {
        return true;
    }
    if (std::isinf(expected)) {
        return false;
    }
    const double absolute = abs.value_or(1e-12);
    double tolerance = absolute;
    if (!(rel == std::nullopt && abs != std::nullopt)) {
        tolerance = std::max(rel.value_or(1e-6) * std::fabs(expected), absolute);
    }
    return std::fabs(actual - expected) <= tolerance;
}

namespace {

bool close(double a, double e, double rtol, double atol) {
    if (std::isnan(a) && std::isnan(e)) {
        return true;
    }
    if (std::isinf(a) || std::isinf(e)) {
        return a == e;
    }
    return std::fabs(a - e) <= atol + rtol * std::fabs(e);
}

}  // namespace

::testing::AssertionResult allclose(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected,
                                    double rtol, double atol) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
        return ::testing::AssertionFailure()
               << "shape mismatch: (" << actual.rows() << ", " << actual.cols() << ") vs ("
               << expected.rows() << ", " << expected.cols() << ")";
    }
    int mismatches = 0;
    double worst = 0.0;
    std::ostringstream first;
    for (Eigen::Index r = 0; r < actual.rows(); ++r) {
        for (Eigen::Index c = 0; c < actual.cols(); ++c) {
            if (!close(actual(r, c), expected(r, c), rtol, atol)) {
                if (mismatches == 0) {
                    first.precision(17);
                    first << "first at (" << r << ", " << c << "): " << actual(r, c) << " vs "
                          << expected(r, c);
                }
                ++mismatches;
                worst = std::max(worst, std::fabs(actual(r, c) - expected(r, c)));
            }
        }
    }
    if (mismatches) {
        return ::testing::AssertionFailure() << mismatches << " element(s) differ (max abs diff "
                                             << worst << ", rtol=" << rtol << ", atol=" << atol
                                             << "); " << first.str();
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult allclose(const std::vector<double>& actual,
                                    const std::vector<double>& expected, double rtol,
                                    double atol) {
    const Eigen::MatrixXd a =
        Eigen::Map<const Eigen::MatrixXd>(actual.data(), static_cast<Eigen::Index>(actual.size()), 1);
    const Eigen::MatrixXd e = Eigen::Map<const Eigen::MatrixXd>(
        expected.data(), static_cast<Eigen::Index>(expected.size()), 1);
    return allclose(a, e, rtol, atol);
}

TempDir::TempDir() {
    static std::atomic<int> counter{0};
    std::random_device rd;
    const auto base = std::filesystem::temp_directory_path();
    path_ = base / ("mvslam-test-" + std::to_string(::getpid()) + "-" +
                    std::to_string(counter++) + "-" + std::to_string(rd()));
    std::filesystem::create_directories(path_);
}

TempDir::~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

CaptureOutput::CaptureOutput()
    : old_out_(std::cout.rdbuf(out_.rdbuf())), old_err_(std::cerr.rdbuf(err_.rdbuf())) {}

CaptureOutput::~CaptureOutput() {
    std::cout.rdbuf(old_out_);
    std::cerr.rdbuf(old_err_);
}

std::string CaptureOutput::out() {
    std::string text = out_.str();
    out_.str("");
    return text;
}

std::string CaptureOutput::err() {
    std::string text = err_.str();
    err_.str("");
    return text;
}

}  // namespace testing_utils
