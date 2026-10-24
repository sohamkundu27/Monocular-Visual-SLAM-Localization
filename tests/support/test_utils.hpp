// Assertion helpers with the semantics of the Python test suite's tools, so
// ported tests keep the original tolerances:
//
// * ``approx(actual, expected, rel, abs)`` is ``pytest.approx``: default
//   tolerance max(1e-6 * |expected|, 1e-12); with only ``abs`` given, only the
//   absolute tolerance applies.
// * ``EXPECT_ALLCLOSE(actual, expected, rtol, atol)`` is
//   ``np.testing.assert_allclose`` (default rtol=1e-7, atol=0, NaN == NaN).
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace testing_utils {

/// pytest.approx(expected, rel=rel, abs=abs) == actual
bool approx(double actual, double expected, std::optional<double> rel = std::nullopt,
            std::optional<double> abs = std::nullopt);

/// Element-wise allclose; returns an explanation on failure.
::testing::AssertionResult allclose(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected,
                                    double rtol = 1e-7, double atol = 0.0);
::testing::AssertionResult allclose(const std::vector<double>& actual,
                                    const std::vector<double>& expected, double rtol = 1e-7,
                                    double atol = 0.0);

/// A temporary directory removed when the object goes out of scope (pytest's
/// ``tmp_path``).
class TempDir {
public:
    TempDir();
    ~TempDir();
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    const std::filesystem::path& path() const { return path_; }
    std::filesystem::path operator/(const std::string& name) const { return path_ / name; }

private:
    std::filesystem::path path_;
};

/// Read a whole file as text.
std::string read_text(const std::filesystem::path& path);
/// Write text to a file, creating parent directories.
void write_text(const std::filesystem::path& path, const std::string& text);

/// Captures what a block writes to stdout/stderr (pytest's ``capsys``).
class CaptureOutput {
public:
    CaptureOutput();
    ~CaptureOutput();
    std::string out();
    std::string err();

private:
    std::streambuf* old_out_;
    std::streambuf* old_err_;
    std::ostringstream out_;
    std::ostringstream err_;
};

}  // namespace testing_utils

// Variadic over all arguments so -Wpedantic accepts calls without tolerances.
#define EXPECT_APPROX(...) EXPECT_TRUE(testing_utils::approx(__VA_ARGS__))

#define EXPECT_ALLCLOSE(...) EXPECT_TRUE(testing_utils::allclose(__VA_ARGS__))
