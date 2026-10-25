// Build and runtime environment information.
//
// Replaces the Python "environment" block (python version, platform) in
// metrics.json with the toolchain and library versions that actually produced
// a C++ run.
#pragma once

#include <string>

namespace mvslam {

/// Project version, e.g. "0.1.0".
std::string version();

/// Compiler identification, e.g. "GNU 13.3.0".
std::string compiler_version();

/// OpenCV version the library was compiled against (CV_VERSION).
std::string opencv_version();

/// GTSAM version the library was compiled against.
std::string gtsam_version();

/// Eigen version the library was compiled against.
std::string eigen_version();

/// Platform string in the style of Python's ``platform.platform()``,
/// e.g. "Linux-6.8.0-45-generic-x86_64-with-glibc2.39".
std::string platform_string();

}  // namespace mvslam
