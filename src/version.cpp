#include "mvslam/version.hpp"

#include <sys/utsname.h>

#include <Eigen/Core>
#include <gtsam/config.h>
#include <opencv2/core/version.hpp>

#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif

namespace mvslam {

std::string version() { return MVSLAM_VERSION; }

std::string compiler_version() { return MVSLAM_CXX_COMPILER; }

std::string opencv_version() { return CV_VERSION; }

std::string gtsam_version() { return GTSAM_VERSION_STRING; }

std::string eigen_version() {
    return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
           std::to_string(EIGEN_MINOR_VERSION);
}

std::string platform_string() {
    struct utsname info {};
    if (uname(&info) != 0) {
        return "unknown";
    }
    std::string out = std::string(info.sysname) + "-" + info.release + "-" + info.machine;
#if defined(__GLIBC__)
    out += std::string("-with-glibc") + gnu_get_libc_version();
#endif
    return out;
}

}  // namespace mvslam
