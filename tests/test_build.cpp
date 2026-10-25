// Build smoke test: the toolchain and every heavy dependency link together.
#include <gtest/gtest.h>

#include "mvslam/version.hpp"

TEST(Build, ReportsDependencyVersions) {
    EXPECT_FALSE(mvslam::version().empty());
    EXPECT_FALSE(mvslam::compiler_version().empty());
    EXPECT_EQ(mvslam::opencv_version().rfind("4.", 0), 0u);
    EXPECT_EQ(mvslam::gtsam_version().rfind("4.2", 0), 0u);
    EXPECT_FALSE(mvslam::platform_string().empty());
}
