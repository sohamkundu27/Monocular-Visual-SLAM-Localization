// Tests for the command line interfaces and the config/logging plumbing
// (mirrors tests/test_cli.py).
#include <gtest/gtest.h>

#include "mvslam/utils/logging.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using testing_utils::TempDir;

// --------------------------------------------------------------------------- //
// TestLogging
// --------------------------------------------------------------------------- //

class TestLogging : public ::testing::Test {
protected:
    void TearDown() override { reset_logging(); }
};

// py: test_cli.py::TestLogging::test_file_log_stays_complete_when_the_console_is_quiet
TEST_F(TestLogging, FileLogStaysCompleteWhenTheConsoleIsQuiet) {
    // run.log must record the run even at a raised console threshold.
    TempDir tmp;
    const auto log_path = tmp / "run.log";
    setup_logging(LogLevel::Error, log_path);
    get_logger("monocular_slam.test").info("an informational message");
    flush_logging();
    EXPECT_NE(testing_utils::read_text(log_path).find("an informational message"), std::string::npos);
}

// py: test_cli.py::TestLogging::test_file_level_can_be_raised_independently
TEST_F(TestLogging, FileLevelCanBeRaisedIndependently) {
    TempDir tmp;
    const auto log_path = tmp / "run.log";
    testing_utils::CaptureOutput capture;  // the console handler still prints it
    setup_logging(LogLevel::Info, log_path, false, LogLevel::Error);
    get_logger("monocular_slam.test").info("should not be written");
    flush_logging();
    EXPECT_EQ(testing_utils::read_text(log_path), "");
}

// py: test_cli.py::TestLogging::test_repeated_setup_does_not_duplicate_handlers
TEST_F(TestLogging, RepeatedSetupDoesNotDuplicateHandlers) {
    TempDir tmp;
    for (int i = 0; i < 3; ++i) {
        setup_logging(LogLevel::Info, tmp / "run.log");
    }
    EXPECT_EQ(logging_handler_count(), 2);
}

// py: test_cli.py::TestLogging::test_quiet_suppresses_only_the_console
TEST_F(TestLogging, QuietSuppressesOnlyTheConsole) {
    TempDir tmp;
    setup_logging(LogLevel::Info, tmp / "run.log", true);
    EXPECT_EQ(logging_handler_count(), 1);
    EXPECT_TRUE(logging_has_only_file_handler());
}

// py: test_cli.py::TestLogging::test_get_logger_strips_the_package_prefix
TEST_F(TestLogging, GetLoggerStripsThePackagePrefix) {
    EXPECT_EQ(get_logger("monocular_slam.features.detector").name(), "monocular_slam.features.detector");
    EXPECT_EQ(get_logger("monocular_slam").name(), "monocular_slam");
}

TEST_F(TestLogging, RecordFormatMatchesPythonLogging) {
    // "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s" with %H:%M:%S.
    TempDir tmp;
    setup_logging(LogLevel::Info, tmp / "run.log", true);
    get_logger("monocular_slam.pipeline").warning("hello");
    flush_logging();
    const std::string line = testing_utils::read_text(tmp / "run.log");
    ASSERT_GE(line.size(), 11u);
    EXPECT_EQ(line.substr(8), " | WARNING | monocular_slam.pipeline      | hello\n");
    EXPECT_EQ(line[2], ':');
    EXPECT_EQ(line[5], ':');
}
