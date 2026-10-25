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

#include "mvslam/cli/common.hpp"
#include "mvslam/geometry/pose.hpp"
#include "support/kitti_fixture.hpp"
#include "support/process.hpp"
using namespace mvslam::cli;
using namespace testing_utils;
namespace {
class Cli : public ::testing::Test {
protected:
    TempDir tmp;
    Config config;
    std::filesystem::path root;
    void SetUp() override {
        root=kitti_fixture::write_kitti_sequence(tmp/"dataset","00",25);
        config=Config().with_overrides({{"dataset.path",root.string()},{"features.max_features",300},
            {"matcher.min_matches",8},{"odometry.min_inliers",6},{"odometry.min_inlier_ratio",0.05},
            {"loop_closure.enabled",false},{"output.plot_3d",false},{"output.root",(tmp/"outputs").string()},
            {"runtime.log_level","ERROR"},{"runtime.progress_interval",0},{"evaluation.rpe_deltas",{1,5}}});
        config.to_yaml(tmp/"test.yaml");
    }
    CommonArgs args(std::vector<std::string> extra={}) {
        std::vector<std::string> argv{"-c",(tmp/"test.yaml").string()}; argv.insert(argv.end(),extra.begin(),extra.end());
        CommonArgs a; const auto result=parse_run_slam_args(argv,a); EXPECT_TRUE(result.proceed); return a;
    }
    ProcessResult run(const std::string& exe=MVSLAM_EXE_RUN_SLAM,std::vector<std::string> extra={}) {
        std::vector<std::string> argv{"-c",(tmp/"test.yaml").string(),"-s","00","-o",(tmp/"run").string()};
        argv.insert(argv.end(),extra.begin(),extra.end()); return run_process(exe,argv);
    }
    Json metrics() { return py::loads(read_text(tmp/"run"/"metrics.json")); }
    std::vector<std::string> eval_args(const std::string& path) {
        return {path,"--sequence","00","--dataset-path",root.string(),"--log-level","ERROR"};
    }
};
}
// py: test_cli.py::TestParsers::test_help_renders[build_parser0]
// py: test_cli.py::TestParsers::test_help_renders[build_parser1]
// py: test_cli.py::TestParsers::test_help_renders[build_parser2]
TEST(CliParsers, Help) {
    for(const auto* exe:{MVSLAM_EXE_RUN_SLAM,MVSLAM_EXE_RUN_VO,MVSLAM_EXE_EVALUATE}) {
        const auto r=run_process(exe,{"--help"}); EXPECT_EQ(r.code,0)<<r.err;
        EXPECT_NE(r.out.find("usage:"),std::string::npos); EXPECT_TRUE(r.err.empty());
    }
}
// py: test_cli.py::TestParsers::test_slam_parser_accepts_the_documented_invocation
TEST(CliParsers, DocumentedFlags) {
    CommonArgs a; EXPECT_TRUE(parse_run_slam_args({"--sequence","00","--dataset-path","/data/kitti","--config","configs/kitti.yaml"},a).proceed);
    EXPECT_EQ(a.sequence,"00"); EXPECT_EQ(a.dataset_path,"/data/kitti");
}
// py: test_cli.py::TestParsers::test_short_flags_work
TEST(CliParsers, ShortFlags) {
    CommonArgs a; EXPECT_TRUE(parse_run_slam_args({"-s","05","-o","/tmp/out","-c","configs/kitti.yaml"},a).proceed);
    EXPECT_EQ(a.sequence,"05"); EXPECT_EQ(a.output_dir,"/tmp/out");
}
// py: test_cli.py::TestParsers::test_evaluate_requires_sequence_and_path
TEST(CliParsers, RequiredEvaluateFlags) { EXPECT_EQ(run_process(MVSLAM_EXE_EVALUATE,{"traj.txt"}).code,2); }
// py: test_cli.py::TestConfigFromArgs::test_flags_override_the_config_file
TEST_F(Cli, FlagsOverrideYaml) {
    const auto c=config_from_args(args({"-s","05","--max-frames","7"})); EXPECT_EQ(c.dataset.sequence,"05"); EXPECT_EQ(c.dataset.max_frames,7);
}
// py: test_cli.py::TestConfigFromArgs::test_set_override_reaches_nested_keys
TEST_F(Cli, NestedOverride) { EXPECT_EQ(config_from_args(args({"--set","odometry.min_inliers=42"})).odometry.min_inliers,42); }
// py: test_cli.py::TestConfigFromArgs::test_multiple_set_overrides
TEST_F(Cli, MultipleOverrides) {
    const auto c=config_from_args(args({"--set","odometry.min_inliers=42","--set","features.max_features=999"}));
    EXPECT_EQ(c.odometry.min_inliers,42); EXPECT_EQ(c.features.max_features,999);
}
// py: test_cli.py::TestConfigFromArgs::test_no_loop_closure_flag_disables_it
TEST_F(Cli, NoLoopFlag) { EXPECT_FALSE(config_from_args(args({"--no-loop-closure"})).loop_closure.enabled); }
// py: test_cli.py::TestConfigFromArgs::test_no_plots_flag_disables_plots
TEST_F(Cli, NoPlotsFlag) { EXPECT_FALSE(config_from_args(args({"--no-plots"})).output.save_plots); }
// py: test_cli.py::TestConfigFromArgs::test_missing_config_file_is_reported_clearly
TEST_F(Cli, MissingConfig) { EXPECT_THROW(config_from_args(args({"-c","/nonexistent/config.yaml"})),ConfigError); }
// py: test_cli.py::TestConfigFromArgs::test_malformed_set_is_rejected
TEST_F(Cli, MalformedSet) { EXPECT_THROW(config_from_args(args({"--set","nonsense"})),ConfigError); }
// py: test_cli.py::TestConfigFromArgs::test_unknown_set_key_is_rejected
TEST_F(Cli, UnknownKey) { EXPECT_THROW(config_from_args(args({"--set","nope.key=1"})),ConfigError); }
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[true-True]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[False-False]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[yes-True]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[off-False]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[42-42]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[-7--7]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[3.5-3.5]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[1e-3-0.001]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[none-None]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[null-None]
// py: test_cli.py::TestScalarParsing::test_parses_expected_types[magsac-magsac]
TEST(CliScalar, ExpectedTypes) {
    for(const auto& p:std::vector<std::pair<std::string,Json>>{{"true",true},{"False",false},{"yes",true},{"off",false},
        {"42",42},{"-7",-7},{"3.5",3.5},{"1e-3",0.001},{"none",nullptr},{"null",nullptr},{"magsac","magsac"}}) {
        EXPECT_EQ(parse_scalar(p.first),p.second)<<p.first; EXPECT_EQ(parse_scalar(p.first).type(),p.second.type());
    }
}
// py: test_cli.py::TestScalarParsing::test_leading_zero_stays_a_string
TEST(CliScalar, LeadingZero) { EXPECT_EQ(parse_scalar("00"),Json("00")); EXPECT_EQ(parse_scalar("05"),Json("05")); }
// py: test_cli.py::TestEndToEndCLI::test_run_slam_writes_results_and_exits_zero
TEST_F(Cli, SlamWritesResults) {
    const auto r=run(); EXPECT_EQ(r.code,0)<<r.err;
    EXPECT_TRUE(std::filesystem::is_regular_file(tmp/"run"/"metrics.json"));
    EXPECT_TRUE(std::filesystem::is_regular_file(tmp/"run"/"run.log"));
    EXPECT_NE(r.out.find("KITTI sequence 00 - results"),std::string::npos);
}
// py: test_cli.py::TestEndToEndCLI::test_run_slam_metrics_reflect_the_run
TEST_F(Cli, SlamMetrics) {
    ASSERT_EQ(run().code,0); const auto m=metrics(); EXPECT_EQ(m["sequence"],"00"); EXPECT_EQ(m["frames"],25); EXPECT_EQ(m["ground_truth_available"],true);
}
// py: test_cli.py::TestEndToEndCLI::test_run_vo_disables_loop_closure
TEST_F(Cli, VoDisablesLoops) {
    const auto r=run(MVSLAM_EXE_RUN_VO,{"--set","loop_closure.enabled=true"}); ASSERT_EQ(r.code,0)<<r.err;
    EXPECT_EQ(metrics()["loop_closure"]["enabled"],false); EXPECT_EQ(metrics()["loop_closures_detected"],0);
}
// py: test_cli.py::TestEndToEndCLI::test_max_frames_is_honoured
TEST_F(Cli, MaxFrames) { ASSERT_EQ(run(MVSLAM_EXE_RUN_SLAM,{"--max-frames","8"}).code,0); EXPECT_EQ(metrics()["frames"],8); }
// py: test_cli.py::TestEndToEndCLI::test_missing_sequence_exits_with_code_two
TEST_F(Cli, MissingSequence) {
    const auto r=run(MVSLAM_EXE_RUN_SLAM,{"-s","42"}); EXPECT_EQ(r.code,2); EXPECT_NE(r.err.find("Dataset error"),std::string::npos);
}
// py: test_cli.py::TestEndToEndCLI::test_bad_config_exits_with_code_two
TEST_F(Cli, BadConfig) {
    const auto r=run_process(MVSLAM_EXE_RUN_SLAM,{"-c","/nonexistent.yaml"}); EXPECT_EQ(r.code,2);
    EXPECT_NE(r.err.find("Configuration error"),std::string::npos);
}
// py: test_cli.py::TestEndToEndCLI::test_evaluate_scores_a_saved_trajectory
TEST_F(Cli, EvaluateTrajectory) {
    ASSERT_EQ(run().code,0); auto argv=eval_args((tmp/"run"/"trajectory_raw.txt").string());
    argv.insert(argv.end(),{"--json",(tmp/"eval.json").string(),"--rpe-deltas","1","5"});
    const auto r=run_process(MVSLAM_EXE_EVALUATE,argv); ASSERT_EQ(r.code,0)<<r.err;
    EXPECT_NE(r.out.find("ATE RMSE"),std::string::npos);
    EXPECT_TRUE(py::loads(read_text(tmp/"eval.json")).contains("trajectory_raw"));
}
// py: test_cli.py::TestEndToEndCLI::test_evaluate_rejects_a_length_mismatch
TEST_F(Cli, EvaluateLengthMismatch) {
    Trajectory(PoseList(3,Eigen::Matrix4d::Identity())).save_kitti(tmp/"short.txt");
    const auto r=run_process(MVSLAM_EXE_EVALUATE,eval_args((tmp/"short.txt").string()));
    EXPECT_EQ(r.code,2); EXPECT_NE(r.err.find("Length mismatch"),std::string::npos);
}
// py: test_cli.py::TestEndToEndCLI::test_evaluate_refuses_a_sequence_without_ground_truth
TEST_F(Cli, EvaluateNoGroundTruth) {
    root=kitti_fixture::write_kitti_sequence(tmp/"d","00",6,false);
    Trajectory(PoseList(6,Eigen::Matrix4d::Identity())).save_kitti(tmp/"t.txt");
    const auto r=run_process(MVSLAM_EXE_EVALUATE,eval_args((tmp/"t.txt").string()));
    EXPECT_EQ(r.code,2); EXPECT_NE(r.err.find("no ground-truth poses"),std::string::npos);
}
TEST_F(Cli, PrecedenceAndPythonIntegerFlags) {
    const auto c=config_from_args(args({"--seq","05","--max-frames","1_0","--set","dataset.sequence=00",
        "--set","loop_closure.enabled=true","--no-loop-closure","--set","output.save_plots=true","--no-plots"}));
    EXPECT_EQ(c.dataset.sequence,"00"); EXPECT_EQ(c.dataset.max_frames,10);
    EXPECT_FALSE(c.loop_closure.enabled); EXPECT_FALSE(c.output.save_plots);
    CommonArgs a; a.config=""; EXPECT_EQ(config_from_args(a).to_dict(),Config().to_dict());
}
TEST_F(Cli, InvalidChoicesAndRunFailureExitCodes) {
    for(const auto& argv:std::vector<std::vector<std::string>>{{"--log-level","nope"},{"--max-frames","1.5"},{"--unknown"},{"--set"}})
        EXPECT_EQ(run_process(MVSLAM_EXE_RUN_SLAM,argv).code,2);
    EXPECT_EQ(run(MVSLAM_EXE_RUN_SLAM,{"--max-frames","1"}).code,1);
    EXPECT_EQ(run_process(MVSLAM_EXE_RUN_VO,{"--no-loop-closure"}).code,2);
}
