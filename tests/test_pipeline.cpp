#include <algorithm>
#include <cmath>
#include <numeric>

#include "mvslam/pipeline.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"
#include "support/golden.hpp"
#include "support/kitti_fixture.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using namespace testing_utils;

namespace {
class Pipeline : public ::testing::Test {
protected:
    TempDir tmp;
    Config config;
    void SetUp() override {
        config = Config().with_overrides({
            {"dataset.path", kitti_fixture::write_kitti_sequence(tmp / "dataset", "00", 40).string()},
            {"output.root", (tmp / "outputs").string()}, {"runtime.log_level", "ERROR"},
            {"runtime.progress_interval", 0}, {"features.max_features", 400},
            {"matcher.min_matches", 8}, {"odometry.min_inliers", 6},
            {"odometry.min_inlier_ratio", 0.05}, {"loop_closure.enabled", false},
            {"output.plot_3d", false}, {"evaluation.rpe_deltas", {1, 5}}});
    }
    SlamResult run() { return SlamPipeline(config).run(); }
};
Trajectory corrected(const Trajectory& dense, const std::vector<std::int64_t>& indices,
                     const Eigen::Matrix4d& shift) {
    PoseList poses;
    for (auto i : indices) poses.push_back(shift * dense[i]);
    return Trajectory(poses);
}
void expect_poses(const Trajectory& a, const Trajectory& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_ALLCLOSE(a[i], b[i], 1e-7, 1e-9);
}
}  // namespace

// py: test_pipeline.py::TestPipelineRun::test_produces_a_trajectory_of_the_right_length
TEST_F(Pipeline, ProducesValidTrajectory) {
    const auto r = run(); EXPECT_EQ(r.raw_trajectory.size(), 40u); EXPECT_TRUE(r.raw_trajectory.is_valid());
}
// py: test_pipeline.py::TestPipelineRun::test_records_one_diagnostic_per_transition
TEST_F(Pipeline, OneDiagnosticPerTransition) {
    const auto r = run(); ASSERT_EQ(r.odometry.diagnostics.size(), 39u);
    for (std::size_t i = 0; i < 39; ++i) EXPECT_EQ(r.odometry.diagnostics[i].index, i + 1);
}
// py: test_pipeline.py::TestPipelineRun::test_ground_truth_is_loaded_and_evaluated
TEST_F(Pipeline, GroundTruthEvaluation) {
    const auto r = run(); EXPECT_TRUE(r.has_ground_truth()); ASSERT_TRUE(r.evaluation_raw);
    EXPECT_TRUE(std::isfinite(r.evaluation_raw->ate_aligned.rmse));
}
// py: test_pipeline.py::TestPipelineRun::test_runs_without_ground_truth
TEST_F(Pipeline, NoGroundTruth) {
    config.dataset.path = kitti_fixture::write_kitti_sequence(tmp / "no_gt", "00", 30, false).string();
    const auto r = run(); EXPECT_FALSE(r.has_ground_truth()); EXPECT_FALSE(r.evaluation_raw);
    EXPECT_EQ(r.odometry.scale_strategy["uses_ground_truth"], false);
    const auto m = build_metrics(r); EXPECT_TRUE(m["ate_rmse_raw_m"].is_null());
}
// py: test_pipeline.py::TestPipelineRun::test_timing_is_recorded_for_each_stage
TEST_F(Pipeline, StageTimings) {
    const auto s = run().timer->summary();
    for (auto key : {"detect", "match", "pose"}) EXPECT_TRUE(s.contains(key));
    EXPECT_EQ(s["detect"]["calls"], 40);
}
// py: test_pipeline.py::TestPipelineRun::test_scale_source_none_is_honoured
TEST_F(Pipeline, UnitScale) {
    config.odometry.scale_source = "none"; const auto r = run();
    EXPECT_EQ(r.odometry.scale_strategy["strategy"], "unit");
    EXPECT_EQ(r.odometry.scale_strategy["uses_ground_truth"], false);
}
// py: test_pipeline.py::TestPipelineRun::test_keyframes_are_harvested_when_loop_closure_is_on
TEST_F(Pipeline, HarvestsKeyframes) {
    config.loop_closure.enabled = true; config.keyframes.every_n_frames = 3;
    const auto r = run(); EXPECT_GT(r.keyframe_frame_indices.size(), 5u);
    EXPECT_TRUE(std::is_sorted(r.keyframe_frame_indices.begin(), r.keyframe_frame_indices.end()));
}
// py: test_pipeline.py::TestPipelineRun::test_no_keyframes_harvested_when_loop_closure_is_off
TEST_F(Pipeline, DisabledHarvest) { EXPECT_TRUE(run().keyframe_frame_indices.empty()); }
// py: test_pipeline.py::TestPipelineRun::test_too_few_frames_raises_clearly
TEST_F(Pipeline, TooFewFrames) {
    config.dataset.path = kitti_fixture::write_kitti_sequence(tmp / "short", "00", 1).string();
    try { run(); FAIL(); } catch (const ValueError& e) {
        EXPECT_NE(std::string(e.what()).find("at least 2 frames"), std::string::npos);
    }
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_identity_correction_leaves_the_trajectory_alone
TEST(Propagation, Identity) {
    Trajectory d(synthetic::straight_drive(30)); std::vector<std::int64_t> ids{0,10,20};
    expect_poses(propagate_keyframe_corrections(d, ids, corrected(d, ids, Eigen::Matrix4d::Identity())), d);
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_correction_is_applied_exactly_at_keyframes
TEST(Propagation, ExactAtKeyframes) {
    Trajectory d(synthetic::straight_drive(30)); std::vector<std::int64_t> ids{0,10,20};
    const auto shift = se3_from_rt(Eigen::Matrix3d::Identity(), {5,0,0});
    const auto r = propagate_keyframe_corrections(d, ids, corrected(d, ids, shift));
    for (auto i : ids) EXPECT_ALLCLOSE(r[i], shift * d[i], 1e-7, 1e-9);
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_relative_motion_between_keyframes_is_preserved
TEST(Propagation, RelativeMotion) {
    golden::Stream rng("test_pipeline", "TestKeyframeCorrectionPropagation::test_relative_motion_between_keyframes_is_preserved");
    Trajectory d(synthetic::straight_drive(30)); std::vector<std::int64_t> ids{0,10,20}; PoseList poses;
    for (auto i : ids) {
        const auto v = rng.next_vec("normal");
        poses.push_back(se3_from_rt(Eigen::Matrix3d::Identity(), {v[0],v[1],v[2]}) * d[i]);
    }
    const auto r = propagate_keyframe_corrections(d, ids, Trajectory(poses));
    for (int i=11; i<20; ++i) EXPECT_ALLCLOSE(invert_se3(r[i-1])*r[i], invert_se3(d[i-1])*d[i], 1e-7, 1e-9);
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_frames_before_the_first_keyframe_use_the_first_correction
TEST(Propagation, BeforeFirst) {
    Trajectory d(synthetic::straight_drive(20)); std::vector<std::int64_t> ids{5,15};
    const auto shift = se3_from_rt(Eigen::Matrix3d::Identity(), {2,0,0});
    const auto r = propagate_keyframe_corrections(d, ids, corrected(d, ids, shift));
    EXPECT_ALLCLOSE(r[0], shift*d[0], 1e-7, 1e-9);
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_output_stays_valid_se3
TEST(Propagation, ValidSe3) {
    golden::Stream rng("test_pipeline", "TestKeyframeCorrectionPropagation::test_output_stays_valid_se3");
    Trajectory d(synthetic::straight_drive(25)); std::vector<std::int64_t> ids{0,12,24}; PoseList poses;
    for (auto i : ids) {
        const auto v = rng.next_vec("normal");
        poses.push_back(se3_from_rt(Eigen::Matrix3d::Identity(), {v[0],v[1],v[2]}) * d[i]);
    }
    EXPECT_TRUE(propagate_keyframe_corrections(d, ids, Trajectory(poses)).is_valid());
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_frame_ids_are_preserved
TEST(Propagation, FrameIds) {
    std::vector<std::int64_t> ids; for(int i=0;i<20;++i) ids.push_back(i*2);
    Trajectory d(synthetic::straight_drive(20), ids);
    EXPECT_EQ(propagate_keyframe_corrections(d, {0,10}, corrected(d,{0,10},Eigen::Matrix4d::Identity())).frame_ids(), ids);
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_no_keyframes_returns_a_copy
TEST(Propagation, Empty) {
    Trajectory d(synthetic::straight_drive(10)); auto r = propagate_keyframe_corrections(d, {}, Trajectory());
    expect_poses(r,d); EXPECT_NE(r.poses().data(), d.poses().data());
}
// py: test_pipeline.py::TestKeyframeCorrectionPropagation::test_count_mismatch_raises
TEST(Propagation, CountMismatch) {
    Trajectory d(synthetic::straight_drive(10));
    EXPECT_THROW(propagate_keyframe_corrections(d, {0,5}, Trajectory(PoseList{d[0]})), ValueError);
}
// py: test_pipeline.py::TestMetrics::test_metrics_are_json_serialisable
TEST_F(Pipeline, SerializableMetrics) { EXPECT_NO_THROW(py::loads(py::dumps(build_metrics(run()), 2))); }
// py: test_pipeline.py::TestMetrics::test_resume_keys_are_all_present
TEST_F(Pipeline, HeadlineSchema) {
    const auto m=build_metrics(run());
    for(auto k:{"sequence","frames","distance_km","successful_pose_rate_pct","ate_rmse_raw_m",
        "ate_rmse_optimized_m","ate_reduction_pct","translational_drift_raw_pct","translational_drift_optimized_pct",
        "rotational_drift_deg_per_100m","loop_closures_detected","avg_features_per_frame","avg_matches_per_pair",
        "avg_inlier_ratio_pct","runtime_fps"}) EXPECT_TRUE(m.contains(k)) << k;
}
// py: test_pipeline.py::TestMetrics::test_unmeasurable_values_are_null_not_invented
TEST_F(Pipeline, MissingMetricsAreNull) {
    const auto m=build_metrics(run()); EXPECT_EQ(m["loop_closures_detected"],0);
    for(auto k:{"ate_rmse_optimized_m","ate_reduction_pct","translational_drift_optimized_pct"}) EXPECT_TRUE(m[k].is_null());
}
// py: test_pipeline.py::TestMetrics::test_ground_truth_scale_use_is_flagged
TEST_F(Pipeline, GroundTruthScaleFlag) {
    const auto m=build_metrics(run()); EXPECT_EQ(m["scale_source"],"ground_truth"); EXPECT_EQ(m["scale_uses_ground_truth"],true);
}
// py: test_pipeline.py::TestMetrics::test_scale_free_run_is_flagged_as_such
TEST_F(Pipeline, ScaleFreeFlag) {
    config.odometry.scale_source="none"; EXPECT_EQ(build_metrics(run())["scale_uses_ground_truth"],false);
}
// py: test_pipeline.py::TestMetrics::test_frame_and_distance_counts_are_real
TEST_F(Pipeline, RealFrameDistanceCounts) {
    const auto r=run(); const auto m=build_metrics(r); EXPECT_EQ(m["frames"],40);
    EXPECT_APPROX(py::as_double(m["distance_km"]), r.ground_truth->path_length()/1000.0, 1e-3);
}
// py: test_pipeline.py::TestMetrics::test_timing_block_is_populated
TEST_F(Pipeline, TimingMetrics) {
    const auto t=build_metrics(run())["timing"]; EXPECT_GT(py::as_double(t["total_runtime_s"]),0);
    EXPECT_GT(py::as_double(t["fps"]),0); EXPECT_TRUE(t["stages"].contains("detect"));
}
// py: test_pipeline.py::TestOutputs::test_writes_the_expected_artifacts
TEST_F(Pipeline, OutputArtifacts) {
    auto r=run(); const auto a=write_outputs(r,tmp/"run");
    for(auto f:{"metrics.json","diagnostics.csv","config.resolved.yaml","trajectory_raw.txt","trajectory_raw.png",
                "trajectory_comparison.png","loop_closures.png","trajectory_raw_data.csv"})
        EXPECT_TRUE(std::filesystem::is_regular_file(tmp/"run"/f)) << f;
    EXPECT_TRUE(a.contains("metrics")); EXPECT_EQ(r.artifacts,a);
}
// py: test_pipeline.py::TestOutputs::test_diagnostics_csv_has_one_row_per_transition
TEST_F(Pipeline, DiagnosticsCsv) {
    auto r=run(); write_outputs(r,tmp/"run"); const auto s=read_text(tmp/"run"/"diagnostics.csv");
    EXPECT_EQ(std::count(s.begin(),s.end(),'\n'),40); EXPECT_EQ(std::count(s.begin(),s.end(),'\r'),40);
    const auto header=s.substr(0,s.find('\n'));
    for(auto k:{"index","frame_id","status","n_matches","n_ransac_inliers"}) EXPECT_NE(header.find(k),std::string::npos);
}
// py: test_pipeline.py::TestOutputs::test_saved_trajectory_round_trips
TEST_F(Pipeline, TrajectoryRoundTrip) {
    auto r=run(); write_outputs(r,tmp/"run");
    EXPECT_ALLCLOSE(Trajectory::load_kitti(tmp/"run"/"trajectory_raw.txt").positions(),r.raw_trajectory.positions(),1e-7,1e-6);
}
// py: test_pipeline.py::TestOutputs::test_resolved_config_can_be_reloaded
TEST_F(Pipeline, ConfigRoundTrip) {
    auto r=run(); write_outputs(r,tmp/"run"); const auto c=Config::from_yaml(tmp/"run"/"config.resolved.yaml");
    EXPECT_EQ(c.dataset.sequence,config.dataset.sequence); EXPECT_EQ(c.odometry.min_inliers,config.odometry.min_inliers);
}
// py: test_pipeline.py::TestOutputs::test_plots_can_be_disabled
TEST_F(Pipeline, DisablePlots) {
    config.output.save_plots=false; auto r=run(); write_outputs(r,tmp/"run");
    EXPECT_FALSE(std::filesystem::exists(tmp/"run"/"trajectory_raw.png"));
    EXPECT_TRUE(std::filesystem::is_regular_file(tmp/"run"/"metrics.json"));
}
// py: test_pipeline.py::TestOutputs::test_run_pipeline_writes_a_log
TEST_F(Pipeline, RunLog) {
    run_pipeline(config,tmp/"run"); EXPECT_GT(std::filesystem::file_size(tmp/"run"/"run.log"),0u);
}
// py: test_pipeline.py::TestLoopClosureIntegration::test_pipeline_survives_a_sequence_with_no_loops
TEST_F(Pipeline, NoLoops) {
    config.loop_closure.enabled=true; config.keyframes.every_n_frames=2;
    config.loop_closure.min_keyframe_separation=5; config.loop_closure.vocabulary_size=32;
    const auto r=run(); EXPECT_TRUE(r.loop_closures.empty()); EXPECT_FALSE(r.optimized_trajectory); EXPECT_FALSE(r.evaluation_optimized);
}
// py: test_pipeline.py::TestLoopClosureIntegration::test_short_sequence_skips_loop_closure_gracefully
TEST_F(Pipeline, TooShortForLoops) {
    config.loop_closure.enabled=true; config.loop_closure.min_keyframe_separation=10000;
    EXPECT_TRUE(run().loop_closures.empty());
}
