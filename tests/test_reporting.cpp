#include "mvslam/reporting.hpp"
#include "support/test_utils.hpp"
using namespace mvslam;
using namespace testing_utils;
namespace {
Json full() {
    return {{"sequence","00"},{"frames",4541},{"distance_km",3.72},{"successful_pose_rate_pct",99.5},
        {"ate_rmse_raw_m",12.5},{"ate_rmse_optimized_m",5.0},{"ate_reduction_pct",60.0},
        {"translational_drift_raw_pct",2.2},{"translational_drift_optimized_pct",1.1},
        {"rotational_drift_deg_per_100m",0.5},{"loop_closures_detected",30},{"avg_features_per_frame",2980.0},
        {"avg_matches_per_pair",900.0},{"avg_inlier_ratio_pct",68.0},{"runtime_fps",14.5},
        {"scale_uses_ground_truth",true},{"alignment","sim3"},{"loop_closure",{{"n_keyframes",1595}}}};
}
Json no_gt() {
    Json m=full(); m["sequence"]="11"; m["frames"]=900;
    for(auto k:{"distance_km","ate_rmse_raw_m","ate_rmse_optimized_m","ate_reduction_pct",
        "translational_drift_raw_pct","translational_drift_optimized_pct","rotational_drift_deg_per_100m"}) m[k]=nullptr;
    m["successful_pose_rate_pct"]=98.0; m["loop_closures_detected"]=0; m["avg_features_per_frame"]=2900.0;
    m["avg_matches_per_pair"]=800.0; m["avg_inlier_ratio_pct"]=65.0; m["runtime_fps"]=15.0;
    m["scale_uses_ground_truth"]=false; m["loop_closure"]["n_keyframes"]=300; return m;
}
Json regressed() {
    Json m=full(); m["sequence"]="01"; m["ate_rmse_raw_m"]=16.2; m["ate_rmse_optimized_m"]=17.7;
    m["ate_reduction_pct"]=-9.0; m["loop_closures_detected"]=1; return m;
}
std::string join(const std::vector<std::string>& v) {
    std::string s; for(const auto& x:v) s+=x+" "; return s;
}
class Reporting : public ::testing::Test {
protected:
    TempDir tmp;
    std::filesystem::path old_figures=figure_dir();
    void SetUp() override { figure_dir()=tmp/"empty"; }
    void TearDown() override { figure_dir()=old_figures; }
    std::filesystem::path write_run(const Json& m=full()) {
        auto dir=tmp/("sequence_"+m.at("sequence").get<std::string>());
        write_text(dir/"metrics.json",py::dumps(m)); return dir;
    }
    std::vector<RunSummary> summaries(const Json& m=full()) { return {RunSummary::from_metrics(m)}; }
};
void contains(const std::string& text,const std::string& part) { EXPECT_NE(text.find(part),std::string::npos)<<part; }
void excludes(const std::string& text,const std::string& part) { EXPECT_EQ(text.find(part),std::string::npos)<<part; }
}
// py: test_reporting.py::TestLoading::test_loads_a_run
TEST_F(Reporting, LoadRun) {
    write_run(); const auto s=load_run_summaries(tmp.path()); ASSERT_EQ(s.size(),1u);
    EXPECT_EQ(s[0].sequence,"00"); EXPECT_EQ(s[0].ate_rmse_optimized_m,5.0);
}
// py: test_reporting.py::TestLoading::test_loads_several_runs_sorted_by_sequence
TEST_F(Reporting, SortedRuns) {
    auto m=full(); m["sequence"]="05"; write_run(m); write_run();
    const auto s=load_run_summaries(tmp.path()); ASSERT_EQ(s.size(),2u);
    EXPECT_EQ(s[0].sequence,"00"); EXPECT_EQ(s[1].sequence,"05");
}
// py: test_reporting.py::TestLoading::test_missing_directory_yields_nothing
TEST_F(Reporting, MissingDirectory) { EXPECT_TRUE(load_run_summaries(tmp/"nope").empty()); }
// py: test_reporting.py::TestLoading::test_corrupt_metrics_file_is_skipped
TEST_F(Reporting, CorruptFile) {
    write_run(); write_text(tmp/"sequence_99"/"metrics.json","{not json");
    EXPECT_EQ(load_run_summaries(tmp.path()).size(),1u);
}
// py: test_reporting.py::TestResultsTable::test_renders_measured_values
TEST_F(Reporting, MeasuredTable) {
    const auto s=results_table(summaries()); for(auto p:{"| 00 ","4541","5.00 m"}) contains(s,p);
}
// py: test_reporting.py::TestResultsTable::test_unmeasured_values_render_as_na
TEST_F(Reporting, NullTable) {
    const auto s=results_table(summaries(no_gt())); contains(s,"n/a"); excludes(s,"0.00 m");
}
// py: test_reporting.py::TestResultsTable::test_empty_input_explains_how_to_populate
TEST_F(Reporting, EmptyTable) { contains(results_table({}),"build/run_slam"); }
// py: test_reporting.py::TestHighlights::test_highlights_use_measured_numbers
TEST_F(Reporting, MeasuredHighlights) {
    const auto s=join(measured_highlights(summaries()));
    for(auto p:{"4,541 KITTI frames","3.72 km","5.00 m optimized ATE RMSE","60.0%","30 loop closures"}) contains(s,p);
}
// py: test_reporting.py::TestHighlights::test_no_ate_claim_without_ground_truth
TEST_F(Reporting, NoUnmeasuredAte) {
    const auto s=join(measured_highlights(summaries(no_gt()))); excludes(s,"ATE"); excludes(s,"drift");
}
// py: test_reporting.py::TestHighlights::test_no_loop_claim_when_none_were_found
TEST_F(Reporting, NoInventedLoops) { excludes(join(measured_highlights(summaries(no_gt()))),"loop closures"); }
// py: test_reporting.py::TestHighlights::test_empty_input_yields_no_highlights
TEST_F(Reporting, EmptyHighlights) { EXPECT_TRUE(measured_highlights({}).empty()); }
// py: test_reporting.py::TestResumeBullets::test_generates_at_most_three
TEST_F(Reporting, ThreeBullets) { EXPECT_LE(resume_bullets(summaries()).size(),3u); }
// py: test_reporting.py::TestResumeBullets::test_bullets_carry_the_scale_caveat
TEST_F(Reporting, BulletScaleCaveat) { contains(join(resume_bullets(summaries())),"scale"); }
// py: test_reporting.py::TestResumeBullets::test_no_scale_caveat_when_scale_was_not_borrowed
TEST_F(Reporting, NoBorrowedScale) {
    auto m=full(); m["scale_uses_ground_truth"]=false; excludes(join(resume_bullets(summaries(m))),"ground-truth-scaled");
}
// py: test_reporting.py::TestResumeBullets::test_no_accuracy_bullet_without_ground_truth
TEST_F(Reporting, NoAccuracyBullet) {
    const auto b=resume_bullets(summaries(no_gt())); EXPECT_FALSE(b.empty()); excludes(join(b),"absolute trajectory error");
}
// py: test_reporting.py::TestResumeBullets::test_no_improvement_claim_without_an_improvement
TEST_F(Reporting, NoImprovementClaim) {
    auto m=full(); m["ate_reduction_pct"]=nullptr; m["loop_closures_detected"]=0;
    excludes(join(resume_bullets(summaries(m))),"Cut absolute trajectory error");
}
// py: test_reporting.py::TestResumeBullets::test_empty_input_yields_no_bullets
TEST_F(Reporting, EmptyBullets) { EXPECT_TRUE(resume_bullets({}).empty()); }
// py: test_reporting.py::TestDocument::test_document_contains_all_sections
TEST_F(Reporting, DocumentSections) {
    const auto s=render_resume_metrics(summaries());
    for(auto p:{"# Resume metrics","## Results","## Measured highlights","## Suggested resume bullets","## Metric definitions"}) contains(s,p);
}
// py: test_reporting.py::TestDocument::test_scale_caveat_is_prominent
TEST_F(Reporting, DocumentCaveat) {
    const auto s=render_resume_metrics(summaries()); contains(s,"Scale caveat"); contains(s,"cannot recover absolute scale");
}
// py: test_reporting.py::TestDocument::test_no_runs_document_explains_itself
TEST_F(Reporting, EmptyDocument) {
    const auto s=render_resume_metrics({}); contains(s,"No runs found"); contains(s,"build/run_slam");
}
// py: test_reporting.py::TestDocument::test_write_creates_the_file
TEST_F(Reporting, WriteDocument) {
    write_run(); const auto p=write_resume_metrics(tmp.path()); ASSERT_TRUE(std::filesystem::is_regular_file(p));
    contains(read_text(p),"# Resume metrics");
}
// py: test_reporting.py::TestDocument::test_custom_destination
TEST_F(Reporting, CustomDestination) {
    write_run(); const auto p=write_resume_metrics(tmp.path(),tmp/"custom"/"out.md");
    EXPECT_EQ(p.filename(),"out.md"); EXPECT_TRUE(std::filesystem::is_regular_file(p));
}
// py: test_reporting.py::TestReadmeInjection::test_replaces_the_marked_block
TEST_F(Reporting, ReadmeBlock) {
    write_text(tmp/"README.md","# Title\n\n## Results\n\n<!-- RESULTS:START -->\nplaceholder\n<!-- RESULTS:END -->\n\n## Next\n");
    write_run(); EXPECT_TRUE(update_readme_results(tmp/"README.md",tmp.path()));
    const auto s=read_text(tmp/"README.md"); excludes(s,"placeholder"); contains(s,"| 00 ");
    EXPECT_EQ(s.substr(0,7),"# Title"); EXPECT_EQ(s.substr(s.size()-8),"## Next\n");
}
// py: test_reporting.py::TestReadmeInjection::test_injection_is_idempotent
TEST_F(Reporting, ReadmeIdempotent) {
    write_text(tmp/"README.md","a\n<!-- RESULTS:START -->\nx\n<!-- RESULTS:END -->\nb\n"); write_run();
    update_readme_results(tmp/"README.md",tmp.path()); const auto first=read_text(tmp/"README.md");
    update_readme_results(tmp/"README.md",tmp.path()); EXPECT_EQ(read_text(tmp/"README.md"),first);
}
// py: test_reporting.py::TestReadmeInjection::test_missing_markers_are_left_alone
TEST_F(Reporting, MissingMarkers) {
    write_text(tmp/"README.md","# No markers here\n"); EXPECT_FALSE(update_readme_results(tmp/"README.md",tmp.path()));
    EXPECT_EQ(read_text(tmp/"README.md"),"# No markers here\n");
}
// py: test_reporting.py::TestReadmeInjection::test_missing_readme_is_not_an_error
TEST_F(Reporting, MissingReadme) { EXPECT_FALSE(update_readme_results(tmp/"nope.md",tmp.path())); }
// py: test_reporting.py::TestReadmeInjection::test_scale_caveat_appears_in_the_block
TEST_F(Reporting, ReadmeCaveat) {
    write_text(tmp/"README.md","<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n"); write_run();
    update_readme_results(tmp/"README.md",tmp.path()); contains(read_text(tmp/"README.md"),"cannot recover absolute scale");
}
// py: test_reporting.py::TestReadmeInjection::test_repo_readme_has_the_markers
TEST_F(Reporting, RepositoryMarkers) {
    const auto s=read_text(std::filesystem::path(MVSLAM_SOURCE_DIR)/"README.md"); contains(s,kReadmeStart); contains(s,kReadmeEnd);
}
// py: test_reporting.py::TestRegressionReporting::test_regression_is_called_out
TEST_F(Reporting, Regression) {
    write_run(regressed()); const auto s=render_resume_metrics(load_run_summaries(tmp.path()));
    for(auto p:{"Where it does not help","Sequence 01","-9.0%"}) contains(s,p);
}
// py: test_reporting.py::TestRegressionReporting::test_no_note_when_everything_improved
TEST_F(Reporting, NoRegression) {
    write_run(); excludes(render_resume_metrics(load_run_summaries(tmp.path())),"Where it does not help");
}
// py: test_reporting.py::TestRegressionReporting::test_highlights_still_use_the_best_run
TEST_F(Reporting, BestHighlights) {
    write_run(); write_run(regressed()); const auto s=join(measured_highlights(load_run_summaries(tmp.path())));
    contains(s,"60.0%"); excludes(s,"-9.0");
}
// py: test_reporting.py::TestRegressionReporting::test_readme_block_also_carries_the_note
TEST_F(Reporting, ReadmeRegression) {
    write_run(regressed()); contains(render_readme_results(load_run_summaries(tmp.path())),"Where it does not help");
}
// py: test_reporting.py::TestFigurePublishing::test_copies_known_figures
TEST_F(Reporting, CopiesFigures) {
    const auto d=write_run(); for(const auto& f:published_figures()) write_text(d/f.first,"fake png bytes");
    write_text(d/"diagnostics.png","not published"); const auto w=publish_figures(tmp.path(),tmp/"docs");
    EXPECT_EQ(w.size(),3u); EXPECT_TRUE(std::filesystem::is_regular_file(tmp/"docs"/"sequence_00"/"trajectory_comparison.png"));
    EXPECT_FALSE(std::filesystem::exists(tmp/"docs"/"sequence_00"/"diagnostics.png"));
    for(const auto& p:w) EXPECT_EQ(read_text(p),"fake png bytes");
}
// py: test_reporting.py::TestFigurePublishing::test_missing_figures_are_skipped
TEST_F(Reporting, MissingFigures) { write_run(); EXPECT_TRUE(publish_figures(tmp.path(),tmp/"docs").empty()); }
// py: test_reporting.py::TestFigurePublishing::test_directories_without_metrics_are_ignored
TEST_F(Reporting, FiguresWithoutMetrics) {
    write_text(tmp/"sequence_99"/"trajectory_comparison.png","x"); EXPECT_TRUE(publish_figures(tmp.path(),tmp/"docs").empty());
}
// py: test_reporting.py::TestFigurePublishing::test_readme_embeds_published_figures
TEST_F(Reporting, EmbedFigures) {
    const auto d=write_run(); for(const auto& f:published_figures()) write_text(d/f.first,"x");
    figure_dir()=tmp/"docs"/"results"; publish_figures(tmp.path(),figure_dir());
    const auto s=render_readme_results(load_run_summaries(tmp.path())); contains(s,"!["); contains(s,"trajectory_comparison.png");
}
// py: test_reporting.py::TestFigurePublishing::test_readme_points_at_the_command_when_nothing_is_published
TEST_F(Reporting, PublishInstructions) {
    write_run(); contains(render_readme_results(load_run_summaries(tmp.path())),"--publish-figures");
}
