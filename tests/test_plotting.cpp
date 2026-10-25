#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include "mvslam/evaluation/plotting.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "support/golden.hpp"
#include "support/synthetic.hpp"
#include "support/test_utils.hpp"
using namespace mvslam;
using namespace mvslam::plot;
using namespace testing_utils;
namespace {
class Plotting : public ::testing::Test {
protected:
    TempDir tmp;
    Trajectory trajectory{synthetic::straight_drive(60)};
    void image(const std::filesystem::path& p) {
        ASSERT_TRUE(std::filesystem::is_regular_file(p)); EXPECT_GT(std::filesystem::file_size(p),1000u);
        EXPECT_FALSE(cv::imread(p.string()).empty());
        ASSERT_TRUE(std::filesystem::is_regular_file(data_path_for(p)));
        EXPECT_EQ(read_text(data_path_for(p)).substr(0,10),"series,x,y");
    }
    Trajectory noisy(const std::string& node) {
        golden::Stream rng("test_plotting",node);
        auto poses=trajectory.poses(); const auto noise=rng.next_mat("normal");
        for(std::size_t i=0;i<poses.size();++i) poses[i].topRightCorner<3,1>()+=noise.row(i).transpose();
        return Trajectory(poses);
    }
    std::vector<FrameDiagnostics> diagnostics(const std::string& node) {
        golden::Stream rng("test_plotting",node); std::vector<FrameDiagnostics> ds;
        for(int i=0;i<120;++i) {
            FrameDiagnostics d; d.index=i; d.frame_id=i; d.status=i%17?"ok":"failed";
            d.n_keypoints=rng.next_int("integers"); d.n_matches=rng.next_int("integers");
            d.n_ransac_inliers=rng.next_int("integers"); d.ransac_inlier_ratio=py::as_double(rng.next("uniform"));
            d.inlier_ratio=py::as_double(rng.next("uniform")); d.rotation_deg=py::as_double(rng.next("uniform")); ds.push_back(d);
        }
        return ds;
    }
    std::vector<double> ramp(int n,double end) {
        std::vector<double> v; for(int i=0;i<n;++i) v.push_back(i*end/(n-1)); return v;
    }
};
}
// py: test_plotting.py::TestTrajectoryPlots::test_single_trajectory
TEST_F(Plotting, SingleTrajectory) {
    image(plot_trajectory(trajectory,tmp/"t.png"));
    const auto csv=read_text(tmp/"t_data.csv");
    EXPECT_EQ(std::count(csv.begin(),csv.end(),'\n'),63);
    const auto p=trajectory.positions();
    EXPECT_NE(csv.find("Estimated,"+py::repr(p(30,0))+","+py::repr(p(30,2))),std::string::npos);
}
// py: test_plotting.py::TestTrajectoryPlots::test_with_reference
TEST_F(Plotting, Reference) {
    image(plot_trajectory(noisy("TestTrajectoryPlots::test_with_reference"),tmp/"t.png",trajectory));
}
// py: test_plotting.py::TestTrajectoryPlots::test_comparison_of_several
TEST_F(Plotting, Comparison) {
    image(plot_trajectory_comparison({{"Raw VO",noisy("TestTrajectoryPlots::test_comparison_of_several")},
        {"Optimized",trajectory}},tmp/"c.png",trajectory));
}
// py: test_plotting.py::TestTrajectoryPlots::test_comparison_without_reference
TEST_F(Plotting, ComparisonNoReference) { image(plot_trajectory_comparison({{"a",trajectory}},tmp/"c.png")); }
// py: test_plotting.py::TestTrajectoryPlots::test_3d_plot
TEST_F(Plotting, ThreeD) {
    const auto n=noisy("TestTrajectoryPlots::test_3d_plot");
    image(plot_trajectory_3d({{"Raw VO",n}},tmp/"t3.png",trajectory));
    const auto p=n.positions();
    EXPECT_NE(read_text(tmp/"t3_data.csv").find("Raw VO,"+py::repr(p(20,0))+","+py::repr(p(20,2))+","+py::repr(-p(20,1))),std::string::npos);
}
// py: test_plotting.py::TestTrajectoryPlots::test_creates_missing_directories
TEST_F(Plotting, CreatesDirectories) { image(plot_trajectory(trajectory,tmp/"deep"/"nested"/"t.png")); }
// py: test_plotting.py::TestTrajectoryPlots::test_single_pose_trajectory_does_not_crash
TEST_F(Plotting, SinglePose) { image(plot_trajectory(Trajectory(PoseList{Eigen::Matrix4d::Identity()}),tmp/"one.png")); }
// py: test_plotting.py::TestLoopClosurePlot::test_draws_edges
TEST_F(Plotting, LoopEdges) {
    image(plot_loop_closures(trajectory,{{5,40},{10,50}},tmp/"l.png"));
    EXPECT_NE(read_text(tmp/"l_data.csv").find("loop_1,"),std::string::npos);
}
// py: test_plotting.py::TestLoopClosurePlot::test_empty_loop_list_is_fine
TEST_F(Plotting, EmptyLoops) { image(plot_loop_closures(trajectory,{},tmp/"l.png")); }
// py: test_plotting.py::TestLoopClosurePlot::test_out_of_range_indices_are_skipped
TEST_F(Plotting, OutOfRangeLoops) {
    image(plot_loop_closures(trajectory,{{0,9999}},tmp/"l.png"));
    EXPECT_EQ(read_text(tmp/"l_data.csv").find("loop_"),std::string::npos);
}
// py: test_plotting.py::TestDiagnosticPlots::test_error_over_time
TEST_F(Plotting, ErrorTime) { image(plot_error_over_time({{"Raw",ramp(100,5)},{"Optimized",ramp(100,2)}},tmp/"e.png")); }
// py: test_plotting.py::TestDiagnosticPlots::test_error_against_distance
TEST_F(Plotting, ErrorDistance) { image(plot_error_over_time({{"Raw",ramp(100,5)}},tmp/"e.png",ramp(100,90))); }
// py: test_plotting.py::TestDiagnosticPlots::test_front_end_diagnostics
TEST_F(Plotting, Diagnostics) {
    image(plot_diagnostics(diagnostics("TestDiagnosticPlots::test_front_end_diagnostics"),tmp/"d.png"));
}
// py: test_plotting.py::TestDiagnosticPlots::test_empty_diagnostics_raise
TEST_F(Plotting, EmptyDiagnostics) { EXPECT_THROW(plot_diagnostics({},tmp/"d.png"),ValueError); }
// py: test_plotting.py::TestResourceHygiene::test_figures_are_closed
TEST_F(Plotting, ResourcesReleased) {
    const auto ds=diagnostics("TestResourceHygiene::test_figures_are_closed");
    const auto fd_count=[] { return std::distance(std::filesystem::directory_iterator("/proc/self/fd"),std::filesystem::directory_iterator{}); };
    const auto before=fd_count();
    for(int i=0;i<3;++i) {
        image(plot_trajectory(trajectory,tmp/"a.png"));
        image(plot_trajectory_comparison({{"x",trajectory}},tmp/"b.png"));
        image(plot_trajectory_3d({{"x",trajectory}},tmp/"c.png"));
        image(plot_loop_closures(trajectory,{{1,2}},tmp/"d.png"));
        image(plot_error_over_time({{"x",ramp(10,9)}},tmp/"e.png"));
        image(plot_diagnostics(ds,tmp/"f.png"));
    }
    EXPECT_EQ(fd_count(),before); // Canvas buffers are automatic cv::Mat owners; no global figure pool.
}
TEST_F(Plotting, CsvEscapesLabelsAndRejectsShortDistances) {
    image(plot_error_over_time({{"raw, \"VO\"",{1,2}}},tmp/"e.png"));
    EXPECT_NE(read_text(tmp/"e_data.csv").find("\"raw, \"\"VO\"\"\""),std::string::npos);
    EXPECT_THROW(plot_error_over_time({{"raw",{1,2}}},tmp/"e.png",std::vector<double>{0}),ValueError);
}
