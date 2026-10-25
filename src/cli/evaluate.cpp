#include "mvslam/cli/apps.hpp"
#include "parser.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/evaluation/metrics.hpp"
#include "mvslam/utils/logging.hpp"
#include <fstream>
namespace mvslam::cli {
int evaluate_main(const std::vector<std::string>& argv) {
    CLI::App app("Evaluate KITTI-format trajectory files against ground truth.\n"
                 "Trajectories are 12-value-per-row KITTI pose files, as written by run_slam.","evaluate");
    std::string trajectory,sequence,dataset_path,alignment="sim3",level="INFO";
    std::optional<std::string> compare,json_path;
    std::vector<std::int64_t> deltas{1,10,100};
    app.add_option("trajectory",trajectory,"KITTI-format trajectory file to evaluate")->required();
    app.add_option("--compare",compare,"Second trajectory to compare against, enabling ATE reduction")->take_last();
    app.add_option("--sequence,-s",sequence,"KITTI sequence id")->required()->take_last();
    app.add_option("--dataset-path",dataset_path,"Root of the KITTI odometry download")->required()->take_last();
    app.add_option("--alignment",alignment,"Umeyama alignment mode (default: sim3)")
        ->check(CLI::IsMember({"sim3","se3","none"}))->take_last();
    app.add_option_function<std::vector<std::string>>("--rpe-deltas",[&](const std::vector<std::string>& values){
        deltas.clear(); for(const auto& value:values) deltas.push_back(integer(value));
    },"Frame gaps for relative pose error (default: 1 10 100)")->expected(1,-1);
    app.add_option("--json",json_path,"Write results to this JSON file")->take_last(); log_option(app,level);
    const auto parsed=parse(app,argv); if(!parsed.proceed) return parsed.exit_code;
    setup_logging(parse_log_level(level));
    std::optional<KittiOdometryDataset> dataset;
    try { dataset=KittiOdometryDataset::from_config(Config().with_overrides({{"dataset.path",dataset_path},{"dataset.sequence",sequence}})); }
    catch(const ConfigError& e) { std::cerr<<"Dataset error: "<<e.what()<<'\n'; return 2; }
    catch(const DatasetError& e) { std::cerr<<"Dataset error: "<<e.what()<<'\n'; return 2; }
    const auto& gt=dataset->ground_truth();
    if(!gt) {
        std::cerr<<"Sequence "<<sequence<<" has no ground-truth poses (sequences 11-21 are the held-out test split); cannot evaluate.\n";
        return 2;
    }
    Trajectory estimated;
    try { estimated=Trajectory::load_kitti(trajectory); }
    catch(const OSError& e) { std::cerr<<"Could not read trajectory: "<<e.what()<<'\n'; return 2; }
    if(estimated.size()!=gt->size()) {
        std::cerr<<"Length mismatch: trajectory has "<<estimated.size()<<" poses, ground truth has "<<gt->size()
                 <<". Was the run truncated with --max-frames?\n"; return 2;
    }
    Json results=Json::object();
    const auto primary=evaluate_trajectory(estimated,*gt,std::filesystem::path(trajectory).stem().string(),alignment,deltas);
    results[primary.label]=primary.to_dict(); std::cout<<primary.summary_line()<<'\n';
    if(compare && !compare->empty()) {
        const auto other=Trajectory::load_kitti(*compare);
        if(other.size()!=gt->size()) { std::cerr<<"Comparison trajectory length does not match ground truth\n"; return 2; }
        const auto secondary=evaluate_trajectory(other,*gt,std::filesystem::path(*compare).stem().string(),alignment,deltas);
        results[secondary.label]=secondary.to_dict(); std::cout<<secondary.summary_line()<<'\n';
        const double reduction=ate_reduction_pct(primary.ate_aligned.rmse,secondary.ate_aligned.rmse);
        results["ate_reduction_pct"]=py::make_float(py::round(reduction,3));
        std::cout<<"ATE reduction: "<<py::format(reduction,".2f")<<"%\n";
    }
    if(json_path && !json_path->empty()) {
        const std::filesystem::path path=*json_path;
        if(path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path); out<<py::dumps(results,2); std::cout<<"Wrote "<<path.string()<<'\n';
    }
    return 0;
}
}  // namespace mvslam::cli
