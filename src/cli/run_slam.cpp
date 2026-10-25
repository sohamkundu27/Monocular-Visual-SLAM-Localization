#include "mvslam/cli/apps.hpp"
#include "mvslam/cli/common.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/pipeline.hpp"
#include <algorithm>
#include <iostream>

namespace mvslam::cli {
namespace {
void print_summary(const SlamResult& result) {
    const auto m=build_metrics(result);
    auto field=[&](const char* label,const char* key,const char* unit="") {
        std::cout<<label<<py::str(m.at(key))<<unit<<'\n';
    };
    std::cout<<'\n'<<std::string(66,'=')<<"\n  KITTI sequence "<<py::str(m["sequence"])<<" - results\n"<<std::string(66,'=')<<'\n';
    field("  frames processed            ","frames");
    field("  distance travelled          ","distance_km"," km");
    field("  successful pose rate        ","successful_pose_rate_pct"," %");
    field("  loop closures detected      ","loop_closures_detected");
    field("  runtime                     ","runtime_fps"," FPS");
    std::cout<<std::string(66,'-')<<'\n';
    if(!m["ate_rmse_raw_m"].is_null()) {
        field("  ATE RMSE (raw VO)           ","ate_rmse_raw_m"," m");
        field("  ATE RMSE (optimized)        ","ate_rmse_optimized_m"," m");
        field("  ATE reduction               ","ate_reduction_pct"," %");
        field("  translational drift (raw)   ","translational_drift_raw_pct"," %");
        field("  translational drift (opt)   ","translational_drift_optimized_pct"," %");
        field("  rotational drift            ","rotational_drift_deg_per_100m"," deg/100 m");
    } else std::cout<<"  no ground truth for this sequence: trajectory metrics unavailable\n";
    if(m["scale_uses_ground_truth"].get<bool>()) {
        std::cout<<std::string(66,'-')<<"\n  NOTE: translation magnitudes came from ground truth\n"
            "        (odometry.scale_source=ground_truth). Absolute scale is\n"
            "        NOT recovered from the images. See README.\n";
    }
    std::cout<<std::string(66,'=')<<"\n\n";
    std::vector<std::string> keys;
    for(const auto& item:result.artifacts.items()) keys.push_back(item.key());
    std::sort(keys.begin(),keys.end());
    for(const auto& key:keys) std::cout<<"  "<<py::ljust(key,24)<<' '<<result.artifacts[key].get<std::string>()<<'\n';
    std::cout<<'\n';
}
}
int run_main(const std::vector<std::string>& argv,bool vo) {
    CommonArgs args; const auto parsed=vo?parse_run_vo_args(argv,args):parse_run_slam_args(argv,args);
    if(!parsed.proceed) return parsed.exit_code;
    Config config;
    try { config=config_from_args(args); }
    catch(const ConfigError& e) { std::cerr<<"Configuration error: "<<e.what()<<'\n'; return 2; }
    if(vo) config.loop_closure.enabled=false; // VO is odometry by definition, even with --set.
    try { print_summary(run_pipeline(config,resolve_output_dir(args,config))); }
    catch(const DatasetError& e) { std::cerr<<"Dataset error: "<<e.what()<<'\n'; return 2; }
    catch(const ValueError& e) { std::cerr<<"Run failed: "<<e.what()<<'\n'; return 1; }
    catch(const RuntimeError& e) { std::cerr<<"Run failed: "<<e.what()<<'\n'; return 1; }
    return 0;
}
int run_slam_main(const std::vector<std::string>& argv) { return run_main(argv,false); }
}  // namespace mvslam::cli
