#include "mvslam/cli/common.hpp"
#include <algorithm>
#include <cctype>
#include "parser.hpp"

namespace mvslam::cli {
namespace {
std::string strip(const std::string& s) {
    const auto first=s.find_first_not_of(" \t\n\r\f\v");
    return first==std::string::npos ? "" : s.substr(first,s.find_last_not_of(" \t\n\r\f\v")-first+1);
}
ParseOutcome parse_common(const std::vector<std::string>& argv, CommonArgs& args, bool slam) {
    CLI::App app(slam ?
        "Run monocular visual SLAM on a KITTI Odometry sequence.\n"
        "Estimates camera motion, detects loop closures, optimizes the keyframe pose graph\n"
        "with GTSAM, and evaluates against ground truth. Monocular vision cannot recover\n"
        "absolute scale: by default translation magnitudes come from ground truth.\n"
        "Use --set odometry.scale_source=none for a scale-free run with sim3 alignment." :
        "Run monocular visual odometry (front end only) on a KITTI Odometry sequence.\n"
        "Loop closure and pose-graph optimization are disabled.", slam ? "run_slam" : "run_vo");
    app.add_option("--sequence,-s",args.sequence,"KITTI sequence id, e.g. 00 (default: from config)")->take_last();
    app.add_option("--dataset-path",args.dataset_path,"Root of the KITTI odometry download (holding sequences/)")->take_last();
    int_option(app,"--max-frames",args.max_frames,"Process at most this many frames");
    int_option(app,"--start-frame",args.start_frame,"Skip this many frames at the start");
    int_option(app,"--frame-step",args.frame_step,"Process every Nth frame");
    app.add_option("--config,-c",args.config,"YAML config file (default: configs/kitti.yaml)")->take_last();
    app.add_option("--output-dir,-o",args.output_dir,"Directory for run artefacts (default: <output.root>/sequence_<seq>)")->take_last();
    int_option(app,"--seed",args.seed,"Random seed");
    app.add_option("--log-level",args.log_level,"Console log level")
        ->check(CLI::IsMember({"DEBUG","INFO","WARNING","ERROR"}))->take_last();
    app.add_option("--set",args.overrides,"Override any config key, e.g. --set odometry.min_inliers=40 (repeatable)")
        ->expected(1)->allow_extra_args(false)->take_all();
    if(slam) app.add_flag("--no-loop-closure",args.no_loop_closure,"Disable loop closure detection and pose-graph optimization");
    app.add_flag("--no-plots",args.no_plots,"Skip figure rendering");
    return parse(app,argv);
}
}
ParseOutcome parse_run_slam_args(const std::vector<std::string>& argv, CommonArgs& args) { return parse_common(argv,args,true); }
ParseOutcome parse_run_vo_args(const std::vector<std::string>& argv, CommonArgs& args) { return parse_common(argv,args,false); }
Json parse_scalar(const std::string& text) {
    std::string lower=text;
    std::transform(lower.begin(),lower.end(),lower.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    if(lower=="true" || lower=="yes" || lower=="on") return true;
    if(lower=="false" || lower=="no" || lower=="off") return false;
    if(lower=="none" || lower=="null" || lower.empty()) return nullptr;
    const auto first=text.find_first_not_of('-');
    const auto digits=first==std::string::npos ? "" : text.substr(first);
    if(!digits.empty() && std::all_of(digits.begin(),digits.end(),[](unsigned char c){return c>='0' && c<='9';})) {
        if(digits.size()>1 && digits.front()=='0') return text;
        if(auto n=py::parse_int(text)) return *n;
        throw ValueError("invalid literal for int() with base 10: " + py::repr_str(text));
    }
    if(auto value=py::parse_float(text)) return py::make_float(*value);
    return text;
}
Config config_from_args(const CommonArgs& args) {
    std::optional<std::filesystem::path> path;
    if(!args.config.empty()) path=args.config;
    if(path && !std::filesystem::is_regular_file(*path))
        throw ConfigError("Config file not found: "+path->string()+"\nPass --config <path>, or run from the repository root where "+kDefaultConfig+" lives.");
    Json overrides=Json::object();
    if(args.sequence) overrides["dataset.sequence"]=*args.sequence;
    if(args.dataset_path) overrides["dataset.path"]=*args.dataset_path;
    if(args.max_frames) overrides["dataset.max_frames"]=*args.max_frames;
    if(args.start_frame) overrides["dataset.start_frame"]=*args.start_frame;
    if(args.frame_step) overrides["dataset.frame_step"]=*args.frame_step;
    if(args.seed) overrides["runtime.seed"]=*args.seed;
    if(args.log_level) overrides["runtime.log_level"]=*args.log_level;
    for(const auto& item:args.overrides) {
        const auto eq=item.find('=');
        if(eq==std::string::npos) throw ConfigError("--set expects KEY=VALUE, got '"+item+"'");
        overrides[strip(item.substr(0,eq))]=parse_scalar(strip(item.substr(eq+1)));
    }
    // Negation flags have final precedence, including over --set, as in Python.
    if(args.no_loop_closure) overrides["loop_closure.enabled"]=false;
    if(args.no_plots) overrides["output.save_plots"]=false;
    return Config::load(path,overrides);
}
std::filesystem::path resolve_output_dir(const CommonArgs& args,const Config& config) {
    return args.output_dir && !args.output_dir->empty() ? std::filesystem::path(*args.output_dir) : config.run_dir();
}
}  // namespace mvslam::cli
