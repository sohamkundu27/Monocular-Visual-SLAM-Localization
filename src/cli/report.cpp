#include "mvslam/cli/apps.hpp"
#include "parser.hpp"
#include "mvslam/reporting.hpp"
#include "mvslam/utils/logging.hpp"
namespace mvslam::cli {
int report_main(const std::vector<std::string>& argv) {
    CLI::App app("Generate outputs/resume_metrics.md from the metrics.json files that completed runs produced.\n"
        "Only measured values are used: unavailable metrics render as n/a and never become claims.","report");
    std::string outputs="outputs",readme="README.md",level="INFO";
    std::optional<std::string> output; bool print=false,update=false,publish=false;
    app.add_option("--outputs",outputs,"Directory containing sequence_XX/metrics.json (default: outputs)")->take_last();
    app.add_option("--output,-o",output,"Destination file (default: <outputs>/resume_metrics.md)")->take_last();
    app.add_flag("--print",print,"Also print the document");
    app.add_flag("--update-readme",update,"Inject the results table into README.md between its RESULTS markers");
    app.add_option("--readme",readme,"README path (default: README.md)")->take_last();
    app.add_flag("--publish-figures",publish,"Copy result figures into docs/results/ for display on GitHub");
    log_option(app,level); const auto parsed=parse(app,argv); if(!parsed.proceed) return parsed.exit_code;
    setup_logging(parse_log_level(level)); const auto summaries=load_run_summaries(outputs);
    if(summaries.empty()) std::cerr<<"No metrics.json files found under '"<<outputs<<"'. Run the pipeline first:\n"
        "  build/run_slam --sequence 00 --dataset-path /path/to/KITTI/dataset\n";
    if(publish) std::cout<<"Published "<<publish_figures(outputs).size()<<" figure(s) to docs/results/\n";
    const auto path=write_resume_metrics(outputs,output ? std::optional<std::filesystem::path>(*output) : std::nullopt);
    std::cout<<"Wrote "<<path.string()<<" from "<<summaries.size()<<" run(s)\n";
    if(update && update_readme_results(readme,outputs)) std::cout<<"Updated results block in "<<readme<<'\n';
    if(print) std::cout<<'\n'<<render_resume_metrics(summaries)<<'\n';
    return summaries.empty()?1:0;
}
}  // namespace mvslam::cli
