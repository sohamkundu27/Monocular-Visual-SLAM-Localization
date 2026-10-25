#include "mvslam/cli/apps.hpp"
namespace mvslam::cli {
int run_main(const std::vector<std::string>& argv, bool vo);
int run_vo_main(const std::vector<std::string>& argv) { return run_main(argv,true); }
}  // namespace mvslam::cli
