#pragma once
#include <string>
#include <vector>
namespace mvslam::cli {
int run_slam_main(const std::vector<std::string>& argv);
int run_vo_main(const std::vector<std::string>& argv);
int evaluate_main(const std::vector<std::string>& argv);
int report_main(const std::vector<std::string>& argv);
}  // namespace mvslam::cli
