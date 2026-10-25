#pragma once
#include <exception>
#include <iostream>
#include "mvslam/cli/apps.hpp"
// Unhandled Python exceptions exit 1; avoid C++ terminate()/SIGABRT for the
// corresponding malformed-input and filesystem failures.
inline int cli_main(int argc, char** argv, int (*run)(const std::vector<std::string>&)) {
    try { return run(std::vector<std::string>(argv+1,argv+argc)); }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
