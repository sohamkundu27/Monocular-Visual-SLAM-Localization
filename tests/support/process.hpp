#pragma once
#include <cerrno>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>
#include "support/test_utils.hpp"

namespace testing_utils {
struct ProcessResult { int code; std::string out; std::string err; };
inline ProcessResult run_process(const std::string& executable,const std::vector<std::string>& args) {
    TempDir tmp;
    const auto stdout_path=tmp/"stdout"; const auto stderr_path=tmp/"stderr";
    std::vector<std::string> storage{executable}; storage.insert(storage.end(),args.begin(),args.end());
    std::vector<char*> argv; for(auto& s:storage) argv.push_back(s.data()); argv.push_back(nullptr);
    const auto child=fork();
    if(child<0) throw std::runtime_error("fork failed");
    if(child==0) {
        if(!std::freopen(stdout_path.c_str(),"w",stdout) || !std::freopen(stderr_path.c_str(),"w",stderr)) _exit(126);
        execv(executable.c_str(),argv.data()); _exit(127);
    }
    int status=0;
    while(waitpid(child,&status,0)<0) if(errno!=EINTR) throw std::runtime_error("waitpid failed");
    return {WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status),read_text(stdout_path),read_text(stderr_path)};
}
}  // namespace testing_utils
