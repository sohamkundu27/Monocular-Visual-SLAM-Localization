#pragma once
#include <algorithm>
#include <iostream>
#include <CLI/CLI.hpp>
#include "mvslam/cli/common.hpp"
#include "mvslam/compat/pyformat.hpp"

namespace mvslam::cli {
inline ParseOutcome parse(CLI::App& app, const std::vector<std::string>& argv) {
    app.get_formatter()->label("Usage", "usage");
    auto args = argv;
    // argparse accepts an unambiguous prefix of a long option, including
    // --name=value, and stops interpreting options at --.
    bool positional = false;
    for (auto& arg : args) {
        if (arg == "--") { positional = true; continue; }
        if (positional || arg.rfind("--", 0) != 0) continue;
        const auto eq = arg.find('=');
        const auto name = arg.substr(2, eq == std::string::npos ? eq : eq - 2);
        std::vector<std::string> matches;
        bool exact = false;
        for (const auto* option : app.get_options()) for (const auto& candidate : option->get_lnames()) {
            exact = exact || candidate == name;
            if (candidate.rfind(name, 0) == 0) matches.push_back(candidate);
        }
        if (!exact && matches.size() == 1) arg = "--" + matches[0] + (eq == std::string::npos ? "" : arg.substr(eq));
    }
    std::reverse(args.begin(), args.end()); // CLI11's vector overload consumes from the back.
    try { app.parse(args); }
    catch (const CLI::CallForHelp&) { std::cout << app.help(); return {false, 0}; }
    catch (const CLI::ParseError& e) {
        std::cerr << "usage: " << app.get_name() << " [options]\n" << app.get_name() << ": error: " << e.what() << '\n';
        return {false, 2};
    }
    return {};
}
inline std::int64_t integer(const std::string& text) {
    const auto value = py::parse_int(text);
    if (!value) throw CLI::ValidationError("invalid int value: " + py::repr_str(text));
    return *value;
}
inline CLI::Option* int_option(CLI::App& app, const std::string& flags,
                               std::optional<std::int64_t>& target, const std::string& help) {
    return app.add_option_function<std::string>(flags, [&target](const std::string& s) { target=integer(s); }, help)->take_last();
}
inline void log_option(CLI::App& app, std::string& target) {
    app.add_option("--log-level",target,"Console log level")
        ->check(CLI::IsMember({"DEBUG","INFO","WARNING","ERROR"}))->take_last();
}
}  // namespace mvslam::cli
