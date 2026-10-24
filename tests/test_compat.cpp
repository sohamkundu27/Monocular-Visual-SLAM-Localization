// Golden tests for the Python-compatibility layer (no Python counterpart:
// these pin the formatting and reduction primitives every artefact relies on
// to their exact CPython / NumPy outputs, from tests/golden/compat.json).
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/compat/pyjson.hpp"
#include "mvslam/errors.hpp"
#include "support/golden.hpp"

using mvslam::Json;
namespace py = mvslam::py;
namespace np = mvslam::np;

namespace {

const Json& compat() { return golden::load("compat.json"); }

bool same_bits(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    return a == b && std::signbit(a) == std::signbit(b);
}

}  // namespace

TEST(CompatRepr, MatchesCPythonFloatRepr) {
    int checked = 0;
    for (const auto& item : compat().at("repr")) {
        const double value = golden::from_hex(item.at(0));
        EXPECT_EQ(py::repr(value), item.at(1).get<std::string>()) << item.at(0);
        ++checked;
    }
    EXPECT_GT(checked, 500);
}

TEST(CompatRound, MatchesCPythonRound) {
    for (const auto& item : compat().at("round")) {
        const double value = golden::from_hex(item.at(0));
        const int digits = item.at(1).get<int>();
        const double expected = golden::from_hex(item.at(2));
        EXPECT_TRUE(same_bits(py::round(value, digits), expected))
            << "round(" << py::repr(value) << ", " << digits << ") = "
            << py::repr(py::round(value, digits)) << ", expected " << py::repr(expected);
    }
}

TEST(CompatFormat, MatchesFormatSpecsAndPercentFormatting) {
    for (const auto& item : compat().at("format")) {
        const double value = golden::from_hex(item.at(0));
        const std::string spec = item.at(1).get<std::string>();
        EXPECT_EQ(py::format(value, spec), item.at(2).get<std::string>())
            << "format(" << py::repr(value) << ", '" << spec << "')";
        const std::string printf_spec = "%" + spec;
        EXPECT_EQ(py::sprintf(printf_spec.c_str(), value), item.at(3).get<std::string>())
            << "'" << printf_spec << "' % " << py::repr(value);
    }
}

TEST(CompatFormat, IntegerGroupingAndWidth) {
    for (const auto& item : compat().at("int_format")) {
        const auto n = item.at(0).get<std::int64_t>();
        EXPECT_EQ(py::format(n, ","), item.at(1).get<std::string>());
        EXPECT_EQ(py::format(n, "7d"), item.at(2).get<std::string>());
    }
}

TEST(CompatNumpy, ReductionsAreBitIdentical) {
    for (const auto& item : compat().at("stats")) {
        const std::vector<double> data = golden::vec(item.at("data"));
        const std::vector<double> positive = golden::vec(item.at("positive"));
        SCOPED_TRACE("n=" + std::to_string(data.size()));
        EXPECT_TRUE(same_bits(np::sum(data), golden::from_hex(item.at("sum"))));
        EXPECT_TRUE(same_bits(np::mean(data), golden::from_hex(item.at("mean"))));
        EXPECT_TRUE(same_bits(np::std_dev(data), golden::from_hex(item.at("std"))));
        EXPECT_TRUE(same_bits(np::median(data), golden::from_hex(item.at("median"))));
        EXPECT_TRUE(same_bits(np::min(data), golden::from_hex(item.at("min"))));
        EXPECT_TRUE(same_bits(np::max(data), golden::from_hex(item.at("max"))));
        EXPECT_TRUE(same_bits(np::rms(data), golden::from_hex(item.at("rms"))));
        EXPECT_TRUE(same_bits(np::percentile(positive, 20),
                              golden::from_hex(item.at("percentile_20_80").at(0))));
        EXPECT_TRUE(same_bits(np::percentile(positive, 80),
                              golden::from_hex(item.at("percentile_20_80").at(1))));
        EXPECT_TRUE(same_bits(np::percentile(positive, 50), golden::from_hex(item.at("percentile_50"))));
    }
}

TEST(CompatNumpy, MedianAndMeanEdgeCases) {
    EXPECT_TRUE(std::isnan(np::median({})));
    EXPECT_TRUE(std::isnan(np::median({1.0, NAN, 2.0})));
    EXPECT_TRUE(std::isnan(np::mean({})));
    EXPECT_EQ(np::median({3.0, 1.0}), 2.0);
    EXPECT_THROW(np::max({}), mvslam::ValueError);
}

TEST(CompatJson, DumpsMatchesJsonDumps) {
    for (const auto& item : compat().at("json")) {
        const Json value = py::loads(item.at(0).get<std::string>());
        EXPECT_EQ(py::dumps(value), item.at(0).get<std::string>());
        EXPECT_EQ(py::dumps(value, 2), item.at(1).get<std::string>());
    }
}

TEST(CompatJson, LoadsRejectsMalformedInput) {
    EXPECT_THROW(py::loads("{not json"), mvslam::JsonDecodeError);
    EXPECT_THROW(py::loads("[1, 2"), mvslam::JsonDecodeError);
    EXPECT_THROW(py::loads(""), mvslam::JsonDecodeError);
    EXPECT_NO_THROW(py::loads("{\"a\": NaN, \"b\": -Infinity}"));
}

TEST(CompatJson, ReprOfPythonObjects) {
    EXPECT_EQ(py::repr(Json::parse(R"({"a": [1, 2.5, null, true, "x'y"]})")),
              "{'a': [1, 2.5, None, True, \"x'y\"]}");
    EXPECT_EQ(py::repr_str("it's"), "\"it's\"");
    EXPECT_EQ(py::repr_str("a\tb"), "'a\\tb'");
}

TEST(CompatCsv, MatchesCsvDictWriter) {
    const Json& csv = compat().at("csv");
    std::string text;
    bool header = true;
    for (const auto& row : csv.at("rows")) {
        std::vector<std::string> names;
        std::vector<std::string> fields;
        for (const auto& pair : row) {
            names.push_back(pair.at(0).get<std::string>());
            const Json& value = pair.at(1);
            const std::string name = pair.at(0).get<std::string>();
            if (value.is_boolean()) {
                fields.push_back(py::repr(value.get<bool>()));
            } else if (value.is_number_integer()) {
                fields.push_back(std::to_string(value.get<std::int64_t>()));
            } else if (name == "ratio" || name == "nan") {
                fields.push_back(py::repr(golden::from_hex(value)));
            } else {
                fields.push_back(value.get<std::string>());
            }
        }
        if (header) {
            text += py::csv_row(names);
            header = false;
        }
        text += py::csv_row(fields);
    }
    EXPECT_EQ(text, csv.at("text").get<std::string>());
}

TEST(CompatSavetxt, MatchesNumpySavetxt) {
    const Json& savetxt = compat().at("savetxt");
    const golden::Array rows = golden::decode(savetxt.at("rows"));
    const auto values = rows.as_doubles();
    std::string text;
    const auto cols = static_cast<std::size_t>(rows.shape[1]);
    for (std::int64_t r = 0; r < rows.shape[0]; ++r) {
        text += py::savetxt_row(values.data() + static_cast<std::size_t>(r) * cols, cols);
    }
    EXPECT_EQ(text, savetxt.at("text").get<std::string>());
}
