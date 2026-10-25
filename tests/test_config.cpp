// Tests for the typed configuration system (mirrors tests/test_config.py),
// plus golden checks against the Python loader's exact behaviour.
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/compat/pyyaml.hpp"
#include "mvslam/config.hpp"
#include "support/golden.hpp"
#include "support/test_utils.hpp"

using mvslam::Config;
using mvslam::ConfigError;
using mvslam::Json;
using testing_utils::TempDir;
namespace py = mvslam::py;

namespace {

const std::filesystem::path kRepoConfig = std::filesystem::path(MVSLAM_SOURCE_DIR) / "configs/kitti.yaml";

// EXPECT that ``fn`` throws ConfigError whose message contains ``fragment``
// (pytest.raises(ConfigError, match=...)).
template <typename Fn>
void expect_config_error(Fn&& fn, const std::string& fragment) {
    try {
        fn();
        ADD_FAILURE() << "expected ConfigError matching '" << fragment << "'";
    } catch (const ConfigError& exc) {
        EXPECT_NE(std::string(exc.what()).find(fragment), std::string::npos)
            << "message was: " << exc.what();
    }
}

Json obj(const std::string& text) { return Json::parse(text); }

}  // namespace

// --------------------------------------------------------------------------- //
// TestDefaults
// --------------------------------------------------------------------------- //

// py: test_config.py::TestDefaults::test_defaults_are_constructible
TEST(TestDefaults, DefaultsAreConstructible) {
    const Config config;
    EXPECT_EQ(config.dataset.sequence, "00");
    EXPECT_EQ(config.features.detector, "orb");
    EXPECT_EQ(config.evaluation.alignment, "sim3");
}

// py: test_config.py::TestDefaults::test_run_dir_substitutes_the_sequence
TEST(TestDefaults, RunDirSubstitutesTheSequence) {
    const Config config = Config().with_override("dataset.sequence", "07");
    EXPECT_EQ(config.run_dir(), std::filesystem::path("outputs/sequence_07"));
}

// py: test_config.py::TestDefaults::test_tuple_fields_are_tuples
TEST(TestDefaults, TupleFieldsAreTuples) {
    // tuple[int, ...] maps to std::vector<int64_t>; the defaults are (1, 10, 100).
    EXPECT_EQ(Config().evaluation.rpe_deltas, (std::vector<std::int64_t>{1, 10, 100}));
}

// --------------------------------------------------------------------------- //
// TestShippedConfig
// --------------------------------------------------------------------------- //

// py: test_config.py::TestShippedConfig::test_repo_config_loads
TEST(TestShippedConfig, RepoConfigLoads) {
    ASSERT_TRUE(std::filesystem::is_regular_file(kRepoConfig))
        << "configs/kitti.yaml is missing from the repository";
    EXPECT_NO_THROW(Config::from_yaml(kRepoConfig));
}

// py: test_config.py::TestShippedConfig::test_repo_config_covers_every_section
TEST(TestShippedConfig, RepoConfigCoversEverySection) {
    // A section silently missing from the YAML would revert to defaults.
    const Json data = py::yaml_safe_load(testing_utils::read_text(kRepoConfig));
    const Json defaults = Config().to_dict();
    for (const auto& [section, _] : defaults.items()) {
        EXPECT_TRUE(data.contains(section)) << section;
    }
}

// py: test_config.py::TestShippedConfig::test_sequence_id_survives_as_a_string
TEST(TestShippedConfig, SequenceIdSurvivesAsAString) {
    // '00' must not be parsed as the integer 0.
    EXPECT_EQ(Config::from_yaml(kRepoConfig).dataset.sequence, "00");
}

// py: test_config.py::TestShippedConfig::test_repo_config_round_trips
TEST(TestShippedConfig, RepoConfigRoundTrips) {
    TempDir tmp;
    const Config original = Config::from_yaml(kRepoConfig);
    const Config reloaded = Config::from_yaml(original.to_yaml(tmp / "copy.yaml"));
    EXPECT_EQ(reloaded.to_dict(), original.to_dict());
}

// --------------------------------------------------------------------------- //
// TestLoading
// --------------------------------------------------------------------------- //

// py: test_config.py::TestLoading::test_missing_file_raises
TEST(TestLoading, MissingFileRaises) {
    TempDir tmp;
    expect_config_error([&] { Config::from_yaml(tmp / "nope.yaml"); }, "not found");
}

// py: test_config.py::TestLoading::test_empty_file_gives_defaults
TEST(TestLoading, EmptyFileGivesDefaults) {
    TempDir tmp;
    testing_utils::write_text(tmp / "empty.yaml", "");
    EXPECT_EQ(Config::from_yaml(tmp / "empty.yaml").dataset.sequence, Config().dataset.sequence);
}

// py: test_config.py::TestLoading::test_non_mapping_root_raises
TEST(TestLoading, NonMappingRootRaises) {
    TempDir tmp;
    testing_utils::write_text(tmp / "list.yaml", "- a\n- b\n");
    expect_config_error([&] { Config::from_yaml(tmp / "list.yaml"); }, "must be a mapping");
}

// py: test_config.py::TestLoading::test_partial_config_keeps_other_defaults
TEST(TestLoading, PartialConfigKeepsOtherDefaults) {
    TempDir tmp;
    testing_utils::write_text(tmp / "partial.yaml", "odometry:\n  min_inliers: 55\n");
    const Config config = Config::from_yaml(tmp / "partial.yaml");
    EXPECT_EQ(config.odometry.min_inliers, 55);
    EXPECT_EQ(config.features.max_features, Config().features.max_features);
}

// --------------------------------------------------------------------------- //
// TestValidation
// --------------------------------------------------------------------------- //

// py: test_config.py::TestValidation::test_unknown_section_is_rejected
TEST(TestValidation, UnknownSectionIsRejected) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometery": {}})")); },
                        "Unknown config key");
}

// py: test_config.py::TestValidation::test_unknown_key_within_a_section_is_rejected
TEST(TestValidation, UnknownKeyWithinASectionIsRejected) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometry": {"min_inlyers": 3}})")); },
                        "Unknown config key");
}

// py: test_config.py::TestValidation::test_error_message_lists_valid_keys
TEST(TestValidation, ErrorMessageListsValidKeys) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometry": {"typo": 1}})")); },
                        "Valid keys");
}

// py: test_config.py::TestValidation::test_wrong_type_is_rejected
TEST(TestValidation, WrongTypeIsRejected) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometry": {"min_inliers": "many"}})")); },
                        "expects an int");
}

// py: test_config.py::TestValidation::test_null_for_a_non_optional_key_is_rejected
TEST(TestValidation, NullForANonOptionalKeyIsRejected) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometry": {"min_inliers": null}})")); },
                        "does not accept null");
}

// py: test_config.py::TestValidation::test_null_is_allowed_for_optional_keys
TEST(TestValidation, NullIsAllowedForOptionalKeys) {
    EXPECT_FALSE(Config::from_dict(obj(R"({"dataset": {"max_frames": null}})")).dataset.max_frames);
    EXPECT_FALSE(
        Config::from_dict(obj(R"({"matcher": {"max_distance": null}})")).matcher.max_distance);
}

// py: test_config.py::TestValidation::test_section_given_a_scalar_is_rejected
TEST(TestValidation, SectionGivenAScalarIsRejected) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometry": 5})")); }, "Expected a mapping");
}

// --------------------------------------------------------------------------- //
// TestCoercion
// --------------------------------------------------------------------------- //

// py: test_config.py::TestCoercion::test_boolean_coercion[true-True]
// py: test_config.py::TestCoercion::test_boolean_coercion[no-False]
// py: test_config.py::TestCoercion::test_boolean_coercion[True-True]
TEST(TestCoercion, BooleanCoercion) {
    const std::vector<std::pair<Json, bool>> cases = {{"true", true}, {"no", false}, {true, true}};
    for (const auto& [value, expected] : cases) {
        Json data = Json::object();
        data["loop_closure"]["enabled"] = value;
        EXPECT_EQ(Config::from_dict(data).loop_closure.enabled, expected) << value;
    }
}

// py: test_config.py::TestCoercion::test_bad_boolean_is_rejected
TEST(TestCoercion, BadBooleanIsRejected) {
    expect_config_error([] { Config::from_dict(obj(R"({"loop_closure": {"enabled": "maybe"}})")); },
                        "expects a boolean");
}

// py: test_config.py::TestCoercion::test_int_is_widened_to_float
TEST(TestCoercion, IntIsWidenedToFloat) {
    const Config config = Config::from_dict(obj(R"({"odometry": {"ransac_threshold_px": 1}})"));
    EXPECT_EQ(config.odometry.ransac_threshold_px, 1.0);
    EXPECT_TRUE(config.to_dict()["odometry"]["ransac_threshold_px"].is_number_float());
}

// py: test_config.py::TestCoercion::test_int_field_rejects_a_bare_bool
TEST(TestCoercion, IntFieldRejectsABareBool) {
    expect_config_error([] { Config::from_dict(obj(R"({"odometry": {"min_inliers": true}})")); },
                        "expects an int");
}

// py: test_config.py::TestCoercion::test_tuple_field_accepts_a_yaml_list
TEST(TestCoercion, TupleFieldAcceptsAYamlList) {
    const Config config = Config::from_dict(obj(R"({"evaluation": {"rpe_deltas": [1, 2, 3]}})"));
    EXPECT_EQ(config.evaluation.rpe_deltas, (std::vector<std::int64_t>{1, 2, 3}));
}

// py: test_config.py::TestCoercion::test_tuple_field_rejects_a_scalar
TEST(TestCoercion, TupleFieldRejectsAScalar) {
    expect_config_error([] { Config::from_dict(obj(R"({"evaluation": {"rpe_deltas": 5}})")); },
                        "expects a list");
}

// --------------------------------------------------------------------------- //
// TestOverrides
// --------------------------------------------------------------------------- //

// py: test_config.py::TestOverrides::test_dotted_override_applies
TEST(TestOverrides, DottedOverrideApplies) {
    EXPECT_EQ(Config().with_override("odometry.min_inliers", 99).odometry.min_inliers, 99);
}

// py: test_config.py::TestOverrides::test_override_leaves_the_original_untouched
TEST(TestOverrides, OverrideLeavesTheOriginalUntouched) {
    const Config config;
    (void)config.with_override("odometry.min_inliers", 99);
    EXPECT_EQ(config.odometry.min_inliers, Config().odometry.min_inliers);
}

// py: test_config.py::TestOverrides::test_none_values_are_skipped
TEST(TestOverrides, NoneValuesAreSkipped) {
    // CLI flags default to None and must not clobber configured values.
    const Config config = Config().with_override("dataset.sequence", "05");
    EXPECT_EQ(config.with_override("dataset.sequence", nullptr).dataset.sequence, "05");
}

// py: test_config.py::TestOverrides::test_unknown_key_is_rejected
TEST(TestOverrides, UnknownKeyIsRejected) {
    expect_config_error([] { Config().with_override("odometry.nope", 1); }, "Unknown config key");
}

// py: test_config.py::TestOverrides::test_unknown_section_is_rejected
TEST(TestOverrides, UnknownSectionIsRejected) {
    expect_config_error([] { Config().with_override("nope.key", 1); }, "Unknown config section");
}

// py: test_config.py::TestOverrides::test_multiple_overrides_all_apply
TEST(TestOverrides, MultipleOverridesAllApply) {
    const Config config = Config().with_overrides(
        obj(R"({"dataset.sequence": "05", "features.max_features": 500, "loop_closure.enabled": false})"));
    EXPECT_EQ(config.dataset.sequence, "05");
    EXPECT_EQ(config.features.max_features, 500);
    EXPECT_FALSE(config.loop_closure.enabled);
}

// --------------------------------------------------------------------------- //
// TestSerialisation
// --------------------------------------------------------------------------- //

// py: test_config.py::TestSerialisation::test_to_dict_is_json_serialisable
TEST(TestSerialisation, ToDictIsJsonSerialisable) {
    const std::string text = py::dumps(Config().to_dict());
    EXPECT_EQ(py::loads(text), Config().to_dict());
}

// py: test_config.py::TestSerialisation::test_tuples_become_lists
TEST(TestSerialisation, TuplesBecomeLists) {
    EXPECT_TRUE(Config().to_dict()["evaluation"]["rpe_deltas"].is_array());
}

// py: test_config.py::TestSerialisation::test_yaml_round_trip_preserves_everything
TEST(TestSerialisation, YamlRoundTripPreservesEverything) {
    TempDir tmp;
    const Config original = Config().with_overrides(
        obj(R"({"dataset.sequence": "09", "odometry.min_inliers": 77, "evaluation.rpe_deltas": [2, 4]})"));
    const Config reloaded = Config::from_yaml(original.to_yaml(tmp / "c.yaml"));
    EXPECT_EQ(reloaded.to_dict(), original.to_dict());
    EXPECT_EQ(reloaded.evaluation.rpe_deltas, (std::vector<std::int64_t>{2, 4}));
}

// py: test_config.py::TestSerialisation::test_to_yaml_creates_parent_directories
TEST(TestSerialisation, ToYamlCreatesParentDirectories) {
    TempDir tmp;
    const auto path = Config().to_yaml(tmp / "deep" / "nested" / "c.yaml");
    EXPECT_TRUE(std::filesystem::is_regular_file(path));
}

// --------------------------------------------------------------------------- //
// Golden parity with the Python loader (tests/golden/config.json)
// --------------------------------------------------------------------------- //

namespace {

const Json& config_golden() { return golden::load("config.json"); }

// Undo export_golden._jsonable: {"float": "inf"} markers and {"tuple": [...]}.
Json unjsonable(const Json& value) {
    if (value.is_object() && value.size() == 1 && value.contains("float")) {
        const std::string text = value.at("float").get<std::string>();
        return py::make_float(text == "nan"   ? NAN
                              : text == "inf" ? INFINITY
                                              : -INFINITY);
    }
    if (value.is_object() && value.size() == 1 && value.contains("tuple")) {
        return unjsonable(value.at("tuple"));
    }
    if (value.is_array()) {
        Json out = Json::array();
        for (const auto& item : value) {
            out.push_back(unjsonable(item));
        }
        return out;
    }
    if (value.is_object()) {
        Json out = Json::object();
        for (const auto& [k, v] : value.items()) {
            out[k] = unjsonable(v);
        }
        return out;
    }
    return value;
}

bool json_equal(const Json& a, const Json& b) {
    if (a.is_number() && b.is_number()) {
        const double x = py::as_double(a);
        const double y = py::as_double(b);
        return (std::isnan(x) && std::isnan(y)) || x == y;
    }
    if (a.is_array() && b.is_array()) {
        if (a.size() != b.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!json_equal(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }
    return a == b;
}

}  // namespace

TEST(ConfigGolden, YamlDumpIsByteIdenticalToPyYaml) {
    for (const auto& [name, entry] : config_golden().at("configs").items()) {
        const Config config = Config::from_dict(unjsonable(entry.at("dict")));
        EXPECT_EQ(py::yaml_safe_dump(config.to_dict()), entry.at("yaml").get<std::string>()) << name;
        // ...and the C++ loader reads Python's text back to the same values.
        const Config reloaded = Config::from_dict(py::yaml_safe_load(entry.at("yaml").get<std::string>()));
        EXPECT_EQ(reloaded.to_dict(), config.to_dict()) << name;
    }
}

TEST(ConfigGolden, KittiYamlLoadsToThePythonValues) {
    const Json expected = unjsonable(config_golden().at("configs").at("kitti").at("dict"));
    EXPECT_EQ(Config::from_yaml(kRepoConfig).to_dict(), expected);
}

TEST(ConfigGolden, CoercionMatchesThePythonLoader) {
    for (const auto& item : config_golden().at("coercion")) {
        const std::string key = item.at("key").get<std::string>();
        const std::string section = key.substr(0, key.find('.'));
        const std::string field = key.substr(key.find('.') + 1);
        Json data = Json::object();
        data[section][field] = unjsonable(item.at("input"));
        const Json& out = item.at("out");
        SCOPED_TRACE(key + " <- " + py::repr(data[section][field]));
        if (out.at("ok").get<bool>()) {
            const Config config = Config::from_dict(data);
            const Json actual = config.to_dict()[section][field];
            const Json expected = unjsonable(out.at("result").at("value"));
            EXPECT_TRUE(json_equal(actual, expected))
                << py::repr(actual) << " vs " << py::repr(expected);
            const std::string type = out.at("result").at("type").get<std::string>();
            if (type == "float") {
                EXPECT_TRUE(actual.is_number_float());
            } else if (type == "int") {
                EXPECT_TRUE(actual.is_number_integer());
            } else if (type == "str") {
                EXPECT_TRUE(actual.is_string());
            }
        } else {
            try {
                Config::from_dict(data);
                ADD_FAILURE() << "expected an error: " << out.at("error");
            } catch (const ConfigError& exc) {
                EXPECT_EQ(std::string(exc.what()), out.at("error").get<std::string>());
            }
        }
    }
}

TEST(ConfigGolden, StructuralErrorsMatchThePythonMessages) {
    for (const auto& item : config_golden().at("structural")) {
        if (item.at("ok").get<bool>()) {
            EXPECT_NO_THROW(Config::from_dict(item.at("input")));
            continue;
        }
        try {
            Config::from_dict(item.at("input"));
            ADD_FAILURE() << "expected: " << item.at("error");
        } catch (const ConfigError& exc) {
            EXPECT_EQ(std::string(exc.what()), item.at("error").get<std::string>());
        }
    }
    for (const auto& item : config_golden().at("override_errors")) {
        if (item.at("ok").get<bool>()) {
            EXPECT_NO_THROW(Config().with_overrides(item.at("input")));
            continue;
        }
        try {
            Config().with_overrides(item.at("input"));
            ADD_FAILURE() << "expected: " << item.at("error");
        } catch (const ConfigError& exc) {
            EXPECT_EQ(std::string(exc.what()), item.at("error").get<std::string>());
        }
    }
}

TEST(ConfigGolden, YamlScalarResolutionFollowsPyYaml) {
    for (const auto& item : config_golden().at("yaml_resolution")) {
        const std::string scalar = item.at("scalar").get<std::string>();
        const std::string type = item.at("type").get<std::string>();
        SCOPED_TRACE(scalar);
        if (type == "date" || type == "datetime" || type == "error") {
            continue;  // constructed as datetime objects in Python; see pyyaml.cpp
        }
        const Json loaded = py::yaml_safe_load("v: " + scalar + "\n").at("v");
        const Json expected = unjsonable(item.at("value"));
        const std::string expected_type = type == "NoneType" ? "NoneType" : type;
        EXPECT_EQ(py::type_name(loaded), expected_type) << py::repr(loaded);
        EXPECT_TRUE(json_equal(loaded, expected)) << py::repr(loaded) << " vs " << py::repr(expected);
    }
}

TEST(ConfigGolden, RunNameFormatting) {
    EXPECT_EQ(mvslam::format_run_name("sequence_{sequence}", "05"), "sequence_05");
    EXPECT_EQ(mvslam::format_run_name("{{literal}}_{sequence}", "00"), "{literal}_00");
    EXPECT_THROW(mvslam::format_run_name("{other}", "00"), mvslam::KeyError);
}
