// Tests for the monocular scale strategies (mirrors tests/test_scale.py).
//
// The central invariant: no strategy may silently fabricate a scale, and any
// strategy that consumes ground truth must say so, because a result produced
// that way cannot be reported as an absolute-scale monocular result.
#include <gtest/gtest.h>

#include <cmath>

#include "mvslam/config.hpp"
#include "mvslam/datasets/kitti.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/odometry/scale.hpp"
#include "support/kitti_fixture.hpp"
#include "support/test_utils.hpp"

using namespace mvslam;
using testing_utils::TempDir;

// --------------------------------------------------------------------------- //
// TestUnitScale
// --------------------------------------------------------------------------- //

// py: test_scale.py::TestUnitScale::test_always_returns_one
TEST(TestUnitScale, AlwaysReturnsOne) {
    const UnitScale estimator;
    EXPECT_EQ(estimator.scale_for(0), 1.0);
    EXPECT_EQ(estimator.scale_for(10'000), 1.0);
}

// py: test_scale.py::TestUnitScale::test_is_flagged_as_not_using_ground_truth
TEST(TestUnitScale, IsFlaggedAsNotUsingGroundTruth) {
    EXPECT_FALSE(UnitScale().uses_ground_truth());
    EXPECT_EQ(UnitScale().describe()["uses_ground_truth"], false);
}

// --------------------------------------------------------------------------- //
// TestConstantScale
// --------------------------------------------------------------------------- //

// py: test_scale.py::TestConstantScale::test_returns_the_configured_value
TEST(TestConstantScale, ReturnsTheConfiguredValue) {
    EXPECT_APPROX(*ConstantScale(0.85).scale_for(5), 0.85);
}

// py: test_scale.py::TestConstantScale::test_is_independent_of_index
TEST(TestConstantScale, IsIndependentOfIndex) {
    const ConstantScale estimator(1.2);
    EXPECT_EQ(estimator.scale_for(0), estimator.scale_for(999));
}

// py: test_scale.py::TestConstantScale::test_non_positive_value_is_rejected
TEST(TestConstantScale, NonPositiveValueIsRejected) {
    try {
        ConstantScale(0.0);
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("must be positive"), std::string::npos);
    }
}

// py: test_scale.py::TestConstantScale::test_describe_reports_the_value
TEST(TestConstantScale, DescribeReportsTheValue) {
    const Json info = ConstantScale(0.7).describe();
    EXPECT_APPROX(info["value_m"].get<double>(), 0.7);
    EXPECT_EQ(info["uses_ground_truth"], false);
}

// --------------------------------------------------------------------------- //
// TestGroundTruthScale
// --------------------------------------------------------------------------- //

namespace {
GroundTruthScale estimator_fixture() { return GroundTruthScale({0.9, 1.1, 0.85, 2.0}); }
}  // namespace

// py: test_scale.py::TestGroundTruthScale::test_returns_the_per_transition_magnitude
TEST(TestGroundTruthScale, ReturnsThePerTransitionMagnitude) {
    const auto estimator = estimator_fixture();
    EXPECT_APPROX(*estimator.scale_for(0), 0.9);
    EXPECT_APPROX(*estimator.scale_for(3), 2.0);
}

// py: test_scale.py::TestGroundTruthScale::test_out_of_range_returns_none_not_a_default
TEST(TestGroundTruthScale, OutOfRangeReturnsNoneNotADefault) {
    // A missing estimate must be distinguishable from a real one.
    const auto estimator = estimator_fixture();
    EXPECT_FALSE(estimator.scale_for(4).has_value());
    EXPECT_FALSE(estimator.scale_for(-1).has_value());
}

// py: test_scale.py::TestGroundTruthScale::test_is_flagged_as_using_ground_truth
TEST(TestGroundTruthScale, IsFlaggedAsUsingGroundTruth) {
    const auto estimator = estimator_fixture();
    EXPECT_TRUE(estimator.uses_ground_truth());
    EXPECT_EQ(estimator.describe()["uses_ground_truth"], true);
}

// py: test_scale.py::TestGroundTruthScale::test_tiny_steps_collapse_to_zero
TEST(TestGroundTruthScale, TinyStepsCollapseToZero) {
    // A stopped vehicle has no motion to scale, so the step is exactly zero.
    const GroundTruthScale estimator({1e-6, 0.9}, 1e-3);
    EXPECT_EQ(*estimator.scale_for(0), 0.0);
    EXPECT_APPROX(*estimator.scale_for(1), 0.9);
}

// py: test_scale.py::TestGroundTruthScale::test_large_steps_are_clipped
TEST(TestGroundTruthScale, LargeStepsAreClipped) {
    EXPECT_APPROX(*GroundTruthScale({50.0}, 1e-3, 10.0).scale_for(0), 10.0);
}

// py: test_scale.py::TestGroundTruthScale::test_non_finite_input_is_rejected
TEST(TestGroundTruthScale, NonFiniteInputIsRejected) {
    try {
        GroundTruthScale({1.0, NAN});
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("non-finite"), std::string::npos);
    }
}

// py: test_scale.py::TestGroundTruthScale::test_describe_summarises_the_scales
TEST(TestGroundTruthScale, DescribeSummarisesTheScales) {
    const Json info = estimator_fixture().describe();
    EXPECT_EQ(info["n_transitions"], 4);
    EXPECT_APPROX(info["median_step_m"].get<double>(), 1.0);
}

// --------------------------------------------------------------------------- //
// TestFactory
// --------------------------------------------------------------------------- //

class TestFactory : public ::testing::Test {
protected:
    void SetUp() override {
        root = kitti_fixture::write_kitti_sequence(tmp / "d", "00", 10, true, true, "image_0", 0.9);
        dataset.emplace(root, "00");
    }
    TempDir tmp;
    std::filesystem::path root;
    std::optional<KittiOdometryDataset> dataset;
};

// py: test_scale.py::TestFactory::test_builds_ground_truth_scale
TEST_F(TestFactory, BuildsGroundTruthScale) {
    const Config config = Config().with_override("odometry.scale_source", "ground_truth");
    const auto estimator = build_scale_estimator(config, &*dataset);
    ASSERT_NE(dynamic_cast<const GroundTruthScale*>(estimator.get()), nullptr);
    EXPECT_APPROX(*estimator->scale_for(0), 0.9, std::nullopt, 1e-6);
}

// py: test_scale.py::TestFactory::test_builds_constant_scale
TEST_F(TestFactory, BuildsConstantScale) {
    const Config config = Config().with_overrides(
        Json::parse(R"({"odometry.scale_source": "constant", "odometry.constant_scale": 1.4})"));
    const auto estimator = build_scale_estimator(config, &*dataset);
    const auto* constant = dynamic_cast<const ConstantScale*>(estimator.get());
    ASSERT_NE(constant, nullptr);
    EXPECT_APPROX(constant->value(), 1.4);
}

// py: test_scale.py::TestFactory::test_builds_unit_scale
TEST_F(TestFactory, BuildsUnitScale) {
    const Config config = Config().with_override("odometry.scale_source", "none");
    EXPECT_NE(dynamic_cast<const UnitScale*>(build_scale_estimator(config, &*dataset).get()), nullptr);
}

// py: test_scale.py::TestFactory::test_falls_back_to_unit_when_ground_truth_is_absent
TEST_F(TestFactory, FallsBackToUnitWhenGroundTruthIsAbsent) {
    // Better an honest scale-free trajectory than a fabricated metric one.
    const auto d = kitti_fixture::write_kitti_sequence(tmp / "nogt", "00", 8, false);
    const KittiOdometryDataset no_gt(d, "00");
    const Config config = Config().with_override("odometry.scale_source", "ground_truth");
    const auto estimator = build_scale_estimator(config, &no_gt);
    EXPECT_NE(dynamic_cast<const UnitScale*>(estimator.get()), nullptr);
    EXPECT_FALSE(estimator->uses_ground_truth());
}

// py: test_scale.py::TestFactory::test_falls_back_without_a_dataset
TEST_F(TestFactory, FallsBackWithoutADataset) {
    const Config config = Config().with_override("odometry.scale_source", "ground_truth");
    EXPECT_NE(dynamic_cast<const UnitScale*>(build_scale_estimator(config, nullptr).get()), nullptr);
}

// py: test_scale.py::TestFactory::test_unknown_source_is_rejected
TEST_F(TestFactory, UnknownSourceIsRejected) {
    const Config config = Config().with_override("odometry.scale_source", "imu");
    try {
        build_scale_estimator(config, &*dataset);
        ADD_FAILURE() << "expected ValueError";
    } catch (const ValueError& exc) {
        EXPECT_NE(std::string(exc.what()).find("Unknown scale_source"), std::string::npos);
    }
}

// --------------------------------------------------------------------------- //
// TestScaleSeparationInvariant
// --------------------------------------------------------------------------- //

// py: test_scale.py::TestScaleSeparationInvariant::test_no_strategy_reports_ground_truth_use_incorrectly
TEST(TestScaleSeparationInvariant, NoStrategyReportsGroundTruthUseIncorrectly) {
    // Exactly the ground-truth strategy may claim ground-truth use.
    EXPECT_TRUE(GroundTruthScale({1.0}).uses_ground_truth());
    EXPECT_FALSE(UnitScale().uses_ground_truth());
    EXPECT_FALSE(ConstantScale(0.9).uses_ground_truth());
}

// py: test_scale.py::TestScaleSeparationInvariant::test_scale_never_silently_defaults_when_unavailable
TEST(TestScaleSeparationInvariant, ScaleNeverSilentlyDefaultsWhenUnavailable) {
    // Out-of-range must return nullopt, never a plausible-looking number.
    const GroundTruthScale estimator({0.9, 0.9});
    EXPECT_FALSE(estimator.scale_for(2).has_value());
    EXPECT_FALSE(estimator.scale_for(100).has_value());
}
