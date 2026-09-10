"""Tests for the monocular scale strategies.

The central invariant: no strategy may silently fabricate a scale, and any
strategy that consumes ground truth must say so, because a result produced
that way cannot be reported as an absolute-scale monocular result.
"""

from __future__ import annotations

import numpy as np
import pytest

from kitti_fixture import write_kitti_sequence
from monocular_slam.config import Config
from monocular_slam.datasets.kitti import KittiOdometryDataset
from monocular_slam.odometry.scale import (
    ConstantScale,
    GroundTruthScale,
    UnitScale,
    build_scale_estimator,
)


class TestUnitScale:
    def test_always_returns_one(self):
        estimator = UnitScale()
        assert estimator.scale_for(0) == 1.0
        assert estimator.scale_for(10_000) == 1.0

    def test_is_flagged_as_not_using_ground_truth(self):
        assert UnitScale().uses_ground_truth is False
        assert UnitScale().describe()["uses_ground_truth"] is False


class TestConstantScale:
    def test_returns_the_configured_value(self):
        assert ConstantScale(0.85).scale_for(5) == pytest.approx(0.85)

    def test_is_independent_of_index(self):
        estimator = ConstantScale(1.2)
        assert estimator.scale_for(0) == estimator.scale_for(999)

    def test_non_positive_value_is_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            ConstantScale(0.0)

    def test_describe_reports_the_value(self):
        info = ConstantScale(0.7).describe()
        assert info["value_m"] == pytest.approx(0.7)
        assert info["uses_ground_truth"] is False


class TestGroundTruthScale:
    @pytest.fixture
    def estimator(self):
        return GroundTruthScale(np.array([0.9, 1.1, 0.85, 2.0]))

    def test_returns_the_per_transition_magnitude(self, estimator):
        assert estimator.scale_for(0) == pytest.approx(0.9)
        assert estimator.scale_for(3) == pytest.approx(2.0)

    def test_out_of_range_returns_none_not_a_default(self, estimator):
        """A missing estimate must be distinguishable from a real one."""
        assert estimator.scale_for(4) is None
        assert estimator.scale_for(-1) is None

    def test_is_flagged_as_using_ground_truth(self, estimator):
        assert estimator.uses_ground_truth is True
        assert estimator.describe()["uses_ground_truth"] is True

    def test_tiny_steps_collapse_to_zero(self):
        """A stopped vehicle has no motion to scale, so the step is exactly zero."""
        estimator = GroundTruthScale(np.array([1e-6, 0.9]), min_scale=1e-3)
        assert estimator.scale_for(0) == 0.0
        assert estimator.scale_for(1) == pytest.approx(0.9)

    def test_large_steps_are_clipped(self):
        estimator = GroundTruthScale(np.array([50.0]), max_scale=10.0)
        assert estimator.scale_for(0) == pytest.approx(10.0)

    def test_non_finite_input_is_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            GroundTruthScale(np.array([1.0, np.nan]))

    def test_describe_summarises_the_scales(self, estimator):
        info = estimator.describe()
        assert info["n_transitions"] == 4
        assert info["median_step_m"] == pytest.approx(1.0)


class TestFactory:
    @pytest.fixture
    def dataset(self, tmp_path):
        root = write_kitti_sequence(tmp_path / "d", "00", n_frames=10, step_m=0.9)
        return KittiOdometryDataset(root, "00")

    def test_builds_ground_truth_scale(self, dataset):
        config = Config().with_overrides({"odometry.scale_source": "ground_truth"})
        estimator = build_scale_estimator(config, dataset)
        assert isinstance(estimator, GroundTruthScale)
        assert estimator.scale_for(0) == pytest.approx(0.9, abs=1e-6)

    def test_builds_constant_scale(self, dataset):
        config = Config().with_overrides(
            {"odometry.scale_source": "constant", "odometry.constant_scale": 1.4}
        )
        estimator = build_scale_estimator(config, dataset)
        assert isinstance(estimator, ConstantScale)
        assert estimator.value == pytest.approx(1.4)

    def test_builds_unit_scale(self, dataset):
        config = Config().with_overrides({"odometry.scale_source": "none"})
        assert isinstance(build_scale_estimator(config, dataset), UnitScale)

    def test_falls_back_to_unit_when_ground_truth_is_absent(self, tmp_path):
        """Better an honest scale-free trajectory than a fabricated metric one."""
        root = write_kitti_sequence(tmp_path / "d", "00", n_frames=8, with_ground_truth=False)
        dataset = KittiOdometryDataset(root, "00")
        config = Config().with_overrides({"odometry.scale_source": "ground_truth"})
        estimator = build_scale_estimator(config, dataset)
        assert isinstance(estimator, UnitScale)
        assert estimator.uses_ground_truth is False

    def test_falls_back_without_a_dataset(self):
        config = Config().with_overrides({"odometry.scale_source": "ground_truth"})
        assert isinstance(build_scale_estimator(config, None), UnitScale)

    def test_unknown_source_is_rejected(self, dataset):
        config = Config().with_overrides({"odometry.scale_source": "imu"})
        with pytest.raises(ValueError, match="Unknown scale_source"):
            build_scale_estimator(config, dataset)


class TestScaleSeparationInvariant:
    def test_no_strategy_reports_ground_truth_use_incorrectly(self):
        """Exactly the ground-truth strategy may claim ground-truth use."""
        assert GroundTruthScale(np.array([1.0])).uses_ground_truth is True
        for estimator in (UnitScale(), ConstantScale(0.9)):
            assert estimator.uses_ground_truth is False

    def test_scale_never_silently_defaults_when_unavailable(self):
        """Out-of-range must return None, never a plausible-looking number."""
        estimator = GroundTruthScale(np.array([0.9, 0.9]))
        assert estimator.scale_for(2) is None
        assert estimator.scale_for(100) is None
