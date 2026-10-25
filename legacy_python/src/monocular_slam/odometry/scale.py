"""Monocular scale strategies.

The central limitation of monocular vision
------------------------------------------
A single moving camera observing a rigid scene cannot recover **absolute
scale**. The essential matrix is defined only up to scale, so decomposing it
yields a translation *direction* with unit norm and nothing more. A car driving
1 m past a 4 m wall produces pixel motion identical to a model car driving
10 cm past a 40 cm wall. No amount of feature matching or bundle adjustment
resolves this: it needs information from outside the image stream — a stereo
baseline, an IMU, a wheel odometer, a known object size, or an assumption such
as constant camera height above a ground plane.

This module therefore keeps scale strictly separate from geometry. The pose
estimator produces unit-norm translation directions; a :class:`ScaleEstimator`
supplies the magnitude; the trajectory builder multiplies them. Swapping in a
different scale source changes one object and nothing else.

Available strategies
--------------------
``GroundTruthScale``
    Takes the per-transition translation magnitude from KITTI ground truth.
    **This is an evaluation aid, not a SLAM capability.** It isolates
    *rotational and directional* drift — the parts monocular VO genuinely
    estimates — so that trajectory shape and loop-closure benefit can be
    benchmarked against ground truth on a common footing. It is the standard
    protocol for reporting monocular VO on KITTI, and it must always be
    reported as such. A trajectory produced this way is **not** an
    absolute-scale monocular result.

``ConstantScale``
    Assumes a fixed distance per frame. Genuinely scale-free with respect to
    ground truth, and a reasonable approximation on constant-speed stretches,
    but it degrades badly through stops and turns.

``UnitScale``
    Pure monocular output: every step has length 1. The trajectory shape is
    meaningful, the units are arbitrary. Combined with Umeyama *similarity*
    alignment during evaluation (which fits a global scale factor), this yields
    honest scale-free metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)


class ScaleEstimator(ABC):
    """Supplies the metric magnitude for a unit-norm translation direction."""

    #: Set to ``True`` by strategies that consume ground truth, so the pipeline
    #: can label results honestly in metrics, logs and the README.
    uses_ground_truth: bool = False

    #: Short identifier recorded in run metadata.
    name: str = "base"

    @abstractmethod
    def scale_for(self, transition_index: int) -> float | None:
        """Return metres for the transition from frame ``i`` to ``i + 1``.

        ``None`` means "no estimate available", which the caller should treat
        as a failed transition rather than silently substituting a default.
        """

    def describe(self) -> dict[str, object]:
        return {"strategy": self.name, "uses_ground_truth": self.uses_ground_truth}


class UnitScale(ScaleEstimator):
    """Pure monocular: every step has unit length, units are arbitrary."""

    uses_ground_truth = False
    name = "unit"

    def scale_for(self, transition_index: int) -> float:
        del transition_index
        return 1.0


class ConstantScale(ScaleEstimator):
    """Assume a fixed travelled distance per transition."""

    uses_ground_truth = False
    name = "constant"

    def __init__(self, value: float = 0.85) -> None:
        if value <= 0:
            raise ValueError(f"constant scale must be positive, got {value}")
        self.value = float(value)

    def scale_for(self, transition_index: int) -> float:
        del transition_index
        return self.value

    def describe(self) -> dict[str, object]:
        return {**super().describe(), "value_m": self.value}


class GroundTruthScale(ScaleEstimator):
    """Read the translation magnitude from ground truth.

    Evaluation-only. See the module docstring: this removes the scale-drift
    component of the error so that rotation and heading drift can be measured
    in isolation. Results derived from it must always be labelled as
    ground-truth-scaled.
    """

    uses_ground_truth = True
    name = "ground_truth"

    def __init__(
        self,
        scales: np.ndarray,
        min_scale: float = 1e-3,
        max_scale: float = 10.0,
    ) -> None:
        self.scales = np.asarray(scales, dtype=np.float64).reshape(-1)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        if np.any(~np.isfinite(self.scales)):
            raise ValueError("Ground-truth scales contain non-finite values")
        n_clipped = int(np.count_nonzero(self.scales > self.max_scale))
        if n_clipped:
            logger.warning(
                "%d ground-truth steps exceed max_scale=%.2f m and will be clipped",
                n_clipped,
                self.max_scale,
            )

    def scale_for(self, transition_index: int) -> float | None:
        if not 0 <= transition_index < len(self.scales):
            return None
        value = float(self.scales[transition_index])
        if value < self.min_scale:
            # The vehicle is effectively stationary; there is no motion to scale.
            return 0.0
        return float(np.clip(value, self.min_scale, self.max_scale))

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "n_transitions": int(len(self.scales)),
            "median_step_m": float(np.median(self.scales)) if len(self.scales) else None,
        }


def build_scale_estimator(config, dataset=None) -> ScaleEstimator:  # noqa: ANN001
    """Construct the scale strategy named by ``config.odometry.scale_source``.

    Falls back to :class:`UnitScale` with a loud warning if ground-truth scale
    is requested but the sequence has no poses — better an honest scale-free
    trajectory than a silently broken one.
    """
    source = config.odometry.scale_source.lower().strip()

    if source == "none":
        return UnitScale()

    if source == "constant":
        return ConstantScale(config.odometry.constant_scale)

    if source == "ground_truth":
        scales = None if dataset is None else dataset.ground_truth_scales()
        if scales is None:
            logger.warning(
                "scale_source='ground_truth' requested but sequence %s has no ground truth; "
                "falling back to unit scale (evaluate with sim3 alignment)",
                config.dataset.sequence,
            )
            return UnitScale()
        logger.info(
            "Scale strategy: ground truth (EVALUATION AID -- absolute scale is not "
            "recovered from the images; see README 'Monocular scale ambiguity')"
        )
        return GroundTruthScale(
            scales, min_scale=config.odometry.min_scale, max_scale=config.odometry.max_scale
        )

    raise ValueError(
        f"Unknown scale_source '{config.odometry.scale_source}'. "
        "Expected one of: 'ground_truth', 'constant', 'none'"
    )
