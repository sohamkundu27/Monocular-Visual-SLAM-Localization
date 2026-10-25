// Monocular scale strategies.
//
// The central limitation of monocular vision
// ------------------------------------------
// A single moving camera observing a rigid scene cannot recover **absolute
// scale**. The essential matrix is defined only up to scale, so decomposing
// it yields a translation *direction* with unit norm and nothing more. A car
// driving 1 m past a 4 m wall produces pixel motion identical to a model car
// driving 10 cm past a 40 cm wall. No amount of feature matching or bundle
// adjustment resolves this: it needs information from outside the image
// stream -- a stereo baseline, an IMU, a wheel odometer, a known object size,
// or an assumption such as constant camera height above a ground plane.
//
// This module therefore keeps scale strictly separate from geometry. The pose
// estimator produces unit-norm translation directions; a ScaleEstimator
// supplies the magnitude; the trajectory builder multiplies them. Swapping in
// a different scale source changes one object and nothing else.
//
// Available strategies
// --------------------
// GroundTruthScale
//     Takes the per-transition translation magnitude from KITTI ground truth.
//     **This is an evaluation aid, not a SLAM capability.** It isolates
//     *rotational and directional* drift -- the parts monocular VO genuinely
//     estimates -- so that trajectory shape and loop-closure benefit can be
//     benchmarked against ground truth on a common footing. It is the
//     standard protocol for reporting monocular VO on KITTI, and it must
//     always be reported as such. A trajectory produced this way is **not** an
//     absolute-scale monocular result.
//
// ConstantScale
//     Assumes a fixed distance per frame. Genuinely scale-free with respect
//     to ground truth, and a reasonable approximation on constant-speed
//     stretches, but it degrades badly through stops and turns.
//
// UnitScale
//     Pure monocular output: every step has length 1. The trajectory shape is
//     meaningful, the units are arbitrary. Combined with Umeyama *similarity*
//     alignment during evaluation (which fits a global scale factor), this
//     yields honest scale-free metrics.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mvslam/compat/pyjson.hpp"

namespace mvslam {

struct Config;
class KittiOdometryDataset;

/// Supplies the metric magnitude for a unit-norm translation direction.
class ScaleEstimator {
public:
    virtual ~ScaleEstimator() = default;

    /// Metres for the transition from frame ``i`` to ``i + 1``.
    ///
    /// ``nullopt`` means "no estimate available", which the caller should
    /// treat as a failed transition rather than silently substituting a
    /// default.
    virtual std::optional<double> scale_for(std::int64_t transition_index) const = 0;

    /// True for strategies that consume ground truth, so the pipeline can
    /// label results honestly in metrics, logs and the README.
    virtual bool uses_ground_truth() const { return false; }

    /// Short identifier recorded in run metadata.
    virtual std::string name() const = 0;

    virtual Json describe() const;
};

/// Pure monocular: every step has unit length, units are arbitrary.
class UnitScale : public ScaleEstimator {
public:
    std::optional<double> scale_for(std::int64_t) const override { return 1.0; }
    std::string name() const override { return "unit"; }
};

/// Assume a fixed travelled distance per transition.
class ConstantScale : public ScaleEstimator {
public:
    explicit ConstantScale(double value = 0.85);
    std::optional<double> scale_for(std::int64_t) const override { return value_; }
    std::string name() const override { return "constant"; }
    Json describe() const override;
    double value() const { return value_; }

private:
    double value_;
};

/// Read the translation magnitude from ground truth.
///
/// Evaluation-only. See the header comment: this removes the scale-drift
/// component of the error so that rotation and heading drift can be measured
/// in isolation. Results derived from it must always be labelled as
/// ground-truth-scaled.
class GroundTruthScale : public ScaleEstimator {
public:
    explicit GroundTruthScale(std::vector<double> scales, double min_scale = 1e-3,
                              double max_scale = 10.0);
    std::optional<double> scale_for(std::int64_t transition_index) const override;
    bool uses_ground_truth() const override { return true; }
    std::string name() const override { return "ground_truth"; }
    Json describe() const override;
    const std::vector<double>& scales() const { return scales_; }

private:
    std::vector<double> scales_;
    double min_scale_;
    double max_scale_;
};

/// Construct the strategy named by ``config.odometry.scale_source``.
///
/// Falls back to UnitScale with a loud warning if ground-truth scale is
/// requested but the sequence has no poses -- better an honest scale-free
/// trajectory than a silently broken one.
std::unique_ptr<ScaleEstimator> build_scale_estimator(const Config& config,
                                                      const KittiOdometryDataset* dataset = nullptr);

}  // namespace mvslam
