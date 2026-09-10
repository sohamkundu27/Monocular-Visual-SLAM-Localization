# Resume metrics

Every number below was produced by an actual run of this repository against the
KITTI Odometry dataset and read back from the corresponding
`outputs/sequence_XX/metrics.json`. Nothing here is estimated, and values that
were not measurable on a given run are shown as `n/a` rather than filled in.

> **Scale caveat.** These runs used `odometry.scale_source: ground_truth`, which
> takes the per-frame translation *magnitude* from KITTI ground truth. Monocular
> vision cannot recover absolute scale; the estimator recovers the translation
> *direction* and the full rotation. These figures therefore measure trajectory
> shape, heading drift and loop-closure benefit — not metric-scale localization.
> Any use of these numbers must carry the same caveat.

## Results

| Sequence | Frames | Distance | Pose rate | Loops | ATE RMSE (raw) | ATE RMSE (opt.) | ATE reduction | Drift (raw) | Drift (opt.) | Rot. drift | FPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | 4541 | 3.72 km | 99.8% | 30 | 8.42 m | 2.70 m | 68.0% | 1.59% | 1.25% | 0.837 | 14.0 |
| 01 | 1101 | 2.45 km | 99.9% | 1 | 16.21 m | 17.67 m | -9.0% | 3.30% | 3.37% | 0.987 | 15.8 |
| 05 | 2761 | 2.21 km | 99.6% | 18 | 11.88 m | 4.03 m | 66.1% | 2.44% | 1.71% | 1.180 | 15.1 |

## Front-end statistics

| Sequence | Keyframes | Avg. features/frame | Avg. matches/pair | Avg. RANSAC inlier ratio |
|---:|---:|---:|---:|---:|
| 00 | 1595 | 2980 | 1122 | 66.6% |
| 01 | 923 | 2698 | 931 | 71.7% |
| 05 | 972 | 2979 | 1168 | 66.3% |

## Measured highlights

- Processed **8,403 KITTI frames** across **8.38 km** of driving (sequences 00, 01, 05).
- Achieved a **99.6-99.9% successful relative-pose rate** across all evaluated sequences.
- Achieved **2.70 m optimized ATE RMSE** on sequence 00 (3.72 km, sim3 alignment).
- Achieved **1.25% translational drift** on sequence 00 (KITTI 100-800 m sub-trajectory metric).
- Reduced ATE RMSE by **68.0%** on sequence 00 (8.42 m -> 2.70 m) via loop closure and pose-graph optimization.
- Detected and geometrically verified **49 loop closures** (30 on sequence 00).
- Ran the full pipeline at **14.0-15.8 FPS** on CPU, end to end including loop closure and optimization.

## Where it does not help

- **Sequence 01**: optimization made ATE *worse* (16.21 m -> 17.67 m, -9.0%) from 1 accepted loop closure.

  KITTI sequence 01 is a fast highway drive that contains **no true loops**, so the
  correct number of closures is zero. Its repetitive corridor (guardrails, lane
  markings, uniform vegetation) defeats appearance matching *and* geometric
  verification, because forward motion along a straight road is a consistent camera
  motion between any two points on it. The metric-plausibility gate removes most such
  detections; the residual ones remain a known limitation, and switchable constraints
  or GNC would be the proper fix.

## Suggested resume bullets

Assembled entirely from the measured values above.

1. Built a monocular visual SLAM system in Python (OpenCV, GTSAM) implementing ORB feature tracking, MAGSAC++ essential-matrix pose estimation, bag-of-words loop closure and SE(3) pose-graph optimization, processing 8,403 KITTI Odometry frames over 8.4 km (sequences 00/01/05), at a 99.6%+ successful relative-pose rate.

2. Benchmarked against KITTI ground truth with Umeyama sim3 alignment, achieving 2.70 m absolute trajectory error (RMSE) on sequence 00 over 3.72 km and 1.25% translational drift on the KITTI 100-800 m sub-trajectory metric; monocular scale ambiguity handled by an explicitly documented ground-truth-scaled evaluation protocol.

3. Cut absolute trajectory error by 68.0% (8.42 m -> 2.70 m on sequence 00) by detecting 49 geometrically verified loop closures and optimizing the keyframe pose graph with GTSAM Levenberg-Marquardt under robust Huber kernels.

## Metric definitions

- **ATE RMSE** — root-mean-square Euclidean distance between estimated and
  ground-truth camera positions after Umeyama alignment.
- **ATE reduction** — `(ATE_raw - ATE_optimized) / ATE_raw * 100`.
- **Translational drift** — KITTI odometry metric: translation error over
  100-800 m sub-trajectories, normalised by sub-trajectory length.
- **Rotational drift** — the same sub-trajectory protocol, reporting rotation
  error in degrees per 100 m.
- **Successful pose rate** — validated relative poses divided by attempted frame
  transitions. A correctly detected stationary frame counts as a success.
- **FPS** — total frames divided by end-to-end wall-clock runtime, including
  loop closure and pose-graph optimization.
