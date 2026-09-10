"""``run_slam`` entry point: full SLAM with loop closure and optimization."""

from __future__ import annotations

import argparse
import sys

from monocular_slam.cli.common import add_common_arguments, config_from_args, resolve_output_dir
from monocular_slam.config import ConfigError
from monocular_slam.datasets.kitti import DatasetError
from monocular_slam.pipeline import run_pipeline

DESCRIPTION = """\
Run monocular visual SLAM on a KITTI Odometry sequence.

Estimates the camera trajectory from a single image stream, detects loop
closures, optimizes the keyframe pose graph with GTSAM, and evaluates both the
raw and optimized trajectories against KITTI ground truth.

Note on scale: monocular geometry cannot recover absolute scale. By default the
per-frame translation magnitude is taken from ground truth
(odometry.scale_source: ground_truth) so trajectory shape and drift can be
benchmarked; results produced this way are labelled in metrics.json and must be
reported as ground-truth-scaled. Use --set odometry.scale_source=none for a
genuinely scale-free run, evaluated with sim3 alignment.
"""

EXAMPLES = """\
examples:
  # Full run on sequence 00
  python scripts/run_slam.py --sequence 00 --dataset-path ~/datasets/kitti/dataset

  # Quick smoke run on 300 frames
  python scripts/run_slam.py -s 05 --dataset-path ~/kitti --max-frames 300

  # Scale-free monocular, no ground-truth assistance
  python scripts/run_slam.py -s 00 --dataset-path ~/kitti \\
      --set odometry.scale_source=none

  # Odometry only, skipping loop closure and optimization
  python scripts/run_slam.py -s 00 --dataset-path ~/kitti --no-loop-closure
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_slam.py",
        description=DESCRIPTION,
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    slam = parser.add_argument_group("slam")
    slam.add_argument(
        "--no-loop-closure", action="store_true",
        help="Disable loop closure detection and pose-graph optimization",
    )
    slam.add_argument("--no-plots", action="store_true", help="Skip figure rendering")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        result = run_pipeline(config, output_dir=resolve_output_dir(args, config))
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(result)
    return 0


def _print_summary(result) -> None:  # noqa: ANN001
    """Print the headline numbers, using only what was actually measured."""
    from monocular_slam.pipeline import build_metrics

    metrics = build_metrics(result)
    lines = [
        "",
        "=" * 66,
        f"  KITTI sequence {metrics['sequence']} - results",
        "=" * 66,
        f"  frames processed            {metrics['frames']}",
        f"  distance travelled          {metrics['distance_km']} km",
        f"  successful pose rate        {metrics['successful_pose_rate_pct']} %",
        f"  loop closures detected      {metrics['loop_closures_detected']}",
        f"  runtime                     {metrics['runtime_fps']} FPS",
    ]
    if metrics["ate_rmse_raw_m"] is not None:
        lines += [
            "-" * 66,
            f"  ATE RMSE (raw VO)           {metrics['ate_rmse_raw_m']} m",
            f"  ATE RMSE (optimized)        {metrics['ate_rmse_optimized_m']} m",
            f"  ATE reduction               {metrics['ate_reduction_pct']} %",
            f"  translational drift (raw)   {metrics['translational_drift_raw_pct']} %",
            f"  translational drift (opt)   {metrics['translational_drift_optimized_pct']} %",
            f"  rotational drift            {metrics['rotational_drift_deg_per_100m']} deg/100 m",
        ]
    else:
        lines += ["-" * 66, "  no ground truth for this sequence: trajectory metrics unavailable"]

    if metrics["scale_uses_ground_truth"]:
        lines += [
            "-" * 66,
            "  NOTE: translation magnitudes came from ground truth",
            "        (odometry.scale_source=ground_truth). Absolute scale is",
            "        NOT recovered from the images. See README.",
        ]
    lines += ["=" * 66, ""]
    for name, path in sorted(result.artifacts.items()):
        lines.append(f"  {name:24s} {path}")
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
