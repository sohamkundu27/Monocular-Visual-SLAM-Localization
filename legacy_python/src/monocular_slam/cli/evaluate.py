"""``evaluate`` entry point: score saved trajectory files against ground truth.

Lets a trajectory be re-evaluated (with different alignment or RPE settings)
without re-running the pipeline, and lets an externally produced trajectory be
scored with the same metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from monocular_slam.config import Config, ConfigError
from monocular_slam.datasets.kitti import DatasetError, KittiOdometryDataset
from monocular_slam.evaluation.metrics import ate_reduction_pct, evaluate_trajectory
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.utils.logging import setup_logging

DESCRIPTION = """\
Evaluate KITTI-format trajectory files against ground truth.

Trajectories are 12-value-per-row KITTI pose files, as written by run_slam.py
(trajectory_raw.txt, trajectory_optimized.txt).
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trajectory", type=str, help="KITTI-format trajectory file to evaluate")
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Second trajectory to compare against (e.g. the optimized one), "
             "which also enables the ATE-reduction figure",
    )
    parser.add_argument("--sequence", "-s", type=str, required=True, help="KITTI sequence id")
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Root of the KITTI odometry download",
    )
    parser.add_argument(
        "--alignment", type=str, default="sim3", choices=["sim3", "se3", "none"],
        help="Umeyama alignment mode (default: sim3, appropriate for monocular)",
    )
    parser.add_argument(
        "--rpe-deltas", type=int, nargs="+", default=[1, 10, 100],
        help="Frame gaps for relative pose error",
    )
    parser.add_argument("--json", type=str, default=None, help="Write results to this JSON file")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    try:
        config = Config().with_overrides(
            {"dataset.path": args.dataset_path, "dataset.sequence": args.sequence}
        )
        dataset = KittiOdometryDataset.from_config(config)
    except (ConfigError, DatasetError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2

    ground_truth = dataset.ground_truth
    if ground_truth is None:
        print(
            f"Sequence {args.sequence} has no ground-truth poses "
            "(sequences 11-21 are the held-out test split); cannot evaluate.",
            file=sys.stderr,
        )
        return 2

    try:
        estimated = Trajectory.load_kitti(args.trajectory)
    except OSError as exc:
        print(f"Could not read trajectory: {exc}", file=sys.stderr)
        return 2

    if len(estimated) != len(ground_truth):
        print(
            f"Length mismatch: trajectory has {len(estimated)} poses, ground truth has "
            f"{len(ground_truth)}. Was the run truncated with --max-frames?",
            file=sys.stderr,
        )
        return 2

    results: dict[str, object] = {}
    primary = evaluate_trajectory(
        estimated,
        ground_truth,
        label=Path(args.trajectory).stem,
        alignment=args.alignment,
        rpe_deltas=tuple(args.rpe_deltas),
    )
    results[primary.label] = primary.to_dict()
    print(primary.summary_line())

    if args.compare:
        other = Trajectory.load_kitti(args.compare)
        if len(other) != len(ground_truth):
            print("Comparison trajectory length does not match ground truth", file=sys.stderr)
            return 2
        secondary = evaluate_trajectory(
            other,
            ground_truth,
            label=Path(args.compare).stem,
            alignment=args.alignment,
            rpe_deltas=tuple(args.rpe_deltas),
        )
        results[secondary.label] = secondary.to_dict()
        print(secondary.summary_line())
        reduction = ate_reduction_pct(primary.ate_aligned.rmse, secondary.ate_aligned.rmse)
        results["ate_reduction_pct"] = round(float(reduction), 3)
        print(f"ATE reduction: {reduction:.2f}%")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
