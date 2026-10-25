#!/usr/bin/env python3
"""Compare measured C++ full-sequence runs with the committed Python references.

Full reference positions are stored with %.9e formatting, so 1e-6 m allows
only the rounding of that text export. Loop count and ATE thresholds are the
port's original 15% and 10% requirements; this tool never adjusts them.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def compare(root: Path, golden: Path, sequences: list[str]) -> dict:
    report = {}
    for sequence in sequences:
        run = root / f"sequence_{sequence}"
        reference = golden / f"sequence_{sequence}"
        cpp = json.loads((run / "metrics.json").read_text())
        python = json.loads((reference / "metrics.json").read_text())
        rows = np.loadtxt(run / "trajectory_raw.txt")
        full_reference = reference / "trajectory_raw.txt"
        poses = np.loadtxt(full_reference) if full_reference.exists() else None
        xyz = poses[:, [3, 7, 11]] if poses is not None else np.loadtxt(reference / "trajectory_raw_xyz.txt")
        if rows.shape[0] != xyz.shape[0]:
            raise ValueError(f"{sequence}: frame count mismatch")
        position_error = float(np.max(np.abs(rows[:, [3, 7, 11]] - xyz)))
        checks = {"raw_positions": position_error <= 1e-6,
                  "odometry_counts": cpp["odometry"] == python["odometry"]}
        values = {}
        for key, tolerance in [("ate_rmse_raw_m", .10), ("ate_rmse_optimized_m", .10),
                               ("loop_closures_detected", .15)]:
            actual, expected = cpp[key], python[key]
            relative = None if actual is None or expected in (None, 0) else abs(actual - expected) / abs(expected)
            checks[key] = actual == expected if relative is None else relative <= tolerance
            values[key] = {"cpp": actual, "python": expected, "relative_difference": relative}
        report[sequence] = {"frames": cpp["frames"], "raw_position_max_abs_m": position_error,
                            "raw_pose_max_abs": None if poses is None else float(np.max(np.abs(rows - poses))),
                            "metrics": values, "runtime_fps": cpp["runtime_fps"],
                            "runtime_s": cpp["timing"]["total_runtime_s"],
                            "stages": cpp["timing"]["stages"], "checks": checks,
                            "passed": all(checks.values())}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("--golden", type=Path, default=Path(__file__).resolve().parents[1] / "tests/golden/kitti")
    parser.add_argument("--sequences", nargs="+", default=["00", "01", "05"])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = compare(args.outputs, args.golden, args.sequences)
    text = json.dumps(result, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0 if all(run["passed"] for run in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
