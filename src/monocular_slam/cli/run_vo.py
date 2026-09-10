"""``run_vo`` entry point: visual odometry only, no loop closure or optimization.

Useful for isolating front-end quality, and for measuring the raw-VO baseline
that the SLAM back end is meant to improve upon.
"""

from __future__ import annotations

import argparse
import sys

from monocular_slam.cli.common import add_common_arguments, config_from_args, resolve_output_dir
from monocular_slam.config import ConfigError
from monocular_slam.datasets.kitti import DatasetError
from monocular_slam.pipeline import run_pipeline

DESCRIPTION = """\
Run monocular visual odometry (front end only) on a KITTI Odometry sequence.

Identical to run_slam.py with loop closure and pose-graph optimization
disabled, so the output is the raw integrated trajectory.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_vo.py",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)
    parser.add_argument("--no-plots", action="store_true", help="Skip figure rendering")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # This entry point is odometry by definition.
    config = config.with_overrides({"loop_closure.enabled": False})

    try:
        result = run_pipeline(config, output_dir=resolve_output_dir(args, config))
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 1

    from monocular_slam.cli.run_slam import _print_summary

    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
