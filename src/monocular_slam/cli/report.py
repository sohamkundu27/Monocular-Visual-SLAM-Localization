"""``report`` entry point: regenerate the resume metrics summary from run outputs."""

from __future__ import annotations

import argparse
import sys

from monocular_slam.reporting import (
    load_run_summaries,
    render_resume_metrics,
    update_readme_results,
    write_resume_metrics,
)
from monocular_slam.utils.logging import setup_logging

DESCRIPTION = """\
Generate outputs/resume_metrics.md from the metrics.json files that completed
runs produced.

Only measured values are used: any metric a run could not compute is rendered
as n/a and is never turned into a claim.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="report.py",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--outputs", type=str, default="outputs",
        help="Directory containing sequence_XX/metrics.json (default: outputs)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Destination file (default: <outputs>/resume_metrics.md)",
    )
    parser.add_argument("--print", action="store_true", help="Also print the document")
    parser.add_argument(
        "--update-readme", action="store_true",
        help="Also inject the results table into README.md between its RESULTS markers",
    )
    parser.add_argument(
        "--readme", type=str, default="README.md", help="README path (default: README.md)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    summaries = load_run_summaries(args.outputs)
    if not summaries:
        print(
            f"No metrics.json files found under '{args.outputs}'. Run the pipeline first:\n"
            "  python scripts/run_slam.py --sequence 00 --dataset-path /path/to/KITTI/dataset",
            file=sys.stderr,
        )

    path = write_resume_metrics(args.outputs, args.output)
    print(f"Wrote {path} from {len(summaries)} run(s)")

    if args.update_readme and update_readme_results(args.readme, args.outputs):
        print(f"Updated results block in {args.readme}")

    if args.print:
        print()
        print(render_resume_metrics(summaries))
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
