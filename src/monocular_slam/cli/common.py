"""Shared CLI argument handling.

Configuration is resolved in three layers, each overriding the last:

1. dataclass defaults in :mod:`monocular_slam.config`
2. the YAML file given by ``--config``
3. explicit command-line flags, plus arbitrary ``--set key=value`` overrides
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from monocular_slam.config import Config, ConfigError

DEFAULT_CONFIG = Path("configs/kitti.yaml")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the flags every entry point shares."""
    dataset = parser.add_argument_group("dataset")
    dataset.add_argument(
        "--sequence", "-s", type=str, default=None,
        help="KITTI sequence id, e.g. 00 (default: from config)",
    )
    dataset.add_argument(
        "--dataset-path", type=str, default=None,
        help="Root of the KITTI odometry download (the directory holding sequences/)",
    )
    dataset.add_argument(
        "--max-frames", type=int, default=None,
        help="Process at most this many frames (useful for a quick smoke run)",
    )
    dataset.add_argument(
        "--start-frame", type=int, default=None, help="Skip this many frames at the start",
    )
    dataset.add_argument(
        "--frame-step", type=int, default=None, help="Process every Nth frame",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--config", "-c", type=str, default=str(DEFAULT_CONFIG),
        help=f"YAML config file (default: {DEFAULT_CONFIG})",
    )
    runtime.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Directory for run artefacts (default: <output.root>/sequence_<seq>)",
    )
    runtime.add_argument("--seed", type=int, default=None, help="Random seed")
    runtime.add_argument(
        "--log-level", type=str, default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level",
    )
    runtime.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override any config key, e.g. --set odometry.min_inliers=40 "
             "(repeatable)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build the resolved :class:`Config` from parsed arguments."""
    config_path = Path(args.config) if args.config else None
    if config_path is not None and not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}\n"
            f"Pass --config <path>, or run from the repository root where "
            f"{DEFAULT_CONFIG} lives."
        )

    overrides: dict[str, Any] = {}
    flag_map = {
        "sequence": "dataset.sequence",
        "dataset_path": "dataset.path",
        "max_frames": "dataset.max_frames",
        "start_frame": "dataset.start_frame",
        "frame_step": "dataset.frame_step",
        "seed": "runtime.seed",
        "log_level": "runtime.log_level",
    }
    for attribute, key in flag_map.items():
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[key] = value

    for item in getattr(args, "overrides", []) or []:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got '{item}'")
        key, _, value = item.partition("=")
        overrides[key.strip()] = _parse_scalar(value.strip())

    for key in ("loop_closure", "optimize", "plots"):
        flag = getattr(args, f"no_{key}", False)
        if flag:
            overrides[_NEGATION_KEYS[key]] = False

    return Config.load(config_path, overrides)


_NEGATION_KEYS = {
    "loop_closure": "loop_closure.enabled",
    "optimize": "loop_closure.enabled",
    "plots": "output.save_plots",
}


def _parse_scalar(text: str) -> Any:
    """Convert a ``--set`` value string to bool/int/float/None where sensible.

    Sequence ids such as ``00`` must survive as strings, which is why the
    integer branch rejects anything with a leading zero.
    """
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    if text.lstrip("-").isdigit() and not (len(text) > 1 and text.lstrip("-").startswith("0")):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def resolve_output_dir(args: argparse.Namespace, config: Config) -> Path:
    """Directory for this run's artefacts."""
    return Path(args.output_dir) if getattr(args, "output_dir", None) else config.run_dir
