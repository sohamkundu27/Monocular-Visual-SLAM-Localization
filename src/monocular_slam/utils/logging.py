"""Logging configuration.

Every run writes a human readable console stream and, when an output directory
is supplied, a verbatim copy to ``<output_dir>/run.log`` so results are
auditable after the fact.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_ROOT_LOGGER_NAME = "monocular_slam"


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    *,
    quiet: bool = False,
) -> logging.Logger:
    """Configure the package logger.

    Parameters
    ----------
    level:
        Logging level for both handlers (``logging.INFO`` by default).
    log_file:
        Optional path to mirror the log stream into. Parent directories are
        created automatically.
    quiet:
        Suppress the console handler (the file handler, if any, still runs).

    Returns
    -------
    logging.Logger
        The configured package-root logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Re-configuring (e.g. across sequences in one process) must not duplicate
    # every message, so tear down previously installed handlers first.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if not quiet:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the package root.

    ``name`` is usually ``__name__``; the ``monocular_slam.`` prefix is stripped
    to keep log lines short.
    """
    short = name
    if short.startswith(_ROOT_LOGGER_NAME + "."):
        short = short[len(_ROOT_LOGGER_NAME) + 1 :]
    elif short == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")
