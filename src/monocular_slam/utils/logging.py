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


def _as_level(level: int | str, default: int = logging.INFO) -> int:
    if isinstance(level, str):
        return getattr(logging, level.upper(), default)
    return int(level)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    *,
    quiet: bool = False,
    file_level: int | str = logging.INFO,
) -> logging.Logger:
    """Configure the package logger.

    Parameters
    ----------
    level:
        Console logging level.
    log_file:
        Optional path to mirror the log stream into. Parent directories are
        created automatically.
    quiet:
        Suppress the console handler (the file handler, if any, still runs).
    file_level:
        Level for the file handler, independent of the console. Defaults to
        ``INFO`` so ``run.log`` is always a complete record of the run even
        when the console is turned down — an empty log next to a set of result
        files is worse than useless when you come back to it later.

    Returns
    -------
    logging.Logger
        The configured package-root logger.
    """
    level = _as_level(level)
    file_level = _as_level(file_level)

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    # The logger itself must pass through whatever the most verbose handler
    # wants; each handler then applies its own threshold.
    logger.setLevel(min(level, file_level))
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
        file_handler.setLevel(file_level)
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
