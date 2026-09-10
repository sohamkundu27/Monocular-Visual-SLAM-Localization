#!/usr/bin/env python3
"""Thin wrapper so the tool can be run from a checkout without installing.

Delegates to monocular_slam.cli.run_slam; the installed console script
(see pyproject.toml [project.scripts]) calls the same entry point.
"""

import sys
from pathlib import Path

# Support running straight from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monocular_slam.cli.run_slam import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
