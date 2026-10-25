"""Deterministic seeding.

RANSAC inside OpenCV draws from its own RNG, so reproducing a run byte-for-byte
requires seeding ``cv2`` in addition to Python and NumPy.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

DEFAULT_SEED = 0


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed Python, NumPy and OpenCV RNGs.

    Returns the seed so callers can record it in run metadata.
    """
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(int(seed))
    return seed
