"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from monocular_slam.geometry.transforms import se3_from_rt


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator so failures are reproducible."""
    return np.random.default_rng(20240115)


def random_se3(rng: np.random.Generator, max_translation: float = 5.0) -> np.ndarray:
    """Sample a random valid SE(3) pose."""
    R = Rotation.random(random_state=int(rng.integers(0, 2**31 - 1))).as_matrix()
    t = rng.uniform(-max_translation, max_translation, size=3)
    return se3_from_rt(R, t)


def small_se3(rng: np.random.Generator, angle_deg: float = 5.0, trans: float = 1.0) -> np.ndarray:
    """Sample a small motion, closer to real frame-to-frame camera movement."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(-angle_deg, angle_deg))
    R = Rotation.from_rotvec(axis * angle).as_matrix()
    t = rng.normal(size=3) * trans
    return se3_from_rt(R, t)
