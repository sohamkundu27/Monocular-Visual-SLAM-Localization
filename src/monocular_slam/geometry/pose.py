"""Trajectory container.

A :class:`Trajectory` is an ordered stack of ``T_w_c`` camera-to-world poses
plus the frame indices they belong to. Keeping frame indices explicit matters
because monocular VO is allowed to drop frames, and evaluation must compare
like-for-like against ground truth rather than assuming dense alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from monocular_slam.geometry.transforms import (
    invert_se3,
    is_valid_se3,
    kitti_row_to_se3,
    project_to_se3,
    relative_pose,
    se3_to_kitti_row,
)


@dataclass
class Trajectory:
    """An ordered sequence of camera-to-world poses.

    Parameters
    ----------
    poses:
        ``(N, 4, 4)`` array of SE(3) camera-to-world transforms.
    frame_ids:
        ``(N,)`` integer frame indices. Defaults to ``0..N-1``.
    timestamps:
        Optional ``(N,)`` array of seconds since sequence start.
    """

    poses: np.ndarray
    frame_ids: np.ndarray = field(default=None)  # type: ignore[assignment]
    timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=np.float64)
        if poses.ndim == 2 and poses.shape == (4, 4):
            poses = poses[None, ...]
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must be (N, 4, 4), got {poses.shape}")
        self.poses = poses

        if self.frame_ids is None:
            self.frame_ids = np.arange(len(poses), dtype=np.int64)
        else:
            self.frame_ids = np.asarray(self.frame_ids, dtype=np.int64).reshape(-1)
            if len(self.frame_ids) != len(poses):
                raise ValueError(
                    f"frame_ids length {len(self.frame_ids)} != poses length {len(poses)}"
                )

        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float64).reshape(-1)
            if len(self.timestamps) != len(poses):
                raise ValueError(
                    f"timestamps length {len(self.timestamps)} != poses length {len(poses)}"
                )

    # ----------------------------------------------------------------- #
    # Sequence protocol
    # ----------------------------------------------------------------- #

    def __len__(self) -> int:
        return int(self.poses.shape[0])

    def __getitem__(self, index: int | slice | np.ndarray) -> np.ndarray | Trajectory:
        """Integer indexing yields a pose; slicing/fancy indexing a sub-trajectory."""
        if isinstance(index, (int, np.integer)):
            return self.poses[int(index)]
        return Trajectory(
            poses=self.poses[index],
            frame_ids=self.frame_ids[index],
            timestamps=None if self.timestamps is None else self.timestamps[index],
        )

    def __iter__(self):
        return iter(self.poses)

    # ----------------------------------------------------------------- #
    # Derived quantities
    # ----------------------------------------------------------------- #

    @property
    def positions(self) -> np.ndarray:
        """``(N, 3)`` camera centres in world coordinates."""
        return self.poses[:, :3, 3]

    @property
    def rotations(self) -> np.ndarray:
        """``(N, 3, 3)`` camera-to-world rotations."""
        return self.poses[:, :3, :3]

    @property
    def xz(self) -> np.ndarray:
        """``(N, 2)`` ground-plane coordinates for top-down plots.

        KITTI camera axes are ``+x`` right, ``+y`` down, ``+z`` forward, so the
        ground plane is x-z rather than the more familiar x-y.
        """
        return self.poses[:, [0, 2], 3]

    def step_lengths(self) -> np.ndarray:
        """``(N-1,)`` Euclidean distance between consecutive camera centres."""
        if len(self) < 2:
            return np.zeros(0, dtype=np.float64)
        return np.linalg.norm(np.diff(self.positions, axis=0), axis=1)

    def cumulative_distance(self) -> np.ndarray:
        """``(N,)`` distance travelled along the path up to each pose."""
        return np.concatenate([[0.0], np.cumsum(self.step_lengths())])

    def path_length(self) -> float:
        """Total travelled path length in metres."""
        return float(self.step_lengths().sum())

    def relative_poses(self, delta: int = 1) -> np.ndarray:
        """``(N-delta, 4, 4)`` relative transforms ``T_i_{i+delta}``."""
        if delta < 1:
            raise ValueError("delta must be >= 1")
        n = len(self) - delta
        if n <= 0:
            return np.zeros((0, 4, 4), dtype=np.float64)
        return np.stack([relative_pose(self.poses[i], self.poses[i + delta]) for i in range(n)])

    # ----------------------------------------------------------------- #
    # Transformations
    # ----------------------------------------------------------------- #

    def transformed(self, T: np.ndarray, scale: float = 1.0) -> Trajectory:
        """Apply a global similarity transform ``(scale, T)`` to every pose.

        Positions map as ``p -> s * R p + t`` and orientations as ``R_i -> R R_i``.
        This is exactly the form produced by Umeyama alignment, so it is how an
        estimated trajectory is brought into the ground-truth frame before
        computing absolute trajectory error.
        """
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T must be 4x4, got {T.shape}")
        R, t = T[:3, :3], T[:3, 3]
        out = np.repeat(np.eye(4, dtype=np.float64)[None, ...], len(self), axis=0)
        out[:, :3, :3] = R @ self.rotations
        out[:, :3, 3] = (scale * (R @ self.positions.T)).T + t
        return Trajectory(out, self.frame_ids.copy(), self._copy_timestamps())

    def relative_to_first(self) -> Trajectory:
        """Re-express every pose relative to the first, so pose 0 is identity."""
        if len(self) == 0:
            return self.copy()
        T0_inv = invert_se3(self.poses[0])
        return Trajectory(
            np.einsum("ij,njk->nik", T0_inv, self.poses),
            self.frame_ids.copy(),
            self._copy_timestamps(),
        )

    def orthonormalized(self) -> Trajectory:
        """Re-project every rotation onto SO(3).

        Long chains of matrix products accumulate small orthonormality errors;
        GTSAM rejects rotations that have drifted, so poses are cleaned before
        they enter the pose graph.
        """
        return Trajectory(
            np.stack([project_to_se3(T) for T in self.poses]),
            self.frame_ids.copy(),
            self._copy_timestamps(),
        )

    def copy(self) -> Trajectory:
        return Trajectory(self.poses.copy(), self.frame_ids.copy(), self._copy_timestamps())

    def _copy_timestamps(self) -> np.ndarray | None:
        return None if self.timestamps is None else self.timestamps.copy()

    # ----------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------- #

    def is_valid(self) -> bool:
        """True when every pose is a finite, right-handed SE(3) member."""
        return all(is_valid_se3(T, tol=1e-4) for T in self.poses)

    # ----------------------------------------------------------------- #
    # Selection
    # ----------------------------------------------------------------- #

    def select_frames(self, frame_ids: np.ndarray) -> Trajectory:
        """Return the sub-trajectory whose frame ids match ``frame_ids``.

        Used to line an estimated trajectory up against dense ground truth when
        frames were skipped or sub-sampled.
        """
        wanted = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
        lookup = {int(f): i for i, f in enumerate(self.frame_ids)}
        missing = [int(f) for f in wanted if int(f) not in lookup]
        if missing:
            raise KeyError(
                f"Trajectory is missing {len(missing)} requested frames, e.g. {missing[:5]}"
            )
        idx = np.array([lookup[int(f)] for f in wanted], dtype=np.int64)
        return self[idx]  # type: ignore[return-value]

    # ----------------------------------------------------------------- #
    # Serialisation (KITTI 12-value rows)
    # ----------------------------------------------------------------- #

    @classmethod
    def from_kitti_rows(cls, rows: np.ndarray, frame_ids: np.ndarray | None = None) -> Trajectory:
        """Build a trajectory from an ``(N, 12)`` array of KITTI pose rows."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[1] != 12:
            raise ValueError(f"KITTI pose rows must be (N, 12), got {rows.shape}")
        poses = np.stack([kitti_row_to_se3(row) for row in rows])
        return cls(poses, frame_ids)

    def to_kitti_rows(self) -> np.ndarray:
        """Flatten to an ``(N, 12)`` KITTI-format array."""
        if len(self) == 0:
            return np.zeros((0, 12), dtype=np.float64)
        return np.stack([se3_to_kitti_row(T) for T in self.poses])

    def save_kitti(self, path: Path | str) -> Path:
        """Write KITTI-format poses so external tools (e.g. evo) can read them."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, self.to_kitti_rows(), fmt="%.12e")
        return out

    @classmethod
    def load_kitti(cls, path: Path | str) -> Trajectory:
        """Read a KITTI-format pose file."""
        rows = np.loadtxt(Path(path), dtype=np.float64)
        return cls.from_kitti_rows(rows)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "Trajectory(empty)"
        return (
            f"Trajectory(n={len(self)}, "
            f"frames={int(self.frame_ids[0])}..{int(self.frame_ids[-1])}, "
            f"path_length={self.path_length():.1f} m)"
        )
