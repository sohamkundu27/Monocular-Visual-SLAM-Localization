"""KITTI Odometry dataset loader.

Expected layout (as produced by unzipping the official odometry downloads)::

    <root>/
      sequences/
        00/
          calib.txt
          times.txt
          image_0/   000000.png 000001.png ...
          image_1/   ...
        01/ ...
      poses/
        00.txt  01.txt ...   # ground truth, sequences 00-10 only

Ground-truth rows are the ``3x4`` camera-to-world transform of the **left
camera** (``image_0``), which is exactly the camera the monocular pipeline
runs on, so no extrinsic conversion is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

from monocular_slam.datasets.calibration import (
    CalibrationError,
    CameraCalibration,
    calibration_for_camera,
    parse_kitti_calib,
)
from monocular_slam.geometry.pose import Trajectory
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)

#: Sequences 11-21 are the held-out test split and ship without ground truth.
SEQUENCES_WITH_GROUND_TRUTH = frozenset(f"{i:02d}" for i in range(11))

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class DatasetError(FileNotFoundError):
    """Raised when the dataset layout is missing or malformed."""


@dataclass(frozen=True)
class FrameSelection:
    """Which frames a run will actually consume."""

    start: int = 0
    step: int = 1
    max_frames: int | None = None

    def apply(self, total: int) -> np.ndarray:
        """Return the selected frame indices within ``[0, total)``."""
        if self.step < 1:
            raise ValueError(f"frame_step must be >= 1, got {self.step}")
        if self.start < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start}")
        if self.start >= total:
            raise ValueError(f"start_frame {self.start} is past the end of the sequence ({total})")
        indices = np.arange(self.start, total, self.step, dtype=np.int64)
        if self.max_frames:
            indices = indices[: self.max_frames]
        return indices


class KittiOdometryDataset:
    """Random-access reader for one KITTI Odometry sequence.

    Images are read lazily so a 4541-frame sequence never has to fit in RAM.
    Indexing is over the *selected* frames: ``dataset[0]`` is the first frame a
    run will process, and ``dataset.frame_ids[0]`` gives its original index in
    the sequence.
    """

    def __init__(
        self,
        root: Path | str,
        sequence: str = "00",
        camera: str = "image_0",
        *,
        grayscale: bool = True,
        start_frame: int = 0,
        frame_step: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.sequence = _normalize_sequence(sequence)
        self.camera = camera
        self.grayscale = grayscale

        self.sequence_dir = self.root / "sequences" / self.sequence
        if not self.sequence_dir.is_dir():
            raise DatasetError(
                f"Sequence directory not found: {self.sequence_dir}\n"
                f"Available sequences under {self.root / 'sequences'}: "
                f"{_available_sequences(self.root)}"
            )

        self.image_dir = self.sequence_dir / camera
        if not self.image_dir.is_dir():
            raise DatasetError(
                f"Image directory not found: {self.image_dir}. "
                f"Present in this sequence: {sorted(p.name for p in self.sequence_dir.iterdir())}"
            )

        all_paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not all_paths:
            raise DatasetError(f"No images found in {self.image_dir}")

        self.selection = FrameSelection(start=start_frame, step=frame_step, max_frames=max_frames)
        self.frame_ids = self.selection.apply(len(all_paths))
        self.image_paths = [all_paths[i] for i in self.frame_ids]
        self._total_frames = len(all_paths)

        logger.info(
            "KITTI sequence %s: %d/%d frames selected from %s (start=%d, step=%d)",
            self.sequence,
            len(self.frame_ids),
            self._total_frames,
            self.image_dir,
            start_frame,
            frame_step,
        )

    # ----------------------------------------------------------------- #
    # Construction helpers
    # ----------------------------------------------------------------- #

    @classmethod
    def from_config(cls, config) -> KittiOdometryDataset:  # noqa: ANN001 - avoid circular import
        """Build a dataset from a :class:`monocular_slam.config.Config`."""
        d = config.dataset
        return cls(
            root=d.path,
            sequence=d.sequence,
            camera=d.camera,
            grayscale=d.grayscale,
            start_frame=d.start_frame,
            frame_step=d.frame_step,
            max_frames=d.max_frames,
        )

    # ----------------------------------------------------------------- #
    # Sequence protocol
    # ----------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.load_image(index)

    def __iter__(self):
        for i in range(len(self)):
            yield self.load_image(i)

    def load_image(self, index: int) -> np.ndarray:
        """Read the ``index``-th selected frame.

        Raises
        ------
        DatasetError
            If the file exists in the listing but cannot be decoded.
        """
        path = self.image_paths[index]
        flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(path), flag)
        if image is None:
            raise DatasetError(f"Failed to decode image: {path}")
        return image

    # ----------------------------------------------------------------- #
    # Calibration
    # ----------------------------------------------------------------- #

    @cached_property
    def calib_raw(self) -> dict[str, np.ndarray]:
        """All projection matrices from this sequence's ``calib.txt``."""
        return parse_kitti_calib(self.sequence_dir / "calib.txt")

    @cached_property
    def calibration(self) -> CameraCalibration:
        """Intrinsics of the configured camera."""
        calib = calibration_for_camera(self.calib_raw, self.camera)
        logger.info("Loaded calibration %r", calib)
        return calib

    @property
    def K(self) -> np.ndarray:
        """``3x3`` intrinsic matrix of the configured camera."""
        return self.calibration.K

    @cached_property
    def image_size(self) -> tuple[int, int]:
        """``(width, height)`` of the first frame."""
        image = self.load_image(0)
        return int(image.shape[1]), int(image.shape[0])

    # ----------------------------------------------------------------- #
    # Timestamps
    # ----------------------------------------------------------------- #

    @cached_property
    def timestamps(self) -> np.ndarray | None:
        """Seconds since sequence start for the selected frames, if available."""
        times_path = self.sequence_dir / "times.txt"
        if not times_path.is_file():
            logger.warning("No times.txt in %s; timestamps unavailable", self.sequence_dir)
            return None
        times = np.loadtxt(times_path, dtype=np.float64).reshape(-1)
        if len(times) < self._total_frames:
            raise DatasetError(
                f"{times_path} has {len(times)} timestamps but the sequence has "
                f"{self._total_frames} images"
            )
        return times[self.frame_ids]

    # ----------------------------------------------------------------- #
    # Ground truth
    # ----------------------------------------------------------------- #

    @property
    def poses_path(self) -> Path:
        return self.root / "poses" / f"{self.sequence}.txt"

    @property
    def has_ground_truth(self) -> bool:
        return self.poses_path.is_file()

    @cached_property
    def ground_truth(self) -> Trajectory | None:
        """Ground-truth trajectory restricted to the selected frames.

        Returns ``None`` for the test-split sequences (11-21), which ship
        without poses. Metrics that require ground truth are skipped rather
        than fabricated in that case.
        """
        if not self.has_ground_truth:
            if self.sequence in SEQUENCES_WITH_GROUND_TRUTH:
                logger.warning(
                    "Sequence %s should have ground truth but %s is missing",
                    self.sequence,
                    self.poses_path,
                )
            else:
                logger.info(
                    "Sequence %s is part of the test split (no ground truth)", self.sequence
                )
            return None

        rows = np.loadtxt(self.poses_path, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.shape[1] != 12:
            raise DatasetError(
                f"{self.poses_path}: expected 12 values per row, "
                f"got {rows.shape[1]}"
            )
        if len(rows) < self._total_frames:
            raise DatasetError(
                f"{self.poses_path} has {len(rows)} poses but the sequence has "
                f"{self._total_frames} images"
            )

        trajectory = Trajectory.from_kitti_rows(rows[self.frame_ids], frame_ids=self.frame_ids)
        if not trajectory.is_valid():
            raise DatasetError(f"{self.poses_path} contains poses that are not valid SE(3)")
        logger.info(
            "Loaded ground truth: %d poses, %.1f m travelled",
            len(trajectory),
            trajectory.path_length(),
        )
        return trajectory

    def ground_truth_scales(self) -> np.ndarray | None:
        """Per-transition ground-truth translation magnitudes, in metres.

        ``scales[i]`` is the distance between selected frames ``i`` and
        ``i + 1``. This is the quantity used by the ground-truth scale strategy
        described in :mod:`monocular_slam.odometry.scale`.
        """
        gt = self.ground_truth
        return None if gt is None else gt.step_lengths()

    def describe(self) -> dict[str, object]:
        """Small summary dict used for run metadata and logging."""
        gt = self.ground_truth
        return {
            "sequence": self.sequence,
            "camera": self.camera,
            "root": str(self.root),
            "frames_available": self._total_frames,
            "frames_selected": len(self),
            "start_frame": int(self.selection.start),
            "frame_step": int(self.selection.step),
            "image_size": list(self.image_size),
            "fx": self.calibration.fx,
            "fy": self.calibration.fy,
            "cx": self.calibration.cx,
            "cy": self.calibration.cy,
            "has_ground_truth": gt is not None,
            "ground_truth_path_length_m": None if gt is None else round(gt.path_length(), 3),
        }

    def __repr__(self) -> str:
        return (
            f"KittiOdometryDataset(sequence={self.sequence!r}, camera={self.camera!r}, "
            f"frames={len(self)})"
        )


def _normalize_sequence(sequence: str | int) -> str:
    """Accept ``0``, ``"0"`` or ``"00"`` and return the canonical ``"00"``."""
    text = str(sequence).strip()
    if not text:
        raise DatasetError("Sequence identifier must not be empty")
    if text.isdigit():
        return f"{int(text):02d}"
    return text


def _available_sequences(root: Path) -> list[str]:
    sequences_dir = root / "sequences"
    if not sequences_dir.is_dir():
        return []
    return sorted(p.name for p in sequences_dir.iterdir() if p.is_dir())


__all__ = [
    "CalibrationError",
    "DatasetError",
    "FrameSelection",
    "KittiOdometryDataset",
    "SEQUENCES_WITH_GROUND_TRUTH",
]
