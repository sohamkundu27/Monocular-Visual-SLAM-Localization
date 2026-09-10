"""Pinhole camera calibration and KITTI ``calib.txt`` parsing.

KITTI Odometry ships one ``calib.txt`` per sequence containing the ``3x4``
projection matrices of the four rectified cameras plus the velodyne-to-camera
extrinsic:

    P0: fx 0 cx 0  0 fy cy 0  0 0 1 0        # left  grayscale (reference)
    P1: ...                                   # right grayscale
    P2: ...                                   # left  colour
    P3: ...                                   # right colour
    Tr: r11 r12 r13 tx  r21 ...               # velodyne -> camera 0

Because the images are already rectified, the intrinsic matrix is simply the
left ``3x3`` block of ``P`` and there is no lens distortion to undo. The fourth
column encodes the stereo baseline as ``-fx * b``; the monocular pipeline never
uses it, but it is parsed so the baseline is available for reference and for
any future stereo-based scale estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CalibrationError(ValueError):
    """Raised when a calibration file cannot be parsed or is inconsistent."""


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics of a single rectified pinhole camera.

    Attributes
    ----------
    K:
        ``3x3`` intrinsic matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
    P:
        ``3x4`` projection matrix as stored by KITTI.
    name:
        Which projection row it came from (e.g. ``"P0"``).
    """

    K: np.ndarray
    P: np.ndarray
    name: str = "P0"

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=np.float64)
        P = np.asarray(self.P, dtype=np.float64)
        if K.shape != (3, 3):
            raise CalibrationError(f"K must be 3x3, got {K.shape}")
        if P.shape != (3, 4):
            raise CalibrationError(f"P must be 3x4, got {P.shape}")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(P)):
            raise CalibrationError("Calibration contains non-finite values")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise CalibrationError(f"Focal lengths must be positive, got {K[0, 0]}, {K[1, 1]}")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "P", P)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def baseline_m(self) -> float:
        """Stereo baseline relative to the reference camera, in metres.

        KITTI encodes it as ``P[0, 3] == -fx * baseline``, so a camera whose
        projection matrix has a zero fourth column (the reference camera)
        reports ``0.0``.
        """
        return float(-self.P[0, 3] / self.fx)

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Convert ``(N, 2)`` pixel coordinates to normalized image coordinates.

        ``x_n = (u - cx) / fx``, ``y_n = (v - cy) / fy``. Useful when a routine
        needs calibrated rays rather than handing ``K`` to OpenCV.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be (N, 2), got {pts.shape}")
        return np.column_stack(
            [(pts[:, 0] - self.cx) / self.fx, (pts[:, 1] - self.cy) / self.fy]
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project ``(N, 3)`` camera-frame points to ``(N, 2)`` pixels.

        Points at or behind the image plane (``z <= 0``) produce ``NaN`` rather
        than silently wrapping around to a bogus pixel.
        """
        pts = np.asarray(points_cam, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_cam must be (N, 3), got {pts.shape}")
        z = pts[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * pts[:, 0] / z + self.cx
            v = self.fy * pts[:, 1] / z + self.cy
        uv = np.column_stack([u, v])
        uv[z <= 0] = np.nan
        return uv

    def __repr__(self) -> str:
        return (
            f"CameraCalibration({self.name}, fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f})"
        )


def parse_kitti_calib(path: Path | str) -> dict[str, np.ndarray]:
    """Parse a KITTI ``calib.txt`` into ``{key: matrix}``.

    ``P0``-``P3`` come back as ``3x4`` matrices and ``Tr`` (when present) as a
    ``4x4`` SE(3) transform with the implicit bottom row filled in.
    """
    calib_path = Path(path)
    if not calib_path.is_file():
        raise CalibrationError(f"Calibration file not found: {calib_path}")

    entries: dict[str, np.ndarray] = {}
    for lineno, raw in enumerate(calib_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise CalibrationError(f"{calib_path}:{lineno}: expected 'key: values', got {raw!r}")
        key, _, values = line.partition(":")
        key = key.strip()
        try:
            numbers = np.fromstring(values, sep=" ", dtype=np.float64)
        except ValueError as exc:  # pragma: no cover - numpy rarely raises here
            raise CalibrationError(f"{calib_path}:{lineno}: unparsable values") from exc
        if numbers.size != 12:
            raise CalibrationError(
                f"{calib_path}:{lineno}: key '{key}' has {numbers.size} values, expected 12"
            )
        if not np.all(np.isfinite(numbers)):
            raise CalibrationError(f"{calib_path}:{lineno}: key '{key}' has non-finite values")

        if key == "Tr":
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = numbers.reshape(3, 4)
            entries[key] = T
        else:
            entries[key] = numbers.reshape(3, 4)

    if not entries:
        raise CalibrationError(f"{calib_path}: no calibration entries found")
    return entries


#: Maps the image directory name to the projection matrix that describes it.
CAMERA_TO_PROJECTION = {
    "image_0": "P0",  # left grayscale (the monocular SLAM camera)
    "image_1": "P1",  # right grayscale
    "image_2": "P2",  # left colour
    "image_3": "P3",  # right colour
}


def calibration_for_camera(
    calib: dict[str, np.ndarray], camera: str = "image_0"
) -> CameraCalibration:
    """Extract the :class:`CameraCalibration` for a KITTI image directory."""
    if camera not in CAMERA_TO_PROJECTION:
        raise CalibrationError(
            f"Unknown camera '{camera}'. Expected one of {sorted(CAMERA_TO_PROJECTION)}"
        )
    key = CAMERA_TO_PROJECTION[camera]
    if key not in calib:
        raise CalibrationError(f"Calibration is missing '{key}' (available: {sorted(calib)})")
    P = calib[key]
    return CameraCalibration(K=P[:3, :3].copy(), P=P.copy(), name=key)
