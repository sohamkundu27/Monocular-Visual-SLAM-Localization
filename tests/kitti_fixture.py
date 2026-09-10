"""Builds a miniature, on-disk KITTI Odometry tree for tests.

Everything the loader touches — ``calib.txt``, ``times.txt``, ``image_0/`` and
``poses/xx.txt`` — is synthesised, so the whole test suite runs without the
20 GB official download. Real numbers from sequence 00 are used for the
intrinsics so the parser is exercised against genuine formatting.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from monocular_slam.geometry.transforms import se3_from_rt, se3_to_kitti_row

# Verbatim from KITTI sequence 00 calib.txt.
CALIB_TEXT = """P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 4.538225000000e+01 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 -1.130887000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.779761000000e-03
P3: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.372877000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 2.369057000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.915215000000e-03
Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01
"""

KITTI_00_FX = 718.856
KITTI_00_CX = 607.1928
KITTI_00_CY = 185.2157


def make_textured_image(
    index: int, width: int = 320, height: int = 120, seed: int = 0
) -> np.ndarray:
    """Render a deterministic, feature-rich grayscale image.

    Random noise alone produces unstable ORB keypoints, so the image is built
    from blobs and edges that shift smoothly with ``index`` — enough structure
    for the detector and matcher to behave like they do on real frames.
    """
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 110, dtype=np.uint8)
    shift = index * 3

    for _ in range(45):
        cx = int(rng.integers(10, width - 10))
        cy = int(rng.integers(10, height - 10))
        radius = int(rng.integers(3, 9))
        shade = int(rng.integers(0, 255))
        cv2.circle(image, ((cx - shift) % width, cy), radius, shade, -1)

    for _ in range(12):
        x0 = int(rng.integers(0, width))
        y0 = int(rng.integers(0, height))
        cv2.rectangle(
            image,
            ((x0 - shift) % width, y0),
            ((x0 - shift + 25) % width, min(y0 + 18, height - 1)),
            int(rng.integers(0, 255)),
            -1,
        )

    return image


def write_kitti_sequence(
    root: Path,
    sequence: str = "00",
    n_frames: int = 12,
    *,
    with_ground_truth: bool = True,
    with_times: bool = True,
    camera: str = "image_0",
    step_m: float = 0.9,
) -> Path:
    """Write a synthetic KITTI sequence under ``root`` and return ``root``.

    The ground truth is a gentle right-hand curve travelling along ``+z``,
    matching KITTI's camera convention (``+x`` right, ``+y`` down, ``+z``
    forward).
    """
    sequence_dir = root / "sequences" / sequence
    image_dir = sequence_dir / camera
    image_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        cv2.imwrite(str(image_dir / f"{i:06d}.png"), make_textured_image(i))

    (sequence_dir / "calib.txt").write_text(CALIB_TEXT, encoding="utf-8")

    if with_times:
        times = np.arange(n_frames) * 0.1037
        np.savetxt(sequence_dir / "times.txt", times, fmt="%.6e")

    if with_ground_truth:
        poses_dir = root / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(poses_dir / f"{sequence}.txt", ground_truth_rows(n_frames, step_m), fmt="%.12e")

    return root


def ground_truth_rows(n_frames: int, step_m: float = 0.9, yaw_rate_deg: float = 1.5) -> np.ndarray:
    """Generate ``(N, 12)`` KITTI pose rows for a constant-curvature drive."""
    rows = []
    T = np.eye(4)
    for i in range(n_frames):
        rows.append(se3_to_kitti_row(T))
        yaw = np.radians(yaw_rate_deg)
        R_step = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ]
        )
        T = T @ se3_from_rt(R_step, np.array([0.0, 0.0, step_m]))
        del i
    return np.stack(rows)
