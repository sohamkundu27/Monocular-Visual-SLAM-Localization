"""Trajectory and diagnostic plots.

All figures use the non-interactive ``Agg`` backend so runs work over SSH and
in CI, and every function returns the path it wrote so the pipeline can record
its artefacts.

Top-down plots use **X against Z**, not X against Y: KITTI camera axes are
``+x`` right, ``+y`` down, ``+z`` forward, so the ground plane is x-z.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from monocular_slam.geometry.pose import Trajectory  # noqa: E402
from monocular_slam.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

#: Consistent colours across every figure so plots can be compared at a glance.
COLOR_GT = "#111111"
COLOR_RAW = "#d1495b"
COLOR_OPTIMIZED = "#2a9d8f"
COLOR_LOOP = "#3d5a80"
COLOR_ACCENT = "#e9c46a"


def _finish(fig, ax, path: Path, dpi: int, title: str) -> Path:
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Wrote %s", path)
    return path


def _setup_topdown(ax) -> None:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    # Equal aspect matters: without it, a drifting trajectory can be squashed
    # into looking accurate.
    ax.set_aspect("equal", adjustable="datalim")


def plot_trajectory(
    trajectory: Trajectory,
    path: Path | str,
    reference: Trajectory | None = None,
    label: str = "Estimated",
    title: str = "Trajectory (top-down)",
    dpi: int = 150,
    color: str = COLOR_RAW,
) -> Path:
    """Top-down x-z plot of one trajectory, optionally against ground truth."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    if reference is not None:
        ax.plot(
            reference.xz[:, 0], reference.xz[:, 1],
            color=COLOR_GT, linewidth=2.0, label="Ground truth", zorder=2,
        )
    ax.plot(
        trajectory.xz[:, 0], trajectory.xz[:, 1],
        color=color, linewidth=1.6, label=label, zorder=3,
    )
    ax.scatter(
        *trajectory.xz[0], marker="o", s=70, color=COLOR_OPTIMIZED, zorder=5, label="Start"
    )
    ax.scatter(*trajectory.xz[-1], marker="s", s=70, color=COLOR_ACCENT, zorder=5, label="End")
    _setup_topdown(ax)
    return _finish(fig, ax, Path(path), dpi, title)


def plot_trajectory_comparison(
    trajectories: dict[str, Trajectory],
    path: Path | str,
    reference: Trajectory | None = None,
    title: str = "Trajectory comparison",
    dpi: int = 150,
) -> Path:
    """Overlay several trajectories (raw VO, optimized SLAM, ground truth)."""
    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    if reference is not None:
        ax.plot(
            reference.xz[:, 0], reference.xz[:, 1],
            color=COLOR_GT, linewidth=2.4, label="Ground truth", zorder=2,
        )
    palette = [COLOR_RAW, COLOR_OPTIMIZED, COLOR_LOOP, COLOR_ACCENT]
    for i, (label, trajectory) in enumerate(trajectories.items()):
        ax.plot(
            trajectory.xz[:, 0], trajectory.xz[:, 1],
            color=palette[i % len(palette)], linewidth=1.6, label=label, zorder=3 + i,
        )
    if reference is not None:
        ax.scatter(
            *reference.xz[0], marker="o", s=80, color=COLOR_OPTIMIZED, zorder=6, label="Start"
        )
    _setup_topdown(ax)
    return _finish(fig, ax, Path(path), dpi, title)


def plot_trajectory_3d(
    trajectories: dict[str, Trajectory],
    path: Path | str,
    reference: Trajectory | None = None,
    title: str = "Trajectory (3D)",
    dpi: int = 150,
) -> Path:
    """3D view, which is where vertical drift becomes visible.

    KITTI's ``y`` axis points *down*, so it is negated here to make the plot
    read the conventional way up.
    """
    fig = plt.figure(figsize=(9.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    if reference is not None:
        p = reference.positions
        ax.plot(p[:, 0], p[:, 2], -p[:, 1], color=COLOR_GT, linewidth=2.0, label="Ground truth")
    palette = [COLOR_RAW, COLOR_OPTIMIZED, COLOR_LOOP, COLOR_ACCENT]
    for i, (label, trajectory) in enumerate(trajectories.items()):
        p = trajectory.positions
        ax.plot(
            p[:, 0], p[:, 2], -p[:, 1],
            color=palette[i % len(palette)], linewidth=1.4, label=label,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_zlabel("height [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _equalize_3d(ax)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def _equalize_3d(ax) -> None:
    """Force an equal aspect ratio on a 3D axis.

    Matplotlib does not do this for 3D axes, and without it the vertical axis
    is stretched enormously, making a few metres of height drift look like a
    catastrophic failure.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centres = limits.mean(axis=1)
    radius = 0.5 * float(np.max(limits[:, 1] - limits[:, 0]))
    radius = max(radius, 1e-3)
    ax.set_xlim3d(centres[0] - radius, centres[0] + radius)
    ax.set_ylim3d(centres[1] - radius, centres[1] + radius)
    ax.set_zlim3d(centres[2] - radius, centres[2] + radius)


def plot_loop_closures(
    trajectory: Trajectory,
    loops: list[tuple[int, int]],
    path: Path | str,
    reference: Trajectory | None = None,
    title: str = "Detected loop closures",
    dpi: int = 150,
) -> Path:
    """Draw accepted loop-closure edges on the estimated trajectory.

    ``loops`` holds ``(index_a, index_b)`` pairs indexing into ``trajectory``.
    """
    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    if reference is not None:
        ax.plot(
            reference.xz[:, 0], reference.xz[:, 1],
            color=COLOR_GT, linewidth=1.6, alpha=0.5, label="Ground truth", zorder=1,
        )
    ax.plot(trajectory.xz[:, 0], trajectory.xz[:, 1], color=COLOR_RAW, linewidth=1.6,
            label="Estimated", zorder=2)

    xz = trajectory.xz
    for k, (i, j) in enumerate(loops):
        if not (0 <= i < len(xz) and 0 <= j < len(xz)):
            continue
        ax.plot(
            [xz[i, 0], xz[j, 0]], [xz[i, 1], xz[j, 1]],
            color=COLOR_LOOP, linewidth=1.1, alpha=0.85, zorder=4,
            label="Loop closure" if k == 0 else None,
        )
        ax.scatter([xz[i, 0], xz[j, 0]], [xz[i, 1], xz[j, 1]], s=14, color=COLOR_LOOP, zorder=5)

    _setup_topdown(ax)
    return _finish(fig, ax, Path(path), dpi, f"{title} ({len(loops)} accepted)")


def plot_error_over_time(
    errors: dict[str, np.ndarray],
    path: Path | str,
    distances: np.ndarray | None = None,
    title: str = "Absolute position error",
    dpi: int = 150,
) -> Path:
    """Plot per-frame position error against frame index or travelled distance."""
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    palette = [COLOR_RAW, COLOR_OPTIMIZED, COLOR_LOOP, COLOR_ACCENT]
    for i, (label, values) in enumerate(errors.items()):
        x = np.arange(len(values)) if distances is None else distances[: len(values)]
        ax.plot(x, values, color=palette[i % len(palette)], linewidth=1.3, label=label)
    ax.set_xlabel("frame index" if distances is None else "distance travelled [m]")
    ax.set_ylabel("position error [m]")
    ax.set_ylim(bottom=0)
    return _finish(fig, ax, Path(path), dpi, title)


def plot_diagnostics(
    diagnostics: list,
    path: Path | str,
    title: str = "Front-end diagnostics",
    dpi: int = 150,
) -> Path:
    """Four-panel summary of feature, match and inlier behaviour over the run."""
    if not diagnostics:
        raise ValueError("No diagnostics to plot")

    frames = np.array([d.index for d in diagnostics])
    matches = np.array([d.n_matches for d in diagnostics], dtype=float)
    inliers = np.array([d.n_ransac_inliers for d in diagnostics], dtype=float)
    ratio = np.array([d.ransac_inlier_ratio for d in diagnostics], dtype=float) * 100.0
    rotation = np.array([d.rotation_deg for d in diagnostics], dtype=float)
    failed = np.array([not d.succeeded for d in diagnostics])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))

    ax = axes[0, 0]
    ax.plot(frames, matches, color=COLOR_LOOP, linewidth=0.9, label="filtered matches")
    ax.plot(frames, inliers, color=COLOR_OPTIMIZED, linewidth=0.9, label="epipolar inliers")
    ax.set_ylabel("count")
    ax.set_title("Matches and inliers")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(frames, ratio, color=COLOR_RAW, linewidth=0.9)
    ax.set_ylabel("inlier ratio [%]")
    ax.set_ylim(0, 100)
    ax.set_title("RANSAC inlier ratio")

    ax = axes[1, 0]
    ax.plot(frames, rotation, color=COLOR_LOOP, linewidth=0.9)
    ax.set_ylabel("rotation [deg]")
    ax.set_xlabel("frame")
    ax.set_title("Per-frame rotation magnitude")

    ax = axes[1, 1]
    ax.plot(frames, np.cumsum(failed), color=COLOR_RAW, linewidth=1.2)
    ax.set_ylabel("cumulative failures")
    ax.set_xlabel("frame")
    ax.set_title(f"Failed transitions ({int(failed.sum())} of {len(failed)})")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
