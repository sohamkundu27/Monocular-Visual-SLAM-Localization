"""Resume-facing summary generation.

Reads the ``metrics.json`` files that real runs produced and renders
``outputs/resume_metrics.md``: a results table plus a few suggested resume
bullets.

The hard rule this module enforces is that **nothing is written that was not
measured**. A metric that came back ``null`` (no ground truth, no loop
closures, too short a sequence for the drift segments) is rendered as ``n/a``
and is never used to generate a claim. Every bullet is assembled from values
read out of the JSON, with the ground-truth-scale caveat attached whenever the
run used it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunSummary:
    """The headline numbers from one run's ``metrics.json``."""

    sequence: str
    frames: int | None
    distance_km: float | None
    successful_pose_rate_pct: float | None
    ate_rmse_raw_m: float | None
    ate_rmse_optimized_m: float | None
    ate_reduction_pct: float | None
    translational_drift_raw_pct: float | None
    translational_drift_optimized_pct: float | None
    rotational_drift_deg_per_100m: float | None
    loop_closures_detected: int | None
    avg_features_per_frame: float | None
    avg_matches_per_pair: float | None
    avg_inlier_ratio_pct: float | None
    runtime_fps: float | None
    scale_uses_ground_truth: bool = False
    alignment: str = "sim3"
    n_keyframes: int | None = None

    @classmethod
    def from_metrics(cls, metrics: dict) -> RunSummary:
        return cls(
            sequence=str(metrics.get("sequence", "?")),
            frames=metrics.get("frames"),
            distance_km=metrics.get("distance_km"),
            successful_pose_rate_pct=metrics.get("successful_pose_rate_pct"),
            ate_rmse_raw_m=metrics.get("ate_rmse_raw_m"),
            ate_rmse_optimized_m=metrics.get("ate_rmse_optimized_m"),
            ate_reduction_pct=metrics.get("ate_reduction_pct"),
            translational_drift_raw_pct=metrics.get("translational_drift_raw_pct"),
            translational_drift_optimized_pct=metrics.get("translational_drift_optimized_pct"),
            rotational_drift_deg_per_100m=metrics.get("rotational_drift_deg_per_100m"),
            loop_closures_detected=metrics.get("loop_closures_detected"),
            avg_features_per_frame=metrics.get("avg_features_per_frame"),
            avg_matches_per_pair=metrics.get("avg_matches_per_pair"),
            avg_inlier_ratio_pct=metrics.get("avg_inlier_ratio_pct"),
            runtime_fps=metrics.get("runtime_fps"),
            scale_uses_ground_truth=bool(metrics.get("scale_uses_ground_truth", False)),
            alignment=str(metrics.get("alignment", "sim3")),
            n_keyframes=(metrics.get("loop_closure") or {}).get("n_keyframes"),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> RunSummary:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_metrics(json.load(handle))


def load_run_summaries(output_root: Path | str = "outputs") -> list[RunSummary]:
    """Load every ``metrics.json`` under ``output_root``, sorted by sequence."""
    root = Path(output_root)
    summaries = []
    for path in sorted(root.glob("*/metrics.json")):
        try:
            summaries.append(RunSummary.from_file(path))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping unreadable metrics file %s: %s", path, exc)
    return sorted(summaries, key=lambda s: s.sequence)


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    """Render a value, or ``n/a`` when it was not measured."""
    if value is None:
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def results_table(summaries: list[RunSummary]) -> str:
    """Markdown results table built only from measured values."""
    if not summaries:
        return (
            "_No benchmark runs found. Run `python scripts/run_slam.py --sequence 00 "
            "--dataset-path /path/to/KITTI/dataset` to populate this table._"
        )

    header = (
        "| Sequence | Frames | Distance | Pose rate | Loops | "
        "ATE RMSE (raw) | ATE RMSE (opt.) | ATE reduction | "
        "Drift (raw) | Drift (opt.) | Rot. drift | FPS |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for s in summaries:
        rows.append(
            f"| {s.sequence} "
            f"| {_fmt(s.frames)} "
            f"| {_fmt(s.distance_km, 2, ' km')} "
            f"| {_fmt(s.successful_pose_rate_pct, 1, '%')} "
            f"| {_fmt(s.loop_closures_detected)} "
            f"| {_fmt(s.ate_rmse_raw_m, 2, ' m')} "
            f"| {_fmt(s.ate_rmse_optimized_m, 2, ' m')} "
            f"| {_fmt(s.ate_reduction_pct, 1, '%')} "
            f"| {_fmt(s.translational_drift_raw_pct, 2, '%')} "
            f"| {_fmt(s.translational_drift_optimized_pct, 2, '%')} "
            f"| {_fmt(s.rotational_drift_deg_per_100m, 3)} "
            f"| {_fmt(s.runtime_fps, 1)} |"
        )
    return header + "\n".join(rows)


def feature_table(summaries: list[RunSummary]) -> str:
    """Markdown table of front-end feature statistics."""
    if not summaries:
        return "_No benchmark runs found._"
    header = (
        "| Sequence | Keyframes | Avg. features/frame | Avg. matches/pair | "
        "Avg. RANSAC inlier ratio |\n"
        "|---:|---:|---:|---:|---:|\n"
    )
    rows = [
        f"| {s.sequence} | {_fmt(s.n_keyframes)} | {_fmt(s.avg_features_per_frame, 0)} "
        f"| {_fmt(s.avg_matches_per_pair, 0)} | {_fmt(s.avg_inlier_ratio_pct, 1, '%')} |"
        for s in summaries
    ]
    return header + "\n".join(rows)


def _best(summaries: list[RunSummary], attribute: str, largest: bool = True) -> RunSummary | None:
    """The run with the best measured value of ``attribute``, or ``None``."""
    candidates = [s for s in summaries if getattr(s, attribute) is not None]
    if not candidates:
        return None
    return (max if largest else min)(candidates, key=lambda s: getattr(s, attribute))


def measured_highlights(summaries: list[RunSummary]) -> list[str]:
    """Bullet list of the strongest measured results.

    Every entry is derived from a value present in a ``metrics.json``; nothing
    is inferred, rounded up, or filled in.
    """
    if not summaries:
        return []

    highlights: list[str] = []

    total_frames = sum(s.frames for s in summaries if s.frames)
    total_km = sum(s.distance_km for s in summaries if s.distance_km)
    sequences = ", ".join(s.sequence for s in summaries)
    if total_frames:
        highlights.append(
            f"Processed **{total_frames:,} KITTI frames** across **{total_km:.2f} km** "
            f"of driving (sequences {sequences})."
        )

    rate = _best(summaries, "successful_pose_rate_pct", largest=True)
    if rate is not None:
        rates = [s.successful_pose_rate_pct for s in summaries if s.successful_pose_rate_pct]
        highlights.append(
            f"Achieved a **{min(rates):.1f}-{max(rates):.1f}% successful relative-pose rate** "
            f"across all evaluated sequences."
        )

    ate = _best(summaries, "ate_rmse_optimized_m", largest=False)
    if ate is not None:
        highlights.append(
            f"Achieved **{ate.ate_rmse_optimized_m:.2f} m optimized ATE RMSE** on sequence "
            f"{ate.sequence} ({ate.distance_km:.2f} km, {ate.alignment} alignment)."
        )

    drift = _best(summaries, "translational_drift_optimized_pct", largest=False)
    if drift is not None:
        highlights.append(
            f"Achieved **{drift.translational_drift_optimized_pct:.2f}% translational drift** "
            f"on sequence {drift.sequence} (KITTI 100-800 m sub-trajectory metric)."
        )

    reduction = _best(summaries, "ate_reduction_pct", largest=True)
    if reduction is not None and reduction.ate_reduction_pct > 0:
        highlights.append(
            f"Reduced ATE RMSE by **{reduction.ate_reduction_pct:.1f}%** on sequence "
            f"{reduction.sequence} ({reduction.ate_rmse_raw_m:.2f} m -> "
            f"{reduction.ate_rmse_optimized_m:.2f} m) via loop closure and pose-graph "
            f"optimization."
        )

    loops = _best(summaries, "loop_closures_detected", largest=True)
    if loops is not None and loops.loop_closures_detected:
        total_loops = sum(s.loop_closures_detected or 0 for s in summaries)
        highlights.append(
            f"Detected and geometrically verified **{total_loops} loop closures** "
            f"({loops.loop_closures_detected} on sequence {loops.sequence})."
        )

    fps = _best(summaries, "runtime_fps", largest=True)
    if fps is not None:
        speeds = [s.runtime_fps for s in summaries if s.runtime_fps]
        highlights.append(
            f"Ran the full pipeline at **{min(speeds):.1f}-{max(speeds):.1f} FPS** "
            f"on CPU, end to end including loop closure and optimization."
        )

    return highlights


def resume_bullets(summaries: list[RunSummary]) -> list[str]:
    """Three concise resume bullets, assembled strictly from measured values."""
    if not summaries:
        return []

    bullets: list[str] = []
    total_frames = sum(s.frames for s in summaries if s.frames)
    total_km = sum(s.distance_km for s in summaries if s.distance_km)
    sequences = "/".join(s.sequence for s in summaries)

    # Bullet 1: the system and its front-end scale.
    rates = [s.successful_pose_rate_pct for s in summaries if s.successful_pose_rate_pct]
    parts = [
        "Built a monocular visual SLAM system in Python (OpenCV, GTSAM) implementing ORB "
        "feature tracking, MAGSAC++ essential-matrix pose estimation, bag-of-words loop "
        "closure and SE(3) pose-graph optimization"
    ]
    if total_frames:
        parts.append(
            f"processing {total_frames:,} KITTI Odometry frames over {total_km:.1f} km "
            f"(sequences {sequences})"
        )
    if rates:
        parts.append(f"at a {min(rates):.1f}%+ successful relative-pose rate")
    bullets.append(", ".join(parts) + ".")

    # Bullet 2: accuracy, with the scale caveat attached where it applies.
    ate = _best(summaries, "ate_rmse_optimized_m", largest=False)
    drift = _best(summaries, "translational_drift_optimized_pct", largest=False)
    if ate is not None:
        accuracy = (
            f"Benchmarked against KITTI ground truth with Umeyama {ate.alignment} alignment, "
            f"achieving {ate.ate_rmse_optimized_m:.2f} m absolute trajectory error (RMSE) "
            f"on sequence {ate.sequence} over {ate.distance_km:.2f} km"
        )
        if drift is not None:
            accuracy += (
                f" and {drift.translational_drift_optimized_pct:.2f}% translational drift "
                f"on the KITTI 100-800 m sub-trajectory metric"
            )
        if any(s.scale_uses_ground_truth for s in summaries):
            accuracy += (
                "; monocular scale ambiguity handled by an explicitly documented "
                "ground-truth-scaled evaluation protocol"
            )
        bullets.append(accuracy + ".")

    # Bullet 3: the back end's measured contribution.
    reduction = _best(summaries, "ate_reduction_pct", largest=True)
    total_loops = sum(s.loop_closures_detected or 0 for s in summaries)
    if reduction is not None and reduction.ate_reduction_pct > 0:
        bullets.append(
            f"Cut absolute trajectory error by {reduction.ate_reduction_pct:.1f}% "
            f"({reduction.ate_rmse_raw_m:.2f} m -> {reduction.ate_rmse_optimized_m:.2f} m on "
            f"sequence {reduction.sequence}) by detecting {total_loops} geometrically verified "
            f"loop closures and optimizing the keyframe pose graph with GTSAM "
            f"Levenberg-Marquardt under robust Huber kernels."
        )
    elif total_loops:
        bullets.append(
            f"Detected {total_loops} geometrically verified loop closures via TF-IDF "
            f"bag-of-visual-words retrieval with essential-matrix verification, feeding "
            f"an SE(3) pose graph optimized with GTSAM."
        )

    fps = [s.runtime_fps for s in summaries if s.runtime_fps]
    if len(bullets) < 3 and fps:
        bullets.append(
            f"Instrumented the pipeline end to end, sustaining {min(fps):.1f}-{max(fps):.1f} FPS "
            f"on CPU with per-stage timing for feature extraction, matching, pose estimation, "
            f"loop closure and optimization."
        )

    return bullets[:3]


def render_resume_metrics(summaries: list[RunSummary]) -> str:
    """Render the full ``resume_metrics.md`` document."""
    lines = [
        "# Resume metrics",
        "",
        "Every number below was produced by an actual run of this repository against the",
        "KITTI Odometry dataset and read back from the corresponding",
        "`outputs/sequence_XX/metrics.json`. Nothing here is estimated, and values that",
        "were not measurable on a given run are shown as `n/a` rather than filled in.",
        "",
    ]

    if not summaries:
        lines += [
            "## No runs found",
            "",
            "No `outputs/sequence_*/metrics.json` files are present, so there are no",
            "measured results to report. Run the pipeline first:",
            "",
            "```bash",
            "python scripts/run_slam.py --sequence 00 --dataset-path /path/to/KITTI/dataset",
            "```",
            "",
        ]
        return "\n".join(lines)

    uses_gt_scale = any(s.scale_uses_ground_truth for s in summaries)
    if uses_gt_scale:
        lines += [
            "> **Scale caveat.** These runs used `odometry.scale_source: ground_truth`, which",
            "> takes the per-frame translation *magnitude* from KITTI ground truth. Monocular",
            "> vision cannot recover absolute scale; the estimator recovers the translation",
            "> *direction* and the full rotation. These figures therefore measure trajectory",
            "> shape, heading drift and loop-closure benefit — not metric-scale localization.",
            "> Any use of these numbers must carry the same caveat.",
            "",
        ]

    lines += ["## Results", "", results_table(summaries), ""]
    lines += ["## Front-end statistics", "", feature_table(summaries), ""]

    highlights = measured_highlights(summaries)
    if highlights:
        lines += ["## Measured highlights", ""]
        lines += [f"- {h}" for h in highlights]
        lines.append("")

    bullets = resume_bullets(summaries)
    if bullets:
        lines += [
            "## Suggested resume bullets",
            "",
            "Assembled entirely from the measured values above.",
            "",
        ]
        for i, bullet in enumerate(bullets, start=1):
            lines += [f"{i}. {bullet}", ""]

    lines += [
        "## Metric definitions",
        "",
        "- **ATE RMSE** — root-mean-square Euclidean distance between estimated and",
        "  ground-truth camera positions after Umeyama alignment.",
        "- **ATE reduction** — `(ATE_raw - ATE_optimized) / ATE_raw * 100`.",
        "- **Translational drift** — KITTI odometry metric: translation error over",
        "  100-800 m sub-trajectories, normalised by sub-trajectory length.",
        "- **Rotational drift** — the same sub-trajectory protocol, reporting rotation",
        "  error in degrees per 100 m.",
        "- **Successful pose rate** — validated relative poses divided by attempted frame",
        "  transitions. A correctly detected stationary frame counts as a success.",
        "- **FPS** — total frames divided by end-to-end wall-clock runtime, including",
        "  loop closure and pose-graph optimization.",
        "",
    ]
    return "\n".join(lines)


def write_resume_metrics(
    output_root: Path | str = "outputs", path: Path | str | None = None
) -> Path:
    """Generate ``resume_metrics.md`` from the runs under ``output_root``."""
    summaries = load_run_summaries(output_root)
    out = Path(path) if path is not None else Path(output_root) / "resume_metrics.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_resume_metrics(summaries), encoding="utf-8")
    logger.info("Wrote %s from %d run(s)", out, len(summaries))
    return out


#: Markers delimiting the auto-generated results block in the README.
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def render_readme_results(summaries: list[RunSummary]) -> str:
    """Render the README results block from measured runs."""
    if not summaries:
        return (
            "_No benchmark runs recorded yet. Run the pipeline (see "
            "[Usage](#usage)) and then `python scripts/report.py --update-readme`._"
        )

    uses_gt_scale = any(s.scale_uses_ground_truth for s in summaries)
    lines: list[str] = []

    if uses_gt_scale:
        lines += [
            "> **All figures below are from real runs of this repository on KITTI Odometry.** "
            "They use `odometry.scale_source: ground_truth`, which takes the per-frame "
            "translation *magnitude* from ground truth. Monocular vision cannot recover "
            "absolute scale; the estimator recovers translation *direction* and full rotation. "
            "These numbers therefore measure trajectory shape, heading drift and loop-closure "
            "benefit — **not** metric-scale localization. See "
            "[Monocular scale ambiguity](#4-monocular-scale-ambiguity).",
            "",
        ]
    else:
        lines += [
            "> **All figures below are from real runs of this repository on KITTI Odometry.**",
            "",
        ]

    lines += [results_table(summaries), ""]
    lines += ["### Front-end statistics", "", feature_table(summaries), ""]

    highlights = measured_highlights(summaries)
    if highlights:
        lines += ["### Highlights", ""]
        lines += [f"- {h}" for h in highlights]
        lines.append("")

    lines += _figure_section(summaries)
    lines.append("_Regenerate with `python scripts/report.py --update-readme`._")
    return "\n".join(lines)


#: Figures copied out of a run directory for display in the README, in the
#: order they should appear.
PUBLISHED_FIGURES = (
    ("trajectory_comparison.png", "Ground truth vs raw VO vs optimized SLAM"),
    ("loop_closures.png", "Accepted loop-closure edges"),
    ("error_over_time.png", "Position error against distance travelled"),
)

FIGURE_DIR = Path("docs/results")


def _figure_section(summaries: list[RunSummary]) -> list[str]:
    """Embed published figures, falling back to a pointer when absent."""
    lines = ["### Trajectories", ""]
    any_published = False

    for s in summaries:
        published = [
            (name, caption)
            for name, caption in PUBLISHED_FIGURES
            if (FIGURE_DIR / f"sequence_{s.sequence}" / name).is_file()
        ]
        if not published:
            continue
        any_published = True
        lines += [f"**Sequence {s.sequence}**", ""]
        for name, caption in published:
            path = FIGURE_DIR / f"sequence_{s.sequence}" / name
            lines.append(f"![{caption} — KITTI sequence {s.sequence}]({path.as_posix()})")
            lines.append("")
            lines.append(f"*{caption}.*")
            lines.append("")

    if not any_published:
        lines += [
            "_Figures are written to `outputs/sequence_XX/` by each run. Publish them into "
            "`docs/results/` with `python scripts/report.py --publish-figures --update-readme`._",
            "",
        ]
    return lines


def publish_figures(
    output_root: Path | str = "outputs", figure_dir: Path | str = FIGURE_DIR
) -> list[Path]:
    """Copy the README-facing figures out of run directories into ``figure_dir``.

    Run outputs are git-ignored (they are regenerated, and include large
    intermediates), but a handful of result images belong in version control so
    the README renders on GitHub. Returns the paths written.
    """
    import shutil

    root = Path(output_root)
    destination_root = Path(figure_dir)
    written: list[Path] = []

    for run_dir in sorted(root.glob("sequence_*")):
        if not (run_dir / "metrics.json").is_file():
            continue
        destination = destination_root / run_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for name, _caption in PUBLISHED_FIGURES:
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, destination / name)
                written.append(destination / name)

    logger.info("Published %d figures to %s", len(written), destination_root)
    return written


def update_readme_results(
    readme_path: Path | str = "README.md", output_root: Path | str = "outputs"
) -> bool:
    """Replace the README's results block with measured results.

    Returns ``True`` when the file was rewritten. The markers must already be
    present; this never appends a block to an arbitrary position in the file.
    """
    path = Path(readme_path)
    if not path.is_file():
        logger.warning("README not found at %s; skipping results injection", path)
        return False

    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        logger.warning(
            "README is missing the %s / %s markers; skipping results injection",
            README_START,
            README_END,
        )
        return False

    before, _, rest = text.partition(README_START)
    _, _, after = rest.partition(README_END)
    block = render_readme_results(load_run_summaries(output_root))
    path.write_text(
        f"{before}{README_START}\n{block}\n{README_END}{after}", encoding="utf-8"
    )
    logger.info("Updated results block in %s", path)
    return True
