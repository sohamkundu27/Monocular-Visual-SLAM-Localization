"""Typed configuration system.

Configuration is expressed as nested dataclasses so every tunable has a
documented default and a static type. YAML files supply overrides, and the CLI
can layer further overrides on top using dotted keys
(``--set odometry.min_inliers=40``).

Unknown keys are rejected rather than silently ignored: a typo in a YAML file
that quietly reverts a threshold to its default is the kind of bug that is very
hard to spot in an experiment log.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file or override is malformed."""


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class DatasetConfig:
    """KITTI Odometry dataset location and frame selection."""

    #: Root of the KITTI odometry download (the directory holding ``sequences/``).
    path: str = "data/kitti/dataset"
    #: Two-character sequence id, e.g. ``"00"``.
    sequence: str = "00"
    #: ``image_0`` is the left grayscale camera used for monocular SLAM.
    camera: str = "image_0"
    #: Process at most this many frames (``0`` or ``None`` means the whole sequence).
    max_frames: int | None = None
    #: Take every Nth frame; ``1`` uses every frame.
    frame_step: int = 1
    #: Skip this many frames at the start of the sequence.
    start_frame: int = 0
    #: Load images as grayscale (monocular front end never needs colour).
    grayscale: bool = True


@dataclass
class FeatureConfig:
    """Keypoint detector and descriptor settings."""

    #: ``"orb"`` (binary, default) or ``"sift"`` (float, requires OpenCV SIFT).
    detector: str = "orb"
    #: Upper bound on retained keypoints per frame.
    max_features: int = 3000
    #: ORB FAST corner threshold; lower detects more (weaker) corners.
    fast_threshold: int = 20
    #: ORB pyramid scale factor between levels.
    scale_factor: float = 1.2
    #: Number of pyramid levels.
    n_levels: int = 8
    #: ORB patch/edge border size in pixels.
    edge_threshold: int = 31
    #: ``0`` selects the HARRIS_SCORE ranking, ``1`` selects FAST_SCORE.
    score_type: int = 0
    #: Apply CLAHE contrast equalisation before detection.
    clahe: bool = False


@dataclass
class MatcherConfig:
    """Descriptor matching and correspondence filtering."""

    #: ``"bf"`` brute force (exact) — the only backend needed at KITTI scale.
    matcher: str = "bf"
    #: Use k-NN matching plus Lowe's ratio test (recommended).
    use_ratio_test: bool = True
    #: Lowe ratio threshold; lower is stricter.
    ratio: float = 0.75
    #: Require mutual best matches in both directions.
    cross_check: bool = True
    #: Drop matches whose descriptor distance exceeds this (``None`` disables).
    #: Hamming units for ORB, L2 units for SIFT.
    max_distance: float | None = 64.0
    #: Frame pairs with fewer surviving matches are treated as failures.
    min_matches: int = 20


@dataclass
class OdometryConfig:
    """Two-view relative pose estimation."""

    #: RANSAC inlier threshold in pixels for the essential matrix.
    ransac_threshold_px: float = 1.0
    #: RANSAC target confidence.
    ransac_confidence: float = 0.999
    #: Maximum RANSAC iterations.
    ransac_max_iters: int = 2000
    #: Reject a relative pose supported by fewer than this many inliers.
    min_inliers: int = 30
    #: Reject a relative pose whose inlier fraction is below this.
    min_inlier_ratio: float = 0.3
    #: Reject rotations larger than this between consecutive frames (degrees).
    max_rotation_deg: float = 30.0
    #: Reject transitions whose rotation-compensated parallax is below this
    #: (degrees). ``0`` disables. Note that parallax cannot detect pure
    #: rotation on its own -- see ``homography_ratio_threshold``.
    min_parallax_deg: float = 0.0
    #: Treat a transition with less median optical flow than this as a
    #: stationary camera and hold the previous pose instead of integrating
    #: noise. KITTI vehicles stop at traffic lights in several sequences.
    min_flow_px: float = 0.7
    #: Homography-vs-essential model selection threshold. Above this, a
    #: homography explains the matches as well as the epipolar model, meaning
    #: the scene is planar or the motion is rotation-only.
    homography_ratio_threshold: float = 0.45
    #: Reject transitions flagged degenerate by model selection. Off by
    #: default: on KITTI the road plane can dominate legitimately, and the
    #: flag is more useful as a diagnostic than as a hard gate.
    reject_degenerate: bool = False
    #: Scale strategy: ``"ground_truth"``, ``"constant"`` or ``"none"``.
    #: See :mod:`monocular_slam.odometry.scale` and the README section on
    #: monocular scale ambiguity.
    scale_source: str = "ground_truth"
    #: Metres per frame when ``scale_source == "constant"``.
    constant_scale: float = 0.85
    #: Discard scales outside this range (metres); guards against GT gaps.
    min_scale: float = 1e-3
    max_scale: float = 10.0


@dataclass
class KeyframeConfig:
    """Keyframe selection for the loop-closure database and pose graph."""

    #: Insert a keyframe at least every N frames.
    every_n_frames: int = 5
    #: ...or sooner if the camera has translated this far since the last one (m).
    min_translation_m: float = 2.0
    #: ...or rotated this much since the last one (degrees).
    min_rotation_deg: float = 10.0


@dataclass
class LoopClosureConfig:
    """Appearance-based loop detection and geometric verification."""

    enabled: bool = True
    #: Size of the visual vocabulary clustered from ORB descriptors.
    vocabulary_size: int = 256
    #: Descriptors sampled to train the vocabulary.
    vocabulary_train_descriptors: int = 60000
    #: Candidate keyframes must be at least this many keyframes in the past.
    min_keyframe_separation: int = 30
    #: ...and at least this many metres of travelled path away, which prevents
    #: a slow/stopped vehicle from matching against itself.
    min_path_separation_m: float = 30.0
    #: Number of top-scoring candidates retained per query.
    top_k: int = 5
    #: Minimum BoW cosine similarity for a candidate to be verified.
    min_similarity: float = 0.20
    #: Reject candidates scoring below this fraction of the best neighbouring
    #: (temporally adjacent) score — the standard DBoW normalisation.
    similarity_ratio: float = 0.85
    #: Geometric verification thresholds.
    min_matches: int = 60
    min_inliers: int = 40
    min_inlier_ratio: float = 0.35
    #: Suppress further detections for this many keyframes after acceptance.
    cooldown_keyframes: int = 10
    #: Cap on accepted loops (0 = unlimited); a safety valve for pathological runs.
    max_loops: int = 0


@dataclass
class PoseGraphConfig:
    """SE(3) pose-graph construction and optimization."""

    #: Prior on the first pose (rotation rad, translation m). Small = strong anchor.
    prior_sigma_rot: float = 1e-4
    prior_sigma_trans: float = 1e-4
    #: Odometry edge noise (rotation rad, translation m).
    odom_sigma_rot: float = 0.02
    odom_sigma_trans: float = 0.10
    #: Loop closure edge noise (rotation rad, translation m). Looser than
    #: odometry because loop constraints come from wide-baseline matches.
    loop_sigma_rot: float = 0.05
    loop_sigma_trans: float = 0.30
    #: Wrap loop factors in a robust kernel so a single false positive cannot
    #: destroy the solution.
    robust_loop_kernel: bool = True
    #: Huber threshold used by that kernel.
    huber_k: float = 1.345
    #: ``"levenberg_marquardt"`` or ``"gauss_newton"``.
    optimizer: str = "levenberg_marquardt"
    max_iterations: int = 100
    relative_error_tol: float = 1e-5
    absolute_error_tol: float = 1e-5
    #: Print GTSAM's per-iteration error.
    verbose: bool = False


@dataclass
class EvaluationConfig:
    """Trajectory error metrics."""

    #: Umeyama alignment mode: ``"sim3"`` (similarity, monocular-appropriate),
    #: ``"se3"`` (rigid) or ``"none"``.
    alignment: str = "sim3"
    #: Frame gaps (in frames) used for relative pose error.
    rpe_deltas: tuple[int, ...] = (1, 10, 100)
    #: Sub-trajectory lengths in metres for the KITTI-style drift metric.
    drift_segment_lengths_m: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0)


@dataclass
class OutputConfig:
    """Where artefacts land and how they are rendered."""

    root: str = "outputs"
    #: Per-run directory name; ``{sequence}`` is substituted.
    run_name: str = "sequence_{sequence}"
    save_plots: bool = True
    save_diagnostics: bool = True
    dpi: int = 150
    plot_format: str = "png"
    #: Also render a 3D trajectory plot (slower, mostly for presentation).
    plot_3d: bool = True


@dataclass
class RuntimeConfig:
    """Process-level behaviour."""

    seed: int = 0
    log_level: str = "INFO"
    #: Emit a progress line every N frames.
    progress_interval: int = 250


@dataclass
class Config:
    """Root configuration object."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    odometry: OdometryConfig = field(default_factory=OdometryConfig)
    keyframes: KeyframeConfig = field(default_factory=KeyframeConfig)
    loop_closure: LoopClosureConfig = field(default_factory=LoopClosureConfig)
    pose_graph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        """Build a config from a nested mapping, validating every key."""
        return _build_dataclass(cls, data or {}, path="")

    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        """Load a config from a YAML file."""
        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise ConfigError(f"Config file not found: {yaml_path}")
        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is not None and not isinstance(data, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> Config:
        """Load ``path`` (or defaults) and apply dotted-key ``overrides``."""
        config = cls.from_yaml(path) if path is not None else cls()
        if overrides:
            config = config.with_overrides(overrides)
        return config

    # ----------------------------------------------------------------- #
    # Mutation / serialisation
    # ----------------------------------------------------------------- #

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a copy with dotted-key overrides applied.

        ``{"odometry.min_inliers": 40}`` sets ``config.odometry.min_inliers``.
        """
        data = self.to_dict()
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            parts = dotted_key.split(".")
            node = data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise ConfigError(f"Unknown config section in override: '{dotted_key}'")
                node = node[part]
            if parts[-1] not in node:
                raise ConfigError(f"Unknown config key in override: '{dotted_key}'")
            node[parts[-1]] = value
        return Config.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain Python types (JSON/YAML friendly)."""
        return _to_plain(dataclasses.asdict(self))

    def to_yaml(self, path: Path | str) -> Path:
        """Write the resolved config to ``path`` for run reproducibility."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)
        return out

    # ----------------------------------------------------------------- #
    # Derived paths
    # ----------------------------------------------------------------- #

    @property
    def run_dir(self) -> Path:
        """Output directory for this configuration's sequence."""
        name = self.output.run_name.format(sequence=self.dataset.sequence)
        return Path(self.output.root) / name


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _build_dataclass(cls: type, data: dict[str, Any], path: str) -> Any:
    """Recursively instantiate nested dataclasses from a mapping."""
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at '{path or 'root'}', got {type(data).__name__}")

    # ``from __future__ import annotations`` stores field types as strings, so
    # resolve them once against this module's namespace.
    hints = get_type_hints(cls)
    field_names = [f.name for f in fields(cls)]
    unknown = set(data) - set(field_names)
    if unknown:
        location = path or "root"
        raise ConfigError(
            f"Unknown config key(s) at '{location}': {sorted(unknown)}. "
            f"Valid keys: {sorted(field_names)}"
        )

    kwargs: dict[str, Any] = {}
    for name in field_names:
        if name not in data:
            continue
        value = data[name]
        annotation = hints[name]
        child_path = f"{path}.{name}" if path else name
        if is_dataclass(annotation):
            kwargs[name] = _build_dataclass(annotation, value, child_path)
        else:
            kwargs[name] = _coerce(value, annotation, child_path)
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Light type coercion so YAML and CLI strings land in the right type."""
    origin = get_origin(annotation)

    # Optional[T] / T | None
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            if type(None) in get_args(annotation):
                return None
            raise ConfigError(f"Config key '{path}' does not accept null")
        return _coerce(value, args[0], path)

    if value is None:
        raise ConfigError(f"Config key '{path}' does not accept null")

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"Config key '{path}' expects a list, got {type(value).__name__}")
        inner = get_args(annotation)[0] if get_args(annotation) else float
        return tuple(_coerce(v, inner, path) for v in value)

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "1", "on"}:
                return True
            if lowered in {"false", "no", "0", "off"}:
                return False
        raise ConfigError(f"Config key '{path}' expects a boolean, got {value!r}")

    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"Config key '{path}' expects an int, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Config key '{path}' expects an int, got {value!r}") from exc

    if annotation is float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Config key '{path}' expects a float, got {value!r}") from exc

    if annotation is str:
        return str(value)

    return value


def _to_plain(value: Any) -> Any:
    """Convert tuples to lists and Paths to strings for YAML/JSON output."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
