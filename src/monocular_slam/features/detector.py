"""Keypoint detection and description.

ORB is the default: it is fast enough to run the full 4541-frame KITTI
sequence 00 in a few minutes on a laptop CPU, and its 256-bit binary
descriptors make the loop-closure database cheap to store and compare.

SIFT is offered as an alternative because its float descriptors are more
discriminative for wide-baseline loop closure, at roughly an order of
magnitude more compute. The rest of the pipeline adapts automatically: the
matcher picks the Hamming or L2 norm from the descriptor dtype, so nothing
downstream needs to know which detector ran.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)

#: Detectors this module knows how to build.
SUPPORTED_DETECTORS = ("orb", "sift")


@dataclass
class Frame:
    """Features extracted from a single image.

    Attributes
    ----------
    index:
        Position within the processed sequence (0-based).
    frame_id:
        Original index in the KITTI sequence (differs when frames are skipped).
    keypoints:
        OpenCV keypoints, kept because drawing utilities and sub-pixel
        refinement need the full structure.
    descriptors:
        ``(N, D)`` descriptor matrix: ``uint8`` for ORB, ``float32`` for SIFT.
        ``None`` when the detector found nothing.
    points:
        ``(N, 2)`` float64 keypoint pixel coordinates, cached because every
        geometric routine wants them in this form.
    """

    index: int
    frame_id: int
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None
    points: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.keypoints = tuple(self.keypoints)
        if self.keypoints:
            self.points = np.array([kp.pt for kp in self.keypoints], dtype=np.float64)
        else:
            self.points = np.zeros((0, 2), dtype=np.float64)
        if self.descriptors is not None and len(self.descriptors) != len(self.keypoints):
            raise ValueError(
                f"descriptor count {len(self.descriptors)} != keypoint count {len(self.keypoints)}"
            )

    def __len__(self) -> int:
        return len(self.keypoints)

    @property
    def is_usable(self) -> bool:
        """True when the frame carries descriptors that can be matched."""
        return self.descriptors is not None and len(self.keypoints) > 0

    @property
    def responses(self) -> np.ndarray:
        """``(N,)`` detector response strengths."""
        return np.array([kp.response for kp in self.keypoints], dtype=np.float64)

    def __repr__(self) -> str:
        return f"Frame(index={self.index}, frame_id={self.frame_id}, keypoints={len(self)})"


class FeatureDetector:
    """Wraps an OpenCV detector behind a stable, configurable interface."""

    def __init__(
        self,
        detector: str = "orb",
        max_features: int = 3000,
        fast_threshold: int = 20,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        score_type: int = 0,
        clahe: bool = False,
    ) -> None:
        self.name = detector.lower().strip()
        if self.name not in SUPPORTED_DETECTORS:
            raise ValueError(
                f"Unsupported detector '{detector}'. Choose from {list(SUPPORTED_DETECTORS)}"
            )
        self.max_features = int(max_features)
        if self.max_features <= 0:
            raise ValueError(f"max_features must be positive, got {max_features}")

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        self._detector = self._build(
            fast_threshold=fast_threshold,
            scale_factor=scale_factor,
            n_levels=n_levels,
            edge_threshold=edge_threshold,
            score_type=score_type,
        )
        logger.info(
            "Feature detector: %s (max_features=%d, clahe=%s)", self.name, self.max_features, clahe
        )

    def _build(
        self,
        fast_threshold: int,
        scale_factor: float,
        n_levels: int,
        edge_threshold: int,
        score_type: int,
    ):
        if self.name == "orb":
            return cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=float(scale_factor),
                nlevels=int(n_levels),
                edgeThreshold=int(edge_threshold),
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE if int(score_type) == 0 else cv2.ORB_FAST_SCORE,
                patchSize=int(edge_threshold),
                fastThreshold=int(fast_threshold),
            )
        if not hasattr(cv2, "SIFT_create"):  # pragma: no cover - depends on OpenCV build
            raise RuntimeError(
                "This OpenCV build has no SIFT. Install opencv-python (not the -headless "
                "contrib-free variant) or set features.detector: orb"
            )
        return cv2.SIFT_create(
            nfeatures=self.max_features,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6,
        )

    # ----------------------------------------------------------------- #
    # Detection
    # ----------------------------------------------------------------- #

    @classmethod
    def from_config(cls, config) -> FeatureDetector:  # noqa: ANN001 - avoid circular import
        f = config.features
        return cls(
            detector=f.detector,
            max_features=f.max_features,
            fast_threshold=f.fast_threshold,
            scale_factor=f.scale_factor,
            n_levels=f.n_levels,
            edge_threshold=f.edge_threshold,
            score_type=f.score_type,
            clahe=f.clahe,
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale and optionally equalise contrast.

        CLAHE helps on the strongly backlit frames in KITTI sequence 01, at a
        small runtime cost, so it is off by default.
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self._clahe is not None:
            image = self._clahe.apply(image)
        return image

    def detect(
        self,
        image: np.ndarray,
        index: int = 0,
        frame_id: int | None = None,
        mask: np.ndarray | None = None,
    ) -> Frame:
        """Detect keypoints and compute descriptors for one image.

        A frame with no detectable structure yields a :class:`Frame` with zero
        keypoints rather than raising, so the caller can record the failure and
        carry on instead of aborting a 4000-frame run.
        """
        prepared = self.preprocess(image)
        keypoints, descriptors = self._detector.detectAndCompute(prepared, mask)

        if keypoints is None or len(keypoints) == 0:
            logger.warning(
                "No keypoints detected in frame %s",
                frame_id if frame_id is not None else index,
            )
            return Frame(index=index, frame_id=index if frame_id is None else frame_id,
                         keypoints=(), descriptors=None)

        # OpenCV's ORB usually respects nfeatures, but SIFT's cap is applied
        # before its final filtering pass, so enforce the budget here too.
        if len(keypoints) > self.max_features:
            keypoints, descriptors = _keep_strongest(keypoints, descriptors, self.max_features)

        return Frame(
            index=index,
            frame_id=index if frame_id is None else frame_id,
            keypoints=keypoints,
            descriptors=descriptors,
        )

    @property
    def descriptor_norm(self) -> int:
        """The OpenCV distance norm that matches this detector's descriptors."""
        return cv2.NORM_HAMMING if self.name == "orb" else cv2.NORM_L2

    def __repr__(self) -> str:
        return f"FeatureDetector({self.name}, max_features={self.max_features})"


def _keep_strongest(
    keypoints: Sequence[cv2.KeyPoint], descriptors: np.ndarray | None, limit: int
) -> tuple[tuple[cv2.KeyPoint, ...], np.ndarray | None]:
    """Retain the ``limit`` highest-response keypoints, descriptors in step."""
    responses = np.array([kp.response for kp in keypoints], dtype=np.float64)
    order = np.argsort(-responses)[:limit]
    kept_kp = tuple(keypoints[i] for i in order)
    kept_desc = None if descriptors is None else descriptors[order]
    return kept_kp, kept_desc


def draw_keypoints(
    image: np.ndarray,
    frame: Frame,
    color: tuple[int, int, int] = (0, 255, 0),
    rich: bool = True,
) -> np.ndarray:
    """Render keypoints for debugging.

    ``rich=True`` draws size and orientation, which is the quickest way to spot
    a detector configured with the wrong pyramid or patch size.
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    return cv2.drawKeypoints(canvas, list(frame.keypoints), None, color=color, flags=flags)
