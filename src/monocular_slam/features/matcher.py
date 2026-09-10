"""Descriptor matching and correspondence filtering.

The filtering cascade is deliberately explicit, because match quality is what
determines whether the essential-matrix estimate is usable:

1. **k-NN search** (k=2) with a brute-force matcher. Exact rather than
   approximate: at 3000 descriptors per frame the exhaustive search is not the
   bottleneck, and approximate search adds a second source of error.
2. **Lowe's ratio test** — keep a match only when the best candidate is
   clearly better than the runner-up (``d1 < ratio * d2``). This is the single
   most effective filter against repetitive structure such as road markings,
   railings and building facades, all of which are everywhere in KITTI.
3. **Mutual consistency (cross-check)** — keep a match only when the two
   descriptors are each other's nearest neighbour. Ratio-test and cross-check
   are combined manually rather than via ``BFMatcher(crossCheck=True)``,
   because OpenCV forbids ``crossCheck`` with ``knnMatch``.
4. **Absolute distance ceiling** — reject matches that survive the relative
   tests but are simply poor in absolute terms.

Everything downstream consumes the resulting :class:`MatchResult`, which
carries the aligned point arrays rather than raw ``DMatch`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from monocular_slam.features.detector import Frame
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Filtered correspondences between two frames.

    Attributes
    ----------
    points_a, points_b:
        ``(M, 2)`` pixel coordinates of matched keypoints, index-aligned.
    indices_a, indices_b:
        ``(M,)`` keypoint indices into the source frames, so a caller can trace
        a correspondence back to its descriptor.
    distances:
        ``(M,)`` descriptor distances of the accepted matches.
    n_raw:
        Matches before filtering, for diagnostics.
    """

    points_a: np.ndarray
    points_b: np.ndarray
    indices_a: np.ndarray
    indices_b: np.ndarray
    distances: np.ndarray
    n_raw: int = 0

    def __len__(self) -> int:
        return int(self.points_a.shape[0])

    @property
    def survival_ratio(self) -> float:
        """Fraction of raw matches that survived filtering."""
        return float(len(self) / self.n_raw) if self.n_raw else 0.0

    @property
    def mean_distance(self) -> float:
        return float(self.distances.mean()) if len(self) else float("nan")

    @property
    def mean_pixel_displacement(self) -> float:
        """Average optical-flow magnitude, a cheap sanity check on motion."""
        if len(self) == 0:
            return float("nan")
        return float(np.linalg.norm(self.points_b - self.points_a, axis=1).mean())

    def select(self, mask: np.ndarray) -> MatchResult:
        """Return the subset selected by a boolean or index mask."""
        mask = np.asarray(mask)
        if mask.dtype == bool and len(mask) != len(self):
            raise ValueError(f"mask length {len(mask)} != match count {len(self)}")
        return MatchResult(
            points_a=self.points_a[mask],
            points_b=self.points_b[mask],
            indices_a=self.indices_a[mask],
            indices_b=self.indices_b[mask],
            distances=self.distances[mask],
            n_raw=self.n_raw,
        )

    @classmethod
    def empty(cls, n_raw: int = 0) -> MatchResult:
        return cls(
            points_a=np.zeros((0, 2), dtype=np.float64),
            points_b=np.zeros((0, 2), dtype=np.float64),
            indices_a=np.zeros(0, dtype=np.int64),
            indices_b=np.zeros(0, dtype=np.int64),
            distances=np.zeros(0, dtype=np.float64),
            n_raw=n_raw,
        )

    def __repr__(self) -> str:
        return f"MatchResult(matches={len(self)}, raw={self.n_raw}, mean_dist={self.mean_distance:.1f})"


class FeatureMatcher:
    """Brute-force descriptor matcher with a configurable filter cascade."""

    def __init__(
        self,
        norm: int = cv2.NORM_HAMMING,
        use_ratio_test: bool = True,
        ratio: float = 0.75,
        cross_check: bool = True,
        max_distance: float | None = 64.0,
        min_matches: int = 20,
    ) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        self.norm = norm
        self.use_ratio_test = bool(use_ratio_test)
        self.ratio = float(ratio)
        self.cross_check = bool(cross_check)
        self.max_distance = None if max_distance is None else float(max_distance)
        self.min_matches = int(min_matches)
        # crossCheck stays False here: it is incompatible with knnMatch, so the
        # mutual-consistency test is applied explicitly in _cross_check_mask.
        self._matcher = cv2.BFMatcher_create(norm, crossCheck=False)
        logger.info(
            "Feature matcher: norm=%s, ratio_test=%s (%.2f), cross_check=%s, max_distance=%s",
            "HAMMING" if norm == cv2.NORM_HAMMING else "L2",
            self.use_ratio_test,
            self.ratio,
            self.cross_check,
            self.max_distance,
        )

    @classmethod
    def from_config(cls, config, norm: int = cv2.NORM_HAMMING) -> FeatureMatcher:  # noqa: ANN001
        m = config.matcher
        if m.matcher.lower() != "bf":
            raise ValueError(f"Unsupported matcher backend '{m.matcher}'; only 'bf' is available")
        return cls(
            norm=norm,
            use_ratio_test=m.use_ratio_test,
            ratio=m.ratio,
            cross_check=m.cross_check,
            max_distance=m.max_distance,
            min_matches=m.min_matches,
        )

    # ----------------------------------------------------------------- #
    # Matching
    # ----------------------------------------------------------------- #

    def match(self, frame_a: Frame, frame_b: Frame) -> MatchResult:
        """Match ``frame_a`` against ``frame_b`` and apply the filter cascade."""
        if not frame_a.is_usable or not frame_b.is_usable:
            return MatchResult.empty()
        return self.match_descriptors(
            frame_a.descriptors,
            frame_b.descriptors,
            frame_a.points,
            frame_b.points,
        )

    def match_descriptors(
        self,
        desc_a: np.ndarray,
        desc_b: np.ndarray,
        points_a: np.ndarray,
        points_b: np.ndarray,
    ) -> MatchResult:
        """Core matching routine working directly on descriptor arrays."""
        desc_a = _as_matcher_dtype(desc_a, self.norm)
        desc_b = _as_matcher_dtype(desc_b, self.norm)
        if len(desc_a) == 0 or len(desc_b) == 0:
            return MatchResult.empty()

        idx_a, idx_b, distances, n_raw = self._forward_matches(desc_a, desc_b)

        if len(idx_a) and self.cross_check:
            keep = self._cross_check_mask(desc_a, desc_b, idx_a, idx_b)
            idx_a, idx_b, distances = idx_a[keep], idx_b[keep], distances[keep]

        if len(idx_a) and self.max_distance is not None:
            keep = distances <= self.max_distance
            idx_a, idx_b, distances = idx_a[keep], idx_b[keep], distances[keep]

        return MatchResult(
            points_a=np.asarray(points_a, dtype=np.float64)[idx_a],
            points_b=np.asarray(points_b, dtype=np.float64)[idx_b],
            indices_a=idx_a,
            indices_b=idx_b,
            distances=distances,
            n_raw=n_raw,
        )

    def _forward_matches(
        self, desc_a: np.ndarray, desc_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Nearest-neighbour search from ``a`` to ``b``, with the ratio test."""
        if self.use_ratio_test and len(desc_b) >= 2:
            knn = self._matcher.knnMatch(desc_a, desc_b, k=2)
            n_raw = len(knn)
            idx_a, idx_b, dist = [], [], []
            for pair in knn:
                if len(pair) < 2:
                    # Only one neighbour exists, so the ratio test cannot be
                    # evaluated; accept it and let later filters decide.
                    if pair:
                        idx_a.append(pair[0].queryIdx)
                        idx_b.append(pair[0].trainIdx)
                        dist.append(pair[0].distance)
                    continue
                best, second = pair[0], pair[1]
                if second.distance > 0 and best.distance < self.ratio * second.distance:
                    idx_a.append(best.queryIdx)
                    idx_b.append(best.trainIdx)
                    dist.append(best.distance)
            return (
                np.asarray(idx_a, dtype=np.int64),
                np.asarray(idx_b, dtype=np.int64),
                np.asarray(dist, dtype=np.float64),
                n_raw,
            )

        matches = self._matcher.match(desc_a, desc_b)
        n_raw = len(matches)
        return (
            np.array([m.queryIdx for m in matches], dtype=np.int64),
            np.array([m.trainIdx for m in matches], dtype=np.int64),
            np.array([m.distance for m in matches], dtype=np.float64),
            n_raw,
        )

    def _cross_check_mask(
        self, desc_a: np.ndarray, desc_b: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray
    ) -> np.ndarray:
        """Keep matches that are mutual nearest neighbours.

        The reverse search runs once over all of ``b`` and is then consulted as
        a lookup table, which is far cheaper than re-querying per match.
        """
        reverse = self._matcher.match(desc_b, desc_a)
        best_from_b = np.full(len(desc_b), -1, dtype=np.int64)
        for m in reverse:
            best_from_b[m.queryIdx] = m.trainIdx
        return best_from_b[idx_b] == idx_a

    def is_sufficient(self, result: MatchResult) -> bool:
        """True when enough matches survived to attempt pose estimation."""
        return len(result) >= self.min_matches


def _as_matcher_dtype(descriptors: np.ndarray, norm: int) -> np.ndarray:
    """Coerce descriptors to the dtype OpenCV's BFMatcher expects.

    Hamming matching requires ``uint8``; L2 matching requires ``float32``.
    Passing the wrong dtype raises a confusing OpenCV error deep in the C++
    layer, so it is normalised here instead.
    """
    descriptors = np.asarray(descriptors)
    if norm == cv2.NORM_HAMMING:
        return descriptors if descriptors.dtype == np.uint8 else descriptors.astype(np.uint8)
    return descriptors if descriptors.dtype == np.float32 else descriptors.astype(np.float32)


def draw_matches(
    image_a: np.ndarray,
    frame_a: Frame,
    image_b: np.ndarray,
    frame_b: Frame,
    result: MatchResult,
    max_draw: int = 150,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render accepted correspondences side by side.

    When ``inlier_mask`` is supplied (typically the RANSAC mask from essential
    matrix estimation), inliers are drawn green and outliers red — the fastest
    way to see whether a bad pose came from bad matching or bad geometry.
    """
    canvas_a = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR) if image_a.ndim == 2 else image_a.copy()
    canvas_b = cv2.cvtColor(image_b, cv2.COLOR_GRAY2BGR) if image_b.ndim == 2 else image_b.copy()

    height = max(canvas_a.shape[0], canvas_b.shape[0])
    width = canvas_a.shape[1] + canvas_b.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: canvas_a.shape[0], : canvas_a.shape[1]] = canvas_a
    canvas[: canvas_b.shape[0], canvas_a.shape[1] :] = canvas_b
    offset = np.array([canvas_a.shape[1], 0.0])

    n = len(result)
    if n == 0:
        return canvas
    order = np.argsort(result.distances)[:max_draw]

    for i in order:
        pa = tuple(np.round(result.points_a[i]).astype(int))
        pb = tuple(np.round(result.points_b[i] + offset).astype(int))
        if inlier_mask is None:
            color = (0, 220, 0)
        else:
            color = (0, 220, 0) if bool(inlier_mask[i]) else (0, 0, 220)
        cv2.line(canvas, pa, pb, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pa, 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pb, 3, color, 1, cv2.LINE_AA)

    label = f"{n} matches"
    if inlier_mask is not None:
        label += f" | {int(np.count_nonzero(inlier_mask))} inliers"
    cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
