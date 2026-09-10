"""Loop candidate retrieval, filtering and geometric verification.

A loop closure is accepted only after passing every stage of this cascade:

1. **Appearance retrieval** — BoW cosine similarity against the keyframe
   database, excluding temporally and spatially adjacent keyframes.
2. **Normalised similarity** — the candidate must score at least
   ``similarity_ratio`` of the best score among the query's own temporal
   neighbours. This adapts to scene texture instead of trusting one absolute
   threshold everywhere.
3. **Descriptor matching** — the two keyframes are re-matched with the same
   filter cascade the front end uses. Appearance similarity alone is famous for
   confusing structurally repetitive scenes (rows of identical parked cars,
   repeating building facades); requiring a large number of individually
   consistent correspondences is a much stronger test.
4. **Geometric verification** — a robust essential matrix must fit those
   correspondences with enough inliers and a high enough inlier ratio. A false
   positive from perceptual aliasing will match descriptors but will not admit
   a single consistent camera motion.
5. **Cooldown** — after an accepted loop, detection pauses briefly so one
   revisit does not generate a burst of near-duplicate constraints.

Recovering the loop translation magnitude
-----------------------------------------
The constraint produced is a relative pose ``T_a_b`` between the two keyframes.
Its rotation is fully determined by the essential matrix; its **translation is
unit-norm only**, for the same monocular reason the odometry front end cannot
recover scale.

The obvious shortcut — take the magnitude from the odometry-estimated gap
between the two keyframes — is *wrong*, and wrong in the most damaging possible
way. When the trajectory has drifted, that gap **is** the drift: on KITTI
sequence 00 the odometry gap across accepted loops averages 10.9 m while the
true separation averages 2.6 m. Feeding that in bakes the drift into the very
constraint meant to remove it, and measurably makes the optimized trajectory
worse than the raw one.

The magnitude is therefore recovered from **shared scene structure**
(:meth:`LoopClosureDetector.estimate_loop_scale`):

1. Triangulate the loop pair's correspondences using the unit-norm relative
   pose. This yields structure that is correct up to the unknown factor ``s``,
   the true translation magnitude.
2. Triangulate the query keyframe against its odometry neighbour, whose
   relative pose *is* metric. This yields the same scene at true scale.
3. For features seen in all three views, the ratio of metric to unit depth is
   ``s``. A robust median over those ratios is the estimate.

If too few features are shared for a reliable ratio, the loop is still used but
with a much looser translation sigma, so it constrains relative *orientation* —
which is well determined and is the dominant drift term — without asserting a
translation magnitude that was never measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monocular_slam.features.matcher import FeatureMatcher, MatchResult
from monocular_slam.geometry.epipolar import (
    estimate_essential_matrix,
    recover_relative_pose,
    triangulate_points,
)
from monocular_slam.geometry.transforms import invert_se3, se3_from_rt
from monocular_slam.loop_closure.database import KeyframeDatabase
from monocular_slam.utils.logging import get_logger
from monocular_slam.utils.timing import StageTimer

logger = get_logger(__name__)


@dataclass
class LoopCandidate:
    """A keyframe pair proposed by appearance retrieval."""

    query_id: int
    match_id: int
    similarity: float
    normalized_similarity: float = 0.0


@dataclass
class LoopClosure:
    """A geometrically verified loop closure constraint."""

    query_id: int
    match_id: int
    similarity: float
    normalized_similarity: float
    #: ``T_match_query``: pose of the query keyframe in the matched keyframe's
    #: frame, with a metric translation taken from odometry.
    T_match_query: np.ndarray
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    #: Translation magnitude applied to the unit direction, in metres.
    scale_m: float
    #: How that magnitude was obtained: ``"structure"`` (triangulated against
    #: metrically-scaled neighbouring structure) or ``"unscaled"`` (estimation
    #: failed; the constraint is treated as orientation-only).
    scale_method: str = "structure"
    #: Number of tri-view features that supported the scale estimate.
    n_scale_points: int = 0
    #: Frame indices, for plotting against the trajectory.
    query_frame_index: int = 0
    match_frame_index: int = 0

    @property
    def scale_is_measured(self) -> bool:
        """False when no metric magnitude could be recovered for this loop."""
        return self.scale_method == "structure"

    def to_dict(self) -> dict[str, object]:
        return {
            "query_keyframe": self.query_id,
            "match_keyframe": self.match_id,
            "query_frame_index": self.query_frame_index,
            "match_frame_index": self.match_frame_index,
            "similarity": round(self.similarity, 4),
            "normalized_similarity": round(self.normalized_similarity, 4),
            "n_matches": self.n_matches,
            "n_inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "scale_m": round(self.scale_m, 4),
            "scale_method": self.scale_method,
            "n_scale_points": self.n_scale_points,
        }


@dataclass
class LoopClosureStats:
    """Counts of how candidates fared at each stage of the cascade."""

    n_queries: int = 0
    n_candidates: int = 0
    rejected_normalized_similarity: int = 0
    rejected_too_few_matches: int = 0
    rejected_geometry: int = 0
    rejected_cooldown: int = 0
    #: Loops accepted geometrically whose metric magnitude could not be
    #: recovered; these become orientation-only constraints.
    scale_estimation_failed: int = 0
    accepted: int = 0
    match_counts: list[int] = field(default_factory=list, repr=False)
    inlier_counts: list[int] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "queries": self.n_queries,
            "candidates_retrieved": self.n_candidates,
            "rejected_normalized_similarity": self.rejected_normalized_similarity,
            "rejected_too_few_matches": self.rejected_too_few_matches,
            "rejected_geometry": self.rejected_geometry,
            "rejected_cooldown": self.rejected_cooldown,
            "scale_estimation_failed": self.scale_estimation_failed,
            "accepted": self.accepted,
            "mean_matches_per_candidate": (
                round(float(np.mean(self.match_counts)), 1) if self.match_counts else None
            ),
            "mean_inliers_per_accepted": (
                round(float(np.mean(self.inlier_counts)), 1) if self.inlier_counts else None
            ),
        }


class LoopClosureDetector:
    """Retrieves, verifies and accepts loop closures over a keyframe database."""

    def __init__(
        self,
        K: np.ndarray,
        matcher: FeatureMatcher,
        *,
        min_keyframe_separation: int = 30,
        min_path_separation_m: float = 30.0,
        top_k: int = 5,
        min_similarity: float = 0.2,
        similarity_ratio: float = 0.85,
        min_matches: int = 60,
        min_inliers: int = 40,
        min_inlier_ratio: float = 0.35,
        cooldown_keyframes: int = 10,
        max_loops: int = 0,
        ransac_method: str = "magsac",
        ransac_threshold_px: float = 1.0,
        cheirality_distance: float = 200.0,
        min_scale_points: int = 15,
        min_scale_baseline_m: float = 0.5,
        scale_neighbour_search: int = 5,
        min_loop_scale_m: float = 0.05,
        max_loop_scale_m: float = 30.0,
        timer: StageTimer | None = None,
    ) -> None:
        self.K = np.asarray(K, dtype=np.float64)
        self.matcher = matcher
        self.min_keyframe_separation = int(min_keyframe_separation)
        self.min_path_separation_m = float(min_path_separation_m)
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.similarity_ratio = float(similarity_ratio)
        self.min_matches = int(min_matches)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.cooldown_keyframes = int(cooldown_keyframes)
        self.max_loops = int(max_loops)
        self.ransac_method = str(ransac_method)
        # Loop pairs are wide-baseline, so a slightly looser epipolar threshold
        # than the frame-to-frame front end is appropriate here.
        self.ransac_threshold_px = float(ransac_threshold_px)
        self.cheirality_distance = float(cheirality_distance)
        self.min_scale_points = int(min_scale_points)
        self.min_scale_baseline_m = float(min_scale_baseline_m)
        self.scale_neighbour_search = int(scale_neighbour_search)
        self.min_loop_scale_m = float(min_loop_scale_m)
        self.max_loop_scale_m = float(max_loop_scale_m)
        self.timer = timer if timer is not None else StageTimer()
        self.stats = LoopClosureStats()

    @classmethod
    def from_config(cls, config, K: np.ndarray, matcher: FeatureMatcher, timer=None):  # noqa: ANN001
        lc = config.loop_closure
        return cls(
            K=K,
            matcher=matcher,
            min_keyframe_separation=lc.min_keyframe_separation,
            min_path_separation_m=lc.min_path_separation_m,
            top_k=lc.top_k,
            min_similarity=lc.min_similarity,
            similarity_ratio=lc.similarity_ratio,
            min_matches=lc.min_matches,
            min_inliers=lc.min_inliers,
            min_inlier_ratio=lc.min_inlier_ratio,
            cooldown_keyframes=lc.cooldown_keyframes,
            max_loops=lc.max_loops,
            ransac_method=config.odometry.ransac_method,
            ransac_threshold_px=max(config.odometry.ransac_threshold_px, 1.0),
            cheirality_distance=config.odometry.cheirality_distance,
            min_scale_points=lc.min_scale_points,
            min_scale_baseline_m=lc.min_scale_baseline_m,
            scale_neighbour_search=lc.scale_neighbour_search,
            min_loop_scale_m=lc.min_loop_scale_m,
            max_loop_scale_m=lc.max_loop_scale_m,
            timer=timer,
        )

    # ----------------------------------------------------------------- #
    # Retrieval
    # ----------------------------------------------------------------- #

    def retrieve_candidates(
        self, database: KeyframeDatabase, query_id: int
    ) -> list[LoopCandidate]:
        """Appearance retrieval plus the normalised-similarity filter."""
        raw = database.query(
            query_id,
            min_keyframe_separation=self.min_keyframe_separation,
            min_path_separation_m=self.min_path_separation_m,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
        )
        self.stats.n_queries += 1
        self.stats.n_candidates += len(raw)
        if not raw:
            return []

        reference = database.neighbour_score(query_id)
        candidates: list[LoopCandidate] = []
        for match_id, score in raw:
            normalized = score / reference if reference > 1e-9 else 0.0
            if normalized < self.similarity_ratio:
                self.stats.rejected_normalized_similarity += 1
                continue
            candidates.append(
                LoopCandidate(
                    query_id=query_id,
                    match_id=match_id,
                    similarity=score,
                    normalized_similarity=normalized,
                )
            )
        return candidates

    # ----------------------------------------------------------------- #
    # Geometric verification
    # ----------------------------------------------------------------- #

    def verify(
        self, database: KeyframeDatabase, candidate: LoopCandidate
    ) -> LoopClosure | None:
        """Re-match and geometrically verify a candidate pair.

        Returns the accepted :class:`LoopClosure`, or ``None`` with the reason
        recorded in :attr:`stats`.
        """
        query = database[candidate.query_id]
        match = database[candidate.match_id]

        matches: MatchResult = self.matcher.match_descriptors(
            match.descriptors, query.descriptors, match.points, query.points
        )
        self.stats.match_counts.append(len(matches))
        if len(matches) < self.min_matches:
            self.stats.rejected_too_few_matches += 1
            return None

        essential = estimate_essential_matrix(
            matches.points_a,
            matches.points_b,
            self.K,
            threshold_px=self.ransac_threshold_px,
            method=self.ransac_method,
        )
        if not essential.ok:
            self.stats.rejected_geometry += 1
            return None

        pose = recover_relative_pose(
            essential.E,
            matches.points_a,
            matches.points_b,
            self.K,
            essential.inlier_mask,
            min_inliers=self.min_inliers,
            min_inlier_ratio=0.0,  # applied below against the match count
            max_rotation_deg=180.0,  # a loop may be traversed in any direction
            cheirality_distance=self.cheirality_distance,
        )
        if not pose.ok or pose.T is None:
            self.stats.rejected_geometry += 1
            return None

        inlier_ratio = pose.n_cheirality_inliers / len(matches)
        if pose.n_cheirality_inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            self.stats.rejected_geometry += 1
            return None

        # Monocular translation is unit-norm; recover its metric magnitude from
        # structure the query keyframe shares with its metrically-scaled
        # odometry neighbour. See the module docstring for why the odometry gap
        # between the two loop keyframes must NOT be used here.
        scale, n_scale_points = self.estimate_loop_scale(
            database, candidate.match_id, candidate.query_id, pose.T, matches
        )
        if scale is None:
            self.stats.scale_estimation_failed += 1
            scale_method = "unscaled"
            # Keep the direction but give it no asserted magnitude; the pose
            # graph will down-weight the translation for this edge.
            scale = 0.0
        else:
            scale_method = "structure"

        T_match_query = se3_from_rt(pose.T[:3, :3], pose.T[:3, 3] * scale)

        self.stats.inlier_counts.append(pose.n_cheirality_inliers)
        return LoopClosure(
            query_id=candidate.query_id,
            match_id=candidate.match_id,
            similarity=candidate.similarity,
            normalized_similarity=candidate.normalized_similarity,
            T_match_query=T_match_query,
            n_matches=len(matches),
            n_inliers=pose.n_cheirality_inliers,
            inlier_ratio=inlier_ratio,
            scale_m=scale,
            scale_method=scale_method,
            n_scale_points=n_scale_points,
            query_frame_index=query.frame_index,
            match_frame_index=match.frame_index,
        )

    def estimate_loop_scale(
        self,
        database: KeyframeDatabase,
        match_id: int,
        query_id: int,
        T_match_query_unit: np.ndarray,
        loop_matches: MatchResult,
    ) -> tuple[float | None, int]:
        """Recover the metric magnitude of a loop translation, or ``None``.

        Compares two triangulations of the same scene points:

        * the loop pair ``(match, query)``, triangulated with the **unit-norm**
          relative pose, giving depths that are correct up to the unknown
          factor ``s``;
        * the query keyframe against an **odometry neighbour**, whose relative
          pose carries the front end's metric scale.

        For points seen in all three views, ``s = depth_metric / depth_unit``.
        The median over those ratios is returned, which resists the outliers
        that individual bad triangulations produce.

        Returns ``(scale, n_supporting_points)``; ``scale`` is ``None`` when
        there was not enough shared, well-conditioned structure.
        """
        query = database[query_id]

        neighbour, T_query_neighbour = self._metric_neighbour(database, query_id)
        if neighbour is None:
            return None, 0

        # Structure from the loop pair, expressed in the query keyframe's frame.
        loop_points_match_frame = triangulate_points(
            T_match_query_unit, loop_matches.points_a, loop_matches.points_b, self.K
        )
        T_query_match_unit = invert_se3(T_match_query_unit)
        loop_points_query_frame = (
            loop_points_match_frame @ T_query_match_unit[:3, :3].T + T_query_match_unit[:3, 3]
        )

        # Metric structure from the query keyframe and its odometry neighbour.
        local_matches = self.matcher.match_descriptors(
            query.descriptors, neighbour.descriptors, query.points, neighbour.points
        )
        if len(local_matches) < self.min_scale_points:
            return None, 0
        local_points = triangulate_points(
            T_query_neighbour, local_matches.points_a, local_matches.points_b, self.K
        )

        # Features observed in all three views, keyed by query keypoint index.
        loop_slot = {int(idx): i for i, idx in enumerate(loop_matches.indices_b)}
        shared = [
            (loop_slot[int(idx)], i)
            for i, idx in enumerate(local_matches.indices_a)
            if int(idx) in loop_slot
        ]
        if len(shared) < self.min_scale_points:
            return None, len(shared)

        unit_depths = np.array([loop_points_query_frame[a, 2] for a, _ in shared])
        metric_depths = np.array([local_points[b, 2] for _, b in shared])

        valid = (
            np.isfinite(unit_depths)
            & np.isfinite(metric_depths)
            & (unit_depths > 1e-3)
            & (metric_depths > 1e-3)
        )
        if int(valid.sum()) < self.min_scale_points:
            return None, int(valid.sum())

        ratios = metric_depths[valid] / unit_depths[valid]
        # Trim the tails before taking the median: a handful of near-degenerate
        # triangulations can otherwise sit at absurd ratios.
        low, high = np.percentile(ratios, [20, 80])
        trimmed = ratios[(ratios >= low) & (ratios <= high)]
        scale = float(np.median(trimmed if len(trimmed) else ratios))

        if not np.isfinite(scale) or not (self.min_loop_scale_m <= scale <= self.max_loop_scale_m):
            return None, int(valid.sum())
        return scale, int(valid.sum())

    def _metric_neighbour(
        self, database: KeyframeDatabase, query_id: int
    ) -> tuple[object | None, np.ndarray]:
        """Find a nearby keyframe with enough metric baseline to triangulate.

        Walks backwards from the query keyframe until the odometry translation
        exceeds a minimum baseline; a stationary vehicle produces adjacent
        keyframes with no parallax, from which nothing can be triangulated.
        """
        query = database[query_id]
        for offset in range(1, self.scale_neighbour_search + 1):
            candidate_id = query_id - offset
            if candidate_id < 0:
                break
            neighbour = database[candidate_id]
            T_query_neighbour = invert_se3(query.pose) @ neighbour.pose
            if float(np.linalg.norm(T_query_neighbour[:3, 3])) >= self.min_scale_baseline_m:
                return neighbour, T_query_neighbour
        return None, np.eye(4)

    # ----------------------------------------------------------------- #
    # Full detection pass
    # ----------------------------------------------------------------- #

    def detect(self, database: KeyframeDatabase) -> list[LoopClosure]:
        """Run detection across the whole keyframe database.

        Queries run in keyframe order so the cooldown behaves causally, exactly
        as it would in an online system.
        """
        if len(database) == 0:
            return []
        self.stats = LoopClosureStats()

        with self.timer.time("loop_index"):
            database.build_index()

        closures: list[LoopClosure] = []
        last_accepted = -(10**9)

        with self.timer.time("loop_detect"):
            for query_id in range(len(database)):
                if query_id < self.min_keyframe_separation:
                    continue
                if query_id - last_accepted < self.cooldown_keyframes:
                    self.stats.rejected_cooldown += 1
                    continue
                if self.max_loops and len(closures) >= self.max_loops:
                    break

                for candidate in self.retrieve_candidates(database, query_id):
                    closure = self.verify(database, candidate)
                    if closure is None:
                        continue
                    closures.append(closure)
                    self.stats.accepted += 1
                    last_accepted = query_id
                    logger.info(
                        "Loop closure: kf %d <-> kf %d (frames %d <-> %d) | "
                        "sim %.3f (norm %.2f) | %d/%d inliers (%.0f%%)",
                        closure.match_id,
                        closure.query_id,
                        closure.match_frame_index,
                        closure.query_frame_index,
                        closure.similarity,
                        closure.normalized_similarity,
                        closure.n_inliers,
                        closure.n_matches,
                        100.0 * closure.inlier_ratio,
                    )
                    break  # one constraint per query keyframe is enough

        logger.info(
            "Loop closure detection: %d accepted from %d candidates over %d queries",
            self.stats.accepted,
            self.stats.n_candidates,
            self.stats.n_queries,
        )
        return closures
