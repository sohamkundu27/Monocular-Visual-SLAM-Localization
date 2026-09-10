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

The constraint produced is a relative pose ``T_a_b`` between the two keyframes.
Its rotation is fully determined; its **translation is unit-norm only**, for
the same monocular reason the odometry front end cannot recover scale. The
magnitude is taken from the current odometry estimate of the gap between the
two keyframes, which is the best available proxy. Because that magnitude is
uncertain, loop factors are given looser translation noise than odometry
factors and are wrapped in a robust kernel — see
:mod:`monocular_slam.optimization.pose_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monocular_slam.features.matcher import FeatureMatcher, MatchResult
from monocular_slam.geometry.epipolar import estimate_essential_matrix, recover_relative_pose
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
    #: Odometry-derived translation magnitude applied to the unit direction.
    scale_m: float
    #: Frame indices, for plotting against the trajectory.
    query_frame_index: int = 0
    match_frame_index: int = 0

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

        # Monocular translation is unit-norm; borrow the magnitude from the
        # odometry estimate of the gap between these two keyframes.
        odometry_gap = invert_se3(match.pose) @ query.pose
        scale = float(np.linalg.norm(odometry_gap[:3, 3]))
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
            query_frame_index=query.frame_index,
            match_frame_index=match.frame_index,
        )

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
