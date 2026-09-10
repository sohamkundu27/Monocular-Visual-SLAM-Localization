"""Keyframe representation and bag-of-visual-words retrieval.

Why a vocabulary at all
-----------------------
Naively comparing a query keyframe against every past keyframe by brute-force
descriptor matching is O(N_kf * 3000 * 3000) Hamming comparisons per query. On
KITTI sequence 00 that is roughly 900 keyframes by the end of the run, and the
cost grows quadratically with sequence length — minutes per query, which makes
the whole idea unusable.

The bag-of-visual-words model reduces each keyframe to a single fixed-length
vector. Descriptors are quantised against a vocabulary of visual words, the
word counts are TF-IDF weighted and L2-normalised, and similarity becomes one
cosine product. Retrieval over the whole database is then a single matrix-vector
multiply: microseconds instead of minutes.

The vocabulary is clustered from the sequence's own descriptors rather than
loaded from a pre-trained file. This keeps the project self-contained (no
external vocabulary download) and adapts the words to the actual imagery. The
trade-off is that it is not transferable between datasets — a real deployment
would train once on a large corpus offline.

Binary k-means
--------------
ORB descriptors are 256-bit strings, so clustering uses Hamming distance with
majority-vote centroids (the "k-majority" variant of k-means) rather than
Euclidean k-means on the raw bytes, which would be meaningless: byte 0xFF and
byte 0x00 differ by 8 bits, not by 255 units. SIFT descriptors, being real
valued, fall back to standard Euclidean k-means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from monocular_slam.geometry.transforms import rotation_angle_deg
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Keyframe:
    """A frame retained for loop-closure search and pose-graph optimization.

    Descriptors are kept because geometric verification needs to re-match a
    candidate pair. Only keyframes are stored — retaining descriptors for all
    4541 KITTI frames would cost roughly 440 MB, while ~900 keyframes cost
    under 100 MB.
    """

    #: Position of this keyframe in the keyframe sequence (0, 1, 2, ...).
    keyframe_id: int
    #: Index into the processed frame sequence.
    frame_index: int
    #: Original KITTI frame number.
    frame_id: int
    #: ``(N, 2)`` keypoint pixel coordinates.
    points: np.ndarray
    #: ``(N, D)`` descriptors.
    descriptors: np.ndarray
    #: Camera-to-world pose at insertion time (pre-optimization).
    pose: np.ndarray
    #: Path length travelled from the start of the sequence, in metres.
    path_distance_m: float = 0.0
    #: TF-IDF weighted, L2-normalised bag-of-words vector; filled in by the
    #: database once a vocabulary exists.
    bow: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.points)


class KeyframeSelector:
    """Decides which frames become keyframes.

    A frame is promoted when *any* of three conditions holds: enough frames
    have elapsed, the camera has translated far enough, or it has rotated far
    enough. Distance and rotation triggers matter because a vehicle stopped at
    a light would otherwise flood the database with near-identical keyframes,
    while a fast turn would leave a gap exactly where appearance changes most.
    """

    def __init__(
        self,
        every_n_frames: int = 5,
        min_translation_m: float = 2.0,
        min_rotation_deg: float = 10.0,
    ) -> None:
        self.every_n_frames = max(1, int(every_n_frames))
        self.min_translation_m = float(min_translation_m)
        self.min_rotation_deg = float(min_rotation_deg)
        self._last_index: int | None = None
        self._last_pose: np.ndarray | None = None

    @classmethod
    def from_config(cls, config) -> KeyframeSelector:  # noqa: ANN001
        k = config.keyframes
        return cls(k.every_n_frames, k.min_translation_m, k.min_rotation_deg)

    def should_select(self, index: int, pose: np.ndarray) -> bool:
        """True when frame ``index`` at ``pose`` should become a keyframe."""
        if self._last_index is None:
            return True
        if index - self._last_index >= self.every_n_frames:
            return True
        if self._last_pose is not None:
            translation = float(np.linalg.norm(pose[:3, 3] - self._last_pose[:3, 3]))
            if translation >= self.min_translation_m:
                return True
            rotation = rotation_angle_deg(self._last_pose[:3, :3].T @ pose[:3, :3])
            if rotation >= self.min_rotation_deg:
                return True
        return False

    def accept(self, index: int, pose: np.ndarray) -> None:
        """Record that ``index`` was selected, resetting the triggers."""
        self._last_index = index
        self._last_pose = np.asarray(pose, dtype=np.float64).copy()

    def reset(self) -> None:
        self._last_index = None
        self._last_pose = None


class VisualVocabulary:
    """A visual vocabulary clustered from descriptors."""

    def __init__(self, centroids: np.ndarray, binary: bool) -> None:
        self.centroids = centroids
        self.binary = bool(binary)
        if self.binary:
            # Pre-unpack centroids to bits once; quantisation then reduces to a
            # matrix product instead of per-descriptor bit twiddling.
            self._centroid_bits = np.unpackbits(centroids, axis=1).astype(np.float32)

    @property
    def size(self) -> int:
        return int(len(self.centroids))

    @classmethod
    def train(
        cls,
        descriptors: np.ndarray,
        vocabulary_size: int = 256,
        iterations: int = 12,
        seed: int = 0,
    ) -> VisualVocabulary:
        """Cluster ``descriptors`` into ``vocabulary_size`` visual words."""
        descriptors = np.asarray(descriptors)
        if len(descriptors) == 0:
            raise ValueError("Cannot train a vocabulary from zero descriptors")
        k = int(min(vocabulary_size, len(descriptors)))
        if k < 2:
            raise ValueError(f"Vocabulary needs at least 2 words, got {k}")

        if descriptors.dtype == np.uint8:
            centroids = _binary_kmeans(descriptors, k, iterations=iterations, seed=seed)
            return cls(centroids, binary=True)

        data = descriptors.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
        _compactness, _labels, centroids = cv2.kmeans(
            data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        return cls(centroids.astype(np.float32), binary=False)

    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        """Assign each descriptor to its nearest word; returns ``(N,)`` word ids."""
        descriptors = np.asarray(descriptors)
        if len(descriptors) == 0:
            return np.zeros(0, dtype=np.int64)

        if self.binary:
            bits = np.unpackbits(descriptors, axis=1).astype(np.float32)
            distances = _hamming_distances(bits, self._centroid_bits)
        else:
            data = descriptors.astype(np.float32)
            distances = (
                np.sum(data**2, axis=1, keepdims=True)
                - 2.0 * (data @ self.centroids.T)
                + np.sum(self.centroids**2, axis=1)[None, :]
            )
        return np.argmin(distances, axis=1).astype(np.int64)

    def histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Raw word-count histogram of length :attr:`size`."""
        words = self.quantize(descriptors)
        return np.bincount(words, minlength=self.size).astype(np.float64)


def _hamming_distances(bits: np.ndarray, centroid_bits: np.ndarray) -> np.ndarray:
    """``(N, k)`` Hamming distances between unpacked bit vectors.

    For 0/1 vectors ``a`` and ``b``, ``||a - b||_1 == sum(a) + sum(b) - 2 a.b``,
    so the whole distance matrix reduces to one BLAS matrix product.
    """
    return (
        bits.sum(axis=1, keepdims=True)
        + centroid_bits.sum(axis=1)[None, :]
        - 2.0 * (bits @ centroid_bits.T)
    )


def _kmeans_plusplus_binary(bits: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding under Hamming distance.

    Seeding matters more than it might appear: with uniformly random seeds,
    k-means routinely converges to a local optimum that splits one true cluster
    across two words while merging two others, which directly costs retrieval
    precision. k-means++ picks each new centre with probability proportional to
    its squared distance from the nearest existing centre, which spreads the
    initial words out and largely removes that failure.
    """
    n = len(bits)
    centres = np.empty((k, bits.shape[1]), dtype=np.float32)
    centres[0] = bits[rng.integers(0, n)]

    closest = _hamming_distances(bits, centres[:1]).ravel()
    for i in range(1, k):
        weights = closest**2
        total = float(weights.sum())
        if total <= 0:
            # All remaining points coincide with a chosen centre.
            centres[i] = bits[rng.integers(0, n)]
        else:
            centres[i] = bits[int(rng.choice(n, p=weights / total))]
        closest = np.minimum(closest, _hamming_distances(bits, centres[i : i + 1]).ravel())
    return centres


def _binary_kmeans(
    descriptors: np.ndarray, k: int, iterations: int = 12, seed: int = 0
) -> np.ndarray:
    """k-means over binary descriptors using Hamming distance.

    Centroids are recomputed by majority vote per bit, which is the correct
    mean under Hamming distance and keeps centroids inside the binary space.
    Euclidean k-means on the raw descriptor bytes would be meaningless, since
    byte 0xFF and 0x00 differ by 8 bits but 255 units.
    """
    rng = np.random.default_rng(seed)
    bits = np.unpackbits(descriptors, axis=1).astype(np.float32)
    n = len(bits)

    centroid_bits = _kmeans_plusplus_binary(bits, k, rng)

    labels = np.zeros(n, dtype=np.int64)
    for iteration in range(iterations):
        new_labels = np.argmin(_hamming_distances(bits, centroid_bits), axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for word in range(k):
            members = bits[labels == word]
            if len(members) == 0:
                # Re-seed an empty cluster so the vocabulary does not silently
                # shrink below the requested size.
                centroid_bits[word] = bits[rng.integers(0, n)]
            else:
                centroid_bits[word] = (members.mean(axis=0) >= 0.5).astype(np.float32)

    return np.packbits(centroid_bits.astype(np.uint8), axis=1)


class KeyframeDatabase:
    """Stores keyframes and retrieves visually similar past keyframes.

    TF-IDF weighting
    ----------------
    Words that appear in nearly every keyframe (sky, road texture, lane
    markings) carry no discriminative information about *where* the camera is.
    The inverse document frequency term ``log(N / n_i)`` down-weights them,
    which is what stops a highway sequence from declaring every frame similar
    to every other. IDF is recomputed lazily from the current database, so it
    reflects the sequence actually seen so far.
    """

    def __init__(self, vocabulary: VisualVocabulary | None = None) -> None:
        self.vocabulary = vocabulary
        self.keyframes: list[Keyframe] = []
        self._document_frequency: np.ndarray | None = None
        self._bow_matrix: np.ndarray | None = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self.keyframes)

    def __getitem__(self, index: int) -> Keyframe:
        return self.keyframes[index]

    def __iter__(self):
        return iter(self.keyframes)

    def set_vocabulary(self, vocabulary: VisualVocabulary) -> None:
        """Install a vocabulary and (re)compute descriptors for stored keyframes."""
        self.vocabulary = vocabulary
        self._dirty = True

    def add(self, keyframe: Keyframe) -> Keyframe:
        """Insert a keyframe, assigning it the next keyframe id."""
        keyframe.keyframe_id = len(self.keyframes)
        self.keyframes.append(keyframe)
        self._dirty = True
        return keyframe

    # ----------------------------------------------------------------- #
    # BoW construction
    # ----------------------------------------------------------------- #

    def build_index(self) -> None:
        """Compute TF-IDF BoW vectors for every stored keyframe.

        Called once after the vocabulary is trained. Doing this in one batch
        rather than incrementally means IDF is computed from the whole
        database, which is both simpler and more stable than updating weights
        as keyframes stream in.
        """
        if self.vocabulary is None:
            raise RuntimeError("A vocabulary must be set before building the index")
        if not self.keyframes:
            self._bow_matrix = np.zeros((0, self.vocabulary.size))
            self._dirty = False
            return

        counts = np.stack(
            [self.vocabulary.histogram(kf.descriptors) for kf in self.keyframes]
        )

        # Document frequency: how many keyframes contain each word at all.
        self._document_frequency = np.count_nonzero(counts > 0, axis=0).astype(np.float64)
        n_documents = len(self.keyframes)
        # +1 inside the log guards against a word that appears nowhere.
        idf = np.log(n_documents / np.maximum(self._document_frequency, 1.0)) + 1e-6

        # Term frequency, normalised per keyframe so that keyframes with more
        # detected features are not systematically scored higher.
        totals = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        tf = counts / totals

        vectors = tf * idf[None, :]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        for keyframe, vector in zip(self.keyframes, vectors):
            keyframe.bow = vector
        self._bow_matrix = vectors
        self._dirty = False
        logger.info(
            "Built BoW index: %d keyframes, %d words, mean %d descriptors per keyframe",
            n_documents,
            self.vocabulary.size,
            int(np.mean([len(kf) for kf in self.keyframes])),
        )

    @property
    def bow_matrix(self) -> np.ndarray:
        """``(N_kf, vocab)`` matrix of normalised BoW vectors."""
        if self._dirty or self._bow_matrix is None:
            self.build_index()
        assert self._bow_matrix is not None
        return self._bow_matrix

    # ----------------------------------------------------------------- #
    # Retrieval
    # ----------------------------------------------------------------- #

    def similarity_scores(self, query_id: int) -> np.ndarray:
        """Cosine similarity of keyframe ``query_id`` against all keyframes.

        Vectors are L2-normalised, so the dot product *is* the cosine.
        """
        matrix = self.bow_matrix
        if not 0 <= query_id < len(matrix):
            raise IndexError(f"query_id {query_id} out of range (0..{len(matrix) - 1})")
        # Clipped because rounding can push a self-similarity fractionally past
        # 1.0, which would otherwise leak into the normalised-similarity ratio.
        return np.clip(matrix @ matrix[query_id], -1.0, 1.0)

    def query(
        self,
        query_id: int,
        min_keyframe_separation: int = 30,
        min_path_separation_m: float = 30.0,
        top_k: int = 5,
        min_similarity: float = 0.2,
    ) -> list[tuple[int, float]]:
        """Return up to ``top_k`` plausible revisits as ``(keyframe_id, score)``.

        Temporal and spatial exclusion
        ------------------------------
        Keyframes adjacent in time are trivially similar — they see the same
        scene — so a raw top-k query returns the immediate neighbours and
        nothing useful. Two exclusions are applied:

        * **Keyframe separation**: candidates must be at least
          ``min_keyframe_separation`` keyframes older.
        * **Path separation**: they must also be at least
          ``min_path_separation_m`` of *travelled distance* away. This second
          test is what protects a stationary vehicle, which can accumulate many
          keyframe ids without moving at all.
        """
        if not self.keyframes:
            return []
        scores = self.similarity_scores(query_id)
        query = self.keyframes[query_id]

        eligible = np.zeros(len(scores), dtype=bool)
        for candidate_id in range(query_id - min_keyframe_separation + 1):
            candidate = self.keyframes[candidate_id]
            travelled = query.path_distance_m - candidate.path_distance_m
            if travelled >= min_path_separation_m:
                eligible[candidate_id] = True

        eligible &= scores >= min_similarity
        candidate_ids = np.flatnonzero(eligible)
        if len(candidate_ids) == 0:
            return []

        order = candidate_ids[np.argsort(-scores[candidate_ids])][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def neighbour_score(self, query_id: int, window: int = 5) -> float:
        """Best similarity among temporally adjacent keyframes.

        Used as a normalising reference: a loop candidate should score
        comparably to what the *same* place looks like a few frames apart. In a
        low-texture scene all scores are low, and in a highly repetitive one all
        scores are high, so an absolute threshold alone generalises poorly.
        This is the normalisation DBoW2 popularised.
        """
        scores = self.similarity_scores(query_id)
        low = max(0, query_id - window)
        neighbours = np.concatenate([scores[low:query_id], scores[query_id + 1 : query_id + 1 + window]])
        finite = neighbours[np.isfinite(neighbours)]
        return float(finite.max()) if len(finite) else 0.0

    def describe(self) -> dict[str, object]:
        return {
            "n_keyframes": len(self.keyframes),
            "vocabulary_size": None if self.vocabulary is None else self.vocabulary.size,
            "binary_vocabulary": None if self.vocabulary is None else self.vocabulary.binary,
        }
