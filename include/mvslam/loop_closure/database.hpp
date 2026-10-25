// Keyframe representation and bag-of-visual-words retrieval.
//
// Why a vocabulary at all
// -----------------------
// Naively comparing a query keyframe against every past keyframe by
// brute-force descriptor matching is O(N_kf * 3000 * 3000) Hamming
// comparisons per query. On KITTI sequence 00 that is roughly 900 keyframes by
// the end of the run, and the cost grows quadratically with sequence length --
// minutes per query, which makes the whole idea unusable.
//
// The bag-of-visual-words model reduces each keyframe to a single
// fixed-length vector. Descriptors are quantised against a vocabulary of
// visual words, the word counts are TF-IDF weighted and L2-normalised, and
// similarity becomes one cosine product. Retrieval over the whole database is
// then a single matrix-vector multiply: microseconds instead of minutes.
//
// The vocabulary is clustered from the sequence's own descriptors rather than
// loaded from a pre-trained file. This keeps the project self-contained (no
// external vocabulary download) and adapts the words to the actual imagery.
// The trade-off is that it is not transferable between datasets -- a real
// deployment would train once on a large corpus offline.
//
// Binary k-means
// --------------
// ORB descriptors are 256-bit strings, so clustering uses Hamming distance
// with majority-vote centroids (the "k-majority" variant of k-means) rather
// than Euclidean k-means on the raw bytes, which would be meaningless: byte
// 0xFF and byte 0x00 differ by 8 bits, not by 255 units. SIFT descriptors,
// being real valued, fall back to standard Euclidean k-means.
//
// Port note: the Python original draws k-means++ seeds from NumPy's PCG64.
// The port uses std::mt19937_64 seeded from runtime.seed with its own
// (library-independent) uniform and weighted sampling, so vocabularies match
// the Python ones statistically, not bit for bit.
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/utils/cv_eigen.hpp"

namespace mvslam {

struct Config;

/// Row-major so each keyframe's BoW vector is contiguous, as in NumPy.
using BowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// A frame retained for loop-closure search and pose-graph optimization.
///
/// Descriptors are kept because geometric verification needs to re-match a
/// candidate pair. Only keyframes are stored -- retaining descriptors for all
/// 4541 KITTI frames would cost roughly 440 MB, while ~900 keyframes cost
/// under 100 MB.
struct Keyframe {
    /// Position of this keyframe in the keyframe sequence (0, 1, 2, ...).
    std::int64_t keyframe_id = -1;
    /// Index into the processed frame sequence.
    std::int64_t frame_index = 0;
    /// Original KITTI frame number.
    std::int64_t frame_id = 0;
    /// (N, 2) keypoint pixel coordinates.
    Points2 points;
    /// (N, D) descriptors.
    cv::Mat descriptors;
    /// Camera-to-world pose at insertion time (pre-optimization).
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    /// Path length travelled from the start of the sequence, in metres.
    double path_distance_m = 0.0;
    /// TF-IDF weighted, L2-normalised bag-of-words vector; filled in by the
    /// database once a vocabulary exists.
    std::optional<Eigen::VectorXd> bow;

    std::size_t size() const { return static_cast<std::size_t>(points.rows()); }
};

/// Decides which frames become keyframes.
///
/// A frame is promoted when *any* of three conditions holds: enough frames
/// have elapsed, the camera has translated far enough, or it has rotated far
/// enough. Distance and rotation triggers matter because a vehicle stopped at
/// a light would otherwise flood the database with near-identical keyframes,
/// while a fast turn would leave a gap exactly where appearance changes most.
class KeyframeSelector {
public:
    explicit KeyframeSelector(std::int64_t every_n_frames = 5, double min_translation_m = 2.0,
                              double min_rotation_deg = 10.0);
    static KeyframeSelector from_config(const Config& config);

    /// True when frame ``index`` at ``pose`` should become a keyframe.
    bool should_select(std::int64_t index, const Eigen::Matrix4d& pose) const;
    /// Record that ``index`` was selected, resetting the triggers.
    void accept(std::int64_t index, const Eigen::Matrix4d& pose);
    void reset();

    std::int64_t every_n_frames() const { return every_n_frames_; }

private:
    std::int64_t every_n_frames_;
    double min_translation_m_;
    double min_rotation_deg_;
    std::optional<std::int64_t> last_index_;
    std::optional<Eigen::Matrix4d> last_pose_;
};

/// Deterministic, standard-library-independent sampling on top of
/// std::mt19937_64 (std::uniform_int_distribution and friends differ between
/// libstdc++ and libc++, which would make runs non-reproducible across
/// toolchains).
class Sampler {
public:
    explicit Sampler(std::uint64_t seed) : engine_(seed) {}
    /// Uniform integer in ``[0, n)`` (rejection sampling, no modulo bias).
    std::uint64_t integer(std::uint64_t n);
    /// Uniform double in ``[0, 1)`` with 53 random bits.
    double uniform();
    /// Index drawn with probability proportional to ``weights``.
    std::size_t weighted(const std::vector<double>& weights, double total);
    /// ``size`` distinct indices out of ``[0, n)`` (partial Fisher-Yates).
    std::vector<std::size_t> sample_without_replacement(std::size_t n, std::size_t size);

private:
    std::mt19937_64 engine_;
};

/// A visual vocabulary clustered from descriptors.
class VisualVocabulary {
public:
    /// ``centroids`` is (k, 32) CV_8U for a binary vocabulary (packed bits, as
    /// np.packbits leaves them) or (k, D) CV_32F otherwise.
    VisualVocabulary(cv::Mat centroids, bool binary);

    std::size_t size() const { return static_cast<std::size_t>(centroids_.rows); }
    bool binary() const { return binary_; }
    const cv::Mat& centroids() const { return centroids_; }

    /// Cluster ``descriptors`` into ``vocabulary_size`` visual words.
    static VisualVocabulary train(const cv::Mat& descriptors, std::int64_t vocabulary_size = 256,
                                  int iterations = 12, std::int64_t seed = 0);

    /// Assign each descriptor to its nearest word; returns (N) word ids.
    std::vector<std::int64_t> quantize(const cv::Mat& descriptors) const;

    /// Raw word-count histogram of length size().
    Eigen::VectorXd histogram(const cv::Mat& descriptors) const;

private:
    cv::Mat centroids_;
    bool binary_;
};

/// Stores keyframes and retrieves visually similar past keyframes.
///
/// TF-IDF weighting
/// ----------------
/// Words that appear in nearly every keyframe (sky, road texture, lane
/// markings) carry no discriminative information about *where* the camera is.
/// The inverse document frequency term ``log(N / n_i)`` down-weights them,
/// which is what stops a highway sequence from declaring every frame similar
/// to every other. IDF is recomputed lazily from the current database, so it
/// reflects the sequence actually seen so far.
class KeyframeDatabase {
public:
    explicit KeyframeDatabase(std::optional<VisualVocabulary> vocabulary = std::nullopt)
        : vocabulary_(std::move(vocabulary)) {}

    std::size_t size() const { return keyframes_.size(); }
    const Keyframe& operator[](std::size_t index) const { return keyframes_.at(index); }
    Keyframe& operator[](std::size_t index) { return keyframes_.at(index); }
    std::vector<Keyframe>::const_iterator begin() const { return keyframes_.begin(); }
    std::vector<Keyframe>::const_iterator end() const { return keyframes_.end(); }
    const std::optional<VisualVocabulary>& vocabulary() const { return vocabulary_; }

    /// Install a vocabulary; the index is rebuilt on next use.
    void set_vocabulary(VisualVocabulary vocabulary);

    /// Insert a keyframe, assigning it the next keyframe id.
    Keyframe& add(Keyframe keyframe);

    /// Compute TF-IDF BoW vectors for every stored keyframe.
    ///
    /// Called once after the vocabulary is trained. Doing this in one batch
    /// rather than incrementally means IDF is computed from the whole
    /// database, which is both simpler and more stable than updating weights
    /// as keyframes stream in. Throws RuntimeError without a vocabulary.
    void build_index();

    /// (N_kf, vocab) matrix of normalised BoW vectors (built on demand).
    const BowMatrix& bow_matrix();

    /// Cosine similarity of keyframe ``query_id`` against all keyframes.
    ///
    /// Vectors are L2-normalised, so the dot product *is* the cosine. Clipped
    /// to [-1, 1] because rounding can push a self-similarity fractionally past
    /// 1.0, which would otherwise leak into the normalised-similarity ratio.
    std::vector<double> similarity_scores(std::int64_t query_id);

    /// Up to ``top_k`` plausible revisits as ``(keyframe_id, score)``.
    ///
    /// Temporal and spatial exclusion: keyframes adjacent in time are
    /// trivially similar -- they see the same scene -- so a raw top-k query
    /// returns the immediate neighbours and nothing useful. Candidates must be
    /// at least ``min_keyframe_separation`` keyframes older *and* at least
    /// ``min_path_separation_m`` of travelled distance away. The second test
    /// is what protects a stationary vehicle, which can accumulate many
    /// keyframe ids without moving at all.
    std::vector<std::pair<std::int64_t, double>> query(std::int64_t query_id,
                                                       std::int64_t min_keyframe_separation = 30,
                                                       double min_path_separation_m = 30.0,
                                                       std::int64_t top_k = 5,
                                                       double min_similarity = 0.2);

    /// Best similarity among temporally adjacent keyframes.
    ///
    /// Used as a normalising reference: a loop candidate should score
    /// comparably to what the *same* place looks like a few frames apart. In a
    /// low-texture scene all scores are low, and in a highly repetitive one all
    /// scores are high, so an absolute threshold alone generalises poorly.
    /// This is the normalisation DBoW2 popularised.
    double neighbour_score(std::int64_t query_id, std::int64_t window = 5);

    Json describe() const;

private:
    std::optional<VisualVocabulary> vocabulary_;
    std::vector<Keyframe> keyframes_;
    Eigen::VectorXd document_frequency_;
    BowMatrix bow_matrix_;
    bool has_matrix_ = false;
    bool dirty_ = true;
};

/// ``np.unpackbits(row)`` of one packed descriptor row (MSB first).
std::vector<std::uint8_t> unpack_bits(const std::uint8_t* row, int bytes);

}  // namespace mvslam
