#include "mvslam/loop_closure/database.hpp"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <limits>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/config.hpp"
#include "mvslam/errors.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "mvslam/utils/logging.hpp"

namespace mvslam {

namespace {

const Logger& logger() {
    static const Logger instance = get_logger("monocular_slam.loop_closure.database");
    return instance;
}

/// Packed binary descriptors as 64-bit words for popcount Hamming distances.
/// For 0/1 vectors, sum(a) + sum(b) - 2 a.b (the Python formulation) equals
/// the popcount of a XOR b exactly, so assignments are identical.
struct Packed {
    std::vector<std::uint64_t> data;
    std::size_t rows = 0;
    int bytes = 0;  // descriptor width in bytes
    int words = 0;  // 64-bit words per row (zero padded)

    const std::uint64_t* row(std::size_t r) const { return &data[r * static_cast<std::size_t>(words)]; }
    std::uint64_t* row(std::size_t r) { return &data[r * static_cast<std::size_t>(words)]; }
};

Packed pack(const cv::Mat& descriptors) {
    Packed out;
    out.rows = static_cast<std::size_t>(descriptors.rows);
    out.bytes = descriptors.cols;
    out.words = (descriptors.cols + 7) / 8;
    out.data.assign(out.rows * static_cast<std::size_t>(out.words), 0);
    for (int r = 0; r < descriptors.rows; ++r) {
        std::memcpy(out.row(static_cast<std::size_t>(r)), descriptors.ptr<std::uint8_t>(r),
                    static_cast<std::size_t>(out.bytes));
    }
    return out;
}

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("popcnt")))
#endif
inline int hamming_words(const std::uint64_t* a, const std::uint64_t* b, int words) {
    int d = 0;
    for (int w = 0; w < words; ++w) {
        d += __builtin_popcountll(a[w] ^ b[w]);
    }
    return d;
}

/// Index of the nearest centroid (first on ties, like np.argmin).
std::size_t nearest(const std::uint64_t* descriptor, const Packed& centroids) {
    std::size_t best = 0;
    int best_distance = std::numeric_limits<int>::max();
    for (std::size_t c = 0; c < centroids.rows; ++c) {
        const int d = hamming_words(descriptor, centroids.row(c), centroids.words);
        if (d < best_distance) {
            best_distance = d;
            best = c;
        }
    }
    return best;
}

/// k-means++ seeding under Hamming distance.
///
/// Seeding matters more than it might appear: with uniformly random seeds,
/// k-means routinely converges to a local optimum that splits one true cluster
/// across two words while merging two others, which directly costs retrieval
/// precision. k-means++ picks each new centre with probability proportional
/// to its squared distance from the nearest existing centre, which spreads the
/// initial words out and largely removes that failure.
Packed kmeans_plusplus_binary(const Packed& data, std::size_t k, Sampler& rng) {
    const std::size_t n = data.rows;
    Packed centres;
    centres.rows = k;
    centres.bytes = data.bytes;
    centres.words = data.words;
    centres.data.assign(k * static_cast<std::size_t>(data.words), 0);
    auto copy_row = [&](std::size_t centre, std::size_t row) {
        std::memcpy(centres.row(centre), data.row(row), static_cast<std::size_t>(data.words) * 8);
    };
    copy_row(0, rng.integer(n));
    std::vector<double> closest(n);
    for (std::size_t j = 0; j < n; ++j) {
        closest[j] = hamming_words(data.row(j), centres.row(0), data.words);
    }
    std::vector<double> weights(n);
    for (std::size_t i = 1; i < k; ++i) {
        double total = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            weights[j] = closest[j] * closest[j];
            total += weights[j];
        }
        if (total <= 0) {
            // All remaining points coincide with a chosen centre.
            copy_row(i, rng.integer(n));
        } else {
            copy_row(i, rng.weighted(weights, total));
        }
        for (std::size_t j = 0; j < n; ++j) {
            closest[j] = std::min(closest[j],
                                  static_cast<double>(hamming_words(data.row(j), centres.row(i), data.words)));
        }
    }
    return centres;
}

/// k-means over binary descriptors using Hamming distance.
///
/// Centroids are recomputed by majority vote per bit, which is the correct
/// mean under Hamming distance and keeps centroids inside the binary space.
/// Euclidean k-means on the raw descriptor bytes would be meaningless, since
/// byte 0xFF and 0x00 differ by 8 bits but 255 units.
cv::Mat binary_kmeans(const cv::Mat& descriptors, std::size_t k, int iterations, std::int64_t seed) {
    Sampler rng(static_cast<std::uint64_t>(seed));
    const Packed data = pack(descriptors);
    const std::size_t n = data.rows;
    const int bits = data.bytes * 8;
    Packed centroids = kmeans_plusplus_binary(data, k, rng);

    std::vector<std::size_t> labels(n, 0);
    std::vector<std::size_t> new_labels(n, 0);
    std::vector<std::int64_t> bit_counts(k * static_cast<std::size_t>(bits));
    std::vector<std::int64_t> members(k);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t j = 0; j < n; ++j) {
            new_labels[j] = nearest(data.row(j), centroids);
        }
        if (iteration > 0 && new_labels == labels) {
            break;
        }
        labels = new_labels;

        std::fill(bit_counts.begin(), bit_counts.end(), 0);
        std::fill(members.begin(), members.end(), 0);
        for (std::size_t j = 0; j < n; ++j) {
            const std::size_t word = labels[j];
            members[word] += 1;
            const std::uint8_t* row = descriptors.ptr<std::uint8_t>(static_cast<int>(j));
            std::int64_t* counts = &bit_counts[word * static_cast<std::size_t>(bits)];
            for (int b = 0; b < data.bytes; ++b) {
                for (int bit = 0; bit < 8; ++bit) {
                    counts[b * 8 + bit] += (row[b] >> (7 - bit)) & 1;  // np.unpackbits order
                }
            }
        }
        for (std::size_t word = 0; word < k; ++word) {
            std::uint8_t* centre = reinterpret_cast<std::uint8_t*>(centroids.row(word));
            if (members[word] == 0) {
                // Re-seed an empty cluster so the vocabulary does not silently
                // shrink below the requested size.
                const std::size_t row = rng.integer(n);
                std::memcpy(centroids.row(word), data.row(row), static_cast<std::size_t>(data.words) * 8);
                continue;
            }
            const std::int64_t* counts = &bit_counts[word * static_cast<std::size_t>(bits)];
            for (int b = 0; b < data.bytes; ++b) {
                std::uint8_t byte = 0;
                for (int bit = 0; bit < 8; ++bit) {
                    // members.mean(axis=0) >= 0.5, i.e. a majority vote.
                    if (2 * counts[b * 8 + bit] >= members[word]) {
                        byte = static_cast<std::uint8_t>(byte | (1u << (7 - bit)));
                    }
                }
                centre[b] = byte;
            }
        }
    }

    cv::Mat out(static_cast<int>(k), data.bytes, CV_8U);
    for (std::size_t word = 0; word < k; ++word) {
        std::memcpy(out.ptr<std::uint8_t>(static_cast<int>(word)), centroids.row(word),
                    static_cast<std::size_t>(data.bytes));
    }
    return out;
}

}  // namespace

std::vector<std::uint8_t> unpack_bits(const std::uint8_t* row, int bytes) {
    std::vector<std::uint8_t> bits(static_cast<std::size_t>(bytes) * 8);
    for (int b = 0; b < bytes; ++b) {
        for (int bit = 0; bit < 8; ++bit) {
            bits[static_cast<std::size_t>(b * 8 + bit)] = (row[b] >> (7 - bit)) & 1;
        }
    }
    return bits;
}

// --------------------------------------------------------------------------- //
// Sampler
// --------------------------------------------------------------------------- //

std::uint64_t Sampler::integer(std::uint64_t n) {
    if (n <= 1) {
        return 0;
    }
    // Rejection sampling: accept draws below the largest multiple of n.
    const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max() -
                                (std::numeric_limits<std::uint64_t>::max() % n);
    std::uint64_t draw = engine_();
    while (draw >= limit) {
        draw = engine_();
    }
    return draw % n;
}

double Sampler::uniform() { return static_cast<double>(engine_() >> 11) * 0x1.0p-53; }

std::size_t Sampler::weighted(const std::vector<double>& weights, double total) {
    // Inverse CDF, like numpy's choice(p=...): first index whose cumulative
    // weight exceeds the uniform draw.
    const double target = uniform() * total;
    double cumulative = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        cumulative += weights[i];
        if (target < cumulative) {
            return i;
        }
    }
    // Rounding can leave the target at the very top; return the last
    // positive-weight index.
    for (std::size_t i = weights.size(); i-- > 0;) {
        if (weights[i] > 0) {
            return i;
        }
    }
    return weights.size() - 1;
}

std::vector<std::size_t> Sampler::sample_without_replacement(std::size_t n, std::size_t size) {
    std::vector<std::size_t> pool(n);
    for (std::size_t i = 0; i < n; ++i) {
        pool[i] = i;
    }
    size = std::min(size, n);
    for (std::size_t i = 0; i < size; ++i) {
        const std::size_t j = i + static_cast<std::size_t>(integer(n - i));
        std::swap(pool[i], pool[j]);
    }
    pool.resize(size);
    return pool;
}

// --------------------------------------------------------------------------- //
// KeyframeSelector
// --------------------------------------------------------------------------- //

KeyframeSelector::KeyframeSelector(std::int64_t every_n_frames, double min_translation_m,
                                   double min_rotation_deg)
    : every_n_frames_(std::max<std::int64_t>(1, every_n_frames)),
      min_translation_m_(min_translation_m),
      min_rotation_deg_(min_rotation_deg) {}

KeyframeSelector KeyframeSelector::from_config(const Config& config) {
    const auto& k = config.keyframes;
    return KeyframeSelector(k.every_n_frames, k.min_translation_m, k.min_rotation_deg);
}

bool KeyframeSelector::should_select(std::int64_t index, const Eigen::Matrix4d& pose) const {
    if (!last_index_) {
        return true;
    }
    if (index - *last_index_ >= every_n_frames_) {
        return true;
    }
    if (last_pose_) {
        const Eigen::Vector3d d = pose.topRightCorner<3, 1>() - last_pose_->topRightCorner<3, 1>();
        const double translation = std::sqrt(d(0) * d(0) + d(1) * d(1) + d(2) * d(2));
        if (translation >= min_translation_m_) {
            return true;
        }
        const double rotation = rotation_angle_deg(
            Eigen::Matrix3d(last_pose_->topLeftCorner<3, 3>().transpose() * pose.topLeftCorner<3, 3>()));
        if (rotation >= min_rotation_deg_) {
            return true;
        }
    }
    return false;
}

void KeyframeSelector::accept(std::int64_t index, const Eigen::Matrix4d& pose) {
    last_index_ = index;
    last_pose_ = pose;
}

void KeyframeSelector::reset() {
    last_index_.reset();
    last_pose_.reset();
}

// --------------------------------------------------------------------------- //
// VisualVocabulary
// --------------------------------------------------------------------------- //

VisualVocabulary::VisualVocabulary(cv::Mat centroids, bool binary)
    : centroids_(std::move(centroids)), binary_(binary) {}

VisualVocabulary VisualVocabulary::train(const cv::Mat& descriptors, std::int64_t vocabulary_size,
                                         int iterations, std::int64_t seed) {
    if (descriptors.rows == 0) {
        throw ValueError("Cannot train a vocabulary from zero descriptors");
    }
    const std::int64_t k = std::min<std::int64_t>(vocabulary_size, descriptors.rows);
    if (k < 2) {
        throw ValueError("Vocabulary needs at least 2 words, got " + std::to_string(k));
    }
    if (descriptors.depth() == CV_8U) {
        return VisualVocabulary(binary_kmeans(descriptors, static_cast<std::size_t>(k), iterations, seed),
                                true);
    }
    cv::Mat data;
    descriptors.convertTo(data, CV_32F);
    cv::Mat labels;
    cv::Mat centroids;
    cv::kmeans(data, static_cast<int>(k), labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, iterations, 1.0),
               3, cv::KMEANS_PP_CENTERS, centroids);
    centroids.convertTo(centroids, CV_32F);
    return VisualVocabulary(centroids, false);
}

std::vector<std::int64_t> VisualVocabulary::quantize(const cv::Mat& descriptors) const {
    std::vector<std::int64_t> words(static_cast<std::size_t>(descriptors.rows));
    if (descriptors.rows == 0) {
        return words;
    }
    if (binary_) {
        const Packed data = pack(descriptors);
        const Packed centroids = pack(centroids_);
        for (std::size_t j = 0; j < data.rows; ++j) {
            words[j] = static_cast<std::int64_t>(nearest(data.row(j), centroids));
        }
        return words;
    }
    // Squared Euclidean distances in float32, in the Python expression's
    // order: |x|^2 - 2 x.c + |c|^2.
    cv::Mat data;
    descriptors.convertTo(data, CV_32F);
    std::vector<float> centroid_sq(size());
    for (int c = 0; c < centroids_.rows; ++c) {
        float s = 0.0f;
        for (int d = 0; d < centroids_.cols; ++d) {
            const float v = centroids_.at<float>(c, d);
            s += v * v;
        }
        centroid_sq[static_cast<std::size_t>(c)] = s;
    }
    for (int j = 0; j < data.rows; ++j) {
        const float* x = data.ptr<float>(j);
        float x_sq = 0.0f;
        for (int d = 0; d < data.cols; ++d) {
            x_sq += x[d] * x[d];
        }
        float best = std::numeric_limits<float>::infinity();
        std::int64_t best_word = 0;
        for (int c = 0; c < centroids_.rows; ++c) {
            const float* cr = centroids_.ptr<float>(c);
            float dot = 0.0f;
            for (int d = 0; d < data.cols; ++d) {
                dot += x[d] * cr[d];
            }
            const float distance = x_sq - 2.0f * dot + centroid_sq[static_cast<std::size_t>(c)];
            if (distance < best) {
                best = distance;
                best_word = c;
            }
        }
        words[static_cast<std::size_t>(j)] = best_word;
    }
    return words;
}

Eigen::VectorXd VisualVocabulary::histogram(const cv::Mat& descriptors) const {
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(size()));
    for (std::int64_t word : quantize(descriptors)) {
        counts(static_cast<Eigen::Index>(word)) += 1.0;
    }
    return counts;
}

// --------------------------------------------------------------------------- //
// KeyframeDatabase
// --------------------------------------------------------------------------- //

void KeyframeDatabase::set_vocabulary(VisualVocabulary vocabulary) {
    vocabulary_ = std::move(vocabulary);
    dirty_ = true;
}

Keyframe& KeyframeDatabase::add(Keyframe keyframe) {
    keyframe.keyframe_id = static_cast<std::int64_t>(keyframes_.size());
    keyframes_.push_back(std::move(keyframe));
    dirty_ = true;
    return keyframes_.back();
}

void KeyframeDatabase::build_index() {
    if (!vocabulary_) {
        throw RuntimeError("A vocabulary must be set before building the index");
    }
    const auto words = static_cast<Eigen::Index>(vocabulary_->size());
    if (keyframes_.empty()) {
        bow_matrix_ = BowMatrix::Zero(0, words);
        has_matrix_ = true;
        dirty_ = false;
        return;
    }
    const auto n_documents = static_cast<Eigen::Index>(keyframes_.size());
    BowMatrix counts(n_documents, words);
    for (Eigen::Index i = 0; i < n_documents; ++i) {
        counts.row(i) = vocabulary_->histogram(keyframes_[static_cast<std::size_t>(i)].descriptors).transpose();
    }

    // Document frequency: how many keyframes contain each word at all.
    document_frequency_ = Eigen::VectorXd::Zero(words);
    for (Eigen::Index i = 0; i < n_documents; ++i) {
        for (Eigen::Index w = 0; w < words; ++w) {
            document_frequency_(w) += counts(i, w) > 0 ? 1.0 : 0.0;
        }
    }
    // +1 inside the log guards against a word that appears nowhere.
    Eigen::VectorXd idf(words);
    for (Eigen::Index w = 0; w < words; ++w) {
        idf(w) = std::log(static_cast<double>(n_documents) / std::max(document_frequency_(w), 1.0)) + 1e-6;
    }

    bow_matrix_.resize(n_documents, words);
    std::vector<double> row(static_cast<std::size_t>(words));
    for (Eigen::Index i = 0; i < n_documents; ++i) {
        // Term frequency, normalised per keyframe so that keyframes with more
        // detected features are not systematically scored higher.
        const double total = std::max(np::sum(counts.row(i).data(), static_cast<std::size_t>(words)), 1.0);
        for (Eigen::Index w = 0; w < words; ++w) {
            row[static_cast<std::size_t>(w)] = (counts(i, w) / total) * idf(w);
        }
        std::vector<double> squared(row.size());
        for (std::size_t w = 0; w < row.size(); ++w) {
            squared[w] = row[w] * row[w];
        }
        const double norm = std::max(std::sqrt(np::sum(squared)), 1e-12);
        for (Eigen::Index w = 0; w < words; ++w) {
            bow_matrix_(i, w) = row[static_cast<std::size_t>(w)] / norm;
        }
        keyframes_[static_cast<std::size_t>(i)].bow = bow_matrix_.row(i).transpose();
    }
    has_matrix_ = true;
    dirty_ = false;

    double total_descriptors = 0.0;
    for (const auto& kf : keyframes_) {
        total_descriptors += static_cast<double>(kf.size());
    }
    logger().info(py::sprintf("Built BoW index: %lld keyframes, %zu words, mean %lld descriptors per keyframe",
                              static_cast<long long>(n_documents), vocabulary_->size(),
                              static_cast<long long>(total_descriptors / static_cast<double>(n_documents))));
}

const BowMatrix& KeyframeDatabase::bow_matrix() {
    if (dirty_ || !has_matrix_) {
        build_index();
    }
    return bow_matrix_;
}

std::vector<double> KeyframeDatabase::similarity_scores(std::int64_t query_id) {
    const BowMatrix& matrix = bow_matrix();
    if (!(0 <= query_id && query_id < matrix.rows())) {
        throw IndexError("query_id " + std::to_string(query_id) + " out of range (0.." +
                         std::to_string(matrix.rows() - 1) + ")");
    }
    std::vector<double> scores(static_cast<std::size_t>(matrix.rows()));
    const double* q = matrix.row(query_id).data();
    const Eigen::Index words = matrix.cols();
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
        const double* r = matrix.row(i).data();
        double dot = 0.0;
        for (Eigen::Index w = 0; w < words; ++w) {
            dot += r[w] * q[w];
        }
        scores[static_cast<std::size_t>(i)] = std::min(1.0, std::max(-1.0, dot));
    }
    return scores;
}

std::vector<std::pair<std::int64_t, double>> KeyframeDatabase::query(std::int64_t query_id,
                                                                     std::int64_t min_keyframe_separation,
                                                                     double min_path_separation_m,
                                                                     std::int64_t top_k,
                                                                     double min_similarity) {
    if (keyframes_.empty()) {
        return {};
    }
    const std::vector<double> scores = similarity_scores(query_id);
    const Keyframe& q = keyframes_[static_cast<std::size_t>(query_id)];

    std::vector<std::int64_t> candidate_ids;
    for (std::int64_t candidate_id = 0; candidate_id < query_id - min_keyframe_separation + 1;
         ++candidate_id) {
        const Keyframe& candidate = keyframes_.at(static_cast<std::size_t>(candidate_id));
        const double travelled = q.path_distance_m - candidate.path_distance_m;
        if (travelled >= min_path_separation_m &&
            scores[static_cast<std::size_t>(candidate_id)] >= min_similarity) {
            candidate_ids.push_back(candidate_id);
        }
    }
    if (candidate_ids.empty()) {
        return {};
    }
    std::vector<double> negated;
    negated.reserve(candidate_ids.size());
    for (std::int64_t id : candidate_ids) {
        negated.push_back(-scores[static_cast<std::size_t>(id)]);
    }
    // np.argsort (quicksort) so ties are ordered exactly as in Python.
    const std::vector<std::int64_t> order = np::argsort(negated);
    std::vector<std::pair<std::int64_t, double>> out;
    for (std::size_t k = 0; k < order.size() && static_cast<std::int64_t>(k) < top_k; ++k) {
        const std::int64_t id = candidate_ids[static_cast<std::size_t>(order[k])];
        out.emplace_back(id, scores[static_cast<std::size_t>(id)]);
    }
    return out;
}

double KeyframeDatabase::neighbour_score(std::int64_t query_id, std::int64_t window) {
    const std::vector<double> scores = similarity_scores(query_id);
    const std::int64_t low = std::max<std::int64_t>(0, query_id - window);
    const auto n = static_cast<std::int64_t>(scores.size());
    double best = -std::numeric_limits<double>::infinity();
    bool any = false;
    auto consider = [&](std::int64_t i) {
        const double v = scores[static_cast<std::size_t>(i)];
        if (std::isfinite(v)) {
            best = any ? std::max(best, v) : v;
            any = true;
        }
    };
    for (std::int64_t i = low; i < query_id; ++i) {
        consider(i);
    }
    for (std::int64_t i = query_id + 1; i < std::min(n, query_id + 1 + window); ++i) {
        consider(i);
    }
    return any ? best : 0.0;
}

Json KeyframeDatabase::describe() const {
    Json out = Json::object();
    out["n_keyframes"] = static_cast<std::int64_t>(keyframes_.size());
    out["vocabulary_size"] = vocabulary_ ? Json(static_cast<std::int64_t>(vocabulary_->size())) : Json(nullptr);
    out["binary_vocabulary"] = vocabulary_ ? Json(vocabulary_->binary()) : Json(nullptr);
    return out;
}

}  // namespace mvslam
