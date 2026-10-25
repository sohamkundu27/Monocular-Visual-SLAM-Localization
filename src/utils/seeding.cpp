#include "mvslam/utils/seeding.hpp"

#include <opencv2/core.hpp>

namespace mvslam {

std::mt19937_64& global_rng() {
    static std::mt19937_64 generator(static_cast<std::uint64_t>(kDefaultSeed));
    return generator;
}

std::int64_t set_global_seed(std::int64_t seed) {
    global_rng().seed(static_cast<std::uint64_t>(seed));
    cv::setRNGSeed(static_cast<int>(seed));
    return seed;
}

}  // namespace mvslam
