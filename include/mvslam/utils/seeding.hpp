// Deterministic seeding.
//
// RANSAC inside OpenCV draws from its own RNG, so reproducing a run
// byte-for-byte requires seeding OpenCV in addition to the project's own
// generator. The Python original seeds ``random``, NumPy and ``cv2``; the
// C++ counterparts are a process-wide ``std::mt19937_64`` and
// ``cv::setRNGSeed``. Note that std::mt19937_64 streams are not NumPy's
// PCG64 streams: anything drawn from them (vocabulary seeding, descriptor
// subsampling) matches the Python run statistically, not bit for bit.
#pragma once

#include <cstdint>
#include <random>

namespace mvslam {

constexpr std::int64_t kDefaultSeed = 0;

/// Seed the global generator and OpenCV's RNG; returns the seed so callers
/// can record it in run metadata.
std::int64_t set_global_seed(std::int64_t seed = kDefaultSeed);

/// The process-wide generator seeded by set_global_seed().
std::mt19937_64& global_rng();

}  // namespace mvslam
