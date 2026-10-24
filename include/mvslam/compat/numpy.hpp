// NumPy-compatible reductions and order statistics.
//
// Metrics such as ATE RMSE are written to metrics.json after rounding, so the
// last bits of a sum can decide the printed digit. These functions therefore
// reproduce NumPy 1.26's float64 algorithms exactly rather than approximately:
//
// * ``sum``: the add-reduce inner loop is called on blocks of at most 8192
//   elements (the ufunc buffer size), and each block is summed with NumPy's
//   pairwise algorithm (8 accumulators below 128 elements, recursive halving
//   above). The block results are accumulated left to right.
// * ``mean`` = sum / n, ``std`` = sqrt(mean((x - mean)^2)) (population, ddof 0).
// * ``median`` averages the two middle values for even n and propagates NaN.
// * ``percentile`` is the default "linear" method (Hyndman & Fan type 7) with
//   NumPy's two-sided lerp.
//
// Each was checked bit-for-bit against NumPy (tests/golden/compat.json).
#pragma once

#include <cstddef>
#include <vector>

namespace mvslam::np {

/// ``np.sum`` of a contiguous float64 array.
double sum(const double* data, std::size_t n);
double sum(const std::vector<double>& values);

/// ``np.mean``; NaN for an empty array (NumPy warns and returns NaN).
double mean(const std::vector<double>& values);

/// ``np.std`` with ddof=0.
double std_dev(const std::vector<double>& values);

/// ``np.median``: NaN-propagating, NaN for an empty array.
double median(std::vector<double> values);

/// ``np.percentile(values, q)`` with the default linear method, ``q`` in [0, 100].
double percentile(std::vector<double> values, double q);

/// ``np.min`` / ``np.max``: NaN-propagating. Throw ValueError on empty input,
/// like NumPy ("zero-size array to reduction operation").
double min(const std::vector<double>& values);
double max(const std::vector<double>& values);

/// ``np.sqrt(np.mean(values ** 2))``.
double rms(const std::vector<double>& values);

/// ``np.degrees`` / ``np.radians``.
double degrees(double radians);
double radians(double degrees);

}  // namespace mvslam::np
