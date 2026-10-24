#include "mvslam/compat/numpy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "mvslam/errors.hpp"

namespace mvslam::np {

namespace {

constexpr std::size_t kPairwiseBlock = 128;  // PW_BLOCKSIZE in numpy/core/src/umath
constexpr std::size_t kBufferSize = 8192;    // NPY_BUFSIZE: inner-loop chunk length

// numpy/core/src/umath/loops_utils.h.src: DOUBLE_pairwise_sum.
double pairwise_sum(const double* a, std::size_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            res += a[i];
        }
        return res;
    }
    if (n <= kPairwiseBlock) {
        double r[8];
        for (std::size_t j = 0; j < 8; ++j) {
            r[j] = a[j];
        }
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8) {
            for (std::size_t j = 0; j < 8; ++j) {
                r[j] += a[i + j];
            }
        }
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) {
            res += a[i];
        }
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool any_nan(const std::vector<double>& values) {
    return std::any_of(values.begin(), values.end(), [](double v) { return std::isnan(v); });
}

}  // namespace

double sum(const double* data, std::size_t n) {
    // The reduction starts from the additive identity and adds each
    // buffer-sized block's pairwise sum in turn.
    double total = 0.0;
    for (std::size_t start = 0; start < n; start += kBufferSize) {
        total += pairwise_sum(data + start, std::min(kBufferSize, n - start));
    }
    return total;
}

double sum(const std::vector<double>& values) { return sum(values.data(), values.size()); }

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return kNaN;
    }
    return sum(values) / static_cast<double>(values.size());
}

double std_dev(const std::vector<double>& values) {
    if (values.empty()) {
        return kNaN;
    }
    const double m = mean(values);
    std::vector<double> squared(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double d = values[i] - m;
        squared[i] = d * d;
    }
    return std::sqrt(sum(squared) / static_cast<double>(values.size()));
}

double median(std::vector<double> values) {
    if (values.empty() || any_nan(values)) {
        return kNaN;
    }
    const std::size_t n = values.size();
    const std::size_t half = n / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(half),
                     values.end());
    const double upper = values[half];
    if (n % 2 == 1) {
        return upper;
    }
    const double lower =
        *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(half));
    // np.mean of the two middle values: (lower + upper) / 2.
    return (lower + upper) / 2.0;
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        throw IndexError("index -1 is out of bounds for axis 0 with size 0");
    }
    if (any_nan(values)) {
        return kNaN;
    }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    const double quantile = q / 100.0;  // np.true_divide(q, 100)
    const double virtual_index = static_cast<double>(n - 1) * quantile;
    std::size_t previous = 0;
    std::size_t next = 0;
    if (virtual_index >= static_cast<double>(n - 1)) {
        previous = next = n - 1;
    } else if (virtual_index < 0) {
        previous = next = 0;
    } else {
        previous = static_cast<std::size_t>(std::floor(virtual_index));
        next = previous + 1;
    }
    const double gamma = virtual_index - std::floor(virtual_index);
    const double a = values[previous];
    const double b = values[next];
    // numpy.lib.function_base._lerp
    const double diff = b - a;
    if (gamma >= 0.5) {
        return b - diff * (1.0 - gamma);
    }
    return a + diff * gamma;
}

double min(const std::vector<double>& values) {
    if (values.empty()) {
        throw ValueError("zero-size array to reduction operation minimum which has no identity");
    }
    double best = values[0];
    for (double v : values) {
        if (std::isnan(v)) {
            return kNaN;
        }
        best = std::min(best, v);
    }
    return best;
}

double max(const std::vector<double>& values) {
    if (values.empty()) {
        throw ValueError("zero-size array to reduction operation maximum which has no identity");
    }
    double best = values[0];
    for (double v : values) {
        if (std::isnan(v)) {
            return kNaN;
        }
        best = std::max(best, v);
    }
    return best;
}

double rms(const std::vector<double>& values) {
    std::vector<double> squared(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        squared[i] = values[i] * values[i];
    }
    return std::sqrt(mean(squared));
}

double degrees(double radians) { return radians * (180.0 / M_PI); }

double radians(double degrees) { return degrees * (M_PI / 180.0); }

}  // namespace mvslam::np
