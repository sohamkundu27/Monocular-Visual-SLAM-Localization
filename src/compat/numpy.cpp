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

namespace {

// npysort/npysort_common.h: DOUBLE_LT, which orders NaNs after everything.
inline bool double_less(double a, double b) { return a < b || (b != b && a == a); }

// npysort/heapsort.cpp: aheapsort_<npy::double_tag>
void aheapsort(const double* v, std::int64_t* tosort, std::int64_t n) {
    std::int64_t* a = tosort - 1;  // 1-based indexing, as in NumPy
    std::int64_t i, j, l, tmp;
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            if (j < n && double_less(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            if (double_less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && double_less(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            if (double_less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[i] = tmp;
    }
}

int get_msb(std::uint64_t unum) {
    int depth = 0;
    while (unum >>= 1) {
        depth++;
    }
    return depth;
}

}  // namespace

std::vector<std::int64_t> argsort(const std::vector<double>& values) {
    // npysort/quicksort.cpp: aquicksort_<npy::double_tag>
    constexpr std::int64_t kSmallQuicksort = 15;
    constexpr int kStack = 128;
    const auto num = static_cast<std::int64_t>(values.size());
    std::vector<std::int64_t> order(values.size());
    for (std::int64_t i = 0; i < num; ++i) {
        order[static_cast<std::size_t>(i)] = i;
    }
    if (num < 2) {
        return order;
    }
    const double* v = values.data();
    std::int64_t* tosort = order.data();
    std::int64_t* pl = tosort;
    std::int64_t* pr = tosort + num - 1;
    std::int64_t* stack[kStack];
    std::int64_t** sptr = stack;
    std::int64_t *pm, *pi, *pj, *pk;
    std::int64_t vi;
    double vp;
    int depth[kStack];
    int* psdepth = depth;
    int cdepth = get_msb(static_cast<std::uint64_t>(num)) * 2;

    for (;;) {
        if (cdepth < 0) {
            aheapsort(v, pl, pr - pl + 1);
            goto stack_pop;
        }
        while ((pr - pl) > kSmallQuicksort) {
            pm = pl + ((pr - pl) >> 1);
            if (double_less(v[*pm], v[*pl])) std::swap(*pm, *pl);
            if (double_less(v[*pr], v[*pm])) std::swap(*pr, *pm);
            if (double_less(v[*pm], v[*pl])) std::swap(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            std::swap(*pm, *pj);
            for (;;) {
                do {
                    ++pi;
                } while (double_less(v[*pi], vp));
                do {
                    --pj;
                } while (double_less(vp, v[*pj]));
                if (pi >= pj) {
                    break;
                }
                std::swap(*pi, *pj);
            }
            pk = pr - 1;
            std::swap(*pi, *pk);
            // push largest partition on stack
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }
        // insertion sort
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && double_less(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
    return order;
}

std::size_t argmin(const double* values, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (std::isnan(values[best])) {
            return best;
        }
        if (values[i] < values[best] || std::isnan(values[i])) {
            best = i;
        }
    }
    return best;
}

double degrees(double radians) { return radians * (180.0 / M_PI); }

double radians(double degrees) { return degrees * (M_PI / 180.0); }

}  // namespace mvslam::np
