#include "support/golden.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace golden {

namespace {

std::vector<unsigned char> b64decode(const std::string& text) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> lookup(256, -1);
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        lookup[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }
    std::vector<unsigned char> out;
    out.reserve(text.size() * 3 / 4);
    int value = 0;
    int bits = -8;
    for (unsigned char c : text) {
        if (lookup[c] < 0) {
            continue;  // '=' padding
        }
        value = (value << 6) + lookup[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<unsigned char>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

std::size_t itemsize(const std::string& dtype) {
    if (dtype == "f8" || dtype == "i8") {
        return 8;
    }
    if (dtype == "f4" || dtype == "i4") {
        return 4;
    }
    return 1;
}

template <typename T>
T read_le(const unsigned char* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));  // x86-64 is little-endian, like the fixtures
    return v;
}

}  // namespace

std::string dir() { return MVSLAM_GOLDEN_DIR; }

const Json& load(const std::string& relative) {
    static std::mutex mutex;
    static std::map<std::string, std::unique_ptr<Json>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(relative);
    if (it != cache.end()) {
        return *it->second;
    }
    std::ifstream in(dir() + "/" + relative, std::ios::binary);
    if (!in) {
        throw std::runtime_error("golden fixture missing: " + relative);
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    auto parsed = std::make_unique<Json>(mvslam::py::loads(buffer.str()));
    const Json& ref = *parsed;
    cache.emplace(relative, std::move(parsed));
    return ref;
}

std::size_t Array::size() const {
    std::size_t n = 1;
    for (auto d : shape) {
        n *= static_cast<std::size_t>(d);
    }
    return n;
}

std::vector<double> Array::as_doubles() const {
    std::vector<double> out(size());
    const std::size_t step = itemsize(dtype);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const unsigned char* p = bytes.data() + i * step;
        if (dtype == "f8") {
            out[i] = read_le<double>(p);
        } else if (dtype == "f4") {
            out[i] = read_le<float>(p);
        } else if (dtype == "i8") {
            out[i] = static_cast<double>(read_le<std::int64_t>(p));
        } else if (dtype == "i4") {
            out[i] = read_le<std::int32_t>(p);
        } else {
            out[i] = *p;
        }
    }
    return out;
}

std::vector<std::int64_t> Array::as_ints() const {
    std::vector<std::int64_t> out(size());
    const std::size_t step = itemsize(dtype);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const unsigned char* p = bytes.data() + i * step;
        if (dtype == "i8") {
            out[i] = read_le<std::int64_t>(p);
        } else if (dtype == "i4") {
            out[i] = read_le<std::int32_t>(p);
        } else if (dtype == "u1" || dtype == "b1") {
            out[i] = *p;
        } else if (dtype == "f8") {
            out[i] = static_cast<std::int64_t>(read_le<double>(p));
        } else {
            out[i] = static_cast<std::int64_t>(read_le<float>(p));
        }
    }
    return out;
}

double Array::scalar() const {
    if (size() != 1) {
        throw std::runtime_error("golden array is not a scalar");
    }
    return as_doubles()[0];
}

Array decode(const Json& encoded, const Json* shared) {
    if (encoded.is_object() && encoded.contains("ref")) {
        if (shared == nullptr) {
            throw std::runtime_error("golden ref without a shared table");
        }
        return decode(shared->at(encoded.at("ref").get<std::string>()), shared);
    }
    Array out;
    out.dtype = encoded.at("dtype").get<std::string>();
    for (const auto& d : encoded.at("shape")) {
        out.shape.push_back(d.get<std::int64_t>());
    }
    if (encoded.contains("v")) {
        const Json& v = encoded.at("v");
        const std::size_t step = itemsize(out.dtype);
        out.bytes.resize(step);
        if (out.dtype == "f8") {
            const double d = mvslam::py::as_double(v);
            std::memcpy(out.bytes.data(), &d, 8);
        } else if (out.dtype == "f4") {
            const float f = static_cast<float>(mvslam::py::as_double(v));
            std::memcpy(out.bytes.data(), &f, 4);
        } else if (out.dtype == "i8") {
            const std::int64_t i = v.is_boolean() ? v.get<bool>() : v.get<std::int64_t>();
            std::memcpy(out.bytes.data(), &i, 8);
        } else if (out.dtype == "i4") {
            const std::int32_t i = static_cast<std::int32_t>(v.get<std::int64_t>());
            std::memcpy(out.bytes.data(), &i, 4);
        } else {
            out.bytes[0] = static_cast<unsigned char>(v.is_boolean() ? v.get<bool>()
                                                                     : v.get<std::int64_t>());
        }
        return out;
    }
    out.bytes = b64decode(encoded.at("b64").get<std::string>());
    if (out.bytes.size() != out.size() * itemsize(out.dtype)) {
        throw std::runtime_error("golden array payload has the wrong size");
    }
    return out;
}

std::vector<double> vec(const Json& encoded, const Json* shared) {
    return decode(encoded, shared).as_doubles();
}

std::vector<std::int64_t> ivec(const Json& encoded, const Json* shared) {
    return decode(encoded, shared).as_ints();
}

Eigen::MatrixXd mat(const Json& encoded, const Json* shared) {
    const Array a = decode(encoded, shared);
    if (a.shape.size() != 2) {
        throw std::runtime_error("golden array is not 2-D");
    }
    const auto values = a.as_doubles();
    Eigen::MatrixXd out(a.shape[0], a.shape[1]);
    for (std::int64_t r = 0; r < a.shape[0]; ++r) {
        for (std::int64_t c = 0; c < a.shape[1]; ++c) {
            out(r, c) = values[static_cast<std::size_t>(r * a.shape[1] + c)];  // row-major source
        }
    }
    return out;
}

Eigen::Matrix3d mat3(const Json& encoded) {
    const Eigen::MatrixXd m = mat(encoded);
    if (m.rows() != 3 || m.cols() != 3) {
        throw std::runtime_error("golden array is not 3x3");
    }
    return m;
}

Eigen::Matrix4d mat4(const Json& encoded) {
    const Eigen::MatrixXd m = mat(encoded);
    if (m.rows() != 4 || m.cols() != 4) {
        throw std::runtime_error("golden array is not 4x4");
    }
    return m;
}

mvslam::PoseList poses(const Json& encoded, const Json* shared) {
    const Array a = decode(encoded, shared);
    if (a.shape.size() != 3 || a.shape[1] != 4 || a.shape[2] != 4) {
        throw std::runtime_error("golden array is not (N, 4, 4)");
    }
    const auto values = a.as_doubles();
    mvslam::PoseList out(static_cast<std::size_t>(a.shape[0]));
    for (std::size_t n = 0; n < out.size(); ++n) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                out[n](r, c) = values[n * 16 + static_cast<std::size_t>(r * 4 + c)];
            }
        }
    }
    return out;
}

mvslam::Points2 points2(const Json& encoded, const Json* shared) {
    const Array a = decode(encoded, shared);
    if (a.shape.size() != 2 || a.shape[1] != 2) {
        throw std::runtime_error("golden array is not (N, 2)");
    }
    const auto values = a.as_doubles();
    mvslam::Points2 out(a.shape[0], 2);
    std::memcpy(out.data(), values.data(), values.size() * sizeof(double));  // both row-major
    return out;
}

mvslam::Points3 points3(const Json& encoded, const Json* shared) {
    const Array a = decode(encoded, shared);
    if (a.shape.size() != 2 || a.shape[1] != 3) {
        throw std::runtime_error("golden array is not (N, 3)");
    }
    const auto values = a.as_doubles();
    mvslam::Points3 out(a.shape[0], 3);
    std::memcpy(out.data(), values.data(), values.size() * sizeof(double));
    return out;
}

mvslam::Mask mask(const Json& encoded) {
    const auto values = decode(encoded).as_ints();
    mvslam::Mask out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = values[i] != 0 ? 1 : 0;
    }
    return out;
}

cv::Mat cvmat(const Json& encoded, const Json* shared) {
    const Array a = decode(encoded, shared);
    if (a.shape.size() != 2) {
        throw std::runtime_error("golden array is not 2-D");
    }
    const int rows = static_cast<int>(a.shape[0]);
    const int cols = static_cast<int>(a.shape[1]);
    int type = CV_8U;
    if (a.dtype == "f4") {
        type = CV_32F;
    } else if (a.dtype == "f8") {
        type = CV_64F;
    } else if (a.dtype == "i4") {
        type = CV_32S;
    } else if (a.dtype != "u1" && a.dtype != "b1") {
        throw std::runtime_error("unsupported dtype for cv::Mat: " + a.dtype);
    }
    cv::Mat out(rows, cols, type);
    if (!a.bytes.empty()) {
        std::memcpy(out.data, a.bytes.data(), a.bytes.size());  // row-major both sides
    }
    return out;
}

double from_hex(const std::string& text) {
    if (text == "nan") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (text == "inf") {
        return std::numeric_limits<double>::infinity();
    }
    if (text == "-inf") {
        return -std::numeric_limits<double>::infinity();
    }
    return std::strtod(text.c_str(), nullptr);  // strtod parses C99 hex floats
}

double from_hex(const Json& value) { return from_hex(value.get<std::string>()); }

// --------------------------------------------------------------------------- //
// Stream
// --------------------------------------------------------------------------- //

Stream::Stream(const std::string& module, const std::string& node)
    : id_(module + ".py::" + node) {
    const std::string relative = "testdata/" + module + ".json";
    const Json& data = load(relative);
    shared_ = &data.at("shared");
    const Json& streams = data.at("streams");
    if (streams.contains(id_)) {
        for (const auto& record : streams.at(id_)) {
            records_.push_back(record);
        }
    }
}

Stream::~Stream() {
    if (position_ != records_.size()) {
        ADD_FAILURE() << "golden stream " << id_ << " not fully consumed: " << position_ << " of "
                      << records_.size() << " records used";
    }
}

const Json& Stream::next(const std::string& fn) {
    if (position_ >= records_.size()) {
        throw std::runtime_error("golden stream " + id_ + " exhausted (wanted " + fn + ")");
    }
    const Json& record = records_[position_++];
    const std::string recorded = record.at("fn").get<std::string>();
    if (recorded != fn) {
        throw std::runtime_error("golden stream " + id_ + " record " +
                                 std::to_string(position_ - 1) + " is " + recorded + ", not " + fn);
    }
    return record.at("value");
}

Array Stream::next_array(const std::string& fn) { return decode(next(fn), shared_); }

std::vector<double> Stream::next_vec(const std::string& fn) { return next_array(fn).as_doubles(); }

Eigen::MatrixXd Stream::next_mat(const std::string& fn) { return mat(next(fn), shared_); }

Eigen::Matrix4d Stream::next_mat4(const std::string& fn) {
    const Eigen::MatrixXd m = next_mat(fn);
    if (m.rows() != 4 || m.cols() != 4) {
        throw std::runtime_error("golden stream value is not 4x4");
    }
    return m;
}

double Stream::next_scalar(const std::string& fn) { return next_array(fn).scalar(); }

std::int64_t Stream::next_int(const std::string& fn) {
    const auto values = next_array(fn).as_ints();
    if (values.size() != 1) {
        throw std::runtime_error("golden stream value is not a scalar");
    }
    return values[0];
}

cv::Mat Stream::next_cvmat(const std::string& fn) { return cvmat(next(fn), shared_); }

}  // namespace golden
