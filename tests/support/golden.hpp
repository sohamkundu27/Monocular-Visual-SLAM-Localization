// Access to the golden fixtures exported from the Python implementation.
//
// tests/golden/<module>.json holds reference inputs/outputs of library
// functions; tests/golden/testdata/<test_module>.json holds the random data
// each Python test consumed (see tools/golden_recorder.py). Arrays are
// ``{"dtype", "shape", "b64"}`` (exact little-endian bytes) or, for 0-d
// values, ``{"dtype", "shape": [], "v"}``; ``{"ref": key}`` points into the
// module's shared table. Exactly-compared floats are ``float.hex()`` strings.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "mvslam/compat/pyjson.hpp"
#include "mvslam/geometry/pose.hpp"
#include "mvslam/utils/cv_eigen.hpp"

namespace golden {

using mvslam::Json;

/// Root of the golden fixtures (``tests/golden``).
std::string dir();

/// Parse ``tests/golden/<relative>`` (cached per path).
const Json& load(const std::string& relative);

/// Decoded n-dimensional array.
struct Array {
    std::string dtype;  // f8, f4, i8, i4, u1, b1
    std::vector<std::int64_t> shape;
    std::vector<unsigned char> bytes;

    std::size_t size() const;
    std::vector<double> as_doubles() const;
    std::vector<std::int64_t> as_ints() const;
    double scalar() const;
};

/// Decode an encoded array (resolving ``{"ref": key}`` through ``shared``).
Array decode(const Json& encoded, const Json* shared = nullptr);

/// Convenience decoders with shape checks.
std::vector<double> vec(const Json& encoded, const Json* shared = nullptr);
std::vector<std::int64_t> ivec(const Json& encoded, const Json* shared = nullptr);
Eigen::MatrixXd mat(const Json& encoded, const Json* shared = nullptr);
Eigen::Matrix3d mat3(const Json& encoded);
Eigen::Matrix4d mat4(const Json& encoded);
mvslam::PoseList poses(const Json& encoded, const Json* shared = nullptr);
mvslam::Points2 points2(const Json& encoded, const Json* shared = nullptr);
mvslam::Points3 points3(const Json& encoded, const Json* shared = nullptr);
mvslam::Mask mask(const Json& encoded);
/// uint8 or float32 2-D array as a cv::Mat (descriptors).
cv::Mat cvmat(const Json& encoded, const Json* shared = nullptr);

/// ``float.fromhex`` (accepts "nan", "inf", "-inf").
double from_hex(const std::string& text);
double from_hex(const Json& value);

/// Replays the random data one Python test consumed, in order.
///
/// Every ``next(fn)`` must name the same generator method or helper the
/// Python test called at that point, which keeps the C++ test's data flow an
/// exact mirror of the original. A stream that is not fully consumed fails
/// the test when it is destroyed.
class Stream {
public:
    /// ``module`` is e.g. "test_transforms"; ``node`` the pytest node id
    /// without the module, e.g. "TestAngles::test_angle_is_bounded".
    Stream(const std::string& module, const std::string& node);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    /// The next record's raw value; fails the test if its name differs.
    const Json& next(const std::string& fn);
    Array next_array(const std::string& fn);
    std::vector<double> next_vec(const std::string& fn);
    Eigen::MatrixXd next_mat(const std::string& fn);
    Eigen::Matrix4d next_mat4(const std::string& fn);
    double next_scalar(const std::string& fn);
    std::int64_t next_int(const std::string& fn);
    cv::Mat next_cvmat(const std::string& fn);
    const Json& shared() const { return *shared_; }

    std::size_t remaining() const { return records_.size() - position_; }

private:
    std::string id_;
    std::vector<Json> records_;
    const Json* shared_ = nullptr;
    std::size_t position_ = 0;
};

}  // namespace golden
