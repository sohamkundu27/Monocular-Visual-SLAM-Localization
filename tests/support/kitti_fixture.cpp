#include "support/kitti_fixture.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "mvslam/compat/numpy.hpp"
#include "mvslam/compat/pyformat.hpp"
#include "mvslam/geometry/transforms.hpp"
#include "support/golden.hpp"

using mvslam::Json;
#include "support/test_utils.hpp"

namespace kitti_fixture {

// The long lines are the real file format and must not be wrapped.
const char* const kCalibText =
    "P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n"
    "P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00\n"
    "P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 4.538225000000e+01 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 -1.130887000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.779761000000e-03\n"
    "P3: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.372877000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 2.369057000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.915215000000e-03\n"
    "Tr: 4.276802385584e-04 -9.999672484946e-01 -8.084491683471e-03 -1.198459927713e-02 -7.210626507497e-03 8.081198471645e-03 -9.999413164504e-01 -5.403984729748e-02 9.999738645903e-01 4.859485810390e-04 -7.206933692422e-03 -2.921968648686e-01\n";

namespace {

// Python's % is a floored modulo: the result has the sign of the divisor.
int py_mod(int a, int b) { return ((a % b) + b) % b; }

}  // namespace

cv::Mat make_textured_image(int index, int width, int height, int seed) {
    static const std::string kFixture = "textured_images.json";
    const Json& fixture = golden::load(kFixture);
    const Json& params = fixture.at("params");
    const std::string key = std::to_string(seed) + "/" + std::to_string(width) + "x" +
                            std::to_string(height);
    if (!params.contains(key)) {
        throw std::runtime_error("no recorded drawing parameters for textured image " + key +
                                 "; re-run tools/export_golden.py");
    }
    const auto& layout = params.at(key);
    cv::Mat image(height, width, CV_8UC1, cv::Scalar(110));
    const int shift = index * 3;
    for (const auto& circle : layout.at("circles")) {
        const int cx = circle.at(0).get<int>();
        const int cy = circle.at(1).get<int>();
        const int radius = circle.at(2).get<int>();
        const int shade = circle.at(3).get<int>();
        cv::circle(image, cv::Point(py_mod(cx - shift, width), cy), radius, cv::Scalar(shade), -1);
    }
    for (const auto& rect : layout.at("rects")) {
        const int x0 = rect.at(0).get<int>();
        const int y0 = rect.at(1).get<int>();
        const int shade = rect.at(2).get<int>();
        cv::rectangle(image, cv::Point(py_mod(x0 - shift, width), y0),
                      cv::Point(py_mod(x0 - shift + 25, width), std::min(y0 + 18, height - 1)),
                      cv::Scalar(shade), -1);
    }
    return image;
}

std::vector<mvslam::KittiRow> ground_truth_rows(int n_frames, double step_m, double yaw_rate_deg) {
    std::vector<mvslam::KittiRow> rows;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const double yaw = mvslam::np::radians(yaw_rate_deg);
    Eigen::Matrix3d R_step;
    R_step << std::cos(yaw), 0.0, std::sin(yaw), 0.0, 1.0, 0.0, -std::sin(yaw), 0.0, std::cos(yaw);
    for (int i = 0; i < n_frames; ++i) {
        rows.push_back(mvslam::se3_to_kitti_row(T));
        T = T * mvslam::se3_from_rt(R_step, Eigen::Vector3d(0.0, 0.0, step_m));
    }
    return rows;
}

std::filesystem::path write_kitti_sequence(const std::filesystem::path& root,
                                           const std::string& sequence, int n_frames,
                                           bool with_ground_truth, bool with_times,
                                           const std::string& camera, double step_m) {
    const auto sequence_dir = root / "sequences" / sequence;
    const auto image_dir = sequence_dir / camera;
    std::filesystem::create_directories(image_dir);
    for (int i = 0; i < n_frames; ++i) {
        cv::imwrite((image_dir / mvslam::py::sprintf("%06d.png", i)).string(), make_textured_image(i));
    }
    testing_utils::write_text(sequence_dir / "calib.txt", kCalibText);
    if (with_times) {
        std::string text;
        for (int i = 0; i < n_frames; ++i) {
            const double t = i * 0.1037;
            text += mvslam::py::savetxt_row(&t, 1, "%.6e");
        }
        testing_utils::write_text(sequence_dir / "times.txt", text);
    }
    if (with_ground_truth) {
        std::string text;
        for (const auto& row : ground_truth_rows(n_frames, step_m)) {
            text += mvslam::py::savetxt_row(row.data(), row.size(), "%.12e");
        }
        testing_utils::write_text(root / "poses" / (sequence + ".txt"), text);
    }
    return root;
}

// --------------------------------------------------------------------------- //
// MD5 (RFC 1321), for comparing rendered images with the Python checksums.
// --------------------------------------------------------------------------- //

std::string md5_hex(const unsigned char* data, std::size_t size) {
    static const std::uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391};
    static const int S[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                              5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                              4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                              6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    std::vector<unsigned char> message(data, data + size);
    const std::uint64_t bit_length = static_cast<std::uint64_t>(size) * 8;
    message.push_back(0x80);
    while (message.size() % 64 != 56) {
        message.push_back(0);
    }
    for (int i = 0; i < 8; ++i) {
        message.push_back(static_cast<unsigned char>(bit_length >> (8 * i)));
    }
    std::uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
    auto rotl = [](std::uint32_t x, int c) { return (x << c) | (x >> (32 - c)); };
    for (std::size_t chunk = 0; chunk < message.size(); chunk += 64) {
        std::uint32_t M[16];
        for (int i = 0; i < 16; ++i) {
            M[i] = static_cast<std::uint32_t>(message[chunk + 4 * i]) |
                   (static_cast<std::uint32_t>(message[chunk + 4 * i + 1]) << 8) |
                   (static_cast<std::uint32_t>(message[chunk + 4 * i + 2]) << 16) |
                   (static_cast<std::uint32_t>(message[chunk + 4 * i + 3]) << 24);
        }
        std::uint32_t A = a0, B = b0, C = c0, D = d0;
        for (int i = 0; i < 64; ++i) {
            std::uint32_t F;
            int g;
            if (i < 16) {
                F = (B & C) | (~B & D);
                g = i;
            } else if (i < 32) {
                F = (D & B) | (~D & C);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                F = B ^ C ^ D;
                g = (3 * i + 5) % 16;
            } else {
                F = C ^ (B | ~D);
                g = (7 * i) % 16;
            }
            F = F + A + K[i] + M[g];
            A = D;
            D = C;
            C = B;
            B = B + rotl(F, S[i]);
        }
        a0 += A;
        b0 += B;
        c0 += C;
        d0 += D;
    }
    std::string out;
    char buffer[3];
    for (std::uint32_t word : {a0, b0, c0, d0}) {
        for (int i = 0; i < 4; ++i) {
            std::snprintf(buffer, sizeof(buffer), "%02x", (word >> (8 * i)) & 0xFF);
            out += buffer;
        }
    }
    return out;
}

}  // namespace kitti_fixture
