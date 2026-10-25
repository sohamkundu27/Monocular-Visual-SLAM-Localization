#include "support/synthetic.hpp"

#include <vector>

#include "mvslam/geometry/transforms.hpp"

namespace synthetic {

using mvslam::Points2;
using mvslam::Points3;

Eigen::Matrix3d kitti_K() {
    Eigen::Matrix3d K;
    K << 718.856, 0.0, 607.1928, 0.0, 718.856, 185.2157, 0.0, 0.0, 1.0;
    return K;
}

Eigen::Matrix4d random_se3(golden::Stream& rng) { return rng.next_mat4("random_se3"); }

Eigen::Matrix4d small_se3(golden::Stream& rng) { return rng.next_mat4("small_se3"); }

Points2 project(const Points3& points_cam, const Eigen::Matrix3d& K) {
    // homogeneous = points_cam @ K.T; uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    Points2 out(points_cam.rows(), 2);
    for (Eigen::Index i = 0; i < points_cam.rows(); ++i) {
        const Eigen::Vector3d h = K * points_cam.row(i).transpose();
        out(i, 0) = h(0) / h(2);
        out(i, 1) = h(1) / h(2);
    }
    return out;
}

Points3 make_scene(golden::Stream& stream) {
    return golden::points3(stream.next("make_scene"), &stream.shared());
}

Correspondences two_view_correspondences(golden::Stream& stream, const Eigen::Matrix4d& T_c1_c2,
                                         const std::optional<Points3>& points_world,
                                         double noise_px,
                                         std::optional<std::pair<int, int>> image_size) {
    const Points3 scene = points_world ? *points_world : make_scene(stream);
    const Eigen::Matrix4d T_c2_c1 = mvslam::invert_se3(T_c1_c2);
    const Eigen::Matrix3d R = T_c2_c1.topLeftCorner<3, 3>();
    const Eigen::Vector3d t = T_c2_c1.topRightCorner<3, 1>();

    std::vector<Eigen::Index> keep;
    Points3 cam2_all(scene.rows(), 3);
    for (Eigen::Index i = 0; i < scene.rows(); ++i) {
        const Eigen::Vector3d p2 = R * scene.row(i).transpose() + t;
        cam2_all.row(i) = p2.transpose();
        if (scene(i, 2) > 0.5 && p2(2) > 0.5) {
            keep.push_back(i);
        }
    }
    Points3 cam1(static_cast<Eigen::Index>(keep.size()), 3);
    Points3 cam2(static_cast<Eigen::Index>(keep.size()), 3);
    for (std::size_t k = 0; k < keep.size(); ++k) {
        cam1.row(static_cast<Eigen::Index>(k)) = scene.row(keep[k]);
        cam2.row(static_cast<Eigen::Index>(k)) = cam2_all.row(keep[k]);
    }
    Points2 a = project(cam1);
    Points2 b = project(cam2);

    if (image_size) {
        const double width = image_size->first;
        const double height = image_size->second;
        std::vector<Eigen::Index> inside;
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            if (a(i, 0) >= 0 && a(i, 0) < width && a(i, 1) >= 0 && a(i, 1) < height &&
                b(i, 0) >= 0 && b(i, 0) < width && b(i, 1) >= 0 && b(i, 1) < height) {
                inside.push_back(i);
            }
        }
        Points2 a2(static_cast<Eigen::Index>(inside.size()), 2);
        Points2 b2(static_cast<Eigen::Index>(inside.size()), 2);
        Points3 c2(static_cast<Eigen::Index>(inside.size()), 3);
        for (std::size_t k = 0; k < inside.size(); ++k) {
            a2.row(static_cast<Eigen::Index>(k)) = a.row(inside[k]);
            b2.row(static_cast<Eigen::Index>(k)) = b.row(inside[k]);
            c2.row(static_cast<Eigen::Index>(k)) = cam1.row(inside[k]);
        }
        a = a2;
        b = b2;
        cam1 = c2;
    }

    if (noise_px > 0) {
        const Points2 noise_a = golden::points2(stream.next("normal"), &stream.shared());
        const Points2 noise_b = golden::points2(stream.next("normal"), &stream.shared());
        a += noise_a;
        b += noise_b;
    }
    return {a, b, cam1};
}

Eigen::Matrix4d forward_motion(double distance, double yaw_deg, double lateral) {
    const Eigen::Matrix3d R = mvslam::rotation::from_euler("y", yaw_deg, true);
    return mvslam::se3_from_rt(R, Eigen::Vector3d(lateral, 0.0, distance));
}

Eigen::Matrix4d pure_rotation(double yaw_deg) {
    return mvslam::se3_from_rt(mvslam::rotation::from_euler("y", yaw_deg, true),
                               Eigen::Vector3d::Zero());
}

mvslam::PoseList straight_drive(int n_frames, double step, double yaw_rate_deg) {
    mvslam::PoseList poses{Eigen::Matrix4d::Identity()};
    const Eigen::Matrix4d motion = forward_motion(step, yaw_rate_deg);
    for (int i = 0; i < n_frames - 1; ++i) {
        poses.push_back(poses.back() * motion);
    }
    return poses;
}

}  // namespace synthetic
