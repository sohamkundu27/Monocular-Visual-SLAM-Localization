"""Classical monocular visual SLAM and localization.

A from-scratch implementation of a monocular SLAM front end (ORB features,
essential-matrix motion estimation) and back end (loop closure detection plus
SE(3) pose-graph optimization with GTSAM), benchmarked against KITTI Odometry
ground truth.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
