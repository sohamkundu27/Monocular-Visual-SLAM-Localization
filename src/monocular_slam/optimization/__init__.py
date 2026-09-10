"""SE(3) pose-graph construction and optimization."""

from monocular_slam.optimization.gtsam_backend import (
    OptimizationResult,
    gtsam_available,
    optimize_pose_graph,
    pose3_to_numpy,
    numpy_to_pose3,
)
from monocular_slam.optimization.pose_graph import (
    LoopEdge,
    OdometryEdge,
    PoseGraph,
    PoseGraphNode,
    build_pose_graph,
)

__all__ = [
    "LoopEdge",
    "OdometryEdge",
    "OptimizationResult",
    "PoseGraph",
    "PoseGraphNode",
    "build_pose_graph",
    "gtsam_available",
    "numpy_to_pose3",
    "optimize_pose_graph",
    "pose3_to_numpy",
]
