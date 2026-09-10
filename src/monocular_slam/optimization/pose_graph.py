"""SE(3) pose-graph representation.

A pose graph is the standard back-end abstraction for SLAM: it discards the
image measurements and keeps only the *relative pose constraints* they implied.

* **Nodes** are keyframe poses ``T_w_kf`` — the quantities being solved for.
* **Odometry edges** connect consecutive keyframes with the relative motion the
  front end integrated between them.
* **Loop edges** connect a revisited keyframe pair with the relative pose that
  geometric verification recovered.

Without a loop edge the graph is a chain, and a chain has exactly one solution:
the odometry itself. Loop edges are what make it a *graph* — they introduce
cycles whose constraints are mutually inconsistent because of accumulated
drift, and optimization redistributes that inconsistency over the whole
trajectory instead of leaving it concentrated at the seam.

Noise model assumptions
-----------------------
Every edge carries a diagonal Gaussian noise model over the 6-dof tangent
space, ordered ``[rx, ry, rz, tx, ty, tz]`` to match GTSAM's ``Pose3``
convention (rotation first).

* **Prior on the first pose.** The graph is only determined up to a global
  rigid transform, so one node must be anchored. A very tight prior fixes the
  gauge freedom without meaningfully constraining the solution.
* **Odometry edges** get moderate sigmas. Frame-to-frame VO is locally
  accurate but its errors are correlated over time; the constant diagonal here
  is a deliberate simplification, since propagating a full covariance would
  require uncertainty estimates the essential-matrix decomposition does not
  provide.
* **Loop edges** get looser translation sigmas than odometry. Their rotation is
  well determined by wide-baseline matching, but their translation *magnitude*
  is recovered indirectly, from a triangulated depth ratio against neighbouring
  structure, so it deserves less confidence than the rotation. A loop whose
  magnitude could not be recovered at all gets a much larger translation sigma
  still, making it an orientation-only constraint.
* **Robust kernel.** Loop factors are optionally wrapped in a Huber kernel. A
  single false-positive loop asserts that two unrelated places are the same,
  and under a pure least-squares cost that one enormous residual can drag the
  entire trajectory with it. Huber bounds its influence, which is cheap
  insurance for a detector that cannot be perfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import (
    invert_se3,
    is_valid_se3,
    project_to_se3,
    rotation_angle_deg,
)
from monocular_slam.utils.logging import get_logger

logger = get_logger(__name__)

#: Translation sigma multiplier for a loop edge whose metric magnitude could
#: not be recovered. Large enough that the translation residual is effectively
#: unconstrained while the rotation residual still acts.
UNSCALED_LOOP_SIGMA_FACTOR = 100.0


@dataclass
class PoseGraphNode:
    """A pose variable in the graph."""

    #: Sequential node id (also the GTSAM variable key).
    node_id: int
    #: Initial estimate of the camera-to-world pose.
    pose: np.ndarray
    #: Index into the processed frame sequence this node corresponds to.
    frame_index: int = 0
    #: Original KITTI frame number.
    frame_id: int = 0

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Node pose must be 4x4, got {pose.shape}")
        self.pose = pose


@dataclass
class OdometryEdge:
    """A sequential relative-motion constraint between two nodes."""

    from_id: int
    to_id: int
    #: ``T_from_to``: pose of ``to`` expressed in ``from``'s frame.
    T: np.ndarray
    #: Rotation sigma in radians and translation sigma in metres.
    sigma_rot: float = 0.02
    sigma_trans: float = 0.10

    def __post_init__(self) -> None:
        self.T = np.asarray(self.T, dtype=np.float64)
        if self.T.shape != (4, 4):
            raise ValueError(f"Edge transform must be 4x4, got {self.T.shape}")


@dataclass
class LoopEdge(OdometryEdge):
    """A loop-closure constraint, carrying its verification evidence."""

    sigma_rot: float = 0.05
    sigma_trans: float = 0.30
    n_inliers: int = 0
    similarity: float = 0.0
    robust: bool = True


@dataclass
class PoseGraph:
    """Nodes plus odometry and loop edges, ready for optimization."""

    nodes: list[PoseGraphNode] = field(default_factory=list)
    odometry_edges: list[OdometryEdge] = field(default_factory=list)
    loop_edges: list[LoopEdge] = field(default_factory=list)
    #: Node anchored by a prior. The gauge freedom has to be fixed somewhere.
    prior_node_id: int = 0
    prior_sigma_rot: float = 1e-4
    prior_sigma_trans: float = 1e-4

    # ----------------------------------------------------------------- #
    # Construction
    # ----------------------------------------------------------------- #

    def add_node(self, pose: np.ndarray, frame_index: int = 0, frame_id: int = 0) -> PoseGraphNode:
        """Append a node and return it."""
        node = PoseGraphNode(
            node_id=len(self.nodes),
            pose=project_to_se3(pose),
            frame_index=frame_index,
            frame_id=frame_id,
        )
        self.nodes.append(node)
        return node

    def add_odometry_edge(
        self, from_id: int, to_id: int, T: np.ndarray, sigma_rot: float = 0.02,
        sigma_trans: float = 0.10,
    ) -> OdometryEdge:
        self._check_ids(from_id, to_id)
        edge = OdometryEdge(from_id, to_id, project_to_se3(T), sigma_rot, sigma_trans)
        self.odometry_edges.append(edge)
        return edge

    def add_loop_edge(
        self,
        from_id: int,
        to_id: int,
        T: np.ndarray,
        sigma_rot: float = 0.05,
        sigma_trans: float = 0.30,
        n_inliers: int = 0,
        similarity: float = 0.0,
        robust: bool = True,
    ) -> LoopEdge:
        self._check_ids(from_id, to_id)
        edge = LoopEdge(
            from_id=from_id,
            to_id=to_id,
            T=project_to_se3(T),
            sigma_rot=sigma_rot,
            sigma_trans=sigma_trans,
            n_inliers=n_inliers,
            similarity=similarity,
            robust=robust,
        )
        self.loop_edges.append(edge)
        return edge

    def _check_ids(self, *ids: int) -> None:
        for node_id in ids:
            if not 0 <= node_id < len(self.nodes):
                raise IndexError(f"Node id {node_id} out of range (0..{len(self.nodes) - 1})")

    # ----------------------------------------------------------------- #
    # Inspection
    # ----------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def edges(self) -> list[OdometryEdge]:
        return [*self.odometry_edges, *self.loop_edges]

    def is_connected(self) -> bool:
        """True when every node is reachable from the prior node.

        A disconnected component has no path to the anchor, so its absolute
        pose is unconstrained and the optimizer's information matrix becomes
        singular. Checking up front turns a cryptic GTSAM ``IndeterminantLinear
        SystemException`` into an actionable message.
        """
        if not self.nodes:
            return True
        if not 0 <= self.prior_node_id < len(self.nodes):
            # Nothing is reachable from an anchor that does not exist.
            return False
        adjacency: dict[int, list[int]] = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            adjacency[edge.from_id].append(edge.to_id)
            adjacency[edge.to_id].append(edge.from_id)

        seen = {self.prior_node_id}
        stack = [self.prior_node_id]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == len(self.nodes)

    def validate(self) -> list[str]:
        """Return a list of structural problems; empty means the graph is sound."""
        problems: list[str] = []
        if not self.nodes:
            problems.append("graph has no nodes")
            return problems
        if not 0 <= self.prior_node_id < len(self.nodes):
            problems.append(f"prior_node_id {self.prior_node_id} is not a valid node")
        for node in self.nodes:
            if not is_valid_se3(node.pose, tol=1e-4):
                problems.append(f"node {node.node_id} pose is not a valid SE(3) member")
        for edge in self.edges:
            if not is_valid_se3(edge.T, tol=1e-4):
                problems.append(f"edge {edge.from_id}->{edge.to_id} transform is not valid SE(3)")
            if edge.from_id == edge.to_id:
                problems.append(f"edge {edge.from_id}->{edge.to_id} is a self-loop")
            if edge.sigma_rot <= 0 or edge.sigma_trans <= 0:
                problems.append(f"edge {edge.from_id}->{edge.to_id} has a non-positive sigma")
        if not self.is_connected():
            problems.append("graph is disconnected; some poses are unconstrained")
        return problems

    # ----------------------------------------------------------------- #
    # Residuals
    # ----------------------------------------------------------------- #

    def edge_residuals(self, trajectory: Trajectory | None = None) -> dict[str, np.ndarray]:
        """Per-edge translation (m) and rotation (deg) disagreement.

        Measures how far each constraint is from being satisfied by the current
        estimate. Loop-edge residuals before optimization are effectively a
        direct readout of accumulated drift around each cycle.
        """
        poses = (
            np.stack([node.pose for node in self.nodes])
            if trajectory is None
            else trajectory.poses
        )
        result: dict[str, np.ndarray] = {}
        for name, edges in (("odometry", self.odometry_edges), ("loop", self.loop_edges)):
            if not edges:
                result[f"{name}_translation_m"] = np.zeros(0)
                result[f"{name}_rotation_deg"] = np.zeros(0)
                continue
            translations = np.zeros(len(edges))
            rotations = np.zeros(len(edges))
            for i, edge in enumerate(edges):
                predicted = invert_se3(poses[edge.from_id]) @ poses[edge.to_id]
                error = invert_se3(edge.T) @ predicted
                translations[i] = float(np.linalg.norm(error[:3, 3]))
                rotations[i] = rotation_angle_deg(error[:3, :3])
            result[f"{name}_translation_m"] = translations
            result[f"{name}_rotation_deg"] = rotations
        return result

    def describe(self) -> dict[str, object]:
        residuals = self.edge_residuals()
        loop_translation = residuals["loop_translation_m"]
        return {
            "n_nodes": len(self.nodes),
            "n_odometry_edges": len(self.odometry_edges),
            "n_loop_edges": len(self.loop_edges),
            "connected": self.is_connected(),
            "mean_loop_residual_m": (
                round(float(loop_translation.mean()), 4) if len(loop_translation) else None
            ),
            "max_loop_residual_m": (
                round(float(loop_translation.max()), 4) if len(loop_translation) else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"PoseGraph(nodes={len(self.nodes)}, odometry={len(self.odometry_edges)}, "
            f"loops={len(self.loop_edges)})"
        )


def build_pose_graph(
    trajectory: Trajectory,
    loop_closures: list,
    node_frame_indices: list[int] | None = None,
    *,
    prior_sigma_rot: float = 1e-4,
    prior_sigma_trans: float = 1e-4,
    odom_sigma_rot: float = 0.02,
    odom_sigma_trans: float = 0.10,
    loop_sigma_rot: float = 0.05,
    loop_sigma_trans: float = 0.30,
    robust_loops: bool = True,
) -> PoseGraph:
    """Assemble a pose graph from a keyframe trajectory and verified loops.

    ``trajectory`` holds one pose per graph node (keyframes, not every frame).
    Odometry edges are derived from the trajectory itself: the relative pose
    between consecutive nodes *is* what the front end estimated for that span.

    ``loop_closures`` are :class:`~monocular_slam.loop_closure.detector.LoopClosure`
    objects whose ``match_id``/``query_id`` index into the same node ordering.
    """
    graph = PoseGraph(
        prior_sigma_rot=prior_sigma_rot,
        prior_sigma_trans=prior_sigma_trans,
    )

    frame_indices = node_frame_indices or list(range(len(trajectory)))
    for i, pose in enumerate(trajectory.poses):
        graph.add_node(
            pose,
            frame_index=frame_indices[i] if i < len(frame_indices) else i,
            frame_id=int(trajectory.frame_ids[i]),
        )

    for i in range(len(trajectory) - 1):
        relative = invert_se3(trajectory[i]) @ trajectory[i + 1]
        graph.add_odometry_edge(i, i + 1, relative, odom_sigma_rot, odom_sigma_trans)

    n_unscaled = 0
    for closure in loop_closures:
        # A loop whose metric magnitude could not be recovered still carries a
        # trustworthy *rotation*. Rather than discard it, or assert a
        # translation that was never measured, its translation sigma is
        # inflated so the factor acts as an orientation constraint. Heading
        # drift is the dominant error term, so this is worth keeping.
        scale_measured = getattr(closure, "scale_is_measured", True)
        sigma_trans = (
            loop_sigma_trans
            if scale_measured
            else loop_sigma_trans * UNSCALED_LOOP_SIGMA_FACTOR
        )
        n_unscaled += 0 if scale_measured else 1
        graph.add_loop_edge(
            closure.match_id,
            closure.query_id,
            closure.T_match_query,
            sigma_rot=loop_sigma_rot,
            sigma_trans=sigma_trans,
            n_inliers=closure.n_inliers,
            similarity=closure.similarity,
            robust=robust_loops,
        )
    if n_unscaled:
        logger.info(
            "%d of %d loop edges have no measured translation magnitude and were "
            "added as orientation-only constraints",
            n_unscaled,
            len(loop_closures),
        )

    logger.info(
        "Built pose graph: %d nodes, %d odometry edges, %d loop edges",
        len(graph.nodes),
        len(graph.odometry_edges),
        len(graph.loop_edges),
    )
    return graph
