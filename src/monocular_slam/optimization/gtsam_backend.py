"""GTSAM pose-graph optimization backend.

Maps the framework-agnostic :class:`~monocular_slam.optimization.pose_graph.PoseGraph`
onto GTSAM's factor graph and solves it.

What the optimizer actually does
--------------------------------
It minimises the sum of squared, noise-weighted residuals of every edge:

    argmin_X  sum_edges || log( Z_ij^-1 (X_i^-1 X_j) ) ||^2_Sigma_ij

Each residual is the SE(3) discrepancy between the measured relative pose
``Z_ij`` and the one implied by the current pose estimates, expressed in the
tangent space and weighted by that edge's covariance. Levenberg-Marquardt
solves it iteratively, damping the Gauss-Newton step so a poor initialisation
cannot cause divergence.

With only odometry edges the graph is a chain and the input trajectory is
already the exact optimum, so nothing moves. Loop edges make it over-determined:
the cycle they create cannot be satisfied simultaneously with all the odometry
edges, and the solver distributes that disagreement over the whole cycle in
proportion to each edge's uncertainty. That is precisely the "rubber sheet"
correction that pulls a drifted trajectory back onto itself.

Conventions
-----------
GTSAM's ``Pose3`` tangent-space ordering is **rotation first**
(``[rx, ry, rz, tx, ty, tz]``), so noise sigma vectors are built in that order.
Rotations must be exact SO(3) members or ``Rot3`` rejects them, which is why
every pose passes through :func:`~monocular_slam.geometry.transforms.project_to_se3`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from monocular_slam.geometry.pose import Trajectory
from monocular_slam.geometry.transforms import project_to_se3
from monocular_slam.optimization.pose_graph import PoseGraph
from monocular_slam.utils.logging import get_logger
from monocular_slam.utils.timing import StageTimer

logger = get_logger(__name__)

_GTSAM_IMPORT_ERROR = (
    "GTSAM is required for pose-graph optimization but is not installed.\n"
    "Install it with:  pip install 'gtsam>=4.2,<4.3'\n"
    "Note that gtsam 4.2 wheels require NumPy < 2.0."
)


def _import_gtsam():
    """Import GTSAM lazily with an actionable error message.

    Deferred so that the front end, metrics and tests that never touch the
    back end remain usable in an environment without GTSAM.
    """
    try:
        import gtsam  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_GTSAM_IMPORT_ERROR) from exc
    return gtsam


def gtsam_available() -> bool:
    """True when GTSAM can be imported."""
    try:
        _import_gtsam()
    except ImportError:  # pragma: no cover - environment dependent
        return False
    return True


@dataclass
class OptimizationResult:
    """Outcome of a pose-graph optimization."""

    trajectory: Trajectory
    initial_error: float
    final_error: float
    iterations: int
    n_nodes: int
    n_odometry_edges: int
    n_loop_edges: int
    converged: bool
    optimizer: str = "levenberg_marquardt"
    residuals_before: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    residuals_after: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def error_reduction_pct(self) -> float:
        """Percentage drop in the total weighted squared error."""
        if not np.isfinite(self.initial_error) or self.initial_error <= 0:
            return float("nan")
        return float((self.initial_error - self.final_error) / self.initial_error * 100.0)

    def to_dict(self) -> dict[str, object]:
        def summarize(residuals: dict[str, np.ndarray]) -> dict[str, float | None]:
            out: dict[str, float | None] = {}
            for key, values in residuals.items():
                out[f"mean_{key}"] = round(float(values.mean()), 5) if len(values) else None
                out[f"max_{key}"] = round(float(values.max()), 5) if len(values) else None
            return out

        return {
            "optimizer": self.optimizer,
            "n_nodes": self.n_nodes,
            "n_odometry_edges": self.n_odometry_edges,
            "n_loop_edges": self.n_loop_edges,
            "iterations": self.iterations,
            "converged": self.converged,
            "initial_error": round(self.initial_error, 6),
            "final_error": round(self.final_error, 6),
            "error_reduction_pct": round(self.error_reduction_pct, 3),
            "residuals_before": summarize(self.residuals_before),
            "residuals_after": summarize(self.residuals_after),
        }


def numpy_to_pose3(T: np.ndarray):
    """Convert a ``4x4`` NumPy transform to a GTSAM ``Pose3``.

    The rotation is re-projected onto SO(3) first: after thousands of matrix
    products a VO pose can drift far enough from orthonormal that ``Rot3``
    rejects it outright.
    """
    gtsam = _import_gtsam()
    T = project_to_se3(np.asarray(T, dtype=np.float64))
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def pose3_to_numpy(pose) -> np.ndarray:
    """Convert a GTSAM ``Pose3`` back to a ``4x4`` NumPy transform."""
    return np.asarray(pose.matrix(), dtype=np.float64)


def _noise_model(
    sigma_rot: float, sigma_trans: float, robust: bool = False, huber_k: float = 1.345
):
    """Build a diagonal (optionally Huber-robust) 6-dof noise model.

    Sigma ordering is rotation-then-translation to match ``Pose3``'s tangent
    space; getting this backwards silently swaps the rotational and
    translational confidence, which produces a plausible-looking but wrong
    solution.
    """
    gtsam = _import_gtsam()
    sigmas = np.array(
        [sigma_rot, sigma_rot, sigma_rot, sigma_trans, sigma_trans, sigma_trans],
        dtype=np.float64,
    )
    model = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    if robust:
        model = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(float(huber_k)), model
        )
    return model


def optimize_pose_graph(
    graph: PoseGraph,
    *,
    optimizer: str = "levenberg_marquardt",
    max_iterations: int = 100,
    relative_error_tol: float = 1e-5,
    absolute_error_tol: float = 1e-5,
    robust_loop_kernel: bool = True,
    huber_k: float = 1.345,
    verbose: bool = False,
    timer: StageTimer | None = None,
) -> OptimizationResult:
    """Optimize ``graph`` with GTSAM and return the corrected trajectory.

    Raises
    ------
    ValueError
        If the graph is structurally unsound (disconnected, invalid poses).
        Failing here produces a clear message instead of GTSAM's
        ``IndeterminantLinearSystemException`` deep inside the solver.
    """
    gtsam = _import_gtsam()

    problems = graph.validate()
    if problems:
        raise ValueError("Pose graph is not optimizable:\n  - " + "\n  - ".join(problems))

    timer = timer if timer is not None else StageTimer()
    residuals_before = graph.edge_residuals()

    factors = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Anchor the gauge freedom: without a prior the whole graph can translate
    # and rotate freely, leaving the linear system rank deficient.
    prior_pose = numpy_to_pose3(graph.nodes[graph.prior_node_id].pose)
    factors.add(
        gtsam.PriorFactorPose3(
            graph.prior_node_id,
            prior_pose,
            _noise_model(graph.prior_sigma_rot, graph.prior_sigma_trans),
        )
    )

    for node in graph.nodes:
        initial.insert(node.node_id, numpy_to_pose3(node.pose))

    for edge in graph.odometry_edges:
        factors.add(
            gtsam.BetweenFactorPose3(
                edge.from_id,
                edge.to_id,
                numpy_to_pose3(edge.T),
                _noise_model(edge.sigma_rot, edge.sigma_trans),
            )
        )

    for edge in graph.loop_edges:
        use_robust = robust_loop_kernel and getattr(edge, "robust", True)
        factors.add(
            gtsam.BetweenFactorPose3(
                edge.from_id,
                edge.to_id,
                numpy_to_pose3(edge.T),
                _noise_model(edge.sigma_rot, edge.sigma_trans, robust=use_robust, huber_k=huber_k),
            )
        )

    initial_error = float(factors.error(initial))

    optimizer_name = str(optimizer).lower().strip()
    if optimizer_name in ("levenberg_marquardt", "lm"):
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(int(max_iterations))
        params.setRelativeErrorTol(float(relative_error_tol))
        params.setAbsoluteErrorTol(float(absolute_error_tol))
        if verbose:
            params.setVerbosityLM("SUMMARY")
        solver = gtsam.LevenbergMarquardtOptimizer(factors, initial, params)
    elif optimizer_name in ("gauss_newton", "gn"):
        params = gtsam.GaussNewtonParams()
        params.setMaxIterations(int(max_iterations))
        params.setRelativeErrorTol(float(relative_error_tol))
        params.setAbsoluteErrorTol(float(absolute_error_tol))
        if verbose:
            params.setVerbosity("SUMMARY")
        solver = gtsam.GaussNewtonOptimizer(factors, initial, params)
    else:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Choose 'levenberg_marquardt' or 'gauss_newton'"
        )

    with timer.time("optimize"):
        result_values = solver.optimize()

    final_error = float(factors.error(result_values))
    iterations = int(solver.iterations())

    poses = np.stack([pose3_to_numpy(result_values.atPose3(node.node_id)) for node in graph.nodes])
    trajectory = Trajectory(
        poses,
        frame_ids=np.array([node.frame_id for node in graph.nodes], dtype=np.int64),
    ).orthonormalized()

    residuals_after = graph.edge_residuals(trajectory)

    result = OptimizationResult(
        trajectory=trajectory,
        initial_error=initial_error,
        final_error=final_error,
        iterations=iterations,
        n_nodes=len(graph.nodes),
        n_odometry_edges=len(graph.odometry_edges),
        n_loop_edges=len(graph.loop_edges),
        converged=iterations < int(max_iterations),
        optimizer=optimizer_name,
        residuals_before=residuals_before,
        residuals_after=residuals_after,
    )

    logger.info(
        "Pose graph optimized: %d nodes, %d loop edges, %d iterations, "
        "error %.4g -> %.4g (%.1f%% reduction)",
        result.n_nodes,
        result.n_loop_edges,
        result.iterations,
        result.initial_error,
        result.final_error,
        result.error_reduction_pct,
    )
    loop_before = residuals_before.get("loop_translation_m", np.zeros(0))
    loop_after = residuals_after.get("loop_translation_m", np.zeros(0))
    if len(loop_before):
        logger.info(
            "Loop edge residual: %.3f m -> %.3f m (mean)",
            float(loop_before.mean()),
            float(loop_after.mean()),
        )
    return result
