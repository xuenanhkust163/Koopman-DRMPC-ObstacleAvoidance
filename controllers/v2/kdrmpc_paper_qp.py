"""V2 adapter for the paper's Koopman distributionally robust QP controller."""

import numpy as np

from config import DT, EPSILON_CVAR, N_DISTURBANCE_SAMPLES, NOMINAL_SIGMA, THETA_WASSERSTEIN
from controllers.kdrmpc_paper_qp_controller import KDRMPCPaperQPController as PaperQPOptimizer
from disturbance.disturbance_generator import DisturbanceGenerator

from .pure_pursuit_baseline import PurePursuitBaselineController


class KDRMPCPaperQPSolveStrategy:
    """Adapt the paper QP optimizer to the shared v2 track-following context."""

    def __init__(
        self,
        koopman_model,
        projection,
        norm_params,
        track,
        distributionally_robust=True,
        theta=THETA_WASSERSTEIN,
        epsilon=EPSILON_CVAR,
        disturbance_sigma=NOMINAL_SIGMA,
        disturbance_samples=N_DISTURBANCE_SAMPLES,
        qp_solver="qpoases",
    ):
        sampler = DisturbanceGenerator(
            sigma=disturbance_sigma,
            seed=42,
            enabled=True,
        )
        empirical_samples = sampler.get_empirical_samples(int(disturbance_samples))
        self.distributionally_robust = bool(distributionally_robust)
        self.optimizer = PaperQPOptimizer(
            koopman_model,
            projection,
            norm_params,
            disturbance_samples=empirical_samples,
            theta=float(theta),
            epsilon=float(epsilon),
            track=track,
            obstacle_strategy="robust" if self.distributionally_robust else "non-robust",
            qp_solver=qp_solver,
        )
        self.solve_count = 0

    @staticmethod
    def _path_xy(context, arc_position):
        x, y, heading, _, _ = context._sample_path(arc_position)
        offset = context._obstacle_lateral_offset(arc_position)
        x -= np.sin(heading) * offset
        y += np.cos(heading) * offset
        return x, y

    @classmethod
    def _path_point(cls, context, arc_position):
        _, _, _, _, speed = context._sample_path(arc_position)
        x, y = cls._path_xy(context, arc_position)
        previous_x, previous_y = cls._path_xy(context, arc_position - 0.5)
        next_x, next_y = cls._path_xy(context, arc_position + 0.5)
        heading = float(np.arctan2(next_y - previous_y, next_x - previous_x))
        heading_in = np.arctan2(y - previous_y, x - previous_x)
        heading_out = np.arctan2(next_y - y, next_x - x)
        curvature = float(np.arctan2(
            np.sin(heading_out - heading_in),
            np.cos(heading_out - heading_in),
        ))
        return x, y, heading, curvature, speed

    def reset(self):
        self.optimizer.reset()
        self.solve_count = 0

    def solve(self, context, x_current, ref_trajectory, obstacles, u_prev=None):
        self.solve_count += 1
        _, current_s, _ = context.track.closest_point(x_current[0], x_current[1])
        arc_positions = np.zeros(len(ref_trajectory), dtype=float)
        arc_positions[0] = current_s
        for index in range(1, len(arc_positions)):
            target_speed = context.target_speed_at_s(arc_positions[index - 1])
            arc_positions[index] = arc_positions[index - 1] + max(target_speed, 1.0) * DT

        reference = np.zeros_like(ref_trajectory, dtype=float)
        for index, arc_position in enumerate(arc_positions):
            x, y, heading, curvature, speed = self._path_point(context, arc_position)
            reference[index] = [x, y, heading, speed, speed * curvature]

        previous_control = self.optimizer.u_prev.copy()
        control, info = self.optimizer.solve(
            x_current,
            reference,
            obstacles,
            u_prev=u_prev,
        )
        if (
            not self.distributionally_robust
            and str(info.get("status", "")).startswith("failed")
            and self.optimizer.hard_obstacle_constraints
            and self.optimizer.halfspace_mode == "track-normal"
        ):
            failed_info = info
            self.optimizer.halfspace_mode = "axis"
            try:
                control, fallback_info = self.optimizer.solve(
                    x_current,
                    reference,
                    obstacles,
                    u_prev=previous_control if u_prev is None else u_prev,
                )
            finally:
                self.optimizer.halfspace_mode = "track-normal"
            if not str(fallback_info.get("status", "")).startswith("failed"):
                fallback_info["status"] = "optimal_axis_halfspace_fallback"
                fallback_info["halfspace_fallback_from"] = "track-normal"
                fallback_info["halfspace_fallback_reason"] = str(failed_info.get("status", ""))[:240]
            info = fallback_info
        info["controller_integration"] = "v2-paper-koopman-qp"
        info["distributionally_robust"] = self.distributionally_robust
        info["solve_count"] = self.solve_count
        return control, info


class KDRMPCPaperQPController(PurePursuitBaselineController):
    """Paper Koopman-QP controller with an isolated DR on/off strategy switch."""

    def __init__(
        self,
        track,
        koopman_model,
        projection,
        norm_params,
        distributionally_robust=True,
        theta=THETA_WASSERSTEIN,
        epsilon=EPSILON_CVAR,
        disturbance_sigma=NOMINAL_SIGMA,
        disturbance_samples=N_DISTURBANCE_SAMPLES,
        qp_solver="qpoases",
        **kwargs,
    ):
        strategy = KDRMPCPaperQPSolveStrategy(
            koopman_model,
            projection,
            norm_params,
            track,
            distributionally_robust=distributionally_robust,
            theta=theta,
            epsilon=epsilon,
            disturbance_sigma=disturbance_sigma,
            disturbance_samples=disturbance_samples,
            qp_solver=qp_solver,
        )
        super().__init__(track, solve_strategy=strategy, **kwargs)
        self.name = "KDRMPCPaperQP"
        self.distributionally_robust = bool(distributionally_robust)
