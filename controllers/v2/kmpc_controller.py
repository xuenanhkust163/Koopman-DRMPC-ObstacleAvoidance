"""K-MPC adapter using the shared v2 track-following controller context."""

import numpy as np
import casadi as ca

from controllers.kmpc_controller import KMPCController as KoopmanMPCOptimizer
from config import DT, N_X

from .pure_pursuit_baseline import PurePursuitBaselineController


class KoopmanMPCSolveStrategy:
    """Adapt the established Koopman MPC optimizer to the v2 solve interface."""

    def __init__(self, koopman_model, D_matrix, norm_params, cost_builder=None, cost_profile="default", trace=False):
        self.trace = bool(trace)
        self.solve_count = 0
        self.optimizer = KoopmanMPCOptimizer(
            koopman_model,
            D_matrix,
            norm_params,
            cost_builder=cost_builder,
            cost_profile=cost_profile,
        )
        self.optimizer.use_physical_state_dynamics = True
        raw_A, _, _ = koopman_model.get_matrices()
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(raw_A))))
        if spectral_radius < 1.0:
            self.optimizer.A = np.asarray(raw_A, dtype=float)
            self.optimizer._A_ca = ca.DM(self.optimizer.A)

        network_encoder = self.optimizer._ca_encode

        def encode_with_physical_passthrough(x_sym):
            lifted = network_encoder(x_sym)
            return ca.vertcat(x_sym, lifted[N_X:])

        self.optimizer._ca_encode = encode_with_physical_passthrough
        if self.trace:
            print(
                "[KMPC trace] prediction model enabled: "
                f"A={self.optimizer.A.shape}, B={self.optimizer.B.shape}, "
                f"Koopman latent dimension={self.optimizer.A.shape[0]}; "
                f"physical state[0:{N_X}] uses bicycle dynamics, "
                f"lifted features[{N_X}:] use A*z+B*u"
            )

        original_encode_state = self.optimizer._encode_state

        def encode_state_with_trace(x_physical):
            latent = original_encode_state(x_physical)
            if self.trace:
                print(
                    f"[KMPC trace] encoder call: physical={np.array2string(np.asarray(x_physical), precision=4)}, "
                    f"normalized latent z0={np.array2string(latent, precision=4)}"
                )
            return latent

        self.optimizer._encode_state = encode_state_with_trace

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

        if self.trace:
            print(
                f"[KMPC trace] solve #{self.solve_count}: "
                f"state={np.array2string(np.asarray(x_current), precision=4)}, "
                f"reference_horizon={len(reference)}, obstacles={len(obstacles)}"
            )
        control, info = self.optimizer.solve(
            x_current,
            reference,
            obstacles,
            u_prev=u_prev,
        )
        if self.trace:
            print(
                f"[KMPC trace] solve #{self.solve_count} result: "
                f"u=[a={control[0]:.4f}, delta={control[1]:.4f}], "
                f"status={info['status']}, solve_time={info['solve_time']:.4f}s"
            )
        return control, info


class KMPCController(PurePursuitBaselineController):
    """K-MPC controller with v2 path context and the trained Koopman model."""

    def __init__(
        self,
        track,
        koopman_model,
        D_matrix,
        norm_params,
        cost_builder=None,
        cost_profile="default",
        trace=False,
        **kwargs,
    ):
        strategy = KoopmanMPCSolveStrategy(
            koopman_model,
            D_matrix,
            norm_params,
            cost_builder=cost_builder,
            cost_profile=cost_profile,
            trace=trace,
        )
        super().__init__(track, solve_strategy=strategy, **kwargs)
        self.name = "KMPCBaseline"