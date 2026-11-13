"""Convex-QP paper controller for Koopman DRMPC.

This controller intentionally avoids the nonlinear physical prediction and
wrapped-heading costs used by the IPOPT paper controller. The optimization is a
convex quadratic program: affine Koopman prediction, quadratic tracking/control
costs, and affine track/obstacle DRCC constraints.
"""

import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Optional

import casadi as ca
import numpy as np

from controllers import kdrmpc_controller as base
from controllers.kdrmpc_paper_controller import KDRMPCPaperController


@dataclass(frozen=True)
class QPEngineeringConfig:
    """Engineering-side QP tuning knobs kept separate from paper-core settings."""

    track_slack_penalty: Optional[float] = None
    speed_slack_penalty: Optional[float] = None
    heading_slack_penalty: float = 600.0
    heading_error_limit_deg: float = 45.0
    hard_constraint_slack_scale: float = 25.0
    track_slack_max: float = 0.35
    heading_slack_max_deg: float = 8.0
    progress_reward_weight: float = 0.25
    terminal_weight_scale: float = 1.0
    lateral_centering_weight: float = 12.0
    steering_rate_weight_scale: float = 1.0
    steering_abs_weight_scale: float = 1.0
    steering_curvature_weight: float = 0.0
    obstacle_lateral_bias_weight: float = 18.0
    obstacle_speed_cap_base: float = 2.0
    obstacle_speed_cap_gain: float = 0.10
    lateral_speed_cap_gain: float = 0.9
    lateral_speed_cap_base: float = 2.4
    linear_physical_consistency: bool = True
    fallback_speed_gain: float = 0.9
    fallback_heading_gain: float = 1.8
    fallback_lateral_gain: float = 0.08

    def resolved_track_slack_penalty(self):
        if self.track_slack_penalty is None:
            return float(base.TRACK_BOUNDARY_SLACK_PENALTY)
        return float(self.track_slack_penalty)

    def resolved_speed_slack_penalty(self):
        if self.speed_slack_penalty is None:
            return float(base.MIN_SPEED_SLACK_PENALTY)
        return float(self.speed_slack_penalty)


_ENGINEERING_KWARG_NAMES = {
    "track_slack_penalty",
    "speed_slack_penalty",
    "heading_slack_penalty",
    "heading_error_limit_deg",
    "hard_constraint_slack_scale",
    "track_slack_max",
    "heading_slack_max_deg",
    "progress_reward_weight",
    "terminal_weight_scale",
    "lateral_centering_weight",
    "steering_rate_weight_scale",
    "steering_abs_weight_scale",
    "steering_curvature_weight",
    "obstacle_lateral_bias_weight",
    "obstacle_speed_cap_base",
    "obstacle_speed_cap_gain",
    "lateral_speed_cap_gain",
    "lateral_speed_cap_base",
    "linear_physical_consistency",
    "fallback_speed_gain",
    "fallback_heading_gain",
    "fallback_lateral_gain",
}


class KDRMPCPaperQPController(KDRMPCPaperController):
    """Paper-faithful convex QP variant solved with CasADi qpsol/OSQP."""

    @staticmethod
    def _resolve_engineering_config(engineering_config, kwargs):
        legacy_overrides = {}
        for key in list(kwargs.keys()):
            if key in _ENGINEERING_KWARG_NAMES:
                legacy_overrides[key] = kwargs.pop(key)

        if engineering_config is None:
            return QPEngineeringConfig(**legacy_overrides)
        if legacy_overrides:
            return replace(engineering_config, **legacy_overrides)
        return engineering_config

    def __init__(
        self,
        *args,
        qp_solver="qpoases",
        engineering_config=None,
        disturbance_trajectories=None,
        residual_scale=None,
        safety_margin_delta=None,
        **kwargs,
    ):
        engineering_config = self._resolve_engineering_config(engineering_config, kwargs)
        super().__init__(*args, **kwargs)
        self.name = "K-DRMPC-Paper-QP"
        self.paper_controller_version = "shifted-hinge-w1-drcc-qp-v1"
        self.qp_solver = str(qp_solver)
        self.engineering_config = engineering_config
        self.track_slack_penalty = self.engineering_config.resolved_track_slack_penalty()
        self.speed_slack_penalty = self.engineering_config.resolved_speed_slack_penalty()
        self.heading_slack_penalty = float(self.engineering_config.heading_slack_penalty)
        self.heading_error_limit = np.deg2rad(float(self.engineering_config.heading_error_limit_deg))
        self.hard_constraint_slack_scale = float(self.engineering_config.hard_constraint_slack_scale)
        self.track_slack_max = float(self.engineering_config.track_slack_max)
        self.heading_slack_max = np.deg2rad(float(self.engineering_config.heading_slack_max_deg))
        self.progress_reward_weight = float(self.engineering_config.progress_reward_weight)
        self.terminal_weight_scale = float(self.engineering_config.terminal_weight_scale)
        self.lateral_centering_weight = float(self.engineering_config.lateral_centering_weight)
        self.steering_rate_weight_scale = float(self.engineering_config.steering_rate_weight_scale)
        self.steering_abs_weight_scale = float(self.engineering_config.steering_abs_weight_scale)
        self.steering_curvature_weight = float(self.engineering_config.steering_curvature_weight)
        self.obstacle_lateral_bias_weight = float(self.engineering_config.obstacle_lateral_bias_weight)
        self.obstacle_speed_cap_base = float(self.engineering_config.obstacle_speed_cap_base)
        self.obstacle_speed_cap_gain = float(self.engineering_config.obstacle_speed_cap_gain)
        self.lateral_speed_cap_gain = float(self.engineering_config.lateral_speed_cap_gain)
        self.lateral_speed_cap_base = float(self.engineering_config.lateral_speed_cap_base)
        self.linear_physical_consistency = bool(self.engineering_config.linear_physical_consistency)
        self.fallback_speed_gain = float(self.engineering_config.fallback_speed_gain)
        self.fallback_heading_gain = float(self.engineering_config.fallback_heading_gain)
        self.fallback_lateral_gain = float(self.engineering_config.fallback_lateral_gain)
        self._qp_warm_start = None

        self.disturbance_trajectories = None
        self.residual_scale = None
        self.safety_margin_delta = None
        if disturbance_trajectories is not None:
            trajectories = np.asarray(disturbance_trajectories, dtype=float)
            expected_tail = (self.prediction_horizon, self.A.shape[0])
            if trajectories.ndim != 3 or trajectories.shape[1:] != expected_tail:
                raise ValueError(
                    "disturbance_trajectories must have shape "
                    f"(N, {expected_tail[0]}, {expected_tail[1]}), got {trajectories.shape}"
                )
            if len(trajectories) == 0 or not np.isfinite(trajectories).all():
                raise ValueError("disturbance_trajectories must be nonempty and finite")
            if residual_scale is None:
                raise ValueError("residual_scale is required with disturbance_trajectories")
            scale = np.asarray(residual_scale, dtype=float).reshape(-1)
            if scale.shape != (self.A.shape[0],) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
                raise ValueError("residual_scale must be a finite positive vector of length n_z")
            if len(trajectories) > base.MAX_OPT_SAMPLES:
                rng = np.random.RandomState(42)
                selected = rng.choice(
                    len(trajectories),
                    size=base.MAX_OPT_SAMPLES,
                    replace=False,
                )
                trajectories = trajectories[selected]
            self.disturbance_sample_count = len(trajectories)
            self.disturbance_trajectories = trajectories
            self.residual_scale = scale

        if safety_margin_delta is not None:
            self.safety_margin_delta = float(safety_margin_delta)
            if not np.isfinite(self.safety_margin_delta) or self.safety_margin_delta <= 0.0:
                raise ValueError("safety_margin_delta must be finite and positive")

        if self.obstacle_strategy == "robust":
            if self.disturbance_trajectories is None:
                raise ValueError("robust paper QP requires full-horizon disturbance_trajectories")
            if self.safety_margin_delta is None:
                raise ValueError("robust paper QP requires a positive safety_margin_delta")

    @staticmethod
    def _wrap_to_pi(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _recovery_control(self, x_current, ref_trajectory, obstacles=None):
        """Fallback tracking policy used when QP solve fails.

        This avoids persistent near-zero-speed stalling from repeatedly reusing
        the previous control under solver failures.
        """
        ref0 = np.asarray(ref_trajectory[min(0, len(ref_trajectory) - 1)], dtype=float)
        v_cur = float(x_current[base.IDX_V])
        psi_cur = float(x_current[base.IDX_PSI])
        v_ref = float(ref0[base.IDX_V])
        psi_ref = float(ref0[base.IDX_PSI])

        dx = float(x_current[0] - ref0[0])
        dy = float(x_current[1] - ref0[1])
        e_lat = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
        e_psi = self._wrap_to_pi(psi_ref - psi_cur)

        a_cmd = self.fallback_speed_gain * (v_ref - v_cur)
        # Slow down when far from lane center to prevent boundary divergence.
        a_cmd -= 0.15 * max(0.0, abs(e_lat) - 2.5)
        delta_cmd = self.fallback_heading_gain * e_psi - self.fallback_lateral_gain * e_lat

        # If the QP is failing near obstacles, bias the fallback away from the
        # nearest obstacle instead of continuing to track the centerline.
        if obstacles and self.track is not None:
            px = float(x_current[0])
            py = float(x_current[1])
            try:
                _, ego_s, ego_lat = self.track.closest_point(px, py)
                total_length = float(self.track.total_length())
            except Exception:
                ego_s = None
                ego_lat = None
                total_length = None

            nearest_obs = None
            for obs in obstacles:
                ox, oy, radius = obs
                dist = float(np.hypot(px - float(ox), py - float(oy)))
                if nearest_obs is None or dist < nearest_obs[0]:
                    nearest_obs = (dist, float(ox), float(oy), float(radius))

            if nearest_obs is not None:
                dist, ox, oy, radius = nearest_obs
                try:
                    _, obs_s, obs_lat = self.track.closest_point(ox, oy)
                    if ego_s is not None and total_length is not None:
                        forward_gap = (float(obs_s) - float(ego_s)) % total_length
                    else:
                        forward_gap = dist
                except Exception:
                    obs_lat = 0.0
                    forward_gap = dist

                safety_margin = radius + base.VEHICLE_RADIUS + base.D_SAFE + 4.0
                if forward_gap <= 60.0 or dist <= safety_margin:
                    away_sign = -np.sign(obs_lat) if obs_lat != 0.0 else 0.0
                    if away_sign == 0.0 and ego_lat is not None:
                        away_sign = -np.sign(float(ego_lat))
                    if away_sign == 0.0:
                        away_sign = -1.0

                    distance_scale = max(dist - radius, 1.0)
                    delta_cmd += away_sign * (1.0 + 3.0 / distance_scale)

                    # Brake harder when an obstacle is ahead so the vehicle has
                    # time to turn before the fallback path reaches the edge.
                    desired_speed_cap = min(v_ref, max(0.8, 1.35 - 0.02 * min(forward_gap, 30.0)))
                    a_cmd -= 0.25 * max(0.0, v_cur - desired_speed_cap)

        a_cmd = float(np.clip(a_cmd, base.A_MIN, base.A_MAX))
        delta_cmd = float(np.clip(delta_cmd, -base.DELTA_MAX, base.DELTA_MAX))

        # Respect steering rate to avoid abrupt flips in fallback mode.
        delta_prev = float(self.u_prev[1])
        max_step = float(base.DELTA_RATE_MAX * base.DT)
        delta_cmd = float(np.clip(delta_cmd, delta_prev - max_step, delta_prev + max_step))

        return np.array([a_cmd, delta_cmd], dtype=float)

    def _qpsol_options(self):
        if self.qp_solver == "osqp":
            return {"osqp": {"verbose": False}, "print_time": False}
        if self.qp_solver == "qpoases":
            return {"printLevel": "none", "print_time": False}
        if self.qp_solver == "qrqp":
            return {"print_header": False, "print_iter": False, "print_info": False, "print_time": False}
        return {"print_time": False}

    @contextmanager
    def _suppress_solver_output(self):
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)

    def _var(self, name, n, decision_blocks):
        x = ca.SX.sym(name, int(n))
        decision_blocks.append(x)
        return x

    def _physical_disturbance_samples_z(self, w_phys_samples):
        samples_z = np.zeros((len(w_phys_samples), self.A.shape[0]), dtype=float)
        if len(w_phys_samples) == 0:
            return samples_z
        samples_z[:, 0] = w_phys_samples[:, 0] / self._px_std
        samples_z[:, 1] = w_phys_samples[:, 1] / self._py_std
        samples_z[:, base.IDX_PSI] = w_phys_samples[:, base.IDX_PSI]
        samples_z[:, base.IDX_V] = w_phys_samples[:, base.IDX_V]
        samples_z[:, base.IDX_OMEGA] = w_phys_samples[:, base.IDX_OMEGA]
        return samples_z

    def _disturbance_state_sensitivity(self, ref_psi, ref_speed, px_std, py_std, T):
        """Map the full lifted disturbance sequence to each predicted lifted state."""
        n_z = self.A.shape[0]
        sensitivities = np.zeros((T + 1, n_z, n_z * T), dtype=float)

        for t in range(T):
            transition = np.array(self.A, dtype=float, copy=True)
            if self.linear_physical_consistency:
                psi_ref_t = float(ref_psi[t])
                v_ref_t = max(float(ref_speed[t]), 0.1)

                transition[0, :] = 0.0
                transition[0, 0] = 1.0
                transition[0, base.IDX_V] = base.DT * np.cos(psi_ref_t) / px_std
                transition[0, base.IDX_PSI] = (
                    -base.DT * v_ref_t * np.sin(psi_ref_t) / px_std
                )
                transition[1, :] = 0.0
                transition[1, 1] = 1.0
                transition[1, base.IDX_V] = base.DT * np.sin(psi_ref_t) / py_std
                transition[1, base.IDX_PSI] = (
                    base.DT * v_ref_t * np.cos(psi_ref_t) / py_std
                )
                transition[2, :] = 0.0
                transition[2, base.IDX_PSI] = 1.0
                transition[3, :] = 0.0
                transition[3, base.IDX_V] = 1.0
                transition[4, :] = 0.0

            sensitivities[t + 1] = transition @ sensitivities[t]
            block = slice(t * n_z, (t + 1) * n_z)
            sensitivities[t + 1, :, block] += np.eye(n_z)

        return sensitivities

    def _select_nearby_obstacles(self, x_current, obstacles):
        px = float(x_current[0])
        py = float(x_current[1])
        preview_distance = max(float(base.OBSTACLE_PROXIMITY), 80.0)

        if self.track is None:
            return [
                obs for obs in obstacles
                if np.hypot(px - float(obs[0]), py - float(obs[1])) < preview_distance
            ]

        try:
            _, ego_s, _ = self.track.closest_point(px, py)
            total_length = float(self.track.total_length())
        except Exception:
            return [
                obs for obs in obstacles
                if np.hypot(px - float(obs[0]), py - float(obs[1])) < preview_distance
            ]

        nearby_obstacles = []
        for obs in obstacles:
            ox, oy, _ = obs
            if np.hypot(px - float(ox), py - float(oy)) >= preview_distance:
                continue

            try:
                _, obs_s, _ = self.track.closest_point(float(ox), float(oy))
                forward_gap = (float(obs_s) - float(ego_s)) % total_length
            except Exception:
                nearby_obstacles.append(obs)
                continue

            if forward_gap > 0.5 * total_length:
                continue
            if forward_gap > preview_distance:
                continue

            nearby_obstacles.append(obs)

        return nearby_obstacles

    def _obstacle_guidance_profile(self, ref_trajectory, obstacles, T):
        lateral_targets = [None] * (T + 1)
        speed_caps = [None] * (T + 1)

        if self.track is None or not obstacles:
            return lateral_targets, speed_caps

        safe_track_lat = max(0.0, base.TRACK_HALF_WIDTH - base.VEHICLE_RADIUS - 0.5)
        total_length = float(self.track.total_length())

        for tc in range(1, T + 1):
            ref_idx = min(max(tc - 1, 0), len(ref_trajectory) - 1)
            ref_x = float(ref_trajectory[ref_idx, 0])
            ref_y = float(ref_trajectory[ref_idx, 1])
            try:
                _, ref_s, _ = self.track.closest_point(ref_x, ref_y)
            except Exception:
                continue

            target_candidates = []
            cap_candidates = []
            for obs in obstacles:
                ox, oy, radius = obs
                try:
                    _, obs_s, obs_lat = self.track.closest_point(float(ox), float(oy))
                except Exception:
                    continue

                forward_gap = (float(obs_s) - float(ref_s)) % total_length
                if forward_gap > 0.5 * total_length:
                    continue

                if forward_gap > 70.0:
                    continue

                if self._obstacle_active_at_step(ref_trajectory, tc, obs):
                    lateral_bias_margin = 0.8
                    if obs_lat >= 0.0:
                        target_lat = float(obs_lat - radius - lateral_bias_margin)
                    else:
                        target_lat = float(obs_lat + radius + lateral_bias_margin)
                    target_lat = float(np.clip(target_lat, -safe_track_lat, safe_track_lat))
                    target_candidates.append(target_lat)

                cap_candidates.append(self.obstacle_speed_cap_base + self.obstacle_speed_cap_gain * max(forward_gap, 0.0))

            if target_candidates:
                lateral_targets[tc] = min(target_candidates) if min(target_candidates) < 0.0 else max(target_candidates)
                speed_caps[tc] = float(max(base.MIN_SPEED_FLOOR, min(cap_candidates)))

        return lateral_targets, speed_caps

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        if u_prev is not None:
            self.u_prev = np.asarray(u_prev, dtype=float)

        T = self.prediction_horizon
        t_start = time.time()
        z0 = self._encode_state(np.asarray(x_current, dtype=float))
        n_z = self.A.shape[0]

        nearby_obstacles = self._select_nearby_obstacles(x_current, obstacles)
        obstacle_lateral_targets, obstacle_speed_caps = self._obstacle_guidance_profile(
            ref_trajectory,
            nearby_obstacles,
            T,
        )

        check_steps = list(range(1, T + 1))
        w_phys_samples = self._physical_disturbance_samples()
        w_z_samples = self._physical_disturbance_samples_z(w_phys_samples)
        track_tightening = self._track_uncertainty_tightening(w_phys_samples, T)

        ref_px_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory) - 1), 0] - self.norm_params["px_mean"])
            / self.norm_params["px_std"]
            for t in range(T)
        ])
        ref_py_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory) - 1), 1] - self.norm_params["py_mean"])
            / self.norm_params["py_std"]
            for t in range(T)
        ])
        ref_psi = np.array([
            ref_trajectory[min(t, len(ref_trajectory) - 1), base.IDX_PSI]
            for t in range(T)
        ])
        y_ref = np.array([
            [
                ref_trajectory[min(t, len(ref_trajectory) - 1), base.IDX_V],
                ref_trajectory[min(t, len(ref_trajectory) - 1), base.IDX_OMEGA],
            ]
            for t in range(T)
        ])

        for t in range(T):
            obstacle_speed_cap = obstacle_speed_caps[t + 1]
            if obstacle_speed_cap is not None:
                y_ref[t, 0] = min(y_ref[t, 0], obstacle_speed_cap)

        px_std = float(self.norm_params["px_std"])
        py_std = float(self.norm_params["py_std"])
        px_mean = float(self.norm_params["px_mean"])
        py_mean = float(self.norm_params["py_mean"])

        disturbance_sensitivities = None
        standardized_residual_samples = None
        residual_scale_full = None
        if self.obstacle_strategy == "robust":
            disturbance_sensitivities = self._disturbance_state_sensitivity(
                ref_psi, y_ref[:, 0], px_std, py_std, T
            )
            residual_scale_full = np.tile(self.residual_scale, T)
            standardized_residual_samples = (
                self.disturbance_trajectories / self.residual_scale[None, None, :]
            ).reshape(len(self.disturbance_trajectories), -1)
            residual_state_samples = np.einsum(
                "tnm,im->itn",
                disturbance_sensitivities,
                self.disturbance_trajectories.reshape(len(self.disturbance_trajectories), -1),
            )

        decision_blocks = []
        U = self._var("U", base.N_U * T, decision_blocks)
        v_slack = self._var("v_slack", T, decision_blocks)
        track_slack = self._var("track_slack", T + 1, decision_blocks)
        heading_slack = self._var("heading_slack", T + 1, decision_blocks)

        obs_slack = None
        active_obstacle_pairs = []
        for j, obs in enumerate(nearby_obstacles):
            for tc in check_steps:
                if self._obstacle_active_at_step(ref_trajectory, tc, obs):
                    active_obstacle_pairs.append((j, tc))

        if active_obstacle_pairs:
            obs_slack = self._var("obs_slack", len(active_obstacle_pairs), decision_blocks)

        s_vars = {}
        if self.obstacle_strategy == "robust" and active_obstacle_pairs:
            for pair in active_obstacle_pairs:
                s_vars[pair] = self._var(
                    f"s_{pair[0]}_{pair[1]}",
                    len(self.disturbance_trajectories),
                    decision_blocks,
                )

        x_dec = ca.vertcat(*decision_blocks)

        A_ca = ca.SX(self.A)
        B_ca = ca.SX(self.B)
        E_ca = ca.SX(self.E)
        F_ca = ca.SX(self.F)
        D_pos_ca = ca.SX(self.D_pos)
        D_psi_ca = ca.SX(np.array([[1.0 if i == base.IDX_PSI else 0.0 for i in range(n_z)]]))

        Z = [ca.SX(z0)]
        for t in range(T):
            u_t = U[base.N_U * t: base.N_U * (t + 1)]
            z_koop = ca.mtimes(A_ca, Z[-1]) + ca.mtimes(B_ca, u_t)
            if self.linear_physical_consistency:
                psi_ref_t = float(ref_psi[t])
                v_ref_t = max(float(y_ref[t, 0]), 0.1)
                psi_dev = Z[-1][base.IDX_PSI] - psi_ref_t
                px_step_phys = (
                    np.cos(psi_ref_t) * Z[-1][base.IDX_V]
                    - v_ref_t * np.sin(psi_ref_t) * psi_dev
                )
                py_step_phys = (
                    np.sin(psi_ref_t) * Z[-1][base.IDX_V]
                    + v_ref_t * np.cos(psi_ref_t) * psi_dev
                )
                px_next = Z[-1][0] + base.DT * px_step_phys / self._px_std
                py_next = Z[-1][1] + base.DT * py_step_phys / self._py_std
                v_next = Z[-1][base.IDX_V] + base.DT * u_t[0]
                omega_next = (v_ref_t / base.L_WHEELBASE) * u_t[1]
                psi_next = Z[-1][base.IDX_PSI] + base.DT * omega_next
                Z.append(ca.vertcat(px_next, py_next, psi_next, v_next, omega_next, z_koop[5:n_z]))
            else:
                Z.append(z_koop)

        Q = ca.SX(base.Q_WEIGHTS)
        r_weights = np.array(base.R_WEIGHTS, dtype=float, copy=True)
        r_abs_weights = np.array(base.R_ABS_WEIGHTS, dtype=float, copy=True)
        r_weights[1, 1] *= self.steering_rate_weight_scale
        r_abs_weights[1, 1] *= self.steering_abs_weight_scale
        R = ca.SX(r_weights)
        R_abs = ca.SX(r_abs_weights)

        cost = 0
        for t in range(T):
            u_t = U[base.N_U * t: base.N_U * (t + 1)]
            if t == 0:
                du = u_t - ca.SX(self.u_prev)
            else:
                u_prev_t = U[base.N_U * (t - 1): base.N_U * t]
                du = u_t - u_prev_t

            if self.obstacle_strategy == "robust":
                scenario_tracking_cost = 0
                for sample_idx in range(len(self.disturbance_trajectories)):
                    z_sample = Z[t] + ca.SX(residual_state_samples[sample_idx, t])
                    y_sample = ca.vertcat(ca.mtimes(E_ca, z_sample), ca.mtimes(F_ca, z_sample))
                    y_error = y_sample - ca.SX(y_ref[t])
                    scenario_tracking_cost += ca.mtimes([y_error.T, Q, y_error])
                cost += scenario_tracking_cost / float(len(self.disturbance_trajectories))
            else:
                y_t = ca.vertcat(ca.mtimes(E_ca, Z[t]), ca.mtimes(F_ca, Z[t]))
                y_err = y_t - ca.SX(y_ref[t])
                cost += ca.mtimes([y_err.T, Q, y_err])
            cost += ca.mtimes([du.T, R, du])
            cost += ca.mtimes([u_t.T, R_abs, u_t])
            if self.steering_curvature_weight > 0.0 and t >= 1:
                prev_delta = U[base.N_U * (t - 1) + 1]
                prev_prev_delta = float(self.u_prev[1]) if t == 1 else U[base.N_U * (t - 2) + 1]
                delta_curvature = u_t[1] - 2.0 * prev_delta + prev_prev_delta
                cost += self.steering_curvature_weight * (delta_curvature ** 2)
            cost -= self.progress_reward_weight * Z[t + 1][base.IDX_V]

            psi_err = ca.mtimes(D_psi_ca, Z[t])[0] - float(ref_psi[t])
            cost += base.Q_PSI * psi_err ** 2

            pos = ca.mtimes(D_pos_ca, Z[t])
            pos_err_x = pos[0] - float(ref_px_norm[t])
            pos_err_y = pos[1] - float(ref_py_norm[t])
            cost += base.Q_POS_TRACK * (pos_err_x ** 2 + pos_err_y ** 2)

            # Direct centerline anchoring in physical meters helps prevent
            # low-speed lateral drift accumulation in long runs.
            d_lat_t = -np.sin(float(ref_psi[t])) * ((pos[0] - float(ref_px_norm[t])) * px_std) + np.cos(float(ref_psi[t])) * ((pos[1] - float(ref_py_norm[t])) * py_std)
            cost += self.lateral_centering_weight * (d_lat_t ** 2)

            obstacle_target_lat = obstacle_lateral_targets[t + 1]
            if obstacle_target_lat is not None:
                cost += self.obstacle_lateral_bias_weight * ((d_lat_t - obstacle_target_lat) ** 2)

            if self.use_worst_case_cost and self.obstacle_strategy == "robust" and self.N_samples > 0:
                sample_cost = 0
                for i in range(self.N_samples):
                    z_dist = Z[t] + ca.SX(w_z_samples[i])
                    y_dist = ca.vertcat(ca.mtimes(E_ca, z_dist), ca.mtimes(F_ca, z_dist))
                    y_dist_err = y_dist - ca.SX(y_ref[t])
                    sample_cost += ca.mtimes([y_dist_err.T, Q, y_dist_err])
                cost += self.dr_cost_weight * sample_cost / float(self.N_samples)

            if self.obstacle_strategy == "robust" and self.robust_obstacle_speed_weight > 0.0:
                if any(self._obstacle_speed_cost_active_at_step(ref_trajectory, t + 1, obs) for obs in nearby_obstacles):
                    cost += self.robust_obstacle_speed_weight * (Z[t + 1][base.IDX_V] ** 2)
                    if self.robust_obstacle_accel_weight > 0.0:
                        cost += self.robust_obstacle_accel_weight * (u_t[0] ** 2)

        terminal = Z[T]
        terminal_pos = ca.mtimes(D_pos_ca, terminal)
        cost += self.terminal_weight_scale * base.Q_TERMINAL_POS * (
            (terminal_pos[0] - float(ref_px_norm[-1])) ** 2 +
            (terminal_pos[1] - float(ref_py_norm[-1])) ** 2
        )
        terminal_psi_err = ca.mtimes(D_psi_ca, terminal)[0] - float(ref_psi[-1])
        cost += self.terminal_weight_scale * base.Q_TERMINAL_HEADING * terminal_psi_err ** 2
        cost += self.speed_slack_penalty * ca.dot(v_slack, v_slack)
        track_slack_penalty = self.track_slack_penalty
        if self.obstacle_strategy == "robust" and self.robust_track_tightening:
            track_slack_penalty *= self.robust_track_slack_penalty_scale
        cost += track_slack_penalty * ca.dot(track_slack, track_slack)
        cost += self.heading_slack_penalty * ca.dot(heading_slack, heading_slack)
        if obs_slack is not None:
            obs_penalty = float(self.obstacle_slack_penalty)
            if self.hard_obstacle_constraints:
                obs_penalty *= self.hard_constraint_slack_scale
            cost += obs_penalty * ca.dot(obs_slack, obs_slack)

        g_expr = []
        lbg = []
        ubg = []

        def add_bound(expr, lb, ub):
            g_expr.append(expr)
            lbg.append(float(lb))
            ubg.append(float(ub))

        for t in range(T):
            u_t = U[base.N_U * t: base.N_U * (t + 1)]
            add_bound(u_t[0], base.A_MIN, base.A_MAX)
            add_bound(u_t[1], -base.DELTA_MAX, base.DELTA_MAX)
            if t == 0:
                delta_rate = u_t[1] - float(self.u_prev[1])
            else:
                delta_rate = u_t[1] - U[base.N_U * (t - 1) + 1]
            add_bound(delta_rate, -base.DELTA_RATE_MAX * base.DT, base.DELTA_RATE_MAX * base.DT)
            add_bound(v_slack[t], 0.0, ca.inf)
            min_speed_floor = base.MIN_SPEED_FLOOR
            if self.obstacle_strategy == "robust":
                min_speed_floor = max(min_speed_floor, self.robust_min_speed_floor)
            v_floor = max(min_speed_floor, base.MIN_SPEED_REF_RATIO * float(y_ref[t, 0]))
            add_bound(Z[t][base.IDX_V] + v_slack[t], v_floor, ca.inf)

            obstacle_speed_cap = obstacle_speed_caps[t + 1]
            if obstacle_speed_cap is not None:
                add_bound(Z[t + 1][base.IDX_V], -ca.inf, obstacle_speed_cap)

        for t in range(T + 1):
            add_bound(track_slack[t], 0.0, self.track_slack_max)
            add_bound(heading_slack[t], 0.0, self.heading_slack_max)
            ref_idx = min(t, T - 1)
            psi_ref = float(ref_psi[ref_idx])
            pos = ca.mtimes(D_pos_ca, Z[t])
            dx_m = (pos[0] - float(ref_px_norm[ref_idx])) * px_std
            dy_m = (pos[1] - float(ref_py_norm[ref_idx])) * py_std
            d_lat = -np.sin(psi_ref) * dx_m + np.cos(psi_ref) * dy_m
            track_limit_t = max(
                base.VEHICLE_RADIUS,
                base.TRACK_HALF_WIDTH - float(track_tightening[t]),
            )
            add_bound(d_lat - track_slack[t], -ca.inf, track_limit_t)
            add_bound(d_lat + track_slack[t], -track_limit_t, ca.inf)

            # Linear adaptive speed cap vs. lateral error magnitude:
            # v <= v_base + k * (track_limit - |d_lat|)
            # implemented via two affine inequalities to keep QP convex.
            v_cap = self.lateral_speed_cap_base + self.lateral_speed_cap_gain * track_limit_t
            add_bound(
                Z[t][base.IDX_V] + self.lateral_speed_cap_gain * d_lat,
                -ca.inf,
                v_cap,
            )
            add_bound(
                Z[t][base.IDX_V] - self.lateral_speed_cap_gain * d_lat,
                -ca.inf,
                v_cap,
            )

            # Keep heading within a reasonable envelope to reduce turn-around behavior.
            psi_err = Z[t][base.IDX_PSI] - psi_ref
            add_bound(psi_err - heading_slack[t], -ca.inf, self.heading_error_limit)
            add_bound(psi_err + heading_slack[t], -self.heading_error_limit, ca.inf)

        obs_slack_idx = 0
        for pair in active_obstacle_pairs:
            j, tc = pair
            obs = nearby_obstacles[j]
            H_obs, h_obs = self._select_obstacle_halfspace(x_current, obs)
            pos = ca.mtimes(D_pos_ca, Z[tc])
            pos_phys = ca.vertcat(pos[0] * px_std + px_mean, pos[1] * py_std + py_mean)
            l_nom = ca.dot(ca.SX(H_obs), pos_phys) - float(h_obs)

            if self.obstacle_strategy == "robust":
                s = s_vars[pair]
                pos_sensitivity = (
                    np.diag([px_std, py_std])
                    @ self.D_pos
                    @ disturbance_sensitivities[tc]
                )
                c_raw = np.asarray(H_obs, dtype=float) @ pos_sensitivity
                c_scaled = c_raw * residual_scale_full
                c_norm = float(np.linalg.norm(c_scaled, ord=2))
                sample_offsets = standardized_residual_samples @ c_scaled
                for i in range(len(self.disturbance_trajectories)):
                    add_bound(s[i], 0.0, ca.inf)
                    add_bound(
                        s[i] - l_nom - float(sample_offsets[i]) - self.safety_margin_delta,
                        0.0,
                        ca.inf,
                    )
                risk_upper = (
                    self.theta * c_norm
                    + ca.sum1(s) / float(len(self.disturbance_trajectories))
                )
                if obs_slack is not None:
                    if self.hard_obstacle_constraints:
                        add_bound(obs_slack[obs_slack_idx], 0.0, 0.0)
                    else:
                        add_bound(obs_slack[obs_slack_idx], 0.0, ca.inf)
                    add_bound(
                        risk_upper - obs_slack[obs_slack_idx],
                        -ca.inf,
                        self.safety_margin_delta * self.epsilon,
                    )
                    obs_slack_idx += 1
                else:
                    add_bound(risk_upper, -ca.inf, self.safety_margin_delta * self.epsilon)
            else:
                if obs_slack is not None:
                    if self.hard_obstacle_constraints:
                        add_bound(obs_slack[obs_slack_idx], 0.0, 0.0)
                    else:
                        add_bound(obs_slack[obs_slack_idx], 0.0, ca.inf)
                    add_bound(l_nom - obs_slack[obs_slack_idx], -ca.inf, 0.0)
                    obs_slack_idx += 1
                else:
                    add_bound(l_nom, -ca.inf, 0.0)

        g = ca.vertcat(*g_expr) if g_expr else ca.SX.zeros(0, 1)
        qp = {"x": x_dec, "f": cost, "g": g}
        arg = {"lbg": np.array(lbg), "ubg": np.array(ubg)}
        if self._qp_warm_start is not None and len(self._qp_warm_start) == int(x_dec.numel()):
            arg["x0"] = self._qp_warm_start

        status = "optimal"
        debug_info = None
        try:
            with self._suppress_solver_output():
                solver = ca.qpsol("paper_qp", self.qp_solver, qp, self._qpsol_options())
                sol = solver(**arg)
                stats = solver.stats()
            x_opt = np.asarray(sol["x"]).reshape(-1)
            self._qp_warm_start = x_opt.copy()
            u_opt = x_opt[0:base.N_U].copy()
            if not bool(stats.get("success", True)):
                status = str(stats.get("return_status", "suboptimal"))
            debug_info = {
                "horizon": {
                    "qp_solver": self.qp_solver,
                    "decision_vars": int(x_dec.numel()),
                    "linear_constraints": int(g.numel()),
                    "active_obstacle_pairs": len(active_obstacle_pairs),
                    "drcc_trajectory_samples": int(self.disturbance_sample_count) if self.obstacle_strategy == "robust" else 0,
                    "track_tightening_max": float(np.max(track_tightening)) if len(track_tightening) else 0.0,
                    "objective": float(sol["f"]),
                },
                "active_constraints": [],
                "u0": [float(u_opt[0]), float(u_opt[1])],
                "v_slack_max": float(np.max(x_opt[base.N_U * T: base.N_U * T + T])) if T else 0.0,
                "obs_slack_max": 0.0,
            }
        except Exception as exc:
            u_opt = self._recovery_control(x_current, ref_trajectory, nearby_obstacles)
            status = f"failed: {str(exc)[:2000]}"
            self._qp_warm_start = None
            debug_info = {
                "fallback": {
                    "mode": "recovery_tracking",
                    "u0": [float(u_opt[0]), float(u_opt[1])],
                }
            }

        u_opt = np.asarray(u_opt, dtype=float).reshape(-1)
        u_opt[0] = np.clip(u_opt[0], base.A_MIN, base.A_MAX)
        u_opt[1] = np.clip(u_opt[1], -base.DELTA_MAX, base.DELTA_MAX)
        self.u_prev = u_opt.copy()

        solve_time = time.time() - t_start
        if not hasattr(self, "_solve_count"):
            self._solve_count = 0
        self._solve_count += 1
        if self._solve_count % 50 == 0:
            gc.collect()

        return u_opt, {
            "solve_time": solve_time,
            "status": status,
            "method": self.name,
            "theta": self.theta,
            "epsilon": self.epsilon,
            "obstacle_strategy": self.obstacle_strategy,
            "paper_controller_version": self.paper_controller_version,
            "paper_formulation": "convex_qp_affine_koopman",
            "paper_obstacle_model": f"preselected_halfspace_{self.halfspace_mode}",
            "paper_risk_model": "wasserstein_shifted_hinge_markov_sufficient_condition",
            "qp_solver": self.qp_solver,
            "linear_physical_consistency": self.linear_physical_consistency,
            "hard_obstacle_constraints": self.hard_obstacle_constraints,
            "obstacle_safety_scale": self.obstacle_safety_scale,
            "obstacle_activation_margin": self.obstacle_activation_margin,
            "robust_track_tightening": self.robust_track_tightening,
            "robust_track_tightening_scale": self.robust_track_tightening_scale,
            "robust_track_tightening_max": self.robust_track_tightening_max,
            "debug": debug_info,
        }
