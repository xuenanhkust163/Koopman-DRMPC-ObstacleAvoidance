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
        **kwargs,
    ):
        engineering_config = self._resolve_engineering_config(engineering_config, kwargs)
        super().__init__(*args, **kwargs)
        self.name = "K-DRMPC-Paper-QP"
        self.paper_controller_version = "halfspace-drcc-convex-qp-v1"
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
        self.lateral_speed_cap_gain = float(self.engineering_config.lateral_speed_cap_gain)
        self.lateral_speed_cap_base = float(self.engineering_config.lateral_speed_cap_base)
        self.linear_physical_consistency = bool(self.engineering_config.linear_physical_consistency)
        self.fallback_speed_gain = float(self.engineering_config.fallback_speed_gain)
        self.fallback_heading_gain = float(self.engineering_config.fallback_heading_gain)
        self.fallback_lateral_gain = float(self.engineering_config.fallback_lateral_gain)
        self._qp_warm_start = None

    @staticmethod
    def _wrap_to_pi(angle):
        return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi

    def _recovery_control(self, x_current, ref_trajectory):
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

    def _select_nearby_obstacles(self, x_current, obstacles):
        px = float(x_current[0])
        py = float(x_current[1])

        if self.track is None:
            return [
                obs for obs in obstacles
                if np.hypot(px - float(obs[0]), py - float(obs[1])) < base.OBSTACLE_PROXIMITY
            ]

        try:
            _, ego_s, _ = self.track.closest_point(px, py)
            total_length = float(self.track.total_length())
        except Exception:
            return [
                obs for obs in obstacles
                if np.hypot(px - float(obs[0]), py - float(obs[1])) < base.OBSTACLE_PROXIMITY
            ]

        nearby_obstacles = []
        for obs in obstacles:
            ox, oy, _ = obs
            if np.hypot(px - float(ox), py - float(oy)) >= base.OBSTACLE_PROXIMITY:
                continue

            try:
                _, obs_s, _ = self.track.closest_point(float(ox), float(oy))
                forward_gap = (float(obs_s) - float(ego_s)) % total_length
            except Exception:
                nearby_obstacles.append(obs)
                continue

            if forward_gap > 0.5 * total_length:
                continue
            if forward_gap > base.OBSTACLE_PROXIMITY:
                continue

            nearby_obstacles.append(obs)

        return nearby_obstacles

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        if u_prev is not None:
            self.u_prev = np.asarray(u_prev, dtype=float)

        T = self.prediction_horizon
        t_start = time.time()
        z0 = self._encode_state(np.asarray(x_current, dtype=float))
        n_z = self.A.shape[0]

        nearby_obstacles = self._select_nearby_obstacles(x_current, obstacles)

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

        px_std = float(self.norm_params["px_std"])
        py_std = float(self.norm_params["py_std"])
        px_mean = float(self.norm_params["px_mean"])
        py_mean = float(self.norm_params["py_mean"])

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

        lambda_vars = {}
        s_vars = {}
        if self.obstacle_strategy == "robust" and active_obstacle_pairs:
            for pair in active_obstacle_pairs:
                lambda_vars[pair] = self._var(f"lam_{pair[0]}_{pair[1]}", 1, decision_blocks)
                s_vars[pair] = self._var(f"s_{pair[0]}_{pair[1]}", self.N_samples, decision_blocks)

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
                px_next = Z[-1][0] + base.DT * np.cos(psi_ref_t) * Z[-1][base.IDX_V] / self._px_std
                py_next = Z[-1][1] + base.DT * np.sin(psi_ref_t) * Z[-1][base.IDX_V] / self._py_std
                v_next = Z[-1][base.IDX_V] + base.DT * u_t[0]
                omega_next = (v_ref_t / base.L_WHEELBASE) * u_t[1]
                psi_next = Z[-1][base.IDX_PSI] + base.DT * omega_next
                Z.append(ca.vertcat(px_next, py_next, psi_next, v_next, omega_next, z_koop[5:n_z]))
            else:
                Z.append(z_koop)

        Q = ca.SX(base.Q_WEIGHTS)
        R = ca.SX(base.R_WEIGHTS)
        R_abs = ca.SX(base.R_ABS_WEIGHTS)

        cost = 0
        for t in range(T):
            u_t = U[base.N_U * t: base.N_U * (t + 1)]
            if t == 0:
                du = u_t - ca.SX(self.u_prev)
            else:
                u_prev_t = U[base.N_U * (t - 1): base.N_U * t]
                du = u_t - u_prev_t

            y_t = ca.vertcat(ca.mtimes(E_ca, Z[t]), ca.mtimes(F_ca, Z[t]))
            y_err = y_t - ca.SX(y_ref[t])
            cost += ca.mtimes([y_err.T, Q, y_err])
            cost += ca.mtimes([du.T, R, du])
            cost += ca.mtimes([u_t.T, R_abs, u_t])
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
                lam = lambda_vars[pair][0]
                s = s_vars[pair]
                add_bound(lam, 0.0, ca.inf)
                c_norm = float(np.linalg.norm(H_obs, ord=2))
                for i in range(self.N_samples):
                    add_bound(s[i], 0.0, ca.inf)
                    sample_loss_i = float(np.dot(H_obs, w_phys_samples[i, 0:2]))
                    add_bound(s[i] - l_nom - sample_loss_i - lam * c_norm, 0.0, ca.inf)
                risk_upper = lam * self.theta + ca.sum1(s) / float(max(self.N_samples, 1))
                if obs_slack is not None:
                    add_bound(obs_slack[obs_slack_idx], 0.0, ca.inf)
                    add_bound(risk_upper - obs_slack[obs_slack_idx], -ca.inf, self.epsilon)
                    obs_slack_idx += 1
                else:
                    add_bound(risk_upper, -ca.inf, self.epsilon)
            else:
                if obs_slack is not None:
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
                    "track_tightening_max": float(np.max(track_tightening)) if len(track_tightening) else 0.0,
                    "objective": float(sol["f"]),
                },
                "active_constraints": [],
                "u0": [float(u_opt[0]), float(u_opt[1])],
                "v_slack_max": float(np.max(x_opt[base.N_U * T: base.N_U * T + T])) if T else 0.0,
                "obs_slack_max": 0.0,
            }
        except Exception as exc:
            u_opt = self._recovery_control(x_current, ref_trajectory)
            status = f"failed: {str(exc)[:80]}"
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
            "paper_risk_model": "wasserstein_per_step_halfspace_drcc_qp",
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
