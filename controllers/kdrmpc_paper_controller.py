"""Parallel paper-faithful Koopman DRMPC controller.

This controller preserves the current runner-facing API while moving the
obstacle-safety layer toward the manuscript formulation:

1. preselected obstacle half-spaces
2. Wasserstein-CVaR auxiliary variables on the half-space safety loss

It intentionally lives beside the practical controller so `main` behavior is
not overwritten.
"""

import gc
import time

import casadi as ca
import numpy as np

from controllers import kdrmpc_controller as base


class KDRMPCPaperController(base.KDRMPCController):
    """Paper branch controller with half-space obstacle reformulation."""

    def __init__(
        self,
        *args,
        obstacle_strategy="robust",
        hard_obstacle_constraints=True,
        halfspace_mode="track-normal",
        track=None,
        obstacle_slack_penalty=None,
        obstacle_safety_scale=1.0,
        obstacle_activation_margin=8.0,
        prediction_horizon=20,
        use_worst_case_cost=True,
        dr_cost_weight=1.0,
        robust_track_tightening=True,
        robust_track_tightening_scale=4.0,
        robust_track_tightening_max=3.0,
        robust_track_slack_penalty_scale=3.0,
        robust_min_speed_floor=1.0,
        robust_obstacle_speed_weight=5.0,
        robust_obstacle_speed_preview=0.0,
        robust_obstacle_accel_weight=80.0,
        safe_suboptimal_fallback=True,
        **kwargs,
    ):
        super().__init__(*args, obstacle_strategy=obstacle_strategy, **kwargs)
        self.name = "K-DRMPC-Paper"
        self.paper_controller_version = "halfspace-drcc-v4"
        self.hard_obstacle_constraints = bool(hard_obstacle_constraints)
        self.halfspace_mode = str(halfspace_mode)
        self.track = track
        self.obstacle_slack_penalty = (
            base.OBSTACLE_SLACK_PENALTY
            if obstacle_slack_penalty is None
            else float(obstacle_slack_penalty)
        )
        self.obstacle_safety_scale = float(obstacle_safety_scale)
        self.obstacle_activation_margin = float(obstacle_activation_margin)
        self.prediction_horizon = int(prediction_horizon)
        self.use_worst_case_cost = bool(use_worst_case_cost)
        self.dr_cost_weight = float(dr_cost_weight)
        self.robust_track_tightening = bool(robust_track_tightening)
        self.robust_track_tightening_scale = float(robust_track_tightening_scale)
        self.robust_track_tightening_max = float(robust_track_tightening_max)
        self.robust_track_slack_penalty_scale = float(robust_track_slack_penalty_scale)
        self.robust_min_speed_floor = float(robust_min_speed_floor)
        self.robust_obstacle_speed_weight = float(robust_obstacle_speed_weight)
        self.robust_obstacle_speed_preview = float(robust_obstacle_speed_preview)
        self.robust_obstacle_accel_weight = float(robust_obstacle_accel_weight)
        self.safe_suboptimal_fallback = bool(safe_suboptimal_fallback)

    def _track_normal_halfspace(self, obstacle, d_min):
        if self.track is None:
            return None

        ox, oy, _ = obstacle
        try:
            idx, _, lat_obs = self.track.closest_point(ox, oy)
            heading = self.track.get_heading()[idx]
        except Exception:
            return None

        side = float(np.sign(lat_obs))
        if side == 0.0:
            return None

        normal = np.array([-np.sin(heading), np.cos(heading)], dtype=float)
        obstacle_projection = float(np.dot(normal, np.array([ox, oy], dtype=float)))
        H_obs = side * normal
        h_obs = side * obstacle_projection - d_min
        return H_obs, h_obs

    def _select_obstacle_halfspace(self, x_current, obstacle):
        """Choose one separating half-space from the ego/obstacle geometry.

        The manuscript states that the safe side is selected before solving the
        MPC problem. The default uses the track normal at the obstacle location,
        keeping the vehicle on the centerline side of edge obstacles. The old
        dominant-axis rule is retained as an explicit fallback/diagnostic mode.
        """
        ox, oy, radius = obstacle
        d_min = self.obstacle_safety_scale * float(radius + base.VEHICLE_RADIUS + base.D_SAFE)

        if self.halfspace_mode == "track-normal":
            track_halfspace = self._track_normal_halfspace(obstacle, d_min)
            if track_halfspace is not None:
                return track_halfspace

        dx = float(x_current[0] - ox)
        dy = float(x_current[1] - oy)

        if abs(dx) >= abs(dy):
            if dx >= 0.0:
                return np.array([-1.0, 0.0]), -(ox + d_min)
            return np.array([1.0, 0.0]), ox - d_min

        if dy >= 0.0:
            return np.array([0.0, -1.0]), -(oy + d_min)
        return np.array([0.0, 1.0]), oy - d_min

    def _physical_disturbance_samples(self):
        """Map empirical generator samples into the simulator's state order.

        DisturbanceGenerator samples are ordered as [px, py, v, psi, omega]
        and the simulator scales them before adding them to the plant state.
        The controller uses the same physical scaling here.
        """
        samples = np.zeros((self.N_samples, base.N_X))
        if self.N_samples == 0:
            return samples
        samples[:, 0] = self.w_samples[:, 0] * 0.1
        samples[:, 1] = self.w_samples[:, 1] * 0.1
        samples[:, base.IDX_PSI] = self.w_samples[:, 3] * 0.01
        samples[:, base.IDX_V] = self.w_samples[:, 2] * 0.05
        samples[:, base.IDX_OMEGA] = self.w_samples[:, 4] * 0.01
        return samples

    def _obstacle_active_at_step(self, ref_trajectory, tc, obstacle):
        ox, oy, radius = obstacle
        ref_idx = min(max(tc - 1, 0), len(ref_trajectory) - 1)
        ref_x = float(ref_trajectory[ref_idx, 0])
        ref_y = float(ref_trajectory[ref_idx, 1])
        activation_radius = (
            self.obstacle_safety_scale * float(radius + base.VEHICLE_RADIUS + base.D_SAFE)
            + self.obstacle_activation_margin
        )
        return np.hypot(ref_x - ox, ref_y - oy) <= activation_radius

    def _obstacle_speed_cost_active_at_step(self, ref_trajectory, tc, obstacle):
        if self.robust_obstacle_speed_preview <= 0.0:
            return self._obstacle_active_at_step(ref_trajectory, tc, obstacle)

        if self.track is None:
            return self._obstacle_active_at_step(ref_trajectory, tc, obstacle)

        ox, oy, _ = obstacle
        ref_idx = min(max(tc - 1, 0), len(ref_trajectory) - 1)
        ref_x = float(ref_trajectory[ref_idx, 0])
        ref_y = float(ref_trajectory[ref_idx, 1])
        try:
            _, ref_s, _ = self.track.closest_point(ref_x, ref_y)
            _, obs_s, _ = self.track.closest_point(ox, oy)
            total_length = float(self.track.total_length())
        except Exception:
            return self._obstacle_active_at_step(ref_trajectory, tc, obstacle)

        forward_gap = (float(obs_s) - float(ref_s)) % total_length
        if forward_gap > 0.5 * total_length:
            return False
        return forward_gap <= self.robust_obstacle_speed_preview

    def _track_uncertainty_tightening(self, w_phys_samples, T):
        """Return per-step track-boundary tightening for robust mode.

        The high-noise failures are track-boundary exits, while the manuscript
        DRCC layer in this branch only constrained obstacle half-spaces. This
        uses the empirical position disturbances to reserve lateral road margin
        over the prediction horizon. The growth is proportional to sqrt(t),
        matching accumulated independent position disturbances without making
        the road constraint infeasible at the first predicted step.
        """
        tightening = np.zeros(T + 1, dtype=float)
        if (
            self.obstacle_strategy != "robust" or
            not self.robust_track_tightening or
            self.N_samples == 0
        ):
            return tightening

        pos_norm = np.linalg.norm(w_phys_samples[:, 0:2], axis=1)
        if len(pos_norm) == 0:
            return tightening

        empirical_radius = float(np.percentile(pos_norm, 95))
        wasserstein_radius = float(max(self.theta, 0.0))
        one_step_radius = empirical_radius + wasserstein_radius
        max_tightening = max(
            0.0,
            min(float(self.robust_track_tightening_max), 0.35 * float(base.TRACK_HALF_WIDTH)),
        )
        for t in range(1, T + 1):
            tightening[t] = min(
                self.robust_track_tightening_scale * one_step_radius * np.sqrt(float(t)),
                max_tightening,
            )
        return tightening

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        if u_prev is not None:
            self.u_prev = u_prev

        T = self.prediction_horizon
        t_start = time.time()
        z0 = self._encode_state(x_current)

        y_ref = np.zeros((T, 2))
        for t in range(T):
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            y_ref[t] = [ref_t[base.IDX_V], ref_t[base.IDX_OMEGA]]

        ref_psi = np.array([
            ref_trajectory[min(t, len(ref_trajectory) - 1), base.IDX_PSI]
            for t in range(T)
        ])

        px, py = x_current[0], x_current[1]
        nearby_obstacles = []
        for obs in obstacles:
            ox, oy, _ = obs
            dist = np.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            if dist < base.OBSTACLE_PROXIMITY:
                nearby_obstacles.append(obs)

        n_obs = len(nearby_obstacles)
        check_steps = list(range(1, T + 1))
        n_check = len(check_steps)

        opti = ca.Opti()
        U = opti.variable(base.N_U, T)

        Z = [ca.DM(z0)]
        n_z = self.A.shape[0]
        for t in range(T):
            z_next = ca.mtimes(self._A_ca, Z[-1]) + ca.mtimes(self._B_ca, U[:, t])

            v_phys = Z[-1][3] + base.DT * U[0, t]
            omega_phys = v_phys * ca.tan(U[1, t]) / base.L_WHEELBASE
            psi_next = Z[-1][2] + base.DT * omega_phys
            px_next = Z[-1][0] + base.DT * v_phys * ca.cos(psi_next) / self._px_std
            py_next = Z[-1][1] + base.DT * v_phys * ca.sin(psi_next) / self._py_std

            z_fixed = ca.vertcat(
                px_next, py_next, psi_next, v_phys, omega_phys, z_next[5:n_z]
            )
            Z.append(z_fixed)

        cost = 0
        Q = ca.DM(base.Q_WEIGHTS)
        R = ca.DM(base.R_WEIGHTS)
        R_abs = ca.DM(base.R_ABS_WEIGHTS)

        v_slack = opti.variable(T)
        opti.subject_to(v_slack >= 0)
        cost += base.MIN_SPEED_SLACK_PENALTY * ca.dot(v_slack, v_slack)

        track_slack = opti.variable(T + 1)
        opti.subject_to(track_slack >= 0)
        track_slack_penalty = base.TRACK_BOUNDARY_SLACK_PENALTY
        if self.obstacle_strategy == "robust" and self.robust_track_tightening:
            track_slack_penalty *= self.robust_track_slack_penalty_scale
        cost += track_slack_penalty * ca.dot(track_slack, track_slack)

        ref_px_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory) - 1), 0] - self.norm_params['px_mean'])
            / self.norm_params['px_std']
            for t in range(T)
        ])
        ref_py_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory) - 1), 1] - self.norm_params['py_mean'])
            / self.norm_params['py_std']
            for t in range(T)
        ])
        ref_psi_terminal = float(ref_trajectory[min(T - 1, len(ref_trajectory) - 1), base.IDX_PSI])
        ref_px_norm_terminal = float(ref_px_norm[T - 1])
        ref_py_norm_terminal = float(ref_py_norm[T - 1])

        px_std = float(self.norm_params['px_std'])
        py_std = float(self.norm_params['py_std'])
        px_mean = float(self.norm_params['px_mean'])
        py_mean = float(self.norm_params['py_mean'])
        w_phys_samples = self._physical_disturbance_samples()
        track_tightening = self._track_uncertainty_tightening(w_phys_samples, T)

        min_speed_floor = base.MIN_SPEED_FLOOR
        if self.obstacle_strategy == "robust":
            min_speed_floor = max(min_speed_floor, self.robust_min_speed_floor)
        min_speed_rule = base.MinSpeedRule(
            floor_abs=min_speed_floor,
            floor_ratio=base.MIN_SPEED_REF_RATIO,
        )

        risk_terms = []
        diag_terms = {}
        dr_empirical_cost = 0
        robust_obstacle_speed_cost = 0
        robust_obstacle_accel_cost = 0

        active_obstacle_steps = set()
        if (
            self.obstacle_strategy == "robust" and
            self.robust_obstacle_speed_weight > 0.0 and
            n_obs > 0
        ):
            for tc in check_steps:
                if any(
                    self._obstacle_speed_cost_active_at_step(ref_trajectory, tc, obs)
                    for obs in nearby_obstacles
                ):
                    active_obstacle_steps.add(tc)

        for t in range(T):
            y_t = ca.vertcat(
                ca.mtimes(self._E_ca, Z[t]),
                ca.mtimes(self._F_ca, Z[t]),
            )
            y_ref_t = ca.DM(y_ref[t])
            cost += self.cost_builder.stage_cost(
                opti=opti,
                t=t,
                z_t=Z[t],
                u_t=U[:, t],
                u_prev=ca.DM(self.u_prev),
                u_prev_step=(U[:, t - 1] if t > 0 else None),
                y_t=y_t,
                y_ref_t=y_ref_t,
                ref_psi_t=float(ref_psi[t]),
                ref_px_norm_t=float(ref_px_norm[t]),
                ref_py_norm_t=float(ref_py_norm[t]),
                d_pos_ca=self._D_pos_ca,
                d_psi_ca=self._D_psi_ca,
                q=Q,
                r=R,
                q_psi=base.Q_PSI,
                q_progress=base.Q_PROGRESS,
                q_pos=base.Q_POS_TRACK,
                add_position_term=(t % base.POSITION_TERM_INTERVAL == 0),
                add_abs_u_term=True,
                r_abs=R_abs,
                min_speed_rule=min_speed_rule,
                v_slack_t=v_slack[t],
                risk_terms=risk_terms,
            )
            self.cost_builder.collect_stage_diagnostics(
                diag_terms,
                opti=opti,
                t=t,
                z_t=Z[t],
                u_t=U[:, t],
                u_prev=ca.DM(self.u_prev),
                u_prev_step=(U[:, t - 1] if t > 0 else None),
                y_t=y_t,
                y_ref_t=y_ref_t,
                ref_psi_t=float(ref_psi[t]),
                ref_px_norm_t=float(ref_px_norm[t]),
                ref_py_norm_t=float(ref_py_norm[t]),
                d_pos_ca=self._D_pos_ca,
                d_psi_ca=self._D_psi_ca,
                q=Q,
                r=R,
                q_psi=base.Q_PSI,
                q_progress=base.Q_PROGRESS,
                q_pos=base.Q_POS_TRACK,
                add_position_term=(t % base.POSITION_TERM_INTERVAL == 0),
                add_abs_u_term=True,
                r_abs=R_abs,
                min_speed_rule=min_speed_rule,
                v_slack_t=v_slack[t],
                risk_terms=risk_terms,
            )

            if self.use_worst_case_cost and self.N_samples > 0:
                stage_dr_cost = 0
                pos_nom = ca.mtimes(self._D_pos_ca, Z[t])
                for i in range(self.N_samples):
                    w_i = w_phys_samples[i]
                    y_disturbed = y_t + ca.DM([w_i[base.IDX_V], w_i[base.IDX_OMEGA]])
                    y_err_disturbed = y_disturbed - y_ref_t
                    stage_dr_cost += ca.mtimes([y_err_disturbed.T, Q, y_err_disturbed])

                    if t % base.POSITION_TERM_INTERVAL == 0:
                        pos_disturbed = pos_nom + ca.DM([w_i[0] / px_std, w_i[1] / py_std])
                        pos_err_x = pos_disturbed[0] - float(ref_px_norm[t])
                        pos_err_y = pos_disturbed[1] - float(ref_py_norm[t])
                        stage_dr_cost += base.Q_POS_TRACK * (pos_err_x ** 2 + pos_err_y ** 2)

                dr_empirical_cost += stage_dr_cost / float(self.N_samples)

        for tc in sorted(active_obstacle_steps):
            robust_obstacle_speed_cost += Z[tc][base.IDX_V] ** 2
            robust_obstacle_accel_cost += ca.fmax(U[0, tc - 1], 0.0) ** 2

        if active_obstacle_steps:
            cost += self.robust_obstacle_speed_weight * robust_obstacle_speed_cost
            diag_terms['cost_robust_obstacle_speed'] = (
                self.robust_obstacle_speed_weight * robust_obstacle_speed_cost
            )
            cost += self.robust_obstacle_accel_weight * robust_obstacle_accel_cost
            diag_terms['cost_robust_obstacle_accel'] = (
                self.robust_obstacle_accel_weight * robust_obstacle_accel_cost
            )

        cost += self.cost_builder.finalize_cost(
            opti=opti,
            horizon=T,
            risk_terms=risk_terms,
            diag_terms=diag_terms,
            z_terminal=Z[T],
            ref_psi_terminal=ref_psi_terminal,
            ref_px_norm_terminal=ref_px_norm_terminal,
            ref_py_norm_terminal=ref_py_norm_terminal,
            d_pos_ca=self._D_pos_ca,
            d_psi_ca=self._D_psi_ca,
            terminal_heading_weight=base.Q_TERMINAL_HEADING,
            terminal_pos_weight=base.Q_TERMINAL_POS,
        )

        if self.use_worst_case_cost:
            cost += self.dr_cost_weight * dr_empirical_cost
            diag_terms['cost_dr_empirical'] = self.dr_cost_weight * dr_empirical_cost

        for t in range(T):
            opti.subject_to(opti.bounded(base.A_MIN, U[0, t], base.A_MAX))
            opti.subject_to(opti.bounded(-base.DELTA_MAX, U[1, t], base.DELTA_MAX))
            if t == 0:
                opti.subject_to(opti.bounded(
                    -base.DELTA_RATE_MAX * base.DT,
                    U[1, t] - self.u_prev[1],
                    base.DELTA_RATE_MAX * base.DT,
                ))
            else:
                opti.subject_to(opti.bounded(
                    -base.DELTA_RATE_MAX * base.DT,
                    U[1, t] - U[1, t - 1],
                    base.DELTA_RATE_MAX * base.DT,
                ))

        for t in range(T + 1):
            ref_idx = min(t, T - 1)
            psi_ref = ref_psi[ref_idx] if t < T else ref_psi_terminal
            pos_t = ca.mtimes(self._D_pos_ca, Z[t])
            dx_m = (pos_t[0] - float(ref_px_norm[ref_idx])) * px_std
            dy_m = (pos_t[1] - float(ref_py_norm[ref_idx])) * py_std
            d_lat = -ca.sin(psi_ref) * dx_m + ca.cos(psi_ref) * dy_m
            track_limit_t = max(
                base.VEHICLE_RADIUS,
                base.TRACK_HALF_WIDTH - float(track_tightening[t]),
            )
            opti.subject_to(d_lat <= track_limit_t + track_slack[t])
            opti.subject_to(-d_lat <= track_limit_t + track_slack[t])

        obs_slack = None
        if n_obs > 0 and n_check > 0:
            n_obs_slack = n_obs * n_check
            if not self.hard_obstacle_constraints:
                obs_slack = opti.variable(n_obs_slack)
                opti.subject_to(obs_slack >= 0)
                cost += self.obstacle_slack_penalty * ca.dot(obs_slack, obs_slack)

            if self.obstacle_strategy == "robust":
                Lambda = {}
                S_vars = {}
                for j in range(n_obs):
                    for tc in check_steps:
                        key = (j, tc)
                        Lambda[key] = opti.variable()
                        opti.subject_to(Lambda[key] >= 0)
                        S_vars[key] = opti.variable(self.N_samples)
                        opti.subject_to(S_vars[key] >= 0)

                obs_slack_idx = 0
                for j, obs in enumerate(nearby_obstacles):
                    H_obs, h_obs = self._select_obstacle_halfspace(x_current, obs)
                    h_row = ca.DM(H_obs.reshape(1, 2))
                    c_norm = float(np.linalg.norm(H_obs, ord=2))

                    for tc in check_steps:
                        if not self._obstacle_active_at_step(ref_trajectory, tc, obs):
                            continue
                        key = (j, tc)
                        lam = Lambda[key]
                        s = S_vars[key]

                        pos = ca.mtimes(self._D_pos_ca, Z[tc])
                        px_phys = pos[0] * px_std + px_mean
                        py_phys = pos[1] * py_std + py_mean
                        pos_phys = ca.vertcat(px_phys, py_phys)
                        l_nom = ca.mtimes(h_row, pos_phys)[0] - float(h_obs)

                        for i in range(self.N_samples):
                            w_pos_i = w_phys_samples[i, 0:2]
                            sample_loss_i = float(np.dot(H_obs, w_pos_i))
                            opti.subject_to(s[i] >= l_nom + sample_loss_i + lam * c_norm)

                        risk_upper = (
                            lam * self.theta +
                            (1.0 / float(self.N_samples)) * ca.sum1(s)
                        )
                        if self.hard_obstacle_constraints:
                            opti.subject_to(risk_upper <= self.epsilon)
                        else:
                            opti.subject_to(risk_upper <= self.epsilon + obs_slack[obs_slack_idx])
                        obs_slack_idx += 1
            else:
                obs_slack_idx = 0
                for obs in nearby_obstacles:
                    H_obs, h_obs = self._select_obstacle_halfspace(x_current, obs)
                    h_row = ca.DM(H_obs.reshape(1, 2))

                    for tc in check_steps:
                        if not self._obstacle_active_at_step(ref_trajectory, tc, obs):
                            continue
                        pos = ca.mtimes(self._D_pos_ca, Z[tc])
                        px_phys = pos[0] * px_std + px_mean
                        py_phys = pos[1] * py_std + py_mean
                        pos_phys = ca.vertcat(px_phys, py_phys)
                        nonrobust_margin = ca.mtimes(h_row, pos_phys)[0] - float(h_obs)
                        if self.hard_obstacle_constraints:
                            opti.subject_to(nonrobust_margin <= 0.0)
                        else:
                            opti.subject_to(nonrobust_margin <= obs_slack[obs_slack_idx])
                        obs_slack_idx += 1

        opti.minimize(cost)

        opts = {
            'ipopt.max_iter': base.IPOPT_MAX_ITER,
            'ipopt.print_level': base.IPOPT_PRINT_LEVEL,
            'print_time': False,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 5,
        }
        opti.solver('ipopt', opts)

        if self._warm_start is not None:
            try:
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                pass

        try:
            sol = opti.solve()
            u_opt = np.array(sol.value(U[:, 0])).flatten()
            status = "optimal"
            debug_info = self._build_debug_info(
                sol=sol,
                U=U,
                T=T,
                diag_terms=diag_terms,
                v_slack=v_slack,
                obs_slack=obs_slack,
            )
            debug_info.setdefault('horizon', {})['active_obstacle_speed_steps'] = len(active_obstacle_steps)
            debug_info['horizon']['robust_obstacle_speed_weight'] = self.robust_obstacle_speed_weight
            debug_info['horizon']['robust_obstacle_speed_preview'] = self.robust_obstacle_speed_preview
            debug_info['horizon']['robust_obstacle_accel_weight'] = self.robust_obstacle_accel_weight
            self._warm_start = {'U': sol.value(U)}
        except Exception as exc:
            try:
                u_opt = np.array(opti.debug.value(U[:, 0])).flatten()
                u_opt[0] = np.clip(u_opt[0], base.A_MIN, base.A_MAX)
                u_opt[1] = np.clip(u_opt[1], -base.DELTA_MAX, base.DELTA_MAX)
                if self.safe_suboptimal_fallback and (n_obs > 0 or len(active_obstacle_steps) > 0):
                    u_opt[0] = min(float(u_opt[0]), 0.0)
                status = "suboptimal"
                debug_info = None
            except Exception:
                u_opt = self.u_prev.copy()
                status = f"failed: {str(exc)[:50]}"
                debug_info = None

        solve_time = time.time() - t_start
        self.u_prev = u_opt.copy()

        try:
            del opti
        except Exception:
            pass

        if not hasattr(self, '_solve_count'):
            self._solve_count = 0
        self._solve_count += 1
        if self._solve_count % 50 == 0:
            gc.collect()

        return u_opt, {
            'solve_time': solve_time,
            'status': status,
            'method': self.name,
            'theta': self.theta,
            'epsilon': self.epsilon,
            'obstacle_strategy': self.obstacle_strategy,
            'paper_controller_version': self.paper_controller_version,
            'paper_obstacle_model': f'preselected_halfspace_{self.halfspace_mode}',
            'paper_risk_model': 'wasserstein_per_step_chance_reformulation',
            'hard_obstacle_constraints': self.hard_obstacle_constraints,
            'obstacle_safety_scale': self.obstacle_safety_scale,
            'obstacle_activation_margin': self.obstacle_activation_margin,
            'robust_track_tightening': self.robust_track_tightening,
            'robust_track_tightening_scale': self.robust_track_tightening_scale,
            'robust_track_tightening_max': self.robust_track_tightening_max,
            'robust_track_slack_penalty_scale': self.robust_track_slack_penalty_scale,
            'robust_min_speed_floor': self.robust_min_speed_floor,
            'robust_obstacle_speed_weight': self.robust_obstacle_speed_weight,
            'robust_obstacle_speed_preview': self.robust_obstacle_speed_preview,
            'robust_obstacle_accel_weight': self.robust_obstacle_accel_weight,
            'safe_suboptimal_fallback': self.safe_suboptimal_fallback,
            'active_obstacle_speed_steps': len(active_obstacle_steps),
            'track_tightening_max': float(np.max(track_tightening)) if len(track_tightening) else 0.0,
            'debug': debug_info,
        }