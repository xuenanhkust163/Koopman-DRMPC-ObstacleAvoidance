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

from config import A_LAT_MAX, REF_SPEED_SCALE
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

    def _tracking_error_metrics(self, x_current, ref_trajectory):
        ref0 = np.asarray(ref_trajectory[min(0, len(ref_trajectory) - 1)], dtype=float)
        dx = float(x_current[0] - ref0[0])
        dy = float(x_current[1] - ref0[1])
        psi_ref = float(ref0[base.IDX_PSI])
        e_lat = -np.sin(psi_ref) * dx + np.cos(psi_ref) * dy
        e_psi = np.arctan2(
            np.sin(float(x_current[base.IDX_PSI]) - psi_ref),
            np.cos(float(x_current[base.IDX_PSI]) - psi_ref),
        )
        return {
            'abs_e_lat': abs(float(e_lat)),
            'abs_e_psi': abs(float(e_psi)),
            'v_ref': float(ref0[base.IDX_V]),
            'v_cur': float(x_current[base.IDX_V]),
        }

    @staticmethod
    def _wrap_angle(angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _reference_preview_metrics(self, ref_trajectory):
        if len(ref_trajectory) == 0:
            return {
                'preview_heading_change_deg': 0.0,
                'preview_heading_rate_deg': 0.0,
                'preview_curvature_max': 0.0,
                'preview_curvature_mean': 0.0,
                'preview_steps': 0,
            }

        preview_steps = min(len(ref_trajectory), 10)
        psi0 = float(ref_trajectory[0, base.IDX_PSI])
        heading_offsets = [
            abs(np.rad2deg(self._wrap_angle(float(ref_trajectory[t, base.IDX_PSI]) - psi0)))
            for t in range(preview_steps)
        ]
        curvature_samples = []
        for t in range(preview_steps):
            v_ref_t = abs(float(ref_trajectory[t, base.IDX_V]))
            omega_ref_t = abs(float(ref_trajectory[t, base.IDX_OMEGA]))
            if v_ref_t > 1e-4:
                curvature_samples.append(omega_ref_t / v_ref_t)
            else:
                curvature_samples.append(0.0)
        max_heading_change_deg = float(max(heading_offsets)) if heading_offsets else 0.0
        heading_rate_deg = max_heading_change_deg / max(float(preview_steps - 1), 1.0)
        preview_curvature_max = float(max(curvature_samples)) if curvature_samples else 0.0
        preview_curvature_mean = float(np.mean(curvature_samples)) if curvature_samples else 0.0
        return {
            'preview_heading_change_deg': max_heading_change_deg,
            'preview_heading_rate_deg': float(heading_rate_deg),
            'preview_curvature_max': preview_curvature_max,
            'preview_curvature_mean': preview_curvature_mean,
            'preview_steps': int(preview_steps),
        }

    def _track_lookahead_metrics(self, x_current, lookahead_distance=25.0):
        if self.track is None:
            return {
                'track_lookahead_curvature_max': 0.0,
                'track_lookahead_curvature_p90': 0.0,
                'track_lookahead_curvature_mean': 0.0,
                'track_lookahead_safe_speed': float(REF_SPEED_SCALE * base.V_MAX),
                'track_lookahead_distance': 0.0,
            }

        try:
            _, current_s, _ = self.track.closest_point(float(x_current[0]), float(x_current[1]))
            arc = self.track.get_arc_length()
            curvature = np.abs(self.track.get_curvature())
            total_length = float(self.track.total_length())
        except Exception:
            return {
                'track_lookahead_curvature_max': 0.0,
                'track_lookahead_curvature_p90': 0.0,
                'track_lookahead_curvature_mean': 0.0,
                'track_lookahead_safe_speed': float(REF_SPEED_SCALE * base.V_MAX),
                'track_lookahead_distance': 0.0,
            }

        current_s = float(current_s)
        lookahead_distance = float(min(max(lookahead_distance, 0.0), total_length))
        end_s = current_s + lookahead_distance

        if end_s <= total_length:
            mask = (arc >= current_s) & (arc <= end_s)
        else:
            wrapped_end = end_s - total_length
            mask = (arc >= current_s) | (arc <= wrapped_end)

        segment_curvature = curvature[mask]
        if len(segment_curvature) == 0:
            idx = int(np.argmin(np.abs(arc - current_s)))
            segment_curvature = curvature[idx:idx + 1]

        max_curvature = float(np.max(segment_curvature))
        if max_curvature > 1e-6:
            lookahead_safe_speed = float(
                REF_SPEED_SCALE * min(base.V_MAX, np.sqrt(A_LAT_MAX / max_curvature))
            )
        else:
            lookahead_safe_speed = float(REF_SPEED_SCALE * base.V_MAX)

        return {
            'track_lookahead_curvature_max': max_curvature,
            'track_lookahead_curvature_p90': float(np.percentile(segment_curvature, 90)),
            'track_lookahead_curvature_mean': float(np.mean(segment_curvature)),
            'track_lookahead_safe_speed': lookahead_safe_speed,
            'track_lookahead_distance': lookahead_distance,
        }

    def _nearest_forward_obstacle_gap(self, x_current, nearby_obstacles):
        if not nearby_obstacles:
            return None

        px = float(x_current[0])
        py = float(x_current[1])

        if self.track is not None:
            try:
                _, ego_s, _ = self.track.closest_point(px, py)
                total_length = float(self.track.total_length())
                forward_gaps = []
                for ox, oy, _ in nearby_obstacles:
                    _, obs_s, _ = self.track.closest_point(float(ox), float(oy))
                    gap = (float(obs_s) - float(ego_s)) % total_length
                    if gap <= 0.5 * total_length:
                        forward_gaps.append(gap)
                if forward_gaps:
                    return float(min(forward_gaps))
            except Exception:
                pass

        euclidean_gaps = [
            float(np.hypot(px - float(ox), py - float(oy)))
            for ox, oy, _ in nearby_obstacles
        ]
        return float(min(euclidean_gaps)) if euclidean_gaps else None

    def _online_strategy_summary(self, x_current, ref_trajectory, nearby_obstacles):
        metrics = self._tracking_error_metrics(x_current, ref_trajectory)
        preview_metrics = self._reference_preview_metrics(ref_trajectory)
        lookahead_metrics = self._track_lookahead_metrics(x_current)
        forward_gap = self._nearest_forward_obstacle_gap(x_current, nearby_obstacles)

        lat_risk = np.clip((metrics['abs_e_lat'] - 0.4) / 1.8, 0.0, 1.0)
        psi_risk = np.clip(
            (metrics['abs_e_psi'] - np.deg2rad(4.0)) / np.deg2rad(14.0),
            0.0,
            1.0,
        )
        obstacle_risk = 0.0
        if forward_gap is not None:
            obstacle_risk = float(np.clip((12.0 - forward_gap) / 12.0, 0.0, 1.0))

        heading_curve_risk = float(np.clip(
            (preview_metrics['preview_heading_change_deg'] - 6.0) / 16.0,
            0.0,
            1.0,
        ))
        curvature_curve_risk = float(np.clip(
            (preview_metrics['preview_curvature_max'] - 0.0105) / 0.0055,
            0.0,
            1.0,
        ))
        lookahead_speed_excess = max(
            0.0,
            metrics['v_cur'] - (lookahead_metrics['track_lookahead_safe_speed'] + 0.15),
        )
        lookahead_curve_risk = float(np.clip(
            lookahead_speed_excess / 1.2,
            0.0,
            1.0,
        ))
        immediate_curve_risk = max(heading_curve_risk, curvature_curve_risk)
        curve_risk = max(immediate_curve_risk, 0.75 * lookahead_curve_risk)

        alignment_risk = float(np.clip(
            0.70 * curve_risk + 0.25 * lat_risk + 0.20 * psi_risk,
            0.0,
            1.2,
        ))

        combined_risk = float(np.clip(
            0.40 * lat_risk + 0.20 * psi_risk + 0.55 * obstacle_risk + 0.55 * curve_risk,
            0.0,
            1.5,
        ))

        min_floor_scale = 1.0 - 0.55 * combined_risk
        min_speed_floor = max(
            base.MIN_SPEED_FLOOR,
            self.robust_min_speed_floor * max(0.35, min_floor_scale),
        )
        effective_preview = max(
            self.robust_obstacle_speed_preview,
            4.0 + 6.0 * obstacle_risk + 2.0 * lat_risk,
        )

        curve_speed_scale = max(
            0.58,
            1.0 - 0.34 * immediate_curve_risk - 0.18 * lookahead_curve_risk - 0.12 * lat_risk - 0.08 * psi_risk,
        )
        curve_speed_cap = max(
            min_speed_floor,
            metrics['v_ref'] * curve_speed_scale + 0.20,
        )
        curve_speed_cap = min(
            curve_speed_cap,
            lookahead_metrics['track_lookahead_safe_speed'] + 0.15,
        )
        curve_accel_penalty = (
            10.0 * max(0.0, lookahead_curve_risk - 0.25)
            + 18.0 * max(0.0, immediate_curve_risk - 0.10)
        ) * (1.0 + 0.5 * lat_risk)
        curve_control_steps = min(
            len(ref_trajectory),
            max(3, int(preview_metrics['preview_steps']))
        )
        curve_speed_buffer = max(0.10, 0.28 - 0.10 * immediate_curve_risk - 0.08 * lookahead_curve_risk)
        curve_hard_speed_cap = max(min_speed_floor, curve_speed_cap + curve_speed_buffer)

        curve_accel_cap = float(base.A_MAX)
        if lookahead_curve_risk > 0.25:
            curve_accel_cap = min(curve_accel_cap, 0.15)
        if immediate_curve_risk > 0.18:
            curve_accel_cap = min(curve_accel_cap, 0.20 - 0.60 * min(1.0, immediate_curve_risk))
        if immediate_curve_risk > 0.60 or alignment_risk > 0.75:
            severe_curve_risk = max(immediate_curve_risk, alignment_risk)
            curve_accel_cap = min(curve_accel_cap, -0.10 - 0.45 * min(1.0, severe_curve_risk))

        return {
            'lat_risk': float(lat_risk),
            'psi_risk': float(psi_risk),
            'obstacle_risk': float(obstacle_risk),
            'immediate_curve_risk': float(immediate_curve_risk),
            'lookahead_curve_risk': float(lookahead_curve_risk),
            'curve_risk': curve_risk,
            'alignment_risk': alignment_risk,
            'combined_risk': combined_risk,
            'forward_gap': forward_gap,
            'min_speed_floor': float(min_speed_floor),
            'obstacle_speed_weight': float(
                self.robust_obstacle_speed_weight * (1.0 + 1.25 * combined_risk)
            ),
            'obstacle_accel_weight': float(
                self.robust_obstacle_accel_weight * (1.0 + 1.00 * combined_risk)
            ),
            'obstacle_speed_preview': float(effective_preview),
            'curve_speed_cap': float(curve_speed_cap),
            'curve_hard_speed_cap': float(curve_hard_speed_cap),
            'curve_accel_penalty': float(curve_accel_penalty),
            'curve_accel_cap': float(curve_accel_cap),
            'curve_control_steps': int(curve_control_steps),
            'preview_heading_change_deg': float(preview_metrics['preview_heading_change_deg']),
            'preview_heading_rate_deg': float(preview_metrics['preview_heading_rate_deg']),
            'preview_curvature_max': float(preview_metrics['preview_curvature_max']),
            'preview_curvature_mean': float(preview_metrics['preview_curvature_mean']),
            'track_lookahead_curvature_max': float(lookahead_metrics['track_lookahead_curvature_max']),
            'track_lookahead_curvature_p90': float(lookahead_metrics['track_lookahead_curvature_p90']),
            'track_lookahead_curvature_mean': float(lookahead_metrics['track_lookahead_curvature_mean']),
            'track_lookahead_safe_speed': float(lookahead_metrics['track_lookahead_safe_speed']),
            'track_lookahead_distance': float(lookahead_metrics['track_lookahead_distance']),
            'lookahead_speed_excess': float(lookahead_speed_excess),
            'curve_preview_steps': int(preview_metrics['preview_steps']),
            'abs_e_lat': float(metrics['abs_e_lat']),
            'abs_e_psi': float(metrics['abs_e_psi']),
        }

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
        online_strategy = self._online_strategy_summary(x_current, ref_trajectory, nearby_obstacles)

        if online_strategy['curve_risk'] > 0.0:
            preview_steps = max(1, min(T, int(online_strategy['curve_preview_steps'])))
            for t in range(preview_steps):
                taper = 1.0 - 0.55 * float(t) / float(preview_steps)
                guided_speed_cap = max(
                    online_strategy['min_speed_floor'],
                    online_strategy['curve_speed_cap'] + 0.25 * taper,
                )
                y_ref[t, 0] = min(y_ref[t, 0], guided_speed_cap)

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
            min_speed_floor = max(min_speed_floor, online_strategy['min_speed_floor'])
        min_speed_rule = base.MinSpeedRule(
            floor_abs=min_speed_floor,
            floor_ratio=base.MIN_SPEED_REF_RATIO,
        )

        risk_terms = []
        diag_terms = {}
        dr_empirical_cost = 0
        robust_obstacle_speed_cost = 0
        robust_obstacle_accel_cost = 0
        curve_positive_accel_cost = 0

        active_obstacle_steps = set()
        if (
            self.obstacle_strategy == "robust" and
            self.robust_obstacle_speed_weight > 0.0 and
            n_obs > 0
        ):
            adaptive_preview = online_strategy['obstacle_speed_preview']
            for tc in check_steps:
                if any(
                    self._obstacle_active_at_step(ref_trajectory, tc, obs)
                    or self._obstacle_speed_cost_active_at_step(ref_trajectory, tc, obs)
                    for obs in nearby_obstacles
                ):
                    if adaptive_preview <= self.robust_obstacle_speed_preview:
                        active_obstacle_steps.add(tc)
                        continue

                    ref_idx = min(max(tc - 1, 0), len(ref_trajectory) - 1)
                    ref_x = float(ref_trajectory[ref_idx, 0])
                    ref_y = float(ref_trajectory[ref_idx, 1])
                    if self.track is not None:
                        try:
                            _, ref_s, _ = self.track.closest_point(ref_x, ref_y)
                            total_length = float(self.track.total_length())
                            close_ahead = False
                            for ox, oy, _ in nearby_obstacles:
                                _, obs_s, _ = self.track.closest_point(float(ox), float(oy))
                                forward_gap = (float(obs_s) - float(ref_s)) % total_length
                                if forward_gap <= 0.5 * total_length and forward_gap <= adaptive_preview:
                                    close_ahead = True
                                    break
                            if close_ahead:
                                active_obstacle_steps.add(tc)
                                continue
                        except Exception:
                            pass

                    if any(
                        np.hypot(ref_x - float(ox), ref_y - float(oy)) <= adaptive_preview
                        for ox, oy, _ in nearby_obstacles
                    ):
                        active_obstacle_steps.add(tc)

        for t in range(T):
            y_t = ca.vertcat(
                ca.mtimes(self._E_ca, Z[t]),
                ca.mtimes(self._F_ca, Z[t]),
            )
            if t < max(1, int(online_strategy['curve_preview_steps'])):
                curve_positive_accel_cost += ca.fmax(U[0, t], 0.0) ** 2
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
            cost += online_strategy['obstacle_speed_weight'] * robust_obstacle_speed_cost
            diag_terms['cost_robust_obstacle_speed'] = (
                online_strategy['obstacle_speed_weight'] * robust_obstacle_speed_cost
            )
            cost += online_strategy['obstacle_accel_weight'] * robust_obstacle_accel_cost
            diag_terms['cost_robust_obstacle_accel'] = (
                online_strategy['obstacle_accel_weight'] * robust_obstacle_accel_cost
            )

        if online_strategy['curve_accel_penalty'] > 0.0:
            cost += online_strategy['curve_accel_penalty'] * curve_positive_accel_cost
            diag_terms['cost_curve_positive_accel'] = (
                online_strategy['curve_accel_penalty'] * curve_positive_accel_cost
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

            if (
                online_strategy['curve_risk'] > 0.20 and
                t < int(online_strategy['curve_control_steps'])
            ):
                opti.subject_to(U[0, t] <= float(online_strategy['curve_accel_cap']))
                opti.subject_to(Z[t + 1][base.IDX_V] <= float(online_strategy['curve_hard_speed_cap']))

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
            debug_info['horizon']['robust_obstacle_speed_weight'] = online_strategy['obstacle_speed_weight']
            debug_info['horizon']['robust_obstacle_speed_preview'] = online_strategy['obstacle_speed_preview']
            debug_info['horizon']['robust_obstacle_accel_weight'] = online_strategy['obstacle_accel_weight']
            debug_info['horizon']['curve_speed_cap'] = online_strategy['curve_speed_cap']
            debug_info['horizon']['curve_hard_speed_cap'] = online_strategy['curve_hard_speed_cap']
            debug_info['horizon']['curve_accel_cap'] = online_strategy['curve_accel_cap']
            debug_info['horizon']['curve_control_steps'] = online_strategy['curve_control_steps']
            debug_info['horizon']['online_strategy'] = online_strategy
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
            'robust_min_speed_floor': online_strategy['min_speed_floor'],
            'robust_obstacle_speed_weight': online_strategy['obstacle_speed_weight'],
            'robust_obstacle_speed_preview': online_strategy['obstacle_speed_preview'],
            'robust_obstacle_accel_weight': online_strategy['obstacle_accel_weight'],
            'curve_speed_cap': online_strategy['curve_speed_cap'],
            'curve_hard_speed_cap': online_strategy['curve_hard_speed_cap'],
            'curve_accel_cap': online_strategy['curve_accel_cap'],
            'curve_control_steps': online_strategy['curve_control_steps'],
            'safe_suboptimal_fallback': self.safe_suboptimal_fallback,
            'active_obstacle_speed_steps': len(active_obstacle_steps),
            'track_tightening_max': float(np.max(track_tightening)) if len(track_tightening) else 0.0,
            'online_strategy': online_strategy,
            'debug': debug_info,
        }