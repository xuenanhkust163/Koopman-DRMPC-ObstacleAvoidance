"""Linearized bicycle-model MPC strategy for the shared v2 track context."""

import time

import casadi as ca
import numpy as np

from config import (
    A_MAX,
    A_MIN,
    D_SAFE,
    DELTA_MAX,
    DELTA_RATE_MAX,
    DT,
    IPOPT_MAX_ITER,
    IPOPT_PRINT_LEVEL,
    L_WHEELBASE,
    TRACK_HALF_WIDTH,
    T_HORIZON,
    V_MAX,
    V_MIN,
    VEHICLE_RADIUS,
)
from vehicle.bicycle_model import linearize

from .solve_strategies import PurePursuitSolveStrategy


class LinearizedMPCSolveStrategy:
    """Re-linearize RK4 bicycle dynamics along a shared reference and solve an MPC NLP."""

    def __init__(self, horizon=T_HORIZON):
        self.horizon = int(horizon)
        self.u_prev = np.zeros(2, dtype=float)
        self._warm_start = None
        self._fallback = PurePursuitSolveStrategy()

    def reset(self):
        self.u_prev = np.zeros(2, dtype=float)
        self._warm_start = None
        self._fallback.reset()

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
        curvature = float(np.arctan2(
            np.sin(np.arctan2(next_y - y, next_x - x) - np.arctan2(y - previous_y, x - previous_x)),
            np.cos(np.arctan2(next_y - y, next_x - x) - np.arctan2(y - previous_y, x - previous_x)),
        ))
        return x, y, heading, curvature, speed

    def _reference(self, context, x_current):
        _, current_s, _ = context.track.closest_point(x_current[0], x_current[1])
        reference = np.zeros((self.horizon + 1, 5), dtype=float)
        arc_positions = np.zeros(self.horizon + 1, dtype=float)
        arc_positions[0] = current_s
        for index in range(1, self.horizon + 1):
            target_speed = context.target_speed_at_s(arc_positions[index - 1])
            arc_positions[index] = arc_positions[index - 1] + max(target_speed, 1.0) * DT

        for index, arc_position in enumerate(arc_positions):
            x, y, heading, curvature, speed = self._path_point(context, arc_position)
            reference[index] = [x, y, heading, speed, speed * curvature]
        return reference, arc_positions

    def _nominal_controls(self, reference):
        controls = np.zeros((self.horizon, 2), dtype=float)
        for index in range(self.horizon):
            controls[index, 0] = (reference[index + 1, 3] - reference[index, 3]) / DT
            controls[index, 1] = np.arctan(L_WHEELBASE * reference[index, 4] / max(reference[index, 3], 0.5))
        controls[:, 0] = np.clip(controls[:, 0], A_MIN, A_MAX)
        controls[:, 1] = np.clip(controls[:, 1], -DELTA_MAX, DELTA_MAX)
        return controls

    def _nearby_rectangles(self, context, current_s, current_speed):
        horizon_distance = max(current_speed, 1.0) * DT * self.horizon + 25.0
        nearby = []
        for rectangle in context._rect_obstacles:
            cx, cy, length, _, _ = rectangle
            _, obstacle_s, _ = context.track.closest_point(cx, cy)
            relative_s = (obstacle_s - current_s + 0.5 * context._lap_length) % context._lap_length
            relative_s -= 0.5 * context._lap_length
            if abs(relative_s) <= horizon_distance + 0.5 * length:
                nearby.append(rectangle)
        return nearby

    def solve(self, context, x_current, ref_trajectory, obstacles, u_prev=None):
        del ref_trajectory, obstacles
        if u_prev is not None:
            self.u_prev = np.asarray(u_prev, dtype=float).copy()

        started = time.perf_counter()
        reference, arc_positions = self._reference(context, x_current)
        nominal_controls = self._nominal_controls(reference)
        opti = ca.Opti()
        states = opti.variable(5, self.horizon + 1)
        controls = opti.variable(2, self.horizon)
        opti.subject_to(states[:, 0] == x_current)

        for index in range(self.horizon):
            A, B, nominal_next = linearize(reference[index], nominal_controls[index])
            affine = nominal_next - A @ reference[index] - B @ nominal_controls[index]
            predicted_next = ca.mtimes(ca.DM(A), states[:, index])
            predicted_next += ca.mtimes(ca.DM(B), controls[:, index]) + ca.DM(affine)
            opti.subject_to(states[:, index + 1] == predicted_next)

        position_weights = ca.diag(ca.DM([2.0, 2.0, 12.0, 2.0, 1.0]))
        input_weights = ca.diag(ca.DM([0.15, 0.3]))
        rate_weights = ca.diag(ca.DM([0.25, 0.8]))
        cost = 0
        for index in range(1, self.horizon + 1):
            state_error = states[:, index] - ca.DM(reference[index])
            heading_error = ca.atan2(ca.sin(state_error[2]), ca.cos(state_error[2]))
            state_error[2] = heading_error
            cost += ca.mtimes([state_error.T, position_weights, state_error])

        for index in range(self.horizon):
            control_error = controls[:, index] - ca.DM(nominal_controls[index])
            cost += ca.mtimes([control_error.T, input_weights, control_error])
            previous = ca.DM(self.u_prev) if index == 0 else controls[:, index - 1]
            control_delta = controls[:, index] - previous
            cost += ca.mtimes([control_delta.T, rate_weights, control_delta])
            opti.subject_to(opti.bounded(A_MIN, controls[0, index], A_MAX))
            opti.subject_to(opti.bounded(-DELTA_MAX, controls[1, index], DELTA_MAX))
            opti.subject_to(opti.bounded(
                -DELTA_RATE_MAX * DT,
                control_delta[1],
                DELTA_RATE_MAX * DT,
            ))

        for index in range(self.horizon + 1):
            opti.subject_to(opti.bounded(V_MIN, states[3, index], V_MAX))
            base_x, base_y, base_heading, _, _ = context._sample_path(arc_positions[index])
            lateral_error = (
                -np.sin(base_heading) * (states[0, index] - base_x)
                + np.cos(base_heading) * (states[1, index] - base_y)
            )
            road_limit = TRACK_HALF_WIDTH - VEHICLE_RADIUS - 0.2
            opti.subject_to(opti.bounded(-road_limit, lateral_error, road_limit))

        _, current_s, _ = context.track.closest_point(x_current[0], x_current[1])
        nearby_rectangles = self._nearby_rectangles(context, current_s, float(x_current[3]))
        required_clearance = VEHICLE_RADIUS + D_SAFE
        for cx, cy, length, width, angle in nearby_rectangles:
            cosine, sine = float(np.cos(angle)), float(np.sin(angle))
            for index in range(1, self.horizon + 1):
                dx = states[0, index] - cx
                dy = states[1, index] - cy
                local_x = cosine * dx + sine * dy
                local_y = -sine * dx + cosine * dy
                outside_x = ca.fmax(ca.fabs(local_x) - 0.5 * length, 0.0)
                outside_y = ca.fmax(ca.fabs(local_y) - 0.5 * width, 0.0)
                opti.subject_to(
                    outside_x**2 + outside_y**2 >= required_clearance**2
                )

        options = {
            "ipopt.max_iter": IPOPT_MAX_ITER,
            "ipopt.print_level": IPOPT_PRINT_LEVEL,
            "ipopt.tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            "print_time": False,
        }
        opti.solver("ipopt", options)
        if self._warm_start is not None:
            try:
                opti.set_initial(states, self._warm_start["states"])
                opti.set_initial(controls, self._warm_start["controls"])
            except Exception:
                pass
        else:
            opti.set_initial(states, reference.T)
            opti.set_initial(controls, nominal_controls.T)

        try:
            solution = opti.solve()
            control = np.asarray(solution.value(controls[:, 0]), dtype=float).reshape(2)
            self._warm_start = {
                "states": solution.value(states),
                "controls": solution.value(controls),
            }
            status = "optimal"
        except Exception as error:
            control, fallback_info = self._fallback.solve(
                context,
                x_current,
                None,
                None,
                u_prev=self.u_prev,
            )
            status = f"fallback_pp: {str(error).splitlines()[0][:80]}"
            self.u_prev = control.copy()
            return control, {
                "solve_time": time.perf_counter() - started,
                "status": status,
                "method": "LMPCBaseline",
                "debug": {"fallback_status": fallback_info["status"]},
            }

        self.u_prev = control.copy()
        return control, {
            "solve_time": time.perf_counter() - started,
            "status": status,
            "method": "LMPCBaseline",
            "debug": {"nearby_obstacles": len(nearby_rectangles)},
        }