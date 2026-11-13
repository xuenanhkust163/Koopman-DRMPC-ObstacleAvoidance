"""Interchangeable control-solve strategies for v2 track-following controllers."""

import numpy as np

from config import A_MAX, A_MIN, DELTA_MAX, DT, L_WHEELBASE


def _wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class PurePursuitSolveStrategy:
    """Geometric Pure Pursuit solve policy for a shared track-following context."""

    def reset(self):
        pass

    def solve(self, context, x_current, ref_trajectory, obstacles, u_prev=None):
        del ref_trajectory, obstacles
        if u_prev is not None:
            context._previous_delta = float(u_prev[1])

        _, current_s, _ = context.track.closest_point(x_current[0], x_current[1])
        speed = max(float(x_current[3]), 0.0)
        lookahead = float(np.clip(
            context.lookahead_base + context.lookahead_time * speed,
            context.lookahead_base,
            18.0,
        ))
        target_x, target_y, target_heading, _, _ = context._sample_path(current_s + lookahead)
        lateral_offset = context._obstacle_lateral_offset(current_s + lookahead)
        target_x -= np.sin(target_heading) * lateral_offset
        target_y += np.cos(target_heading) * lateral_offset

        bearing = np.arctan2(target_y - x_current[1], target_x - x_current[0])
        alpha = _wrap_angle(bearing - float(x_current[2]))
        desired_delta = np.arctan2(2.0 * L_WHEELBASE * np.sin(alpha), lookahead)
        desired_delta = float(np.clip(desired_delta, -DELTA_MAX, DELTA_MAX))

        delta_step = context.steering_rate_limit * DT
        delta = float(np.clip(
            desired_delta,
            context._previous_delta - delta_step,
            context._previous_delta + delta_step,
        ))
        delta = float(np.clip(delta, -DELTA_MAX, DELTA_MAX))
        context._previous_delta = delta

        target_speed = context.target_speed_at_s(current_s)
        acceleration = context.speed_gain * (target_speed - speed)
        acceleration = float(np.clip(acceleration, A_MIN, A_MAX))
        return np.array([acceleration, delta], dtype=float), {
            "solve_time": 0.0,
            "status": "pure_pursuit",
            "debug": {
                "target_speed": target_speed,
                "lookahead": lookahead,
                "desired_delta": desired_delta,
                "obstacle_lateral_offset": lateral_offset,
            },
        }