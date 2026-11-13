"""Lightweight Pure Pursuit baseline for obstacle-free and obstacle courses."""

import numpy as np

from config import (
    A_MAX,
    A_MIN,
    DELTA_MAX,
    DELTA_RATE_MAX,
    DT,
    L_WHEELBASE,
    TRACK_HALF_WIDTH,
    V_MAX,
    VEHICLE_RADIUS,
)
from .solve_strategies import PurePursuitSolveStrategy


def _wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class PurePursuitBaselineController:
    """Follow a closed track with curvature-aware speed planning and obstacle offsets."""

    def __init__(
        self,
        track,
        max_speed=8.0,
        lateral_acceleration=2.5,
        acceleration=1.8,
        braking=2.8,
        lookahead_base=5.0,
        lookahead_time=0.8,
        speed_gain=1.4,
        steering_rate_limit=None,
        solve_strategy=None,
    ):
        self.name = "PurePursuitBaseline"
        self.track = track
        self.max_speed = min(float(max_speed), float(V_MAX))
        self.lateral_acceleration = float(lateral_acceleration)
        self.acceleration = float(acceleration)
        self.braking = float(braking)
        self.lookahead_base = float(lookahead_base)
        self.lookahead_time = float(lookahead_time)
        self.speed_gain = float(speed_gain)
        self.steering_rate_limit = (
            float(steering_rate_limit)
            if steering_rate_limit is not None
            else float(DELTA_RATE_MAX)
        )

        self._x, self._y = track.get_centerline()
        self._arc = track.get_arc_length()
        self._curvature = track.get_curvature()
        self._rect_obstacles = list(getattr(track, "_rect_obstacles", []))
        closing_distance = float(np.hypot(self._x[0] - self._x[-1], self._y[0] - self._y[-1]))
        self._lap_length = float(self._arc[-1] + closing_distance)
        self._speed_profile = self._build_speed_profile()
        self._previous_delta = 0.0
        self.solve_strategy = solve_strategy or PurePursuitSolveStrategy()

    def reset(self):
        self._previous_delta = 0.0
        self.solve_strategy.reset()

    def _build_speed_profile(self):
        curvature = np.abs(self._curvature)
        curvature = np.convolve(
            np.pad(curvature, (4, 4), mode="wrap"),
            np.ones(9, dtype=float) / 9.0,
            mode="valid",
        )
        speed = np.minimum(
            self.max_speed,
            np.sqrt(self.lateral_acceleration / np.maximum(curvature, 1e-4)),
        )

        segment_lengths = np.diff(np.append(self._arc, self._lap_length))
        segment_lengths = np.maximum(segment_lengths, 1e-6)
        for _ in range(3):
            for index in range(len(speed)):
                next_index = (index + 1) % len(speed)
                reachable = np.sqrt(speed[index] ** 2 + 2.0 * self.acceleration * segment_lengths[index])
                speed[next_index] = min(speed[next_index], reachable)
            for index in range(len(speed) - 1, -1, -1):
                next_index = (index + 1) % len(speed)
                brake_reachable = np.sqrt(
                    speed[next_index] ** 2 + 2.0 * self.braking * segment_lengths[index]
                )
                speed[index] = min(speed[index], brake_reachable)
        return speed

    def _sample_path(self, arc_position):
        position = float(arc_position) % self._lap_length
        extended_arc = np.append(self._arc, self._lap_length)
        x_values = np.append(self._x, self._x[0])
        y_values = np.append(self._y, self._y[0])
        heading_values = np.unwrap(np.append(self.track.get_heading(), self.track.get_heading()[0]))
        curvature_values = np.append(self._curvature, self._curvature[0])
        speed_values = np.append(self._speed_profile, self._speed_profile[0])

        x = float(np.interp(position, extended_arc, x_values))
        y = float(np.interp(position, extended_arc, y_values))
        heading = float(np.interp(position, extended_arc, heading_values))
        curvature = float(np.interp(position, extended_arc, curvature_values))
        target_speed = float(np.interp(position, extended_arc, speed_values))
        return x, y, heading, curvature, target_speed

    def target_speed_at_s(self, arc_position):
        """Return the planned speed at an arc-length position."""
        return self._sample_path(arc_position)[4]

    def _obstacle_lateral_offset(self, arc_position):
        """Return a smooth lane offset that clears nearby rectangular obstacles."""
        offset = 0.0
        max_offset = max(TRACK_HALF_WIDTH - VEHICLE_RADIUS - 0.5, 0.0)
        for cx, cy, length, width, angle in self._rect_obstacles:
            _, obstacle_s, obstacle_lat = self.track.closest_point(cx, cy)
            relative_s = (obstacle_s - arc_position + 0.5 * self._lap_length) % self._lap_length
            relative_s -= 0.5 * self._lap_length

            target_offset = -np.sign(obstacle_lat) * max(
                0.0,
                0.5 * float(width) + VEHICLE_RADIUS + 2.0 - abs(obstacle_lat),
            )
            target_offset = float(np.clip(target_offset, -max_offset, max_offset))
            ramp_length = max(22.0, 4.0 * abs(target_offset))
            longitudinal_extent = 0.5 * float(length) + max(30.0, ramp_length + 8.0)
            progress = relative_s + longitudinal_extent
            if progress <= 0.0 or progress >= 2.0 * longitudinal_extent:
                continue

            ramp_in = float(np.clip(progress / ramp_length, 0.0, 1.0))
            ramp_out = float(np.clip((2.0 * longitudinal_extent - progress) / ramp_length, 0.0, 1.0))
            blend = min(ramp_in, ramp_out)
            blend = blend * blend * (3.0 - 2.0 * blend)
            offset += target_offset * blend

        return float(np.clip(offset, -max_offset, max_offset))

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        return self.solve_strategy.solve(
            self,
            x_current,
            ref_trajectory,
            obstacles,
            u_prev=u_prev,
        )