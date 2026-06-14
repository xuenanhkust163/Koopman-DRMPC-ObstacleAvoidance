"""Straight track for controller sanity checks and obstacle experiments."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack


class StraightTrack(BaseTrack):
    """Simple straight closed-loop corridor represented as an out-and-back oval with long straights."""

    def __init__(self, num_points=600, straight_length=600.0, turn_radius=12.0):
        super().__init__()
        self._target_points = int(num_points)
        self._straight_length = float(straight_length)
        self._turn_radius = float(turn_radius)
        self._build_track()

    def _build_track(self):
        n_line = self._target_points // 2
        n_arc = max(20, (self._target_points - 2 * n_line) // 2)
        remaining = self._target_points - 2 * n_line - n_arc
        if remaining < 20:
            remaining = 20
            n_line = max(40, (self._target_points - n_arc - remaining) // 2)

        half_straight = 0.5 * self._straight_length
        radius = self._turn_radius

        top_x = np.linspace(-half_straight, half_straight, n_line, endpoint=False)
        top_y = np.full_like(top_x, radius)

        theta_r = np.linspace(np.pi / 2, -np.pi / 2, n_arc, endpoint=False)
        right_x = half_straight + radius * np.cos(theta_r)
        right_y = radius * np.sin(theta_r)

        bot_x = np.linspace(half_straight, -half_straight, n_line, endpoint=False)
        bot_y = np.full_like(bot_x, -radius)

        theta_l = np.linspace(-np.pi / 2, np.pi / 2, remaining, endpoint=False)
        left_x = -half_straight - radius * np.cos(theta_l)
        left_y = radius * np.sin(theta_l)

        x = np.concatenate([top_x, right_x, bot_x, left_x])
        y = np.concatenate([top_y, right_y, bot_y, left_y])

        start_idx = 0
        self._centerline_x = np.roll(x, -start_idx)
        self._centerline_y = np.roll(y, -start_idx)
        self._compute_geometry(self._centerline_x, self._centerline_y)
        self._obstacles = []
        self._rect_obstacles = []

        print(
            f"Straight Track: {self._total_length:.0f}m, "
            f"{self._num_points} points, {len(self._obstacles)} obstacles"
        )