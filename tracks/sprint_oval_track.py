"""Compact, gentle sprint oval for rapid closed-loop validation."""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
import config as _config


class SprintOvalTrack(BaseTrack):
    """Compact stadium-like oval with mild corner curvature."""

    def __init__(self, num_points=420, straight_length=180.0, turn_radius=70.0):
        super().__init__()
        self._target_points = num_points
        self._straight_length = float(straight_length)
        self._turn_radius = float(turn_radius)
        self._build_track()

    def _build_track(self):
        n_line = self._target_points // 4
        n_arc = self._target_points // 4
        half_straight = 0.5 * self._straight_length
        radius = self._turn_radius

        # Top straight: left -> right
        top_x = np.linspace(-half_straight, half_straight, n_line, endpoint=False)
        top_y = np.full_like(top_x, radius)

        # Right semicircle: top -> bottom
        theta_r = np.linspace(np.pi / 2, -np.pi / 2, n_arc, endpoint=False)
        right_x = half_straight + radius * np.cos(theta_r)
        right_y = radius * np.sin(theta_r)

        # Bottom straight: right -> left
        bot_x = np.linspace(half_straight, -half_straight, n_line, endpoint=False)
        bot_y = np.full_like(bot_x, -radius)

        # Left semicircle: bottom -> top
        theta_l = np.linspace(-np.pi / 2, np.pi / 2, self._target_points - 2 * n_line - n_arc, endpoint=False)
        left_x = -half_straight - radius * np.cos(theta_l)
        left_y = radius * np.sin(theta_l)

        x = np.concatenate([top_x, right_x, bot_x, left_x])
        y = np.concatenate([top_y, right_y, bot_y, left_y])

        # Rotate the oval to make the long axis vertical.
        x_rot = -y
        y_rot = x

        # Start from a short side (arc) so heading changes appear early.
        # The right arc in the unrotated frame becomes the top short side after rotation.
        start_idx = n_line
        self._centerline_x = np.roll(x_rot, -start_idx)
        self._centerline_y = np.roll(y_rot, -start_idx)

        self._compute_geometry(self._centerline_x, self._centerline_y)
        self._rect_obstacles = []
        if _config.ENABLE_OBSTACLES:
            self._obstacles = [
                (px, py, _config.OBSTACLE_RADIUS)
                for px, py in _config.OBSTACLE_POSITIONS
            ]
        else:
            self._obstacles = []

        print(
            f"Sprint Oval Track: {self._total_length:.0f}m, "
            f"{self._num_points} points, {len(self._obstacles)} obstacles"
        )