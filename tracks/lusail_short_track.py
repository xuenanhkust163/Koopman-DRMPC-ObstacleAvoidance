"""
Lusail-like short circuit approximation (~2.6 km).

This track keeps the same geometric style as LusailTrack but scales to a
shorter lap length for faster closed-loop validation and analysis.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import OBSTACLE_RADIUS, TRACK_HALF_WIDTH


class LusailShortTrack(BaseTrack):
    """Short Lusail-style circuit (~2.6 km)."""

    def __init__(self, num_points=1200, target_length=2600.0):
        super().__init__()
        self._target_points = num_points
        self._target_length = target_length
        self._build_track()

    def _build_track(self):
        """Build short Lusail-style centerline from the same base waypoints."""
        waypoints = np.array([
            [0, 0],
            [200, 5],
            [400, 15],
            [600, 30],
            [800, 50],
            [1000, 60],
            [1150, 80],
            [1250, 140],
            [1280, 220],
            [1250, 320],
            [1180, 380],
            [1150, 450],
            [1180, 520],
            [1250, 580],
            [1280, 650],
            [1240, 730],
            [1150, 790],
            [1030, 830],
            [900, 820],
            [780, 770],
            [680, 690],
            [620, 600],
            [580, 500],
            [560, 380],
            [510, 300],
            [430, 260],
            [350, 290],
            [300, 360],
            [250, 440],
            [200, 500],
            [120, 540],
            [30, 530],
            [-60, 480],
            [-120, 400],
            [-150, 300],
            [-130, 200],
            [-80, 130],
            [-30, 60],
            [0, 0],
        ], dtype=np.float64)

        raw_length = 0.0
        for i in range(len(waypoints) - 1):
            raw_length += np.linalg.norm(waypoints[i + 1] - waypoints[i])
        scale = self._target_length / raw_length
        waypoints *= scale

        t_param = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            t_param[i] = t_param[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])
        t_param /= t_param[-1]

        cs_x = CubicSpline(t_param, waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t_param, waypoints[:, 1], bc_type='periodic')

        t_fine = np.linspace(0, 1, self._target_points, endpoint=False)
        self._centerline_x = cs_x(t_fine)
        self._centerline_y = cs_y(t_fine)

        self._compute_geometry(self._centerline_x, self._centerline_y)
        self._place_obstacles()

        print(
            f"Lusail Short Track: {self._total_length:.0f}m, "
            f"{self._num_points} points, {len(self._rect_obstacles)} rectangles"
        )

    def _place_obstacles(self):
        """Place rectangular obstacles evenly around the lap while keeping the start clear."""
        N = self._num_points
        curvature = self._curvature

        # Keep the start/finish region clear so the car does not spawn beside an obstacle.
        start_exclusion = max(80, N // 10)
        num_rectangles = 10
        corner_indices = np.linspace(
            start_exclusion,
            N - start_exclusion - 1,
            num_rectangles,
            dtype=int,
        )

        self._obstacles = []
        self._rect_obstacles = []

        rng = np.random.default_rng(20260427)
        min_center_spacing = 120.0
        placed_centers = []
        for rect_idx, idx in enumerate(corner_indices):
            heading = self._heading[idx]
            side_hint = float(np.sign(curvature[idx]))
            if side_hint == 0.0:
                side_hint = 1.0 if rect_idx % 2 == 0 else -1.0

            best_candidate = None
            best_spacing = -np.inf
            for _ in range(32):
                rect_length = rng.uniform(22.0, 34.0)
                rect_width = rng.uniform(10.0, 18.0)
                min_offset = 8.0 + 0.25 * rect_length
                max_offset = TRACK_HALF_WIDTH - 0.5 * rect_length - 3.0
                if max_offset <= min_offset:
                    continue

                side = rng.choice((side_hint, -side_hint))
                offset_distance = rng.uniform(min_offset, max_offset)
                nx = -np.sin(heading) * side
                ny = np.cos(heading) * side
                cx = self._centerline_x[idx] + offset_distance * nx
                cy = self._centerline_y[idx] + offset_distance * ny

                if placed_centers:
                    spacing = min(np.hypot(cx - px, cy - py) for px, py in placed_centers)
                else:
                    spacing = np.inf

                candidate = (cx, cy, rect_length, rect_width, spacing)
                if spacing >= min_center_spacing:
                    best_candidate = candidate
                    break
                if spacing > best_spacing:
                    best_candidate = candidate
                    best_spacing = spacing

            if best_candidate is None:
                raise RuntimeError("Failed to place Lusail Short rectangular obstacle")

            cx, cy, rect_length, rect_width, _ = best_candidate
            rect_angle = heading + 0.5 * np.pi  # 长边与中线切向垂直（沿法向）

            self._rect_obstacles.append((cx, cy, rect_length, rect_width, rect_angle))
            placed_centers.append((cx, cy))
            self._rasterize_rect_to_circles(
                cx=cx,
                cy=cy,
                length=rect_length,
                width=rect_width,
                angle=rect_angle,
                nx_cells=4,
                ny_cells=2,
            )

    def _rasterize_rect_to_circles(self, cx, cy, length, width, angle, nx_cells=4, ny_cells=2):
        """Approximate a rectangle by circles so existing circular constraints remain usable."""
        dx = float(length) / float(nx_cells)
        dy = float(width) / float(ny_cells)
        r = 0.5 * np.hypot(dx, dy)

        c = np.cos(angle)
        s = np.sin(angle)

        for ix in range(nx_cells):
            x_local = -0.5 * length + (ix + 0.5) * dx
            for iy in range(ny_cells):
                y_local = -0.5 * width + (iy + 0.5) * dy
                ox = cx + c * x_local - s * y_local
                oy = cy + s * x_local + c * y_local
                self._obstacles.append((ox, oy, max(r, OBSTACLE_RADIUS)))
