"""Monaco-inspired hairpin street circuit with dense low-speed turns."""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import TRACK_HALF_WIDTH


class MonacoHairpinTrack(BaseTrack):
    """Tight street-style circuit with repeated hairpins and short connectors."""

    def __init__(self, num_points=1300, target_length=1600.0):
        super().__init__()
        self._target_points = int(num_points)
        self._target_length = float(target_length)
        self._build_track()

    def _build_track(self):
        waypoints = np.array([
            [0, 0],
            [90, -10],
            [170, 5],
            [225, 45],
            [235, 120],
            [190, 180],
            [120, 210],
            [60, 250],
            [30, 320],
            [70, 380],
            [150, 400],
            [215, 440],
            [240, 520],
            [195, 590],
            [120, 620],
            [40, 610],
            [-40, 650],
            [-90, 720],
            [-65, 790],
            [10, 830],
            [90, 860],
            [130, 930],
            [95, 1010],
            [10, 1040],
            [-80, 1010],
            [-150, 950],
            [-220, 900],
            [-285, 830],
            [-320, 745],
            [-300, 660],
            [-235, 610],
            [-175, 545],
            [-185, 460],
            [-255, 410],
            [-315, 340],
            [-320, 250],
            [-260, 180],
            [-175, 165],
            [-95, 125],
            [-70, 55],
            [-20, 20],
            [0, 0],
        ], dtype=np.float64)

        raw_length = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        scale = self._target_length / max(raw_length, 1e-6)
        waypoints *= scale

        t_param = np.zeros(len(waypoints), dtype=np.float64)
        for i in range(1, len(waypoints)):
            t_param[i] = t_param[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])
        t_param /= max(t_param[-1], 1e-9)

        cs_x = CubicSpline(t_param, waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t_param, waypoints[:, 1], bc_type='periodic')

        t_fine = np.linspace(0.0, 1.0, self._target_points, endpoint=False)
        self._centerline_x = cs_x(t_fine)
        self._centerline_y = cs_y(t_fine)
        self._compute_geometry(self._centerline_x, self._centerline_y)

        self._place_rect_obstacles()
        print(
            f"Monaco Hairpin Track: {self._total_length:.0f}m, "
            f"{self._num_points} points, {len(self._rect_obstacles)} obstacles"
        )

    def _place_rect_obstacles(self):
        self._obstacles = []
        self._rect_obstacles = []

        curvature_mag = np.abs(self._curvature)
        candidate_idx = np.argsort(curvature_mag)[::-1]
        min_idx_gap = max(40, self._num_points // 12)
        selected = []
        rng = np.random.default_rng(20260501)

        for idx in candidate_idx:
            if len(selected) >= 10:
                break
            if any(min((idx - j) % self._num_points, (j - idx) % self._num_points) < min_idx_gap for j in selected):
                continue
            selected.append(int(idx))

        selected.sort()
        for idx in selected:
            heading = self._heading[idx]
            side = 1.0 if self._curvature[idx] >= 0 else -1.0
            full_track_width = 2.0 * TRACK_HALF_WIDTH
            width = float(rng.uniform(0.30 * full_track_width, 0.55 * full_track_width))
            length = float(rng.uniform(9.0, 16.0))
            max_offset = max(TRACK_HALF_WIDTH - 0.5 * width - 0.2, 0.8)
            offset = float(rng.uniform(0.7, max_offset))
            nx = -np.sin(heading) * side
            ny = np.cos(heading) * side
            cx = float(self._centerline_x[idx] + offset * nx)
            cy = float(self._centerline_y[idx] + offset * ny)
            self._rect_obstacles.append((cx, cy, length, width, float(heading)))
            self._obstacles.append((cx, cy, 0.5 * max(length, width)))
