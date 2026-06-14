"""Suzuka-inspired flowing circuit with linked S-curves and medium-speed bends."""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import TRACK_HALF_WIDTH


class SuzukaFlowTrack(BaseTrack):
    """Figure-eight feeling track without crossings, emphasizing rhythm sections."""

    def __init__(self, num_points=1600, target_length=2300.0):
        super().__init__()
        self._target_points = int(num_points)
        self._target_length = float(target_length)
        self._build_track()

    def _build_track(self):
        waypoints = np.array([
            [0, 0],
            [120, -40],
            [250, -20],
            [340, 50],
            [365, 150],
            [320, 250],
            [230, 300],
            [120, 330],
            [40, 390],
            [20, 490],
            [80, 570],
            [190, 610],
            [310, 595],
            [430, 645],
            [500, 740],
            [470, 860],
            [365, 930],
            [250, 950],
            [175, 1010],
            [180, 1120],
            [260, 1190],
            [390, 1210],
            [510, 1260],
            [560, 1360],
            [525, 1470],
            [430, 1540],
            [290, 1570],
            [150, 1540],
            [20, 1490],
            [-100, 1410],
            [-200, 1300],
            [-260, 1160],
            [-275, 1000],
            [-250, 850],
            [-190, 720],
            [-95, 620],
            [-20, 520],
            [-35, 420],
            [-130, 360],
            [-235, 290],
            [-295, 185],
            [-280, 70],
            [-200, -10],
            [-90, -25],
            [0, 0],
        ], dtype=np.float64)

        raw_length = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        scale = self._target_length / max(raw_length, 1e-6)
        waypoints *= scale

        s = np.zeros(len(waypoints), dtype=np.float64)
        for i in range(1, len(waypoints)):
            s[i] = s[i - 1] + np.linalg.norm(waypoints[i] - waypoints[i - 1])
        s /= max(s[-1], 1e-9)

        cs_x = CubicSpline(s, waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(s, waypoints[:, 1], bc_type='periodic')
        t = np.linspace(0.0, 1.0, self._target_points, endpoint=False)
        self._centerline_x = cs_x(t)
        self._centerline_y = cs_y(t)
        self._compute_geometry(self._centerline_x, self._centerline_y)

        self._place_rect_obstacles()
        print(
            f"Suzuka Flow Track: {self._total_length:.0f}m, "
            f"{self._num_points} points, {len(self._rect_obstacles)} obstacles"
        )

    def _place_rect_obstacles(self):
        self._obstacles = []
        self._rect_obstacles = []
        rng = np.random.default_rng(20260502)

        kappa = np.abs(self._curvature)
        order = np.argsort(kappa)[::-1]
        chosen = []
        min_gap = max(45, self._num_points // 14)

        for idx in order:
            if len(chosen) >= 12:
                break
            if any(min((idx - j) % self._num_points, (j - idx) % self._num_points) < min_gap for j in chosen):
                continue
            chosen.append(int(idx))

        chosen.sort()
        for n, idx in enumerate(chosen):
            heading = float(self._heading[idx])
            side = 1.0 if (n % 2 == 0) else -1.0
            full_track_width = 2.0 * TRACK_HALF_WIDTH
            width = float(rng.uniform(0.30 * full_track_width, 0.55 * full_track_width))
            length = float(rng.uniform(10.0, 18.0))
            max_offset = max(TRACK_HALF_WIDTH - 0.5 * width - 0.2, 0.8)
            offset = float(rng.uniform(0.9, max_offset))
            nx = -np.sin(heading) * side
            ny = np.cos(heading) * side
            cx = float(self._centerline_x[idx] + offset * nx)
            cy = float(self._centerline_y[idx] + offset * ny)
            self._rect_obstacles.append((cx, cy, length, width, heading))
            self._obstacles.append((cx, cy, 0.5 * max(length, width)))
