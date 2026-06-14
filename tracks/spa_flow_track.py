"""Spa-inspired high-speed flowing circuit with long-radius corners."""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import TRACK_HALF_WIDTH


class SpaFlowTrack(BaseTrack):
    """Long, sweeping circuit inspired by classic fast European layouts."""

    def __init__(self, num_points=600, target_length=1000.0):
        super().__init__()
        self._target_points = int(num_points)
        self._target_length = float(target_length)
        self._build_track()

    def _build_track(self):
        waypoints = np.array([
            [0, 0],
            [180, -35],
            [380, -50],
            [560, -10],
            [710, 80],
            [820, 220],
            [870, 400],
            [840, 590],
            [730, 760],
            [560, 860],
            [390, 930],
            [340, 1040],
            [420, 1160],
            [580, 1230],
            [770, 1290],
            [920, 1420],
            [980, 1610],
            [930, 1790],
            [790, 1910],
            [610, 1960],
            [410, 2000],
            [230, 2060],
            [80, 2170],
            [20, 2340],
            [80, 2510],
            [240, 2600],
            [440, 2620],
            [620, 2550],
            [780, 2460],
            [900, 2480],
            [970, 2590],
            [940, 2730],
            [830, 2810],
            [680, 2850],
            [500, 2880],
            [300, 2870],
            [120, 2800],
            [-40, 2660],
            [-130, 2480],
            [-180, 2290],
            [-240, 2110],
            [-330, 1960],
            [-470, 1860],
            [-620, 1780],
            [-730, 1650],
            [-760, 1470],
            [-700, 1300],
            [-560, 1190],
            [-400, 1110],
            [-300, 980],
            [-300, 810],
            [-390, 670],
            [-550, 590],
            [-700, 500],
            [-760, 340],
            [-700, 180],
            [-540, 90],
            [-340, 40],
            [-150, 15],
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
            f"Spa Flow Track: {self._total_length:.0f}m, "
            f"{self._num_points} points, {len(self._rect_obstacles)} obstacles"
        )

    def _place_rect_obstacles(self):
        self._obstacles = []
        self._rect_obstacles = []
        rng = np.random.default_rng(20260503)

        kappa = np.abs(self._curvature)
        order = np.argsort(kappa)[::-1]
        chosen = []
        min_gap = max(60, self._num_points // 16)

        for idx in order:
            if len(chosen) >= 9:
                break
            if any(min((idx - j) % self._num_points, (j - idx) % self._num_points) < min_gap for j in chosen):
                continue
            chosen.append(int(idx))

        chosen.sort()
        for idx in chosen:
            heading = float(self._heading[idx])
            side = 1.0 if self._curvature[idx] >= 0 else -1.0
            full_track_width = 2.0 * TRACK_HALF_WIDTH
            width = float(rng.uniform(0.30 * full_track_width, 0.55 * full_track_width))
            length = float(rng.uniform(11.0, 21.0))
            max_offset = max(TRACK_HALF_WIDTH - 0.5 * width - 0.2, 1.0)
            offset = float(rng.uniform(1.0, max_offset))
            nx = -np.sin(heading) * side
            ny = np.cos(heading) * side
            cx = float(self._centerline_x[idx] + offset * nx)
            cy = float(self._centerline_y[idx] + offset * ny)
            self._rect_obstacles.append((cx, cy, length, width, heading))
            self._obstacles.append((cx, cy, 0.5 * max(length, width)))
