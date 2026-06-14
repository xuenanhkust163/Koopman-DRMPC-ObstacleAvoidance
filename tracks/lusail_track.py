"""
Lusail International Circuit approximation (~5.38 km, 16 turns).

The track is approximated using waypoints fit with a periodic cubic spline.
4 static obstacles are placed at strategic corner locations.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import OBSTACLE_RADIUS, D_SAFE


class LusailTrack(BaseTrack):
    """
    Approximation of the Lusail International Circuit (Qatar).
    ~5.38 km length, 16 turns (10 right, 6 left).
    """

    def __init__(self, num_points=2000):
        super().__init__()
        self._target_points = num_points
        self._build_track()

    def _build_track(self):
        """Build the Lusail circuit approximation from waypoints."""
        # Define waypoints approximating the Lusail circuit shape
        # Coordinates in meters, roughly centered at origin
        # The circuit has a characteristic "figure-8 like" shape
        waypoints = np.array([
            # Start/Finish straight
            [0, 0],
            [200, 5],
            [400, 15],
            [600, 30],
            [800, 50],
            [1000, 60],
            # T1 - Right turn
            [1150, 80],
            [1250, 140],
            [1280, 220],
            # T2-T3 - Left-Right chicane
            [1250, 320],
            [1180, 380],
            [1150, 450],
            [1180, 520],
            # T4-T5 - Double right
            [1250, 580],
            [1280, 650],
            [1240, 730],
            # T6 - Sharp left
            [1150, 790],
            [1030, 830],
            [900, 820],
            # T7-T8 - Right-Left sequence (back section)
            [780, 770],
            [680, 690],
            [620, 600],
            [580, 500],
            # T9 - Right hairpin
            [560, 380],
            [510, 300],
            [430, 260],
            [350, 290],
            # T10-T11 - Left-Right through middle
            [300, 360],
            [250, 440],
            [200, 500],
            [120, 540],
            # T12-T13 - Right turns
            [30, 530],
            [-60, 480],
            [-120, 400],
            # T14-T15 - Left-Right before straight
            [-150, 300],
            [-130, 200],
            [-80, 130],
            # T16 - Final corner back to straight
            [-30, 60],
            [0, 0],  # Close the loop
        ], dtype=np.float64)

        # Scale to achieve approximately 5.38 km total length
        # First compute raw length
        raw_length = 0
        for i in range(len(waypoints) - 1):
            raw_length += np.linalg.norm(waypoints[i+1] - waypoints[i])
        scale = 5380.0 / raw_length
        waypoints *= scale

        # Fit periodic cubic spline
        t_param = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            t_param[i] = t_param[i-1] + np.linalg.norm(
                waypoints[i] - waypoints[i-1])

        # Normalize parameter
        t_param /= t_param[-1]

        # Use periodic boundary conditions
        cs_x = CubicSpline(t_param, waypoints[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t_param, waypoints[:, 1], bc_type='periodic')

        # Resample at uniform parameter values
        t_fine = np.linspace(0, 1, self._target_points, endpoint=False)
        self._centerline_x = cs_x(t_fine)
        self._centerline_y = cs_y(t_fine)

        # Compute geometry
        self._compute_geometry(self._centerline_x, self._centerline_y)

        # Place 4 obstacles at corners T1, T6, T9, T16
        self._place_obstacles()

        print(f"Lusail Track: {self._total_length:.0f}m, "
              f"{self._num_points} points, {len(self._obstacles)} obstacles")

    def _place_obstacles(self):
        """Place 4 static obstacles at strategic corner locations."""
        N = self._num_points
        curvature = np.abs(self._curvature)

        # Find high-curvature regions (corners)
        # Smooth curvature for peak detection
        from scipy.ndimage import uniform_filter1d
        smooth_curv = uniform_filter1d(curvature, size=N // 20, mode='wrap')

        # Find peaks
        corner_indices = []
        min_separation = N // 8  # Minimum distance between corners

        for _ in range(4):
            idx = np.argmax(smooth_curv)
            corner_indices.append(idx)
            # Suppress nearby peaks
            start = max(0, idx - min_separation)
            end = min(N, idx + min_separation)
            smooth_curv[start:end] = 0

        corner_indices.sort()

        # Place obstacles slightly off the racing line at each corner
        offset_distance = 5.0  # meters from centerline
        for idx in corner_indices:
            # Offset perpendicular to heading (inside of corner)
            heading = self._heading[idx]
            sign = np.sign(self._curvature[idx])  # Inside of corner
            nx = -np.sin(heading) * sign
            ny = np.cos(heading) * sign
            ox = self._centerline_x[idx] + offset_distance * nx
            oy = self._centerline_y[idx] + offset_distance * ny
            self._obstacles.append((ox, oy, OBSTACLE_RADIUS))
