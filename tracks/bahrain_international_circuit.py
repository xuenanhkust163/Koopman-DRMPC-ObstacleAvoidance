"""Bahrain International Circuit approximation based on the supplied 11-turn map."""

import numpy as np
from scipy.interpolate import CubicSpline

from tracks.base_track import BaseTrack
from config import TRACK_HALF_WIDTH


class BahrainInternationalCircuit(BaseTrack):
    """Closed Bahrain-style circuit with a long start/finish straight and 11 turns."""

    def __init__(self, num_points=1800, target_bounds=(350.0, 300.0), target_origin=(0.0, -100.0)):
        super().__init__()
        self._target_points = int(num_points)
        self._target_bounds = np.asarray(target_bounds, dtype=np.float64)
        self._target_origin = np.asarray(target_origin, dtype=np.float64)
        if self._target_bounds.shape != (2,) or self._target_origin.shape != (2,):
            raise ValueError("target_bounds and target_origin must each contain two values")
        self._usable_bounds = self._target_bounds - 2.0 * TRACK_HALF_WIDTH
        if (
            not np.isfinite(self._usable_bounds).all()
            or not np.isfinite(self._target_origin).all()
            or np.any(self._usable_bounds <= 0.0)
        ):
            raise ValueError("target_bounds must exceed the unchanged road width in both dimensions")
        self._plot_bounds = (
            float(self._target_origin[0]),
            float(self._target_origin[0] + self._target_bounds[0]),
            float(self._target_origin[1]),
            float(self._target_origin[1] + self._target_bounds[1]),
        )
        self._build_track()

    def _build_track(self):
        waypoints = np.array([
            [-280, -140],
            [0, -140],
            [300, -140],
            [470, -140],
            [525, -90],
            [520, 10],
            [475, 110],
            [410, 155],
            [365, 205],
            [345, 270],
            [300, 340],
            [250, 420],
            [220, 510],
            [180, 580],
            [135, 620],
            [85, 640],
            [35, 620],
            [-5, 580],
            [-75, 535],
            [-150, 480],
            [-205, 410],
            [-220, 340],
            [-190, 280],
            [-135, 235],
            [-65, 215],
            [15, 195],
            [100, 180],
            [190, 165],
            [255, 140],
            [270, 105],
            [245, 75],
            [185, 60],
            [95, 55],
            [-10, 55],
            [-120, 55],
            [-230, 55],
            [-330, 55],
            [-405, 75],
            [-440, 105],
            [-500, 115],
            [-555, 60],
            [-580, -30],
            [-560, -130],
            [-510, -205],
            [-450, -230],
            [-400, -200],
            [-350, -180],
            [-280, -140],
        ], dtype=np.float64)
        arc = np.zeros(len(waypoints), dtype=np.float64)
        arc[1:] = np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
        arc /= arc[-1]

        spline_x = CubicSpline(arc, waypoints[:, 0], bc_type="periodic")
        spline_y = CubicSpline(arc, waypoints[:, 1], bc_type="periodic")
        parameter = np.linspace(0.0, 1.0, self._target_points, endpoint=False)
        preview_parameter = np.linspace(0.0, 1.0, self._target_points, endpoint=False)
        preview_x = spline_x(preview_parameter)
        preview_y = spline_y(preview_parameter)
        center_x = 0.5 * (float(np.min(preview_x)) + float(np.max(preview_x)))
        center_y = 0.5 * (float(np.min(preview_y)) + float(np.max(preview_y)))
        target_center = self._target_origin + 0.5 * self._target_bounds
        scale = min(
            self._usable_bounds[0] / float(np.ptp(preview_x)),
            self._usable_bounds[1] / float(np.ptp(preview_y)),
        )
        self._centerline_x = (spline_x(parameter) - center_x) * scale + target_center[0]
        self._centerline_y = (spline_y(parameter) - center_y) * scale + target_center[1]
        self._compute_geometry(self._centerline_x, self._centerline_y)
        self._place_obstacles()

        self.turn_waypoints = (
            (waypoints[3:-1] - np.array([center_x, center_y])) * scale
            + target_center
        )
        print(
            f"Bahrain International Circuit: {self._total_length:.0f} m, "
            f"{np.ptp(self._centerline_x):.0f} x {np.ptp(self._centerline_y):.0f} m, "
            f"{self._num_points} points, 11-turn topology"
        )

    def _place_obstacles(self):
        """Place 13 reproducible rectangular blocks around the circuit edges."""
        self._rect_obstacles = []
        self._obstacles = []
        start_fraction = 0.055
        spacing = 0.89 / 13.0

        for obstacle_index in range(13):
            arc_position = (start_fraction + obstacle_index * spacing) * self._total_length
            sample_index = int(np.argmin(np.abs(self._arc_length - arc_position)))
            heading = float(self._heading[sample_index])
            side = 1.0 if obstacle_index % 2 == 0 else -1.0
            offset = 0.62 * TRACK_HALF_WIDTH
            normal_x = -np.sin(heading) * side
            normal_y = np.cos(heading) * side
            center_x = float(self._centerline_x[sample_index] + offset * normal_x)
            center_y = float(self._centerline_y[sample_index] + offset * normal_y)
            length = 10.0
            width = 10.0
            self._rect_obstacles.append((center_x, center_y, length, width, heading))
            self._obstacles.append((center_x, center_y, 0.5 * max(length, width)))