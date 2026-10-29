"""
Serpentine Track (~1.4 km).

A highly winding closed circuit with dense switchbacks and S-curves, designed
to stress-test lateral tracking performance:
  - Multiple tight hairpin turns
  - Back-to-back S-curve chicanes
  - A compact layout with high average curvature
  - Several direction reversals

13 rectangular obstacles placed at braking / apex zones on an equi-arc-length grid.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack
from config import OBSTACLE_RADIUS, TRACK_HALF_WIDTH, OBSTACLE_LAYOUT_MODE, OBSTACLE_EDGE_MARGIN


class SerpentineTrack(BaseTrack):
    """Highly winding serpentine circuit (~1.4 km) with dense switchbacks."""

    def __init__(self, num_points=1400, target_length=1400.0):
        super().__init__()
        self._target_points = num_points
        self._target_length = target_length
        self._display_name = "Serpentine Track"
        self._build_track()

    def _build_track(self):
        """Build the serpentine centerline from hand-crafted waypoints."""

        # Raw waypoints (arbitrary scale); they will be uniformly rescaled to
        # self._target_length. The path weaves back and forth with tight
        # hairpins and compact S-curves to form a strongly serpentine layout.
        waypoints = np.array([
            # -- Start / Finish (bottom-left) --
            [   0,    0],
            # -- Head right along the bottom --
            [ 140,  -15],
            [ 280,    5],
            # -- First hairpin (bottom-right) --
            [ 400,   70],
            [ 450,  160],
            [ 400,  250],
            # -- S-curve heading left --
            [ 300,  300],
            [ 180,  280],
            [ 100,  320],
            # -- Second hairpin (left side, lower-mid) --
            [  40,  400],
            [ 100,  470],
            [ 220,  460],
            # -- S-curve heading right --
            [ 340,  430],
            [ 430,  480],
            [ 460,  570],
            # -- Third hairpin (right side, upper-mid) --
            [ 420,  660],
            [ 320,  700],
            [ 210,  680],
            # -- S-curve heading left-up --
            [ 120,  720],
            [  20,  780],
            [ -70,  820],
            # -- Fourth hairpin (top-left) --
            [-140,  780],
            [-130,  700],
            [ -60,  650],
            # -- Descend left side --
            [ -20,  580],
            [ -70,  510],
            [-140,  450],
            # -- Fifth hairpin (far-left) --
            [-180,  370],
            [-140,  280],
            [ -70,  220],
            # -- Final sweep back to start --
            [ -30,  150],
            [  10,   80],
            [   0,    0],
        ], dtype=np.float64)

        waypoints = waypoints[:, [1, 0]]

        # --- Scale to target length ---
        raw_length = sum(
            np.linalg.norm(waypoints[i + 1] - waypoints[i])
            for i in range(len(waypoints) - 1)
        )
        scale = self._target_length / raw_length
        waypoints *= scale

        # --- Fit periodic cubic spline ---
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
            f"{self._display_name}: {self._total_length:.0f} m, "
            f"{self._num_points} points, {len(self._rect_obstacles)} obstacles"
        )

    # ------------------------------------------------------------------
    # Obstacle placement
    # ------------------------------------------------------------------

    def _place_obstacles(self, start_clearance_m=0.0):
        """Place 13 rectangular obstacles, optionally clearing the start/finish zone."""
        N = self._num_points
        curvature = self._curvature

        num_rectangles = 13
        start_clearance_m = float(start_clearance_m)
        if start_clearance_m < 0.0 or 2.0 * start_clearance_m >= self._total_length:
            raise ValueError("start_clearance_m must be non-negative and less than half the track length")
        if start_clearance_m == 0.0:
            start_offset = 62.0
            arc_spacing = self._total_length / num_rectangles
            obstacle_s = start_offset + np.arange(num_rectangles) * arc_spacing
        else:
            obstacle_s = np.linspace(
                start_clearance_m,
                self._total_length - start_clearance_m,
                num_rectangles,
            )
        corner_indices = np.mod(
            np.rint(np.interp(obstacle_s, self._arc_length, np.arange(N))).astype(int),
            N,
        )

        self._obstacles = []
        self._rect_obstacles = []

        rng = np.random.default_rng(20260430)
        min_center_spacing = min(80.0, max(20.0, 4.0 * TRACK_HALF_WIDTH))
        placed_centers = []

        for rect_idx, idx in enumerate(corner_indices):
            heading = self._heading[idx]
            side_hint = float(np.sign(curvature[idx]))
            if side_hint == 0.0:
                side_hint = 1.0 if rect_idx % 2 == 0 else -1.0

            best_candidate = None
            best_spacing = -np.inf

            for _ in range(32):
                usable_cross_span = max(TRACK_HALF_WIDTH - 1.0, 2.0)
                rect_length_max = min(34.0, max(4.0, usable_cross_span - 1.0))
                rect_length_min = min(rect_length_max, max(3.0, 0.65 * rect_length_max))
                full_track_width = 2.0 * TRACK_HALF_WIDTH
                rect_width_min = 0.30 * full_track_width
                rect_width_max = 0.55 * full_track_width

                rect_length = rng.uniform(rect_length_min, rect_length_max)
                rect_width = rng.uniform(rect_width_min, rect_width_max)
                min_offset = 0.1
                max_offset = TRACK_HALF_WIDTH - 0.5 * rect_width - 0.1
                if max_offset <= min_offset:
                    continue

                if OBSTACLE_LAYOUT_MODE == "edge":
                    # Edge mode: stick obstacle close to one boundary and keep a tiny edge gap,
                    # so the optimizer avoids trying to squeeze through narrow side passages.
                    side = side_hint
                    edge_offset = max_offset - max(0.0, OBSTACLE_EDGE_MARGIN)
                    offset_distance = np.clip(edge_offset, min_offset, max_offset)
                else:
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

                candidate = (cx, cy, rect_length, rect_width, heading, spacing)
                if spacing >= min_center_spacing:
                    best_candidate = candidate
                    break
                if spacing > best_spacing:
                    best_candidate = candidate
                    best_spacing = spacing

            if best_candidate is None:
                continue

            cx, cy, rect_length, rect_width, angle, _ = best_candidate

            # Keep blocks out of the one U-turn tighter than the vehicle can negotiate.
            if abs(curvature[idx]) > 1.0 / 30.0:
                continue

            placed_centers.append((cx, cy))
            self._rect_obstacles.append((cx, cy, rect_length, rect_width, angle))
            # Approximate circular footprint for collision checks
            r_approx = 0.5 * max(rect_length, rect_width)
            self._obstacles.append((cx, cy, r_approx))
