"""
Abstract base class for tracks (BaseTrack).
This module defines the common interface and shared implementations for all tracks, including:
- Geometric calculations for the track centerline, heading, and curvature
- Reference trajectory generation, including speed profile planning
- Nearest-point queries and lateral deviation calculations
- Obstacle management

Subclasses only need to implement _build_track() to define the specific track shape.
"""

import numpy as np              # Numerical computing library for vectorized and matrix operations
from abc import ABC, abstractmethod  # Abstract base class utilities for enforcing subclass methods
from scipy.interpolate import CubicSpline  # Cubic spline interpolation (currently unused; reserved)
from config import (
    IDX_PX, IDX_PY, IDX_PSI, IDX_V, IDX_OMEGA,  # State vector dimension index constants
    A_LAT_MAX,       # Maximum lateral acceleration [m/s^2], used to limit cornering speed
    V_MAX,           # Maximum vehicle speed [m/s]
    REF_SPEED_SCALE, # Global reference speed scale factor (reduces aggressiveness)
    REF_ACCEL_MAX,   # Maximum acceleration for the reference speed profile [m/s^2]
    REF_DECEL_MAX,   # Maximum deceleration for the reference speed profile [m/s^2]
    ENABLE_OBSTACLES,  # Global obstacle toggle
    DT,              # Sampling time [s]
)


# ============================================================================
# BaseTrack: Abstract base class for tracks
# ============================================================================
class BaseTrack(ABC):
    """
    Abstract base class for racing tracks.
    Every concrete track must inherit from this class and implement _build_track()
    to define the track centerline coordinates.
    """

    def __init__(self):
        """Initialize placeholders for track geometry, populated by the subclass's _build_track()."""
        self._centerline_x = None   # Track centerline x-coordinate array (N,)
        self._centerline_y = None   # Track centerline y-coordinate array (N,)
        self._heading = None        # Tangential heading angle at each centerline point [rad] (N,)
        self._curvature = None      # Curvature at each centerline point [1/m] (N,)
        self._arc_length = None     # Cumulative arc length at each centerline point [m] (N,)
        self._obstacles = []        # Circular obstacles as (ox, oy, radius) tuples
        self._rect_obstacles = []   # Rectangular obstacles as (cx, cy, length, width, angle_rad) tuples
        self._total_length = 0.0    # Total track length [m]
        self._num_points = 0        # Number of discrete centerline points

    @abstractmethod
    def _build_track(self):
        """
        Build the track geometry. Subclasses must override this method.
        The method must set _centerline_x and _centerline_y, then call _compute_geometry().
        """
        pass

    # ------------------------------------------------------------------
    # Query methods: return copies of track geometry to protect internal data.
    # ------------------------------------------------------------------
    def get_centerline(self):
        """Return copies of the track centerline x- and y-coordinate arrays."""
        return self._centerline_x.copy(), self._centerline_y.copy()

    def get_heading(self):
        """Return a copy of the tangential heading angle array [rad]."""
        return self._heading.copy()

    def get_curvature(self):
        """Return a copy of the curvature array [1/m]."""
        return self._curvature.copy()

    def get_arc_length(self):
        """Return a copy of the cumulative arc length array [m]."""
        return self._arc_length.copy()

    def get_obstacles(self):
        """
        Return the list of circular obstacles.
        Return an empty list if the global ENABLE_OBSTACLES toggle is False.
        """
        if not ENABLE_OBSTACLES:
            return []
        return list(self._obstacles)

    def get_rect_obstacles(self):
        """
        Return rectangular obstacles as (cx, cy, length, width, angle_rad) tuples.
        Return an empty list if the global ENABLE_OBSTACLES toggle is False.
        """
        if not ENABLE_OBSTACLES:
            return []
        return list(self._rect_obstacles)

    def total_length(self):
        """Return the total track length [m]."""
        return self._total_length

    def num_points(self):
        """Return the number of discrete centerline points."""
        return self._num_points

    def closest_point(self, px, py):
        """
        Find the point on the track nearest to the given coordinates (px, py).
        Uses a global nearest-neighbor search based on Euclidean distance over the discrete centerline.

        Important: The returned arc length s is not the arc length of the nearest
        discrete point. It is the continuously interpolated arc length after projecting
        the vehicle onto an adjacent track segment. This prevents accumulated lateral
        error caused by reference trajectory stagnation at low speeds.

        Args:
            px: Query point x-coordinate [m]
            py: Query point y-coordinate [m]

        Returns:
            idx: Index of the nearest point in the centerline arrays
            s: Continuous cumulative arc length at the projected point [m]
            lateral_error: Signed lateral deviation [m]
                Positive means the vehicle is left of the centerline (the normal points left).
                Negative means the vehicle is right of the centerline.
        """
        # Compute x/y differences from the query point to every centerline point.
        dx = self._centerline_x - px
        dy = self._centerline_y - py
        # Compute Euclidean distances.
        dist = np.sqrt(dx**2 + dy**2)
        # Find the index of the nearest point.
        idx = np.argmin(dist)

        # ------------------------------------------------------------------
        # Compute the precise projected arc length by interpolating on adjacent segments.
        # ------------------------------------------------------------------
        N = self._num_points
        prev_idx = (idx - 1) % N
        next_idx = (idx + 1) % N

        best_s = self._arc_length[idx]
        best_dist = float('inf')

        for i, j in [(prev_idx, idx), (idx, next_idx)]:
            x1, y1 = self._centerline_x[i], self._centerline_y[i]
            x2, y2 = self._centerline_x[j], self._centerline_y[j]
            dx_seg = x2 - x1
            dy_seg = y2 - y1
            seg_len_sq = dx_seg**2 + dy_seg**2
            if seg_len_sq < 1e-12:
                continue
            # Projection ratio: t = ((P - P1) dot (P2 - P1)) / |P2 - P1|^2.
            t = ((px - x1) * dx_seg + (py - y1) * dy_seg) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            # Projected point.
            proj_x = x1 + t * dx_seg
            proj_y = y1 + t * dy_seg
            d = (px - proj_x)**2 + (py - proj_y)**2
            if d < best_dist:
                best_dist = d
                # Interpolate arc length, accounting for wraparound.
                s1 = self._arc_length[i]
                s2 = self._arc_length[j]
                if j < i:  # Wrap across the end of the track.
                    s2 += self._total_length
                best_s = s1 + t * (s2 - s1)
                if best_s >= self._total_length:
                    best_s -= self._total_length

        # ------------------------------------------------------------------
        # Compute the signed lateral error (signed point-to-line distance).
        # ------------------------------------------------------------------
        heading = self._heading[idx]   # Tangential heading at the nearest point.
        # Compute the unit normal to the centerline, pointing left (the tangent rotated 90 degrees counterclockwise).
        # Tangent: (cos(heading), sin(heading))
        # Normal: (-sin(heading), cos(heading))
        nx = -np.sin(heading)
        ny = np.cos(heading)
        # Signed lateral error = dot product of the centerline-to-vehicle vector and the unit normal.
        # A positive dot product means the vehicle is in the normal direction (left).
        # A negative dot product means the vehicle is in the opposite direction (right).
        lateral_error = (px - self._centerline_x[idx]) * nx + \
                        (py - self._centerline_y[idx]) * ny

        return idx, best_s, lateral_error

    def get_reference_trajectory(self, start_idx, horizon, v_ref=None,
                                 a_lat_max=A_LAT_MAX, v_max=V_MAX,
                                 current_speed=None,
                                 accel_limit=REF_ACCEL_MAX,
                                 decel_limit=REF_DECEL_MAX,
                                 start_s=None):
        """
        Generate a reference trajectory over the MPC prediction horizon.
        The trajectory contains position (px, py), heading (psi), speed (v), and yaw rate (omega).
        The speed profile uses a three-stage algorithm: curvature limits, forward acceleration
        limits, and backward deceleration limits. This ensures the reference speed is physically
        reachable and safe.

        Important: Reference positions are no longer sampled at fixed index intervals.
        Instead, track points are sampled according to the actual per-step travel distance
        v*dt from the speed profile, keeping the reference timeline consistent with vehicle dynamics.

        Args:
            start_idx: Starting track point index (used to query upcoming curvature)
            horizon: Prediction horizon length T (steps)
            v_ref: Fixed reference speed; when provided, curvature-based speed limits are ignored
            a_lat_max: Maximum lateral acceleration [m/s^2], setting the cornering speed limit: v <= sqrt(a_lat_max / kappa)
            v_max: Maximum vehicle speed [m/s]
            current_speed: Current vehicle speed [m/s], used for a smooth transition from the current speed
            accel_limit: Maximum acceleration for forward speed planning [m/s^2]
            decel_limit: Maximum deceleration for backward speed planning [m/s] (positive value)
            start_s: Starting arc length [m]. If provided, stage 3 starts the reference positions
                     at this exact arc length; otherwise it falls back to the arc length at start_idx.
                     Passing the vehicle's actual arc length prevents the MPC from chasing a reference
                     whose first point lags behind the vehicle.

        Returns:
            ref: NumPy array of shape (horizon, 5), with rows [px, py, psi, v, omega]
        """
        ref = np.zeros((horizon, 5))     # Preallocate the reference trajectory array.
        N = self._num_points             # Total number of discrete track points.
        target_speeds = np.zeros(horizon)  # Raw target speeds limited by curvature.

        # ------------------------------------------------------------------
        # Stage 1: Compute curvature-limited speeds at fixed index intervals to query upcoming curvature.
        # ------------------------------------------------------------------
        idx_sequence_temp = []
        for t in range(horizon):
            idx = (start_idx + t) % N
            idx_sequence_temp.append(idx)
            kappa = abs(self._curvature[idx])  # Absolute curvature [1/m].
            if v_ref is not None:
                target_speeds[t] = REF_SPEED_SCALE * v_ref
            elif kappa > 1e-6:
                target_speeds[t] = REF_SPEED_SCALE * min(v_max, np.sqrt(a_lat_max / kappa))
            else:
                target_speeds[t] = REF_SPEED_SCALE * v_max

        # ------------------------------------------------------------------
        # Stage 2: Smooth the speed profile while accounting for longitudinal acceleration limits.
        # ------------------------------------------------------------------
        if current_speed is None:
            speed_profile = target_speeds.copy()
        else:
            speed_profile = np.zeros(horizon)
            speed_profile[0] = float(np.clip(current_speed, 0.0, v_max))

            # ---- Forward pass ----
            for t in range(1, horizon):
                prev_idx = idx_sequence_temp[t - 1]
                idx = idx_sequence_temp[t]
                ds = self._arc_length[idx] - self._arc_length[prev_idx]
                if ds < 0:
                    ds += self._total_length
                v_reachable = np.sqrt(max(speed_profile[t - 1] ** 2 + 2.0 * accel_limit * max(ds, 0.0), 0.0))
                speed_profile[t] = min(target_speeds[t], v_reachable)

            # ---- Backward pass ----
            for t in range(horizon - 2, -1, -1):
                idx = idx_sequence_temp[t]
                next_idx = idx_sequence_temp[t + 1]
                ds = self._arc_length[next_idx] - self._arc_length[idx]
                if ds < 0:
                    ds += self._total_length
                v_brake_cap = np.sqrt(max(speed_profile[t + 1] ** 2 + 2.0 * decel_limit * max(ds, 0.0), 0.0))
                speed_profile[t] = min(speed_profile[t], v_brake_cap)

        # ------------------------------------------------------------------
        # Stage 3: Regenerate reference positions using actual travel distance from the speed profile.
        # Use linear interpolation instead of nearest-neighbor lookup to prevent position stagnation at low speeds.
        # Use the exact start_s when supplied to prevent ref[0] from lagging behind the vehicle.
        # ------------------------------------------------------------------
        ref_positions = []
        distance_covered = 0.0
        if start_s is None:
            start_s = self._arc_length[start_idx]

        # Unwrap heading angles to avoid wraparound issues during interpolation.
        psi_unwrapped = np.unwrap(self._heading)

        for t in range(horizon):
            if t == 0:
                target_s = start_s
            else:
                distance_covered += speed_profile[t - 1] * DT
                target_s = (start_s + distance_covered) % self._total_length

            # Linearly interpolate the reference position to avoid jumps from nearest-neighbor lookup.
            px = np.interp(target_s, self._arc_length, self._centerline_x)
            py = np.interp(target_s, self._arc_length, self._centerline_y)
            psi = np.interp(target_s, self._arc_length, psi_unwrapped)
            kappa = np.interp(target_s, self._arc_length, self._curvature)

            ref_positions.append((px, py, psi, kappa))

        # ------------------------------------------------------------------
        # Stage 4: Fill in the reference trajectory (position, heading, speed, and yaw rate).
        # ------------------------------------------------------------------
        for t, (px, py, psi, kappa) in enumerate(ref_positions):
            ref[t, IDX_PX] = px
            ref[t, IDX_PY] = py
            # Wrap heading back to [-pi, pi].
            ref[t, IDX_PSI] = ((psi + np.pi) % (2.0 * np.pi)) - np.pi
            ref[t, IDX_V] = speed_profile[t]
            ref[t, IDX_OMEGA] = speed_profile[t] * kappa

        return ref

    def get_reference_v_omega(self, start_idx, horizon, a_lat_max=A_LAT_MAX, v_max=V_MAX):
        """
        Get reference speed and yaw-rate trajectories for the Koopman MPC cost function.
        This convenience wrapper around get_reference_trajectory returns only the [v, omega] columns.

        Args:
            start_idx: Starting track point index
            horizon: Prediction horizon length
            a_lat_max: Maximum lateral acceleration
            v_max: Maximum speed

        Returns:
            y_ref: NumPy array of shape (horizon, 2), with rows [v, omega]
        """
        ref = self.get_reference_trajectory(
            start_idx, horizon, a_lat_max=a_lat_max, v_max=v_max)
        return ref[:, [IDX_V, IDX_OMEGA]]  # Extract the speed and yaw-rate columns.

    def _compute_geometry(self, x, y):
        """
        Compute geometric properties from the track centerline (x, y) coordinates.
        These include cumulative arc length, tangential heading, and curvature.
        Subclasses should call this method after setting the centerline in _build_track().

        Args:
            x: Centerline x-coordinate array (N,)
            y: Centerline y-coordinate array (N,)
        """
        N = len(x)

        # ------------------------------------------------------------------
        # 1. Compute cumulative arc length.
        # ------------------------------------------------------------------
        # Compute Euclidean distances between adjacent points.
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        # Accumulate distances to get the arc length at each point (the first point is zero).
        self._arc_length = np.zeros(N)
        self._arc_length[1:] = np.cumsum(ds)
        self._total_length = self._arc_length[-1]  # The final point gives the total length.

        # ------------------------------------------------------------------
        # 2. Compute tangential heading.
        # ------------------------------------------------------------------
        # Approximate derivatives with gradients: heading = arctan2(dy/ds, dx/ds).
        # edge_order=2 uses second-order accuracy at the boundaries.
        self._heading = np.arctan2(
            np.gradient(y, self._arc_length, edge_order=2),
            np.gradient(x, self._arc_length, edge_order=2)
        )

        # ------------------------------------------------------------------
        # 3. Compute curvature.
        # ------------------------------------------------------------------
        # Curvature = d(heading) / ds.
        # Unwrap heading with np.unwrap first to remove artificial jumps at +/-pi crossings.
        heading_unwrapped = np.unwrap(self._heading)
        dheading = np.gradient(heading_unwrapped)
        # Curvature = heading change / arc-length change; clamp the denominator to avoid division by zero.
        self._curvature = dheading / np.maximum(np.gradient(self._arc_length), 1e-6)

        self._num_points = N
