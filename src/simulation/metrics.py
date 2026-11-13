"""
Performance metric calculations for simulation results.
Implements all performance metrics in Table 8 of the paper.
"""

import numpy as np  # NumPy library for numerical computations
import os  # Operating system interface
import sys  # System module for path operations

# Add the parent directory to the system path so project modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the safety distance and vehicle radius constants from the configuration.
from config import D_SAFE, VEHICLE_RADIUS, IDX_V, IDX_PSI


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_all_metrics(result, track, obstacles=None, w_samples=None):
    """
    Calculate all performance metrics for a simulation result.

    Args:
        result: SimResult instance containing the simulation data
        track: BaseTrack instance representing the track
        obstacles: List of obstacles in the form [(ox, oy, radius), ...]
        w_samples: Disturbance samples used to calculate CVaR
    Returns:
        metrics: Dictionary containing all computed performance metrics
    """
    if obstacles is None:  # Use the track's obstacles if none were provided.
        obstacles = track.get_obstacles()  # Get obstacles from the track object.

    data = result.to_arrays()  # Convert the simulation result to arrays.
    states = data['states']        # State array, shape (T+1, 5), containing five state variables
    controls = data['controls']    # Control input array, shape (T, 2), containing acceleration and steering angle
    solve_times = data['solve_times']  # Solver time array, shape (T,)

    positions = states[:, :2]  # Extract positions, shape (T+1, 2), containing [px, py].
    T = len(controls)  # Get the number of control steps.

    metrics = {}  # Initialize the dictionary for computed metrics.

    # Lap time metrics
    metrics['lap_time'] = result.lap_time if result.lap_completed else None  # Record the lap time only if a lap was completed.
    metrics['lap_completed'] = result.lap_completed  # Record whether a lap was completed.
    metrics['total_steps'] = result.total_steps  # Record the total number of simulation steps.

    # Tracking errors: compute the root mean square (RMS) lateral deviation.
    lat_errors = []  # Initialize the lateral error list.
    heading_errors = []  # Initialize the heading error list.
    heading_ref = track.get_heading()
    for t in range(len(positions)):  # Iterate over positions at each time step.
        idx, _, lat_err = track.closest_point(positions[t, 0], positions[t, 1])  # Compute lateral error from the track centerline.
        lat_errors.append(lat_err)  # Append the lateral error to the list.
        psi_err = _wrap_angle(states[t, IDX_PSI] - heading_ref[idx])
        heading_errors.append(psi_err)
    lat_errors = np.array(lat_errors)  # Convert to a NumPy array.
    heading_errors = np.array(heading_errors)  # Convert to a NumPy array.
    abs_lat_errors = np.abs(lat_errors)  # Absolute lateral errors, used to measure proximity to the centerline.
    abs_heading_errors = np.abs(heading_errors)
    metrics['tracking_error_rms'] = np.sqrt(np.mean(lat_errors**2))  # Compute the RMS lateral error.
    metrics['tracking_error_max'] = np.max(abs_lat_errors)  # Compute the maximum absolute lateral error.
    metrics['tracking_error_mean_abs'] = np.mean(abs_lat_errors)  # Mean absolute lateral error.
    metrics['tracking_error_p95_abs'] = np.percentile(abs_lat_errors, 95)  # 95th-percentile absolute lateral error.
    metrics['tracking_within_1m_pct'] = 100.0 * np.mean(abs_lat_errors <= 1.0)  # Percentage of samples within 1 m of the centerline.
    metrics['tracking_within_2m_pct'] = 100.0 * np.mean(abs_lat_errors <= 2.0)  # Percentage of samples within 2 m of the centerline.
    metrics['heading_error_rms_rad'] = np.sqrt(np.mean(heading_errors**2))
    metrics['heading_error_max_abs_rad'] = np.max(abs_heading_errors)
    metrics['heading_error_mean_abs_rad'] = np.mean(abs_heading_errors)
    metrics['heading_error_p95_abs_rad'] = np.percentile(abs_heading_errors, 95)
    metrics['heading_error_mean_abs_deg'] = np.degrees(metrics['heading_error_mean_abs_rad'])
    metrics['heading_error_p95_abs_deg'] = np.degrees(metrics['heading_error_p95_abs_rad'])
    metrics['heading_error_max_abs_deg'] = np.degrees(metrics['heading_error_max_abs_rad'])

    # Find the longest consecutive run of steps more than 2 m from the centerline.
    off_center_mask = abs_lat_errors > 2.0
    max_off_center_run = 0
    current_run = 0
    for is_off_center in off_center_mask:
        if is_off_center:
            current_run += 1
            if current_run > max_off_center_run:
                max_off_center_run = current_run
        else:
            current_run = 0
    metrics['tracking_offcenter_max_steps'] = int(max_off_center_run)

    # Speed metrics
    velocities = states[:, IDX_V]  # Extract the speed column.
    metrics['max_speed'] = np.max(velocities)  # Record the maximum speed.
    metrics['mean_speed'] = np.mean(velocities)  # Record the mean speed.
    metrics['min_speed'] = np.min(velocities)  # Record the minimum speed.

    # Constraint violations: compute the percentage of time steps when obstacle clearance is below the safety distance.
    n_violations = 0  # Initialize the constraint-violation counter.
    min_obs_distances = []  # Initialize the list of obstacle clearances.
    for t in range(len(positions)):  # Iterate over each time step.
        for obs in obstacles:  # Iterate over all obstacles.
            ox, oy, r = obs  # Unpack obstacle x-coordinate, y-coordinate, and radius.
            d_min = r + VEHICLE_RADIUS + D_SAFE  # Minimum safe separation: obstacle radius + vehicle radius + safety margin.
            dist = np.sqrt((positions[t, 0] - ox)**2 + (positions[t, 1] - oy)**2)  # Compute the distance from the vehicle to the obstacle center.
            min_obs_distances.append(dist - d_min)  # Record clearance relative to the safe distance (positive is safe; negative is a violation).
            if dist < d_min:  # Check whether the distance is below the safe separation.
                n_violations += 1  # Increment the violation counter.
                break  # Count at most one violation per time step, even if multiple obstacles are too close.

    metrics['constraint_violation_pct'] = 100.0 * n_violations / max(len(positions), 1)  # Compute the violation percentage, guarding against division by zero.
    metrics['min_obstacle_clearance'] = min(min_obs_distances) if min_obs_distances else float('inf')  # Record the minimum obstacle clearance.

    # CVaR safety margin, when disturbance data is available.
    if w_samples is not None:  # Calculate CVaR when disturbance samples are provided.
        from disturbance.wasserstein import compute_cvar_margin  # Import the CVaR calculation function.
        cvar, cvar_per_obs = compute_cvar_margin(
            positions, obstacles, w_samples  # Compute the conditional value at risk (CVaR) safety margin.
        )
        metrics['cvar_safety_margin'] = cvar  # Store the CVaR safety margin.
    else:
        metrics['cvar_safety_margin'] = None  # Set to None when disturbance data is unavailable.

    # Timing metrics
    metrics['solve_time_mean'] = np.mean(solve_times) * 1000  # Mean solve time in milliseconds.
    metrics['solve_time_max'] = np.max(solve_times) * 1000    # Maximum solve time in milliseconds.
    metrics['solve_time_std'] = np.std(solve_times) * 1000    # Solve-time standard deviation in milliseconds.
    metrics['real_time_feasible'] = metrics['solve_time_mean'] < 100  # Check real-time feasibility (mean solve time < 100 ms).

    # Control effort (magnitude of control inputs)
    if len(controls) > 0:  # Compute control metrics when control inputs are available.
        metrics['control_effort_a'] = np.sqrt(np.mean(controls[:, 0]**2))  # RMS acceleration.
        metrics['control_effort_delta'] = np.sqrt(np.mean(controls[:, 1]**2))  # RMS steering angle.

        # Control smoothness (rate of change in control inputs)
        if len(controls) > 1:  # Compute smoothness when at least two control steps are available.
            du = np.diff(controls, axis=0)  # Compute differences between consecutive control inputs.
            metrics['control_smoothness'] = np.sqrt(np.mean(du**2))  # Compute the RMS control change.

    return metrics


def format_metrics_table(all_metrics, methods):
    """
    Format metrics as a printable table.

    Args:
        all_metrics: Dictionary in the form {method_name: metrics_dictionary}
        methods: List of method names in the desired order

    Returns:
        table_str: Formatted table string
    """
    lines = []  # Initialize the list of table rows.
    sep = "-" * 80  # Create an 80-character separator.

    lines.append(sep)  # Add the top separator.
    header = f"{'Metric':<30}"  # Create the header with a 30-character, left-aligned Metric column.
    for m in methods:  # Iterate over all method names.
        header += f"{'  ' + m:>12}"  # Add each method name, right-aligned in 12 characters with two leading spaces.
    lines.append(header)  # Add the header row.
    lines.append(sep)  # Add a separator.

    # Metrics to display: (key, display label, format specifier)
    metric_names = [
        ('lap_time', 'Lap Time (s)', '.1f'),  # Lap time, to one decimal place
        ('tracking_error_rms', 'Tracking Error RMS (m)', '.2f'),  # RMS tracking error, to two decimal places
        ('tracking_error_p95_abs', 'Tracking Error P95 |e_y| (m)', '.2f'),  # 95th-percentile tracking error
        ('tracking_within_2m_pct', 'Tracking Within ±2m (%)', '.1f'),  # Percentage within 2 m of the centerline
        ('heading_error_mean_abs_deg', 'Heading Mean |e_psi| (deg)', '.2f'),
        ('heading_error_p95_abs_deg', 'Heading P95 |e_psi| (deg)', '.2f'),
        ('max_speed', 'Max Speed (m/s)', '.1f'),  # Maximum speed, to one decimal place
        ('constraint_violation_pct', 'Constraint Violation (%)', '.1f'),  # Constraint violation percentage, to one decimal place
        ('cvar_safety_margin', 'CVaR Safety Margin', '.3f'),  # CVaR safety margin, to three decimal places
        ('solve_time_mean', 'Solve Time Mean (ms)', '.1f'),  # Mean solve time, to one decimal place
        ('solve_time_max', 'Solve Time Max (ms)', '.1f'),  # Maximum solve time, to one decimal place
    ]

    for key, label, fmt in metric_names:  # Iterate over each metric.
        row = f"{label:<30}"  # Start the row with the metric label, left-aligned in 30 characters.
        for m in methods:  # Iterate over all methods.
            val = all_metrics.get(m, {}).get(key, None)  # Get the metric value for this method, or None if unavailable.
            if val is None:  # Handle missing values.
                row += f"{'N/A':>12}"  # Add N/A, right-aligned in 12 characters.
            else:
                row += f"{val:>12{fmt}}"  # Add the formatted value, right-aligned in 12 characters.
        lines.append(row)  # Add the row to the table.

    lines.append(sep)  # Add the bottom separator.
    return "\n".join(lines)  # Join all rows with newlines to create the complete table string.


def format_latex_table(all_metrics, methods, caption="", label=""):
    """
    Format metrics as a LaTeX table.

    Args:
        all_metrics: Dictionary in the form {method_name: metrics_dictionary}
        methods: List of method names
        caption: Table caption
        label: Table label for references

    Returns:
        latex_str: LaTeX table source string
    """
    n_methods = len(methods)  # Get the number of methods.
    col_spec = "l" + "r" * n_methods  # Left-align the first column and right-align the remaining columns.

    lines = [  # Initialize the opening lines of the LaTeX table.
        "\\begin{table}[htbp]",  # Start the table environment; [htbp] specifies float placement options.
        f"\\caption{{{caption}}}",  # Set the table caption.
        f"\\label{{{label}}}",  # Set the table label.
        "\\centering",  # Center the table.
        f"\\begin{{tabular}}{{{col_spec}}}",  # Start the tabular environment with the specified column layout.
        "\\toprule",  # Add the top rule (requires the booktabs package).
    ]

    header = "Method"  # Initialize the header; the first column is Method.
    for m in methods:  # Iterate over all method names.
        header += f" & {m}"  # Add each method name, separated by the LaTeX column delimiter &.
    header += " \\\\"  # Add a line break.
    lines.append(header)  # Add the header row.
    lines.append("\\midrule")  # Add a rule separating the header from the data.

    # Metrics to display in the LaTeX table.
    metric_names = [
        ('lap_time', 'Lap Time (s)', '.1f'),  # Lap time
        ('tracking_error_rms', 'Tracking Error (m)', '.2f'),  # Tracking error
        ('heading_error_mean_abs_deg', 'Heading Mean (deg)', '.2f'),
        ('max_speed', 'Max Speed (m/s)', '.1f'),  # Maximum speed
        ('constraint_violation_pct', 'Constraint Viol. (\\%)', '.1f'),  # Constraint violations (escape % in LaTeX).
        ('solve_time_mean', 'Solve Time (ms)', '.1f'),  # Solve time
    ]

    for key, label_tex, fmt in metric_names:  # Iterate over each metric.
        row = label_tex  # Start the row with the metric's LaTeX label.
        for m in methods:  # Iterate over all methods.
            val = all_metrics.get(m, {}).get(key, None)  # Get the metric value.
            if val is None:  # Handle missing values.
                row += " & N/A"  # Add N/A.
            else:
                row += f" & {val:{fmt}}"  # Add the formatted value.
        row += " \\\\"  # Add a line break.
        lines.append(row)  # Add the row.

    lines.extend([  # Add the closing lines of the table.
        "\\bottomrule",  # Add the bottom rule.
        "\\end{tabular}",  # End the tabular environment.
        "\\end{table}",  # End the table environment.
    ])

    return "\n".join(lines)  # Join all lines into the complete LaTeX source string.
