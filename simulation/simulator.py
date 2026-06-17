"""Closed-loop simulation engine for running MPC controllers on a track."""

import numpy as np
import time
import os
import sys
import pickle
import subprocess
from collections import Counter

# ============================================================================
# Path configuration and imports
# ============================================================================
# Put the project root at the front of the Python module search path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simulation-related constants from the project configuration.
from config import (
    DT,
    MAX_SIM_STEPS,
    V_MIN, V_MAX,
    A_MIN, A_MAX,
    DELTA_MAX,
    RESULTS_DIR,
    FIGURES_DIR,
    IDX_PX, IDX_PY,
    IDX_PSI,
    IDX_V, IDX_OMEGA,
    CONTROL_UPDATE_INTERVAL,
    TRACK_HALF_WIDTH,
    VEHICLE_RADIUS,
    EXPORT_STATIC_FIGURES,
    EXPORT_ANIMATION,
    ANIMATION_FPS,
)
# Import the discrete-time bicycle model step function.
from vehicle.bicycle_model import discrete_step


# ============================================================================
# SimResult: simulation result container
# ============================================================================
class SimResult:
    """Container for all data produced by one closed-loop simulation run."""

    def __init__(self, method_name, track_name):
        """Initialize empty storage for one simulation run."""
        # ------------------------------------------------------------------
        # Basic identifiers
        # ------------------------------------------------------------------
        self.method_name = method_name
        self.track_name = track_name

        # ------------------------------------------------------------------
        # Time-series data, one element appended per simulation step
        # ------------------------------------------------------------------
        self.states = []
        self.controls = []
        self.solve_times = []
        self.solve_statuses = []
        self.solve_debug = []
        self.ref_states = []
        self.timestamps = []

        # ------------------------------------------------------------------
        # Event flags and metadata
        # ------------------------------------------------------------------
        self.lap_completed = False
        self.lap_time = None
        self.total_steps = 0
        self.crashed = False
        self.crash_step = None
        self.crash_time = None
        self.crash_reason = None

    def to_arrays(self):
        """Convert stored Python lists into NumPy arrays for analysis."""
        return {
            'states': np.array(self.states),
            'controls': np.array(self.controls),
            'solve_times': np.array(self.solve_times),
            'timestamps': np.array(self.timestamps),
            'solve_debug': list(self.solve_debug),
            'crashed': self.crashed,
            'crash_step': self.crash_step,
            'crash_time': self.crash_time,
            'crash_reason': self.crash_reason,
        }


# ============================================================================
# Simulator: closed-loop simulation engine
# ============================================================================
class Simulator:
    """Closed-loop simulation orchestrator for track, controller, and noise."""

    def __init__(self, track, controller, disturbance_gen=None):
        """Store references to the track, controller, and optional noise model."""
        self.track = track
        self.controller = controller
        self.disturbance_gen = disturbance_gen

    def run(self, x0=None, max_steps=MAX_SIM_STEPS, lap_fraction=0.95,
            verbose=True, detailed_step_log=False,
            detailed_step_log_max_steps=None,
            control_update_interval=CONTROL_UPDATE_INTERVAL):
        """Execute the full closed-loop MPC simulation loop."""
        # ------------------------------------------------------------------
        # Validate inputs and grab local references.
        # ------------------------------------------------------------------
        if control_update_interval <= 0:
            raise ValueError("control_update_interval must be a positive integer")

        track = self.track
        controller = self.controller
        obstacles = track.get_obstacles()

        # Stage 1: initialize the starting state.
        if x0 is None:
            # Build a default initial state from the track start.
            cx, cy = track.get_centerline()
            heading = track.get_heading()
            ref_init = track.get_reference_v_omega(0, 1)
            x0 = np.array([cx[0], cy[0], heading[0], ref_init[0, 0], ref_init[0, 1]])

        result = SimResult(controller.name, track.__class__.__name__)
        controller.reset()

        # ------------------------------------------------------------------
        # Stage 2: initialize loop variables.
        # ------------------------------------------------------------------
        x = x0.copy()
        result.states.append(x.copy())

        start_idx, start_s, _ = track.closest_point(x[0], x[1])
        max_s = track.total_length()
        cumulative_s = 0.0
        prev_s = start_s
        crash_lat_limit = max(TRACK_HALF_WIDTH - VEHICLE_RADIUS, 0.0)

        # ------------------------------------------------------------------
        # Stage 3: print run metadata.
        # ------------------------------------------------------------------
        if verbose:
            print(f"\nRunning {controller.name} on {track.__class__.__name__}...")
            print(f"  Track length: {max_s:.0f}m, obstacles: {len(obstacles)}")
            print(f"  Control update interval: solve MPC every {control_update_interval} step(s)")

        # ------------------------------------------------------------------
        # Stage 4: initialize held-control state for rate reduction.
        # ------------------------------------------------------------------
        held_u = np.zeros(2)
        held_info = {'solve_time': 0.0, 'status': 'hold', 'debug': None}

        # ==================================================================
        # Main simulation loop.
        # ==================================================================
        for step in range(max_steps):
            t_sim = step * DT  # Current simulated time [s].

            # ------------------------------------------------------------------
            # Step A: locate the vehicle projection on the track.
            # ------------------------------------------------------------------
            # closest_point returns:
            #   idx: nearest centerline index
            #   current_s: cumulative arc length along the centerline [m]
            #   lat_err: lateral error [m], positive on the left, negative on the right
            idx, current_s, lat_err = track.closest_point(x[0], x[1])

            # ------------------------------------------------------------------
            # Step B: update cumulative travel distance with wrap handling.
            # ------------------------------------------------------------------
            ds = current_s - prev_s
            # Handle wraparound when the vehicle crosses the start/finish line.
            if ds < -max_s / 2:
                ds += max_s
            elif ds > max_s / 2:
                ds -= max_s
            cumulative_s += abs(ds)
            prev_s = current_s

            # ------------------------------------------------------------------
            # Step C: build the MPC reference trajectory.
            # ------------------------------------------------------------------
            from config import T_HORIZON
            # Use the current arc length directly to avoid lagging the first reference point.
            ref = track.get_reference_trajectory(
                idx, T_HORIZON, start_s=current_s)

            # ------------------------------------------------------------------
            # Step D: solve MPC or reuse the held control.
            # ------------------------------------------------------------------
            should_solve = (step % control_update_interval == 0)
            if should_solve:
                try:
                    # Prefer passing u_prev so controllers can model rate penalties consistently.
                    try:
                        u_opt, info = controller.solve(x, ref, obstacles, u_prev=held_u)
                    except TypeError:
                        # Fall back to the legacy interface when u_prev is unsupported.
                        u_opt, info = controller.solve(x, ref, obstacles)
                except Exception as e:
                    if verbose:
                        print(f"  Step {step}: controller error: {e}")
                    # Hold the previous command to avoid abrupt control jumps on failure.
                    u_opt = held_u.copy()
                    info = {'solve_time': 0, 'status': f'error: {str(e)[:30]}', 'debug': None}
            else:
                # In reduced solve-rate mode, reuse the previous control input.
                u_opt = held_u.copy()
                info = held_info.copy()
                info['solve_time'] = 0.0
                info['status'] = f"hold({held_info.get('status', 'unknown')})"

            # ------------------------------------------------------------------
            # Step E: clip control inputs to actuator limits.
            # ------------------------------------------------------------------
            u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)
            u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)

            # Update held values only after a real solve step.
            if should_solve:
                held_u = u_opt.copy()
                held_info = info.copy()

            # ------------------------------------------------------------------
            # Step F: sample environment disturbance noise.
            # ------------------------------------------------------------------
            noise = np.zeros(5)
            if self.disturbance_gen is not None:
                w = self.disturbance_gen.sample_single()
                noise[0] = w[0] * 0.1
                noise[1] = w[1] * 0.1
                noise[IDX_PSI] = w[3] * 0.01
                noise[IDX_V] = w[2] * 0.05
                noise[IDX_OMEGA] = w[4] * 0.01

            # ------------------------------------------------------------------
            # Step G: propagate dynamics to the next state.
            # ------------------------------------------------------------------
            x_next = discrete_step(x, u_opt) + noise
            x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)

            # ------------------------------------------------------------------
            # Step H: record this step into the result container.
            # ------------------------------------------------------------------
            result.controls.append(u_opt.copy())
            result.solve_times.append(info.get('solve_time', 0))
            result.solve_statuses.append(info.get('status', 'unknown'))
            result.solve_debug.append(info.get('debug'))
            result.ref_states.append(ref[0].copy())
            result.timestamps.append(t_sim)

            x = x_next
            result.states.append(x.copy())

            # ------------------------------------------------------------------
            # Step I: print detailed per-step logs for debugging.
            # ------------------------------------------------------------------
            if detailed_step_log and (
                detailed_step_log_max_steps is None or step < detailed_step_log_max_steps
            ):
                # Format key variables as readable strings.
                x_str = np.array2string(result.states[-2], precision=4, suppress_small=True)
                ref_str = np.array2string(ref[0], precision=4, suppress_small=True)
                u_str = np.array2string(u_opt, precision=4, suppress_small=True)
                noise_str = np.array2string(noise, precision=4, suppress_small=True)
                x_next_str = np.array2string(x_next, precision=4, suppress_small=True)
                # Print one step summary line.
                print(
                    f"[Step {step:04d}] t={t_sim:7.2f}s "
                    f"idx={idx:4d} s={current_s:8.2f}m ds={ds:7.3f}m "
                    f"cum={cumulative_s:8.2f}m prog={cumulative_s/max_s*100:6.2f}% lat={lat_err:8.4f}m"
                )
                print(f"  x      = {x_str}")
                print(f"  ref[0] = {ref_str}")
                print(f"  u_opt  = {u_str}")
                print(
                    f"  solve  = status={info.get('status','unknown')} "
                    f"time={info.get('solve_time', 0.0) * 1000:.2f}ms"
                )
                # Expand cost terms and active constraints when debug data exists.
                debug = info.get('debug')
                if debug:
                    step0 = debug.get('step0', {})
                    horizon = debug.get('horizon', {})
                    active = ','.join(debug.get('active_constraints', [])) or 'none'
                    key_parts = []
                    for key in (
                        'cost_track_vomega', 'cost_contour', 'cost_lag',
                        'cost_position',
                        'cost_heading', 'cost_heading_mpcc', 'cost_speed',
                        'cost_progress', 'cost_progress_mpcc', 'cost_du',
                        'cost_abs_u'
                    ):
                        if key in step0:
                            key_parts.append(f"{key}={step0[key]:.3f}")
                    if 'cost_cvar' in horizon:
                        key_parts.append(f"cost_cvar={horizon['cost_cvar']:.3f}")
                    if 'risk_eta' in horizon:
                        key_parts.append(f"risk_eta={horizon['risk_eta']:.3f}")
                    key_parts.append(f"v_slack_max={debug.get('v_slack_max', 0.0):.3f}")
                    key_parts.append(f"obs_slack_max={debug.get('obs_slack_max', 0.0):.3f}")
                    print(f"  diag   = {'; '.join(key_parts)}")
                    print(f"  active = {active}")
                print(f"  noise  = {noise_str}")
                print(f"  x_next = {x_next_str}")
                print(
                    f"  speed  = v:{result.states[-2][IDX_V]:.4f} -> {x_next[IDX_V]:.4f} m/s, "
                    f"omega:{result.states[-2][IDX_OMEGA]:.4f} -> {x_next[IDX_OMEGA]:.4f} rad/s"
                )

            # ------------------------------------------------------------------
            # Step J: periodic progress summaries.
            # ------------------------------------------------------------------
            if verbose:
                solve_t = info.get('solve_time', 0) * 1000
                if (step + 1) % 100 == 0:
                    print(f"  Step {step+1}/{max_steps}: "
                          f"speed={x[IDX_V]:.1f}m/s, "
                          f"lat_err={lat_err:.1f}m, "
                          f"progress={cumulative_s/max_s*100:.1f}%, "
                          f"solve={solve_t:.1f}ms")
                elif (step + 1) % 10 == 0:
                    # Lightweight progress point to avoid long silent periods.
                    status_tag = info.get('status', '?')
                    print(f"  ... Step {step+1}/{max_steps} "
                          f"(s={current_s:.1f}m, lat={lat_err:.2f}m, "
                          f"solve={solve_t:.0f}ms, status={status_tag})")

            # ------------------------------------------------------------------
            # Step K: termination checks.
            # ------------------------------------------------------------------
            # K1. Check for lap completion.
            if cumulative_s >= max_s * lap_fraction:
                result.lap_completed = True
                result.lap_time = (step + 1) * DT
                if verbose:
                    print(f"  Lap completed at step {step+1}, time={result.lap_time:.1f}s")
                break

            # K2. Check whether the vehicle hits the track boundary.
            if abs(lat_err) >= crash_lat_limit:
                result.crashed = True
                result.crash_step = step + 1
                result.crash_time = (step + 1) * DT
                result.crash_reason = (
                    f"track boundary hit: |lat_err|={abs(lat_err):.3f}m >= {crash_lat_limit:.3f}m"
                )
                if verbose:
                    print(
                        f"  Vehicle hit the track boundary at step {step+1} "
                        f"(lat_err={lat_err:.2f}m, threshold={crash_lat_limit:.2f}m)"
                    )
                break

            # K3. Check for large divergence to avoid wasting compute.
            if abs(lat_err) > 500:
                if verbose:
                    print(f"  Vehicle diverged at step {step+1} (lat_err={lat_err:.0f}m)")
                break

        # ==================================================================
        # End of simulation: finalize and return.
        # ==================================================================
        # Record the number of executed control steps.
        result.total_steps = len(result.controls)

        if verbose:
            # Print the average MPC solve time for quick runtime checks.
            avg_solve = np.mean(result.solve_times) if result.solve_times else 0
            print(f"  Simulation finished: {result.total_steps} steps, "
                  f"average solve time={avg_solve*1000:.1f}ms")

        return result

    # ==================================================================
    # Static helpers: result export and visualization
    # ==================================================================

    @staticmethod
    def _export_result_to_step_log(result, output_path):
        """Export simulation results as an aligned line-by-line text log."""
        # Convert SimResult storage into array form for indexed access.
        data = result.to_arrays()
        states = data['states']
        controls = data['controls']
        solve_times = data['solve_times']
        timestamps = data['timestamps']
        solve_debug = list(data.get('solve_debug', []))
        ref_states = np.array(result.ref_states)
        solve_statuses = list(result.solve_statuses)

        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            # Build the fixed-width header.
            header = (
                f"{'step':>5} {'t(s)':>8} {'status':<12} {'solve_ms':>9} "
                f"{'x':>9} {'y':>9} {'psi':>9} {'v':>8} {'omega':>8} "
                f"{'ref_v':>8} {'ref_omega':>10} {'a':>8} {'delta':>8} "
                f"{'next_v':>8} {'next_omega':>10}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # Write metadata lines prefixed with '#'.
            f.write(f"# method={result.method_name}\n")
            f.write(f"# track={result.track_name}\n")
            f.write(f"# lap_completed={result.lap_completed}\n")
            f.write(f"# lap_time={result.lap_time}\n")
            f.write(f"# total_steps={result.total_steps}\n")
            f.write(f"# crashed={result.crashed}\n")
            f.write(f"# crash_step={result.crash_step}\n")
            f.write(f"# crash_time={result.crash_time}\n")
            f.write(f"# crash_reason={result.crash_reason}\n")

            # Write the initial state for reproducibility.
            if len(states) > 0:
                f.write(
                    "# init_state="
                    f"[{states[0, 0]:.6f}, {states[0, 1]:.6f}, {states[0, 2]:.6f}, "
                    f"{states[0, 3]:.6f}, {states[0, 4]:.6f}]\n"
                )
            f.write("\n")

            # Write one row per simulation step.
            for step in range(len(controls)):
                x_t = states[step]
                x_next = states[step + 1]
                u_t = controls[step]
                # Fill with NaN if reference states are unexpectedly missing.
                ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
                solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
                status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
                debug = solve_debug[step] if step < len(solve_debug) else None
                t_sim = timestamps[step] if step < len(timestamps) else float(step)

                # Write the main data row.
                f.write(
                    f"{step:5d} {t_sim:8.3f} {status:<12.12} {solve_time_ms:9.3f} "
                    f"{x_t[0]:9.3f} {x_t[1]:9.3f} {x_t[2]:9.4f} {x_t[3]:8.3f} {x_t[4]:8.4f} "
                    f"{ref_t[3]:8.3f} {ref_t[4]:10.4f} {u_t[0]:8.4f} {u_t[1]:8.4f} "
                    f"{x_next[3]:8.3f} {x_next[4]:10.4f}\n"
                )
                # Append a diagnostic line when debug data exists.
                if debug:
                    step0 = debug.get('step0', {})
                    horizon = debug.get('horizon', {})
                    active = ','.join(debug.get('active_constraints', [])) or 'none'
                    f.write(
                        f"  # debug step0={step0} horizon={horizon} active={active} "
                        f"v_slack_max={debug.get('v_slack_max', 0.0):.6f} "
                        f"obs_slack_max={debug.get('obs_slack_max', 0.0):.6f}\n"
                    )

    @staticmethod
    def _export_result_to_compact_log(result, output_path):
        """Export a more compact one-line-per-step simulation log."""
        data = result.to_arrays()
        states = data['states']
        controls = data['controls']
        solve_times = data['solve_times']
        timestamps = data['timestamps']
        solve_debug = list(data.get('solve_debug', []))
        ref_states = np.array(result.ref_states)
        solve_statuses = list(result.solve_statuses)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            # Write the compact column header.
            f.write("# step t(s) v ref_v omega ref_omega a delta solve_ms status\n")

            for step in range(len(controls)):
                x_t = states[step]
                u_t = controls[step]
                ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
                solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
                status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
                debug = solve_debug[step] if step < len(solve_debug) else None
                t_sim = timestamps[step] if step < len(timestamps) else float(step)

                # Extract a short diagnostic summary with the main cost terms.
                diag_excerpt = ""
                if debug:
                    step0 = debug.get('step0', {})
                    important = []
                    for key in ('cost_track_vomega', 'cost_contour', 'cost_lag', 'cost_progress', 'cost_cvar'):
                        if key in step0:
                            important.append(f"{key}={step0[key]:.3f}")
                    if 'horizon' in debug and 'cost_cvar' in debug['horizon']:
                        important.append(f"cost_cvar={debug['horizon']['cost_cvar']:.3f}")
                    diag_excerpt = (" " + " ".join(important)) if important else ""

                # Write one compact data row.
                f.write(
                    f"{step:04d} "
                    f"{t_sim:8.3f} "
                    f"{x_t[3]:9.4f} "
                    f"{ref_t[3]:9.4f} "
                    f"{x_t[4]:9.4f} "
                    f"{ref_t[4]:9.4f} "
                    f"{u_t[0]:9.4f} "
                    f"{u_t[1]:9.4f} "
                    f"{solve_time_ms:9.3f} "
                    f"{status}{diag_excerpt}\n"
                )

    @staticmethod
    def _summarize_debug_diagnostics(result, top_k=5):
        """Aggregate debug diagnostics across all recorded steps."""
        # Filter out empty debug rows because some steps may not return them.
        debug_rows = [row for row in getattr(result, 'solve_debug', []) if row]
        if not debug_rows:
            return None

        step_acc = {}
        horizon_acc = {}
        active_counter = Counter()

        # Accumulate scalar diagnostics and active constraint frequencies.
        for row in debug_rows:
            for key, value in row.get('step0', {}).items():
                if isinstance(value, (int, float)):
                    step_acc.setdefault(key, []).append(float(value))
            for key, value in row.get('horizon', {}).items():
                if isinstance(value, (int, float)):
                    horizon_acc.setdefault(key, []).append(float(value))
            active_counter.update(row.get('active_constraints', []))

        # Compute mean and max statistics for every scalar field.
        def build_stats(acc):
            stats = {}
            for key, values in acc.items():
                if values:
                    stats[key] = {
                        'mean': float(np.mean(values)),
                        'max': float(np.max(values)),
                    }
            return stats

        step_stats = build_stats(step_acc)
        horizon_stats = build_stats(horizon_acc)
        # Rank dominant cost terms by mean value.
        dominant = sorted(
            [
                (key, vals['mean'])
                for key, vals in step_stats.items()
                if key.startswith('cost_')
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return {
            'step_stats': step_stats,
            'horizon_stats': horizon_stats,
            'dominant_costs': dominant,
            'active_constraints': dict(active_counter.most_common()),
        }

    @staticmethod
    def _export_result_debug_summary(result, output_path, top_k=5):
        """Append an aggregated debug summary to an existing log file."""
        summary = Simulator._summarize_debug_diagnostics(result, top_k=top_k)
        if summary is None:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'a') as f:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("DEBUG SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"method={result.method_name}\n")
            f.write(f"track={result.track_name}\n")
            f.write(f"total_steps={result.total_steps}\n\n")
            f.write(f"crashed={result.crashed}\n")
            f.write(f"crash_step={result.crash_step}\n")
            f.write(f"crash_time={result.crash_time}\n")
            f.write(f"crash_reason={result.crash_reason}\n\n")

            # Write dominant cost terms.
            f.write("[Top Dominant Cost Terms]\n")
            for key, mean_val in summary['dominant_costs']:
                max_val = summary['step_stats'][key]['max']
                f.write(f"  {key}: mean={mean_val:.6f}, max={max_val:.6f}\n")

            # Write active-constraint frequencies.
            f.write("\n[Active Constraints Frequency]\n")
            if summary['active_constraints']:
                for key, count in summary['active_constraints'].items():
                    f.write(f"  {key}: {count}\n")
            else:
                f.write("  none\n")

            # Write horizon-level diagnostics.
            f.write("\n[Horizon Diagnostics]\n")
            for key, vals in sorted(summary['horizon_stats'].items()):
                f.write(f"  {key}: mean={vals['mean']:.6f}, max={vals['max']:.6f}\n")

        return True

    @staticmethod
    def _build_track_for_result(result):
        """Rebuild the track instance referenced by a saved simulation result."""
        track_name = result.track_name
        if track_name == 'LusailShortTrack':
            from tracks.lusail_short_track import LusailShortTrack
            return LusailShortTrack()
        if track_name == 'LusailTrack':
            from tracks.lusail_track import LusailTrack
            return LusailTrack()
        if track_name == 'CustomWindingTrack':
            from tracks.custom_track import CustomWindingTrack
            return CustomWindingTrack()
        if track_name == 'SprintOvalTrack':
            from tracks.sprint_oval_track import SprintOvalTrack
            return SprintOvalTrack()
        return None

    @staticmethod
    def _export_result_figures(result, base_name):
        """Export PDF trajectory, state, and control figures for one result."""
        # Rebuild the track so plots can use the track geometry as background.
        track = Simulator._build_track_for_result(result)
        if track is None:
            return []

        # Delay plotting imports to avoid startup overhead and circular imports.
        from visualization.plot_trajectories import (
            plot_trajectory_comparison,
            plot_state_comparison,
            plot_control_comparison,
        )

        # Plotting functions expect a mapping from method name to result object.
        results = {result.method_name: result}
        trajectory_name = f"{base_name}_trajectory.pdf"
        states_name = f"{base_name}_states.pdf"
        controls_name = f"{base_name}_controls.pdf"

        plot_trajectory_comparison(
            results,
            track,
            title=f"{result.method_name} Trajectory on {result.track_name}",
            filename=trajectory_name,
        )
        plot_state_comparison(
            results,
            track,
            filename=states_name,
        )
        plot_control_comparison(
            results,
            filename=controls_name,
        )

        return [
            os.path.join(FIGURES_DIR, trajectory_name),
            os.path.join(FIGURES_DIR, states_name),
            os.path.join(FIGURES_DIR, controls_name),
        ]

    @staticmethod
    @staticmethod
    def _export_result_animation(result_path, base_name):
        """Call the external animation script and export a GIF."""
        output_path = os.path.join(FIGURES_DIR, f"{base_name}_animation.gif")
        # Build the path to the animation script.
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'visualization',
            'animate_simulation.py',
        )

        # Build subprocess arguments.
        cmd = [
            sys.executable,
            script_path,
            '--result', result_path,
            '--fps', str(int(ANIMATION_FPS)),
            '--fast-gif',
            '--save', output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[Warn] Animation export failed: {proc.stderr.strip() or proc.stdout.strip()}")
            return None
        return output_path

    @staticmethod
    def save_result(result, filename=None, save_dir=RESULTS_DIR):
        """Persist a simulation result and export associated artifacts."""
        # Ensure the save directory exists.
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            # Auto-generate a filename such as K-DRMPC_SprintOvalTrack.pkl.
            filename = f"{result.method_name}_{result.track_name}.pkl"
        path = os.path.join(save_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(result, f)

        # Build output basenames from the saved filename.
        base_name = os.path.splitext(os.path.basename(path))[0]
        log_dir = os.path.join(save_dir, 'logs')
        step_log_path = os.path.join(log_dir, f"{base_name}.log")

        Simulator._export_result_to_step_log(result, step_log_path)
        Simulator._export_result_debug_summary(result, step_log_path)
        figure_paths = Simulator._export_result_figures(result, base_name) if EXPORT_STATIC_FIGURES else []
        animation_path = Simulator._export_result_animation(path, base_name) if EXPORT_ANIMATION else None

        # Print generated artifact paths for quick inspection.
        print(f"Saved result to {path}")
        print(f"Log file: {step_log_path}")
        for figure_path in figure_paths:
            print(f"Figure: {figure_path}")
        if animation_path:
            print(f"Animation: {animation_path}")

    @staticmethod
    def load_result(filepath):
        """Load a previously saved simulation result from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
