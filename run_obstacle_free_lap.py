"""Run a v2 tracking baseline on a track, optionally with rectangular obstacles."""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONTROL_UPDATE_INTERVAL, FIGURES_DIR, MAX_SIM_STEPS, MODEL_DIR, NOMINAL_SIGMA, PLOT_TRACK_HALF_WIDTH, RESULTS_DIR, TRACK_HALF_WIDTH, VEHICLE_RADIUS
from controllers.v2 import KMPCController, LMPCBaselineController, PurePursuitBaselineController
from disturbance.disturbance_generator import DisturbanceGenerator
from simulation.simulator import Simulator
from tracks import base_track
from tracks.serpentine_track import SerpentineTrack
from tracks.sprint_oval_track import SprintOvalTrack
from tracks.bahrain_international_circuit import BahrainInternationalCircuit


TRACK_BUILDERS = {
    "SerpentineTrack": SerpentineTrack,
    "SprintOvalTrack": SprintOvalTrack,
    "BahrainInternationalCircuit": BahrainInternationalCircuit,
}
PAPER_THETA = 0.10
PAPER_EPSILON = 0.10


def build_obstacle_free_track(
    track_name,
    obstacle_count=0,
    obstacle_width_increase=0.0,
    obstacle_width_ratio=None,
    obstacle_width_seed=20261028,
    obstacle_start_clearance=0.0,
):
    track = TRACK_BUILDERS[track_name]()
    obstacle_start_clearance = float(obstacle_start_clearance)
    if obstacle_start_clearance > 0.0:
        if not isinstance(track, SerpentineTrack):
            raise ValueError("obstacle_start_clearance is currently supported only for Serpentine tracks")
        track._place_obstacles(start_clearance_m=obstacle_start_clearance)
    obstacle_count = int(obstacle_count)
    if obstacle_count < 0 or obstacle_count > len(track._rect_obstacles):
        raise ValueError(f"obstacle_count must be between 0 and {len(track._rect_obstacles)}")
    obstacle_width_increase = float(obstacle_width_increase)
    if obstacle_width_increase < 0.0:
        raise ValueError("obstacle_width_increase must be non-negative")
    original_rectangles = track._rect_obstacles[:obstacle_count]
    if obstacle_width_ratio is not None:
        ratio_min, ratio_max = map(float, obstacle_width_ratio)
        if not (0.0 < ratio_min <= ratio_max <= 1.0):
            raise ValueError("obstacle_width_ratio must satisfy 0 < min <= max <= 1")
        rng = np.random.default_rng(obstacle_width_seed)
        center_x, center_y = track.get_centerline()
        heading = track.get_heading()
        rectangles = []
        for cx, cy, length, _, angle in original_rectangles:
            index, _, lateral = track.closest_point(cx, cy)
            side = float(np.sign(lateral)) or (1.0 if len(rectangles) % 2 == 0 else -1.0)
            width = float(rng.uniform(ratio_min, ratio_max) * 2.0 * TRACK_HALF_WIDTH)
            lateral_offset = side * (TRACK_HALF_WIDTH - 0.5 * width - 0.5)
            normal_x = -np.sin(heading[index])
            normal_y = np.cos(heading[index])
            new_cx = center_x[index] + lateral_offset * normal_x
            new_cy = center_y[index] + lateral_offset * normal_y
            rectangles.append((new_cx, new_cy, length, width, angle))
        track._rect_obstacles = rectangles
        track._obstacle_width_ratio = (ratio_min, ratio_max)
        track._obstacle_width_seed = int(obstacle_width_seed)
    else:
        track._rect_obstacles = [
            (cx, cy, length, width + obstacle_width_increase, angle)
            for cx, cy, length, width, angle in original_rectangles
        ]
        track._obstacle_width_ratio = None
    track._obstacles = [
        (cx, cy, 0.5 * max(length, width))
        for cx, cy, length, width, angle in track._rect_obstacles
    ]
    track._obstacle_width_increase = obstacle_width_increase
    base_track.ENABLE_OBSTACLES = obstacle_count > 0
    return track


def reverse_track_direction(track):
    point_indices = np.concatenate(([0], np.arange(track.num_points() - 1, 0, -1)))
    track._centerline_x = track._centerline_x[point_indices].copy()
    track._centerline_y = track._centerline_y[point_indices].copy()
    track._compute_geometry(track._centerline_x, track._centerline_y)
    return track


def minimum_rect_obstacle_clearance(states, track):
    minimum_clearance = float("inf")
    for cx, cy, length, width, angle in track._rect_obstacles:
        dx = states[:, 0] - cx
        dy = states[:, 1] - cy
        local_x = np.cos(angle) * dx + np.sin(angle) * dy
        local_y = -np.sin(angle) * dx + np.cos(angle) * dy
        outside_x = np.maximum(np.abs(local_x) - 0.5 * length, 0.0)
        outside_y = np.maximum(np.abs(local_y) - 0.5 * width, 0.0)
        clearance = np.hypot(outside_x, outside_y) - VEHICLE_RADIUS
        minimum_clearance = min(minimum_clearance, float(np.min(clearance)))
    return minimum_clearance


def draw_rect_obstacles(ax, track):
    from matplotlib.patches import Polygon

    for index, (cx, cy, length, width, angle) in enumerate(track._rect_obstacles):
        half_length = 0.5 * length
        half_width = 0.5 * width
        local_corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ])
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        corners = local_corners @ rotation.T + np.array([cx, cy])
        ax.add_patch(Polygon(
            corners,
            closed=True,
            facecolor="#d95f02",
            edgecolor="#8c2d04",
            linewidth=1.2,
            alpha=0.8,
            zorder=4,
            label="Obstacle block" if index == 0 else None,
        ))


def _draw_track_geometry(ax, track):
    center_x, center_y = track.get_centerline()
    heading = track.get_heading()
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    half_width = float(PLOT_TRACK_HALF_WIDTH)
    left_x = center_x + half_width * normal_x
    left_y = center_y + half_width * normal_y
    right_x = center_x - half_width * normal_x
    right_y = center_y - half_width * normal_y

    ax.plot(center_x, center_y, "--", color="#a0a0a0", linewidth=0.9, label="Centerline")
    ax.plot(left_x, left_y, color="#444444", linewidth=0.9, label="Track boundary")
    ax.plot(right_x, right_y, color="#444444", linewidth=0.9)
    draw_rect_obstacles(ax, track)
    return center_x, center_y, left_x, left_y, right_x, right_y


def _configure_track_axes(ax, geometry, track=None, margin=12.0):
    from matplotlib.ticker import MultipleLocator

    center_x, center_y, left_x, left_y, right_x, right_y = geometry
    plot_bounds = getattr(track, "_plot_bounds", None)
    if plot_bounds is None:
        all_x = np.concatenate([center_x, left_x, right_x])
        all_y = np.concatenate([center_y, left_y, right_y])
        ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
        ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))
    else:
        ax.set_xlim(plot_bounds[0], plot_bounds[1])
        ax.set_ylim(plot_bounds[2], plot_bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.xaxis.set_major_locator(MultipleLocator(50.0))
    ax.yaxis.set_major_locator(MultipleLocator(50.0))


def export_track_preview(track, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    geometry = _draw_track_geometry(ax, track)
    center_x, center_y = geometry[:2]
    ax.plot(center_x[0], center_y[0], "o", color="#238b45", markersize=7, label="Start / finish")
    obstacle_count = len(track._rect_obstacles)
    title = "Track preview"
    if obstacle_count:
        title += f" with {obstacle_count} obstacle block{'s' if obstacle_count != 1 else ''}"
    ax.set_title(f"{title}: {track.__class__.__name__}")
    _configure_track_axes(ax, geometry, track=track)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=90, pil_kwargs={"optimize": True})
    plt.close(fig)


def export_obstacle_free_plot(
    result,
    track,
    output_path,
    trajectory_sample_step=6,
    distributionally_robust=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = np.asarray(result.states)
    robust_state = distributionally_robust
    if robust_state is None:
        robust_state = getattr(result, "distributionally_robust", None)
    trajectory_color = "#1769aa" if robust_state is not False else "#d62728"
    if trajectory_sample_step < 1:
        raise ValueError("trajectory_sample_step must be at least 1")
    trajectory_indices = np.arange(0, len(states), trajectory_sample_step)
    if trajectory_indices[-1] != len(states) - 1:
        trajectory_indices = np.append(trajectory_indices, len(states) - 1)

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    geometry = _draw_track_geometry(ax, track)
    center_x, center_y = geometry[:2]
    ax.plot(
        states[trajectory_indices, 0],
        states[trajectory_indices, 1],
        color=trajectory_color,
        linewidth=1.8,
        label=f"{result.method_name} (1/{trajectory_sample_step} samples)",
    )
    ax.plot(states[0, 0], states[0, 1], "o", color="#238b45", markersize=7, label="Start")
    if result.crashed:
        ax.plot(
            states[-1, 0],
            states[-1, 1],
            marker="x",
            color="#d62728",
            markersize=12,
            markeredgewidth=3,
            linestyle="None",
            label="",
            zorder=6,
        )
    else:
        endpoint_label = "Finish" if result.lap_completed else "Current position"
        ax.plot(states[-1, 0], states[-1, 1], "o", color="#d62728", markersize=7, label=endpoint_label)

    _configure_track_axes(ax, geometry, track=track)
    obstacle_count = len(track._rect_obstacles)
    if result.lap_completed:
        title = "Obstacle-free lap" if not obstacle_count else f"Lap with {obstacle_count} obstacle block{'s' if obstacle_count != 1 else ''}"
    else:
        title = f"Partial run ({result.total_steps} steps)"
        if obstacle_count:
            title += f" with {obstacle_count} obstacle block{'s' if obstacle_count != 1 else ''}"
    if obstacle_count and track._obstacle_width_ratio is not None:
        lower, upper = track._obstacle_width_ratio
        title += f" ({lower:.0%}-{upper:.0%} road width)"
    ax.set_title(f"{title}: {track.__class__.__name__}")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=90, pil_kwargs={"optimize": True})
    plt.close(fig)


def export_obstacle_free_gif(
    result,
    track,
    output_path,
    max_frames=80,
    trajectory_sample_step=6,
    fps=20,
    distributionally_robust=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    states = np.asarray(result.states)
    robust_state = distributionally_robust
    if robust_state is None:
        robust_state = getattr(result, "distributionally_robust", None)
    trajectory_color = "#1769aa" if robust_state is not False else "#d62728"
    full_trajectory_color = "#8bb7d8" if robust_state is not False else "#e6a0a0"
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if trajectory_sample_step < 1:
        raise ValueError("trajectory_sample_step must be at least 1")

    frame_indices = np.linspace(
        0,
        len(states) - 1,
        min(max_frames, len(states)),
        dtype=int,
    )
    trajectory_indices = np.arange(0, len(states), trajectory_sample_step)
    if trajectory_indices[-1] != len(states) - 1:
        trajectory_indices = np.append(trajectory_indices, len(states) - 1)
    fig, ax = plt.subplots(figsize=(8.8, 6.4), constrained_layout=True)
    geometry = _draw_track_geometry(ax, track)
    center_x, center_y = geometry[:2]
    ax.plot(states[0, 0], states[0, 1], "o", color="#238b45", markersize=6, label="Start")
    ax.plot(
        states[trajectory_indices, 0],
        states[trajectory_indices, 1],
        color=full_trajectory_color,
        linewidth=1.3,
        alpha=0.55,
        label="Full trajectory",
    )
    trajectory, = ax.plot([], [], color=trajectory_color, linewidth=1.8, label=result.method_name)
    vehicle, = ax.plot([], [], "o", color="#d62728", markersize=7, zorder=5, label="Vehicle")
    progress_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    _configure_track_axes(ax, geometry, track=track)
    obstacle_count = len(track._rect_obstacles)
    title = "Obstacle-free lap" if not obstacle_count else f"Lap with {obstacle_count} obstacle block{'s' if obstacle_count != 1 else ''}"
    ax.set_title(f"{title}: {track.__class__.__name__}")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    def update(frame):
        current_index = int(frame_indices[frame])
        sampled_end = np.searchsorted(trajectory_indices, current_index, side="right")
        shown_indices = trajectory_indices[:sampled_end]
        if len(shown_indices) == 0 or shown_indices[-1] != current_index:
            shown_indices = np.append(shown_indices, current_index)
        trajectory.set_data(states[shown_indices, 0], states[shown_indices, 1])
        vehicle.set_data([states[current_index, 0]], [states[current_index, 1]])
        progress_text.set_text(
            f"Step {current_index}/{len(states) - 1}  |  v={states[current_index, 3]:.1f} m/s"
        )
        return trajectory, vehicle, progress_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=True,
        repeat=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-name", choices=sorted(TRACK_BUILDERS), default="SerpentineTrack")
    parser.add_argument("--reverse-direction", action="store_true", help="drive the closed track in the opposite direction")
    parser.add_argument("--steps", type=int, default=MAX_SIM_STEPS * 4)
    parser.add_argument("--control-update-interval", type=int, default=CONTROL_UPDATE_INTERVAL, help="re-solve the controller every N simulation steps")
    parser.add_argument("--lap-fraction", type=float, default=0.995)
    parser.add_argument("--max-speed", type=float, default=8.0)
    parser.add_argument("--controller", choices=("pure-pursuit", "lmpc", "kmpc", "paper-qp"), default="pure-pursuit")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="directory containing Koopman checkpoint, projection, and normalization files")
    parser.add_argument("--trace-kmpc", action="store_true", help="print Koopman model and per-solve trace logs")
    parser.add_argument(
        "--distributionally-robust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable distributionally robust terms for paper-qp (default: enabled)",
    )
    parser.add_argument("--theta", type=float, default=PAPER_THETA, help="Wasserstein radius for paper-qp (paper default: 0.10)")
    parser.add_argument("--epsilon", type=float, default=PAPER_EPSILON, help="DR chance-constraint risk level for paper-qp (paper default: 0.10)")
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--measurement-noise", action="store_true", help="diagnostic only: perturb controller observations")
    noise_group.add_argument("--process-disturbance", action="store_true", help="add paper-style disturbance to the plant state transition")
    parser.add_argument("--noise-sigma", type=float, default=NOMINAL_SIGMA, help="GMM base scale (paper settings: nominal 0.5, shifted 0.75 or 1.0)")
    parser.add_argument("--noise-seed", type=int, default=42, help="disturbance random seed")
    parser.add_argument("--obstacle-count", type=int, default=0, help="enable the first N track obstacles")
    parser.add_argument("--obstacle-width-increase", type=float, default=0.0, help="increase each block's total width by this many meters")
    parser.add_argument("--obstacle-width-ratio", nargs=2, type=float, metavar=("MIN", "MAX"), help="randomly scale each block width by a fraction of total road width")
    parser.add_argument("--obstacle-width-seed", type=int, default=20261028, help="seed for reproducible random block widths")
    parser.add_argument("--obstacle-start-clearance", type=float, default=0.0, help="reserve this arc length after/before start-finish without obstacle centers")
    parser.add_argument("--name-suffix", default="pure_pursuit_obstacle_free")
    parser.add_argument("--no-gif", action="store_true", help="skip GIF export")
    parser.add_argument("--gif-only", action="store_true", help="regenerate GIF from an existing result pickle")
    parser.add_argument("--static-only", action="store_true", help="generate a static PNG from an existing result pickle")
    parser.add_argument("--preview-only", action="store_true", help="render the selected track with the shared track style and exit")
    parser.add_argument("--static-plot", action="store_true", help="also save a static PNG after simulation")
    parser.add_argument("--gif-frames", type=int, default=80, help="maximum number of GIF frames")
    parser.add_argument("--gif-sample-step", type=int, default=6, help="plot every Nth simulation state")
    parser.add_argument("--static-sample-step", type=int, default=6, help="plot every Nth simulation state in the static image")
    args = parser.parse_args()

    if args.gif_only and args.no_gif:
        parser.error("--gif-only cannot be combined with --no-gif")
    if args.gif_only and args.static_only:
        parser.error("--gif-only cannot be combined with --static-only")
    if args.static_only and args.static_plot:
        parser.error("--static-only cannot be combined with --static-plot")
    if args.gif_frames < 2:
        parser.error("--gif-frames must be at least 2")
    if args.gif_sample_step < 1:
        parser.error("--gif-sample-step must be at least 1")
    if args.static_sample_step < 1:
        parser.error("--static-sample-step must be at least 1")
    if args.control_update_interval < 1:
        parser.error("--control-update-interval must be at least 1")
    if (args.measurement_noise or args.process_disturbance) and args.noise_sigma <= 0.0:
        parser.error("--noise-sigma must be positive when a disturbance mode is enabled")
    if args.obstacle_width_increase < 0.0:
        parser.error("--obstacle-width-increase must be non-negative")
    if args.obstacle_start_clearance < 0.0:
        parser.error("--obstacle-start-clearance must be non-negative")
    if args.obstacle_start_clearance > 0.0 and args.track_name != "SerpentineTrack":
        parser.error("--obstacle-start-clearance is currently supported only for Serpentine tracks")
    if args.obstacle_width_increase > 0.0 and args.obstacle_count == 0:
        parser.error("--obstacle-width-increase requires --obstacle-count")
    if args.obstacle_width_ratio is not None and args.obstacle_width_increase > 0.0:
        parser.error("--obstacle-width-ratio cannot be combined with --obstacle-width-increase")
    if args.obstacle_width_ratio is not None:
        ratio_min, ratio_max = args.obstacle_width_ratio
        if not (0.0 < ratio_min <= ratio_max <= 1.0):
            parser.error("--obstacle-width-ratio must satisfy 0 < MIN <= MAX <= 1")
        if args.obstacle_count == 0:
            parser.error("--obstacle-width-ratio requires --obstacle-count")

    track = build_obstacle_free_track(
        args.track_name,
        args.obstacle_count,
        obstacle_width_increase=args.obstacle_width_increase,
        obstacle_width_ratio=args.obstacle_width_ratio,
        obstacle_width_seed=args.obstacle_width_seed,
        obstacle_start_clearance=args.obstacle_start_clearance,
    )
    if args.reverse_direction:
        track = reverse_track_direction(track)
    if args.preview_only:
        preview_name = f"{args.track_name}_preview"
        if args.reverse_direction:
            preview_name += "_reverse_direction"
        if args.obstacle_count:
            preview_name += f"_obs{args.obstacle_count}"
        if args.obstacle_width_ratio is not None:
            lower, upper = (str(round(100.0 * value)) for value in args.obstacle_width_ratio)
            preview_name += f"_w{lower}-{upper}pct"
        os.makedirs(FIGURES_DIR, exist_ok=True)
        preview_path = os.path.join(FIGURES_DIR, f"{preview_name}.png")
        export_track_preview(track, preview_path)
        print(f"Track preview saved to {preview_path}")
        return
    controller_type = {
        "pure-pursuit": PurePursuitBaselineController,
        "lmpc": LMPCBaselineController,
    }.get(args.controller)
    if args.controller in ("kmpc", "paper-qp"):
        from model.koopman_trainer import load_trained_model

        checkpoint_path = os.path.join(args.model_dir, "best_koopman_model.pth")
        projection_path = os.path.join(args.model_dir, "projection_D.npy")
        norm_path = os.path.join(args.model_dir, "norm_params.json")
        for required_path in (checkpoint_path, projection_path, norm_path):
            if not os.path.isfile(required_path):
                parser.error(f"{args.controller} model file not found: {required_path}")
        model = load_trained_model(checkpoint_path)
        projection = np.load(projection_path)
        with open(norm_path) as handle:
            norm_params = json.load(handle)
        if args.trace_kmpc and args.controller == "kmpc":
            koopman_A, koopman_B, _ = model.get_matrices()
            print(f"[KMPC trace] loaded checkpoint: {checkpoint_path}")
            print(f"[KMPC trace] loaded projection: {projection_path}, shape={projection.shape}")
            print(f"[KMPC trace] loaded normalization: {norm_path}, keys={sorted(norm_params)}")
            print(f"[KMPC trace] learned Koopman matrices: A={koopman_A.shape}, B={koopman_B.shape}")
        if args.controller == "kmpc":
            controller = KMPCController(
                track,
                model,
                projection,
                norm_params,
                max_speed=args.max_speed,
                trace=args.trace_kmpc,
            )
        else:
            from controllers.v2.kdrmpc_paper_qp import KDRMPCPaperQPController

            controller = KDRMPCPaperQPController(
                track,
                model,
                projection,
                norm_params,
                distributionally_robust=args.distributionally_robust,
                theta=args.theta,
                epsilon=args.epsilon,
                max_speed=args.max_speed,
            )
    else:
        controller = controller_type(track, max_speed=args.max_speed)
    suffix = args.name_suffix.strip().replace(" ", "_")
    if args.reverse_direction:
        suffix += "_reverse_direction"
    if args.process_disturbance:
        sigma_tag = f"{args.noise_sigma:g}".replace(".", "p")
        suffix += f"_process_disturbance_sigma{sigma_tag}_seed{args.noise_seed}"
    elif args.measurement_noise:
        sigma_tag = f"{args.noise_sigma:g}".replace(".", "p")
        suffix += f"_measurement_noise_sigma{sigma_tag}_seed{args.noise_seed}"
    if args.controller == "paper-qp":
        suffix += f"_theta{args.theta:g}_epsilon{args.epsilon:g}".replace(".", "p")
        suffix += "_dr" if args.distributionally_robust else "_non_dr"
    obstacle_suffix = f"_obs{args.obstacle_count}" if args.obstacle_count else ""
    if args.obstacle_width_increase:
        width_tag = f"{args.obstacle_width_increase:g}".replace(".", "p")
        obstacle_suffix += f"_wplus{width_tag}m"
    if args.obstacle_width_ratio is not None:
        lower_tag, upper_tag = (str(round(100.0 * value)) for value in args.obstacle_width_ratio)
        obstacle_suffix += f"_w{lower_tag}-{upper_tag}pct"
    if args.obstacle_start_clearance > 0.0:
        clearance_tag = f"{args.obstacle_start_clearance:g}".replace(".", "p")
        obstacle_suffix += f"_startclear{clearance_tag}m"
    base_name = f"{controller.name}_{args.track_name}{obstacle_suffix}_{suffix}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{base_name}.pkl")

    if args.gif_only or args.static_only:
        with open(result_path, "rb") as handle:
            result = pickle.load(handle)
    else:
        center_x, center_y = track.get_centerline()
        heading = track.get_heading()
        curvature = track.get_curvature()
        initial_speed = controller.target_speed_at_s(0.0)
        x0 = np.array([
            center_x[0],
            center_y[0],
            heading[0],
            initial_speed,
            initial_speed * curvature[0],
        ])

        disturbance_gen = None
        noise_mode = "process"
        if args.measurement_noise or args.process_disturbance:
            disturbance_gen = DisturbanceGenerator(
                sigma=args.noise_sigma,
                seed=args.noise_seed,
                enabled=True,
            )
            noise_mode = "process" if args.process_disturbance else "measurement"
        simulator = Simulator(track, controller, disturbance_gen, noise_mode=noise_mode)
        result = simulator.run(
            x0=x0,
            max_steps=args.steps,
            lap_fraction=args.lap_fraction,
            verbose=True,
            control_update_interval=args.control_update_interval,
        )
        with open(result_path, "wb") as handle:
            pickle.dump(result, handle)

    states = np.asarray(result.states)
    summary = {
        "controller": controller.name,
        "distributionally_robust": (
            bool(args.distributionally_robust) if args.controller == "paper-qp" else None
        ),
        "wasserstein_theta": args.theta if args.controller == "paper-qp" else None,
        "chance_constraint_epsilon": args.epsilon if args.controller == "paper-qp" else None,
        "track": args.track_name,
        "reverse_direction": bool(args.reverse_direction),
        "obstacles": len(track._rect_obstacles),
        "obstacle_width_increase_m": args.obstacle_width_increase,
        "obstacle_start_clearance_m": args.obstacle_start_clearance,
        "measurement_noise_enabled": bool(args.measurement_noise),
        "measurement_noise_sigma": args.noise_sigma if args.measurement_noise else None,
        "measurement_noise_seed": args.noise_seed if args.measurement_noise else None,
        "process_disturbance_enabled": bool(args.process_disturbance),
        "process_disturbance_sigma": args.noise_sigma if args.process_disturbance else None,
        "process_disturbance_seed": args.noise_seed if args.process_disturbance else None,
        "disturbance_injection_mode": (
            "process" if args.process_disturbance else "measurement" if args.measurement_noise else "none"
        ),
        "lap_completed": bool(result.lap_completed),
        "lap_time_s": result.lap_time,
        "steps": result.total_steps,
        "control_update_interval": args.control_update_interval,
        "crashed": bool(result.crashed),
        "crash_reason": result.crash_reason,
        "mean_speed_mps": float(np.mean(states[:, 3])),
        "max_lateral_error_m": float(max(
            abs(track.closest_point(x[0], x[1])[2]) for x in states
        )),
        "result_pickle": result_path,
    }
    if track._obstacle_width_ratio is not None:
        summary["obstacle_width_ratio_of_road"] = list(track._obstacle_width_ratio)
        summary["obstacle_width_seed"] = track._obstacle_width_seed
    if track._rect_obstacles:
        summary["minimum_obstacle_clearance_m"] = minimum_rect_obstacle_clearance(states, track)

    plot_path = os.path.join(FIGURES_DIR, f"{base_name}_trajectory.png")
    if args.static_only or args.static_plot:
        export_obstacle_free_plot(
            result,
            track,
            plot_path,
            trajectory_sample_step=args.static_sample_step,
            distributionally_robust=(
                bool(args.distributionally_robust) if args.controller == "paper-qp" else None
            ),
        )
        summary["static_plot"] = plot_path
    elif os.path.isfile(plot_path):
        summary["static_plot"] = plot_path

    gif_path = os.path.join(FIGURES_DIR, f"{base_name}_animation.gif")
    if not args.no_gif and not args.static_only:
        export_obstacle_free_gif(
            result,
            track,
            gif_path,
            max_frames=args.gif_frames,
            trajectory_sample_step=args.gif_sample_step,
            distributionally_robust=(
                bool(args.distributionally_robust) if args.controller == "paper-qp" else None
            ),
        )
        summary["gif"] = gif_path
    elif os.path.isfile(gif_path):
        summary["gif"] = gif_path

    summary_path = os.path.join(RESULTS_DIR, f"{base_name}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nSaved result: {result_path}")
    print(f"Saved summary: {summary_path}")
    if summary.get("gif"):
        print(f"Saved GIF: {summary['gif']}")
    if summary.get("static_plot"):
        print(f"Saved static plot: {summary['static_plot']}")
    print(f"Lap completed: {summary['lap_completed']}")

    if not result.lap_completed or (track._rect_obstacles and summary["minimum_obstacle_clearance_m"] <= 0.0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()