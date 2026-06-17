"""Noise comparison experiment for the paper-faithful K-DRMPC controller.

This script mirrors the practical comparison entry point, but uses the
parallel paper controller implementation so the main experiment path remains
untouched.
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as _cfg

_cfg.ENABLE_DISTURBANCE = True
_cfg.ENABLE_OBSTACLES = True

from config import CONTROL_UPDATE_INTERVAL, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from controllers.kdrmpc_paper_controller import KDRMPCPaperController  # noqa: E402
from disturbance.disturbance_generator import DisturbanceGenerator  # noqa: E402
from run_noise_comparison import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_SEED,
    DEFAULT_SIGMA,
    DEFAULT_STEPS,
    DEFAULT_THETA,
    N_DISTURBANCE_SAMPLES,
    TRACK_NAME,
    generate_dual_line_animation,
    load_koopman,
    print_comparison,
)
from simulation.simulator import Simulator  # noqa: E402
from tracks.serpentine_track import SerpentineTrack  # noqa: E402
import numpy as np  # noqa: E402

PAPER_DUAL_GIF_TEMPLATE = "Comparison_{track}_paper_dual_line{suffix}.gif"
PAPER_SUMMARY_TEMPLATE = "noise_comparison_paper_summary{suffix}.json"


def persist_result(result, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "wb") as handle:
        pickle.dump(result, handle)
    print(f"Saved result artifact: {path}")
    return path


def cleanup_previous_outputs(tags, suffix=""):
    removed = []
    for tag in tags:
        base = f"{tag}_{TRACK_NAME}"
        path = os.path.join(RESULTS_DIR, f"{base}.pkl")
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    comparison_gif = os.path.join(
        FIGURES_DIR,
        PAPER_DUAL_GIF_TEMPLATE.format(track=TRACK_NAME, suffix=suffix),
    )
    if os.path.exists(comparison_gif):
        os.remove(comparison_gif)
        removed.append(comparison_gif)

    summary_path = os.path.join(
        RESULTS_DIR,
        PAPER_SUMMARY_TEMPLATE.format(suffix=suffix if suffix else ""),
    )
    if os.path.exists(summary_path):
        os.remove(summary_path)
        removed.append(summary_path)

    if removed:
        print("\nRemoved stale paper files:")
        for path in removed:
            print(f"  - {path}")
    else:
        print("\nNo stale paper files to remove")


def initial_state_from_s(track, start_s):
    if start_s is None:
        return None

    arc_length = track.get_arc_length()
    start_s = float(start_s) % float(track.total_length())
    idx = int(np.argmin(np.abs(arc_length - start_s)))
    cx, cy = track.get_centerline()
    heading = track.get_heading()
    ref_init = track.get_reference_v_omega(idx, 1)
    return np.array([cx[idx], cy[idx], heading[idx], ref_init[0, 0], ref_init[0, 1]])


def run_experiment(
    model,
    D,
    norm_params,
    robust,
    tag,
    sigma,
    theta,
    epsilon,
    seed,
    max_steps,
    start_s,
    halfspace_mode,
    hard_obstacle_constraints,
    obstacle_slack_penalty,
    obstacle_safety_scale,
    obstacle_activation_margin,
    prediction_horizon,
    use_worst_case_cost,
    robust_track_tightening,
    robust_track_tightening_scale,
    robust_track_tightening_max,
    robust_track_slack_penalty_scale,
    robust_min_speed_floor,
    robust_obstacle_speed_weight,
    robust_obstacle_speed_preview,
    robust_obstacle_accel_weight,
    control_update_interval,
):
    track = SerpentineTrack()
    dist_gen = DisturbanceGenerator(sigma=sigma, seed=seed)
    w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

    obstacle_strategy = "robust" if robust else "non-robust"
    controller = KDRMPCPaperController(
        model,
        D,
        norm_params,
        disturbance_samples=w_empirical,
        theta=theta,
        epsilon=epsilon,
        obstacle_strategy=obstacle_strategy,
        hard_obstacle_constraints=hard_obstacle_constraints,
        halfspace_mode=halfspace_mode,
        track=track,
        obstacle_slack_penalty=obstacle_slack_penalty,
        obstacle_safety_scale=obstacle_safety_scale,
        obstacle_activation_margin=obstacle_activation_margin,
        prediction_horizon=prediction_horizon,
        use_worst_case_cost=(use_worst_case_cost and robust),
        robust_track_tightening=(robust_track_tightening and robust),
        robust_track_tightening_scale=robust_track_tightening_scale,
        robust_track_tightening_max=robust_track_tightening_max,
        robust_track_slack_penalty_scale=robust_track_slack_penalty_scale,
        robust_min_speed_floor=robust_min_speed_floor,
        robust_obstacle_speed_weight=(robust_obstacle_speed_weight if robust else 0.0),
        robust_obstacle_speed_preview=robust_obstacle_speed_preview,
        robust_obstacle_accel_weight=(robust_obstacle_accel_weight if robust else 0.0),
    )

    simulator = Simulator(track, controller, dist_gen)
    x0 = initial_state_from_s(track, start_s)
    result = simulator.run(
        x0=x0,
        max_steps=max_steps,
        verbose=True,
        lap_fraction=1.0,
        detailed_step_log=True,
        detailed_step_log_max_steps=10,
        control_update_interval=control_update_interval,
    )

    persist_result(result, f"{tag}_{TRACK_NAME}.pkl")
    return result, track


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper controller comparison: robust vs non-robust K-DRMPC"
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="measurement noise sigma")
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA, help="Wasserstein theta")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="CVaR epsilon")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="disturbance RNG seed")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="max simulation steps")
    parser.add_argument(
        "--control-update-interval",
        type=int,
        default=CONTROL_UPDATE_INTERVAL,
        help="solve MPC every N simulation steps; held control is used between solves",
    )
    parser.add_argument("--start-s", type=float, default=None, help="optional initial arclength position [m]")
    parser.add_argument(
        "--halfspace-mode",
        choices=["track-normal", "axis"],
        default="track-normal",
        help="paper obstacle half-space selection rule",
    )
    parser.add_argument(
        "--soft-obstacle-constraints",
        action="store_true",
        help="diagnostic mode: allow paper obstacle constraints to use slack",
    )
    parser.add_argument(
        "--obstacle-slack-penalty",
        type=float,
        default=None,
        help="diagnostic slack penalty for soft obstacle constraints",
    )
    parser.add_argument(
        "--obstacle-safety-scale",
        type=float,
        default=1.0,
        help="diagnostic scale applied to inflated obstacle clearance",
    )
    parser.add_argument(
        "--obstacle-activation-margin",
        type=float,
        default=8.0,
        help="extra radius for prediction-step obstacle constraint activation",
    )
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=20,
        help="paper controller prediction horizon; manuscript uses 20 steps",
    )
    parser.add_argument(
        "--no-worst-case-cost",
        action="store_true",
        help="diagnostic mode: disable robust empirical worst-case cost term",
    )
    parser.add_argument(
        "--no-robust-track-tightening",
        action="store_true",
        help="diagnostic mode: disable robust track-boundary uncertainty tightening",
    )
    parser.add_argument(
        "--robust-track-tightening-scale",
        type=float,
        default=4.0,
        help="scale for robust track-boundary uncertainty tightening",
    )
    parser.add_argument(
        "--robust-track-slack-penalty-scale",
        type=float,
        default=3.0,
        help="robust-only multiplier for the track-boundary slack penalty",
    )
    parser.add_argument(
        "--robust-track-tightening-max",
        type=float,
        default=3.0,
        help="maximum robust track-boundary tightening in meters",
    )
    parser.add_argument(
        "--robust-min-speed-floor",
        type=float,
        default=1.0,
        help="robust-only soft minimum predicted speed floor in m/s",
    )
    parser.add_argument(
        "--robust-obstacle-speed-weight",
        type=float,
        default=5.0,
        help="robust-only objective weight on predicted speed during active obstacle windows",
    )
    parser.add_argument(
        "--robust-obstacle-speed-preview",
        type=float,
        default=0.0,
        help="diagnostic forward arclength preview distance [m] for robust obstacle-window speed objective; 0 uses the obstacle activation window",
    )
    parser.add_argument(
        "--robust-obstacle-accel-weight",
        type=float,
        default=80.0,
        help="robust-only objective weight on positive acceleration during active obstacle windows",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="skip dual-line GIF generation for long validation runs",
    )
    parser.add_argument("--name-suffix", type=str, default="", help="output filename suffix")
    return parser.parse_args()


def main():
    args = parse_args()
    sigma = float(args.sigma)
    theta = float(args.theta)
    epsilon = float(args.epsilon)
    seed = int(args.seed)
    max_steps = int(args.steps)
    control_update_interval = int(args.control_update_interval)
    start_s = args.start_s
    halfspace_mode = args.halfspace_mode
    hard_obstacle_constraints = not bool(args.soft_obstacle_constraints)
    obstacle_slack_penalty = args.obstacle_slack_penalty
    obstacle_safety_scale = float(args.obstacle_safety_scale)
    obstacle_activation_margin = float(args.obstacle_activation_margin)
    prediction_horizon = int(args.prediction_horizon)
    use_worst_case_cost = not bool(args.no_worst_case_cost)
    robust_track_tightening = not bool(args.no_robust_track_tightening)
    robust_track_tightening_scale = float(args.robust_track_tightening_scale)
    robust_track_tightening_max = float(args.robust_track_tightening_max)
    robust_track_slack_penalty_scale = float(args.robust_track_slack_penalty_scale)
    robust_min_speed_floor = float(args.robust_min_speed_floor)
    robust_obstacle_speed_weight = float(args.robust_obstacle_speed_weight)
    robust_obstacle_speed_preview = float(args.robust_obstacle_speed_preview)
    robust_obstacle_accel_weight = float(args.robust_obstacle_accel_weight)
    suffix = args.name_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    robust_tag = f"Robust_K-DRMPC-Paper{suffix}"
    nonrobust_tag = f"NonRobust_K-DRMPC-Paper{suffix}"
    cleanup_previous_outputs([robust_tag, nonrobust_tag], suffix=suffix)

    print("=" * 60)
    print("Paper controller comparison: Robust vs Non-Robust K-DRMPC")
    print(f"Track: {TRACK_NAME} | {max_steps} steps | sigma: {sigma}")
    print(f"control_update_interval={control_update_interval}")
    if start_s is not None:
        print(f"start_s={float(start_s):.2f} m")
    print(
        f"halfspace_mode={halfspace_mode}, "
        f"hard_obstacles={hard_obstacle_constraints}, "
        f"safety_scale={obstacle_safety_scale:.3f}"
    )
    print(f"obstacle_activation_margin={obstacle_activation_margin:.3f}")
    print(f"prediction_horizon={prediction_horizon}")
    print(f"robust_use_worst_case_cost={use_worst_case_cost}; nonrobust_use_worst_case_cost=False")
    print(
        f"robust_track_tightening={robust_track_tightening}, "
        f"scale={robust_track_tightening_scale:.3f}, "
        f"max={robust_track_tightening_max:.3f}, "
        f"slack_penalty_scale={robust_track_slack_penalty_scale:.3f}, "
        f"min_speed_floor={robust_min_speed_floor:.3f}, "
        f"obstacle_speed_weight={robust_obstacle_speed_weight:.3f}, "
        f"obstacle_speed_preview={robust_obstacle_speed_preview:.3f}, "
        f"obstacle_accel_weight={robust_obstacle_accel_weight:.3f}"
    )
    if obstacle_slack_penalty is not None:
        print(f"obstacle_slack_penalty={float(obstacle_slack_penalty):.3f}")
    print(f"theta={theta}, epsilon={epsilon}, seed={seed}, suffix='{suffix}'")
    print("=" * 60)

    model, D, norm_params = load_koopman()

    print("\n\n" + "#" * 60)
    print("# Experiment 1: Paper robust mode")
    print("#" * 60)
    result_robust, track_r = run_experiment(
        model, D, norm_params,
        robust=True,
        tag=robust_tag,
        sigma=sigma,
        theta=theta,
        epsilon=epsilon,
        seed=seed,
        max_steps=max_steps,
        start_s=start_s,
        halfspace_mode=halfspace_mode,
        hard_obstacle_constraints=hard_obstacle_constraints,
        obstacle_slack_penalty=obstacle_slack_penalty,
        obstacle_safety_scale=obstacle_safety_scale,
        obstacle_activation_margin=obstacle_activation_margin,
        prediction_horizon=prediction_horizon,
        use_worst_case_cost=use_worst_case_cost,
        robust_track_tightening=robust_track_tightening,
        robust_track_tightening_scale=robust_track_tightening_scale,
        robust_track_tightening_max=robust_track_tightening_max,
        robust_track_slack_penalty_scale=robust_track_slack_penalty_scale,
        robust_min_speed_floor=robust_min_speed_floor,
        robust_obstacle_speed_weight=robust_obstacle_speed_weight,
        robust_obstacle_speed_preview=robust_obstacle_speed_preview,
        robust_obstacle_accel_weight=robust_obstacle_accel_weight,
        control_update_interval=control_update_interval,
    )

    print("\n\n" + "#" * 60)
    print("# Experiment 2: Paper non-robust mode")
    print("#" * 60)
    result_nonrobust, _ = run_experiment(
        model, D, norm_params,
        robust=False,
        tag=nonrobust_tag,
        sigma=sigma,
        theta=theta,
        epsilon=epsilon,
        seed=seed,
        max_steps=max_steps,
        start_s=start_s,
        halfspace_mode=halfspace_mode,
        hard_obstacle_constraints=hard_obstacle_constraints,
        obstacle_slack_penalty=obstacle_slack_penalty,
        obstacle_safety_scale=obstacle_safety_scale,
        obstacle_activation_margin=obstacle_activation_margin,
        prediction_horizon=prediction_horizon,
        use_worst_case_cost=use_worst_case_cost,
        robust_track_tightening=robust_track_tightening,
        robust_track_tightening_scale=robust_track_tightening_scale,
        robust_track_tightening_max=robust_track_tightening_max,
        robust_track_slack_penalty_scale=robust_track_slack_penalty_scale,
        robust_min_speed_floor=robust_min_speed_floor,
        robust_obstacle_speed_weight=robust_obstacle_speed_weight,
        robust_obstacle_speed_preview=robust_obstacle_speed_preview,
        robust_obstacle_accel_weight=robust_obstacle_accel_weight,
        control_update_interval=control_update_interval,
    )

    print("\n\n" + "█" * 60)
    print("█  Paper comparison summary")
    print("█" * 60)
    metrics_r = print_comparison("Robust K-DRMPC-Paper", result_robust, track_r)
    metrics_nr = print_comparison("Non-Robust K-DRMPC-Paper", result_nonrobust, track_r)

    print("\n" + "=" * 60)
    print("  Difference analysis (Robust - NonRobust)")
    print("=" * 60)
    for key in [
        "steps",
        "ey_mean",
        "ey_p95",
        "ey_max",
        "epsi_mean_deg",
        "v_mean",
        "solve_mean_ms",
        "within_2m_pct",
    ]:
        v_r = metrics_r[key]
        v_nr = metrics_nr[key]
        diff = v_r - v_nr
        label = key.replace("_", " ")
        print(f"  {label:20s}: {v_r:>10.3f} vs {v_nr:>10.3f}  (Δ = {diff:+.3f})")

    dual_gif_path = os.path.join(
        FIGURES_DIR,
        PAPER_DUAL_GIF_TEMPLATE.format(track=TRACK_NAME, suffix=suffix),
    )
    if args.skip_animation:
        dual_gif_path = None
        print("\nSkipped paper dual-line animation generation.")
    else:
        print("\n\n" + "#" * 60)
        print("# Generating paper dual-line comparison animation...")
        print("#" * 60)
        generate_dual_line_animation(result_robust, result_nonrobust, track_r, dual_gif_path, fps=20, max_frames=300)

    summary = {
        "robust": metrics_r,
        "nonrobust": metrics_nr,
        "sigma": sigma,
        "theta": theta,
        "epsilon": epsilon,
        "seed": seed,
        "max_steps": max_steps,
        "control_update_interval": control_update_interval,
        "start_s": start_s,
        "halfspace_mode": halfspace_mode,
        "hard_obstacle_constraints": hard_obstacle_constraints,
        "obstacle_slack_penalty": obstacle_slack_penalty,
        "obstacle_safety_scale": obstacle_safety_scale,
        "obstacle_activation_margin": obstacle_activation_margin,
        "prediction_horizon": prediction_horizon,
        "robust_use_worst_case_cost": use_worst_case_cost,
        "nonrobust_use_worst_case_cost": False,
        "robust_track_tightening": robust_track_tightening,
        "robust_track_tightening_scale": robust_track_tightening_scale,
        "robust_track_tightening_max": robust_track_tightening_max,
        "robust_track_slack_penalty_scale": robust_track_slack_penalty_scale,
        "robust_min_speed_floor": robust_min_speed_floor,
        "robust_obstacle_speed_weight": robust_obstacle_speed_weight,
        "robust_obstacle_speed_preview": robust_obstacle_speed_preview,
        "robust_obstacle_accel_weight": robust_obstacle_accel_weight,
        "suffix": suffix,
        "robust_tag": robust_tag,
        "nonrobust_tag": nonrobust_tag,
        "dual_gif": dual_gif_path,
        "controller": "KDRMPCPaperController",
    }
    suffix_name = suffix if suffix else ""
    summary_path = os.path.join(
        RESULTS_DIR,
        PAPER_SUMMARY_TEMPLATE.format(suffix=suffix_name),
    )
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved paper comparison summary: {summary_path}")


if __name__ == "__main__":
    main()