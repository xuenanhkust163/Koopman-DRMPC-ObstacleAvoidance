"""Noise comparison experiment for the paper-faithful K-DRMPC controller.

This script mirrors the practical comparison entry point, but uses the
parallel paper controller implementation so the main experiment path remains
untouched.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys

from _bootstrap import bootstrap

bootstrap()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as _cfg

_cfg.ENABLE_DISTURBANCE = True
_cfg.ENABLE_OBSTACLES = True

from config import CONTROL_UPDATE_INTERVAL, FIGURES_DIR, MODEL_DIR, RESULTS_DIR  # noqa: E402
from controllers.kdrmpc_paper_controller import KDRMPCPaperController  # noqa: E402
from controllers.kdrmpc_shifted_hinge_qp_experimental import KDRMPCPaperQPController  # noqa: E402
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
PAPER_DUAL_PNG_TEMPLATE = "Comparison_{track}_paper_dual_trajectory{suffix}.png"
PAPER_SUMMARY_TEMPLATE = "noise_comparison_paper_summary{suffix}.json"
TRACK_BUILDERS = {
    "SerpentineTrack": SerpentineTrack,
}


def build_track(track_name):
    try:
        return TRACK_BUILDERS[track_name]()
    except KeyError as exc:
        supported = ", ".join(sorted(TRACK_BUILDERS))
        raise ValueError(f"Unsupported track name: {track_name}. Supported: {supported}") from exc


def persist_result(result, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "wb") as handle:
        pickle.dump(result, handle)
    print(f"Saved result artifact: {path}")
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_residual_bank(path, horizon):
    """Load a residual bank and reject stale or dimensionally mismatched files."""
    with np.load(path, allow_pickle=False) as bank:
        trajectories = np.asarray(bank["residual_trajectories"], dtype=float)
        scale = np.asarray(bank["residual_scale"], dtype=float).reshape(-1)
        metadata = json.loads(str(bank["metadata_json"].item()))

    checkpoint_path = os.path.join(MODEL_DIR, "best_koopman_model.pth")
    expected_hash = _sha256_file(checkpoint_path)
    if metadata.get("checkpoint_sha256") != expected_hash:
        raise ValueError("Residual bank checkpoint hash does not match the loaded Koopman model")
    if trajectories.ndim != 3 or trajectories.shape[1] != int(horizon):
        raise ValueError(
            f"Residual bank must have shape (N, {horizon}, n_z), got {trajectories.shape}"
        )
    if trajectories.shape[0] == 0 or not np.isfinite(trajectories).all():
        raise ValueError("Residual bank must contain finite trajectories")
    if scale.shape != (trajectories.shape[2],) or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Residual bank scale must be positive, finite, and match n_z")
    return trajectories, scale, metadata


def generate_dual_trajectory_png(result_robust, result_nonrobust, track, save_path):
    """Render a single static red/blue full-trajectory comparison PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    robust_states = result_robust.to_arrays()["states"]
    nonrobust_states = result_nonrobust.to_arrays()["states"]

    cx, cy = track.get_centerline()
    heading = track.get_heading()
    nx = -np.sin(heading)
    ny = np.cos(heading)
    half_width = float(_cfg.PLOT_TRACK_HALF_WIDTH)
    left_x = cx + half_width * nx
    left_y = cy + half_width * ny
    right_x = cx - half_width * nx
    right_y = cy - half_width * ny

    fig, ax = plt.subplots(1, 1, figsize=(13, 8), constrained_layout=True)
    ax.plot(left_x, left_y, color="#4a4a4a", linewidth=1.0, alpha=0.9)
    ax.plot(right_x, right_y, color="#4a4a4a", linewidth=1.0, alpha=0.9)
    ax.plot(cx, cy, linestyle="--", color="#9a9a9a", linewidth=1.0, alpha=0.7)

    for ox, oy, radius in track.get_obstacles():
        obstacle = Circle((ox, oy), radius, facecolor="#7a7a7a", edgecolor="#555555", alpha=0.25, linewidth=0.8)
        ax.add_patch(obstacle)

    ax.plot(nonrobust_states[:, 0], nonrobust_states[:, 1], color="#d62728", linewidth=0.85, alpha=0.95, label="Non-robust")
    ax.plot(robust_states[:, 0], robust_states[:, 1], color="#1f77b4", linewidth=0.85, alpha=0.95, label="Robust")

    ax.plot(nonrobust_states[0, 0], nonrobust_states[0, 1], marker="o", color="#d62728", markersize=5)
    ax.plot(robust_states[0, 0], robust_states[0, 1], marker="o", color="#1f77b4", markersize=5)

    if getattr(result_nonrobust, "crashed", False):
        ax.plot(nonrobust_states[-1, 0], nonrobust_states[-1, 1], marker="x", color="#d62728", markersize=9, mew=2.0)
    if getattr(result_robust, "crashed", False):
        ax.plot(robust_states[-1, 0], robust_states[-1, 1], marker="x", color="#1f77b4", markersize=9, mew=2.0)

    ax.set_title(f"Paper Trajectory Comparison: {track.__class__.__name__}")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="best")

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paper trajectory PNG: {save_path}")


def cleanup_previous_outputs(tags, suffix="", track_name=TRACK_NAME):
    removed = []
    for tag in tags:
        base = f"{tag}_{track_name}"
        path = os.path.join(RESULTS_DIR, f"{base}.pkl")
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    comparison_gif = os.path.join(
        FIGURES_DIR,
        PAPER_DUAL_GIF_TEMPLATE.format(track=track_name, suffix=suffix),
    )
    if os.path.exists(comparison_gif):
        os.remove(comparison_gif)
        removed.append(comparison_gif)

    comparison_png = os.path.join(
        FIGURES_DIR,
        PAPER_DUAL_PNG_TEMPLATE.format(track=track_name, suffix=suffix),
    )
    if os.path.exists(comparison_png):
        os.remove(comparison_png)
        removed.append(comparison_png)

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
    obstacle_lateral_bias_weight,
    obstacle_speed_cap_base,
    obstacle_speed_cap_gain,
    steering_blend,
    adaptive_steering_relief,
    steering_relief_lateral_threshold,
    steering_relief_obstacle_distance,
    control_update_interval,
    controller_formulation,
    qp_solver,
    track_name,
    residual_trajectories=None,
    residual_scale=None,
    safety_margin_delta=None,
):
    track = build_track(track_name)
    dist_gen = DisturbanceGenerator(sigma=sigma, seed=seed)
    w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

    obstacle_strategy = "robust" if robust else "non-robust"
    controller_cls = (
        KDRMPCPaperQPController
        if controller_formulation == "qp"
        else KDRMPCPaperController
    )
    controller_kwargs = {}
    if controller_formulation == "qp":
        controller_kwargs["qp_solver"] = qp_solver
        controller_kwargs["linear_physical_consistency"] = True
        if robust:
            if residual_trajectories is None or residual_scale is None:
                raise ValueError("robust QP requires a calibrated full-horizon residual bank")
            if safety_margin_delta is None or float(safety_margin_delta) <= 0.0:
                raise ValueError("robust QP requires a positive safety-margin delta")
            controller_kwargs["disturbance_trajectories"] = residual_trajectories
            controller_kwargs["residual_scale"] = residual_scale
            controller_kwargs["safety_margin_delta"] = float(safety_margin_delta)

    controller = controller_cls(
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
        obstacle_lateral_bias_weight=obstacle_lateral_bias_weight,
        obstacle_speed_cap_base=obstacle_speed_cap_base,
        obstacle_speed_cap_gain=obstacle_speed_cap_gain,
        **controller_kwargs,
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
        steering_blend=steering_blend,
        adaptive_steering_relief=adaptive_steering_relief,
        steering_relief_lateral_threshold=steering_relief_lateral_threshold,
        steering_relief_obstacle_distance=steering_relief_obstacle_distance,
        control_update_interval=control_update_interval,
    )

    persist_result(result, f"{tag}_{track_name}.pkl")
    return result, track


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper controller comparison: robust vs non-robust K-DRMPC"
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="measurement noise sigma")
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA, help="Wasserstein theta")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="distributionally robust chance-constraint risk level")
    parser.add_argument("--residual-bank", type=str, default=None, help="calibrated full-horizon lifted residual bank (.npz)")
    parser.add_argument("--safety-margin-delta", type=float, default=None, help="positive shifted-hinge margin in meters")
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
        "--track-name",
        choices=sorted(TRACK_BUILDERS),
        default=TRACK_NAME,
        help="track geometry used for the robust/nonrobust comparison",
    )
    parser.add_argument(
        "--controller-formulation",
        choices=["qp", "nlp"],
        default="qp",
        help="paper controller formulation: convex QP or legacy IPOPT NLP",
    )
    parser.add_argument(
        "--qp-solver",
        choices=["osqp", "qpoases", "qrqp"],
        default="qpoases",
        help="CasADi qpsol plugin used when --controller-formulation=qp",
    )
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
        "--obstacle-lateral-bias-weight",
        type=float,
        default=18.0,
        help="QP engineering weight for obstacle-driven lateral target bias",
    )
    parser.add_argument(
        "--obstacle-speed-cap-base",
        type=float,
        default=2.0,
        help="base obstacle-window speed cap in m/s",
    )
    parser.add_argument(
        "--obstacle-speed-cap-gain",
        type=float,
        default=0.10,
        help="linear gain on forward-gap for obstacle-window speed cap",
    )
    parser.add_argument(
        "--steering-blend",
        type=float,
        default=1.0,
        help="blend factor for newly solved steering; lower values produce smoother paths",
    )
    parser.add_argument(
        "--adaptive-steering-relief",
        action="store_true",
        help="increase steering responsiveness near risk while keeping smoothing in safer regions",
    )
    parser.add_argument(
        "--steering-relief-lateral-threshold",
        type=float,
        default=6.0,
        help="lateral error threshold [m] where adaptive steering relief reaches full strength",
    )
    parser.add_argument(
        "--steering-relief-obstacle-distance",
        type=float,
        default=10.0,
        help="obstacle distance [m] where adaptive steering relief fades out to baseline blend",
    )
    parser.add_argument(
        "--render-gif",
        action="store_true",
        help="render the dual-line GIF comparison animation",
    )
    parser.add_argument(
        "--name-suffix",
        type=str,
        default="shifted_hinge_v1",
        help="output filename suffix for the revised DRCC experiment",
    )
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
    obstacle_lateral_bias_weight = float(args.obstacle_lateral_bias_weight)
    obstacle_speed_cap_base = float(args.obstacle_speed_cap_base)
    obstacle_speed_cap_gain = float(args.obstacle_speed_cap_gain)
    steering_blend = float(args.steering_blend)
    adaptive_steering_relief = bool(args.adaptive_steering_relief)
    steering_relief_lateral_threshold = float(args.steering_relief_lateral_threshold)
    steering_relief_obstacle_distance = float(args.steering_relief_obstacle_distance)
    controller_formulation = args.controller_formulation
    qp_solver = args.qp_solver
    track_name = args.track_name
    suffix = args.name_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"

    controller_tag = "Paper-QP" if controller_formulation == "qp" else "Paper"
    robust_tag = f"Robust_K-DRMPC-{controller_tag}{suffix}"
    nonrobust_tag = f"NonRobust_K-DRMPC-{controller_tag}{suffix}"

    print("=" * 60)
    print("Paper controller comparison: Robust vs Non-Robust K-DRMPC")
    print(f"Track: {track_name} | {max_steps} steps | sigma: {sigma}")
    print(f"controller_formulation={controller_formulation}, qp_solver={qp_solver}")
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
        f"obstacle_accel_weight={robust_obstacle_accel_weight:.3f}, "
        f"lateral_bias_weight={obstacle_lateral_bias_weight:.3f}, "
        f"speed_cap_base={obstacle_speed_cap_base:.3f}, "
        f"speed_cap_gain={obstacle_speed_cap_gain:.3f}, "
        f"steering_blend={steering_blend:.3f}, "
        f"adaptive_steering_relief={adaptive_steering_relief}, "
        f"relief_lateral_threshold={steering_relief_lateral_threshold:.3f}, "
        f"relief_obstacle_distance={steering_relief_obstacle_distance:.3f}"
    )
    if obstacle_slack_penalty is not None:
        print(f"obstacle_slack_penalty={float(obstacle_slack_penalty):.3f}")
    print(f"theta={theta}, epsilon={epsilon}, seed={seed}, suffix='{suffix}'")
    print("=" * 60)

    model, D, norm_params = load_koopman()
    if controller_formulation != "qp":
        raise ValueError("The revised shifted-hinge chance constraint is implemented only by the QP controller")
    if not hard_obstacle_constraints:
        raise ValueError("The revised DRCC guarantee requires hard obstacle constraints")
    if args.residual_bank is None:
        raise ValueError("Pass --residual-bank from run_residual_calibration.py")
    if args.safety_margin_delta is None or args.safety_margin_delta <= 0.0:
        raise ValueError("Pass a positive --safety-margin-delta in meters")

    residual_trajectories, residual_scale, residual_metadata = load_residual_bank(
        args.residual_bank,
        prediction_horizon,
    )
    cleanup_previous_outputs([robust_tag, nonrobust_tag], suffix=suffix, track_name=track_name)

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
        obstacle_lateral_bias_weight=obstacle_lateral_bias_weight,
        obstacle_speed_cap_base=obstacle_speed_cap_base,
        obstacle_speed_cap_gain=obstacle_speed_cap_gain,
        steering_blend=steering_blend,
        adaptive_steering_relief=adaptive_steering_relief,
        steering_relief_lateral_threshold=steering_relief_lateral_threshold,
        steering_relief_obstacle_distance=steering_relief_obstacle_distance,
        control_update_interval=control_update_interval,
        controller_formulation=controller_formulation,
        qp_solver=qp_solver,
        track_name=track_name,
        residual_trajectories=residual_trajectories,
        residual_scale=residual_scale,
        safety_margin_delta=args.safety_margin_delta,
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
        obstacle_lateral_bias_weight=obstacle_lateral_bias_weight,
        obstacle_speed_cap_base=obstacle_speed_cap_base,
        obstacle_speed_cap_gain=obstacle_speed_cap_gain,
        steering_blend=steering_blend,
        adaptive_steering_relief=adaptive_steering_relief,
        steering_relief_lateral_threshold=steering_relief_lateral_threshold,
        steering_relief_obstacle_distance=steering_relief_obstacle_distance,
        control_update_interval=control_update_interval,
        controller_formulation=controller_formulation,
        qp_solver=qp_solver,
        track_name=track_name,
    )

    print("\n\n" + "█" * 60)
    print("█  Paper comparison summary")
    print("█" * 60)
    metrics_r = print_comparison(f"Robust K-DRMPC-{controller_tag}", result_robust, track_r)
    metrics_nr = print_comparison(f"Non-Robust K-DRMPC-{controller_tag}", result_nonrobust, track_r)

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
        PAPER_DUAL_GIF_TEMPLATE.format(track=track_name, suffix=suffix),
    )
    dual_png_path = os.path.join(
        FIGURES_DIR,
        PAPER_DUAL_PNG_TEMPLATE.format(track=track_name, suffix=suffix),
    )
    if args.render_gif:
        print("\n\n" + "#" * 60)
        print("# Generating paper dual-line comparison animation...")
        print("#" * 60)
        generate_dual_line_animation(result_robust, result_nonrobust, track_r, dual_gif_path, fps=20, max_frames=300)
    else:
        dual_gif_path = None
        print("\n\n" + "#" * 60)
        print("# Skipping paper dual-line animation (PNG-only mode)...")
        print("#" * 60)

    print("\n\n" + "#" * 60)
    print("# Generating paper full-trajectory PNG...")
    print("#" * 60)
    generate_dual_trajectory_png(result_robust, result_nonrobust, track_r, dual_png_path)

    summary = {
        "robust": metrics_r,
        "nonrobust": metrics_nr,
        "sigma": sigma,
        "theta": theta,
        "epsilon": epsilon,
        "safety_margin_delta": args.safety_margin_delta,
        "residual_bank": os.path.abspath(args.residual_bank),
        "residual_bank_trajectory_count": int(len(residual_trajectories)),
        "drcc_optimization_sample_count": int(min(len(residual_trajectories), 10)),
        "drcc_optimization_sample_seed": 42,
        "wasserstein_ground_norm": "Euclidean norm after per-coordinate residual standardization",
        "residual_bank_checkpoint_sha256": residual_metadata["checkpoint_sha256"],
        "seed": seed,
        "max_steps": max_steps,
        "track_name": track_name,
        "controller_formulation": controller_formulation,
        "qp_solver": qp_solver if controller_formulation == "qp" else None,
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
        "obstacle_lateral_bias_weight": obstacle_lateral_bias_weight,
        "obstacle_speed_cap_base": obstacle_speed_cap_base,
        "obstacle_speed_cap_gain": obstacle_speed_cap_gain,
        "steering_blend": steering_blend,
        "adaptive_steering_relief": adaptive_steering_relief,
        "steering_relief_lateral_threshold": steering_relief_lateral_threshold,
        "steering_relief_obstacle_distance": steering_relief_obstacle_distance,
        "suffix": suffix,
        "robust_tag": robust_tag,
        "nonrobust_tag": nonrobust_tag,
        "dual_gif": dual_gif_path,
        "dual_png": dual_png_path,
        "controller": "KDRMPCPaperQPController" if controller_formulation == "qp" else "KDRMPCPaperController",
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