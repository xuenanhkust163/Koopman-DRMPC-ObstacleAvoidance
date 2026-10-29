"""Collect lifted-model residuals for the revised distributionally robust MPC."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from config import DT, L_WHEELBASE, MODEL_DIR, N_DISTURBANCE_SAMPLES, RESULTS_DIR
from controllers.kdrmpc_controller import KDRMPCController
from controllers.kdrmpc_paper_qp_controller import KDRMPCPaperQPController
from data.data_loader import normalize_state
from disturbance.disturbance_generator import DisturbanceGenerator
from run_noise_comparison import DEFAULT_EPSILON, DEFAULT_SEED, DEFAULT_SIGMA, DEFAULT_THETA
from run_noise_comparison_paper import (
    TRACK_BUILDERS,
    TRACK_NAME,
    build_track,
    initial_state_from_s,
    load_koopman,
    persist_result,
)
from simulation.simulator import Simulator


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lifted_residual_bank(result, model, norm_params, horizon, max_trajectories):
    """Build non-overlapping horizon windows from one closed-loop calibration run."""
    arrays = result.to_arrays()
    states = np.asarray(arrays["states"], dtype=float)
    controls = np.asarray(arrays["controls"], dtype=float)
    ref_states = np.asarray(result.ref_states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 5:
        raise ValueError(f"Expected states with shape (n, 5), got {states.shape}")
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError(f"Expected controls with shape (n, 2), got {controls.shape}")
    if len(states) != len(controls) + 1:
        raise ValueError("A calibration run must have one more state than control")
    if ref_states.shape != states[:-1].shape:
        raise ValueError("Calibration result must contain one reference state per transition")
    if len(controls) < horizon:
        raise ValueError(f"Need at least {horizon} transitions; got {len(controls)}")

    normalized_states = normalize_state(states, norm_params)
    with torch.no_grad():
        state_tensor = torch.as_tensor(normalized_states, dtype=torch.float32)
        lifted_states = model.encode(state_tensor).cpu().numpy()

    A, B, _ = model.get_matrices()
    A = KDRMPCController._clip_matrix_eigenvalues(A, radius=0.9)
    nominal_next = lifted_states[:-1] @ A.T + controls @ B.T

    psi_ref = ref_states[:, 2]
    v_ref = np.maximum(ref_states[:, 3], 0.1)
    psi_dev = lifted_states[:-1, 2] - psi_ref
    px_step = (
        np.cos(psi_ref) * lifted_states[:-1, 3]
        - v_ref * np.sin(psi_ref) * psi_dev
    )
    py_step = (
        np.sin(psi_ref) * lifted_states[:-1, 3]
        + v_ref * np.cos(psi_ref) * psi_dev
    )
    nominal_next[:, 0] = lifted_states[:-1, 0] + DT * px_step / float(norm_params["px_std"])
    nominal_next[:, 1] = lifted_states[:-1, 1] + DT * py_step / float(norm_params["py_std"])
    nominal_next[:, 2] = lifted_states[:-1, 2] + DT * (v_ref / L_WHEELBASE) * controls[:, 1]
    nominal_next[:, 3] = lifted_states[:-1, 3] + DT * controls[:, 0]
    nominal_next[:, 4] = (v_ref / L_WHEELBASE) * controls[:, 1]
    residuals = lifted_states[1:] - nominal_next
    if not np.isfinite(residuals).all():
        raise ValueError("Lifted residuals contain non-finite values")

    starts = np.arange(0, len(residuals) - horizon + 1, horizon, dtype=int)
    if max_trajectories > 0:
        starts = starts[:max_trajectories]
    if len(starts) == 0:
        raise ValueError("The calibration run did not contain a complete residual window")

    trajectories = np.stack([residuals[start:start + horizon] for start in starts])
    residual_scale = np.std(residuals, axis=0, ddof=1) if len(residuals) > 1 else np.ones(residuals.shape[1])
    residual_scale = np.maximum(residual_scale, 1e-8)
    return trajectories, residual_scale, residuals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a non-robust calibration rollout and save lifted residual trajectories"
    )
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-trajectories", type=int, default=N_DISTURBANCE_SAMPLES)
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--qp-solver", choices=("qpoases", "osqp", "qrqp"), default="qpoases")
    parser.add_argument("--track-name", choices=sorted(TRACK_BUILDERS), default=TRACK_NAME)
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--tag", default="residual_calibration")
    return parser.parse_args()


def main():
    args = parse_args()
    model, D, norm_params = load_koopman()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    trajectory_banks = []
    residual_sets = []
    run_metadata = []

    for calibration_seed in seeds:
        track = build_track(args.track_name)
        disturbance = DisturbanceGenerator(sigma=args.sigma, seed=calibration_seed)
        empirical_samples = disturbance.get_empirical_samples(N_DISTURBANCE_SAMPLES)
        controller = KDRMPCPaperQPController(
            model,
            D,
            norm_params,
            disturbance_samples=empirical_samples,
            theta=args.theta,
            epsilon=args.epsilon,
            obstacle_strategy="non-robust",
            track=track,
            prediction_horizon=args.horizon,
            use_worst_case_cost=False,
            robust_track_tightening=False,
            linear_physical_consistency=True,
            qp_solver=args.qp_solver,
        )
        simulator = Simulator(track, controller, disturbance)
        x0 = initial_state_from_s(track, args.start_s)
        result = simulator.run(
            x0=x0,
            max_steps=args.steps,
            lap_fraction=1.0,
            verbose=True,
            detailed_step_log=False,
        )

        failed_steps = [
            step for step, status in enumerate(result.solve_statuses)
            if str(status).startswith("failed")
        ]
        valid_transitions = failed_steps[0] if failed_steps else len(result.controls)
        if valid_transitions < args.horizon:
            run_metadata.append({
                "seed": int(calibration_seed),
                "valid_transitions": int(valid_transitions),
                "failed_solve_count": int(len(failed_steps)),
                "used": False,
            })
            print(
                f"Skipping seed {calibration_seed}: only {valid_transitions} "
                "valid transitions before the first failed solve"
            )
            continue

        result.states = result.states[:valid_transitions + 1]
        result.controls = result.controls[:valid_transitions]
        result.solve_times = result.solve_times[:valid_transitions]
        result.solve_statuses = result.solve_statuses[:valid_transitions]
        result.solve_debug = result.solve_debug[:valid_transitions]
        result.ref_states = result.ref_states[:valid_transitions]
        result.timestamps = result.timestamps[:valid_transitions]
        result.total_steps = valid_transitions

        trajectories, _, residuals = build_lifted_residual_bank(
            result, model, norm_params, args.horizon, args.max_trajectories
        )
        trajectory_banks.append(trajectories)
        residual_sets.append(residuals)
        run_metadata.append({
            "seed": int(calibration_seed),
            "valid_transitions": int(valid_transitions),
            "failed_solve_count": int(len(failed_steps)),
            "used": True,
            "trajectory_count": int(len(trajectories)),
        })
        persist_result(
            result,
            f"{args.tag}_seed{calibration_seed}_sigma{args.sigma:g}.pkl",
        )

    if not trajectory_banks:
        raise RuntimeError("No calibration seed produced a complete valid residual window")

    trajectories = np.concatenate(trajectory_banks, axis=0)[:args.max_trajectories]
    residuals = np.concatenate(residual_sets, axis=0)
    residual_scale = np.std(residuals, axis=0, ddof=1) if len(residuals) > 1 else np.ones(residuals.shape[1])
    residual_scale = np.maximum(residual_scale, 1e-8)
    checkpoint_path = Path(MODEL_DIR) / "best_koopman_model.pth"
    metadata = {
        "method": "non-robust paper QP rollout; residuals relative to its affine nominal predictor; truncate at first failed solve",
        "track_name": args.track_name,
        "seeds": [int(seed) for seed in seeds],
        "runs": run_metadata,
        "sigma": args.sigma,
        "dt": float(_get_dt()),
        "horizon": args.horizon,
        "trajectory_count": int(len(trajectories)),
        "residual_dimension": int(residuals.shape[1]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "trajectory_sampling": "non-overlapping consecutive residual blocks within each seed",
        "residual_definition": "encoded next state minus A/B prediction with the controller's first-five-state affine consistency update",
        "residual_scale_definition": "per-coordinate standard deviation over the calibration rollout",
        "wasserstein_ground_norm": "Euclidean norm after per-coordinate residual standardization",
    }

    output_dir = Path(RESULTS_DIR) / "residual_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = "_".join(str(int(seed)) for seed in seeds)
    output_path = output_dir / f"{args.tag}_seeds{seed_tag}_sigma{args.sigma:g}.npz"
    np.savez_compressed(
        output_path,
        residual_trajectories=trajectories,
        residual_scale=residual_scale,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved residual bank: {output_path}")
    print(f"Residual bank shape: {trajectories.shape}; scale shape: {residual_scale.shape}")
    print(f"Checkpoint SHA-256: {metadata['checkpoint_sha256']}")


def _get_dt():
    from config import DT

    return DT


if __name__ == "__main__":
    main()
