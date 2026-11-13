"""Minimal standalone smoke runner for the paper-faithful K-DRMPC controller."""

import argparse
import json
import os
import pickle
import sys

from _bootstrap import bootstrap

bootstrap()

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Keep behavior aligned with the main comparison script.
import config as _cfg
_cfg.ENABLE_DISTURBANCE = True
_cfg.ENABLE_OBSTACLES = True

from config import (  # noqa: E402
    MODEL_DIR,
    RESULTS_DIR,
    N_DISTURBANCE_SAMPLES,
    THETA_WASSERSTEIN,
    EPSILON_CVAR,
    NOMINAL_SIGMA,
)
from controllers.kdrmpc_paper_controller import KDRMPCPaperController  # noqa: E402
from disturbance.disturbance_generator import DisturbanceGenerator  # noqa: E402
from model.koopman_trainer import load_trained_model  # noqa: E402
from model.projection import load_projection_matrix  # noqa: E402
from simulation.simulator import Simulator  # noqa: E402
from tracks.serpentine_track import SerpentineTrack  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone smoke test for the paper-faithful K-DRMPC controller"
    )
    parser.add_argument("--steps", type=int, default=80, help="max simulation steps")
    parser.add_argument("--sigma", type=float, default=NOMINAL_SIGMA, help="disturbance sigma")
    parser.add_argument("--theta", type=float, default=THETA_WASSERSTEIN, help="Wasserstein radius")
    parser.add_argument("--epsilon", type=float, default=EPSILON_CVAR, help="CVaR epsilon")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--obstacle-strategy",
        choices=["robust", "non-robust"],
        default="robust",
        help="paper controller obstacle mode",
    )
    parser.add_argument(
        "--lap-fraction",
        type=float,
        default=0.10,
        help="fraction of one lap to target in the smoke run",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="paper_smoke",
        help="prefix for saved result files",
    )
    return parser.parse_args()


def load_koopman_assets():
    model = load_trained_model()
    projection = load_projection_matrix()
    norm_path = os.path.join(MODEL_DIR, "norm_params.json")
    with open(norm_path, "r") as handle:
        norm_params = json.load(handle)
    return model, projection, norm_params


def summarize_result(result):
    states = np.array(result.states)
    solve_times = np.array(result.solve_times) if result.solve_times else np.array([0.0])
    last_state = states[-1] if len(states) else np.zeros(5)
    return {
        "method": result.method_name,
        "steps": int(result.total_steps),
        "crashed": bool(result.crashed),
        "crash_step": result.crash_step,
        "lap_completed": bool(result.lap_completed),
        "lap_time": result.lap_time,
        "final_state": [float(v) for v in last_state.tolist()],
        "solve_mean_ms": float(np.mean(solve_times) * 1000.0),
        "solve_p95_ms": float(np.percentile(solve_times, 95) * 1000.0),
        "solve_status_last": result.solve_statuses[-1] if result.solve_statuses else None,
    }


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, projection, norm_params = load_koopman_assets()
    track = SerpentineTrack()
    disturbance = DisturbanceGenerator(sigma=float(args.sigma), seed=int(args.seed))
    empirical_samples = disturbance.get_empirical_samples(N_DISTURBANCE_SAMPLES)

    controller = KDRMPCPaperController(
        model,
        projection,
        norm_params,
        disturbance_samples=empirical_samples,
        theta=float(args.theta),
        epsilon=float(args.epsilon),
        obstacle_strategy=args.obstacle_strategy,
    )

    simulator = Simulator(track, controller, disturbance)
    result = simulator.run(
        max_steps=int(args.steps),
        lap_fraction=float(args.lap_fraction),
        verbose=True,
        detailed_step_log=True,
        detailed_step_log_max_steps=5,
    )

    suffix = f"{args.output_tag}_{args.obstacle_strategy}_seed{args.seed}"
    artifact_name = f"{suffix}_{track.__class__.__name__}.pkl"
    artifact_path = os.path.join(RESULTS_DIR, artifact_name)
    with open(artifact_path, "wb") as handle:
        pickle.dump(result, handle)

    summary = summarize_result(result)
    summary_path = os.path.join(RESULTS_DIR, f"{suffix}_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Paper controller smoke run complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved result artifact: {artifact_path}")
    print(f"Saved summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()