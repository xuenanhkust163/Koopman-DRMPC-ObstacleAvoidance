"""Sweep QP paper-controller robustness over increasing disturbance levels.

The script wraps ``run_noise_comparison_paper.py`` so each sweep point uses the
same robust-vs-nonrobust artifact pipeline as the main comparison run. It then
adds lightweight safety-margin metrics from the saved pickle artifacts and
writes one compact JSON/CSV summary for finding a displayable noise threshold.
"""

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR  # noqa: E402
from controllers import kdrmpc_controller as base  # noqa: E402
from run_noise_comparison import TRACK_NAME  # noqa: E402
from run_noise_comparison_paper import PAPER_SUMMARY_TEMPLATE, TRACK_BUILDERS, build_track  # noqa: E402


DEFAULT_SIGMAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
DEFAULT_SEEDS = [1, 2, 3]


def _tag_value(value):
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _result_path(tag, track_name):
    return os.path.join(RESULTS_DIR, f"{tag}_{track_name}.pkl")


def _safety_metrics(result, track):
    states = np.asarray(result.states, dtype=float)
    if states.size == 0:
        return {
            "track_margin_min": None,
            "track_margin_p05": None,
            "obstacle_margin_min": None,
            "obstacle_margin_p05": None,
        }

    obstacles = track.get_obstacles()
    track_margins = []
    obstacle_margins = []

    for x in states:
        px = float(x[0])
        py = float(x[1])
        _, _, lat = track.closest_point(px, py)
        track_margins.append(float(base.TRACK_HALF_WIDTH - abs(float(lat)) - base.VEHICLE_RADIUS))

        if obstacles:
            margins = []
            for ox, oy, radius in obstacles:
                center_dist = float(np.hypot(px - float(ox), py - float(oy)))
                margins.append(center_dist - (float(radius) + base.VEHICLE_RADIUS + base.D_SAFE))
            obstacle_margins.append(min(margins))

    out = {
        "track_margin_min": float(np.min(track_margins)),
        "track_margin_p05": float(np.percentile(track_margins, 5)),
    }
    if obstacle_margins:
        out.update({
            "obstacle_margin_min": float(np.min(obstacle_margins)),
            "obstacle_margin_p05": float(np.percentile(obstacle_margins, 5)),
        })
    else:
        out.update({"obstacle_margin_min": None, "obstacle_margin_p05": None})
    return out


def _classify_case(row):
    robust_ok = not row["robust_crashed"]
    nonrobust_ok = not row["nonrobust_crashed"]
    if robust_ok and not nonrobust_ok:
        return "candidate_robust_advantage"
    if robust_ok and nonrobust_ok:
        return "both_survive"
    if not robust_ok and not nonrobust_ok:
        return "both_fail"
    return "nonrobust_survives_robust_fails"


def run_one(args, sigma, seed):
    suffix = f"{args.prefix}_sigma{_tag_value(sigma)}_seed{seed}"
    cmd = [
        sys.executable,
        "run_noise_comparison_paper.py",
        "--controller-formulation", "qp",
        "--qp-solver", args.qp_solver,
        "--sigma", str(float(sigma)),
        "--theta", str(float(args.theta)),
        "--epsilon", str(float(args.epsilon)),
        "--seed", str(int(seed)),
        "--prediction-horizon", str(int(args.prediction_horizon)),
        "--steps", str(int(args.steps)),
        "--track-name", args.track_name,
        "--name-suffix", suffix,
    ]
    if not args.skip_animation:
        cmd.append("--render-gif")

    print("\n" + "=" * 72)
    print(f"Running QP noise sweep point: sigma={sigma:g}, seed={seed}")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), check=True)

    suffix_name = suffix if suffix.startswith("_") else f"_{suffix}"
    summary_path = os.path.join(RESULTS_DIR, PAPER_SUMMARY_TEMPLATE.format(suffix=suffix_name))
    with open(summary_path, "r") as handle:
        summary = json.load(handle)

    robust_tag = summary["robust_tag"]
    nonrobust_tag = summary["nonrobust_tag"]
    track_name = summary.get("track_name", args.track_name)
    robust_result = _load_pickle(_result_path(robust_tag, track_name))
    nonrobust_result = _load_pickle(_result_path(nonrobust_tag, track_name))
    track = build_track(track_name)
    robust_safety = _safety_metrics(robust_result, track)
    nonrobust_safety = _safety_metrics(nonrobust_result, track)

    row = {
        "sigma": float(sigma),
        "seed": int(seed),
        "track_name": track_name,
        "robust_crashed": bool(summary["robust"]["crashed"]),
        "robust_crash_step": summary["robust"]["crash_step"],
        "robust_steps": int(summary["robust"]["steps"]),
        "robust_v_mean": float(summary["robust"]["v_mean"]),
        "robust_ey_max": float(summary["robust"]["ey_max"]),
        "robust_track_margin_min": robust_safety["track_margin_min"],
        "robust_track_margin_p05": robust_safety["track_margin_p05"],
        "robust_obstacle_margin_min": robust_safety["obstacle_margin_min"],
        "robust_obstacle_margin_p05": robust_safety["obstacle_margin_p05"],
        "nonrobust_crashed": bool(summary["nonrobust"]["crashed"]),
        "nonrobust_crash_step": summary["nonrobust"]["crash_step"],
        "nonrobust_steps": int(summary["nonrobust"]["steps"]),
        "nonrobust_v_mean": float(summary["nonrobust"]["v_mean"]),
        "nonrobust_ey_max": float(summary["nonrobust"]["ey_max"]),
        "nonrobust_track_margin_min": nonrobust_safety["track_margin_min"],
        "nonrobust_track_margin_p05": nonrobust_safety["track_margin_p05"],
        "nonrobust_obstacle_margin_min": nonrobust_safety["obstacle_margin_min"],
        "nonrobust_obstacle_margin_p05": nonrobust_safety["obstacle_margin_p05"],
        "summary_path": summary_path,
        "dual_png": summary.get("dual_png"),
        "dual_gif": summary.get("dual_gif"),
    }
    row["case"] = _classify_case(row)
    print(
        f"Result: {row['case']} | robust steps={row['robust_steps']} crashed={row['robust_crashed']} | "
        f"nonrobust steps={row['nonrobust_steps']} crashed={row['nonrobust_crashed']}"
    )
    return row


def write_outputs(rows, output_prefix):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{output_prefix}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{output_prefix}.csv")
    candidates = [row for row in rows if row["case"] == "candidate_robust_advantage"]

    payload = {
        "rows": rows,
        "candidate_robust_advantage": candidates,
    }
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if rows:
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "=" * 72)
    print(f"Saved sweep JSON: {json_path}")
    print(f"Saved sweep CSV:  {csv_path}")
    if candidates:
        print("Candidate robust-advantage cases:")
        for row in candidates:
            print(
                f"  sigma={row['sigma']:g}, seed={row['seed']}, "
                f"nonrobust crash step={row['nonrobust_crash_step']}, robust steps={row['robust_steps']}"
            )
    else:
        print("No robust-advantage candidate found in this grid.")


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep QP robust/nonrobust comparison over noise levels.")
    parser.add_argument("--sigmas", nargs="+", type=float, default=DEFAULT_SIGMAS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--prediction-horizon", type=int, default=20)
    parser.add_argument("--qp-solver", choices=["qpoases", "osqp", "qrqp"], default="qpoases")
    parser.add_argument("--track-name", choices=sorted(TRACK_BUILDERS), default=TRACK_NAME)
    parser.add_argument("--prefix", default="_qp_sweep")
    parser.add_argument("--output-prefix", default="qp_noise_sweep_summary")
    parser.add_argument("--skip-animation", action="store_true", default=True)
    parser.add_argument("--with-animation", action="store_false", dest="skip_animation")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for sigma in args.sigmas:
        for seed in args.seeds:
            rows.append(run_one(args, sigma, seed))
            write_outputs(rows, args.output_prefix)


if __name__ == "__main__":
    main()