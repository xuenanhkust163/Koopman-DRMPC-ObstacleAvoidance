"""Step-level diagnostics for robust vs non-robust noise comparison runs.

This script reads saved result artifacts (.pkl), exports step-level CSV diagnostics,
and generates intuitive figures for troubleshooting:
1) Full-run overview (progress, speed, lateral error)
2) Focused windows (robust stuck zone, non-robust pre-crash zone)
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import TRACK_HALF_WIDTH
from tracks.serpentine_track import SerpentineTrack


@dataclass
class StepRecord:
    controller: str
    step: int
    s: float
    lat: float
    abs_lat: float
    v: float
    omega: float
    x: float
    y: float
    psi: float
    solve_time_ms: Optional[float]
    solve_status: str
    solve_failed: int
    active_obstacle_pairs: Optional[int]
    objective: Optional[float]


def _safe_get(mapping: Any, keys: Sequence[str], default: Any = None) -> Any:
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_result(path: str) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _resolve_obstacle_list(track: SerpentineTrack) -> List[Tuple[float, float, float]]:
    # In this codebase, public getters may return empty lists after runtime masking.
    public = track.get_obstacles()
    if public:
        return list(public)
    private = getattr(track, "_obstacles", [])
    return list(private)


def build_step_records(result: Any, track: SerpentineTrack, controller_name: str) -> List[StepRecord]:
    states = np.asarray(result.states)
    solve_times = list(getattr(result, "solve_times", []))
    solve_statuses = list(getattr(result, "solve_statuses", []))
    solve_debug = list(getattr(result, "solve_debug", []))

    records: List[StepRecord] = []
    for i, st in enumerate(states):
        _, s, lat = track.closest_point(float(st[0]), float(st[1]))

        solve_time_ms = None
        if i < len(solve_times):
            solve_time_ms = float(solve_times[i]) * 1000.0

        status = ""
        if i < len(solve_statuses):
            status = str(solve_statuses[i])

        dbg = solve_debug[i] if i < len(solve_debug) else None
        active_pairs = _safe_get(dbg, ["horizon", "active_obstacle_pairs"], None)
        objective = _safe_get(dbg, ["horizon", "objective"], None)

        records.append(
            StepRecord(
                controller=controller_name,
                step=i,
                s=float(s),
                lat=float(lat),
                abs_lat=abs(float(lat)),
                v=float(st[3]),
                omega=float(st[4]),
                x=float(st[0]),
                y=float(st[1]),
                psi=float(st[2]),
                solve_time_ms=solve_time_ms,
                solve_status=status,
                solve_failed=int(bool(status) and status != "optimal"),
                active_obstacle_pairs=(int(active_pairs) if active_pairs is not None else None),
                objective=(float(objective) if objective is not None else None),
            )
        )
    return records


def write_csv(path: str, records: List[StepRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "controller",
                "step",
                "s",
                "lat",
                "abs_lat",
                "v",
                "omega",
                "x",
                "y",
                "psi",
                "solve_time_ms",
                "solve_status",
                "solve_failed",
                "active_obstacle_pairs",
                "objective",
            ],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row.__dict__)


def _steps_to_arrays(records: List[StepRecord]) -> Dict[str, np.ndarray]:
    return {
        "step": np.asarray([r.step for r in records]),
        "s": np.asarray([r.s for r in records]),
        "abs_lat": np.asarray([r.abs_lat for r in records]),
        "v": np.asarray([r.v for r in records]),
        "solve_failed": np.asarray([r.solve_failed for r in records]),
    }


def _find_robust_focus(records: List[StepRecord], obstacle_s: float, margin_s: float) -> Tuple[int, int]:
    arr = _steps_to_arrays(records)
    mask = np.where(np.abs(arr["s"] - obstacle_s) <= margin_s)[0]
    if len(mask) == 0:
        return max(0, len(records) - 200), len(records) - 1
    return int(mask[0]), int(mask[-1])


def _find_nonrobust_focus(records: List[StepRecord], crash_step: Optional[int], tail_steps: int) -> Tuple[int, int]:
    end = (crash_step if crash_step is not None else len(records) - 1)
    start = max(0, end - tail_steps)
    return int(start), int(end)


def plot_overview(
    output_path: str,
    robust: List[StepRecord],
    nonrobust: List[StepRecord],
    robust_focus: Tuple[int, int],
    nonrobust_focus: Tuple[int, int],
) -> None:
    r = _steps_to_arrays(robust)
    n = _steps_to_arrays(nonrobust)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(r["step"], r["s"], label="Robust", color="C0", linewidth=1.6)
    axes[0].plot(n["step"], n["s"], label="Non-robust", color="C3", linewidth=1.6)
    axes[0].set_ylabel("Arc-length s [m]")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(r["step"], r["v"], color="C0", linewidth=1.4)
    axes[1].plot(n["step"], n["v"], color="C3", linewidth=1.4)
    axes[1].set_ylabel("Speed v [m/s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(r["step"], r["abs_lat"], color="C0", linewidth=1.4)
    axes[2].plot(n["step"], n["abs_lat"], color="C3", linewidth=1.4)
    axes[2].axhline(TRACK_HALF_WIDTH, linestyle="--", color="k", linewidth=1.0, alpha=0.75, label="Boundary")
    axes[2].set_ylabel("|lat error| [m]")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)

    # Mark failed solve points
    r_bad = np.where(r["solve_failed"] > 0)[0]
    n_bad = np.where(n["solve_failed"] > 0)[0]
    axes[1].scatter(r["step"][r_bad], r["v"][r_bad], s=8, color="C0", alpha=0.45)
    axes[1].scatter(n["step"][n_bad], n["v"][n_bad], s=8, color="C3", alpha=0.45)

    # Highlight focus windows
    for ax in axes:
        ax.axvspan(robust_focus[0], robust_focus[1], color="C0", alpha=0.08)
        ax.axvspan(nonrobust_focus[0], nonrobust_focus[1], color="C3", alpha=0.08)

    fig.suptitle("Run Diagnostics Overview: Robust vs Non-robust", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_focus(
    output_path: str,
    robust: List[StepRecord],
    nonrobust: List[StepRecord],
    robust_focus: Tuple[int, int],
    nonrobust_focus: Tuple[int, int],
) -> None:
    r = _steps_to_arrays(robust)
    n = _steps_to_arrays(nonrobust)

    rs = slice(robust_focus[0], robust_focus[1] + 1)
    ns = slice(nonrobust_focus[0], nonrobust_focus[1] + 1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

    axes[0].plot(r["step"][rs], r["s"][rs], color="C0", linewidth=1.7, label="s")
    axes[0].set_ylabel("s [m]")
    ax0b = axes[0].twinx()
    ax0b.plot(r["step"][rs], r["v"][rs], color="C2", linewidth=1.2, label="v")
    ax0b.set_ylabel("v [m/s]")
    axes[0].set_title("Robust focus near obstacle zone")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n["step"][ns], n["abs_lat"][ns], color="C3", linewidth=1.7, label="|lat|")
    axes[1].axhline(TRACK_HALF_WIDTH, linestyle="--", color="k", linewidth=1.0, alpha=0.75)
    axes[1].set_ylabel("|lat error| [m]")
    ax1b = axes[1].twinx()
    ax1b.plot(n["step"][ns], n["s"][ns], color="C1", linewidth=1.2, label="s")
    ax1b.set_ylabel("s [m]")
    axes[1].set_xlabel("Step")
    axes[1].set_title("Non-robust pre-crash window")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSV and diagnostic charts for one robust/non-robust run pair.")
    parser.add_argument("--robust-pkl", required=True, help="Path to robust result .pkl")
    parser.add_argument("--nonrobust-pkl", required=True, help="Path to non-robust result .pkl")
    parser.add_argument("--output-dir", default="_output/diagnostics", help="Diagnostics output directory")
    parser.add_argument("--name", default="run", help="Name suffix for output artifacts")
    parser.add_argument("--obstacle-index", type=int, default=1, help="Obstacle index used as robust focus center")
    parser.add_argument("--obstacle-margin-s", type=float, default=8.0, help="Arc-length margin around selected obstacle [m]")
    parser.add_argument("--nonrobust-tail-steps", type=int, default=80, help="Tail window length before non-robust crash")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    robust_result = _load_result(args.robust_pkl)
    nonrobust_result = _load_result(args.nonrobust_pkl)
    track = SerpentineTrack()

    robust_records = build_step_records(robust_result, track, "robust")
    nonrobust_records = build_step_records(nonrobust_result, track, "nonrobust")

    obstacles = _resolve_obstacle_list(track)
    if len(obstacles) == 0:
        raise RuntimeError("No obstacles found on track; cannot build obstacle-centered diagnostics.")
    if args.obstacle_index < 0 or args.obstacle_index >= len(obstacles):
        raise ValueError(f"obstacle-index {args.obstacle_index} out of range [0, {len(obstacles)-1}]")

    obs = obstacles[args.obstacle_index]
    _, obstacle_s, _ = track.closest_point(float(obs[0]), float(obs[1]))

    robust_focus = _find_robust_focus(robust_records, float(obstacle_s), args.obstacle_margin_s)
    nonrobust_focus = _find_nonrobust_focus(
        nonrobust_records,
        getattr(nonrobust_result, "crash_step", None),
        args.nonrobust_tail_steps,
    )

    all_csv = os.path.join(args.output_dir, f"step_diagnostics_{args.name}.csv")
    robust_focus_csv = os.path.join(args.output_dir, f"step_diagnostics_robust_focus_{args.name}.csv")
    nonrobust_focus_csv = os.path.join(args.output_dir, f"step_diagnostics_nonrobust_focus_{args.name}.csv")
    overview_png = os.path.join(args.output_dir, f"diagnostics_overview_{args.name}.png")
    focus_png = os.path.join(args.output_dir, f"diagnostics_focus_{args.name}.png")

    write_csv(all_csv, robust_records + nonrobust_records)
    write_csv(robust_focus_csv, robust_records[robust_focus[0] : robust_focus[1] + 1])
    write_csv(nonrobust_focus_csv, nonrobust_records[nonrobust_focus[0] : nonrobust_focus[1] + 1])

    plot_overview(overview_png, robust_records, nonrobust_records, robust_focus, nonrobust_focus)
    plot_focus(focus_png, robust_records, nonrobust_records, robust_focus, nonrobust_focus)

    print("Diagnostics exported:")
    print(f"- all_csv: {all_csv}")
    print(f"- robust_focus_csv: {robust_focus_csv}")
    print(f"- nonrobust_focus_csv: {nonrobust_focus_csv}")
    print(f"- overview_png: {overview_png}")
    print(f"- focus_png: {focus_png}")
    print(f"- obstacle_index: {args.obstacle_index}")
    print(f"- obstacle_s: {float(obstacle_s):.3f}")
    print(f"- robust_focus_steps: {robust_focus[0]}..{robust_focus[1]}")
    print(f"- nonrobust_focus_steps: {nonrobust_focus[0]}..{nonrobust_focus[1]}")


if __name__ == "__main__":
    main()
