"""Animate trajectory and key states from simulation outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DT, PLOT_TRACK_HALF_WIDTH
from simulation.simulator import Simulator
from tracks.custom_track import CustomWindingTrack
from tracks.lusail_short_track import LusailShortTrack
from tracks.lusail_track import LusailTrack
from tracks.sprint_oval_track import SprintOvalTrack
from tracks.serpentine_track import SerpentineTrack
from tracks.monaco_hairpin_track import MonacoHairpinTrack
from tracks.suzuka_flow_track import SuzukaFlowTrack
from tracks.spa_flow_track import SpaFlowTrack

IDX_PX = 0
IDX_PY = 1
IDX_PSI = 2
IDX_V = 3
IDX_OMEGA = 4


def _get_visible_rect_obstacles(track):
    """Return rectangular obstacles even when global plotting flags are currently off."""
    rects = track.get_rect_obstacles() if hasattr(track, "get_rect_obstacles") else []
    if rects:
        return rects
    return list(getattr(track, "_rect_obstacles", []))


def _compute_track_boundaries(track, half_width=PLOT_TRACK_HALF_WIDTH):
    cx, cy = track.get_centerline()
    heading = track.get_heading()
    nx = -np.sin(heading)
    ny = np.cos(heading)
    left_x = cx + half_width * nx
    left_y = cy + half_width * ny
    right_x = cx - half_width * nx
    right_y = cy - half_width * ny
    return (left_x, left_y), (right_x, right_y)


def _build_track(track_name: str):
    if track_name == "LusailTrack":
        return LusailTrack()
    if track_name == "LusailShortTrack":
        return LusailShortTrack()
    if track_name == "CustomWindingTrack":
        return CustomWindingTrack()
    if track_name == "SprintOvalTrack":
        return SprintOvalTrack()
    if track_name == "SerpentineTrack":
        return SerpentineTrack()
    if track_name == "MonacoHairpinTrack":
        return MonacoHairpinTrack()
    if track_name == "SuzukaFlowTrack":
        return SuzukaFlowTrack()
    if track_name == "SpaFlowTrack":
        return SpaFlowTrack()
    raise ValueError(f"Unsupported track name: {track_name}")


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _compute_errors(states: np.ndarray, track) -> Dict[str, np.ndarray]:
    lat_err = np.zeros(len(states), dtype=float)
    psi_err = np.zeros(len(states), dtype=float)
    ref_psi = np.zeros(len(states), dtype=float)
    heading = track.get_heading()

    for i in range(len(states)):
        x = float(states[i, IDX_PX])
        y = float(states[i, IDX_PY])
        idx, _, lat = track.closest_point(x, y)
        lat_err[i] = float(lat)
        psi_ref = float(heading[idx])
        ref_psi[i] = psi_ref
        psi_err[i] = float(_wrap_angle(np.array([states[i, IDX_PSI] - psi_ref]))[0])

    return {"lat_err": lat_err, "psi_err": psi_err, "ref_psi": ref_psi}


def _load_from_result(path: Path) -> Dict[str, np.ndarray]:
    result = Simulator.load_result(str(path))
    data = result.to_arrays()
    states = np.asarray(data["states"], dtype=float)
    timestamps = np.asarray(data["timestamps"], dtype=float)
    if len(timestamps) == len(states) - 1:
        if len(timestamps) > 0:
            timestamps = np.concatenate([timestamps, [timestamps[-1] + DT]])
        else:
            timestamps = np.arange(len(states), dtype=float) * DT
    elif len(timestamps) != len(states):
        timestamps = np.arange(len(states), dtype=float) * DT

    ref_states = np.asarray(result.ref_states, dtype=float) if len(result.ref_states) else np.empty((0, 5))
    ref_v = np.full(len(states), np.nan, dtype=float)
    ref_omega = np.full(len(states), np.nan, dtype=float)
    if len(ref_states) > 0:
        n = min(len(states) - 1, len(ref_states))
        ref_v[:n] = ref_states[:n, IDX_V]
        ref_omega[:n] = ref_states[:n, IDX_OMEGA]

    return {
        "method": result.method_name,
        "track": result.track_name,
        "states": states,
        "time": timestamps,
        "ref_v": ref_v,
        "ref_omega": ref_omega,
    }


def _load_from_step_log(path: Path) -> Dict[str, np.ndarray]:
    method = "Unknown"
    track_name = "Unknown"
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# method="):
                method = line.split("=", 1)[1]
                continue
            if line.startswith("# track="):
                track_name = line.split("=", 1)[1]
                continue
            if line.startswith("#"):
                continue
            if line.startswith("step") or line.startswith("-"):
                continue
            if not line[0].isdigit():
                continue

            cols = line.split()
            if len(cols) < 15:
                continue

            # step t status solve_ms x y psi v omega ref_v ref_omega a delta next_v next_omega
            rows.append(
                (
                    int(cols[0]),
                    float(cols[1]),
                    float(cols[4]),
                    float(cols[5]),
                    float(cols[6]),
                    float(cols[7]),
                    float(cols[8]),
                    float(cols[9]),
                    float(cols[10]),
                )
            )

    if not rows:
        raise ValueError(f"No step rows parsed from log: {path}")

    rows = np.asarray(rows, dtype=float)
    states = rows[:, 2:7]
    time = rows[:, 1]
    ref_v = rows[:, 7]
    ref_omega = rows[:, 8]

    return {
        "method": method,
        "track": track_name,
        "states": states,
        "time": time,
        "ref_v": ref_v,
        "ref_omega": ref_omega,
    }


def _resolve_input(result_path: Optional[str], log_path: Optional[str]) -> Dict[str, np.ndarray]:
    if result_path:
        return _load_from_result(Path(result_path))
    if log_path:
        return _load_from_step_log(Path(log_path))

    latest_result = sorted(Path("_output/results").glob("*.pkl"), key=lambda p: p.stat().st_mtime)
    if latest_result:
        return _load_from_result(latest_result[-1])

    latest_log = sorted(Path("_output/results/logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    if latest_log:
        return _load_from_step_log(latest_log[-1])

    raise FileNotFoundError("No input found. Provide --result or --log.")


def _build_animation(
    run: Dict[str, np.ndarray],
    track,
    fps: int,
    speed: float,
    tail: int,
    psi_highlight_threshold_deg: Optional[float],
    max_frames: Optional[int] = None,
    crashed: bool = False,
    crash_step: Optional[int] = None,
    crash_time: Optional[float] = None,
    crash_reason: Optional[str] = None,
):
    states = run["states"]
    t = run["time"]
    ref_v = run["ref_v"]
    controls = np.asarray(run.get("controls", np.empty((0, 2))), dtype=float)

    accel = np.full(len(states), np.nan, dtype=float)
    delta = np.full(len(states), np.nan, dtype=float)
    if len(controls) > 0:
        n_ctrl = min(len(states), len(controls))
        accel[:n_ctrl] = controls[:n_ctrl, 0]
        delta[:n_ctrl] = controls[:n_ctrl, 1]
    delta_rate = np.full(len(states), np.nan, dtype=float)
    if np.any(np.isfinite(delta)):
        delta_filled = np.nan_to_num(delta, nan=0.0)
        delta_rate[0] = 0.0
        if len(delta_filled) > 1:
            delta_rate[1:] = np.diff(delta_filled) / DT

    errs = _compute_errors(states, track)
    lat_err = errs["lat_err"]
    psi_err_deg = np.rad2deg(errs["psi_err"])
    ref_psi = errs["ref_psi"]

    cx, cy = track.get_centerline()
    (lx, ly), (rx, ry) = _compute_track_boundaries(track)

    # Determine trajectory color: render robust controllers in blue
    source_path = run.get("source_path", "") if isinstance(run, dict) else ""
    source_l = source_path.lower() if source_path else ""
    method_name = run.get("method", "") if isinstance(run, dict) else ""
    ml = method_name.lower() if method_name else ""
    # Detect robust either from method name or from filename/path
    is_robust = any(tok in ml for tok in ("k-dr", "kdr", "drmpc", "dr-mpc", "robust")) or any(
        tok in source_l for tok in ("_robust", "robust", "k-dr", "kdr", "drmpc")
    )
    traj_color = "#1f77b4" if is_robust else "#d62728"

    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[2.15, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
    )

    ax_xy = fig.add_subplot(gs[:, 0])
    # Right panel layout (top to bottom):
    # Row 1: lateral error
    # Row 2: speed + acceleration/deceleration
    # Row 3: heading error + steering angle + steering rate
    ax_lat = fig.add_subplot(gs[0, 1:4])
    ax_v = fig.add_subplot(gs[1, 1:3])
    ax_a = fig.add_subplot(gs[1, 3])
    ax_psi = fig.add_subplot(gs[2, 1])
    ax_delta = fig.add_subplot(gs[2, 2])
    ax_drate = fig.add_subplot(gs[2, 3])

    ax_xy.plot(lx, ly, "-", color="#444444", linewidth=1.2, alpha=0.8)
    ax_xy.plot(rx, ry, "-", color="#444444", linewidth=1.2, alpha=0.8)
    ax_xy.plot(cx, cy, "--", color="gray", linewidth=1.2, alpha=0.6)

    rect_obstacles = _get_visible_rect_obstacles(track)
    for cx_r, cy_r, length, width, angle in rect_obstacles:
        c = math.cos(angle)
        s = math.sin(angle)
        hl = 0.5 * length
        hw = 0.5 * width
        local_corners = np.array([
            [-hl, -hw],
            [hl, -hw],
            [hl, hw],
            [-hl, hw],
        ])
        rot = np.array([[c, -s], [s, c]])
        corners = (local_corners @ rot.T) + np.array([cx_r, cy_r])
        ax_xy.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor="black",
                edgecolor="black",
                linewidth=1.6,
                alpha=0.65,
                zorder=3,
            )
        )

    traj_line, = ax_xy.plot([], [], color=traj_color, linewidth=1.8, label=run["method"])
    car_dot, = ax_xy.plot([], [], "o", color=traj_color, markersize=6)
    car_arrow = ax_xy.quiver([], [], [], [], color=traj_color, scale=20, width=0.004)

    ax_xy.set_title(f"{run['method']} on {run['track']}")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(loc="upper right")

    v_line, = ax_v.plot(t, states[:, IDX_V], color="#1f77b4", linewidth=1.2, label="v")
    vref_line, = ax_v.plot(t, ref_v, "--", color="#2ca02c", linewidth=1.0, label="ref v")
    v_now = ax_v.axvline(t[0], color="black", linewidth=1.0, alpha=0.6)
    ax_v.set_title("Speed")
    ax_v.set_xlabel("t (s)")
    ax_v.set_ylabel("m/s")
    ax_v.grid(alpha=0.3)
    ax_v.legend(loc="upper right")

    a_hist, = ax_a.plot([], [], color="#17becf", linewidth=1.2)
    a_now = ax_a.axvline(t[0], color="black", linewidth=1.0, alpha=0.6)
    ax_a.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax_a.set_title("Acceleration / Deceleration")
    ax_a.set_xlabel("t (s)")
    ax_a.set_ylabel("m/s$^2$")
    ax_a.grid(alpha=0.3)

    lat_hist, = ax_lat.plot([], [], color="#ff7f0e", linewidth=1.2)
    lat_now = ax_lat.axvline(t[0], color="black", linewidth=1.0, alpha=0.6)
    ax_lat.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax_lat.set_title("Lateral Error")
    ax_lat.set_xlabel("t (s)")
    ax_lat.set_ylabel("m")
    ax_lat.grid(alpha=0.3)

    drate_hist, = ax_drate.plot([], [], color="#8c564b", linewidth=1.2)
    drate_now = ax_drate.axvline(t[0], color="black", linewidth=1.0, alpha=0.6)
    ax_drate.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax_drate.set_title("Steering Rate")
    ax_drate.set_xlabel("t (s)")
    ax_drate.set_ylabel("rad/s")
    ax_drate.grid(alpha=0.3)

    delta_hist, = ax_delta.plot([], [], color="#bcbd22", linewidth=1.2)
    delta_now = ax_delta.axvline(t[0], color="black", linewidth=1.0, alpha=0.6)
    ax_delta.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax_delta.set_title("Steering Angle")
    ax_delta.set_xlabel("t (s)")
    ax_delta.set_ylabel("deg")
    ax_delta.grid(alpha=0.3)

    psi_hist, = ax_psi.plot([], [], color="#9467bd", linewidth=1.2)
    psi_now = ax_psi.axvline(t[0], color="black", linewidth=1.0, alpha=0.6)
    ax_psi.axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    ax_psi.set_title("Heading Error")
    ax_psi.set_xlabel("t (s)")
    ax_psi.set_ylabel("deg")
    ax_psi.grid(alpha=0.3)

    ax_lat.set_xlim(t[0], t[-1])
    ax_a.set_xlim(t[0], t[-1])
    ax_drate.set_xlim(t[0], t[-1])
    ax_delta.set_xlim(t[0], t[-1])
    ax_psi.set_xlim(t[0], t[-1])
    if np.any(np.isfinite(accel)):
        m = np.nanmax(np.abs(accel))
        ax_a.set_ylim(-1.1 * m - 1e-6, 1.1 * m + 1e-6)
    if np.isfinite(np.nanmax(np.abs(lat_err))):
        m = np.nanmax(np.abs(lat_err))
        ax_lat.set_ylim(-1.1 * m - 1e-6, 1.1 * m + 1e-6)
    if np.any(np.isfinite(delta_rate)):
        m = np.nanmax(np.abs(delta_rate))
        ax_drate.set_ylim(-1.1 * m - 1e-6, 1.1 * m + 1e-6)
    if np.any(np.isfinite(delta)):
        delta_deg = np.rad2deg(delta)
        m = np.nanmax(np.abs(delta_deg))
        ax_delta.set_ylim(-1.1 * m - 1e-6, 1.1 * m + 1e-6)
    if np.isfinite(np.nanmax(np.abs(psi_err_deg))):
        m = np.nanmax(np.abs(psi_err_deg))
        ax_psi.set_ylim(-1.1 * m - 1e-6, 1.1 * m + 1e-6)

    text_box = ax_xy.text(
        0.02,
        0.98,
        "",
        transform=ax_xy.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    # Crash visualization
    crash_text = ax_xy.text(
        0.5,
        0.5,
        "",
        transform=ax_xy.transAxes,
        va="center",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color="red",
        bbox={"facecolor": "yellow", "alpha": 0.9, "edgecolor": "red", "linewidth": 2},
        visible=False,
    )
    crash_marker, = ax_xy.plot([], [], "X", color="red", markersize=15, markeredgewidth=2, markeredgecolor="darkred")

    frame_interval_ms = 1000.0 / max(1, fps) / max(1e-6, speed)
    if max_frames is not None and max_frames > 0 and len(states) > max_frames:
        frame_indices = np.unique(
            np.linspace(0, len(states) - 1, num=int(max_frames), dtype=int)
        )
    else:
        frame_indices = np.arange(len(states), dtype=int)

    def _update(frame_idx: int):
        start = max(0, frame_idx - tail) if tail > 0 else 0
        xs = states[start : frame_idx + 1, IDX_PX]
        ys = states[start : frame_idx + 1, IDX_PY]
        traj_line.set_data(xs, ys)

        x = states[frame_idx, IDX_PX]
        y = states[frame_idx, IDX_PY]
        psi = states[frame_idx, IDX_PSI]
        psi_target = ref_psi[frame_idx]
        car_dot.set_data([x], [y])
        car_arrow.set_offsets(np.array([[x, y]]))
        car_arrow.set_UVC(np.array([math.cos(psi)]), np.array([math.sin(psi)]))

        lat_hist.set_data(t[: frame_idx + 1], lat_err[: frame_idx + 1])
        a_hist.set_data(t[: frame_idx + 1], accel[: frame_idx + 1])
        drate_hist.set_data(t[: frame_idx + 1], delta_rate[: frame_idx + 1])
        delta_deg = np.rad2deg(delta)
        delta_hist.set_data(t[: frame_idx + 1], delta_deg[: frame_idx + 1])
        psi_hist.set_data(t[: frame_idx + 1], psi_err_deg[: frame_idx + 1])
        v_now.set_xdata([t[frame_idx], t[frame_idx]])
        a_now.set_xdata([t[frame_idx], t[frame_idx]])
        lat_now.set_xdata([t[frame_idx], t[frame_idx]])
        drate_now.set_xdata([t[frame_idx], t[frame_idx]])
        delta_now.set_xdata([t[frame_idx], t[frame_idx]])
        psi_now.set_xdata([t[frame_idx], t[frame_idx]])

        text_box.set_text(
            f"step: {frame_idx}\n"
            f"t: {t[frame_idx]:.2f}s\n"
            f"v: {states[frame_idx, IDX_V]:.2f} m/s\n"
            f"ref v: {ref_v[frame_idx]:.2f} m/s\n"
            f"a: {accel[frame_idx]:.3f} m/s^2\n"
            f"delta: {delta_deg[frame_idx]:.2f} deg\n"
            f"delta_dot: {delta_rate[frame_idx]:.3f} rad/s\n"
            f"e_y: {lat_err[frame_idx]:.2f} m\n"
            f"e_psi: {psi_err_deg[frame_idx]:.1f} deg\n"
            f"psi_ref: {np.rad2deg(psi_target):.1f} deg"
        )

        # Handle crash visualization
        if crashed and crash_step is not None and frame_idx >= crash_step - 1:
            crash_text.set_visible(True)
            crash_text.set_text(f"💥 CRASH 💥\n{crash_reason or 'Track boundary hit'}")
            crash_marker.set_data([x], [y])
        else:
            crash_text.set_visible(False)
            crash_marker.set_data([], [])

        return (
            traj_line,
            car_dot,
            car_arrow,
            lat_hist,
            psi_hist,
            a_hist,
            drate_hist,
            delta_hist,
            v_line,
            vref_line,
            v_now,
            a_now,
            lat_now,
            drate_now,
            delta_now,
            psi_now,
            text_box,
            crash_text,
            crash_marker,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=frame_indices,
        interval=frame_interval_ms,
        blit=False,
        repeat=False,
    )

    fig.tight_layout()
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Animate trajectory and key states from simulation outputs.")
    parser.add_argument("--result", type=str, default=None, help="Path to saved .pkl result")
    parser.add_argument("--log", type=str, default=None, help="Path to step log (.log)")
    parser.add_argument("--track", type=str, default=None, help="Override track name")
    parser.add_argument("--fps", type=int, default=20, help="Animation frames per second")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--tail", type=int, default=0, help="Show only last N points (0 means full history)")
    parser.add_argument("--save", type=str, default=None, help="Save animation to .gif or .mp4")
    parser.add_argument(
        "--fast-gif",
        action="store_true",
        help="Use faster GIF export settings (fewer frames and lower DPI).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of rendered frames. If omitted, GIF export auto-limits frame count.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Output DPI for saved animation. If omitted, GIF export uses a lower DPI automatically.",
    )
    parser.add_argument(
        "--psi-highlight-threshold-deg",
        type=float,
        default=None,
        help="If set, highlight target heading arrow when |heading error| exceeds this degree threshold.",
    )
    args = parser.parse_args()

    run = _resolve_input(args.result, args.log)
    track_name = args.track or run["track"]
    track = _build_track(track_name)

    save_ext = Path(args.save).suffix.lower() if args.save else ""
    save_is_gif = save_ext == ".gif"
    max_frames = args.max_frames
    if save_is_gif and max_frames is None:
        max_frames = 120 if args.fast_gif else 240
    dpi = args.dpi
    if save_is_gif and dpi is None:
        dpi = 72 if args.fast_gif else 80

    fig, anim = _build_animation(
        run,
        track,
        fps=args.fps,
        speed=args.speed,
        tail=args.tail,
        psi_highlight_threshold_deg=args.psi_highlight_threshold_deg,
        max_frames=max_frames,
        crashed=run.get("crashed", False),
        crash_step=run.get("crash_step", None),
        crash_time=run.get("crash_time", None),
        crash_reason=run.get("crash_reason", None),
    )

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = out.suffix.lower()
        if ext == ".gif":
            anim.save(str(out), writer="pillow", fps=max(1, args.fps), dpi=dpi)
        else:
            anim.save(str(out), writer="ffmpeg", fps=max(1, args.fps), dpi=dpi)
        print(f"Saved animation to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
