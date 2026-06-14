"""
Trajectory visualization for Figures 6 and 7.
Overlays all 4 methods' trajectories on the track with obstacles.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    METHOD_COLORS, METHOD_LABELS, FIGURES_DIR,
    FIGURE_DPI, FIGURE_FORMAT, D_SAFE, VEHICLE_RADIUS,
    PLOT_TRACK_HALF_WIDTH
)
from visualization.plot_utils import add_figure_timestamp


def _compute_track_boundaries(track, half_width=PLOT_TRACK_HALF_WIDTH):
    """Build left/right track boundaries from centerline and heading."""
    cx, cy = track.get_centerline()
    heading = track.get_heading()
    nx = -np.sin(heading)
    ny = np.cos(heading)
    left_x = cx + half_width * nx
    left_y = cy + half_width * ny
    right_x = cx - half_width * nx
    right_y = cy - half_width * ny
    return (left_x, left_y), (right_x, right_y)


def _add_crashed_banner(fig, ax, result, color='red'):
    """Overlay a visible crash marker when a run ended by boundary hit."""
    if not getattr(result, 'crashed', False):
        return

    states = result.to_arrays()['states']
    if len(states) > 0:
        ax.plot(states[-1, 0], states[-1, 1], marker='x', color=color,
                markersize=12, mew=3, linestyle='None', zorder=6)
        ax.text(states[-1, 0], states[-1, 1], 'CRASHED', color=color,
                fontsize=12, fontweight='bold', ha='left', va='bottom', zorder=7)

    fig.text(
        0.5,
        0.965,
        'CRASHED',
        color=color,
        fontsize=24,
        fontweight='bold',
        ha='center',
        va='top',
        alpha=0.9,
    )


def plot_trajectory_comparison(results, track, title="Trajectory Comparison",
                               filename=None, save_dir=FIGURES_DIR):
    """
    Plot all methods' trajectories on a single figure.

    Args:
        results: dict {method_name: SimResult}
        track: BaseTrack instance
        title: figure title
        filename: output filename
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Plot track centerline
    cx, cy = track.get_centerline()
    (left_x, left_y), (right_x, right_y) = _compute_track_boundaries(track)
    ax.plot(left_x, left_y, '-', color='#444444', linewidth=1.2, alpha=0.8,
            label='Track boundary')
    ax.plot(right_x, right_y, '-', color='#444444', linewidth=1.2, alpha=0.8)
    ax.plot(cx, cy, '--', color='gray', linewidth=1.5, alpha=0.6, label='Track centerline')

    # Plot rectangular obstacles first (if track provides them)
    rect_obstacles = track.get_rect_obstacles() if hasattr(track, 'get_rect_obstacles') else []
    for i, (cx_r, cy_r, length, width, angle) in enumerate(rect_obstacles):
        c = np.cos(angle)
        s = np.sin(angle)
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
        poly = Polygon(corners, closed=True, facecolor='red', edgecolor='darkred',
                       linewidth=1.2, alpha=0.25, zorder=2)
        ax.add_patch(poly)
        ax.annotate(f'Rect {i+1}', (cx_r, cy_r), fontsize=8, ha='center', va='center')

    # Plot obstacles (circular approximation used by controller)
    obstacles = track.get_obstacles()
    for i, (ox, oy, r) in enumerate(obstacles):
        # Obstacle body
        circle = Circle((ox, oy), r, color='red', alpha=0.4)
        ax.add_patch(circle)
        # Safety margin
        circle_safe = Circle((ox, oy), r + VEHICLE_RADIUS + D_SAFE,
                            fill=False, edgecolor='red', linestyle='--',
                            linewidth=1, alpha=0.5)
        ax.add_patch(circle_safe)
        ax.annotate(f'Obs {i+1}', (ox, oy), fontsize=8, ha='center', va='center')

    # Plot each method's trajectory
    for method_name, result in results.items():
        data = result.to_arrays()
        states = data['states']
        color = METHOD_COLORS.get(method_name, 'black')
        label = METHOD_LABELS.get(method_name, method_name)
        ax.plot(states[:, 0], states[:, 1], color=color, linewidth=1.5,
                label=label, alpha=0.8)

        # Mark start position
        ax.plot(states[0, 0], states[0, 1], 'o', color=color, markersize=8)
        _add_crashed_banner(fig, ax, result, color='red')

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    add_figure_timestamp(fig)

    if filename is None:
        filename = f"trajectory_{track.__class__.__name__}.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectory plot saved to {filepath}")


def plot_state_comparison(results, track, filename=None, save_dir=FIGURES_DIR):
    """
    Plot state variable comparisons over time for all methods.
    """
    os.makedirs(save_dir, exist_ok=True)

    state_labels = ['$p_x$ (m)', '$p_y$ (m)', '$\\psi$ (rad)',
                    '$v$ (m/s)', '$\\omega$ (rad/s)']

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)

    for method_name, result in results.items():
        data = result.to_arrays()
        states = data['states']
        t = np.arange(len(states)) * 0.1  # DT = 0.1s
        color = METHOD_COLORS.get(method_name, 'black')
        label = METHOD_LABELS.get(method_name, method_name)

        for i in range(5):
            axes[i].plot(t, states[:, i], color=color, linewidth=1,
                        label=label, alpha=0.8)

    for i in range(5):
        axes[i].set_ylabel(state_labels[i], fontsize=11)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='best', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle(f'State Evolution: {track.__class__.__name__}', fontsize=14)

    if any(getattr(result, 'crashed', False) for result in results.values()):
        fig.text(0.5, 0.985, 'CRASHED', color='red', fontsize=22,
                 fontweight='bold', ha='center', va='top', alpha=0.9)

    plt.tight_layout()
    add_figure_timestamp(fig)

    if filename is None:
        filename = f"states_{track.__class__.__name__}.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"State comparison plot saved to {filepath}")


def plot_control_comparison(results, filename=None, save_dir=FIGURES_DIR):
    """
    Plot control inputs over time for all methods.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    control_labels = ['Acceleration $a$ (m/s$^2$)', 'Steering $\\delta$ (rad)']

    for method_name, result in results.items():
        data = result.to_arrays()
        controls = data['controls']
        t = np.arange(len(controls)) * 0.1
        color = METHOD_COLORS.get(method_name, 'black')
        label = METHOD_LABELS.get(method_name, method_name)

        for i in range(2):
            axes[i].plot(t, controls[:, i], color=color, linewidth=1,
                        label=label, alpha=0.8)

    for i in range(2):
        axes[i].set_ylabel(control_labels[i], fontsize=11)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc='best', fontsize=9)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.suptitle('Control Input Comparison', fontsize=14)

    if any(getattr(result, 'crashed', False) for result in results.values()):
        fig.text(0.5, 0.985, 'CRASHED', color='red', fontsize=22,
                 fontweight='bold', ha='center', va='top', alpha=0.9)

    plt.tight_layout()
    add_figure_timestamp(fig)

    if filename is None:
        filename = f"controls.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Control comparison plot saved to {filepath}")
