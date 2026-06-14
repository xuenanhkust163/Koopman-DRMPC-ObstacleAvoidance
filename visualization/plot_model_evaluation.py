"""Plots for trained Koopman model evaluation summaries."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURE_DPI, FIGURE_FORMAT, FIGURES_DIR
from visualization.plot_utils import add_figure_timestamp


STATE_NAMES = ["px_norm", "py_norm", "psi", "v", "omega"]


def _state_values(summary, key):
    return np.array([summary[key][state_name] for state_name in STATE_NAMES], dtype=float)


def plot_rmse_comparison(summary, filename=None, save_dir=FIGURES_DIR):
    """Plot one-step and multi-step RMSE against a hold-state baseline."""
    os.makedirs(save_dir, exist_ok=True)

    one_step_model = _state_values(summary, "one_step_rmse")
    one_step_baseline = _state_values(summary, "baseline_one_step_rmse")
    multi_step_model = _state_values(summary, "multi_step_rmse")
    multi_step_baseline = _state_values(summary, "baseline_multi_step_rmse")

    x = np.arange(len(STATE_NAMES))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=False)

    panels = [
        (axes[0], one_step_model, one_step_baseline, "One-step RMSE"),
        (axes[1], multi_step_model, multi_step_baseline, f"{summary['dataset_sizes']['val_batches']} Validation Batches Multi-step RMSE"),
    ]

    for ax, model_vals, baseline_vals, title in panels:
        ax.bar(x - width / 2, model_vals, width=width, label="Koopman model", color="#1f77b4")
        ax.bar(x + width / 2, baseline_vals, width=width, label="Hold-state baseline", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_NAMES, rotation=20)
        ax.set_title(title)
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)

    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Koopman Model Evaluation: RMSE Comparison", fontsize=14)
    plt.tight_layout()
    add_figure_timestamp(fig, prefix="Evaluated")

    if filename is None:
        model_stem = os.path.splitext(os.path.basename(summary["model_path"]))[0]
        filename = f"{model_stem}_rmse_comparison.{FIGURE_FORMAT}"

    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath