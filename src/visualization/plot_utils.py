"""Shared plotting utilities."""

from datetime import datetime


def add_figure_timestamp(fig, prefix="Generated"):
    """Add a visible generation timestamp on the figure canvas."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(
        0.995,
        0.005,
        f"{prefix}: {ts}",
        ha="right",
        va="bottom",
        fontsize=8,
        alpha=0.8,
    )
