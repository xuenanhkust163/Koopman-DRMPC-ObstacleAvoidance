"""
t-SNE visualization of the Koopman latent space (Figure 5).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT
from visualization.plot_utils import add_figure_timestamp


def plot_tsne_latent_space(model, X_data, filename=None, save_dir=FIGURES_DIR,
                           perplexity=30, n_samples=5000):
    """
    Visualize the latent space using t-SNE (Figure 5 of paper).

    Args:
        model: trained DeepKoopmanPaper model
        X_data: (N, 5) normalized state data
        filename: output filename
        save_dir: output directory
        perplexity: t-SNE perplexity
        n_samples: number of samples to use (subsampled for speed)
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    # Subsample if needed
    N = X_data.shape[0]
    if N > n_samples:
        indices = np.random.choice(N, n_samples, replace=False)
        X_sub = X_data[indices]
    else:
        X_sub = X_data

    # Encode to latent space
    X_tensor = torch.tensor(X_sub, dtype=torch.float32)
    with torch.no_grad():
        Z = model.encode(X_tensor).numpy()  # (n_samples, 32)

    # Run t-SNE
    print(f"Running t-SNE on {len(Z)} latent vectors (dim={Z.shape[1]})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                max_iter=1000)
    Z_2d = tsne.fit_transform(Z)  # (n_samples, 2)
    print(f"t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")

    # Color by velocity (index 2)
    velocities = X_sub[:, 2]

    # Classify driving regimes for coloring
    # Based on curvature of trajectory (use omega as proxy)
    omega = X_sub[:, 4]
    v = X_sub[:, 2]

    # Regime classification
    regime = np.zeros(len(X_sub), dtype=int)
    omega_thresh = np.percentile(np.abs(omega), 66)
    v_thresh = np.percentile(v, 50)

    # 0: Straight (low omega, high v)
    # 1: Corner entry (increasing omega)
    # 2: Apex (high omega)
    # 3: Corner exit (decreasing omega)
    regime[(np.abs(omega) < omega_thresh) & (v > v_thresh)] = 0   # Straight
    regime[(np.abs(omega) >= omega_thresh) & (v > v_thresh)] = 1  # Corner entry
    regime[(np.abs(omega) >= omega_thresh) & (v <= v_thresh)] = 2  # Apex
    regime[(np.abs(omega) < omega_thresh) & (v <= v_thresh)] = 3   # Corner exit

    regime_names = ['Straight', 'Corner Entry', 'Apex', 'Corner Exit']
    regime_colors = ['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']

    # === Figure 5a: Colored by driving regime ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for r in range(4):
        mask = regime == r
        axes[0].scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                       c=regime_colors[r], s=5, alpha=0.5,
                       label=regime_names[r])

    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title('Latent Space (Colored by Driving Regime)', fontsize=13)
    axes[0].legend(markerscale=5, fontsize=10)

    # === Figure 5b: Colored by velocity ===
    scatter = axes[1].scatter(Z_2d[:, 0], Z_2d[:, 1],
                             c=velocities, cmap='viridis', s=5, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Velocity (m/s)', fontsize=11)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title('Latent Space (Colored by Velocity)', fontsize=13)

    plt.tight_layout()
    add_figure_timestamp(fig)

    if filename is None:
        filename = f"fig5_tsne.{FIGURE_FORMAT}"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"t-SNE visualization saved to {filepath}")
