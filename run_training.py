"""
Entry point: Train the Deep Koopman model.
Implements the full training pipeline and computes the projection matrix D.
"""

import os
import sys
import json
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    N_X, N_U, N_Z, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    K_PRED, VAL_SPLIT, MODEL_DIR, GAMMA_RIDGE,
    DATA_NPZ_PATH, NORM_JSON_PATH, ORIGINAL_TS,
    LAMBDA_PHYSICS,
    LAMBDA_REG_HIGH_DIM,
    LAMBDA_SPECTRAL
)
from data.data_loader import load_and_subsample, create_datasets
from model.koopman_network import DeepKoopmanPaper
from model.koopman_trainer import train_model
from model.projection import (
    compute_projection_matrix,
    save_projection_matrix,
    get_fixed_selector_matrices,
)
from visualization.plot_tsne import plot_tsne_latent_space


def main():
    print("=" * 60)
    print("Deep Koopman Model Training (Paper Algorithm 1)")
    print("=" * 60)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Check if training data exists; if not, generate it automatically
    if not os.path.exists(DATA_NPZ_PATH) or not os.path.exists(NORM_JSON_PATH):
        print("\n[Info] Training data not found. Generating synthetic training data...")
        from generate_training_data import generate_data
        os.makedirs(MODEL_DIR, exist_ok=True)
        X_t, U_t, W_t, X_t1, norm_params_gen = generate_data(
            n_samples=500000, dt=0.01, seed=42
        )
        np.savez(DATA_NPZ_PATH, X_t=X_t, U_t=U_t, W_t=W_t,
                 X_t1=X_t1, Ts=ORIGINAL_TS)
        with open(NORM_JSON_PATH, 'w') as f:
            json.dump(norm_params_gen, f, indent=2)
        print(f"[Info] Training data saved to: {DATA_NPZ_PATH}")
        print(f"[Info] Norm params saved to: {NORM_JSON_PATH}")

    # Step 1: Load and subsample data
    print("\n--- Step 1: Loading data ---")
    X_sub, U_sub, W_sub, norm_params = load_and_subsample()

    # Step 2: Create datasets
    print("\n--- Step 2: Creating datasets ---")
    train_loader, val_loader = create_datasets(
        X_sub, U_sub, W_sub,
        window_len=K_PRED,
        val_split=VAL_SPLIT,
        batch_size=BATCH_SIZE
    )

    # Step 3: Initialize model
    print("\n--- Step 3: Initializing model ---")
    model = DeepKoopmanPaper(n_x=N_X, n_u=N_U, n_z=N_Z)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture:")
    print(f"  Encoder: z = [x, Lx], x:{N_X} -> z:{N_Z} (linear passthrough)")
    print(f"  Decoder: x_hat = z[:{N_X}] (linear selector)")
    print(f"  Linear: A({N_Z}x{N_Z}), B({N_Z}x{N_U})")

    # Freeze physically-initialized B elements during training
    # to prevent the optimizer from breaking the sign/magnitude priors.
    # z 的前5维顺序为 [px, py, psi, v, omega]
    def _freeze_b_grad_hook(grad):
        grad_clone = grad.clone()
        grad_clone[3, 0] = 0.0   # a -> v  (dt=0.1), v 在 z 中索引为 3
        grad_clone[4, 1] = 0.0   # delta -> omega, omega 在 z 中索引为 4
        return grad_clone

    model.B.register_hook(_freeze_b_grad_hook)
    print("  [Init] B[3,0] (a->v) and B[4,1] (delta->omega) gradients frozen.")

    # Freeze physically-initialized A elements to prevent drift
    # z 的前5维顺序为 [px, py, psi, v, omega]
    def _freeze_a_grad_hook(grad):
        grad_clone = grad.clone()
        grad_clone[0, 0] = 0.0   # px -> px
        grad_clone[1, 1] = 0.0   # py -> py
        grad_clone[2, 2] = 0.0   # psi -> psi
        grad_clone[2, 4] = 0.0   # omega -> psi (dt=0.1)
        grad_clone[3, 3] = 0.0   # v -> v
        grad_clone[4, 4] = 0.0   # omega -> omega
        return grad_clone

    model.A.register_hook(_freeze_a_grad_hook)
    print("  [Init] A physical elements (px/py/psi/v/omega) gradients frozen.")

    # Step 4: Train
    print("\n--- Step 4: Training ---")
    model, training_log = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=device,
        lambda_physics=LAMBDA_PHYSICS,
        lambda_reg_high_dim=LAMBDA_REG_HIGH_DIM,
        lambda_spectral=LAMBDA_SPECTRAL
    )

    # Step 5: Compute projection matrix D (diagnostic only)
    print("\n--- Step 5: Computing projection matrix D (diagnostic) ---")
    model = model.to('cpu')
    model.eval()
    D, r2 = compute_projection_matrix(model, X_sub, gamma=GAMMA_RIDGE)
    save_projection_matrix(D)

    # Paper-style fixed linear selectors for controller core path
    D_pos, E_v, F_omg, D_vomg = get_fixed_selector_matrices(N_Z)

    # Step 6: Save Koopman matrices
    print("\n--- Step 6: Saving Koopman matrices ---")
    A, B, C = model.get_matrices()
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(os.path.join(MODEL_DIR, 'koopman_matrices.npz'),
             A=A, B=B, C=C, D=D,
             D_pos=D_pos, E_v=E_v, F_omg=F_omg, D_vomg=D_vomg)
    print(f"  A: {A.shape}, max|eig|={np.max(np.abs(np.linalg.eigvals(A))):.4f}")
    print(f"  B: {B.shape}")
    print(f"  C: {C.shape}")
    print(f"  D(ridge, diagnostic): {D.shape}, R^2={r2.mean():.4f}")
    print(f"  Fixed selectors: D_pos{D_pos.shape}, E_v{E_v.shape}, F_omg{F_omg.shape}")

    # Save normalization params in output
    with open(os.path.join(MODEL_DIR, 'norm_params.json'), 'w') as f:
        json.dump(norm_params, f, indent=2)

    # Step 7: Generate t-SNE visualization (Figure 5)
    print("\n--- Step 7: Generating t-SNE visualization ---")
    plot_tsne_latent_space(model, X_sub)

    # Step 8: Print training summary (Table 6)
    print("\n--- Step 8: Training Summary (Table 6) ---")
    from visualization.plot_tables import print_table_6
    print_table_6(training_log)

    print("\nTraining pipeline complete.")
    print(f"Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
