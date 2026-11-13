"""
Compute the projection matrix D using ridge regression.

The projection matrix D ∈ R^{2 × n_z} maps the Koopman latent state z to [v, omega]:
    [v, omega]^T ≈ D @ z

Paper Section 3.5: D = Y @ Z^T @ (Z @ Z^T + gamma * I)^{-1}

The projection matrix is used to:
1. Extract control-related variables (speed and yaw rate) from Koopman space
2. Construct control-related cost terms in MPC optimization
3. Ensure that the Koopman-space representation accurately reflects physical quantities
4. Bridge the linear Koopman model and nonlinear control objectives

Benefits of ridge regression:
- Regularization helps prevent overfitting
- Ensures (Z^T Z + gamma * I) is invertible
- Improves numerical stability
"""

import os  # Operating system interfaces for file and directory operations
import sys  # System module for modifying the Python path
import numpy as np  # NumPy for numerical computing and matrix operations
import torch  # PyTorch deep learning framework for model inference

# Add the parent directory to the system path so sibling modules can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import projection-matrix parameters from the configuration.
from config import (
    N_Z,  # Koopman-space dimension
    GAMMA_RIDGE,  # Ridge-regression regularization parameter
    MODEL_DIR,  # Model output directory
    IDX_V,
    IDX_OMEGA
)


def get_fixed_selector_matrices(n_z=N_Z):
    """
    Build the fixed, non-learned linear selectors used by the paper controller.

    The first five latent-state dimensions are assumed to correspond to physical states in this order:
        z[0:5] <-> [px, py, psi, v, omega]

    Returns:
        D_pos: (2, n_z), selects [px, py]
        E_v:   (1, n_z), selects [v]
        F_omg: (1, n_z), selects [omega]
        D_vomg:(2, n_z), selects [v, omega]
    """
    D_pos = np.zeros((2, n_z))
    D_pos[0, 0] = 1.0  # px
    D_pos[1, 1] = 1.0  # py

    E_v = np.zeros((1, n_z))
    E_v[0, IDX_V] = 1.0

    F_omg = np.zeros((1, n_z))
    F_omg[0, IDX_OMEGA] = 1.0

    D_vomg = np.zeros((2, n_z))
    D_vomg[0, IDX_V] = 1.0
    D_vomg[1, IDX_OMEGA] = 1.0

    return D_pos, E_v, F_omg, D_vomg


def compute_projection_matrix(model, X_data, gamma=GAMMA_RIDGE, device='cpu'):
    """
    Compute the projection matrix D using ridge regression.

    The projection matrix D maps the Koopman latent state z to the control-related
    variables [v, omega]:
        [v, omega]^T ≈ D @ z

    Procedure:
    1. Encode all states x into Koopman-space representations z using the trained encoder.
    2. Extract the target variables Y = [v, omega].
    3. Solve for the optimal projection matrix D using ridge regression.
    4. Compute the R^2 score to evaluate the fit quality.

    Ridge-regression formula:
        D = Y @ Z^T @ (Z @ Z^T + gamma * I)^{-1}
    Or, equivalently, in a more computationally efficient form:
        D^T = (Z^T @ Z + gamma * I)^{-1} @ Z^T @ Y

    Args:
        model: Trained DeepKoopmanPaper model used to encode states into Koopman space.
        X_data: NumPy array of shape (N, 5) containing states [px, py, psi, v, omega],
                where N is the number of samples.
        gamma: Ridge-regression regularization parameter. Defaults to GAMMA_RIDGE from
               config.py. Larger values apply stronger regularization to reduce overfitting.
        device: Computation device, either 'cpu' or 'cuda'.

    Returns:
        D: NumPy array of shape (2, n_z), mapping the n_z-dimensional Koopman state
           to the two variables [v, omega].
        r2_score: NumPy array of shape (2,) containing the R^2 fit scores:
                  r2_score[0] is the score for speed v, and r2_score[1] is the score
                  for yaw rate omega. Values closer to 1 indicate a better fit.
    """
    # Set the model to evaluation mode, disabling training-only behavior such as dropout.
    model.eval()

    # Move the model to the specified device.
    model = model.to(device)

    # ================================================================
    # Step 1: Encode all states into Koopman latent space.
    # ================================================================
    # Convert the NumPy array to a PyTorch tensor.
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    # Map states x to Koopman-space representations z using the model encoder.
    # torch.no_grad() disables gradient tracking to save memory and computation time.
    with torch.no_grad():
        # model.encode returns Koopman states of shape (N, n_z).
        Z = model.encode(X_tensor).cpu().numpy()

    # ================================================================
    # Step 2: Extract the target variables.
    # ================================================================
    # Extract speed v and yaw rate omega from the state data.
    # State order: [px(0), py(1), psi(2), v(3), omega(4)].
    Y = X_data[:, [IDX_V, IDX_OMEGA]]  # Shape (N, 2), with columns [v, omega].

    # ================================================================
    # Step 3: Solve for projection matrix D using ridge regression.
    # ================================================================
    # Standard ridge-regression form:
    #     D = Y @ Z^T @ (Z @ Z^T + gamma * I)^{-1}
    # Direct computation is inefficient because it requires inverting an (N x N) matrix.
    #
    # Use the equivalent, more efficient form:
    #     D^T = (Z^T @ Z + gamma * I)^{-1} @ Z^T @ Y
    # This form only requires solving an (n_z x n_z) system, where n_z << N.

    # Get the data dimensions.
    # N = Z.shape[0]  # Number of samples (unused; retained for explanation).
    n_z = Z.shape[1]  # Koopman-space dimension.

    # Compute Z^T @ Z, shape (n_z, n_z), the Koopman-state autocorrelation matrix.
    ZtZ = Z.T @ Z

    # Compute Z^T @ Y, shape (n_z, 2), the cross-correlation matrix of Koopman states and targets.
    ZtY = Z.T @ Y

    # Solve (Z^T Z + gamma * I) D^T = Z^T Y.
    # np.linalg.solve is more stable and efficient than explicitly computing an inverse.
    # In A X = B form:
    #   A = ZtZ + gamma * I  (n_z x n_z)
    #   B = ZtY              (n_z x 2)
    #   X = D^T              (n_z x 2)
    D_T = np.linalg.solve(ZtZ + gamma * np.eye(n_z), ZtY)

    # Transpose to obtain D, shape (2, n_z).
    D = D_T.T

    # ================================================================
    # Step 4: Compute the R^2 goodness-of-fit scores.
    # ================================================================
    # Make predictions using the computed projection matrix D.
    Y_pred = Z @ D.T  # Shape (N, 2), predicted [v, omega].

    # Compute the sum of squared residuals (SS_res = sum((y - y_pred)^2))
    # separately for each target variable (v and omega).
    ss_res = np.sum((Y - Y_pred) ** 2, axis=0)

    # Compute the total sum of squares (SS_tot = sum((y - y_mean)^2)),
    # representing the total variance in the data.
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)

    # Compute R^2 = 1 - SS_res / SS_tot.
    # R^2 = 1 indicates a perfect fit; R^2 = 0 indicates a fit equal to the mean.
    # Add 1e-12 to avoid division by zero.
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # Print projection-matrix results.
    print(f"Projection matrix D computed: shape={D.shape}")
    print(f"  R^2 for v:     {r2[0]:.6f}")
    print(f"  R^2 for omega: {r2[1]:.6f}")
    print(f"  Mean R^2:      {r2.mean():.6f}")

    # Return the projection matrix and R^2 scores.
    return D, r2


def save_projection_matrix(D, save_dir=MODEL_DIR, filename='projection_D.npy'):
    """
    Save the projection matrix D to disk.

    The saved matrix can be loaded directly in subsequent MPC simulations,
    avoiding repeated computation.

    Args:
        D: Projection matrix to save, a NumPy array of shape (2, n_z).
        save_dir: Output directory; defaults to MODEL_DIR from config.py.
        filename: Output filename; defaults to 'projection_D.npy'.
    """
    # Create the output directory if it does not exist.
    os.makedirs(save_dir, exist_ok=True)

    # Build the full file path.
    path = os.path.join(save_dir, filename)

    # Save as .npy using NumPy's efficient binary format, which preserves the data type.
    np.save(path, D)

    # Print a confirmation message.
    print(f"Projection matrix saved to {path}")


def load_projection_matrix(save_dir=MODEL_DIR, filename='projection_D.npy'):
    """
    Load the projection matrix D from disk.

    Load a previously computed and saved projection matrix for use by the MPC controller.

    Args:
        save_dir: Directory containing the matrix; defaults to MODEL_DIR from config.py.
        filename: Matrix filename; defaults to 'projection_D.npy'.

    Returns:
        D: Loaded projection matrix, a NumPy array of shape (2, n_z).

    Example:
        D = load_projection_matrix()
        v_omega = D @ z  # Extract [v, omega] from the Koopman state.
    """
    # Build the full file path.
    path = os.path.join(save_dir, filename)

    # Load the projection matrix from the .npy file.
    D = np.load(path)

    return D
