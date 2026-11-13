"""
Deep Koopman network architecture precisely as in the paper.

Implements the encoder-decoder structure described in Section 3.3 of the paper,
as well as the linear dynamics matrices A and B.

Network architecture (MLP version, Table 3 of the paper):
    Encoder: x ∈ R^5 -> z ∈ R^32
        MLP with 3 hidden layers [64, 128, 64]
        Lifts the physical state x to the Koopman space z
    Decoder: z ∈ R^32 -> x ∈ R^5
        MLP with 2 hidden layers [64, 32]
        Reconstructs the physical state x from the Koopman space z
    Linear Dynamics: z_{t+1} = A @ z_t + B @ u_t

Koopman Operator Theory:
    The Koopman operator is an infinite-dimensional linear operator that can
    precisely describe the evolution of nonlinear systems. Deep Koopman uses
    neural networks to learn a finite-dimensional approximation that lifts
    the nonlinear system to a high-dimensional linear space (Koopman space),
    where linear models are used for prediction and control.
"""

import torch  # PyTorch deep learning framework
import torch.nn as nn  # PyTorch neural network module
import numpy as np  # NumPy library for numerical computations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODER_LAYERS, DECODER_LAYERS, DT, PX_STD, PY_STD


class DeepKoopmanPaper(nn.Module):
    """
    Deep Koopman model matching the paper specifications (Table 3).

    This model implements the core idea of the Koopman operator theory:
    lifting the state of a nonlinear system to a high-dimensional linear space
    via an encoder, evolving in this space using linear dynamics, and then
    mapping back to the original physical space through a decoder.

    Components:
        - Encoder phi: R^5 -> R^32
          MLP with 3 hidden layers [64, 128, 64]
          Lifts the physical state x to the Koopman space z

        - Linear dynamics matrices:
          A ∈ R^{32x32}: State evolution matrix (autonomous dynamics)
          B ∈ R^{32x2}:  Control input matrix (effect of control on the system)

        - Decoder psi: R^32 -> R^5
          MLP with 2 hidden layers [64, 32]
          Reconstructs the physical state x from the Koopman space z

    Forward process:
        1. Encoding: z_t = phi(x_t)
        2. Linear evolution: z_{t+1} = A @ z_t + B @ u_t
        3. Decoding: x_{t+1} = psi(z_{t+1})
    """

    def __init__(self, n_x=5, n_u=2, n_z=32, n_w=5):
        """
        Initialize the Deep Koopman network.

        Parameters:
            n_x: int, dimension of the physical state, default 5 [px, py, psi, v, omega]
            n_u: int, dimension of the control input, default 2 [a, delta]
            n_z: int, dimension of the Koopman space, default 32
            n_w: int, dimension of the disturbance, default 5 (same as state dimension)
        """
        super().__init__()  # Call the parent class nn.Module's initializer
        # Save the dimension parameters for later use
        self.n_x = n_x  # Physical state dimension
        self.n_u = n_u  # Control input dimension
        self.n_z = n_z  # Koopman space dimension
        self.n_w = n_w  # Disturbance dimension

        # MLP Encoder and Decoder (according to Table 3 in the paper)
        self.encoder = self._build_mlp(ENCODER_LAYERS)
        self.decoder = self._build_mlp(DECODER_LAYERS)

        # ============================================================
        # Linear dynamics matrices (Section 3.3.3)
        # ============================================================
        # Linear evolution equation in the Koopman space: z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        # Use nn.Parameter to make the matrices trainable parameters
        self.A = nn.Parameter(torch.empty(n_z, n_z))  # State evolution matrix 32x32
        self.B = nn.Parameter(torch.empty(n_z, n_u))  # Control input matrix 32x2
        self.C = nn.Parameter(torch.empty(n_z, n_w))  # Disturbance matrix 32x5

        # Initialize network weights and dynamics matrices
        self._init_weights()

    def _build_mlp(self, layer_dims):
        """
        Build an MLP network.

        Parameters:
            layer_dims: list, number of neurons in each layer
        Returns:
            nn.Sequential: The constructed MLP network
        """
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Initialize network weights and dynamics matrices.

        The weight initialization strategy is crucial for training stability and convergence speed:
        1. Encoder/Decoder: Xavier initialization to maintain variance
        2. Matrix A: Initialized as an identity matrix + small noise to ensure initial stability
        3. Matrix B: Xavier initialization
        """
        # MLP Encoder/Decoder Xavier Initialization
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Physically initialize matrix A
        # The first 5 dimensions of z are ordered as [px, py, psi, v, omega] (consistent with state vector x)
        with torch.no_grad():
            self.A.data = torch.eye(self.n_z)
            self.A.data[0, 0] = 1.0   # px -> px
            self.A.data[1, 1] = 1.0   # py -> py
            self.A.data[2, 2] = 1.0   # psi -> psi
            self.A.data[2, 4] = 0.1   # omega -> psi (dt=0.1)
            self.A.data[3, 3] = 1.0   # v -> v
            self.A.data[4, 4] = 1.0   # omega -> omega
            noise = torch.randn_like(self.A.data) * 0.01
            noise[0, 0] = 0
            noise[1, 1] = 0
            noise[2, 2] = 0
            noise[2, 4] = 0
            noise[3, 3] = 0
            noise[4, 4] = 0
            self.A.data += noise

        # Physically initialize matrix B
        # The first 5 dimensions of z are ordered as [px, py, psi, v, omega]
        # v_{t+1} = v_t + dt * a_t,  omega_{t+1} ≈ omega_t + (v/L)*dt * delta_t
        with torch.no_grad():
            self.B.data.zero_()
            self.B.data[3, 0] = 0.1   # a -> v (dt=0.1), v is at index 3 in z
            self.B.data[4, 1] = 0.15  # delta -> omega, omega is at index 4 in z

        # Xavier initialization for matrix C
        nn.init.xavier_uniform_(self.C)  # Disturbance matrix

    def encode(self, x):
        """
        Encoder: Lift the physical state to the Koopman space.

        Uses a mixed passthrough structure: z = [x, phi(x)]
        The first n_x dimensions are forced to be the physical state x (to ensure position/orientation information is not lost),
        and the last n_z-n_x dimensions are learned by the MLP as nonlinear features (e.g., cos(psi), sin(psi), etc.).

        Parameters:
            x: Tensor of shape (batch, n_x), the physical state

        Returns:
            z: Tensor of shape (batch, n_z), the Koopman space state
        """
        z = self.encoder(x)
        # Force the first n_x dimensions to be the physical state x, preventing the encoder from distorting position/orientation information
        z[:, :self.n_x] = x
        return z

    def decode(self, z):
        """
        Decoder: Reconstruct the physical state from the Koopman space.

        Since the encoder has forced the first n_x dimensions to be x, the decoder can directly take the first n_x dimensions,
        achieving perfect reconstruction and avoiding additional distortion from the decoder.

        Parameters:
            z: Tensor of shape (batch, n_z), the Koopman space state

        Returns:
            x_hat: Tensor of shape (batch, n_x), the reconstructed physical state
        """
        return z[:, :self.n_x]

    def linear_step(self, z, u, w=None):
        """
        Linear dynamics evolution in the Koopman space.

        Implements the linear state transition equation:
            z_{t+1} = A @ z_t + B @ u_t + C @ w_t

        This is the core of the Koopman theory: using a linear model in a high-dimensional space
        to describe the evolution of a nonlinear system. Matrix A captures the autonomous dynamics,
        B captures the effect of the control input, and C captures the effect of the disturbance.

        Parameters:
            z: Tensor of shape (batch, n_z), the current Koopman space state
            u: Tensor of shape (batch, n_u), the control input
            w: Tensor of shape (batch, n_w), the disturbance input (optional, default None)
        Returns:
            z_next: Tensor of shape (batch, n_z), the predicted next Koopman space state
        """
        # Compute z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        # Note: here we use z @ A.T instead of A @ z.T, because the batch is the first dimension
        # Mathematically equivalent to computing A @ z_i for each sample
        z_next = z @ self.A.T + u @ self.B.T
        if w is not None:
            z_next = z_next + w @ self.C.T
        return z_next

    def forward(self, x, u, x_next=None):
        """
        Complete forward pass, returning all intermediate variables needed for loss computation.

        This function performs the complete forward process of the Koopman model:
        1. Encode the current state
        2. Reconstruct the current state (for reconstruction loss)
        3. Predict using linear dynamics (for linear dynamics loss)
        4. Decode the prediction results (for multi-step prediction loss)

        Parameters:
            x: Tensor of shape (batch, n_x), the current state
            u: Tensor of shape (batch, n_u), the control input
            x_next: Tensor of shape (batch, n_x), the true next state (optional)
                   If provided, used to compute the linear dynamics loss
        Returns:
            result: Dictionary containing the following keys:
                'z': Encoded current state (batch, n_z)
                'z_next_linear': Next Koopman state predicted by linear model (batch, n_z)
                'z_next_true': Encoded true next state (batch, n_z), if x_next is provided
                'x_recon': Decoded reconstruction of the current state (batch, n_x)
                'x_next_recon': Decoded reconstruction of the true next state (batch, n_x), if x_next is provided
                'x_next_pred': Prediction decoded from linear dynamics (batch, n_x)
        """
        # Step 1: Encode the current state x_t -> z_t
        z = self.encode(x)

        # Step 2: Reconstruct the current state z_t -> x_t_recon
        # Used to compute reconstruction loss L_recon = ||x - psi(phi(x))||^2
        x_recon = self.decode(z)

        # Step 3: Predict the next state in the Koopman space using linear dynamics
        # z_{t+1} = A @ z_t + B @ u_t
        z_next_linear = self.linear_step(z, u)

        # Step 4: Decode the linear prediction result to the physical space
        # x_{t+1}_pred = psi(z_{t+1})
        x_next_pred = self.decode(z_next_linear)

        # Build the result dictionary
        result = {
            'z': z,                      # Current state in the Koopman space
            'z_next_linear': z_next_linear,  # Next state in the Koopman space predicted by linear model
            'x_recon': x_recon,          # Current state reconstruction
            'x_next_pred': x_next_pred,  # Next state prediction
        }

        # If the true next state is provided, encode it for the linear dynamics loss
        if x_next is not None:
            # Encode the true next state: x_{t+1} -> z_{t+1}_true
            z_next_true = self.encode(x_next)
            # Decode the true next state: z_{t+1}_true -> x_{t+1}_recon
            x_next_recon = self.decode(z_next_true)
            # Add to the result dictionary
            result['z_next_true'] = z_next_true
            result['x_next_recon'] = x_next_recon

        return result

    def multi_step_predict(self, x0, u_seq, w_seq=None):
        """
        Multi-step prediction in the Koopman space.

        This function performs K-step ahead prediction using the linear dynamics model,
        to compute the multi-step prediction loss L_pred, helping the model to learn
        the long-term dynamic evolution characteristics, not just single-step predictions.

        Prediction process (without disturbance):
            z_0 = phi(x_0)
            z_1 = A @ z_0 + B @ u_0 (+ C @ w_0)
            z_2 = A @ z_1 + B @ u_1 (+ C @ w_1)
            ...
            z_K = A @ z_{K-1} + B @ u_{K-1} (+ C @ w_{K-1})

            x_k = psi(z_k), for k = 1, 2, ..., K

        Parameters:
            x0: Tensor of shape (batch, n_x), the initial state
            u_seq: Tensor of shape (batch, K, n_u), the control sequence
                  K is the prediction horizon (number of steps)
            w_seq: Tensor of shape (batch, K, n_w), the disturbance sequence (optional)
        Returns:
            x_preds: Tensor of shape (batch, K, n_x), the predicted state sequence for t=1..K
            z_preds: Tensor of shape (batch, K, n_z), the predicted Koopman state sequence for t=1..K
        """
        K = u_seq.shape[1]  # Get the prediction horizon (number of steps)
        has_w = w_seq is not None

        # Encode the initial state: x_0 -> z_0
        z = self.encode(x0)  # Shape: (batch, n_z)

        # Initialize the prediction result lists
        x_preds = []  # Store predictions in the physical space
        z_preds = []  # Store predictions in the Koopman space

        # Perform K-step predictions step by step
        for k in range(K):
            # Extract the control input for the k-th step
            u_k = u_seq[:, k, :]  # Shape: (batch, n_u)
            # Extract the disturbance input for the k-th step (if any)
            w_k = w_seq[:, k, :] if has_w else None

            # Linear dynamics evolution: z_{k+1} = A @ z_k + B @ u_k + C @ w_k
            z = self.linear_step(z, u_k, w_k)
            # Decode to the physical space: x_{k+1} = psi(z_{k+1})
            x_pred = self.decode(z)

            # Store the prediction results
            z_preds.append(z)
            x_preds.append(x_pred)

        # Convert the lists to tensors
        # Stack along dim=1, resulting in shapes (batch, K, n_x) and (batch, K, n_z)
        x_preds = torch.stack(x_preds, dim=1)
        z_preds = torch.stack(z_preds, dim=1)

        return x_preds, z_preds

    def get_matrices(self):
        """
        Extract A, B, C matrices as NumPy arrays.

        This function is used to export the trained dynamics matrices,
        for use in the MPC controller.

        Returns:
            A: numpy array, shape(n_z, n_z), state evolution matrix
            B: numpy array, shape(n_z, n_u), control input matrix
            C: numpy array, shape(n_z, n_w), disturbance matrix
        """
        # Detach from the computation graph, then move to CPU and convert to NumPy
        A = self.A.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        C = self.C.detach().cpu().numpy()
        return A, B, C

    def get_network_weights(self):
        """
        Extract all encoder/decoder weights for CasADi reconstruction.

        This function exports the weights of the PyTorch network as NumPy arrays,
        to reconstruct the same network structure in CasADi, for
        nonlinear optimization in the NMPC controller.

        Returns:
            dict: Dictionary containing the following keys:
                'encoder_weights': List of encoder weights, each element is a weight matrix of a linear layer
                'encoder_biases': List of encoder biases
                'decoder_weights': List of decoder weights
                'decoder_biases': List of decoder biases
        """
        enc_W = []
        enc_b = []
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                enc_W.append(layer.weight.detach().cpu().numpy())
                enc_b.append(layer.bias.detach().cpu().numpy())
        dec_W = []
        dec_b = []
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                dec_W.append(layer.weight.detach().cpu().numpy())
                dec_b.append(layer.bias.detach().cpu().numpy())
        return {
            'mode': 'mlp',
            'encoder_weights': enc_W,
            'encoder_biases': enc_b,
            'decoder_weights': dec_W,
            'decoder_biases': dec_b,
        }


def koopman_loss(model, x_windows, u_windows, w_windows=None,
                 lambda_recon=1.0, lambda_linear=1.0, lambda_pred=0.5,
                 lambda_physics=0.0, lambda_reg_high_dim=0.0,
                 lambda_spectral=5.0):
    """
    Compute the three-part loss function from the paper (Section 3.3.5) + physical consistency loss.

    The training of Deep Koopman uses a weighted sum of four loss terms:
    1. Reconstruction loss (L_recon): Ensures the encoder-decoder can accurately reconstruct the original state
    2. Linear dynamics loss (L_linear): Ensures the accuracy of linear evolution in the Koopman space
    3. Multi-step prediction loss (L_pred): Improves long-term prediction capability
    4. Physical consistency loss (L_physics): Forces the A matrix to learn the correct position update

    Total loss: L_total = λ_recon * L_recon + λ_linear * L_linear
                     + λ_pred * L_pred + λ_physics * L_physics

    Parameters:
        model: DeepKoopmanPaper instance, the Koopman model in training
        x_windows: Tensor of shape (batch, K+1, n_x), state sequence
                  [x_0, x_1, ..., x_K], containing K+1 continuous states
        u_windows: Tensor of shape (batch, K, n_u), control sequence
                  [u_0, u_1, ..., u_{K-1}], containing K continuous controls
        w_windows: Tensor of shape (batch, K, n_w), disturbance sequence (optional)
                  [w_0, w_1, ..., w_{K-1}], containing K continuous disturbances
        lambda_recon: float, weight for the reconstruction loss, default 1.0
        lambda_linear: float, weight for the linear dynamics loss, default 1.0
        lambda_pred: float, weight for the multi-step prediction loss, default 0.5
        lambda_physics: float, weight for the physical consistency loss, default 0.0
        lambda_reg_high_dim: float, weight for the high-dimensional regularization, default 0.0

    Returns:
        total_loss: tensor, the total loss value
        loss_dict: Dictionary containing the values of each loss term:
            - 'total': Total loss
            - 'recon': Reconstruction loss
            - 'linear': Linear dynamics loss
            - 'pred': Multi-step prediction loss
            - 'physics': Physical consistency loss
    """
    # Get the batch size and prediction horizon (number of steps).
    # batch = x_windows.shape[0]  # Unused, retained for reference.
    # K = u_windows.shape[1]      # Unused, retained for reference.

    # Extract the initial state, next state, and first control input.
    x_0 = x_windows[:, 0, :]       # Shape (batch, n_x), initial state x_t
    x_1 = x_windows[:, 1, :]       # Shape (batch, n_x), next state x_{t+1}
    u_0 = u_windows[:, 0, :]       # Shape (batch, n_u), first control u_t
    w_0 = w_windows[:, 0, :] if w_windows is not None else None

    # ================================================================
    # 1. Reconstruction loss (L_recon)
    # ================================================================
    # Formula: L_recon = (1/M) * Σ(||x_t - psi(phi(x_t))||^2 +
    #                            ||x_{t+1} - psi(phi(x_{t+1}))||^2)
    #
    # The reconstruction loss ensures that the encoder-decoder accurately reconstructs
    # input states, preserving the important information from the original state space.

    # Encode the current state: z_0 = phi(x_0).
    z_0 = model.encode(x_0)
    # Reconstruct the current state: x_0_recon = psi(z_0).
    x_0_recon = model.decode(z_0)

    # Encode the next state: z_1 = phi(x_1).
    z_1 = model.encode(x_1)
    # Reconstruct the next state: x_1_recon = psi(z_1).
    x_1_recon = model.decode(z_1)

    # Compute the reconstruction loss using MSE for both x_t and x_{t+1}.
    loss_recon = (
        nn.functional.mse_loss(x_0_recon, x_0) +  # ||x_0 - psi(phi(x_0))||^2
        nn.functional.mse_loss(x_1_recon, x_1)    # ||x_1 - psi(phi(x_1))||^2
    )

    # ================================================================
    # 2. Linear dynamics loss (L_linear)
    # ================================================================
    # Formula: L_linear = (1/M) * Σ(||phi(x_{t+1}) - A*phi(x_t) - B*u_t||^2)
    #
    # The linear dynamics loss enforces accurate linear evolution in Koopman space:
    # z_{t+1} ≈ A @ z_t + B @ u_t.

    # Predict the next Koopman state with the linear model: z_1_pred = A @ z_0 + B @ u_0 (+ C @ w_0).
    z_1_pred = model.linear_step(z_0, u_0, w_0)
    # Compute the error between the linear prediction and the encoded target state.
    loss_linear = nn.functional.mse_loss(z_1_pred, z_1)

    # ================================================================
    # 2b. Physical consistency loss (L_physics)
    # ================================================================
    # Encourage the A matrix to learn the correct position updates:
    #   px_next = px + dt * v * cos(psi)
    #   py_next = py + dt * v * sin(psi)
    # Compute this in the original coordinate space (meters), keeping its scale
    # comparable to the linear loss.
    if lambda_physics > 0.0:
        # Decode the linear prediction into physical state space.
        x_next_pred = model.decode(z_1_pred)

        # Predicted position changes, de-normalized to the original space in meters.
        px_diff_pred = (x_next_pred[:, 0] - x_0[:, 0]) * PX_STD
        py_diff_pred = (x_next_pred[:, 1] - x_0[:, 1]) * PY_STD

        # Physical integration in the original space: dt * v * cos(psi).
        px_diff_phys = DT * x_0[:, 3] * torch.cos(x_0[:, 2])
        py_diff_phys = DT * x_0[:, 3] * torch.sin(x_0[:, 2])

        loss_physics = torch.mean(
            (px_diff_pred - px_diff_phys) ** 2 +
            (py_diff_pred - py_diff_phys) ** 2
        )
    else:
        loss_physics = torch.tensor(0.0, device=x_0.device)

    # ================================================================
    # 3. Multi-step prediction loss (L_pred)
    # ================================================================
    # Formula: L_pred = (1/MK) * Σ_k(||x_{t+k} - psi(A^k*phi(x_t) + Σ...))||^2)
    #
    # The multi-step prediction loss encourages accurate long-term predictions,
    # not just one-step predictions, improving model prediction and MPC control performance.

    # Extract the target state sequence: x_1, x_2, ..., x_K.
    x_targets = x_windows[:, 1:, :]   # Shape (batch, K, n_x), target states at t=1..K
    # Control sequence: u_0, u_1, ..., u_{K-1}.
    u_seq = u_windows                  # Shape (batch, K, n_u)
    # Optional disturbance sequence: w_0, w_1, ..., w_{K-1}.
    w_seq = w_windows                  # Shape (batch, K, n_w), or None

    # Predict K steps from x_0 using the control sequence u_seq.
    x_preds, _ = model.multi_step_predict(x_0, u_seq, w_seq)
    # x_preds has shape (batch, K, n_x), with predicted states at t=1..K.

    # Compute the multi-step prediction loss (MSE across all K steps).
    K = x_preds.shape[1]
    # Exponentially decaying weights give greater importance to near-term accuracy.
    weights = torch.exp(-0.02 * torch.arange(K, device=x_0.device, dtype=torch.float32))
    # Compute the per-step MSE, averaged over the batch and state dimensions.
    per_step_mse = torch.mean((x_preds - x_targets) ** 2, dim=(0, 2))
    loss_pred = torch.sum(weights * per_step_mse) / torch.sum(weights)

    # ================================================================
    # 4. Total loss
    # ================================================================
    # Weighted sum: L_total = λ_recon * L_recon + λ_linear * L_linear
    #                     + λ_pred * L_pred + λ_physics * L_physics
    total_loss = (lambda_recon * loss_recon +
                  lambda_linear * loss_linear +
                  lambda_pred * loss_pred +
                  lambda_physics * loss_physics)

    # ================================================================
    # 5. High-dimensional regularization loss (L_reg)
    # ================================================================
    # Penalize high-dimensional columns (5+) in the physical rows of A (psi/v/omega, indices 2-4).
    # This prevents high-dimensional features from dominating v/psi predictions.
    if lambda_reg_high_dim > 0.0:
        loss_reg = torch.sum(model.A[2:5, 5:] ** 2)
    else:
        loss_reg = torch.tensor(0.0, device=x_0.device)

    # ================================================================
    # 5b. Spectral-radius penalty (L_spectral)
    # ================================================================
    # Enforce spectral_radius(A) <= 1.0 to prevent divergent high-dimensional features
    # from causing ill-conditioned MPC optimization.
    if lambda_spectral > 0.0:
        eigenvalues = torch.linalg.eigvals(model.A)
        spectral_radius = torch.max(torch.abs(eigenvalues))
        loss_spectral = lambda_spectral * torch.relu(spectral_radius - 1.0) ** 2
    else:
        loss_spectral = torch.tensor(0.0, device=x_0.device)

    # ================================================================
    # 6. Total loss
    # ================================================================
    # Weighted sum: L_total = λ_recon * L_recon + λ_linear * L_linear
    #                     + λ_pred * L_pred + λ_physics * L_physics
    #                     + λ_reg_high_dim * L_reg + L_spectral
    total_loss = (lambda_recon * loss_recon +
                  lambda_linear * loss_linear +
                  lambda_pred * loss_pred +
                  lambda_physics * loss_physics +
                  lambda_reg_high_dim * loss_reg +
                  loss_spectral)

    # Build the loss dictionary for logging.
    loss_dict = {
        'total': total_loss.item(),     # Total loss
        'recon': loss_recon.item(),     # Reconstruction loss
        'linear': loss_linear.item(),   # Linear dynamics loss
        'pred': loss_pred.item(),       # Multi-step prediction loss
        'physics': loss_physics.item(), # Physical consistency loss
        'reg': loss_reg.item(),         # High-dimensional regularization loss
        'spectral': loss_spectral.item(), # Spectral-radius penalty
    }

    return total_loss, loss_dict
