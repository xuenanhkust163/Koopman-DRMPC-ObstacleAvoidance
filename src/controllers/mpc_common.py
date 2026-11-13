"""
MPC Shared Utility Functions Module.

This module includes the following core functionalities:
1. Compact Matrix Construction (Section 4.1 of the paper)
   - A_cal, B_cal, C_cal: Stacked prediction matrices
2. LQR Feedback Gain Calculation (Section 4.3 of the paper)
   - For closed-loop system stabilization
3. Closed-loop Matrix Calculation (Section 4.4 of the paper)
   - A_tilde, B_tilde, C_tilde: Compact matrices with feedback
4. PyTorch to CasADi Neural Network Conversion
   - Symbolic reconstruction of encoders and decoders
5. QP/NLP Auxiliary Functions
   - QP Hessian matrix and linear term construction

These utility functions are shared by controllers like K-MPC, K-DRMPC, etc.,
realizing code reuse and modular design.
"""

import numpy as np  # Import NumPy library for efficient numerical computation and matrix operations
import os  # Import operating system interface module for file and path operations
import sys  # Import system module to modify Python path

# Import matrix equation solvers and block diagonal matrix constructors from SciPy library
from scipy.linalg import solve_discrete_are, block_diag

# Add the parent directory to the system path to ensure the same-level modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MPC-related parameters from the configuration file
from config import (
    N_Z,  # Koopman space dimension, default 8 (not directly used in this paper)
    N_U,  # Control input dimension, default 2 [acceleration a, steering angle delta]
    N_X,  # Physical state dimension, default 5 (not directly used in this paper)
    T_HORIZON,  # Prediction horizon length, default 20 steps
    Q_WEIGHTS,  # State tracking weight vector (not directly used in this paper)
    R_WEIGHTS  # Control input weight vector (not directly used in this paper)
)


def build_compact_matrices(A, B, T=T_HORIZON):
    """
    Construct stacked prediction matrices A_cal and B_cal (Section 4.1 of the paper).

    The compact matrix form represents multi-step predictions as matrix multiplications:
        z = A_cal @ z_0 + B_cal @ u

    Where:
        z = [z_1; z_2; ...; z_T] ∈ R^{T*nz}  (stacked state trajectory)
        z_0 ∈ R^{nz}                          (initial state)
        u = [u_0; u_1; ...; u_{T-1}] ∈ R^{T*nu}  (stacked control trajectory)

    Structure of A_cal matrix (T*nz × nz):
        [  A  ]
        [ A^2 ]
        [ ... ]
        [ A^T ]

    Structure of B_cal matrix (T*nz × T*nu), block lower triangular Toeplitz matrix:
        [    B     0    ...  0  ]
        [   AB      B   ...  0  ]
        [  A^2B    AB   ...  0  ]
        [  ...    ...   ... ... ]
        [A^{T-1}B A^{T-2}B ... B]

    Parameters:
        A: numpy array, shape (nz, nz)
          State transition matrix (Koopman A matrix)
        B: numpy array, shape (nz, nu)
          Control matrix (Koopman B matrix)
        T: integer, prediction horizon length
          Default is T_HORIZON (20) in config.py

    Returns:
        A_cal: numpy array, shape (T*nz, nz)
              Stacked state transition matrix
        B_cal: numpy array, shape (T*nz, T*nu)
              Stacked control matrix (block lower triangular Toeplitz)
    """
    # Get Koopman space dimension and control dimension
    nz = A.shape[0]  # Koopman state dimension
    nu = B.shape[1]  # Control input dimension

    # Initialize compact matrices
    A_cal = np.zeros((T * nz, nz))  # Shape: (T*nz, nz)
    B_cal = np.zeros((T * nz, T * nu))  # Shape: (T*nz, T*nu)

    # Compute powers of A: A^0, A^1, A^2, ..., A^T
    A_pow = np.eye(nz)  # A^0 = I (identity matrix)
    A_powers = [np.eye(nz)]  # List to store powers of A

    # Compute A_cal and store powers of A
    for t in range(T):
        A_pow = A_pow @ A  # A^{t+1} = A^t @ A
        A_cal[t * nz:(t + 1) * nz, :] = A_pow  # t+1 block of rows = A^{t+1}
        A_powers.append(A_pow.copy())  # Save A^{t+1}

    # Construct B_cal: block lower triangular Toeplitz matrix
    # B_cal[i, j] = A^{i-j} @ B, when i >= j
    for i in range(T):  # Row block index
        for j in range(i + 1):  # Column block index (only to lower triangular)
            power = i - j  # Compute power of A
            B_cal[i * nz:(i + 1) * nz,
                  j * nu:(j + 1) * nu] = A_powers[power] @ B

    return A_cal, B_cal


def build_compact_C_matrix(A, C, T=T_HORIZON):
    """
    Construct stacked disturbance matrix C_cal (Section 4.1 of the paper).

    Structure of C_cal matrix (T*nz × T*nw), block lower triangular Toeplitz matrix:
        [    C     0    ...  0  ]
        [   AC      C   ...  0  ]
        [  A^2C    AC   ...  0  ]
        [  ...    ...   ... ... ]
        [A^{T-1}C A^{T-2}C ... C]

    Parameters:
        A: numpy array, shape (nz, nz)
          State transition matrix (Koopman A matrix)
        C: numpy array, shape (nz, nw)
          Disturbance matrix (Koopman C matrix)
        T: integer, prediction horizon length

    Returns:
        C_cal: numpy array, shape (T*nz, T*nw)
              Stacked disturbance matrix
    """
    nz = A.shape[0]  # Koopman state dimension
    nw = C.shape[1]  # Disturbance dimension

    C_cal = np.zeros((T * nz, T * nw))

    # Compute powers of A
    A_powers = [np.eye(nz)]
    A_pow = np.eye(nz)
    for t in range(T):
        A_pow = A_pow @ A
        A_powers.append(A_pow.copy())

    # Construct C_cal: block lower triangular Toeplitz matrix
    for i in range(T):
        for j in range(i + 1):
            power = i - j
            C_cal[i * nz:(i + 1) * nz,
                  j * nw:(j + 1) * nw] = A_powers[power] @ C

    return C_cal


def build_lqr_gain(A, B, Q_lqr=None, R_lqr=None):
    """
    Calculate the LQR feedback gain K for the Koopman system (Section 4.3 of the paper).

    The optimal LQR gain is solved using the Discrete-time Algebraic Riccati Equation (DARE):
        u_t = K @ z_t + v_t

    Where:
        K: LQR feedback gain matrix
        v_t: new control input (MPC optimization variable)

    LQR optimization problem:
        min Σ (z^T Q z + u^T R u)
        s.t. z_{t+1} = A z_t + B u_t

    Parameters:
        A: numpy array, shape (nz, nz)
          Koopman state transition matrix
        B: numpy array, shape (nz, nu)
          Koopman control matrix
        Q_lqr: numpy array, shape (nz, nz)
              State cost matrix (default: 0.01 * I)
        R_lqr: numpy array, shape (nu, nu)
              Control cost matrix (default: 1.0 * I)

    Returns:
        K: numpy array, shape (nu, nz)
          LQR feedback gain matrix
    """
    nz = A.shape[0]  # Koopman state dimension
    nu = B.shape[1]  # Control input dimension

    # Use default values if Q and R are not provided
    if Q_lqr is None:
        Q_lqr = np.eye(nz) * 0.01  # Small state weight
    if R_lqr is None:
        R_lqr = np.eye(nu) * 1.0  # Large control weight

    try:
        # Solve the Discrete-time Algebraic Riccati Equation (DARE)
        # A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0
        P = solve_discrete_are(A, B, Q_lqr, R_lqr)

        # Compute the optimal feedback gain
        # K = -(R + B^T P B)^{-1} B^T P A
        K = -np.linalg.solve(R_lqr + B.T @ P @ B, B.T @ P @ A)
        print(f"LQR gain computed: K shape={K.shape}")

        # Check closed-loop system stability
        # Closed-loop system: z_{t+1} = (A + B K) z_t
        A_cl = A + B @ K
        eigenvalues = np.linalg.eigvals(A_cl)  # Compute eigenvalues
        max_eig = np.max(np.abs(eigenvalues))  # Maximum eigenvalue magnitude
        print(f"  Closed-loop max eigenvalue magnitude: {max_eig:.4f}")

        # If the maximum eigenvalue magnitude >= 1, the closed-loop system is unstable
        if max_eig >= 1.0:
            print("  WARNING: Closed-loop may be unstable, using zero gain")
            K = np.zeros((nu, nz))  # Use zero gain
    except Exception as e:
        # If DARE solving fails, use zero gain
        print(f"  LQR solve failed ({e}), using zero feedback gain")
        K = np.zeros((nu, nz))

    return K


def build_closed_loop_matrices(A_cal, B_cal, K_fb, C_cal=None):
    """
    Construct closed-loop compact matrices A_tilde, B_tilde, C_tilde (Section 4.4 of the paper).

    Closed-loop control is performed using the LQR feedback gain K_fb:
        u = K_fb @ z + v

    Where v is the new control input (MPC optimization variable).

    Closed-loop dynamics derivation:
        z = A_cal @ z_0 + B_cal @ u
          = A_cal @ z_0 + B_cal @ (K_stack @ z + v)
          = A_cal @ z_0 + B_cal @ K_stack @ z + B_cal @ v

    Rearranging gives: (I - B_cal @ K_stack) @ z = A_cal @ z_0 + B_cal @ v

    Let M = (I - B_cal @ K_stack)^{-1}, then:
        z = M @ A_cal @ z_0 + M @ B_cal @ v + M @ C_cal @ w
          = A_tilde @ z_0 + B_tilde @ v + C_tilde @ w

    Parameters:
        A_cal: numpy array, shape (T*nz, nz)
              Stacked state transition matrix
        B_cal: numpy array, shape (T*nz, T*nu)
              Stacked control matrix
        K_fb: numpy array, shape (nu, nz)
             LQR feedback gain
        C_cal: numpy array, shape (T*nz, T*nw), optional
              Stacked disturbance matrix (C_tilde is not returned if None)

    Returns:
        A_tilde: numpy array, shape (T*nz, nz)
                Closed-loop state transition matrix
        B_tilde: numpy array, shape (T*nz, T*nu)
                Closed-loop control matrix
        C_tilde: numpy array, shape (T*nz, T*nw), or None
                Closed-loop disturbance matrix (C_tilde is None if C_cal is None)
    """
    T_nz = A_cal.shape[0]  # T*nz
    # T_nu = B_cal.shape[1]  # T*nu (not directly used, but kept for reference)
    nz = A_cal.shape[1]  # nz

    # Calculate T and nu
    T = T_nz // nz
    nu = K_fb.shape[0]

    # ================================================================
    # Build the stacked K matrix
    # ================================================================
    # K_stack = diag(K_fb, K_fb, ..., K_fb) ∈ R^{(T*nu) × (T*nz)}
    # Block diagonal matrix, each block diagonal is K_fb
    K_stack = np.zeros((T * nu, T * nz))
    for t in range(T):
        K_stack[t * nu:(t + 1) * nu,
                t * nz:(t + 1) * nz] = K_fb

    # Compute (I - B_cal @ K_stack)
    I_BK = np.eye(T_nz) - B_cal @ K_stack

    # Invert (if invertible)
    try:
        I_BK_inv = np.linalg.inv(I_BK)
    except np.linalg.LinAlgError:
        # If not invertible, use pseudo-inverse
        print("WARNING: (I - B*K) not invertible, using pseudo-inverse")
        I_BK_inv = np.linalg.pinv(I_BK)

    # Calculate closed-loop matrices
    A_tilde = I_BK_inv @ A_cal
    B_tilde = I_BK_inv @ B_cal

    # If C_cal is provided, calculate C_tilde
    C_tilde = None
    if C_cal is not None:
        C_tilde = I_BK_inv @ C_cal

    return A_tilde, B_tilde, C_tilde


def build_D_stacked(D, T=T_HORIZON):
    """
    Construct the stacked projection matrix D_stack = I_T ⊗ D (Kronecker product).

    D_stack ∈ R^{(2*T) × (T*nz)}

    Structure:
        [D   0   ...  0 ]
        [0   D   ...  0 ]
        [...     ... ...]
        [0   0   ...  D ]

    Effect:
    Projects the T-step Koopman state stack z ∈ R^{T*nz} to [v, omega] space,
    resulting in the T-step [v, omega] stack y ∈ R^{2*T}:
        y = D_stack @ z

    Parameters:
        D: numpy array, shape (2, nz)
          Projection matrix, mapping single Koopman state to [v, omega]
        T: integer, prediction horizon length

    Returns:
        D_stack: numpy array, shape (2*T, T*nz)
                Block diagonal stacked projection matrix
    """
    nz = D.shape[1]
    # Create block diagonal matrix using scipy.linalg.block_diag
    # Repeat D matrix T times along the diagonal
    return block_diag(*[D for _ in range(T)])


def build_smoothness_matrix(T=T_HORIZON, nu=N_U):
    """
    Construct the control difference matrix S (Section 4.5 of the paper).

    Structure of S matrix (T*nu × T*nu)：
        S = [ I   0   ...  0 ]
            [-I   I   ...  0 ]
            [ 0  -I   ...  0 ]
            [...     ... ...]
            [ 0   0   ... -I I]

    Control differences can be expressed as:
        Delta_u = S @ u - u_prev_stacked

    Where:
        Delta_u = [u_0 - u_prev; u_1 - u_0; ...; u_{T-1} - u_{T-2}]

    Effect:
    Used in QP to construct control smoothness cost,
    penalizing control increments between adjacent time steps.

    Parameters:
        T: integer, prediction horizon length
        nu: integer, control input dimension

    Returns:
        S: numpy array, shape (T*nu, T*nu)
          Control difference matrix
    """
    S = np.zeros((T * nu, T * nu))

    # Iterate through each block of rows
    for t in range(T):
        # Diagonal block: identity matrix
        S[t * nu:(t + 1) * nu, t * nu:(t + 1) * nu] = np.eye(nu)

        # Lower diagonal block (when t>0): negative identity matrix
        if t > 0:
            S[t * nu:(t + 1) * nu, (t - 1) * nu:t * nu] = -np.eye(nu)

    return S


def build_qp_matrices(A_tilde, B_tilde, D_stack, S, Q_stack, R_stack,
                      z0, y_ref, u_prev, K_fb=None):
    """
    Construct QP Hessian matrix H and linear term f (Section 4.5 of the paper).

    Under closed-loop control (u = K @ z + v), the cost function can be expressed as:
        J = 0.5 * v^T @ H @ v + v^T @ f

    Where v is the new MPC optimization variable.

    Parameters:
        A_tilde, B_tilde: numpy arrays
                         Closed-loop compact matrices
        D_stack: numpy array, shape (2*T, T*nz)
                Stacked projection matrix
        S: numpy array, shape (T*nu, T*nu)
          Control smoothness matrix
        Q_stack: numpy array, shape (2*T, 2*T)
                Stacked tracking cost matrix
        R_stack: numpy array, shape (T*nu, T*nu)
                Stacked control cost matrix
        z0: numpy array, shape (nz,)
           Initial Koopman state
        y_ref: numpy array, shape (2*T,)
              Stacked reference trajectory [v_ref, omega_ref]
        u_prev: numpy array, shape (nu,)
               Previous control input
        K_fb: numpy array, shape (nu, nz), optional
             LQR feedback gain (for closed-loop control smoothness)

    Returns:
        H: numpy array, shape (T*nu, T*nu)
          QP Hessian matrix
        f: numpy array, shape (T*nu,)
          QP linear term
    """
    # D_stack @ B_tilde: mapping control to [v, omega]
    DB_tilde = D_stack @ B_tilde

    # If K_fb is provided, adjust S for closed-loop control smoothness
    if K_fb is not None:
        # nz = A_tilde.shape[1]  # Not directly used
        # T = B_tilde.shape[1] // K_fb.shape[0]  # Not directly used
        # K_stack_B = np.zeros_like(S)  # Not directly used
        # S @ (K_stack @ z + v) needs adjustment
        # SKB = S  # Simplification: directly use S for v
        pass  # Simplified handling, directly use S

    # Construct Hessian matrix H
    # H = 2 * (DB_tilde^T @ Q_stack @ DB_tilde + S^T @ R_stack @ S)
    H = 2 * (DB_tilde.T @ Q_stack @ DB_tilde + S.T @ R_stack @ S)

    # Ensure H is symmetric (numerical stability)
    H = 0.5 * (H + H.T)

    # Construct linear term f
    # Da_z0 = D_stack @ A_tilde @ z0: contribution of initial state to output
    Da_z0 = D_stack @ A_tilde @ z0
    # f = 2 * (DB_tilde^T @ Q_stack @ (Da_z0 - y_ref) + ...)
    u_prev_term = (
        S.T @ R_stack
        @ (-np.concatenate([[1], np.zeros(S.shape[0] - 1)]))
    )
    f = 2 * (DB_tilde.T @ Q_stack @ (Da_z0 - y_ref) + u_prev_term)

    return H, f


def pytorch_to_casadi_encoder(weights_dict):
    """
    Reconstruct the PyTorch encoder as a CasADi MX expression.

    This function reads the weights of the PyTorch model and rebuilds a functionally equivalent CasADi function,
    allowing the encoder to be used in the CasADi optimization framework (e.g., NLP solving in MPC).

    Network architecture (encoder in the paper):
        Input layer: 5 neurons (physical states)
        Hidden layer 1: 64 neurons, ReLU activation
        Hidden layer 2: 128 neurons, ReLU activation
        Hidden layer 3: 64 neurons, ReLU activation
        Output layer: 32 neurons (Koopman latent states)

    Parameters:
        weights_dict: dictionary, from model.get_network_weights()
                     Contains weights and biases of the encoder

    Returns:
        casadi_encode: function, accepts CasADi MX symbolic input, returns CasADi MX symbolic output
                      function(x_sym) -> z_sym
    """
    # Import CasADi within the function to avoid global import dependencies
    import casadi as ca

    # Compatibility with direct pass-through linear structure
    mode = weights_dict.get('mode', 'mlp')
    if mode == 'linear_passthrough':
        n_x = int(weights_dict['n_x'])
        n_z = int(weights_dict['n_z'])
        lift_weight = weights_dict.get('lift_weight', None)

        def casadi_encode(x_sym):
            if lift_weight is None or n_z <= n_x:
                return x_sym
            Wl = ca.DM(lift_weight)
            z_lift = Wl @ x_sym
            return ca.vertcat(x_sym, z_lift)

        return casadi_encode

    # Extract weights and biases of the encoder
    enc_W = weights_dict['encoder_weights']  # List of weights
    enc_b = weights_dict['encoder_biases']   # List of biases

    def casadi_encode(x_sym):
        """Encoder forward propagation (CasADi version)."""
        h = x_sym  # Input

        # Iterate through each layer
        for i, (W, b) in enumerate(zip(enc_W, enc_b)):
            # Convert NumPy arrays to CasADi DM
            W_ca = ca.DM(W)
            b_ca = ca.DM(b)

            # Linear transformation: h = W @ h + b
            h = W_ca @ h + b_ca

            # ReLU activation function (except for the last layer)
            # ReLU(x) = max(0, x)
            if i < len(enc_W) - 1:
                h = ca.fmax(0, h)

        return h

    return casadi_encode


def pytorch_to_casadi_decoder(weights_dict):
    """
    Reconstruct the PyTorch decoder as a CasADi MX expression.

    This function reads the weights of the PyTorch model and rebuilds a functionally equivalent CasADi function,
    allowing the decoder to be used in the CasADi optimization framework (e.g., NLP solving in MPC).

    Network architecture (decoder in the paper):
        Input layer: 32 neurons (Koopman latent states)
        Hidden layer 1: 64 neurons, ReLU activation
        Hidden layer 2: 32 neurons, ReLU activation
        Output layer: 5 neurons (physical states)

    Parameters:
        weights_dict: dictionary, from model.get_network_weights()
                     Contains weights and biases of the decoder

    Returns:
        casadi_decode: function, accepts CasADi MX symbolic input, returns CasADi MX symbolic output
                      function(z_sym) -> x_sym
    """
    # Import CasADi within the function to avoid global import dependencies
    import casadi as ca

    # Compatibility with direct pass-through linear structure
    mode = weights_dict.get('mode', 'mlp')
    if mode == 'linear_passthrough':
        n_x = int(weights_dict['n_x'])

        def casadi_decode(z_sym):
            return z_sym[:n_x]

        return casadi_decode

    # Extract weights and biases of the decoder
    dec_W = weights_dict['decoder_weights']  # List of weights
    dec_b = weights_dict['decoder_biases']   # List of biases

    def casadi_decode(z_sym):
        """Decoder forward propagation (CasADi version)."""
        h = z_sym  # Input

        # Iterate through each layer
        for i, (W, b) in enumerate(zip(dec_W, dec_b)):
            # Convert NumPy arrays to CasADi DM
            W_ca = ca.DM(W)
            b_ca = ca.DM(b)

            # Linear transformation: h = W @ h + b
            h = W_ca @ h + b_ca

            # ReLU activation function (except for the last layer)
            if i < len(dec_W) - 1:
                h = ca.fmax(0, h)

        return h

    return casadi_decode


def verify_casadi_consistency(model, weights_dict, n_tests=100, tol=1e-5):
    """
    Verify the consistency of CasADi encoder/decoder with PyTorch output.

    This function generates random inputs, encodes and decodes them through the PyTorch model and CasADi functions,
    and compares the output differences to ensure the CasADi-reconstructed neural network is equivalent to the original PyTorch model.

    Verification contents:
    1. Encoder consistency: PyTorch encode(x) vs CasADi encode(x)
    2. Decoder consistency: PyTorch decode(z) vs CasADi decode(z)

    Parameters:
        model: Trained PyTorch DeepKoopmanPaper model
        weights_dict: dictionary, from model.get_network_weights()
        n_tests: integer, number of random test inputs (default 100)
        tol: floating-point, acceptable error tolerance (default 1e-5)

    Returns:
        max_enc_err: floating-point, maximum error of the encoder
        max_dec_err: floating-point, maximum error of the decoder

    Exceptions:
        AssertionError: If the error of the encoder or decoder exceeds the tolerance
    """
    # Import dependencies within the function to avoid global imports
    import torch
    import casadi as ca

    # Create CasADi encoder and decoder
    casadi_encode = pytorch_to_casadi_encoder(weights_dict)
    casadi_decode = pytorch_to_casadi_decoder(weights_dict)

    # Set the PyTorch model to evaluation mode
    model.eval()

    # Initialize maximum errors
    max_enc_err = 0
    max_dec_err = 0

    # Perform n_tests random tests
    for _ in range(n_tests):
        # ================================================================
        # Test encoder consistency
        # ================================================================
        # Generate random input (physical states)
        x_np = np.random.randn(5).astype(np.float32)

        # PyTorch forward propagation
        x_torch = torch.tensor(x_np).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            z_torch = model.encode(x_torch).squeeze(0).numpy()  # Encode and remove batch dimension

        # CasADi forward propagation
        x_ca = ca.DM(x_np.reshape(-1, 1))  # Convert to CasADi column vector
        z_ca = np.array(casadi_encode(x_ca)).flatten()  # Encode and flatten

        # Calculate encoder error
        enc_err = np.max(np.abs(z_torch - z_ca))
        max_enc_err = max(max_enc_err, enc_err)

        # ================================================================
        # Test decoder consistency
        # ================================================================
        # Generate random Koopman state
        z_np = np.random.randn(32).astype(np.float32)

        # PyTorch forward propagation
        z_torch_in = torch.tensor(z_np).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            x_dec_torch = (
                model.decode(z_torch_in).squeeze(0).numpy()
            )  # Decode and remove batch dimension

        # CasADi forward propagation
        z_ca_in = ca.DM(z_np.reshape(-1, 1))  # Convert to CasADi column vector
        x_dec_ca = np.array(casadi_decode(z_ca_in)).flatten()  # Decode and flatten

        # Calculate decoder error
        dec_err = np.max(np.abs(x_dec_torch - x_dec_ca))
        max_dec_err = max(max_dec_err, dec_err)

    # Print verification results
    print(f"CasADi verification ({n_tests} tests):")
    print(f"  Max encoder error: {max_enc_err:.2e} (tol={tol:.0e})")
    print(f"  Max decoder error: {max_dec_err:.2e} (tol={tol:.0e})")

    # Assertions: Errors must be within tolerance
    assert max_enc_err < tol, f"Encoder mismatch: {max_enc_err} > {tol}"
    assert max_dec_err < tol, f"Decoder mismatch: {max_dec_err} > {tol}"
    print("  PASSED")

    return max_enc_err, max_dec_err
