"""
Koopman MPC (K-MPC) controller without distributed robust constraints.
Uses linear Koopman dynamics in the latent space (see Sec. 3.5 and 4 in the paper).

Core idea of K-MPC:
1. Elevate the nonlinear system to a linear space using the trained Koopman model
2. Perform prediction and optimization in the Koopman space using linear dynamics
3. Extract control-related variables [v, omega] through the projection matrix D
4. Handle obstacle avoidance through the decoder (nonlinear constraints)

Differences with K-DRMPC:
- K-MPC: Uses a deterministic linear Koopman model, ignoring disturbance uncertainties
- K-DRMPC: Utilizes Wasserstein fuzzy sets and CVaR constraints for distributional robustness
"""

import numpy as np  # Import NumPy for efficient numerical computation and multi-dimensional array operations
import casadi as ca  # Import CasADi for symbolic computation and nonlinear optimization (IPOPT solver)
import os  # Import operating system interface module for file and path operations
import sys  # Import system module to modify Python path
import time  # Import time module to calculate solving time

# Add parent directory to system path to ensure import of modules at the same level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MPC-related parameters from the configuration file
from config import (
    N_X,  # Physical state dimension, default 5 [px, py, psi, v, omega]
    N_U,  # Control input dimension, default 2 [acceleration a, steering angle delta]
    N_Z,  # Koopman space dimension, default 8
    T_HORIZON,  # Prediction horizon length, default 20 steps
    DT,  # Time step, default 0.1 seconds
    L_WHEELBASE,
    Q_WEIGHTS,  # State tracking weight vector
    R_WEIGHTS,  # Control input weight vector
    V_MIN,  # Minimum speed [m/s]
    V_MAX,  # Maximum speed [m/s]
    A_MIN,  # Minimum acceleration [m/s^2]
    A_MAX,  # Maximum acceleration [m/s^2]
    DELTA_MAX,  # Maximum steering angle [radians]
    DELTA_RATE_MAX,  # Maximum steering angle rate [radians/second]
    D_SAFE,  # Safety distance margin [meters]
    VEHICLE_RADIUS,  # Vehicle radius [meters]
    IPOPT_MAX_ITER,  # IPOPT solver maximum number of iterations
    IPOPT_PRINT_LEVEL,  # IPOPT solver print level
    Q_PSI_TRACK,
    Q_PROGRESS_TRACK,
    Q_POS_TRACK,
    POSITION_TERM_INTERVAL,
    Q_TERMINAL_HEADING,
    Q_TERMINAL_POS,
    IDX_PSI,
    IDX_V,
    IDX_OMEGA
)

# Import encoder/decoder transformation functions from the MPC common module
from controllers.mpc_common import (
    pytorch_to_casadi_encoder,  # PyTorch encoder to CasADi function
    pytorch_to_casadi_decoder  # PyTorch decoder to CasADi function
)
from controllers.tracking_costs import resolve_tracking_cost_builder
from model.projection import get_fixed_selector_matrices

# Obstacle proximity threshold: only add obstacle constraints if the vehicle is within this distance
# This reduces the complexity of the NLP, focusing only on nearby obstacles
OBSTACLE_PROXIMITY = 200.0  # Unit: meters

# Penalty weight for obstacle relaxation variables
# A larger value forces the satisfaction of obstacle constraints but may lead to infeasibility
OBSTACLE_SLACK_PENALTY = 1000.0
# Yaw angle tracking weight: enhances the sense of direction "along the reference direction"
Q_PSI = Q_PSI_TRACK
# Forward progress weight: explicitly encourages progress along the reference tangent
Q_PROGRESS = Q_PROGRESS_TRACK


class KMPCController:
    """
    Koopman MPC controller using linearly lifted dynamics.

    Core features:
    1. The cost function is constructed in the latent space through the projection matrix D
       - Extracts [v, omega] from the Koopman state z using the D matrix
       - Tracks the reference trajectory's speed and angular velocity

    2. Obstacle avoidance uses the decoder (nonlinear -> NLP)
       - The decoder maps the Koopman state z back to the physical space [px, py]
       - Computes the distance to obstacles in the physical space
       - Since the decoder is nonlinear, this makes the MPC a nonlinear programming problem

    3. Position tracking also uses the decoder
       - Computes position error using the decoder every 4 steps
       - Balances accuracy and computational complexity

    Optimization problem form:
        min_u Σ_t ||D*z_t - y_ref_t||_Q^2 + ||u_t||_R^2 + position tracking + obstacle penalty
        s.t.  z_{t+1} = A*z_t + B*u_t  (linear Koopman dynamics)
              u_min <= u_t <= u_max    (control constraints)
              ||decoded_pos - obs|| >= d_min  (obstacle constraints, softened)
    """

    @staticmethod
    def _clip_matrix_eigenvalues(A, radius=0.999):
        """Clip all eigenvalues of matrix A with magnitude greater than radius back inside the unit circle, preventing exponential divergence in the prediction horizon."""
        eigenvalues, V = np.linalg.eig(A)
        magnitudes = np.abs(eigenvalues)
        clipped = np.where(
            magnitudes > radius,
            eigenvalues / magnitudes * radius,
            eigenvalues
        )
        A_clipped = V @ np.diag(clipped) @ np.linalg.inv(V)
        return np.real(A_clipped)

    def __init__(self, koopman_model, D_matrix, norm_params,
                 cost_builder=None, cost_profile="default"):
        """
        Initialize the K-MPC controller.

        Parameters:
            koopman_model: Trained DeepKoopmanPaper model
                          Includes encoder, decoder, and Koopman matrices A, B
            D_matrix: numpy array, shape (2, n_z)
                     Projection matrix, used to extract [v, omega] from the Koopman state
            norm_params: Dictionary containing normalization parameters
                        {'px_mean', 'px_std', 'py_mean', 'py_std'}
        """
        # Controller name, used for logging and result identification
        self.name = "K-MPC"
        self.use_physical_state_dynamics = False

        # Initialize the last control input as a zero vector
        # Used for calculating control increments and smoothness constraints
        self.u_prev = np.zeros(N_U)

        # Warm start information, used to accelerate the next solve
        # Stores the previous optimization solution as an initial guess
        self._warm_start = None

        # ================================================================
        # Extract Koopman model parameters
        # ================================================================
        # Get Koopman linear dynamics matrices
        # A: State transition matrix (n_z x n_z)
        # B: Control matrix (n_z x n_u)
        # C: Disturbance matrix (n_z x n_w)
        self.A, self.B, self.C = koopman_model.get_matrices()
        # Clip eigenvalues of A matrix to prevent high-dimensional Koopman features from exponentially diverging in the prediction horizon
        self.A = self._clip_matrix_eigenvalues(self.A, radius=0.9)

        # Fixed linear selectors from the paper (used for controlling the main path)
        # D_pos: z->[px,py], E: z->[v], F: z->[omega], D_vomega: z->[v,omega]
        self.D_pos, self.E, self.F, self.D_vomega = get_fixed_selector_matrices(
            self.A.shape[0]
        )
        # Keep the externally passed D for diagnosis only, not used for core cost/constraint
        self.D_diag = D_matrix

        # Save normalization parameters for state encoding and decoding
        self.norm_params = norm_params

        # ================================================================
        # Build CasADi functions
        # ================================================================
        # Get neural network weights
        weights = koopman_model.get_network_weights()

        # Convert PyTorch encoder to CasADi function
        # This allows the encoder to be used in CasADi optimization
        self._ca_encode = pytorch_to_casadi_encoder(weights)

        # Convert PyTorch decoder to CasADi function
        # Used for obstacle avoidance and position tracking
        self._ca_decode = pytorch_to_casadi_decoder(weights)

        # Store matrices as CasADi DM (dense matrix) type
        # This allows direct use in CasADi symbolic computation
        self._A_ca = ca.DM(self.A)  # CasADi format A matrix
        self._B_ca = ca.DM(self.B)  # CasADi format B matrix
        self._D_pos_ca = ca.DM(self.D_pos)
        self._E_ca = ca.DM(self.E)
        self._F_ca = ca.DM(self.F)
        self._D_vomega_ca = ca.DM(self.D_vomega)
        D_psi = np.zeros((1, self.A.shape[0]))
        D_psi[0, IDX_PSI] = 1.0
        self._D_psi_ca = ca.DM(D_psi)
        # Replaceable cost builder: allows external injection of different stage cost definitions
        self.cost_builder = resolve_tracking_cost_builder(
            cost_builder,
            profile=cost_profile,
        )

    def _encode_state(self, x_physical):
        """
        Encode the physical state into the Koopman latent state.

        Encoding process:
        1. Normalize px, py (using the mean and standard deviation from training)
        2. Other states v, psi, omega are already normalized, no need to process
        3. Map the normalized state to the Koopman space using the neural network encoder

        Parameters:
            x_physical: numpy array, shape (5,)
                       Physical state [px, py, psi, v, omega]

        Returns:
            z: numpy array, shape (n_z,)
              Koopman latent state
        """
        # Copy physical state to avoid modifying the original data
        x_norm = x_physical.copy()

        # Normalize px (index 0)
        # Formula: px_norm = (px - mean) / std
        x_norm[0] = (
            x_norm[0] - self.norm_params['px_mean']
        ) / self.norm_params['px_std']

        # Normalize py (index 1)
        # Formula: py_norm = (py - mean) / std
        x_norm[1] = (
            x_norm[1] - self.norm_params['py_mean']
        ) / self.norm_params['py_std']

        # Convert NumPy array to CasADi column vector
        # reshape(-1, 1) ensures the shape is (5, 1)
        x_ca = ca.DM(x_norm.reshape(-1, 1))

        # Compute latent state using CasADi encoder
        # This is the forward propagation of the neural network
        z_ca = self._ca_encode(x_ca)

        # Convert CasADi DM back to NumPy array and flatten
        # Returns a one-dimensional array with shape (n_z,)
        return np.array(z_ca).flatten()

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        Solve the K-MPC optimization problem.

        Optimization problem construction process:
        1. Encode the current state to the Koopman space
        2. Construct the reference trajectory (speed and angular velocity)
        3. Create decision variables (control sequence)
        4. Propagate the state trajectory using linear Koopman dynamics
        5. Construct the cost function (tracking + smoothness + position + obstacle)
        6. Add constraints (control input + rate + obstacle)
        7. Solve the NLP problem using IPOPT
        8. Return the optimal control input

        Parameters:
            x_current: numpy array, shape (5,)
                      Current physical state [px, py, psi, v, omega]
            ref_trajectory: numpy array, shape (T, 5)
                          Reference trajectory (physical coordinates)
            obstacles: List, each element is (ox, oy, radius)
                      Obstacle position and radius (physical coordinates)
            u_prev: numpy array, shape (2,)
                   Previous control input [a, delta]
                   If None, use the internally stored u_prev

        Returns:
            u_opt: numpy array, shape (2,)
                  Optimal control input [acceleration a, steering angle delta]
            solve_info: Dictionary containing solving information
                       {'solve_time', 'status', 'method'}
        """
        # If the previous control input is provided, update the internal storage
        if u_prev is not None:
            self.u_prev = u_prev

        # Get the prediction horizon length (default 20 steps)
        T = T_HORIZON

        # Record the start time of the solving process
        t_start = time.time()

        # ================================================================
        # Step 1: Encode the current state
        # ================================================================
        # Encode the current physical state to the Koopman latent state
        z0 = self._encode_state(x_current)

        # ================================================================
        # Step 2: Construct the reference trajectory (in [v, omega] space)
        # ================================================================
        # Initialize the reference trajectory array, shape (T, 2)
        # Each row contains [v_ref, omega_ref]
        y_ref = np.zeros((T, 2))

        # Iterate over each step of the prediction horizon
        for t in range(T):
            # Get the reference state for the current time step
            # Use min to ensure it does not exceed the reference trajectory length
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            # Extract speed v (index 2) and angular velocity omega (index 4)
            y_ref[t] = [ref_t[IDX_V], ref_t[IDX_OMEGA]]

        # ================================================================
        # Step 3: Create CasADi optimization problem
        # ================================================================
        # Create Opti object for constructing the nonlinear programming (NLP) problem
        opti = ca.Opti()

        # Decision variables: control sequence over the prediction horizon
        # U's shape is (N_U, T) = (2, 20)
        # U[0, :] = acceleration sequence [a_0, a_1, ..., a_{T-1}]
        # U[1, :] = steering angle sequence [delta_0, delta_1, ..., delta_{T-1}]
        U = opti.variable(N_U, T)

        # ================================================================
        # Step 4: Propagate the state trajectory using linear Koopman dynamics
        # ================================================================
        # List of latent state trajectories
        # z_0 is fixed (encoding of the current state), no need to optimize
        Z = [ca.DM(z0)]

        # Iterate over the prediction horizon, propagate the state using linear dynamics
        # Formula: z_{t+1} = A*z_t + B*u_t
        for t in range(T):
            # Compute the next Koopman state
            # ca.mtimes indicates matrix multiplication
            z_next = ca.mtimes(self._A_ca, Z[-1]) + \
                     ca.mtimes(self._B_ca, U[:, t])
            if self.use_physical_state_dynamics:
                v_next = Z[-1][IDX_V] + DT * U[0, t]
                omega_next = v_next * ca.tan(U[1, t]) / L_WHEELBASE
                psi_next = Z[-1][IDX_PSI] + DT * omega_next
                px_next = Z[-1][0] + DT * v_next * ca.cos(psi_next) / self.norm_params['px_std']
                py_next = Z[-1][1] + DT * v_next * ca.sin(psi_next) / self.norm_params['py_std']
                z_next = ca.vertcat(
                    px_next,
                    py_next,
                    psi_next,
                    v_next,
                    omega_next,
                    z_next[N_X:],
                )
            # Add to the state trajectory list
            Z.append(z_next)

        # ================================================================
        # Step 5: Construct the cost function
        # ================================================================
        # Initialize the total cost to 0
        cost = 0

        # Create weight matrices (CasADi format)
        Q = ca.DM(Q_WEIGHTS)  # [v, omega] tracking weights
        R = ca.DM(R_WEIGHTS)  # Control input weights

        # Position tracking weight
        # Balances the importance of [v, omega] tracking and position tracking
        Q_pos = Q_POS_TRACK

        # Normalize the reference trajectory's position
        # For comparison with the decoder output (normalized space)
        ref_px_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory)-1), 0]
             - self.norm_params['px_mean'])
            / self.norm_params['px_std']
            for t in range(T)
        ])
        ref_py_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory)-1), 1]
             - self.norm_params['py_mean'])
            / self.norm_params['py_std']
            for t in range(T)
        ])
        ref_psi = np.array([
            ref_trajectory[min(t, len(ref_trajectory)-1), IDX_PSI]
            for t in range(T)
        ])
        ref_psi_terminal = float(ref_trajectory[min(T - 1, len(ref_trajectory) - 1), IDX_PSI])
        ref_px_norm_terminal = float(ref_px_norm[T - 1])
        ref_py_norm_terminal = float(ref_py_norm[T - 1])

        # Optional risk term container: some builders append per-step losses for tail terms like CVaR here
        risk_terms = []

        # Iterate over the prediction horizon, accumulating the cost of each step (computed by the replaceable cost builder)
        for t in range(T):
            y_t = ca.mtimes(self._D_vomega_ca, Z[t])
            y_ref_t = ca.DM(y_ref[t])
            cost += self.cost_builder.stage_cost(
                opti=opti,
                t=t,
                z_t=Z[t],
                u_t=U[:, t],
                u_prev=ca.DM(self.u_prev),
                u_prev_step=(U[:, t - 1] if t > 0 else None),
                y_t=y_t,
                y_ref_t=y_ref_t,
                ref_psi_t=float(ref_psi[t]),
                ref_px_norm_t=float(ref_px_norm[t]),
                ref_py_norm_t=float(ref_py_norm[t]),
                d_pos_ca=self._D_pos_ca,
                d_psi_ca=self._D_psi_ca,
                q=Q,
                r=R,
                q_psi=Q_PSI,
                q_progress=Q_PROGRESS,
                q_pos=Q_pos,
                add_position_term=(t % POSITION_TERM_INTERVAL == 0),
                risk_terms=risk_terms,
            )

        # Optional terminal cost (e.g., CVaR tail risk)
        cost += self.cost_builder.finalize_cost(
            opti=opti,
            horizon=T,
            risk_terms=risk_terms,
            z_terminal=Z[T],
            ref_psi_terminal=ref_psi_terminal,
            ref_px_norm_terminal=ref_px_norm_terminal,
            ref_py_norm_terminal=ref_py_norm_terminal,
            d_pos_ca=self._D_pos_ca,
            d_psi_ca=self._D_psi_ca,
            terminal_heading_weight=Q_TERMINAL_HEADING,
            terminal_pos_weight=Q_TERMINAL_POS,
        )

        # ================================================================
        # Step 6: Add constraints
        # ================================================================
        # 6.1 Control input constraints
        for t in range(T):
            # Acceleration constraints: A_MIN <= a_t <= A_MAX
            opti.subject_to(opti.bounded(A_MIN, U[0, t], A_MAX))
            # Steering angle constraints: -DELTA_MAX <= delta_t <= DELTA_MAX
            opti.subject_to(
                opti.bounded(-DELTA_MAX, U[1, t], DELTA_MAX)
            )

            # Steering angle rate constraints
            # Limits the change in steering angle between adjacent time steps
            if t == 0:
                # First step compared to the last control input
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - self.u_prev[1],
                    DELTA_RATE_MAX * DT
                ))
            else:
                # Other steps compared to the previous step
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - U[1, t - 1],
                    DELTA_RATE_MAX * DT
                ))

        # 6.2 Obstacle avoidance constraints (with proximity filtering and relaxation variables)
        # Only add constraints for obstacles within the proximity threshold
        nearby_obstacles = []
        px, py = x_current[0], x_current[1]

        # Iterate over all obstacles
        for obs in obstacles:
            ox, oy, r = obs
            # Compute the distance between the vehicle and the obstacle
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            # Only consider obstacles within the proximity threshold
            if dist < OBSTACLE_PROXIMITY:
                nearby_obstacles.append(obs)

        # Check steps: check obstacle constraints every 4 steps
        # This balances safety and computational complexity
        check_steps = list(range(1, T + 1, 4))

        # If there are nearby obstacles, add avoidance constraints
        if nearby_obstacles:
            # Create relaxation variables for softening obstacle constraints
            # Number of relaxation variables = Number of obstacles × Number of check steps
            n_slack = len(nearby_obstacles) * len(check_steps)
            slack = opti.variable(n_slack)

            # Relaxation variables must be non-negative
            opti.subject_to(slack >= 0)

            # Add penalty for relaxation variables to the cost function
            # A large penalty weight forces satisfaction of obstacle constraints
            cost += OBSTACLE_SLACK_PENALTY * ca.dot(slack, slack)

            # Iterate over each obstacle
            slack_idx = 0
            for obs in nearby_obstacles:
                ox, oy, r = obs
                # Compute the minimum safe distance
                # d_min = obstacle radius + vehicle radius + safety margin
                d_min = r + VEHICLE_RADIUS + D_SAFE

                # Normalize the obstacle position
                # For comparison with the decoder output (normalized space)
                ox_norm = (
                    (ox - self.norm_params['px_mean'])
                    / self.norm_params['px_std']
                )
                oy_norm = (
                    (oy - self.norm_params['py_mean'])
                    / self.norm_params['py_std']
                )

                # Iterate over each check step
                for tc in check_steps:
                    # Extract position using fixed selector (linear)
                    pos = ca.mtimes(self._D_pos_ca, Z[tc])
                    dx = pos[0] - ox_norm
                    dy = pos[1] - oy_norm

                    # Compute the physical distance squared by de-normalization
                    dist_phys_sq = (
                        (dx * self.norm_params['px_std'])**2
                        + (dy * self.norm_params['py_std'])**2
                    )

                    # Softened constraint: dist^2 >= d_min^2 - slack
                    # Relaxation variables allow slight constraint violations but are penalized
                    opti.subject_to(
                        dist_phys_sq + slack[slack_idx] >= d_min**2
                    )
                    slack_idx += 1

        # ================================================================
        # Step 7: Set the optimization objective
        # ================================================================
        # Minimize the total cost
        opti.minimize(cost)

        # ================================================================
        # Step 8: Configure IPOPT solver
        # ================================================================
        # IPOPT configuration options
        opts = {
            # Maximum number of iterations
            'ipopt.max_iter': IPOPT_MAX_ITER,
            # Print level (0=silent, 5=detailed)
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            # Do not print solving time
            'print_time': False,
            # Enable warm start
            'ipopt.warm_start_init_point': 'yes',
            # Convergence tolerance
            'ipopt.tol': 1e-4,
            # Acceptable convergence tolerance (looser)
            'ipopt.acceptable_tol': 1e-3,
            # Acceptable number of iterations
            'ipopt.acceptable_iter': 5,
        }

        # Set the solver to IPOPT (Interior Point Optimizer)
        # IPOPT is an open-source solver for large-scale nonlinear optimization problems
        opti.solver('ipopt', opts)

        # ================================================================
        # Step 9: Warm start (optional)
        # ================================================================
        # If there is a previous solution, use it as an initial guess
        # Warm starting can accelerate the solving process
        if self._warm_start is not None:
            try:
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                # If setting fails, ignore and continue
                pass

        # ================================================================
        # Step 10: Solve the optimization problem
        # ================================================================
        try:
            # Call IPOPT to solve
            sol = opti.solve()

            # Extract the optimal control input for the first step
            # MPC only applies the control for the first step, then re-optimizes in the next step
            u_opt = np.array(sol.value(U[:, 0])).flatten()

            # Solving status
            status = "optimal"

            # Save the current solution for the next warm start
            self._warm_start = {'U': sol.value(U)}

        except Exception as e:
            # If solving fails, attempt to extract a suboptimal solution
            try:
                # Extract solution from debug object
                u_opt = np.array(opti.debug.value(U[:, 0])).flatten()

                # Clip the control inputs to valid ranges
                u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)  # Acceleration
                u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)  # Steering angle

                # Mark as suboptimal solution
                status = "suboptimal"

            except Exception:
                # If unable to extract a suboptimal solution, use the previous control input
                u_opt = self.u_prev.copy()
                # Truncate error message (max 50 characters)
                status = f"failed: {str(e)[:50]}"

        # Calculate solving time
        solve_time = time.time() - t_start

        # Update the last control input
        self.u_prev = u_opt.copy()

        # Return the optimal control input and solving information
        return u_opt, {
            'solve_time': solve_time,  # Solving time (seconds)
            'status': status,  # Solving status
            'method': self.name,  # Controller name
        }

    def reset(self):
        """
        Reset the controller state.

        Called at the start of each new simulation, clears the last control input and warm start information.
        """
        # Reset the last control input to a zero vector
        self.u_prev = np.zeros(N_U)
        # Clear warm start information
        self._warm_start = None
