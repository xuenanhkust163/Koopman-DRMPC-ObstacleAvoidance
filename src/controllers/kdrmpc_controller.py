"""
Distributionally Robust MPC (K-DRMPC) controller based on Koopman.
This is the main contribution of the paper (Section 5).

Extension on top of K-MPC:
- Modeling disturbance uncertainty using Wasserstein fuzzy sets
- CVaR-based safety constraints (Equations 4.10-4.11)
- Manageable convex reformulation via Lagrangian duality

Core innovations:
1. Distributional robustness: does not assume a specific distribution of disturbances, but considers a set of possible distributions
2. Wasserstein fuzzy sets: probability distribution sets centered around empirical distributions
3. CVaR constraints: Conditional Value-at-Risk, ensuring safety in the worst case
4. Convex reformulation: transforming infinite-dimensional optimization problems into finite-dimensional convex optimization problems
"""

import numpy as np  # Import NumPy library for efficient numerical computation and multi-dimensional array operations
import gc              # Garbage collection to prevent segfaults due to long-term memory accumulation in CasADi
import casadi as ca  # Import CasADi library for symbolic computation and nonlinear optimization (IPOPT solver)
import os  # Import operating system interface module for file and path operations
import sys  # Import system module to modify Python path
import time  # Import time module to calculate solving time

# Add parent directory to system path to ensure import of modules at the same level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MPC and distributionally robust optimization related parameters from config file
from config import (
    N_X,  # Physical state dimension, default 5 [px, py, psi, v, omega]
    N_U,  # Control input dimension, default 2 [acceleration a, steering angle delta]
    N_Z,  # Koopman space dimension, default 8
    N_W,  # Disturbance dimension, used for sampling historical disturbances
    T_HORIZON,  # Prediction horizon length, default 20 steps
    DT,  # Time step, default 0.1 seconds
    L_WHEELBASE,  # Vehicle wheelbase [meters], used for physical omega calculation
    Q_WEIGHTS,  # State tracking weight vector
    R_WEIGHTS,  # Control input weight vector
    V_MIN,  # Minimum speed [meters/second]
    V_MAX,  # Maximum speed [meters/second]
    A_MIN,  # Minimum acceleration [meters/second^2]
    A_MAX,  # Maximum acceleration [meters/second^2]
    DELTA_MAX,  # Maximum steering angle [radians]
    DELTA_RATE_MAX,  # Maximum steering angle rate [radians/second]
    D_SAFE,  # Safety distance margin [meters]
    TRACK_HALF_WIDTH,
    TRACK_BOUNDARY_SLACK_PENALTY,
    VEHICLE_RADIUS,  # Vehicle radius [meters]
    N_DISTURBANCE_SAMPLES,  # Number of disturbance samples, default 100
    THETA_WASSERSTEIN,  # Wasserstein ball radius, default 0.1
    EPSILON_CVAR,  # CVaR risk level, default 0.05 (95% confidence)
    IPOPT_MAX_ITER,  # IPOPT solver maximum number of iterations
    IPOPT_PRINT_LEVEL,  # IPOPT solver print level
    Q_PSI_TRACK,
    Q_PROGRESS_TRACK,
    Q_POS_TRACK,
    POSITION_TERM_INTERVAL,
    R_ABS_A,
    R_ABS_DELTA,
    Q_TERMINAL_HEADING,
    Q_TERMINAL_POS,
    IDX_PSI,
    IDX_V,
    IDX_OMEGA
)

# Import encoder/decoder conversion functions from MPC common module
from controllers.mpc_common import (
    pytorch_to_casadi_encoder,  # PyTorch encoder to CasADi function
    pytorch_to_casadi_decoder  # PyTorch decoder to CasADi function
)
from controllers.tracking_costs import (
    MinSpeedRule,
    resolve_tracking_cost_builder,
)
from model.projection import get_fixed_selector_matrices

# Obstacle proximity threshold: only add obstacle constraints if the vehicle is within this distance
OBSTACLE_PROXIMITY = 50.0  # Unit: meters (SerpentineTrack compact, 50m enough to cover nearby obstacles)
OBSTACLE_STRATEGY_CHOICES = ("robust", "non-robust")
DEFAULT_OBSTACLE_STRATEGY = "robust"

# Maximum number of disturbance samples used in optimization (to ensure tractability)
# If the number of samples is too large, it makes the optimization problem too complex, so subsampling is performed
MAX_OPT_SAMPLES = 10

# Penalty weight for slack variables in obstacle avoidance
# A larger value forces the satisfaction of obstacle constraints, but may make the problem unsolvable
OBSTACLE_SLACK_PENALTY = 1000.0

# Soft minimum speed constraint parameters: avoid optimization falling into "stop solution"
MIN_SPEED_FLOOR = V_MIN         # Consistent with config, avoid hard coding
MIN_SPEED_REF_RATIO = 0.25      # Reference speed lower limit ratio
MIN_SPEED_SLACK_PENALTY = 1200.0  # Speed lower limit slack penalty (soft tuning)
# Yaw angle tracking weight: enhance the sense of direction "moving forward along the reference direction"
Q_PSI = Q_PSI_TRACK
# Forward progress weight: explicitly encourage progress along the reference tangent
Q_PROGRESS = Q_PROGRESS_TRACK
# Absolute control quantity penalty (soft constraint): suppress long-term full brake and full steering
R_ABS_WEIGHTS = np.diag([R_ABS_A, R_ABS_DELTA])


class KDRMPCController:
    """
    Distributionally robust Koopman MPC controller with CVaR constraints.

    Core features:
    1. Linear Koopman dynamics: z_{t+1} = A*z + B*u
       - A: State transition matrix (learned)
       - B: Control matrix (learned)

    2. Wasserstein fuzzy sets centered around empirical disturbance distributions
       - Fuzzy set: P ∈ {Q : W(P, P_N) <= theta}
       - W: Wasserstein distance
       - P_N: Empirical distribution (constructed from N samples)
       - theta: Fuzzy set radius (controls robustness level)

    3. CVaR safety constraint reformulation (Theorem, Section 5.5)
       - CVaR_epsilon[l(z, w)] <= 0
       - Transforms probabilistic constraints into deterministic constraints
       - Introduces auxiliary variables Lambda and S

    Optimization problem form (paper Equation 20):
        min_u Σ_t ||D*z_t - y_ref_t||_Q^2 + ||u_t||_R^2
        s.t.  z_{t+1} = A*z_t + B*u_t  (nominal dynamics)
              u_min <= u_t <= u_max
              CVaR constraint: lambda*theta + (1/(epsilon*N))*Σs_i <= slack
                        s_i >= l_nom + lambda*||w_i||
    """

    @staticmethod
    def _clip_matrix_eigenvalues(A, radius=0.999):
        """Clip all eigenvalues of matrix A with magnitude greater than radius back inside the unit circle, to prevent exponential divergence in the prediction horizon."""
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
                 disturbance_samples=None, theta=THETA_WASSERSTEIN,
                 epsilon=EPSILON_CVAR, cost_builder=None,
                 cost_profile="default", obstacle_strategy=DEFAULT_OBSTACLE_STRATEGY):
        """
        Initialize K-DRMPC controller.

        Parameters:
            koopman_model: Trained DeepKoopmanPaper model
                          Includes encoder, decoder, and Koopman matrices A, B
            D_matrix: numpy array, shape (2, n_z)
                     Projection matrix, used to extract [v, omega] from Koopman state
            norm_params: Dictionary, contains normalization parameters
                        {'px_mean', 'px_std', 'py_mean', 'py_std'}
            disturbance_samples: numpy array, shape (N, 5)
                                Disturbance samples used to construct fuzzy sets
                                If None, randomly generated samples are used
            theta: Float, Wasserstein ball radius
                  Default is THETA_WASSERSTEIN (0.25) in config.py
                  A larger theta means a wider range of considered distributions, and a more conservative controller
            epsilon: Float, CVaR risk level
                    Default is EPSILON_CVAR (0.05) in config.py
                    epsilon=0.05 indicates a risk constraint at 95% confidence level
        """
        # Controller name, used for logging and result identification
        self.name = "K-DRMPC"

        # Initialize the last control input as a zero vector
        self.u_prev = np.zeros(N_U)

        # Warm start information, used to accelerate the next solve
        self._warm_start = None

        # ================================================================
        # Koopman model components
        # ================================================================
        # Get Koopman linear dynamics matrices
        # A: State transition matrix (n_z x n_z)
        # B: Control matrix (n_z x n_u)
        # C: Disturbance matrix (n_z x n_w)
        self.A, self.B, self.C = koopman_model.get_matrices()
        # Clip eigenvalues of A matrix to prevent exponential divergence of high-dimensional Koopman features in the prediction horizon
        self.A = self._clip_matrix_eigenvalues(self.A, radius=0.9)

        # Fixed linear selectors from the paper (for controlling the main path)
        self.D_pos, self.E, self.F, self.D_vomega = get_fixed_selector_matrices(
            self.A.shape[0]
        )
        # Keep the externally passed D only for diagnosis
        self.D_diag = D_matrix

        # Save normalization parameters for state encoding and decoding
        self.norm_params = norm_params
        # Cache std of px/py for normalization compensation during mixed model position updates
        self._px_std = float(norm_params['px_std'])
        self._py_std = float(norm_params['py_std'])

        # ================================================================
        # Distributionally robust optimization parameters
        # ================================================================
        # Wasserstein ball radius
        self.theta = theta

        # CVaR risk level
        self.epsilon = epsilon

        if obstacle_strategy not in OBSTACLE_STRATEGY_CHOICES:
            raise ValueError(
                f"obstacle_strategy must be one of {OBSTACLE_STRATEGY_CHOICES}, "
                f"but got: {obstacle_strategy}"
            )
        self.obstacle_strategy = obstacle_strategy

        # ================================================================
        # Disturbance sample handling (subsampling to ensure tractability)
        # ================================================================
        if disturbance_samples is not None:
            # If the number of samples exceeds the maximum, perform random subsampling
            if len(disturbance_samples) > MAX_OPT_SAMPLES:
                # Use fixed seed to ensure reproducibility
                rng = np.random.RandomState(42)
                # Randomly select MAX_OPT_SAMPLES samples (without replacement)
                idx = rng.choice(
                    len(disturbance_samples),
                    MAX_OPT_SAMPLES,
                    replace=False
                )
                self.w_samples = disturbance_samples[idx]
            else:
                # The number of samples is appropriate, use directly
                self.w_samples = disturbance_samples
        else:
            # If no samples are provided, generate random samples as default
            # Using zero-mean, standard deviation 0.05 normal distribution
            self.w_samples = np.random.randn(MAX_OPT_SAMPLES, N_W) * 0.05

        # Save the actual number of used samples
        self.N_samples = len(self.w_samples)

        # ================================================================
        # Pre-compute norms required for CVaR constraints (to improve optimization efficiency)
        # ================================================================
        # Pre-compute ||w_i||: the norm of the disturbance itself
        # This is used for calculating Wasserstein distance
        self.w_norms = np.array([
            np.linalg.norm(self.w_samples[i])
            for i in range(self.N_samples)
        ])

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

        # ================================================================
        # Build CasADi functions (encoder/decoder)
        # ================================================================
        # Get neural network weights
        weights = koopman_model.get_network_weights()

        # Convert PyTorch encoder to CasADi function
        self._ca_encode = pytorch_to_casadi_encoder(weights)

        # Convert PyTorch decoder to CasADi function
        # Used for obstacle avoidance and position tracking
        self._ca_decode = pytorch_to_casadi_decoder(weights)

    def _encode_state(self, x_physical):
        """Encode physical state to Koopman latent state (with normalization)."""
        # Copy physical state to avoid modifying the original data
        x_norm = x_physical.copy()

        # Normalize px (index 0)
        x_norm[0] = (
            x_norm[0] - self.norm_params['px_mean']
        ) / self.norm_params['px_std']

        # Normalize py (index 1)
        x_norm[1] = (
            x_norm[1] - self.norm_params['py_mean']
        ) / self.norm_params['py_std']

        # Convert NumPy array to CasADi column vector
        x_ca = ca.DM(x_norm.reshape(-1, 1))

        # Compute latent state using CasADi encoder
        z_ca = self._ca_encode(x_ca)

        # === Key fix: _ca_encode directly returns MLP output, without covering the first N_X dimensions as physical state.
        #  But in PyTorch's model.encode, z[:, :n_x] = x_norm, ensures the first 5 dimensions are normalized physical states.
        #  If not covered, in MPC, the first 5 dimensions of Z[0] are MLP outputs (possibly up to 351), leading to complete distortion of position/v/omega.
        z_ca[:N_X] = x_ca

        # Convert back to NumPy array and flatten
        return np.array(z_ca).flatten()

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        Solve K-DRMPC optimization problem (paper Equation 20).

        The optimization includes:
        1. Tracking cost in latent space through D projection
        2. Control smoothness cost
        3. Input/state constraints
        4. Distributionally robust CVaR obstacle avoidance constraints

        Parameters:
            x_current: numpy array, shape (5,)
                      Current physical state [px, py, psi, v, omega]
            ref_trajectory: numpy array, shape (T, 5)
                          Reference trajectory (physical coordinates)
            obstacles: List, each element is (ox, oy, radius)
                      Obstacle position and radius
            u_prev: numpy array, shape (2,)
                   Previous control input [a, delta]

        Returns:
            u_opt: numpy array, shape (2,)
                  Optimal control input [acceleration a, steering angle delta]
            solve_info: Dictionary, containing solving information and DR parameters
        """
        if u_prev is not None:
            self.u_prev = u_prev

        # Get prediction horizon length (default 20 steps)
        T = T_HORIZON

        # Record the start time of solving
        t_start = time.time()

        # Step 1: Encode current state to Koopman space
        z0 = self._encode_state(x_current)

        # Step 2: Construct reference trajectory (in [v, omega] space)
        y_ref = np.zeros((T, 2))
        for t in range(T):
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            y_ref[t] = [ref_t[IDX_V], ref_t[IDX_OMEGA]]  # [v, omega]

        # Pre-compute reference yaw angles (for yaw tracking and terminal constraints in the cost function)
        ref_psi = np.array([
            ref_trajectory[min(t, len(ref_trajectory)-1), IDX_PSI]
            for t in range(T)
        ])

        # Step 3: Filter nearby obstacles
        px, py = x_current[0], x_current[1]
        nearby_obstacles = []
        for obs in obstacles:
            ox, oy, r = obs
            # Calculate the distance between the vehicle and the obstacle
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            # Only consider obstacles within the proximity threshold (200 meters)
            if dist < OBSTACLE_PROXIMITY:
                nearby_obstacles.append(obs)

        # Number of obstacles and check steps
        n_obs = len(nearby_obstacles)
        # Robust CVaR constraint computation is more expensive, use sparser checkpoints; non-robust can be denser
        if self.obstacle_strategy == "robust":
            check_steps = list(range(4, T + 1, 12))
        else:
            check_steps = list(range(1, T + 1, 6))
        n_check = len(check_steps)

        opti = ca.Opti()

        # Decision variables: controls
        U = opti.variable(N_U, T)

        # Propagate latent dynamics (nominal, no disturbance in prediction)
        Z = [ca.DM(z0)]
        n_z = self.A.shape[0]
        for t in range(T):
            z_next = ca.mtimes(self._A_ca, Z[-1]) + \
                     ca.mtimes(self._B_ca, U[:, t])

            # === Mixed model correction: overwrite the first 5 dimensions with physical formulas, Koopman only provides high-dimensional features ===
            # Koopman A matrix cannot reliably predict omega (pure Koopman omega changes from -0.036 to +0.015),
            # Must use physical formulas to ensure the correct steering direction.
            v_phys = Z[-1][3] + DT * U[0, t]
            omega_phys = v_phys * ca.tan(U[1, t]) / L_WHEELBASE
            psi_next = Z[-1][2] + DT * omega_phys
            # Note: Z[0], Z[1] are in normalized space, position increments need to be divided by corresponding std
            px_next = Z[-1][0] + DT * v_phys * ca.cos(psi_next) / self._px_std
            py_next = Z[-1][1] + DT * v_phys * ca.sin(psi_next) / self._py_std

            # Overwrite the first 5 dimensions (px,py,psi,v,omega), retain high-dimensional features predicted by Koopman z[5:]
            z_fixed = ca.vertcat(px_next, py_next, psi_next, v_phys, omega_phys,
                                z_next[5:n_z])
            Z.append(z_fixed)

        # === Tracking Cost ===
        cost = 0
        Q = ca.DM(Q_WEIGHTS)
        R = ca.DM(R_WEIGHTS)
        R_abs = ca.DM(R_ABS_WEIGHTS)

        # Soft constraint slack variables for minimum speed lower bound (one for each prediction step)
        v_slack = opti.variable(T)
        opti.subject_to(v_slack >= 0)
        cost += MIN_SPEED_SLACK_PENALTY * ca.dot(v_slack, v_slack)

        # Soft constraint slack variables for track boundary (paper |d(s)|<=W/2; added slack to avoid infeasibility)
        track_slack = opti.variable(T + 1)
        opti.subject_to(track_slack >= 0)
        cost += TRACK_BOUNDARY_SLACK_PENALTY * ca.dot(track_slack, track_slack)

        # Position tracking weight (decoded position vs reference)
        Q_pos = Q_POS_TRACK

        # Normalize reference positions for decoder comparison
        ref_px_norm = np.array([(ref_trajectory[min(t, len(ref_trajectory)-1), 0]
                                 - self.norm_params['px_mean']) / self.norm_params['px_std']
                                for t in range(T)])
        ref_py_norm = np.array([(ref_trajectory[min(t, len(ref_trajectory)-1), 1]
                                 - self.norm_params['py_mean']) / self.norm_params['py_std']
                                for t in range(T)])
        # Terminal constraint also needs reference yaw
        ref_psi_terminal = float(ref_trajectory[min(T - 1, len(ref_trajectory) - 1), IDX_PSI])
        ref_px_norm_terminal = float(ref_px_norm[T - 1])
        ref_py_norm_terminal = float(ref_py_norm[T - 1])

        px_std = float(self.norm_params['px_std'])
        py_std = float(self.norm_params['py_std'])

        min_speed_rule = MinSpeedRule(
            floor_abs=MIN_SPEED_FLOOR,
            floor_ratio=MIN_SPEED_REF_RATIO,
        )

        risk_terms = []
        diag_terms = {}

        for t in range(T):
            y_t = ca.vertcat(
                ca.mtimes(self._E_ca, Z[t]),
                ca.mtimes(self._F_ca, Z[t])
            )
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
                add_abs_u_term=True,
                r_abs=R_abs,
                min_speed_rule=min_speed_rule,
                v_slack_t=v_slack[t],
                risk_terms=risk_terms,
            )
            self.cost_builder.collect_stage_diagnostics(
                diag_terms,
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
                add_abs_u_term=True,
                r_abs=R_abs,
                min_speed_rule=min_speed_rule,
                v_slack_t=v_slack[t],
                risk_terms=risk_terms,
            )

        # Optional terminal cost (e.g., CVaR tail risk)
        cost += self.cost_builder.finalize_cost(
            opti=opti,
            horizon=T,
            risk_terms=risk_terms,
            diag_terms=diag_terms,
            z_terminal=Z[T],
            ref_psi_terminal=ref_psi_terminal,
            ref_px_norm_terminal=ref_px_norm_terminal,
            ref_py_norm_terminal=ref_py_norm_terminal,
            d_pos_ca=self._D_pos_ca,
            d_psi_ca=self._D_psi_ca,
            terminal_heading_weight=Q_TERMINAL_HEADING,
            terminal_pos_weight=Q_TERMINAL_POS,
        )

        # === Input Constraints ===
        for t in range(T):
            opti.subject_to(opti.bounded(A_MIN, U[0, t], A_MAX))
            opti.subject_to(opti.bounded(-DELTA_MAX, U[1, t], DELTA_MAX))
            if t == 0:
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - self.u_prev[1],
                    DELTA_RATE_MAX * DT))
            else:
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - U[1, t - 1],
                    DELTA_RATE_MAX * DT))

        # === Track Boundary Constraints (paper Eq. |d(s)| <= W/2) ===
        for t in range(T + 1):
            ref_idx = min(t, T - 1)
            psi_ref = ref_psi[ref_idx] if t < T else ref_psi_terminal

            # Linear selector extracts position (normalized coordinates), then converts to metric deviation
            pos_t = ca.mtimes(self._D_pos_ca, Z[t])
            dx_m = (pos_t[0] - float(ref_px_norm[ref_idx])) * px_std
            dy_m = (pos_t[1] - float(ref_py_norm[ref_idx])) * py_std

            # Signed lateral deviation d(s) in the paper
            d_lat = -ca.sin(psi_ref) * dx_m + ca.cos(psi_ref) * dy_m

            # Softened bilateral constraint: |d_lat| <= TRACK_HALF_WIDTH + slack
            opti.subject_to(d_lat <= TRACK_HALF_WIDTH + track_slack[t])
            opti.subject_to(-d_lat <= TRACK_HALF_WIDTH + track_slack[t])

        obs_slack = None
        if n_obs > 0 and n_check > 0:
            n_obs_slack = n_obs * n_check
            obs_slack = opti.variable(n_obs_slack)
            opti.subject_to(obs_slack >= 0)
            cost += OBSTACLE_SLACK_PENALTY * ca.dot(obs_slack, obs_slack)

            if self.obstacle_strategy == "robust":
                # === Distributionally Robust CVaR Constraints (Eq. 4.10-4.11) ===
                Lambda = {}
                S_vars = {}
                for j in range(n_obs):
                    for tc in check_steps:
                        key = (j, tc)
                        Lambda[key] = opti.variable()
                        opti.subject_to(Lambda[key] >= 0)
                        S_vars[key] = opti.variable(self.N_samples)
                        opti.subject_to(S_vars[key] >= 0)

                obs_slack_idx = 0
                for j, obs in enumerate(nearby_obstacles):
                    ox, oy, r = obs
                    d_min = r + VEHICLE_RADIUS + D_SAFE

                    ox_norm = (ox - self.norm_params['px_mean']) / self.norm_params['px_std']
                    oy_norm = (oy - self.norm_params['py_mean']) / self.norm_params['py_std']

                    for tc in check_steps:
                        key = (j, tc)
                        lam = Lambda[key]
                        s = S_vars[key]

                        pos = ca.mtimes(self._D_pos_ca, Z[tc])
                        px_pred = pos[0]
                        py_pred = pos[1]

                        dx_phys = (px_pred - ox_norm) * self.norm_params['px_std']
                        dy_phys = (py_pred - oy_norm) * self.norm_params['py_std']
                        dist_physical = ca.sqrt(dx_phys**2 + dy_phys**2 + 1e-6)
                        l_nom = d_min - dist_physical

                        for i in range(self.N_samples):
                            w_norm_i = float(self.w_norms[i])
                            opti.subject_to(s[i] >= l_nom + lam * w_norm_i)

                        sum_s = ca.sum1(s)
                        opti.subject_to(
                            lam * self.theta +
                            (1.0 / (self.epsilon * self.N_samples)) * sum_s
                            <= obs_slack[obs_slack_idx]
                        )
                        obs_slack_idx += 1
            else:
                # === Non-robust deterministic obstacle constraints ===
                obs_slack_idx = 0
                for obs in nearby_obstacles:
                    ox, oy, r = obs
                    d_min = r + VEHICLE_RADIUS + D_SAFE

                    ox_norm = (ox - self.norm_params['px_mean']) / self.norm_params['px_std']
                    oy_norm = (oy - self.norm_params['py_mean']) / self.norm_params['py_std']

                    for tc in check_steps:
                        pos = ca.mtimes(self._D_pos_ca, Z[tc])
                        dx_phys = (pos[0] - ox_norm) * self.norm_params['px_std']
                        dy_phys = (pos[1] - oy_norm) * self.norm_params['py_std']
                        dist_phys_sq = dx_phys**2 + dy_phys**2

                        opti.subject_to(
                            dist_phys_sq + obs_slack[obs_slack_idx] >= d_min**2
                        )
                        obs_slack_idx += 1

        opti.minimize(cost)

        # === Solver ===
        opts = {
            'ipopt.max_iter': IPOPT_MAX_ITER,
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            'print_time': False,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 5,
        }
        opti.solver('ipopt', opts)

        # Warm start
        if self._warm_start is not None:
            try:
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                pass

        # Solve
        try:
            sol = opti.solve()
            u_opt = np.array(sol.value(U[:, 0])).flatten()
            status = "optimal"
            debug_info = self._build_debug_info(
                sol=sol,
                U=U,
                T=T,
                diag_terms=diag_terms,
                v_slack=v_slack,
                obs_slack=obs_slack,
            )

            # Store warm start
            self._warm_start = {'U': sol.value(U)}

        except Exception as e:
            # Try suboptimal solution from debug
            try:
                u_opt = np.array(opti.debug.value(U[:, 0])).flatten()
                u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)
                u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)
                status = "suboptimal"
                debug_info = None
            except Exception:
                u_opt = self.u_prev.copy()
                status = f"failed: {str(e)[:50]}"
                debug_info = None

        solve_time = time.time() - t_start
        self.u_prev = u_opt.copy()

        # Explicitly release CasADi symbolic graph, to prevent long-term memory accumulation leading to segfault
        try:
            del opti
        except Exception:
            pass
        # Force GC every 50 steps
        if not hasattr(self, '_solve_count'):
            self._solve_count = 0
        self._solve_count += 1
        if self._solve_count % 50 == 0:
            gc.collect()

        return u_opt, {
            'solve_time': solve_time,
            'status': status,
            'method': self.name,
            'theta': self.theta,
            'epsilon': self.epsilon,
            'obstacle_strategy': self.obstacle_strategy,
            'debug': debug_info,
        }

    def _build_debug_info(self, sol, U, T, diag_terms, v_slack, obs_slack):
        """Build numeric debug diagnostics from symbolic expressions and solver solution."""
        u_seq = sol.value(U)
        v_slack_vals = sol.value(v_slack) if v_slack is not None else None
        obs_slack_vals = sol.value(obs_slack) if obs_slack is not None else None

        step_terms = {}
        horizon_terms = {}
        for name, expr in diag_terms.items():
            if isinstance(expr, list):
                values = [float(sol.value(item)) for item in expr]
                step_terms[name] = values
            else:
                horizon_terms[name] = float(sol.value(expr))

        tol = 1e-3
        active = []
        if abs(float(u_seq[0, 0]) - A_MIN) < tol:
            active.append("a_min")
        if abs(float(u_seq[0, 0]) - A_MAX) < tol:
            active.append("a_max")
        if abs(float(u_seq[1, 0]) + DELTA_MAX) < tol:
            active.append("delta_min")
        if abs(float(u_seq[1, 0]) - DELTA_MAX) < tol:
            active.append("delta_max")
        delta_rate0 = float(u_seq[1, 0] - self.u_prev[1])
        if abs(delta_rate0 + DELTA_RATE_MAX * DT) < tol:
            active.append("delta_rate_min")
        if abs(delta_rate0 - DELTA_RATE_MAX * DT) < tol:
            active.append("delta_rate_max")
        if v_slack_vals is not None and float(np.max(v_slack_vals)) > 1e-6:
            active.append("speed_floor_slack")
        if obs_slack_vals is not None and float(np.max(obs_slack_vals)) > 1e-6:
            active.append("obs_cvar_slack")

        step0 = {name: values[0] for name, values in step_terms.items() if values}
        summary = {
            'step0': step0,
            'horizon': horizon_terms,
            'active_constraints': active,
            'u0': [float(u_seq[0, 0]), float(u_seq[1, 0])],
            'v_slack_max': float(np.max(v_slack_vals)) if v_slack_vals is not None else 0.0,
            'obs_slack_max': float(np.max(obs_slack_vals)) if obs_slack_vals is not None else 0.0,
        }
        return summary

    def update_disturbance_samples(self, new_samples):
        """Update the disturbance sample set (e.g., from online data)."""
        if len(new_samples) > MAX_OPT_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(new_samples), MAX_OPT_SAMPLES, replace=False)
            self.w_samples = new_samples[idx]
        else:
            self.w_samples = new_samples
        self.N_samples = len(self.w_samples)
        self.w_norms = np.array([
            np.linalg.norm(self.w_samples[i])
            for i in range(self.N_samples)
        ])

    def reset(self):
        self.u_prev = np.zeros(N_U)
        self._warm_start = None
