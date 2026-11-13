"""
Linear MPC (LMPC) baseline controller.
Uses Jacobian linearized bicycle model at each operating point.

Characteristics of LMPC as a baseline method:
1. Uses traditional Jacobian linearization, instead of Koopman lift linearization
2. Linearizes around the current operating point at each time step
3. Suitable for scenarios with small deviations from the operating point
4. Moderate computational complexity, but accuracy is limited by the linearization assumption

Differences with K-MPC/K-DRMPC:
- LMPC: Uses Jacobian linearization in physical space
- K-MPC: Uses global linear model (learned) in Koopman space
- K-DRMPC: Considers robust control with disturbance uncertainty in Koopman space
"""

import numpy as np  # Import NumPy library for efficient numerical computation and multi-dimensional array operations
import casadi as ca  # Import CasADi library for symbolic computation and nonlinear optimization (IPOPT solver)
import os  # Import operating system interface module for file and path operations
import sys  # Import system module to modify Python path
import time  # Import time module to calculate solving time

# Add parent directory to system path to ensure modules at the same level can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MPC related parameters from configuration file
from config import (
    N_X,  # Physical state dimension, default 5 [px, py, psi, v, omega]
    N_U,  # Control input dimension, default 2 [acceleration a, steering angle delta]
    T_HORIZON,  # Prediction horizon length, default 20 steps
    DT,  # Time step, default 0.1 seconds
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
    IPOPT_MAX_ITER,  # IPOPT solver maximum iterations
    IPOPT_PRINT_LEVEL,  # IPOPT solver print level
    IDX_V,
    IDX_OMEGA
)

# Import linearization function from bicycle model module
from vehicle.bicycle_model import linearize


class LMPCController:
    """
    Linear MPC controller using Jacobian linearized bicycle model.

    Core idea:
    1. At each time step, linearize the nonlinear bicycle model around the current operating point (state, control)
    2. Use the linearized affine system for prediction and optimization
    3. Linearization formula: x_{t+1} ≈ A_t * x_t + B_t * u_t + c_t
       Where: A_t = ∂f/∂x|_{x_op, u_op} (Jacobian matrix)
              B_t = ∂f/∂u|_{x_op, u_op} (Control Jacobian matrix)
              c_t = f(x_op, u_op) - A_t*x_op - B_t*u_op (Constant term)

    Optimization problem form:
        min_u Σ_t ||[v_t, omega_t] - [v_ref, omega_ref]||_Q^2 + ||Δu_t||_R^2
        s.t.  x_{t+1} = A_t*x_t + B_t*u_t + c_t  (Linearized dynamics)
              x_0 = x_current                    (Initial condition)
              u_min <= u_t <= u_max              (Control constraints)
              |Δdelta_t| <= delta_rate_max * dt  (Steering angle rate constraint)
              v_min <= v_t <= v_max              (Speed constraints)
              ||pos_t - obs|| >= d_min           (Obstacle constraints)
    """

    def __init__(self):
        """
        Initialize LMPC controller.
        LMPC does not require Koopman model, directly uses linearization in physical space.
        """
        # Controller name, used for logging and result identification
        self.name = "LMPC"

        # Initialize the last control input as a zero vector
        # Used for calculating control increments and smoothness constraints
        self.u_prev = np.zeros(N_U)

        # Warm start information, used to accelerate the next solve
        # Stores the last state and control trajectory as initial guess
        self._warm_start = None

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        Solve LMPC optimization problem.

        Optimization process:
        1. Create decision variables (state and control trajectories)
        2. Linearize the bicycle model at the current operating point
        3. Construct linearized dynamics constraints
        4. Construct tracking cost and control smoothness cost
        5. Add control, state, and obstacle constraints
        6. Solve using IPOPT

        Parameters:
            x_current: numpy array, shape (5,)
                      Current state [px, py, psi, v, omega]
            ref_trajectory: numpy array, shape (T, 5)
                          Reference trajectory
            obstacles: list, each element is (ox, oy, radius)
                      Obstacle position and radius
            u_prev: numpy array, shape (2,)
                   Previous control input [a, delta]

        Returns:
            u_opt: numpy array, shape (2,)
                  Optimal control input [acceleration a, steering angle delta]
            solve_info: dict, contains solving information
                       {'solve_time', 'status', 'method'}
        """
        # If previous control input is provided, update internal storage
        if u_prev is not None:
            self.u_prev = u_prev

        # Get the prediction horizon length (default 20 steps)
        T = T_HORIZON

        # Record the start time of solving
        t_start = time.time()

        # ================================================================
        # Step 1: Create CasADi optimization problem
        # ================================================================
        # Create Opti object for constructing nonlinear programming (NLP) problem
        opti = ca.Opti()

        # Decision variables
        # X: State trajectory, shape (N_X, T+1) = (5, 21)
        # Contains initial state and all states in the prediction horizon
        X = opti.variable(N_X, T + 1)

        # U: Control trajectory, shape (N_U, T) = (2, 20)
        # All control inputs in the prediction horizon
        U = opti.variable(N_U, T)

        # Parameter: initial state
        # Use parameter instead of fixed value for easy warm start and parameter update
        x0_param = opti.parameter(N_X)
        opti.set_value(x0_param, x_current)

        # Initial condition constraint: x_0 = x_current
        opti.subject_to(X[:, 0] == x0_param)

        # ================================================================
        # Step 2: Linearize the bicycle model at the current state
        # ================================================================
        # linearize function returns:
        #   A_d: Discrete-time state transition matrix (N_X x N_X)
        #   B_d: Discrete-time control matrix (N_X x N_U)
        #   x_nom: Nominal state (for linearization)
        A_d, B_d, x_nom = linearize(x_current, self.u_prev)

        # ================================================================
        # Step 3: Construct linearized dynamics constraints
        # ================================================================
        # Iterate over each step in the prediction horizon, applying linearized dynamics
        for t in range(T):
            # Get the current step's operating point (for linearization)
            if t == 0:
                # First step: use current true state and last control input
                x_op = x_current
                u_op = self.u_prev
            else:
                # Subsequent steps: use reference trajectory and zero control (approximation)
                x_op = ref_trajectory[min(t, len(ref_trajectory) - 1)]
                u_op = np.zeros(N_U)

            # Linearize the bicycle model at the operating point
            A_t, B_t, _ = linearize(x_op, u_op)

            # Compute the constant term c_t for the affine linearized dynamics
            # Formula: c_t = f(x_op, u_op) - A_t*x_op - B_t*u_op
            # Where f(x_op, u_op) is the one-step evolution of the discrete-time bicycle model
            from vehicle.bicycle_model import discrete_step
            c_t = discrete_step(x_op, u_op) - A_t @ x_op - B_t @ u_op

            # Construct affine dynamics constraint: x_{t+1} = A_t*x_t + B_t*u_t + c_t
            x_next = (
                ca.mtimes(ca.DM(A_t), X[:, t])
                + ca.mtimes(ca.DM(B_t), U[:, t])
                + ca.DM(c_t)
            )

            # Add dynamics constraint to the optimization problem
            opti.subject_to(X[:, t + 1] == x_next)

        # ================================================================
        # Step 4: Construct cost function
        # ================================================================
        # Cost function: tracking [v, omega] + control smoothness
        cost = 0
        Q = ca.DM(Q_WEIGHTS)  # State tracking weights
        R = ca.DM(R_WEIGHTS)  # Control input weights

        # Iterate over the prediction horizon, accumulating the cost at each step
        for t in range(T):
            # Get the reference state for the current step
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]

            # Track speed v (index 2) and angular speed omega (index 4)
            y_t = ca.vertcat(X[IDX_V, t], X[IDX_OMEGA, t])  # [v, omega]
            y_ref = ca.DM([ref_t[IDX_V], ref_t[IDX_OMEGA]])  # [v_ref, omega_ref]

            # Quadratic tracking cost: (y - y_ref)^T * Q * (y - y_ref)
            cost += ca.mtimes([(y_t - y_ref).T, Q, (y_t - y_ref)])

            # Control smoothness (penalize control increments)
            if t == 0:
                # First step uses the last control input
                du = U[:, t] - ca.DM(self.u_prev)
            else:
                # Other steps use the control input from the previous step
                du = U[:, t] - U[:, t - 1]

            # Quadratic cost for control increment: du^T * R * du
            cost += ca.mtimes([du.T, R, du])

        # Set the optimization objective: minimize total cost
        opti.minimize(cost)

        # ================================================================
        # Step 5: Add constraints
        # ================================================================
        # 5.1 Control input constraints
        for t in range(T):
            # Acceleration constraint: A_MIN <= a_t <= A_MAX
            opti.subject_to(opti.bounded(A_MIN, U[0, t], A_MAX))
            # Steering angle constraint: -DELTA_MAX <= delta_t <= DELTA_MAX
            opti.subject_to(
                opti.bounded(-DELTA_MAX, U[1, t], DELTA_MAX)
            )

            # Steering angle rate constraint
            # Limit the change in steering angle between adjacent time steps
            if t == 0:
                # First step compares with the last control input
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - self.u_prev[1],
                    DELTA_RATE_MAX * DT
                ))
            else:
                # Other steps compare with the previous step
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - U[1, t - 1],
                    DELTA_RATE_MAX * DT
                ))

        # 5.2 State constraints
        for t in range(T + 1):
            # Speed constraint: V_MIN <= v_t <= V_MAX
            opti.subject_to(opti.bounded(V_MIN, X[IDX_V, t], V_MAX))

        # 5.3 Obstacle avoidance constraints (linearized distance)
        # Iterate over all obstacles
        for obs in obstacles:
            ox, oy, r = obs
            # Calculate minimum safe distance
            d_min = r + VEHICLE_RADIUS + D_SAFE

            # Check obstacle constraints at each step in the prediction horizon
            for t in range(T + 1):
                # Calculate the distance components between the vehicle and the obstacle
                dx = X[0, t] - ox  # x-direction distance
                dy = X[1, t] - oy  # y-direction distance

                # Distance constraint: ||pos - obs||^2 >= d_min^2
                # This is a non-convex constraint, IPOPT can handle but may get stuck in local optima
                opti.subject_to(dx**2 + dy**2 >= d_min**2)

        # ================================================================
        # Step 6: Configure IPOPT solver
        # ================================================================
        # IPOPT configuration options
        opts = {
            # Maximum number of iterations
            'ipopt.max_iter': IPOPT_MAX_ITER,
            # Print level (0=silent, 5=detailed)
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            # Do not print solving time
            'print_time': False,
            # Enable warm start (use the previous solution as initial guess)
            'ipopt.warm_start_init_point': 'yes',
        }

        # Set the solver to IPOPT (Interior Point Optimizer)
        opti.solver('ipopt', opts)

        # ================================================================
        # Step 7: Warm start (optional)
        # ================================================================
        # If there is a previous solution, use it as initial guess
        # Warm start can accelerate the solving process, especially for similar problems
        if self._warm_start is not None:
            try:
                # Set initial guess for state trajectory
                opti.set_initial(X, self._warm_start['X'])
                # Set initial guess for control trajectory
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                # If setting fails, ignore and continue
                pass

        # ================================================================
        # Step 8: Solve the optimization problem
        # ================================================================
        try:
            # Call IPOPT solver
            sol = opti.solve()

            # Extract the optimal control input for the first step
            # MPC only applies the control for the first step, then re-optimizes in the next step
            u_opt = np.array(sol.value(U[:, 0])).flatten()

            # Solving status: optimal
            status = "optimal"

            # Save the current solution for the next warm start
            # Also save state and control trajectories for complete warm start
            self._warm_start = {
                'X': sol.value(X),
                'U': sol.value(U),
            }

        except Exception as e:
            # If solving fails, use the last control input as fallback
            # LMPC does not have a suboptimal solution extraction mechanism, directly use fallback strategy
            u_opt = self.u_prev.copy()
            # Truncate error message (maximum 50 characters)
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
        # Reset the last control input to zero vector
        self.u_prev = np.zeros(N_U)

        # Clear warm start information
        self._warm_start = None
