"""
Kinematic bicycle model for vehicle dynamics (paper Section 2).

State: x = [px, py, psi, v, omega]^T ∈ R^5
    - px: Vehicle x-position in the global frame [m]
    - py: Vehicle y-position in the global frame [m]
    - v:  Longitudinal vehicle speed [m/s]
    - psi: Heading angle (angle between vehicle orientation and global x-axis) [rad]
    - omega: Yaw rate (heading-angle derivative) [rad/s]

Control input: u = [a, delta]^T ∈ R^2
    - a: Longitudinal acceleration [m/s^2] (positive accelerates, negative decelerates)
    - delta: Front-wheel steering angle [rad] (positive turns left, negative turns right)

Continuous-time dynamics:
    px_dot  = v * cos(psi)              # x-position rate is the x-component of velocity
    py_dot  = v * sin(psi)              # y-position rate is the y-component of velocity
    v_dot   = a                          # Speed rate equals acceleration
    psi_dot = v * tan(delta) / L        # Heading rate depends on speed and steering angle
    omega   = v * tan(delta) / L        # Yaw rate equals heading rate (kinematic constraint)

Applications:
    - LMPC (linear model predictive control): uses the linearized model
    - NMPC (nonlinear model predictive control): uses the full nonlinear model
    - Simulation: serves as the plant dynamics model
"""

import numpy as np  # NumPy for efficient numerical computing and array operations
import os  # Operating system interfaces for file and path operations
import sys  # System module for modifying the Python path

# Add the project root to the system path so sibling modules such as config.py can be imported.
# os.path.dirname(os.path.abspath(__file__)) obtains the parent directory of this file.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import vehicle and simulation parameters from the configuration.
from config import (  # noqa: F401  # A_MIN and A_MAX are unused here but may be needed elsewhere
    L_WHEELBASE,  # Wheelbase [m], the distance between the front and rear axles
    DT,           # Sampling interval [s], the discrete-time step
    V_MIN,        # Minimum speed [m/s], lower speed bound
    V_MAX,        # Maximum speed [m/s], upper speed bound
    A_MIN,        # Maximum deceleration [m/s^2]; negative values indicate braking
    A_MAX,        # Maximum acceleration [m/s^2]; positive values indicate acceleration
    DELTA_MAX,    # Maximum steering angle [rad], front-wheel steering limit
    IDX_PSI,      # Heading-angle index
    IDX_V,        # Speed index
    IDX_OMEGA     # Yaw-rate index
)


def continuous_dynamics(x, u, L=L_WHEELBASE):
    """
    Continuous-time kinematic bicycle model.

    Implements the continuous-time vehicle dynamics and computes the state derivative
    for a given state and control input.

    Args:
        x: NumPy array of shape (5,) representing the current state
           [px, py, psi, v, omega]
              - px: x-position [m]
              - py: y-position [m]
              - v:  Longitudinal speed [m/s]
              - psi: Heading angle [rad]
              - omega: Yaw rate [rad/s]
          u: NumPy array of shape (2,) representing the control input
           [a, delta]
              - a: Longitudinal acceleration [m/s^2]
              - delta: Front-wheel steering angle [rad]
          L: Wheelbase [m]; defaults to L_WHEELBASE from the configuration.

    Returns:
        x_dot: NumPy array of shape (5,) containing the state derivative
               [px_dot, py_dot, v_dot, psi_dot, omega_dot]
    """
    # Unpack the state variables for readability.
    px, py, psi, v, omega = x
    # Unpack the control inputs.
    a, delta = u

    # Clip the steering angle for numerical stability, keeping a 1e-6 margin
    # from the tangent singularities at ±π/2.
    delta = np.clip(delta, -DELTA_MAX + 1e-6, DELTA_MAX - 1e-6)
    # Keep speed nonzero to avoid division-by-zero and numerical issues.
    # Use 1e-3 when the speed is close to zero.
    v = max(v, 1e-3)

    # Compute the state derivative from the continuous-time dynamics.
    # x-position rate = speed × cos(heading angle)
    px_dot = v * np.cos(psi)
    # y-position rate = speed × sin(heading angle)
    py_dot = v * np.sin(psi)
    # Speed rate = acceleration (directly controlled).
    v_dot = a
    # Heading rate = (speed × tan(steering angle)) / wheelbase.
    # This is the core equation of the kinematic bicycle model.
    psi_dot = v * np.tan(delta) / L
    # Yaw-rate dynamics: a first-order system tracks the heading rate.
    # tau is the time constant that controls how quickly omega converges to psi_dot.
    # tau=0.05 gives fast convergence relative to dt=0.1.
    tau = 0.05  # Fast time constant relative to the 0.1 s sampling interval
    omega_dot = (psi_dot - omega) / tau  # First-order dynamic equation

    # Return the state-derivative vector.
    return np.array([px_dot, py_dot, psi_dot, v_dot, omega_dot])


def discrete_step(x, u, dt=DT, L=L_WHEELBASE):
    """
    Integrate one discrete-time step using the fourth-order Runge-Kutta method (RK4).

    RK4 is a high-accuracy numerical integration method that is more accurate than
    the simple Euler method, particularly for discretizing nonlinear systems.

    RK4 formula:
        k1 = f(x, u)                        # Slope at the start
        k2 = f(x + dt/2 * k1, u)            # Midpoint slope predicted using k1
        k3 = f(x + dt/2 * k2, u)            # Midpoint slope predicted using k2
        k4 = f(x + dt * k3, u)              # Slope at the endpoint
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        x: Current state vector, a NumPy array of shape (5,).
        u: Control input vector, a NumPy array of shape (2,).
        dt: Time step [s]; defaults to DT from the configuration.
        L: Wheelbase [m]; defaults to L_WHEELBASE from the configuration.

    Returns:
        x_next: State at the next time step, a NumPy array of shape (5,).
    """
    # Compute the four slopes, the core of the RK4 method.
    # k1: Derivative at the current state.
    k1 = continuous_dynamics(x, u, L)
    # k2: Derivative at the midpoint predicted using k1.
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, L)
    # k3: Derivative at the midpoint predicted using k2 (a more accurate midpoint estimate).
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, L)
    # k4: Derivative at the endpoint predicted using k3.
    k4 = continuous_dynamics(x + dt * k3, u, L)

    # RK4 weighted average: midpoint slopes receive twice the endpoint weight.
    # This gives RK4 fourth-order accuracy (local error O(dt^5)).
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Enforce speed limits for physical plausibility.
    # Speed must remain between V_MIN (typically 0) and V_MAX.
    x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)

    return x_next


def discrete_step_with_disturbance(x, u, w, dt=DT, L=L_WHEELBASE):
    """
    Integrate one discrete-time step with disturbance using RK4.

    The disturbance w is added directly to the continuous-time state derivative
    and propagated to the next state through RK4:
        x_dot = f(x, u) + w

    Args:
        x: Current state [px, py, psi, v, omega], a NumPy array of shape (5,).
        u: Control input [a, delta], a NumPy array of shape (2,).
        w: Disturbance vector [w_px, w_py, w_psi, w_v, w_omega], shape (5,).
        dt: Time step [s].
        L: Wheelbase [m].

    Returns:
        x_next: Disturbance-affected state at the next time step, a NumPy array of shape (5,).
    """
    # Compute the four slopes for the deterministic dynamics.
    k1 = continuous_dynamics(x, u, L)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, L)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, L)
    k4 = continuous_dynamics(x + dt * k3, u, L)

    # Integrate the deterministic dynamics using RK4.
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # Add the disturbance contribution directly to the state increment.
    x_next = x_next + w * dt

    # Enforce the speed limits.
    x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)
    return x_next


def discrete_step_batch_with_disturbance(x_batch, u_batch, w_batch,
                                         dt=DT, L=L_WHEELBASE):
    """
    Compute discrete-time steps with disturbances for a batch of samples.

    Args:
        x_batch: NumPy array of shape (N, 5) containing N state vectors.
        u_batch: NumPy array of shape (N, 2) containing N control inputs.
        w_batch: NumPy array of shape (N, 5) containing N disturbance vectors.
        dt: Time step [s].
        L: Wheelbase [m].

    Returns:
        x_next_batch: NumPy array of shape (N, 5) containing the N next states.
    """
    N = x_batch.shape[0]
    x_next = np.zeros_like(x_batch)
    for i in range(N):
        x_next[i] = discrete_step_with_disturbance(
            x_batch[i], u_batch[i], w_batch[i], dt, L
        )
    return x_next


def discrete_step_batch(x_batch, u_batch, dt=DT, L=L_WHEELBASE):
    """
    Compute discrete-time steps in a batch to process multiple state-control pairs.

    This function is useful for data processing and model training, efficiently
    generating a large number of state-transition samples.

    Args:
        x_batch: NumPy array of shape (N, 5) containing N state vectors;
                 each row is a 5D state [px, py, psi, v, omega].
        u_batch: NumPy array of shape (N, 2) containing N control inputs;
                 each row is a 2D control [a, delta].
        dt: Time step [s].
        L: Wheelbase [m].

    Returns:
        x_next_batch: NumPy array of shape (N, 5) containing N next states;
                      x_next_batch[i] is the next state from applying u_batch[i]
                      to x_batch[i].
    """
    # Get the batch size (number of samples).
    N = x_batch.shape[0]
    # Preallocate a zero-filled output array with the same shape as the input.
    x_next = np.zeros_like(x_batch)
    # Advance each sample by one discrete-time step.
    # Use a loop rather than vectorization because discrete_step contains nonlinear operations.
    for i in range(N):
        x_next[i] = discrete_step(x_batch[i], u_batch[i], dt, L)
    return x_next


def linearize(x_op, u_op, dt=DT, L=L_WHEELBASE, eps=1e-5):
    """
    Linearize the discrete-time dynamics around the operating point (x_op, u_op).

    Linearized model:
        x_{t+1} ≈ A_d @ (x - x_op) + B_d @ (u - u_op) + x_op_next

    where:
        - A_d is the state Jacobian ∂f/∂x
        - B_d is the control Jacobian ∂f/∂u
        - x_op_next is the nominal next state at the operating point

    Compute the Jacobians using central differences, which are more accurate than
    forward differences:
        ∂f/∂x_i ≈ [f(x + eps*e_i) - f(x - eps*e_i)] / (2*eps)

    Args:
        x_op: State at the linearization operating point, a NumPy array of shape (5,).
        u_op: Control input at the operating point, a NumPy array of shape (2,).
        dt: Time step [s].
        L: Wheelbase [m].
        eps: Finite-difference perturbation, defaulting to 1e-5. A value that is too
             large reduces linearization accuracy; one that is too small increases
             numerical error.

        Returns:
           A_d: Discrete state Jacobian, a NumPy array of shape (5, 5), where
               A_d[i, j] = ∂x_next[i] / ∂x[j].
           B_d: Discrete control Jacobian, a NumPy array of shape (5, 2), where
               B_d[i, j] = ∂x_next[i] / ∂u[j].
           x_op_next: Nominal next state at the operating point, a NumPy array of shape (5,),
                    with x_op_next = f(x_op, u_op).
    """
    # Get the state and control dimensions.
    n_x = len(x_op)  # State dimension; expected to be 5
    n_u = len(u_op)  # Control dimension; expected to be 2

    # Compute the nominal next state at the operating point.
    x_op_next = discrete_step(x_op, u_op, dt, L)

    # Compute state Jacobian A_d using central differences.
    # A_d is a 5x5 matrix describing each state variable's effect on each next-state variable.
    A_d = np.zeros((n_x, n_x))  # Initialize to zero
    for i in range(n_x):  # Perturb each state dimension
        # Positive perturbation: increase state i by eps.
        x_plus = x_op.copy()
        x_plus[i] += eps
        # Negative perturbation: decrease state i by eps.
        x_minus = x_op.copy()
        x_minus[i] -= eps
        # Central difference: [f(x+) - f(x-)] / (2*eps).
        A_d[:, i] = (discrete_step(x_plus, u_op, dt, L) -
                     discrete_step(x_minus, u_op, dt, L)) / (2 * eps)

    # Compute control Jacobian B_d using central differences.
    # B_d is a 5x2 matrix describing each control input's effect on each next-state variable.
    B_d = np.zeros((n_x, n_u))  # Initialize to zero
    for i in range(n_u):  # Perturb each control dimension
        # Positive perturbation: increase control i by eps.
        u_plus = u_op.copy()
        u_plus[i] += eps
        # Negative perturbation: decrease control i by eps.
        u_minus = u_op.copy()
        u_minus[i] -= eps
        # Central difference: [f(u+) - f(u-)] / (2*eps).
        B_d[:, i] = (discrete_step(x_op, u_plus, dt, L) -
                     discrete_step(x_op, u_minus, dt, L)) / (2 * eps)

    # Return the linearized model.
    return A_d, B_d, x_op_next


def casadi_dynamics():
    """
    Create a CasADi symbolic dynamics function for NMPC (nonlinear model predictive control).

    CasADi is a symbolic-computation and automatic-differentiation library that can:
    1. Build symbolic expressions
    2. Automatically compute Jacobian and Hessian matrices
    3. Generate efficient C code
    4. Interface with optimization solvers such as IPOPT

    This function discretizes the continuous-time dynamics using RK4 and returns
    a CasADi function object that can be called within an optimization problem.

    Returns:
        f: CasADi function mapping (x, u) -> x_next, ready for use in CasADi optimization.
        x_sym: CasADi symbolic state variable of shape (5,).
        u_sym: CasADi symbolic control variable of shape (2,).
    """
    import casadi as ca  # Import the CasADi symbolic-computation library

    # Create symbolic variables.
    # MX.sym creates a 5D symbolic column vector for the state.
    x_sym = ca.MX.sym('x', 5)
    # MX.sym creates a 2D symbolic column vector for the control input.
    u_sym = ca.MX.sym('u', 2)

    # Use vehicle parameters from the configuration.
    L = L_WHEELBASE  # Wheelbase [m]
    tau = 0.05  # Time constant for yaw-rate dynamics [s]

    def _continuous(xx, uu):
        """
        Helper function that computes continuous-time dynamics for a given state and control.

        Encapsulates the continuous-time dynamics for RK4 integration.
        Uses CasADi symbolic operations instead of NumPy to enable automatic differentiation.

        Args:
            xx: CasADi symbolic state vector [px, py, psi, v, omega].
            uu: CasADi symbolic control vector [a, delta].

        Returns:
            CasADi symbolic state derivative [px_dot, py_dot, v_dot, psi_dot, omega_dot].
        """
        # Use ca.fmax to keep speed nonzero (the symbolic equivalent of max).
        # 1e-3 is the minimum speed threshold to avoid division by zero.
        vv = ca.fmax(xx[IDX_V], 1e-3)
        # Compute heading rate from the kinematic constraint.
        psi_dot = vv * ca.tan(uu[1]) / L
        # Return the state-derivative vector, vertically concatenated with ca.vertcat.
        return ca.vertcat(
            vv * ca.cos(xx[IDX_PSI]),      # px_dot = v * cos(psi)
            vv * ca.sin(xx[IDX_PSI]),      # py_dot = v * sin(psi)
            psi_dot,                        # psi_dot = v * tan(delta) / L
            uu[0],                          # v_dot = a
            (psi_dot - xx[IDX_OMEGA]) / tau  # omega_dot = (psi_dot - omega) / tau
        )

    # Use the sampling interval from the configuration.
    dt = DT

    # Symbolic RK4 integration.
    # k1: Derivative at the current state.
    k1 = _continuous(x_sym, u_sym)
    # k2: Derivative at the midpoint predicted using k1.
    k2 = _continuous(x_sym + 0.5 * dt * k1, u_sym)
    # k3: Derivative at the midpoint predicted using k2.
    k3 = _continuous(x_sym + 0.5 * dt * k2, u_sym)
    # k4: Derivative at the endpoint predicted using k3.
    k4 = _continuous(x_sym + dt * k3, u_sym)

    # Use the RK4 weighted-average formula to compute the next state.
    x_next = x_sym + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Create the CasADi function object.
    # Arguments:
    #   'bicycle_dynamics': Function name
    #   [x_sym, u_sym]: Input symbolic variables
    #   [x_next]: Output symbolic expression
    #   ['x', 'u']: Input argument names (for documentation and debugging)
    #   ['x_next']: Output argument name
    f = ca.Function(
        'bicycle_dynamics', [x_sym, u_sym], [x_next],
        ['x', 'u'], ['x_next']
    )

    # Return the function and symbolic variables for building optimization problems.
    return f, x_sym, u_sym
