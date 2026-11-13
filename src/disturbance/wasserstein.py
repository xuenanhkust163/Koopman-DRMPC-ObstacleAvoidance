"""
Wasserstein fuzzy set and CVaR constraint computation module.
Corresponding to Sections 5.2-5.5 of the paper.

Provides post-hoc evaluation of CVaR safety margins for performance metric computation.
"""

import numpy as np  # Import NumPy library for numerical computations
import os  # Import operating system interface module
import sys  # Import system module for path operations

# Add parent directory to system path for project module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import Wasserstein radius, CVaR risk level, safety distance, and vehicle radius constants from config file
from config import THETA_WASSERSTEIN, EPSILON_CVAR, D_SAFE, VEHICLE_RADIUS


def compute_cvar_margin(positions, obstacles, w_samples,
                        theta=THETA_WASSERSTEIN, epsilon=EPSILON_CVAR):
    """
    Compute the worst-case CVaR safety margin for given trajectory positions.

    For each obstacle and each time step, compute:
        sup_{P in D} CVaR_epsilon^P [l(x_t, w_t)]

    Using the dual reconstruction formula (Theorem, Section 5.5):
        = inf_{lambda>=0} { lambda*theta + 1/(epsilon*N) * sum [l_i + lambda*||w_i||]_+ }

    Args:
        positions: Array of predicted positions with shape (T, 2), containing [px, py] (physical coordinates)
        obstacles: List of obstacles in the format [(ox, oy, radius), ...]
        w_samples: Array of disturbance samples with shape (N, 5)
        theta: Wasserstein radius (fuzzy set size)
        epsilon: CVaR risk level (usually small, like 0.05 or 0.1)

    Returns:
        max_cvar: Maximum CVaR margin across all obstacles and time steps
                  (negative indicates safety, positive indicates danger)
        cvar_per_obs: List of CVaR values for each obstacle
    """
    from scipy.optimize import minimize_scalar  # Import scalar optimization function to solve dual problem

    N = len(w_samples)  # Get the number of disturbance samples

    # Pre-compute ||w_i|| (norm) to avoid redundant calculations
    w_norms = np.array([np.linalg.norm(w_samples[i]) for i in range(N)])  # Compute the norm of each raw sample

    cvar_per_obs = []  # Initialize the list of CVaR values for each obstacle

    for obs in obstacles:  # Iterate over each obstacle
        ox, oy, r = obs  # Unpack obstacle information: x-coordinate, y-coordinate, radius
        d_min = r + VEHICLE_RADIUS + D_SAFE  # Compute the minimum safety distance (obstacle radius + vehicle radius + safety margin)
        max_cvar_obs = -np.inf  # Initialize the maximum CVaR value for this obstacle to negative infinity

        for t in range(len(positions)):  # Iterate over each time step
            px, py = positions[t]  # Get the current position coordinates
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)  # Compute the Euclidean distance from the vehicle to the obstacle center
            l_nom = d_min - dist  # Compute the nominal safety margin (positive indicates safety, negative indicates collision)

            # Dual reconstruction: find the optimal lambda value
            def dual_objective(lam):
                """Dual objective function: lambda*theta + (1/epsilon)*E[max(l_nom + lambda*||w_i||, 0)]"""
                terms = np.maximum(l_nom + lam * w_norms, 0)  # Compute [l_nom + lambda*||w_i||]_+
                return lam * theta + np.mean(terms) / epsilon  # Return the dual objective function value

            # Use scalar optimizer to find the optimal lambda in the range [0, 100]
            result = minimize_scalar(dual_objective, bounds=(0, 100), method='bounded')
            cvar_value = result.fun  # Get the optimal objective function value (i.e., CVaR value)

            max_cvar_obs = max(max_cvar_obs, cvar_value)  # Update the maximum CVaR value for this obstacle

        cvar_per_obs.append(max_cvar_obs)  # Add the maximum CVaR value of this obstacle to the list

    max_cvar = max(cvar_per_obs) if cvar_per_obs else 0.0  # Get the maximum CVaR value among all obstacles

    return max_cvar, cvar_per_obs  # Return the maximum CVaR value and the list of CVaR values for each obstacle


def check_constraint_violation(positions, obstacles):
    """
    Check if any distance constraints with obstacles are violated.

    Args:
        positions: Trajectory array with shape (T, 2), containing [px, py]
        obstacles: List of obstacles in the format [(ox, oy, radius), ...]

    Returns:
        violations: Boolean array with shape (T,), True if constraint is violated at time step t
        min_distances: Array with shape (T, n_obs), recording the minimum distance to each obstacle
    """
    T = len(positions)  # Get the number of time steps
    n_obs = len(obstacles)  # Get the number of obstacles
    violations = np.zeros(T, dtype=bool)  # Initialize the violation flag array, all set to False
    min_distances = np.zeros((T, n_obs))  # Initialize the distance array, recording the distance to each obstacle at each time step

    for j, obs in enumerate(obstacles):  # Iterate over each obstacle, j is the index
        ox, oy, r = obs  # Unpack obstacle information
        d_min = r + VEHICLE_RADIUS + D_SAFE  # Compute the minimum safety distance

        for t in range(T):  # Iterate over each time step
            px, py = positions[t]  # Get the current position
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)  # Compute the distance to the obstacle center
            min_distances[t, j] = dist  # Record this distance
            if dist < d_min:  # If the distance is less than the safety distance
                violations[t] = True  # Mark this time step as violating the constraint

    return violations, min_distances  # Return the violation flag array and the distance matrix
