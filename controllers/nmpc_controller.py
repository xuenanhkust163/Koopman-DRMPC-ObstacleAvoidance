"""
Nonlinear MPC (NMPC) baseline controller.
Uses the full nonlinear bicycle model via CasADi + IPOPT.
"""

import numpy as np
import casadi as ca
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    N_X, N_U, T_HORIZON, DT, Q_WEIGHTS, R_WEIGHTS,
    V_MIN, V_MAX, A_MIN, A_MAX, DELTA_MAX, DELTA_RATE_MAX, D_SAFE,
    VEHICLE_RADIUS, IPOPT_MAX_ITER, IPOPT_PRINT_LEVEL
    , IDX_V, IDX_OMEGA
)
from vehicle.bicycle_model import casadi_dynamics


class NMPCController:
    """
    Nonlinear MPC using full bicycle model dynamics.
    Solved as NLP via CasADi Opti + IPOPT.
    """

    def __init__(self):
        self.name = "NMPC"
        self.u_prev = np.zeros(N_U)
        self._warm_start = None
        self._setup_solver()

    def _setup_solver(self):
        """Pre-build the NLP structure."""
        self._f_dyn, _, _ = casadi_dynamics()

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        Solve the NMPC problem.

        Args:
            x_current: (5,) current state
            ref_trajectory: (T, 5) reference states
            obstacles: list of (ox, oy, radius)
            u_prev: (2,) previous control input

        Returns:
            u_opt: (2,) optimal control
            solve_info: dict
        """
        if u_prev is not None:
            self.u_prev = u_prev

        T = T_HORIZON
        t_start = time.time()

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(N_X, T + 1)
        U = opti.variable(N_U, T)

        # Initial condition
        opti.subject_to(X[:, 0] == x_current)

        # Nonlinear dynamics constraints
        for t in range(T):
            x_next = self._f_dyn(X[:, t], U[:, t])
            opti.subject_to(X[:, t + 1] == x_next)

        # Cost function
        cost = 0
        Q = ca.DM(Q_WEIGHTS)
        R = ca.DM(R_WEIGHTS)

        for t in range(T):
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            y_t = ca.vertcat(X[IDX_V, t], X[IDX_OMEGA, t])
            y_ref = ca.DM([ref_t[IDX_V], ref_t[IDX_OMEGA]])
            cost += ca.mtimes([(y_t - y_ref).T, Q, (y_t - y_ref)])

            if t == 0:
                du = U[:, t] - ca.DM(self.u_prev)
            else:
                du = U[:, t] - U[:, t - 1]
            cost += ca.mtimes([du.T, R, du])

        opti.minimize(cost)

        # Input constraints
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

        # State constraints
        for t in range(T + 1):
            opti.subject_to(opti.bounded(V_MIN, X[IDX_V, t], V_MAX))

        # Obstacle avoidance (nonlinear)
        for obs in obstacles:
            ox, oy, r = obs
            d_min = r + VEHICLE_RADIUS + D_SAFE
            for t in range(1, T + 1):
                dx = X[0, t] - ox
                dy = X[1, t] - oy
                opti.subject_to(dx**2 + dy**2 >= d_min**2)

        # Solver options
        opts = {
            'ipopt.max_iter': IPOPT_MAX_ITER,
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            'print_time': False,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
        }
        opti.solver('ipopt', opts)

        # Warm start
        if self._warm_start is not None:
            try:
                opti.set_initial(X, self._warm_start['X'])
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                pass
        else:
            # Initialize with reference trajectory
            for t in range(T + 1):
                ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
                opti.set_initial(X[:, t], ref_t)

        # Solve
        try:
            sol = opti.solve()
            u_opt = np.array(sol.value(U[:, 0])).flatten()
            status = "optimal"
            self._warm_start = {
                'X': sol.value(X),
                'U': sol.value(U),
            }
        except Exception as e:
            u_opt = self.u_prev.copy()
            status = f"failed: {str(e)[:50]}"

        solve_time = time.time() - t_start
        self.u_prev = u_opt.copy()

        return u_opt, {
            'solve_time': solve_time,
            'status': status,
            'method': self.name,
        }

    def reset(self):
        self.u_prev = np.zeros(N_U)
        self._warm_start = None
