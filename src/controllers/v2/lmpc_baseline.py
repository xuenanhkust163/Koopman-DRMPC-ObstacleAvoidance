"""Linearized MPC baseline using the shared v2 Pure Pursuit track context."""

from .lmpc_solve_strategy import LinearizedMPCSolveStrategy
from .pure_pursuit_baseline import PurePursuitBaselineController


class LMPCBaselineController(PurePursuitBaselineController):
    """Use the shared path/speed/obstacle context with a linearized MPC solve strategy."""

    def __init__(self, track, **kwargs):
        super().__init__(track, solve_strategy=LinearizedMPCSolveStrategy(), **kwargs)
        self.name = "LMPCBaseline"