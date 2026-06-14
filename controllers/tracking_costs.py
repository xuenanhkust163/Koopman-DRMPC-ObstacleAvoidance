"""Reusable tracking-cost builders for Koopman MPC controllers."""

from dataclasses import dataclass
from typing import Optional, Union

import casadi as ca


def _append_diag(diag_terms, name, expr):
    diag_terms.setdefault(name, []).append(expr)


@dataclass
class MinSpeedRule:
    """Soft minimum-speed rule used by DR formulations."""

    floor_abs: float
    floor_ratio: float

    def floor(self, v_ref: float) -> float:
        return max(self.floor_abs, self.floor_ratio * float(v_ref))


class DefaultTrackingCostBuilder:
    """Default stage-cost builder for Koopman trajectory tracking."""

    def stage_cost(
        self,
        *,
        opti,
        t: int,
        z_t,
        u_t,
        u_prev,
        u_prev_step,
        y_t,
        y_ref_t,
        ref_psi_t: float,
        ref_px_norm_t: float,
        ref_py_norm_t: float,
        d_pos_ca,
        d_psi_ca,
        q,
        r,
        q_psi: float,
        q_progress: float,
        q_pos: float,
        add_position_term: bool,
        add_abs_u_term: bool = False,
        r_abs=None,
        min_speed_rule: Optional[MinSpeedRule] = None,
        v_slack_t=None,
        risk_terms=None,
        **_extra,
    ):
        """Return stage cost term and add optional soft constraints to opti."""
        stage = 0

        # [v, omega] tracking
        y_err = y_t - y_ref_t
        stage += ca.mtimes([y_err.T, q, y_err])

        # Heading tracking with wrapped angle error
        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - float(ref_psi_t)),
                           ca.cos(psi_t - float(ref_psi_t)))
        stage += q_psi * (psi_err ** 2)

        # Forward-progress tracking
        v_forward = y_t[0] * ca.cos(psi_err)
        stage += q_progress * ((v_forward - float(y_ref_t[0])) ** 2)

        # Optional soft minimum-speed constraint
        if min_speed_rule is not None and v_slack_t is not None:
            v_floor = min_speed_rule.floor(float(y_ref_t[0]))
            opti.subject_to(y_t[0] + v_slack_t >= v_floor)

        # Position term (typically every 4th step)
        if add_position_term:
            pos = ca.mtimes(d_pos_ca, z_t)
            pos_err_x = pos[0] - float(ref_px_norm_t)
            pos_err_y = pos[1] - float(ref_py_norm_t)
            stage += q_pos * (pos_err_x ** 2 + pos_err_y ** 2)

        # Input increment penalty
        if t == 0:
            du = u_t - u_prev
        else:
            du = u_t - u_prev_step
        stage += ca.mtimes([du.T, r, du])

        # Optional absolute input penalty
        if add_abs_u_term and r_abs is not None:
            stage += ca.mtimes([u_t.T, r_abs, u_t])

        return stage

    def finalize_cost(self, **ctx):
        """Optional horizon-level terminal anchor for heading and position."""
        z_terminal = ctx.get("z_terminal")
        if z_terminal is None:
            return 0

        terminal_cost = 0

        terminal_heading_weight = float(ctx.get("terminal_heading_weight", 0.0))
        ref_psi_terminal = ctx.get("ref_psi_terminal")
        d_psi_ca = ctx.get("d_psi_ca")
        if terminal_heading_weight > 0.0 and ref_psi_terminal is not None and d_psi_ca is not None:
            psi_terminal = ca.mtimes(d_psi_ca, z_terminal)[0]
            psi_err_terminal = ca.atan2(
                ca.sin(psi_terminal - float(ref_psi_terminal)),
                ca.cos(psi_terminal - float(ref_psi_terminal)),
            )
            terminal_cost += terminal_heading_weight * (psi_err_terminal ** 2)

        terminal_pos_weight = float(ctx.get("terminal_pos_weight", 0.0))
        ref_px_norm_terminal = ctx.get("ref_px_norm_terminal")
        ref_py_norm_terminal = ctx.get("ref_py_norm_terminal")
        d_pos_ca = ctx.get("d_pos_ca")
        if (
            terminal_pos_weight > 0.0
            and ref_px_norm_terminal is not None
            and ref_py_norm_terminal is not None
            and d_pos_ca is not None
        ):
            pos_terminal = ca.mtimes(d_pos_ca, z_terminal)
            pos_err_x_terminal = pos_terminal[0] - float(ref_px_norm_terminal)
            pos_err_y_terminal = pos_terminal[1] - float(ref_py_norm_terminal)
            terminal_cost += terminal_pos_weight * (
                pos_err_x_terminal ** 2 + pos_err_y_terminal ** 2
            )

        return terminal_cost

    def collect_stage_diagnostics(self, diag_terms, **ctx):
        """Collect symbolic stage diagnostics for later numeric evaluation."""
        t = int(ctx["t"])
        z_t = ctx["z_t"]
        u_t = ctx["u_t"]
        u_prev = ctx["u_prev"]
        u_prev_step = ctx["u_prev_step"]
        y_t = ctx["y_t"]
        y_ref_t = ctx["y_ref_t"]
        ref_psi_t = float(ctx["ref_psi_t"])
        ref_px_norm_t = float(ctx["ref_px_norm_t"])
        ref_py_norm_t = float(ctx["ref_py_norm_t"])
        d_pos_ca = ctx["d_pos_ca"]
        d_psi_ca = ctx["d_psi_ca"]
        q = ctx["q"]
        r = ctx["r"]
        q_psi = float(ctx["q_psi"])
        q_progress = float(ctx["q_progress"])
        q_pos = float(ctx["q_pos"])

        y_err = y_t - y_ref_t
        track_term = ca.mtimes([y_err.T, q, y_err])
        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - ref_psi_t), ca.cos(psi_t - ref_psi_t))
        heading_term = q_psi * (psi_err ** 2)
        v_forward = y_t[0] * ca.cos(psi_err)
        progress_term = q_progress * ((v_forward - float(y_ref_t[0])) ** 2)

        if t == 0:
            du = u_t - u_prev
        else:
            du = u_t - u_prev_step
        du_term = ca.mtimes([du.T, r, du])

        _append_diag(diag_terms, "cost_track_vomega", track_term)
        _append_diag(diag_terms, "cost_heading", heading_term)
        _append_diag(diag_terms, "cost_progress", progress_term)
        _append_diag(diag_terms, "cost_du", du_term)
        _append_diag(diag_terms, "psi_err", psi_err)
        _append_diag(diag_terms, "v_ref", y_ref_t[0])
        _append_diag(diag_terms, "v_pred", y_t[0])

        if bool(ctx["add_position_term"]):
            pos = ca.mtimes(d_pos_ca, z_t)
            pos_err_x = pos[0] - ref_px_norm_t
            pos_err_y = pos[1] - ref_py_norm_t
            pos_term = q_pos * (pos_err_x ** 2 + pos_err_y ** 2)
            _append_diag(diag_terms, "cost_position", pos_term)
            _append_diag(diag_terms, "pos_err_x", pos_err_x)
            _append_diag(diag_terms, "pos_err_y", pos_err_y)

        if bool(ctx["add_abs_u_term"]) and (ctx.get("r_abs") is not None):
            abs_u_term = ca.mtimes([u_t.T, ctx["r_abs"], u_t])
            _append_diag(diag_terms, "cost_abs_u", abs_u_term)

        min_speed_rule = ctx.get("min_speed_rule")
        v_slack_t = ctx.get("v_slack_t")
        if min_speed_rule is not None and v_slack_t is not None:
            _append_diag(diag_terms, "v_slack", v_slack_t)
            _append_diag(diag_terms, "v_floor", min_speed_rule.floor(float(y_ref_t[0])))

    def finalize_diagnostics(self, diag_terms, **_ctx):
        """Optional horizon-level diagnostics hook."""
        return diag_terms


class WeightedTrackingCostBuilder(DefaultTrackingCostBuilder):
    """Default builder with simple multiplicative profile scales."""

    def __init__(
        self,
        *,
        q_psi_scale: float = 1.0,
        q_progress_scale: float = 1.0,
        q_pos_scale: float = 1.0,
        du_scale: float = 1.0,
        abs_u_scale: float = 1.0,
    ):
        self.q_psi_scale = float(q_psi_scale)
        self.q_progress_scale = float(q_progress_scale)
        self.q_pos_scale = float(q_pos_scale)
        self.du_scale = float(du_scale)
        self.abs_u_scale = float(abs_u_scale)

    def stage_cost(self, **ctx):
        stage = super().stage_cost(**ctx)

        # Rebuild key terms with profile multipliers by applying incremental deltas.
        # This keeps API compatibility while allowing fast profile switching.
        y_t = ctx["y_t"]
        y_ref_t = ctx["y_ref_t"]
        z_t = ctx["z_t"]
        d_psi_ca = ctx["d_psi_ca"]
        d_pos_ca = ctx["d_pos_ca"]

        q_psi = float(ctx["q_psi"])
        q_progress = float(ctx["q_progress"])
        q_pos = float(ctx["q_pos"])
        ref_psi_t = float(ctx["ref_psi_t"])
        ref_px_norm_t = float(ctx["ref_px_norm_t"])
        ref_py_norm_t = float(ctx["ref_py_norm_t"])

        t = int(ctx["t"])
        u_t = ctx["u_t"]
        u_prev = ctx["u_prev"]
        u_prev_step = ctx["u_prev_step"]
        r = ctx["r"]

        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - ref_psi_t), ca.cos(psi_t - ref_psi_t))
        v_forward = y_t[0] * ca.cos(psi_err)

        base_psi = q_psi * (psi_err ** 2)
        base_progress = q_progress * ((v_forward - float(y_ref_t[0])) ** 2)

        stage += (self.q_psi_scale - 1.0) * base_psi
        stage += (self.q_progress_scale - 1.0) * base_progress

        if bool(ctx["add_position_term"]):
            pos = ca.mtimes(d_pos_ca, z_t)
            pos_err_x = pos[0] - ref_px_norm_t
            pos_err_y = pos[1] - ref_py_norm_t
            base_pos = q_pos * (pos_err_x ** 2 + pos_err_y ** 2)
            stage += (self.q_pos_scale - 1.0) * base_pos

        if t == 0:
            du = u_t - u_prev
        else:
            du = u_t - u_prev_step
        base_du = ca.mtimes([du.T, r, du])
        stage += (self.du_scale - 1.0) * base_du

        if bool(ctx["add_abs_u_term"]) and (ctx.get("r_abs") is not None):
            base_abs_u = ca.mtimes([u_t.T, ctx["r_abs"], u_t])
            stage += (self.abs_u_scale - 1.0) * base_abs_u

        return stage


class MPCCPaperCostBuilder(DefaultTrackingCostBuilder):
    """MPCC-style stage cost: contouring/lag + heading + speed + progress."""

    def __init__(
        self,
        *,
        q_contour: float = 8.0,
        q_lag: float = 2.0,
        q_speed: float = 1.4,
        q_heading_scale: float = 1.3,
        q_progress_scale: float = 1.1,
        du_scale: float = 1.0,
        abs_u_scale: float = 1.0,
    ):
        self.q_contour = float(q_contour)
        self.q_lag = float(q_lag)
        self.q_speed = float(q_speed)
        self.q_heading_scale = float(q_heading_scale)
        self.q_progress_scale = float(q_progress_scale)
        self.du_scale = float(du_scale)
        self.abs_u_scale = float(abs_u_scale)

    def stage_cost(self, **ctx):
        opti = ctx["opti"]
        t = int(ctx["t"])
        z_t = ctx["z_t"]
        u_t = ctx["u_t"]
        u_prev = ctx["u_prev"]
        u_prev_step = ctx["u_prev_step"]
        y_t = ctx["y_t"]
        y_ref_t = ctx["y_ref_t"]
        ref_psi_t = float(ctx["ref_psi_t"])
        ref_px_norm_t = float(ctx["ref_px_norm_t"])
        ref_py_norm_t = float(ctx["ref_py_norm_t"])
        d_pos_ca = ctx["d_pos_ca"]
        d_psi_ca = ctx["d_psi_ca"]
        q = ctx["q"]
        r = ctx["r"]
        q_psi = float(ctx["q_psi"])
        q_progress = float(ctx["q_progress"])

        stage = 0

        # Base [v, omega] tracking term
        y_err = y_t - y_ref_t
        stage += ca.mtimes([y_err.T, q, y_err])

        # MPCC-style contouring/lag errors in local Frenet-like frame
        pos = ca.mtimes(d_pos_ca, z_t)
        dx = pos[0] - ref_px_norm_t
        dy = pos[1] - ref_py_norm_t
        e_contour = -ca.sin(ref_psi_t) * dx + ca.cos(ref_psi_t) * dy
        e_lag = ca.cos(ref_psi_t) * dx + ca.sin(ref_psi_t) * dy
        stage += self.q_contour * (e_contour ** 2)
        stage += self.q_lag * (e_lag ** 2)

        # Heading and forward-progress consistency
        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - ref_psi_t), ca.cos(psi_t - ref_psi_t))
        stage += (self.q_heading_scale * q_psi) * (psi_err ** 2)

        v_ref = float(y_ref_t[0])
        stage += self.q_speed * ((y_t[0] - v_ref) ** 2)
        v_forward = y_t[0] * ca.cos(psi_err)
        stage += (self.q_progress_scale * q_progress) * ((v_forward - v_ref) ** 2)

        # Optional soft minimum-speed rule (for DR setting)
        min_speed_rule = ctx.get("min_speed_rule")
        v_slack_t = ctx.get("v_slack_t")
        if min_speed_rule is not None and v_slack_t is not None:
            v_floor = min_speed_rule.floor(v_ref)
            opti.subject_to(y_t[0] + v_slack_t >= v_floor)

        # Input increment / magnitude regularization
        if t == 0:
            du = u_t - u_prev
        else:
            du = u_t - u_prev_step
        stage += self.du_scale * ca.mtimes([du.T, r, du])

        if bool(ctx["add_abs_u_term"]) and (ctx.get("r_abs") is not None):
            stage += self.abs_u_scale * ca.mtimes([u_t.T, ctx["r_abs"], u_t])

        return stage

    def collect_stage_diagnostics(self, diag_terms, **ctx):
        super().collect_stage_diagnostics(diag_terms, **ctx)

        z_t = ctx["z_t"]
        y_t = ctx["y_t"]
        y_ref_t = ctx["y_ref_t"]
        ref_psi_t = float(ctx["ref_psi_t"])
        ref_px_norm_t = float(ctx["ref_px_norm_t"])
        ref_py_norm_t = float(ctx["ref_py_norm_t"])
        d_pos_ca = ctx["d_pos_ca"]
        d_psi_ca = ctx["d_psi_ca"]
        q_psi = float(ctx["q_psi"])
        q_progress = float(ctx["q_progress"])

        pos = ca.mtimes(d_pos_ca, z_t)
        dx = pos[0] - ref_px_norm_t
        dy = pos[1] - ref_py_norm_t
        e_contour = -ca.sin(ref_psi_t) * dx + ca.cos(ref_psi_t) * dy
        e_lag = ca.cos(ref_psi_t) * dx + ca.sin(ref_psi_t) * dy
        psi_t = ca.mtimes(d_psi_ca, z_t)[0]
        psi_err = ca.atan2(ca.sin(psi_t - ref_psi_t), ca.cos(psi_t - ref_psi_t))
        v_ref = float(y_ref_t[0])
        v_forward = y_t[0] * ca.cos(psi_err)

        _append_diag(diag_terms, "e_contour", e_contour)
        _append_diag(diag_terms, "e_lag", e_lag)
        _append_diag(diag_terms, "cost_contour", self.q_contour * (e_contour ** 2))
        _append_diag(diag_terms, "cost_lag", self.q_lag * (e_lag ** 2))
        _append_diag(diag_terms, "cost_speed", self.q_speed * ((y_t[0] - v_ref) ** 2))
        _append_diag(diag_terms, "cost_heading_mpcc", (self.q_heading_scale * q_psi) * (psi_err ** 2))
        _append_diag(diag_terms, "cost_progress_mpcc", (self.q_progress_scale * q_progress) * ((v_forward - v_ref) ** 2))


class MPCCPaperCVARCostBuilder(MPCCPaperCostBuilder):
    """MPCC-style cost with CVaR tail-risk penalty over horizon tracking loss."""

    def __init__(
        self,
        *,
        cvar_alpha: float = 0.90,
        cvar_lambda: float = 6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_lambda = float(cvar_lambda)

    def stage_cost(self, **ctx):
        stage = super().stage_cost(**ctx)

        risk_terms = ctx.get("risk_terms")
        if risk_terms is not None:
            # Use MPCC-style geometry loss as per-step risk sample l_t.
            z_t = ctx["z_t"]
            d_pos_ca = ctx["d_pos_ca"]
            ref_px_norm_t = float(ctx["ref_px_norm_t"])
            ref_py_norm_t = float(ctx["ref_py_norm_t"])
            ref_psi_t = float(ctx["ref_psi_t"])
            d_psi_ca = ctx["d_psi_ca"]

            pos = ca.mtimes(d_pos_ca, z_t)
            dx = pos[0] - ref_px_norm_t
            dy = pos[1] - ref_py_norm_t
            e_contour = -ca.sin(ref_psi_t) * dx + ca.cos(ref_psi_t) * dy
            e_lag = ca.cos(ref_psi_t) * dx + ca.sin(ref_psi_t) * dy

            psi_t = ca.mtimes(d_psi_ca, z_t)[0]
            psi_err = ca.atan2(ca.sin(psi_t - ref_psi_t), ca.cos(psi_t - ref_psi_t))

            l_t = 1.2 * (e_contour ** 2) + 0.4 * (e_lag ** 2) + 0.6 * (psi_err ** 2)
            risk_terms.append(l_t)

        return stage

    def finalize_cost(self, **ctx):
        opti = ctx["opti"]
        risk_terms = ctx.get("risk_terms")
        horizon = int(ctx.get("horizon", 0))
        diag_terms = ctx.get("diag_terms")

        if not risk_terms or horizon <= 0:
            return 0

        # Standard CVaR epigraph:
        # CVaR_alpha(l) = eta + 1/((1-alpha)T) * sum xi_t
        # s.t. xi_t >= l_t - eta, xi_t >= 0
        eta = opti.variable()
        xi = opti.variable(horizon)
        opti.subject_to(xi >= 0)
        for t in range(horizon):
            opti.subject_to(xi[t] >= risk_terms[t] - eta)

        denom = max(1e-6, (1.0 - self.cvar_alpha) * float(horizon))
        cvar = eta + (1.0 / denom) * ca.sum1(xi)
        if diag_terms is not None:
            diag_terms["cost_cvar"] = cvar
            diag_terms["risk_eta"] = eta
            diag_terms["risk_xi_mean"] = ca.sum1(xi) / horizon
        return self.cvar_lambda * cvar


TRACKING_COST_PROFILES = {
    "default": DefaultTrackingCostBuilder,
    # 更强调中心线/姿态/平滑，抑制过度激进推进。
    "tracking-first": lambda: WeightedTrackingCostBuilder(
        q_psi_scale=3.0,
        q_progress_scale=0.05,
        q_pos_scale=9.0,
        du_scale=2.0,
        abs_u_scale=1.8,
    ),
    # 更强调前进效率，适度放松位置/平滑保守性。
    "progress-first": lambda: WeightedTrackingCostBuilder(
        q_psi_scale=0.9,
        q_progress_scale=1.7,
        q_pos_scale=0.8,
        du_scale=0.9,
        abs_u_scale=0.9,
    ),
    # 高引用轨迹跟踪论文常见结构：contouring/lag + heading + speed + progress
    "mpcc-paper": MPCCPaperCostBuilder,
    # 在mpcc-paper上增加尾部风险项：E[J] + lambda * CVaR_alpha(loss)
    "mpcc-paper-cvar": MPCCPaperCVARCostBuilder,
    # 稳定优先：强contour/lag与姿态，极弱进度项，避免大偏航后继续“硬推前进”。
    "stabilize-first": lambda: MPCCPaperCostBuilder(
        q_contour=22.0,
        q_lag=6.0,
        q_speed=0.6,
        q_heading_scale=2.0,
        q_progress_scale=0.05,
        du_scale=1.8,
        abs_u_scale=1.8,
    ),
}


def resolve_tracking_cost_builder(
    builder: Optional[Union[str, DefaultTrackingCostBuilder]] = None,
    profile: str = "default",
):
    """Resolve a builder name/object to a cost builder instance."""
    if builder is None:
        if profile not in TRACKING_COST_PROFILES:
            raise ValueError(
                f"Unknown cost profile: {profile}. "
                f"Available: {', '.join(TRACKING_COST_PROFILES.keys())}"
            )
        factory = TRACKING_COST_PROFILES[profile]
        return factory() if callable(factory) else factory
    if isinstance(builder, str):
        if builder in TRACKING_COST_PROFILES:
            factory = TRACKING_COST_PROFILES[builder]
            return factory() if callable(factory) else factory
        raise ValueError(f"Unknown tracking cost builder: {builder}")
    if hasattr(builder, "stage_cost"):
        return builder
    raise TypeError(
        "cost_builder must be None, a known profile name, "
        "or an object with stage_cost(...)"
    )
