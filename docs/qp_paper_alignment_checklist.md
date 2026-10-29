# QP Paper Alignment Checklist

This note summarizes which parts of `controllers/kdrmpc_paper_qp_controller.py` are aligned with the paper's convex QP theory and which parts are engineering safeguards added for numerical robustness and closed-loop stability.

## Paper-Core Alignment

1. Convex QP formulation
   - The controller is explicitly implemented as a convex QP rather than the NLP/IPOPT paper controller variant.
   - See `controllers/kdrmpc_paper_qp_controller.py`.

2. Affine Koopman prediction
   - The prediction backbone uses affine Koopman dynamics of the form `z_{t+1} = A z_t + B u_t`, which is the key structural requirement for the convex QP formulation.

3. Quadratic objective
   - The main tracking, control increment, control magnitude, and terminal penalties are all quadratic.

4. Preselected obstacle half-space model
   - Obstacles are represented through a preselected separating half-space rather than a nonconvex geometric collision model inside the optimizer.

5. Wasserstein/DRCC obstacle risk constraints
   - The QP includes the `lambda` and sample-wise `s_i` auxiliary variables together with `theta` and `epsilon`, matching the paper-style Wasserstein distributionally robust chance-constraint structure.

6. Affine constraint structure
   - Input bounds, steering-rate bounds, track-boundary constraints, and obstacle DRCC constraints are all written in affine form, preserving QP convexity.

## Engineering Safeguards

1. `QPEngineeringConfig`
   - This configuration layer isolates engineering-side tuning knobs from the paper-core formulation.

2. Legacy compatibility mapping
   - `_resolve_engineering_config(...)` keeps older constructor overrides working and is purely an interface-layer convenience.

3. `linear_physical_consistency`
   - This adds a more physically sensible linearized state update for the physical coordinates and headings, which improves closed-loop behavior but is not part of the minimal paper statement.

4. Fallback recovery controller
   - `_recovery_control(...)` is used when the QP solve fails, preventing persistent solver-failure stalling.

5. Heading slack and heading envelope
   - `heading_slack`, `heading_slack_penalty`, and the bounded heading-error envelope are engineering protections against turn-around and heading divergence.

6. Lateral centering penalty
   - The extra centerline-anchoring penalty in physical meters is added to reduce long-run lateral drift.

7. Lateral adaptive speed cap
   - The additional affine speed cap that tightens with growing lateral error is an engineering stabilization device.

8. Slack caps and penalty scaling
   - `track_slack_max`, `heading_slack_max_deg`, and `hard_constraint_slack_scale` are numerical robustness parameters, not paper-core theoretical ingredients.

9. Progress reward and terminal weight scaling
   - `progress_reward_weight` and `terminal_weight_scale` are implementation-side tuning knobs unless they are explicitly present in the manuscript.

## Recommended Interpretation

The current QP controller should be described as:

"Paper-core aligned convex QP controller with a small set of engineering safeguards for numerical robustness and closed-loop stability."

That is the most accurate wording if the goal is to claim theoretical alignment without overstating that the implementation is a line-by-line transcription of the manuscript.

## Disturbance Experiment Protocol

The manuscript models the physical disturbance additively in the plant transition and then represents its combined lifted effect as `hat{w}`. Use `run_obstacle_free_lap.py --process-disturbance` for that experiment path. The existing `--measurement-noise` option perturbs only controller observations and is a diagnostic mode, not the manuscript disturbance protocol.

The runner's paper-QP defaults are `theta=0.10` and `epsilon=0.10`. The nominal disturbance scale is `NOMINAL_SIGMA=0.5`; shifted plant settings are `0.75` and `1.0`. The robust controller's empirical ambiguity samples remain at the nominal scale while the plant disturbance is shifted, as required for the distribution-shift comparison.

The state-space scaling currently shared by plant disturbance injection and the QP disturbance samples is `[0.1, 0.1, 0.01, 0.05, 0.01]` in state order `[px, py, psi, v, omega]`. The manuscript should report these physical units/scales and define how the five-dimensional physical disturbance maps to the lifted disturbance `hat{w}`. This is an underspecification to resolve in the text; no mathematical error in the additive disturbance model has been identified.

Earlier runs made with `--measurement-noise` should be retained as sensor-noise diagnostics, but not reported as reproductions of the paper's disturbance-shift experiments.