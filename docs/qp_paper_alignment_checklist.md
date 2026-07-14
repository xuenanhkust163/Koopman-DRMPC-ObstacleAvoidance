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