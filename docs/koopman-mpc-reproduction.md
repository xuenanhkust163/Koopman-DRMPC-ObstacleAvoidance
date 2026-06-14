# Plan: Reproduce K-DRMPC Paper in Experiment/

## Context

The paper "Distributionally Robust Model Predictive Control with Disturbance Uncertainty Adaptiveness for Obstacle Avoidance of Autonomous Driving" proposes a Koopman-based DR-MPC (K-DRMPC) framework. The existing codebase has data preprocessing and a basic Deep Koopman trainer with a **different architecture** from the paper. The MPC simulation is marked as incomplete. This plan implements the full paper algorithm framework in `Experiment/`, including retraining the Koopman model per the paper's exact specs, implementing 4 MPC controllers (LMPC, NMPC, K-MPC, K-DRMPC), running simulations on 2 tracks, and generating all figures/tables.

## Directory Structure

```
Experiment/
  __init__.py
  config.py                         # All hyperparameters from paper
  data/
    __init__.py
    data_loader.py                  # Load existing data, subsample to dt=0.1s
  model/
    __init__.py
    koopman_network.py              # Paper-exact DNN (encoder x-only, A/B/C)
    koopman_trainer.py              # Algorithm 1 training loop
    projection.py                   # Ridge regression for D matrix
  tracks/
    __init__.py
    base_track.py                   # Abstract track interface
    lusail_track.py                 # Lusail Circuit approximation (~5.38km)
    custom_track.py                 # Custom winding track (~3.2km)
  vehicle/
    __init__.py
    bicycle_model.py                # Kinematic bicycle model (plant + linearization)
  controllers/
    __init__.py
    mpc_common.py                   # Compact matrices, LQR, PyTorch->CasADi utils
    lmpc_controller.py              # Linear MPC baseline
    nmpc_controller.py              # Nonlinear MPC baseline
    kmpc_controller.py              # Koopman MPC (no DR)
    kdrmpc_controller.py            # K-DRMPC (paper's main contribution)
  disturbance/
    __init__.py
    disturbance_generator.py        # Mixture-of-Gaussians sampling
    wasserstein.py                  # Wasserstein ambiguity set + CVaR constraints
  simulation/
    __init__.py
    simulator.py                    # Closed-loop simulation engine
    metrics.py                      # Performance metric computation
  visualization/
    __init__.py
    plot_trajectories.py            # Figs 6, 7: trajectory comparisons
    plot_tsne.py                    # Fig 5: t-SNE latent space
    plot_tables.py                  # Tables 6, 9-15: formatted output
  run_training.py                   # Entry: train Deep Koopman model
  run_simulation.py                 # Entry: run all 4 methods on both tracks
  run_analysis.py                   # Entry: generate all figures and tables
  run_all.py                        # Master entry: train -> simulate -> analyze
```

Output directory: `Experiment/_output/` (model checkpoints, simulation results, figures)

## Implementation Steps

### Step 1: config.py
Central configuration with all paper constants:
- Dimensions: `N_X=5, N_U=2, N_Z=32, N_W=5`
- Network: encoder `[5,64,128,64,32]`, decoder `[32,64,32,5]`, ReLU
- Training: `BATCH_SIZE=256, EPOCHS=500, LR=1e-3, lambda1=1.0, lambda2=1.0, lambda3=0.5, K_PRED=10`
- Vehicle: `L=2.6m, DT=0.1s`
- MPC: `T=40, Q=diag(1.5,3.0), R=diag(1.5,3.0), V_MIN=0, V_MAX=40, A_MIN=-5, A_MAX=3, DELTA_MAX=pi/4, DELTA_RATE_MAX=0.5`
- DR: `D_SAFE=0.5m, N_SAMPLES=100, THETA=0.05, EPSILON=0.1`
- Sensitivity sweeps: `THETA_VALUES=[0.00,0.02,0.05,0.10,0.20]`, `EPSILON_VALUES=[0.01,0.05,0.10,0.20,0.30]`, `SIGMA_VALUES=[0.01,0.05,0.10,0.15]`
- Paths: to existing data `../_output/_data_process/training_data.npz` and norm params

### Step 2: data/data_loader.py
- Load existing `training_data.npz` (already preprocessed 5D states, normalized px/py)
- **Critical**: Existing data has Ts=0.01s, paper needs dt=0.1s. Reconstruct full trajectory, subsample every 10 steps, average controls over 10-step windows
- Build sliding-window sequences of length K_PRED=10 for multi-step prediction loss
- Return PyTorch DataLoaders (train 90% / val 10%)
- Expose `normalize(x)` and `denormalize(x)` using stored norm params

### Step 3: model/koopman_network.py
Paper-exact Deep Koopman architecture (**differs from existing code**):

| Aspect | Existing Code | Paper Spec |
|--------|--------------|------------|
| Encoder input | [x; u] (7D) | x only (5D) |
| Encoder layers | 7->128->128->48 | 5->64->128->64->32 |
| Decoder layers | 48->128->128->5 | 32->64->32->5 |
| Linear matrices | K(48x48), L(48x2) | A(32x32), B(32x2), C(32x5) |
| Disturbance matrix | None | C in R^{32x5} |

Class `DeepKoopmanPaper`:
- `encode(x) -> z`: x only input (not [x;u])
- `decode(z) -> x_hat`
- `linear_step(z, u, w=None) -> z_next`: z_next = A@z + B@u + C@w
- `forward(x, u, x_next) -> (x_recon, x_next_recon, z, z_next_linear, z_next_true)`: returns all intermediates for loss

Loss function (3 components per paper Section 3.3.5):
- L_recon = MSE(decode(encode(x)), x) + MSE(decode(encode(x_next)), x_next)
- L_linear = MSE(encode(x_next), A @ encode(x) + B @ u)
- L_pred = Multi-step: roll forward K steps via A,B in latent space, decode, compare to true
- Total = 1.0 * L_recon + 1.0 * L_linear + 0.5 * L_pred

### Step 4: model/koopman_trainer.py
Implements Algorithm 1:
- Adam optimizer, lr=1e-3, ReduceLROnPlateau (patience=20, factor=0.5)
- Use sliding-window batches: each window provides (x_0..x_K, u_0..u_{K-1})
- First transition gives single-step data for L_recon and L_linear
- Full window gives multi-step data for L_pred
- Save best model checkpoint by validation loss
- Log per-component losses for Table 6

### Step 5: model/projection.py
After training, compute projection matrix D in R^{2x32}:
- Collect z_i = encode(x_i) for all training data
- Collect y_i = [v_i, omega_i] (indices 2,4 of x)
- D = Y @ Z^T @ inv(Z @ Z^T + gamma*I) via ridge regression (gamma=1e-4)
- Save D alongside model checkpoint

### Step 6: vehicle/bicycle_model.py
Kinematic bicycle model (5-state):
```
px_dot = v * cos(psi)
py_dot = v * sin(psi)
v_dot = a
psi_dot = v * tan(delta) / L
omega = v * tan(delta) / L
```
Functions:
- `discrete_step(x, u, dt)`: RK4 integration
- `linearize(x_op, u_op, dt) -> (A_d, B_d)`: Jacobian linearization + ZOH discretization
- `casadi_dynamics(x_sym, u_sym)`: CasADi symbolic version for NMPC

### Step 7: tracks/base_track.py + lusail_track.py + custom_track.py
Base class interface: `get_centerline()`, `get_reference()`, `get_obstacles()`, `get_curvature()`, `closest_point()`

**Lusail Circuit**: ~40 waypoints approximating the 16-turn, 5.38km F1 circuit. Periodic cubic spline. 4 static obstacles at turns T1, T6, T9, T16.

**Custom Winding Track**: Parametric sinusoidal curves, 3.2km. `y(s) = A1*sin(2*pi*s/P1) + A2*sin(2*pi*s/P2)`. 3 static obstacles at apex points.

Speed profile from curvature: `v_ref = min(V_MAX, sqrt(a_lat_max / kappa))`

### Step 8: controllers/mpc_common.py
Shared utilities:
- `build_compact_matrices(A, B, T)`: Stacked A_cal, B_cal block matrices
- `build_lqr_gain(A, B, Q, R)`: DARE solve for feedback gain K
- `build_closed_loop_matrices(A_cal, B_cal, C_cal, K)`: Compute A_tilde, B_tilde, C_tilde
- `pytorch_to_casadi(model)`: Extract weights, rebuild encoder/decoder as CasADi MX expressions. **Verify numerically** against PyTorch output.
- `build_qp_hessian(...)`: Compute H, f matrices for QP formulation

### Step 9: controllers/lmpc_controller.py
At each step:
1. Linearize bicycle model at current operating point
2. Build QP with linearized dynamics, tracking cost on [v, omega], input constraints
3. Linearized obstacle avoidance constraints
4. Solve via CasADi qpsol or IPOPT
5. Apply first control

### Step 10: controllers/nmpc_controller.py
Full nonlinear MPC using CasADi Opti:
1. Define state/control trajectories as decision variables
2. Nonlinear dynamics via RK4 (CasADi symbolic)
3. Quadratic tracking cost on [v, omega]
4. Nonlinear obstacle avoidance: ||pos - obs|| >= r + d_safe
5. Solve NLP with IPOPT, warm-start from previous solution

### Step 11: controllers/kmpc_controller.py
Koopman MPC (deterministic, no DR):
1. Encode current state: z_0 = encode(x_0)
2. Compact dynamics: z = A_cal*z_0 + B_cal*u (linear)
3. Cost in latent space: ||D*z - y_ref||^2_Q + ||delta_u||^2_R
4. Decode for obstacle constraints (makes it NLP due to nonlinear decoder)
5. Solve via CasADi Opti + IPOPT

### Step 12: controllers/kdrmpc_controller.py (Paper's main contribution)
Extends K-MPC with Wasserstein DR-CVaR constraints:
1. Maintain N=100 disturbance samples
2. For each obstacle j, each timestep t, add reformulated CVaR constraint:
   - Auxiliary variables: lambda >= 0, s_i >= 0 (i=1..N)
   - s_i >= -l(x_t, w_i) + lambda * ||w_i||
   - lambda * theta + 1/(epsilon*N) * sum(s_i) <= 0
   - Safety margin: l = ||decoded_pos - obs_j|| - d_safe
3. Full NLP with ~16K+ variables, solved via IPOPT with warm-start
4. C matrix propagates disturbance through lifted dynamics

### Step 13: disturbance/disturbance_generator.py + wasserstein.py
- Generator: Mixture of 3 Gaussians, zero-mean, covariance sigma*I with small component offsets
- Wasserstein: Build CasADi constraints implementing the CVaR dual reformulation
- Post-hoc CVaR margin computation for metrics

### Step 14: simulation/simulator.py + metrics.py
Closed-loop simulation engine:
1. Get state -> find track reference -> call controller.solve() -> sample disturbance -> propagate plant -> record
2. Track: lap time, tracking error (RMS lateral + speed), max speed, constraint violations, CVaR margin, solve time

### Step 15: visualization/ (all plots)
- **Fig 5** (plot_tsne.py): t-SNE of 32D latent vectors, colored by driving regime
- **Fig 6** (plot_trajectories.py): 4-method trajectory comparison on Lusail Circuit
- **Fig 7** (plot_trajectories.py): 4-method trajectory comparison on Custom Winding Track
- **Tables 6, 9-15** (plot_tables.py): Formatted console output + optional LaTeX

### Step 16: Entry scripts
- `run_training.py`: Train model, compute D, save checkpoints
- `run_simulation.py`: Run all 4 methods on both tracks, save results
- `run_analysis.py`: Generate all figures and tables from saved results
- `run_all.py`: Sequential pipeline: train -> simulate -> analyze

## Key Technical Challenges

1. **Data timestep mismatch**: Existing Ts=0.01s, paper needs 0.1s. Subsample x10, average controls.
2. **PyTorch -> CasADi translation**: Must manually extract weights and rebuild NN as CasADi MX. Verify numerically.
3. **Normalization consistency**: px/py normalized in training data. Must normalize/denormalize correctly at every model boundary.
4. **Heading angle wrapping**: psi wraps around 2pi on closed tracks. Use atan2(sin,cos) for angular differences.
5. **LQR stabilizability**: Learned (A,B) may not be stabilizable. Add fallback to zero gain.
6. **IPOPT convergence**: K-DRMPC has ~16K variables. Requires good warm-starts and variable scaling.
7. **C matrix training**: Learn C as nn.Parameter during training. Residual w provides implicit supervision.

## Critical Files to Create/Modify

All new files in `Experiment/`:
- `config.py` - foundation for all other modules
- `model/koopman_network.py` - paper-exact architecture (most critical)
- `controllers/kdrmpc_controller.py` - paper's main algorithmic contribution
- `controllers/mpc_common.py` - PyTorch-to-CasADi bridge
- `simulation/simulator.py` - produces all experimental results

Existing files read (not modified):
- `_output/_data_process/training_data.npz` - training data
- `_output/_data_process/training_data_norm_params.json` - normalization

## Verification Plan

1. **Model training**: Run `run_training.py`. Check that all 3 loss components decrease. Validate reconstruction error < 0.01 MSE. Compare 1-step and 10-step prediction accuracy.
2. **Projection matrix D**: Verify that D@z approximates [v, omega] with R^2 > 0.95 on validation data.
3. **PyTorch-CasADi consistency**: For 100 random inputs, verify CasADi encoder/decoder output matches PyTorch within 1e-6.
4. **Individual controllers**: Test each controller on a short straight segment (no obstacles) to verify basic tracking.
5. **Full simulation**: Run `run_simulation.py` on both tracks. Verify all 4 methods complete at least 1 lap without divergence.
6. **Figures and tables**: Run `run_analysis.py`. Visually inspect trajectory plots. Check that K-DRMPC shows 0% constraint violations and competitive lap times.
7. **Sensitivity analysis**: Verify monotonic relationship: larger theta -> more conservative (slower but safer).
