"""
Central configuration file for the K-DRMPC paper implementation.
All paper hyperparameters are defined here.
"""

import os  # Operating system interfaces for path operations
import numpy as np  # NumPy for numerical computing and array operations

# ============================================================
# Project paths
# ============================================================
# Repository root is the parent of src/.
EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = EXPERIMENT_ROOT

# Output directory paths
OUTPUT_DIR = os.path.join(EXPERIMENT_ROOT, "_output")  # Root output directory
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")  # Model output directory
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")  # Simulation result directory
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")  # Figure output directory
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")  # Table output directory

# Training data file paths (within the project)
DATA_NPZ_PATH = os.path.join(MODEL_DIR, "training_data.npz")
NORM_JSON_PATH = os.path.join(MODEL_DIR, "norm_params.json")

# Export mode: use simulation plus animation by default; do not export static figures for now.
EXPORT_STATIC_FIGURES = False
EXPORT_ANIMATION = True
ANIMATION_FPS = 12

# ============================================================
# State and control dimensions
# ============================================================
N_X = 5       # State dimension: [px, py, psi, v, omega] = [x-position, y-position, heading, speed, yaw rate]
N_U = 2       # Control input dimension: [a, delta] = [acceleration, steering angle]
N_Z = 32      # Koopman latent-space dimension (lifted linear-system state dimension)
N_W = 5       # Disturbance dimension (same as the state dimension)

# Shared state indices used throughout the repository to avoid hard-coded values.
IDX_PX = 0
IDX_PY = 1
IDX_PSI = 2
IDX_V = 3
IDX_OMEGA = 4

# ============================================================
# Deep Koopman network architecture (paper Section 3.3)
# ============================================================
ENCODER_LAYERS = [N_X, 64, 128, 64, N_Z]    # Encoder layer sizes: 5 -> 64 -> 128 -> 64 -> 32
DECODER_LAYERS = [N_Z, 64, 32, N_X]         # Decoder layer sizes: 32 -> 64 -> 32 -> 5
ACTIVATION = "relu"  # Activation function: ReLU

# ============================================================
# Training hyperparameters (paper Section 3.4)
# ============================================================
BATCH_SIZE = 256  # Number of samples per training batch
EPOCHS = 500  # Number of epochs (full passes through the dataset)
LEARNING_RATE = 1e-3  # Initial learning rate: 0.001
LR_PATIENCE = 20  # Reduce the learning rate after 20 epochs without validation-loss improvement
LR_FACTOR = 0.5  # Learning-rate decay factor: multiply by 50% each time
EARLY_STOP_PATIENCE = 50  # Stop training after 50 epochs without validation-loss improvement
VAL_SPLIT = 0.1  # Validation-set fraction: 10% of the data

# Loss weights (paper Section 3.3.5)
LAMBDA_RECON = 1.0  # Reconstruction loss: encourage accurate reconstruction of the original state
LAMBDA_LINEAR = 1.0  # Linear evolution loss: encourage accurate linear evolution in Koopman space
LAMBDA_PRED = 2.0  # Multi-step prediction loss: improve long-term prediction
LAMBDA_PHYSICS = 1.0  # Physics-consistency loss: encourage A to learn correct position updates
LAMBDA_REG_HIGH_DIM = 2.0  # High-dimensional regularization: penalize high-dimensional columns in physical rows of A to prevent feature leakage into v/psi predictions
LAMBDA_SPECTRAL = 1.5          # Spectral-radius penalty for A (enforce spectral_radius < 1.0)

# Normalization parameters computed from training data for de-normalization in the physics-consistency loss.
# These values come from norm_params.json and are hard-coded to avoid repeatedly reading the file during training.
PX_STD = 481.95896444159337   # Standard deviation of px
PY_STD = 273.6795009788262    # Standard deviation of py

# Multi-step prediction horizon for the training loss
K_PRED = 25  # Number of forward prediction steps

# Ridge-regression regularization parameter for matrix D
GAMMA_RIDGE = 1e-4  # Regularization strength to reduce overfitting and improve numerical stability

# Data subsampling (original Ts=0.01s -> paper dt=0.1s)
ORIGINAL_TS = 0.01  # Original data sampling interval [s]
SUBSAMPLE_RATE = 10  # Select every tenth sample to obtain an effective interval of 0.1s

# ============================================================
# Vehicle parameters (paper Section 2, Table 5)
# ============================================================
DT = 0.1                   # Sampling time [s]
CONTROL_UPDATE_INTERVAL = 1  # Solve the MPC once every N simulation steps
L_WHEELBASE = 2.6          # Vehicle wheelbase [m]
V_MIN = 0.0                # Original paper Table 5 value: minimum speed [m/s]
V_MAX = 40.0               # Maximum speed [m/s]
A_MIN = -5.0               # Original paper Table 5 value: maximum deceleration [m/s^2]
A_MAX = 3.0                # Original paper Table 5 value: maximum acceleration [m/s^2]
DELTA_MAX = np.pi / 4      # Original paper Table 5 value: maximum steering angle = 45 degrees [rad]
DELTA_RATE_MAX = 0.5       # Original paper Table 5 value: maximum steering rate [rad/s]
D_SAFE = 1.5               # Safety margin [m] (starts obstacle avoidance earlier and smooths steering)
TRACK_HALF_WIDTH = 12.0    # Track half-width W/2 [m] (consistent with the PLUS version)
TRACK_BOUNDARY_SLACK_PENALTY = 500.0   # Track-boundary slack penalty, comparable to obstacle penalties, to prevent leaving the track during avoidance

# Lateral acceleration limits for speed-profile planning
A_LAT_MAX = 4.0            # Maximum lateral acceleration [m/s^2] (limits cornering speed)
REF_SPEED_SCALE = 0.25     # Straight-line target is about 10 m/s; lowers cornering demand and reduces lateral loss of control during avoidance
REF_ACCEL_MAX = 0.4        # Maximum acceleration in the reference speed profile [m/s^2]
REF_DECEL_MAX = 1.2        # Maximum deceleration in the reference speed profile [m/s^2]

# ============================================================
# MPC parameters (paper Section 6, Table 5)
# ============================================================
T_HORIZON = 40             # MPC prediction horizon (40 steps = 4 s; the normalization bug is fixed)

# Cost-function weights (paper Section 6.1.2)
Q_WEIGHTS = np.diag([6.0, 1.0])     # Q: speed tracking weight 6.0; yaw-rate weight reduced to 1.0 to avoid over-tracking omega when psi and omega conflict
R_WEIGHTS = np.diag([0.5, 1.5])     # R: acceleration-change penalty reduced to 0.5 for more flexible throttle; steering penalty remains 1.5

# Shared trajectory-tracking tuning parameters for K-MPC and K-DRMPC
Q_PSI_TRACK = 15.0                  # Heading-error weight, increased to improve heading tracking and prevent lateral-error divergence
Q_PROGRESS_TRACK = 0.15             # Forward-progress weight, increased to encourage movement along the track
Q_POS_TRACK = 5000.0               # Position-error weight compensating for normalized coordinate scale (px_std ~= 80)
POSITION_TERM_INTERVAL = 1          # Frequency of position terms (1 = every step)
R_ABS_A = 0.5                       # Lower acceleration penalty to encourage tracking the reference speed
R_ABS_DELTA = 2.0                   # Absolute steering-angle penalty
Q_TERMINAL_HEADING = 15.0           # Terminal heading-error weight (reduced)
Q_TERMINAL_POS = 30.0               # Terminal position-error weight

# ============================================================
# Distributionally robust parameters (paper Section 6.1.3)
# ============================================================
ENABLE_DISTURBANCE = False    # Global noise toggle: False disables online disturbances and empirical disturbance samples
N_DISTURBANCE_SAMPLES = 100    # Number of historical disturbance samples used to build the Wasserstein ambiguity set
THETA_WASSERSTEIN = 0.25       # Wasserstein-ball radius controlling the ambiguity-set size (uncertainty level)
NOMINAL_SIGMA = 0.5            # Nominal disturbance standard deviation in training data (single source of truth)
                               # Paper setting scale references:
                               #   Setting A (Nominal,  x1.0) = 0.5
                               #   Setting B (Moderate, x1.5) = 0.75
                               #   Setting C (Large,    x2.0) = 1.0
                               # Training and simulation defaults derive from this value; retrain the Koopman model if it changes.
EPSILON_CVAR = 0.05             # CVaR risk level: 0.05 considers the worst 5% of cases

# ============================================================
# Sensitivity-analysis ranges (paper Section 6.7)
# ============================================================
THETA_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]  # Candidate Wasserstein radii
EPSILON_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30]  # Candidate CVaR risk levels
SIGMA_VALUES = [0.01, 0.05, 0.10, 0.15]  # Candidate noise standard deviations

# ============================================================
# Simulation parameters
# ============================================================
MAX_SIM_STEPS = 3000       # Maximum simulation steps per lap (3000 steps x 0.1 s = 300 s)
IPOPT_MAX_ITER = 500       # Maximum number of IPOPT solver iterations
IPOPT_TOL = 1e-6           # IPOPT solver convergence tolerance
IPOPT_PRINT_LEVEL = 0      # IPOPT output level: 0 suppresses detailed output

# ============================================================
# Obstacle configuration
# ============================================================
ENABLE_OBSTACLES = False  # Global obstacle toggle: False disables obstacles in simulation and plots
OBSTACLE_RADIUS = 3.0      # Default obstacle radius [m]
OBSTACLE_LAYOUT_MODE = "edge"  # Obstacle layout: edge = near the boundary; center = closer to the centerline
OBSTACLE_EDGE_MARGIN = 0.2  # Small gap between edge-mode obstacles and the track boundary [m]
VEHICLE_RADIUS = 2.0       # Approximate vehicle footprint radius [m] (used for collision checking)

# Predefined obstacle coordinates on the track, kept away from the start.
# Legacy fixed obstacle coordinates; track-specific layouts are defined by each track.
OBSTACLE_POSITIONS = [
    (100.0, 50.0),    # Upper-right section of the track (about 25% progress)
    (-50.0, 120.0),    # Upper section of the track (about 50% progress)
    (-120.0, -50.0),   # Lower-left section of the track (about 75% progress)
]

# ============================================================
# Plot configuration
# ============================================================
PLOT_TRACK_HALF_WIDTH = 12.0  # Track-boundary half-width for visualization [m]; matches the constraint width but is used only for plotting
FIGURE_DPI = 300  # Figure resolution: 300 DPI for high-quality printing
FIGURE_FORMAT = "pdf"  # Figure format: PDF vector graphics, suitable for papers
# Method colors for distinguishing methods in plots
METHOD_COLORS = {
    "LMPC": "#1f77b4",    # Blue: linear MPC
    "NMPC": "#ff7f0e",    # Orange: nonlinear MPC
    "K-MPC": "#2ca02c",   # Green: Koopman MPC
    "K-DRMPC": "#d62728", # Red: Koopman distributionally robust MPC (our method)
}
# Method labels for plot legends
METHOD_LABELS = {
    "LMPC": "LMPC",                    # Linear MPC
    "NMPC": "NMPC",                    # Nonlinear MPC
    "K-MPC": "K-MPC",                  # Koopman MPC
    "K-DRMPC": "K-DRMPC (Ours)",      # Koopman distributionally robust MPC (our method)
}
