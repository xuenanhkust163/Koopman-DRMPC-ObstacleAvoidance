"""
Data loading and preprocessing for Deep Koopman model training.

This module handles:
1. Loading preprocessed training data (in .npz format)
2. Subsampling data from the original dt=0.01s to the paper's required dt=0.1s
3. Creating sliding-window sequences for multi-step prediction loss calculation
4. Building PyTorch DataLoaders with training/validation splits

Data pipeline:
Original data (Ts=0.01s) -> reconstruct full trajectories -> subsample (dt=0.1s) ->
split into sliding windows -> train/validation split -> DataLoader
"""

import numpy as np  # NumPy for numerical computing and multidimensional arrays
import json  # JSON support for loading normalization parameters
import os  # Operating system interfaces for file and path operations
import torch  # PyTorch deep learning framework
from torch.utils.data import DataLoader, TensorDataset  # PyTorch data-loading utilities

import sys  # System module for modifying the Python path
# Add the project root to the system path so sibling modules such as config.py can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data-loading and training parameters from the configuration.
from config import (
    DATA_NPZ_PATH,  # Path to the training data .npz file
    NORM_JSON_PATH,  # Path to the normalization parameter .json file
    SUBSAMPLE_RATE,  # Subsampling rate (10 selects every tenth sample, 0.01s -> 0.1s)
    K_PRED,  # Multi-step prediction horizon (sliding-window length)
    VAL_SPLIT,  # Validation-set fraction (0.1 means 10% of the data)
    BATCH_SIZE  # Batch size (number of samples per training batch)
)


def _legacy_to_canonical_state_order(x):
    """Convert [px, py, v, psi, omega] -> [px, py, psi, v, omega]."""
    x_new = x.copy()
    x_new[..., 2] = x[..., 3]
    x_new[..., 3] = x[..., 2]
    return x_new


def load_norm_params(json_path=NORM_JSON_PATH):
    """
    Load normalization parameters (means and standard deviations for px and py).

    The normalization parameters are computed and saved during preprocessing. They are used to:
    1. Standardize position coordinates (px, py) before training
    2. De-standardize position coordinates after inference
    3. Keep state-variable ranges comparable and improve training stability

    Args:
        json_path: Path to the normalization parameter JSON file.
                   Defaults to NORM_JSON_PATH from config.py.

    Returns:
        params: Dictionary containing 'px_mean', 'px_std', 'py_mean', and 'py_std'.
    """
    # Open the JSON file in read-only mode.
    with open(json_path, 'r') as f:
        params = json.load(f)  # Load the parameter dictionary from the JSON file.
    return params


def normalize_state(x, norm_params):
    """
    Normalize (standardize) px and py in the state vector.

    Use z-score standardization:
        x_normalized = (x - mean) / std

    This function supports both individual state vectors and batches of state vectors.

    Args:
        x: NumPy state array, either shape (5,) for one state [px, py, psi, v, omega]
           or shape (N, 5) for a batch of N states.
        norm_params: Dictionary containing px_mean, px_std, py_mean, and py_std.

    Returns:
        Normalized state array with the same shape as the input. Only px and py are
        normalized; all other state variables are unchanged.
    """
    x = x.copy()  # Make a copy to avoid modifying the original data.
    # Determine whether the input is a single state or a batch.
    if x.ndim == 1:
        # Single state vector (shape (5,)).
        # Normalize px: px = (px - px_mean) / px_std.
        x[0] = (x[0] - norm_params['px_mean']) / norm_params['px_std']
        # Normalize py: py = (py - py_mean) / py_std.
        x[1] = (x[1] - norm_params['py_mean']) / norm_params['py_std']
    else:
        # Batch of state vectors (shape (N, 5)).
        # Normalize px for all samples.
        x[:, 0] = (x[:, 0] - norm_params['px_mean']) / norm_params['px_std']
        # Normalize py for all samples.
        x[:, 1] = (x[:, 1] - norm_params['py_mean']) / norm_params['py_std']
    return x


def denormalize_state(x, norm_params):
    """
    De-normalize (reverse standardize) px and py in the state vector.

    Apply the inverse of z-score standardization:
        x_original = x_normalized * std + mean

    This function is used to:
    1. Convert normalized model predictions back to the original coordinates
    2. Use physical coordinates for visualization and evaluation

    Args:
        x: Normalized NumPy state array, either shape (5,) for one state
           or shape (N, 5) for a batch of N states.
        norm_params: Dictionary containing px_mean, px_std, py_mean, and py_std.

    Returns:
        De-normalized state array with the same shape as the input. Only px and py
        are de-normalized; all other state variables are unchanged.
    """
    x = x.copy()  # Make a copy to avoid modifying the original data.
    # Determine whether the input is a single state or a batch.
    if x.ndim == 1:
        # Single state vector (shape (5,)).
        # De-normalize px: px = px * px_std + px_mean.
        x[0] = x[0] * norm_params['px_std'] + norm_params['px_mean']
        # De-normalize py: py = py * py_std + py_mean.
        x[1] = x[1] * norm_params['py_std'] + norm_params['py_mean']
    else:
        # Batch of state vectors (shape (N, 5)).
        # De-normalize px for all samples.
        x[:, 0] = x[:, 0] * norm_params['px_std'] + norm_params['px_mean']
        # De-normalize py for all samples.
        x[:, 1] = x[:, 1] * norm_params['py_std'] + norm_params['py_mean']
    return x


def load_and_subsample(npz_path=DATA_NPZ_PATH, norm_json_path=NORM_JSON_PATH,
                       subsample_rate=SUBSAMPLE_RATE):
    """
    Load preprocessed training data and subsample it to match the paper's dt=0.1s.

    Data conversion pipeline:
    Original data (Ts=0.01s) -> reconstruct full trajectories -> subsample (dt=0.1s) -> average controls/disturbances

    The original sampling time is 0.01s, while the paper requires 0.1s.
    Therefore, every tenth sample is selected, and control inputs and disturbance
    vectors are averaged within each subsampling window to preserve their aggregate effect.

    Args:
        npz_path: Path to the .npz data file.
        norm_json_path: Path to the normalization parameter JSON file.
        subsample_rate: Subsampling rate (default 10, converting 0.01s to 0.1s).

    Returns:
        X_sub: Subsampled state trajectory, shape (M, 5).
        U_sub: Averaged control inputs, shape (M-1, 2).
        W_sub: Averaged disturbance vectors, shape (M-1, 5), or None.
        norm_params: Dictionary containing px_mean, px_std, py_mean, and py_std.
    """
    # Load data from the .npz file.
    data = np.load(npz_path)
    # X_t: Current state, shape (N, 5), already normalized.
    X_t = data['X_t']
    # U_t: Current control input, shape (N, 2).
    U_t = data['U_t']
    # X_t1: Next state, shape (N, 5), already normalized.
    X_t1 = data['X_t1']
    # W_t: Optional disturbance vectors, present in newer datasets.
    has_w = 'W_t' in data.files
    W_t = data['W_t'] if has_w else None

    # Convert the data file's legacy order [px, py, v, psi, omega] to the
    # internal code order [px, py, psi, v, omega].
    X_t = _legacy_to_canonical_state_order(X_t)
    X_t1 = _legacy_to_canonical_state_order(X_t1)

    # Reconstruct the full trajectory from consecutive state-transition pairs.
    # Since X_t1[i] = X_t[i+1], the full sequence is
    # [X_t[0], X_t1[0], X_t1[1], ..., X_t1[N-1]].
    N = X_t.shape[0]  # Get the number of original data points.
    X_full_orig = np.zeros((N + 1, 5))  # Preallocate the full trajectory array.
    X_full_orig[0] = X_t[0]  # First state.
    X_full_orig[1:] = X_t1  # All subsequent states (X_t1 contains X_t[1:]).

    # Subsample the states, selecting one sample every subsample_rate steps.
    # For example, subsample_rate=10 selects indices 0, 10, 20, 30, ... .
    M = (N + 1) // subsample_rate  # Compute the number of subsampled data points.
    X_sub = X_full_orig[::subsample_rate][:M]  # Subsample and truncate to M points.

    # Average control inputs within each subsampling window.
    U_sub = np.zeros((M - 1, 2))  # Preallocate the control input array.
    W_sub = np.zeros((M - 1, 5)) if has_w else None  # Preallocate the disturbance array.
    for i in range(M - 1):  # Iterate over each subsampling interval.
        start = i * subsample_rate  # Window start index.
        end = min(start + subsample_rate, N)  # Window end index, capped at N.
        # Average all control inputs within the window.
        U_sub[i] = U_t[start:end].mean(axis=0)
        if has_w:
            W_sub[i] = W_t[start:end].mean(axis=0)

    # Load normalization parameters for standardizing px and py.
    norm_params = load_norm_params(norm_json_path)

    # Print data-loading information.
    print(f"Data loaded: {N+1} original steps -> {M} subsampled "
          f"steps (rate={subsample_rate})")
    print(f"  X_sub shape: {X_sub.shape}")
    print(f"  U_sub shape: {U_sub.shape}")
    if has_w:
        print(f"  W_sub shape: {W_sub.shape}")

    return X_sub, U_sub, W_sub, norm_params


def create_sequence_windows(X, U, W=None, window_len=K_PRED):
    """
    Create sliding-window sequences for multi-step prediction loss calculation.

    This function divides a long trajectory into overlapping windows. Each window contains:
    - window_len + 1 consecutive states: [x_t, x_{t+1}, ..., x_{t+K}]
    - window_len consecutive controls: [u_t, u_{t+1}, ..., u_{t+K-1}]
    - Optionally, window_len consecutive disturbances: [w_t, w_{t+1}, ..., w_{t+K-1}]

    These windows support multi-step prediction loss during training, helping the model
    learn long-term dynamics rather than only one-step predictions.

    Args:
        X: State trajectory, NumPy array of shape (M, 5).
        U: Control trajectory, NumPy array of shape (M-1, 2).
        W: Optional disturbance trajectory, NumPy array of shape (M-1, 5).
        window_len: Prediction horizon K (window length). Defaults to K_PRED from
                    config.py (typically 10).

    Returns:
        X_windows: NumPy array of shape (num_windows, window_len+1, 5).
        U_windows: NumPy array of shape (num_windows, window_len, 2).
        W_windows: NumPy array of shape (num_windows, window_len, 5), or None.
    """
    M = X.shape[0]  # Get the trajectory length.
    # Each window requires window_len + 1 states, so num_windows = M - window_len.
    num_windows = M - window_len
    # Check that enough data is available.
    if num_windows <= 0:
        raise ValueError(
            f"Not enough data for window_len={window_len}, "
            f"have {M} steps"
        )

    # Preallocate output arrays.
    # X_windows contains window_len + 1 states per window (from t through t+K).
    X_windows = np.zeros((num_windows, window_len + 1, 5))
    # U_windows contains window_len controls per window (from t through t+K-1).
    U_windows = np.zeros((num_windows, window_len, 2))
    # W_windows contains window_len disturbances per window (from t through t+K-1).
    has_w = W is not None
    W_windows = np.zeros((num_windows, window_len, 5)) if has_w else None

    # Split the trajectory with a sliding window.
    for i in range(num_windows):  # Iterate over each window.
        # Extract states from index i through i+window_len, inclusive.
        X_windows[i] = X[i:i + window_len + 1]
        # Extract controls from index i through i+window_len-1.
        U_windows[i] = U[i:i + window_len]
        if has_w:
            W_windows[i] = W[i:i + window_len]

    # Print the number of windows created.
    print(f"Created {num_windows} sliding windows of length {window_len}")
    return X_windows, U_windows, W_windows


def create_datasets(X_sub, U_sub, W_sub=None, window_len=K_PRED,
                    val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    """
    Create PyTorch DataLoaders for training and validation.

    This function:
    1. Splits the trajectory into sequence samples using sliding windows
    2. Converts NumPy arrays to PyTorch tensors
    3. Randomly shuffles and splits the data into training and validation sets
    4. Creates DataLoaders for batched loading and shuffling

    Each sample is a window containing:
    - window_len + 1 states: [x_t, x_{t+1}, ..., x_{t+K}]
    - window_len controls: [u_t, u_{t+1}, ..., u_{t+K-1}]
    - Optionally, window_len disturbances: [w_t, w_{t+1}, ..., w_{t+K-1}]

    Args:
        X_sub: Subsampled state trajectory, shape (M, 5).
        U_sub: Subsampled control trajectory, shape (M-1, 2).
        W_sub: Optional subsampled disturbance trajectory, shape (M-1, 5).
        window_len: Prediction horizon K (window length).
        val_split: Validation-set fraction (0.1 means 10%).
        batch_size: Batch size (default 256).

    Returns:
        train_loader: PyTorch DataLoader for the training set.
        val_loader: PyTorch DataLoader for the validation set.
    """
    # Create sequence samples using sliding windows.
    X_windows, U_windows, W_windows = create_sequence_windows(
        X_sub, U_sub, W_sub, window_len
    )

    # Convert NumPy arrays to PyTorch float32 tensors.
    X_tensor = torch.tensor(X_windows, dtype=torch.float32)
    U_tensor = torch.tensor(U_windows, dtype=torch.float32)

    # Randomly shuffle and split into training and validation sets.
    N = X_tensor.shape[0]  # Get the total number of samples.
    indices = np.random.permutation(N)  # Generate a random permutation of indices.
    split = int(N * (1 - val_split))  # Compute the split point.
    train_idx = indices[:split]  # Training-set indices.
    val_idx = indices[split:]  # Validation-set indices.

    # Create TensorDatasets pairing X and U, and W when available.
    if W_windows is not None:
        W_tensor = torch.tensor(W_windows, dtype=torch.float32)
        train_dataset = TensorDataset(
            X_tensor[train_idx], U_tensor[train_idx], W_tensor[train_idx]
        )
        val_dataset = TensorDataset(
            X_tensor[val_idx], U_tensor[val_idx], W_tensor[val_idx]
        )
    else:
        train_dataset = TensorDataset(X_tensor[train_idx], U_tensor[train_idx])
        val_dataset = TensorDataset(X_tensor[val_idx], U_tensor[val_idx])

    # Create DataLoaders.
    # Shuffle the training set each epoch and drop the final incomplete batch.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    # Do not shuffle or drop any validation samples.
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False
    )

    # Print dataset information.
    print(f"Train: {len(train_dataset)} samples, "
          f"Val: {len(val_dataset)} samples")
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    return train_loader, val_loader


def create_single_step_datasets(X_sub, U_sub, W_sub=None, val_split=VAL_SPLIT,
                                batch_size=BATCH_SIZE):
    """
    Create a one-step (x_t, u_t, w_t, x_{t+1}) DataLoader.

    Unlike create_datasets, this function creates one-step transition samples
    rather than multi-step sequences. Each sample contains:
    - Current state x_t
    - Current control u_t
    - Optional disturbance w_t
    - Next state x_{t+1}

    This format is suitable for:
    1. Simple one-step prediction training
    2. Model evaluation and testing
    3. Scenarios that do not require a multi-step prediction loss

    Args:
        X_sub: State trajectory, NumPy array of shape (M, 5).
        U_sub: Control trajectory, NumPy array of shape (M-1, 2).
        W_sub: Optional disturbance trajectory, NumPy array of shape (M-1, 5).
        val_split: Validation-set fraction.
        batch_size: Batch size.

    Returns:
        train_loader: Training DataLoader; each batch is (X_t, U_t, X_t1) or
                      (X_t, U_t, W_t, X_t1).
        val_loader: Validation DataLoader.
    """
    # Create one-step transition pairs.
    # X_t: States at all time steps except the last.
    X_t = torch.tensor(X_sub[:-1], dtype=torch.float32)
    # U_t: Controls at all time steps.
    U_t = torch.tensor(U_sub, dtype=torch.float32)
    # X_t1: States at all time steps except the first.
    X_t1 = torch.tensor(X_sub[1:], dtype=torch.float32)

    # Randomly shuffle and split into training and validation sets.
    N = X_t.shape[0]  # Get the total number of samples.
    indices = np.random.permutation(N)  # Generate a random permutation of indices.
    split = int(N * (1 - val_split))  # Compute the split point.

    # Create training and validation datasets.
    if W_sub is not None:
        W_t = torch.tensor(W_sub, dtype=torch.float32)
        train_dataset = TensorDataset(
            X_t[indices[:split]], U_t[indices[:split]],
            W_t[indices[:split]], X_t1[indices[:split]]
        )
        val_dataset = TensorDataset(
            X_t[indices[split:]], U_t[indices[split:]],
            W_t[indices[split:]], X_t1[indices[split:]]
        )
    else:
        train_dataset = TensorDataset(X_t[indices[:split]], U_t[indices[:split]],
                                      X_t1[indices[:split]])
        val_dataset = TensorDataset(X_t[indices[split:]], U_t[indices[split:]],
                                    X_t1[indices[split:]])

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """
    Main program for testing data loading and preprocessing.

    When this script is run directly (python data_loader.py), it:
    1. Loads and subsamples the data
    2. Displays the normalization parameters
    3. Displays the ranges of states and control inputs
    4. Creates datasets and tests the DataLoaders
    """
    # Load and subsample the data.
    X_sub, U_sub, W_sub, norm_params = load_and_subsample()
    # Print normalization parameters.
    print(f"\nNorm params: {norm_params}")

    # Display the ranges of normalized states.
    print("State ranges (normalized):")
    for i, name in enumerate(['px', 'py', 'v', 'psi', 'omega']):
        print(f"  {name}: [{X_sub[:, i].min():.4f}, "
              f"{X_sub[:, i].max():.4f}]")

    # Display the ranges of control inputs.
    print("Control ranges:")
    for i, name in enumerate(['a', 'delta']):
        print(f"  {name}: [{U_sub[:, i].min():.4f}, "
              f"{U_sub[:, i].max():.4f}]")

    if W_sub is not None:
        print("Disturbance ranges:")
        for i, name in enumerate(['w_px', 'w_py', 'w_v', 'w_psi', 'w_omega']):
            print(f"  {name}: [{W_sub[:, i].min():.4f}, "
                  f"{W_sub[:, i].max():.4f}]")

    # Create training and validation DataLoaders.
    train_loader, val_loader = create_datasets(X_sub, U_sub, W_sub)

    # Test retrieving a batch from the DataLoader.
    first_batch = next(iter(train_loader))
    batch_shapes = [f"{t.shape}" for t in first_batch]
    print(f"\nBatch shapes: {', '.join(batch_shapes)}")
    print(f"  X contains states at t=0..{K_PRED}, "
          f"U contains controls at t=0..{K_PRED-1}")
    if len(first_batch) == 3:
        print(f"  W contains disturbances at t=0..{K_PRED-1}")
