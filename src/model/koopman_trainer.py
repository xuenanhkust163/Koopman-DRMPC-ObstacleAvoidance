"""
Training procedure for the Deep Koopman model (Algorithm 1 in the paper).

This module implements the complete training process, including:
1. Model training (forward pass, loss computation, backward pass, parameter update)
2. Validation (evaluating model performance on the validation set)
3. Learning rate scheduling (dynamically adjusting the learning rate based on validation loss)
4. Model saving (saving the best and final models)
5. Early stopping (preventing overfitting)
6. Training log recording (for analysis and visualization)

The training process follows Algorithm 1 in the paper, using a three-part loss function:
- Reconstruction loss (L_recon)
- Linear dynamics loss (L_linear)
- Multi-step prediction loss (L_pred)
"""

import os  # Operating system interfaces for file and directory operations
import sys  # System-specific parameters and functions
import time  # Time access and conversions
import json  # JSON encoder and decoder
import numpy as np  # NumPy: array computing for Python
import torch  # PyTorch: tensor computing and deep learning framework
import torch.optim as optim  # PyTorch optimization algorithms

# Add the parent directory to the system path, ensuring same-level module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import training-related parameters from the config file
from config import (
    EPOCHS,  # Number of training epochs, default 500
    LEARNING_RATE,  # Initial learning rate, default 0.001
    LR_PATIENCE,  # Patience value for the learning rate scheduler, default 20
    LR_FACTOR,  # Learning rate decay factor, default 0.5
    EARLY_STOP_PATIENCE,  # Patience value for early stopping, default 50
    LAMBDA_RECON,  # Weight for reconstruction loss, default 1.0
    LAMBDA_LINEAR,  # Weight for linear dynamics loss, default 1.0
    LAMBDA_PRED,  # Weight for multi-step prediction loss, default 0.5
    LAMBDA_PHYSICS,  # Weight for physics consistency loss, default 1.0
    LAMBDA_SPECTRAL,  # Weight for spectral radius penalty, default 5.0
    MODEL_DIR  # Model saving directory
)

# Import model and loss function from the koopman_network module
from model.koopman_network import DeepKoopmanPaper, koopman_loss


def clip_spectral_radius(A_numpy, max_radius=0.99):
    """Clip the eigenvalues of matrix A to within the unit circle to ensure long-term prediction stability."""
    eigenvalues, V = np.linalg.eig(A_numpy)
    magnitudes = np.abs(eigenvalues)
    mask = magnitudes > max_radius
    if mask.any():
        eigenvalues[mask] *= max_radius / magnitudes[mask]
        A_clipped = (V @ np.diag(eigenvalues) @ np.linalg.inv(V)).real
        return A_clipped, True  # Return the clipped matrix and the clipping flag
    return A_numpy, False


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE,
                device='cpu', save_dir=MODEL_DIR,
                lambda_physics=LAMBDA_PHYSICS,
                lambda_reg_high_dim=0.0,
                lambda_spectral=LAMBDA_SPECTRAL):
    """
    Train the Deep Koopman model according to Algorithm 1 in the paper.

    Training procedure:
    1. Initialize the optimizer and learning rate scheduler
    2. For each epoch:
       a. Training phase: forward pass → loss computation → backward pass → parameter update
       b. Validation phase: compute validation loss (no parameter update)
       c. Learning rate scheduling: adjust learning rate based on validation loss
       d. Model saving: if validation loss improves, save the best model
       e. Early stopping check: if validation loss does not improve for a long time, stop training
    3. Save the final model and training log

    Parameters:
        model: DeepKoopmanPaper instance, the Koopman model to be trained
        train_loader: PyTorch DataLoader, training set data loader
                     Returns (x_windows, u_windows) for each batch
        val_loader: PyTorch DataLoader, validation set data loader
        epochs: int, number of training epochs, default is EPOCHS (500) in config.py
        lr: float, initial learning rate, default is LEARNING_RATE (0.001) in config.py
        device: str, training device, 'cpu' or 'cuda'
        save_dir: str, model checkpoint saving directory

    Returns:
        model: Trained Deep Koopman model
        training_log: dict, contains training history:
            - 'train_loss': list of total training losses
            - 'val_loss': list of total validation losses
            - 'train_recon': list of training reconstruction losses
            - 'train_linear': list of training linear dynamics losses
            - 'train_pred': list of training multi-step prediction losses
            - 'val_recon': list of validation reconstruction losses
            - 'val_linear': list of validation linear dynamics losses
            - 'val_pred': list of validation multi-step prediction losses
            - 'lr': list of learning rates for each epoch
            - 'epoch_time': list of epoch durations
            - 'total_time': total training time
            - 'best_val_loss': best validation loss
    """
    # Create model saving directory (if not exists)
    # exist_ok=True means no error if the directory already exists
    os.makedirs(save_dir, exist_ok=True)

    # Move the model to the specified device (CPU or GPU)
    model = model.to(device)

    # Create Adam optimizer
    # Adam is an adaptive learning rate optimization algorithm that combines the advantages of momentum and RMSprop
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create learning rate scheduler (ReduceLROnPlateau)
    # Automatically reduce the learning rate when the validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # The monitored metric should be minimized (loss)
        patience=LR_PATIENCE,  # Wait for LR_PATIENCE epochs without improvement to reduce the learning rate
        factor=LR_FACTOR  # Reduce to LR_FACTOR times (0.5 means halve) the learning rate each time
    )

    # Initialize the best validation loss to positive infinity
    best_val_loss = float('inf')
    # Initialize the patience counter (for early stopping)
    patience_counter = 0

    # Initialize training log dictionary
    # Records various metrics during training and validation
    training_log = {
        'train_loss': [],  # Total training loss
        'val_loss': [],  # Total validation loss
        'train_recon': [],  # Training reconstruction loss
        'train_linear': [],  # Training linear dynamics loss
        'train_pred': [],  # Training multi-step prediction loss
        'train_physics': [],  # Training physics consistency loss
        'train_spectral': [],  # Training spectral radius penalty
        'train_reg': [],  # Training high-dimensional regularization loss
        'val_recon': [],  # Validation reconstruction loss
        'val_linear': [],  # Validation linear dynamics loss
        'val_pred': [],  # Validation multi-step prediction loss
        'val_physics': [],  # Validation physics consistency loss
        'val_spectral': [],  # Validation spectral radius penalty
        'val_reg': [],  # Validation high-dimensional regularization loss
        'lr': [],  # Learning rate
        'epoch_time': [],  # Time taken for each epoch
    }

    # Record the start time of training
    total_start = time.time()

    # Main training loop: iterate over all epochs
    for epoch in range(epochs):
        # Record the start time of the current epoch
        epoch_start = time.time()

        # ================================================================
        # Stage 1: Training Phase
        # ================================================================
        # Set the model to training mode
        # This enables behaviors specific to training, such as dropout and batchnorm
        model.train()

        # Initialize training loss accumulator
        train_losses = {'total': 0, 'recon': 0, 'linear': 0, 'pred': 0, 'physics': 0, 'reg': 0, 'spectral': 0}
        # Count the number of batches in the training set
        n_train_batches = 0

        # Iterate over all batches in the training set
        for batch in train_loader:
            # Check if the batch contains interference based on its length
            if len(batch) == 3:
                batch_x, batch_u, batch_w = batch
                batch_w = batch_w.to(device)
            else:
                batch_x, batch_u = batch
                batch_w = None

            # Move data to the specified device
            batch_x = batch_x.to(device)  # State sequence, shape(batch, K+1, 5)
            batch_u = batch_u.to(device)  # Control sequence, shape(batch, K, 2)

            # Zero the gradients (must be done before each backward pass)
            # PyTorch accumulates gradients by default, so we need to zero them manually
            optimizer.zero_grad()

            # Compute the loss (forward pass)
            # koopman_loss returns the total loss and the component losses
            loss, loss_dict = koopman_loss(
                model, batch_x, batch_u, batch_w,
                lambda_recon=LAMBDA_RECON,  # Weight for reconstruction loss
                lambda_linear=LAMBDA_LINEAR,  # Weight for linear dynamics loss
                lambda_pred=LAMBDA_PRED,  # Weight for multi-step prediction loss
                lambda_physics=lambda_physics,  # Weight for physics consistency loss
                lambda_reg_high_dim=lambda_reg_high_dim,  # Weight for high-dimensional regularization
                lambda_spectral=lambda_spectral  # Weight for spectral radius penalty
            )

            # Backward pass: compute gradients
            # Automatically compute the gradients of all parameters starting from the loss
            loss.backward()

            # Gradient clipping: prevent gradient explosion
            # Limit the L2 norm of the gradients to max_norm=10.0
            # This is crucial for training stability, especially for RNNs and deep networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # Update model parameters
            # Use the computed gradients to update all trainable parameters with the Adam optimizer
            optimizer.step()

            # Force reset the physical key elements to prevent drift caused by Adam momentum
            # The first 5 dimensions of z are [px, py, psi, v, omega]
            with torch.no_grad():
                model.B.data[3, 0] = 0.1   # a -> v (dt=0.1), v is at index 3 in z
                model.B.data[4, 1] = 0.15  # delta -> omega, omega is at index 4 in z
                model.A.data[0, 0] = 1.0   # px -> px
                model.A.data[1, 1] = 1.0   # py -> py
                model.A.data[2, 2] = 1.0   # psi -> psi
                model.A.data[2, 4] = 0.1   # omega -> psi (dt=0.1)
                model.A.data[3, 3] = 1.0   # v -> v
                model.A.data[4, 4] = 1.0   # omega -> omega

            # Accumulate the loss for the current batch
            for k in train_losses:
                train_losses[k] += loss_dict[k]
            n_train_batches += 1

        # Compute the average training loss (divide by the number of batches)
        for k in train_losses:
            train_losses[k] /= max(n_train_batches, 1)  # Use max to avoid division by zero

        # ================================================================
        # Stage 2: Validation Phase
        # ================================================================
        # Set the model to evaluation mode
        # This disables dropout and uses the running averages for batchnorm
        model.eval()

        # Initialize validation loss accumulator
        val_losses = {'total': 0, 'recon': 0, 'linear': 0, 'pred': 0, 'physics': 0, 'reg': 0, 'spectral': 0}
        # Count the number of batches in the validation set
        n_val_batches = 0

        # Disable gradient computation with torch.no_grad()
        # No need for backward pass during validation, so gradients are not computed
        # This saves memory and computation time
        with torch.no_grad():
            # Iterate over all batches in the validation set
            for batch in val_loader:
                # Check if the batch contains interference based on its length
                if len(batch) == 3:
                    batch_x, batch_u, batch_w = batch
                    batch_w = batch_w.to(device)
                else:
                    batch_x, batch_u = batch
                    batch_w = None

                # Move data to the specified device
                batch_x = batch_x.to(device)
                batch_u = batch_u.to(device)

                # Compute the validation loss (forward pass only, no backward pass)
                _, loss_dict = koopman_loss(
                    model, batch_x, batch_u, batch_w,
                    lambda_recon=LAMBDA_RECON,
                    lambda_linear=LAMBDA_LINEAR,
                    lambda_pred=LAMBDA_PRED,
                    lambda_physics=lambda_physics,
                    lambda_reg_high_dim=lambda_reg_high_dim,
                    lambda_spectral=lambda_spectral
                )

                # Accumulate the loss for the current batch
                for k in val_losses:
                    val_losses[k] += loss_dict[k]
                n_val_batches += 1

        # Compute the average validation loss
        for k in val_losses:
            val_losses[k] /= max(n_val_batches, 1)

        # ================================================================
        # Stage 3: Learning Rate Scheduling
        # ================================================================
        # Adjust the learning rate based on the validation loss
        # If the validation loss does not improve for LR_PATIENCE epochs, multiply the learning rate by LR_FACTOR
        scheduler.step(val_losses['total'])

        # Calculate the time taken for the current epoch
        epoch_time = time.time() - epoch_start

        # ================================================================
        # Stage 4: Record Training Log
        # ================================================================
        # Add the metrics of the current epoch to the training log
        training_log['train_loss'].append(train_losses['total'])
        training_log['val_loss'].append(val_losses['total'])
        training_log['train_recon'].append(train_losses['recon'])
        training_log['train_linear'].append(train_losses['linear'])
        training_log['train_pred'].append(train_losses['pred'])
        training_log['train_physics'].append(train_losses['physics'])
        training_log['train_reg'].append(train_losses['reg'])
        training_log['train_spectral'].append(train_losses['spectral'])
        training_log['val_recon'].append(val_losses['recon'])
        training_log['val_linear'].append(val_losses['linear'])
        training_log['val_pred'].append(val_losses['pred'])
        training_log['val_physics'].append(val_losses['physics'])
        training_log['val_reg'].append(val_losses['reg'])
        training_log['val_spectral'].append(val_losses['spectral'])
        # Record the current learning rate
        training_log['lr'].append(optimizer.param_groups[0]['lr'])
        # Record the time taken for the current epoch
        training_log['epoch_time'].append(epoch_time)

        # ================================================================
        # Stage 5: Print Training Progress
        # ================================================================
        # Print every 10 epochs, or also print the first epoch
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_losses['total']:.6f} "
                  f"(R:{train_losses['recon']:.4f} L:{train_losses['linear']:.4f} "
                  f"P:{train_losses['pred']:.4f} Ph:{train_losses['physics']:.4f} "
                  f"Rg:{train_losses['reg']:.4f} Sp:{train_losses['spectral']:.4f}) | "
                  f"Val: {val_losses['total']:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        # ================================================================
        # Stage 6: Save Best Model
        # ================================================================
        # If the current validation loss is better than the historical best, save the model
        if val_losses['total'] < best_val_loss:
            # Update the best validation loss
            best_val_loss = val_losses['total']
            # Reset the patience counter
            patience_counter = 0

            # Create a checkpoint dictionary, containing the model and optimizer states
            checkpoint = {
                'epoch': epoch + 1,  # Current epoch number
                'model_state_dict': model.state_dict(),  # Model parameters
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                'val_loss': best_val_loss,  # Best validation loss
                # Save the Koopman matrices (for subsequent MPC optimization)
                'A': model.A.detach().cpu().numpy().tolist(),  # State transition matrix
                'B': model.B.detach().cpu().numpy().tolist(),  # Control matrix
            }
            # Save the best model checkpoint
            model_path = os.path.join(save_dir, 'best_koopman_model.pth')
            torch.save(checkpoint, model_path)

            # Print the saving information every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
        else:
            # If the validation loss did not improve, increase the patience counter
            patience_counter += 1

        # ================================================================
        # Stage 7: Early Stopping Check
        # ================================================================
        # If the validation loss has not improved for EARLY_STOP_PATIENCE epochs, stop training
        # This is a regularization technique to prevent overfitting
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(patience={EARLY_STOP_PATIENCE})")
            break

    # Calculate the total training time
    total_time = time.time() - total_start

    # ================================================================
    # Training Finished: Load Best Model and Clip A Matrix Spectral Radius
    # ================================================================
    best_model_path = os.path.join(save_dir, 'best_koopman_model.pth')
    if os.path.exists(best_model_path):
        print("\n[INFO] 加载最佳模型进行特征值裁剪...")
        best_model = load_trained_model(best_model_path, device=device)

        # Eigenvalue clipping: ensure stable deployment of the model
        A_numpy = best_model.A.detach().cpu().numpy()
        A_clipped, was_clipped = clip_spectral_radius(A_numpy, max_radius=0.99)
        if was_clipped:
            orig_radius = np.max(np.abs(np.linalg.eigvals(A_numpy)))
            print(f"[INFO] A矩阵特征值已裁剪: 原谱半径={orig_radius:.4f} -> 0.99")
            best_model.A.data = torch.tensor(A_clipped, dtype=torch.float32, device=best_model.A.device)

        # Save the clipped best model (overwriting the original file)
        checkpoint_best = {
            'epoch': epoch + 1,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'A': best_model.A.detach().cpu().numpy().tolist(),
            'B': best_model.B.detach().cpu().numpy().tolist(),
        }
        torch.save(checkpoint_best, best_model_path)
        print("  -> Saved clipped best model")

        # Save the clipped final model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': best_model.state_dict(),
            'val_loss': best_val_loss,
        }, os.path.join(save_dir, 'final_koopman_model.pth'))
        print("  -> Saved clipped final model")

        # Save the Koopman matrices to a npz file
        A, B, C = best_model.get_matrices()
        np.savez(os.path.join(save_dir, 'koopman_matrices.npz'),
                 A=A, B=B, C=C)
        print(f"  -> Saved koopman_matrices.npz (A max|eig|={np.max(np.abs(np.linalg.eigvals(A))):.4f})")

        # Use the clipped model as the return value
        model = best_model
    else:
        # If no best model, save the current final model
        print("\n[WARN] 未找到最佳模型，保存当前最终模型（未裁剪）")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': val_losses['total'],
        }, os.path.join(save_dir, 'final_koopman_model.pth'))

    # ================================================================
    # Save Training Log
    # ================================================================
    # Add the training log to the log dictionary
    training_log['total_time'] = total_time  # Total training time
    training_log['best_val_loss'] = best_val_loss  # Best validation loss

    # Save the training log as a JSON file
    # indent=2 makes the JSON file more readable
    log_path = os.path.join(save_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    # Print training completion information
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Return the trained model and training log
    return model, training_log


def load_trained_model(model_path=None, device='cpu'):
    """
    Load a trained DeepKoopmanPaper model from a checkpoint.

    This function is used during the inference phase to load a pre-trained model for simulation or control.

    Parameters:
        model_path: str, path to the model checkpoint
                   If None, the default best_koopman_model.pth is used
        device: str, device for loading, 'cpu' or 'cuda'

    Returns:
        model: Loaded DeepKoopmanPaper model (in evaluation mode)

    Usage example:
        model = load_trained_model()
        z = model.encode(x)  # Encode state to Koopman space
    """
    # If no path is specified, use the default best model path
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_koopman_model.pth')

    # Create a model instance (using default parameters)
    model = DeepKoopmanPaper()

    # Load the checkpoint
    # weights_only=False allows loading checkpoints that contain numpy arrays
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False
    )

    # Check the checkpoint format and load the model weights
    if 'model_state_dict' in checkpoint:
        # Standard format: contains 'model_state_dict' key
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # Old format: the checkpoint itself is the state_dict
        model.load_state_dict(checkpoint, strict=False)

    # Set the model to evaluation mode
    # This disables behaviors specific to training, such as dropout
    model.eval()

    return model
