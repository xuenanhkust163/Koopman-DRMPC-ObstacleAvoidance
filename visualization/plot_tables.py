"""
Table generation for all paper tables (Tables 6, 9-15).
Outputs formatted console tables and optional LaTeX.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TABLES_DIR
from simulation.metrics import format_metrics_table, format_latex_table


def print_table_6(training_log, save_dir=TABLES_DIR):
    """
    Print Table 6: Deep Koopman model training performance.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Table 6: Deep Koopman Model Training Performance")
    print("=" * 60)

    metrics = [
        ("Reconstruction Error (MSE)", training_log.get('val_recon', [0])[-1]),
        ("Linear Dynamics Error (MSE)", training_log.get('val_linear', [0])[-1]),
        ("Multi-step Prediction Error (MSE)", training_log.get('val_pred', [0])[-1]),
        ("Total Validation Loss", training_log.get('best_val_loss', 0)),
        ("Training Time (s)", training_log.get('total_time', 0)),
        ("Total Epochs", len(training_log.get('train_loss', []))),
    ]

    for name, value in metrics:
        if isinstance(value, float):
            print(f"  {name:<40} {value:.6f}")
        else:
            print(f"  {name:<40} {value}")

    print("=" * 60)

    # Save to file
    with open(os.path.join(save_dir, 'table6_training.txt'), 'w') as f:
        for name, value in metrics:
            f.write(f"{name}: {value}\n")


def print_performance_tables(all_metrics, methods, track_name,
                             table_num="9", save_dir=TABLES_DIR):
    """
    Print performance comparison tables (Tables 9, 10, 13).
    """
    os.makedirs(save_dir, exist_ok=True)

    table_str = format_metrics_table(all_metrics, methods)
    print(f"\nTable {table_num}: Performance on {track_name}")
    print(table_str)

    # Save
    with open(os.path.join(save_dir, f'table{table_num}_{track_name}.txt'), 'w') as f:
        f.write(table_str)

    # LaTeX version
    latex_str = format_latex_table(
        all_metrics, methods,
        caption=f"Performance comparison on {track_name}",
        label=f"tab:perf_{track_name}"
    )
    with open(os.path.join(save_dir, f'table{table_num}_{track_name}.tex'), 'w') as f:
        f.write(latex_str)


def print_robustness_table(robustness_results, sigma_values, table_num="11",
                           save_dir=TABLES_DIR):
    """
    Print Table 11: Robustness analysis under different disturbance levels.

    Args:
        robustness_results: dict {sigma: metrics_dict} for K-DRMPC
        sigma_values: list of sigma values tested
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nTable {table_num}: Robustness Analysis (K-DRMPC)")
    print("-" * 70)
    header = f"{'Disturbance Level':<20}"
    metric_names = ['tracking_error_rms', 'constraint_violation_pct',
                    'max_speed', 'solve_time_mean']
    metric_labels = ['Track Err (m)', 'Viol. (%)', 'Max Speed (m/s)', 'Time (ms)']
    for label in metric_labels:
        header += f"{label:>15}"
    print(header)
    print("-" * 70)

    for sigma in sigma_values:
        key = f"sigma_{sigma}"
        if key in robustness_results:
            m = robustness_results[key]
            row = f"sigma = {sigma:<13.2f}"
            for name in metric_names:
                val = m.get(name, None)
                if val is None:
                    row += f"{'N/A':>15}"
                else:
                    row += f"{val:>15.2f}"
            print(row)

    print("-" * 70)


def print_sensitivity_table(sensitivity_results, param_values, param_name,
                            table_num="14", save_dir=TABLES_DIR):
    """
    Print Tables 14-15: Sensitivity analysis for theta or epsilon.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nTable {table_num}: Sensitivity Analysis for {param_name}")
    print("-" * 60)
    header = f"{param_name:<15}"
    metric_labels = ['Viol. (%)', 'Lap Time (s)', 'Track Err (m)', 'Time (ms)']
    for label in metric_labels:
        header += f"{label:>15}"
    print(header)
    print("-" * 60)

    metric_keys = ['constraint_violation_pct', 'lap_time',
                   'tracking_error_rms', 'solve_time_mean']

    for val in param_values:
        key = f"{param_name}_{val}"
        if key in sensitivity_results:
            m = sensitivity_results[key]
            row = f"{val:<15.2f}"
            for mk in metric_keys:
                v = m.get(mk, None)
                if v is None:
                    row += f"{'N/A':>15}"
                else:
                    row += f"{v:>15.2f}"
            print(row)

    print("-" * 60)

    # Save
    with open(os.path.join(save_dir, f'table{table_num}_{param_name}.txt'), 'w') as f:
        f.write(f"Sensitivity analysis for {param_name}\n")
        for val in param_values:
            key = f"{param_name}_{val}"
            if key in sensitivity_results:
                f.write(f"{param_name}={val}: {sensitivity_results[key]}\n")
