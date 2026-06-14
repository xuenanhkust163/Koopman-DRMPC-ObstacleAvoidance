# K-DRMPC: Distributionally Robust Koopman Model Predictive Control

> Experimental snapshot dated 2026-05-21, including the SerpentineTrack benchmark and the latest robust versus non-robust comparison scripts.

## Overview

This repository contains the research code for Koopman operator-based distributionally robust model predictive control applied to vehicle trajectory tracking and dynamic obstacle avoidance. The implementation focuses on learning-based linear prediction, disturbance-aware robust optimization, and comparative evaluation against multiple MPC baselines.

More specifically, the framework combines a deep Koopman lifting model with Wasserstein distributionally robust optimization to improve feasibility and tail-error robustness under disturbance shifts while retaining the computational advantages of linear MPC. The repository is intended to support reproducible experiments on autonomous driving obstacle avoidance benchmarks and controlled comparisons against non-robust and alternative MPC formulations.

The codebase supports the study reported in the manuscript *Distributionally Robust Model Predictive Control with Disturbance Uncertainty Adaptiveness for Obstacle Avoidance of Autonomous Driving*.

## Key Features

- **Deep Koopman network** for lifting nonlinear vehicle dynamics into a linear latent space using data-driven learning.
- **Distributionally robust MPC** based on Wasserstein ambiguity sets and CVaR constraints to explicitly handle measurement noise and model uncertainty.
- **Unified controller benchmarking** across LMPC, NMPC, K-MPC, and K-DRMPC.
- **Robust vs non-robust evaluation** through `run_noise_comparison.py`, using the same track and disturbance realizations to produce comparative metrics and dual-trajectory GIF visualizations.
- **Multiple track configurations** including SprintOval, LusailShort, Lusail, SerpentineTrack, MonacoHairpin, SpaFlow, and SuzukaFlow.
- **Complete simulation pipeline** covering spline-based track modeling, rectangular obstacle generation, Gaussian disturbance injection, and performance evaluation.

## Installation

Install the Python dependencies and prepare the runtime environment with:

```bash
bash setup.sh
```

If you prefer a manual setup workflow, install the required packages listed in `requirements.txt` inside your own Python environment.

## Quick Start

### 1. Generate training data

```bash
python generate_training_data.py
```

### 2. Train the Koopman model

```bash
python run_training.py
```

### 3. Run the robust vs non-robust comparison

```bash
bash run_all_seeds.sh
```

This script runs comparison experiments on SerpentineTrack under three random seeds and multiple disturbance levels.

Expected outputs:

- `_output/results/Robust_K-DRMPC_SerpentineTrack.pkl`
- `_output/results/NonRobust_K-DRMPC_SerpentineTrack.pkl`
- `_output/figures/Comparison_SerpentineTrack_dual_line.gif`
- `_output/results/noise_comparison_summary.json`

### 4. Run sensitivity analysis

```bash
python run_sensitivity_analysis.py
```

## Repository Structure

- `controllers/`: controller implementations, including LMPC, NMPC, K-MPC, and K-DRMPC.
- `model/`: Koopman model definitions and training-related components.
- `simulation/`: simulation logic, closed-loop rollout, and experiment orchestration.
- `tracks/`: track geometry and benchmark track definitions.
- `vehicle/`: vehicle dynamics and state propagation models.
- `disturbance/`: disturbance generation and uncertainty-related components.
- `visualization/`: plotting, animation, and result rendering utilities.
- `data/`: data loading and dataset support.
- `docs/`: supplementary notes and internal documentation.
- `_output/`: generated models, figures, and experiment results.

## Reproducibility Workflow

For a standard end-to-end workflow, run the steps in the following order:

1. Generate training data with `python generate_training_data.py`.
2. Train the Koopman model with `python run_training.py`.
3. Evaluate robust and non-robust controllers with `bash run_all_seeds.sh` or `python run_noise_comparison.py`.
4. Run parameter sensitivity studies with `python run_sensitivity_analysis.py`.

## Citation

If you use this repository in academic work, please cite the associated manuscript:

```text
Nan Xue and Lei Zhang.
Distributionally Robust Model Predictive Control with Disturbance Uncertainty Adaptiveness for Obstacle Avoidance of Autonomous Driving.
Manuscript in preparation / under submission.
```

BibTeX entry:

```bibtex
@misc{xue2026drmpc,
  title={Distributionally Robust Model Predictive Control with Disturbance Uncertainty Adaptiveness for Obstacle Avoidance of Autonomous Driving},
  author={Nan Xue and Lei Zhang},
  year={2026},
  note={Manuscript in preparation / under submission}
}
```

This citation entry can be updated once a formal publication record is available.

## License

This repository now includes a formal [LICENSE](LICENSE). At present, the project is released as academic-reference-only with all rights reserved. If you need reuse, redistribution, or collaboration permission, please contact the authors.
