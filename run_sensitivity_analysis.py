"""
论文 6.6.2 节：Wasserstein Radius 和 CVaR Level 敏感性分析

扫描 theta × epsilon 参数网格，评估 Robust K-DRMPC 性能

用法:
    python run_sensitivity_analysis.py --sigma 0.75 --steps 3500
"""

import os
import sys
import json
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_noise_comparison import (
    run_experiment, load_koopman, print_comparison,
    TRACK_NAME, N_DISTURBANCE_SAMPLES, DEFAULT_STEPS,
)
from simulation.simulator import Simulator
from tracks.serpentine_track import SerpentineTrack

# 论文 6.6.2 节的参数网格
THETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]
EPSILON_GRID = [0.01, 0.05, 0.10]


def run_single_config(model, D, norm_params, theta, epsilon, sigma, seed, max_steps):
    """运行单个 (theta, epsilon) 配置，只跑 Robust 模式。"""
    tag = f"Robust_K-DRMPC_T{theta:.2f}_E{epsilon:.2f}_S{sigma:.2f}"
    print(f"\n{'='*60}")
    print(f"Running: theta={theta}, epsilon={epsilon}, sigma={sigma}")
    print(f"{'='*60}")

    try:
        result, track = run_experiment(
            model, D, norm_params,
            robust=True,
            tag=tag,
            sigma=sigma,
            theta=theta,
            epsilon=epsilon,
            seed=seed,
            max_steps=max_steps,
        )

        # 计算关键指标
        states = np.array(result.states)
        controls = np.array(result.controls)
        solve_times = np.array(result.solve_times)

        # 横向误差
        e_y_list = []
        for st in states:
            _, s, lat_err = track.closest_point(st[0], st[1])
            e_y_list.append(abs(lat_err))
        e_y_arr = np.array(e_y_list)

        # 检查是否完成一圈
        lap_completed = result.lap_completed
        total_steps = result.total_steps
        crashed = result.crashed
        crash_step = result.crash_step if result.crashed else None

        metrics = {
            'theta': theta,
            'epsilon': epsilon,
            'sigma': sigma,
            'lap_completed': lap_completed,
            'total_steps': total_steps,
            'crashed': crashed,
            'crash_step': crash_step,
            'ey_mean': float(np.mean(e_y_arr)),
            'ey_p95': float(np.percentile(e_y_arr, 95)),
            'ey_max': float(np.max(e_y_arr)),
            'solve_mean_ms': float(np.mean(solve_times)),
            'solve_max_ms': float(np.max(solve_times)),
            'v_mean': float(np.mean(states[:, 3])),
            'tag': tag,
        }
        return metrics

    except Exception as e:
        print(f"ERROR running theta={theta}, epsilon={epsilon}: {e}")
        return {
            'theta': theta,
            'epsilon': epsilon,
            'sigma': sigma,
            'error': str(e),
            'lap_completed': False,
            'total_steps': 0,
            'crashed': True,
            'crash_step': 0,
            'ey_mean': np.nan,
            'ey_p95': np.nan,
            'ey_max': np.nan,
            'solve_mean_ms': np.nan,
            'solve_max_ms': np.nan,
            'v_mean': np.nan,
            'tag': tag,
        }


def generate_heatmaps(results, output_dir):
    """生成 2D 热力图。"""
    os.makedirs(output_dir, exist_ok=True)

    # 构建网格数据
    n_theta = len(THETA_GRID)
    n_eps = len(EPSILON_GRID)

    success_grid = np.zeros((n_theta, n_eps))
    ey_mean_grid = np.zeros((n_theta, n_eps))
    solve_mean_grid = np.zeros((n_theta, n_eps))

    for r in results:
        if 'error' in r:
            continue
        ti = THETA_GRID.index(r['theta'])
        ei = EPSILON_GRID.index(r['epsilon'])

        success_grid[ti, ei] = 1.0 if r['lap_completed'] else 0.0
        ey_mean_grid[ti, ei] = r['ey_mean'] if not np.isnan(r['ey_mean']) else np.nan
        solve_mean_grid[ti, ei] = r['solve_mean_ms'] if not np.isnan(r['solve_mean_ms']) else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 热力图 1: 成功率
    ax = axes[0]
    im = ax.imshow(success_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(n_eps))
    ax.set_yticks(range(n_theta))
    ax.set_xticklabels([f'{e:.2f}' for e in EPSILON_GRID])
    ax.set_yticklabels([f'{t:.1f}' for t in THETA_GRID])
    ax.set_xlabel('CVaR Level ε')
    ax.set_ylabel('Wasserstein Radius θ')
    ax.set_title('Success Rate')
    for i in range(n_theta):
        for j in range(n_eps):
            text = '✓' if success_grid[i, j] > 0.5 else '✗'
            ax.text(j, i, text, ha='center', va='center', fontsize=16)
    plt.colorbar(im, ax=ax, label='Success')

    # 热力图 2: 平均横向误差
    ax = axes[1]
    # 使用 masked array 处理 NaN
    ey_masked = np.ma.masked_where(np.isnan(ey_mean_grid), ey_mean_grid)
    im = ax.imshow(ey_masked, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(n_eps))
    ax.set_yticks(range(n_theta))
    ax.set_xticklabels([f'{e:.2f}' for e in EPSILON_GRID])
    ax.set_yticklabels([f'{t:.1f}' for t in THETA_GRID])
    ax.set_xlabel('CVaR Level ε')
    ax.set_ylabel('Wasserstein Radius θ')
    ax.set_title('Mean |e_y| (m)')
    for i in range(n_theta):
        for j in range(n_eps):
            if not np.isnan(ey_mean_grid[i, j]):
                ax.text(j, i, f'{ey_mean_grid[i, j]:.2f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax, label='|e_y| (m)')

    # 热力图 3: 平均求解时间
    ax = axes[2]
    solve_masked = np.ma.masked_where(np.isnan(solve_mean_grid), solve_mean_grid)
    im = ax.imshow(solve_masked, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(n_eps))
    ax.set_yticks(range(n_theta))
    ax.set_xticklabels([f'{e:.2f}' for e in EPSILON_GRID])
    ax.set_yticklabels([f'{t:.1f}' for t in THETA_GRID])
    ax.set_xlabel('CVaR Level ε')
    ax.set_ylabel('Wasserstein Radius θ')
    ax.set_title('Mean Solve Time (ms)')
    for i in range(n_theta):
        for j in range(n_eps):
            if not np.isnan(solve_mean_grid[i, j]):
                ax.text(j, i, f'{solve_mean_grid[i, j]:.0f}', ha='center', va='center', fontsize=10)
    plt.colorbar(im, ax=ax, label='ms')

    plt.suptitle(f'K-DRMPC Sensitivity Analysis (σ={results[0]["sigma"]:.2f})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'sensitivity_heatmap_sigma{results[0]["sigma"]:.2f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ 热力图已保存: {output_path}")


def generate_line_plots(results, output_dir):
    """生成线图（性能 vs theta，分 epsilon 曲线）。"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # red, orange, green

    # 图 1: 成功率 vs theta
    ax = axes[0]
    for ei, eps in enumerate(EPSILON_GRID):
        vals = []
        for t in THETA_GRID:
            r = next((x for x in results if x['theta'] == t and x['epsilon'] == eps), None)
            if r and 'error' not in r:
                vals.append(1.0 if r['lap_completed'] else 0.0)
            else:
                vals.append(0.0)
        ax.plot(THETA_GRID, vals, 'o-', color=colors[ei], label=f'ε={eps:.2f}', linewidth=2, markersize=8)
    ax.set_xlabel('Wasserstein Radius θ')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate vs θ')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    # 图 2: ey_mean vs theta
    ax = axes[1]
    for ei, eps in enumerate(EPSILON_GRID):
        vals = []
        for t in THETA_GRID:
            r = next((x for x in results if x['theta'] == t and x['epsilon'] == eps), None)
            if r and 'error' not in r and not np.isnan(r['ey_mean']):
                vals.append(r['ey_mean'])
            else:
                vals.append(np.nan)
        ax.plot(THETA_GRID, vals, 'o-', color=colors[ei], label=f'ε={eps:.2f}', linewidth=2, markersize=8)
    ax.set_xlabel('Wasserstein Radius θ')
    ax.set_ylabel('Mean |e_y| (m)')
    ax.set_title('Tracking Error vs θ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图 3: solve time vs theta
    ax = axes[2]
    for ei, eps in enumerate(EPSILON_GRID):
        vals = []
        for t in THETA_GRID:
            r = next((x for x in results if x['theta'] == t and x['epsilon'] == eps), None)
            if r and 'error' not in r and not np.isnan(r['solve_mean_ms']):
                vals.append(r['solve_mean_ms'])
            else:
                vals.append(np.nan)
        ax.plot(THETA_GRID, vals, 'o-', color=colors[ei], label=f'ε={eps:.2f}', linewidth=2, markersize=8)
    ax.set_xlabel('Wasserstein Radius θ')
    ax.set_ylabel('Mean Solve Time (ms)')
    ax.set_title('Solve Time vs θ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'K-DRMPC Performance vs Robustness Parameters (σ={results[0]["sigma"]:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'sensitivity_lines_sigma{results[0]["sigma"]:.2f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 线图已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="论文6.6.2节：theta/epsilon敏感性分析")
    parser.add_argument("--sigma", type=float, default=0.75,
                        help="干扰强度（Setting B = 训练σ × 1.5, 默认0.75）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--steps", type=int, default=3500, help="最大仿真步数")
    parser.add_argument("--output-dir", type=str, default="_output/sensitivity",
                        help="结果输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("论文 6.6.2 节：Wasserstein Radius 和 CVaR Level 敏感性分析")
    print(f"sigma={args.sigma}, steps={args.steps}, seed={args.seed}")
    print("="*60)

    model, D, norm_params = load_koopman()

    results = []
    total_configs = len(THETA_GRID) * len(EPSILON_GRID)
    for idx, (theta, epsilon) in enumerate(itertools.product(THETA_GRID, EPSILON_GRID)):
        print(f"\n[{idx+1}/{total_configs}] theta={theta:.1f}, epsilon={epsilon:.2f}")
        metrics = run_single_config(
            model, D, norm_params,
            theta=theta, epsilon=epsilon,
            sigma=args.sigma, seed=args.seed, max_steps=args.steps,
        )
        results.append(metrics)

    # 保存结果
    summary_path = os.path.join(args.output_dir, f'sensitivity_results_sigma{args.sigma:.2f}.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果摘要已保存: {summary_path}")

    # 打印摘要表格
    print("\n" + "="*80)
    print(f"{'θ':>6} {'ε':>6} {'完成':>6} {'步数':>6} {'撞车':>6} {'ey_mean':>10} {'solve_ms':>10}")
    print("-"*80)
    for r in results:
        status = "✓" if r['lap_completed'] else "✗"
        crash = f"@{r['crash_step']}" if r['crashed'] else "-"
        ey = f"{r['ey_mean']:.2f}" if not np.isnan(r['ey_mean']) else "N/A"
        sol = f"{r['solve_mean_ms']:.0f}" if not np.isnan(r['solve_mean_ms']) else "N/A"
        print(f"{r['theta']:>6.1f} {r['epsilon']:>6.2f} {status:>6} {r['total_steps']:>6} {crash:>6} {ey:>10} {sol:>10}")
    print("="*80)

    # 生成可视化
    generate_heatmaps(results, args.output_dir)
    generate_line_plots(results, args.output_dir)

    print("\n完成！")


if __name__ == '__main__':
    main()
