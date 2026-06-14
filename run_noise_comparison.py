"""
噪音对照实验：鲁棒 K-DRMPC vs 非鲁棒 K-DRMPC
在蛇形赛道上，开启适度观测噪声 (measurement noise, sigma=0.5)，各跑完整圈。
观测噪声模式：真实轨迹不受噪声影响，控制器收到含噪状态估计。

本版本包含：
1. 鲁棒/非鲁棒双模式对比
2. 详细的性能表格对比
3. 红蓝双线 GIF 动画对比
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 先开启噪声 ----
import config as _cfg
_cfg.ENABLE_DISTURBANCE = True
_cfg.ENABLE_OBSTACLES = True

from config import (
    MODEL_DIR, RESULTS_DIR, FIGURES_DIR,
    N_DISTURBANCE_SAMPLES, THETA_WASSERSTEIN, EPSILON_CVAR, TRACK_HALF_WIDTH,
    NOMINAL_SIGMA,
)
from model.koopman_trainer import load_trained_model
from model.projection import load_projection_matrix
from tracks.serpentine_track import SerpentineTrack
from controllers.kdrmpc_controller import KDRMPCController
from disturbance.disturbance_generator import DisturbanceGenerator
from simulation.simulator import Simulator
from simulation.metrics import compute_all_metrics

# 同步 ENABLE_DISTURBANCE 到已导入模块
for mod_name in ['disturbance.disturbance_generator', 'simulation.simulator']:
    mod = sys.modules.get(mod_name)
    if mod and hasattr(mod, 'ENABLE_DISTURBANCE'):
        mod.ENABLE_DISTURBANCE = True

# 同步 ENABLE_OBSTACLES 到基类模块（BaseTrack 在导入时缓存了该配置）
base_track_mod = sys.modules.get('tracks.base_track')
if base_track_mod and hasattr(base_track_mod, 'ENABLE_OBSTACLES'):
    base_track_mod.ENABLE_OBSTACLES = True

# ---- 默认参数 ----
DEFAULT_SIGMA = NOMINAL_SIGMA  # Setting A (Nominal, x1.0)
DEFAULT_THETA = THETA_WASSERSTEIN
DEFAULT_EPSILON = EPSILON_CVAR
DEFAULT_SEED = 42
DEFAULT_STEPS = 3000
NUM_LAPS = 1
LAP_FRACTION = 1.0 * NUM_LAPS
TRACK_NAME = "SerpentineTrack"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_koopman():
    model = load_trained_model()
    D = load_projection_matrix()
    norm_path = os.path.join(MODEL_DIR, 'norm_params.json')
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    return model, D, norm_params


def cleanup_previous_outputs(tags, suffix=""):
    """每次开跑前清理旧结果文件（log/pkl/gif/png）。

    suffix 参与 tag/gif 的清理匹配，保证 run_all_seeds.sh 多 seed 跑时，
    只清理当前 suffix 对应的旧输出，不会误删其他 seed 的结果。
    """
    removed = []
    for tag in tags:
        base = f"{tag}_{TRACK_NAME}"
        candidates = [
            os.path.join(RESULTS_DIR, f"{base}.pkl"),
            os.path.join(RESULTS_DIR, "logs", f"{base}.log"),
            os.path.join(FIGURES_DIR, f"{base}_animation.gif"),
            os.path.join(FIGURES_DIR, f"{base}_animation.png"),
            os.path.join(FIGURES_DIR, f"{base}_final_frame.png"),
        ]
        for path in candidates:
            if os.path.exists(path):
                os.remove(path)
                removed.append(path)

    # 清理双线对比图（带 suffix）
    comparison_gif = os.path.join(
        FIGURES_DIR, f"Comparison_{TRACK_NAME}_dual_line{suffix}.gif"
    )
    if os.path.exists(comparison_gif):
        os.remove(comparison_gif)
        removed.append(comparison_gif)
    
    if removed:
        print("\n清理旧文件:")
        for p in removed:
            print(f"  - {p}")
    else:
        print("\n清理旧文件: 无需清理")


def run_experiment(model, D, norm_params, robust: bool, tag: str, sigma: float, theta: float, epsilon: float, seed: int, max_steps: int):
    """运行单次实验并返回结果。"""
    track = SerpentineTrack()
    dist_gen = DisturbanceGenerator(sigma=sigma, seed=seed)
    w_empirical = dist_gen.get_empirical_samples(N_DISTURBANCE_SAMPLES)

    obstacle_strategy = "robust" if robust else "non-robust"
    controller = KDRMPCController(
        model, D, norm_params,
        disturbance_samples=w_empirical,
        theta=theta,
        epsilon=epsilon,
        obstacle_strategy=obstacle_strategy,
    )

    simulator = Simulator(track, controller, dist_gen)
    result = simulator.run(max_steps=max_steps, verbose=True,
                           lap_fraction=LAP_FRACTION,
                           detailed_step_log=True,
                           detailed_step_log_max_steps=10)

    # 保存结果
    fname = f"{tag}_{TRACK_NAME}.pkl"
    Simulator.save_result(result, fname)

    return result, track


def print_comparison(tag, result, track):
    """打印单个实验的关键指标。"""
    states = np.array(result.states)
    controls = np.array(result.controls)
    solve_times = np.array(result.solve_times)

    # 横向误差
    e_y_list = []
    e_psi_list = []
    for i, st in enumerate(states):
        idx, s, lat_err = track.closest_point(st[0], st[1])
        ref_psi = track._heading[idx]
        e_psi = st[2] - ref_psi
        # 规范化角度到 [-pi, pi]
        e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi
        e_y_list.append(abs(lat_err))
        e_psi_list.append(abs(e_psi))

    e_y_arr = np.array(e_y_list)
    e_psi_arr = np.array(e_psi_list)

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(f"  总步数:        {result.total_steps}")
    print(f"  是否碰撞:      {'是 (step {})'.format(result.crash_step) if result.crashed else '否'}")
    if result.crashed:
        print(f"  碰撞原因:      {result.crash_reason}")
    print(f"  ---")
    print(f"  |e_y| Mean:    {np.mean(e_y_arr):.3f} m")
    print(f"  |e_y| P95:     {np.percentile(e_y_arr, 95):.3f} m")
    print(f"  |e_y| Max:     {np.max(e_y_arr):.3f} m")
    print(f"  |e_ψ| Mean:    {np.degrees(np.mean(e_psi_arr)):.2f}°")
    print(f"  |e_ψ| P95:     {np.degrees(np.percentile(e_psi_arr, 95)):.2f}°")
    print(f"  |e_ψ| Max:     {np.degrees(np.max(e_psi_arr)):.2f}°")
    print(f"  ---")
    print(f"  速度 Mean:     {np.mean(states[:, 3]):.3f} m/s")
    print(f"  速度 Max:      {np.max(states[:, 3]):.3f} m/s")
    print(f"  ---")
    print(f"  求解时间 Mean:  {np.mean(solve_times)*1000:.1f} ms")
    print(f"  求解时间 P95:   {np.percentile(solve_times, 95)*1000:.1f} ms")
    print(f"  求解时间 Max:   {np.max(solve_times)*1000:.1f} ms")
    within_2m = np.sum(e_y_arr <= 2.0) / len(e_y_arr) * 100
    print(f"  ±2m覆盖率:     {within_2m:.1f}%")

    return {
        'tag': tag,
        'steps': result.total_steps,
        'crashed': result.crashed,
        'crash_step': result.crash_step if result.crashed else None,
        'lap_completed': bool(getattr(result, 'lap_completed', False)),
        'lap_time': float(result.lap_time) if getattr(result, 'lap_time', None) is not None else None,
        'ey_mean': float(np.mean(e_y_arr)),
        'ey_p95': float(np.percentile(e_y_arr, 95)),
        'ey_max': float(np.max(e_y_arr)),
        'epsi_mean_deg': float(np.degrees(np.mean(e_psi_arr))),
        'epsi_p95_deg': float(np.degrees(np.percentile(e_psi_arr, 95))),
        'v_mean': float(np.mean(states[:, 3])),
        'solve_mean_ms': float(np.mean(solve_times) * 1000),
        'solve_p95_ms': float(np.percentile(solve_times, 95) * 1000),
        'within_2m_pct': float(within_2m),
    }


def generate_dual_line_animation(result_robust, result_nonrobust, track, output_path, fps=20, max_frames=300):
    """生成红蓝双线对比动画（鲁棒=蓝色，非鲁棒=红色）。

    对齐策略：取较长轨迹为基准，较短的用最后一帧填充到相同长度，
    这样两条轨迹都能完整显示到各自终点（早停策略的终止帧用 X 标出）。
    动画文本显示的 Step 是原始仿真步（非抽帧索引）。
    """
    states_r = np.array(result_robust.states)
    states_nr = np.array(result_nonrobust.states)

    # 记录原始终止步，用于 X 标记和信息文本
    orig_len_r = len(states_r)
    orig_len_nr = len(states_nr)

    # 对齐长度：以较长为准，较短的最后一帧重复填充
    max_len = max(orig_len_r, orig_len_nr)
    if orig_len_r < max_len:
        states_r = np.vstack([states_r, np.tile(states_r[-1], (max_len - orig_len_r, 1))])
    if orig_len_nr < max_len:
        states_nr = np.vstack([states_nr, np.tile(states_nr[-1], (max_len - orig_len_nr, 1))])

    # 采样帧数（记录下采样步长 stride，以便在动画里显示原始 step 编号）
    original_len = max_len
    stride = 1
    if max_frames and len(states_r) > max_frames:
        stride = len(states_r) // max_frames
        states_r = states_r[::stride]
        states_nr = states_nr[::stride]
    
    # 基于 0506 的通用赛道接口推导边界，避免依赖 PLUS 私有属性
    track_x, track_y = track.get_centerline()
    heading = track.get_heading()
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    boundary_left_x = track_x + TRACK_HALF_WIDTH * normal_x
    boundary_left_y = track_y + TRACK_HALF_WIDTH * normal_y
    boundary_right_x = track_x - TRACK_HALF_WIDTH * normal_x
    boundary_right_y = track_y - TRACK_HALF_WIDTH * normal_y
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # 固定全图视角，避免仅显示已行驶局部区域（例如起点左下角）
    all_track_x = np.concatenate([track_x, boundary_left_x, boundary_right_x])
    all_track_y = np.concatenate([track_y, boundary_left_y, boundary_right_y])
    map_margin = 20.0
    ax.set_xlim(np.min(all_track_x) - map_margin, np.max(all_track_x) + map_margin)
    ax.set_ylim(np.min(all_track_y) - map_margin, np.max(all_track_y) + map_margin)
    
    # 绘制轨道和边界
    ax.plot(track_x, track_y, 'k--', linewidth=1, alpha=0.3, label='Centerline')
    ax.plot(boundary_left_x, boundary_left_y, 'k-', linewidth=1, alpha=0.3)
    ax.plot(boundary_right_x, boundary_right_y, 'k-', linewidth=1, alpha=0.3)

    # 绘制障碍物（优先绘制矩形，回退到圆形）
    rect_obstacles = track.get_rect_obstacles()
    if rect_obstacles:
        for (cx, cy, length, width, angle) in rect_obstacles:
            # 手动计算旋转后的角点，确保围绕中心 (cx, cy) 旋转
            c = np.cos(angle)
            s = np.sin(angle)
            hl = length / 2.0
            hw = width / 2.0
            local_corners = np.array([
                [-hl, -hw],
                [hl, -hw],
                [hl, hw],
                [-hl, hw],
            ])
            rot = np.array([[c, -s], [s, c]])
            corners = (local_corners @ rot.T) + np.array([cx, cy])
            poly = patches.Polygon(
                corners,
                closed=True,
                linewidth=1.0,
                edgecolor='dimgray',
                facecolor='lightgray',
                alpha=0.45,
                zorder=1,
            )
            ax.add_patch(poly)
    else:
        for (ox, oy, r) in track.get_obstacles():
            circ = patches.Circle(
                (ox, oy),
                r,
                linewidth=1.0,
                edgecolor='dimgray',
                facecolor='lightgray',
                alpha=0.45,
                zorder=1,
            )
            ax.add_patch(circ)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Robust (Blue) vs Non-Robust (Red) Comparison on {TRACK_NAME}')
    
    # 初始化轨迹线
    line_r, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Robust K-DRMPC')
    line_nr, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label='Non-Robust K-DRMPC')
    point_r, = ax.plot([], [], 'bo', markersize=8)
    point_nr, = ax.plot([], [], 'ro', markersize=8)
    
    text_info = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right')

    n_r = len(states_r)
    n_nr = len(states_nr)
    n_frames = max(n_r, n_nr)

    # 撞车/终止标记（X），在初始化时创建，默认不可见
    crash_r, = ax.plot([], [], 'bx', markersize=12, markeredgewidth=2)
    crash_nr, = ax.plot([], [], 'rx', markersize=12, markeredgewidth=2)

    # 原始 step 下的终止点（未在原始仿真中终止的策略，取 None 不标记）
    # 只要该策略未完整跑完 original_len 就视为提前终止
    term_step_r = orig_len_r if orig_len_r < original_len else None
    term_step_nr = orig_len_nr if orig_len_nr < original_len else None

    def init():
        line_r.set_data([], [])
        line_nr.set_data([], [])
        point_r.set_data([], [])
        point_nr.set_data([], [])
        crash_r.set_data([], [])
        crash_nr.set_data([], [])
        return line_r, line_nr, point_r, point_nr, crash_r, crash_nr, text_info

    def animate(frame):
        # 原始 step（将抽帧索引 frame 换算回原始仿真步）
        current_original = min((frame + 1) * stride, original_len)

        # 绘制轨迹
        line_r.set_data(states_r[:min(frame+1, n_r), 0], states_r[:min(frame+1, n_r), 1])
        line_nr.set_data(states_nr[:min(frame+1, n_nr), 0], states_nr[:min(frame+1, n_nr), 1])

        # Robust 当前位置 / 终止标记
        if term_step_r is not None and current_original >= term_step_r:
            point_r.set_data([], [])
            idx_r = min(term_step_r // stride, n_r - 1)
            crash_r.set_data([states_r[idx_r, 0]], [states_r[idx_r, 1]])
        else:
            f_r = min(frame, n_r - 1)
            point_r.set_data([states_r[f_r, 0]], [states_r[f_r, 1]])
            crash_r.set_data([], [])

        # Non-Robust 当前位置 / 终止标记
        if term_step_nr is not None and current_original >= term_step_nr:
            point_nr.set_data([], [])
            idx_nr = min(term_step_nr // stride, n_nr - 1)
            crash_nr.set_data([states_nr[idx_nr, 0]], [states_nr[idx_nr, 1]])
        else:
            f_nr = min(frame, n_nr - 1)
            point_nr.set_data([states_nr[f_nr, 0]], [states_nr[f_nr, 1]])
            crash_nr.set_data([], [])

        # 信息文本（显示原始 step）
        info_text = f"Step: {current_original}/{original_len}\n"
        if term_step_r is not None and current_original >= term_step_r:
            info_text += f"Robust (B):    END at step {term_step_r}\n"
        else:
            f_r = min(frame, n_r - 1)
            info_text += f"Robust (B):    x={states_r[f_r, 0]:.1f}, y={states_r[f_r, 1]:.1f}\n"
        if term_step_nr is not None and current_original >= term_step_nr:
            info_text += f"NonRobust (R): END at step {term_step_nr}"
        else:
            f_nr = min(frame, n_nr - 1)
            info_text += f"NonRobust (R): x={states_nr[f_nr, 0]:.1f}, y={states_nr[f_nr, 1]:.1f}"
        text_info.set_text(info_text)

        return line_r, line_nr, point_r, point_nr, crash_r, crash_nr, text_info

    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_frames, interval=1000/fps,
                        blit=True, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"✓ 红蓝双线 GIF 已保存: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="噪音对照实验: Robust vs Non-Robust K-DRMPC")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="measurement noise sigma")
    parser.add_argument("--theta", type=float, default=DEFAULT_THETA, help="Wasserstein theta")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="CVaR epsilon")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="disturbance RNG seed")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="max simulation steps (default 3000)")
    parser.add_argument("--name-suffix", type=str, default="", help="输出文件名后缀（如 _S0p8_T0p08）")
    return parser.parse_args()


def main():
    args = parse_args()
    sigma = float(args.sigma)
    theta = float(args.theta)
    epsilon = float(args.epsilon)
    seed = int(args.seed)
    max_steps = int(args.steps)
    suffix = args.name_suffix

    robust_tag = f"Robust_K-DRMPC{suffix}"
    nonrobust_tag = f"NonRobust_K-DRMPC{suffix}"
    cleanup_previous_outputs([robust_tag, nonrobust_tag], suffix=suffix)

    print("=" * 60)
    print("噪音对照实验: Robust vs Non-Robust K-DRMPC")
    print(f"赛道: {TRACK_NAME} | {max_steps} 步 | sigma: {sigma}")
    print(f"theta={theta}, epsilon={epsilon}, seed={seed}, suffix='{suffix}'")
    print("=" * 60)

    model, D, norm_params = load_koopman()

    # ---- 实验1: 鲁棒模式 (CVaR + Wasserstein DR) ----
    print("\n\n" + "#" * 60)
    print("# 实验1: K-DRMPC 鲁棒模式 (CVaR + Wasserstein DR)")
    print("#" * 60)
    result_robust, track_r = run_experiment(
        model, D, norm_params,
        robust=True,
        tag=robust_tag,
        sigma=sigma,
        theta=theta,
        epsilon=epsilon,
        seed=seed,
        max_steps=max_steps,
    )

    # ---- 实验2: 非鲁棒模式 (确定性障碍约束) ----
    print("\n\n" + "#" * 60)
    print("# 实验2: K-DRMPC 非鲁棒模式 (确定性约束)")
    print("#" * 60)
    result_nonrobust, track_nr = run_experiment(
        model, D, norm_params,
        robust=False,
        tag=nonrobust_tag,
        sigma=sigma,
        theta=theta,
        epsilon=epsilon,
        seed=seed,
        max_steps=max_steps,
    )

    # ---- 对比报告 ----
    print("\n\n")
    print("█" * 60)
    print("█  对照实验结果汇总")
    print("█" * 60)
    metrics_r = print_comparison("Robust K-DRMPC (CVaR+Wasserstein DR)", result_robust, track_r)
    metrics_nr = print_comparison("Non-Robust K-DRMPC (确定性约束)", result_nonrobust, track_nr)

    # ---- 差异分析 ----
    print("\n\n" + "=" * 60)
    print("  差异分析 (Robust - NonRobust)")
    print("=" * 60)
    diff_data = []
    for key in ['steps', 'ey_mean', 'ey_p95', 'ey_max', 'epsi_mean_deg', 'v_mean', 'solve_mean_ms', 'within_2m_pct']:
        v_r = metrics_r[key]
        v_nr = metrics_nr[key]
        if v_r is not None and v_nr is not None:
            diff = v_r - v_nr
            label = key.replace('_', ' ')
            print(f"  {label:20s}: {v_r:>10.3f} vs {v_nr:>10.3f}  (Δ = {diff:+.3f})")
            diff_data.append((label, v_r, v_nr, diff))

    print("\n" + "=" * 60)
    print("完成！结果已保存到 _output/results/")
    print("=" * 60)

    # ---- 生成红蓝双线动画 ----
    print("\n\n" + "#" * 60)
    print("# 生成红蓝双线对比动画...")
    print("#" * 60)
    dual_gif_path = os.path.join(
        FIGURES_DIR, f"Comparison_{TRACK_NAME}_dual_line{suffix}.gif"
    )
    generate_dual_line_animation(result_robust, result_nonrobust, track_r, dual_gif_path, fps=20, max_frames=300)

    # 保存对比摘要
    summary = {
        'robust': metrics_r,
        'nonrobust': metrics_nr,
        'sigma': sigma,
        'theta': theta,
        'epsilon': epsilon,
        'seed': seed,
        'max_steps': max_steps,
        'suffix': suffix,
        'robust_tag': robust_tag,
        'nonrobust_tag': nonrobust_tag,
        'dual_gif': dual_gif_path,
    }
    suffix_name = suffix if suffix else ""
    summary_path = os.path.join(RESULTS_DIR, f'noise_comparison_summary{suffix_name}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n对比摘要已保存: {summary_path}")


if __name__ == '__main__':
    main()
