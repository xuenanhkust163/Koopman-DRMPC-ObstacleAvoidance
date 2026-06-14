"""
训练数据生成脚本。

使用车辆动力学模型在多条赛道上进行开放环仿真，
收集状态转移对 (x_t, u_t, x_{t+1})，用于 Deep Koopman 模型训练。

本脚本生成的数据格式与 data_loader.py 兼容：
    - X_t:   当前状态，形状 (N, 5)，顺序 [px, py, v, psi, omega]（legacy order）
    - U_t:   当前控制，形状 (N, 2)，顺序 [a, delta]
    - X_t1:  下一状态，形状 (N, 5)，顺序 [px, py, v, psi, omega]
    - Ts:    采样时间

其中 px, py 已经过 z-score 归一化处理。
"""

import argparse
import json
import os
import sys

import numpy as np

# 将项目根目录添加到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    A_MAX,
    A_MIN,
    DELTA_MAX,
    MODEL_DIR,
    ORIGINAL_TS,
    TRACK_HALF_WIDTH,
    V_MAX,
    V_MIN,
)
from tracks.custom_track import CustomWindingTrack
from tracks.lusail_short_track import LusailShortTrack
from tracks.lusail_track import LusailTrack
from tracks.sprint_oval_track import SprintOvalTrack
from tracks.straight_track import StraightTrack
from vehicle.bicycle_model import (
    discrete_step,
    discrete_step_batch,
    discrete_step_batch_with_disturbance,
    discrete_step_with_disturbance,
)

from config import NOMINAL_SIGMA

# 训练数据中的干扰强度（统一从 config.NOMINAL_SIGMA 取，避免多处硬编码）
DISTURBANCE_SIGMA = NOMINAL_SIGMA

# 轨迹模式下控制序列的随机游走标准差
DU_SIGMA = np.array([0.2, 0.01])


def _canonical_to_legacy(x):
    """Convert [px, py, psi, v, omega] -> [px, py, v, psi, omega]."""
    x_new = x.copy()
    x_new[..., 2] = x[..., 3]  # v
    x_new[..., 3] = x[..., 2]  # psi
    return x_new


def sample_state_on_track(track, rng):
    """
    在指定赛道上随机采样一个合法初始状态。

    参数:
        track: BaseTrack 子类实例
        rng:   numpy.random.Generator

    返回:
        x: 形状 (5,) 的状态向量 [px, py, psi, v, omega]
    """
    cx, cy = track.get_centerline()
    N = len(cx)

    # 随机选择一个中心线索引
    idx = rng.integers(0, N)

    # 获取该点的几何属性
    px, py = cx[idx], cy[idx]
    heading = track.get_heading()[idx]
    curvature = track.get_curvature()[idx]

    # 随机横向偏移（限制在赛道宽度 80% 以内，避免出界）
    lat_offset = rng.uniform(-TRACK_HALF_WIDTH * 0.8, TRACK_HALF_WIDTH * 0.8)

    # 法向量（指向赛道左侧）
    nx = -np.sin(heading)
    ny = np.cos(heading)
    px += lat_offset * nx
    py += lat_offset * ny

    # 随机速度 [m/s]
    v = rng.uniform(V_MIN + 1.0, V_MAX * 0.9)

    # 航向角：以赛道切向为基准，添加随机扰动
    psi = heading + rng.uniform(-np.deg2rad(15), np.deg2rad(15))

    # 角速度：基于 v * curvature，添加随机扰动
    omega = v * curvature + rng.uniform(-1.0, 1.0)
    omega = np.clip(omega, -10.0, 10.0)

    return np.array([px, py, psi, v, omega])


def is_out_of_bounds(x, track):
    px, py = x[0], x[1]
    try:
        _, _, lat_err = track.closest_point(px, py)
        return abs(lat_err) > TRACK_HALF_WIDTH
    except Exception:
        return True


def sample_smooth_controls(seq_len, rng, du_sigma=None):
    if du_sigma is None:
        du_sigma = DU_SIGMA
    u_seq = np.zeros((seq_len, 2))
    u_seq[0] = [
        rng.uniform(A_MIN * 0.5, A_MAX * 0.5),
        rng.uniform(-DELTA_MAX * 0.5, DELTA_MAX * 0.5),
    ]
    for t in range(1, seq_len):
        du = rng.normal(0.0, du_sigma)
        u_seq[t] = u_seq[t - 1] + du
        u_seq[t, 0] = np.clip(u_seq[t, 0], A_MIN, A_MAX)
        u_seq[t, 1] = np.clip(u_seq[t, 1], -DELTA_MAX, DELTA_MAX)
    return u_seq


def sample_adversarial_state_on_track(track, rng, scenario="hard_braking"):
    """
    采样对抗性初始状态。

    scenario:
        - "hard_braking": 在弯道处高速（超过安全速度）
        - "deceleration": 在直道处高速，然后持续刹车
        - "large_deviation": 大横向偏离（3~5米）
    """
    cx, cy = track.get_centerline()
    N = len(cx)
    curvatures = np.abs(track.get_curvature())

    if scenario == "hard_braking":
        # 选择曲率最大的前30%位置
        threshold = np.percentile(curvatures, 70)
        valid_idx = np.where(curvatures >= threshold)[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(N)
        idx = rng.choice(valid_idx)
        v = rng.uniform(6.0, 8.0)  # 超过弯道安全速度
    elif scenario == "deceleration":
        # 选择曲率最小的前30%位置（直道）
        threshold = np.percentile(curvatures, 30)
        valid_idx = np.where(curvatures <= threshold)[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(N)
        idx = rng.choice(valid_idx)
        v = rng.uniform(6.0, 8.0)
    elif scenario == "sustained_turn":
        # 选择曲率适中的位置（弯道但非急弯）
        threshold_low = np.percentile(curvatures, 30)
        threshold_high = np.percentile(curvatures, 70)
        valid_idx = np.where((curvatures >= threshold_low) & (curvatures <= threshold_high))[0]
        if len(valid_idx) == 0:
            valid_idx = np.arange(N)
        idx = rng.choice(valid_idx)
        v = rng.uniform(4.0, 6.0)  # 适中速度
    else:  # large_deviation
        idx = rng.integers(0, N)
        v = rng.uniform(V_MIN + 1.0, V_MAX * 0.5)

    px, py = cx[idx], cy[idx]
    heading = track.get_heading()[idx]
    curvature = track.get_curvature()[idx]

    if scenario == "large_deviation":
        # 大横向偏离 3~5 米
        lat_offset = rng.choice([-1, 1]) * rng.uniform(3.0, 5.0)
    else:
        lat_offset = rng.uniform(-TRACK_HALF_WIDTH * 0.5, TRACK_HALF_WIDTH * 0.5)

    # 法向量（指向赛道左侧）
    nx = -np.sin(heading)
    ny = np.cos(heading)
    px += lat_offset * nx
    py += lat_offset * ny

    # 航向角：以赛道切向为基准，添加随机扰动
    psi = heading + rng.uniform(-np.deg2rad(20), np.deg2rad(20))

    # 角速度：基于 v * curvature，添加随机扰动
    omega = v * curvature + rng.uniform(-1.0, 1.0)
    omega = np.clip(omega, -10.0, 10.0)

    return np.array([px, py, psi, v, omega])


def sample_adversarial_controls(seq_len, rng, scenario="hard_braking"):
    """采样对抗性控制序列。"""
    u_seq = np.zeros((seq_len, 2))

    if scenario == "hard_braking":
        # 持续强刹车 + 大转向
        for t in range(seq_len):
            a = rng.uniform(-1.0, -0.3)
            delta = rng.uniform(-DELTA_MAX * 0.8, DELTA_MAX * 0.8)
            u_seq[t] = [a, delta]
    elif scenario == "deceleration":
        # 持续刹车，偶尔松一下，小转向
        for t in range(seq_len):
            a = rng.uniform(-1.0, 0.0)
            delta = rng.uniform(-DELTA_MAX * 0.3, DELTA_MAX * 0.3)
            u_seq[t] = [a, delta]
    elif scenario == "sustained_turn":
        # 持续固定方向转向，建立 omega->psi 明确相关性
        turn_dir = rng.choice([-1, 1])
        for t in range(seq_len):
            a = rng.uniform(0.0, 0.3)
            if t < seq_len // 3:
                delta = turn_dir * rng.uniform(DELTA_MAX * 0.5, DELTA_MAX * 0.9)
            elif t < 2 * seq_len // 3:
                delta = turn_dir * rng.uniform(DELTA_MAX * 0.3, DELTA_MAX * 0.7)
            else:
                delta = turn_dir * rng.uniform(0.0, DELTA_MAX * 0.3)
            u_seq[t] = [a, delta]
    else:  # large_deviation
        # 大转向回正 + 小油门维持速度
        for t in range(seq_len):
            a = rng.uniform(0.0, 0.3)
            delta = rng.uniform(-DELTA_MAX, DELTA_MAX)
            u_seq[t] = [a, delta]

    return u_seq


# 对抗性场景列表
ADVERSARIAL_SCENARIOS = ["hard_braking", "deceleration", "large_deviation", "sustained_turn"]


def generate_data(n_samples=100000, dt=0.1, seed=42, mode="trajectory", seq_len=30):
    """
    生成训练数据。

    参数:
        n_samples: 总样本数
        dt:        离散时间步长 [秒]
        seed:      随机种子
        mode:      'random' 单步随机采样，'trajectory' 短轨迹 rollout
        seq_len:   轨迹模式下的最大步数

    返回:
        X_t:         numpy 数组，形状 (N, 5)，legacy order，px/py 已归一化
        U_t:         numpy 数组，形状 (N, 2)
        W_t:         numpy 数组，形状 (N, 5)，干扰向量
        X_t1:        numpy 数组，形状 (N, 5)，legacy order，px/py 已归一化
        norm_params: dict，包含 px_mean, px_std, py_mean, py_std
    """
    rng = np.random.default_rng(seed)
    tracks = [
        LusailTrack(),
        SprintOvalTrack(),
        CustomWindingTrack(),
        # LusailShortTrack(),  # 暂移除：其 _place_obstacles 参数与 TRACK_HALF_WIDTH=12 不兼容（rect_length=[22,34] 远超半宽）
        StraightTrack(),
    ]

    X_t_list = []
    U_t_list = []
    W_t_list = []
    X_t1_list = []

    print(f"Generating {n_samples} samples with dt={dt}s, mode={mode}, disturbance_sigma={DISTURBANCE_SIGMA} ...")

    if mode == "random":
        batch_size = 1000
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            current_batch = min(batch_size, n_samples - batch_idx * batch_size)
            track_choices = rng.integers(0, len(tracks), size=current_batch)

            x_batch = np.zeros((current_batch, 5))
            u_batch = np.zeros((current_batch, 2))

            for i in range(current_batch):
                track = tracks[track_choices[i]]
                x_batch[i] = sample_state_on_track(track, rng)
                a = rng.uniform(A_MIN, A_MAX)
                delta = rng.uniform(-DELTA_MAX, DELTA_MAX)
                u_batch[i] = [a, delta]

            w_batch = rng.normal(0.0, DISTURBANCE_SIGMA, size=(current_batch, 5))
            x1_batch = discrete_step_batch_with_disturbance(
                x_batch, u_batch, w_batch, dt=dt
            )

            X_t_list.append(x_batch)
            U_t_list.append(u_batch)
            W_t_list.append(w_batch)
            X_t1_list.append(x1_batch)

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches} done")
    else:
        # 轨迹模式：短轨迹 rollout（混合 10% 对抗性样本）
        collected = 0
        batch_size = 1000
        n_batches = (n_samples + batch_size - 1) // batch_size
        adversarial_ratio = 0.10  # 10% 对抗性样本

        for batch_idx in range(n_batches):
            current_batch = min(batch_size, n_samples - collected)
            x_batch = np.zeros((current_batch, 5))
            u_batch = np.zeros((current_batch, 2))
            w_batch = np.zeros((current_batch, 5))
            x1_batch = np.zeros((current_batch, 5))

            # 决定本批次中对抗性样本的数量
            n_adversarial = max(1, int(current_batch * adversarial_ratio))
            adversarial_indices = set(rng.choice(current_batch, size=n_adversarial, replace=False))

            i = 0
            while i < current_batch:
                track = tracks[rng.integers(0, len(tracks))]

                if i in adversarial_indices:
                    # 对抗性样本
                    scenario = rng.choice(ADVERSARIAL_SCENARIOS)
                    x = sample_adversarial_state_on_track(track, rng, scenario=scenario)
                    u_seq = sample_adversarial_controls(seq_len, rng, scenario=scenario)
                else:
                    # 正常样本
                    x = sample_state_on_track(track, rng)
                    u_seq = sample_smooth_controls(seq_len, rng)

                for t in range(seq_len):
                    w = rng.normal(0.0, DISTURBANCE_SIGMA, size=5)
                    x1 = discrete_step_with_disturbance(x, u_seq[t], w, dt=dt)
                    x_batch[i] = x
                    u_batch[i] = u_seq[t]
                    w_batch[i] = w
                    x1_batch[i] = x1
                    i += 1
                    x = x1
                    if i >= current_batch:
                        break
                    if is_out_of_bounds(x, track):
                        break

            X_t_list.append(x_batch)
            U_t_list.append(u_batch)
            W_t_list.append(w_batch)
            X_t1_list.append(x1_batch)
            collected += current_batch

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches} done ({collected}/{n_samples})")

    # 合并所有 batch
    X_t = np.vstack(X_t_list)
    U_t = np.vstack(U_t_list)
    W_t = np.vstack(W_t_list)
    X_t1 = np.vstack(X_t1_list)

    # 精确截断到目标数量
    X_t = X_t[:n_samples]
    U_t = U_t[:n_samples]
    W_t = W_t[:n_samples]
    X_t1 = X_t1[:n_samples]

    # 计算 px, py 的 z-score 归一化参数
    px_mean = float(X_t[:, 0].mean())
    px_std = float(X_t[:, 0].std())
    py_mean = float(X_t[:, 1].mean())
    py_std = float(X_t[:, 1].std())

    # 避免标准差为 0
    if px_std < 1e-6:
        px_std = 1.0
    if py_std < 1e-6:
        py_std = 1.0

    norm_params = {
        "px_mean": px_mean,
        "px_std": px_std,
        "py_mean": py_mean,
        "py_std": py_std,
    }

    # 对 px, py 进行 z-score 归一化
    X_t_norm = X_t.copy()
    X_t1_norm = X_t1.copy()
    X_t_norm[:, 0] = (X_t[:, 0] - px_mean) / px_std
    X_t_norm[:, 1] = (X_t[:, 1] - py_mean) / py_std
    X_t1_norm[:, 0] = (X_t1[:, 0] - px_mean) / px_std
    X_t1_norm[:, 1] = (X_t1[:, 1] - py_mean) / py_std

    # 转换为 legacy order [px, py, v, psi, omega]
    X_t_legacy = _canonical_to_legacy(X_t_norm)
    X_t1_legacy = _canonical_to_legacy(X_t1_norm)

    return X_t_legacy, U_t, W_t, X_t1_legacy, norm_params


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for Deep Koopman model"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500000,
        help="Number of samples to generate (default: 500000)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step [s] (default: 0.01)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="trajectory",
        choices=["random", "trajectory"],
        help="Data generation mode: 'random' or 'trajectory' (default: trajectory)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=500,
        help="Max trajectory length in steps for trajectory mode (default: 500)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=MODEL_DIR,
        help=f"Output directory (default: {MODEL_DIR})",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_t, U_t, W_t, X_t1, norm_params = generate_data(
        n_samples=args.n_samples,
        dt=args.dt,
        seed=args.seed,
        mode=args.mode,
        seq_len=args.seq_len,
    )

    # 保存 .npz 数据文件
    npz_path = os.path.join(args.output_dir, "training_data.npz")
    np.savez(npz_path, X_t=X_t, U_t=U_t, W_t=W_t, X_t1=X_t1, Ts=args.dt)
    print(f"Saved training data to {npz_path}")
    print(f"  X_t:   {X_t.shape}")
    print(f"  U_t:   {U_t.shape}")
    print(f"  W_t:   {W_t.shape}")
    print(f"  X_t1:  {X_t1.shape}")

    # 保存归一化参数
    json_path = os.path.join(args.output_dir, "norm_params.json")
    with open(json_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Saved norm params to {json_path}")
    print(f"  {norm_params}")


if __name__ == "__main__":
    main()
