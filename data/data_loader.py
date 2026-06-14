"""
数据加载和预处理，用于Deep Koopman模型训练。

本模块负责：
1. 加载已有的预处理训练数据（格式为.npz）
2. 对数据进行子采样，将原始dt=0.01s转换为论文要求的dt=0.1s
3. 创建滑动窗口序列，用于多步预测损失计算
4. 构建PyTorch DataLoader，支持训练集和验证集划分

数据流程：
原始数据(Ts=0.01s) -> 重构完整轨迹 -> 子采样(dt=0.1s) -> 
滑动窗口分割 -> 训练/验证集划分 -> DataLoader
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和多维数组操作
import json  # 导入JSON模块，用于读取归一化参数文件
import os  # 导入操作系统接口模块，用于文件和路径操作
import torch  # 导入PyTorch深度学习框架
from torch.utils.data import DataLoader, TensorDataset  # 导入PyTorch数据加载工具

import sys  # 导入系统模块，用于修改Python路径
# 将项目根目录添加到系统路径，确保可以导入同级别的模块（如config.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入数据加载和训练相关参数
from config import (
    DATA_NPZ_PATH,  # 训练数据.npz文件的路径
    NORM_JSON_PATH,  # 归一化参数.json文件的路径
    SUBSAMPLE_RATE,  # 子采样率（10表示每10个样本取1个，0.01s->0.1s）
    K_PRED,  # 多步预测时域（滑动窗口长度）
    VAL_SPLIT,  # 验证集比例（0.1表示10%数据用于验证）
    BATCH_SIZE  # 批次大小（每次训练的样本数）
)


def _legacy_to_canonical_state_order(x):
    """Convert [px, py, v, psi, omega] -> [px, py, psi, v, omega]."""
    x_new = x.copy()
    x_new[..., 2] = x[..., 3]
    x_new[..., 3] = x[..., 2]
    return x_new


def load_norm_params(json_path=NORM_JSON_PATH):
    """
    加载归一化参数（px和py的均值和标准差）。

    归一化参数在数据预处理阶段计算并保存，用于：
    1. 训练前对位置坐标(px, py)进行标准化
    2. 推理后对位置坐标进行反标准化
    3. 确保不同状态的数值范围相近，提高训练稳定性

    参数:
        json_path: 字符串，归一化参数JSON文件的路径
                   默认使用config.py中的NORM_JSON_PATH

    返回:
        params: 字典，包含以下键值对：
            - 'px_mean': px的均值
            - 'px_std': px的标准差
            - 'py_mean': py的均值
            - 'py_std': py的标准差
    """
    # 以只读模式打开JSON文件
    with open(json_path, 'r') as f:
        params = json.load(f)  # 从JSON文件加载参数字典
    return params


def normalize_state(x, norm_params):
    """
    对状态向量中的px和py进行归一化（标准化）。

    使用z-score标准化方法：
        x_normalized = (x - mean) / std

    该函数支持单个状态向量和批量状态向量。

    参数:
        x: numpy数组，状态向量
           - 形状(5,): 单个状态 [px, py, psi, v, omega]
           - 形状(N, 5): N个状态的批量数据
        norm_params: 字典，包含px_mean, px_std, py_mean, py_std

    返回:
        x: numpy数组，归一化后的状态向量（形状与输入相同）
           只有px和py被归一化，其他状态保持不变
    """
    x = x.copy()  # 创建副本，避免修改原始数据
    # 判断是单个状态还是批量状态
    if x.ndim == 1:
        # 单个状态向量（形状为(5,)）
        # 对px进行归一化：px = (px - px_mean) / px_std
        x[0] = (x[0] - norm_params['px_mean']) / norm_params['px_std']
        # 对py进行归一化：py = (py - py_mean) / py_std
        x[1] = (x[1] - norm_params['py_mean']) / norm_params['py_std']
    else:
        # 批量状态向量（形状为(N, 5)）
        # 对所有样本的px进行归一化
        x[:, 0] = (x[:, 0] - norm_params['px_mean']) / norm_params['px_std']
        # 对所有样本的py进行归一化
        x[:, 1] = (x[:, 1] - norm_params['py_mean']) / norm_params['py_std']
    return x


def denormalize_state(x, norm_params):
    """
    对状态向量中的px和py进行反归一化（逆标准化）。

    使用z-score标准化的逆运算：
        x_original = x_normalized * std + mean

    该函数用于：
    1. 将模型预测的归一化结果转换回原始坐标
    2. 在可视化和评估时使用真实的物理坐标

    参数:
        x: numpy数组，归一化的状态向量
           - 形状(5,): 单个状态
           - 形状(N, 5): N个状态的批量数据
        norm_params: 字典，包含px_mean, px_std, py_mean, py_std

    返回:
        x: numpy数组，反归一化后的状态向量（形状与输入相同）
           只有px和py被反归一化，其他状态保持不变
    """
    x = x.copy()  # 创建副本，避免修改原始数据
    # 判断是单个状态还是批量状态
    if x.ndim == 1:
        # 单个状态向量（形状为(5,)）
        # 对px进行反归一化：px = px * px_std + px_mean
        x[0] = x[0] * norm_params['px_std'] + norm_params['px_mean']
        # 对py进行反归一化：py = py * py_std + py_mean
        x[1] = x[1] * norm_params['py_std'] + norm_params['py_mean']
    else:
        # 批量状态向量（形状为(N, 5)）
        # 对所有样本的px进行反归一化
        x[:, 0] = x[:, 0] * norm_params['px_std'] + norm_params['px_mean']
        # 对所有样本的py进行反归一化
        x[:, 1] = x[:, 1] * norm_params['py_std'] + norm_params['py_mean']
    return x


def load_and_subsample(npz_path=DATA_NPZ_PATH, norm_json_path=NORM_JSON_PATH,
                       subsample_rate=SUBSAMPLE_RATE):
    """
    加载预处理训练数据并进行子采样，以匹配论文的dt=0.1s。

    数据转换流程：
    原始数据(Ts=0.01s) -> 重构完整轨迹 -> 子采样(dt=0.1s) -> 控制/干扰平均

    原始数据的采样时间为0.01s，但论文要求使用0.1s的采样时间。
    因此需要对数据进行子采样（每10个样本取1个），并对控制输入
    和干扰向量在子采样窗口内进行平均，以保持能量守恒。

    参数:
        npz_path: 字符串，.npz数据文件的路径
        norm_json_path: 字符串，归一化参数JSON文件的路径
        subsample_rate: 整数，子采样率（默认10，即0.01s->0.1s）

    返回:
        X_sub: numpy数组，形状(M, 5)，子采样后的状态轨迹
        U_sub: numpy数组，形状(M-1, 2)，平均后的控制输入
        W_sub: numpy数组，形状(M-1, 5) 或 None，平均后的干扰向量
        norm_params: 字典，包含px_mean, px_std, py_mean, py_std
    """
    # 从.npz文件加载数据
    data = np.load(npz_path)
    # X_t: 当前时刻的状态，形状(N, 5)，已经归一化
    X_t = data['X_t']
    # U_t: 当前时刻的控制输入，形状(N, 2)
    U_t = data['U_t']
    # X_t1: 下一时刻的状态，形状(N, 5)，已经归一化
    X_t1 = data['X_t1']
    # W_t: 干扰向量（可选，新版本数据包含）
    has_w = 'W_t' in data.files
    W_t = data['W_t'] if has_w else None

    # 数据文件是历史顺序 [px, py, v, psi, omega]，统一转换为
    # 代码内部顺序 [px, py, psi, v, omega]
    X_t = _legacy_to_canonical_state_order(X_t)
    X_t1 = _legacy_to_canonical_state_order(X_t1)

    # 重构完整轨迹
    # 原始数据是连续的状态转移对：(X_t[0], X_t1[0]), (X_t[1], X_t1[1]), ...
    # 由于X_t1[i] = X_t[i+1]（连续数据），可以重构完整轨迹：
    # X_full_orig = [X_t[0], X_t1[0], X_t1[1], ..., X_t1[N-1]]
    N = X_t.shape[0]  # 获取原始数据点数
    X_full_orig = np.zeros((N + 1, 5))  # 预分配完整轨迹数组
    X_full_orig[0] = X_t[0]  # 第一个状态
    X_full_orig[1:] = X_t1  # 后续所有状态（X_t1包含了X_t[1:]）

    # 对状态进行子采样，每subsample_rate步取一个样本
    # 例如：subsample_rate=10，取索引0, 10, 20, 30, ...
    M = (N + 1) // subsample_rate  # 计算子采样后的数据点数
    X_sub = X_full_orig[::subsample_rate][:M]  # 子采样并截断到M个点

    # 对控制输入在子采样窗口内进行平均
    # 每个子采样间隔内的控制输入取平均值，以保持能量守恒
    U_sub = np.zeros((M - 1, 2))  # 预分配控制输入数组
    W_sub = np.zeros((M - 1, 5)) if has_w else None  # 预分配干扰数组
    for i in range(M - 1):  # 遍历每个子采样间隔
        start = i * subsample_rate  # 窗口起始索引
        end = min(start + subsample_rate, N)  # 窗口结束索引（不超过N）
        # 计算窗口内所有控制输入的平均值
        U_sub[i] = U_t[start:end].mean(axis=0)
        if has_w:
            W_sub[i] = W_t[start:end].mean(axis=0)

    # 加载归一化参数（用于px和py的标准化）
    norm_params = load_norm_params(norm_json_path)

    # 打印数据加载信息
    print(f"Data loaded: {N+1} original steps -> {M} subsampled "
          f"steps (rate={subsample_rate})")
    print(f"  X_sub shape: {X_sub.shape}")
    print(f"  U_sub shape: {U_sub.shape}")
    if has_w:
        print(f"  W_sub shape: {W_sub.shape}")

    return X_sub, U_sub, W_sub, norm_params


def create_sequence_windows(X, U, W=None, window_len=K_PRED):
    """
    创建滑动窗口序列，用于多步预测损失计算。

    该函数将长轨迹切分为多个重叠的窗口，每个窗口包含：
    - (window_len + 1)个连续状态：[x_t, x_{t+1}, ..., x_{t+K}]
    - window_len个连续控制：[u_t, u_{t+1}, ..., u_{t+K-1}]
    - window_len个连续干扰（可选）：[w_t, w_{t+1}, ..., w_{t+K-1}]

    这些窗口用于训练时的多步预测损失计算，帮助模型
    学习长期动态演化特性，而不仅是单步预测。

    参数:
        X: numpy数组，形状(M, 5)，状态轨迹
        U: numpy数组，形状(M-1, 2)，控制轨迹
        W: numpy数组，形状(M-1, 5)，干扰轨迹（可选）
        window_len: 整数，预测时域K（窗口长度）
                   默认使用config.py中的K_PRED（通常为10）

    返回:
        X_windows: numpy数组，形状(num_windows, window_len+1, 5)
        U_windows: numpy数组，形状(num_windows, window_len, 2)
        W_windows: numpy数组，形状(num_windows, window_len, 5) 或 None
    """
    M = X.shape[0]  # 获取轨迹长度
    # 计算可以创建的窗口数量
    # 每个窗口需要window_len+1个状态，所以num_windows = M - window_len
    num_windows = M - window_len
    # 检查数据量是否足够
    if num_windows <= 0:
        raise ValueError(
            f"Not enough data for window_len={window_len}, "
            f"have {M} steps"
        )

    # 预分配输出数组
    # X_windows: 每个窗口包含window_len+1个状态（从t到t+K）
    X_windows = np.zeros((num_windows, window_len + 1, 5))
    # U_windows: 每个窗口包含window_len个控制（从t到t+K-1）
    U_windows = np.zeros((num_windows, window_len, 2))
    # W_windows: 每个窗口包含window_len个干扰（从t到t+K-1）
    has_w = W is not None
    W_windows = np.zeros((num_windows, window_len, 5)) if has_w else None

    # 使用滑动窗口切分轨迹
    for i in range(num_windows):  # 遍历每个窗口
        # 提取状态窗口：从索引i到i+window_len（包含两端）
        X_windows[i] = X[i:i + window_len + 1]
        # 提取控制窗口：从索引i到i+window_len-1
        U_windows[i] = U[i:i + window_len]
        if has_w:
            W_windows[i] = W[i:i + window_len]

    # 打印创建的窗口数量
    print(f"Created {num_windows} sliding windows of length {window_len}")
    return X_windows, U_windows, W_windows


def create_datasets(X_sub, U_sub, W_sub=None, window_len=K_PRED,
                    val_split=VAL_SPLIT, batch_size=BATCH_SIZE):
    """
    创建PyTorch DataLoader，用于训练和验证。

    该函数执行以下操作：
    1. 使用滑动窗口将轨迹切分为多个序列样本
    2. 将numpy数组转换为PyTorch张量
    3. 随机打乱并划分训练集和验证集
    4. 创建DataLoader，支持批量加载和shuffle

    每个样本是一个窗口，包含：
    - (window_len+1)个状态：[x_t, x_{t+1}, ..., x_{t+K}]
    - window_len个控制：[u_t, u_{t+1}, ..., u_{t+K-1}]
    - window_len个干扰（可选）：[w_t, w_{t+1}, ..., w_{t+K-1}]

    参数:
        X_sub: numpy数组，形状(M, 5)，子采样后的状态轨迹
        U_sub: numpy数组，形状(M-1, 2)，子采样后的控制轨迹
        W_sub: numpy数组，形状(M-1, 5)，子采样后的干扰轨迹（可选）
        window_len: 整数，预测时域K（窗口长度）
        val_split: 浮点数，验证集比例（0.1表示10%）
        batch_size: 整数，批次大小（默认256）

    返回:
        train_loader: PyTorch DataLoader，训练集数据加载器
        val_loader: PyTorch DataLoader，验证集数据加载器
    """
    # 使用滑动窗口创建序列样本
    X_windows, U_windows, W_windows = create_sequence_windows(
        X_sub, U_sub, W_sub, window_len
    )

    # 将numpy数组转换为PyTorch张量（float32类型）
    X_tensor = torch.tensor(X_windows, dtype=torch.float32)
    U_tensor = torch.tensor(U_windows, dtype=torch.float32)

    # 随机打乱并划分训练集和验证集
    N = X_tensor.shape[0]  # 获取样本总数
    indices = np.random.permutation(N)  # 生成随机排列的索引
    split = int(N * (1 - val_split))  # 计算训练集和验证集的分割点
    train_idx = indices[:split]  # 训练集索引
    val_idx = indices[split:]  # 验证集索引

    # 创建TensorDataset（将X和U配对，如有W则一并配对）
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

    # 创建DataLoader
    # 训练集：shuffle=True（每个epoch打乱）
    # drop_last=True（丢弃最后不完整的batch）
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    # 验证集：shuffle=False（不需要打乱）
    # drop_last=False（保留所有样本）
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False
    )

    # 打印数据集信息
    print(f"Train: {len(train_dataset)} samples, "
          f"Val: {len(val_dataset)} samples")
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}")

    return train_loader, val_loader


def create_single_step_datasets(X_sub, U_sub, W_sub=None, val_split=VAL_SPLIT,
                                batch_size=BATCH_SIZE):
    """
    创建单步(x_t, u_t, w_t, x_{t+1}) DataLoader。

    与create_datasets不同，该函数创建的是单步转移样本，
    而不是多步序列。每个样本包含：
    - 当前状态 x_t
    - 当前控制 u_t
    - 干扰 w_t（可选）
    - 下一状态 x_{t+1}

    这种格式适用于：
    1. 简单的单步预测训练
    2. 模型评估和测试
    3. 不需要多步预测损失的场景

    参数:
        X_sub: numpy数组，形状(M, 5)，状态轨迹
        U_sub: numpy数组，形状(M-1, 2)，控制轨迹
        W_sub: numpy数组，形状(M-1, 5)，干扰轨迹（可选）
        val_split: 浮点数，验证集比例
        batch_size: 整数，批次大小

    返回:
        train_loader: PyTorch DataLoader，训练集
                     每个batch为(X_t, U_t, X_t1) 或 (X_t, U_t, W_t, X_t1)
        val_loader: PyTorch DataLoader，验证集
    """
    # 创建单步转移对
    # X_t: 所有时刻t的状态（除了最后一个）
    X_t = torch.tensor(X_sub[:-1], dtype=torch.float32)
    # U_t: 所有时刻t的控制
    U_t = torch.tensor(U_sub, dtype=torch.float32)
    # X_t1: 所有时刻t+1的状态（除了第一个）
    X_t1 = torch.tensor(X_sub[1:], dtype=torch.float32)

    # 随机打乱并划分训练集和验证集
    N = X_t.shape[0]  # 获取样本总数
    indices = np.random.permutation(N)  # 生成随机排列的索引
    split = int(N * (1 - val_split))  # 计算分割点

    # 创建训练集和验证集
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

    # 创建DataLoader
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
    主程序：测试数据加载和预处理功能。

    当直接运行此脚本时（python data_loader.py），
    执行以下测试：
    1. 加载并子采样数据
    2. 显示归一化参数
    3. 显示状态和控制输入的范围
    4. 创建数据集并测试DataLoader
    """
    # 加载并子采样数据
    X_sub, U_sub, W_sub, norm_params = load_and_subsample()
    # 打印归一化参数
    print(f"\nNorm params: {norm_params}")

    # 显示归一化后状态的范围
    print("State ranges (normalized):")
    for i, name in enumerate(['px', 'py', 'v', 'psi', 'omega']):
        print(f"  {name}: [{X_sub[:, i].min():.4f}, "
              f"{X_sub[:, i].max():.4f}]")

    # 显示控制输入的范围
    print("Control ranges:")
    for i, name in enumerate(['a', 'delta']):
        print(f"  {name}: [{U_sub[:, i].min():.4f}, "
              f"{U_sub[:, i].max():.4f}]")

    if W_sub is not None:
        print("Disturbance ranges:")
        for i, name in enumerate(['w_px', 'w_py', 'w_v', 'w_psi', 'w_omega']):
            print(f"  {name}: [{W_sub[:, i].min():.4f}, "
                  f"{W_sub[:, i].max():.4f}]")

    # 创建训练和验证DataLoader
    train_loader, val_loader = create_datasets(X_sub, U_sub, W_sub)

    # 测试从DataLoader中获取一个batch
    first_batch = next(iter(train_loader))
    batch_shapes = [f"{t.shape}" for t in first_batch]
    print(f"\nBatch shapes: {', '.join(batch_shapes)}")
    print(f"  X contains states at t=0..{K_PRED}, "
          f"U contains controls at t=0..{K_PRED-1}")
    if len(first_batch) == 3:
        print(f"  W contains disturbances at t=0..{K_PRED-1}")
