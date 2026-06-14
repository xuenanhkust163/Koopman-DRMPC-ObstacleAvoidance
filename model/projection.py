"""
通过岭回归（Ridge Regression）计算投影矩阵D。

投影矩阵D ∈ R^{2 × n_z}将Koopman潜在状态z映射到[v, omega]：
    [v, omega]^T ≈ D @ z

论文第3.5节：D = Y @ Z^T @ (Z @ Z^T + gamma * I)^{-1}

投影矩阵的作用：
1. 从Koopman空间提取控制相关变量（速度和角速度）
2. 在MPC优化中，用于构建控制相关的代价函数
3. 确保Koopman空间的表示能够准确反映物理量
4. 是连接线性Koopman模型和非线性控制目标的关键桥梁

岭回归的优势：
- 通过正则化防止过拟合
- 保证(Z^T Z + gamma * I)可逆
- 提高数值稳定性
"""

import os  # 导入操作系统接口模块，用于文件和目录操作
import sys  # 导入系统模块，用于修改Python路径
import numpy as np  # 导入NumPy库，用于高效的数值计算和矩阵运算
import torch  # 导入PyTorch深度学习框架，用于模型推理

# 将父目录添加到系统路径，确保可以导入同级别的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入投影矩阵相关参数
from config import (
    N_Z,  # Koopman空间维度，默认8
    GAMMA_RIDGE,  # 岭回归正则化参数，默认0.001
    MODEL_DIR,  # 模型保存目录
    IDX_V,
    IDX_OMEGA
)


def get_fixed_selector_matrices(n_z=N_Z):
    """
    构建论文控制段使用的固定线性选择器（非学习）。

    约定潜在状态前5维与物理状态同序对应：
        z[0:5] <-> [px, py, psi, v, omega]

    返回:
        D_pos: (2, n_z), 提取 [px, py]
        E_v:   (1, n_z), 提取 [v]
        F_omg: (1, n_z), 提取 [omega]
        D_vomg:(2, n_z), 提取 [v, omega]
    """
    D_pos = np.zeros((2, n_z))
    D_pos[0, 0] = 1.0  # px
    D_pos[1, 1] = 1.0  # py

    E_v = np.zeros((1, n_z))
    E_v[0, IDX_V] = 1.0

    F_omg = np.zeros((1, n_z))
    F_omg[0, IDX_OMEGA] = 1.0

    D_vomg = np.zeros((2, n_z))
    D_vomg[0, IDX_V] = 1.0
    D_vomg[1, IDX_OMEGA] = 1.0

    return D_pos, E_v, F_omg, D_vomg


def compute_projection_matrix(model, X_data, gamma=GAMMA_RIDGE, device='cpu'):
    """
    通过岭回归计算投影矩阵D。

    投影矩阵D将Koopman潜在状态z映射到控制相关变量[v, omega]：
        [v, omega]^T ≈ D @ z

    计算流程：
    1. 使用训练好的编码器将所有状态x编码为Koopman空间表示z
    2. 提取目标变量Y = [v, omega]
    3. 使用岭回归求解最优投影矩阵D
    4. 计算R^2分数评估拟合质量

    岭回归公式：
        D = Y @ Z^T @ (Z @ Z^T + gamma * I)^{-1}
    或者等价地（更高效的计算方式）：
        D^T = (Z^T @ Z + gamma * I)^{-1} @ Z^T @ Y

    参数:
        model: 训练好的DeepKoopmanPaper模型
              用于将状态编码到Koopman空间
        X_data: numpy数组，形状为(N, 5)
               状态数据 [px, py, psi, v, omega]
               N是样本数量
        gamma: 浮点数，岭回归正则化参数
              默认使用config.py中的GAMMA_RIDGE（0.001）
              gamma越大，正则化越强，防止过拟合
        device: 字符串，计算设备，'cpu'或'cuda'

    返回:
        D: numpy数组，形状为(2, n_z)
          投影矩阵，将n_z维Koopman状态映射到2维[v, omega]
        r2_score: numpy数组，形状为(2,)
                 R^2拟合优度分数
                 r2_score[0]: 速度v的R^2
                 r2_score[1]: 角速度omega的R^2
                 R^2越接近1表示拟合越好
    """
    # 将模型设置为评估模式
    # 这会禁用dropout等训练时特有的行为
    model.eval()

    # 将模型移动到指定设备
    model = model.to(device)

    # ================================================================
    # 步骤1：将所有状态编码到Koopman潜在空间
    # ================================================================
    # 将NumPy数组转换为PyTorch张量
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    # 使用模型的编码器将状态x映射到Koopman空间z
    # torch.no_grad()禁用梯度计算，节省内存和计算时间
    with torch.no_grad():
        # model.encode返回形状为(N, n_z)的Koopman状态
        Z = model.encode(X_tensor).cpu().numpy()

    # ================================================================
    # 步骤2：提取目标变量
    # ================================================================
    # 从状态数据中提取速度v和角速度omega
    # 状态顺序：[px(0), py(1), psi(2), v(3), omega(4)]
    Y = X_data[:, [IDX_V, IDX_OMEGA]]  # 形状：(N, 2)，列分别为[v, omega]

    # ================================================================
    # 步骤3：使用岭回归求解投影矩阵D
    # ================================================================
    # 岭回归公式（标准形式）：
    #     D = Y @ Z^T @ (Z @ Z^T + gamma * I)^{-1}
    # 但是直接计算这个公式效率较低，因为需要求逆(N × N)矩阵
    #
    # 使用等价形式（更高效）：
    #     D^T = (Z^T @ Z + gamma * I)^{-1} @ Z^T @ Y
    # 这个形式只需要求逆(n_z × n_z)矩阵，n_z << N

    # 获取数据维度
    # N = Z.shape[0]  # 样本数量（未使用，但保留注释说明）
    n_z = Z.shape[1]  # Koopman空间维度

    # 计算Z^T @ Z，形状为(n_z, n_z)
    # 这是Koopman状态的自相关矩阵
    ZtZ = Z.T @ Z

    # 计算Z^T @ Y，形状为(n_z, 2)
    # 这是Koopman状态和目标变量的互相关矩阵
    ZtY = Z.T @ Y

    # 求解线性方程组：(Z^T Z + gamma * I) D^T = Z^T Y
    # 使用np.linalg.solve比直接求逆更稳定、更高效
    # 方程形式: A X = B，求解X
    # 其中: A = ZtZ + gamma * I  (n_z × n_z)
    #       B = ZtY              (n_z × 2)
    #       X = D^T              (n_z × 2)
    D_T = np.linalg.solve(ZtZ + gamma * np.eye(n_z), ZtY)

    # 转置得到D，形状为(2, n_z)
    D = D_T.T

    # ================================================================
    # 步骤4：计算R^2拟合优度分数
    # ================================================================
    # 使用求得的投影矩阵D进行预测
    Y_pred = Z @ D.T  # 形状：(N, 2)，预测的[v, omega]

    # 计算残差平方和（Sum of Squares of Residuals）
    # ss_res = Σ(y - y_pred)^2
    # 对每个目标变量（v和omega）分别计算
    ss_res = np.sum((Y - Y_pred) ** 2, axis=0)

    # 计算总平方和（Total Sum of Squares）
    # ss_tot = Σ(y - y_mean)^2
    # 表示数据的总方差
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)

    # 计算R^2分数
    # R^2 = 1 - ss_res / ss_tot
    # R^2 = 1表示完美拟合，R^2 = 0表示拟合等于均值
    # 添加1e-12避免除零错误
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # 打印投影矩阵计算结果
    print(f"Projection matrix D computed: shape={D.shape}")
    print(f"  R^2 for v:     {r2[0]:.6f}")
    print(f"  R^2 for omega: {r2[1]:.6f}")
    print(f"  Mean R^2:      {r2.mean():.6f}")

    # 返回投影矩阵和R^2分数
    return D, r2


def save_projection_matrix(D, save_dir=MODEL_DIR, filename='projection_D.npy'):
    """
    将投影矩阵D保存到磁盘。

    保存的矩阵可以在后续的MPC仿真中直接加载，
    避免重复计算。

    参数:
        D: numpy数组，形状为(2, n_z)
          要保存的投影矩阵
        save_dir: 字符串，保存目录
                 默认使用config.py中的MODEL_DIR
        filename: 字符串，文件名
                 默认为'projection_D.npy'
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 构建完整的文件路径
    path = os.path.join(save_dir, filename)

    # 使用NumPy保存为.npy格式
    # .npy是NumPy的二进制格式，高效且保留数据类型
    np.save(path, D)

    # 打印保存成功信息
    print(f"Projection matrix saved to {path}")


def load_projection_matrix(save_dir=MODEL_DIR, filename='projection_D.npy'):
    """
    从磁盘加载投影矩阵D。

    该函数用于加载之前计算并保存的投影矩阵，
    供MPC控制器使用。

    参数:
        save_dir: 字符串，保存目录
                 默认使用config.py中的MODEL_DIR
        filename: 字符串，文件名
                 默认为'projection_D.npy'

    返回:
        D: numpy数组，形状为(2, n_z)
          加载的投影矩阵

    使用示例:
        D = load_projection_matrix()
        v_omega = D @ z  # 从Koopman状态提取[v, omega]
    """
    # 构建完整的文件路径
    path = os.path.join(save_dir, filename)

    # 从.npy文件加载投影矩阵
    D = np.load(path)

    return D
