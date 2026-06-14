"""
Wasserstein模糊集和CVaR约束计算模块。
对应论文第5.2-5.5节。

提供CVaR安全裕度的事后评估功能，用于性能指标计算。
"""

import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入操作系统接口模块
import sys  # 导入系统模块，用于路径操作

# 将父目录添加到系统路径，以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从配置文件导入Wasserstein半径、CVaR风险水平、安全距离和车辆半径常量
from config import THETA_WASSERSTEIN, EPSILON_CVAR, D_SAFE, VEHICLE_RADIUS


def compute_cvar_margin(positions, obstacles, w_samples,
                        theta=THETA_WASSERSTEIN, epsilon=EPSILON_CVAR):
    """
    计算给定轨迹位置的最坏情况CVaR安全裕度。

    对每个障碍物和每个时间步，计算：
        sup_{P in D} CVaR_epsilon^P [l(x_t, w_t)]

    使用对偶重构公式（定理，第5.5节）：
        = inf_{lambda>=0} { lambda*theta + 1/(epsilon*N) * sum [l_i + lambda*||w_i||]_+ }

    Args:
        positions: 形状为(T, 2)的预测位置数组，包含[px, py]（物理坐标）
        obstacles: 障碍物列表，格式为[(ox, oy, radius), ...]
        w_samples: 形状为(N, 5)的干扰样本数组
        theta: Wasserstein半径（模糊集大小）
        epsilon: CVaR风险水平（通常很小，如0.05或0.1）

    Returns:
        max_cvar: 所有障碍物和时间步中的最大CVaR裕度
                  （负数表示安全，正数表示不安全）
        cvar_per_obs: 每个障碍物的CVaR值列表
    """
    from scipy.optimize import minimize_scalar  # 导入标量优化函数，用于求解对偶问题

    N = len(w_samples)  # 获取干扰样本数量

    # 预计算||w_i||（范数），避免重复计算
    w_norms = np.array([np.linalg.norm(w_samples[i]) for i in range(N)])  # 计算每个原始样本的范数

    cvar_per_obs = []  # 初始化每个障碍物的CVaR值列表

    for obs in obstacles:  # 遍历每个障碍物
        ox, oy, r = obs  # 解包障碍物信息：x坐标、y坐标、半径
        d_min = r + VEHICLE_RADIUS + D_SAFE  # 计算最小安全距离（障碍物半径+车辆半径+安全裕度）
        max_cvar_obs = -np.inf  # 初始化该障碍物的最大CVaR值为负无穷

        for t in range(len(positions)):  # 遍历每个时间步的位置
            px, py = positions[t]  # 获取当前位置坐标
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)  # 计算车辆到障碍物中心的欧氏距离
            l_nom = d_min - dist  # 计算标称安全裕度（正数表示安全，负数表示碰撞）

            # 对偶重构：寻找最优的lambda值
            def dual_objective(lam):
                """对偶目标函数：lambda*theta + (1/epsilon)*E[max(l_nom + lambda*||w_i||, 0)]"""
                terms = np.maximum(l_nom + lam * w_norms, 0)  # 计算[l_nom + lambda*||w_i||]_+
                return lam * theta + np.mean(terms) / epsilon  # 返回对偶目标函数值

            # 使用标量优化器在[0, 100]范围内寻找最优lambda
            result = minimize_scalar(dual_objective, bounds=(0, 100), method='bounded')
            cvar_value = result.fun  # 获取最优目标函数值（即CVaR值）

            max_cvar_obs = max(max_cvar_obs, cvar_value)  # 更新该障碍物的最大CVaR值

        cvar_per_obs.append(max_cvar_obs)  # 将该障碍物的最大CVaR值添加到列表

    max_cvar = max(cvar_per_obs) if cvar_per_obs else 0.0  # 获取所有障碍物中的最大CVaR值

    return max_cvar, cvar_per_obs  # 返回最大CVaR值和每个障碍物的CVaR值列表


def check_constraint_violation(positions, obstacles):
    """
    检查是否违反了任何障碍物距离约束。

    Args:
        positions: 形状为(T, 2)的轨迹数组，包含[px, py]
        obstacles: 障碍物列表，格式为[(ox, oy, radius), ...]

    Returns:
        violations: 形状为(T,)的布尔数组，如果在时间步t违反约束则为True
        min_distances: 形状为(T, n_obs)的数组，记录到每个障碍物的最小距离
    """
    T = len(positions)  # 获取时间步数
    n_obs = len(obstacles)  # 获取障碍物数量
    violations = np.zeros(T, dtype=bool)  # 初始化违反标志数组，全为False
    min_distances = np.zeros((T, n_obs))  # 初始化距离数组，记录每个时间步到每个障碍物的距离

    for j, obs in enumerate(obstacles):  # 遍历每个障碍物，j为索引
        ox, oy, r = obs  # 解包障碍物信息
        d_min = r + VEHICLE_RADIUS + D_SAFE  # 计算最小安全距离

        for t in range(T):  # 遍历每个时间步
            px, py = positions[t]  # 获取当前位置
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)  # 计算到障碍物中心的距离
            min_distances[t, j] = dist  # 记录该距离
            if dist < d_min:  # 如果距离小于安全距离
                violations[t] = True  # 标记该时间步为违反约束

    return violations, min_distances  # 返回违反标志数组和距离矩阵
