"""
运动学自行车模型用于车辆动力学建模（论文第2节）。

状态变量: x = [px, py, psi, v, omega]^T ∈ R^5
    - px: 车辆在全局坐标系中的x位置 [米]
    - py: 车辆在全局坐标系中的y位置 [米]
    - v:  车辆纵向速度 [米/秒]
    - psi: 航向角（车辆朝向与全局x轴的夹角）[弧度]
    - omega: 横摆角速度（航向角变化率）[弧度/秒]

控制输入: u = [a, delta]^T ∈ R^2
    - a: 纵向加速度 [米/秒^2]（正值为加速，负值为减速）
    - delta: 前轮转向角 [弧度]（正值为左转，负值为右转）

动力学方程（连续时间）:
    px_dot  = v * cos(psi)              # x位置变化率由速度在x方向的分量决定
    py_dot  = v * sin(psi)              # y位置变化率由速度在y方向的分量决定
    v_dot   = a                          # 速度变化率等于加速度
    psi_dot = v * tan(delta) / L        # 航向角变化率由速度和转向角决定
    omega   = v * tan(delta) / L        # 横摆角速度等于航向角变化率（运动学约束）

应用场景:
    - LMPC（线性模型预测控制）：使用线性化模型
    - NMPC（非线性模型预测控制）：使用完整非线性模型
    - 仿真环境：作为被控对象的真实动力学模型
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和多维数组操作
import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径

# 将项目根目录添加到系统路径，确保可以导入同级别的模块（如config.py）
# os.path.dirname(os.path.abspath(__file__)) 获取当前文件的父目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件中导入车辆和仿真参数
from config import (  # noqa: F401  # A_MIN和A_MAX在当前文件中未使用，但在其他地方可能需要
    L_WHEELBASE,  # 车辆轴距 [米]，前后轴之间的距离
    DT,           # 采样时间 [秒]，离散化时间步长
    V_MIN,        # 最小速度 [米/秒]，速度下限约束
    V_MAX,        # 最大速度 [米/秒]，速度上限约束
    A_MIN,        # 最大减速度 [米/秒^2]，负值表示制动
    A_MAX,        # 最大加速度 [米/秒^2]，正值表示加速
    DELTA_MAX,    # 最大转向角 [弧度]，前轮转向角度限制
    IDX_PSI,      # 航向角索引
    IDX_V,        # 速度索引
    IDX_OMEGA     # 角速度索引
)


def continuous_dynamics(x, u, L=L_WHEELBASE):
    """
    连续时间运动学自行车模型。

    该函数实现车辆的连续时间动力学方程，描述了
    在给定状态和控制输入下的状态导数（变化率）。

    参数:
        x: 形状为(5,)的numpy数组，表示当前状态
           [px, py, psi, v, omega]
           - px: x位置坐标 [米]
           - py: y位置坐标 [米]
           - v:  纵向速度 [米/秒]
           - psi: 航向角 [弧度]
           - omega: 横摆角速度 [弧度/秒]
        u: 形状为(2,)的numpy数组，表示控制输入
           [a, delta]
           - a: 纵向加速度 [米/秒^2]
           - delta: 前轮转向角 [弧度]
        L: 标量，车辆轴距 [米]，默认使用配置中的L_WHEELBASE

    返回:
        x_dot: 形状为(5,)的numpy数组，表示状态导数
               [px_dot, py_dot, v_dot, psi_dot, omega_dot]
    """
    # 解包状态变量，提高代码可读性
    px, py, psi, v, omega = x
    # 解包控制输入变量
    a, delta = u

    # 对转向角进行裁剪，防止数值不稳定
    # 保留1e-6的余量避免tan函数在±π/2处的奇点
    delta = np.clip(delta, -DELTA_MAX + 1e-6, DELTA_MAX - 1e-6)
    # 确保速度不为零，避免除零错误和数值问题
    # 当速度接近零时，使用最小值1e-3代替
    v = max(v, 1e-3)

    # 计算状态导数（连续时间动力学方程）
    # x方向位置变化率 = 速度 × cos(航向角)
    px_dot = v * np.cos(psi)
    # y方向位置变化率 = 速度 × sin(航向角)
    py_dot = v * np.sin(psi)
    # 速度变化率 = 加速度（直接控制）
    v_dot = a
    # 航向角变化率 = (速度 × tan(转向角)) / 轴距
    # 这是运动学自行车模型的核心方程
    psi_dot = v * np.tan(delta) / L
    # 横摆角速度动态：一阶系统跟踪航向角变化率
    # tau是时间常数，控制omega收敛到psi_dot的速度
    # tau=0.05表示快速收敛（相对于dt=0.1）
    tau = 0.05  # 快速收敛时间常数，相对于采样时间dt=0.1秒
    omega_dot = (psi_dot - omega) / tau  # 一阶动态方程

    # 返回状态导数向量
    return np.array([px_dot, py_dot, psi_dot, v_dot, omega_dot])


def discrete_step(x, u, dt=DT, L=L_WHEELBASE):
    """
    使用四阶龙格-库塔法（RK4）进行离散时间单步积分。

    RK4是一种高精度的数值积分方法，比简单的欧拉法
    更准确，特别适合非线性系统的离散化。

    RK4公式:
        k1 = f(x, u)                        # 起点斜率
        k2 = f(x + dt/2 * k1, u)            # 中点斜率（用k1预测）
        k3 = f(x + dt/2 * k2, u)            # 中点斜率（用k2预测）
        k4 = f(x + dt * k3, u)              # 终点斜率
        x_next = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    参数:
        x: 形状为(5,)的numpy数组，当前状态向量
        u: 形状为(2,)的numpy数组，控制输入向量
        dt: 标量，时间步长 [秒]，默认使用配置中的DT
        L: 标量，车辆轴距 [米]，默认使用配置中的L_WHEELBASE

    返回:
        x_next: 形状为(5,)的numpy数组，下一个时间步的状态
    """
    # 计算四个斜率点（RK4方法的核心）
    # k1: 在当前状态处计算导数
    k1 = continuous_dynamics(x, u, L)
    # k2: 用k1预测的中点状态处计算导数
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, L)
    # k3: 用k2预测的中点状态处计算导数（更准确的中点估计）
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, L)
    # k4: 用k3预测的终点状态处计算导数
    k4 = continuous_dynamics(x + dt * k3, u, L)

    # RK4加权平均：中点斜率权重是端点的两倍
    # 这个公式使得RK4达到四阶精度（局部误差O(dt^5)）
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 强制速度约束，确保物理合理性
    # 速度不能低于V_MIN（通常为0），不能超过V_MAX
    x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)

    return x_next


def discrete_step_with_disturbance(x, u, w, dt=DT, L=L_WHEELBASE):
    """
    使用RK4进行带干扰的离散时间单步积分。

    干扰 w 直接叠加到连续时间状态导数上，通过RK4传播到下一状态：
        x_dot = f(x, u) + w

    参数:
        x: 形状为(5,)的numpy数组，当前状态 [px, py, psi, v, omega]
        u: 形状为(2,)的numpy数组，控制输入 [a, delta]
        w: 形状为(5,)的numpy数组，干扰向量 [w_px, w_py, w_psi, w_v, w_omega]
        dt: 标量，时间步长 [秒]
        L: 标量，车辆轴距 [米]

    返回:
        x_next: 形状为(5,)的numpy数组，下一时刻状态（已受干扰影响）
    """
    # 计算确定性部分的四个斜率
    k1 = continuous_dynamics(x, u, L)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, L)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, L)
    k4 = continuous_dynamics(x + dt * k3, u, L)

    # RK4积分确定性动力学
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    # 叠加干扰影响（干扰直接作用于状态增量）
    x_next = x_next + w * dt

    # 强制速度约束
    x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)
    return x_next


def discrete_step_batch_with_disturbance(x_batch, u_batch, w_batch,
                                         dt=DT, L=L_WHEELBASE):
    """
    批量带干扰离散时间步计算。

    参数:
        x_batch: 形状为(N, 5)的numpy数组，N个状态向量
        u_batch: 形状为(N, 2)的numpy数组，N个控制输入
        w_batch: 形状为(N, 5)的numpy数组，N个干扰向量
        dt: 标量，时间步长 [秒]
        L: 标量，车辆轴距 [米]

    返回:
        x_next_batch: 形状为(N, 5)的numpy数组，N个下一状态
    """
    N = x_batch.shape[0]
    x_next = np.zeros_like(x_batch)
    for i in range(N):
        x_next[i] = discrete_step_with_disturbance(
            x_batch[i], u_batch[i], w_batch[i], dt, L
        )
    return x_next


def discrete_step_batch(x_batch, u_batch, dt=DT, L=L_WHEELBASE):
    """
    批量离散时间步计算，用于同时处理多个状态-控制对。

    该函数在数据处理和模型训练中非常有用，可以
    高效地生成大量状态转移样本。

    参数:
        x_batch: 形状为(N, 5)的numpy数组，包含N个状态向量
                 每一行是一个5维状态 [px, py, psi, v, omega]
        u_batch: 形状为(N, 2)的numpy数组，包含N个控制输入
                 每一行是一个2维控制 [a, delta]
        dt: 标量，时间步长 [秒]
        L: 标量，车辆轴距 [米]

    返回:
        x_next_batch: 形状为(N, 5)的numpy数组，包含N个下一状态
                      x_next_batch[i] 是 x_batch[i] 在 u_batch[i] 作用下的下一状态
    """
    # 获取批量大小（样本数量）
    N = x_batch.shape[0]
    # 预分配输出数组，与输入形状相同，初始化为零
    x_next = np.zeros_like(x_batch)
    # 逐个样本进行离散时间步进
    # 注意：这里使用循环而不是向量化，因为discrete_step包含非线性操作
    for i in range(N):
        x_next[i] = discrete_step(x_batch[i], u_batch[i], dt, L)
    return x_next


def linearize(x_op, u_op, dt=DT, L=L_WHEELBASE, eps=1e-5):
    """
    在工作点(x_op, u_op)附近对离散时间动力学进行线性化。

    线性化模型形式:
        x_{t+1} ≈ A_d @ (x - x_op) + B_d @ (u - u_op) + x_op_next

    其中:
        - A_d是状态雅可比矩阵 ∂f/∂x
        - B_d是控制雅可比矩阵 ∂f/∂u
        - x_op_next是工作点处的名义下一状态

    使用中心差分法计算雅可比矩阵，比前向差分更精确：
        ∂f/∂x_i ≈ [f(x + eps*e_i) - f(x - eps*e_i)] / (2*eps)

    参数:
        x_op: 形状为(5,)的numpy数组，线性化工作点的状态
        u_op: 形状为(2,)的numpy数组，线性化工作点的控制输入
        dt: 标量，时间步长 [秒]
        L: 标量，车辆轴距 [米]
        eps: 标量，有限差分的扰动量，默认1e-5
             太大会导致线性化不准确，太小会导致数值误差

    返回:
        A_d: 形状为(5, 5)的numpy数组，离散状态雅可比矩阵
             A_d[i,j] = ∂x_next[i] / ∂x[j]
        B_d: 形状为(5, 2)的numpy数组，离散控制雅可比矩阵
             B_d[i,j] = ∂x_next[i] / ∂u[j]
        x_op_next: 形状为(5,)的numpy数组，工作点处的名义下一状态
                   x_op_next = f(x_op, u_op)
    """
    # 获取状态和控制输入的维度
    n_x = len(x_op)  # 状态维度，应为5
    n_u = len(u_op)  # 控制维度，应为2

    # 计算工作点处的名义下一状态
    x_op_next = discrete_step(x_op, u_op, dt, L)

    # 使用中心差分法计算状态雅可比矩阵A_d
    # A_d是一个5x5矩阵，表示每个状态变量对每个下一状态变量的影响
    A_d = np.zeros((n_x, n_x))  # 初始化为零矩阵
    for i in range(n_x):  # 对每个状态维度进行扰动
        # 正向扰动：第i个状态增加eps
        x_plus = x_op.copy()
        x_plus[i] += eps
        # 负向扰动：第i个状态减少eps
        x_minus = x_op.copy()
        x_minus[i] -= eps
        # 中心差分公式：[f(x+) - f(x-)] / (2*eps)
        A_d[:, i] = (discrete_step(x_plus, u_op, dt, L) -
                     discrete_step(x_minus, u_op, dt, L)) / (2 * eps)

    # 使用中心差分法计算控制雅可比矩阵B_d
    # B_d是一个5x2矩阵，表示每个控制输入对每个下一状态变量的影响
    B_d = np.zeros((n_x, n_u))  # 初始化为零矩阵
    for i in range(n_u):  # 对每个控制维度进行扰动
        # 正向扰动：第i个控制增加eps
        u_plus = u_op.copy()
        u_plus[i] += eps
        # 负向扰动：第i个控制减少eps
        u_minus = u_op.copy()
        u_minus[i] -= eps
        # 中心差分公式：[f(u+) - f(u-)] / (2*eps)
        B_d[:, i] = (discrete_step(x_op, u_plus, dt, L) -
                     discrete_step(x_op, u_minus, dt, L)) / (2 * eps)

    # 返回线性化结果
    return A_d, B_d, x_op_next


def casadi_dynamics():
    """
    创建CasADi符号动力学函数，用于NMPC（非线性模型预测控制）。

    CasADi是一个符号计算和自动微分库，可以：
    1. 构建符号表达式
    2. 自动计算雅可比矩阵和Hessian矩阵
    3. 生成高效的C代码
    4. 与IPOPT等优化求解器接口

    该函数使用RK4方法离散化连续时间动力学，
    并返回一个CasADi函数对象，可以在优化问题中调用。

    返回:
        f: CasADi函数对象，映射 (x, u) -> x_next
           可以直接在CasADi优化问题中使用
        x_sym: CasADi符号变量，形状为(5,)的状态符号
        u_sym: CasADi符号变量，形状为(2,)的控制符号
    """
    import casadi as ca  # 导入CasADi符号计算库

    # 创建符号变量
    # MX.sym创建一个5维的符号列向量，用于表示状态
    x_sym = ca.MX.sym('x', 5)
    # MX.sym创建一个2维的符号列向量，用于表示控制输入
    u_sym = ca.MX.sym('u', 2)

    # 使用配置中的车辆参数
    L = L_WHEELBASE  # 车辆轴距 [米]
    tau = 0.05  # 横摆角速度动态的时间常数 [秒]

    def _continuous(xx, uu):
        """
        辅助函数：计算给定状态和控制的连续时间动力学。

        该函数封装了连续时间动力学方程，用于RK4积分。
        使用CasADi符号运算而不是NumPy，以便后续自动微分。

        参数:
            xx: CasADi符号向量，状态 [px, py, psi, v, omega]
            uu: CasADi符号向量，控制 [a, delta]

        返回:
            CasADi符号向量，状态导数 [px_dot, py_dot, v_dot, psi_dot, omega_dot]
        """
        # 使用ca.fmax确保速度不为零（符号版本的max函数）
        # 1e-3是最小速度阈值，避免除零错误
        vv = ca.fmax(xx[IDX_V], 1e-3)
        # 计算航向角变化率（运动学约束）
        psi_dot = vv * ca.tan(uu[1]) / L
        # 返回状态导数向量（使用ca.vertcat垂直拼接）
        return ca.vertcat(
            vv * ca.cos(xx[IDX_PSI]),      # px_dot = v * cos(psi)
            vv * ca.sin(xx[IDX_PSI]),      # py_dot = v * sin(psi)
            psi_dot,                        # psi_dot = v * tan(delta) / L
            uu[0],                          # v_dot = a
            (psi_dot - xx[IDX_OMEGA]) / tau  # omega_dot = (psi_dot - omega) / tau
        )

    # 使用配置中的采样时间
    dt = DT

    # RK4四阶龙格-库塔积分（符号版本）
    # k1: 在当前状态处计算导数
    k1 = _continuous(x_sym, u_sym)
    # k2: 用k1预测的中点状态处计算导数
    k2 = _continuous(x_sym + 0.5 * dt * k1, u_sym)
    # k3: 用k2预测的中点状态处计算导数
    k3 = _continuous(x_sym + 0.5 * dt * k2, u_sym)
    # k4: 用k3预测的终点状态处计算导数
    k4 = _continuous(x_sym + dt * k3, u_sym)

    # RK4加权平均公式，计算下一状态
    x_next = x_sym + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # 创建CasADi函数对象
    # 参数说明:
    #   'bicycle_dynamics': 函数名称
    #   [x_sym, u_sym]: 输入符号变量列表
    #   [x_next]: 输出符号表达式列表
    #   ['x', 'u']: 输入参数名称（用于文档和调试）
    #   ['x_next']: 输出参数名称
    f = ca.Function(
        'bicycle_dynamics', [x_sym, u_sym], [x_next],
        ['x', 'u'], ['x_next']
    )

    # 返回函数对象和符号变量（符号变量可用于后续的优化问题构建）
    return f, x_sym, u_sym
