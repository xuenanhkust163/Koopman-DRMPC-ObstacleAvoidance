"""
线性MPC (LMPC)基线控制器。
在每个工作点使用Jacobian线性化的自行车模型。

LMPC作为基线方法的特点：
1. 使用传统的Jacobian线性化，而非Koopman提升线性化
2. 在每个时间步围绕当前工作点进行线性化
3. 适用于小幅偏离工作点的场景
4. 计算复杂度适中，但精度受限于线性化假设

与K-MPC/K-DRMPC的区别：
- LMPC：在物理空间中使用Jacobian线性化
- K-MPC：在Koopman空间中使用全局线性模型（学习得到）
- K-DRMPC：在Koopman空间中考虑干扰不确定性的鲁棒控制
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和多维数组操作
import casadi as ca  # 导入CasADi库，用于符号计算和非线性优化（IPOPT求解器）
import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径
import time  # 导入时间模块，用于计算求解耗时

# 将父目录添加到系统路径，确保可以导入同级别的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入MPC相关参数
from config import (
    N_X,  # 物理状态维度，默认5 [px, py, psi, v, omega]
    N_U,  # 控制输入维度，默认2 [加速度a, 转向角delta]
    T_HORIZON,  # 预测时域长度，默认20步
    DT,  # 时间步长，默认0.1秒
    Q_WEIGHTS,  # 状态跟踪权重向量
    R_WEIGHTS,  # 控制输入权重向量
    V_MIN,  # 最小速度 [米/秒]
    V_MAX,  # 最大速度 [米/秒]
    A_MIN,  # 最小加速度 [米/秒^2]
    A_MAX,  # 最大加速度 [米/秒^2]
    DELTA_MAX,  # 最大转向角 [弧度]
    DELTA_RATE_MAX,  # 最大转向角速率 [弧度/秒]
    D_SAFE,  # 安全距离余量 [米]
    VEHICLE_RADIUS,  # 车辆半径 [米]
    IPOPT_MAX_ITER,  # IPOPT求解器最大迭代次数
    IPOPT_PRINT_LEVEL,  # IPOPT求解器打印级别
    IDX_V,
    IDX_OMEGA
)

# 从自行车模型模块导入线性化函数
from vehicle.bicycle_model import linearize


class LMPCController:
    """
    使用Jacobian线性化自行车模型的线性MPC控制器。

    核心思想：
    1. 在每个时间步，围绕当前工作点（状态、控制）线性化非线性自行车模型
    2. 使用线性化后的仿射系统进行预测和优化
    3. 线性化公式: x_{t+1} ≈ A_t * x_t + B_t * u_t + c_t
       其中: A_t = ∂f/∂x|_{x_op, u_op} (Jacobian矩阵)
            B_t = ∂f/∂u|_{x_op, u_op} (控制Jacobian矩阵)
            c_t = f(x_op, u_op) - A_t*x_op - B_t*u_op (常数项)

    优化问题形式：
        min_u Σ_t ||[v_t, omega_t] - [v_ref, omega_ref]||_Q^2 + ||Δu_t||_R^2
        s.t.  x_{t+1} = A_t*x_t + B_t*u_t + c_t  (线性化动力学)
              x_0 = x_current                    (初始条件)
              u_min <= u_t <= u_max              (控制约束)
              |Δdelta_t| <= delta_rate_max * dt  (转向角速率约束)
              v_min <= v_t <= v_max              (速度约束)
              ||pos_t - obs|| >= d_min           (障碍物约束)
    """

    def __init__(self):
        """
        初始化LMPC控制器。
        LMPC不需要Koopman模型，直接使用物理空间的线性化。
        """
        # 控制器名称，用于日志和结果标识
        self.name = "LMPC"

        # 初始化上一次控制输入为零向量
        # 用于计算控制增量和平滑性约束
        self.u_prev = np.zeros(N_U)

        # 热启动信息，用于加速下一次求解
        # 存储上一次的状态和控制轨迹作为初始猜测
        self._warm_start = None

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        求解LMPC优化问题。

        优化流程：
        1. 创建决策变量（状态和控制轨迹）
        2. 在当前工作点线性化自行车模型
        3. 构建线性化动力学约束
        4. 构建跟踪代价和控制平滑性代价
        5. 添加控制、状态和障碍物约束
        6. 使用IPOPT求解

        参数:
            x_current: numpy数组，形状为(5,)
                      当前状态 [px, py, psi, v, omega]
            ref_trajectory: numpy数组，形状为(T, 5)
                          参考轨迹
            obstacles: 列表，每个元素为(ox, oy, radius)
                      障碍物位置和半径
            u_prev: numpy数组，形状为(2,)
                   上一次的控制输入 [a, delta]

        返回:
            u_opt: numpy数组，形状为(2,)
                  最优控制输入 [加速度a, 转向角delta]
            solve_info: 字典，包含求解信息
                       {'solve_time', 'status', 'method'}
        """
        # 如果提供了上一次控制输入，更新内部存储
        if u_prev is not None:
            self.u_prev = u_prev

        # 获取预测时域长度（默认20步）
        T = T_HORIZON

        # 记录求解开始时间
        t_start = time.time()

        # ================================================================
        # 步骤1：创建CasADi优化问题
        # ================================================================
        # 创建Opti对象，用于构建非线性规划（NLP）问题
        opti = ca.Opti()

        # 决策变量
        # X: 状态轨迹，形状为(N_X, T+1) = (5, 21)
        # 包含初始状态和预测时域内的所有状态
        X = opti.variable(N_X, T + 1)

        # U: 控制轨迹，形状为(N_U, T) = (2, 20)
        # 预测时域内的所有控制输入
        U = opti.variable(N_U, T)

        # 参数：初始状态
        # 使用parameter而不是固定值，便于热启动和参数更新
        x0_param = opti.parameter(N_X)
        opti.set_value(x0_param, x_current)

        # 初始条件约束: x_0 = x_current
        opti.subject_to(X[:, 0] == x0_param)

        # ================================================================
        # 步骤2：在当前状态线性化自行车模型
        # ================================================================
        # linearize函数返回:
        #   A_d: 离散时间状态转移矩阵 (N_X x N_X)
        #   B_d: 离散时间控制矩阵 (N_X x N_U)
        #   x_nom: 标称状态（用于线性化）
        A_d, B_d, x_nom = linearize(x_current, self.u_prev)

        # ================================================================
        # 步骤3：构建线性化动力学约束
        # ================================================================
        # 遍历预测时域的每一步，应用线性化动力学
        for t in range(T):
            # 获取当前步的工作点（用于线性化）
            if t == 0:
                # 第一步：使用当前真实状态和上一次控制输入
                x_op = x_current
                u_op = self.u_prev
            else:
                # 后续步：使用参考轨迹和零控制（近似）
                x_op = ref_trajectory[min(t, len(ref_trajectory) - 1)]
                u_op = np.zeros(N_U)

            # 在工作点线性化自行车模型
            A_t, B_t, _ = linearize(x_op, u_op)

            # 计算仿射线性化动力学的常数项c_t
            # 公式: c_t = f(x_op, u_op) - A_t*x_op - B_t*u_op
            # 其中f(x_op, u_op)是离散时间自行车模型的一步演化
            from vehicle.bicycle_model import discrete_step
            c_t = discrete_step(x_op, u_op) - A_t @ x_op - B_t @ u_op

            # 构建仿射动力学约束: x_{t+1} = A_t*x_t + B_t*u_t + c_t
            x_next = (
                ca.mtimes(ca.DM(A_t), X[:, t])
                + ca.mtimes(ca.DM(B_t), U[:, t])
                + ca.DM(c_t)
            )

            # 添加动力学约束到优化问题
            opti.subject_to(X[:, t + 1] == x_next)

        # ================================================================
        # 步骤4：构建代价函数
        # ================================================================
        # 代价函数：跟踪[v, omega] + 控制平滑性
        cost = 0
        Q = ca.DM(Q_WEIGHTS)  # 状态跟踪权重
        R = ca.DM(R_WEIGHTS)  # 控制输入权重

        # 遍历预测时域，累加每一步的代价
        for t in range(T):
            # 获取当前步的参考状态
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]

            # 跟踪速度v（索引2）和角速度omega（索引4）
            y_t = ca.vertcat(X[IDX_V, t], X[IDX_OMEGA, t])  # [v, omega]
            y_ref = ca.DM([ref_t[IDX_V], ref_t[IDX_OMEGA]])  # [v_ref, omega_ref]

            # 二次跟踪代价: (y - y_ref)^T * Q * (y - y_ref)
            cost += ca.mtimes([(y_t - y_ref).T, Q, (y_t - y_ref)])

            # 控制平滑性（惩罚控制增量）
            if t == 0:
                # 第一步使用上一次的控制输入
                du = U[:, t] - ca.DM(self.u_prev)
            else:
                # 其他步使用前一步的控制输入
                du = U[:, t] - U[:, t - 1]

            # 控制增量的二次代价: du^T * R * du
            cost += ca.mtimes([du.T, R, du])

        # 设置优化目标：最小化总代价
        opti.minimize(cost)

        # ================================================================
        # 步骤5：添加约束
        # ================================================================
        # 5.1 控制输入约束
        for t in range(T):
            # 加速度约束: A_MIN <= a_t <= A_MAX
            opti.subject_to(opti.bounded(A_MIN, U[0, t], A_MAX))
            # 转向角约束: -DELTA_MAX <= delta_t <= DELTA_MAX
            opti.subject_to(
                opti.bounded(-DELTA_MAX, U[1, t], DELTA_MAX)
            )

            # 转向角速率约束
            # 限制相邻时间步的转向角变化量
            if t == 0:
                # 第一步与上一次控制输入比较
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - self.u_prev[1],
                    DELTA_RATE_MAX * DT
                ))
            else:
                # 其他步与前一步比较
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - U[1, t - 1],
                    DELTA_RATE_MAX * DT
                ))

        # 5.2 状态约束
        for t in range(T + 1):
            # 速度约束: V_MIN <= v_t <= V_MAX
            opti.subject_to(opti.bounded(V_MIN, X[IDX_V, t], V_MAX))

        # 5.3 障碍物避免约束（线性化距离）
        # 遍历所有障碍物
        for obs in obstacles:
            ox, oy, r = obs
            # 计算最小安全距离
            d_min = r + VEHICLE_RADIUS + D_SAFE

            # 在预测时域的每一步检查障碍物约束
            for t in range(T + 1):
                # 计算车辆与障碍物的距离分量
                dx = X[0, t] - ox  # x方向距离
                dy = X[1, t] - oy  # y方向距离

                # 距离约束: ||pos - obs||^2 >= d_min^2
                # 这是一个非凸约束，IPOPT可以处理但可能陷入局部最优
                opti.subject_to(dx**2 + dy**2 >= d_min**2)

        # ================================================================
        # 步骤6：配置IPOPT求解器
        # ================================================================
        # IPOPT配置选项
        opts = {
            # 最大迭代次数
            'ipopt.max_iter': IPOPT_MAX_ITER,
            # 打印级别（0=静默，5=详细）
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            # 不打印求解时间
            'print_time': False,
            # 启用热启动（使用上一次的解作为初始猜测）
            'ipopt.warm_start_init_point': 'yes',
        }

        # 设置求解器为IPOPT（Interior Point Optimizer）
        opti.solver('ipopt', opts)

        # ================================================================
        # 步骤7：热启动（可选）
        # ================================================================
        # 如果有上一次的解，用作初始猜测
        # 热启动可以加速求解过程，特别是对于相似的问题
        if self._warm_start is not None:
            try:
                # 设置状态轨迹的初始猜测
                opti.set_initial(X, self._warm_start['X'])
                # 设置控制轨迹的初始猜测
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                # 如果设置失败，忽略并继续
                pass

        # ================================================================
        # 步骤8：求解优化问题
        # ================================================================
        try:
            # 调用IPOPT求解
            sol = opti.solve()

            # 提取第一步的最优控制输入
            # MPC只应用第一步的控制，然后在下一步重新优化
            u_opt = np.array(sol.value(U[:, 0])).flatten()

            # 求解状态：最优
            status = "optimal"

            # 保存当前解用于下一次热启动
            # 同时保存状态和控制轨迹，以便完全热启动
            self._warm_start = {
                'X': sol.value(X),
                'U': sol.value(U),
            }

        except Exception as e:
            # 如果求解失败，使用上一次的控制输入作为回退
            # LMPC没有次优解提取机制，直接使用回退策略
            u_opt = self.u_prev.copy()
            # 截断错误信息（最多50字符）
            status = f"failed: {str(e)[:50]}"

        # 计算求解耗时
        solve_time = time.time() - t_start

        # 更新上一次控制输入
        self.u_prev = u_opt.copy()

        # 返回最优控制输入和求解信息
        return u_opt, {
            'solve_time': solve_time,  # 求解耗时（秒）
            'status': status,  # 求解状态
            'method': self.name,  # 控制器名称
        }

    def reset(self):
        """
        重置控制器状态。

        在每次新的仿真开始时调用，清除上一次的控制输入和热启动信息。
        """
        # 重置上一次控制输入为零向量
        self.u_prev = np.zeros(N_U)

        # 清除热启动信息
        self._warm_start = None
