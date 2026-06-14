"""
Koopman MPC (K-MPC)控制器，不使用分布鲁棒约束。
在潜在空间中使用线性Koopman动力学（论文第3.5节、第4节）。

K-MPC的核心思想：
1. 使用训练好的Koopman模型将非线性系统提升到线性空间
2. 在Koopman空间中使用线性动力学进行预测和优化
3. 通过投影矩阵D提取控制相关变量[v, omega]
4. 通过解码器处理障碍物避免（非线性约束）

与K-DRMPC的区别：
- K-MPC：使用确定性的线性Koopman模型，不考虑干扰不确定性
- K-DRMPC：使用Wasserstein模糊集和CVaR约束，具有分布鲁棒性
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
    N_Z,  # Koopman空间维度，默认8
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
    Q_PSI_TRACK,
    Q_PROGRESS_TRACK,
    Q_POS_TRACK,
    POSITION_TERM_INTERVAL,
    Q_TERMINAL_HEADING,
    Q_TERMINAL_POS,
    IDX_PSI,
    IDX_V,
    IDX_OMEGA
)

# 从MPC公共模块导入编解码器转换函数
from controllers.mpc_common import (
    pytorch_to_casadi_encoder,  # PyTorch编码器转CasADi函数
    pytorch_to_casadi_decoder  # PyTorch解码器转CasADi函数
)
from controllers.tracking_costs import resolve_tracking_cost_builder
from model.projection import get_fixed_selector_matrices

# 障碍物接近阈值：只有车辆在这个距离内才添加障碍物约束
# 这样可以减少NLP的复杂度，只关注附近的障碍物
OBSTACLE_PROXIMITY = 200.0  # 单位：米

# 障碍物松弛变量的惩罚权重
# 较大的值会强制满足障碍物约束，但可能导致问题不可解
OBSTACLE_SLACK_PENALTY = 1000.0
# 航向角跟踪权重：增强“沿参考方向前进”的方向感
Q_PSI = Q_PSI_TRACK
# 前向进度权重：显式鼓励沿参考切向前进
Q_PROGRESS = Q_PROGRESS_TRACK


class KMPCController:
    """
    使用线性提升动力学的Koopman MPC控制器。

    核心特性：
    1. 代价函数通过投影矩阵D在潜在空间中构建
       - 使用D矩阵从Koopman状态z提取[v, omega]
       - 跟踪参考轨迹的速度和角速度

    2. 障碍物避免使用解码器（非线性->NLP）
       - 解码器将Koopman状态z映射回物理空间[px, py]
       - 在物理空间中计算与障碍物的距离
       - 由于解码器是非线性的，这使MPC成为非线性规划问题

    3. 位置跟踪也使用解码器
       - 每隔4步使用解码器计算位置误差
       - 平衡精度和计算复杂度

    优化问题形式：
        min_u Σ_t ||D*z_t - y_ref_t||_Q^2 + ||u_t||_R^2 + 位置跟踪 + 障碍物惩罚
        s.t.  z_{t+1} = A*z_t + B*u_t  (线性Koopman动力学)
              u_min <= u_t <= u_max    (控制约束)
              ||decoded_pos - obs|| >= d_min  (障碍物约束，软化)
    """

    @staticmethod
    def _clip_matrix_eigenvalues(A, radius=0.999):
        """将矩阵A所有模大于radius的特征值裁剪回单位圆内，防止预测时域内指数发散。"""
        eigenvalues, V = np.linalg.eig(A)
        magnitudes = np.abs(eigenvalues)
        clipped = np.where(
            magnitudes > radius,
            eigenvalues / magnitudes * radius,
            eigenvalues
        )
        A_clipped = V @ np.diag(clipped) @ np.linalg.inv(V)
        return np.real(A_clipped)

    def __init__(self, koopman_model, D_matrix, norm_params,
                 cost_builder=None, cost_profile="default"):
        """
        初始化K-MPC控制器。

        参数:
            koopman_model: 训练好的DeepKoopmanPaper模型
                          包含编码器、解码器和Koopman矩阵A, B
            D_matrix: numpy数组，形状为(2, n_z)
                     投影矩阵，用于从Koopman状态提取[v, omega]
            norm_params: 字典，包含归一化参数
                        {'px_mean', 'px_std', 'py_mean', 'py_std'}
        """
        # 控制器名称，用于日志和结果标识
        self.name = "K-MPC"

        # 初始化上一次控制输入为零向量
        # 用于计算控制增量和平滑性约束
        self.u_prev = np.zeros(N_U)

        # 热启动信息，用于加速下一次求解
        # 存储上一次的优化解作为初始猜测
        self._warm_start = None

        # ================================================================
        # 提取Koopman模型参数
        # ================================================================
        # 获取Koopman线性动力学矩阵
        # A: 状态转移矩阵 (n_z x n_z)
        # B: 控制矩阵 (n_z x n_u)
        # C: 干扰矩阵 (n_z x n_w)
        self.A, self.B, self.C = koopman_model.get_matrices()
        # 对A矩阵进行特征值裁剪，防止高维Koopman特征在预测时域内指数发散
        self.A = self._clip_matrix_eigenvalues(self.A, radius=0.9)

        # 论文固定线性选择器（控制主路径使用）
        # D_pos: z->[px,py], E: z->[v], F: z->[omega], D_vomega: z->[v,omega]
        self.D_pos, self.E, self.F, self.D_vomega = get_fixed_selector_matrices(
            self.A.shape[0]
        )
        # 保留外部传入D仅作诊断，不用于核心代价/约束
        self.D_diag = D_matrix

        # 保存归一化参数，用于状态编码和解码
        self.norm_params = norm_params

        # ================================================================
        # 构建CasADi函数
        # ================================================================
        # 获取神经网络权重
        weights = koopman_model.get_network_weights()

        # 将PyTorch编码器转换为CasADi函数
        # 这样可以在CasADi优化中使用编码器
        self._ca_encode = pytorch_to_casadi_encoder(weights)

        # 将PyTorch解码器转换为CasADi函数
        # 用于障碍物避免和位置跟踪
        self._ca_decode = pytorch_to_casadi_decoder(weights)

        # 将矩阵存储为CasADi DM（密集矩阵）类型
        # 这样可以在CasADi符号计算中直接使用
        self._A_ca = ca.DM(self.A)  # CasADi格式的A矩阵
        self._B_ca = ca.DM(self.B)  # CasADi格式的B矩阵
        self._D_pos_ca = ca.DM(self.D_pos)
        self._E_ca = ca.DM(self.E)
        self._F_ca = ca.DM(self.F)
        self._D_vomega_ca = ca.DM(self.D_vomega)
        D_psi = np.zeros((1, self.A.shape[0]))
        D_psi[0, IDX_PSI] = 1.0
        self._D_psi_ca = ca.DM(D_psi)
        # 可替换代价构建器：允许外部注入不同的stage cost定义
        self.cost_builder = resolve_tracking_cost_builder(
            cost_builder,
            profile=cost_profile,
        )

    def _encode_state(self, x_physical):
        """
        将物理状态编码为Koopman潜在状态。

        编码流程：
        1. 对px, py进行归一化（使用训练时的均值和标准差）
        2. 其他状态v, psi, omega已经归一化，不需要处理
        3. 使用神经网络编码器将归一化状态映射到Koopman空间

        参数:
            x_physical: numpy数组，形状为(5,)
                       物理状态 [px, py, psi, v, omega]

        返回:
            z: numpy数组，形状为(n_z,)
              Koopman潜在状态
        """
        # 复制物理状态，避免修改原始数据
        x_norm = x_physical.copy()

        # 对px（索引0）进行归一化
        # 公式: px_norm = (px - mean) / std
        x_norm[0] = (
            x_norm[0] - self.norm_params['px_mean']
        ) / self.norm_params['px_std']

        # 对py（索引1）进行归一化
        # 公式: py_norm = (py - mean) / std
        x_norm[1] = (
            x_norm[1] - self.norm_params['py_mean']
        ) / self.norm_params['py_std']

        # 将NumPy数组转换为CasADi列向量
        # reshape(-1, 1)确保形状为(5, 1)
        x_ca = ca.DM(x_norm.reshape(-1, 1))

        # 使用CasADi编码器计算潜在状态
        # 这是神经网络的前向传播
        z_ca = self._ca_encode(x_ca)

        # 将CasADi DM转换回NumPy数组并展平
        # 返回形状为(n_z,)的一维数组
        return np.array(z_ca).flatten()

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        求解K-MPC优化问题。

        优化问题构建流程：
        1. 编码当前状态到Koopman空间
        2. 构建参考轨迹（速度和角速度）
        3. 创建决策变量（控制序列）
        4. 使用线性Koopman动力学传播状态轨迹
        5. 构建代价函数（跟踪+平滑性+位置+障碍物）
        6. 添加约束（控制输入+速率+障碍物）
        7. 使用IPOPT求解NLP问题
        8. 返回最优控制输入

        参数:
            x_current: numpy数组，形状为(5,)
                      当前物理状态 [px, py, psi, v, omega]
            ref_trajectory: numpy数组，形状为(T, 5)
                          参考轨迹（物理坐标）
            obstacles: 列表，每个元素为(ox, oy, radius)
                      障碍物位置和半径（物理坐标）
            u_prev: numpy数组，形状为(2,)
                   上一次的控制输入 [a, delta]
                   如果为None，使用内部存储的u_prev

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
        # 步骤1：编码当前状态
        # ================================================================
        # 将当前物理状态编码为Koopman潜在状态
        z0 = self._encode_state(x_current)

        # ================================================================
        # 步骤2：构建参考轨迹（在[v, omega]空间）
        # ================================================================
        # 初始化参考轨迹数组，形状为(T, 2)
        # 每行包含[v_ref, omega_ref]
        y_ref = np.zeros((T, 2))

        # 遍历预测时域的每一步
        for t in range(T):
            # 获取当前时间步的参考状态
            # 使用min确保不会超出参考轨迹长度
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            # 提取速度v（索引2）和角速度omega（索引4）
            y_ref[t] = [ref_t[IDX_V], ref_t[IDX_OMEGA]]

        # ================================================================
        # 步骤3：创建CasADi优化问题
        # ================================================================
        # 创建Opti对象，用于构建非线性规划（NLP）问题
        opti = ca.Opti()

        # 决策变量：预测时域内的控制序列
        # U的形状为(N_U, T) = (2, 20)
        # U[0, :] = 加速度序列 [a_0, a_1, ..., a_{T-1}]
        # U[1, :] = 转向角序列 [delta_0, delta_1, ..., delta_{T-1}]
        U = opti.variable(N_U, T)

        # ================================================================
        # 步骤4：使用线性Koopman动力学传播状态轨迹
        # ================================================================
        # 潜在状态轨迹列表
        # z_0是固定的（当前状态的编码），不需要优化
        Z = [ca.DM(z0)]

        # 遍历预测时域，使用线性动力学传播状态
        # 公式: z_{t+1} = A*z_t + B*u_t
        for t in range(T):
            # 计算下一步的Koopman状态
            # ca.mtimes表示矩阵乘法
            z_next = ca.mtimes(self._A_ca, Z[-1]) + \
                     ca.mtimes(self._B_ca, U[:, t])
            # 添加到状态轨迹列表
            Z.append(z_next)

        # ================================================================
        # 步骤5：构建代价函数
        # ================================================================
        # 初始化总代价为0
        cost = 0

        # 创建权重矩阵（CasADi格式）
        Q = ca.DM(Q_WEIGHTS)  # [v, omega]跟踪权重
        R = ca.DM(R_WEIGHTS)  # 控制输入权重

        # 位置跟踪权重
        # 用于平衡[v, omega]跟踪和位置跟踪的重要性
        Q_pos = Q_POS_TRACK

        # 对参考轨迹的位置进行归一化
        # 用于与解码器输出（归一化空间）进行比较
        ref_px_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory)-1), 0]
             - self.norm_params['px_mean'])
            / self.norm_params['px_std']
            for t in range(T)
        ])
        ref_py_norm = np.array([
            (ref_trajectory[min(t, len(ref_trajectory)-1), 1]
             - self.norm_params['py_mean'])
            / self.norm_params['py_std']
            for t in range(T)
        ])
        ref_psi = np.array([
            ref_trajectory[min(t, len(ref_trajectory)-1), IDX_PSI]
            for t in range(T)
        ])
        ref_psi_terminal = float(ref_trajectory[min(T - 1, len(ref_trajectory) - 1), IDX_PSI])
        ref_px_norm_terminal = float(ref_px_norm[T - 1])
        ref_py_norm_terminal = float(ref_py_norm[T - 1])

        # 可选的风险项容器：某些builder会在其中追加每步损失用于CVaR等尾部项
        risk_terms = []

        # 遍历预测时域，累加每一步的代价（由可替换cost builder计算）
        for t in range(T):
            y_t = ca.mtimes(self._D_vomega_ca, Z[t])
            y_ref_t = ca.DM(y_ref[t])
            cost += self.cost_builder.stage_cost(
                opti=opti,
                t=t,
                z_t=Z[t],
                u_t=U[:, t],
                u_prev=ca.DM(self.u_prev),
                u_prev_step=(U[:, t - 1] if t > 0 else None),
                y_t=y_t,
                y_ref_t=y_ref_t,
                ref_psi_t=float(ref_psi[t]),
                ref_px_norm_t=float(ref_px_norm[t]),
                ref_py_norm_t=float(ref_py_norm[t]),
                d_pos_ca=self._D_pos_ca,
                d_psi_ca=self._D_psi_ca,
                q=Q,
                r=R,
                q_psi=Q_PSI,
                q_progress=Q_PROGRESS,
                q_pos=Q_pos,
                add_position_term=(t % POSITION_TERM_INTERVAL == 0),
                risk_terms=risk_terms,
            )

        # 可选的时域级代价（例如CVaR尾部风险）
        cost += self.cost_builder.finalize_cost(
            opti=opti,
            horizon=T,
            risk_terms=risk_terms,
            z_terminal=Z[T],
            ref_psi_terminal=ref_psi_terminal,
            ref_px_norm_terminal=ref_px_norm_terminal,
            ref_py_norm_terminal=ref_py_norm_terminal,
            d_pos_ca=self._D_pos_ca,
            d_psi_ca=self._D_psi_ca,
            terminal_heading_weight=Q_TERMINAL_HEADING,
            terminal_pos_weight=Q_TERMINAL_POS,
        )

        # ================================================================
        # 步骤6：添加约束
        # ================================================================
        # 6.1 控制输入约束
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

        # 6.2 障碍物避免约束（带接近过滤和松弛变量）
        # 只对在接近阈值内的障碍物添加约束
        nearby_obstacles = []
        px, py = x_current[0], x_current[1]

        # 遍历所有障碍物
        for obs in obstacles:
            ox, oy, r = obs
            # 计算车辆与障碍物的距离
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            # 只关注在接近阈值内的障碍物
            if dist < OBSTACLE_PROXIMITY:
                nearby_obstacles.append(obs)

        # 检查步骤：每隔4步检查一次障碍物约束
        # 这样可以在安全性和计算复杂度之间取得平衡
        check_steps = list(range(1, T + 1, 4))

        # 如果有附近的障碍物，添加避免约束
        if nearby_obstacles:
            # 创建松弛变量用于软化障碍物约束
            # 松弛变量数量 = 障碍物数量 × 检查步骤数量
            n_slack = len(nearby_obstacles) * len(check_steps)
            slack = opti.variable(n_slack)

            # 松弛变量必须非负
            opti.subject_to(slack >= 0)

            # 在代价函数中添加松弛变量的惩罚
            # 大的惩罚权重会强制满足障碍物约束
            cost += OBSTACLE_SLACK_PENALTY * ca.dot(slack, slack)

            # 遍历每个障碍物
            slack_idx = 0
            for obs in nearby_obstacles:
                ox, oy, r = obs
                # 计算最小安全距离
                # d_min = 障碍物半径 + 车辆半径 + 安全余量
                d_min = r + VEHICLE_RADIUS + D_SAFE

                # 对障碍物位置进行归一化
                # 用于与解码器输出（归一化空间）比较
                ox_norm = (
                    (ox - self.norm_params['px_mean'])
                    / self.norm_params['px_std']
                )
                oy_norm = (
                    (oy - self.norm_params['py_mean'])
                    / self.norm_params['py_std']
                )

                # 遍历每个检查步骤
                for tc in check_steps:
                    # 使用固定选择器提取位置（线性）
                    pos = ca.mtimes(self._D_pos_ca, Z[tc])
                    dx = pos[0] - ox_norm
                    dy = pos[1] - oy_norm

                    # 通过反归一化计算物理距离的平方
                    dist_phys_sq = (
                        (dx * self.norm_params['px_std'])**2
                        + (dy * self.norm_params['py_std'])**2
                    )

                    # 软化约束: dist^2 >= d_min^2 - slack
                    # 松弛变量允许轻微违反约束，但会受到惩罚
                    opti.subject_to(
                        dist_phys_sq + slack[slack_idx] >= d_min**2
                    )
                    slack_idx += 1

        # ================================================================
        # 步骤7：设置优化目标
        # ================================================================
        # 最小化总代价
        opti.minimize(cost)

        # ================================================================
        # 步骤8：配置IPOPT求解器
        # ================================================================
        # IPOPT配置选项
        opts = {
            # 最大迭代次数
            'ipopt.max_iter': IPOPT_MAX_ITER,
            # 打印级别（0=静默，5=详细）
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            # 不打印求解时间
            'print_time': False,
            # 启用热启动
            'ipopt.warm_start_init_point': 'yes',
            # 收敛容差
            'ipopt.tol': 1e-4,
            # 可接受的收敛容差（宽松）
            'ipopt.acceptable_tol': 1e-3,
            # 可接受的迭代次数
            'ipopt.acceptable_iter': 5,
        }

        # 设置求解器为IPOPT（Interior Point Optimizer）
        # IPOPT是求解大规模非线性优化问题的开源求解器
        opti.solver('ipopt', opts)

        # ================================================================
        # 步骤9：热启动（可选）
        # ================================================================
        # 如果有上一次的解，用作初始猜测
        # 热启动可以加速求解过程
        if self._warm_start is not None:
            try:
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                # 如果设置失败，忽略并继续
                pass

        # ================================================================
        # 步骤10：求解优化问题
        # ================================================================
        try:
            # 调用IPOPT求解
            sol = opti.solve()

            # 提取第一步的最优控制输入
            # MPC只应用第一步的控制，然后在下一步重新优化
            u_opt = np.array(sol.value(U[:, 0])).flatten()

            # 求解状态
            status = "optimal"

            # 保存当前解用于下一次热启动
            self._warm_start = {'U': sol.value(U)}

        except Exception as e:
            # 如果求解失败，尝试提取次优解
            try:
                # 从debug对象中提取解
                u_opt = np.array(opti.debug.value(U[:, 0])).flatten()

                # 将控制输入裁剪到合法范围内
                u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)  # 加速度
                u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)  # 转向角

                # 标记为次优解
                status = "suboptimal"

            except Exception:
                # 如果无法提取次优解，使用上一次的控制输入
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
