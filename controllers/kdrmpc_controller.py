"""
基于Koopman的分布鲁棒MPC (K-DRMPC)控制器。
这是论文的主要贡献（第5节）。

在K-MPC基础上扩展：
- 使用Wasserstein模糊集建模干扰不确定性
- 基于CVaR的安全约束（公式4.10-4.11）
- 通过拉格朗日对偶实现可处理的凸重构

核心创新：
1. 分布鲁棒性：不假设干扰的具体分布，而是考虑一组可能的分布
2. Wasserstein模糊集：以经验分布为中心的概率分布集合
3. CVaR约束：条件风险价值，保证在最坏情况下的安全性
4. 凸重构：将无穷维优化问题转化为有限维凸优化问题
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和多维数组操作
import gc              # 垃圾回收，防止 CasADi 长时间运行内存累积导致 segfault
import casadi as ca  # 导入CasADi库，用于符号计算和非线性优化（IPOPT求解器）
import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径
import time  # 导入时间模块，用于计算求解耗时

# 将父目录添加到系统路径，确保可以导入同级别的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入MPC和分布鲁棒优化相关参数
from config import (
    N_X,  # 物理状态维度，默认5 [px, py, psi, v, omega]
    N_U,  # 控制输入维度，默认2 [加速度a, 转向角delta]
    N_Z,  # Koopman空间维度，默认8
    N_W,  # 干扰维度，用于采样历史扰动
    T_HORIZON,  # 预测时域长度，默认20步
    DT,  # 时间步长，默认0.1秒
    L_WHEELBASE,  # 车辆轴距 [米]，用于物理omega计算
    Q_WEIGHTS,  # 状态跟踪权重向量
    R_WEIGHTS,  # 控制输入权重向量
    V_MIN,  # 最小速度 [米/秒]
    V_MAX,  # 最大速度 [米/秒]
    A_MIN,  # 最小加速度 [米/秒^2]
    A_MAX,  # 最大加速度 [米/秒^2]
    DELTA_MAX,  # 最大转向角 [弧度]
    DELTA_RATE_MAX,  # 最大转向角速率 [弧度/秒]
    D_SAFE,  # 安全距离余量 [米]
    TRACK_HALF_WIDTH,
    TRACK_BOUNDARY_SLACK_PENALTY,
    VEHICLE_RADIUS,  # 车辆半径 [米]
    N_DISTURBANCE_SAMPLES,  # 干扰样本数量，默认100
    THETA_WASSERSTEIN,  # Wasserstein球半径，默认0.1
    EPSILON_CVAR,  # CVaR风险水平，默认0.05（95%置信度）
    IPOPT_MAX_ITER,  # IPOPT求解器最大迭代次数
    IPOPT_PRINT_LEVEL,  # IPOPT求解器打印级别
    Q_PSI_TRACK,
    Q_PROGRESS_TRACK,
    Q_POS_TRACK,
    POSITION_TERM_INTERVAL,
    R_ABS_A,
    R_ABS_DELTA,
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
from controllers.tracking_costs import (
    MinSpeedRule,
    resolve_tracking_cost_builder,
)
from model.projection import get_fixed_selector_matrices

# 障碍物接近阈值：只有车辆在这个距离内才添加障碍物约束
OBSTACLE_PROXIMITY = 50.0  # 单位：米（SerpentineTrack 紧凑，50m 足够覆盖附近障碍）
OBSTACLE_STRATEGY_CHOICES = ("robust", "non-robust")
DEFAULT_OBSTACLE_STRATEGY = "robust"

# 优化中使用的最大干扰样本数（为了保证可处理性）
# 如果样本太多，会使优化问题过于复杂，因此进行子采样
MAX_OPT_SAMPLES = 10

# 障碍物松弛变量的惩罚权重
# 较大的值会强制满足障碍物约束，但可能导致问题不可解
OBSTACLE_SLACK_PENALTY = 1000.0

# 软最小速度约束参数：避免优化落入"停车解"
MIN_SPEED_FLOOR = V_MIN         # 与 config 保持一致，避免硬编码
MIN_SPEED_REF_RATIO = 0.25      # 参考速度下限比例
MIN_SPEED_SLACK_PENALTY = 1200.0  # 速度下限松弛惩罚（软调优）
# 航向角跟踪权重：增强“沿参考方向前进”的方向感
Q_PSI = Q_PSI_TRACK
# 前向进度权重：显式鼓励沿参考切向前进
Q_PROGRESS = Q_PROGRESS_TRACK
# 绝对控制量惩罚（软约束）：抑制长期满刹和满舵
R_ABS_WEIGHTS = np.diag([R_ABS_A, R_ABS_DELTA])


class KDRMPCController:
    """
    带有CVaR约束的分布鲁棒Koopman MPC控制器。

    核心特性：
    1. 线性Koopman动力学: z_{t+1} = A*z + B*u
       - A: 状态转移矩阵（学习得到）
       - B: 控制矩阵（学习得到）

    2. 以经验干扰分布为中心的Wasserstein模糊集
       - 模糊集: P ∈ {Q : W(P, P_N) <= theta}
       - W: Wasserstein距离
       - P_N: 经验分布（从N个样本构建）
       - theta: 模糊集半径（控制鲁棒性程度）

    3. CVaR安全约束重构（定理，第5.5节）
       - CVaR_epsilon[l(z, w)] <= 0
       - 将概率约束转化为确定性约束
       - 引入辅助变量Lambda和S

    优化问题形式（论文公式20）：
        min_u Σ_t ||D*z_t - y_ref_t||_Q^2 + ||u_t||_R^2
        s.t.  z_{t+1} = A*z_t + B*u_t  (标称动力学)
              u_min <= u_t <= u_max
              CVaR约束: lambda*theta + (1/(epsilon*N))*Σs_i <= slack
                        s_i >= l_nom + lambda*||w_i||
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
                 disturbance_samples=None, theta=THETA_WASSERSTEIN,
                 epsilon=EPSILON_CVAR, cost_builder=None,
                 cost_profile="default", obstacle_strategy=DEFAULT_OBSTACLE_STRATEGY):
        """
        初始化K-DRMPC控制器。

        参数:
            koopman_model: 训练好的DeepKoopmanPaper模型
                          包含编码器、解码器和Koopman矩阵A, B
            D_matrix: numpy数组，形状为(2, n_z)
                     投影矩阵，用于从Koopman状态提取[v, omega]
            norm_params: 字典，包含归一化参数
                        {'px_mean', 'px_std', 'py_mean', 'py_std'}
            disturbance_samples: numpy数组，形状为(N, 5)
                                用于构建模糊集的干扰样本
                                如果为None，则使用随机生成的样本
            theta: 浮点数，Wasserstein球半径
                  默认使用config.py中的THETA_WASSERSTEIN（0.25）
                  theta越大，考虑的分布范围越广，控制器越保守
            epsilon: 浮点数，CVaR风险水平
                    默认使用config.py中的EPSILON_CVAR（0.05）
                    epsilon=0.05表示95%置信度下的风险约束
        """
        # 控制器名称，用于日志和结果标识
        self.name = "K-DRMPC"

        # 初始化上一次控制输入为零向量
        self.u_prev = np.zeros(N_U)

        # 热启动信息，用于加速下一次求解
        self._warm_start = None

        # ================================================================
        # Koopman模型组件
        # ================================================================
        # 获取Koopman线性动力学矩阵
        # A: 状态转移矩阵 (n_z x n_z)
        # B: 控制矩阵 (n_z x n_u)
        # C: 干扰矩阵 (n_z x n_w)
        self.A, self.B, self.C = koopman_model.get_matrices()
        # 对A矩阵进行特征值裁剪，防止高维Koopman特征在预测时域内指数发散
        self.A = self._clip_matrix_eigenvalues(self.A, radius=0.9)

        # 论文固定线性选择器（控制主路径使用）
        self.D_pos, self.E, self.F, self.D_vomega = get_fixed_selector_matrices(
            self.A.shape[0]
        )
        # 保留外部传入D仅作诊断
        self.D_diag = D_matrix

        # 保存归一化参数，用于状态编码和解码
        self.norm_params = norm_params
        # 缓存px/py的std，用于混合模型位置更新时的归一化补偿
        self._px_std = float(norm_params['px_std'])
        self._py_std = float(norm_params['py_std'])

        # ================================================================
        # 分布鲁棒优化参数
        # ================================================================
        # Wasserstein球半径
        self.theta = theta

        # CVaR风险水平
        self.epsilon = epsilon

        if obstacle_strategy not in OBSTACLE_STRATEGY_CHOICES:
            raise ValueError(
                f"obstacle_strategy 必须是 {OBSTACLE_STRATEGY_CHOICES} 之一，"
                f"收到: {obstacle_strategy}"
            )
        self.obstacle_strategy = obstacle_strategy

        # ================================================================
        # 干扰样本处理（子采样以保证可处理性）
        # ================================================================
        if disturbance_samples is not None:
            # 如果样本数量超过最大值，进行随机子采样
            if len(disturbance_samples) > MAX_OPT_SAMPLES:
                # 使用固定种子保证可复现性
                rng = np.random.RandomState(42)
                # 随机选择MAX_OPT_SAMPLES个样本（不重复）
                idx = rng.choice(
                    len(disturbance_samples),
                    MAX_OPT_SAMPLES,
                    replace=False
                )
                self.w_samples = disturbance_samples[idx]
            else:
                # 样本数量合适，直接使用
                self.w_samples = disturbance_samples
        else:
            # 如果没有提供样本，生成随机样本作为默认值
            # 使用零均值、标准差0.05的正态分布
            self.w_samples = np.random.randn(MAX_OPT_SAMPLES, N_W) * 0.05

        # 保存实际使用的样本数量
        self.N_samples = len(self.w_samples)

        # ================================================================
        # 预计算CVaR约束所需的范数（提高优化效率）
        # ================================================================
        # 预计算||w_i||：干扰本身的范数
        # 这用于Wasserstein距离的计算
        self.w_norms = np.array([
            np.linalg.norm(self.w_samples[i])
            for i in range(self.N_samples)
        ])

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

        # ================================================================
        # 构建CasADi函数（编解码器）
        # ================================================================
        # 获取神经网络权重
        weights = koopman_model.get_network_weights()

        # 将PyTorch编码器转换为CasADi函数
        self._ca_encode = pytorch_to_casadi_encoder(weights)

        # 将PyTorch解码器转换为CasADi函数
        # 用于障碍物避免和位置跟踪
        self._ca_decode = pytorch_to_casadi_decoder(weights)

    def _encode_state(self, x_physical):
        """将物理状态编码为Koopman潜在状态（带归一化）。"""
        # 复制物理状态，避免修改原始数据
        x_norm = x_physical.copy()

        # 对px（索引0）进行归一化
        x_norm[0] = (
            x_norm[0] - self.norm_params['px_mean']
        ) / self.norm_params['px_std']

        # 对py（索引1）进行归一化
        x_norm[1] = (
            x_norm[1] - self.norm_params['py_mean']
        ) / self.norm_params['py_std']

        # 将NumPy数组转换为CasADi列向量
        x_ca = ca.DM(x_norm.reshape(-1, 1))

        # 使用CasADi编码器计算潜在状态
        z_ca = self._ca_encode(x_ca)

        # === 关键修复：_ca_encode 直接返回 MLP 输出，没有覆盖前 N_X 维为物理状态。
        # 但 PyTorch 的 model.encode 中 z[:, :n_x] = x_norm，保证前 5 维是归一化后的物理状态。
        # 如果不覆盖，MPC 中 Z[0] 的前 5 维是 MLP 输出（可能高达 351），导致位置/v/omega 完全失真。
        z_ca[:N_X] = x_ca

        # 转换回NumPy数组并展平
        return np.array(z_ca).flatten()

    def solve(self, x_current, ref_trajectory, obstacles, u_prev=None):
        """
        求解K-DRMPC优化问题（论文公式20）。

        优化包含：
        1. 通过D投影在潜在空间中的跟踪代价
        2. 控制平滑性代价
        3. 输入/状态约束
        4. 分布鲁棒的CVaR障碍物避免约束

        参数:
            x_current: numpy数组，形状为(5,)
                      当前物理状态 [px, py, psi, v, omega]
            ref_trajectory: numpy数组，形状为(T, 5)
                          参考轨迹（物理坐标）
            obstacles: 列表，每个元素为(ox, oy, radius)
                      障碍物位置和半径
            u_prev: numpy数组，形状为(2,)
                   上一次的控制输入 [a, delta]

        返回:
            u_opt: numpy数组，形状为(2,)
                  最优控制输入 [加速度a, 转向角delta]
            solve_info: 字典，包含求解信息和DR参数
        """
        if u_prev is not None:
            self.u_prev = u_prev

        # 获取预测时域长度（默认20步）
        T = T_HORIZON

        # 记录求解开始时间
        t_start = time.time()

        # 步骤1：编码当前状态到Koopman空间
        z0 = self._encode_state(x_current)

        # 步骤2：构建参考轨迹（在[v, omega]空间）
        y_ref = np.zeros((T, 2))
        for t in range(T):
            ref_t = ref_trajectory[min(t, len(ref_trajectory) - 1)]
            y_ref[t] = [ref_t[IDX_V], ref_t[IDX_OMEGA]]  # [v, omega]

        # 预计算参考航向角（用于代价函数中的航向跟踪和终端约束）
        ref_psi = np.array([
            ref_trajectory[min(t, len(ref_trajectory)-1), IDX_PSI]
            for t in range(T)
        ])

        # 步骤3：过滤附近的障碍物
        px, py = x_current[0], x_current[1]
        nearby_obstacles = []
        for obs in obstacles:
            ox, oy, r = obs
            # 计算车辆与障碍物的距离
            dist = np.sqrt((px - ox)**2 + (py - oy)**2)
            # 只关注在接近阈值内的障碍物（200米）
            if dist < OBSTACLE_PROXIMITY:
                nearby_obstacles.append(obs)

        # 障碍物数量和检查步骤
        n_obs = len(nearby_obstacles)
        # 鲁棒CVaR约束计算成本更高，使用更稀疏的检查点；非鲁棒可更密集
        if self.obstacle_strategy == "robust":
            check_steps = list(range(4, T + 1, 12))
        else:
            check_steps = list(range(1, T + 1, 6))
        n_check = len(check_steps)

        opti = ca.Opti()

        # Decision variables: controls
        U = opti.variable(N_U, T)

        # Propagate latent dynamics (nominal, no disturbance in prediction)
        Z = [ca.DM(z0)]
        n_z = self.A.shape[0]
        for t in range(T):
            z_next = ca.mtimes(self._A_ca, Z[-1]) + \
                     ca.mtimes(self._B_ca, U[:, t])

            # === 混合模型修正：用物理公式覆盖前5维，Koopman只提供高维特征 ===
            # Koopman A矩阵无法可靠预测omega（纯Koopman时omega从-0.036变+0.015），
            # 必须用物理公式确保转向方向正确。
            v_phys = Z[-1][3] + DT * U[0, t]
            omega_phys = v_phys * ca.tan(U[1, t]) / L_WHEELBASE
            psi_next = Z[-1][2] + DT * omega_phys
            # 注意：Z[0], Z[1] 在归一化空间中，位置增量需除以对应std
            px_next = Z[-1][0] + DT * v_phys * ca.cos(psi_next) / self._px_std
            py_next = Z[-1][1] + DT * v_phys * ca.sin(psi_next) / self._py_std

            # 覆盖前5维(px,py,psi,v,omega)，保留Koopman预测的高维特征z[5:]
            z_fixed = ca.vertcat(px_next, py_next, psi_next, v_phys, omega_phys,
                                z_next[5:n_z])
            Z.append(z_fixed)

        # === Tracking Cost ===
        cost = 0
        Q = ca.DM(Q_WEIGHTS)
        R = ca.DM(R_WEIGHTS)
        R_abs = ca.DM(R_ABS_WEIGHTS)

        # 速度下限软约束松弛变量（每个预测步一个）
        v_slack = opti.variable(T)
        opti.subject_to(v_slack >= 0)
        cost += MIN_SPEED_SLACK_PENALTY * ca.dot(v_slack, v_slack)

        # 赛道边界软约束松弛变量（论文 |d(s)|<=W/2；加入松弛避免不可行）
        track_slack = opti.variable(T + 1)
        opti.subject_to(track_slack >= 0)
        cost += TRACK_BOUNDARY_SLACK_PENALTY * ca.dot(track_slack, track_slack)

        # Position tracking weight (decoded position vs reference)
        Q_pos = Q_POS_TRACK

        # Normalize reference positions for decoder comparison
        ref_px_norm = np.array([(ref_trajectory[min(t, len(ref_trajectory)-1), 0]
                                 - self.norm_params['px_mean']) / self.norm_params['px_std']
                                for t in range(T)])
        ref_py_norm = np.array([(ref_trajectory[min(t, len(ref_trajectory)-1), 1]
                                 - self.norm_params['py_mean']) / self.norm_params['py_std']
                                for t in range(T)])
        # 终端约束也需要参考航向
        ref_psi_terminal = float(ref_trajectory[min(T - 1, len(ref_trajectory) - 1), IDX_PSI])
        ref_px_norm_terminal = float(ref_px_norm[T - 1])
        ref_py_norm_terminal = float(ref_py_norm[T - 1])

        px_std = float(self.norm_params['px_std'])
        py_std = float(self.norm_params['py_std'])

        min_speed_rule = MinSpeedRule(
            floor_abs=MIN_SPEED_FLOOR,
            floor_ratio=MIN_SPEED_REF_RATIO,
        )

        risk_terms = []
        diag_terms = {}

        for t in range(T):
            y_t = ca.vertcat(
                ca.mtimes(self._E_ca, Z[t]),
                ca.mtimes(self._F_ca, Z[t])
            )
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
                add_abs_u_term=True,
                r_abs=R_abs,
                min_speed_rule=min_speed_rule,
                v_slack_t=v_slack[t],
                risk_terms=risk_terms,
            )
            self.cost_builder.collect_stage_diagnostics(
                diag_terms,
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
                add_abs_u_term=True,
                r_abs=R_abs,
                min_speed_rule=min_speed_rule,
                v_slack_t=v_slack[t],
                risk_terms=risk_terms,
            )

        # 可选的时域级代价（例如CVaR尾部风险）
        cost += self.cost_builder.finalize_cost(
            opti=opti,
            horizon=T,
            risk_terms=risk_terms,
            diag_terms=diag_terms,
            z_terminal=Z[T],
            ref_psi_terminal=ref_psi_terminal,
            ref_px_norm_terminal=ref_px_norm_terminal,
            ref_py_norm_terminal=ref_py_norm_terminal,
            d_pos_ca=self._D_pos_ca,
            d_psi_ca=self._D_psi_ca,
            terminal_heading_weight=Q_TERMINAL_HEADING,
            terminal_pos_weight=Q_TERMINAL_POS,
        )

        # === Input Constraints ===
        for t in range(T):
            opti.subject_to(opti.bounded(A_MIN, U[0, t], A_MAX))
            opti.subject_to(opti.bounded(-DELTA_MAX, U[1, t], DELTA_MAX))
            if t == 0:
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - self.u_prev[1],
                    DELTA_RATE_MAX * DT))
            else:
                opti.subject_to(opti.bounded(
                    -DELTA_RATE_MAX * DT,
                    U[1, t] - U[1, t - 1],
                    DELTA_RATE_MAX * DT))

        # === Track Boundary Constraints (paper Eq. |d(s)| <= W/2) ===
        for t in range(T + 1):
            ref_idx = min(t, T - 1)
            psi_ref = ref_psi[ref_idx] if t < T else ref_psi_terminal

            # 线性选择器提取位置（归一化坐标），再换算为米制偏差
            pos_t = ca.mtimes(self._D_pos_ca, Z[t])
            dx_m = (pos_t[0] - float(ref_px_norm[ref_idx])) * px_std
            dy_m = (pos_t[1] - float(ref_py_norm[ref_idx])) * py_std

            # 论文中的 signed lateral deviation d(s)
            d_lat = -ca.sin(psi_ref) * dx_m + ca.cos(psi_ref) * dy_m

            # 软化的双边约束：|d_lat| <= TRACK_HALF_WIDTH + slack
            opti.subject_to(d_lat <= TRACK_HALF_WIDTH + track_slack[t])
            opti.subject_to(-d_lat <= TRACK_HALF_WIDTH + track_slack[t])

        obs_slack = None
        if n_obs > 0 and n_check > 0:
            n_obs_slack = n_obs * n_check
            obs_slack = opti.variable(n_obs_slack)
            opti.subject_to(obs_slack >= 0)
            cost += OBSTACLE_SLACK_PENALTY * ca.dot(obs_slack, obs_slack)

            if self.obstacle_strategy == "robust":
                # === Distributionally Robust CVaR Constraints (Eq. 4.10-4.11) ===
                Lambda = {}
                S_vars = {}
                for j in range(n_obs):
                    for tc in check_steps:
                        key = (j, tc)
                        Lambda[key] = opti.variable()
                        opti.subject_to(Lambda[key] >= 0)
                        S_vars[key] = opti.variable(self.N_samples)
                        opti.subject_to(S_vars[key] >= 0)

                obs_slack_idx = 0
                for j, obs in enumerate(nearby_obstacles):
                    ox, oy, r = obs
                    d_min = r + VEHICLE_RADIUS + D_SAFE

                    ox_norm = (ox - self.norm_params['px_mean']) / self.norm_params['px_std']
                    oy_norm = (oy - self.norm_params['py_mean']) / self.norm_params['py_std']

                    for tc in check_steps:
                        key = (j, tc)
                        lam = Lambda[key]
                        s = S_vars[key]

                        pos = ca.mtimes(self._D_pos_ca, Z[tc])
                        px_pred = pos[0]
                        py_pred = pos[1]

                        dx_phys = (px_pred - ox_norm) * self.norm_params['px_std']
                        dy_phys = (py_pred - oy_norm) * self.norm_params['py_std']
                        dist_physical = ca.sqrt(dx_phys**2 + dy_phys**2 + 1e-6)
                        l_nom = d_min - dist_physical

                        for i in range(self.N_samples):
                            w_norm_i = float(self.w_norms[i])
                            opti.subject_to(s[i] >= l_nom + lam * w_norm_i)

                        sum_s = ca.sum1(s)
                        opti.subject_to(
                            lam * self.theta +
                            (1.0 / (self.epsilon * self.N_samples)) * sum_s
                            <= obs_slack[obs_slack_idx]
                        )
                        obs_slack_idx += 1
            else:
                # === Non-robust deterministic obstacle constraints ===
                obs_slack_idx = 0
                for obs in nearby_obstacles:
                    ox, oy, r = obs
                    d_min = r + VEHICLE_RADIUS + D_SAFE

                    ox_norm = (ox - self.norm_params['px_mean']) / self.norm_params['px_std']
                    oy_norm = (oy - self.norm_params['py_mean']) / self.norm_params['py_std']

                    for tc in check_steps:
                        pos = ca.mtimes(self._D_pos_ca, Z[tc])
                        dx_phys = (pos[0] - ox_norm) * self.norm_params['px_std']
                        dy_phys = (pos[1] - oy_norm) * self.norm_params['py_std']
                        dist_phys_sq = dx_phys**2 + dy_phys**2

                        opti.subject_to(
                            dist_phys_sq + obs_slack[obs_slack_idx] >= d_min**2
                        )
                        obs_slack_idx += 1

        opti.minimize(cost)

        # === Solver ===
        opts = {
            'ipopt.max_iter': IPOPT_MAX_ITER,
            'ipopt.print_level': IPOPT_PRINT_LEVEL,
            'print_time': False,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 5,
        }
        opti.solver('ipopt', opts)

        # Warm start
        if self._warm_start is not None:
            try:
                opti.set_initial(U, self._warm_start['U'])
            except Exception:
                pass

        # Solve
        try:
            sol = opti.solve()
            u_opt = np.array(sol.value(U[:, 0])).flatten()
            status = "optimal"
            debug_info = self._build_debug_info(
                sol=sol,
                U=U,
                T=T,
                diag_terms=diag_terms,
                v_slack=v_slack,
                obs_slack=obs_slack,
            )

            # Store warm start
            self._warm_start = {'U': sol.value(U)}

        except Exception as e:
            # Try suboptimal solution from debug
            try:
                u_opt = np.array(opti.debug.value(U[:, 0])).flatten()
                u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)
                u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)
                status = "suboptimal"
                debug_info = None
            except Exception:
                u_opt = self.u_prev.copy()
                status = f"failed: {str(e)[:50]}"
                debug_info = None

        solve_time = time.time() - t_start
        self.u_prev = u_opt.copy()

        # 显式释放 CasADi 符号图，防止长时间运行内存积累导致 segfault
        try:
            del opti
        except Exception:
            pass
        # 每 50 步强制 GC
        if not hasattr(self, '_solve_count'):
            self._solve_count = 0
        self._solve_count += 1
        if self._solve_count % 50 == 0:
            gc.collect()

        return u_opt, {
            'solve_time': solve_time,
            'status': status,
            'method': self.name,
            'theta': self.theta,
            'epsilon': self.epsilon,
            'obstacle_strategy': self.obstacle_strategy,
            'debug': debug_info,
        }

    def _build_debug_info(self, sol, U, T, diag_terms, v_slack, obs_slack):
        """Build numeric debug diagnostics from symbolic expressions and solver solution."""
        u_seq = sol.value(U)
        v_slack_vals = sol.value(v_slack) if v_slack is not None else None
        obs_slack_vals = sol.value(obs_slack) if obs_slack is not None else None

        step_terms = {}
        horizon_terms = {}
        for name, expr in diag_terms.items():
            if isinstance(expr, list):
                values = [float(sol.value(item)) for item in expr]
                step_terms[name] = values
            else:
                horizon_terms[name] = float(sol.value(expr))

        tol = 1e-3
        active = []
        if abs(float(u_seq[0, 0]) - A_MIN) < tol:
            active.append("a_min")
        if abs(float(u_seq[0, 0]) - A_MAX) < tol:
            active.append("a_max")
        if abs(float(u_seq[1, 0]) + DELTA_MAX) < tol:
            active.append("delta_min")
        if abs(float(u_seq[1, 0]) - DELTA_MAX) < tol:
            active.append("delta_max")
        delta_rate0 = float(u_seq[1, 0] - self.u_prev[1])
        if abs(delta_rate0 + DELTA_RATE_MAX * DT) < tol:
            active.append("delta_rate_min")
        if abs(delta_rate0 - DELTA_RATE_MAX * DT) < tol:
            active.append("delta_rate_max")
        if v_slack_vals is not None and float(np.max(v_slack_vals)) > 1e-6:
            active.append("speed_floor_slack")
        if obs_slack_vals is not None and float(np.max(obs_slack_vals)) > 1e-6:
            active.append("obs_cvar_slack")

        step0 = {name: values[0] for name, values in step_terms.items() if values}
        summary = {
            'step0': step0,
            'horizon': horizon_terms,
            'active_constraints': active,
            'u0': [float(u_seq[0, 0]), float(u_seq[1, 0])],
            'v_slack_max': float(np.max(v_slack_vals)) if v_slack_vals is not None else 0.0,
            'obs_slack_max': float(np.max(obs_slack_vals)) if obs_slack_vals is not None else 0.0,
        }
        return summary

    def update_disturbance_samples(self, new_samples):
        """Update the disturbance sample set (e.g., from online data)."""
        if len(new_samples) > MAX_OPT_SAMPLES:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(new_samples), MAX_OPT_SAMPLES, replace=False)
            self.w_samples = new_samples[idx]
        else:
            self.w_samples = new_samples
        self.N_samples = len(self.w_samples)
        self.w_norms = np.array([
            np.linalg.norm(self.w_samples[i])
            for i in range(self.N_samples)
        ])

    def reset(self):
        self.u_prev = np.zeros(N_U)
        self._warm_start = None
