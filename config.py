"""
K-DRMPC论文实现的中心配置文件。
所有论文中的超参数都在此定义。
"""

import os  # 导入操作系统接口模块，用于路径操作
import numpy as np  # 导入NumPy库，用于数值计算和数组操作

# ============================================================
# 项目路径配置
# ============================================================
# K-DRMPC目录（当前文件所在目录）
EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 独立项目，K-DRMPC即为根目录
PROJECT_ROOT = EXPERIMENT_ROOT

# 输出目录路径
OUTPUT_DIR = os.path.join(EXPERIMENT_ROOT, "_output")  # 总输出目录
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")  # 模型保存目录
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")  # 仿真结果保存目录
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")  # 图表保存目录
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")  # 表格保存目录

# 训练数据文件路径（项目内部）
DATA_NPZ_PATH = os.path.join(MODEL_DIR, "training_data.npz")
NORM_JSON_PATH = os.path.join(MODEL_DIR, "norm_params.json")

# 结果导出模式：后续默认采用 simulation + animate，暂不导出静态图
EXPORT_STATIC_FIGURES = False
EXPORT_ANIMATION = True
ANIMATION_FPS = 12

# ============================================================
# 状态和控制维度
# ============================================================
N_X = 5       # 状态维度：[px, py, psi, v, omega] = [位置x, 位置y, 航向角, 速度, 角速度]
N_U = 2       # 控制输入维度：[a, delta] = [加速度, 转向角]
N_Z = 32      # Koopman隐空间维度（提升后的线性系统状态维度）
N_W = 5       # 干扰维度（与状态维度相同）

# 统一状态索引（全仓库使用，避免硬编码）
IDX_PX = 0
IDX_PY = 1
IDX_PSI = 2
IDX_V = 3
IDX_OMEGA = 4

# ============================================================
# Deep Koopman网络架构（论文第3.3节）
# ============================================================
ENCODER_LAYERS = [N_X, 64, 128, 64, N_Z]    # 编码器层结构：5 -> 64 -> 128 -> 64 -> 32
DECODER_LAYERS = [N_Z, 64, 32, N_X]          # 解码器层结构：32 -> 64 -> 32 -> 5
ACTIVATION = "relu"  # 激活函数：ReLU

# ============================================================
# 训练超参数（论文第3.4节）
# ============================================================
BATCH_SIZE = 256  # 批次大小：每次训练使用的样本数
EPOCHS = 500  # 训练轮数：完整遍历数据集的次数
LEARNING_RATE = 1e-3  # 初始学习率：0.001
LR_PATIENCE = 20  # 学习率调度器耐心值：验证损失20轮不改善则降低学习率
LR_FACTOR = 0.5  # 学习率衰减因子：每次降低为原来的50%
EARLY_STOP_PATIENCE = 50  # 早停耐心值：验证损失50轮不改善则停止训练
VAL_SPLIT = 0.1  # 验证集比例：10%的数据用作验证集

# 损失函数权重（论文第3.3.5节）
LAMBDA_RECON = 1.0  # 重构损失权重：保证编解码器能准确重构原始状态
LAMBDA_LINEAR = 1.0  # 线性演化损失权重：保证Koopman空间中的线性演化准确性
LAMBDA_PRED = 2.0  # 多步预测损失权重：提高长期预测能力
LAMBDA_PHYSICS = 1.0  # 物理一致性损失权重：强制A矩阵学习正确的位置更新
LAMBDA_REG_HIGH_DIM = 2.0  # 高维正则化权重：惩罚A矩阵物理行的高维列，防止高维特征劫持v/psi预测
LAMBDA_SPECTRAL = 1.5          # A矩阵谱半径惩罚权重（强制spectral_radius < 1.0）

# 归一化参数（从训练数据计算得到，用于物理一致性损失反归一化）
# 这些值来自 norm_params.json，硬编码以避免训练时重复读取文件
PX_STD = 481.95896444159337   # px的标准差
PY_STD = 273.6795009788262    # py的标准差

# 训练损失的多步预测时域
K_PRED = 25  # 前向预测步数：计算100步的预测误差

# D矩阵的岭回归正则化参数
GAMMA_RIDGE = 1e-4  # 正则化强度：防止过拟合，提高数值稳定性

# 数据子采样（原始Ts=0.01s -> 论文dt=0.1s）
ORIGINAL_TS = 0.01  # 原始数据采样时间间隔：0.01秒
SUBSAMPLE_RATE = 10  # 子采样率：每10个样本取1个，得到有效采样时间0.1秒

# ============================================================
# 车辆参数（论文第2节，表5）
# ============================================================
DT = 0.1                   # 采样时间 [秒]
CONTROL_UPDATE_INTERVAL = 1  # 控制更新间隔：每N个仿真步求解一次MPC，1表示每步求解
L_WHEELBASE = 2.6          # 车辆轴距 [米]
V_MIN = 0.0                # 论文Table 5原始值：最小速度 [米/秒]
V_MAX = 40.0               # 最大速度 [米/秒]
A_MIN = -5.0               # 论文Table 5原始值：最大减速度 [米/秒^2]
A_MAX = 3.0                # 论文Table 5原始值：最大加速度 [米/秒^2]
DELTA_MAX = np.pi / 4      # 论文Table 5原始值：最大转向角 = 45° [弧度]
DELTA_RATE_MAX = 0.5       # 论文Table 5原始值：最大转向速率 [弧度/秒]
D_SAFE = 1.5               # 安全裕度 [米]（更早开始避障，转向更平缓）
TRACK_HALF_WIDTH = 12.0    # 赛道半宽 W/2 [米]（与 PLUS 版本保持一致）
TRACK_BOUNDARY_SLACK_PENALTY = 500.0   # 赛道边界松弛惩罚（与障碍物约束同级，防止避障时冲出赛道）

# 横向加速度限制（用于速度曲线规划）
A_LAT_MAX = 4.0            # 最大横向加速度 [米/秒^2]（限制过弯速度）
REF_SPEED_SCALE = 0.25     # 直道目标~10m/s，降低过弯压力，减少避障时的横向失控
REF_ACCEL_MAX = 0.4        # 参考速度曲线的最大加速度 [米/秒^2]
REF_DECEL_MAX = 1.2        # 参考速度曲线的最大减速度 [米/秒^2]

# ============================================================
# MPC参数（论文第6节，表5）
# ============================================================
T_HORIZON = 40             # MPC预测时域（40步 = 4秒，归一化bug已修复，可安全使用长时域）

# 代价函数权重（论文第6.1.2节）
Q_WEIGHTS = np.diag([6.0, 1.0])     # Q矩阵：v跟踪权重6.0，omega跟踪权重降到1.0（避免psi/omega矛盾时过度跟踪omega）
R_WEIGHTS = np.diag([0.5, 1.5])     # R矩阵：a变化惩罚降到0.5（允许更灵活的油门调整），delta保持1.5

# 轨迹跟踪调优参数（供K-MPC/K-DRMPC统一读取）
Q_PSI_TRACK = 15.0                  # 航向误差权重（提高，增强航向跟踪能力，防止航向偏离导致lat发散）
Q_PROGRESS_TRACK = 0.15             # 前向进度权重（增大，鼓励沿赛道前进）
Q_POS_TRACK = 5000.0               # 位置误差权重（补偿归一化空间缩放：px_std≈80，需Q*err_norm²有效）
POSITION_TERM_INTERVAL = 1          # 位置项加入频率（1=每步）
R_ABS_A = 0.5                       # 降低加速惩罚，鼓励MPC积极提速跟踪参考速度
R_ABS_DELTA = 2.0                   # 绝对转角惩罚
Q_TERMINAL_HEADING = 15.0           # 终端航向误差权重（降低）
Q_TERMINAL_POS = 30.0               # 终端位置误差权重

# ============================================================
# 分布鲁棒参数（论文第6.1.3节）
# ============================================================
ENABLE_DISTURBANCE = False    # 全局噪声开关：False 表示关闭在线扰动和经验扰动样本
N_DISTURBANCE_SAMPLES = 100    # 历史干扰样本数量：用于构建Wasserstein模糊集
THETA_WASSERSTEIN = 0.25       # Wasserstein球半径：控制模糊集大小（不确定性程度）
NOMINAL_SIGMA = 0.5            # 训练数据的 nominal 干扰标准差（单点真理源）
                               # 论文 Setting 缩放基准：
                               #   Setting A (Nominal,   x1.0) = 0.5
                               #   Setting B (Moderate,  x1.5) = 0.75
                               #   Setting C (Large,     x2.0) = 1.0
                               # 训练和仿真默认均从此值派生，修改后需重训 Koopman 模型
EPSILON_CVAR = 0.05             # CVaR风险水平：0.05表示考虑最坏的5%情况

# ============================================================
# 敏感性分析范围（论文第6.7节）
# ============================================================
THETA_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]  # Wasserstein半径的测试值列表
EPSILON_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30]  # CVaR风险水平的测试值列表
SIGMA_VALUES = [0.01, 0.05, 0.10, 0.15]  # 噪声标准差的测试值列表

# ============================================================
# 仿真参数
# ============================================================
MAX_SIM_STEPS = 3000       # 每圈最大仿真步数（3000步 × 0.1秒 = 300秒）
IPOPT_MAX_ITER = 500       # IPOPT求解器最大迭代次数
IPOPT_TOL = 1e-6           # IPOPT求解器收敛容差
IPOPT_PRINT_LEVEL = 0      # IPOPT输出级别：0表示抑制输出（不打印详细信息）

# ============================================================
# 障碍物配置
# ============================================================
ENABLE_OBSTACLES = False  # 全局障碍物开关：False 表示仿真与绘图都不使用任何障碍物
OBSTACLE_RADIUS = 3.0      # 默认障碍物半径 [米]
OBSTACLE_LAYOUT_MODE = "edge"  # 障碍物布局模式：edge=贴边，center=更靠中线
OBSTACLE_EDGE_MARGIN = 0.2  # edge模式下障碍物与路边保留的小间隙 [米]
VEHICLE_RADIUS = 2.0       # 车辆近似足迹半径 [米]（用于碰撞检测）

# 预定义的障碍物位置列表（赛道上的具体坐标，避开起点附近）
# SprintOvalTrack: 椭圆形赛道，起点约 (72.3, -99.1)，总长 798m
OBSTACLE_POSITIONS = [
    (100.0, 50.0),    # 赛道右侧上方（进度 ~25%）
    (-50.0, 120.0),    # 赛道上侧（进度 ~50%）
    (-120.0, -50.0),   # 赛道左侧下方（进度 ~75%）
]

# ============================================================
# 绘图配置
# ============================================================
PLOT_TRACK_HALF_WIDTH = 12.0  # 轨迹边界可视化半宽 [米]（与约束半宽一致；仅用于画图，不进入约束）
FIGURE_DPI = 300  # 图像分辨率：300 DPI（高质量打印）
FIGURE_FORMAT = "pdf"  # 图像格式：PDF（矢量图，适合论文）
# 方法颜色映射（用于绘图区分不同方法）
METHOD_COLORS = {
    "LMPC": "#1f77b4",    # 蓝色：线性MPC
    "NMPC": "#ff7f0e",    # 橙色：非线性MPC
    "K-MPC": "#2ca02c",   # 绿色：Koopman MPC
    "K-DRMPC": "#d62728", # 红色：Koopman分布鲁棒MPC（本文方法）
}
# 方法标签映射（用于图例显示）
METHOD_LABELS = {
    "LMPC": "LMPC",                    # 线性MPC
    "NMPC": "NMPC",                    # 非线性MPC
    "K-MPC": "K-MPC",                  # Koopman MPC
    "K-DRMPC": "K-DRMPC (Ours)",      # Koopman分布鲁棒MPC（本文方法）
}
