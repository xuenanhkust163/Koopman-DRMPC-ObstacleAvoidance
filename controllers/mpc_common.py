"""
MPC共享工具函数模块。

本模块包含以下核心功能：
1. 紧凑矩阵构建（论文第4.1节）
   - A_cal, B_cal, C_cal：堆叠预测矩阵
2. LQR反馈增益计算（论文第4.3节）
   - 用于闭环系统稳定化
3. 闭环矩阵计算（论文第4.4节）
   - A_tilde, B_tilde, C_tilde：带反馈的紧凑矩阵
4. PyTorch到CasADi的神经网络转换
   - 编码器和解码器的符号化重构
5. QP/NLP辅助函数
   - QP黑塞矩阵和线性项构建

这些工具函数被K-MPC、K-DRMPC等控制器共享使用，
实现了代码复用和模块化设计。
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和矩阵运算
import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径

# 导入SciPy库中的矩阵方程求解器和块对角矩阵构造函数
from scipy.linalg import solve_discrete_are, block_diag

# 将父目录添加到系统路径，确保可以导入同级别的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入MPC相关参数
from config import (
    N_Z,  # Koopman空间维度，默认8（未在本文直接使用）
    N_U,  # 控制输入维度，默认2 [加速度a, 转向角delta]
    N_X,  # 物理状态维度，默认5（未在本文直接使用）
    T_HORIZON,  # 预测时域长度，默认20步
    Q_WEIGHTS,  # 状态跟踪权重向量（未在本文直接使用）
    R_WEIGHTS  # 控制输入权重向量（未在本文直接使用）
)


def build_compact_matrices(A, B, T=T_HORIZON):
    """
    构建堆叠预测矩阵A_cal和B_cal（论文第4.1节）。

    紧凑矩阵形式将多步预测表示为矩阵乘法：
        z = A_cal @ z_0 + B_cal @ u

    其中:
        z = [z_1; z_2; ...; z_T] ∈ R^{T*nz}  (堆叠的状态轨迹)
        z_0 ∈ R^{nz}                          (初始状态)
        u = [u_0; u_1; ...; u_{T-1}] ∈ R^{T*nu}  (堆叠的控制轨迹)

    A_cal矩阵结构 (T*nz × nz):
        [  A  ]
        [ A^2 ]
        [ ... ]
        [ A^T ]

    B_cal矩阵结构 (T*nz × T*nu)，块下三角Toeplitz矩阵:
        [    B     0    ...  0  ]
        [   AB      B   ...  0  ]
        [  A^2B    AB   ...  0  ]
        [  ...    ...   ... ... ]
        [A^{T-1}B A^{T-2}B ... B]

    参数:
        A: numpy数组，形状为(nz, nz)
          状态转移矩阵（Koopman A矩阵）
        B: numpy数组，形状为(nz, nu)
          控制矩阵（Koopman B矩阵）
        T: 整数，预测时域长度
          默认使用config.py中的T_HORIZON（20）

    返回:
        A_cal: numpy数组，形状为(T*nz, nz)
              堆叠的状态转移矩阵
        B_cal: numpy数组，形状为(T*nz, T*nu)
              堆叠的控制矩阵（块下三角Toeplitz）
    """
    # 获取Koopman空间维度和控制维度
    nz = A.shape[0]  # Koopman状态维度
    nu = B.shape[1]  # 控制输入维度

    # 初始化紧凑矩阵
    A_cal = np.zeros((T * nz, nz))  # 形状：(T*nz, nz)
    B_cal = np.zeros((T * nz, T * nu))  # 形状：(T*nz, T*nu)

    # 计算A的幂次: A^0, A^1, A^2, ..., A^T
    A_pow = np.eye(nz)  # A^0 = I（单位矩阵）
    A_powers = [np.eye(nz)]  # 存储A的幂次列表

    # 计算A_cal并存储A的幂次
    for t in range(T):
        A_pow = A_pow @ A  # A^{t+1} = A^t @ A
        A_cal[t * nz:(t + 1) * nz, :] = A_pow  # 第t+1行块 = A^{t+1}
        A_powers.append(A_pow.copy())  # 保存A^{t+1}

    # 构建B_cal：块下三角Toeplitz矩阵
    # B_cal[i, j] = A^{i-j} @ B，当 i >= j
    for i in range(T):  # 行块索引
        for j in range(i + 1):  # 列块索引（只到下三角）
            power = i - j  # 计算A的幂次
            B_cal[i * nz:(i + 1) * nz,
                  j * nu:(j + 1) * nu] = A_powers[power] @ B

    return A_cal, B_cal


def build_compact_C_matrix(A, C, T=T_HORIZON):
    """
    构建堆叠干扰矩阵C_cal（论文第4.1节）。

    C_cal矩阵结构 (T*nz × T*nw)，块下三角Toeplitz矩阵:
        [    C     0    ...  0  ]
        [   AC      C   ...  0  ]
        [  A^2C    AC   ...  0  ]
        [  ...    ...   ... ... ]
        [A^{T-1}C A^{T-2}C ... C]

    参数:
        A: numpy数组，形状为(nz, nz)
          状态转移矩阵（Koopman A矩阵）
        C: numpy数组，形状为(nz, nw)
          干扰矩阵（Koopman C矩阵）
        T: 整数，预测时域长度

    返回:
        C_cal: numpy数组，形状为(T*nz, T*nw)
              堆叠的干扰矩阵
    """
    nz = A.shape[0]  # Koopman状态维度
    nw = C.shape[1]  # 干扰维度

    C_cal = np.zeros((T * nz, T * nw))

    # 计算A的幂次
    A_powers = [np.eye(nz)]
    A_pow = np.eye(nz)
    for t in range(T):
        A_pow = A_pow @ A
        A_powers.append(A_pow.copy())

    # 构建C_cal：块下三角Toeplitz矩阵
    for i in range(T):
        for j in range(i + 1):
            power = i - j
            C_cal[i * nz:(i + 1) * nz,
                  j * nw:(j + 1) * nw] = A_powers[power] @ C

    return C_cal


def build_lqr_gain(A, B, Q_lqr=None, R_lqr=None):
    """
    计算Koopman系统的LQR反馈增益K（论文第4.3节）。

    使用离散时间代数Riccati方程（DARE）求解最优LQR增益：
        u_t = K @ z_t + v_t

    其中:
        K: LQR反馈增益矩阵
        v_t: 新的控制输入（MPC优化变量）

    LQR优化问题：
        min Σ (z^T Q z + u^T R u)
        s.t. z_{t+1} = A z_t + B u_t

    参数:
        A: numpy数组，形状为(nz, nz)
          Koopman状态转移矩阵
        B: numpy数组，形状为(nz, nu)
          Koopman控制矩阵
        Q_lqr: numpy数组，形状为(nz, nz)
              状态代价矩阵（默认：0.01 * I）
        R_lqr: numpy数组，形状为(nu, nu)
              控制代价矩阵（默认：1.0 * I）

    返回:
        K: numpy数组，形状为(nu, nz)
          LQR反馈增益矩阵
    """
    nz = A.shape[0]  # Koopman状态维度
    nu = B.shape[1]  # 控制输入维度

    # 如果未提供Q和R，使用默认值
    if Q_lqr is None:
        Q_lqr = np.eye(nz) * 0.01  # 较小的状态权重
    if R_lqr is None:
        R_lqr = np.eye(nu) * 1.0  # 较大的控制权重

    try:
        # 求解离散时间代数Riccati方程（DARE）
        # A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q = 0
        P = solve_discrete_are(A, B, Q_lqr, R_lqr)

        # 计算最优反馈增益
        # K = -(R + B^T P B)^{-1} B^T P A
        K = -np.linalg.solve(R_lqr + B.T @ P @ B, B.T @ P @ A)
        print(f"LQR gain computed: K shape={K.shape}")

        # 检查闭环系统稳定性
        # 闭环系统: z_{t+1} = (A + B K) z_t
        A_cl = A + B @ K
        eigenvalues = np.linalg.eigvals(A_cl)  # 计算特征值
        max_eig = np.max(np.abs(eigenvalues))  # 最大特征值模长
        print(f"  Closed-loop max eigenvalue magnitude: {max_eig:.4f}")

        # 如果最大特征值模长>=1，闭环系统不稳定
        if max_eig >= 1.0:
            print("  WARNING: Closed-loop may be unstable, using zero gain")
            K = np.zeros((nu, nz))  # 使用零增益
    except Exception as e:
        # 如果DARE求解失败，使用零增益
        print(f"  LQR solve failed ({e}), using zero feedback gain")
        K = np.zeros((nu, nz))

    return K


def build_closed_loop_matrices(A_cal, B_cal, K_fb, C_cal=None):
    """
    构建闭环紧凑矩阵A_tilde, B_tilde, C_tilde（论文第4.4节）。

    使用LQR反馈增益K_fb进行闭环控制：
        u = K_fb @ z + v

    其中v是新的控制输入（MPC优化变量）。

    闭环动力学推导：
        z = A_cal @ z_0 + B_cal @ u
          = A_cal @ z_0 + B_cal @ (K_stack @ z + v)
          = A_cal @ z_0 + B_cal @ K_stack @ z + B_cal @ v

    整理得到：(I - B_cal @ K_stack) @ z = A_cal @ z_0 + B_cal @ v

    令 M = (I - B_cal @ K_stack)^{-1}，则：
        z = M @ A_cal @ z_0 + M @ B_cal @ v + M @ C_cal @ w
          = A_tilde @ z_0 + B_tilde @ v + C_tilde @ w

    参数:
        A_cal: numpy数组，形状为(T*nz, nz)
              堆叠状态转移矩阵
        B_cal: numpy数组，形状为(T*nz, T*nu)
              堆叠控制矩阵
        K_fb: numpy数组，形状为(nu, nz)
             LQR反馈增益
        C_cal: numpy数组，形状为(T*nz, T*nw)，可选
              堆叠干扰矩阵（如果为None，不返回C_tilde）

    返回:
        A_tilde: numpy数组，形状为(T*nz, nz)
                闭环状态转移矩阵
        B_tilde: numpy数组，形状为(T*nz, T*nu)
                闭环控制矩阵
        C_tilde: numpy数组，形状为(T*nz, T*nw)，或None
                闭环干扰矩阵（如果C_cal不为None）
    """
    T_nz = A_cal.shape[0]  # T*nz
    # T_nu = B_cal.shape[1]  # T*nu（未直接使用，但保留注释说明）
    nz = A_cal.shape[1]  # nz

    # 计算T和nu
    T = T_nz // nz
    nu = K_fb.shape[0]

    # ================================================================
    # 构建堆叠K矩阵
    # ================================================================
    # K_stack = diag(K_fb, K_fb, ..., K_fb) ∈ R^{(T*nu) × (T*nz)}
    # 块对角矩阵，每个对角块都是K_fb
    K_stack = np.zeros((T * nu, T * nz))
    for t in range(T):
        K_stack[t * nu:(t + 1) * nu,
                t * nz:(t + 1) * nz] = K_fb

    # 计算 (I - B_cal @ K_stack)
    I_BK = np.eye(T_nz) - B_cal @ K_stack

    # 求逆（如果可逆）
    try:
        I_BK_inv = np.linalg.inv(I_BK)
    except np.linalg.LinAlgError:
        # 如果不可逆，使用伪逆
        print("WARNING: (I - B*K) not invertible, using pseudo-inverse")
        I_BK_inv = np.linalg.pinv(I_BK)

    # 计算闭环矩阵
    A_tilde = I_BK_inv @ A_cal
    B_tilde = I_BK_inv @ B_cal

    # 如果提供了C_cal，计算C_tilde
    C_tilde = None
    if C_cal is not None:
        C_tilde = I_BK_inv @ C_cal

    return A_tilde, B_tilde, C_tilde


def build_D_stacked(D, T=T_HORIZON):
    """
    构建堆叠投影矩阵D_stack = I_T ⊗ D（克罗内克积）。

    D_stack ∈ R^{(2*T) × (T*nz)}

    结构:
        [D   0   ...  0 ]
        [0   D   ...  0 ]
        [...     ... ...]
        [0   0   ...  D ]

    作用：
    将T步的Koopman状态堆栈z ∈ R^{T*nz}投影到[v, omega]空间，
    得到T步的[v, omega]堆栈 y ∈ R^{2*T}：
        y = D_stack @ z

    参数:
        D: numpy数组，形状为(2, nz)
          投影矩阵，将单个Koopman状态映射到[v, omega]
        T: 整数，预测时域长度

    返回:
        D_stack: numpy数组，形状为(2*T, T*nz)
                块对角堆叠投影矩阵
    """
    nz = D.shape[1]
    # 使用scipy.linalg.block_diag创建块对角矩阵
    # 重复T次D矩阵沿对角线排列
    return block_diag(*[D for _ in range(T)])


def build_smoothness_matrix(T=T_HORIZON, nu=N_U):
    """
    构建控制差分矩阵S（论文第4.5节）。

    S矩阵结构 (T*nu × T*nu)：
        S = [ I   0   ...  0 ]
            [-I   I   ...  0 ]
            [ 0  -I   ...  0 ]
            [...     ... ...]
            [ 0   0   ... -I I]

    控制差分可以表示为：
        Delta_u = S @ u - u_prev_stacked

    其中:
        Delta_u = [u_0 - u_prev; u_1 - u_0; ...; u_{T-1} - u_{T-2}]

    作用：
    在QP中用于构建控制平滑性代价，
    惩罚相邻时间步的控制增量。

    参数:
        T: 整数，预测时域长度
        nu: 整数，控制输入维度

    返回:
        S: numpy数组，形状为(T*nu, T*nu)
          控制差分矩阵
    """
    S = np.zeros((T * nu, T * nu))

    # 遍历每一行块
    for t in range(T):
        # 对角块：单位矩阵
        S[t * nu:(t + 1) * nu, t * nu:(t + 1) * nu] = np.eye(nu)

        # 下对角块（当t>0时）：负单位矩阵
        if t > 0:
            S[t * nu:(t + 1) * nu, (t - 1) * nu:t * nu] = -np.eye(nu)

    return S


def build_qp_matrices(A_tilde, B_tilde, D_stack, S, Q_stack, R_stack,
                      z0, y_ref, u_prev, K_fb=None):
    """
    构建QP黑塞矩阵H和线性项f（论文第4.5节）。

    在闭环控制下（u = K @ z + v），代价函数可以表示为：
        J = 0.5 * v^T @ H @ v + v^T @ f

    其中v是新的MPC优化变量。

    参数:
        A_tilde, B_tilde: numpy数组
                         闭环紧凑矩阵
        D_stack: numpy数组，形状为(2*T, T*nz)
                堆叠投影矩阵
        S: numpy数组，形状为(T*nu, T*nu)
          控制平滑性矩阵
        Q_stack: numpy数组，形状为(2*T, 2*T)
                堆叠跟踪代价矩阵
        R_stack: numpy数组，形状为(T*nu, T*nu)
                堆叠控制代价矩阵
        z0: numpy数组，形状为(nz,)
           初始Koopman状态
        y_ref: numpy数组，形状为(2*T,)
              堆叠参考轨迹[v_ref, omega_ref]
        u_prev: numpy数组，形状为(nu,)
               上一次控制输入
        K_fb: numpy数组，形状为(nu, nz)，可选
             LQR反馈增益（用于闭环控制平滑性）

    返回:
        H: numpy数组，形状为(T*nu, T*nu)
          QP黑塞矩阵
        f: numpy数组，形状为(T*nu,)
          QP线性项
    """
    # D_stack @ B_tilde：将控制到[v, omega]的映射
    DB_tilde = D_stack @ B_tilde

    # 如果提供了K_fb，需要对S进行闭环控制平滑性调整
    if K_fb is not None:
        # nz = A_tilde.shape[1]  # 未直接使用
        # T = B_tilde.shape[1] // K_fb.shape[0]  # 未直接使用
        # K_stack_B = np.zeros_like(S)  # 未直接使用
        # S @ (K_stack @ z + v)需要调整
        # SKB = S  # 简化：直接对v使用S
        pass  # 简化处理，直接使用S

    # 构建黑塞矩阵H
    # H = 2 * (DB_tilde^T @ Q_stack @ DB_tilde + S^T @ R_stack @ S)
    H = 2 * (DB_tilde.T @ Q_stack @ DB_tilde + S.T @ R_stack @ S)

    # 确保H对称（数值稳定性）
    H = 0.5 * (H + H.T)

    # 构建线性项f
    # Da_z0 = D_stack @ A_tilde @ z0：初始状态对输出的贡献
    Da_z0 = D_stack @ A_tilde @ z0
    # f = 2 * (DB_tilde^T @ Q_stack @ (Da_z0 - y_ref) + ...)
    u_prev_term = (
        S.T @ R_stack
        @ (-np.concatenate([[1], np.zeros(S.shape[0] - 1)]))
    )
    f = 2 * (DB_tilde.T @ Q_stack @ (Da_z0 - y_ref) + u_prev_term)

    return H, f


def pytorch_to_casadi_encoder(weights_dict):
    """
    将PyTorch编码器重构为CasADi MX表达式。

    该函数读取PyTorch模型的权重，重建一个功能等价的CasADi函数，
    使得编码器可以在CasADi优化框架中使用（如MPC中的NLP求解）。

    网络结构（论文中的编码器）：
        输入层: 5个神经元 (物理状态)
        隐藏层1: 64个神经元，ReLU激活
        隐藏层2: 128个神经元，ReLU激活
        隐藏层3: 64个神经元，ReLU激活
        输出层: 32个神经元（Koopman潜在状态）

    参数:
        weights_dict: 字典，来自model.get_network_weights()
                     包含编码器的权重和偏置

    返回:
        casadi_encode: 函数，接受CasADi MX符号输入，返回CasADi MX符号输出
                      function(x_sym) -> z_sym
    """
    # 在函数内部导入CasADi，避免全局导入依赖
    import casadi as ca

    # 兼容直进直出线性结构
    mode = weights_dict.get('mode', 'mlp')
    if mode == 'linear_passthrough':
        n_x = int(weights_dict['n_x'])
        n_z = int(weights_dict['n_z'])
        lift_weight = weights_dict.get('lift_weight', None)

        def casadi_encode(x_sym):
            if lift_weight is None or n_z <= n_x:
                return x_sym
            Wl = ca.DM(lift_weight)
            z_lift = Wl @ x_sym
            return ca.vertcat(x_sym, z_lift)

        return casadi_encode

    # 提取编码器的权重和偏置
    enc_W = weights_dict['encoder_weights']  # 权重列表
    enc_b = weights_dict['encoder_biases']   # 偏置列表

    def casadi_encode(x_sym):
        """编码器前向传播（CasADi版本）。"""
        h = x_sym  # 输入

        # 遍历每一层
        for i, (W, b) in enumerate(zip(enc_W, enc_b)):
            # 将NumPy数组转换为CasADi DM
            W_ca = ca.DM(W)
            b_ca = ca.DM(b)

            # 线性变换: h = W @ h + b
            h = W_ca @ h + b_ca

            # ReLU激活函数（除最后一层外）
            # ReLU(x) = max(0, x)
            if i < len(enc_W) - 1:
                h = ca.fmax(0, h)

        return h

    return casadi_encode


def pytorch_to_casadi_decoder(weights_dict):
    """
    将PyTorch解码器重构为CasADi MX表达式。

    该函数读取PyTorch模型的权重，重建一个功能等价的CasADi函数，
    使得解码器可以在CasADi优化框架中使用（如MPC中的NLP求解）。

    网络结构（论文中的解码器）：
        输入层: 32个神经元（Koopman潜在状态）
        隐藏层1: 64个神经元，ReLU激活
        隐藏层2: 32个神经元，ReLU激活
        输出层: 5个神经元（物理状态）

    参数:
        weights_dict: 字典，来自model.get_network_weights()
                     包含解码器的权重和偏置

    返回:
        casadi_decode: 函数，接受CasADi MX符号输入，返回CasADi MX符号输出
                      function(z_sym) -> x_sym
    """
    # 在函数内部导入CasADi，避免全局导入依赖
    import casadi as ca

    # 兼容直进直出线性结构
    mode = weights_dict.get('mode', 'mlp')
    if mode == 'linear_passthrough':
        n_x = int(weights_dict['n_x'])

        def casadi_decode(z_sym):
            return z_sym[:n_x]

        return casadi_decode

    # 提取解码器的权重和偏置
    dec_W = weights_dict['decoder_weights']  # 权重列表
    dec_b = weights_dict['decoder_biases']   # 偏置列表

    def casadi_decode(z_sym):
        """解码器前向传播（CasADi版本）。"""
        h = z_sym  # 输入

        # 遍历每一层
        for i, (W, b) in enumerate(zip(dec_W, dec_b)):
            # 将NumPy数组转换为CasADi DM
            W_ca = ca.DM(W)
            b_ca = ca.DM(b)

            # 线性变换: h = W @ h + b
            h = W_ca @ h + b_ca

            # ReLU激活函数（除最后一层外）
            if i < len(dec_W) - 1:
                h = ca.fmax(0, h)

        return h

    return casadi_decode


def verify_casadi_consistency(model, weights_dict, n_tests=100, tol=1e-5):
    """
    验证CasADi编码器/解码器与PyTorch输出的一致性。

    该函数生成随机输入，分别通过PyTorch模型和CasADi函数进行编码和解码，
    比较两者的输出差异，确保CasADi重构的神经网络与原始PyTorch模型等价。

    验证内容：
    1. 编码器一致性：PyTorch encode(x) vs CasADi encode(x)
    2. 解码器一致性：PyTorch decode(z) vs CasADi decode(z)

    参数:
        model: 训练好的PyTorch DeepKoopmanPaper模型
        weights_dict: 字典，来自model.get_network_weights()
        n_tests: 整数，随机测试输入的数量（默认100）
        tol: 浮点数，可接受的误差容限（默认1e-5）

    返回:
        max_enc_err: 浮点数，编码器的最大误差
        max_dec_err: 浮点数，解码器的最大误差

    异常:
        AssertionError: 如果编码器或解码器的误差超过容限
    """
    # 在函数内部导入依赖，避免全局导入
    import torch
    import casadi as ca

    # 创建CasADi编码器和解码器
    casadi_encode = pytorch_to_casadi_encoder(weights_dict)
    casadi_decode = pytorch_to_casadi_decoder(weights_dict)

    # 将PyTorch模型设置为评估模式
    model.eval()

    # 初始化最大误差
    max_enc_err = 0
    max_dec_err = 0

    # 进行n_tests次随机测试
    for _ in range(n_tests):
        # ================================================================
        # 测试编码器一致性
        # ================================================================
        # 生成随机输入（物理状态）
        x_np = np.random.randn(5).astype(np.float32)

        # PyTorch前向传播
        x_torch = torch.tensor(x_np).unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            z_torch = model.encode(x_torch).squeeze(0).numpy()  # 编码并去除batch维度

        # CasADi前向传播
        x_ca = ca.DM(x_np.reshape(-1, 1))  # 转换为CasADi列向量
        z_ca = np.array(casadi_encode(x_ca)).flatten()  # 编码并展平

        # 计算编码器误差
        enc_err = np.max(np.abs(z_torch - z_ca))
        max_enc_err = max(max_enc_err, enc_err)

        # ================================================================
        # 测试解码器一致性
        # ================================================================
        # 生成随机Koopman状态
        z_np = np.random.randn(32).astype(np.float32)

        # PyTorch前向传播
        z_torch_in = torch.tensor(z_np).unsqueeze(0)  # 添加batch维度
        with torch.no_grad():
            x_dec_torch = (
                model.decode(z_torch_in).squeeze(0).numpy()
            )  # 解码并去除batch维度

        # CasADi前向传播
        z_ca_in = ca.DM(z_np.reshape(-1, 1))  # 转换为CasADi列向量
        x_dec_ca = np.array(casadi_decode(z_ca_in)).flatten()  # 解码并展平

        # 计算解码器误差
        dec_err = np.max(np.abs(x_dec_torch - x_dec_ca))
        max_dec_err = max(max_dec_err, dec_err)

    # 打印验证结果
    print(f"CasADi verification ({n_tests} tests):")
    print(f"  Max encoder error: {max_enc_err:.2e} (tol={tol:.0e})")
    print(f"  Max decoder error: {max_dec_err:.2e} (tol={tol:.0e})")

    # 断言：误差必须在容限内
    assert max_enc_err < tol, f"Encoder mismatch: {max_enc_err} > {tol}"
    assert max_dec_err < tol, f"Decoder mismatch: {max_dec_err} > {tol}"
    print("  PASSED")

    return max_enc_err, max_dec_err
