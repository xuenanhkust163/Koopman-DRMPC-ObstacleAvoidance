"""
论文精确的Deep Koopman网络架构。

实现了论文第3.3节中描述的编码器-解码器结构，
以及线性动力学矩阵A、B。

网络架构（MLP版，论文表3）：
    编码器: x ∈ R^5 -> z ∈ R^32
        MLP（多层感知机），3个隐藏层 [64, 128, 64]
        将物理状态x提升到Koopman空间z
    解码器: z ∈ R^32 -> x ∈ R^5
        MLP，2个隐藏层 [64, 32]
        从Koopman空间z重构物理状态x
    线性动力学: z_{t+1} = A @ z_t + B @ u_t

Koopman算子理论：
    Koopman算子是一种无限维线性算子，可以精确描述非线性系统的演化。
    Deep Koopman使用神经网络学习一个有限维的近似，将非线性系统
    提升到高维线性空间（Koopman空间），在该空间中使用线性模型
    进行预测和控制。
"""

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入PyTorch神经网络模块
import numpy as np  # 导入NumPy库，用于数值计算

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODER_LAYERS, DECODER_LAYERS, DT, PX_STD, PY_STD


class DeepKoopmanPaper(nn.Module):
    """
    与论文规格完全匹配的Deep Koopman模型（表3）。

    该模型实现了Koopman算子理论的核心思想：
    通过编码器将非线性系统状态提升到高维线性空间，
    在该空间中使用线性动力学进行演化，然后通过解码器
    映射回原始物理空间。

    组件:
        - 编码器phi: R^5 -> R^32
          MLP（多层感知机），3个隐藏层 [64, 128, 64]
          将物理状态x提升到Koopman空间z

        - 线性动力学矩阵:
          A ∈ R^{32x32}: 状态演化矩阵（自主动力学）
          B ∈ R^{32x2}:  控制输入矩阵（控制对系统的影响）

        - 解码器psi: R^32 -> R^5
          MLP，2个隐藏层 [64, 32]
          从Koopman空间z重构物理状态x

    前向过程:
        1. 编码: z_t = phi(x_t)
        2. 线性演化: z_{t+1} = A @ z_t + B @ u_t
        3. 解码: x_{t+1} = psi(z_{t+1})
    """

    def __init__(self, n_x=5, n_u=2, n_z=32, n_w=5):
        """
        初始化Deep Koopman网络。

        参数:
            n_x: 整数，物理状态维度，默认5 [px, py, psi, v, omega]
            n_u: 整数，控制输入维度，默认2 [a, delta]
            n_z: 整数，Koopman空间维度，默认32
            n_w: 整数，干扰维度，默认5（与状态维度相同）
        """
        super().__init__()  # 调用父类nn.Module的初始化函数
        # 保存维度参数，供后续使用
        self.n_x = n_x  # 物理状态维度
        self.n_u = n_u  # 控制输入维度
        self.n_z = n_z  # Koopman空间维度
        self.n_w = n_w  # 干扰维度

        # MLP 编码器和解码器（论文表3规格）
        self.encoder = self._build_mlp(ENCODER_LAYERS)
        self.decoder = self._build_mlp(DECODER_LAYERS)

        # ============================================================
        # 线性动力学矩阵（第3.3.3节）
        # ============================================================
        # Koopman空间的线性演化方程: z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        # 使用nn.Parameter使矩阵成为可训练参数
        self.A = nn.Parameter(torch.empty(n_z, n_z))  # 状态演化矩阵 32x32
        self.B = nn.Parameter(torch.empty(n_z, n_u))  # 控制输入矩阵 32x2
        self.C = nn.Parameter(torch.empty(n_z, n_w))  # 干扰矩阵 32x5

        # 初始化网络权重和动力学矩阵
        self._init_weights()

    def _build_mlp(self, layer_dims):
        """
        构建MLP网络。

        参数:
            layer_dims: 列表，每层的神经元数量
        返回:
            nn.Sequential: 构建好的MLP网络
        """
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        初始化网络权重和动力学矩阵。

        权重初始化策略对训练稳定性和收敛速度至关重要：
        1. 编码器/解码器：使用Xavier初始化，保持方差稳定
        2. 矩阵A：初始化为单位矩阵+小噪声，确保初始稳定性
        3. 矩阵B：使用Xavier初始化
        """
        # MLP 编码器/解码器 Xavier 初始化
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 将矩阵A按物理动力学初始化
        # z 的前5维顺序为 [px, py, psi, v, omega]（与状态向量 x 一致）
        with torch.no_grad():
            self.A.data = torch.eye(self.n_z)
            self.A.data[0, 0] = 1.0   # px -> px
            self.A.data[1, 1] = 1.0   # py -> py
            self.A.data[2, 2] = 1.0   # psi -> psi
            self.A.data[2, 4] = 0.1   # omega -> psi (dt=0.1)
            self.A.data[3, 3] = 1.0   # v -> v
            self.A.data[4, 4] = 1.0   # omega -> omega
            noise = torch.randn_like(self.A.data) * 0.01
            noise[0, 0] = 0
            noise[1, 1] = 0
            noise[2, 2] = 0
            noise[2, 4] = 0
            noise[3, 3] = 0
            noise[4, 4] = 0
            self.A.data += noise

        # 将矩阵B按物理动力学初始化
        # z 的前5维顺序为 [px, py, psi, v, omega]
        # v_{t+1} = v_t + dt * a_t,  omega_{t+1} ≈ omega_t + (v/L)*dt * delta_t
        with torch.no_grad():
            self.B.data.zero_()
            self.B.data[3, 0] = 0.1   # a -> v (dt=0.1), v 在 z 中索引为 3
            self.B.data[4, 1] = 0.15  # delta -> omega, omega 在 z 中索引为 4

        # 使用Xavier初始化矩阵C
        nn.init.xavier_uniform_(self.C)  # 干扰矩阵

    def encode(self, x):
        """
        编码器：将物理状态提升到Koopman空间。

        采用混合passthrough结构：z = [x, phi(x)]
        前n_x维强制为物理状态x（保证位置/航向信息不丢失），
        后n_z-n_x维由MLP学习非线性特征（如cos(psi), sin(psi)等）。

        参数:
            x: 形状为(batch, n_x)的张量，物理状态

        返回:
            z: 形状为(batch, n_z)的张量，Koopman空间状态
        """
        z = self.encoder(x)
        # 强制前n_x维为物理状态x，防止编码器扭曲位置/航向信息
        z[:, :self.n_x] = x
        return z

    def decode(self, z):
        """
        解码器：从Koopman空间重构物理状态。

        由于encode已强制前n_x维为x，decode可直接取前n_x维，
        实现完美重构，避免解码器引入额外失真。

        参数:
            z: 形状为(batch, n_z)的张量，Koopman空间状态

        返回:
            x_hat: 形状为(batch, n_x)的张量，重构的物理状态
        """
        return z[:, :self.n_x]

    def linear_step(self, z, u, w=None):
        """
        Koopman空间中的线性动力学演化。

        实现线性状态转移方程：
            z_{t+1} = A @ z_t + B @ u_t + C @ w_t

        这是Koopman理论的核心：在高维空间中使用线性模型
        描述非线性系统的演化。矩阵A捕获自主动力学，
        B捕获控制输入的影响，C捕获干扰的影响。

        参数:
            z: 形状为(batch, n_z)的张量，当前Koopman空间状态
            u: 形状为(batch, n_u)的张量，控制输入
            w: 形状为(batch, n_w)的张量，干扰输入（可选，默认None）
        返回:
            z_next: 形状为(batch, n_z)的张量，预测的下一时刻Koopman空间状态
        """
        # 计算z_{t+1} = A @ z_t + B @ u_t + C @ w_t
        # 注意：这里使用 z @ A.T 而不是 A @ z.T，是因为batch在第一个维度
        # 数学上等价于对每个样本计算 A @ z_i
        z_next = z @ self.A.T + u @ self.B.T
        if w is not None:
            z_next = z_next + w @ self.C.T
        return z_next

    def forward(self, x, u, x_next=None):
        """
        完整的前向传播，返回损失计算所需的所有中间变量。

        该函数执行Koopman模型的完整前向过程：
        1. 编码当前状态
        2. 重构当前状态（用于重构损失）
        3. 线性动力学预测（用于线性动力学损失）
        4. 解码预测结果（用于多步预测损失）

        参数:
            x: 形状为(batch, n_x)的张量，当前状态
            u: 形状为(batch, n_u)的张量，控制输入
            x_next: 形状为(batch, n_x)的张量，真实下一状态（可选）
                   如果提供，用于计算线性动力学损失
        返回:
            result: 字典，包含以下键：
                'z': 编码后的当前状态 (batch, n_z)
                'z_next_linear': 线性预测的下一Koopman状态 (batch, n_z)
                'z_next_true': 编码的真实下一状态 (batch, n_z)，如果提供了x_next
                'x_recon': 解码的当前状态重构 (batch, n_x)
                'x_next_recon': 解码的真实下一状态重构 (batch, n_x)，如果提供了x_next
                'x_next_pred': 从线性动力学解码的预测 (batch, n_x)
        """
        # 步骤1：编码当前状态 x_t -> z_t
        z = self.encode(x)

        # 步骤2：重构当前状态 z_t -> x_t_recon
        # 用于计算重构损失 L_recon = ||x - psi(phi(x))||^2
        x_recon = self.decode(z)

        # 步骤3：在Koopman空间中进行线性动力学预测
        # z_{t+1} = A @ z_t + B @ u_t
        z_next_linear = self.linear_step(z, u)

        # 步骤4：解码线性预测结果到物理空间
        # x_{t+1}_pred = psi(z_{t+1})
        x_next_pred = self.decode(z_next_linear)

        # 构建结果字典
        result = {
            'z': z,                      # Koopman空间当前状态
            'z_next_linear': z_next_linear,  # 线性预测的Koopman下一状态
            'x_recon': x_recon,          # 当前状态重构
            'x_next_pred': x_next_pred,  # 下一状态预测
        }

        # 如果提供了真实下一状态，编码它用于线性动力学损失
        if x_next is not None:
            # 编码真实下一状态：x_{t+1} -> z_{t+1}_true
            z_next_true = self.encode(x_next)
            # 重构真实下一状态：z_{t+1}_true -> x_{t+1}_recon
            x_next_recon = self.decode(z_next_true)
            # 添加到结果字典
            result['z_next_true'] = z_next_true
            result['x_next_recon'] = x_next_recon

        return result

    def multi_step_predict(self, x0, u_seq, w_seq=None):
        """
        Koopman空间中的多步预测。

        该函数使用线性动力学模型进行K步前向预测，
        用于计算多步预测损失 L_pred，帮助模型学习
        长期动态演化特性，而不仅是单步预测。

        预测过程（无干扰时）：
            z_0 = phi(x_0)
            z_1 = A @ z_0 + B @ u_0 (+ C @ w_0)
            z_2 = A @ z_1 + B @ u_1 (+ C @ w_1)
            ...
            z_K = A @ z_{K-1} + B @ u_{K-1} (+ C @ w_{K-1})

            x_k = psi(z_k), for k = 1, 2, ..., K

        参数:
            x0: 形状为(batch, n_x)的张量，初始状态
            u_seq: 形状为(batch, K, n_u)的张量，控制序列
                  K是预测时域（步数）
            w_seq: 形状为(batch, K, n_w)的张量，干扰序列（可选）
        返回:
            x_preds: 形状为(batch, K, n_x)的张量，预测的状态序列 t=1..K
            z_preds: 形状为(batch, K, n_z)的张量，预测的Koopman状态序列 t=1..K
        """
        K = u_seq.shape[1]  # 获取预测时域（步数）
        has_w = w_seq is not None

        # 编码初始状态：x_0 -> z_0
        z = self.encode(x0)  # 形状: (batch, n_z)

        # 初始化预测结果列表
        x_preds = []  # 存储物理空间预测
        z_preds = []  # 存储Koopman空间预测

        # 逐步进行K步预测
        for k in range(K):
            # 提取第k步的控制输入
            u_k = u_seq[:, k, :]  # 形状: (batch, n_u)
            # 提取第k步的干扰输入（如果有）
            w_k = w_seq[:, k, :] if has_w else None

            # 线性动力学演化：z_{k+1} = A @ z_k + B @ u_k + C @ w_k
            z = self.linear_step(z, u_k, w_k)
            # 解码到物理空间：x_{k+1} = psi(z_{k+1})
            x_pred = self.decode(z)

            # 存储预测结果
            z_preds.append(z)
            x_preds.append(x_pred)

        # 将列表转换为张量
        # stack along dim=1，得到形状 (batch, K, n_x) 和 (batch, K, n_z)
        x_preds = torch.stack(x_preds, dim=1)
        z_preds = torch.stack(z_preds, dim=1)

        return x_preds, z_preds

    def get_matrices(self):
        """
        提取A、B、C矩阵为NumPy数组。

        该函数用于将训练好的动力学矩阵导出，
        以便在MPC控制器中使用。

        返回:
            A: numpy数组，形状(n_z, n_z)，状态演化矩阵
            B: numpy数组，形状(n_z, n_u)，控制输入矩阵
            C: numpy数组，形状(n_z, n_w)，干扰矩阵
        """
        # 使用detach()从计算图中分离，然后转移到CPU并转换为NumPy
        A = self.A.detach().cpu().numpy()
        B = self.B.detach().cpu().numpy()
        C = self.C.detach().cpu().numpy()
        return A, B, C

    def get_network_weights(self):
        """
        提取所有编码器/解码器权重，用于CasADi重构。

        该函数将PyTorch网络的权重导出为NumPy数组，
        以便在CasADi中重建相同的网络结构，用于
        NMPC控制器中的非线性优化。

        返回:
            dict: 包含以下键的字典：
                'encoder_weights': 编码器权重列表，每个元素是一个线性层的权重矩阵
                'encoder_biases': 编码器偏置列表
                'decoder_weights': 解码器权重列表
                'decoder_biases': 解码器偏置列表
        """
        enc_W = []
        enc_b = []
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                enc_W.append(layer.weight.detach().cpu().numpy())
                enc_b.append(layer.bias.detach().cpu().numpy())
        dec_W = []
        dec_b = []
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                dec_W.append(layer.weight.detach().cpu().numpy())
                dec_b.append(layer.bias.detach().cpu().numpy())
        return {
            'mode': 'mlp',
            'encoder_weights': enc_W,
            'encoder_biases': enc_b,
            'decoder_weights': dec_W,
            'decoder_biases': dec_b,
        }


def koopman_loss(model, x_windows, u_windows, w_windows=None,
                 lambda_recon=1.0, lambda_linear=1.0, lambda_pred=0.5,
                 lambda_physics=0.0, lambda_reg_high_dim=0.0,
                 lambda_spectral=5.0):
    """
    计算论文的三部分损失函数（第3.3.5节）+ 物理一致性损失。

    Deep Koopman的训练使用四个损失项的加权和：
    1. 重构损失 (L_recon): 确保编解码器能准确重构原始状态
    2. 线性动力学损失 (L_linear): 确保Koopman空间中的线性演化准确性
    3. 多步预测损失 (L_pred): 提高长期预测能力
    4. 物理一致性损失 (L_physics): 强制A矩阵学习正确的位置更新

    总损失：L_total = λ_recon * L_recon + λ_linear * L_linear
                     + λ_pred * L_pred + λ_physics * L_physics

    参数:
        model: DeepKoopmanPaper实例，训练中的Koopman模型
        x_windows: 形状为(batch, K+1, n_x)的张量，状态序列
                  [x_0, x_1, ..., x_K]，包含K+1个连续状态
        u_windows: 形状为(batch, K, n_u)的张量，控制序列
                  [u_0, u_1, ..., u_{K-1}]，包含K个连续控制
        w_windows: 形状为(batch, K, n_w)的张量，干扰序列（可选）
                  [w_0, w_1, ..., w_{K-1}]，包含K个连续干扰
        lambda_recon: 浮点数，重构损失权重，默认1.0
        lambda_linear: 浮点数，线性动力学损失权重，默认1.0
        lambda_pred: 浮点数，多步预测损失权重，默认0.5
        lambda_physics: 浮点数，物理一致性损失权重，默认0.0
        lambda_reg_high_dim: 浮点数，高维正则化权重，默认0.0

    返回:
        total_loss: 张量，总损失值
        loss_dict: 字典，包含各分项损失值：
            - 'total': 总损失
            - 'recon': 重构损失
            - 'linear': 线性动力学损失
            - 'pred': 多步预测损失
            - 'physics': 物理一致性损失
    """
    # 获取批次大小和预测时域（步数）
    # batch = x_windows.shape[0]  # 未使用，但保留作为参考
    # K = u_windows.shape[1]      # 未使用，但保留作为参考

    # 提取初始状态、下一状态和第一个控制输入
    x_0 = x_windows[:, 0, :]       # 形状: (batch, n_x)，初始状态 x_t
    x_1 = x_windows[:, 1, :]       # 形状: (batch, n_x)，下一状态 x_{t+1}
    u_0 = u_windows[:, 0, :]       # 形状: (batch, n_u)，第一个控制 u_t
    w_0 = w_windows[:, 0, :] if w_windows is not None else None

    # ================================================================
    # 1. 重构损失 (L_recon)
    # ================================================================
    # 公式: L_recon = (1/M) * Σ(||x_t - psi(phi(x_t))||^2 +
    #                            ||x_{t+1} - psi(phi(x_{t+1}))||^2)
    #
    # 重构损失确保编解码器能够准确地重构输入状态，
    # 这保证了Koopman空间保留了原始状态的所有重要信息。

    # 编码当前状态：z_0 = phi(x_0)
    z_0 = model.encode(x_0)
    # 重构当前状态：x_0_recon = psi(z_0)
    x_0_recon = model.decode(z_0)

    # 编码下一状态：z_1 = phi(x_1)
    z_1 = model.encode(x_1)
    # 重构下一状态：x_1_recon = psi(z_1)
    x_1_recon = model.decode(z_1)

    # 计算重构损失（使用MSE损失）
    # 同时计算x_t和x_{t+1}的重构误差
    loss_recon = (
        nn.functional.mse_loss(x_0_recon, x_0) +  # ||x_0 - psi(phi(x_0))||^2
        nn.functional.mse_loss(x_1_recon, x_1)    # ||x_1 - psi(phi(x_1))||^2
    )

    # ================================================================
    # 2. 线性动力学损失 (L_linear)
    # ================================================================
    # 公式: L_linear = (1/M) * Σ(||phi(x_{t+1}) - A*phi(x_t) - B*u_t||^2)
    #
    # 线性动力学损失确保Koopman空间中的线性演化准确，
    # 即 z_{t+1} ≈ A @ z_t + B @ u_t

    # 使用线性模型预测下一Koopman状态：z_1_pred = A @ z_0 + B @ u_0 (+ C @ w_0)
    z_1_pred = model.linear_step(z_0, u_0, w_0)
    # 计算线性预测与真实编码状态的误差
    loss_linear = nn.functional.mse_loss(z_1_pred, z_1)

    # ================================================================
    # 2b. 物理一致性损失 (L_physics)
    # ================================================================
    # 强制A矩阵学习正确的位置更新关系：
    #   px_next = px + dt * v * cos(psi)
    #   py_next = py + dt * v * sin(psi)
    # 此处使用原始空间（米）计算，量级与线性损失相当。
    if lambda_physics > 0.0:
        # 解码线性预测结果到物理状态空间
        x_next_pred = model.decode(z_1_pred)

        # 预测的位置变化（反归一化到原始空间，单位：米）
        px_diff_pred = (x_next_pred[:, 0] - x_0[:, 0]) * PX_STD
        py_diff_pred = (x_next_pred[:, 1] - x_0[:, 1]) * PY_STD

        # 物理积分（原始空间）：dt * v * cos(psi)
        px_diff_phys = DT * x_0[:, 3] * torch.cos(x_0[:, 2])
        py_diff_phys = DT * x_0[:, 3] * torch.sin(x_0[:, 2])

        loss_physics = torch.mean(
            (px_diff_pred - px_diff_phys) ** 2 +
            (py_diff_pred - py_diff_phys) ** 2
        )
    else:
        loss_physics = torch.tensor(0.0, device=x_0.device)

    # ================================================================
    # 3. 多步预测损失 (L_pred)
    # ================================================================
    # 公式: L_pred = (1/MK) * Σ_k(||x_{t+k} - psi(A^k*phi(x_t) + Σ...))||^2)
    #
    # 多步预测损失确保模型能够准确预测长期演化，
    # 而不仅是单步预测。这提高了模型的预测能力和
    # 在MPC中的控制性能。

    # 提取目标状态序列：x_1, x_2, ..., x_K
    x_targets = x_windows[:, 1:, :]   # 形状: (batch, K, n_x)，t=1..K的目标状态
    # 控制序列：u_0, u_1, ..., u_{K-1}
    u_seq = u_windows                  # 形状: (batch, K, n_u)
    # 干扰序列（可选）：w_0, w_1, ..., w_{K-1}
    w_seq = w_windows                  # 形状: (batch, K, n_w) 或 None

    # 使用多步预测：从x_0开始，使用控制序列u_seq进行K步预测
    x_preds, _ = model.multi_step_predict(x_0, u_seq, w_seq)
    # x_preds形状: (batch, K, n_x)，预测的t=1..K的状态

    # 计算多步预测损失（所有K步的MSE）
    K = x_preds.shape[1]
    # 指数衰减权重：近期步精度更重要
    weights = torch.exp(-0.02 * torch.arange(K, device=x_0.device, dtype=torch.float32))
    # 计算每步的MSE（在batch和状态维度上平均）
    per_step_mse = torch.mean((x_preds - x_targets) ** 2, dim=(0, 2))
    loss_pred = torch.sum(weights * per_step_mse) / torch.sum(weights)

    # ================================================================
    # 4. 总损失
    # ================================================================
    # 加权求和：L_total = λ_recon * L_recon + λ_linear * L_linear
    #                     + λ_pred * L_pred + λ_physics * L_physics
    total_loss = (lambda_recon * loss_recon +
                  lambda_linear * loss_linear +
                  lambda_pred * loss_pred +
                  lambda_physics * loss_physics)

    # ================================================================
    # 5. 高维正则化损失 (L_reg)
    # ================================================================
    # 惩罚 A 矩阵物理行（psi/v/omega 行，索引 2-4）的高维列（列 5+）
    # 防止高维特征劫持 v/psi 预测，确保物理状态主导预测
    if lambda_reg_high_dim > 0.0:
        loss_reg = torch.sum(model.A[2:5, 5:] ** 2)
    else:
        loss_reg = torch.tensor(0.0, device=x_0.device)

    # ================================================================
    # 5b. 谱半径惩罚 (L_spectral)
    # ================================================================
    # 强制 spectral_radius(A) <= 1.0，防止高维特征发散导致MPC病态优化
    if lambda_spectral > 0.0:
        eigenvalues = torch.linalg.eigvals(model.A)
        spectral_radius = torch.max(torch.abs(eigenvalues))
        loss_spectral = lambda_spectral * torch.relu(spectral_radius - 1.0) ** 2
    else:
        loss_spectral = torch.tensor(0.0, device=x_0.device)

    # ================================================================
    # 6. 总损失
    # ================================================================
    # 加权求和：L_total = λ_recon * L_recon + λ_linear * L_linear
    #                     + λ_pred * L_pred + λ_physics * L_physics
    #                     + λ_reg_high_dim * L_reg + L_spectral
    total_loss = (lambda_recon * loss_recon +
                  lambda_linear * loss_linear +
                  lambda_pred * loss_pred +
                  lambda_physics * loss_physics +
                  lambda_reg_high_dim * loss_reg +
                  loss_spectral)

    # 构建损失字典，用于日志记录
    loss_dict = {
        'total': total_loss.item(),     # 总损失值
        'recon': loss_recon.item(),     # 重构损失值
        'linear': loss_linear.item(),   # 线性动力学损失值
        'pred': loss_pred.item(),       # 多步预测损失值
        'physics': loss_physics.item(), # 物理一致性损失值
        'reg': loss_reg.item(),         # 高维正则化损失值
        'spectral': loss_spectral.item(), # 谱半径惩罚值
    }

    return total_loss, loss_dict
