"""
Deep Koopman模型的训练流程（论文中的算法1）。

本模块实现了完整的训练流程，包括：
1. 模型训练（前向传播、损失计算、反向传播、参数更新）
2. 验证（评估模型在验证集上的性能）
3. 学习率调度（根据验证损失动态调整学习率）
4. 模型保存（保存最佳模型和最终模型）
5. 早停机制（防止过拟合）
6. 训练日志记录（用于分析和可视化）

训练过程遵循论文Algorithm 1，使用三部分损失函数：
- 重构损失（L_recon）
- 线性动力学损失（L_linear）
- 多步预测损失（L_pred）
"""

import os  # 导入操作系统接口模块，用于文件和目录操作
import sys  # 导入系统模块，用于修改Python路径
import time  # 导入时间模块，用于计算训练耗时
import json  # 导入JSON模块，用于保存训练日志
import numpy as np  # 导入NumPy库，用于数值计算
import torch  # 导入PyTorch深度学习框架
import torch.optim as optim  # 导入PyTorch优化器模块

# 将父目录添加到系统路径，确保可以导入同级别的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入训练相关参数
from config import (
    EPOCHS,  # 训练轮数，默认500
    LEARNING_RATE,  # 初始学习率，默认0.001
    LR_PATIENCE,  # 学习率调度器耐心值，默认20
    LR_FACTOR,  # 学习率衰减因子，默认0.5
    EARLY_STOP_PATIENCE,  # 早停耐心值，默认50
    LAMBDA_RECON,  # 重构损失权重，默认1.0
    LAMBDA_LINEAR,  # 线性动力学损失权重，默认1.0
    LAMBDA_PRED,  # 多步预测损失权重，默认0.5
    LAMBDA_PHYSICS,  # 物理一致性损失权重，默认1.0
    LAMBDA_SPECTRAL,  # 谱半径惩罚权重，默认5.0
    MODEL_DIR  # 模型保存目录
)

# 从koopman_network模块导入模型和损失函数
from model.koopman_network import DeepKoopmanPaper, koopman_loss


def clip_spectral_radius(A_numpy, max_radius=0.99):
    """将A矩阵特征值裁剪到单位圆内，确保长期预测稳定性"""
    eigenvalues, V = np.linalg.eig(A_numpy)
    magnitudes = np.abs(eigenvalues)
    mask = magnitudes > max_radius
    if mask.any():
        eigenvalues[mask] *= max_radius / magnitudes[mask]
        A_clipped = (V @ np.diag(eigenvalues) @ np.linalg.inv(V)).real
        return A_clipped, True  # 返回裁剪后的矩阵和是否裁剪的标志
    return A_numpy, False


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE,
                device='cpu', save_dir=MODEL_DIR,
                lambda_physics=LAMBDA_PHYSICS,
                lambda_reg_high_dim=0.0,
                lambda_spectral=LAMBDA_SPECTRAL):
    """
    按照论文算法1训练Deep Koopman模型。

    训练流程：
    1. 初始化优化器和学习率调度器
    2. 对每个epoch：
       a. 训练阶段：前向传播→计算损失→反向传播→更新参数
       b. 验证阶段：计算验证损失（不更新参数）
       c. 学习率调度：根据验证损失调整学习率
       d. 模型保存：如果验证损失改善，保存最佳模型
       e. 早停检查：如果验证损失长时间不改善，停止训练
    3. 保存最终模型和训练日志

    参数:
        model: DeepKoopmanPaper实例，待训练的Koopman模型
        train_loader: PyTorch DataLoader，训练集数据加载器
                     每个batch返回(x_windows, u_windows)
        val_loader: PyTorch DataLoader，验证集数据加载器
        epochs: 整数，训练轮数，默认使用config.py中的EPOCHS（500）
        lr: 浮点数，初始学习率，默认使用config.py中的LEARNING_RATE（0.001）
        device: 字符串，训练设备，'cpu'或'cuda'
        save_dir: 字符串，模型检查点保存目录

    返回:
        model: 训练好的Deep Koopman模型
        training_log: 字典，包含训练历史记录：
            - 'train_loss': 训练总损失列表
            - 'val_loss': 验证总损失列表
            - 'train_recon': 训练重构损失列表
            - 'train_linear': 训练线性动力学损失列表
            - 'train_pred': 训练多步预测损失列表
            - 'val_recon': 验证重构损失列表
            - 'val_linear': 验证线性动力学损失列表
            - 'val_pred': 验证多步预测损失列表
            - 'lr': 每个epoch的学习率
            - 'epoch_time': 每个epoch的耗时
            - 'total_time': 总训练时间
            - 'best_val_loss': 最佳验证损失
    """
    # 创建模型保存目录（如果不存在）
    # exist_ok=True表示如果目录已存在不会报错
    os.makedirs(save_dir, exist_ok=True)

    # 将模型移动到指定设备（CPU或GPU）
    model = model.to(device)

    # 创建Adam优化器
    # Adam是一种自适应学习率优化算法，结合了动量和RMSprop的优点
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建学习率调度器（ReduceLROnPlateau）
    # 当验证损失停止改善时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控指标越小越好（损失）
        patience=LR_PATIENCE,  # 等待LR_PATIENCE个epoch不改善才降低学习率
        factor=LR_FACTOR  # 每次降低为原来的LR_FACTOR倍（0.5即减半）
    )

    # 初始化最佳验证损失为正无穷
    best_val_loss = float('inf')
    # 初始化耐心计数器（用于早停）
    patience_counter = 0

    # 初始化训练日志字典
    # 记录训练和验证过程中的各种指标
    training_log = {
        'train_loss': [],  # 训练总损失
        'val_loss': [],  # 验证总损失
        'train_recon': [],  # 训练重构损失
        'train_linear': [],  # 训练线性动力学损失
        'train_pred': [],  # 训练多步预测损失
        'train_physics': [],  # 训练物理一致性损失
        'train_spectral': [],  # 训练谱半径惩罚
        'train_reg': [],  # 训练高维正则化损失
        'val_recon': [],  # 验证重构损失
        'val_linear': [],  # 验证线性动力学损失
        'val_pred': [],  # 验证多步预测损失
        'val_physics': [],  # 验证物理一致性损失
        'val_spectral': [],  # 验证谱半径惩罚
        'val_reg': [],  # 验证高维正则化损失
        'lr': [],  # 学习率
        'epoch_time': [],  # 每个epoch的耗时
    }

    # 记录训练开始时间
    total_start = time.time()

    # 训练主循环：遍历所有epoch
    for epoch in range(epochs):
        # 记录当前epoch的开始时间
        epoch_start = time.time()

        # ================================================================
        # 阶段1：训练阶段
        # ================================================================
        # 将模型设置为训练模式
        # 这会启用dropout、batchnorm等训练时特有的行为
        model.train()

        # 初始化训练损失累加器
        train_losses = {'total': 0, 'recon': 0, 'linear': 0, 'pred': 0, 'physics': 0, 'reg': 0, 'spectral': 0}
        # 记录训练的batch数量
        n_train_batches = 0

        # 遍历训练集的所有batch
        for batch in train_loader:
            # 根据batch长度判断是否包含干扰
            if len(batch) == 3:
                batch_x, batch_u, batch_w = batch
                batch_w = batch_w.to(device)
            else:
                batch_x, batch_u = batch
                batch_w = None

            # 将数据移动到指定设备
            batch_x = batch_x.to(device)  # 状态序列，形状(batch, K+1, 5)
            batch_u = batch_u.to(device)  # 控制序列，形状(batch, K, 2)

            # 清零梯度（必须在每次反向传播前执行）
            # PyTorch默认会累积梯度，所以需要手动清零
            optimizer.zero_grad()

            # 计算损失（前向传播）
            # koopman_loss返回总损失和各个分项损失
            loss, loss_dict = koopman_loss(
                model, batch_x, batch_u, batch_w,
                lambda_recon=LAMBDA_RECON,  # 重构损失权重
                lambda_linear=LAMBDA_LINEAR,  # 线性动力学损失权重
                lambda_pred=LAMBDA_PRED,  # 多步预测损失权重
                lambda_physics=lambda_physics,  # 物理一致性损失权重
                lambda_reg_high_dim=lambda_reg_high_dim,  # 高维正则化权重
                lambda_spectral=lambda_spectral  # 谱半径惩罚权重
            )

            # 反向传播：计算梯度
            # 从loss开始，自动计算所有参数的梯度
            loss.backward()

            # 梯度裁剪：防止梯度爆炸
            # 将梯度的L2范数限制在max_norm=10.0以内
            # 这对训练稳定性非常重要，特别是对于RNN和深度网络
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # 更新模型参数
            # 使用计算出的梯度和Adam优化器更新所有可训练参数
            optimizer.step()

            # 强制重置物理关键元素，防止Adam动量导致漂移
            # z 的前5维顺序为 [px, py, psi, v, omega]
            with torch.no_grad():
                model.B.data[3, 0] = 0.1   # a -> v (dt=0.1), v 在 z 中索引为 3
                model.B.data[4, 1] = 0.15  # delta -> omega, omega 在 z 中索引为 4
                model.A.data[0, 0] = 1.0   # px -> px
                model.A.data[1, 1] = 1.0   # py -> py
                model.A.data[2, 2] = 1.0   # psi -> psi
                model.A.data[2, 4] = 0.1   # omega -> psi (dt=0.1)
                model.A.data[3, 3] = 1.0   # v -> v
                model.A.data[4, 4] = 1.0   # omega -> omega

            # 累加当前batch的损失
            for k in train_losses:
                train_losses[k] += loss_dict[k]
            n_train_batches += 1

        # 计算平均训练损失（除以batch数量）
        for k in train_losses:
            train_losses[k] /= max(n_train_batches, 1)  # 使用max避免除零

        # ================================================================
        # 阶段2：验证阶段
        # ================================================================
        # 将模型设置为评估模式
        # 这会禁用dropout、使用batchnorm的运行均值等
        model.eval()

        # 初始化验证损失累加器
        val_losses = {'total': 0, 'recon': 0, 'linear': 0, 'pred': 0, 'physics': 0, 'reg': 0, 'spectral': 0}
        # 记录验证的batch数量
        n_val_batches = 0

        # 使用torch.no_grad()禁用梯度计算
        # 验证阶段不需要反向传播，所以不需要计算梯度
        # 这样可以节省内存和计算时间
        with torch.no_grad():
            # 遍历验证集的所有batch
            for batch in val_loader:
                # 根据batch长度判断是否包含干扰
                if len(batch) == 3:
                    batch_x, batch_u, batch_w = batch
                    batch_w = batch_w.to(device)
                else:
                    batch_x, batch_u = batch
                    batch_w = None

                # 将数据移动到指定设备
                batch_x = batch_x.to(device)
                batch_u = batch_u.to(device)

                # 计算验证损失（只前向传播，不反向传播）
                _, loss_dict = koopman_loss(
                    model, batch_x, batch_u, batch_w,
                    lambda_recon=LAMBDA_RECON,
                    lambda_linear=LAMBDA_LINEAR,
                    lambda_pred=LAMBDA_PRED,
                    lambda_physics=lambda_physics,
                    lambda_reg_high_dim=lambda_reg_high_dim,
                    lambda_spectral=lambda_spectral
                )

                # 累加当前batch的损失
                for k in val_losses:
                    val_losses[k] += loss_dict[k]
                n_val_batches += 1

        # 计算平均验证损失
        for k in val_losses:
            val_losses[k] /= max(n_val_batches, 1)

        # ================================================================
        # 阶段3：学习率调度
        # ================================================================
        # 根据验证损失调整学习率
        # 如果验证损失在LR_PATIENCE个epoch内没有改善，学习率乘以LR_FACTOR
        scheduler.step(val_losses['total'])

        # 计算当前epoch的耗时
        epoch_time = time.time() - epoch_start

        # ================================================================
        # 阶段4：记录训练日志
        # ================================================================
        # 将当前epoch的各项指标添加到训练日志中
        training_log['train_loss'].append(train_losses['total'])
        training_log['val_loss'].append(val_losses['total'])
        training_log['train_recon'].append(train_losses['recon'])
        training_log['train_linear'].append(train_losses['linear'])
        training_log['train_pred'].append(train_losses['pred'])
        training_log['train_physics'].append(train_losses['physics'])
        training_log['train_reg'].append(train_losses['reg'])
        training_log['train_spectral'].append(train_losses['spectral'])
        training_log['val_recon'].append(val_losses['recon'])
        training_log['val_linear'].append(val_losses['linear'])
        training_log['val_pred'].append(val_losses['pred'])
        training_log['val_physics'].append(val_losses['physics'])
        training_log['val_reg'].append(val_losses['reg'])
        training_log['val_spectral'].append(val_losses['spectral'])
        # 记录当前学习率
        training_log['lr'].append(optimizer.param_groups[0]['lr'])
        # 记录当前epoch耗时
        training_log['epoch_time'].append(epoch_time)

        # ================================================================
        # 阶段5：打印训练进度
        # ================================================================
        # 每10个epoch打印一次，或者第一个epoch也打印
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"Train: {train_losses['total']:.6f} "
                  f"(R:{train_losses['recon']:.4f} L:{train_losses['linear']:.4f} "
                  f"P:{train_losses['pred']:.4f} Ph:{train_losses['physics']:.4f} "
                  f"Rg:{train_losses['reg']:.4f} Sp:{train_losses['spectral']:.4f}) | "
                  f"Val: {val_losses['total']:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")

        # ================================================================
        # 阶段6：保存最佳模型
        # ================================================================
        # 如果当前验证损失优于历史最佳，则保存模型
        if val_losses['total'] < best_val_loss:
            # 更新最佳验证损失
            best_val_loss = val_losses['total']
            # 重置耐心计数器
            patience_counter = 0

            # 创建检查点字典，包含模型和优化器的状态
            checkpoint = {
                'epoch': epoch + 1,  # 当前epoch数
                'model_state_dict': model.state_dict(),  # 模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
                'val_loss': best_val_loss,  # 最佳验证损失
                # 保存Koopman矩阵（用于后续MPC优化）
                'A': model.A.detach().cpu().numpy().tolist(),  # 状态转移矩阵
                'B': model.B.detach().cpu().numpy().tolist(),  # 控制矩阵
            }
            # 保存最佳模型检查点
            model_path = os.path.join(save_dir, 'best_koopman_model.pth')
            torch.save(checkpoint, model_path)

            # 每10个epoch打印一次保存信息
            if (epoch + 1) % 10 == 0:
                print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
        else:
            # 如果验证损失没有改善，增加耐心计数器
            patience_counter += 1

        # ================================================================
        # 阶段7：早停检查
        # ================================================================
        # 如果验证损失连续EARLY_STOP_PATience个epoch没有改善，停止训练
        # 这是一种防止过拟合的正则化技术
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(patience={EARLY_STOP_PATIENCE})")
            break

    # 计算总训练时间
    total_time = time.time() - total_start

    # ================================================================
    # 训练结束：加载最佳模型并裁剪A矩阵谱半径
    # ================================================================
    best_model_path = os.path.join(save_dir, 'best_koopman_model.pth')
    if os.path.exists(best_model_path):
        print("\n[INFO] 加载最佳模型进行特征值裁剪...")
        best_model = load_trained_model(best_model_path, device=device)

        # 特征值裁剪：确保部署模型稳定
        A_numpy = best_model.A.detach().cpu().numpy()
        A_clipped, was_clipped = clip_spectral_radius(A_numpy, max_radius=0.99)
        if was_clipped:
            orig_radius = np.max(np.abs(np.linalg.eigvals(A_numpy)))
            print(f"[INFO] A矩阵特征值已裁剪: 原谱半径={orig_radius:.4f} -> 0.99")
            best_model.A.data = torch.tensor(A_clipped, dtype=torch.float32, device=best_model.A.device)

        # 保存裁剪后的最佳模型（覆盖原文件）
        checkpoint_best = {
            'epoch': epoch + 1,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'A': best_model.A.detach().cpu().numpy().tolist(),
            'B': best_model.B.detach().cpu().numpy().tolist(),
        }
        torch.save(checkpoint_best, best_model_path)
        print("  -> Saved clipped best model")

        # 保存裁剪后的最终模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': best_model.state_dict(),
            'val_loss': best_val_loss,
        }, os.path.join(save_dir, 'final_koopman_model.pth'))
        print("  -> Saved clipped final model")

        # 保存Koopman矩阵到npz文件
        A, B, C = best_model.get_matrices()
        np.savez(os.path.join(save_dir, 'koopman_matrices.npz'),
                 A=A, B=B, C=C)
        print(f"  -> Saved koopman_matrices.npz (A max|eig|={np.max(np.abs(np.linalg.eigvals(A))):.4f})")

        # 使用裁剪后的模型作为返回值
        model = best_model
    else:
        # 如果没有最佳模型，保存当前最终模型
        print("\n[WARN] 未找到最佳模型，保存当前最终模型（未裁剪）")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_loss': val_losses['total'],
        }, os.path.join(save_dir, 'final_koopman_model.pth'))

    # ================================================================
    # 保存训练日志
    # ================================================================
    # 将训练日志添加到日志字典中
    training_log['total_time'] = total_time  # 总训练时间
    training_log['best_val_loss'] = best_val_loss  # 最佳验证损失

    # 将训练日志保存为JSON文件
    # indent=2使得JSON文件可读性更好
    log_path = os.path.join(save_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    # 打印训练完成信息
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 返回训练好的模型和训练日志
    return model, training_log


def load_trained_model(model_path=None, device='cpu'):
    """
    从检查点加载训练好的DeepKoopmanPaper模型。

    该函数用于推理阶段，加载已训练好的模型进行仿真或控制。

    参数:
        model_path: 字符串，模型检查点的路径
                   如果为None，则使用默认的best_koopman_model.pth
        device: 字符串，加载设备，'cpu'或'cuda'

    返回:
        model: 加载好的DeepKoopmanPaper模型（评估模式）

    使用示例:
        model = load_trained_model()
        z = model.encode(x)  # 编码状态到Koopman空间
    """
    # 如果未指定路径，使用默认的最佳模型路径
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_koopman_model.pth')

    # 创建模型实例（使用默认参数）
    model = DeepKoopmanPaper()

    # 加载检查点
    # weights_only=False允许加载包含numpy数组的检查点
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False
    )

    # 检查检查点格式并加载模型权重
    if 'model_state_dict' in checkpoint:
        # 标准格式：包含'model_state_dict'键
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        # 旧格式：检查点本身就是state_dict
        model.load_state_dict(checkpoint, strict=False)

    # 将模型设置为评估模式
    # 这会禁用dropout等训练时特有的行为
    model.eval()

    return model
