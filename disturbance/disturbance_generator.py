"""
使用高斯混合模型（Gaussian Mixture Model, GMM）生成干扰信号。

论文第6.1.3节：干扰w_t由高斯混合分布生成，用于模拟真实的不确定性。

高斯混合分布的优势：
1. 能够建模多模态分布（多种不同的干扰模式）
2. 比单一高斯分布更灵活，可以捕获复杂的干扰特性
3. 更贴近真实场景中的不确定性（如不同的路况、风速等）

干扰的应用场景：
- 在仿真环境中添加到车辆动力学模型
- 用于构建Wasserstein模糊集（分布鲁棒优化）
- 评估MPC控制器在不确定性下的鲁棒性
"""

import numpy as np  # 导入NumPy库，用于高效的数值计算和多维数组操作
import os  # 导入操作系统接口模块，用于文件和路径操作
import sys  # 导入系统模块，用于修改Python路径

# 将父目录添加到系统路径，确保可以导入同级别的模块（如config.py）
# os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 获取当前文件的祖父目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从配置文件导入干扰相关参数
from config import (
    N_W,  # 干扰维度，默认为5 [w_px, w_py, w_v, w_psi, w_omega]
    N_DISTURBANCE_SAMPLES,  # 干扰样本数量，默认100，用于构建Wasserstein模糊集
    ENABLE_DISTURBANCE,
    NOMINAL_SIGMA,  # 训练 nominal 干扰标准差（单点真理源）
)


class DisturbanceGenerator:
    """
    从高斯混合分布（Gaussian Mixture Distribution）生成干扰样本。

    干扰向量 w = [w_px, w_py, w_v, w_psi, w_omega] ∈ R^5
    各个分量的物理含义：
        - w_px: 位置x方向的干扰 [米]（如侧风引起的横向偏移）
        - w_py: 位置y方向的干扰 [米]
        - w_v: 速度干扰 [米/秒]（如路面摩擦力变化）
        - w_psi: 航向角干扰 [弧度]（如转向系统不确定性）
        - w_omega: 角速度干扰 [弧度/秒]（如横摆力矩扰动）

    高斯混合模型公式：
        p(w) = Σ_{k=1}^{K} π_k * N(w | μ_k, Σ_k)
    其中：
        - K: 高斯分量数量（n_components）
        - π_k: 第k个分量的权重（weights），Σπ_k = 1
        - μ_k: 第k个分量的均值向量（means）
        - Σ_k: 第k个分量的协方差矩阵（使用对角阵简化）

    该方法生成的干扰具有：
    - 多模态特性：可以模拟不同类型的干扰场景
    - 可控强度：通过sigma参数调节干扰大小
    - 可实现性：使用固定种子保证结果可复现
    """

    def __init__(self, sigma=None, n_components=3, seed=42):
        """
        初始化干扰生成器，设置高斯混合分布的参数。

        该函数创建高斯混合模型的参数：
        1. 每个高斯分量的均值（小偏移量）
        2. 每个高斯分量的协方差（通过尺度因子控制）
        3. 每个高斯分量的权重（均匀分布）

        参数:
            sigma: 浮点数，基础噪声标准差，控制干扰强度
                  默认0.05，表示5%的噪声水平
                  sigma越大，干扰越强
            n_components: 整数，高斯混合分量的数量
                         默认3，表示使用3个不同的高斯分布
                         分量越多，干扰分布越复杂
            seed: 整数，随机种子
                 默认42，确保结果可复现
                 相同的seed会生成相同的干扰序列
        """
        self.sigma = sigma if sigma is not None else NOMINAL_SIGMA  # 默认回落到 config.NOMINAL_SIGMA
        sigma = self.sigma  # 后续公式统一使用本地 sigma
        self.n_components = n_components  # 保存高斯混合分量数量

        # 创建随机数生成器实例
        # 使用RandomState而不是全局numpy.random，确保随机数序列独立可控
        self.rng = np.random.RandomState(seed)

        # ================================================================
        # 初始化高斯混合模型的参数
        # ================================================================

        # 1. 混合分量的均值（小偏移量）
        # 每个分量有一个均值向量，形状为(n_components, N_W)
        # 使用标准正态分布生成随机均值，然后缩放为sigma*0.5
        # 这样确保均值很小，干扰以零为中心但略有偏移
        self.means = self.rng.randn(n_components, N_W) * sigma * 0.5

        # 2. 混合分量的协方差（对角阵，不同尺度）
        # 使用不同的尺度因子[0.8, 1.0, 1.5]来创建不同方差的成分
        # 这模拟了不同类型的干扰：
        #   - 尺度0.8：较小的干扰（如轻微的路面不平）
        #   - 尺度1.0：中等干扰（如正常的风速变化）
        #   - 尺度1.5：较大的干扰（如强风或湿滑路面）
        # 只取前n_components个尺度因子
        self.scales = np.array([0.8, 1.0, 1.5])[:n_components]

        # 3. 混合权重（均匀分布）
        # 每个高斯分量的权重相等，总和为1
        # 例如n_components=3时，weights=[1/3, 1/3, 1/3]
        # 这意味着采样时每个分量被选中的概率相等
        self.weights = np.ones(n_components) / n_components

    def sample(self, n_samples=N_DISTURBANCE_SAMPLES):
        """
        从高斯混合分布中抽取指定数量的干扰样本。

        采样过程（对每个样本）：
        1. 根据权重概率随机选择一个高斯分量k
        2. 从选中的分量N(μ_k, σ_k^2*I)中采样：
           w = μ_k + ε * sigma * scale_k
           其中ε ~ N(0, I)是标准正态噪声

        该方法适用于：
        - 在线仿真：在每个时间步生成新的干扰
        - 蒙特卡洛测试：生成大量样本评估统计特性

        参数:
            n_samples: 整数，要生成的样本数量
                      默认使用config.py中的N_DISTURBANCE_SAMPLES（100）

        返回:
            w_samples: numpy数组，形状为(n_samples, N_W)
                      每一行是一个5维干扰向量
                      例如：w_samples[0] = [w_px, w_py, w_v, w_psi, w_omega]
        """
        if not ENABLE_DISTURBANCE:
            return np.zeros((n_samples, N_W))

        # 初始化样本数组，全部为零
        # 形状：(n_samples, N_W) = (100, 5)
        w_samples = np.zeros((n_samples, N_W))

        # 逐个生成样本
        for i in range(n_samples):
            # 步骤1：根据权重概率选择一个高斯分量
            # 例如：如果weights=[0.33, 0.33, 0.34]，则每个分量被选中的概率约为1/3
            k = self.rng.choice(self.n_components, p=self.weights)

            # 步骤2：从选中的高斯分量中采样
            # 公式: w = μ_k + ε * sigma * scale_k
            # 其中:
            #   - μ_k: 第k个分量的均值向量 self.means[k]
            #   - ε: 标准正态噪声 self.rng.randn(N_W)
            #   - sigma: 基础噪声标准差
            #   - scale_k: 第k个分量的尺度因子 self.scales[k]
            w_samples[i] = (
                self.means[k] +
                self.rng.randn(N_W) * self.sigma * self.scales[k]
            )

        return w_samples  # 返回生成的干扰样本数组

    def sample_single(self):
        """
        抽取单个干扰样本。

        该方法是sample(1)的简化版本，直接返回一维数组而不是二维数组。
        适用于：
        - 在线仿真中每个时间步添加干扰
        - 实时控制系统中的扰动生成

        返回:
            w: numpy数组，形状为(N_W,) = (5,)
               单个5维干扰向量 [w_px, w_py, w_v, w_psi, w_omega]
        """
        if not ENABLE_DISTURBANCE:
            return np.zeros(N_W)

        # 步骤1：根据权重随机选择一个高斯分量
        k = self.rng.choice(self.n_components, p=self.weights)

        # 步骤2：从选中的分量生成样本
        # 公式: w = μ_k + ε * sigma * scale_k
        # 返回形状为(N_W,)的一维数组
        return (
            self.means[k] +
            self.rng.randn(N_W) * self.sigma * self.scales[k]
        )

    def get_empirical_samples(self, n_samples=N_DISTURBANCE_SAMPLES):
        """
        获取一组固定的经验样本，用于Wasserstein模糊集构建。

        该方法与sample()的关键区别：
        1. 使用独立的随机数生成器（seed=0），不受其他采样操作影响
        2. 保证相同的n_samples总是返回相同的样本集
        3. 专门用于分布鲁棒优化中的Wasserstein模糊集构建

        在K-DRMPC中的应用：
        - 这些经验样本作为历史干扰数据
        - 用于构建Wasserstein球（模糊集）的中心分布
        - MPC优化时考虑该模糊集内所有可能的分布
        - 提高控制器对干扰不确定性的鲁棒性

        参数:
            n_samples: 整数，要生成的样本数量
                      默认使用config.py中的N_DISTURBANCE_SAMPLES（100）

        返回:
            w_empirical: numpy数组，形状为(n_samples, N_W)
                        固定的经验样本集，用于分布鲁棒优化
                        例如：(100, 5)表示100个5维干扰样本
        """
        if not ENABLE_DISTURBANCE:
            return np.zeros((n_samples, N_W))

        # 使用独立的随机数生成器以保证可实现性
        # 固定种子为0，确保结果完全可复现
        # 这个生成器不受self.rng的影响，保证了经验样本的稳定性
        rng_fixed = np.random.RandomState(0)

        # 初始化样本数组
        samples = np.zeros((n_samples, N_W))

        # 逐个生成样本（与sample方法相同的逻辑）
        for i in range(n_samples):
            # 选择高斯分量
            k = rng_fixed.choice(self.n_components, p=self.weights)
            # 生成样本：w = μ_k + ε * sigma * scale_k
            samples[i] = (
                self.means[k] +
                rng_fixed.randn(N_W) * self.sigma * self.scales[k]
            )

        return samples  # 返回固定的经验样本集
