"""
自定义蜿蜒赛道 (CustomWindingTrack)
- 赛道长度约 3.2 公里
- 包含S型弯道和减速弯 (chicanes)
- 在弯心处放置 3 个静态圆形障碍物
"""

import numpy as np                  # 数值计算库
from scipy.interpolate import CubicSpline  # 三次样条插值，用于生成平滑赛道
import os                            # 操作系统接口，用于路径处理
import sys                           # 系统接口，用于修改模块搜索路径

# ------------------------------------------------------------------
# 将项目根目录加入Python模块搜索路径，确保能导入同项目其他模块
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracks.base_track import BaseTrack  # 导入赛道抽象基类
from config import OBSTACLE_RADIUS       # 障碍物半径配置参数


# ============================================================================
# CustomWindingTrack: 自定义蜿蜒赛道
# ============================================================================
class CustomWindingTrack(BaseTrack):
    """
    自定义蜿蜒赛道，通过参数化椭圆叠加正弦扰动生成S型弯道。
    赛道长度约 3.2 公里，在弯心（曲率最大处）放置 3 个静态障碍物。
    """

    def __init__(self, num_points=1500):
        """
        初始化赛道。
        Args:
            num_points: 最终赛道中心线的离散点数量（默认1500点）
        """
        super().__init__()            # 调用父类BaseTrack的构造函数
        self._target_points = num_points  # 保存目标离散点数
        self._build_track()           # 构建赛道几何形状

    def _build_track(self):
        """
        构建蜿蜒赛道。
        算法流程:
        1. 以参数化椭圆为基础形状
        2. 叠加多频正弦扰动产生S型弯道
        3. 缩放到目标长度 (~3.2 km)
        4. 用周期三次样条平滑并均匀重采样
        5. 计算几何属性（航向、曲率、弧长）
        6. 在弯心处放置障碍物
        """
        # ------------------------------------------------------------------
        # 步骤1: 高分辨率参数化曲线生成（原始构造）
        # ------------------------------------------------------------------
        n_build = 5000   # 初始构造使用5000个高分辨率点，保证形状精度

        # 基础形状: 拉长椭圆
        # t 为参数角，范围 [0, 2*pi)，endpoint=False 确保闭合但不重复首尾
        t = np.linspace(0, 2 * np.pi, n_build, endpoint=False)

        # 椭圆半轴长度（任意单位，后续会缩放到实际米制）
        a_major = 500    # 长半轴
        b_minor = 200    # 短半轴

        # 多频正弦扰动: 叠加3个、5个、7个周期的正弦波
        # 产生复杂的S型弯道和减速弯效果
        perturbation = (60 * np.sin(3 * t) +   # 低频大振幅: 大S弯
                        40 * np.sin(5 * t) +   # 中频中振幅: 中幅波动
                        25 * np.sin(7 * t))    # 高频小振幅: 小幅急弯

        # 将扰动叠加到椭圆半径上，x/y方向权重不同产生非对称变形
        x_raw = (a_major + perturbation * 0.3) * np.cos(t)
        y_raw = (b_minor + perturbation * 0.5) * np.sin(t)

        # ------------------------------------------------------------------
        # 步骤2: 缩放到目标长度 (~3200 米)
        # ------------------------------------------------------------------
        # 计算当前原始曲线的总长度（首尾闭合）
        dx = np.diff(np.append(x_raw, x_raw[0]))  # x方向差分，append闭合首尾
        dy = np.diff(np.append(y_raw, y_raw[0]))  # y方向差分，append闭合首尾
        ds = np.sqrt(dx**2 + dy**2)               # 每段弧长
        raw_length = np.sum(ds)                   # 原始总长度
        scale = 3200.0 / raw_length               # 缩放因子: 目标长度 / 当前长度

        x_raw *= scale   # 统一缩放x坐标
        y_raw *= scale   # 统一缩放y坐标

        # ------------------------------------------------------------------
        # 步骤3: 用周期三次样条进行平滑并均匀重采样
        # ------------------------------------------------------------------
        # 计算归一化累积弧长参数 t_param ∈ [0, 1]
        # 以弧长为参数可使样条更均匀地贴合曲线形状
        t_param = np.zeros(n_build + 1)
        for i in range(n_build):
            # 相邻点之间的欧几里得距离（循环索引处理闭合）
            t_param[i+1] = t_param[i] + np.sqrt(
                (x_raw[(i+1) % n_build] - x_raw[i])**2 +
                (y_raw[(i+1) % n_build] - y_raw[i])**2
            )
        # 归一化到 [0, 1] 区间（最后一个点对应1.0）
        t_param_norm = t_param[:-1] / t_param[-1]

        # 为周期性样条添加环绕点: 将第一个点复制到末尾 (t=1.0处)
        # 这样CubicSpline的 bc_type='periodic' 能确保曲线首尾光滑连接
        x_ext = np.append(x_raw, x_raw[0])
        y_ext = np.append(y_raw, y_raw[0])
        t_ext = np.append(t_param_norm, 1.0)

        # 构建周期三次样条: x(t) 和 y(t)
        # bc_type='periodic' 保证曲线在 t=0 和 t=1 处位置和一阶导数连续
        cs_x = CubicSpline(t_ext, x_ext, bc_type='periodic')
        cs_y = CubicSpline(t_ext, y_ext, bc_type='periodic')

        # 在 [0, 1) 区间均匀重采样为 target_points 个点
        t_fine = np.linspace(0, 1, self._target_points, endpoint=False)
        self._centerline_x = cs_x(t_fine)   # 平滑后的x坐标数组
        self._centerline_y = cs_y(t_fine)   # 平滑后的y坐标数组

        # ------------------------------------------------------------------
        # 步骤4: 计算几何属性（调用父类方法）
        # ------------------------------------------------------------------
        self._compute_geometry(self._centerline_x, self._centerline_y)

        # ------------------------------------------------------------------
        # 步骤5: 在弯心处放置障碍物
        # ------------------------------------------------------------------
        self._place_obstacles()

        # 打印赛道基本信息
        print(f"Custom Winding Track: {self._total_length:.0f}m, "
              f"{self._num_points} points, {len(self._obstacles)} obstacles")

    def _place_obstacles(self):
        """
        在赛道最急的 3 个弯心处放置静态圆形障碍物。
        算法流程:
        1. 对曲率绝对值进行平滑处理（消除噪声）
        2. 迭代寻找曲率最大的 3 个点（确保彼此间隔足够远）
        3. 在每个弯心沿法向偏移一定距离放置障碍物
        """
        N = self._num_points          # 赛道离散点总数
        curvature = np.abs(self._curvature)  # 取曲率绝对值（只关心弯的急缓，不关心左右）

        # 使用一维均匀滤波平滑曲率，消除高频噪声和毛刺
        # size=N//15 表示平滑窗口覆盖约 1/15 的赛道周长
        # mode='wrap' 处理环形赛道的边界环绕
        from scipy.ndimage import uniform_filter1d
        smooth_curv = uniform_filter1d(curvature, size=N // 15, mode='wrap')

        corner_indices = []           # 存储找到的弯心索引
        min_separation = N // 6       # 两个障碍物之间的最小点间距（约1/6圈）

        # ------------------------------------------------------------------
        # 迭代寻找 3 个弯心: 每次找当前曲率最大点，然后清空其邻域避免重复
        # ------------------------------------------------------------------
        for _ in range(3):
            idx = np.argmax(smooth_curv)   # 找到当前平滑曲率最大的点索引
            corner_indices.append(idx)
            # 将该点邻域的曲率置零，确保下一个找到的弯心距离足够远
            start = max(0, idx - min_separation)
            end = min(N, idx + min_separation)
            smooth_curv[start:end] = 0

        corner_indices.sort()         # 按索引排序（沿赛道顺序）

        # ------------------------------------------------------------------
        # 在每个弯心沿曲率法向偏移放置障碍物
        # ------------------------------------------------------------------
        offset_distance = 4.0         # 障碍物偏离中心线的距离 [米]
        for idx in corner_indices:
            heading = self._heading[idx]            # 弯心处的切向航向角
            sign = np.sign(self._curvature[idx])    # 曲率符号（左弯/右弯）
            # 计算法向量（指向弯道内侧）
            # 单位法向量 nx = -sin(heading), ny = cos(heading) 指向左侧
            # 乘以 sign 后指向弯道内侧（曲率中心方向）
            nx = -np.sin(heading) * sign
            ny = np.cos(heading) * sign
            # 障碍物坐标 = 中心线坐标 + 偏移距离 * 指向内侧的单位法向量
            ox = self._centerline_x[idx] + offset_distance * nx
            oy = self._centerline_y[idx] + offset_distance * ny
            # 将障碍物添加到列表: (x, y, radius)
            self._obstacles.append((ox, oy, OBSTACLE_RADIUS))
