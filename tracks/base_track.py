"""
赛道抽象基类 (BaseTrack)
该模块定义了所有赛道的通用接口和共享实现，包括:
- 赛道中心线、航向角、曲率的几何计算
- 参考轨迹生成（含速度曲线规划）
- 最近点查询和横向偏差计算
- 障碍物管理

子类只需实现 _build_track() 方法定义具体赛道形状。
"""

import numpy as np              # 数值计算库，用于向量化和矩阵运算
from abc import ABC, abstractmethod  # 抽象基类工具，强制子类实现特定方法
from scipy.interpolate import CubicSpline  # 三次样条插值（当前未使用，预留）
from config import (
    IDX_PX, IDX_PY, IDX_PSI, IDX_V, IDX_OMEGA,  # 状态向量各维度的索引常量
    A_LAT_MAX,       # 最大横向加速度 [m/s^2]，用于过弯速度限制
    V_MAX,           # 车辆最大速度 [m/s]
    REF_SPEED_SCALE, # 全局参考速度缩放因子（降低激进性）
    REF_ACCEL_MAX,   # 参考速度曲线的最大加速度 [m/s^2]
    REF_DECEL_MAX,   # 参考速度曲线的最大减速度 [m/s^2]
    ENABLE_OBSTACLES,  # 全局障碍物开关
    DT,              # 采样时间 [秒]
)


# ============================================================================
# BaseTrack: 赛道抽象基类
# ============================================================================
class BaseTrack(ABC):
    """
    赛车赛道的抽象基类。
    所有具体赛道（如LusailTrack、SprintOvalTrack）都必须继承此类
    并实现 _build_track() 方法来定义赛道的中心线坐标。
    """

    def __init__(self):
        """初始化赛道几何数据的占位符（在子类_build_track中填充）。"""
        self._centerline_x = None   # 赛道中心线x坐标数组 (N,)
        self._centerline_y = None   # 赛道中心线y坐标数组 (N,)
        self._heading = None        # 中心线各点的切向航向角数组 [rad] (N,)
        self._curvature = None      # 中心线各点的曲率数组 [1/m] (N,)
        self._arc_length = None     # 中心线各点的累积弧长数组 [m] (N,)
        self._obstacles = []        # 圆形障碍物列表，元素为(ox, oy, radius)三元组
        self._rect_obstacles = []   # 矩形障碍物列表，元素为(cx, cy, length, width, angle_rad)五元组
        self._total_length = 0.0    # 赛道总长度 [米]
        self._num_points = 0        # 中心线离散点的数量

    @abstractmethod
    def _build_track(self):
        """
        构建赛道几何形状。子类必须重写此方法。
        方法内必须设置: _centerline_x, _centerline_y, 并调用 _compute_geometry()
        """
        pass

    # ------------------------------------------------------------------
    # 查询方法: 获取赛道几何属性（返回副本以保护内部数据）
    # ------------------------------------------------------------------
    def get_centerline(self):
        """返回赛道中心线的(x, y)坐标数组副本。"""
        return self._centerline_x.copy(), self._centerline_y.copy()

    def get_heading(self):
        """返回中心线各点的切向航向角数组副本 [弧度]。"""
        return self._heading.copy()

    def get_curvature(self):
        """返回中心线各点的曲率数组副本 [1/米]。"""
        return self._curvature.copy()

    def get_arc_length(self):
        """返回中心线各点的累积弧长数组副本 [米]。"""
        return self._arc_length.copy()

    def get_obstacles(self):
        """
        返回圆形障碍物列表。
        若全局障碍物开关ENABLE_OBSTACLES为False，则返回空列表。
        """
        if not ENABLE_OBSTACLES:
            return []
        return list(self._obstacles)

    def get_rect_obstacles(self):
        """
        返回矩形障碍物列表，格式为(cx, cy, length, width, angle_rad)。
        若全局障碍物开关ENABLE_OBSTACLES为False，则返回空列表。
        """
        if not ENABLE_OBSTACLES:
            return []
        return list(self._rect_obstacles)

    def total_length(self):
        """返回赛道总长度 [米]。"""
        return self._total_length

    def num_points(self):
        """返回中心线离散点的数量。"""
        return self._num_points

    def closest_point(self, px, py):
        """
        查找赛道上离给定坐标(px, py)最近的点。
        使用欧几里得距离的全局最近邻搜索（适用于离散中心线点）。

        关键修复: 返回的弧长s不再是离散最近点的弧长，而是车辆投影到
        相邻赛道线段后的连续插值弧长。这消除了低速时参考轨迹 stagnation
        导致的横向偏差累积问题。

        Args:
            px: 查询点的x坐标 [米]
            py: 查询点的y坐标 [米]

        Returns:
            idx: 最近点在中心线数组中的索引
            s: 投影点处的连续累积弧长 [米]
            lateral_error: 有符号横向偏差 [米]
                正值表示车辆在中心线左侧（法向量指向左侧）
                负值表示车辆在中心线右侧
        """
        # 计算查询点到所有中心线点的x/y方向差值
        dx = self._centerline_x - px
        dy = self._centerline_y - py
        # 计算欧几里得距离
        dist = np.sqrt(dx**2 + dy**2)
        # 找到距离最小的点的索引
        idx = np.argmin(dist)

        # ------------------------------------------------------------------
        # 计算精确投影弧长（线性插值相邻线段，替代离散最近点弧长）
        # ------------------------------------------------------------------
        N = self._num_points
        prev_idx = (idx - 1) % N
        next_idx = (idx + 1) % N

        best_s = self._arc_length[idx]
        best_dist = float('inf')

        for i, j in [(prev_idx, idx), (idx, next_idx)]:
            x1, y1 = self._centerline_x[i], self._centerline_y[i]
            x2, y2 = self._centerline_x[j], self._centerline_y[j]
            dx_seg = x2 - x1
            dy_seg = y2 - y1
            seg_len_sq = dx_seg**2 + dy_seg**2
            if seg_len_sq < 1e-12:
                continue
            # 投影比例 t = ((P - P1) · (P2 - P1)) / |P2 - P1|^2
            t = ((px - x1) * dx_seg + (py - y1) * dy_seg) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            # 投影点
            proj_x = x1 + t * dx_seg
            proj_y = y1 + t * dy_seg
            d = (px - proj_x)**2 + (py - proj_y)**2
            if d < best_dist:
                best_dist = d
                # 弧长插值（处理环绕）
                s1 = self._arc_length[i]
                s2 = self._arc_length[j]
                if j < i:  # 环绕跨越终点
                    s2 += self._total_length
                best_s = s1 + t * (s2 - s1)
                if best_s >= self._total_length:
                    best_s -= self._total_length

        # ------------------------------------------------------------------
        # 计算有符号横向误差（点到直线的有符号距离）
        # ------------------------------------------------------------------
        heading = self._heading[idx]   # 最近点处的切向航向角
        # 计算中心线的单位法向量（指向赛道左侧，即逆时针90度旋转切向量）
        # 切向量: (cos(heading), sin(heading))
        # 法向量: (-sin(heading), cos(heading))
        nx = -np.sin(heading)
        ny = np.cos(heading)
        # 有符号横向误差 = 从中心线指向车辆的向量 与 单位法向量 的点积
        #   若点积为正，说明车辆在法向量方向（左侧）
        #   若点积为负，说明车辆在法向量反方向（右侧）
        lateral_error = (px - self._centerline_x[idx]) * nx + \
                        (py - self._centerline_y[idx]) * ny

        return idx, best_s, lateral_error

    def get_reference_trajectory(self, start_idx, horizon, v_ref=None,
                                 a_lat_max=A_LAT_MAX, v_max=V_MAX,
                                 current_speed=None,
                                 accel_limit=REF_ACCEL_MAX,
                                 decel_limit=REF_DECEL_MAX,
                                 start_s=None):
        """
        生成MPC预测时域内的参考轨迹。
        参考轨迹包含位置(px, py)、航向(psi)、速度(v)和横摆角速度(omega)。
        速度曲线通过"曲率限制 + 前向加速限制 + 后向减速限制"三阶段算法生成，
        确保参考速度在物理上可达且安全。

        关键修复: 参考位置不再按固定索引步进，而是根据速度曲线每步实际行驶
        距离 v*dt 来采样赛道点，确保参考轨迹的时间轴与车辆动力学匹配。

        Args:
            start_idx: 赛道上的起始点索引（用于前方曲率查询）
            horizon: 预测时域长度T（步数）
            v_ref: 固定参考速度（如果提供，则忽略曲率计算的速度）
            a_lat_max: 最大横向加速度 [m/s^2]，决定过弯速度上限: v <= sqrt(a_lat_max / kappa)
            v_max: 车辆最大速度 [m/s]
            current_speed: 当前实际速度 [m/s]，用于从当前速度平滑过渡
            accel_limit: 前向速度规划时的最大加速度 [m/s^2]
            decel_limit: 后向速度规划时的最大减速度 [m/s^2]（正值）
            start_s: 起始弧长位置 [m]。如果提供，阶段3的参考位置从该精确弧长开始；
                     否则回退到 start_idx 对应的弧长。传入车辆实际弧长可消除
                     "ref[0]滞后于车辆位置"导致的MPC追赶效应。

        Returns:
            ref: numpy数组，形状 (horizon, 5)，每行为 [px, py, psi, v, omega]
        """
        ref = np.zeros((horizon, 5))     # 预分配参考轨迹数组
        N = self._num_points             # 赛道离散点总数
        target_speeds = np.zeros(horizon)  # 曲率限制的原始目标速度

        # ------------------------------------------------------------------
        # 阶段1: 按固定索引步进计算曲率限制速度（用于获取前方曲率信息）
        # ------------------------------------------------------------------
        idx_sequence_temp = []
        for t in range(horizon):
            idx = (start_idx + t) % N
            idx_sequence_temp.append(idx)
            kappa = abs(self._curvature[idx])  # 曲率绝对值 [1/米]
            if v_ref is not None:
                target_speeds[t] = REF_SPEED_SCALE * v_ref
            elif kappa > 1e-6:
                target_speeds[t] = REF_SPEED_SCALE * min(v_max, np.sqrt(a_lat_max / kappa))
            else:
                target_speeds[t] = REF_SPEED_SCALE * v_max

        # ------------------------------------------------------------------
        # 阶段2: 速度曲线平滑（考虑纵向加减速能力约束）
        # ------------------------------------------------------------------
        if current_speed is None:
            speed_profile = target_speeds.copy()
        else:
            speed_profile = np.zeros(horizon)
            speed_profile[0] = float(np.clip(current_speed, 0.0, v_max))

            # ---- 前向扫描 (Forward Pass) ----
            for t in range(1, horizon):
                prev_idx = idx_sequence_temp[t - 1]
                idx = idx_sequence_temp[t]
                ds = self._arc_length[idx] - self._arc_length[prev_idx]
                if ds < 0:
                    ds += self._total_length
                v_reachable = np.sqrt(max(speed_profile[t - 1] ** 2 + 2.0 * accel_limit * max(ds, 0.0), 0.0))
                speed_profile[t] = min(target_speeds[t], v_reachable)

            # ---- 后向扫描 (Backward Pass) ----
            for t in range(horizon - 2, -1, -1):
                idx = idx_sequence_temp[t]
                next_idx = idx_sequence_temp[t + 1]
                ds = self._arc_length[next_idx] - self._arc_length[idx]
                if ds < 0:
                    ds += self._total_length
                v_brake_cap = np.sqrt(max(speed_profile[t + 1] ** 2 + 2.0 * decel_limit * max(ds, 0.0), 0.0))
                speed_profile[t] = min(speed_profile[t], v_brake_cap)

        # ------------------------------------------------------------------
        # 阶段3: 根据速度曲线按实际行驶距离重新生成参考位置
        # 关键修复1: 使用线性插值替代最近邻查找，避免低速时参考位置停滞
        # 关键修复2: 若调用方提供了精确弧长start_s，则以此为准，消除ref[0]滞后
        # ------------------------------------------------------------------
        ref_positions = []
        distance_covered = 0.0
        if start_s is None:
            start_s = self._arc_length[start_idx]

        # 展开航向角以避免插值时的角度环绕问题
        psi_unwrapped = np.unwrap(self._heading)

        for t in range(horizon):
            if t == 0:
                target_s = start_s
            else:
                distance_covered += speed_profile[t - 1] * DT
                target_s = (start_s + distance_covered) % self._total_length

            # 线性插值获取参考位置（避免最近邻导致的位置跳跃）
            px = np.interp(target_s, self._arc_length, self._centerline_x)
            py = np.interp(target_s, self._arc_length, self._centerline_y)
            psi = np.interp(target_s, self._arc_length, psi_unwrapped)
            kappa = np.interp(target_s, self._arc_length, self._curvature)

            ref_positions.append((px, py, psi, kappa))

        # ------------------------------------------------------------------
        # 阶段4: 填充参考轨迹（位置、航向、速度、omega）
        # ------------------------------------------------------------------
        for t, (px, py, psi, kappa) in enumerate(ref_positions):
            ref[t, IDX_PX] = px
            ref[t, IDX_PY] = py
            # wrap航向角回 [-pi, pi]
            ref[t, IDX_PSI] = ((psi + np.pi) % (2.0 * np.pi)) - np.pi
            ref[t, IDX_V] = speed_profile[t]
            ref[t, IDX_OMEGA] = speed_profile[t] * kappa

        return ref

    def get_reference_v_omega(self, start_idx, horizon, a_lat_max=A_LAT_MAX, v_max=V_MAX):
        """
        获取参考速度和横摆角速度轨迹（用于Koopman MPC的代价函数计算）。
        这是 get_reference_trajectory 的便捷包装，仅提取 [v, omega] 两列。

        Args:
            start_idx: 赛道起始点索引
            horizon: 预测时域长度
            a_lat_max: 最大横向加速度
            v_max: 最大速度

        Returns:
            y_ref: numpy数组，形状 (horizon, 2)，每行为 [v, omega]
        """
        ref = self.get_reference_trajectory(
            start_idx, horizon, a_lat_max=a_lat_max, v_max=v_max)
        return ref[:, [IDX_V, IDX_OMEGA]]  # 提取速度和角速度两列

    def _compute_geometry(self, x, y):
        """
        从赛道中心线的(x, y)坐标计算几何属性。
        包括: 累积弧长、切向航向角、曲率。
        子类在 _build_track() 中设置完中心线后应调用此方法。

        Args:
            x: 中心线x坐标数组 (N,)
            y: 中心线y坐标数组 (N,)
        """
        N = len(x)

        # ------------------------------------------------------------------
        # 1. 计算累积弧长 (Arc Length)
        # ------------------------------------------------------------------
        # 计算相邻点之间的欧几里得距离
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        # 累积求和得到各点的弧长（第一个点为0）
        self._arc_length = np.zeros(N)
        self._arc_length[1:] = np.cumsum(ds)
        self._total_length = self._arc_length[-1]  # 最后一个点即为总长度

        # ------------------------------------------------------------------
        # 2. 计算切向航向角 (Heading)
        # ------------------------------------------------------------------
        # 使用梯度近似导数: heading = arctan2(dy/ds, dx/ds)
        # edge_order=2 使用二阶精度处理边界
        self._heading = np.arctan2(
            np.gradient(y, self._arc_length, edge_order=2),
            np.gradient(x, self._arc_length, edge_order=2)
        )

        # ------------------------------------------------------------------
        # 3. 计算曲率 (Curvature)
        # ------------------------------------------------------------------
        # 曲率 = d(heading) / ds
        # 先用 np.unwrap 展开航向角，消除 ±π 环绕点的虚假跳变
        heading_unwrapped = np.unwrap(self._heading)
        dheading = np.gradient(heading_unwrapped)
        # 曲率 = 航向角变化 / 弧长变化，防止除以0加入极小值保护
        self._curvature = dheading / np.maximum(np.gradient(self._arc_length), 1e-6)

        self._num_points = N
