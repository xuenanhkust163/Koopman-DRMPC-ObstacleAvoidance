"""
闭环仿真引擎，用于在赛道上运行MPC控制器。
该模块是整个K-DRMPC项目的核心仿真组件，负责将赛道、控制器和干扰模型整合在一起，
执行完整的闭环MPC控制仿真循环。
"""

import numpy as np              # 导入NumPy数值计算库，用于高效的数组运算和矩阵操作
import time                     # 导入时间模块，用于测量MPC求解耗时
import os                       # 导入操作系统接口模块，用于文件路径和目录操作
import sys                      # 导入系统模块，用于修改Python搜索路径
import pickle                   # 导入pickle模块，用于对象的序列化和反序列化（保存/加载仿真结果）
import subprocess               # 导入子进程模块，用于调用外部程序（如动画生成脚本）
from collections import Counter # 从collections模块导入Counter，用于统计约束出现频率

# ============================================================================
# 路径配置与模块导入
# ============================================================================
# 将当前文件的上级目录（即项目根目录）添加到Python模块搜索路径最前面
# 这样可以确保后续导入的config、vehicle等模块来自正确的位置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从项目配置文件导入所有仿真相关的超参数和常量
from config import (
    DT,                        # 仿真时间步长 [秒]，默认0.1s
    MAX_SIM_STEPS,             # 单次仿真的最大步数上限，防止无限循环
    V_MIN, V_MAX,              # 车辆速度的下限和上限 [m/s]
    A_MIN, A_MAX,              # 纵向加速度的下限和上限 [m/s^2]
    DELTA_MAX,                 # 前轮最大转向角 [弧度]
    RESULTS_DIR,               # 仿真结果（.pkl文件）的保存目录
    FIGURES_DIR,               # 图表和动画的保存目录
    IDX_PX, IDX_PY,            # 状态向量中x位置和y位置的索引
    IDX_PSI,                   # 状态向量中航向角的索引
    IDX_V, IDX_OMEGA,          # 状态向量中速度和横摆角速度的索引
    CONTROL_UPDATE_INTERVAL,   # 默认控制更新间隔（每N步重新求解一次MPC）
    TRACK_HALF_WIDTH,          # 赛道半宽 [米]，用于碰撞检测
    VEHICLE_RADIUS,            # 车辆近似半径 [米]，用于碰撞检测
    EXPORT_STATIC_FIGURES,     # 布尔标志：是否导出静态PDF图表
    EXPORT_ANIMATION,          # 布尔标志：是否导出GIF动画
    ANIMATION_FPS,             # 导出动画的帧率 [帧/秒]
)
# 从车辆动力学模块导入离散时间自行车模型步进函数
from vehicle.bicycle_model import discrete_step


# ============================================================================
# SimResult: 仿真结果数据容器
# ============================================================================
class SimResult:
    """
    仿真结果容器类，用于存储和封装单次闭环仿真的全部数据。
    该类是一个纯数据结构（DTO），本身不包含业务逻辑，
    只负责收集仿真过程中各时间步的状态、控制、求解器信息等原始数据。
    """

    def __init__(self, method_name, track_name):
        """
        初始化仿真结果对象，创建所有用于存储数据的空列表和标志位。

        Args:
            method_name: 控制器/方法名称字符串，例如 "K-DRMPC" 或 "LMPC"
            track_name: 赛道名称字符串，例如 "SprintOvalTrack"
        """
        # ------------------------------------------------------------------
        # 基本标识信息
        # ------------------------------------------------------------------
        self.method_name = method_name     # 控制器或方法的名称，用于结果归档和图表标注
        self.track_name = track_name       # 赛道的名称，用于区分不同赛道上的实验结果

        # ------------------------------------------------------------------
        # 时间序列数据列表（每仿真步追加一个元素）
        # ------------------------------------------------------------------
        # 状态轨迹列表，每个元素为形状(5,)的numpy数组，状态向量: [px, py, psi, v, omega]
        self.states = []
        # 控制输入轨迹列表，每个元素为形状(2,)的numpy数组，控制向量: [a, delta] = [纵向加速度, 前轮转向角]
        self.controls = []
        # MPC求解耗时列表 [秒]，用于评估控制器实时性能
        self.solve_times = []
        # 求解器返回状态字符串列表，如 "optimal"/"suboptimal"
        self.solve_statuses = []
        # 每步求解的详细诊断信息字典列表，包含: cost分项、活跃约束、松弛变量最大值等
        self.solve_debug = []
        # 参考状态列表，每个元素为形状(5,)的参考轨迹点
        self.ref_states = []
        # 仿真时间戳列表 [秒]，从0开始以DT为间隔递增
        self.timestamps = []

        # ------------------------------------------------------------------
        # 事件标志与元信息
        # ------------------------------------------------------------------
        # 布尔标志: 车辆是否成功完成了一圈（达到lap_fraction阈值）
        self.lap_completed = False
        # 完成一圈的耗时 [秒]，仅在lap_completed=True时有效
        self.lap_time = None
        # 本次仿真实际执行的总步数（提前终止时也记录）
        self.total_steps = 0
        # 布尔标志: 车辆是否因撞到赛道边界而提前终止
        self.crashed = False
        # 撞车/出界时对应的仿真步数索引（从1开始计数）
        self.crash_step = None
        # 撞车/出界时刻的仿真时间 [秒]
        self.crash_time = None
        # 撞车原因描述字符串，用于调试和分析
        self.crash_reason = None

    def to_arrays(self):
        """
        将SimResult中存储的Python列表转换为NumPy数组格式。
        转换后的数组更便于后续的数值分析、绘图和向量化计算。

        Returns:
            dict: 包含以下键值对的字典:
                - 'states':     np.ndarray, 形状 (N+1, 5)，完整状态轨迹
                - 'controls':   np.ndarray, 形状 (N, 2)，控制输入序列
                - 'solve_times':np.ndarray, 形状 (N,)，每步MPC求解耗时
                - 'timestamps': np.ndarray, 形状 (N,)，仿真时间戳
                - 'solve_debug':list, 长度为N的调试信息字典列表
                - 'crashed':    bool, 是否撞车
                - 'crash_step': int or None, 撞车步数
                - 'crash_time': float or None, 撞车时间
                - 'crash_reason':str or None, 撞车原因
        """
        return {
            'states': np.array(self.states),              # 将状态列表堆叠为二维NumPy数组
            'controls': np.array(self.controls),          # 将控制列表堆叠为二维NumPy数组
            'solve_times': np.array(self.solve_times),    # 将求解时间列表转换为一维数组
            'timestamps': np.array(self.timestamps),      # 将时间戳列表转换为一维数组
            'solve_debug': list(self.solve_debug),        # 保持为列表（元素为字典，结构不统一）
            'crashed': self.crashed,                      # 直接传递布尔标志
            'crash_step': self.crash_step,                # 直接传递整数或None
            'crash_time': self.crash_time,                # 直接传递浮点数或None
            'crash_reason': self.crash_reason,            # 直接传递字符串或None
        }


# ============================================================================
# Simulator: 闭环仿真引擎
# ============================================================================
class Simulator:
    """
    在赛车赛道上对MPC控制器进行闭环仿真的核心引擎类。
    该类是仿真流程的编排者（Orchestrator），负责将三个核心组件整合在一起：
      1. 赛道 (Track): 提供赛道几何、参考轨迹和障碍物信息
      2. 控制器 (Controller): 求解MPC优化问题，生成控制指令
      3. 干扰生成器 (DisturbanceGenerator): 模拟环境不确定性
    主循环按时间步推进：传感器测量 -> MPC求解 -> 执行控制 -> 动力学传播 -> 记录数据
    """

    def __init__(self, track, controller, disturbance_gen=None):
        """
        初始化仿真器，保存三个核心组件的引用。

        Args:
            track: BaseTrack子类实例，表示赛道对象，提供几何和参考轨迹
            controller: MPC控制器对象，必须实现 solve(x, ref, obstacles) 接口
            disturbance_gen: DisturbanceGenerator实例（可选），如果为None则仿真无干扰
        """
        self.track = track                      # 保存赛道对象引用
        self.controller = controller            # 保存控制器对象引用
        self.disturbance_gen = disturbance_gen  # 保存干扰生成器引用（允许为None）

    def run(self, x0=None, max_steps=MAX_SIM_STEPS, lap_fraction=0.95,
            verbose=True, detailed_step_log=False,
            detailed_step_log_max_steps=None,
            control_update_interval=CONTROL_UPDATE_INTERVAL):
        """
        执行完整的闭环MPC仿真主循环。

        仿真流程如下（每步迭代）：
        1. 定位：根据当前位置(x,y)找到赛道上的最近点
        2. 参考：从最近点向前预测时域长度，获取参考轨迹
        3. 求解：调用控制器求解MPC优化问题（若开启降频则复用上一步控制）
        4. 限幅：将优化出的控制量裁剪到车辆物理执行范围内
        5. 干扰：若启用干扰生成器，则采样噪声叠加到下一状态
        6. 传播：调用自行车模型计算下一时刻状态
        7. 记录：将状态、控制、求解信息存入SimResult
        8. 检测：检查是否完成一圈、是否撞墙、是否发散

        Args:
            x0: 形状为(5,)的初始状态向量 [px, py, psi, v, omega]。
                如果为None，则自动从赛道起点（索引0）构建初始状态。
            max_steps: 最大仿真步数上限，防止无限循环，默认使用config中的MAX_SIM_STEPS
            lap_fraction: 判定"完成一圈"所需的累积行驶距离占赛道总长的比例（默认0.95即95%）
            verbose: 是否在终端打印进度摘要信息
            detailed_step_log: 是否每一步都打印详细的数值日志（用于调试）
            detailed_step_log_max_steps: 限制详细日志打印的最大步数（None表示打印全部步）
            control_update_interval: 控制降频间隔，每N步重新求解一次MPC，中间保持控制不变

        Returns:
            SimResult: 包含完整仿真轨迹、控制序列和事件标志的结果对象
        """
        # ------------------------------------------------------------------
        # 参数校验与组件引用获取
        # ------------------------------------------------------------------
        if control_update_interval <= 0:
            raise ValueError("control_update_interval 必须为正整数")

        track = self.track              # 获取赛道对象的本地引用（加速属性访问）
        controller = self.controller    # 获取控制器对象的本地引用
        obstacles = track.get_obstacles()  # 获取赛道上当前激活的障碍物列表

        # ------------------------------------------------------------------
        # 阶段1: 初始化起始状态
        # ------------------------------------------------------------------
        if x0 is None:
            # 用户未提供初始状态，则从赛道起点自动构建
            cx, cy = track.get_centerline()     # 获取赛道中心线所有点的x、y坐标数组
            heading = track.get_heading()       # 获取中心线各点的切向航向角数组
            # 获取起点处的参考速度和参考横摆角速度（返回形状为(1,2)的数组）
            ref_init = track.get_reference_v_omega(0, 1)
            # 构建初始状态向量: [起点x, 起点y, 起点航向, 参考速度, 参考角速度]
            x0 = np.array([cx[0], cy[0], heading[0], ref_init[0, 0], ref_init[0, 1]])

        # 创建仿真结果容器，自动记录方法名和赛道名
        result = SimResult(controller.name, track.__class__.__name__)
        # 重置控制器内部状态（清除历史控制、状态估计等）
        controller.reset()

        # ------------------------------------------------------------------
        # 阶段2: 初始化仿真循环变量
        # ------------------------------------------------------------------
        x = x0.copy()                   # 复制初始状态，避免修改外部传入的x0
        result.states.append(x.copy())  # 将初始状态(t=0)加入结果记录

        # 查询车辆在赛道上的初始投影信息
        # closest_point返回: (最近点索引, 沿中心线弧长s, 横向偏差lat_err)
        start_idx, start_s, _ = track.closest_point(x[0], x[1])
        max_s = track.total_length()    # 获取赛道总长度 [米]
        cumulative_s = 0.0              # 初始化累积行驶距离（用于判断是否完成一圈）
        prev_s = start_s                # 记录上一步的弧长位置，用于计算步进距离ds
        # 计算碰撞检测阈值: 赛道半宽减去车辆半径，小于等于0时取0
        crash_lat_limit = max(TRACK_HALF_WIDTH - VEHICLE_RADIUS, 0.0)

        # ------------------------------------------------------------------
        # 阶段3: 仿真前信息输出
        # ------------------------------------------------------------------
        if verbose:
            print(f"\n正在仿真 {controller.name} 在 {track.__class__.__name__} 赛道上...")
            print(f"  赛道长度: {max_s:.0f}m, 障碍物数量: {len(obstacles)}")
            print(f"  控制更新间隔: 每 {control_update_interval} 步求解一次MPC")

        # ------------------------------------------------------------------
        # 阶段4: 控制保持变量初始化（用于降频模式）
        # ------------------------------------------------------------------
        held_u = np.zeros(2)                              # 保存上一次实际施加的控制输入
        held_info = {'solve_time': 0.0, 'status': 'hold', 'debug': None}  # 保存对应求解信息

        # ==================================================================
        # 主仿真循环: 按时间步迭代推进仿真
        # ==================================================================
        for step in range(max_steps):
            t_sim = step * DT  # 计算当前仿真时间 [秒]

            # ------------------------------------------------------------------
            # 步骤A: 定位 — 查询车辆在赛道上的投影位置
            # ------------------------------------------------------------------
            # closest_point返回三元组:
            #   idx: 中心线上最近点的索引
            #   current_s: 沿中心线的累积弧长 [米]
            #   lat_err: 横向偏差 [米]，正值表示中心线左侧，负值表示右侧
            idx, current_s, lat_err = track.closest_point(x[0], x[1])

            # ------------------------------------------------------------------
            # 步骤B: 更新累积行驶距离（处理赛道环绕）
            # ------------------------------------------------------------------
            ds = current_s - prev_s  # 计算单步弧长变化量
            # 处理环绕情况: 当车辆跨越起点时，current_s会从max_s跳变到0（或反向）
            if ds < -max_s / 2:
                ds += max_s  # 负向跨越起点: 弧长变化加上赛道总长
            elif ds > max_s / 2:
                ds -= max_s  # 正向跨越起点: 弧长变化减去赛道总长
            cumulative_s += abs(ds)  # 累加绝对行驶距离（用于判断是否完成一圈）
            prev_s = current_s       # 保存当前弧长供下一步使用

            # ------------------------------------------------------------------
            # 步骤C: 获取参考轨迹（MPC预测时域内的目标路径）
            # ------------------------------------------------------------------
            from config import T_HORIZON  # 导入预测时域长度（默认40步=4秒）
            # 从当前最近点idx向前获取T_HORIZON步的参考轨迹
            # start_s传入车辆实际弧长，消除"ref[0]滞后于最近点索引"的追赶效应
            # 关键修复: 不再传入current_speed，避免参考速度曲线与车辆速度绑定形成正反馈陷阱
            ref = track.get_reference_trajectory(
                idx, T_HORIZON, start_s=current_s)

            # ------------------------------------------------------------------
            # 步骤D: 求解MPC优化问题（或复用保持控制）
            # ------------------------------------------------------------------
            should_solve = (step % control_update_interval == 0)
            if should_solve:
                try:
                    # 优先尝试传入u_prev参数，告知控制器上一步实际执行的控制
                    # 这有助于某些控制器（如带控制速率惩罚的MPC）更准确地建模
                    try:
                        u_opt, info = controller.solve(x, ref, obstacles, u_prev=held_u)
                    except TypeError:
                        # 若控制器不支持u_prev参数（兼容旧接口），则降级调用
                        u_opt, info = controller.solve(x, ref, obstacles)
                except Exception as e:
                    # MPC求解失败时的容错处理
                    if verbose:
                        print(f"  步骤 {step}: 控制器错误: {e}")
                    # 保持上一步控制不变，避免控制跳变导致车辆失稳
                    u_opt = held_u.copy()
                    info = {'solve_time': 0, 'status': f'error: {str(e)[:30]}', 'debug': None}
            else:
                # 降频模式: 非求解步直接复用上一步的控制输入
                u_opt = held_u.copy()
                info = held_info.copy()
                info['solve_time'] = 0.0
                info['status'] = f"hold({held_info.get('status', 'unknown')})"

            # ------------------------------------------------------------------
            # 步骤E: 控制输入物理限幅（执行器饱和约束）
            # ------------------------------------------------------------------
            # 加速度限幅: [A_MIN, A_MAX]，防止要求过大驱动力或制动力
            u_opt[0] = np.clip(u_opt[0], A_MIN, A_MAX)
            # 转向角限幅: [-DELTA_MAX, DELTA_MAX]，防止要求超出行驶机构物理极限
            u_opt[1] = np.clip(u_opt[1], -DELTA_MAX, DELTA_MAX)

            # 仅在求解步更新保持控制，确保非求解步复用的是经过限幅的实际控制量
            if should_solve:
                held_u = u_opt.copy()
                held_info = info.copy()

            # ------------------------------------------------------------------
            # 步骤F: 采样环境干扰噪声（模拟传感器误差和外部扰动）
            # ------------------------------------------------------------------
            noise = np.zeros(5)  # 初始化5维零噪声向量
            if self.disturbance_gen is not None:
                w = self.disturbance_gen.sample_single()  # 从干扰分布采样一个样本
                # 对不同状态维度的噪声进行差异化缩放，以符合物理量级
                noise[0] = w[0] * 0.1         # x位置噪声: 0.1倍缩放 [米]
                noise[1] = w[1] * 0.1         # y位置噪声: 0.1倍缩放 [米]
                noise[IDX_PSI] = w[3] * 0.01  # 航向角噪声: 0.01倍缩放 [弧度]
                noise[IDX_V] = w[2] * 0.05    # 速度噪声: 0.05倍缩放 [m/s]
                noise[IDX_OMEGA] = w[4] * 0.01  # 角速度噪声: 0.01倍缩放 [rad/s]

            # ------------------------------------------------------------------
            # 步骤G: 动力学传播（计算下一时刻状态）
            # ------------------------------------------------------------------
            # 使用离散时间自行车模型计算无干扰的下一状态，然后叠加噪声
            x_next = discrete_step(x, u_opt) + noise
            # 对速度进行硬约束截断，防止数值异常导致速度越界
            x_next[IDX_V] = np.clip(x_next[IDX_V], V_MIN, V_MAX)

            # ------------------------------------------------------------------
            # 步骤H: 记录当前步数据到结果容器
            # ------------------------------------------------------------------
            result.controls.append(u_opt.copy())         # 记录实际施加的控制输入
            result.solve_times.append(info.get('solve_time', 0))      # 记录求解耗时
            result.solve_statuses.append(info.get('status', 'unknown'))  # 记录求解器状态
            result.solve_debug.append(info.get('debug'))  # 记录调试诊断信息
            result.ref_states.append(ref[0].copy())       # 记录当前步的参考状态（仅第1个参考点）
            result.timestamps.append(t_sim)               # 记录仿真时间戳

            # 状态推进: 将当前状态更新为下一时刻状态
            x = x_next
            result.states.append(x.copy())  # 记录新的状态到轨迹

            # ------------------------------------------------------------------
            # 步骤I: 逐步详细日志输出（用于深度调试）
            # ------------------------------------------------------------------
            if detailed_step_log and (
                detailed_step_log_max_steps is None or step < detailed_step_log_max_steps
            ):
                # 将关键变量格式化为可读字符串
                x_str = np.array2string(result.states[-2], precision=4, suppress_small=True)
                ref_str = np.array2string(ref[0], precision=4, suppress_small=True)
                u_str = np.array2string(u_opt, precision=4, suppress_small=True)
                noise_str = np.array2string(noise, precision=4, suppress_small=True)
                x_next_str = np.array2string(x_next, precision=4, suppress_small=True)
                # 打印一行汇总信息
                print(
                    f"[Step {step:04d}] t={t_sim:7.2f}s "
                    f"idx={idx:4d} s={current_s:8.2f}m ds={ds:7.3f}m "
                    f"cum={cumulative_s:8.2f}m prog={cumulative_s/max_s*100:6.2f}% lat={lat_err:8.4f}m"
                )
                print(f"  x      = {x_str}")       # 当前状态
                print(f"  ref[0] = {ref_str}")      # 参考状态
                print(f"  u_opt  = {u_str}")        # 优化控制
                print(
                    f"  solve  = status={info.get('status','unknown')} "
                    f"time={info.get('solve_time', 0.0) * 1000:.2f}ms"
                )
                # 若控制器提供了调试信息，则展开打印cost分项和活跃约束
                debug = info.get('debug')
                if debug:
                    step0 = debug.get('step0', {})
                    horizon = debug.get('horizon', {})
                    active = ','.join(debug.get('active_constraints', [])) or 'none'
                    key_parts = []
                    for key in (
                        'cost_track_vomega', 'cost_contour', 'cost_lag',
                        'cost_position',
                        'cost_heading', 'cost_heading_mpcc', 'cost_speed',
                        'cost_progress', 'cost_progress_mpcc', 'cost_du',
                        'cost_abs_u'
                    ):
                        if key in step0:
                            key_parts.append(f"{key}={step0[key]:.3f}")
                    if 'cost_cvar' in horizon:
                        key_parts.append(f"cost_cvar={horizon['cost_cvar']:.3f}")
                    if 'risk_eta' in horizon:
                        key_parts.append(f"risk_eta={horizon['risk_eta']:.3f}")
                    key_parts.append(f"v_slack_max={debug.get('v_slack_max', 0.0):.3f}")
                    key_parts.append(f"obs_slack_max={debug.get('obs_slack_max', 0.0):.3f}")
                    print(f"  diag   = {'; '.join(key_parts)}")
                    print(f"  active = {active}")
                print(f"  noise  = {noise_str}")      # 干扰噪声
                print(f"  x_next = {x_next_str}")     # 下一状态
                print(
                    f"  speed  = v:{result.states[-2][IDX_V]:.4f} -> {x_next[IDX_V]:.4f} m/s, "
                    f"omega:{result.states[-2][IDX_OMEGA]:.4f} -> {x_next[IDX_OMEGA]:.4f} rad/s"
                )

            # ------------------------------------------------------------------
            # 步骤J: 定期进度摘要（每10步轻量输出，每100步详细输出）
            # ------------------------------------------------------------------
            if verbose:
                solve_t = info.get('solve_time', 0) * 1000
                if (step + 1) % 100 == 0:
                    print(f"  步骤 {step+1}/{max_steps}: "
                          f"速度={x[IDX_V]:.1f}m/s, "
                          f"横向误差={lat_err:.1f}m, "
                          f"进度={cumulative_s/max_s*100:.1f}%, "
                          f"求解时间={solve_t:.1f}ms")
                elif (step + 1) % 10 == 0:
                    # 轻量进度点，避免长时静默导致用户误以为卡死
                    status_tag = info.get('status', '?')
                    print(f"  ... 步骤 {step+1}/{max_steps} "
                          f"(s={current_s:.1f}m, lat={lat_err:.2f}m, "
                          f"solve={solve_t:.0f}ms, status={status_tag})")

            # ------------------------------------------------------------------
            # 步骤K: 终止条件检测
            # ------------------------------------------------------------------
            # K1. 检查是否完成一圈（成功条件）
            if cumulative_s >= max_s * lap_fraction:
                result.lap_completed = True
                result.lap_time = (step + 1) * DT
                if verbose:
                    print(f"  在第 {step+1} 步完成一圈, 时间={result.lap_time:.1f}s")
                break  # 正常终止仿真

            # K2. 检查是否撞到赛道边界（失败条件1）
            if abs(lat_err) >= crash_lat_limit:
                result.crashed = True
                result.crash_step = step + 1
                result.crash_time = (step + 1) * DT
                result.crash_reason = (
                    f"track boundary hit: |lat_err|={abs(lat_err):.3f}m >= {crash_lat_limit:.3f}m"
                )
                if verbose:
                    print(
                        f"  车辆在第 {step+1} 步撞到赛道边界 "
                        f"(横向误差={lat_err:.2f}m, 阈值={crash_lat_limit:.2f}m)"
                    )
                break  # 异常终止仿真

            # K3. 检查是否严重发散（失败条件2，防止无限偏离浪费计算资源）
            if abs(lat_err) > 500:
                if verbose:
                    print(f"  车辆在第 {step+1} 步发散 (横向误差={lat_err:.0f}m)")
                break  # 异常终止仿真

        # ==================================================================
        # 仿真结束: 收尾与结果返回
        # ==================================================================
        # 统计实际执行的总步数（等于控制输入列表的长度）
        result.total_steps = len(result.controls)

        if verbose:
            # 计算并打印平均MPC求解时间，用于评估实时性能
            avg_solve = np.mean(result.solve_times) if result.solve_times else 0
            print(f"  仿真完成: {result.total_steps} 步, "
                  f"平均求解时间={avg_solve*1000:.1f}ms")

        return result

    # ==================================================================
    # 静态工具方法: 结果导出与可视化
    # ==================================================================

    @staticmethod
    def _export_result_to_step_log(result, output_path):
        """
        将仿真结果导出为对齐的逐行文本日志文件。
        该日志格式适合用文本编辑器或脚本快速查阅每步的详细数值。
        """
        # 将SimResult中的列表数据转换为NumPy数组以便索引
        data = result.to_arrays()
        states = data['states']
        controls = data['controls']
        solve_times = data['solve_times']
        timestamps = data['timestamps']
        solve_debug = list(data.get('solve_debug', []))
        ref_states = np.array(result.ref_states)
        solve_statuses = list(result.solve_statuses)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            # 构建表头: 定义每列的名称和宽度
            header = (
                f"{'step':>5} {'t(s)':>8} {'status':<12} {'solve_ms':>9} "
                f"{'x':>9} {'y':>9} {'psi':>9} {'v':>8} {'omega':>8} "
                f"{'ref_v':>8} {'ref_omega':>10} {'a':>8} {'delta':>8} "
                f"{'next_v':>8} {'next_omega':>10}"
            )
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            # 写入元信息注释行（以#开头）
            f.write(f"# method={result.method_name}\n")
            f.write(f"# track={result.track_name}\n")
            f.write(f"# lap_completed={result.lap_completed}\n")
            f.write(f"# lap_time={result.lap_time}\n")
            f.write(f"# total_steps={result.total_steps}\n")
            f.write(f"# crashed={result.crashed}\n")
            f.write(f"# crash_step={result.crash_step}\n")
            f.write(f"# crash_time={result.crash_time}\n")
            f.write(f"# crash_reason={result.crash_reason}\n")

            # 写入初始状态（便于复现仿真）
            if len(states) > 0:
                f.write(
                    "# init_state="
                    f"[{states[0, 0]:.6f}, {states[0, 1]:.6f}, {states[0, 2]:.6f}, "
                    f"{states[0, 3]:.6f}, {states[0, 4]:.6f}]\n"
                )
            f.write("\n")

            # 逐行写入每一步的仿真数据
            for step in range(len(controls)):
                x_t = states[step]           # 当前步状态
                x_next = states[step + 1]    # 下一步状态
                u_t = controls[step]         # 当前步控制
                # 若参考状态不足则用NaN填充（异常情况保护）
                ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
                # 求解时间转换为毫秒
                solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
                status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
                debug = solve_debug[step] if step < len(solve_debug) else None
                t_sim = timestamps[step] if step < len(timestamps) else float(step)

                # 写入主数据行
                f.write(
                    f"{step:5d} {t_sim:8.3f} {status:<12.12} {solve_time_ms:9.3f} "
                    f"{x_t[0]:9.3f} {x_t[1]:9.3f} {x_t[2]:9.4f} {x_t[3]:8.3f} {x_t[4]:8.4f} "
                    f"{ref_t[3]:8.3f} {ref_t[4]:10.4f} {u_t[0]:8.4f} {u_t[1]:8.4f} "
                    f"{x_next[3]:8.3f} {x_next[4]:10.4f}\n"
                )
                # 如果该步有调试信息，则额外写入诊断行
                if debug:
                    step0 = debug.get('step0', {})
                    horizon = debug.get('horizon', {})
                    active = ','.join(debug.get('active_constraints', [])) or 'none'
                    f.write(
                        f"  # debug step0={step0} horizon={horizon} active={active} "
                        f"v_slack_max={debug.get('v_slack_max', 0.0):.6f} "
                        f"obs_slack_max={debug.get('obs_slack_max', 0.0):.6f}\n"
                    )

    @staticmethod
    def _export_result_to_compact_log(result, output_path):
        """
        将仿真结果导出为紧凑的单行-per-step日志，便于快速浏览。
        相比 _export_result_to_step_log，该格式列数更少、文件体积更小。
        """
        data = result.to_arrays()
        states = data['states']
        controls = data['controls']
        solve_times = data['solve_times']
        timestamps = data['timestamps']
        solve_debug = list(data.get('solve_debug', []))
        ref_states = np.array(result.ref_states)
        solve_statuses = list(result.solve_statuses)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            # 写入紧凑格式的列标题
            f.write("# step t(s) v ref_v omega ref_omega a delta solve_ms status\n")

            for step in range(len(controls)):
                x_t = states[step]
                u_t = controls[step]
                ref_t = ref_states[step] if step < len(ref_states) else np.full(5, np.nan)
                solve_time_ms = solve_times[step] * 1000.0 if step < len(solve_times) else float('nan')
                status = solve_statuses[step] if step < len(solve_statuses) else 'unknown'
                debug = solve_debug[step] if step < len(solve_debug) else None
                t_sim = timestamps[step] if step < len(timestamps) else float(step)

                # 提取关键诊断信息片段（仅包含最重要的cost项）
                diag_excerpt = ""
                if debug:
                    step0 = debug.get('step0', {})
                    important = []
                    for key in ('cost_track_vomega', 'cost_contour', 'cost_lag', 'cost_progress', 'cost_cvar'):
                        if key in step0:
                            important.append(f"{key}={step0[key]:.3f}")
                    if 'horizon' in debug and 'cost_cvar' in debug['horizon']:
                        important.append(f"cost_cvar={debug['horizon']['cost_cvar']:.3f}")
                    diag_excerpt = (" " + " ".join(important)) if important else ""

                # 写入紧凑数据行（仅包含最核心的状态和控制变量）
                f.write(
                    f"{step:04d} "
                    f"{t_sim:8.3f} "
                    f"{x_t[3]:9.4f} "
                    f"{ref_t[3]:9.4f} "
                    f"{x_t[4]:9.4f} "
                    f"{ref_t[4]:9.4f} "
                    f"{u_t[0]:9.4f} "
                    f"{u_t[1]:9.4f} "
                    f"{solve_time_ms:9.3f} "
                    f"{status}{diag_excerpt}\n"
                )

    @staticmethod
    def _summarize_debug_diagnostics(result, top_k=5):
        """
        汇总每步的调试诊断信息，提取主导代价项和活跃约束频率。
        该统计信息有助于理解控制器在仿真过程中的行为模式。

        Args:
            result: SimResult实例，包含solve_debug列表
            top_k: 返回的主导代价项数量，默认前5项

        Returns:
            dict or None: 包含统计信息的字典，若无调试数据则返回None
        """
        # 过滤掉空的调试行（某些步可能未返回debug信息）
        debug_rows = [row for row in getattr(result, 'solve_debug', []) if row]
        if not debug_rows:
            return None

        # 初始化累加器字典和计数器
        step_acc = {}       # 用于累加step0级别的标量诊断项
        horizon_acc = {}    # 用于累加horizon级别的标量诊断项
        active_counter = Counter()  # 用于统计各约束的激活频率

        # 遍历所有调试行，累加数值并统计约束
        for row in debug_rows:
            # 累加step0级别的数值项（单步诊断）
            for key, value in row.get('step0', {}).items():
                if isinstance(value, (int, float)):
                    step_acc.setdefault(key, []).append(float(value))
            # 累加horizon级别的数值项（时域级诊断）
            for key, value in row.get('horizon', {}).items():
                if isinstance(value, (int, float)):
                    horizon_acc.setdefault(key, []).append(float(value))
            # 统计本步的活跃约束
            active_counter.update(row.get('active_constraints', []))

        # 局部辅助函数: 计算均值和最大值的统计字典
        def build_stats(acc):
            stats = {}
            for key, values in acc.items():
                if values:
                    stats[key] = {
                        'mean': float(np.mean(values)),  # 该诊断项在所有步上的平均值
                        'max': float(np.max(values)),    # 该诊断项在所有步上的最大值
                    }
            return stats

        step_stats = build_stats(step_acc)
        horizon_stats = build_stats(horizon_acc)
        # 找出平均代价最大的top_k个代价项（用于判断哪个代价项主导了优化行为）
        dominant = sorted(
            [
                (key, vals['mean'])
                for key, vals in step_stats.items()
                if key.startswith('cost_')
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        return {
            'step_stats': step_stats,          # step0级别诊断统计
            'horizon_stats': horizon_stats,    # horizon级别诊断统计
            'dominant_costs': dominant,        # 主导代价项列表（按均值降序）
            'active_constraints': dict(active_counter.most_common()),  # 活跃约束频率
        }

    @staticmethod
    def _export_result_debug_summary(result, output_path, top_k=5):
        """
        将调试诊断摘要追加到现有日志文件末尾。
        该摘要包含主导代价项、活跃约束频率和时域诊断统计。

        Args:
            result: SimResult实例
            output_path: 要追加摘要的日志文件路径
            top_k: 主导代价项数量

        Returns:
            bool: 是否成功写入摘要
        """
        summary = Simulator._summarize_debug_diagnostics(result, top_k=top_k)
        if summary is None:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'a') as f:
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("DEBUG SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"method={result.method_name}\n")
            f.write(f"track={result.track_name}\n")
            f.write(f"total_steps={result.total_steps}\n\n")
            f.write(f"crashed={result.crashed}\n")
            f.write(f"crash_step={result.crash_step}\n")
            f.write(f"crash_time={result.crash_time}\n")
            f.write(f"crash_reason={result.crash_reason}\n\n")

            # 写入主导代价项（帮助分析控制器优化重点）
            f.write("[Top Dominant Cost Terms]\n")
            for key, mean_val in summary['dominant_costs']:
                max_val = summary['step_stats'][key]['max']
                f.write(f"  {key}: mean={mean_val:.6f}, max={max_val:.6f}\n")

            # 写入活跃约束频率（帮助分析哪些约束经常起作用）
            f.write("\n[Active Constraints Frequency]\n")
            if summary['active_constraints']:
                for key, count in summary['active_constraints'].items():
                    f.write(f"  {key}: {count}\n")
            else:
                f.write("  none\n")

            # 写入时域级诊断统计
            f.write("\n[Horizon Diagnostics]\n")
            for key, vals in sorted(summary['horizon_stats'].items()):
                f.write(f"  {key}: mean={vals['mean']:.6f}, max={vals['max']:.6f}\n")

        return True

    @staticmethod
    def _build_track_for_result(result):
        """
        根据仿真结果中保存的赛道名称，重建对应的赛道实例。
        该方法用于结果后处理阶段（绘图、动画），此时需要从.pkl中恢复赛道几何。

        Args:
            result: SimResult实例，包含track_name属性

        Returns:
            BaseTrack子类实例，或None（若赛道名称未识别）
        """
        track_name = result.track_name
        if track_name == 'LusailShortTrack':
            from tracks.lusail_short_track import LusailShortTrack
            return LusailShortTrack()
        if track_name == 'LusailTrack':
            from tracks.lusail_track import LusailTrack
            return LusailTrack()
        if track_name == 'CustomWindingTrack':
            from tracks.custom_track import CustomWindingTrack
            return CustomWindingTrack()
        if track_name == 'SprintOvalTrack':
            from tracks.sprint_oval_track import SprintOvalTrack
            return SprintOvalTrack()
        return None

    @staticmethod
    def _export_result_figures(result, base_name):
        """
        将单个仿真结果导出为PDF图表（轨迹图、状态对比图、控制对比图）。

        Args:
            result: SimResult实例
            base_name: 输出文件名的基础前缀（不含扩展名）

        Returns:
            list: 生成的PDF文件路径列表
        """
        # 根据结果中的赛道名重建赛道对象（绘图需要赛道几何作为背景）
        track = Simulator._build_track_for_result(result)
        if track is None:
            return []

        # 延迟导入绘图模块（避免循环导入和启动开销）
        from visualization.plot_trajectories import (
            plot_trajectory_comparison,
            plot_state_comparison,
            plot_control_comparison,
        )

        # 包装为字典格式（绘图函数支持多方法对比，但此处仅有一种方法）
        results = {result.method_name: result}
        trajectory_name = f"{base_name}_trajectory.pdf"
        states_name = f"{base_name}_states.pdf"
        controls_name = f"{base_name}_controls.pdf"

        # 生成轨迹对比图（车辆路径叠加在赛道上）
        plot_trajectory_comparison(
            results,
            track,
            title=f"{result.method_name} Trajectory on {result.track_name}",
            filename=trajectory_name,
        )
        # 生成状态随时间变化图
        plot_state_comparison(
            results,
            track,
            filename=states_name,
        )
        # 生成控制输入随时间变化图
        plot_control_comparison(
            results,
            filename=controls_name,
        )

        # 返回生成的PDF文件完整路径列表
        return [
            os.path.join(FIGURES_DIR, trajectory_name),
            os.path.join(FIGURES_DIR, states_name),
            os.path.join(FIGURES_DIR, controls_name),
        ]

    @staticmethod
    @staticmethod
    def _export_result_animation(result_path, base_name):
        """
        调用外部动画生成脚本，将仿真结果导出为GIF动画。

        Args:
            result_path: 仿真结果.pkl文件的完整路径
            base_name: 输出文件名的基础前缀

        Returns:
            str or None: 生成的GIF文件路径，若失败则返回None
        """
        output_path = os.path.join(FIGURES_DIR, f"{base_name}_animation.gif")
        # 构建动画脚本的路径（位于visualization/animate_simulation.py）
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'visualization',
            'animate_simulation.py',
        )

        # 构建子进程命令行参数
        cmd = [
            sys.executable,          # 当前Python解释器路径
            script_path,             # 动画生成脚本
            '--result', result_path, # 输入结果文件
            '--fps', str(int(ANIMATION_FPS)),  # 动画帧率
            '--fast-gif',            # 使用快速GIF导出设置
            '--save', output_path,   # 输出GIF路径
        ]
        # 执行子进程并捕获输出
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[Warn] 动画导出失败: {proc.stderr.strip() or proc.stdout.strip()}")
            return None
        return output_path

    @staticmethod
    def save_result(result, filename=None, save_dir=RESULTS_DIR):
        """
        将仿真结果持久化保存到磁盘，并触发关联的日志、图表和动画导出。
        这是仿真完成后调用的主入口，一站式完成所有结果归档工作。

        Args:
            result: SimResult实例，包含完整的仿真轨迹和数据
            filename: 保存文件名（可选），默认为"{method_name}_{track_name}.pkl"
            save_dir: 保存目录，默认为config中的RESULTS_DIR
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            # 自动生成文件名: 例如 "K-DRMPC_SprintOvalTrack.pkl"
            filename = f"{result.method_name}_{result.track_name}.pkl"
        path = os.path.join(save_dir, filename)
        # 使用pickle将SimResult对象序列化保存（保留完整数据结构）
        with open(path, 'wb') as f:
            pickle.dump(result, f)

        # 基于文件名（不含扩展名）生成各类输出的基础名称
        base_name = os.path.splitext(os.path.basename(path))[0]
        # 日志文件保存到results/logs/子目录
        log_dir = os.path.join(save_dir, 'logs')
        step_log_path = os.path.join(log_dir, f"{base_name}.log")

        # 导出详细步进日志
        Simulator._export_result_to_step_log(result, step_log_path)
        # 追加调试诊断摘要
        Simulator._export_result_debug_summary(result, step_log_path)
        # 根据配置决定是否导出静态PDF图表
        figure_paths = Simulator._export_result_figures(result, base_name) if EXPORT_STATIC_FIGURES else []
        # 根据配置决定是否导出GIF动画
        animation_path = Simulator._export_result_animation(path, base_name) if EXPORT_ANIMATION else None

        # 在终端输出所有生成文件的路径，方便用户查看
        print(f"结果已保存到 {path}")
        print(f"报表链接: {step_log_path}")
        for figure_path in figure_paths:
            print(f"图片链接: {figure_path}")
        if animation_path:
            print(f"动画链接: {animation_path}")

    @staticmethod
    def load_result(filepath):
        """
        从磁盘加载之前保存的仿真结果（.pkl文件）。

        Args:
            filepath: 仿真结果文件的完整路径

        Returns:
            SimResult: 反序列化后的仿真结果对象
        """
        with open(filepath, 'rb') as f:
            # 使用pickle反序列化，恢复SimResult对象的完整状态
            return pickle.load(f)
