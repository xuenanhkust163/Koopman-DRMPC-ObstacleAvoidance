"""
仿真结果的性能指标计算模块。
实现了论文表8中的所有性能指标。
"""

import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入操作系统接口模块
import sys  # 导入系统模块，用于路径操作

# 将父目录添加到系统路径，以便导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 从配置文件导入安全距离和车辆半径常量
from config import D_SAFE, VEHICLE_RADIUS, IDX_V, IDX_PSI


def _wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_all_metrics(result, track, obstacles=None, w_samples=None):
    """
    计算仿真结果的所有性能指标。

    Args:
        result: SimResult实例，包含仿真结果数据
        track: BaseTrack实例，表示赛道对象
        obstacles: 障碍物列表，格式为[(ox, oy, radius), ...]
        w_samples: 用于计算CVaR的干扰样本
    Returns:
        metrics: 字典，包含所有计算得到的性能指标
    """
    if obstacles is None:  # 如果没有提供障碍物列表
        obstacles = track.get_obstacles()  # 从赛道对象获取障碍物

    data = result.to_arrays()  # 将仿真结果转换为数组格式
    states = data['states']        # 状态数组，形状为(T+1, 5)，包含5个状态变量
    controls = data['controls']    # 控制输入数组，形状为(T, 2)，包含加速度和转向角
    solve_times = data['solve_times']  # 求解时间数组，形状为(T,)

    positions = states[:, :2]  # 提取位置信息，形状为(T+1, 2)，包含[px, py]
    T = len(controls)  # 获取控制步数

    metrics = {}  # 初始化指标字典，用于存储所有计算结果

    # 圈速时间指标
    metrics['lap_time'] = result.lap_time if result.lap_completed else None  # 如果完成一圈则记录圈速，否则为None
    metrics['lap_completed'] = result.lap_completed  # 记录是否完成一圈
    metrics['total_steps'] = result.total_steps  # 记录总仿真步数

    # 跟踪误差：计算横向偏差的均方根（RMS）
    lat_errors = []  # 初始化横向误差列表
    heading_errors = []  # 初始化航向误差列表
    heading_ref = track.get_heading()
    for t in range(len(positions)):  # 遍历每个时间点的位置
        idx, _, lat_err = track.closest_point(positions[t, 0], positions[t, 1])  # 计算到赛道中心线的横向误差
        lat_errors.append(lat_err)  # 将横向误差添加到列表
        psi_err = _wrap_angle(states[t, IDX_PSI] - heading_ref[idx])
        heading_errors.append(psi_err)
    lat_errors = np.array(lat_errors)  # 转换为NumPy数组
    heading_errors = np.array(heading_errors)  # 转换为NumPy数组
    abs_lat_errors = np.abs(lat_errors)  # 绝对横向误差，用于衡量是否贴近中线
    abs_heading_errors = np.abs(heading_errors)
    metrics['tracking_error_rms'] = np.sqrt(np.mean(lat_errors**2))  # 计算横向误差的均方根
    metrics['tracking_error_max'] = np.max(abs_lat_errors)  # 计算最大绝对横向误差
    metrics['tracking_error_mean_abs'] = np.mean(abs_lat_errors)  # 平均绝对横向误差
    metrics['tracking_error_p95_abs'] = np.percentile(abs_lat_errors, 95)  # 95分位绝对横向误差
    metrics['tracking_within_1m_pct'] = 100.0 * np.mean(abs_lat_errors <= 1.0)  # 落在中线±1m内的比例
    metrics['tracking_within_2m_pct'] = 100.0 * np.mean(abs_lat_errors <= 2.0)  # 落在中线±2m内的比例
    metrics['heading_error_rms_rad'] = np.sqrt(np.mean(heading_errors**2))
    metrics['heading_error_max_abs_rad'] = np.max(abs_heading_errors)
    metrics['heading_error_mean_abs_rad'] = np.mean(abs_heading_errors)
    metrics['heading_error_p95_abs_rad'] = np.percentile(abs_heading_errors, 95)
    metrics['heading_error_mean_abs_deg'] = np.degrees(metrics['heading_error_mean_abs_rad'])
    metrics['heading_error_p95_abs_deg'] = np.degrees(metrics['heading_error_p95_abs_rad'])
    metrics['heading_error_max_abs_deg'] = np.degrees(metrics['heading_error_max_abs_rad'])

    # 统计连续偏离中线（>2m）的最长持续步数
    off_center_mask = abs_lat_errors > 2.0
    max_off_center_run = 0
    current_run = 0
    for is_off_center in off_center_mask:
        if is_off_center:
            current_run += 1
            if current_run > max_off_center_run:
                max_off_center_run = current_run
        else:
            current_run = 0
    metrics['tracking_offcenter_max_steps'] = int(max_off_center_run)

    # 速度指标
    velocities = states[:, IDX_V]  # 提取速度列
    metrics['max_speed'] = np.max(velocities)  # 记录最大速度
    metrics['mean_speed'] = np.mean(velocities)  # 记录平均速度
    metrics['min_speed'] = np.min(velocities)  # 记录最小速度

    # 约束违反：计算障碍物距离小于安全距离的时间步百分比
    n_violations = 0  # 初始化违反约束的步数计数器
    min_obs_distances = []  # 初始化最小障碍物距离列表
    for t in range(len(positions)):  # 遍历每个时间点
        for obs in obstacles:  # 遍历所有障碍物
            ox, oy, r = obs  # 解包障碍物信息：x坐标、y坐标、半径
            d_min = r + VEHICLE_RADIUS + D_SAFE  # 计算最小安全距离（障碍物半径+车辆半径+安全裕度）
            dist = np.sqrt((positions[t, 0] - ox)**2 + (positions[t, 1] - oy)**2)  # 计算车辆到障碍物中心的距离
            min_obs_distances.append(dist - d_min)  # 记录实际距离与安全距离的差值（正数表示安全，负数表示违反）
            if dist < d_min:  # 如果距离小于安全距离
                n_violations += 1  # 违反计数器加1
                break  # 即使多个障碍物违反，也只计数一次

    metrics['constraint_violation_pct'] = 100.0 * n_violations / max(len(positions), 1)  # 计算违反百分比（防止除以0）
    metrics['min_obstacle_clearance'] = min(min_obs_distances) if min_obs_distances else float('inf')  # 记录最小障碍物间隙

    # CVaR安全裕度（如果有干扰数据可用）
    if w_samples is not None:  # 如果提供了干扰样本
        from disturbance.wasserstein import compute_cvar_margin  # 导入CVaR计算函数
        cvar, cvar_per_obs = compute_cvar_margin(
            positions, obstacles, w_samples  # 计算条件风险价值（CVaR）安全裕度
        )
        metrics['cvar_safety_margin'] = cvar  # 存储CVaR安全裕度
    else:
        metrics['cvar_safety_margin'] = None  # 如果数据不足，设为None

    # 计算时间指标
    metrics['solve_time_mean'] = np.mean(solve_times) * 1000  # 平均求解时间（转换为毫秒）
    metrics['solve_time_max'] = np.max(solve_times) * 1000    # 最大求解时间（转换为毫秒）
    metrics['solve_time_std'] = np.std(solve_times) * 1000    # 求解时间标准差（转换为毫秒）
    metrics['real_time_feasible'] = metrics['solve_time_mean'] < 100  # 判断是否满足实时性要求（平均求解时间<100ms）

    # 控制努力度（衡量控制输入的大小）
    if len(controls) > 0:  # 如果有控制输入数据
        metrics['control_effort_a'] = np.sqrt(np.mean(controls[:, 0]**2))  # 加速度的RMS值
        metrics['control_effort_delta'] = np.sqrt(np.mean(controls[:, 1]**2))  # 转向角的RMS值

        # 控制平滑度（衡量控制输入的变化率）
        if len(controls) > 1:  # 如果至少有2个控制步
            du = np.diff(controls, axis=0)  # 计算控制输入的差分（相邻步的变化量）
            metrics['control_smoothness'] = np.sqrt(np.mean(du**2))  # 计算控制变化的RMS值

    return metrics


def format_metrics_table(all_metrics, methods):
    """
    将指标格式化为可打印的表格。

    Args:
        all_metrics: 字典，格式为{方法名: 指标字典}
        methods: 方法名称列表，按指定顺序排列

    Returns:
        table_str: 格式化后的表格字符串
    """
    lines = []  # 初始化行列表，用于构建表格
    sep = "-" * 80  # 创建80个字符的分隔线

    lines.append(sep)  # 添加顶部分隔线
    header = f"{'Metric':<30}"  # 创建表头，第一列为"Metric"，左对齐占30字符
    for m in methods:  # 遍历所有方法名
        header += f"{'  ' + m:>12}"  # 添加方法名到表头，右对齐占12字符，前面加2个空格
    lines.append(header)  # 添加表头行
    lines.append(sep)  # 添加分隔线

    # 定义要显示的指标列表：(键名, 显示标签, 格式化字符串)
    metric_names = [
        ('lap_time', 'Lap Time (s)', '.1f'),  # 圈速时间，保留1位小数
        ('tracking_error_rms', 'Tracking Error RMS (m)', '.2f'),  # 跟踪误差RMS，保留2位小数
        ('tracking_error_p95_abs', 'Tracking Error P95 |e_y| (m)', '.2f'),  # 跟踪误差95分位
        ('tracking_within_2m_pct', 'Tracking Within ±2m (%)', '.1f'),  # 中线±2m覆盖率
        ('heading_error_mean_abs_deg', 'Heading Mean |e_psi| (deg)', '.2f'),
        ('heading_error_p95_abs_deg', 'Heading P95 |e_psi| (deg)', '.2f'),
        ('max_speed', 'Max Speed (m/s)', '.1f'),  # 最大速度，保留1位小数
        ('constraint_violation_pct', 'Constraint Violation (%)', '.1f'),  # 约束违反百分比，保留1位小数
        ('cvar_safety_margin', 'CVaR Safety Margin', '.3f'),  # CVaR安全裕度，保留3位小数
        ('solve_time_mean', 'Solve Time Mean (ms)', '.1f'),  # 平均求解时间，保留1位小数
        ('solve_time_max', 'Solve Time Max (ms)', '.1f'),  # 最大求解时间，保留1位小数
    ]

    for key, label, fmt in metric_names:  # 遍历每个指标
        row = f"{label:<30}"  # 创建行，以指标标签开始，左对齐占30字符
        for m in methods:  # 遍历所有方法
            val = all_metrics.get(m, {}).get(key, None)  # 获取对应方法的指标值，如果不存在则为None
            if val is None:  # 如果值为None
                row += f"{'N/A':>12}"  # 添加"N/A"，右对齐占12字符
            else:
                row += f"{val:>12{fmt}}"  # 添加格式化后的数值，右对齐占12字符
        lines.append(row)  # 添加该行到表格

    lines.append(sep)  # 添加底部分隔线
    return "\n".join(lines)  # 将所有行用换行符连接成完整表格字符串


def format_latex_table(all_metrics, methods, caption="", label=""):
    """
    将指标格式化为LaTeX表格。

    Args:
        all_metrics: 字典，格式为{方法名: 指标字典}
        methods: 方法名称列表
        caption: 表格标题
        label: 表格标签（用于引用）

    Returns:
        latex_str: LaTeX表格代码字符串
    """
    n_methods = len(methods)  # 获取方法数量
    col_spec = "l" + "r" * n_methods  # 创建列格式字符串：第一列左对齐(l)，其余列右对齐(r)

    lines = [  # 初始化LaTeX表格开始部分
        "\\begin{table}[htbp]",  # 开始table环境，[htbp]指定浮动位置
        f"\\caption{{{caption}}}",  # 设置表格标题
        f"\\label{{{label}}}",  # 设置表格标签
        "\\centering",  # 居中对齐
        f"\\begin{{tabular}}{{{col_spec}}}",  # 开始tabular环境，指定列格式
        "\\toprule",  # 添加顶部粗线（需要booktabs宏包）
    ]

    header = "Method"  # 初始化表头，第一列为"Method"
    for m in methods:  # 遍历所有方法名
        header += f" & {m}"  # 添加方法名，用&分隔（LaTeX表格列分隔符）
    header += " \\\\"  # 添加换行符
    lines.append(header)  # 添加表头行
    lines.append("\\midrule")  # 添加中间线（分隔表头和数据）

    # 定义LaTeX表格中要显示的指标列表
    metric_names = [
        ('lap_time', 'Lap Time (s)', '.1f'),  # 圈速时间
        ('tracking_error_rms', 'Tracking Error (m)', '.2f'),  # 跟踪误差
        ('heading_error_mean_abs_deg', 'Heading Mean (deg)', '.2f'),
        ('max_speed', 'Max Speed (m/s)', '.1f'),  # 最大速度
        ('constraint_violation_pct', 'Constraint Viol. (\\%)', '.1f'),  # 约束违反（注意%需要转义）
        ('solve_time_mean', 'Solve Time (ms)', '.1f'),  # 求解时间
    ]

    for key, label_tex, fmt in metric_names:  # 遍历每个指标
        row = label_tex  # 以指标的LaTeX标签开始行
        for m in methods:  # 遍历所有方法
            val = all_metrics.get(m, {}).get(key, None)  # 获取指标值
            if val is None:  # 如果值为None
                row += " & N/A"  # 添加"N/A"
            else:
                row += f" & {val:{fmt}}"  # 添加格式化后的数值
        row += " \\\\"  # 添加换行符
        lines.append(row)  # 添加该行

    lines.extend([  # 添加表格结束部分
        "\\bottomrule",  # 添加底部粗线
        "\\end{tabular}",  # 结束tabular环境
        "\\end{table}",  # 结束table环境
    ])

    return "\n".join(lines)  # 将所有行连接成完整的LaTeX代码字符串
