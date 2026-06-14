#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多 seed 实验结果聚合脚本。

扫描 _output/results/noise_comparison_summary_Setting*_seed*.json
按 (Setting, Method) 分组，计算 mean ± std 和 crash_rate。

输出：
    1. 控制台表格（论文可直接引用）
    2. _output/results/multi_seed_aggregated.json（机器可读）
    3. _output/results/multi_seed_table.csv（Excel 可读）

用法：
    python aggregate_seeds.py
    python aggregate_seeds.py --pattern 'Setting*_seed*'  # 自定义 suffix 模式
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "_output", "results")

# 聚合指标列表（key -> 显示名, 单位, 小数位, 方向: 越大越好=+1 / 越小越好=-1 / N/A=0）
METRICS = [
    ('lap_time',      'Lap Time',      's',   2, +1),   # 越大说明更慢但这里只在 lap_completed=True 时才有效，方向给 -1（越小越好）
    ('steps',         'Steps',         '',    0,  0),
    ('ey_mean',       '|e_y| Mean',    'm',   3, -1),
    ('ey_p95',        '|e_y| P95',     'm',   3, -1),
    ('ey_max',        '|e_y| Max',     'm',   3, -1),
    ('epsi_mean_deg', '|e_ψ| Mean',    '°',   2, -1),
    ('epsi_p95_deg', '|e_ψ| P95',      '°',   2, -1),
    ('v_mean',        'v Mean',        'm/s', 3, +1),
    ('solve_mean_ms', 'Solve Mean',    'ms',  1, -1),
    ('solve_p95_ms',  'Solve P95',     'ms',  1, -1),
    ('within_2m_pct', '±2m Coverage',  '%',   1, +1),
]
# lap_time 方向实际应为 -1（越小越好），上表已矫正（行顺序对，最后修正如下）
METRICS[0] = ('lap_time', 'Lap Time', 's', 2, -1)


def parse_suffix(suffix):
    """从 suffix 提取 setting 和 seed。例如 '_SettingA_seed42' -> ('A', 42)"""
    m = re.match(r'_Setting([A-Za-z])_seed(\d+)', suffix)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def aggregate_values(values):
    """对一列数值做聚合：返回 dict(mean/std/count/raw)。None 值会被剔除。"""
    valid = [v for v in values if v is not None]
    if not valid:
        return {'mean': None, 'std': None, 'count': 0, 'raw': values}
    arr = np.asarray(valid, dtype=float)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'count': len(valid),
        'raw': values,
    }


def format_mean_std(agg, decimals, unit):
    """格式化 'mean ± std'。"""
    if agg['mean'] is None:
        return 'N/A'
    if agg['count'] == 1:
        return f"{agg['mean']:.{decimals}f}"
    return f"{agg['mean']:.{decimals}f} ± {agg['std']:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description='多 seed 实验聚合')
    parser.add_argument('--pattern', type=str, default='Setting*_seed*',
                        help='suffix 模式（glob 形式，不含 noise_comparison_summary_ 前缀）')
    args = parser.parse_args()

    glob_pattern = os.path.join(RESULTS_DIR,
                                f'noise_comparison_summary_{args.pattern}.json')
    files = sorted(glob.glob(glob_pattern))
    if not files:
        print(f"[ERROR] 未找到任何匹配文件: {glob_pattern}")
        print(f"        请先运行 ./run_all_seeds.sh")
        return

    print(f"扫描到 {len(files)} 个结果文件")
    print()

    # ---- 按 (setting, method) 分组收集 ----
    # grouped[(setting, 'robust')] = list of (seed, metrics_dict)
    grouped = defaultdict(list)

    for fp in files:
        fname = os.path.basename(fp)
        # 移除前缀 'noise_comparison_summary' 和扩展名 '.json'
        suffix = fname[len('noise_comparison_summary'):-len('.json')]
        setting, seed = parse_suffix(suffix)
        if setting is None:
            print(f"  [WARN] 跳过无法解析的文件: {fname}")
            continue

        with open(fp, 'r') as f:
            data = json.load(f)

        for method in ('robust', 'nonrobust'):
            if method in data and data[method]:
                grouped[(setting, method)].append((seed, data[method]))

    if not grouped:
        print("[ERROR] 未解析到任何 (setting, method) 组合")
        return

    # ---- 聚合每组 ----
    # aggregated[(setting, method)][metric_key] = aggregate_values dict
    aggregated = {}
    for (setting, method), entries in grouped.items():
        metric_cols = defaultdict(list)
        seeds_seen = []
        for seed, md in entries:
            seeds_seen.append(seed)
            for key, _, _, _, _ in METRICS:
                metric_cols[key].append(md.get(key))
            # crash & lap_completed 也收集
            metric_cols['crashed'].append(1 if md.get('crashed') else 0)
            metric_cols['lap_completed'].append(1 if md.get('lap_completed') else 0)

        agg = {}
        for key, _, _, _, _ in METRICS:
            agg[key] = aggregate_values(metric_cols[key])
        # crash_rate: 撞车次数 / 总次数
        n_total = len(entries)
        n_crash = sum(metric_cols['crashed'])
        n_lap_ok = sum(metric_cols['lap_completed'])
        agg['_meta'] = {
            'n_seeds': n_total,
            'seeds': sorted(seeds_seen),
            'crash_count': n_crash,
            'crash_rate': n_crash / n_total if n_total else 0.0,
            'lap_completed_count': n_lap_ok,
            'lap_completed_rate': n_lap_ok / n_total if n_total else 0.0,
        }
        aggregated[(setting, method)] = agg

    # ---- 打印表格 ----
    print("=" * 100)
    print("多 Seed 聚合结果 (mean ± std)")
    print("=" * 100)

    settings = sorted({s for (s, _) in aggregated.keys()})
    methods = ['robust', 'nonrobust']
    method_names = {'robust': 'Robust K-DRMPC', 'nonrobust': 'Non-Robust'}

    # 每行一个 (setting, method)
    header = f"{'Setting':<8} {'Method':<18} {'Seeds':<8} {'Crash':<10} {'LapOK':<10} "
    for _, disp, unit, _, _ in METRICS:
        head = f"{disp}({unit})" if unit else disp
        header += f"{head:<20} "
    print(header)
    print("-" * len(header))

    for setting in settings:
        for method in methods:
            if (setting, method) not in aggregated:
                continue
            agg = aggregated[(setting, method)]
            meta = agg['_meta']
            crash_str = f"{meta['crash_count']}/{meta['n_seeds']}"
            lap_str = f"{meta['lap_completed_count']}/{meta['n_seeds']}"
            row = f"{setting:<8} {method_names[method]:<18} {meta['n_seeds']:<8} {crash_str:<10} {lap_str:<10} "
            for key, _, unit, dec, _ in METRICS:
                row += f"{format_mean_std(agg[key], dec, unit):<20} "
            print(row)
        print()  # setting 之间空一行

    # ---- 保存 JSON ----
    out_json = {
        str((s, m)): {
            k: v for k, v in agg.items()
        }
        for (s, m), agg in aggregated.items()
    }
    json_path = os.path.join(RESULTS_DIR, 'multi_seed_aggregated.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON 已保存: {json_path}")

    # ---- 保存 CSV ----
    csv_path = os.path.join(RESULTS_DIR, 'multi_seed_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # header
        header_row = ['Setting', 'Method', 'n_seeds', 'crash_rate', 'lap_ok_rate']
        for key, disp, unit, _, _ in METRICS:
            header_row.append(f"{disp}_mean" + (f"_{unit}" if unit else ""))
            header_row.append(f"{disp}_std"  + (f"_{unit}" if unit else ""))
        writer.writerow(header_row)
        # body
        for setting in settings:
            for method in methods:
                if (setting, method) not in aggregated:
                    continue
                agg = aggregated[(setting, method)]
                meta = agg['_meta']
                row = [setting, method_names[method],
                       meta['n_seeds'],
                       f"{meta['crash_rate']:.3f}",
                       f"{meta['lap_completed_rate']:.3f}"]
                for key, _, _, _, _ in METRICS:
                    m_ = agg[key]['mean']
                    s_ = agg[key]['std']
                    row.append('' if m_ is None else f"{m_:.4f}")
                    row.append('' if s_ is None else f"{s_:.4f}")
                writer.writerow(row)
    print(f"✓ CSV 已保存: {csv_path}")

    # ---- 核心结论自动生成 ----
    print()
    print("=" * 100)
    print("核心观察")
    print("=" * 100)
    for setting in settings:
        print(f"\nSetting {setting}:")
        for method in methods:
            if (setting, method) not in aggregated:
                continue
            agg = aggregated[(setting, method)]
            meta = agg['_meta']
            lap_time_agg = agg['lap_time']
            ey_max_agg = agg['ey_max']
            lap_str = f"{meta['lap_completed_count']}/{meta['n_seeds']}"
            crash_str = f"{meta['crash_count']}/{meta['n_seeds']}"
            lt = format_mean_std(lap_time_agg, 2, 's') if lap_time_agg['mean'] is not None else 'N/A'
            ey = format_mean_std(ey_max_agg, 3, 'm')
            print(f"  {method_names[method]:<18} | LapOK: {lap_str} | Crash: {crash_str} | LapTime: {lt} | MaxErr: {ey}")


if __name__ == '__main__':
    main()
