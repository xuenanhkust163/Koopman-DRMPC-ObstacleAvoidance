#!/usr/bin/env bash
# ----------------------------------------------------------------------
# 多 seed × 多 Setting 批量实验脚本
#   - Setting A (Nominal, x1.0): sigma = 0.5
#   - Setting B (Moderate, x1.5): sigma = 0.75
#   - Setting C (Large,    x2.0): sigma = 1.0
#   每个 Setting 跑 5 个随机种子，共 15 次实验
#
# 用法：
#   chmod +x run_all_seeds.sh
#   ./run_all_seeds.sh              # 正常跑（已存在的结果自动跳过）
#   ./run_all_seeds.sh --force      # 强制重跑所有（覆盖现有结果）
#
# 断点续跑：检测 _output/results/noise_comparison_summary_<suffix>.json 是否存在
# ----------------------------------------------------------------------

set -e  # 遇到错误立即退出

# ------- 参数 -------
SEEDS=(42 123 2024)
SIGMAS=(0.5 0.75 1.0)
SETTINGS=(A B C)
THETA=0.10
EPSILON=0.10
STEPS=3500

RESULTS_DIR="_output/results"
FORCE=0
if [[ "$1" == "--force" ]]; then
  FORCE=1
  echo "[WARN] --force 已启用：将覆盖已存在的结果"
fi

# ------- 主循环 -------
total=$(( ${#SIGMAS[@]} * ${#SEEDS[@]} ))
cnt=0
skipped=0
start_time=$(date +%s)

for i in "${!SIGMAS[@]}"; do
  sigma="${SIGMAS[$i]}"
  setting="${SETTINGS[$i]}"

  for seed in "${SEEDS[@]}"; do
    cnt=$((cnt + 1))
    suffix="_Setting${setting}_seed${seed}"
    json_file="${RESULTS_DIR}/noise_comparison_summary${suffix}.json"

    echo ""
    echo "============================================================"
    echo "[${cnt}/${total}] Setting ${setting} (sigma=${sigma}) seed=${seed}"
    echo "  -> suffix: ${suffix}"
    echo "============================================================"

    # 断点续跑：已存在则跳过
    if [[ $FORCE -eq 0 && -f "$json_file" ]]; then
      echo "  [SKIP] 已存在 $json_file（用 --force 强制重跑）"
      skipped=$((skipped + 1))
      continue
    fi

    # 实际执行
    python run_noise_comparison.py \
      --sigma "$sigma" \
      --theta "$THETA" \
      --epsilon "$EPSILON" \
      --seed "$seed" \
      --steps "$STEPS" \
      --name-suffix "$suffix"

    elapsed=$(( $(date +%s) - start_time ))
    echo "  [OK] 累计耗时: $((elapsed / 60)) 分钟"
  done
done

elapsed=$(( $(date +%s) - start_time ))
echo ""
echo "============================================================"
echo "全部完成: 共 ${total} 次实验, 跳过 ${skipped} 次, 实际用时 $((elapsed / 60)) 分钟"
echo "============================================================"
echo "下一步: python aggregate_seeds.py"
