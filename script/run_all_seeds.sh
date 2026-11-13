#!/usr/bin/env bash
# ----------------------------------------------------------------------
# Multi-seed x Multi-setting Batch Experiment Script
#   - Setting A (Nominal, x1.0): sigma = 0.5
#   - Setting B (Moderate, x1.5): sigma = 0.75
#   - Setting C (Large,    x2.0): sigma = 1.0
#   A total of 15 experiments with 5 random seeds for each setting
#
# Usage：
#   chmod +x run_all_seeds.sh
#   ./run_all_seeds.sh              # Run normally (existing results will be skipped automatically)
#   ./run_all_seeds.sh --force      # Force rerun all (overwrite existing results)
#
# Breakpoint Resume: Check if _output/results/noise_comparison_summary_<suffix>.json exists
# ----------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ------- Parameters -------
SEEDS=(42 123 2024)
SIGMAS=(0.5 0.75 1.0)
SETTINGS=(A B C)
THETA=0.10
EPSILON=0.10
STEPS=3500

RESULTS_DIR="$PROJECT_ROOT/_output/results"
FORCE=0
if [[ "$1" == "--force" ]]; then
  FORCE=1
  echo "[WARN] --force 已启用：将覆盖已存在的结果"
fi

# ------- Main Loop -------
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

    # Breakpoint Resume: skip if exists
    if [[ $FORCE -eq 0 && -f "$json_file" ]]; then
      echo "  [SKIP] 已存在 $json_file（用 --force 强制重跑）"
      skipped=$((skipped + 1))
      continue
    fi

    # Actual execution
    python "$SCRIPT_DIR/run_noise_comparison.py" \
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
echo "下一步: python $SCRIPT_DIR/aggregate_seeds.py"
