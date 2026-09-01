#!/bin/bash
# CoIN+Replay 全量扫描：顺序跑多个比例（默认 0.1 0.01），fail-fast。
# 用法:
#   bash scripts/CoIN_Replay/run_sweep.sh            # 跑 0.1 0.01
#   bash scripts/CoIN_Replay/run_sweep.sh 0.1 0.01
set -euo pipefail

RATIOS=("$@")
[[ ${#RATIOS[@]} -gt 0 ]] || RATIOS=(0.1 0.01)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for r in "${RATIOS[@]}"; do
  echo "=================================================================="
  echo "[sweep] 开始 ratio=$r"
  echo "=================================================================="
  bash "$ROOT/scripts/CoIN_Replay/run_replay_exp.sh" "$r"
  echo "[sweep] ratio=$r 完成"
done

echo "[sweep] 全部比例完成"
