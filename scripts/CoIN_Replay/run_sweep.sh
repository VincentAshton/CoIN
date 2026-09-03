#!/bin/bash
# CoIN+Replay 全量扫描：顺序跑多个比例（默认 0.1 0.01），fail-fast。
# 用法:
#   bash scripts/CoIN_Replay/run_sweep.sh            # 跑 0.1 0.01
#   bash scripts/CoIN_Replay/run_sweep.sh 0.1 0.01
# 方案 D（2026-09-04）：REPLAY_ACCUM 必须由调用方在外部显式 export（正式值=1），
# 本脚本【禁止】按 ratio 隐式设置不同 REPLAY_ACCUM（ratio 专属 accum 已被评审拒绝，
# 会造成实验混杂）。run_sweep 也不得覆盖已 export 的 REPLAY_ACCUM。
#   export REPLAY_ACCUM=1 && bash scripts/CoIN_Replay/run_sweep.sh 0.1
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
