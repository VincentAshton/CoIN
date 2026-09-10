#!/bin/bash
# 四任务 preflight（2026-09-04 方案 D 审计重跑；layout-map 引号在脚本内安全）
set -uo pipefail
export PATH=/root/data/coin/conda_envs/coin/bin:$PATH
cd /root/data/coin/project
python3 scripts/CoIN_Replay/preflight_data.py \
  --data-dir playground/Instructions_Original \
  --image-dir cl_dataset \
  --out-report results/CoIN_Replay/preflight_report.json \
  --tasks ScienceQA TextVQA ImageNet GQA \
  --layout-map '{"ImageNet":"ImageNet_withlabel"}' 2>&1 | tee /root/data/coin/logs/canary_20260904/preflight_4task.log
rc=${PIPESTATUS[0]}
echo "PREFLIGHT_RC=$rc"
exit $rc
