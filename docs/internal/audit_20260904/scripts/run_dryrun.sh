#!/bin/bash
# presweep dry-run 0.1/0.01（方案 D 代码验证，隔离目录 + REPLAY_ACCUM=1 + layout-map）
set -uo pipefail
export PATH=/root/data/coin/conda_envs/coin/bin:$PATH
cd /root/data/coin/project
export REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1 DRY_RUN=1 GPUS=0,1,2,3
export PREFLIGHT_ARGS='--skip-pil --layout-map {"ImageNet":"ImageNet_withlabel"}'
mkdir -p /root/data/coin/tmp/dryrun_20260904
for r in 0.1 0.01; do
  echo "===== DRY_RUN ratio=$r ====="
  export CKPT_ROOT=/root/data/coin/tmp/dryrun_20260904/ckpt_$r
  export RES_ROOT=/root/data/coin/tmp/dryrun_20260904/res_$r
  export REPLAY_DATA_DIR=/root/data/coin/tmp/dryrun_20260904/replay_$r
  export PREFLIGHT_REPORT=/root/data/coin/tmp/dryrun_20260904/preflight_$r.json
  rm -rf "$CKPT_ROOT" "$RES_ROOT" "$REPLAY_DATA_DIR"
  bash scripts/CoIN_Replay/run_replay_exp.sh "$r" > /root/data/coin/logs/canary_20260904/dryrun_${r}.log 2>&1
  echo "RC_$r=$?"
done
echo "DRYRUN_ALL_DONE"
