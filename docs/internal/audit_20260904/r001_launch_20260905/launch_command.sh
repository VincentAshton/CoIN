#!/bin/bash
# CoIN+Replay ratio=0.01 正式 sweep launch wrapper（2026-09-05，用户任务书批准）
# 用途：source env.sh → 锁定 conda env → 唯一命令 run_sweep.sh 0.01 → 记录退出码与状态
set -uo pipefail

export COIN_ROOT=/root/data/coin
# env.sh：持久缓存目录 + CUDA/nvcc PATH + HF_ENDPOINT（不含 secret，见 launch_environment.txt）
source /root/data/coin/env.sh
export PATH=/root/data/coin/conda_envs/coin/bin:$PATH
cd /root/data/coin/project

# 正式环境变量（任务书阶段 2，勿增删）
export REPLAY_ACCUM=1
export ENFORCE_MIN_STEPS=1
export GPUS=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
# 禁止设置 ALLOW_SINGLE_STEP_REPLAY（方案 D 下 0.01 不需要，0.01 全 replay 段 accum=1 真步达标）

LROOT=/root/data/coin/logs/formal/ratio_0.01_20260905_035722
LOG="$LROOT/sweep.log"

echo $$ > "$LROOT/main_pid.txt"
echo "$(date '+%F %T %Z')" > "$LROOT/start_time.txt"
echo "$(git rev-parse HEAD)" > "$LROOT/git_commit.txt"

{
  echo "===== launch $(date '+%F %T') ====="
  echo "HEAD=$(git rev-parse HEAD)"
  echo "cmd: bash scripts/CoIN_Replay/run_sweep.sh 0.01"
  echo "REPLAY_ACCUM=$REPLAY_ACCUM ENFORCE_MIN_STEPS=$ENFORCE_MIN_STEPS GPUS=$GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "PREFLIGHT_ARGS=$PREFLIGHT_ARGS"
  echo "ALLOW_SINGLE_STEP_REPLAY=${ALLOW_SINGLE_STEP_REPLAY:-UNSET}"
} >> "$LOG"

bash scripts/CoIN_Replay/run_sweep.sh 0.01 >> "$LOG" 2>&1
rc=$?

echo "$rc" > "$LROOT/exit_code.txt"
if [ "$rc" -eq 0 ]; then
  echo "STATUS: DONE (rc=0) $(date '+%F %T')" >> "$LROOT/STATUS.md"
else
  echo "STATUS: FAILED (rc=$rc) $(date '+%F %T')" >> "$LROOT/STATUS.md"
fi
echo "wrapper exit rc=$rc"
exit "$rc"
