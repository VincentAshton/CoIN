#!/bin/bash
# 阶段 II 门禁复现：single-step replay（0.01 round2 N=127 / round3 N=473 场景）。
# 配置与正式 run_replay_exp 完全一致（4×A100 / batch14×4×accum16 / bf16+tf32 /
# LoRA r192·a256 / lr2e-4 / mm 2e-5 / cosine / warmup 0.03 / 1 epoch / zero3_offload /
# grad_ckpt / seed1234 / max_len 2048）。
# 用法（coin env 内，从仓库根）:
#   bash scripts/CoIN_Replay/tools/single_step_gate.sh <gate_dir> <log_dir>
# 环境变量: REPLAY_ACCUM（默认 16；方案 D 验证用 REPLAY_ACCUM=1）
# 2026-09-03 实测结论：N<896（1 有效 batch）时 DS engine（accum16 "auto"）0 次真实更新，
# HF global_step=1 为假象——见 EXPERIMENT_LOG 4.15 / SINGLE_STEP_GATE_REPORT。
# 2026-09-04 方案 D：REPLAY_ACCUM=1 时每 micro 即 boundary，无 0 真步场景。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="${1:-/root/data/coin/tmp/single_step_gate}"
LDIR="${2:-/root/data/coin/logs/presweep/single_step_replay}"
REPLAY_ACCUM="${REPLAY_ACCUM:-16}"   # 方案 D 验证用 1
PY="$(command -v python3)"
DS="$(command -v deepspeed)"
COIN="$PY $ROOT/scripts/CoIN_Replay/coin_lib.py"
mkdir -p "$GATE/data" "$GATE/ckpt" "$LDIR"

echo "===== [gate] 阶段 II single-step replay 门禁 $(date '+%F %T') ====="

echo "--- build replay r2 (N=127 期望) ---"
"$PY" "$ROOT/scripts/CoIN_Replay/build_replay_data.py" --tasks ScienceQA \
  --data-dir "$ROOT/playground/Instructions_Original" --image-dir "$ROOT/cl_dataset" \
  --round 2 --ratio 0.01 --seed 1234 \
  --out "$GATE/data/r2.json" || exit 90
echo "--- build replay r3 (N=473 期望) ---"
"$PY" "$ROOT/scripts/CoIN_Replay/build_replay_data.py" --tasks ScienceQA TextVQA \
  --data-dir "$ROOT/playground/Instructions_Original" --image-dir "$ROOT/cl_dataset" \
  --round 3 --ratio 0.01 --seed 1234 \
  --out "$GATE/data/r3.json" || exit 91

for spec in "r2:$GATE/data/r2.json" "r3:$GATE/data/r3.json"; do
  n="${spec%%:*}"; j="${spec#*:}"
  echo "--- plan $n (accum=$REPLAY_ACCUM) ---"
  $COIN train-plan --data-json "$j" --batch 14 --accum "$REPLAY_ACCUM" --world 4 --lr 2e-4 \
    --warmup-ratio 0.03 --epochs 1 --name "replay_$n" | "$PY" -m json.tool
done

TRAIN_ARGS=(
  --deepspeed "$ROOT/scripts/zero3_offload.json"
  --lora_enable True --lora_r 192 --lora_alpha 256 --lora_dropout 0.05
  --mm_projector_lr 2e-5
  --model_name_or_path "$ROOT/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5"
  --pretrain_mm_mlp_adapter "$ROOT/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin"
  --version v1
  --image_folder "$ROOT/cl_dataset"
  --vision_tower "$ROOT/checkpoints/LLaVA/clip-vit-large-patch14-336"
  --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2
  --mm_use_im_start_end False --mm_use_im_patch_token False
  --image_aspect_ratio pad --group_by_modality_length True
  --bf16 True --tf32 True
  --per_device_train_batch_size 14 --per_device_eval_batch_size 16
  --evaluation_strategy no --save_strategy epoch
  --num_train_epochs 1
  --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03
  --lr_scheduler_type cosine --logging_steps 1
  --seed 1234 --data_seed 1234
  --model_max_length 2048
  --gradient_checkpointing True
  --dataloader_num_workers 4 --lazy_preprocess True
  --report_to none
)

train() {  # name data out [prev] [gas]
  local name="$1" data="$2" out="$3" prev="${4:-}" gas="${5:-16}"
  local args=("${TRAIN_ARGS[@]}" --gradient_accumulation_steps "$gas" --data_path "$data" --output_dir "$out")
  [[ -n "$prev" ]] && args+=(--previous_task_model_path "$prev")
  echo "===== [gate] train $name $(date '+%F %T') prev=${prev:-none} accum=$gas ====="
  $DS --include localhost:0,1,2,3 --master_port 29600 \
    "$ROOT/ETrain/Train/LLaVA/train_mem.py" "${args[@]}" 2>&1 | tee "$LDIR/train_${name}.log"
  local rc=${PIPESTATUS[0]}
  echo "[gate] train $name rc=$rc"
  return $rc
}

# task 段与正式配置一致（accum16）；replay 段用 REPLAY_ACCUM（方案 D 验证=1）
train task_sqa "$ROOT/playground/Instructions_Original/ScienceQA/train.json" "$GATE/ckpt/task_sqa" "" 16 || exit 92
train replay_r2 "$GATE/data/r2.json" "$GATE/ckpt/replay_r2" "$GATE/ckpt/task_sqa" "$REPLAY_ACCUM" || exit 93
train replay_r3 "$GATE/data/r3.json" "$GATE/ckpt/replay_r3" "$GATE/ckpt/task_sqa" "$REPLAY_ACCUM" || exit 94

echo "===== [gate] 训练完成，收集校验证据 ====="
"$PY" "$HERE/single_step_summary.py" "$GATE" "$LDIR"
exit $?
