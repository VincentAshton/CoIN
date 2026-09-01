#!/bin/bash
# ============================================================================
# CoIN + TRACE 式 Replay 实验编排（4 任务 × 单比例，Truth Alignment）
#
# 设计（与 TRACE 实验口径对齐）：
#   对 round j (1..4)：
#     1) 顺序微调任务 j：LoRA(r=192/alpha=256) 续接上一轮 checkpoint，1 epoch
#     2) 若 j>=2：构建 replay 数据（前 j-1 任务按 ratio 抽样）+ replay 训练 1 epoch
#        （同 LR，checkpoint 写回本轮的输出目录，作为下一轮的 previous_task_model_path）
#     3) 评估任务 1..j（Truth Alignment，温度 0）→ results/CoIN_Replay/ratio_<r>/<Task>/round<j>/
#   全部轮次完成后：aggregate_coin.py 严格聚合 → coin_metrics.json（MAA/BWT）+ .complete
#
# TRACE 教训落实：
#   - fail-fast（set -euo pipefail）；任一环节失败立即退出，不继续烧卡
#   - .complete 是整组完成的权威标志；.round<j>_done 是每轮断点（可续跑）
#   - run_manifest.json 记录配置快照（run ID / 冻结配置 / 环境版本）
#   - 评估产物严格校验（缺失/无效即失败），绝不产出半成品指标
#   - 不自动删除 checkpoint
#
# 用法:
#   bash scripts/CoIN_Replay/run_replay_exp.sh 0.1
#   bash scripts/CoIN_Replay/run_replay_exp.sh 0.01
#
# 常用环境变量（全部有默认值）:
#   GPUS=0,1,2,3   BATCH=14   ACCUM=16（4 卡下保持有效 batch 与论文 8卡×8 一致: 14*4*16=896）
#   LR=2e-4  LORA_R=192  LORA_ALPHA=256  EPOCHS=1  REPLAY_EPOCHS=1
# SEED=1234  SAMPLE_MODE=prefix（与 TRACE 一致：取前序任务数据前缀）
# ============================================================================
set -euo pipefail

RATIO="${1:?用法: run_replay_exp.sh <ratio>，如 0.1 或 0.01}"

# ---- 路径与配置 ------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29600}"

TASKS=(ScienceQA TextVQA ImageNet GQA)
EVAL_SCRIPTS=(1_eval_sqa 2_eval_textqa 3_eval_ImageNet 4_eval_gqa)

DATA_DIR="${DATA_DIR:-$ROOT/playground/Instructions_Original}"
IMG_DIR="${IMG_DIR:-$ROOT/cl_dataset}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/checkpoints/CoIN_Replay/ratio_${RATIO}}"
RES_ROOT="${RES_ROOT:-$ROOT/results/CoIN_Replay/ratio_${RATIO}}"
REPLAY_DATA_DIR="${REPLAY_DATA_DIR:-$ROOT/playground/Replay/ratio_${RATIO}}"

BASE_MODEL="${BASE_MODEL:-$ROOT/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5}"
VISION_TOWER="${VISION_TOWER:-$ROOT/checkpoints/LLaVA/clip-vit-large-patch14-336}"
PROJECTOR="${PROJECTOR:-$ROOT/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin}"
DS_CONFIG="${DS_CONFIG:-$ROOT/scripts/zero3_offload.json}"

# 训练超参（默认与论文 scripts/LLaVA/Train/*.sh 一致，仅 ACCUM 按 4 卡调整）
LORA_R="${LORA_R:-192}"
LORA_ALPHA="${LORA_ALPHA:-256}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
LR="${LR:-2e-4}"
BATCH="${BATCH:-14}"
ACCUM="${ACCUM:-16}"
EPOCHS="${EPOCHS:-1}"
REPLAY_EPOCHS="${REPLAY_EPOCHS:-1}"
SEED="${SEED:-1234}"
SAMPLE_MODE="${SAMPLE_MODE:-prefix}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"

log()  { echo "[$(date '+%F %T')] $*"; }

# ---- 前置检查（fail-fast） -------------------------------------------------
preflight() {
  local missing=0
  for p in "$BASE_MODEL" "$VISION_TOWER" "$PROJECTOR" "$DS_CONFIG"; do
    [[ -e "$p" ]] || { echo "ERROR: 缺路径 $p"; missing=1; }
  done
  for t in "${TASKS[@]}"; do
    [[ -f "$DATA_DIR/$t/train.json" ]] || { echo "ERROR: 缺 $DATA_DIR/$t/train.json"; missing=1; }
    if [[ "$t" == "TextVQA" ]]; then
      [[ -f "$DATA_DIR/$t/val.json" ]] || { echo "ERROR: 缺 $DATA_DIR/$t/val.json"; missing=1; }
    else
      [[ -f "$DATA_DIR/$t/test.json" ]] || { echo "ERROR: 缺 $DATA_DIR/$t/test.json"; missing=1; }
    fi
  done
  for s in "${EVAL_SCRIPTS[@]}"; do
    [[ -f "$ROOT/scripts/LLaVA/Eval/$s.sh" ]] || { echo "ERROR: 缺评估脚本 $s.sh"; missing=1; }
  done
  command -v deepspeed >/dev/null || { echo "ERROR: deepspeed 不在 PATH"; missing=1; }
  (( missing == 0 )) || { echo "前置检查失败，终止"; exit 1; }
  log "前置检查通过 (GPUS=$GPUS ratio=$RATIO)"
}

# ---- 配置快照（run_manifest.json，原子写） ----------------------------------
write_manifest() {
  python3 - "$RES_ROOT" "$RATIO" <<'EOF'
import json, os, subprocess, sys, time

res_root, ratio = sys.argv[1], sys.argv[2]
def ver(pkg):
    try:
        return subprocess.check_output([sys.executable, "-c",
            f"import {pkg}; print({pkg}.__version__)"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"
try:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     stderr=subprocess.DEVNULL, text=True).strip()
except Exception:
    commit = "unknown"
manifest = {
    "run_id": f"coin_replay_r{ratio}_{time.strftime('%Y%m%d_%H%M%S')}",
    "ratio": float(ratio),
    "tasks": ["ScienceQA", "TextVQA", "ImageNet", "GQA"],
    "T": 4,
    "order": "ScienceQA -> TextVQA -> ImageNet -> GQA",
    "model": {"base": os.environ.get("BASE_MODEL"), "vision_tower": os.environ.get("VISION_TOWER")},
    "lora": {"r": int(os.environ.get("LORA_R", 192)), "alpha": int(os.environ.get("LORA_ALPHA", 256))},
    "train": {
        "per_device_batch": int(os.environ.get("BATCH", 14)),
        "grad_accum": int(os.environ.get("ACCUM", 16)),
        "effective_batch": int(os.environ.get("BATCH", 14)) * int(os.environ.get("ACCUM", 16)),
        "lr": float(os.environ.get("LR", 2e-4)),
        "mm_projector_lr": float(os.environ.get("MM_PROJECTOR_LR", 2e-5)),
        "epochs_per_task": int(os.environ.get("EPOCHS", 1)),
        "replay_epochs": int(os.environ.get("REPLAY_EPOCHS", 1)),
        "seed": int(os.environ.get("SEED", 1234)),
        "sample_mode": os.environ.get("SAMPLE_MODE", "random"),
        "ds_config": os.environ.get("DS_CONFIG"),
        "gpus": os.environ.get("GPUS", "0,1,2,3"),
    },
    "env": {"torch": ver("torch"), "transformers": ver("transformers"),
            "deepspeed": ver("deepspeed"), "peft": ver("peft")},
    "git_commit": commit,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
os.makedirs(res_root, exist_ok=True)
tmp = os.path.join(res_root, "run_manifest.json.tmp")
with open(tmp, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
os.replace(tmp, os.path.join(res_root, "run_manifest.json"))
print("manifest written:", os.path.join(res_root, "run_manifest.json"))
EOF
}

# ---- 训练（任务微调 / replay 共用） -----------------------------------------
train_one() {  # $1=name $2=data_path $3=ckpt $4=prev(可空) $5=epochs
  local name="$1" data="$2" ckpt="$3" prev="$4" epochs="$5" logf="$RES_ROOT/logs/${name}.log"
  local args=(
    --deepspeed "$DS_CONFIG"
    --lora_enable True --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --mm_projector_lr "$MM_PROJECTOR_LR"
    --model_name_or_path "$BASE_MODEL"
    --pretrain_mm_mlp_adapter "$PROJECTOR"
    --version "$PROMPT_VERSION"
    --data_path "$data"
    --image_folder "$IMG_DIR"
    --vision_tower "$VISION_TOWER"
    --mm_projector_type mlp2x_gelu --mm_vision_select_layer -2
    --mm_use_im_start_end False --mm_use_im_patch_token False
    --image_aspect_ratio pad --group_by_modality_length True
    --bf16 True --tf32 True
    --output_dir "$ckpt"
    --num_train_epochs "$epochs"
    --per_device_train_batch_size "$BATCH"
    --per_device_eval_batch_size 16
    --gradient_accumulation_steps "$ACCUM"
    --evaluation_strategy no --save_strategy epoch
    --learning_rate "$LR" --weight_decay 0. --warmup_ratio 0.03
    --lr_scheduler_type cosine --logging_steps 1
    --model_max_length "$MODEL_MAX_LENGTH"
    --gradient_checkpointing True
    --dataloader_num_workers 4 --lazy_preprocess True
    --report_to none
  )
  [[ -n "$prev" ]] && args+=(--previous_task_model_path "$prev")
  mkdir -p "$(dirname "$logf")"
  log "[train:$name] 启动 (epochs=$epochs data=$data)"
  deepspeed --include localhost:"$GPUS" --master_port "$MASTER_PORT" \
    "$ROOT/ETrain/Train/LLaVA/train_mem.py" "${args[@]}" 2>&1 | tee -a "$logf"
  log "[train:$name] 结束"
}

# ---- 评估（单任务，Truth Alignment） -----------------------------------------
eval_one() {  # $1=task $2=ckpt $3=stage
  local etask="$1" ckpt="$2" stage="$3" idx
  for i in "${!TASKS[@]}"; do [[ "${TASKS[$i]}" == "$etask" ]] && idx=$((i + 1)); done
  local script="$ROOT/scripts/LLaVA/Eval/${EVAL_SCRIPTS[$((idx - 1))]}.sh"
  local art="$RES_ROOT/$etask/$stage"
  [[ -f "$script" ]] || { echo "ERROR: 缺 $script"; exit 1; }
  log "[eval:$etask@$stage] $script"
  # create_prompt 失败不影响 Truth Alignment（脚本内已做容错），准确性产物为准
  RESULT_DIR="$RES_ROOT" bash "$script" "$stage" "$ckpt" || \
    { echo "WARN: $script 退出码非零，按产物校验判定"; }
  # 严格校验准确性产物
  if [[ "$etask" == "ScienceQA" ]]; then
    python3 -c "import json,sys; d=json.load(open('$art/output_result.jsonl')); assert 'acc' in d" \
      || { echo "ERROR: $etask@$stage 缺 output_result.jsonl 或无效"; exit 1; }
  else
    grep -qE "Accuracy:" "$art/Result.text" \
      || { echo "ERROR: $etask@$stage 缺 Result.text 或无效"; exit 1; }
  fi
  log "[eval:$etask@$stage] 产物校验通过"
}

# ---- 单轮 -------------------------------------------------------------------
run_round() {
  local j="$1"
  local task="${TASKS[$((j - 1))]}"
  local ckpt="$CKPT_ROOT/${task}_llava_lora"
  local marker="$RES_ROOT/.round${j}_done"
  local prev=""
  [[ -f "$marker" ]] && { log "round$j 已完成（$marker 存在），跳过"; return 0; }

  if (( j > 1 )); then
    prev="$CKPT_ROOT/${TASKS[$((j - 2))]}_llava_lora"
    [[ -d "$prev" ]] || { echo "ERROR: 缺上一轮 checkpoint $prev"; exit 1; }
  fi

  # 1) 任务微调
  train_one "round${j}_${task}_task" "$DATA_DIR/$task/train.json" "$ckpt" "$prev" "$EPOCHS"

  # 2) replay 训练（j>=2）
  if (( j > 1 )); then
    local replay_json="$REPLAY_DATA_DIR/round${j}_train.json"
    log "round$j 构建 replay 数据 (ratio=$RATIO, prev=${TASKS[*]:0:$((j-1))})"
    python3 "$ROOT/scripts/CoIN_Replay/build_replay_data.py" \
      --tasks "${TASKS[@]}" --data-dir "$DATA_DIR" --round "$j" \
      --ratio "$RATIO" --seed "$SEED" --mode "$SAMPLE_MODE" \
      --out "$replay_json"
    train_one "round${j}_replay" "$replay_json" "$ckpt" "$ckpt" "$REPLAY_EPOCHS"
  fi

  # 3) 评估任务 1..j
  for (( i = 1; i <= j; i++ )); do
    eval_one "${TASKS[$((i - 1))]}" "$ckpt" "round${j}"
  done

  touch "$marker"
  log "round$j 完成"
}

# ---- 主流程 -----------------------------------------------------------------
main() {
  [[ -f "$RES_ROOT/.complete" ]] && { log "整组已完成（$RES_ROOT/.complete 存在），跳过"; exit 0; }
  preflight
  mkdir -p "$RES_ROOT/logs"
  write_manifest
  for j in 1 2 3 4; do run_round "$j"; done
  log "全部轮次完成，聚合 MAA/BWT"
  python3 "$ROOT/scripts/CoIN_Replay/aggregate_coin.py" \
    --results-dir "$RES_ROOT" --tasks "${TASKS[@]}"
  touch "$RES_ROOT/.complete"
  log "DONE: $RES_ROOT（coin_metrics.json + .complete）"
}

main "$@"
