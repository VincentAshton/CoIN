#!/bin/bash
# ============================================================================
# CoIN + TRACE 式 Replay 实验编排（加固版）
#
# 目录契约（工单 1，与 aggregate_coin.py / eval 脚本一致）：
#   results/CoIN_Replay/ratio_<r>/<Task>/round<j>/{merge.jsonl, output_result.jsonl|Result.text,...}
# checkpoint 链（工单 4）：
#   checkpoints/CoIN_Replay/ratio_<r>/round<j>_task_llava_lora   （微调后）
#   checkpoints/CoIN_Replay/ratio_<r>/round<j>_replay_llava_lora （replay 后；下一轮只加载这个）
# 完成判定：.complete（权威）；每轮 .round<j>_done + round<j>_manifest.json（跳过前必须 validate_round）
#
# 用法:
#   bash scripts/CoIN_Replay/run_replay_exp.sh 0.1
# 环境变量（全部显式 export 后进入 manifest，不依赖未 export 的默认值）:
#   GPUS=0,1,2,3  BATCH=14  ACCUM=16  LR=2e-4  LORA_R=192  LORA_ALPHA=256
#   EPOCHS=1  REPLAY_EPOCHS=1  SEED=1234  DATA_SEED=1234  SAMPLE_MODE=prefix
#   DS_CONFIG=scripts/zero3_offload.json  ENFORCE_MIN_STEPS=0  DRY_RUN=0
#   TASKS_JSON='["ScienceQA","TextVQA","ImageNet","GQA"]'  PREFLIGHT_ARGS=""
# ============================================================================
set -euo pipefail

RATIO="${1:?用法: run_replay_exp.sh <ratio>，如 0.1 或 0.01}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ---- 显式配置（全部 export，manifest 从这里读） -----------------------------
export GPUS="${GPUS:-0,1,2,3}"
export RATIO
export BASE_MODEL="${BASE_MODEL:-$ROOT/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5}"
export VISION_TOWER="${VISION_TOWER:-$ROOT/checkpoints/LLaVA/clip-vit-large-patch14-336}"
export PROJECTOR="${PROJECTOR:-$ROOT/checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin}"
export DS_CONFIG="${DS_CONFIG:-$ROOT/scripts/zero3_offload.json}"
export LORA_R="${LORA_R:-192}"
export LORA_ALPHA="${LORA_ALPHA:-256}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
export LR="${LR:-2e-4}"
export BATCH="${BATCH:-14}"
export ACCUM="${ACCUM:-16}"
export EPOCHS="${EPOCHS:-1}"
export REPLAY_EPOCHS="${REPLAY_EPOCHS:-1}"
export SEED="${SEED:-1234}"
export DATA_SEED="${DATA_SEED:-1234}"
export SAMPLE_MODE="${SAMPLE_MODE:-prefix}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
export PRECISION="${PRECISION:-bf16+tf32}"
export GRAD_CKPT="${GRAD_CKPT:-true}"
export TASKS_JSON="${TASKS_JSON:-[\"ScienceQA\",\"TextVQA\",\"ImageNet\",\"GQA\"]}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export ENFORCE_MIN_STEPS="${ENFORCE_MIN_STEPS:-0}"
export ALLOW_SINGLE_STEP_REPLAY="${ALLOW_SINGLE_STEP_REPLAY:-0}"
# 方案 D（2026-09-04 批准设计）：replay 段 gradient_accumulation 覆盖值。
# 所有 ratio / round 必须用同一值（正式 sweep 要求显式 export REPLAY_ACCUM=1）。
# 空 = 不覆盖（replay 段继承 ACCUM=16，旧语义；兼容测试/回退路径，正式跑禁止留空）。
export REPLAY_ACCUM="${REPLAY_ACCUM:-}"
export DRY_RUN="${DRY_RUN:-0}"
export PREFLIGHT_ARGS="${PREFLIGHT_ARGS:-}"

DATA_DIR="${DATA_DIR:-$ROOT/playground/Instructions_Original}"
IMG_DIR="${IMG_DIR:-$ROOT/cl_dataset}"
CKPT_ROOT="${CKPT_ROOT:-$ROOT/checkpoints/CoIN_Replay/ratio_${RATIO}}"
RES_ROOT="${RES_ROOT:-$ROOT/results/CoIN_Replay/ratio_${RATIO}}"
REPLAY_DATA_DIR="${REPLAY_DATA_DIR:-$ROOT/playground/Replay/ratio_${RATIO}}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-$ROOT/results/CoIN_Replay/preflight_report.json}"

WORLD=$(awk -F',' '{print NF}' <<< "$GPUS")
mapfile -t TASKS < <(python3 -c "import json,sys; print('\n'.join(json.loads(sys.argv[1])))" "$TASKS_JSON")
T=${#TASKS[@]}

EVAL_SCRIPTS=("1_eval_sqa" "2_eval_textqa" "3_eval_ImageNet" "4_eval_gqa")
# 每个任务的问题文件（verify_predictions 用；与 eval 脚本内硬编码一致）
declare -A TASK_QF
TASK_QF[ScienceQA]="$DATA_DIR/ScienceQA/test.json"
TASK_QF[TextVQA]="$DATA_DIR/TextVQA/val.json"
TASK_QF[ImageNet]="$DATA_DIR/ImageNet/test.json"
TASK_QF[GQA]="$DATA_DIR/GQA/test.json"
declare -A TASK_EVAL_SCRIPT
for i in "${!TASKS[@]}"; do
  case "${TASKS[$i]}" in
    ScienceQA) TASK_EVAL_SCRIPT[${TASKS[$i]}]="1_eval_sqa" ;;
    TextVQA)   TASK_EVAL_SCRIPT[${TASKS[$i]}]="2_eval_textqa" ;;
    ImageNet)  TASK_EVAL_SCRIPT[${TASKS[$i]}]="3_eval_ImageNet" ;;
    GQA)       TASK_EVAL_SCRIPT[${TASKS[$i]}]="4_eval_gqa" ;;
    *) echo "ERROR: 不支持的任务 ${TASKS[$i]}"; exit 1 ;;
  esac
done

COIN_LIB="$ROOT/scripts/CoIN_Replay/coin_lib.py"
log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

py() { python3 "$COIN_LIB" "$@"; }

# ---- 前置检查（工单 7 + 模型/脚本存在性） ----------------------------------
preflight() {
  local missing=0
  for p in "$BASE_MODEL" "$VISION_TOWER" "$PROJECTOR" "$DS_CONFIG"; do
    [[ -e "$p" ]] || { echo "ERROR: 缺路径 $p"; missing=1; }
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    command -v deepspeed >/dev/null || { echo "ERROR: deepspeed 不在 PATH"; missing=1; }
  fi
  [[ -f "$COIN_LIB" ]] || { echo "ERROR: 缺 $COIN_LIB"; missing=1; }
  for s in "${EVAL_SCRIPTS[@]}"; do
    [[ -f "$ROOT/scripts/LLaVA/Eval/$s.sh" ]] || { echo "ERROR: 缺评估脚本 $s.sh"; missing=1; }
  done
  (( missing == 0 )) || die "前置文件检查失败"

  # 数据 preflight（缓存：report 存在且 data_sha256 一致则跳过）
  local quick_sha
  quick_sha=$(python3 - "$DATA_DIR" "$TASKS_JSON" <<'EOF'
import hashlib, json, os, sys
data_dir, tasks = sys.argv[1], json.loads(sys.argv[2])
h = hashlib.sha256()
for t in tasks:
    for f in ("train.json", "val.json" if t == "TextVQA" else "test.json"):
        p = os.path.join(data_dir, t, f)
        if os.path.isfile(p):
            h.update(f"{t}/{f}".encode())
            h.update(hashlib.sha256(open(p, "rb").read()).hexdigest().encode())
print(h.hexdigest())
EOF
)
  local reuse=0
  if [[ -f "$PREFLIGHT_REPORT" ]]; then
    local cached
    cached=$(python3 -c "import json; print(json.load(open('$PREFLIGHT_REPORT')).get('data_sha256',''))")
    [[ "$cached" == "$quick_sha" ]] && reuse=1
  fi
  if [[ "$reuse" == "1" ]]; then
    log "preflight report 缓存命中（data_sha256=$quick_sha），跳过图片重检"
  else
    log "运行数据 preflight（$PREFLIGHT_ARGS）..."
    python3 "$ROOT/scripts/CoIN_Replay/preflight_data.py" \
      --data-dir "$DATA_DIR" --image-dir "$IMG_DIR" \
      --out-report "$PREFLIGHT_REPORT" \
      --tasks "${TASKS[@]}" $PREFLIGHT_ARGS
  fi
  export DATA_SHA256="$quick_sha"
  export DS_CONFIG_SHA256=$(sha256sum "$DS_CONFIG" | awk '{print $1}')
  export MODEL_CONFIG_HASH=$(python3 - "$BASE_MODEL" "$VISION_TOWER" "$PROJECTOR" <<'EOF'
import hashlib, os, sys
h = hashlib.sha256()
for p in sys.argv[1:]:
    c = os.path.join(p, "config.json")
    if os.path.isfile(c):
        h.update(hashlib.sha256(open(c, "rb").read()).hexdigest().encode())
    elif os.path.isfile(p):
        h.update(hashlib.sha256(open(p, "rb").read()).hexdigest().encode())
    else:
        h.update(b"missing")
print(h.hexdigest() if os.path.isfile(os.path.join(sys.argv[1], "config.json")) else "unknown")
EOF
)
  log "前置检查通过 (GPUS=$GPUS ratio=$RATIO tasks=${TASKS[*]} world=$WORLD)"
}

# ---- manifest（工单 6）-----------------------------------------------------
write_manifest() {
  if [[ -f "$RES_ROOT/run_manifest.json" ]]; then
    # 恢复运行：config hash 校验（不一致即退出），不覆盖
    log "run_manifest.json 已存在，执行恢复配置校验..."
    py manifest-resume-check --res-root "$RES_ROOT" --root "$ROOT"
    log "配置 hash 一致，继续恢复运行"
  else
    mkdir -p "$RES_ROOT/logs"
    local out
    out=$(py manifest-write --res-root "$RES_ROOT" --root "$ROOT")
    log "manifest 写入: $out"
  fi
}

# ---- 训练（工单 3/8）-------------------------------------------------------
train_one() {  # $1=name $2=data_path $3=ckpt $4=prev(可空) $5=epochs $6=replay_k(可空) $7=accum_override(可空)
  local name="$1" data="$2" ckpt="$3" prev="$4" epochs="$5" replay_k="${6:-}"
  # 方案 D：replay 段经第 7 参覆盖 accum（task 段不传 → 恒 $ACCUM=16）；空串回退 $ACCUM
  local accum="${7:-$ACCUM}"
  local logf="$RES_ROOT/logs/${name}.log"
  # 训练分辨率报告（工单 8；accum 用覆盖后值，plan 与实参同源防错位）
  local plan
  plan=$(py train-plan --data-json "$data" --batch "$BATCH" --accum "$accum" \
         --world "$WORLD" --lr "$LR" --warmup-ratio "$WARMUP_RATIO" \
         --epochs "$epochs" --name "$name" ${replay_k:+--replay-k "$replay_k"})
  echo "==== 训练分辨率报告 [$name] ===="
  echo "$plan" | python3 -m json.tool
  echo "$plan" > "$RES_ROOT/logs/${name}.plan.json"   # 留痕（round manifest 引用）
  if [[ "$ENFORCE_MIN_STEPS" == "1" ]]; then
    local flags
    flags=$(echo "$plan" | python3 -c "import json,sys; p=json.load(sys.stdin); print(p['flag_replay_single_step'], p['flag_first_lr_zero'], p['flag_warmup_covers_all'])" 2>/dev/null || echo "True False False")
    read -r f1 f2 f3 <<< "$flags"
    if [[ "$f1" == "True" || "$f2" == "True" || "$f3" == "True" ]]; then
      if [[ "$ALLOW_SINGLE_STEP_REPLAY" != "1" ]]; then
        die "ENFORCE_MIN_STEPS=1 且训练计划异常（ds_single_step=$f1 first_lr_zero=$f2 warmup_covers_all=$f3）。如需强制继续请显式 ALLOW_SINGLE_STEP_REPLAY=1"
      fi
      log "WARN: 训练计划异常但 ALLOW_SINGLE_STEP_REPLAY=1，继续"
    fi
  fi

  local args=(
    --deepspeed "$DS_CONFIG"
    --lora_enable True --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT"
    --mm_projector_lr "$MM_PROJECTOR_LR"
    --model_name_or_path "$BASE_MODEL"
    --pretrain_mm_mlp_adapter "$PROJECTOR"
    --version v1
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
    --gradient_accumulation_steps "$accum"
    --evaluation_strategy no --save_strategy epoch
    --learning_rate "$LR" --weight_decay 0. --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" --logging_steps 1
    --seed "$SEED" --data_seed "$DATA_SEED"
    --model_max_length "$MODEL_MAX_LENGTH"
    --gradient_checkpointing True
    --dataloader_num_workers 4 --lazy_preprocess True
    --report_to none
  )
  [[ -n "$prev" ]] && args+=(--previous_task_model_path "$prev")
  mkdir -p "$(dirname "$logf")"
  log "[train:$name] 启动 (epochs=$epochs data=$data ckpt=$ckpt prev=${prev:-none})"
  if [[ "$DRY_RUN" == "1" ]]; then
    mkdir -p "$ckpt"
    for f in config.json adapter_config.json adapter_model.bin non_lora_trainables.bin; do
      printf '{}' > "$ckpt/config.json"
      printf '{}' > "$ckpt/adapter_config.json"
      head -c 64 /dev/urandom > "$ckpt/adapter_model.bin"
      head -c 64 /dev/urandom > "$ckpt/non_lora_trainables.bin"
    done
    log "[train:$name] DRY_RUN：跳过 deepspeed，生成假 checkpoint $ckpt"
  else
    deepspeed --include localhost:"$GPUS" --master_port "$MASTER_PORT" \
      "$ROOT/ETrain/Train/LLaVA/train_mem.py" "${args[@]}" 2>&1 | tee -a "$logf"
  fi
  # checkpoint 校验（工单 4）
  local ckrep
  ckrep=$(py ckpt-validate "$ckpt")
  echo "==== checkpoint 校验 [$ckpt] ===="
  echo "$ckrep" | python3 -m json.tool
  log "[train:$name] checkpoint 校验通过"
}

# ---- 评估（工单 1/5）-------------------------------------------------------
eval_one() {  # $1=task $2=ckpt $3=stage
  local etask="$1" ckpt="$2" stage="$3"
  local script="$ROOT/scripts/LLaVA/Eval/${TASK_EVAL_SCRIPT[$etask]}.sh"
  local final_dir="$RES_ROOT/$etask/$stage"
  local tmp_dir="$RES_ROOT/.tmp_eval_${etask}_${stage}_$$"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"
  log "[eval:$etask@$stage] 全新临时目录 $tmp_dir"
  # 失败必须非零退出，绝不吞错
  if [[ "$DRY_RUN" == "1" ]]; then
    EVAL_DRY_RUN=1 CUDA_VISIBLE_DEVICES="${GPUS%%,*}" RESULT_DIR="$tmp_dir" \
      QUESTION_FILE="${TASK_QF[$etask]}" \
      bash "$script" "$stage" "$ckpt"
  else
    CUDA_VISIBLE_DEVICES="$GPUS" RESULT_DIR="$tmp_dir" \
      QUESTION_FILE="${TASK_QF[$etask]}" bash "$script" "$stage" "$ckpt"
  fi
  # 预测校验：总数 / 唯一 ID / 集合 / 顺序
  py verify-predictions \
    --questions "${TASK_QF[$etask]}" \
    --predictions "$tmp_dir/$stage/merge.jsonl" \
    --order-check 1 >/dev/null
  log "[eval:$etask@$stage] 预测校验通过"
  # 产物校验
  py artifact-check --task "$etask" --stage-dir "$tmp_dir/$stage" >/dev/null
  log "[eval:$etask@$stage] 产物校验通过"
  # 成功后原子换入正式目录（旧的先移走，防半成品混用）
  mkdir -p "$(dirname "$final_dir")"
  if [[ -e "$final_dir" ]]; then
    mv "$final_dir" "$RES_ROOT/.stale_${etask}_${stage}_$$"
  fi
  mv "$tmp_dir/$stage" "$final_dir"
  rm -rf "$RES_ROOT/.tmp_eval_${etask}_${stage}_$$" "$RES_ROOT/.stale_${etask}_${stage}_$$"
  log "[eval:$etask@$stage] 已原子换入 $final_dir"
}

# ---- 单轮（工单 4/6）-------------------------------------------------------
run_round() {
  local j="$1"
  local task="${TASKS[$((j - 1))]}"
  local task_ckpt="$CKPT_ROOT/round${j}_task_llava_lora"
  local replay_ckpt="$CKPT_ROOT/round${j}_replay_llava_lora"
  local eval_ckpt="$task_ckpt"
  local marker="$RES_ROOT/.round${j}_done"
  local replay_json="$REPLAY_DATA_DIR/round${j}_train.json"

  if [[ -f "$marker" ]]; then
    # .roundj_done 不能单独作为成功依据（工单 6）
    log "round$j 存在标记，执行 validate_round..."
    py validate-round --res-root "$RES_ROOT" --tasks-json "$TASKS_JSON" \
      --round "$j" --ckpt-dir "$eval_ckpt" --replay-data "$replay_json" >/dev/null \
      || { log "validate_round 失败，重跑 round$j"; rm -f "$marker"; }
    if [[ -f "$marker" ]]; then
      log "round$j 校验通过，跳过"
      return 0
    fi
  fi

  local prev=""
  if (( j > 1 )); then
    prev="$CKPT_ROOT/round$((j - 1))_replay_llava_lora"
    # round1 无 replay → 上一轮加载其 task ckpt
    [[ -d "$prev" ]] || prev="$CKPT_ROOT/round$((j - 1))_task_llava_lora"
    [[ -d "$prev" ]] || die "缺上一轮 checkpoint $prev"
  fi

  # 1) 任务微调
  train_one "round${j}_${task}_task" "$DATA_DIR/$task/train.json" "$task_ckpt" "$prev" "$EPOCHS"

  # 2) replay（j>=2；写独立目录，禁止与 task 同目录）
  if (( j > 1 )); then
    local nested_args=()
    # 0.01 嵌套验证：若 0.10 的 replay 数据已存在，断言 0.01 ⊆ 0.10
    if [[ "$SAMPLE_MODE" == "prefix" && "$RATIO" == "0.01" ]]; then
      local outer_json="$REPLAY_DATA_DIR/../ratio_0.1/round${j}_train.json"
      [[ -f "$outer_json.manifest.json" ]] && nested_args=(--nested-with "$outer_json.manifest.json")
    fi
    log "round$j 构建 replay 数据 (ratio=$RATIO prefix)"
    python3 "$ROOT/scripts/CoIN_Replay/build_replay_data.py" \
      --tasks "${TASKS[@]}" --data-dir "$DATA_DIR" --image-dir "$IMG_DIR" \
      --round "$j" --ratio "$RATIO" --seed "$SEED" \
      --out "$replay_json" "${nested_args[@]}"
    local k
    k=$(python3 -c "import json; m=json.load(open('$replay_json.manifest.json')); print(m['output']['N'])")
    # 方案 D：replay 段 accum = REPLAY_ACCUM（正式跑=1；task 段不受影响恒 16）
    train_one "round${j}_replay" "$replay_json" "$replay_ckpt" "$task_ckpt" "$REPLAY_EPOCHS" "$k" "${REPLAY_ACCUM:-}"
    # 断言 replay 真实更新（评审 C-1/C-3 铁证，2026-09-04 方案 D 强化）：
    # 非 DRY_RUN 下走 ckpt-tensor-diff（changed≥1 + 规范化 hash 不同 + 全 finite），
    # 替代弱 param_hash 比较（bf16 无 torch 时可能静默跳过——禁止）
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN：跳过 tensor-diff 强化断言（假 checkpoint 无真实训练语义）"
    else
      if ! py ckpt-tensor-diff "$task_ckpt" "$replay_ckpt"; then
        die "replay 未产生真实参数更新（ckpt-tensor-diff 失败）：task=$task_ckpt replay=$replay_ckpt"
      fi
      log "ckpt-tensor-diff 通过：replay 真实更新已确认（changed>0 + hash≠ + finite）"
    fi
    eval_ckpt="$replay_ckpt"
  fi

  # 3) 评估任务 1..j（每轮评估用本轮最终 checkpoint：j>=2 为 replay ckpt）
  for (( i = 1; i <= j; i++ )); do
    eval_one "${TASKS[$((i - 1))]}" "$eval_ckpt" "round${j}"
  done

  # round manifest（原子）+ 轮次标记
  local ckrep
  ckrep=$(py ckpt-validate "$eval_ckpt")
  # 方案 D：round manifest 并入 replay 段 train-plan（per-rank micro/HF/DS 步数/LR 摘要）
  local rp_plan="{}"
  if [[ -f "$RES_ROOT/logs/round${j}_replay.plan.json" ]]; then
    rp_plan=$(cat "$RES_ROOT/logs/round${j}_replay.plan.json")
  fi
  py round-manifest-write --res-root "$RES_ROOT" --round "$j" --info-json \
    "$(python3 -c "import json,sys; print(json.dumps({'task': '$task', 'eval_ckpt': '$eval_ckpt', 'task_ckpt': '$task_ckpt', 'replay_ckpt': '$replay_ckpt', 'replay_data': '$replay_json', 'ckpt': json.loads(sys.argv[1]), 'replay_plan': json.loads(sys.argv[2])}))" "$ckrep" "$rp_plan")" >/dev/null
  touch "$marker"
  log "round$j 完成（round${j}_manifest.json + .round${j}_done）"
}

# ---- 主流程 ----------------------------------------------------------------
main() {
  # 配置校验必须先于 .complete 短路：已完成目录用不同配置重跑必须失败（防混用）
  if [[ -f "$RES_ROOT/run_manifest.json" ]]; then
    log "存在 run_manifest.json，先执行恢复配置校验..."
    py manifest-resume-check --res-root "$RES_ROOT" --root "$ROOT"
    log "配置 hash 一致"
    if [[ -f "$RES_ROOT/.complete" ]]; then
      log "整组已完成（.complete 存在），跳过"
      exit 0
    fi
  else
    if [[ -f "$RES_ROOT/.complete" ]]; then
      die "存在 .complete 但缺 run_manifest.json——结果目录状态损坏，拒绝继续"
    fi
  fi
  preflight
  write_manifest
  mkdir -p "$RES_ROOT/logs"
  for (( j = 1; j <= T; j++ )); do
    run_round "$j"
  done
  log "全部轮次完成，聚合 MAA/BWT"
  python3 "$ROOT/scripts/CoIN_Replay/aggregate_coin.py" \
    --results-dir "$RES_ROOT" --tasks "${TASKS[@]}"
  touch "$RES_ROOT/.complete"
  log "DONE: $RES_ROOT（coin_metrics.json + .complete）"
}

main "$@"
