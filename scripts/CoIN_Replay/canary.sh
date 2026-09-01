#!/bin/bash
# ============================================================================
# CoIN+Replay canary（工单 9，云端 4×A100 上执行）
#
# A: bash -n + py_compile + 全部单测（零 GPU）
# B: GPU 冒烟：NCCL all-reduce + flash-attn fwd/bwd + DeepSpeed 最小任务
#    （同时断言 zero3_offload.json 无 scheduler/optimizer 段）
# C: 迷你数据 round1→round2 真实训练（触发 previous-task LoRA 加载/replay/下一轮加载/两任务评估）
# D: 故意杀死一个真实 eval chunk → 证明 fail-fast、无 .complete
# E: 完整 ScienceQA round1 训练 + probe logits 一致性 + 两次评估结果一致
#
# 用法（在仓库根目录）:
#   bash scripts/CoIN_Replay/canary.sh
# 环境变量: ROOT 之外的路径默认取仓库默认布局（checkpoints/LLaVA/...、cl_dataset、playground）
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
FAIL=0
step() { echo; echo "############ $* ############"; }
ok()   { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; FAIL=1; }

PY="${PYTHON:-python3}"
GPUS="${GPUS:-0,1,2,3}"
CANARY_ROOT="${CANARY_ROOT:-/dev/shm/coin_canary}"
mkdir -p "$CANARY_ROOT"

# ---------------- A：静态门禁 + 单测 ----------------
step "A: 静态检查 + 单测"
if bash scripts/CoIN_Replay/run_tests.sh "$PY"; then ok "A"; else bad "A"; fi

# ---------------- B：GPU 冒烟 ----------------
step "B: GPU 冒烟（NCCL / flash-attn / DeepSpeed）"
if grep -qE '"scheduler"|"optimizer"' scripts/zero3_offload.json; then
    bad "zero3_offload.json 含 scheduler/optimizer 段"
else
    ok "zero3_offload.json 无 scheduler/optimizer 段"
fi
if torchrun --nproc_per_node="$(awk -F',' '{print NF}' <<<"$GPUS")" --master_port=29517 \
        scripts/CoIN_Replay/smoke/smoke_gpu.py; then ok "NCCL/flash-attn"; else bad "NCCL/flash-attn"; fi
if "$PY" scripts/CoIN_Replay/smoke/smoke_ds.py --ds-config "$ROOT/scripts/zero3_offload.json" \
        --gpus "$GPUS"; then ok "DeepSpeed 最小任务"; else bad "DeepSpeed 最小任务"; fi

# ---------------- C：迷你数据 round1→round2 ----------------
step "C: 迷你数据真实训练（round1→round2，触发 LoRA 链 + replay）"
CD="$CANARY_ROOT/data"
"$PY" scripts/CoIN_Replay/smoke/build_canary_data.py \
    --data-dir playground/Instructions_Original --out "$CD" \
    --tasks ScienceQA TextVQA --train-n 16 --test-n 8
if DRY_RUN=0 GPUS="$GPUS" BATCH=2 ACCUM=1 EPOCHS=1 REPLAY_EPOCHS=1 \
       BASE_MODEL=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
       VISION_TOWER=checkpoints/LLaVA/clip-vit-large-patch14-336 \
       PROJECTOR=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin \
       DATA_DIR="$CD" IMG_DIR=cl_dataset \
       CKPT_ROOT="$CANARY_ROOT/ckpt" RES_ROOT="$CANARY_ROOT/res_c" \
       REPLAY_DATA_DIR="$CANARY_ROOT/replay" \
       PREFLIGHT_REPORT="$CANARY_ROOT/preflight_c.json" \
       PREFLIGHT_ARGS="--skip-pil" \
       TASKS_JSON='["ScienceQA","TextVQA"]' \
       bash scripts/CoIN_Replay/run_replay_exp.sh 0.1 \
    && [ -f "$CANARY_ROOT/res_c/.complete" ]; then
    ok "canary C 全链路（含 task!=replay hash 断言若 torch 可用）"
    echo "  训练日志: $CANARY_ROOT/res_c/logs/"
    # 记录 task/replay 参数 hash 供人工核对（工单 4 canary 证明点）
    "$PY" scripts/CoIN_Replay/coin_lib.py ckpt-validate "$CANARY_ROOT/ckpt/round2_task_llava_lora" | tee "$CANARY_ROOT/task_hash.json"
    "$PY" scripts/CoIN_Replay/coin_lib.py ckpt-validate "$CANARY_ROOT/ckpt/round2_replay_llava_lora" | tee "$CANARY_ROOT/replay_hash.json"
else
    bad "canary C"
fi

# ---------------- D：真实 eval chunk 被杀 → fail-fast ----------------
step "D: 故障注入（kill 一个真实 eval chunk）"
D_RES="$CANARY_ROOT/res_d"
rm -rf "$D_RES"
(
    set +e
    CUDA_VISIBLE_DEVICES="0,1" RESULT_DIR="$D_RES" \
      bash scripts/LLaVA/Eval/1_eval_sqa.sh roundX "$CANARY_ROOT/ckpt/round2_task_llava_lora" &
    EPID=$!
    # 等 chunk 进程出现后杀掉其中一个
    for _ in $(seq 1 60); do
        CPID=$(pgrep -f "[m]odel_vqa_science" | head -1)
        [ -n "$CPID" ] && break
        sleep 1
    done
    if [ -z "${CPID:-}" ]; then
        echo "FAIL D: 未观察到 chunk 进程"; exit 1
    fi
    kill -9 "$CPID"
    wait "$EPID"; RC=$?
    [ "$RC" -ne 0 ] || { echo "FAIL D: eval 未失败（rc=0）"; exit 1; }
    # 无准确性产物
    [ ! -f "$D_RES/roundX/output_result.jsonl" ] || { echo "FAIL D: 存在产物"; exit 1; }
    echo "PASS  D: chunk 被杀 → eval 非零退出且无产物"
) && ok "D" || bad "D"

# ---------------- E：完整 ScienceQA round1 + probe 一致性 ----------------
step "E: 完整 ScienceQA round1 + probe logits + 双评估一致性"
E_RES="$CANARY_ROOT/res_e"
if [ -f "$E_RES/.complete" ]; then
    ok "E: round1 已完成（.complete 存在），跳过训练"
else
    if GPUS="$GPUS" BATCH=14 ACCUM=16 EPOCHS=1 \
           BASE_MODEL=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
           VISION_TOWER=checkpoints/LLaVA/clip-vit-large-patch14-336 \
           PROJECTOR=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin \
           DATA_DIR=playground/Instructions_Original IMG_DIR=cl_dataset \
           CKPT_ROOT="$CANARY_ROOT/ckpt_e" RES_ROOT="$E_RES" \
           REPLAY_DATA_DIR="$CANARY_ROOT/replay_e" \
           PREFLIGHT_REPORT="$CANARY_ROOT/preflight_e.json" \
           TASKS_JSON='["ScienceQA"]' \
           bash scripts/CoIN_Replay/run_replay_exp.sh 0.1 \
        && [ -f "$E_RES/.complete" ]; then
        ok "E: 完整 ScienceQA round1 训练完成"
    else
        bad "E: ScienceQA round1"
    fi
fi
# probe logits（加载两次一致性 + finite）
E_CKPT="$CANARY_ROOT/ckpt_e/round1_task_llava_lora"
if "$PY" scripts/CoIN_Replay/smoke/probe_logits.py \
        --ckpt "$E_CKPT" \
        --model-base checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --vision-tower checkpoints/LLaVA/clip-vit-large-patch14-336 \
        --probe-json playground/Instructions_Original/ScienceQA/test.json; then
    ok "E: probe logits 一致性"
else
    bad "E: probe logits 一致性"
fi
# 双评估一致性（温度 0 → 两次结果必须完全一致）
E1="$E_RES/ScienceQA/round1"
rm -rf "$E1" "$CANARY_ROOT/eval2"
CUDA_VISIBLE_DEVICES="$GPUS" RESULT_DIR="$CANARY_ROOT/eval2" \
  bash scripts/LLaVA/Eval/1_eval_sqa.sh round1 "$E_CKPT" || bad "E: 第一次评估"
if diff -q "$E1/merge.jsonl" "$CANARY_ROOT/eval2/round1/merge.jsonl" >/dev/null \
   && diff -q "$E1/output_result.jsonl" "$CANARY_ROOT/eval2/round1/output_result.jsonl" >/dev/null; then
    ok "E: 两次评估结果逐字节一致（温度 0 确定性）"
else
    bad "E: 两次评估结果不一致"
fi

# ---------------- 结论 ----------------
echo
echo "======================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "CANARY 全部通过 → 允许启动正式 sweep"
    echo "正式命令:"
    echo "  bash scripts/CoIN_Replay/run_sweep.sh 0.1 0.01"
else
    echo "CANARY 存在失败（FAIL=$FAIL）→ 禁止启动正式 sweep"
fi
echo "======================================================"
exit "$FAIL"
