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
# smoke_ds 由 torchrun --standalone 启动（评审 2026-09-02 方案 A：与正式四卡分布式路径一致，
# 避免 deepspeed 单进程 MPI 探测）；timeout 防进程挂死持续烧卡
if timeout 900 torchrun --standalone --nproc_per_node="$(awk -F',' '{print NF}' <<<"$GPUS")" \
        --master_port=29518 scripts/CoIN_Replay/smoke/smoke_ds.py \
        --ds-config "$ROOT/scripts/zero3_offload.json"; then ok "DeepSpeed 最小任务"; else bad "DeepSpeed 最小任务"; fi

# ---------------- C：迷你数据 round1→round2（评审 2026-09-02 重审版） ----------------
step "C: 迷你数据真实训练（round1→round2，触发 LoRA 链 + replay）"
CD="$CANARY_ROOT/data"
# C 必须从头真实训练：旧的 .round*_done/.complete 不得跳过新断言（评审 C-6/C-7）。
# C 与 E 使用独立根（ckpt/ckpt_e、res_c/res_e），只清 C 自己的目录，不影响 E 断点复用。
rm -rf "$CD" "$CANARY_ROOT/ckpt" "$CANARY_ROOT/res_c" "$CANARY_ROOT/replay" \
       "$CANARY_ROOT/preflight_c.json"
# 数据量公式化（评审 C-1）：replay 样本数必须 >= 2 个 optimizer step 的有效 batch。
# replay k = floor(round1_train_N * ratio)；有效 batch = world * per_device_batch * grad_accum。
WORLD=$(awk -F',' '{print NF}' <<< "$GPUS")
C_BATCH=2 C_ACCUM=1 C_RATIO=0.1
EFF=$((WORLD * C_BATCH * C_ACCUM))          # 每 optimizer step 消耗样本
REPLAY_STEP_TARGET=3                         # 目标 ≥2，取 3 留余量（防采样/分桶边界）
REPLAY_MIN=$((REPLAY_STEP_TARGET * EFF))     # replay 最少样本数
TRAIN_N=$(python3 -c "import math; print(int(math.ceil($REPLAY_MIN / $C_RATIO)))")
echo "[canary C] world=$WORLD batch=$C_BATCH accum=$C_ACCUM eff_batch=$EFF"
echo "[canary C] replay_target=${REPLAY_STEP_TARGET} optimizer_steps -> replay_min=$REPLAY_MIN samples, ratio=$C_RATIO -> round1_train_n=$TRAIN_N"
"$PY" scripts/CoIN_Replay/smoke/build_canary_data.py \
    --data-dir playground/Instructions_Original --out "$CD" \
    --tasks ScienceQA TextVQA --train-n "$TRAIN_N" --test-n 8
if DRY_RUN=0 GPUS="$GPUS" BATCH=$C_BATCH ACCUM=$C_ACCUM EPOCHS=1 REPLAY_EPOCHS=1 \
       ENFORCE_MIN_STEPS=1 \
       BASE_MODEL=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
       VISION_TOWER=checkpoints/LLaVA/clip-vit-large-patch14-336 \
       PROJECTOR=checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin \
       DATA_DIR="$CD" IMG_DIR=cl_dataset \
       CKPT_ROOT="$CANARY_ROOT/ckpt" RES_ROOT="$CANARY_ROOT/res_c" \
       REPLAY_DATA_DIR="$CANARY_ROOT/replay" \
       PREFLIGHT_REPORT="$CANARY_ROOT/preflight_c.json" \
       PREFLIGHT_ARGS="--skip-pil" \
       TASKS_JSON='["ScienceQA","TextVQA"]' \
       bash scripts/CoIN_Replay/run_replay_exp.sh "$C_RATIO" \
    && [ -f "$CANARY_ROOT/res_c/.complete" ]; then
    ok "canary C run_replay_exp 完成（round1→round2 + replay）"
else
    bad "canary C run_replay_exp"
fi
# ---- 强制断言 A（评审 C-2）：replay 训练分辨率 + optimizer steps + LR ----
REPLAY_CKPT="$CANARY_ROOT/ckpt/round2_replay_llava_lora"
TASK_CKPT="$CANARY_ROOT/ckpt/round2_task_llava_lora"
REPLAY_MAN="$CANARY_ROOT/replay/round2_train.json.manifest.json"
REPLAY_TS="$REPLAY_CKPT/trainer_state.json"
if [ -f "$REPLAY_TS" ] && [ -f "$REPLAY_MAN" ] && [ -f "$TASK_CKPT/adapter_model.bin" ]; then
    if python3 - "$REPLAY_TS" "$REPLAY_MAN" "$EFF" "$C_ACCUM" "$WORLD" <<'PYEOF'
import json, sys
ts, man, eff, accum, world = (json.load(open(sys.argv[1])),
                              json.load(open(sys.argv[2])),
                              int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
n = man["output"]["N"]
gs = int(ts.get("global_step", 0))
hist = ts.get("log_history", [])
lr_last = None
for h in reversed(hist):
    if "learning_rate" in h:
        lr_last = h["learning_rate"]; break
print(f"[canary C] replay 数据: N={n} 条（manifest），有效 batch/step={eff}")
print(f"[canary C] replay 训练: global_step={gs} optimizer_steps={gs} "
      f"(要求 >=2)，grad_accum={accum} world={world} per_device_batch={eff // (accum * world)}")
print(f"[canary C] replay 末次 LR={lr_last}")
ok = gs >= 2 and lr_last is not None
print(f"[canary C] replay optimizer_steps>=2 断言: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
    then
        ok "C: replay optimizer_steps>=2（global_step 增加 + LR 记录）"
    else
        bad "C: replay optimizer_steps<2（replay 未产生有效训练步）"
    fi
else
    bad "C: 缺 replay trainer_state/manifest/task adapter"
fi
# ---- 强制断言 B（评审 C-3）：tensor 级 task vs replay 比较（禁止 metadata 假阳性）----
if "$PY" scripts/CoIN_Replay/coin_lib.py ckpt-tensor-diff "$TASK_CKPT" "$REPLAY_CKPT"; then
    ok "C: task/replay tensor 级差异（changed>0 + hash 不同 + finite + keys/shapes 一致）"
else
    bad "C: task/replay tensor 相同或结构不符（replay 未生效）"
fi
# ---- 强制断言 C（评审 C-4）：Round 3 previous-task 加载（源=round2_replay）----
if "$PY" scripts/CoIN_Replay/smoke/verify_round3_load.py \
        --model-base checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --vision-tower checkpoints/LLaVA/clip-vit-large-patch14-336 \
        --projector checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin \
        --previous-task "$REPLAY_CKPT" --task-ckpt "$TASK_CKPT"; then
    ok "C: round3 previous-task 加载验证（missing=0/unexpected=0 + hash==replay != task）"
else
    bad "C: round3 previous-task 加载验证失败"
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
# 双评估一致性（温度 0 → 两次结果必须逐题一致）
# 修复（评审 2026-09-02）：① 原 rm -rf "$E1" 删掉了比对参照，检查永远失败——E1 不再删，
#    .complete 跳训导致 E1 缺失时自动重建；② merge.jsonl 含随机 answer_id（shortuuid），
#    逐字节 diff 永远不等——比对排除 answer_id，其余字段+顺序逐题一致才算 PASS
#    （预测确定性已由 eval3 双跑诊断证实：4241 行除 answer_id 外全部一致）
E1="$E_RES/ScienceQA/round1"
rm -rf "$CANARY_ROOT/eval2"
if [ ! -f "$E1/merge.jsonl" ]; then
    mkdir -p "$(dirname "$E1")"
    CUDA_VISIBLE_DEVICES="$GPUS" RESULT_DIR="$(dirname "$E1")" \
      bash scripts/LLaVA/Eval/1_eval_sqa.sh round1 "$E_CKPT" || bad "E: 第一次评估"
fi
CUDA_VISIBLE_DEVICES="$GPUS" RESULT_DIR="$CANARY_ROOT/eval2" \
  bash scripts/LLaVA/Eval/1_eval_sqa.sh round1 "$E_CKPT" || bad "E: 第二次评估"
if [ -f "$E1/merge.jsonl" ] && [ -f "$CANARY_ROOT/eval2/round1/merge.jsonl" ]; then
    if python3 - "$E1/merge.jsonl" "$CANARY_ROOT/eval2/round1/merge.jsonl" <<'PYEOF'
import json, sys
def load(p):
    rows = []
    for ln, line in enumerate(open(p, encoding="utf-8"), 1):
        line = line.strip()
        if line:
            r = json.loads(line)
            r.pop("answer_id", None)  # shortuuid 每次随机，非预测内容
            rows.append((ln, r))
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"[E] 两次评估 {len(a)} vs {len(b)} 条（排除 answer_id）逐题一致: {a == b}")
sys.exit(0 if a == b else 1)
PYEOF
    then
        ok "E: 两次评估逐题一致（温度 0 确定性，排除随机 answer_id）"
    else
        bad "E: 两次评估结果不一致"
    fi
else
    bad "E: 缺少评估产物（E1 或 eval2）"
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
