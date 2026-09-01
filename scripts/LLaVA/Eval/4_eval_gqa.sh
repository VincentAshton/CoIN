#!/bin/bash
# GQA 评估（Truth Alignment）。本 fork 改动同 1_eval_sqa.sh：PID 收集 / 去 create_prompt / dry-run。
# 依赖 cl_dataset/GQA/testdev_balanced_questions.json（preflight 已校验）。
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
if [ ! -n "$1" ] ;then
    STAGE='Finetune'
else
    STAGE=$1
fi
if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/Instruction/Only_Pretrain_1.5/GQA/llava-1.5-7b-lora'
else
    MODELPATH=$2
fi
RESULT_DIR="${RESULT_DIR:-./results/CoIN/LLaVA/GQA}"
QF="${QUESTION_FILE:-./playground/Instructions_Original/GQA/test.json}"
STAGE_DIR="$RESULT_DIR/$STAGE"

if [ "${EVAL_DRY_RUN:-0}" = "1" ]; then
    python3 - "$STAGE_DIR" "$QF" "$CHUNKS" <<'PYEOF'
import json, os, sys
stage, qf, chunks = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(stage, exist_ok=True)
qs = json.load(open(qf))
if os.environ.get("EVAL_FAULT_INJECT") == "1":
    with open(os.path.join(stage, f"{chunks}_0.jsonl"), "w") as f:
        for q in qs[: max(1, len(qs) // 2)]:
            f.write(json.dumps({"question_id": q["question_id"], "text": "dummy"}) + "\n")
    print("[dry-run] 注入故障: chunk 未完成即退出", flush=True)
    sys.exit(1)
with open(os.path.join(stage, f"{chunks}_0.jsonl"), "w") as f:
    for q in qs:
        f.write(json.dumps({"question_id": q["question_id"], "text": "dummy"}) + "\n")
merged = []
for idx in range(chunks):
    fn = os.path.join(stage, f"{chunks}_{idx}.jsonl")
    if not os.path.isfile(fn):
        print(f"[dry-run] 缺 chunk {fn}", flush=True)
        sys.exit(1)
    merged.extend(open(fn, encoding="utf-8").read().splitlines())
with open(os.path.join(stage, "merge.jsonl"), "w", encoding="utf-8") as f:
    f.write("\n".join(merged) + "\n")
with open(os.path.join(stage, "testdev_balanced_predictions.json"), "w", encoding="utf-8") as f:
    json.dump([], f)
with open(os.path.join(stage, "Result.text"), "w", encoding="utf-8") as f:
    f.write(f"Samples: {len(qs)}\nAccuracy: 42.00%\n")
print(f"[dry-run] GQA OK -> {stage}")
PYEOF
    exit $?
fi

mkdir -p "$STAGE_DIR"
pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.LLaVA.CoIN.model_gqa \
        --model-path "$MODELPATH" \
        --model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
        --question-file "$QF" \
        --image-folder ./cl_dataset \
        --answers-file "$STAGE_DIR/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    pids+=($!)
done
for p in "${pids[@]}"; do
    wait "$p" || { echo "[eval] chunk 进程失败 (pid=$p)"; exit 1; }
done

output_file="$STAGE_DIR/merge.jsonl"
: > "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    chunk_file="$STAGE_DIR/${CHUNKS}_${IDX}.jsonl"
    [ -f "$chunk_file" ] || { echo "[eval] 缺 chunk 文件 $chunk_file"; exit 1; }
    cat "$chunk_file" >> "$output_file"
done

python -m ETrain.Eval.LLaVA.CoIN.convert_gqa_for_eval \
    --src "$output_file" \
    --dst "$STAGE_DIR/testdev_balanced_predictions.json" || exit 1
python -m ETrain.Eval.LLaVA.CoIN.eval_gqa \
    --tier testdev_balanced --path "$STAGE_DIR" --output-dir "$STAGE_DIR" || exit 1
