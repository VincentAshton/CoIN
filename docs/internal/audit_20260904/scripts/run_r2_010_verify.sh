#!/bin/bash
# 方案 D 实机验证：0.1 round2 replay（N=1272）gas1 单跑验证（2026-09-04）
# 复用 gas1_gate 的 task_sqa ckpt 作 prev；期望 gs=23、23 真步、tensor 448 变
set -uo pipefail
export PATH=/root/data/coin/conda_envs/coin/bin:$PATH
ROOT=/root/data/coin/project
GATE=/root/data/coin/tmp/gas1_gate_20260904
LDIR=/root/data/coin/logs/presweep/gas1_gate_20260904
DS=$(command -v deepspeed)
PY=/root/data/coin/conda_envs/coin/bin/python

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
  --gradient_accumulation_steps 1
  --data_path "$GATE/data/r2_010.json"
  --output_dir "$GATE/ckpt/replay_r2_010"
  --previous_task_model_path "$GATE/ckpt/task_sqa"
)
echo "===== [r2_010] train N=1272 gas1 $(date '+%F %T') ====="
$DS --include localhost:0,1,2,3 --master_port 29601 \
  "$ROOT/ETrain/Train/LLaVA/train_mem.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$LDIR/train_replay_r2_010.log"
rc=${PIPESTATUS[0]}
echo "[r2_010] train rc=$rc"
if [ "$rc" -ne 0 ]; then exit 92; fi

echo "===== [r2_010] 断言收集 ====="
"$PY" - "$GATE" <<'PYEOF'
import json, os, sys
GATE = sys.argv[1]
sys.path.insert(0, "/root/data/coin/project/scripts/CoIN_Replay")
from coin_lib import ckpt_tensor_compare, per_rank_micro_batches

ts = json.load(open(os.path.join(GATE, "ckpt/replay_r2_010/trainer_state.json")))
gs = int(ts.get("global_step", 0))
hist = ts.get("log_history", [])
lrs = [h.get("learning_rate") for h in hist if "learning_rate" in h]
man = json.load(open(os.path.join(GATE, "data/r2_010.json.manifest.json")))
N = man["output"]["N"]
M = per_rank_micro_batches(N, 14, 4)[0]
print(f"N={N} M(per-rank micro)={M} gs={gs} exp_gs={M} lr_first={lrs[0] if lrs else None} lr_last={lrs[-1] if lrs else None} lr_n={len(lrs)}")
ok = gs == M
diff = ckpt_tensor_compare(os.path.join(GATE, "ckpt/task_sqa"),
                           os.path.join(GATE, "ckpt/replay_r2_010"))
th = diff.get("tensor_hash") or {}
fin = diff.get("finite") or {}
print(f"tensor-diff pass={diff['pass']} changed={diff['changed_tensor_count']} "
      f"l2={diff['l2_norm_diff']} max_abs={diff['max_abs_diff']} "
      f"hash_differs={th.get('differs')} finite={fin}")
ok = ok and diff["pass"] is True and diff["changed_tensor_count"] > 0 and th.get("differs") is True
print("R2_010_GATE_PASS" if ok else "R2_010_GATE_FAIL")
sys.exit(0 if ok else 1)
PYEOF
