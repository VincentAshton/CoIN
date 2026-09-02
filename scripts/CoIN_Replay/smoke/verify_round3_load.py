#!/usr/bin/env python3
"""verify_round3_load：验证 Round 3 previous-task 加载入口（canary C 评审 2026-09-02）。

关键链路：round2_replay_llava_lora → round3 的 previous_task_model_path。
（仅验证 round2 加载 round1 不够——round1 无 replay；本脚本用与正式 Round 3 完全相同的
加载入口：train.py 的 HfArgumentParser 解析 → create_LLaVA_model 建模型 →
llava_trainer.load_model_from_previous_task(model, model_args)，前向验证：

  - previous_task_model_path 明确指向 Round 2 replay checkpoint；
  - LoRA keys missing=0 / unexpected=0（入口内部硬校验，此处独立复算）；
  - 加载后模型 LoRA 权重的规范化 float32 tensor hash == Round 2 replay 权重 hash；
  - 该 hash != Round 2 task 权重 hash。

单进程 CPU（--use_cpu True），无需 deepspeed/分布式；不改任何训练/评估代码。

用法（仓库根目录执行）:
  python scripts/CoIN_Replay/smoke/verify_round3_load.py \
      --model-base checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5 \
      --vision-tower checkpoints/LLaVA/clip-vit-large-patch14-336 \
      --projector checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin \
      --previous-task <round2_replay_llava_lora> \
      --task-ckpt <round2_task_llava_lora> \
      --lora-r 192 --lora-alpha 256 --lora-dropout 0.05
exit: 0 全部断言通过 / 1 任一断言失败 / 2 结构错误
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from transformers import HfArgumentParser  # noqa: E402

from ETrain.Train.LLaVA.train import ModelArguments, TrainingArguments  # noqa: E402
from ETrain.Dataset.dataset import DataArguments  # noqa: E402
from ETrain.Models.LLaVA.utils import create_LLaVA_model  # noqa: E402
from ETrain.Train.LLaVA.llava_trainer import load_model_from_previous_task  # noqa: E402


def _norm(k: str) -> str:
    """与 llava_trainer.load_model_from_previous_task 的 _norm 一致：去 base_model. 前缀 + 适配器名段。"""
    if k.startswith("base_model."):
        k = k[len("base_model."):]
    return k.replace(".default.", ".")


def _f32_hash(state: dict) -> str:
    """规范化 float32 值 hash：sorted normalized keys + key 名 + float32 字节。

    用 float32 规范型，使 bf16/fp16/fp32 的文件张量与加载后参数可比（bf16/fp16→fp32 无损；
    若保存 fp32 而加载被截成 bf16 则值不等 → 断言如实失败，暴露 dtype 截断问题）。
    """
    buf = hashlib.sha256()
    for k in sorted(state.keys()):
        buf.update(k.encode("utf-8"))
        t = state[k].detach().cpu().contiguous().float()
        buf.update(t.view(torch.uint8).numpy().tobytes())
    return buf.hexdigest()


def _load_adapter_dict(ckpt_dir: str) -> dict:
    p = os.path.join(ckpt_dir, "adapter_model.bin")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{ckpt_dir} 缺少 adapter_model.bin")
    w = torch.load(p, map_location="cpu")
    if not isinstance(w, dict):
        raise ValueError(f"{p}: adapter 非 state dict")
    return {_norm(k): v for k, v in w.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-base", required=True)
    ap.add_argument("--vision-tower", required=True)
    ap.add_argument("--projector", required=True)
    ap.add_argument("--previous-task", required=True,
                    help="Round 2 replay checkpoint（Round 3 的 previous_task_model_path）")
    ap.add_argument("--task-ckpt", required=True,
                    help="Round 2 task checkpoint（对照：加载后 hash 必须 != task hash）")
    ap.add_argument("--lora-r", type=int, default=192)
    ap.add_argument("--lora-alpha", type=int, default=256)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    a = ap.parse_args()

    for p in (a.model_base, a.vision_tower, a.projector, a.previous_task, a.task_ckpt):
        if not os.path.exists(p):
            raise SystemExit(f"[verify_round3] 路径不存在: {p}")

    # ---- 与正式训练完全相同的解析入口（train.py 同款 HfArgumentParser + 三组 dataclass）----
    argv = [
        "--model_name_or_path", a.model_base,
        "--vision_tower", a.vision_tower,
        "--pretrain_mm_mlp_adapter", a.projector,
        "--previous_task_model_path", a.previous_task,
        "--version", "v1",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False",
        "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "pad",
        "--bf16", "True",
        "--tf32", "True",
        "--lora_enable", "True",
        "--lora_r", str(a.lora_r),
        "--lora_alpha", str(a.lora_alpha),
        "--lora_dropout", str(a.lora_dropout),
        "--model_max_length", "2048",
        "--gradient_checkpointing", "False",
        "--use_cpu", "True",
        "--output_dir", os.path.join(a.previous_task, ".round3_load_verify_tmp"),
    ]
    model_args, data_args, training_args = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(argv)
    # 镜像 train.py 第 115 行：transformers TrainingArguments __post_init__ 后冻结，
    # 而 create_LLaVA_model 会写 training_args.tune_mm_mlp_adapter——不解冻即 FrozenInstanceError
    training_args._frozen = False

    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    print(f"[verify_round3] previous_task_model_path={model_args.previous_task_model_path}")
    model, _tokenizer = create_LLaVA_model(
        training_args, model_args, data_args, {}, compute_dtype, local_rank=None)

    # ---- 正式 Round 3 的 previous-task 加载入口（train.py 同款调用）----
    load_model_from_previous_task(model, model_args)

    # ---- 独立复算断言 ----
    replay_sd = _load_adapter_dict(a.previous_task)   # Round 2 replay 权重（Round 3 要加载的）
    task_sd = _load_adapter_dict(a.task_ckpt)         # Round 2 task 权重（对照）
    # 模型加载后的 LoRA 权重（named_parameters 带 base_model. 前缀与 .default. 段 → _norm）
    model_sd = {_norm(n): p.detach().cpu()
                for n, p in model.named_parameters()
                if (".lora_A." in n or ".lora_B." in n) and ".default." in n}

    keys_replay = set(replay_sd)
    keys_model = set(model_sd)
    missing = sorted(keys_replay - keys_model)
    unexpected = sorted(keys_model - keys_replay)
    keys_ok = not missing and not unexpected
    all_finite = all(bool(torch.isfinite(v).all().item()) for v in model_sd.values())
    shape_ok = all(tuple(model_sd[k].shape) == tuple(replay_sd[k].shape)
                   for k in keys_replay & keys_model)
    h_model = _f32_hash(model_sd)
    h_replay = _f32_hash(replay_sd)
    h_task = _f32_hash(task_sd)
    eq_replay = h_model == h_replay
    neq_task = h_model != h_task
    # task 与 replay 自身必须已不同（canary C 的 tensor 断言在此得到交叉验证）
    replay_neq_task = h_replay != h_task

    report = {
        "pass": bool(keys_ok and shape_ok and all_finite and eq_replay and neq_task and replay_neq_task),
        "previous_task": a.previous_task,
        "task_ckpt": a.task_ckpt,
        "n_model_lora_tensors": len(model_sd),
        "n_replay_tensors": len(replay_sd),
        "load_keys": {"missing": missing[:10], "unexpected": unexpected[:10],
                      "missing_count": len(missing), "unexpected_count": len(unexpected)},
        "all_finite": all_finite,
        "shapes_match": shape_ok,
        "float32_tensor_hash": {
            "model_after_load": h_model,
            "round2_replay": h_replay,
            "round2_task": h_task,
            "model_eq_replay": eq_replay,
            "model_neq_task": neq_task,
            "replay_neq_task": replay_neq_task,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
