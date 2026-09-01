#!/usr/bin/env python3
"""CoIN+Replay: 构建 replay 训练数据（LLaVA 指令格式，原样保留样本）。

设计（对应 TRACE 的 past_task_ratio 语义）：
  - 对 round j (2..T)，取前 j-1 个任务的 train.json，每个任务按 ratio 抽样，
    合并为一个 replay json，供 replay 训练阶段使用（1 epoch，与任务训练同 LR）。
  - 抽样方式默认 random + 固定 seed（可 --mode prefix 切换为 TRACE 式前缀抽样）。
  - 样本字段（id/image/conversations）原样复制，image 相对路径不变
    （image_folder=./cl_dataset 下解析），因此无需关心各数据集的目录布局。

用法:
  python scripts/CoIN_Replay/build_replay_data.py \
      --tasks ScienceQA TextVQA ImageNet GQA \
      --data-dir playground/Instructions_Original \
      --round 3 --ratio 0.1 --seed 1234 --mode random \
      --out playground/Replay/ratio_0.1/round3_train.json

退出码: 0=成功; 1=任一任务数据缺失或抽样结果为空。
"""
import argparse
import json
import random
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", required=True, help="任务名列表（顺序微调顺序的前缀）")
    p.add_argument("--data-dir", type=Path, required=True, help="Instructions_Original 目录")
    p.add_argument("--round", type=int, required=True, help="当前 round j（取任务 1..j-1 作为 replay 源）")
    p.add_argument("--ratio", type=float, required=True, help="每任务回放比例 (0,1]")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mode", choices=["random", "prefix"], default="random",
                   help="random=固定 seed 随机抽样（默认）；prefix=取前 N*ratio 条（TRACE 原代码行为）")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_samples(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    assert isinstance(samples, list), f"{path} 不是 json 数组"
    for s in samples:
        assert isinstance(s, dict) and "image" in s and "conversations" in s, \
            f"{path} 中存在非 LLaVA 指令格式样本: {str(s)[:120]}"
    return samples


def main():
    args = parse_args()
    assert 0 < args.ratio <= 1.0, "ratio 必须在 (0,1]"
    assert args.round >= 2, "round 1 无需 replay（没有历史任务）"
    prev_tasks = args.tasks[: args.round - 1]
    if not prev_tasks:
        print(f"[build_replay] round {args.round}: 无历史任务，跳过")
        return

    rng = random.Random(args.seed)
    replay = []
    per_task = {}
    for task in prev_tasks:
        src = args.data_dir / task / "train.json"
        if not src.exists():
            print(f"[build_replay] ERROR: 缺少 {src}")
            sys.exit(1)
        samples = load_samples(src)
        n = len(samples)
        k = max(1, round(n * args.ratio))  # 每任务至少 1 条，避免空抽样
        if args.mode == "prefix":
            picked = samples[:k]
        else:
            picked = rng.sample(samples, k)
        replay.extend(picked)
        per_task[task] = {"total": n, "picked": len(picked)}
        print(f"[build_replay] {task}: {len(picked)}/{n} (ratio {args.ratio})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False)
    print(f"[build_replay] round {args.round} replay 数据: {len(replay)} 条 -> {args.out}")
    print(f"[build_replay] per-task: {json.dumps(per_task, ensure_ascii=False)}")
    if not replay:
        print("[build_replay] ERROR: replay 数据为空")
        sys.exit(1)


if __name__ == "__main__":
    main()
