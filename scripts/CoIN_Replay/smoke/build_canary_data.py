#!/usr/bin/env python3
"""canary C：从真实数据切出每任务 8-32 条的迷你集（仅 json，图片仍引用 cl_dataset）。"""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="真实 playground/Instructions_Original")
    p.add_argument("--out", required=True, help="迷你数据目录（json 子集）")
    p.add_argument("--tasks", nargs="+", default=["ScienceQA", "TextVQA", "ImageNet", "GQA"])
    p.add_argument("--train-n", type=int, default=16)
    p.add_argument("--test-n", type=int, default=8)
    args = p.parse_args()

    for task in args.tasks:
        src = Path(args.data_dir) / task
        dst = Path(args.out) / task
        dst.mkdir(parents=True, exist_ok=True)
        test_name = "val.json" if task == "TextVQA" else "test.json"
        for role, n in (("train.json", args.train_n), (test_name, args.test_n)):
            sp = src / role
            if not sp.is_file():
                print(f"[build_canary] 跳过缺失 {sp}")
                continue
            data = json.load(open(sp))
            # 取数据前缀（与 replay prefix 口径一致）
            picked = data[:n]
            json.dump(picked, open(dst / role, "w"))
            print(f"[build_canary] {task}/{role}: {len(picked)}/{len(data)} 条 -> {dst / role}")
    print("[build_canary] done")


if __name__ == "__main__":
    main()
