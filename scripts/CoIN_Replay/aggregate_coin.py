#!/usr/bin/env python3
"""CoIN+Replay: 聚合 4 任务 Truth Alignment 准确率 -> A 矩阵 + MAA/BWT。

严格模式（TRACE 教训 P0-1）：任一 (round j, task i) 的准确性产物缺失/解析失败
即非零退出，且不写结果文件 —— 绝不输出半成品。

准确性产物位置（results/CoIN_Replay/ratio_<r>/<Task>/round<j>/）：
  ScienceQA  -> output_result.jsonl  {"acc": 0-100}
  TextVQA    -> Result.text          "Accuracy: XX.XX%"
  ImageNet   -> Result.text          "Accuracy: XX.XX%"   (注意官方脚本拼写 eval_ImagetNet)
  GQA        -> Result.text          "Accuracy: XX.XX%"

指标口径（论文 Section 3.1.3）：
  MAA = (1/T) * sum_j ( (1/j) * sum_{i<=j} A_{j,i} )
  BWT = (1/T) * sum_i ( A_{T,i} - A_{i,i} )
A_{j,i} = 第 j 轮训练完的 checkpoint 在第 i 个任务测试集上的准确率（0-100）。

用法:
  python scripts/CoIN_Replay/aggregate_coin.py \
      --results-dir results/CoIN_Replay/ratio_0.1 \
      --tasks ScienceQA TextVQA ImageNet GQA
输出: <results-dir>/coin_metrics.json （原子写 tmp+os.replace）
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ACC_TEXT_RE = re.compile(r"Accuracy:\s*([\d.]+)%")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    return p.parse_args()


def read_accuracy(task: str, results_dir: Path, j: int) -> float:
    """读取第 j 轮训练后任务 task 的 Truth Alignment 准确率。"""
    stage = results_dir / task / f"round{j}"
    if task == "ScienceQA":
        f = stage / "output_result.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"缺失 {f}")
        data = json.loads(f.read_text(encoding="utf-8"))
        acc = data.get("acc")
        if acc is None:
            raise ValueError(f"{f} 中无 acc 字段")
        return float(acc)
    f = stage / "Result.text"
    if not f.exists():
        raise FileNotFoundError(f"缺失 {f}")
    text = f.read_text(encoding="utf-8")
    m = ACC_TEXT_RE.search(text)
    if not m:
        raise ValueError(f"{f} 中未找到 Accuracy: XX.XX%")
    return float(m.group(1))


def atomic_write(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    args = parse_args()
    T = len(args.tasks)
    A = [[0.0] * T for _ in range(T)]  # A[j][i] = 第 j+1 轮后在任务 i+1 上的准确率
    for j in range(1, T + 1):
        for i in range(1, j + 1):
            task = args.tasks[i - 1]
            acc = read_accuracy(task, args.results_dir, j)
            if not (0.0 <= acc <= 100.0):
                raise ValueError(f"{task} round{j} 准确率越界: {acc}")
            A[j - 1][i - 1] = acc
            print(f"[aggregate] A[{j}][{i}] {task} round{j}: {acc:.2f}")

    # MAA / BWT（与论文公式一致；A 为 0-100 量纲，直接平均）
    maa = sum(sum(A[j - 1][:j]) / j for j in range(1, T + 1)) / T
    bwt = sum(A[T - 1][i] - A[i][i] for i in range(T)) / T

    # ratio 优先取 run_manifest.json 的配置（工单 6：manifest 是显式配置的唯一权威）
    ratio = None
    man_path = args.results_dir / "run_manifest.json"
    if man_path.is_file():
        try:
            ratio = json.loads(man_path.read_text(encoding="utf-8"))["config"]["ratio"]
        except Exception:
            ratio = None
    if ratio is None:
        try:
            ratio = float(os.path.basename(str(args.results_dir)).split("_")[-1])
        except Exception:
            raise ValueError(
                f"无法从 {args.results_dir} 确定 ratio：run_manifest.json 缺失且目录名不含 ratio")
    metrics = {
        "ratio": ratio,
        "tasks": args.tasks,
        "T": T,
        "A_matrix": A,
        "MAA": round(maa, 4),
        "BWT": round(bwt, 4),
    }
    out = args.results_dir / "coin_metrics.json"
    atomic_write(out, metrics)
    print(f"[aggregate] MAA={maa:.4f}  BWT={bwt:.4f}")
    print(f"[aggregate] 已写入 {out}")


if __name__ == "__main__":
    main()
