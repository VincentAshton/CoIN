#!/usr/bin/env python3
"""独立重算 CoIN+Replay ratio=0.1 指标（2026-09-04，GitHub 可复现版）。

输入：acc_sources.json（10 个 eval 单元的聚合 accuracy，去敏、不含原始预测）。
独立从单元 accuracy 构造 A 矩阵并计算 MAA/BWT，与 coin_metrics.json（runtime 产物）
交叉验证——不读取 coin_metrics.json 中的 A_matrix/MAA/BWT 参与计算。

用法: python3 recompute_metrics.py [acc_sources.json 所在目录，默认脚本目录]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else sys.argv[1]
TASKS = ["ScienceQA", "TextVQA", "ImageNet", "GQA"]
# 每 round 评估的任务子集（与 run_replay_exp.sh run_round 语义一致）
EVAL_MAP = {1: ["ScienceQA"], 2: ["ScienceQA", "TextVQA"],
            3: ["ScienceQA", "TextVQA", "ImageNet"],
            4: ["ScienceQA", "TextVQA", "ImageNet", "GQA"]}


def main():
    src = json.load(open(os.path.join(ROOT, "acc_sources.json")))
    assert src["ratio"] == 0.1
    T = len(TASKS)
    A = [[0.0] * T for _ in range(T)]
    for u in src["units"]:
        i = TASKS.index(u["task"])
        j = u["round"] - 1
        A[j][i] = float(u["accuracy"])

    maa = sum(sum(A[j - 1][:j]) / j for j in range(1, T + 1)) / T
    bwt = sum(A[T - 1][i] - A[i][i] for i in range(T)) / T
    final_mean = sum(A[T - 1]) / T

    out = {
        "tasks": TASKS, "ratio": 0.1, "A_matrix": A,
        "MAA": round(maa, 4), "BWT": round(bwt, 4),
        "final_average_accuracy": round(final_mean, 4),
        "row_averages": [round(sum(A[j][:j + 1]) / (j + 1), 4) for j in range(T)],
    }
    json.dump(out, open(os.path.join(ROOT, "recomputed_metrics.json"), "w"),
              indent=2, ensure_ascii=False)

    print("=== A 矩阵（独立重算，来自 acc_sources.json） ===")
    for j in range(T):
        print(f"round{j+1}: " + ", ".join(f"{v:.4f}" for v in A[j]))
    print(f"MAA = {out['MAA']}  BWT = {out['BWT']}  final avg = {out['final_average_accuracy']}")

    ref = os.path.join(ROOT, "coin_metrics.json")
    if os.path.isfile(ref):
        m = json.load(open(ref))
        dA = max(abs(a - b) for ra, rb in zip(A, m["A_matrix"]) for a, b in zip(ra, rb))
        print(f"与 coin_metrics.json 最大 A 差异: {dA:.2e}  "
              f"MAA diff {abs(out['MAA'] - m['MAA']):.2e}  "
              f"BWT diff {abs(out['BWT'] - m['BWT']):.2e}")
        assert dA < 1e-6 and abs(out["MAA"] - m["MAA"]) < 1e-4 \
            and abs(out["BWT"] - m["BWT"]) < 1e-4
        print("CROSS-VALIDATION PASS")
    else:
        print("（无 coin_metrics.json，跳过交叉验证）")


if __name__ == "__main__":
    main()
