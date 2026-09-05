#!/usr/bin/env python3
"""CoIN+Replay ratio=0.10 / 0.01 公开指标重算脚本（GitHub 可复现版，2026-09-05）。

输入：<root>/ratio_0.10/acc_sources.json 与 <root>/ratio_0.01/acc_sources.json
      （每文件 = 10 个三角矩阵 eval 单元的聚合 accuracy，去敏、不含原始预测）。
输出（不读取 coin_metrics.json 的 A_matrix/MAA/BWT 参与计算，仅用于事后交叉验证）：
  - 每 ratio 目录 recomputed_metrics.json（A 矩阵与全部指标）
  - <root>/recomputed_metrics.json（双 ratio 汇总）
  - <root>/comparison.json（0.01 vs 0.10 结构化逐项对比）

指标公式（论文 Section 3.1.3 口径；A[j,i] = 第 j 轮完成 task+replay 后的 checkpoint
在第 i 个任务测试集上的准确率，0-100；T=4）：
  MAA        = (1/T) * Σ_j [ (1/j) * Σ_{i<=j} A[j,i] ]
  CoIN BWT   = (1/T) * Σ_i [ A[T,i] - A[i,i] ]          （正文官方口径）
  Final Avg  = (1/T) * Σ_i A[T,i]
  T-1 BWT    = (1/(T-1)) * Σ_{i<T} [ A[T,i] - A[i,i] ]  （附录补充口径）
 终局旧任务均值 = (1/(T-1)) * Σ_{i<T} A[T,i]
 对角项均值     = (1/T) * Σ_i A[i,i]
 每任务 final-diagonal_i = A[T,i] - A[i,i]

用法: python3 recompute_metrics.py [coin_replay 根目录，默认脚本所在目录]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__)) if len(sys.argv) < 2 else sys.argv[1]
TASKS = ["ScienceQA", "TextVQA", "ImageNet", "GQA"]
T = len(TASKS)
RATIOS = ["ratio_0.10", "ratio_0.01"]


def load_acc_sources(ratio_dir: str) -> dict:
    p = os.path.join(ROOT, ratio_dir, "acc_sources.json")
    src = json.load(open(p))
    A = [[0.0] * T for _ in range(T)]
    for u in src["units"]:
        i = TASKS.index(u["task"])
        j = u["round"] - 1
        A[j][i] = float(u["accuracy"])
    return src["ratio"], A


def metrics(A):
    maa = sum(sum(A[j][:j + 1]) / (j + 1) for j in range(T)) / T
    bwt = sum(A[T - 1][i] - A[i][i] for i in range(T)) / T
    bwt_tminus1 = sum(A[T - 1][i] - A[i][i] for i in range(T - 1)) / (T - 1)
    final_avg = sum(A[T - 1]) / T
    old_mean = sum(A[T - 1][i] for i in range(T - 1)) / (T - 1)
    diag_mean = sum(A[i][i] for i in range(T)) / T
    row_avg = [round(sum(A[j][:j + 1]) / (j + 1), 4) for j in range(T)]
    return {
        "MAA": round(maa, 4), "BWT_coin": round(bwt, 4),
        "BWT_Tminus1": round(bwt_tminus1, 4), "final_avg": round(final_avg, 4),
        "final_old_task_mean": round(old_mean, 4),
        "diagonal_mean": round(diag_mean, 4),
        "row_averages": row_avg,
        "per_task_final_minus_diagonal": {
            t: round(A[T - 1][i] - A[i][i], 4) for i, t in enumerate(TASKS)},
    }


def cross_validate(ratio_dir: str, A, m):
    ref_path = os.path.join(ROOT, ratio_dir, "coin_metrics.json")
    out = {"cross_validation": None}
    if os.path.isfile(ref_path):
        ref = json.load(open(ref_path))
        dA = max(abs(a - b) for ra, rb in zip(A, ref["A_matrix"])
                 for a, b in zip(ra, rb))
        dmaa = abs(m["MAA"] - ref["MAA"])
        dbwt = abs(m["BWT_coin"] - ref["BWT"])
        ok = dA < 1e-6 and dmaa < 1e-4 and dbwt < 1e-4
        out["cross_validation"] = {
            "PASS": ok, "max_A_diff": dA, "MAA_diff": dmaa, "BWT_diff": dbwt,
            "ref": {"MAA": ref["MAA"], "BWT": ref["BWT"]}}
        print(f"[{ratio_dir}] CROSS-VALIDATION {'PASS' if ok else 'FAIL'}  "
              f"maxA={dA:.2e} dMAA={dmaa:.2e} dBWT={dbwt:.2e}")
    return out


def main():
    results = {}
    for rd in RATIOS:
        ratio, A = load_acc_sources(rd)
        m = metrics(A)
        entry = {"ratio": ratio, "tasks": TASKS,
                 "A_matrix": [[round(v, 4) for v in row] for row in A],
                 "A_matrix_full_precision": A, **m}
        entry.update(cross_validate(rd, A, m))
        results[rd] = entry
        with open(os.path.join(ROOT, rd, "recomputed_metrics.json"), "w") as f:
            json.dump({k: v for k, v in entry.items()
                       if k != "A_matrix_full_precision"}, f, indent=2, ensure_ascii=False)
        print(f"=== {rd} (ratio={ratio}) ===")
        for j, row in enumerate(A):
            print(f"round{j+1}: " + ", ".join(f"{v:.4f}" for v in row))
        for k in ("MAA", "BWT_coin", "BWT_Tminus1", "final_avg",
                  "final_old_task_mean", "diagonal_mean"):
            print(f"  {k} = {entry[k]}")

    # comparison（0.01 - 0.10 逐项）
    a10, a01 = results["ratio_0.10"], results["ratio_0.01"]
    unit_units = []  # 逐单元 delta
    for u10 in json.load(open(os.path.join(ROOT, "ratio_0.10", "acc_sources.json")))["units"]:
        u01 = next(u for u in json.load(open(os.path.join(ROOT, "ratio_0.01", "acc_sources.json")))["units"]
                   if u["task"] == u10["task"] and u["round"] == u10["round"])
        unit_units.append({"task": u10["task"], "round": u10["round"],
                           "acc_0.01": u01["accuracy"], "acc_0.10": u10["accuracy"],
                           "delta_0.01_minus_0.10": round(u01["accuracy"] - u10["accuracy"], 4)})
    metric_keys = ("MAA", "BWT_coin", "BWT_Tminus1", "final_avg",
                   "final_old_task_mean", "diagonal_mean")
    cmp = {
        "tasks": TASKS, "formulas": {
            "MAA": "(1/T)*sum_j[(1/j)*sum_{i<=j} A[j,i]]",
            "BWT_coin": "(1/T)*sum_i[A[T,i]-A[i,i]]",
            "final_avg": "(1/T)*sum_i A[T,i]",
            "BWT_Tminus1_appendix": "(1/(T-1))*sum_{i<T}[A[T,i]-A[i,i]]"},
        "per_unit_accuracy": unit_units,
        "metrics": {k: {"ratio_0.01": a01[k], "ratio_0.10": a10[k],
                        "delta_0.01_minus_0.10": round(a01[k] - a10[k], 4)}
                    for k in metric_keys},
        "A_matrix_0.01": a01["A_matrix"], "A_matrix_0.10": a10["A_matrix"],
        "A_matrix_delta_0.01_minus_0.10": [[round(b - a, 4) for a, b in zip(r10, r01)]
                                           for r10, r01 in zip(a10["A_matrix"], a01["A_matrix"])],
    }
    with open(os.path.join(ROOT, "comparison.json"), "w") as f:
        json.dump(cmp, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ROOT, "recomputed_metrics.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "A_matrix_full_precision"}
                   for k, v in results.items()}, f, indent=2, ensure_ascii=False)
    print("WROTE recomputed_metrics.json (per-ratio + top)  and  comparison.json")


if __name__ == "__main__":
    main()
