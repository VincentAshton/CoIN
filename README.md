# CoIN + Replay — 实验结果分支

本分支只保存**实验结果的对外呈现**（去敏、无 checkpoint/模型/数据/原始预测/日志）。
运行代码在 `experiment/coin-replay-presweep-20260903` 分支（锁定，代码与运行时一致）。

## 实验摘要

ScienceQA → TextVQA → ImageNet → GQA 顺序 LoRA 微调 + TRACE 式回放（LLaVA-1.5 7B，
4×A100，task effective batch 896 / replay 56，方案 D replay accum=1，runtime commit
17cfa66）。回放比例扫描：ratio ∈ {0.10, 0.01}，各一轮完整运行，单 seed=1234。

| 指标 | ratio=0.10 | ratio=0.01 |
|---|---|---|
| MAA | 57.5057 | 60.4406 |
| CoIN BWT | +17.2306 | −13.6299 |
| Final Avg | 55.7834 | 46.1925 |
| 终局旧任务均值 | 61.7445 | 47.7000 |

结论（推荐口径）：ratio=0.01 提高部分中间轮次及新任务表现（MAA/对角项），但终局旧任务
均值降低约 14.04 个百分点、最终平均降低约 9.59 个百分点——0.01 不足以满足终局保持目标，
体现稳定性—可塑性权衡。单 seed 描述性证据，非统计显著性结论。

## 文件

```
docs/experiments/coin_replay/
  README.md                      总览（双 A 矩阵/公式/对比表/限制/状态）
  comparison.json                0.01 vs 0.10 结构化逐项对比
  recompute_metrics.py           公开独立重算脚本（双 ratio，CROSS-VALIDATION PASS）
  recomputed_metrics.json        双 ratio 重算汇总
  ratio_0.10/                    0.10 结果包（2026-09-04 COMPLETE）
    README.md / coin_metrics.json / acc_sources.json / run_manifest.sanitized.json /
    validation_report.md / results_sha256.txt / recomputed_metrics.json
  ratio_0.01/                    0.01 结果包（2026-09-05 COMPLETE，同构）
  analysis/
    comparison_r001_vs_r010.md   详细对比分析
    postrun_manifest_diff.json   0.10 vs 0.01 白名单配置 diff
    id_hash_comparison.json      20 个 eval 单元 ID 级 hash 对比
    checkpoint_inventory.json    两组 7 checkpoint 逐路径组件/SHA256 清单
    imagenet_round3_diagnostic.md ImageNet round3=4.02 只读诊断（0.10）
```

## 状态

- ratio=0.10：COMPLETE（2026-09-04，验收通过）
- ratio=0.01：COMPLETE（2026-09-05，验收通过）
- 两组均未重跑、未修改原始结果；runtime HEAD 保持 17cfa66 未前进
