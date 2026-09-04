# CoIN + Replay — 实验结果分支

本分支只保存**实验结果的对外呈现**（去敏、无 checkpoint/模型/数据/原始预测/日志）。
运行代码在 `experiment/coin-replay-presweep-20260903` 分支（锁定，代码与 0.1 运行时一致）。

## ratio=0.1 结果（COMPLETE，2026-09-04）

ScienceQA → TextVQA → ImageNet → GQA 顺序 LoRA 微调 + 0.1 比例 TRACE 式回放（4×A100，
task effective batch 896 / replay 56，replay 真步 23/85/317，runtime commit 17cfa66）

| | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | **73.26** | — | — | — |
| round2 | 72.86 | **39.03** | — | — |
| round3 | 73.52 | 57.56 | **4.02** | — |
| round4 | 74.13 | 55.91 | 55.19 | **37.90** |

- **MAA = 57.51**（平均每轮已学任务精度）
- **BWT = 17.23**（平均末轮 vs 初学差值）
- round3 ImageNet 4.02 已诊断 = 方法行为（replay 不含当前任务所致，初学 96.93%）——
  见下方 diagnostic

## 文件

```
docs/experiments/coin_replay/ratio_0.1/
  README.md                      实验说明 + 配置 + 状态
  coin_metrics.json              权威指标（A 矩阵/MAA/BWT）
  acc_sources.json               10 个评估单元聚合 accuracy（重算输入）
  recompute_metrics.py           独立重算脚本（交叉验证通过）
  recomputed_metrics.json        重算输出
  run_manifest.sanitized.json    去敏配置快照（路径相对化、无凭据）
  validation_report.md           完整性验收（全项 PASS）
  imagenet_round3_diagnostic.md  ImageNet 4.02 只读诊断
  results_sha256.txt             结果文件 SHA256 清单
```

## 状态

- ratio=0.01：未运行（待批准；将使用与 0.1 相同的运行代码 commit）
- 其他比例/实验：按需在此仓库追加独立结果分支
