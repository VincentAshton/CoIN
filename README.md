# CoIN（fork）— 持续学习 + 回放比例实验

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350，
LLaVA-1.5 顺序 LoRA 微调持续学习基准）的个人 fork，用于运行 **TRACE 式 Replay 回放比例
扫描实验**：ScienceQA → TextVQA → ImageNet → GQA 四任务顺序学习 + 前序任务按比例回放，
观测 Truth Alignment（MAA/BWT）随回放比例的变化。

## 快速导航

| 想看什么 | 去哪 |
|---|---|
| **实验结果（ratio=0.1 已完成）** | → `results/coin-replay-r010-20260904` 分支<br>[github.com/…/tree/results/coin-replay-r010-20260904](https://github.com/VincentAshton/CoIN/tree/results/coin-replay-r010-20260904)（A 矩阵 / MAA=57.51 / BWT=17.23 / 验收 / 诊断） |
| **实验运行代码与工具** | → `experiment/coin-replay-presweep-20260903` 分支（编排/门禁/测试 + 实验导向 README） |
| **内部过程记录**（交接/逐日日志/运行手册） | `experiment` 分支的 `docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md / RUNBOOK.md） |

## 实验状态（2026-09-04）

- ratio=0.1：**COMPLETE**（MAA=57.51，BWT=17.23，验收全项通过）
- ratio=0.01：未运行（待批准；将与 0.1 使用同一运行代码 commit）
- 运行代码锁定于 commit `17cfa66`（仅文档后续整理，代码未变）

## 复现要点

task 段 gradient_accumulation=16（effective batch 896）；replay 段 accum=1 全比例统一
（`REPLAY_ACCUM=1`，effective batch 56——修复 DS 0.14 短 replay 尾部 0 真实更新的设计修订）。
完整步骤见 experiment 分支 `docs/internal/RUNBOOK.md`。

上游 CoIN 代码（模型/数据管线/评估）保持原样，未修改。
