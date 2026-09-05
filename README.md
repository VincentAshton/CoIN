# CoIN（fork）— 持续学习 + 回放比例实验

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350，
LLaVA-1.5 顺序 LoRA 微调持续学习基准）的个人 fork，用于运行 **TRACE 式 Replay 回放比例
扫描实验**：ScienceQA → TextVQA → ImageNet → GQA 四任务顺序学习 + 前序任务按比例回放，
观测 Truth Alignment（MAA/BWT）随回放比例的变化。

## 快速导航

| 想看什么 | 去哪 |
|---|---|
| **实验结果（ratio=0.10 与 0.01 均完成）** | → `results/coin-replay-r010-20260904` 分支<br>[github.com/…/tree/results/coin-replay-r010-20260904](https://github.com/VincentAshton/CoIN/tree/results/coin-replay-r010-20260904)（双 A 矩阵 / MAA / BWT / 对比分析 / 验收） |
| **实验运行代码与工具** | → `experiment/coin-replay-presweep-20260903` 分支（编排/门禁/测试 + 实验导向 README） |
| **内部过程记录**（交接/逐日日志/运行手册） | `experiment` 分支的 `docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md / RUNBOOK.md） |

## 实验状态（2026-09-05）

- ratio=0.10：**COMPLETE**（MAA=57.5057，CoIN BWT=+17.2306，final avg=55.7834）
- ratio=0.01：**COMPLETE**（MAA=60.4406，CoIN BWT=−13.6299，final avg=46.1925）
- 对比结论（单 seed 描述性证据）：0.01 提高部分中间轮次/新任务表现，但终局旧任务均值
  低 14.04 个百分点、最终平均低 9.59 个百分点——不足以满足终局保持目标，体现
  稳定性—可塑性权衡。详见 results 分支 `docs/experiments/coin_replay/README.md`。
- 运行代码锁定于 commit `17cfa66`（结果分支与 runtime 分离，代码未变）

## 复现要点

task 段 gradient_accumulation=16（effective batch 896）；replay 段 accum=1 全比例统一
（`REPLAY_ACCUM=1`，effective batch 56——修复 DS 0.14 短 replay 尾部 0 真实更新的设计修订）。
完整步骤见 experiment 分支 `docs/internal/RUNBOOK.md`。

上游 CoIN 代码（模型/数据管线/评估）保持原样，未修改。
