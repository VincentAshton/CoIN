# CoIN + Replay — 持续学习回放比例实验（LLaVA-1.5 7B）

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350）的 fork，
用于跑 **CoIN 顺序 LoRA 微调 + TRACE 式 Replay** 的回放比例扫描实验
（研究问题：回放比例降到多少时 Truth Alignment 明显下降）。

> 实验记录与完整内部文档已归档至 `docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md /
> RUNBOOK.md），本 README 只保留对外可见的导航与状态。

## 实验概况

- 任务顺序（4 任务持续学习，CoIN 前 4 任务）：
  ScienceQA → TextVQA → ImageNet → GQA
- 每轮：新任务全量 LoRA 微调（1 epoch）→ 前序任务按比例 prefix 回放（round≥2）→ 评估全部已学任务
- 模型：LLaVA-1.5-7B（vicuna-7b-v1.5 + CLIP-L/14-336），LoRA r=192 α=256
- 训练：4×A100-80G，DeepSpeed ZeRO-3 + CPU offload，bf16+tf32，grad checkpoint
- **task 段**：accum=16 → effective batch 896（论文口径）
- **replay 段**：accum=1（`REPLAY_ACCUM=1`，全比例统一）→ effective batch 56
  —— 设计修订：DS 0.14 在 accum=16 下不提交短 replay 尾部（N<896 时 0 真实更新），
  方案 D 以全比例统一的 replay accum=1 修复（详见 docs/internal/EXPERIMENT_LOG.md §4.16）
- 评估口径：Truth Alignment（每任务官方 eval + 严格 prediction 校验）
- 指标：A 矩阵 → MAA / BWT

## 分支导航

| 分支 | 内容 | 位置 |
|---|---|---|
| **CoIN**（默认） | 本入口页 + 上游代码 | 当前页 |
| `experiment/coin-replay-presweep-20260903` | **运行代码（锁定）** + 工具/测试 | 本分支（当前） |
| `results/coin-replay-r010-20260904` | **ratio=0.1 正式结果**（矩阵/验收/诊断） | [点此](https://github.com/VincentAshton/CoIN/tree/results/coin-replay-r010-20260904/docs/experiments/coin_replay/ratio_0.1) |

## 当前状态（2026-09-04）

- **ratio=0.1：COMPLETE** —— MAA=57.51，BWT=17.23；
  A 矩阵与完整验收见 results 分支（上方链接）
- ratio=0.01：**未运行**（需另行批准；运行代码与 0.1 保持同 commit）
- 0.1 运行代码 commit：`17cfa66`（experiment 分支，代码自此后未变）
- 已知方法行为：replay 不含当前任务 → 新学任务被回放干扰（round3 ImageNet 96.93% →
  replay 后 4.02 → round4 恢复 55.19），诊断见结果分支 imagenet_round3_diagnostic.md

## 复现

```bash
# 数据/模型准备与完整环境步骤：docs/internal/RUNBOOK.md
cd CoIN
export REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1 GPUS=0,1,2,3
export PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
bash scripts/CoIN_Replay/run_sweep.sh 0.1     # 单比例；0.01 需另行批准
```

结果落盘：`results/CoIN_Replay/ratio_<r>/`（A 矩阵/coin_metrics.json + 每轮评估单元 +
run_manifest.json 配置快照）；`scripts/CoIN_Replay/` 下编排/门禁/测试可独立运行（run_tests.sh）。
