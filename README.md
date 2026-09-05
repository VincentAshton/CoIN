# CoIN + Replay — 持续学习回放比例实验（LLaVA-1.5 7B）

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350）的 fork，
用于跑 **CoIN 顺序 LoRA 微调 + TRACE 式 Replay** 的回放比例扫描实验
（研究问题：回放比例降到多少时 Truth Alignment 明显下降）。

> 实验记录与完整内部文档已归档至 `docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md /
> RUNBOOK.md / dataset.md）；本 README 只保留对外可见的导航与状态。
> **想从零复现本实验 → 直接读 [REPRODUCE.md](REPRODUCE.md)**（环境/数据/门禁/运行/验收全流程）。

## 实验概况

- 任务顺序（4 任务持续学习，CoIN 前 4 任务）：
  ScienceQA → TextVQA → ImageNet → GQA
- 每轮：新任务全量 LoRA 微调（1 epoch）→ 前序任务按比例 prefix 回放（round≥2）→ 评估全部已学任务
- 模型：LLaVA-1.5-7B（vicuna-7b-v1.5 + CLIP-L/14-336），LoRA r=192 α=256
- 训练：4×A100-80G，DeepSpeed ZeRO-3 + CPU offload，bf16+tf32，grad checkpoint
- **task 段**：accum=16 → effective batch 896（论文口径）
- **replay 段**：accum=1（`REPLAY_ACCUM=1`，全比例统一）→ effective batch 56
  —— 设计修订：DS 0.14 在 accum=16 下不提交短 replay 尾部（N<896 时 0 真实更新），
  方案 D 以全比例统一的 replay accum=1 修复（详见 docs/internal/EXPERIMENT_LOG.md）
- 评估口径：Truth Alignment（每任务官方 eval + 严格 prediction 校验）
- 指标：A 矩阵 → MAA / CoIN BWT（单 seed=1234；10 个 eval 单元为三角矩阵交集）

## 分支导航

| 分支 | 内容 | 位置 |
|---|---|---|
| **CoIN**（默认） | 入口页 + 上游代码 | [GitHub 首页](https://github.com/VincentAshton/CoIN) |
| `experiment/coin-replay-presweep-20260903` | **运行代码（锁定）+ 工具/测试 + 复现手册** | 本分支（当前） |
| `results/coin-replay-r010-20260904` | **正式结果**（双 ratio 矩阵/验收/对比分析） | [点此](https://github.com/VincentAshton/CoIN/tree/results/coin-replay-r010-20260904/docs/experiments/coin_replay) |

## 当前状态（2026-09-05）

- **ratio=0.10：COMPLETE** —— MAA=57.5057，CoIN BWT=+17.2306，final avg=55.7834
- **ratio=0.01：COMPLETE** —— MAA=60.4406，CoIN BWT=−13.6299，final avg=46.1925
- 对比结论：0.01 提高部分中间轮次/新任务表现但终局旧任务均值低 ~14.04 点、最终平均低
  ~9.59 点 → 不足以满足终局保持目标（稳定性—可塑性权衡）；单 seed 描述性证据
- 运行代码锁定 commit：`17cfa66`（= 069b608，experiment 分支代码自此后未变）
- 已知方法行为：replay 不含当前任务 → 新学任务被回放干扰（0.10 round3 ImageNet 初学
  96.93% → replay 后 4.02 → round4 恢复 55.19），诊断见结果分支 analysis/

## 快速运行（细节与门禁见 REPRODUCE.md）

```bash
export REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1 GPUS=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3
export PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
bash scripts/CoIN_Replay/run_sweep.sh 0.10     # 一次一个比例；先门禁后正式
```

结果落盘：`results/CoIN_Replay/ratio_<r>/`（10 个 eval 单元 + coin_metrics.json +
run_manifest.json + markers）；`scripts/CoIN_Replay/` 下编排/门禁/测试可独立运行
（run_tests.sh 期望 Ran 89）。
