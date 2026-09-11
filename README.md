# CoIN（fork）— 持续学习 + 回放比例实验

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350，
LLaVA-1.5 顺序 LoRA 微调持续学习基准）的个人 fork。

**当前主线 = random replay 抽样实验**（长期分支 `codex/coin-replay-random`）：固定
CoIN 四任务顺序学习 + TRACE 式 replay，只改变**回放样本抽取方式与回放比例**，
观测 Truth Alignment（MAA / CoIN BWT / final avg）。任务序 ScienceQA → TextVQA →
ImageNet → GQA；抽样算法 `sha256_task_seed_python_shuffle_v1`（同 seed 下 0.01 样本集
⊆ 0.10 样本集，构成嵌套配对设计）。

## 快速导航

| 想看什么 | 去哪 |
|---|---|
| **Random replay 总入口**（0.01 系列 / 0.10 系列 / 同 seed 配对） | → `codex/coin-replay-random` 分支 [docs/experiments/coin_replay_random/README.md](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random/README.md) |
| random ratio=0.01 系列（r001；当前 5 个 COMPLETE） | [系列 README](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r001/README.md) ｜ [五次运行快照报告](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r001/SERIES_REPORT.md) |
| random ratio=0.10 系列（r010；当前 1 个 COMPLETE，MAA 60.122 / CoIN BWT +28.7287 / final_avg 66.402） | [系列 README（含 A 矩阵与提交链）](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r010/README.md) |
| 同 seed 配对比较（0.10 − 0.01；当前 1 对，seed 358341059） | [PAIRED_REPORT.md（ΔMAA −2.7310 / ΔBWT +20.7933 / Δfinal_avg +4.9499）](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random/PAIRED_REPORT.md) ｜ [paired_comparison.json](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random/paired_comparison.json) |
| **历史探索 / legacy prefix baseline**（回放集=按时序取前 k，非 random 主比较） | → `results/coin-replay-r010-20260904` 分支 [结果包](https://github.com/VincentAshton/CoIN/tree/results/coin-replay-r010-20260904/docs/experiments/coin_replay) ｜ 运行代码与手册：`experiment/coin-replay-presweep-20260903` 分支（[REPRODUCE.md](https://github.com/VincentAshton/CoIN/blob/experiment/coin-replay-presweep-20260903/REPRODUCE.md)） |
| 内部过程记录（prefix 阶段） | `experiment` 分支的 `docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md / RUNBOOK.md） |

## 实验状态（2026-09-11）

**random replay 抽样（主线，开放式系列：不设固定总运行次数，全部运行含 FAILED 永久公开）**

| 系列 | ratio | ratio tag | 状态 |
|---|---|---|---|
| r001 | 0.01 | r001 | 当前 5 个 COMPLETE、0 个 RUNNING |
| r010 | 0.10 | r010 | 当前 1 个 COMPLETE、0 个 RUNNING（MAA 60.122 / CoIN BWT +28.7287 / final_avg 66.402） |
| 配对（同 seed，0.10 − 0.01） | — | — | 当前 1 对（seed 358341059）：ΔMAA −2.7310 / ΔBWT **+20.7933** / Δfinal_avg +4.9499 |

- r001（ratio=0.01，首批五次运行，2026-09-09~10）：MAA 61.9287 ± 1.3164、
  CoIN BWT **+11.3442 ± 5.0428**、final_avg 62.0781 ± 1.4895（mean ± sample sd，n=5）；
  五次 BWT 全部为正（+6.5542 ~ +16.8100）。对照单次 prefix 0.01 的 −13.6299 为
  **描述性证据**（非同 seed 配对、n=5），不构成统计显著性结论。
- 固定项（两个 ratio 一致）：`SAMPLE_MODE=random`、`SEED=1234`、`DATA_SEED=1234`、
  `REPLAY_ACCUM=1`；task 段 accum=16（effective batch 896）、replay 段 accum=1
  （effective batch 56，修复 DS 0.14 短 replay 尾部 0 真实更新的设计修订）。

**prefix 抽样（历史探索 / legacy baseline，均单次运行，非分布）**

- ratio=0.10：MAA 57.5057、CoIN BWT +17.2306、final_avg 55.7834
- ratio=0.01：MAA 60.4406、CoIN BWT −13.6299、final_avg 46.1925
- 结论（单 seed 描述性证据）：0.01 提高部分中间轮次/新任务表现，但终局旧任务均值低
  14.04 个百分点、最终平均低 9.59 个百分点——不足以满足终局保持目标（稳定性—可塑性权衡）。

## 复现与工具（`codex/coin-replay-random` 分支）

- 单次 random run：`run_replay_exp.sh <ratio>` + 显式 `RANDOM_REPLAY_RUN_ID` 与三个
  per-run 目录（启动门在 preflight 前 fail-fast 校验 ratio/tag/seed/目录/accum/port/GPU）。
- 门禁：`bash scripts/CoIN_Replay/run_tests.sh`（bash -n / py_compile / 全套单测；
  本地零依赖环境 18 skip = 9 torch + 9 PIL，云端完整依赖环境须真实执行）。
- 发布：`tools/random_replay_finalize.py assemble`（全门 PASS 才原子换入六件套）→
  registry A/C/D 提交协议 → `tools/random_replay_pair.py` 重算配对报告。
- prefix 阶段的完整复现步骤见 `experiment` 分支的 `docs/internal/RUNBOOK.md`；上游 CoIN
  代码（模型/数据管线/评估）保持原样，未修改。
