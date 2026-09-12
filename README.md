# CoIN（fork）— 持续学习 + 回放比例实验

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350，LLaVA-1.5 顺序
LoRA 微调持续学习基准）的个人 fork。

**当前主线 = random replay 抽样实验**（长期分支 `codex/coin-replay-random`）：固定 CoIN 四任务顺序
学习 + TRACE 式 replay，只改变**回放样本的抽取方式与回放比例**，观测 Truth Alignment
（MAA / CoIN BWT / final_avg）。

> 本文件是**稳定导航页**：不复制任何指标数字——指标会随每次新 run 变化，写在这里必然与
> registry 漂移。状态与数值一律去下面「随机抽样主线」的链接里看（权威数据 = `index.json`）。

## 随机抽样主线（`codex/coin-replay-random` 分支）

| 想看什么 | 去哪 |
|---|---|
| **总入口**（random 0.01 系列 / random 0.10 系列 / 同 seed 配对 / 口径与限制） | [docs/experiments/coin_replay_random/README.md](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random/README.md) |
| ratio=0.01 系列 **r001** registry（状态与逐 run 数值） | [index.json](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r001/index.json) ｜ [系列 README](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r001/README.md) ｜ [首批五次运行快照报告](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r001/SERIES_REPORT.md) |
| ratio=0.10 系列 **r010** registry（状态与逐 run 数值） | [index.json](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r010/index.json) ｜ [系列 README（A 矩阵 / 提交链 / BWT 口径）](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r010/README.md) ｜ [执行报告](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r010/EXECUTION_REPORT.md) ｜ [预启动审计](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random_r010/PRELAUNCH_AUDIT.json) |
| **同 seed 配对比较**（0.10 − 0.01） | [PAIRED_REPORT.md](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random/PAIRED_REPORT.md) ｜ [paired_comparison.json](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/docs/experiments/coin_replay_random/paired_comparison.json) |
| 运行/注册/验收/发布工具与门禁 | [活动分支 README](https://github.com/VincentAshton/CoIN/blob/codex/coin-replay-random/README.md) |

抽样算法：`sha256_task_seed_python_shuffle_v1` —— `task_seed = sha256("<replay_sample_seed>:<task>")`，
对完整索引排列取前 `k = floor(N×ratio)`；同 seed 下 0.01 样本集 ⊆ 0.10 样本集（嵌套配对设计）。
系列为**开放式**：无固定总运行次数，全部运行（含 FAILED）永久公开，不按指标筛选或隐藏。

## 历史探索 / legacy prefix baseline

早期 prefix 抽样探索（回放集 = 按原顺序取前 k），**保留但不作为 random 比例的主结果**，
只作历史对照基线（均为单次运行，非分布）：

| 分支 | 内容 |
|---|---|
| [`experiment/coin-replay-presweep-20260903`](https://github.com/VincentAshton/CoIN/tree/experiment/coin-replay-presweep-20260903) | prefix 运行代码（锁定 `17cfa66`）+ [REPRODUCE.md](https://github.com/VincentAshton/CoIN/blob/experiment/coin-replay-presweep-20260903/REPRODUCE.md) + `docs/internal/`（HANDOFF / EXPERIMENT_LOG / RUNBOOK） |
| [`results/coin-replay-r010-20260904`](https://github.com/VincentAshton/CoIN/tree/results/coin-replay-r010-20260904) | prefix ratio=0.10 与 0.01 的单次结果包（A 矩阵 / 验收 / 分析；不携带运行代码） |

## 分支角色

| 分支 | 内容 |
|---|---|
| `codex/coin-replay-random` | **长期活动分支**：random 运行代码（ratio 通用 0.01/0.10）+ 两个系列 registry + 配对/嵌套/一致性工具 + 实验文档 |
| `codex/coin-replay-random-r001` | r001 首次系列的已审核快照（冻结，不再追加） |
| `CoIN`（默认分支，本分支） | 上游代码 + 本稳定导航 README |
| `experiment/coin-replay-presweep-20260903` | prefix 阶段运行代码 + 复现手册 + 内部记录 |
| `results/coin-replay-r010-20260904` | prefix 单次结果包 |

## 口径提醒（方法层面，与具体数值无关）

- 指标为 A 矩阵派生的 MAA / CoIN BWT / final_avg；单训练 seed，属**描述性证据**，
  不作统计显著性结论；prefix 基线是单次运行而非分布，差值只能描述性比较。
- **配对限制**：同 seed 配对的两个 ratio 控制变量一致（模型 / 数据 / 训练超参 / 训练 seed /
  replay sample seed / 抽样算法），但两侧可能来自**不同代码提交**（逐对提交见 PAIRED_REPORT）。
- **BWT 解释**：BWT 由各任务 `final − diagonal` 合成，可能被阶段性波动支配，正 BWT 不等于
  「几乎没有遗忘」；应与 MAA / final_avg 一起解读为稳定性—可塑性权衡。
- 上游 CoIN 代码（模型/数据管线/评估）保持原样，未修改。
