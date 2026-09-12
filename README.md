# CoIN + Replay — Random Replay 抽样实验（LLaVA-1.5 7B）

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350）的 fork。
**当前主线是 random replay 抽样实验**：固定 CoIN 顺序 LoRA 微调 + TRACE 式 replay，
只改变**回放样本的抽取方式与回放比例**，观测 Truth Alignment（MAA / CoIN BWT / final avg）。

> 本文件只做**导航**：不复制指标数字（同一组数字写多处是历史漂移源）。
> 状态与数值一律以机器数据为准 → 展示层 → 人工说明，见下「文档权威层级」。

## 快速导航

| 想看什么 | 去哪 |
|---|---|
| **主实验总入口**（random 0.01 系列 / random 0.10 系列 / 同 seed 配对） | [docs/experiments/coin_replay_random/README.md](docs/experiments/coin_replay_random/README.md) |
| random ratio=0.01 系列 **r001**（开放式系列） | [系列 README](docs/experiments/coin_replay_random_r001/README.md) ｜ [首批五次运行快照报告](docs/experiments/coin_replay_random_r001/SERIES_REPORT.md) |
| random ratio=0.10 系列 **r010**（开放式系列） | [系列 README（A 矩阵 / 提交链 / BWT 口径）](docs/experiments/coin_replay_random_r010/README.md) ｜ [执行报告](docs/experiments/coin_replay_random_r010/EXECUTION_REPORT.md) ｜ [预启动审计](docs/experiments/coin_replay_random_r010/PRELAUNCH_AUDIT.json) |
| 同 seed 配对比较（0.10 − 0.01） | [PAIRED_REPORT.md](docs/experiments/coin_replay_random/PAIRED_REPORT.md) ｜ [paired_comparison.json](docs/experiments/coin_replay_random/paired_comparison.json) |
| 历史探索 / legacy prefix baseline | `results/coin-replay-r010-20260904` 分支（单次结果包）｜`experiment/coin-replay-presweep-20260903` 分支（[REPRODUCE.md](REPRODUCE.md) + `docs/internal/`） |
| 抽样算法 | `sha256_task_seed_python_shuffle_v1`：`task_seed = sha256("<replay_sample_seed>:<task>")`，对完整索引排列取前 `k = floor(N×ratio)`；同 seed 下 0.01 样本集 ⊆ 0.10 样本集（嵌套配对设计） |

## 文档权威层级（避免同一数字多处漂移）

```
权威机器数据      <series>/index.json（+ runs/<run_id>/summary.json）
                  docs/experiments/coin_replay_random/paired_comparison.json
自动生成展示      <series>/README.md 状态行与 registry 表（registry 工具维护）
                  docs/experiments/coin_replay_random/PAIRED_REPORT.md（pair 工具生成）
人工说明          docs/experiments/coin_replay_random/README.md（总入口）
                  SERIES_REPORT.md / EXECUTION_REPORT.md / PRELAUNCH_AUDIT.json
本文件 + 默认分支 README   只做稳定导航，不写指标
```

- push 前自查：`python3 scripts/CoIN_Replay/tools/random_replay_sync_check.py`
  —— 校验各 README/CSV 的 COMPLETE·RUNNING 计数与指标同 index 一致、配对产物与重算一致、
  相对链接可解析、新增文档已脱敏（legacy 已发布审核文档只警告，`--strict` 升级为失败）。
  已接入门禁：`bash scripts/CoIN_Replay/run_tests.sh`（步骤 A7）。

## 分支角色

| 分支 | 内容 |
|---|---|
| `codex/coin-replay-random`（**长期活动分支**） | random 运行代码（ratio 通用 = 0.01/0.10）+ 两个系列 registry + 配对/嵌套/一致性工具 + 全部实验文档 |
| `codex/coin-replay-random-r001` | r001 首次系列的**已审核快照**（冻结，不再追加提交） |
| `CoIN`（默认分支） | 上游代码 + 稳定导航 README（不复制指标） |
| `experiment/coin-replay-presweep-20260903` | prefix 阶段运行代码（锁定 `17cfa66`）+ REPRODUCE.md + `docs/internal/` |
| `results/coin-replay-r010-20260904` | prefix 0.01 / 0.10 单次结果包（纯结果，不携带运行代码） |

已发布结果不得改写：run 六件套、指标、seed、commit 链与注册信息只追加，不删除、不隐藏、
不按指标重排；**系列无固定总运行次数**，FAILED 记录同样永久保留。

## 运行与工具

```bash
# 门禁（bash -n / py_compile / 全部单测 / 文档一致性+脱敏自查；零 GPU 可本地跑）
bash scripts/CoIN_Replay/run_tests.sh

# 单次 random run（示例：ratio=0.10；seed 与 run id 由 registry 决定，详见系列 README）
export SAMPLE_MODE=random RATIO=0.10 REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1
export REPLAY_SAMPLE_SEED=<seed>
export RANDOM_REPLAY_RUN_ID=run_NNNN_seed_<seed>
export CKPT_ROOT=<...>/checkpoints/CoIN_Replay_random/r010/$RANDOM_REPLAY_RUN_ID
export RES_ROOT=<...>/results/CoIN_Replay_random/r010/$RANDOM_REPLAY_RUN_ID
export REPLAY_DATA_DIR=<...>/playground/Replay_random/r010/$RANDOM_REPLAY_RUN_ID
bash scripts/CoIN_Replay/run_replay_exp.sh 0.10
```

启动门在 preflight 之前 fail-fast（SAMPLE_MODE / ratio 允许集合 / ratio 派生 tag 与目录一致 /
seed 31-bit / REPLAY_ACCUM=1 / run ID 格式与内嵌 seed / 三目录互异 / GPUS·WORLD·PORT 自洽）。
评估阶段的路径另有训练前门禁 `coin_lib.py eval-path-audit`（覆盖钩子缺失 / 路径不存在 /
残余相对路径不可解析 → 训练前失败，避免「训练数小时后评估才炸」）。

| 工具 | 作用 |
|---|---|
| `tools/random_replay_registry.py` | 系列注册表（index.json/csv/README 表）；ratio 元数据驱动，追加无上限 |
| `tools/random_replay_finalize.py` | assemble（云端全门验收 + 脱敏导出六件套）/ fill-delta（prefix 与同 seed 配对差值） |
| `tools/random_replay_pair.py` | 生成 paired_comparison.json + PAIRED_REPORT.md（按 seed 交集配对） |
| `tools/random_replay_nested_check.py` | 真实数据 A/B 重建 + 0.01⊆0.10 嵌套验证 |
| `tools/random_replay_sync_check.py` | 文档一致性 + 脱敏自查（push 前 / 门禁 A7） |

## 实验口径

- 任务顺序：ScienceQA → TextVQA → ImageNet → GQA；每轮新任务全量 LoRA 微调（1 epoch）→
  前序任务按比例 replay → 评估全部已学任务（round-end 评估，10 个 eval 单元构成下三角）。
- 训练：4×A100-80G，DeepSpeed ZeRO-3 + CPU offload，bf16+tf32，gradient checkpointing。
- 指标：A 矩阵 → MAA / CoIN BWT / final_avg；单训练 seed，属描述性证据，不作显著性结论。
- **配对口径**：两个 ratio 的控制变量一致（模型 / 数据 / 训练超参 / 训练 seed / replay sample seed /
  抽样算法），但不同系列可能来自**不同代码提交**——该差异经向后兼容测试未见训练语义变化，
  仍作为配对设计限制保留（逐对提交见 PAIRED_REPORT 的 `code_commit` 列）。
- **BWT 口径**：BWT 由各任务 `final − diagonal` 合成，可能被阶段性波动支配（如强 replay 干扰后
  又在终局恢复）——不应把正 BWT 直接解读为「几乎没有遗忘」，须结合 A 矩阵的阶段性变化，
  与 MAA / final_avg 一起报告为稳定性—可塑性权衡。
- 上游 CoIN 代码（模型/数据管线/评估）保持原样，未修改。

> 内部过程记录（prefix 阶段）：`docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md / RUNBOOK.md）
> 与 [REPRODUCE.md](REPRODUCE.md)。
