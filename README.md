# CoIN + Replay — Random Replay 抽样实验（LLaVA-1.5 7B）

本仓库是 [zackschen/CoIN](https://github.com/zackschen/CoIN)（arXiv:2403.08350）的 fork。
**当前主线是 random replay 抽样实验**：固定 CoIN 顺序 LoRA 微调 + TRACE 式 replay，
只改变**回放样本的抽取方式与回放比例**，观测 Truth Alignment（MAA / CoIN BWT / final avg）。

- **主实验总入口**：[docs/experiments/coin_replay_random/README.md](docs/experiments/coin_replay_random/README.md)
  （random ratio=0.01 系列 + random ratio=0.10 系列 + 同 seed 配对比较）
- 长期分支：**`codex/coin-replay-random`**（同一分支维护两个 ratio、配对工具与运行/注册/发布工具）
- 抽样算法：`sha256_task_seed_python_shuffle_v1` —— `task_seed = sha256("<replay_sample_seed>:<task>")`，
  对完整索引排列取前 `k = floor(N×ratio)`；同 seed 下 0.01 样本集 ⊆ 0.10 样本集（嵌套配对设计）

## 当前状态（随机抽样主线）

| 系列 | ratio | ratio tag | 状态 |
|---|---|---|---|
| random replay | 0.01 | r001 | 当前 5 个 COMPLETE、0 个 RUNNING（开放式系列，可继续追加） |
| random replay | 0.10 | r010 | 见 [r010 README](docs/experiments/coin_replay_random_r010/README.md) 状态节 |
| 同 seed 配对（0.10 − 0.01） | — | — | [paired_comparison.json](docs/experiments/coin_replay_random/paired_comparison.json) / [PAIRED_REPORT.md](docs/experiments/coin_replay_random/PAIRED_REPORT.md) |

- 两个 ratio 的固定项完全一致：`SAMPLE_MODE=random`、`SEED=1234`、`DATA_SEED=1234`、
  `REPLAY_ACCUM=1`；t 段 accum=16（effective batch 896）/ replay 段 accum=1（effective batch 56）。
- **无固定总运行次数**：系列开放；全部运行（含 FAILED）永久公开，不按指标筛选或隐藏。

## 历史探索 / legacy prefix baseline（不属于 random 比例主比较）

以下分支是早期 prefix 抽样探索（回放集 = 历史样本按原顺序取前 k），**保留但不作为
random 比例的主结果**，只作历史对照基线（均为单次运行，非分布）：

| 分支 | 内容 | 数值 |
|---|---|---|
| `results/coin-replay-r010-20260904` | prefix 0.10 与 0.01 单次结果包（A 矩阵/验收/分析） | 0.10：MAA 57.5057 / CoIN BWT +17.2306 / final_avg 55.7834；0.01：MAA 60.4406 / CoIN BWT −13.6299 / final_avg 46.1925 |
| `experiment/coin-replay-presweep-20260903` | prefix 运行代码（锁定 17cfa66）+ 复现手册 REPRODUCE.md + docs/internal/ | — |

口径提示：prefix 基线各为单次运行，与 random 多次运行的差值只能作描述性比较。

## 运行与工具

```bash
# 门禁（bash -n / py_compile / 全部单测；零 GPU 可本地跑）
bash scripts/CoIN_Replay/run_tests.sh

# 单次 random run（示例：ratio=0.10, seed=358341059；详见系列 README）
export SAMPLE_MODE=random RATIO=0.10 REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1
export REPLAY_SAMPLE_SEED=358341059
export RANDOM_REPLAY_RUN_ID=run_0001_seed_358341059
export CKPT_ROOT=<...>/checkpoints/CoIN_Replay_random/r010/$RANDOM_REPLAY_RUN_ID
export RES_ROOT=<...>/results/CoIN_Replay_random/r010/$RANDOM_REPLAY_RUN_ID
export REPLAY_DATA_DIR=<...>/playground/Replay_random/r010/$RANDOM_REPLAY_RUN_ID
bash scripts/CoIN_Replay/run_replay_exp.sh 0.10
```

启动门在 preflight 之前 fail-fast（SAMPLE_MODE / ratio 允许集合 / ratio 派生 tag 与目录一致 /
seed 31-bit / REPLAY_ACCUM=1 / run ID 格式与内嵌 seed / 三目录互异 / GPUS·WORLD·PORT 自洽）。

| 工具 | 作用 |
|---|---|
| `tools/random_replay_registry.py` | 系列注册表（index.json/csv/README 表）；ratio 元数据驱动，追加无上限 |
| `tools/random_replay_finalize.py` | assemble（云端全门验收 + 脱敏导出六件套）/ fill-delta（prefix 与同 seed 配对差值） |
| `tools/random_replay_pair.py` | 生成 paired_comparison.json + PAIRED_REPORT.md（按 seed 交集配对） |
| `tools/random_replay_nested_check.py` | 真实数据 A/B 重建 + 0.01⊆0.10 嵌套验证 |

## 实验口径

- 任务顺序：ScienceQA → TextVQA → ImageNet → GQA；每轮新任务全量 LoRA 微调（1 epoch）→
  前序任务按比例 replay → 评估全部已学任务（round-end 评估，10 个 eval 单元构成下三角）。
- 训练：4×A100-80G，DeepSpeed ZeRO-3 + CPU offload，bf16+tf32，gradient checkpointing。
- 指标：A 矩阵 → MAA / CoIN BWT / final avg；单训练 seed（描述性证据，不作显著性结论）。
- 上游 CoIN 代码（模型/数据管线/评估）保持原样，未修改。

> 详细内部记录（prefix 阶段）：`docs/internal/`（HANDOFF.md / EXPERIMENT_LOG.md /
> RUNBOOK.md / dataset.md）与 [REPRODUCE.md](REPRODUCE.md)。
