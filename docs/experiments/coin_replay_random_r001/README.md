# Random Replay ratio=0.01 —— ongoing experiments（codex 系列）

长期实验系列：**ratio=0.01 随机 replay 样本选择**（固定 SAMPLE_MODE=random、
RATIO=0.01、REPLAY_ACCUM=1、SEED=1234、DATA_SEED=1234；每次只变
REPLAY_SAMPLE_SEED）。基线分支 `experiment/coin-replay-presweep-20260903`；
本系列长期分支 **`codex/coin-replay-random-r001`**（不按 seed 建分支）。

## 运行纪律（单次运行）

- 每次收到运行指令最多完成**一个**正式实验：登记 RUNNING → push 确认 → 训练 →
  验收 → 发布 COMPLETE → push → **停止**。绝不自动开始下一次。
- seed 用 `secrets.SystemRandom` 生成 31-bit 正整数，登记前查 index 确保未用过；
  不以实验结果修改/重生成 seed。
- 失败恢复：不换 seed、不新建 run、不删 checkpoint/round marker；用 validate-round
  判断进度，同 seed 同目录恢复；语义性改动（算法/超参/数据/模型/任务序/ratio/指标）
  必须另开 r002 系列。
- index 只追加或 RUNNING → COMPLETE/FAILED；失败记录保留；run_number 单调递增。

## 抽样算法（sha256_task_seed_python_shuffle_v1）

对每个任务（round≥2 的历史任务）：

1. `task_seed = sha256("<replay_sample_seed>:<task>")`（UTF-8 hexdigest；**不用** Python
   `hash(task)`；task seed 不含 round → 同任务在不同 round 用同一排列）。
2. `rng = random.Random(int(task_seed, 16))`；对完整 `indices = list(range(N))` 洗牌。
3. `selected_indices = shuffled[:k]`，`k = floor(N*ratio)` 严格不变；
   `picked = samples[i] for i in selected_indices`。
4. 先全排列再取前 k ⇒ 同 seed 下 ratio=0.01 集合 ⊆ ratio=0.10 集合（nested-with 校验）。

prefix 模式（`--sample-mode prefix`）与旧实现完全兼容（`samples[:k]`，输出逐字节一致）。
实现：`scripts/CoIN_Replay/build_replay_data.py`；编排：`run_replay_exp.sh`
（export REPLAY_SAMPLE_SEED，构建器收 `--sample-mode/--seed`）；
manifest/config hash/resume 校验见 `coin_lib.py`（CONFIG_FIELDS 含 replay_sample_seed）。

## 每 run 布局（训练机云端；本地 .gitignore 已覆盖不入库）

```
checkpoints/CoIN_Replay_random/r001/<run_id>/   round<j>_{task,replay}_llava_lora
results/     CoIN_Replay_random/r001/<run_id>/   eval 产物 + run/round manifests + coin_metrics.json
playground/  Replay_random/r001/<run_id>/        round{2..4}_train.json(+.manifest.json)
```

## 发布物（本分支 docs/experiments/coin_replay_random_r001/runs/<run_id>/）

只允许：`summary.json`、`coin_metrics.json`、`acc_sources.json`、
`run_manifest.sanitized.json`、`replay_selection_summary.json`、`validation_report.md`。
绝对禁止提交 checkpoint/adapter 权重/optimizer 态/完整 replay JSON/完整预测/原始日志/
图片模型数据集/cache/含绝对路径·主机名·IP·凭据文件。完整 sidecar manifest 留在训练机，
摘要记录其 SHA256（抽样算法+seed+source SHA 已记录 ⇒ 样本集合可重建）。

## 工具

- `scripts/CoIN_Replay/tools/random_replay_registry.py` —— index.json/csv/README 表维护
  （register / complete / list；append-only 校验）。
- `scripts/CoIN_Replay/tools/random_replay_finalize.py` —— assemble（云端验收+脱敏瘦身导出，
  任务书十一全部检查通过才写发布目录）/ fill-delta（本地填 prefix 基线差值）。

## 状态

<!-- registry-table:start -->
| Run | Replay sample seed | MAA | BWT | Status | Result commit |
|-----|--------------------|-----|-----|--------|---------------|
<!-- registry-table:end -->

（表由 random_replay_registry.py 自动维护；index.json 为权威记录，含
started_at/completed_at/result_directory/code_commit/config_hash/error_summary/
registration_commit/result_commit 全字段 + prefix 基线数值。）

## 对照基线（prefix 系列，results/coin-replay-r010-20260904 分支）

| 基线 | MAA | CoIN BWT | final avg |
|---|---|---|---|
| prefix ratio=0.10（2026-09-04） | 57.5057 | +17.2306 | 55.7834 |
| prefix ratio=0.01（2026-09-05） | 60.4406 | −13.6299 | 46.1925 |

差值口径（summary.json deltas）：本 run − 对应基线，四项全精度指标逐项。
