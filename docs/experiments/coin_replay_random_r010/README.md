# Random Replay ratio=0.10 —— 系列 r010（开放式系列）

长期实验系列：**ratio=0.10 随机 replay 样本选择**（固定 SAMPLE_MODE=random、
RATIO=0.10、REPLAY_ACCUM=1、SEED=1234、DATA_SEED=1234；每次只变
`REPLAY_SAMPLE_SEED`）。与 [ratio=0.01 系列 r001](../coin_replay_random_r001/README.md)
构成**同 seed 嵌套配对**（同 seed 下 0.01 的样本集是 0.10 的子集），用于区分
「抽样方式」与「回放比例」两个效应。

- 系列总入口：[docs/experiments/coin_replay_random/README.md](../coin_replay_random/README.md)
- 同 seed 配对结果：[paired_comparison.json](../coin_replay_random/paired_comparison.json) /
  [PAIRED_REPORT.md](../coin_replay_random/PAIRED_REPORT.md)
- 长期分支：`codex/coin-replay-random`（同一分支同时维护 0.01 与 0.10）
- ratio tag：**r010**（由 ratio 数值派生，`coin_lib.ratio_tag`：0.01→r001、0.10→r010）

## 系列纪律

- 每次收到运行指令最多完成**一个**正式实验：登记 RUNNING → push → 训练 → 验收 →
  发布 COMPLETE → push → 停止（除非用户一次性授权连续多轮）。
- **无预设目标运行总数**：系列开放，可继续追加；不设「最多 N 次」限制。
- seed 由 registry 在启动前以 `SystemRandom` 生成 31-bit 正整数；首次 0.10 运行按设计
  复用 r001 的已公开 seed（同 seed 配对），**不是**根据实验结果选 seed。
- 同一 ratio 内禁止复用 seed；**跨 ratio 允许同 seed**（配对设计需要）。
- 失败按技术原因记 FAILED；记录只追加，不删除、不隐藏、不按指标重排。
- 失败恢复：不换 seed、不新建 run、不删 checkpoint/round marker；用 validate-round
  判断进度，同 seed 同目录恢复。语义性改动（算法/超参/数据/模型/任务序/ratio/指标）
  必须另开新系列，且停止当前实验并如实记录。

## 启动姿势（fail-fast 门）

正式运行必须显式提供 `RANDOM_REPLAY_RUN_ID=run_NNNN_seed_<seed>`（与
`REPLAY_SAMPLE_SEED` 一致）；`run_replay_exp.sh` 在 preflight 之前执行通用
random 门：SAMPLE_MODE=random、ratio ∈ {0.01, 0.10}（数值归一化）、ratio 派生
tag 与目录一致、seed 为正 31-bit、REPLAY_ACCUM=1、run ID 格式与内嵌 seed 一致、
CKPT/RES/REPLAY 三目录互异且同属该 tag 与 run_id、GPUS/WORLD/MASTER_PORT 自洽。
不设 `RANDOM_REPLAY_RUN_ID` = 旧 prefix 语义（行为不变）。

目录布局（`<run_id>` = run_NNNN_seed_<seed>）：

```
checkpoints/CoIN_Replay_random/r010/<run_id>/   round<j>_{task,replay}_llava_lora
results/     CoIN_Replay_random/r010/<run_id>/   eval 产物 + round/run manifests + coin_metrics.json
playground/  Replay_random/r010/<run_id>/        round{2..4}_train.json(+.manifest.json)
```

## 抽样算法（sha256_task_seed_python_shuffle_v1）

与 r001 完全一致（同一算法常量、同一 task seed 派生规则）：

1. `task_seed = sha256("<replay_sample_seed>:<task>")`（不含 round）。
2. `rng = random.Random(int(task_seed, 16))`；对完整 `indices = list(range(N))` 洗牌。
3. `selected_indices = shuffled[:k]`，`k = floor(N*ratio)`（0.10 时 k 为 0.01 的 10 倍量级）。
4. 先全排列再取前 k ⇒ 同 seed 下 0.01 集合 ⊆ 0.10 集合。

实现与验证：`scripts/CoIN_Replay/build_replay_data.py`（`--sample-mode random --seed`）、
`tools/random_replay_nested_check.py`（真实数据 A/B 重建 + 0.01⊆0.10 前缀/子集断言）。

## 提交协议（A → C → D）

- **A** = RUNNING 注册提交（`tools/random_replay_registry.py <本目录> register
  --seed <seed> [--paired-with run_0001_seed_358341059 --paired-index ../coin_replay_random_r001/index.json]`）
- **C** = 六件套结果提交（`tools/random_replay_finalize.py assemble` 产物 commit+push）
- **D** = registry COMPLETE 提交（携带 A、C 的真实 hash；禁伪造「当前提交自己的 SHA」）

发布物只允许六个轻量文件：`coin_metrics.json`、`acc_sources.json`、
`replay_selection_summary.json`、`run_manifest.sanitized.json`、`summary.json`、
`validation_report.md`（存放于 `runs/<run_id>/`）。禁止提交 checkpoint/权重/
优化器态/完整 replay JSON/完整预测/原始日志/缓存/含绝对云端路径的文件。

## 状态

当前 0 个 COMPLETE，0 个 RUNNING；无预设目标运行总数。

<!-- registry-table:start -->
| Run | Replay sample seed | MAA | BWT | Status | Result commit |
|-----|--------------------|-----|-----|--------|---------------|
<!-- registry-table:end -->

（表由 `tools/random_replay_registry.py` 自动维护；`index.json` 为权威记录，含
started_at/completed_at/result_directory/code_commit/config_hash/error_summary/
registration_commit/result_commit/ratio/ratio_tag/paired_with 全字段 + prefix 基线数值。）

## 对照基线（prefix 系列；历史探索，非 random 比例主比较）

| 基线 | MAA | CoIN BWT | final avg |
|---|---|---|---|
| prefix ratio=0.10（2026-09-04） | 57.5057 | +17.2306 | 55.7834 |
| prefix ratio=0.01（2026-09-05） | 60.4406 | −13.6299 | 46.1925 |

差值口径：`summary.json` deltas 中 `vs_prefix_*` = 本 run − 对应 prefix 基线；
`vs_paired_*` = 高 ratio − 低 ratio（即 random-0.10 − random-0.01，同 seed 配对）。
