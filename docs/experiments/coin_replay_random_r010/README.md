# Random Replay ratio=0.10 —— 系列 r010（开放式系列）

长期实验系列：**ratio=0.10 随机 replay 样本选择**（固定 SAMPLE_MODE=random、
RATIO=0.10、REPLAY_ACCUM=1、SEED=1234、DATA_SEED=1234；每次只变
`REPLAY_SAMPLE_SEED`）。与 [ratio=0.01 系列 r001](../coin_replay_random_r001/README.md)
构成**同 seed 嵌套配对**（同 seed 下 0.01 的样本集是 0.10 的子集），用于区分
「抽样方式」与「回放比例」两个效应。

- 系列总入口：[docs/experiments/coin_replay_random/README.md](../coin_replay_random/README.md)
- 同 seed 配对结果：[paired_comparison.json](../coin_replay_random/paired_comparison.json) /
  [PAIRED_REPORT.md](../coin_replay_random/PAIRED_REPORT.md)
- 执行与预启动审计：[EXECUTION_REPORT.md](EXECUTION_REPORT.md) /
  [PRELAUNCH_AUDIT.json](PRELAUNCH_AUDIT.json)
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

当前 1 个 COMPLETE，0 个 RUNNING（`run_0001_seed_358341059`，与 r001 的
`run_0001_seed_358341059` 同 seed 配对：ΔMAA −2.7310 / ΔCoIN BWT **+20.7933** /
Δfinal_avg +4.9499，方向 = random-0.10 − random-0.01）；无预设目标运行总数。

<!-- registry-table:start -->
| Run | Replay sample seed | MAA | BWT | Status | Result commit |
|-----|--------------------|-----|-----|--------|---------------|
| run_0001_seed_358341059 | 358341059 | 60.122 | 28.7287 | COMPLETE | 1cdb09ae3d858a0209e6eea565d4f2226fc3e37b |
<!-- registry-table:end -->

（表由 `tools/random_replay_registry.py` 自动维护；`index.json` 为权威记录，含
started_at/completed_at/result_directory/code_commit/config_hash/error_summary/
registration_commit/result_commit/ratio/ratio_tag/paired_with 全字段 + prefix 基线数值。）

## 结果（`run_0001_seed_358341059`）

状态：**COMPLETE**（2026-09-11）；与 r001 的 `run_0001_seed_358341059` **同 seed 配对**
（同 seed 下 0.01 样本集是 0.10 的子集，嵌套对照）。

- A 矩阵（round 行 × task 列；4 位显示，全精度以 `coin_metrics.json` 原始浮点为准）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.1431 | — | — | — |
| round2 | 73.4497 | 36.5500 | — | — |
| round3 | 74.6994 | 58.4600 | 4.6700 | — |
| round4 | 76.4678 | 56.2800 | 96.5300 | 36.3300 |

- 指标：**MAA = 60.122**、**CoIN BWT = +28.7287**、**final_avg = 66.402**
  （终局旧任务均值 76.4259、对角均值 37.6733；独立重算与 coin_metrics 最大差 maxA=0、dMAA=7.6e-6、dBWT=2.8e-5）
- **BWT 分解（各任务 `final − diagonal`）**：ScienceQA **+3.3247**、TextVQA **+19.7300**、
  ImageNet **+91.8600**、GQA **0**（四项均值 = 28.7287 = BWT）。
  ⚠️ 其中 ImageNet 在 round3 被强 replay 严重干扰（4.67）而在 round4 大幅恢复（96.53）——
  **BWT 的正值主要由该阶段性恢复贡献，不能表述为「几乎没有遗忘」**。
  更准确的结论：random-0.10 相比同 seed random-0.01 **提高了 final_avg 与 BWT、降低了 MAA**，
  呈现「更强的终局保持 ↔ 更严重的阶段性新任务干扰」的**稳定性—可塑性权衡**。
- 与同 seed random-0.01 配对差值（**random-0.10 − random-0.01**）：
  ΔMAA **−2.7310**、ΔCoIN BWT **+20.7933**、Δfinal_avg **+4.9499**
  （逐对明细见 [PAIRED_REPORT.md](../coin_replay_random/PAIRED_REPORT.md) / [paired_comparison.json](../coin_replay_random/paired_comparison.json)）
- 与历史 prefix 基线差值：vs prefix-0.10 ΔMAA +2.6163 / ΔBWT +11.4981 / Δfinal_avg +10.6186；
  vs prefix-0.01 ΔMAA −0.3186 / ΔBWT +42.3586 / Δfinal_avg +20.2095
  （prefix 基线为**单次运行**，非分布，差值仅描述性）
- 跨 run 可比性证据：`model_config_hash=5fe5a4b3…`、`data_revision=bf6bd4ee…` 与 r001 各 run **完全一致**
  （同模型、同数据）；`code_commit=41abc5af222ba5eed7c6d25edab4a2bb2d110bcc`
- **代码提交差异（配对设计限制）**：本 run 的 `code_commit=41abc5a`，而 r001 各 run 为 `16f27d4`。
  因此「固定项一致」严格指**模型 / 数据 / 训练超参数 / 训练 seed / replay sample seed / 抽样算法**一致；
  新增代码经 r001 向后兼容测试（结果字段 / 六件套 / `config_hash` 逐项一致）未发现训练语义变化，
  但**不是同一代码提交**，解释配对差值时保留该限制。
- 说明：六件套 `summary.json` 中 `paired.this_result_commit=null` 属**正常**——提交无法预知自身 SHA；
  真实 C SHA 记录在 registry 的 `result_commit` 字段与本文件提交链表（不回头修改已发布六件套）。

提交链（A → C → D，顺序在 Git 历史中可核验）：

| 环节 | commit |
|---|---|
| A 注册 RUNNING | `6419a6866ef25ba3888282d21e2ef1e296277ba0` |
| C 六件套发布 | `1cdb09ae3d858a0209e6eea565d4f2226fc3e37b` |
| D registry COMPLETE | `3f6e13e8bdfa37f1b5435495fd7ec3f9e9ef6375` |

发布物：[runs/run_0001_seed_358341059/](runs/run_0001_seed_358341059/)（六件套，`summary.json` 内含
`repro` 全量 sha256、`paired` 配对元数据、`deltas` 全部差值）。验收：
`validation_report.md` **20 项全 PASS**（含 7 ckpt 参数级 finite、tensor-diff changed=448、
源数据独立重建、10 eval 单元预测校验、独立重算、脱敏扫描 clean）。

口径提示：n=1 的配对差值是**描述性证据**（单训练 seed、单任务序），不作显著性结论；
r010 与 r001 的 replay 样本量不同（0.10 是 0.01 的 10 倍）且构成嵌套，差值含「样本量」与
「样本组成」两个不可完全分离的效应。

## 对照基线（prefix 系列；历史探索，非 random 比例主比较）

| 基线 | MAA | CoIN BWT | final avg |
|---|---|---|---|
| prefix ratio=0.10（2026-09-04） | 57.5057 | +17.2306 | 55.7834 |
| prefix ratio=0.01（2026-09-05） | 60.4406 | −13.6299 | 46.1925 |

差值口径：`summary.json` deltas 中 `vs_prefix_*` = 本 run − 对应 prefix 基线；
`vs_paired_*` = 高 ratio − 低 ratio（即 random-0.10 − random-0.01，同 seed 配对）。
