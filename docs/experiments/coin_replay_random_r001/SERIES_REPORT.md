# CoIN + Replay「随机 replay 抽样」系列 r001 —— 五次独立运行报告

> 状态：**5/5 COMPLETE**（2026-09-09 ~ 2026-09-10）
> **审核结论：实验数据与证据链通过；本文档已完成科学表述与统计口径修订。**
> 本报告为文档层汇总（docs-only）。所有指标数值以各 run 已发布的 `coin_metrics.json`
> 的原始 `A_matrix` 浮点值为计算基准；`summary.json` 中本身已按 4 位发布的指标按其发布精度展示。
> 标注「计算」的项由脚本从原始数据复算得到；标注「推导」的项为报告作者计算。

---

## 1. 实验问题

CoIN（arXiv:2403.08350）前 4 任务顺序 LoRA 持续学习 + TRACE 式 replay。
此前 prefix 抽样（回放集 = 历史样本按原顺序取前 k）在 **ratio=0.01** 时观察到严重旧任务遗忘
（单次运行，BWT = −13.6299，终局旧任务均值 47.70），本系列考察：相同比例下改为随机抽样，
五次独立运行的结果分布如何。

- 对照基线（同环境、同代码基线，已在 results 分支发布；**均为单次运行**）：
  - prefix 0.10：MAA 57.5057 / BWT +17.2306 / final_avg 55.7834
  - prefix 0.01：MAA 60.4406 / BWT −13.6299 / final_avg 46.1925

## 2. 设计（固定项 vs 变量）

| 项 | 值 |
|---|---|
| 模型 | LLaVA-1.5-7B（vicuna-7b-v1.5 + CLIP-L/14-336），LoRA r=192 α=256 dropout=0.05 |
| 任务序 | ScienceQA → TextVQA → ImageNet → GQA（4 任务） |
| 回放集选择 | `SAMPLE_MODE=random`，`RATIO=0.01`，`REPLAY_ACCUM=1` |
| 训练 seed | `SEED=1234`、`DATA_SEED=1234`（五次运行固定） |
| **五次运行之间唯一有意改变的配置** | `REPLAY_SAMPLE_SEED`（registry 启动前以 `secrets.SystemRandom` 即时生成 31-bit，查重后使用；禁止人工指定/筛选/复用） |
| task 段 | accum=16（有效 batch 896，论文口径）；replay 段 accum=1（方案 D，全比例统一） |
| 优化 | lr 2e-4、cosine、warmup 3%、bf16+tf32、grad ckpt、ZeRO-3 + CPU offload、1 epoch/task |
| 代码 | commit `16f27d4`（云端 project_random 精确 checkout；五次运行的 run manifest `code_commit` 全为该值） |
| 硬件 | 4×A100-80GB（ebcloud） |
| 评估 | Truth Alignment：每任务官方 eval，A 矩阵（round 为行、task 为列，有效单元为**下三角**） |

**抽样算法 `sha256_task_seed_python_shuffle_v1`**（每个 round≥2 的历史任务）：
1. `task_seed = sha256("<replay_sample_seed>:<task>")`（UTF-8 hexdigest；**task seed 不含 round**）
2. `rng = random.Random(int(task_seed,16))` 对完整 `indices=list(range(N))` 洗牌
3. `selected = shuffled[:k]`，`k = floor(N*ratio)` 严格不变
4. 先全排列再取前 k ⇒ 同 seed 下 0.01 集合 ⊆ 0.10 集合（nested 性质）

**实际 replay 样本规模**（五次运行的源 N 与 k 经发布产物 `replay_selection_summary.json` 逐项核验一致）：

| 历史任务 | 源训练集 N | k = floor(N×0.01) |
|---|---:|---:|
| ScienceQA | 12726 | 127 |
| TextVQA | 34602 | 346 |
| ImageNet | 129833 | 1298 |

由此，每次运行内的 replay 总量为：round 2 = 127；round 3 = 473（127+346）；round 4 = 1771（127+346+1298）。
同一任务在后续 round 中**复用同一 seed 派生的同一组选样**（task seed 不含 round），
不是每个 round 重新随机抽取。

## 3. 指标定义（CoIN 官方口径）

设 A 为 T×T 矩阵，`A_{j,i}` = 第 j 轮结束后任务 i 的得分（T=4，下标 1..T，未评估单元计 0，不参与求和）：

```
MAA = (1/T) · Σ(j=1..T) [ (1/j) · Σ(i=1..j) A_{j,i} ]
BWT = (1/T) · Σ(i=1..T) ( A_{T,i} − A_{i,i} )
final_avg    = (1/T) · Σ(i=1..T) A_{T,i}
final_old_task_mean = (1/(T−1)) · Σ(i=1..T−1) A_{T,i}
diagonal_mean       = (1/T) · Σ(i=1..T) A_{i,i}
```

- MAA：**先计算每一轮结束后所有已学习任务的平均准确率，再对 T 个轮次等权平均**；
  不是把下三角全部有效单元直接等权平均。（因各轮有效单元数不同，两种算法结果不同。）
- BWT：最后一个任务的差值项恒为 0，但按 CoIN 定义**仍包含在求和中并除以 T**；
  本报告统一称 **CoIN BWT**，以区别于部分文献采用 1/(T−1) 的定义。

## 4. 执行纪律（防止结果被人为筛选）

- 累计 **5 个 COMPLETE 后停止**，不注册第 6 个；全部运行公开（RUNNING/FAILED/COMPLETE 永久保留）
- **禁止以指标（MAA/BWT）为导向停止或隐藏结果**；seed 不得因结果重新生成
- registry「A/C/D」提交协议：A=注册 RUNNING 提交、C=六件套发布提交、D=registry COMPLETE 提交
  （hash 反查自 git log，禁伪造）
- 发布前置：`random_replay_finalize.py assemble` 在训练机执行 11 类验收门
  （.complete/归属/配置一致性/sidecar 一致 + 源数据独立重建【SHA、N、k、排列、ids、replay 顺序】/
  目录契约/7 ckpt 参数级校验/tensor-diff【changed + hash≠ + finite】/non_lora 差异/
  10 eval 单元 + 预测校验/独立重算 == coin_metrics/6 文件脱敏扫描），**全部 PASS 才原子换入发布目录**

## 5. 五次独立运行的结果

### 5.1 A 矩阵（round 为行、task 为列；表中四舍五入显示到小数点后 4 位）

| run | round1 | round2 | round3 | round4 |
|---|---|---|---|---|
| 0001 | [73.1667] | [68.6631, 56.3400] | [69.4647, 47.0500, 46.3600] | [61.1884, 52.0000, 94.4200, 38.2000] |
| 0002 | [72.9073] | [71.7991, 54.0400] | [67.1068, 49.6200, 21.6600] | [70.8324, 50.8800, 94.1200, 39.9400] |
| 0003 | [73.0960] | [68.5687, 55.0500] | [65.6449, 46.1600, 37.0700] | [54.6569, 50.7400, 94.2800, 40.7600] |
| 0004 | [73.1903] | [70.3608, 53.9400] | [69.9599, 48.5000, 54.5100] | [65.1969, 49.3500, 93.3100, 39.2900] |
| 0005 | [72.9781] | [67.3190, 53.2900] | [70.3843, 46.4900, 19.5000] | [69.1582, 50.4200, 93.4300, 39.3900] |

### 5.2 各 run 指标

| run | seed | MAA | CoIN BWT | final_avg | 终局旧任务均值 | 对角均值 |
|---|---|---|---|---|---|---|
| 0001 | 358341059 | 62.8530 | 7.9354 | 61.4521 | 69.2028 | 53.5167 |
| 0002 | 1608950547 | 61.4747 | 16.8063 | 63.9431 | 71.9441 | 47.1368 |
| 0003 | 2089198544 | 61.1599 | 8.6152 | 60.1092 | 66.5590 | 51.4940 |
| 0004 | 1212218766 | 63.6960 | 6.5542 | 61.7867 | 69.2856 | 55.2326 |
| 0005 | 232675848 | 60.4601 | 16.8100 | 63.0996 | 71.0027 | 46.2895 |

（各 run 的复算值与其 summary.json 发布值最大差异 ≤ 5×10⁻⁵，来源于 summary 的 4 位发布精度。）

### 5.3 五次运行的分布统计（计算；输入为五个 coin_metrics.json 的原始 A_matrix，sample sd 使用 n−1 分母）

| 指标 | mean | sample sd | median | [min, max] |
|---|---|---|---|---|
| MAA | 61.9287 | 1.3164 | 61.4747 | [60.4601, 63.6960] |
| CoIN BWT | 11.3442 | 5.0428 | 8.6152 | [6.5542, 16.8100] |
| final_avg | 62.0781 | 1.4895 | 61.7867 | [60.1092, 63.9431] |
| 终局旧任务均值 | 69.5989 | 2.0596 | 69.2856 | [66.5590, 71.9441] |
| 对角均值 | 50.7339 | 3.9132 | 51.4940 | [46.2895, 55.2326] |

（n=5 保留完整逐 run 列表、min/max 与中位数；不给出显著性 p 值或依赖正态假设的置信区间。）

### 5.4 相对 prefix 基线的逐 run 差值（各 run summary.json 的 deltas 字段，本 run − 基线）

| run | vs prefix 0.01: MAA / BWT / final | vs prefix 0.10: MAA / BWT / final |
|---|---|---|
| 0001 | +2.4124 / +21.5653 / +15.2596 | +5.3473 / −9.2952 / +5.6687 |
| 0002 | +1.0341 / +30.4362 / +17.7506 | +3.9690 / −0.4243 / +8.1597 |
| 0003 | +0.7193 / +22.2451 / +13.9167 | +3.6542 / −8.6154 / +4.3258 |
| 0004 | +3.2554 / +20.1841 / +15.5942 | +6.1903 / −10.6764 / +6.0033 |
| 0005 | +0.0195 / +30.4399 / +16.9071 | +2.9544 / −0.4206 / +7.3162 |

### 5.5 五次运行均值相对基线的描述性差值（计算）

口径声明：**统一使用由原始 A_matrix 复算出的五次运行均值**减去基线发布值（基线为单次运行发布值）。

| 对比 | ΔMAA | ΔBWT | Δfinal_avg |
|---|---|---|---|
| 随机 0.01 五次均值 vs prefix 0.01 | +1.4881 | +24.9741 | +15.8856 |
| 随机 0.01 五次均值 vs prefix 0.10 | +4.4230 | −5.8864 | +6.2947 |

（ΔMAA vs prefix 0.10 实算为 +4.423032 → 显示 +4.4230；若改用「先各自舍入到 4 位再相减」的口径，
结果同为 +4.4230。本报告采用前一种口径，全程不混用。）

**以上是五次 random 运行均值与单次 prefix 基线之间的描述性差值。prefix 基线不是一个多 seed
分布，因此这些差值不能解释为配对效应、置信区间或统计显著性。**

### 5.6 终局相对对角的变化（per_task_final_minus_diagonal，A_{T,i} − A_{i,i}）

| run | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| 0001 | −11.9783 | −4.3400 | +48.0600 | 0.0 |
| 0002 | −2.0750 | −3.1600 | +72.4600 | 0.0 |
| 0003 | −18.4390 | −4.3100 | +57.2100 | 0.0 |
| 0004 | −7.9934 | −4.5900 | +38.8000 | 0.0 |
| 0005 | −3.8199 | −2.8700 | +73.9300 | 0.0 |

（GQA 恒 0 = 最后一轮新学任务，A_{T,T} 即其对角项。）

## 6. 结果解读（报告作者观点，供审核挑战）

### 6.1 BWT 与 MAA / final_avg

- 在固定模型、任务顺序、训练 seed 与训练超参数的条件下，五个独立生成的 replay 抽样 seed 均得到正
  的 CoIN BWT（6.5542 ~ 16.8100），而单次 prefix-0.01 基线为 −13.6299。该结果提供了**描述性证据**：
  ratio=0.01 时观察到的严重遗忘并非必然，实验结果对 replay 样本组成高度敏感。
  由于 prefix 基线只有一次运行、没有 random-0.10 配对对照且样本量仅为 n=5，本系列尚不能证明
  prefix 取样是遗忘的主要原因，也不构成统计显著性结论。
- 结果方向**不支持**「ratio=0.01 必然导致旧任务崩溃」（限定于当前模型、任务序、训练 seed 与实验环境）。
- MAA：五次运行全部 ≥ 60.4601（均值 61.9287），高于单次 prefix 0.01（60.4406）与 prefix 0.10（57.5057）。
- final_avg：五次均值 62.0781，高于单次 prefix 0.10（55.7834）与 prefix 0.01（46.1925）。

### 6.2 运行间变异

与 replay 抽样 seed 相关的运行间变异不可忽略（BWT sd=5.0428、跨度 6.5542~16.8100；对角均值 sd=3.9132）。
需要说明：`REPLAY_SAMPLE_SEED` 是五次运行之间唯一有意改变的配置变量，但固定 `SEED`/`DATA_SEED`
不代表 CUDA、Flash Attention、TF32 与分布式训练能够做到逐 bit 确定，因此当前观察到的方差
**不能完全分解**为 replay 样本组成效应与训练非确定性。若要估计纯训练噪声，需要对同一个
replay seed 重复完整训练。

### 6.3 共同现象（与 replay 干扰/后续恢复假设一致的观察现象）

- ImageNet 的 **round-3 对角值**，即 round 3 完成 ImageNet task 段和随后旧任务 replay 段之后的评估结果，
  在五次运行中为 19.50 ~ 54.51，而 round 4 终局为 93.31 ~ 94.42（终局相对对角 +38.80 ~ +73.93）。
- ScienceQA 终局低于其对角（−2.08 ~ −18.44），呈明显遗忘。
- 与 prefix 系列此前的观察模式一致，属**与 replay 干扰/后续恢复假设一致的观察现象**，
  尚无独立消融证据支持机制性结论。
- 现有 A 矩阵只包含**每轮最终 checkpoint** 的评估，不能区分 task 段与 replay 段分别造成的变化。
  更直接的验证需要比较同一轮的 `roundj_task` 与 `roundj_replay` checkpoint，
  例如评估 ImageNet@round3_task、ImageNet@round3_replay、ImageNet@round4_task 与 ImageNet@round4_replay。
  （此处为未来验证需求，本报告不启动任何新评估。）

## 7. 工程事件与恢复（全部留痕，供审计）

1. **启动脚本 seed 变量覆盖 bug（被门禁两次拦下）**：
   云端启动脚本中 replay sample seed 的顶层变量名曾用 `SEED`，被随后的 `export SEED=1234` 覆盖 →
   `REPLAY_SAMPLE_SEED` 错成 1234 → `random_r001_gate` 报「run ID 内嵌 seed ≠ REPLAY_SAMPLE_SEED」
   并**拒绝进入 preflight/训练**（run_0002 与 run_0003 各一次）。修复：变量改名 `RS_SEED`。
   该事件验证了启动 fail-fast 门的有效性。
2. **run_0004 磁盘满崩溃与恢复（2026-09-10）**：
   round3 ImageNet task **144/144 步跑完**（2h07m）后存 checkpoint 时崩溃：
   `RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos X vs Y`
   （PytorchStreamWriter 写盘失败 = 磁盘满签名）；根因 `/root/data` 1T Lustre 100%（剩 14M）。
   恢复：清理 tmp 历史门禁残留（167G）+ 已完成运行内未参与发布的 HF 单 epoch 存档 `checkpoint-N`
   → 同一 seed、同一目录 resume（manifest 配置 hash 校验一致 → round1/2 `validate-round` 跳过 →
   round3 重训）→ 一次通过。
   **关于被删的 `checkpoint-N` 的准确表述**：这些存档**不参与已发布指标计算、跨轮加载链或最终
   checkpoint 验收**（加载链与 assemble 只读 ckpt 顶层 adapter_model.bin，
   `best_model_checkpoint=None`），删除后仍可使用顶层 adapter 复核已发布结果；
   **但会失去对应训练段的 optimizer/scheduler 中途恢复能力，且被删除文件不可直接恢复，
   只能重新训练生成。** 此清理针对已发布完成的 run，不影响任何已发布指标。
3. **未来清理的保留策略建议**（仅在以下条件全部满足后，才清理 completed run 内的 `checkpoint-*` 存档）：
   - 该 run 状态为 COMPLETE；
   - finalize 全部门通过；
   - 六件套的结果提交 C 已 push；
   - registry COMPLETE 提交 D 已 push；
   - 顶层 adapter / non_lora 文件完成参数级校验；
   - 确认没有 RUNNING run 依赖该目录。
   同时建议未来启动门设置**可配置的磁盘余量阈值**；根据本系列观察可先采用不少于约 **200GB**
   的安全余量（工程经验阈值，不是实验定义）。
4. **preflight 图片检查缓存永不命中**（脚本 quick_sha 算法 ≠ preflight_data 报告 data_sha256）：
   每 run 全量重检约 13 分钟，**未影响实验结果，仅产生约 13 分钟额外开销**；
   该问题应在未来实验系列启动前修复，**但不得为已经完成的 r001 改写训练代码或历史 manifest**。
5. **两处 commit message 文本笔误（errata，权威数据不受影响）**：
   - run_0002 的 C 提交文本 `final_avg=63.9424`，权威值 **63.9431**
   - run_0003 的 C 提交文本 `final_avg=60.1142`，权威值 **60.1092**
   说明：**权威值由 `summary.json`、`coin_metrics.json`、`index.json` 决定；commit message 仅为
   非权威文本标签。** 按「不重写 Git 历史、不修改旧提交」原则保留原提交，本报告已完成勘误。
6. **云端空间现状**：清理后 637G 可用；五次运行各保留 6.6G（顶层 adapter + 完整结果 + manifests）。

## 8. 证据索引（可核验）

分支：`codex/coin-replay-random-r001`。每个 run 的提交链（A → C → D，顺序已在 Git 历史中核验）：

| run | A 注册 | C 发布六件套 | D registry COMPLETE |
|---|---|---|---|
| 0001 | `c9f8c02` | `de8bfca` | `d7a8ec8` |
| 0002 | `d959e32` | `d640ebd` | `a249d2a` |
| 0003 | `b5a829d` | `d8d361c` | `371d9a7` |
| 0004 | `095392b` | `d31d8b6` | `deb6009` |
| 0005 | `e46ee3c` | `97023b8` | `0d5b0d0` |

- 发布物目录：`docs/experiments/coin_replay_random_r001/runs/<run_id>/`，每次运行六件套：
  `summary.json`（含 repro 字段：其余五文件 sha256 全值）、`coin_metrics.json`、`acc_sources.json`、
  `run_manifest.sanitized.json`、`replay_selection_summary.json`、`validation_report.md`（18 项 PASS、0 FAIL）。
  25 个非 summary 发布文件的 SHA256 已按各 run summary.json 的 repro 字段逐一复核，mismatch = 0。
- 权威注册表：`docs/experiments/coin_replay_random_r001/index.json`（5 个 run 全 COMPLETE、seed 唯一、
  `code_commit`/`data_revision`/`model_config_hash` 五项一致；含 registration_commit/result_commit 等全字段）
- 抽样可重建性：`replay_selection_summary.json` 记录每个 round×task 的 `N`、`k`、`task_seed`、
  源文件 sha256、`selected_ids/indices` 的 sha256 与 example_ids；配合源 `train.json` 可逐字节重建样本集
- 训练机完整 sidecar manifest / ckpt 顶层 adapter / 原始 eval 产物：保留于云端
  `/root/data/coin/formal/<run_id>/`（既有审计位置；不入 Git）

## 9. 局限（请审核时按此打折）

1. n=5，仅提供**描述性分布**，不构成统计显著性结论。
2. prefix 0.01 与 prefix 0.10 均为**单次确定性基线**，不是多 seed 分布，跨运行差值不可作配对解释。
3. **未运行 random 0.10**，无法通过同 seed 的 nested 配对分离「抽样方式」与「回放比例」效应。
4. 单训练 seed（SEED=1234）、单任务顺序、单模型（LLaVA-1.5-7B），结论外推受限。
5. **未对同一 replay seed 重复训练**，无法估计训练非确定性，故运行间方差不能唯一归因于 replay 抽样。
6. A 矩阵只包含 round-end 评估，**不能直接分解 task 段与 replay 段**各自造成的变化。
7. 4×A100 环境仅用于本项目内部比较（含与 prefix 基线），**不与论文原始数值做未经校准的直接比较**。
8. 已删除部分 completed run 的 `checkpoint-N`，因此不能从这些中间点精确恢复优化器状态，
   **但不影响现有发布指标、顶层 adapter 与结果复核**。

## 10. 提交审核的问题

1. §5.3 的统计口径（mean ± sample sd + median + [min,max]，不做检验）是否恰当？
2. §6.1 的描述性推断边界是否合适？升级为可写进论文的结论需要哪些额外证据
   （random-0.10 对照、跨模型复制、同 seed 重复训练的噪声估计、配对设计）？
3. §6.3 的「观察现象」表述是否足够严谨？是否需要单独的 task/replay 分段消融来支持机制解释？
4. §7.3 的清理保留策略与磁盘余量阈值建议是否可采纳？是否应写入运行手册的前置门？
5. 本报告的数值、提交链、证据索引是否与各 run 发布物逐项一致？是否存在遗漏或过度声明？
6. 是否建议将本报告（或其精简版）作为系列公开摘要随 registry 表一起发布？

---

*报告生成：2026-09-10（Hermes agent）；2026-09-10 完成审核后文档修订（docs-only）。
数据来源：五次运行的已发布产物（`coin_metrics.json` 原始 A_matrix、`summary.json`、
`replay_selection_summary.json`、`validation_report.md`）与 registry `index.json`。
§5.3 / §5.5 的统计与差值由脚本从原始 A_matrix 复算；其余为产物原值。*
