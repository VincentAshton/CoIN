# CoIN + Replay「随机 replay 抽样」系列 r001 —— 五轮实验报告

> 状态：**5/5 COMPLETE**（2026-09-09 ~ 2026-09-10）
> 本报告供外部审核 agent（Codex）审阅。所有数值均取自已发布的权威产物
> （`runs/<run_id>/summary.json` / `coin_metrics.json`，逐个 run 已 push 到本分支），
> 未经二次修约；报告中标注「推导」的项为报告作者计算，其余为产物原值。

---

## 1. 实验问题

CoIN（arXiv:2403.08350）前 4 任务顺序 LoRA 持续学习 + TRACE 式 replay。
此前 prefix 抽样（回放集 = 历史样本按原顺序取前 k）在 **ratio=0.01** 时出现严重旧任务遗忘
（BWT = −13.6299，终局旧任务均值 47.70），而同比例下随机抽样是否缓解，是本系列要回答的问题。

- 对照基线（同环境、同代码基线，results 分支已发布）：
  - prefix 0.10：MAA 57.5057 / BWT +17.2306 / final_avg 55.7834
  - prefix 0.01：MAA 60.4406 / BWT −13.6299 / final_avg 46.1925

## 2. 设计（固定项 vs 变量）

| 项 | 值 |
|---|---|
| 模型 | LLaVA-1.5-7B（vicuna-7b-v1.5 + CLIP-L/14-336），LoRA r=192 α=256 dropout=0.05 |
| 任务序 | ScienceQA → TextVQA → ImageNet → GQA（4 任务） |
| 回放集选择 | `SAMPLE_MODE=random`，`RATIO=0.01`，`REPLAY_ACCUM=1` |
| 训练 seed | `SEED=1234`、`DATA_SEED=1234`（固定） |
| 每 run 唯一变量 | `REPLAY_SAMPLE_SEED`（registry 启动前以 `secrets.SystemRandom` 即时生成 31-bit，查重后使用，禁止人工指定/筛选/复用） |
| task 段 | accum=16（有效 batch 896，论文口径）；replay 段 accum=1（方案 D，全比例统一） |
| 优化 | lr 2e-4、cosine、warmup 3%、bf16+tf32、grad ckpt、ZeRO-3 + CPU offload、1 epoch/task |
| 代码 | commit `16f27d4`（云端 project_random 精确 checkout；run manifest 内 code_commit 全为 16f27d4…） |
| 硬件 | 4×A100-80GB（ebcloud） |
| 评估 | Truth Alignment：每任务官方 eval，A 矩阵（10 个 round×task 单元，下三角） |

**抽样算法 `sha256_task_seed_python_shuffle_v1`**（每个 round≥2 的历史任务）：
1. `task_seed = sha256("<replay_sample_seed>:<task>")`（UTF-8 hexdigest；task seed 不含 round）
2. `rng = random.Random(int(task_seed,16))` 对完整 `indices=list(range(N))` 洗牌
3. `selected = shuffled[:k]`，`k = floor(N*ratio)` 严格不变
4. 先全排列再取前 k ⇒ 同 seed 下 0.01 集合 ⊆ 0.10 集合（nested）

## 3. 执行纪律（防止结果被人为筛选）

- 累计 **5 个 COMPLETE 后停止**，不注册第 6 个；全部运行公开（RUNNING/FAILED/COMPLETE 永久保留）
- **禁止以指标（MAA/BWT）为导向停止或隐藏结果**；seed 不得因结果重新生成
- registry「A/C/D」提交协议：A=注册 RUNNING 提交、C=六件套发布提交、D=registry COMPLETE 提交
  （hash 反查自 git log，禁伪造）
- 发布前置：`random_replay_finalize.py assemble` 在训练机执行 11 道验收门
  （.complete/归属/配置一致性/sidecar 一致 + 源数据独立重建【SHA、N、k、排列、ids、replay 顺序】/
  目录契约/7 ckpt 参数级校验/tensor-diff【changed + hash≠ + finite】/non_lora 差异/
  10 eval 单元 + 预测校验/独立重算 == coin_metrics/6 文件脱敏扫描），**全部 PASS 才原子换入发布目录**

## 4. 五轮结果

### 4.1 A 矩阵（round×task，全精度 4 位）

| run | round1 | round2 | round3 | round4 |
|---|---|---|---|---|
| 0001 | [73.1667] | [68.6631, 56.3400] | [69.4647, 47.0500, 46.3600] | [61.1884, 52.0000, 94.4200, 38.2000] |
| 0002 | [72.9073] | [71.7991, 54.0400] | [67.1068, 49.6200, 21.6600] | [70.8324, 50.8800, 94.1200, 39.9400] |
| 0003 | [73.0960] | [68.5687, 55.0500] | [65.6449, 46.1600, 37.0700] | [54.6569, 50.7400, 94.2800, 40.7600] |
| 0004 | [73.1903] | [70.3608, 53.9400] | [69.9599, 48.5000, 54.5100] | [65.1969, 49.3500, 93.3100, 39.2900] |
| 0005 | [72.9781] | [67.3190, 53.2900] | [70.3843, 46.4900, 19.5000] | [69.1582, 50.4200, 93.4300, 39.3900] |

列顺序：round j 行的第 i 项 = R[j][i]（第 j 轮结束后任务 i 的得分）。

### 4.2 指标（权威值，来自各 run summary.json / coin_metrics.json）

| run | seed | MAA | BWT | final_avg | 终局旧任务均值 | 对角均值 |
|---|---|---|---|---|---|---|
| 0001 | 358341059 | 62.8530 | 7.9354 | 61.4521 | 69.2028 | 53.5167 |
| 0002 | 1608950547 | 61.4747 | 16.8063 | 63.9431 | 71.9441 | 47.1368 |
| 0003 | 2089198544 | 61.1599 | 8.6152 | 60.1092 | 66.5590 | 51.4940 |
| 0004 | 1212218766 | 63.6960 | 6.5542 | 61.7867 | 69.2856 | 55.2326 |
| 0005 | 232675848 | 60.4601 | 16.8100 | 63.0996 | 71.0027 | 46.2895 |

配对指标：MAA=上三角单元平均（含对角）；BWT=平均 (R[4][i] − R[i][i])；final_avg=R[4] 行均值。

### 4.3 统计（n=5，报告作者计算）

| 指标 | mean | sd | min | max |
|---|---|---|---|---|
| MAA | 61.9287 | 1.3164 | 60.4601 | 63.6960 |
| BWT | 11.3442 | 5.0428 | 6.5542 | 16.8100 |
| final_avg | 62.0781 | 1.4896 | 60.1092 | 63.9431 |
| 终局旧任务均值 | 69.5988 | 2.0595 | 66.5590 | 71.9441 |
| 对角均值 | 50.7339 | 3.9132 | 46.2895 | 55.2326 |

### 4.4 相对 prefix 基线（各 run summary.json 的 deltas 字段，本 run − 基线）

| run | vs prefix 0.01: MAA / BWT / final | vs prefix 0.10: MAA / BWT / final |
|---|---|---|
| 0001 | +2.4124 / +21.5653 / +15.2596 | +5.3473 / −9.2952 / +5.6687 |
| 0002 | +1.0341 / +30.4362 / +17.7506 | +3.9690 / −0.4243 / +8.1597 |
| 0003 | +0.7193 / +22.2451 / +13.9167 | +3.6542 / −8.6154 / +4.3258 |
| 0004 | +3.2554 / +20.1841 / +15.5942 | +6.1903 / −10.6764 / +6.0033 |
| 0005 | +0.0195 / +30.4399 / +16.9071 | +2.9544 / −0.4206 / +7.3162 |

### 4.5 终局相对对角的变化（per_task_final_minus_diagonal，R[4][i] − R[i][i]）

| run | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| 0001 | −11.9783 | −4.3400 | +48.0600 | 0.0 |
| 0002 | −2.0750 | −3.1600 | +72.4600 | 0.0 |
| 0003 | −18.4390 | −4.3100 | +57.2100 | 0.0 |
| 0004 | −7.9934 | −4.5900 | +38.8000 | 0.0 |
| 0005 | −3.8199 | −2.8700 | +73.9300 | 0.0 |

（GQA 恒 0 = 最后一轮新学任务，R[4][3] 即其对角。）

## 5. 结果解读（报告作者观点，供审核挑战）

1. **BWT 方向一致为正**：5/5 全为正（+6.5542 ~ +16.8100），与 prefix 0.01 的 −13.6299 形成对比；
   即使最差一轮（0004）也显著优于 prefix 0.01。**方向上不支持**「0.01 必然导致旧任务崩溃」，
   即 prefix 0.01 的崩溃至少部分来自「按原顺序取前 k」的样本选择方式，而非单纯比例过低。
2. **MAA**：5/5 ≥ 60.4601，均值 61.93，高于 prefix 0.01（60.4406）与 prefix 0.10（57.5057）。
3. **final_avg**：均值 62.08，高于 prefix 0.10（55.78）与 prefix 0.01（46.19）。
4. **seed 效应不可忽略**：BWT 的 sd=5.04（跨度 6.55~16.81），对角均值 sd=3.91，
   说明单 seed 结果不可当结论用——本系列的 5 seed 是产生该统计的最小设计。
5. **共同现象（方法行为，非工程故障）**：
   - ImageNet 对角（初学）在 4 轮中出现「初学低、终局高」：初学 19.50~54.51 →
     round4 终局 93.31~94.42（+38.80 ~ +73.93）。
   - ScienceQA 终局低于其初学（−2.08 ~ −18.44），呈明显遗忘。
   - 与 prefix 系列的已知镜像一致：**replay 集不含当前任务** ⇒ 新学任务在 replay 段被旧数据干扰，
     而后续轮次 replay 中含该任务时又大幅回升。此现象在 prefix 0.10/0.01 上已被诊断记录。

## 6. 工程事件与恢复（全部留痕，供审计）

1. **启动脚本 seed 变量覆盖 bug（被门禁两次拦下）**：
   云端启动脚本中 replay sample seed 的顶层变量名曾用 `SEED`，被随后的 `export SEED=1234` 覆盖 →
   `REPLAY_SAMPLE_SEED` 错成 1234 → `random_r001_gate` 报「run ID 内嵌 seed ≠ REPLAY_SAMPLE_SEED」
   并**拒绝进入 preflight/训练**（run_0002 与 run_0003 各一次）。修复：变量改名 `RS_SEED`。
   这验证了启动 fail-fast 门的有效性。
2. **run_0004 磁盘满崩溃与恢复（2026-09-10）**：
   round3 ImageNet task **144/144 步跑完**（2h07m）后存 checkpoint 时崩溃：
   `RuntimeError: [enforce fail at inline_container.cc:337] . unexpected pos X vs Y`
   （PytorchStreamWriter 写盘失败 = 磁盘满签名）；根因 `/root/data` 1T Lustre 100%（剩 14M）。
   恢复：清理 tmp 历史门禁残留（167G）+ 已发布 run 的 HF 单 epoch 存档 `checkpoint-N`
   （每个 ~20G、每 run 7 个；加载链与 assemble 只读 ckpt 顶层 adapter_model.bin，
   `best_model_checkpoint=None`，该存档零依赖）→ 同 seed 同目录 resume
   （manifest 配置 hash 校验一致 → round1/2 `validate-round` 跳过 → round3 重训）→ 一次通过。
3. **preflight 图片检查缓存永不命中**（脚本 quick_sha 算法 ≠ preflight_data 报告 data_sha256）：
   每 run 全量重检 ~13 分钟，属已知非阻塞开销（上游记录在案，未影响结果）。
4. **两处 commit message 文本笔误（数据不受影响）**：
   - run_0002 的 C 提交文本 `final_avg=63.9424`，权威值 **63.9431**
   - run_0003 的 C 提交文本 `final_avg=60.1142`，权威值 **60.1092**
   两处均为手工转述派生值时的笔误；`summary.json`/`coin_metrics.json` 及全部发布物正确，
   报告与 registry 表使用权威值。
5. **云端空间现状**：清理后 637G 可用；5 个 run 各保留 6.6G（顶层 adapter + 完整结果 + manifests）。

## 7. 证据索引（可核验）

分支：`codex/coin-replay-random-r001`（本报告所在分支）。每个 run 的提交链：

| run | A 注册 | C 发布六件套 | D registry COMPLETE |
|---|---|---|---|
| 0001 | `c9f8c02` | `de8bfca` | `d7a8ec8` |
| 0002 | `d959e32` | `d640ebd` | `a249d2a` |
| 0003 | `b5a829d` | `d8d361c` | `371d9a7` |
| 0004 | `095392b` | `d31d8b6` | `deb6009` |
| 0005 | `e46ee3c` | `97023b8` | `0d5b0d0` |

- 发布物目录：`docs/experiments/coin_replay_random_r001/runs/<run_id>/`（每 run 六件套：
  `summary.json`（含 repro 字段：其余五文件 sha256 全值）、`coin_metrics.json`、`acc_sources.json`、
  `run_manifest.sanitized.json`、`replay_selection_summary.json`、`validation_report.md`（11 门 PASS 证据））
- 权威注册表：`docs/experiments/coin_replay_random_r001/index.json`（含 started_at/completed_at/
  code_commit/config_hash/data_revision/model_config_hash/registration_commit/result_commit 全字段）
- 抽样可重建性：`replay_selection_summary.json` 记录每个 round×task 的 `N`、`k`、`task_seed`、
  源文件 sha256、`selected_ids/indices` 的 sha256 与 example_ids；配合源 `train.json` 可逐字节重建样本集
- 训练机完整 sidecar manifest / ckpt 顶层 adapter / 原始 eval 产物：保留于云端
  `/root/data/coin/formal/<run_id>/`（不入 Git）

## 8. 局限（请审核时按此打折）

- **样本量**：5 个 replay 抽样 seed（单模型、单任务序、单训练 seed）。描述性证据，未做统计检验
  （n=5 下 p 值意义有限）；BWT sd≈5 说明均值差异需谨慎表述。
- **对照口径**：与 prefix 的比较是「跨运行」比较（prefix 基线为单 run 单 seed，2026-09-04/05 发布），
  非同 seed 配对；random/prefix 的分布差异未做配对检验。
- **仅 ratio=0.01**：未跑 random 0.10（无法分离「抽样方式」与「比例」交互）。
- **评测面**：仅为 4 任务 Truth Alignment（MAA/BWT/final_avg）；未评估生成质量、指令跟随等。
- **硬件差异**：4×A100 vs 论文 8 卡（prefix 系列同环境，故内部可比；与论文数值不可直接并排）。
- **ImageNet 对角现象**未独立复核（沿用 prefix 系列的「方法行为」判定，见 §5.5）；如需强证据需做
  「replay 集不含当前任务」的消融。

## 9. 提交审核的具体问题

1. §4.3 的统计口径是否恰当？是否应改为「逐 run 列表 + 中位数/区间」而非 mean±sd（n=5）？
2. §5.1 的推断（「prefix 0.01 的崩溃主要来自取样方式而非比例」）是否过强？需要什么额外证据
   才能从「方向性信号」升级为可写进论文的结论（例如 random 0.10 对照、跨模型复制、pairwise 检验）？
3. §5.5 的 ImageNet/ScienceQA 现象判定为「方法行为」是否成立？是否需要单独的消融实验补证？
4. 报告 §6 的工程事件（尤其 run_0004 崩溃→清理→resume）是否满足可复现性/审计要求？
   清理掉的 `checkpoint-N` 是否需要在发布协议中改为「发布后即删」以免再次撑满磁盘？
5. 本报告的数值、commit 链、证据索引是否与各 run 发布物逐项一致？是否存在遗漏项或过度声明？
6. 是否建议把本报告（或其精简版）作为系列公开摘要，随 registry 表一起发布？

---

*报告生成：2026-09-10（Hermes agent）；数据来源：五轮已发布产物 + registry index.json。
报告作者声明：§4.3 与 §4.4 中的差值/统计为计算所得；其余为产物原值；所有数值保留产物精度，未四舍五入修约。*
