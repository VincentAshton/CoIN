# CoIN + Replay —— Random Replay 实验总入口

本目录是 **random replay 抽样实验**的统一入口（长期分支 `codex/coin-replay-random`）。
主要实验只有三个，全部按同一套运行/注册/验收/发布工具执行：

| 主实验 | 系列目录 | ratio tag | 状态 |
|---|---|---|---|
| random replay **ratio=0.01** | [coin_replay_random_r001](../coin_replay_random_r001/README.md) | r001 | 当前 5 个 COMPLETE，0 个 RUNNING（开放式，可继续追加） |
| random replay **ratio=0.10** | [coin_replay_random_r010](../coin_replay_random_r010/README.md) | r010 | 当前 1 个 COMPLETE，0 个 RUNNING（MAA 60.122 / CoIN BWT +28.7287 / final_avg 66.402） |
| **同 seed 配对比较**（0.10 − 0.01） | [paired_comparison.json](paired_comparison.json) / [PAIRED_REPORT.md](PAIRED_REPORT.md) | — | 当前 1 对（seed 358341059）：ΔMAA −2.7310 / ΔBWT **+20.7933** / Δfinal_avg +4.9499 |

> **BWT 口径**：BWT 由各任务 `final − diagonal` 合成，受阶段性波动支配——本次 ImageNet 在 round3 被
> 强 replay 干扰到 4.67、round4 恢复到 96.53，对 BWT 贡献极大。**+28.7287 不宜简化为「几乎没有遗忘」**；
> 准确表述见 [r010 结果节](../coin_replay_random_r010/README.md)（稳定性—可塑性权衡）。
> 执行与预启动审计：[EXECUTION_REPORT.md](../coin_replay_random_r010/EXECUTION_REPORT.md) /
> [PRELAUNCH_AUDIT.json](../coin_replay_random_r010/PRELAUNCH_AUDIT.json)。

控制变量（两个 ratio 一致）：**模型与数据、训练超参数、训练 seed（SEED/DATA_SEED）、replay sample
seed 与抽样算法**。`SAMPLE_MODE=random`、`SEED=1234`、`DATA_SEED=1234`、`REPLAY_ACCUM=1`、
LLaVA-1.5-7B + LoRA(r=192/α=256)，任务序 ScienceQA → TextVQA → ImageNet → GQA。

**注意：两次运行使用不同代码提交**（r001 各 run = `16f27d4`，r010 `run_0001` = `41abc5a`）。新增代码
（ratio 通用化 + 门禁泛化）经 r001 向后兼容测试：结果字段 / 六件套 / `config_hash` 逐项一致，未发现
训练语义变化；但严格来说**不是同一代码提交**，该差异作为配对设计限制保留（后续新配对尽量让两侧
使用同一冻结 commit）。

抽样算法（两个 ratio 同一实现）：
`sha256_task_seed_python_shuffle_v1` —— `task_seed = sha256("<replay_sample_seed>:<task>")`，
对完整索引排列取前 `k = floor(N×ratio)`；同 seed 下 **0.01 的样本集是 0.10 的子集**
（先全排列再取前 k），构成嵌套配对设计。

## 运行纪律

- 单次运行闭环：注册 RUNNING（提交 A）→ 训练 → `finalize assemble` 全门验收 →
  发布六件套（提交 C）→ registry COMPLETE（提交 D）→ push → 停止。
- **无固定总运行次数**：系列开放，可继续追加；不设「最多 N 次」限制，也不允许按指标
  决定是否停止或隐藏结果；FAILED 记录同样永久保留。
- 同一 ratio 内禁止复用 replay sample seed；**跨 ratio 允许同 seed**（配对设计需要）。
- 配对比较一律**按 seed 取交集**（`random_replay_pair.py`），禁止按 run_number 猜测配对；
  未配对 run 标记 UNPAIRED，不参与配对均值。

## 文档权威层级与自查（防止同一数字多处漂移）

```
权威机器数据      <series>/index.json（+ runs/<run_id>/summary.json）
                  paired_comparison.json
自动生成展示      本目录各系列 README 的状态行与 registry 表、PAIRED_REPORT.md
人工说明          本文件（总入口）、SERIES_REPORT.md / EXECUTION_REPORT.md / PRELAUNCH_AUDIT.json
活动分支根 README + 默认分支 README   只做稳定导航，不写指标
```

- push 前 / 门禁自查：`python3 scripts/CoIN_Replay/tools/random_replay_sync_check.py`
  （读数一致性 + 配对产物重算 + 相对链接 + 脱敏扫描；已接入 `run_tests.sh` 步骤 A7）。
- 本文件中的系列计数与指标、系列 README 的状态表均由该工具与 registry 工具校验；
  若与 `index.json` 不一致，工具会直接 FAIL——**不要手改数字，改 registry**。

## 工具（`scripts/CoIN_Replay/tools/`）

| 工具 | 作用 |
|---|---|
| `random_replay_registry.py` | 系列注册表（index.json/csv/README 表）register / complete / list；ratio 元数据驱动，追加无上限 |
| `random_replay_finalize.py` | assemble（云端全门验收 + 脱敏导出六件套）/ fill-delta（prefix 基线与同 seed 配对差值） |
| `random_replay_pair.py` | 生成 paired_comparison.json + PAIRED_REPORT.md（seed 交集配对） |
| `random_replay_nested_check.py` | 真实数据 A/B 重建 + 0.01⊆0.10 嵌套/前缀验证 |

## 历史探索 / legacy prefix baseline（不属于 random 比例主比较）

以下两个 prefix 分支是早期探索，**保留但不作为 random 比例的主结果**；它们的数值只作为
历史对照基线（单次运行，非分布）：

| 分支 | 内容 |
|---|---|
| `experiment/coin-replay-presweep-20260903` | prefix 抽样运行代码 + 内部手册（docs/internal/） |
| `results/coin-replay-r010-20260904` | prefix ratio=0.10 与 0.01 的单次结果包 |

- prefix ratio=0.10（2026-09-04）：MAA 57.5057 / CoIN BWT +17.2306 / final_avg 55.7834
- prefix ratio=0.01（2026-09-05）：MAA 60.4406 / CoIN BWT −13.6299 / final_avg 46.1925

口径提示：prefix 基线各为**单次运行**，不是分布；与 random 系列（多次运行）的差值
只能作描述性比较，不能解释为配对效应或显著性结论。
