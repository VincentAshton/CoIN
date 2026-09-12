# Random Replay 同 seed 配对比较（ratio 对照组）

- 生成: 2026-09-12T16:11:30+08:00（`scripts/CoIN_Replay/tools/random_replay_pair.py`）
- 配对方式: **按 replay sample seed 取两个系列的交集**（不按 run_number 猜测配对）
- 配对差值定义: **random-0.10 − random-0.01**（高 ratio 为被减数）
- 配对 n（完整 COMPLETE 对）: **1**；seed 交集内 run 对（含未完成）: 1

## 1. 两个 ratio 系列各自规模（不做配对合并）

| ratio | ratio tag | 系列目录 | n_total | COMPLETE | FAILED | RUNNING |
|---|---|---|---|---|---|---|
| 0.01 | r001 | `coin_replay_random_r001` | 5 | 5 | 0 | 0 |
| 0.10 | r010 | `coin_replay_random_r010` | 1 | 1 | 0 | 0 |

### 1.1 各系列全部 run（按注册顺序，含 FAILED / RUNNING，不做指标筛选）

| ratio tag | run_id | seed | status | MAA | CoIN BWT | final_avg | result commit |
|---|---|---|---|---|---|---|---|
| r001 | run_0001_seed_358341059 | 358341059 | COMPLETE | 62.8530 | 7.9354 | 61.4521 | de8bfcac25bf |
| r001 | run_0002_seed_1608950547 | 1608950547 | COMPLETE | 61.4747 | 16.8063 | 63.9431 | d640ebdbe350 |
| r001 | run_0003_seed_2089198544 | 2089198544 | COMPLETE | 61.1599 | 8.6152 | 60.1092 | d8d361cbebac |
| r001 | run_0004_seed_1212218766 | 1212218766 | COMPLETE | 63.6960 | 6.5542 | 61.7867 | d31d8b67f6dc |
| r001 | run_0005_seed_232675848 | 232675848 | COMPLETE | 60.4601 | 16.8100 | 63.0996 | 97023b8a06b5 |
| r010 | run_0001_seed_358341059 | 358341059 | COMPLETE | 60.1220 | 28.7287 | 66.4020 | 1cdb09ae3d85 |

## 2. 同 seed 配对（seed 交集）

| seed | r001 run | r010 run | ΔMAA | ΔCoIN BWT | Δfinal_avg | 两边均 COMPLETE |
|---|---|---|---|---|---|---|
| 358341059 | run_0001_seed_358341059 (COMPLETE) | run_0001_seed_358341059 (COMPLETE) | -2.7310 | 20.7933 | 4.9499 | True |

### 2.1 配对明细（run_id / commit / 指标）

| seed | 项 | run_id | status | MAA | CoIN BWT | final_avg | result commit | code commit |
|---|---|---|---|---|---|---|---|---|
| 358341059 | r001 | run_0001_seed_358341059 | COMPLETE | 62.8530 | 7.9354 | 61.4521 | de8bfcac25bf | 16f27d4469b0 |
| 358341059 | r010 | run_0001_seed_358341059 | COMPLETE | 60.1220 | 28.7287 | 66.4020 | 1cdb09ae3d85 | 41abc5af222b |

### 2.2 配对均值与差值（仅完整 COMPLETE 对）

| 指标 | r001 均值 | r010 均值 | 均值差 | 差值范围 | n |
|---|---|---|---|---|---|
| MAA | 62.853 | 60.122 | -2.731 | [-2.731, -2.731] | 1 |
| BWT | 7.9354 | 28.7287 | 20.7933 | [20.7933, 20.7933] | 1 |
| final_avg | 61.4521 | 66.402 | 4.9499 | [4.9499, 4.9499] | 1 |

## 3. 未配对 run（UNPAIRED）

| ratio tag | run_id | seed | status | MAA | CoIN BWT | final_avg |
|---|---|---|---|---|---|---|
| r001 | run_0002_seed_1608950547 | 1608950547 | COMPLETE | 61.4747 | 16.8063 | 63.9431 |
| r001 | run_0003_seed_2089198544 | 2089198544 | COMPLETE | 61.1599 | 8.6152 | 60.1092 |
| r001 | run_0004_seed_1212218766 | 1212218766 | COMPLETE | 63.6960 | 6.5542 | 61.7867 |
| r001 | run_0005_seed_232675848 | 232675848 | COMPLETE | 60.4601 | 16.8100 | 63.0996 |

未配对 run 只在单一 ratio 存在，其数值**不参与**配对均值/差值，也不得当作配对结论使用。

## 4. 限制与口径

1. 配对差值为 4 位显示精度；统计量由 `paired_comparison.json` 的逐对差值计算。
2. n 为 seed 交集大小，属**描述性证据**，不作显著性推断。
3. 固定训练 SEED 不等于逐 bit 确定性（CUDA / FlashAttention / TF32 / 分布式），差值不能唯一归因于 replay 比例。
4. 同 seed 的 0.01 样本集是 0.10 的子集（嵌套），因此差值含「样本量」与「样本组成」两个无法完全分离的效应。
5. 同一对两侧 run 的代码提交见逐对表 `code_commit` 列（不假定两侧相同）。「固定项一致」仅指模型/数据/训练超参/训练 seed/replay sample seed/抽样算法；若两侧提交不同（例如 r001 `16f27d4` vs r010 `41abc5a`），该差异经低 ratio 系列的向后兼容测试（结果字段/六件套/config_hash 逐项一致）未发现训练语义变化，但仍作为配对设计限制保留。
6. BWT 由各任务 `final − diagonal` 合成，可能被阶段性波动支配（例如 ImageNet round3 受强 replay 干扰、round4 恢复）：不应把正 BWT 直接解读为「几乎没有遗忘」，需结合 A 矩阵的阶段性变化。
