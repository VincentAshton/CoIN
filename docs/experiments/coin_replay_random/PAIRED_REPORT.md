# Random Replay 同 seed 配对比较（ratio 对照组）

- 生成: 2026-09-11T02:17:52+08:00（`scripts/CoIN_Replay/tools/random_replay_pair.py`）
- 配对方式: **按 replay sample seed 取两个系列的交集**（不按 run_number 猜测配对）
- 配对差值定义: **random-0.10 − random-0.01**（高 ratio 为被减数）
- 配对 n（完整 COMPLETE 对）: **0**；seed 交集内 run 对（含未完成）: 0

## 1. 两个 ratio 系列各自规模（不做配对合并）

| ratio | ratio tag | 系列目录 | n_total | COMPLETE | FAILED | RUNNING |
|---|---|---|---|---|---|---|
| 0.01 | r001 | `coin_replay_random_r001` | 5 | 5 | 0 | 0 |
| 0.10 | r010 | `coin_replay_random_r010` | 0 | 0 | 0 | 0 |

### 1.1 各系列全部 run（按注册顺序，含 FAILED / RUNNING，不做指标筛选）

| ratio tag | run_id | seed | status | MAA | CoIN BWT | final_avg | result commit |
|---|---|---|---|---|---|---|---|
| r001 | run_0001_seed_358341059 | 358341059 | COMPLETE | 62.8530 | 7.9354 | 61.4521 | de8bfcac25bf |
| r001 | run_0002_seed_1608950547 | 1608950547 | COMPLETE | 61.4747 | 16.8063 | 63.9431 | d640ebdbe350 |
| r001 | run_0003_seed_2089198544 | 2089198544 | COMPLETE | 61.1599 | 8.6152 | 60.1092 | d8d361cbebac |
| r001 | run_0004_seed_1212218766 | 1212218766 | COMPLETE | 63.6960 | 6.5542 | 61.7867 | d31d8b67f6dc |
| r001 | run_0005_seed_232675848 | 232675848 | COMPLETE | 60.4601 | 16.8100 | 63.0996 | 97023b8a06b5 |

## 2. 同 seed 配对（seed 交集）

（当前无同 seed 配对）

## 3. 未配对 run（UNPAIRED）

| ratio tag | run_id | seed | status | MAA | CoIN BWT | final_avg |
|---|---|---|---|---|---|---|
| r001 | run_0001_seed_358341059 | 358341059 | COMPLETE | 62.8530 | 7.9354 | 61.4521 |
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
