# CoIN+Replay ratio=0.01 vs ratio=0.10 — 详细对比分析（2026-09-05）

数据来源：两组 acc_sources.json（10 个三角矩阵 eval 单元的聚合 accuracy，SQA 为
output_result.jsonl 原始 acc + correct/count，其余为 Result.text 原始值）与
coin_metrics.json（runtime 权威产物）。全部数值经 recompute_metrics.py 从 acc_sources
独立构造 A 矩阵重算并与 coin_metrics.json 交叉验证（CROSS-VALIDATION PASS，A 差异
≤5e-7，MAA/BWT 差异 0）。本实验为**单 seed（seed=data_seed=1234）描述性证据**：
10 个 eval 单元是「(round, 任务)」三角矩阵交集（非 10 次独立重复），不构成统计显著性
或跨任务顺序的普遍因果证明；文中差异均为「观察到的大幅差异」表述。

## 1. 配置一致性（白名单 diff，详见 postrun_manifest_diff.json）
唯一差异 = ratio 及其派生项：replay 样本数 N 与真实 optimizer steps（0.10 replay 真步
23/85/317 vs 0.01 的 3/9/32——ratio 同时改变这两个量）、run_id/时间戳/输出路径。
以下必须相同项全部一致：runtime commit 17cfa66（运行代码逐字节相同）、git dirty=clean、
数据/模型 revision hash、任务顺序、seed/data_seed=1234、task/replay batch（14 / eff
896 与 56）、task accum=16、replay accum=1、LR 2e-4、mm_projector_lr 2e-5、cosine
scheduler、DeepSpeed 配置（zero3_offload.json）、每任务 1 epoch、eval 入口与 generation
参数（同一运行代码）。

## 2. 逐单元 Accuracy（A 矩阵元素，0.01 vs 0.10；Δ=0.01−0.10）
| 单元 (任务/round) | 0.01 | 0.10 | Δ |
|---|---|---|---|
| ScienceQA r1 | 73.1195 | 73.2610 | −0.1415 |
| ScienceQA r2 | 66.5173 | 72.8602 | −6.3429 |
| ScienceQA r3 | 70.2900 | 73.5204 | −3.2304 |
| ScienceQA r4 | 61.6600 | 74.1335 | −12.4735 |
| TextVQA r2 | 53.67 | 39.03 | +14.64 |
| TextVQA r3 | 45.95 | 57.56 | −11.61 |
| TextVQA r4 | 51.84 | 55.91 | −4.07 |
| ImageNet r3 | 70.83 | 4.02 | +66.81 |
| ImageNet r4 | 29.60 | 55.19 | −25.59 |
| GQA r4 | 41.67 | 37.90 | +3.77 |

（SQA 为 6 位舍入源值；其余为 Result.text 两位原始值。）

## 3. 总体指标（公式见 coin_replay/README.md）
| 指标 | 0.01 | 0.10 | Δ (0.01−0.10) |
|---|---|---|---|
| MAA | 60.4406 | 57.5057 | +2.9350 |
| CoIN BWT | −13.6299 | +17.2306 | −30.8605 |
| Final Avg | 46.1925 | 55.7834 | −9.5909 |
| 终局旧任务均值 | 47.7000 | 61.7445 | −14.0445 |
| 对角项均值 | 59.8224 | 38.5528 | +21.2696 |
| T−1 BWT（附录口径） | −18.1732 | +22.9741 | −41.1473 |

## 4. 每任务遗忘/恢复（final−diagonal；A[T,i] − A[i,i]）
| 任务 | 0.01 | 0.10 |
|---|---|---|
| ScienceQA | −11.46（73.12→61.66）| +0.87（73.26→74.13）|
| TextVQA | −1.83（53.67→51.84）| +16.88（39.03→55.91）|
| ImageNet | −41.23（70.83→29.60）| +51.17（4.02→55.19）|
| GQA | 0（终局即对角）| 0 |
| 均值（CoIN BWT） | −13.63 | +17.23 |

注意：对角项 A[i,i] 是在「该 round 完成 task 训练 + replay 微调后的 checkpoint」上
评估（replay 不含当前任务），0.10 的对角项因此被当轮 replay 干扰压低（如 ImageNet
39.03→57.56 段、4.02 起点），其 BWT 正值部分来自该低起点；0.01 的负 BWT 为缺乏复习
的单调遗忘。

## 5. 逐任务轨迹与 ImageNet round3/4 行为
- ScienceQA：0.10 全程稳定（73.26→72.86→73.52→74.13）；0.01 先降后**短暂回升**再降
  （73.12→66.52→70.29→61.66），最终净下降 11.46 点（round3 的 70.29 高于 round2 的
  66.52，非单调衰减）。
- TextVQA：0.10 round2 低（39.03，被当轮 replay 干扰）后经大 replay 复习回升并保持
  （57.56→55.91）；0.01 起点高（53.67）但后续低于 0.10（45.95→51.84）。
- ImageNet：0.10 round3 自评 4.02——只读诊断（analysis/imagenet_round3_diagnostic.md）
  排除评估/checkpoint 工程故障（round3 task checkpoint 初学自评 96.93%，5050/5050 ID
  完整），与「round3 replay 不含 ImageNet、干扰刚学权重」的机制一致（机制层面为与该
  解释一致，非已证明事实）；round4 replay 重新含 ImageNet → 55.19。0.01 round3 干扰小
  保留 70.83，但 round4 replay 仅含少量 ImageNet 样本（N=1771 的约 1/4），GQA 全量 task
  训练后复习不足 → 29.60。
- GQA：round4 自评 0.10=37.90（被当轮 replay 干扰）vs 0.01=41.67。

## 6. 主结论（推荐口径）
在固定任务顺序、数据划分、seed 和训练预算下，ratio=0.01 提高了部分中间轮次及新任务
表现（中间/对角项：MAA +2.94、对角均值 +21.27，如 TextVQA r2 +14.64、ImageNet r3
+66.81、GQA r4 +3.77），但使**终局旧任务均值降低约 14.04 个百分点**（61.74→47.70）、
**最终平均降低约 9.59 个百分点**（55.78→46.19）。因此结果支持 0.01 不足以满足终局
保持目标，同时体现稳定性—可塑性权衡：更小 replay 对新任务干扰更小（可塑性↑），但
对已学任务的复习巩固不足（稳定性↓，CoIN BWT 由正转负、终局旧任务失守）。本实验为
单 seed 描述性证据，不构成统计显著性或跨任务顺序的普遍因果证明。

## 7. 口径说明
- 主指标采用 CoIN BWT（论文 Section 3.1.3 官方口径，含全部 T 个任务项）；T−1 BWT
  仅在附录补充，不用于正文结论。
- 终局旧任务均值排除最后一任务 GQA（其 final−diagonal 恒为 0，会稀释遗忘信号）。
- checkpoint 逐路径 inventory 见 checkpoint_inventory.json（两组各 7 个：4 task +
  3 replay，非 8）。
