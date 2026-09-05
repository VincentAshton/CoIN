# CoIN + Replay — 回放比例扫描正式结果（ratio=0.10 与 ratio=0.01）

持续学习基准 CoIN（arXiv:2403.08350，LLaVA-1.5 7B）前 4 任务顺序 LoRA 微调 +
TRACE 式 Replay 的回放比例扫描实验：ratio ∈ {0.10, 0.01} 各一轮完整运行（单 seed=1234，
固定任务顺序 ScienceQA → TextVQA → ImageNet → GQA；每轮：新任务全量微调 → 前序任务按
ratio 抽样回放（round≥2）→ 在完成 task+replay 后的 checkpoint 上评估全部已学任务）。
10 个评估单元是「(round, 任务)」三角矩阵交集（r1:1、r2:2、r3:3、r4:4），**不是 10 次
独立重复运行**；本实验为单 seed 描述性证据，不构成统计显著性或跨任务顺序的普遍因果证明。

## A 矩阵（Truth Accuracy %，行 = 评估轮，列 = 任务；对角项为「该轮 task+replay 后」自评）

ratio=0.10（COMPLETE 2026-09-04，~7.6h）
| | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.26 | — | — | — |
| round2 | 72.86 | 39.03 | — | — |
| round3 | 73.52 | 57.56 | 4.02 | — |
| round4 | 74.13 | 55.91 | 55.19 | 37.90 |

ratio=0.01（COMPLETE 2026-09-05，~7.1h）
| | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.12 | — | — | — |
| round2 | 66.52 | 53.67 | — | — |
| round3 | 70.29 | 45.95 | 70.83 | — |
| round4 | 61.66 | 51.84 | 29.60 | 41.67 |

## 指标公式（论文 Section 3.1.3 口径；A[j,i] 见上，T=4）
- MAA = (1/T) Σ_j [ (1/j) Σ_{i≤j} A[j,i] ]
- CoIN BWT = (1/T) Σ_i [ A[T,i] − A[i,i] ]（正文官方口径）
- Final Avg = (1/T) Σ_i A[T,i]
- 附录补充：T−1 BWT = (1/(T−1)) Σ_{i<T} [ A[T,i] − A[i,i] ]
- 终局旧任务均值 = (1/(T−1)) Σ_{i<T} A[T,i]（排除最后一任务的终局均值）
- 对角项均值 = (1/T) Σ_i A[i,i]

## 汇总对比（0.01 − 0.10）
| 指标 | ratio=0.01 | ratio=0.10 | Δ (0.01−0.10) |
|---|---|---|---|
| MAA | 60.4406 | 57.5057 | +2.9350 |
| CoIN BWT | −13.6299 | +17.2306 | −30.8605 |
| Final Avg | 46.1925 | 55.7834 | −9.5909 |
| 终局旧任务均值 | 47.7000 | 61.7445 | −14.0445 |
| 对角项均值 | 59.8224 | 38.5528 | +21.2696 |
| T−1 BWT（附录） | −18.1732 | +22.9741 | −41.1473 |

逐单元 delta 与完整结构化对比见 comparison.json；数值全部经 recompute_metrics.py
从两组 acc_sources.json 独立重算并与 coin_metrics.json 交叉验证（A 差异 ≤5e-7、
MAA/BWT 差异 0，CROSS-VALIDATION PASS）。

## 逐任务轨迹（每列随轮次）
- ScienceQA：0.10 全程稳定（73.26→72.86→73.52→74.13）；0.01 先降后短暂回升再降
  （73.12→66.52→70.29→61.66），终局净下降 11.46 点（73.12→61.66）。
- TextVQA：0.10 从低起点回升（39.03→57.56→55.91）；0.01 起点高但后续低于 0.10
  （53.67→45.95→51.84）。
- ImageNet：0.10 round3 自评仅 4.02（其 task checkpoint 初学自评 96.93%，只读诊断见
  analysis/imagenet_round3_diagnostic.md——与「replay 不含当前任务、干扰刚学权重」的
  机制一致，并排除评估工程故障），round4 大 replay 复习回升至 55.19；0.01 round3 保留
  70.83（干扰小）但 round4 跌至 29.60（复习不足）。
- GQA：round4 自评 0.10=37.90 / 0.01=41.67。

## 解读：稳定性—可塑性权衡
ratio 同时改变两个量：replay 样本数 N 与真实 optimizer steps（0.10 replay 真步
23/85/317 vs 0.01 的 3/9/32；配置其余全部相同——见 analysis/postrun_manifest_diff.json
白名单 diff）。观察到的差异与如下机制一致：更大的 replay 对新学任务（当轮刚训完的权重）
干扰更强（压低对角项：0.10 对角均值 38.55 vs 0.01 的 59.82），但对已学任务提供更强
复习巩固（抬高终局旧任务与遗忘控制：0.10 CoIN BWT 为正 +17.23、0.01 为负 −13.63）。
主结论（推荐口径）：**在固定任务顺序、数据划分、seed 与训练预算下，ratio=0.01 提高了
部分中间轮次及新任务表现，但使终局旧任务均值降低约 14.04 个百分点、最终平均降低约
9.59 个百分点。因此结果支持 0.01 不足以满足终局保持目标，同时体现稳定性—可塑性权衡。
本实验为单 seed 描述性证据，不构成统计显著性或跨任务顺序的普遍因果证明。**

## 配置与可复现
- runtime commit（运行代码，本地=GitHub=云端三端一致，结果分支不携带运行代码）：
  `17cfa66f009bd1fd1f5d360307f97d4249bf2c5c`
- 数据/模型 revision：两 ratio 完全一致（run_manifest.sanitized.json 的 data_revision /
  model_config_hash / ds_config_hash 相同；data_revision = bf6bd4ee…2516196d）
- 其余配置（task accum=16/eff 896、replay accum=1/eff 56、LR 2e-4 / mm 2e-5、cosine、
  zero3+CPU offload、每任务 1 epoch）两 ratio 相同；唯一差异 = ratio → replay N/steps
  → 输出路径
- 完整配置快照：ratio_0.10/run_manifest.sanitized.json、ratio_0.01/run_manifest.sanitized.json
- 重算：`python3 recompute_metrics.py .`（读两组 acc_sources.json，输出各目录
  recomputed_metrics.json + 顶层汇总 + comparison.json）

## 完整性验收（两组均 PASS）
- exit_code=0；.complete；round1-4 markers；7 个 checkpoint（4 task + 3 replay）全部
  可加载、参数 finite（逐路径 inventory 见 analysis/checkpoint_inventory.json）
- task→replay tensor 级 diff：round2/3/4 各 448/448 参数 changed（missing=0/unexpected=0/
  shape 一致/finite/规范化 hash 不同）——replay 确实改变权重，非元数据差异
- DS 真步与 manifest 一致：0.10 = 23/85/317；0.01 = 3/9/32
- prediction 数量/ID 与问题集逐一吻合（4241/5000/5050/12578；ID 级 hash 对比见
  analysis/id_hash_comparison.json）
- 0.10 原结果在 0.01 运行期间未被修改（SHA256 复核，见 ratio_0.10/results_sha256.txt
  与 analysis/）

## 目录
```
docs/experiments/coin_replay/
  README.md                        本文件（总览/矩阵/公式/对比/限制）
  comparison.json                  0.01 vs 0.10 结构化逐项对比（recompute 输出）
  recompute_metrics.py             公开独立重算脚本（双 ratio）
  recomputed_metrics.json          双 ratio 重算汇总
  ratio_0.10/                      0.10 结果包（README/coin_metrics/acc_sources/
                                   run_manifest.sanitized/validation_report/results_sha256/
                                   recomputed_metrics）
  ratio_0.01/                      0.01 结果包（同构）
  analysis/
    comparison_r001_vs_r010.md     详细对比分析（逐单元表/遗忘曲线/结论）
    postrun_manifest_diff.json     0.10 vs 0.01 白名单配置 diff（工程交叉验证）
    id_hash_comparison.json        20 个 eval 单元 ID 级 hash 对比
    checkpoint_inventory.json      两组 7 checkpoint 逐路径组件/SHA256 清单
    imagenet_round3_diagnostic.md  ImageNet round3=4.02 只读诊断（0.10）
```
