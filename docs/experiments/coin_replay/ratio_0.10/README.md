# CoIN+Replay — ratio=0.1 正式结果（2026-09-04）

持续学习（Continual Learning）基准 CoIN（arXiv:2403.08350，LLaVA-1.5 7B）前 4 任务
顺序 LoRA 微调 + TRACE 式 Replay 扫描实验 —— ratio=0.1 一轮完整运行结果。

## 研究目标
扫描回放比例对持续学习 Truth Alignment（MAA/BWT）的影响；本目录为 ratio=0.10
基线结果。ratio=0.01 结果见上级目录 ratio_0.01/（COMPLETE 2026-09-05）。

## 任务顺序
ScienceQA → TextVQA → ImageNet → GQA（每轮：新任务全量微调 → 前序任务按
ratio=0.1 prefix 抽样回放（round≥2）→ 评估已学全部任务）

## A 矩阵（Truth Accuracy %，行=评估轮，列=任务）

| | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | **73.26** | — | — | — |
| round2 | 72.86 | **39.03** | — | — |
| round3 | 73.52 | 57.56 | **4.02** | — |
| round4 | 74.13 | 55.91 | 55.19 | **37.90** |

- **MAA = 57.51**（平均每轮已学任务准确率）
- **BWT = 17.23**（平均末轮 vs 初学差值）
- 最终平均准确率 = 55.78
- 独立重算：`recompute_metrics.py`（从 acc_sources.json 构造 A 并验证，
  cross-validation 与 coin_metrics.json 逐位一致）

## 配置摘要（sanitized manifest 见 run_manifest.sanitized.json）
- 模型：LLaVA-1.5 7B（vicuna-7b-v1.5 + clip-vit-large-patch14-336），LoRA r=192 α=256
- 4×A100-SXM4-80GB，DeepSpeed ZeRO-3 + CPU offload，bf16+tf32，grad checkpoint
- 每任务 1 epoch；LR 2e-4（LoRA）/ 2e-5（mm_projector）；cosine，warmup 0.03；
  seed 1234；max_len 2048
- **task 段：gradient_accumulation=16 → effective batch = 896**（论文口径 8卡×8×14）
- **replay 段：gradient_accumulation=1（方案 D，REPLAY_ACCUM=1）→ effective batch = 56**
  —— 设计修订：DS 0.14 在 accum16 下不提交短 replay 尾部（N<896 时 0 真实更新），
  方案 D 以全 ratio/round 统一的 replay accum=1 修复，保证小回放段真实训练
- replay：prefix 子集、1 epoch、无 LIMA
- **三轮 replay 真实 optimizer 步数：round2 = 23（N=1272）/ round3 = 85（N=4732）/
  round4 = 317（N=17715）**（trainer_state.global_step 与 manifest 精确一致，tensor
  级 448/448 参数变化验证）
- runtime commit：`17cfa66f009bd1fd1f5d360307f97d4249bf2c5c`
- data_sha256：见 run_manifest.sanitized.json `data_revision`
- 运行时长：~7.6h 墙钟（2026-09-04 03:30 → 11:04 CST）

## 完整性验收（validation_report.md 全项 PASS）
- exit_code=0；.complete + round1-4 markers；7 个 checkpoint（4 task：round1-4 +
  3 replay：round2-4）可加载且参数 finite
- task/replay tensor diff 448/448 changed（round2-4）
- 10 个 eval 单元（三角矩阵交集）齐全；prediction 数量/ID 与问题集逐一吻合
  （ScienceQA 4241 / TextVQA 5000 / ImageNet 5050 / GQA 12578）
- 结果 SHA256 清单：results_sha256.txt

## ImageNet round3 诊断
round3 的 ImageNet 自评 4.02 显著低于 round4 的 55.19——只读诊断见
../analysis/imagenet_round3_diagnostic.md（结论：与 replay 内容与方法行为一致，
并排除评估工程故障）。

## 状态
- ratio=0.10：COMPLETE（验收通过，2026-09-04）
- ratio=0.01：COMPLETE（2026-09-05，见 ../README.md 与 ratio_0.01/）
- 运行代码锁定 commit 17cfa66（结果分支与 runtime 分离，runtime HEAD 未前进）
- 双 ratio 对比分析见 ../analysis/comparison_r001_vs_r010.md
