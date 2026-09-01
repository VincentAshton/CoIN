# CoIN 实验记录（EXPERIMENT_LOG）

> 权威记录：本文件 + `HANDOFF.md`（交接口径）+ `RUNBOOK.md`（运行手册）。
> 仓库：本地 `/home/vincent/CoIN`（clone 自 zackschen/CoIN，commit `41411ab`，分支 CoIN）；
> GitHub 远程待建（本地 fork 计划见 HANDOFF.md）。

---

## 1. 项目概述（2026-08-31 立项侦察）

- **论文**：CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model（arXiv:2403.08350, 2024-03）
- **代码**：https://github.com/zackschen/CoIN（Apache-2.0，80 stars）
- **一句话**：MLLM 持续指令微调 benchmark——10 数据集 8 任务顺序 LoRA 微调，评估灾难性遗忘；
  发现遗忘主因是指令跟随能力丧失而非推理知识遗忘；提出 MoELoRA（多专家 LoRA）缓解。
- **与 TRACE 的关系**：同属持续学习（CL）范式，用户按 TRACE 流程推进（租卡 → 数据/模型/代码 →
  改进调控 → 本地+GitHub 留痕）。

### 1.1 论文关键事实（已核实）

- **8 任务顺序（随机序）**：ScienceQA → TextVQA → ImageNet → GQA → VizWiz → Grounding → VQAv2 → OCR-VQA
- **训练设置**：LoRA 参数高效微调（论文 vanilla r=192/alpha=256；MoELoRA r=128 + expert_num N，
  每 expert rank r/N，gate=Softmax(xWg)），base LLM + vision encoder 冻结，每任务 1 epoch，
  lr 2e-4 + mm_projector_lr 2e-5，cosine，bf16，gradient checkpointing，model_max_length 2048
- **指标**：
  - Truth Alignment（传统 ground-truth 匹配，各任务口径不同）
  - Reasoning Capability（Qwen1.5-32B-Chat 用 vLLM 打分 0-10，prompt 见 ETrain/Eval/rule.json）
  - MAA（全程平均精度；需每轮训练后评估**之前所有任务** → 每序列 8 任务共 36 次评估）
  - BWT（向后迁移，负值=遗忘）
- **论文结论**：LLaVA/Qwen-VL 严重遗忘（BWT -32.62 / -16.94），MiniGPT-v2 几乎不遗忘；
  Reasoning Capability 遗忘远小于 Truth Alignment → 遗忘≈指令跟随能力丧失；
  MoELoRA expert N=2/4/8 单调改善（N=8: MAA 42.76 / BWT -25.91 vs LoRA 32.97 / -32.62）；
  对比 LwF/EWC（lambda=0.1）也最优；消融：任务顺序、指令模板（Original/Diverse/10Type）、
  数据量（0.1~1.0，0.4 时最佳）。
- **论文基线**：Multi-task（联合微调）、Zero-shot（预训练直接测）。

### 1.2 代码结构（已核实，本地 clone commit 41411ab）

- `ETrain/`：基于 LLaVA-1.5 的 MLLM 训练/评估框架（Train: LLaVA/Qwen/LAVIS；Models: LLaVA/Qwen/MiniGPT；Eval: 各任务 model_vqa_* + eval_*）
- `scripts/LLaVA/Train|Train_MOE|Train_EWC|Train_LWF`：8 任务顺序训练脚本
- 顺序训练机制：`--previous_task_model_path <上一任务 output_dir>` 把上一任务 LoRA 传给下一任务
- MoELoRA 入口：`--expert_num N`；EWC：`--EWC`；LwF：`--LWF`
- 评估：温度 0、多卡分块并行；`evaluate_score.py` 汇总分数；`create_prompt.py` + `eval_GenealKnowledge.py`（vLLM + Qwen1.5-32B）做 Reasoning Capability
- 训练脚本写死 8 卡（`--include localhost:0,1,2,3,4,5,6,7`）+ zero3_offload + per-GPU batch 14

### 1.3 数据需求（已核实来源）

- 指令 JSON：HF `Zacks-Chen/CoIN`，含 Instructions_Original / _Diverse / _10Type / _Qwen 四套模板；
  train 合计约 57 万条、test 约 26 万条（表 1）
- 图片（须自行下载，总量 **~300G+**）：
  - COCO2014 train2014(~13G)+val2014(~6G)+test2015(~13G)（VQAv2/RefCOCO 系/Grounding）
  - GQA images ~20G；TextVQA train+test ~30G；OCR-VQA ~60G（Google Drive）；ImageNet ~150G+（需注册）
  - ScienceQA / VizWiz 较小；RefCOCO 注释来自 bvisionweb1
- 统一目录 `./cl_dataset`，指令 JSON 的 image 字段为相对路径

### 1.4 模型需求

- 主模型：LLaVA-1.5-7B = vicuna-7b-v1.5（HF）+ CLIP ViT-L/14-336 + LLaVA Model Zoo 的 mm_projector.bin（预训练投影器）
- 论文另测：Qwen-VL-7B、MiniGPT-v2（可选）
- Reasoning evaluator：Qwen1.5-32B-Chat（HF 公开，bf16 ~64G），vLLM 推理

### 1.5 环境坑清单（TRACE 教训对位，2026-08-31 发现）

1. requirements.txt 钉死老栈：torch 2.0.1 / transformers 4.32.0 / deepspeed 0.14.0 / flash-attn 2.5.6（需编译）→ 与现代 CUDA12 镜像大概率冲突，需适配；vllm 不在 requirements 里（Reasoning 评估需另装）
2. requirements.txt 含 `-e git+ssh://git@github.com/zackschen/Easy_Train_MLLM.git@...`：无该作者 SSH 权限装不了；但 ETrain 代码已 vendored 在仓库内 → 删该行直接 `pip install -e .`
3. 磁盘：图片 ~300G+，**不能**用 TRACE 的 28G 盘 + 200G tmpfs 配置，需 ≥500G 数据盘
4. 仓库含未解决 merge conflict 标记（`scripts/Eval_GeneralKnowledge/eval_prompt_slim.sh` 有 `<<<<<<< HEAD`）
5. 模块名拼写：`ETrain/Eval/eval_GenealKnowledge.py`（Geneal 拼错，import 时注意）
6. 每序列 36 次任务评估（MAA 需要）→ 推理量大，类似 TRACE 的 36 项预测校验

### 1.6 进度

- [x] 2026-08-31 立项侦察：论文全文 + 仓库代码/脚本/依赖已通读，本文档建立
- [x] 2026-09-01 实验设计确认（用户）：CoIN 基础上加 TRACE 式 Replay，验证回放比例下降阈值
- [x] 2026-09-01 本地改造：Replay 编排/聚合脚本 + 官方脚本 bug 修复 + 文档三件套（见第 2 节）
- [x] 2026-09-01 功能测试：build_replay_data（抽样数量/seed 可复现/round1 拒绝）+ aggregate_coin（MAA/BWT 与手算一致/严格模式）12/12 通过
- [x] 2026-09-01 推送 GitHub fork：commit `3fbedc9` → VincentAshton/CoIN（CoIN 分支，ls-remote 已验证）
- [ ] 租卡（4×A100 80G，数据盘 ≥500G）
- [ ] 云端环境 + 数据 + 模型下载
- [ ] 冒烟 → canary → 正式组（ratio 0.1 / 0.01）

---

## 2. 实验设计（2026-09-01 确认，Replay 比例下降阈值）

### 2.1 一句话

在 CoIN 基准（LLaVA-1.5-7B，前 4 任务）的顺序微调基础上加入 TRACE 式 Replay 训练，
验证：**回放比例 ratio=0.01 时 MAA/BWT 相对 ratio=0.1 是否明显下降**（找到下降阈值，与 TRACE 同思路）。

### 2.2 实验矩阵

| 组 | 模型 | 任务 | 比例 | 说明 |
|---|---|---|---|---|
| 1 | LLaVA-1.5-7B | ScienceQA→TextVQA→ImageNet→GQA | 0.10 | replay 基线 |
| 2 | LLaVA-1.5-7B | 同上 | 0.01 | 低比例 |

- 指标：Truth Alignment 的 MAA/BWT（论文 Section 3.1.3，T=4）
- 只做 Truth Alignment（不做 Reasoning Capability，无需 Qwen1.5-32B 评估器）
- 基线口径：用自己环境跑的 ratio=0.1 作基线（同 TRACE option B），不对比论文绝对值

### 2.3 Replay 机制（本 fork 新增，scripts/CoIN_Replay/）

- 每轮 j：① 顺序微调任务 j（LoRA r=192/alpha=256，1 epoch，续接上一轮 checkpoint）
  ② 若 j≥2：前 j-1 任务各取数据前缀的 ratio 子集（**与 TRACE 一致：prefix 抽样**，
  不重新随机抽样；seed 仅用于记录）合并为 replay 数据集，replay 训练 1 epoch（同 LR 2e-4，
  checkpoint 写回本轮目录）③ 评估任务 1..j（温度 0）→ 每序列共 10 次评估
- 与 TRACE 的差异：TRACE 的 replay 数据 = 历史任务 ratio 子集 + **完整 LIMA**；
  CoIN 无 LIMA 类额外记忆语料，replay 数据 = 前序任务 ratio 子集（其余规则一致）
- 备选：`SAMPLE_MODE=random` 可切随机抽样（统计上更均匀，但口径与 TRACE 不同，需说明）

### 2.4 冻结配置（4×A100 80G）

| 项 | 值 | 备注 |
|---|---|---|
| GPU | 4×A100 80G | 脚本默认 --include localhost:0,1,2,3 |
| per_device_train_batch_size | 14 | |
| gradient_accumulation_steps | 16 | 4 卡下有效 batch=14×4×16=896，与论文 8卡×8 一致 |
| LoRA | r=192, alpha=256 | 论文 vanilla LoRA 配置 |
| lr / mm_projector_lr | 2e-4 / 2e-5 | cosine |
| 每任务 epoch | 1；replay 1 | |
| seed / 抽样 | 1234 / prefix（TRACE 一致） | |
| precision | bf16 + tf32 + grad ckpt | 与论文脚本一致 |
| deepspeed | zero3_offload.json | 稳妥；冒烟后可选 zero3.json 提速 |

> ⚠️ 改变 GPU 数 / batch / accum / epoch / seed / 抽样方式 / 数据版本 / 模型版本，
> 必须重跑两个比例，不得与旧配置结果混合。

### 2.5 代码改动清单（2026-09-01，相对上游 zackschen/CoIN commit 41411ab）

| 文件 | 改动 |
|---|---|
| `scripts/CoIN_Replay/run_replay_exp.sh` | 新增：实验编排（训练+replay+评估+聚合），fail-fast/.complete/manifest/断点续跑 |
| `scripts/CoIN_Replay/run_sweep.sh` | 新增：多比例顺序扫描 |
| `scripts/CoIN_Replay/build_replay_data.py` | 新增：replay 数据集构建（random/prefix 抽样） |
| `scripts/CoIN_Replay/aggregate_coin.py` | 新增：A 矩阵 + MAA/BWT 严格聚合（原子写） |
| `scripts/LLaVA/Eval/{1,2,3,4}_eval_*.sh` | RESULT_DIR 环境变量化（实验隔离）；create_prompt 容错（只影响 Reasoning，不影响 TA） |
| `scripts/LLaVA/Eval/3_eval_ImageNet.sh` | 修复 `./layground` 拼写 bug + `RESULT_DIR//` 双斜杠 |
| `scripts/Eval_GeneralKnowledge/eval_prompt_slim.sh` | 修复未解决的 merge conflict 标记（保留新路径侧） |
| `ETrain/Models/LLaVA/builder.py` 等 5 文件 | 移除作者机器硬编码 `sys.path.append('/home/chencheng/...')` |
| `requirements.txt` | 注释 git+ssh 私有依赖行（ETrain 已 vendored） |
| `.gitignore` | 新增：checkpoints/cl_dataset/playground/results 不入库 |
| `RUNBOOK.md` / `HANDOFF.md` / 本文件 | 文档三件套 |

