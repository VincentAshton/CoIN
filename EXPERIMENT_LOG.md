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

## 3. 工程加固（2026-09-01 第二轮，工单 1-9，未 push）

### 3.1 修改文件清单（相对 855bfb2）

| 文件 | 改动 |
|---|---|
| `scripts/CoIN_Replay/coin_lib.py` | 新增核心库：train_plan（分辨率报告）、ckpt_validate（文件+参数 hash+finite）、verify_predictions（数量/唯一/集合/顺序）、artifact_check、manifest 写/恢复校验（config hash）、validate_round、round_manifest（原子写） |
| `scripts/CoIN_Replay/build_replay_data.py` | floor(N*ratio) 前缀（N*ratio<1→k=0，全 0→退出）；无图样本允许（ScienceQA）；conversations 合法性；有图时校验存在/非空/PIL；sidecar manifest（源 SHA256/N/k/选中 ID/索引/输出 SHA256）；--nested-with 断言 0.01⊆0.10 |
| `scripts/CoIN_Replay/preflight_data.py` | 新增：四任务 train/test/val + 评估辅助文件（pid_splits/problems/TextVQA_0.5.1_val/testdev_balanced_questions）、图片存在/大小/PIL/越界/大小写、布局首段匹配、SHA256 报告（缓存复用） |
| `scripts/CoIN_Replay/run_replay_exp.sh` | 目录契约统一；checkpoint 链 task/replay 分离（round<j>_{task,replay}_llava_lora，下一轮只加载 replay）；manifest 显式配置 export + 恢复 config hash 校验（先于 .complete 短路）；每轮 round_manifest + validate_round；评估临时目录+原子 rename；QUESTION_FILE 传递；--seed/--data_seed；ENFORCE_MIN_STEPS 门禁；DRY_RUN 模式；TASKS_JSON 可配；preflight 集成 |
| `scripts/CoIN_Replay/aggregate_coin.py` | ratio 优先读 run_manifest.json（目录名解析为兜底） |
| `scripts/LLaVA/Eval/{1,2,3,4}_eval_*.sh` | chunk PID 逐个收集（任一失败整次失败）；去 create_prompt；EVAL_DRY_RUN/EVAL_FAULT_INJECT；QUESTION_FILE 环境覆盖；dry-run merge 只收 chunk 文件 |
| `ETrain/Train/LLaVA/llava_trainer.py` | create_optimizer 增加 mm_projector 独立分组（lr=mm_projector_lr）+ 完整性断言 + 分组明细打印；log_optimizer_scheduler；load_model_from_previous_task LoRA key 完整性检查（missing/unexpected 即失败）；LrLogCallback 逐 step LR |
| `ETrain/Train/LLaVA/train.py` | seed/data_seed/lr/lora 配置真实性日志；LrLogCallback 注册；训练后 optimizer/scheduler 实况 |
| `scripts/zero3_offload.json` | **移除 optimizer+scheduler 段**（上游会让 DeepSpeed 用单组 AdamW + WarmupLR，顶掉冻结配置的 mm_projector_lr 与 cosine） |
| `scripts/LLaVA/Train/{3,4,5,8}_*.sh`、`Train_MOE/{2,6,7}_*.sh`、`Qwen/Eval/3_eval_ImageNet.sh` | 解决上游 0.4_MOE 分支残留的 8 处 merge conflict（保留 HEAD 侧规范路径）——原文件无法通过 bash -n |
| `scripts/CoIN_Replay/tests/` | 新增 5 个测试模块共 46 个用例 |
| `scripts/CoIN_Replay/run_tests.sh` | 门禁 A：bash -n + py_compile + unittest + ds 配置无 scheduler 检查 |
| `scripts/CoIN_Replay/canary.sh` | 门禁 B-E（云端 4×A100）：GPU 冒烟 / 迷你 round1→round2 / 真实 chunk kill / 完整 ScienceQA round1 + probe |
| `scripts/CoIN_Replay/smoke/` | smoke_gpu.py（NCCL+flash-attn）、smoke_ds.py（deepspeed 最小任务）、probe_logits.py（加载一致性）、build_canary_data.py |
| `scripts/CoIN_Replay/env/requirements_coin.txt` | 云端依赖锁定（未实测） |

### 3.2 冻结配置层面的修复（不是改超参，是让冻结配置真实生效）

| 发现 | 上游实际行为 | 修复后 |
|---|---|---|
| mm_projector_lr 静默失效 | create_optimizer 只建 decay/no_decay 两组 | mm_projector 独立分组 lr=2e-5，无参数即报错 |
| cosine 被替换 | zero3_offload.json 带 WarmupLR scheduler + AdamW optimizer 段 | 移除两段，trainer 的 cosine + 分组优化器生效 |
| seed 未显式 | 未传 --seed/--data_seed | 显式 1234/1234 并日志确认 |

### 3.3 测试与验证状态（2026-09-01 本地）

- ✅ 门禁 A：bash -n 全部脚本、py_compile、46/46 单测（build_replay 12 / coin_lib 16 / eval 契约 3 / preflight 8 / orchestrator dry-run 4 + 其他）、zero3_offload 无 scheduler 检查 —— 全过
- ✅ 端到端 DRY_RUN：2 任务全链路（manifest/两轮训练假 ckpt/replay 构建/评估契约/聚合/.complete/恢复跳过/配置不一致拒绝/故障注入无 .complete）
- ⏸ 门禁 B-E（canary）：**阻塞——云端 4×A100 实例未租用**（ebcloud 32433 连接拒绝）；canary.sh 已就绪，租卡后 `bash scripts/CoIN_Replay/canary.sh` 一键执行

---

## 4. 云端部署 + canary probe 修复（2026-09-02）

### 4.1 云端实例与代码部署

- 实例：ebcloud cs-66731-55d7d-server（ssh-cn-huabei1.ebcloud.com:30267，4×A100-80GB SXM4，NVLink P2P，~31.26 元/时；cgroup：80 CPU / 400GB 内存）
- 代码：本地 bundle（coin-6ff09a5.bundle）→ `/root/data/coin/project`，detached HEAD=6ff09a5（2026-09-02 部署）
- 备份：`/root/data/coin/git_backup/`（bundle + binary patch）；审计报告 `coin-deploy-audit-20260902.md`
- 目录：`/root/data/coin/{conda_envs,conda_pkgs,hf_cache,torch_cache,datasets,models,checkpoints,predictions,logs,tmp,pip_cache,cache}` 全在 1T Lustre 持久卷；env.sh 已补缓存变量（PIP_CACHE_DIR/XDG_CACHE_HOME/TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR/CUDA_CACHE_PATH + CUDA_HOME + nvcc/conda PATH + HF_ENDPOINT=hf-mirror，因 hf.co/GDrive 从实例不可达）；diff 见 logs
- 环境：conda env 已建（`/root/data/coin/conda_envs/coin` python=3.10）；torch 2.0.1 cu118 等按 requirements_coin.txt 安装（见 logs/environment/）

### 4.2 canary E probe_logits 缺陷（评审批准修复）

- **根因**：`smoke/probe_logits.py` 无条件 `q['image']`；ScienceQA test.json 含纯文本题（question_id=4 无 image 字段）→ KeyError → canary E probe 必然失败（本地 DRY_RUN 无法覆盖该路径，需真实 torch+ckpt）
- **修复（commit `9be6312`）**：与官方 `ETrain/Eval/LLaVA/CoIN/model_vqa_science.py` 语义严格对齐——
  - 无图题：images=None 传给模型，prompt 不含 `<image>` token（不注入空白图/伪造路径）
  - 有图题：加载并预处理图片，prompt=`<image>\n`+text；text 统一 `replace('<image>','').strip()`
  - image 字段声明但路径越界/缺失/空/损坏 → 立即非零退出（严格失败，不跳过）
  - 固定 probe 集（question_id 4 无图 + 5 有图）；probe manifest 原子写：question_id/has_image/图片相对路径/prompt+input_ids+logits hash/数据文件 hash
  - 两次加载 input_ids/logits hash 完全一致才 PASS；断言 mm_use_im_start_end=False（LLaVA-1.5）
- **测试**：新增 `tests/test_probe_logits.py` 18 用例（本地 17 过 + 1 PIL 依赖跳过；云端环境 PIL 齐全）
- ⚠️ 本地门禁注意：8 个既有用例（test_preflight_data ×7 + test_build_replay_data ×1）在**无 PIL 的本地机器**失败（`ModuleNotFoundError: PIL`），与本次修复无关；云端 Phase 5 以 run_tests.sh 全绿为准
- **锁定 hash 更新**：`6ff09a5` → 部署 HEAD `9be6312`（云端 /root/data/coin/project）；文档 commit 另见 git log
- **云端同步**：bundle `coin-probe-fix.bundle` + binary patch → git_backup；`git fetch <bundle>` + checkout `9be6312`；验证 HEAD/status/bundle verify/probe_logits.py sha256

### 4.3 数据/模型准备状态（截至 2026-09-02）

- 指令数据（hf-mirror 已下载，sha256 见部署记录）：ScienceQA 12726/4241、TextVQA 34602/5000、ImageNet 129833/5050、GQA 72140/12578（train/test|val）
- 模型三件套：全部 gated:false，待下载（lmsys/vicuna-7b-v1.5、openai/clip-vit-large-patch14-336、liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5）
- 图片：ScienceQA（GDrive 实例不可达 → 需 HF 镜像替代）、TextVQA（fbail GET 206 可下）、GQA（stanford 200 可下）、ImageNet（官方注册凭据 → **阻塞，待用户提供途径**；canary 不需要）
- ⚠️ 正式 sweep 必须 `PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel","GQA":"."}'`（json 图片路径首段分别为 ImageNet_withlabel 与 `./`）；canary（ScienceQA/TextVQA）不受影响

### 4.4 部署日志

- `/root/data/coin/logs/`：audit_report、environment/（pip freeze/conda list/pip check）、run_tests.log、canary.log、deployment_lock_hashes.txt

### 4.5 canary 首跑缺陷修复（2026-09-02，评审二轮批准）

canary B–E 云端首跑暴露 3 个问题（本地零 GPU 无法覆盖），评审逐项批准修复：

1. **protobuf 版本缺口（环境，C 训练崩溃）**：transformers 4.32 加载 llama tokenizer 时
   `sentencepiece_model_pb2` 导入在 protobuf 5/6 下静默失败 → UnboundLocalError。
   requirements_coin.txt 原漏 pin；作者 requirements.txt 锁定 protobuf==4.25.3。
   修复：requirements_coin.txt 补 `protobuf==4.25.3` + run_tests.sh 新增 [A5] 版本断言
   （protobuf 可导入时必须 ==4.25.3，零依赖环境跳过）。
2. **smoke_gpu.py flash-attn bwd 必失败（测试脚本）**：q/k/v 未设 requires_grad →
   `o.sum().backward()` 无 grad_fn。修复：三个张量 requires_grad=True，backward 后严格断言
   q/k/v grads 均存在、全部 finite、至少一个非零；若修复后仍失败视为真实环境阻塞，不削弱测试。
3. **smoke_ds.py 单进程 MPI 探测失败（测试脚本）**：deepspeed.initialize 无分布式环境走
   MPI（缺 mpi4py）。修复（评审方案 A）：canary.sh 改 `torchrun --standalone --nproc_per_node=4`
   启动（与正式四卡路径一致），smoke_ds.py 分布式化：
   - WORLD_SIZE==4 断言、LOCAL_RANK 选 GPU、每 rank 不同卡、bf16（对齐正式 bf16+tf32）
   - 显式解析正式配置的 auto 字段（fp16 off/bf16 on、gradient_clipping=1.0、
     zero_force_ds_cpu_optimizer=False——均与 HF trainer 在正式路径的解析一致，config 文件不改）
   - GatheredParameters（deepspeed 公共 API）验证：loss finite + 参数 finite +
     step 后至少一个参数非零变化；全部 rank 通过才 exit 0；finally 清理 process group
   - canary.sh 以 timeout 900 包裹防挂死
- 验证（云端 4×A100 实机）：smoke_gpu 4 卡 fwd/bwd/grads 全 PASS；smoke_ds 4 卡
  loss=0.917969 finite / param_changed=True / 全局汇总 PASS
- 代码 commit：见 4.6；不改变训练参数/replay ratio/正式评估口径/数据

### 4.6 canary B 修复 commit（2026-09-02）

- 代码 commit：`7bab671`（5 文件：smoke_gpu/smoke_ds/canary.sh/requirements_coin.txt/run_tests.sh）
- 文档 commit（本段）：见 git log HEAD（部署锁定 HEAD）
- 实机验证（修复后、同步前先行独立跑通）：
  - smoke_gpu：4 rank NCCL sum=4 ✓；flash-attn 2.5.6 fwd OK ✓；q/k/v grads 存在+finite+非零 ✓；bwd OK ✓
  - smoke_ds：4 rank rank/local_rank/gpu 映射正确；loss=0.917969 loss_finite ✓ params_finite ✓
    n_params=4 param_changed=True（全 rank）；全局汇总 PASS
- requirements_coin.txt SHA256 已更新（见 deployment_lock_hashes.txt）


