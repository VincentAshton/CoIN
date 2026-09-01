# CoIN + Replay 实验交接文档（HANDOFF）

> 目标读者：接手实验恢复、结果验收的 AI agent / 工程协作者。
> 仓库：<https://github.com/VincentAshton/CoIN>（fork 自 zackschen/CoIN，上游 commit `41411ab`）
> 运行手册见 `RUNBOOK.md`，实验记录见 `EXPERIMENT_LOG.md`。

## 1. 一句话概述

在 **CoIN 基准**（arXiv:2403.08350，LLaVA-1.5-7B，前 4 任务：ScienceQA → TextVQA → ImageNet → GQA）
的顺序 LoRA 微调上加入 **TRACE 式 Replay**（每轮微调后对前序任务数据按比例回放 1 epoch），
验证：**回放比例从 0.10 降到 0.01 时，Truth Alignment 的 MAA/BWT 是否明显下降**（下降阈值）。

## 2. 实验设计

- **方法**：顺序微调（LoRA r=192/alpha=256）+ 每轮后 replay 1 epoch（前序任务按 ratio 取数据前缀）
- **模型**：LLaVA-1.5-7B（vicuna-7b-v1.5 + CLIP ViT-L/14-336 + LLaVA-1.5 projector）
- **任务**：前 4 任务（论文随机序开头 4 个）；每序列评估 1+2+3+4=10 次
- **比例**：0.10（基线）、0.01 —— 共 2 组
- **指标**（论文 Section 3.1.3，Truth Alignment，0-100）：
  - `MAA = (1/T)·Σ_j (1/j)·Σ_{i≤j} A_{j,i}`（全程平均精度）
  - `BWT = (1/T)·Σ_i (A_{T,i} − A_{i,i})`（负值=遗忘）
- **基线策略**：以自己环境跑的 ratio=0.10 为基线（TRACE option B），绝不回退论文绝对值
- **抽样**：默认 prefix + seed 1234（与 TRACE 完全一致：取前序任务数据前缀，不重新随机抽样）；
  `SAMPLE_MODE=random` 为备选（统计更均匀但口径与 TRACE 不同）
- **Replay 训练**：1 epoch，同任务 LR 2e-4（与 TRACE 的 replay 规则一致；
  差异：CoIN 无 LIMA 类额外语料，replay 数据仅含前序任务子集）

### 2.1 冻结配置（不可随意改，改了必须重跑两组）

| 项 | 固定值 |
|---|---|
| GPU | 4 × A100-80GB（`--include localhost:0,1,2,3`） |
| per_device_train_batch_size | 14 |
| gradient_accumulation_steps | 16（4 卡有效 batch=896，对齐论文 8 卡 ×8） |
| LoRA | r=192, alpha=256, dropout 0.05 |
| lr / mm_projector_lr | 2e-4 / 2e-5，cosine，warmup 0.03 |
| 每任务 epoch / replay epoch | 1 / 1 |
| seed / sample_mode | 1234 / prefix（TRACE 一致） |
| precision | bf16 + tf32 + gradient_checkpointing |
| deepspeed | zero3_offload.json（冒烟后可试 zero3.json 提速） |
| 任务顺序 | ScienceQA → TextVQA → ImageNet → GQA（固定） |
| 数据 | HF Zacks-Chen/CoIN Instructions_Original（train/test|val.json） |
| 评估 | 温度 0，官方各任务评估器，create_prompt 容错（不做 Reasoning） |

## 3. 运行环境

- **云端**：4× A100 80G，数据盘 ≥500G（ebcloud 等，按小时计费）
- **软件**：Python 3.10、torch 2.0.1+cu118、transformers 4.32.0、deepspeed 0.14.0、
  peft 0.4.0、flash-attn 2.5.6、bitsandbytes 0.41.0（老栈，勿升版本）
- **关键路径**（云端，均可被环境变量覆盖）：
  - 代码 `<repo>/`（git 仓库，可 clone）
  - 模型 `checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5`、`checkpoints/LLaVA/clip-vit-large-patch14-336`、
    `checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector/mm_projector.bin`
  - 指令数据 `playground/Instructions_Original/<Task>/{train,test|val}.json`
  - 图片 `cl_dataset/`（~200G+，按 json image 字段的相对路径组织）
  - 结果 `results/CoIN_Replay/ratio_<r>/`，checkpoint `checkpoints/CoIN_Replay/ratio_<r>/<Task>_llava_lora/`
- **Git 仓库不含数据/模型/结果**（`.gitignore` 忽略 `/checkpoints/ /cl_dataset/ /playground/ /results/`）

## 4. 执行流程（脚本链路）

```
run_sweep.sh（顺序扫 0.1 0.01，fail-fast）
  └─ run_replay_exp.sh <ratio>
       ├─ preflight（模型/数据/脚本/deepspeed 存在性，失败即终止）
       ├─ run_manifest.json（run ID + 冻结配置 + 环境版本 + git commit，原子写）
       └─ 每轮 j=1..4（.round<j>_done 断点，已完成跳过）：
            ├─ train_one：任务 j 顺序微调（prev=上一轮 ckpt）→ <Task>_llava_lora/
            ├─ (j≥2) build_replay_data.py → playground/Replay/ratio_<r>/round<j>_train.json
            │         → train_one：replay 1 epoch（prev=本轮 ckpt，写回同一目录）
            └─ 评估任务 1..j（eval 脚本 + 产物严格校验，缺失即失败）
       └─ aggregate_coin.py（严格模式：任一产物缺失/损坏即非零退出）→ coin_metrics.json
       └─ touch .complete（完成的权威标志）
```

**完成判定 = `.complete` 存在**（不是 coin_metrics.json 存在）。

## 5. 当前进度（2026-09-01）

- ✅ 立项侦察 + 实验设计确认（2 组：0.1 / 0.01）
- ✅ 本地改造完成：Replay 编排/聚合脚本、官方脚本 bug 修复（layground 拼写、merge conflict、
  chencheng 硬编码路径、git+ssh 依赖）、文档三件套（待 push 到 fork）
- ⏳ 待办：push 到 GitHub → 租卡（4×A100 + ≥500G 盘）→ 云端装环境 → 数据/模型下载 →
  图片完整性核对 → 冒烟（极小数据）→ canary（0.1 的 round1 ScienceQA）→ 正式 sweep

## 6. 已修复的问题（相对上游 41411ab）

| 问题 | 修复 |
|---|---|
| requirements.txt 含 git+ssh 私有依赖，装不了 | 注释该行（ETrain 已 vendored） |
| eval_prompt_slim.sh 有未解决 merge conflict | 保留新路径侧，清除标记 |
| 3_eval_ImageNet.sh 拼写 `./layground` + `RESULT_DIR//` | 修复（否则 ImageNet 评估直接失败） |
| eval 脚本 RESULT_DIR 写死，实验无法隔离 | 环境变量化 `${RESULT_DIR:-默认}` |
| eval 脚本 create_prompt 失败会导致整链失败 | 容错（仅 Reasoning 需要，TA 不受影响） |
| builder.py/llava_trainer.py 等 5 处作者机器硬编码 sys.path | 移除 |
| 无实验编排/断点/清单/聚合 | 新增 scripts/CoIN_Replay/* |

## 7. 尚未解决 / 潜在风险

1. **老栈兼容**：torch 2.0.1 + flash-attn 2.5.6 在云端镜像上的编译/兼容未实测（A100 sm_80 理论 OK）；
   失败预案：flash-attn 降级 eager（LLaVA 支持）。
2. **ImageNet 图片**：~150G 且需官方注册下载，是数据准备的大头；务必先核对 json 引用图片完整性。
3. **Replay 超参未调**：replay 用任务同 LR 2e-4 是默认选择，未验证最优；如 0.1 vs 0.01 差异不显著，
   可考虑 replay LR 减半或 epoch 增加（但改了配置要重跑两组，先按默认跑）。
4. **zero3_offload 性能**：CPU offload 会拖慢训练；冒烟后评估 zero3.json（无 offload）是否放得下。
5. **推理时间**：每序列 10 次评估 × 温度 0 自回归，4 卡并行 chunked；GQA test 1k 条较快，
   TextVQA 5k / ImageNet 5k / ScienceQA 4k 为主（参考 TRACE 经验：7B 推理 ~10-28s/step 属正常）。

## 8. 恢复实验的操作步骤（实例恢复后）

```bash
# 1. clone / pull 仓库
git clone https://github.com/VincentAshton/CoIN.git && cd CoIN

# 2. 环境（见 RUNBOOK 第 3 节）——老栈，装完跑一次 import 冒烟
python -c "import torch, transformers, deepspeed, peft; print(torch.__version__, transformers.__version__)"

# 3. 模型三件套 + 指令数据 + 图片（RUNBOOK 3.3-3.5）；核对图片完整性

# 4. 单任务冒烟（极小数据：把 ScienceQA train.json 截 50 条 → 1 卡 1 epoch）
#    再 canary：bash scripts/CoIN_Replay/run_replay_exp.sh 0.1（round1 即 ScienceQA 全量）

# 5. 正式：bash scripts/CoIN_Replay/run_sweep.sh 0.1 0.01
#    断点续跑：脚本自动跳过 .round<j>_done / .complete 的已完成部分

# 6. 每组完成立即校验 coin_metrics.json + 拉回本地 + 回填 EXPERIMENT_LOG
```

## 9. 关键约束（务必遵守）

- 改 GPU 数/batch/accum/epoch/seed/抽样/数据版本/模型版本 → 重跑两组，不混用
- 完成判定只看 `.complete`；失败即 fail-fast，不继续烧卡
- 不自动删 checkpoint；结果先拉回本地再清理
- 不做 Reasoning Capability（省 vllm/Qwen1.5-32B）；如后续要做，README 的 Eval_GeneralKnowledge 流程可用
