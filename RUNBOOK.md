# CoIN + TRACE 式 Replay 实验 — 运行手册（RUNBOOK）

> ⚠️ 权威最新状态见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)，交接口径见 [HANDOFF.md](HANDOFF.md)。
> 本手册面向云端环境搭建与实验执行。仓库：<https://github.com/VincentAshton/CoIN>

## 1. 实验设计（摘要）

- 基准：CoIN（arXiv:2403.08350），LLaVA-1.5-7B，前 4 任务顺序微调：ScienceQA → TextVQA → ImageNet → GQA
- 新增：TRACE 式 Replay —— 每轮微调后，对前序任务数据按比例抽样回放训练 1 epoch
- 扫描：ratio ∈ {0.10（基线）, 0.01}，共 2 组
- 指标：Truth Alignment 的 MAA / BWT（T=4；每序列 10 次评估）
- 目标：验证 ratio=0.01 相对 0.10 是否明显下降（下降阈值）

## 2. 租卡要求（重要）

- **4× A100 80G**（与 TRACE 同规格；LoRA 微调 7B MLLM + ZeRO-3）
- **数据盘 ≥ 500G**：前 4 任务图片约 200G+（ImageNet 训练图是大头，~150G），
  另有模型 ~30G（vicuna + CLIP + projector）、checkpoint 与缓存余量
- 不用 TRACE 的 28G 盘 + 200G tmpfs 配置（图片必须落盘）

## 3. 云端环境搭建

```bash
# 1) 代码（云端可 git clone，与 TRACE 不同——本仓库无大文件）
git clone https://github.com/VincentAshton/CoIN.git && cd CoIN

# 2) Python 3.10 + 老版本栈（论文 2024-03 环境；A100 sm_80 兼容）
conda create -n coin python=3.10 -y && conda activate coin
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e . --no-deps          # 关键：--no-deps，跳过 pyproject 里的 torch 2.0.1 重装
pip install transformers==4.32.0 deepspeed==0.14.0 peft==0.4.0 \
    bitsandbytes==0.41.0 accelerate==0.21.0 sentencepiece==0.1.99 \
    tokenizers==0.13.3 safetensors==0.4.2 tqdm shortuuid einops timm \
    pillow opencv-python numpy==1.26.4
# flash-attn（需编译，gcc + CUDA toolkit；sm_80）
pip install flash-attn==2.5.6 --no-build-isolation
# 注意：requirements.txt 中 git+ssh 私有依赖行已注释（ETrain 已 vendored）

# 3) 指令数据（HF）
pip install -U huggingface_hub
huggingface-cli download Zacks-Chen/CoIN \
    --repo-type dataset --include "Instructions_Original/ScienceQA/*" \
    --local-dir playground
# 同理下载 TextVQA / ImageNet / GQA 子目录（或整仓下载后只保留 4 任务）
# 最终布局：playground/Instructions_Original/<Task>/{train,test|val}.json

# 4) 图片：按 train/test json 的 image 字段组织到 cl_dataset/（字段是相对路径）
#    前 4 任务来源：ScienceQA 图（官方 drive）、TextVQA train+test 图（~30G）、
#    ImageNet（ILSVRC2012，需注册）、GQA images（~20G）
#    提示：先在云端写个小脚本核对 json 引用的图片 100% 存在再开跑（缺失=白烧卡）

# 5) 模型（LLaVA-1.5-7B 三件套）
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir checkpoints/LLaVA/clip-vit-large-patch14-336
huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 \
    --local-dir checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5-projector
# 目录名必须与 scripts/CoIN_Replay/run_replay_exp.sh 的默认 BASE_MODEL/VISION_TOWER/PROJECTOR 一致
# 注意：checkpoint 目录名必须含 "lora"（builder.py 按名字判断加载方式），脚本已遵守
```

## 4. 跑实验

```bash
cd CoIN
# 前置检查（脚本内 preflight：模型/数据/脚本/环境缺失即终止）
bash scripts/CoIN_Replay/run_replay_exp.sh 0.1     # 先跑基线
bash scripts/CoIN_Replay/run_replay_exp.sh 0.01    # 再跑低比例
# 或一条命令顺序扫：
bash scripts/CoIN_Replay/run_sweep.sh 0.1 0.01

# 常用覆盖（默认值见脚本头部）：
# GPUS=0,1,2,3  ACCUM=16（4卡保持有效batch=896与论文一致）  SAMPLE_MODE=random
# SEED=1234  LORA_R=192  LR=2e-4  REPLAY_EPOCHS=1
# DS_CONFIG=scripts/zero3.json   # 冒烟通过后可选：去掉 CPU offload 提速
```

结果落盘结构：
```
results/CoIN_Replay/ratio_<r>/
  ├── run_manifest.json      # 配置快照（run ID/超参/环境版本/git commit）
  ├── coin_metrics.json      # A 矩阵 + MAA/BWT（严格聚合，原子写）
  ├── .round1_done ... .round4_done   # 每轮断点（存在=该轮完成，可续跑）
  ├── .complete              # 整组完成的权威标志
  ├── logs/                  # 各轮训练日志
  └── <Task>/round<j>/       # 评估产物（output_result.jsonl / Result.text）
checkpoints/CoIN_Replay/ratio_<r>/<Task>_llava_lora/   # 每轮 LoRA（下一轮续接）
playground/Replay/ratio_<r>/round<j>_train.json        # replay 数据（可审计）
```

**完成判定 = `.complete` 存在**（不是 coin_metrics.json 存在）。

## 5. 看结果

```bash
cat results/CoIN_Replay/ratio_0.1/coin_metrics.json
cat results/CoIN_Replay/ratio_0.01/coin_metrics.json
# 对比 0.1 vs 0.01 的 MAA/BWT，找下降阈值（看 ≥1% 量级差异，评估有随机性）
```

## 6. 关键约束与坑（务必读）

1. **不要动论文评估口径**：四个任务的 question-file/评估器原样使用（ScienceQA=test.json、
   TextVQA=val.json + TextVQA_0.5.1_val.json、ImageNet=test.json、GQA=test.json + testdev_balanced）。
2. **Replay 数据默认 random+seed=1234**（与 TRACE 的前缀抽样不同，统计上更稳）；
   如要复刻 TRACE 口径：`SAMPLE_MODE=prefix`。
3. **有效 batch 一致性**：4 卡必须 ACCUM=16（14×4×16=896 = 论文 8 卡×8×14）。
   改 GPU 数/超参 = 重跑两个比例。
4. **eval 脚本里 create_prompt 已容错**（只影响 Reasoning Capability，不影响 Truth Alignment）。
5. **checkpoint 目录名必须含 "lora"**（builder.py 按名字判断），脚本已用 `<Task>_llava_lora`。
6. **老版本栈**：transformers 4.32.0 的 TrainingArguments 与新版不兼容（如 use_cache 行为），
   严格按第 3 节版本装；flash-attn 编译失败可先降级跑 eager（LLaVA 支持）。
7. **磁盘**：cl_dataset 图片 ~200G+；每组完成建议先拉回本地再清 checkpoint（不自动删）。
8. **图片完整性**：开跑前核对 json 引用图片 100% 存在（缺失会静默跳图导致准确率失真）。
9. 本实验**不需要** vllm / Qwen1.5-32B（不做 Reasoning Capability 评估）。

## 7. 成本估算（粗略）

- 前 4 任务 LoRA 训练：ScienceQA 12k / TextVQA 34k / ImageNet 129k / GQA 72k 样本，
  4×A100 上每任务约 0.5~1.5h；replay 阶段 <0.5h；每轮评估（1..j 任务）约 0.5~1h
- 单组（4 轮）约 6~10h；两组约 15~20h（4 卡并行）
- 按 A100 80G $1~2/卡/时：约 $60~160
