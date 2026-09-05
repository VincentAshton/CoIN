# CoIN + Replay — 完整复现手册（REPRODUCE）

本手册面向**全新环境从零复现**本仓库的持续学习实验（CoIN 前 4 任务顺序 LoRA 微调 +
TRACE 式 Replay 回放比例扫描）。目标：照此执行一次得到与已发布结果一致（同 seed/同配置下
确定性数值）的完整产物。约需 1 台 4×A100-80GB（每比例一轮 ~7-8h 墙钟 + 数据准备时间）。

> 若已持有作者现成环境（云端持久卷含全部数据/模型/conda env），可跳过 §3-4，
> 直接从 §5 门禁开始。本手册不含任何主机/账号/凭据信息。

## 0. 这是什么、成果是什么

- 基准：CoIN（arXiv:2403.08350，LLaVA-1.5-7B 顺序持续学习；上游 zackschen/CoIN）
- 方法：4 任务顺序全量 LoRA 微调（1 epoch/任务），每轮完成后用**前序任务**按比例
  `ratio` 做 prefix 回放（round≥2），再评估全部已学任务；只算 Truth Alignment（A 矩阵）
- 任务顺序：ScienceQA → TextVQA → ImageNet → GQA
- 已发布成果（单 seed=1234，固定任务顺序；**10 个 eval 单元 = 三角矩阵交集，非 10 次
  独立重复；描述性证据，非统计显著性结论**）：

| 指标 | ratio=0.10 | ratio=0.01 |
|---|---|---|
| MAA | 57.5057 | 60.4406 |
| CoIN BWT（(1/T)Σ_i[A[T,i]−A[i,i]]） | +17.2306 | −13.6299 |
| Final Avg | 55.7834 | 46.1925 |
| 终局旧任务均值 | 61.7445 | 47.7000 |
| replay 真步（r2/r3/r4） | 23 / 85 / 317 | 3 / 9 / 32 |

主结论：ratio=0.01 提高部分中间轮次/新任务表现（replay 干扰小），但终局旧任务均值低
~14.04 点、最终平均低 ~9.59 点——不足以满足终局保持目标（稳定性—可塑性权衡）。
完整结果与对比分析在 `results/coin-replay-r010-20260904` 分支
`docs/experiments/coin_replay/`。

分支角色：默认分支 = 入口页 + 上游代码；本分支（experiment）= 运行代码 + 工具 + 本手册 +
内部记录 `docs/internal/`（HANDOFF/EXPERIMENT_LOG/RUNBOOK/dataset）；results 分支 = 结果。

## 1. 仓库与锁定 commit

```bash
git clone https://github.com/VincentAshton/CoIN.git
cd CoIN && git checkout experiment/coin-replay-presweep-20260903
# 运行代码锁定：069b608 == 17cfa66（逐字节一致，git diff 0 行）。复现必须用本分支代码。
```

## 2. 硬件与预算

- 4×A100-80GB（NVLink 互联），CPU 内存 ≥200GB 建议（zero3 + CPU offload）
- 每比例一轮 ~7-8h；两个比例 ≈ 16h 算力 + 每轮后评估穿插在内（预算已含）
- 吞吐锚点：task 段 accum16 ~48-61s/optimizer step；replay 段 accum1 ~4-8s/step；
  GQA task 段最慢 ~111s/step（长序列）

## 3. 环境搭建（conda，python 3.10）

```bash
conda create -p $ENV_ROOT/coin python=3.10 -y   # ENV_ROOT 自定（≥30GB 空间）
conda activate $ENV_ROOT/coin   # 或 export PATH=$ENV_ROOT/coin/bin:$PATH
# torch 2.0.1 cu118（必须此版本：LLaVA/DS/transformers 4.32 老栈锁死）
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd CoIN && pip install -e . --no-deps && pip install -r requirements_coin.txt
# requirements_coin.txt 之外必须手动 pin：
pip install "protobuf==4.25.3" "ninja==1.11.1.4" "huggingface-hub==0.22.2"
# flash-attn 2.5.6：走 Dao-AILab GitHub releases 官方预编译 wheel
#（精确匹配 cu118 + torch2.0 + cp310 标签），勿源码编译
```

关键环境变量（写进 env.sh 并 source）：
`PIP_CACHE_DIR` `XDG_CACHE_HOME` `TORCH_EXTENSIONS_DIR` `TRITON_CACHE_DIR`
`CUDA_CACHE_PATH` `HF_HOME` `TRANSFORMERS_CACHE` `TMPDIR` 全部指到持久盘；
`HF_ENDPOINT=https://hf-mirror.com`（国内；hf.co 直连不可达时）。
验证：`pip check` = 0；`python -c "import torch,deepspeed,transformers,peft"` 无错。

## 4. 数据与模型（~55-60GB 落盘；全部 gated:false 无需 token）

仓库相对路径约定（脚本硬编码相对路径，用符号链接指向实际存储即可）：
`playground/Instructions_Original/`、`cl_dataset/`、`checkpoints/LLaVA/`。

模型三件套（HF repo → 目标目录）：
| 组件 | repo | 文件 |
|---|---|---|
| LLM | `lmsys/vicuna-7b-v1.5` | pytorch_model-00001/00002-of-00002.bin + tokenizer |
| vision | `openai/clip-vit-large-patch14-336` | pytorch_model.bin |
| projector | `liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5` | mm_projector.bin |

指令数据（`HF Zacks-Chen/CoIN` → `playground/Instructions_Original/<Task>/`）：
ScienceQA train 12,726/test 4,241；TextVQA train 34,602/val 5,000；ImageNet train
129,833/test 5,050；GQA train 72,140/test 12,578。eval 问题文件 = ScienceQA/ImageNet/GQA
用 test.json、TextVQA 用 val.json。

图片（`cl_dataset/<Task>/...`，json 的 `image` 字段相对此根）：
- ScienceQA：官方包 `ScienceQA.zip`，**内层布局 `train/<id>/image.png` 需重排成
  `images/{train,val,test}/<id>/image.png`**；辅助 metadata：`pid_splits.json`、`problems.json`
- TextVQA：`train_images/<hex>.jpg`（~25,119 文件；**val 也引用 train_images**）；
  辅助：`TextVQA_0.5.1_val.json`
- ImageNet：ILSVRC2012 官方 tar（或 Kaggle `imagenet-object-localization-challenge`）；
  只需 **101 个 synset 子集**（train.json 引用即类内全量 129,833）+ val 5,050，布局
  `ImageNet_withlabel/{train/<synset>/,val/}`——用 `scripts/CoIN_Replay/tools/imagenet_extract.py`
  流式提取（约 9 分钟），勿逐类解压全 tar
- GQA：`GQA/images/<id>.jpg`；辅助：`testdev_balanced_questions.json`

**布局正确性以 `preflight_data.py` 为准**（missing=0/corrupt=0），勿用 find 数文件
（TextVQA 多题共享一图是正常的）。ImageNet 必须在 preflight 带
`--layout-map {"ImageNet":"ImageNet_withlabel"}`（json 无 withlabel 前缀）。

## 5. 门禁（正式运行前必须全绿；完成后**停止等人工验收**，严禁自动串联 sweep）

```bash
# ① 数据完整性（canary 任务集；输出 missing=0/corrupt=0）
python scripts/CoIN_Replay/preflight_data.py --data-dir cl_dataset \
  --layout-map '{"ImageNet":"ImageNet_withlabel"}' --json-root playground/Instructions_Original
# ② 单元测试（期望 Ran 89 tests OK）
bash scripts/CoIN_Replay/run_tests.sh
# ③ 端到端 canary（B→E；含 tensor 级断言/断点复用）
bash scripts/CoIN_Replay/canary.sh
```

canary 通过 = 门禁闭环（真 optimizer steps、tensor 级 diff、round 加载）已验证，
此时才被允许跑正式 sweep。

## 6. 正式运行（一次一个比例；两个比例分别批准/验收）

```bash
cd CoIN
export REPLAY_ACCUM=1          # 方案 D：replay 段 accum 恒 1（全比例统一；禁止按 ratio 变）
export ENFORCE_MIN_STEPS=1
export GPUS=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
bash scripts/CoIN_Replay/run_sweep.sh 0.10    # 或 0.01（严禁一次两个/重跑已发布比例）
```

建议 tmux 唯一会话 + `set -o pipefail` + 记录 commit/env/PID/退出码（参考本仓库正式运行
日志结构：`logs/formal/ratio_<r>_<ts>/`）。产物落盘（仓库相对）：
`results/CoIN_Replay/ratio_<r>/`（10 个 eval 单元 + `coin_metrics.json` +
`run_manifest.json` + `.roundN_done`/`.complete`）、`checkpoints/CoIN_Replay/ratio_<r>/`
（每轮 task + replay adapter）、`playground/Replay/ratio_<r>/`。

## 7. 验收（对照已发布数值前必须全过）

1. 退出码 0、`.complete` + 4 个 round markers
2. checkpoint：每比例 **7 个 = 4 task(round1-4) + 3 replay(round2-4)**，全部可加载 +
   参数 finite（`coin_lib.py ckpt-validate`）
3. task→replay tensor 级 diff：round2/3/4 各 **448/448** 参数 changed
   （adapter_model.bin = 224 lora_A + 224 lora_B；missing/unexpected=0；规范化 hash 不同。
   **mm_projector 权重在独立文件 non_lora_trainables.bin，需单独对比**：4/4 changed）
4. DS 真步 = round manifest `replay_plan.ds_expected_updates` = trainer_state.global_step
   （0.10: 23/85/317；0.01: 3/9/32。**HF 日志 global_step/LR 不可信，以 DS engine 计数 +
   tensor diff 为权威**——DS 0.14 accum16 下短 replay 尾部不提交是已知坑）
5. 10 个 eval 单元齐全；prediction 数量==唯一 ID==问题集（4241/5000/5050/12578）
6. 独立重算：用结果分支 `docs/experiments/coin_replay/recompute_metrics.py` 从
   acc_sources.json 构造 A 矩阵 → MAA/BWT 与 `coin_metrics.json` 交叉验证零差异
7. 无残留 GPU 进程；ratio=0.1 与 0.01 的 run_manifest 白名单 diff 仅 ratio 派生项

## 8. 关键坑速查（均在本仓库实验中被实证踩过）

- HF `global_step`/LR callback ≠ 真实更新：真步 = per-rank micro // gas；铁证 = tensor diff
- LoRA key 双格式：单适配器 ckpt 无 `.default.` 段、`base_model.` 前缀 11 字符
- bf16 tensor 不能 `.numpy()`：参数 hash 用 `.view(torch.uint8)` 原始字节
- transformers TrainingArguments `__post_init__` 冻结：复刻入口须先 `_frozen = False`
- `pkill -f` 会杀掉含同 token 的自身 SSH：用 `[x]` 括号且把 pkill 拆到独立命令
- `torch.load` 旧版 peft `adapter_model.bin`；`import protobuf` 必失败（无顶层模块），
  版本用 `importlib.metadata.version`
- 环境/依赖版本记录在 run_manifest `env`（DS 日志污染时正则抽版本号）
- 严格模式：aggregate/验收脚本任一步失败即非零退出且不写半成品

## 9. 发布协议（结果对外呈现）

- 结果（去敏小文件）推 `results/coin-replay-r010-20260904` 分支
  `docs/experiments/coin_replay/`；**永不推** checkpoint/模型/数据/原始 predictions/完整日志
- manifest 入档前 sanitize（路径相对化、env 版本清理、无凭据）
- 提交后需全新 clone + 只靠公开包重算交叉验证（CROSS-VALIDATION PASS）

## 10. 文档索引

- `docs/internal/RUNBOOK.md` — 运行手册（环境/流程细节）
- `docs/internal/HANDOFF.md` — 交接与状态（含每阶段结论）
- `docs/internal/EXPERIMENT_LOG.md` — 逐日实验记录（含方案 D 设计决策）
- `docs/internal/dataset.md` — 数据集 card
- results 分支 `docs/experiments/coin_replay/README.md` — 结果总览（A 矩阵/公式/对比/限制）
