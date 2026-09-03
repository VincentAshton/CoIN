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

## 5. 当前进度（2026-09-01 工程加固完成）

- ✅ 立项侦察 + 实验设计确认（2 组：0.1 / 0.01，prefix replay，与 TRACE 口径一致）
- ✅ 第一轮本地改造 + 推送 fork：commit `855bfb2`（GitHub 当前 HEAD）
- ✅ 第二轮工程加固（工单 1-9）：commit `6ff09a5`（**仅本地，未 push**——用户约束）
  - 修复：目录契约、mm_projector_lr 失效、zero3_offload 顶掉 cosine、上游 8 处 merge conflict、
    eval chunk 吞错、checkpoint 链、manifest/恢复校验、preflight、分辨率报告（详见 EXPERIMENT_LOG 3 节）
  - 门禁 A 全绿：bash -n + py_compile + **46/46 单测**（`bash scripts/CoIN_Replay/run_tests.sh`）
  - 端到端 DRY_RUN 全链路验证通过（含故障注入）
- ✅ **云端实例已租并部署**（ebcloud 30267，cs-66731-55d7d-server）：代码经 bundle 恢复至
  `/root/data/coin/project`，HEAD = `9be6312`（含 6ff09a5 加固 + probe 修复），
  备份 `/root/data/coin/git_backup/`（bundle + binary patch）
- ✅ **canary E probe_logits 缺陷已修复（commit 9be6312，评审批准）**：兼容 ScienceQA 纯文本样本，
  与官方评估语义对齐；固定 probe 集（question_id 4 无图 + 5 有图）+ probe manifest + 18 单测
  （详见 EXPERIMENT_LOG 4.2 节）
- ✅ **canary B 缺陷修复（评审二轮批准，commit 见 EXPERIMENT_LOG 4.6）**：smoke_gpu requires_grad+梯度断言；
  smoke_ds 改 torchrun --standalone 四卡分布式（ZeRO-3 bf16 实测 PASS）；requirements_coin 补
  protobuf==4.25.3 + run_tests [A5] 版本断言（详见 EXPERIMENT_LOG 4.5 节）
- ✅ **DRY_RUN flake 修复（评审方案 A，commit 见 EXPERIMENT_LOG 4.8）**：coin_lib ckpt_validate
  <1MB 尺寸守卫（DRY_RUN 假文件确定性降级，真实 checkpoint 校验不变）+ A5 import 写法修正
- ⏳ 进行中（2026-09-02）：云端装环境（requirements_coin.txt）→ 模型三件套 + 图片下载
  （ScienceQA GDrive 不可达需 HF 镜像替代；ImageNet 需官方注册凭据——阻塞）→
  preflight → run_tests → canary B-E
- ⏸ 未启动正式 run_sweep.sh（canary 完成后必须人工验收）

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
# 1. clone / pull 仓库（GitHub 当前 855bfb2；本地 HEAD 9be6312（含 6ff09a5 加固 + probe 修复）未 push，
#    云端 /root/data/coin/project 已部署 9be6312；恢复/同步以 git_backup/coin-probe-fix.bundle 为准）
git clone https://github.com/VincentAshton/CoIN.git && cd CoIN

# 2. 环境（RUNBOOK 第 3 节 + scripts/CoIN_Replay/env/requirements_coin.txt）——老栈，装完冒烟
python -c "import torch, transformers, deepspeed, peft; print(torch.__version__, transformers.__version__)"

# 3. 模型三件套 + 指令数据 + 图片（RUNBOOK 3.3-3.5）；preflight 核对图片完整性
python scripts/CoIN_Replay/preflight_data.py --data-dir playground/Instructions_Original \
    --image-dir cl_dataset --out-report results/CoIN_Replay/preflight_report.json

# 4. 零 GPU 门禁（可在任何机器先跑）
bash scripts/CoIN_Replay/run_tests.sh

# 5. canary（4×A100，全部通过才允许正式 sweep）
bash scripts/CoIN_Replay/canary.sh

# 6. 正式 sweep（不混配置；0.01 round2 replay 是单 step，需显式确认）
ENFORCE_MIN_STEPS=1 ALLOW_SINGLE_STEP_REPLAY=1 bash scripts/CoIN_Replay/run_sweep.sh 0.1 0.01

# 7. 每组完成立即校验 coin_metrics.json + 拉回本地 + 回填 EXPERIMENT_LOG
```

## 9. 关键约束（务必遵守）

- 改 GPU 数/batch/accum/epoch/seed/抽样/数据版本/模型版本 → 重跑两组，不混用
- 完成判定只看 `.complete`；失败即 fail-fast，不继续烧卡
- 不自动删 checkpoint；结果先拉回本地再清理
- 不做 Reasoning Capability（省 vllm/Qwen1.5-32B）；如后续要做，README 的 Eval_GeneralKnowledge 流程可用

## 10. 当前进度快照（2026-09-02 封存，本段取代 §8/§9 的旧状态描述）

### 10.1 部署锁定状态
- **代码 HEAD = `a29b29d`**（本段所在的最终提交；GitHub origin/CoIN 已同步至此，2026-09-02 解除"不推送"约束）
- 云端实例 ebcloud:30267（CoIN 实例）`/root/data/coin/project` 工作树 = a29b29d、干净；
  备份 bundle 在 `/root/data/coin/git_backup/`（6ff09a5 → coin-canaryv5.bundle 全链）
- **canary v5 全绿**（A/B/C/D/E；C 含评审重审三重断言）；run_tests v4 = **Ran 73 tests OK**
- 完整修复链（均评审批准）：probe_logits 9be6312 → smoke_gpu/smoke_ds 7bab671 →
  DRY_RUN flake coin_lib bfbc1b0 → previous-task LoRA `_norm` f8ede2f → canary E 双评估 6e40594 →
  **canary C 重审** 89067ea（coin_lib `ckpt-tensor-diff` + canary.sh C 公式化数据/三重断言 +
  smoke/verify_round3_load.py）+ 2434a8b（_frozen 镜像）。细节：EXPERIMENT_LOG 4.6–4.13

### 10.2 数据与环境（全在 /root/data 持久卷，实例重租不丢）
- 模型：/root/data/coin/models（vicuna-7b-v1.5 13G + clip 1.6G + projector 42M）
- 图片：/root/data/coin/datasets/cl_dataset（project/cl_dataset 符号链接指向）；
  **preflight 权威 PASS（2026-09-02）**：SQA refs 8235 / TextVQA refs 39602（unique 25119，
  多题共享图）/ GQA refs 84718，全部 missing=0 corrupt=0
- 环境：/root/data/coin/conda_envs/coin（torch 2.0.1+cu118 / transformers 4.32.0 / peft 0.4.0 /
  deepspeed 0.14.0 / flash-attn 2.5.6 / protobuf 4.25.3 / python 3.10.21）
- 封存：/root/data/coin/logs/archive_final_20260902/（sha256_manifest_final.txt + run_tests_v4/canary_v5 日志）
- **正式 sweep 数据门禁已全 PASS（2026-09-03）**：ImageNet 就位（公共卷官方 tar 提取 101 类
  129,833 + val 5,050 → cl_dataset/ImageNet_withlabel/）；四任务全量 preflight 三连 PASS
  （logs/full_preflight/four_tasks_20260903_114629/、presweep_four_tasks_20260903_123141/）；
  data_sha256=30a878a24653f942f50af1ed8bd6c7118a6fd407dcb1a7f2a84b215ac69c4602；
  manifest logs/full_preflight/manifest_20260903.json（sha256 b67771670f63…）

### 10.3 恢复/续接步骤
```bash
# 代码：直接 GitHub（已同步 7c45394）或本地/云端 git_backup bundle
git clone https://github.com/VincentAshton/CoIN.git && cd CoIN   # HEAD=7c45394

# canary 已验收通过（v5 全绿，证据在持久卷 logs/archive_final_20260902/canary_v5.log，
# 不随实例关闭丢失）——同节点重租后【不需要】重跑 canary。
# 仅当环境有实质变动（不同实例配置/重装依赖）才重验门禁：
#   bash scripts/CoIN_Replay/run_tests.sh   # 期望 Ran 73 OK（零 GPU，5 分钟环境健全性快检）
#   bash scripts/CoIN_Replay/canary.sh      # 全绿（C 自清重训/E 断点复用）——环境变动时

# 下一步正式实验（2026-09-03 起分阶段放行，见 §11）：四任务 preflight 已 PASS → presweep GO → 阶段
# I-IV：同步 → single-step 门禁 → 启动前检查 → 仅 ratio=0.1
# preflight 命令（layout-map 已实证修正——GQA json 带 ./ 但 Path.parts 折叠 → 首段=GQA=默认期望，
# 不可配 GQA:"."（永不匹配）；仅 ImageNet 需映射）：
python scripts/CoIN_Replay/preflight_data.py --data-dir playground/Instructions_Original \
    --image-dir cl_dataset --out-report results/CoIN_Replay/preflight_report.json \
    --tasks ScienceQA TextVQA ImageNet GQA \
    --layout-map '{"ImageNet":"ImageNet_withlabel"}'
# 正式运行（分阶段，IV 只启动 0.1，0.01 另行批准）：
export ENFORCE_MIN_STEPS=1 ALLOW_SINGLE_STEP_REPLAY=1 \
  PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
bash scripts/CoIN_Replay/run_sweep.sh 0.1
```
- canary C 数据公式：round1_train_n = ceil(3 × world × batch × accum / ratio)（当前 = 240，
  replay 24 条 = 3 optimizer steps）；断言含 replay global_step≥2、tensor 级 task!=replay
  （coin_lib ckpt-tensor-diff）、round3 previous-task 加载（verify_round3_load.py）
- 已知观察：replay 训练末步 LR=0.0 是 cosine 终点（canary 3 steps 下第 3 步为 lr 0，无碍；
  正式 14+ steps 同标准调度）；0.01 组 replay 若 N×ratio<1 会由 build_replay_data 拒绝（空 replay 禁止）

## 11. 正式实验分阶段放行（2026-09-03，权威分支 experiment/coin-replay-presweep-20260903）

- 状态：四任务 preflight PASS、presweep GO（logs/presweep/PRESWEEP_GO_NOGO_20260903.md）、
  正式 sweep **未启动**；用户批准分阶段：I 同步（本分支）→ II single-step replay 门禁
  （0.01 round2 N=127 / round3 N=473 的 1-step replay 实测 LR>0 + 参数真变）→ III 启动前检查
  → IV 仅 `bash scripts/CoIN_Replay/run_sweep.sh 0.1` → 验收后停止，0.01 另行批准
- 0.01 single-step 背景：有效 batch 896（4×14×16），replay N=127/473 < 896 → ceil=1 step；
  warmup_steps=int(0.03×1)=0 → 唯一 update 步 LR=lr=2e-4（cosine 首步=峰值，update 先于
  scheduler step）——理论非零，阶段 II 以 tensor 差异为最终实证
- 运行 env：ENFORCE_MIN_STEPS=1、ALLOW_SINGLE_STEP_REPLAY=1（0.01 的 1-step replay 显式放行，
  进 manifest 可审计）、PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
- 权威代码版本 = 本分支 HEAD（实验记录见 EXPERIMENT_LOG 4.14/4.15）；云端 detached HEAD 需与
  本地/GitHub 三端 hash 一致后再动实验

### 11.1 续接状态（2026-09-03 晚，阶段 II No-Go 后停机点）

- **阶段 I 完成**：本分支 HEAD 已含 tools(imagenet)+docs 记录并三端同步（见 EXPERIMENT_LOG 4.15 首条）
- **阶段 II = No-Go**：0.01 round2/3 replay（N=127/473）在正式配置（accum16/zero3 "auto"）下
  **0 次真实权重更新**（DS engine 需 896 样本/真步；HF global_step=1 是假象；adapter==task ckpt
  逐字节相同 sha256 13255ed6…）。0.1 replay 真步 1/5/19（r2 低于 C-1 ≥2）。未启动 0.1/0.01。
  完整证据：logs/presweep/single_step_replay/ + EXPERIMENT_LOG 4.15 + 复现工具 tools/single_step_gate.*
- **停机续接点 = 候选方案决策**（推荐 A：run_replay_exp 加 REPLAY_ACCUM env——0.01→1、0.1→8，
  仅 replay 段覆盖 accum、manifest 留痕；需评审批准改锁定代码）→ run_tests 补用例 →
  重跑门禁（预期 changed=448）→ 阶段 III 检查 → IV 仅 0.1（run_sweep.sh 0.1 +
  ENFORCE_MIN_STEPS=1 ALLOW_SINGLE_STEP_REPLAY=1 PREFLIGHT_ARGS='--layout-map
  {"ImageNet":"ImageNet_withlabel"}'）
- 实例关机安全：数据/代码/日志全在 coinssd（/root/data）持久卷；gate 临时产物在
  /root/data/coin/tmp/single_step_gate（保留，评审可能需要）；重租 4×A100 挂 coinssd 即可续接
