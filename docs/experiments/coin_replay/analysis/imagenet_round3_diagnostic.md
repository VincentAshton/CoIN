# IMAGE_NET_R3_DIAGNOSTIC — Round 3 ImageNet 4.02 只读诊断（2026-09-04）

## 问题
正式 ratio=0.1 结果中 round3 的 ImageNet 自评 acc = **4.02**，而 round4 的
ImageNet = 55.19。需判定：方法行为（replay 干扰/遗忘）还是评估工程故障。

## 方法
不重训。使用与正式实验**完全相同的评估入口**（ETrain.Eval.LLaVA.CoIN.model_vqa、
temperature=0、4-chunk 并行、官方 substring 评分）对只读 checkpoint 额外评估，
输出独立目录 /root/data/coin/results/diagnostics/ratio_0.1/（未触碰正式 10 个 eval 单元）。

| 诊断 eval | checkpoint | 路径 |
|---|---|---|
| ImageNet @ round3 task ckpt | round3_task_llava_lora | checkpoints/CoIN_Replay/ratio_0.1/round3_task_llava_lora |
| GQA @ round4 task ckpt（同类对照） | round4_task_llava_lora | checkpoints/CoIN_Replay/ratio_0.1/round4_task_llava_lora |

- checkpoint 加载：与正式 previous-task 加载同路径（adapter_model.bin +
  non_lora_trainables.bin，keys missing=0/unexpected=0 语义由训练链保证）
- prediction 校验：ImageNet 5050/5050（行数=唯一 ID=问题集）；GQA 12578/12578 ✓
- evaluator/数据/temperature 与正式一致；无旧预测混入（全新输出目录）

## 结果

| checkpoint | 任务 | acc | 对照（正式） |
|---|---|---|---|
| round3 task（刚学完 ImageNet） | ImageNet | **96.93%** | — |
| round3 replay（replay 仅含 SQA+TVQA 0.1） | ImageNet | 4.02（正式 A[3][3]） | 96.93 → 4.02 |
| round4 task（刚学完 GQA） | GQA | **56.70%** | — |
| round4 replay（replay 含 SQA+TVQA+INet 0.1） | GQA | 37.90（正式 A[4][4]） | 56.70 → 37.90 |

## 判定：A —— 方法行为，非评估工程故障
- Round 3 task checkpoint 本身 ImageNet = 96.93%（训练正常、评估链路正常、无标签/路径问题）
- round3 replay 数据只含前序任务（ScienceQA 1273 + TextVQA 3460 条，0.1 prefix 子集），
  不含 ImageNet → replay 训练把刚学的 ImageNet 适配器权重覆盖性干扰，导致自评骤降至 4.02
- round4 replay 重新包含 ImageNet（0.1 × 129,833 ≈ 12,983 条）→ ImageNet 恢复至 55.19
  （55.19 < 96.93：0.1 比例 1 epoch 回放只能部分恢复，且叠加 GQA 任务训练后的干扰）
- GQA 佐证同一模式：round4 task 后 56.70 → 不含 GQA 的 replay 后 37.90
  —— 新任务学完即被「不含自己的 replay」干扰，是 TRACE 式 replay 顺序学习的内在行为

## 对指标解读的影响（供组会/评审）
- A[3][3]=4.02 与 A[4][4]=37.90 反映的是「replay 对新学任务的即时干扰」，不是任务本身
  学不会（初学 96.93% / 56.70%）
- BWT 计算含此类低初值点（BWT = Σ(A[T,i] − A[i,i])/T；A[i,i] 取 replay 后值）
- 若研究问题关注「纯净初学精度」，需另行定义 A[i,i] 口径（task ckpt 直接评估）——
  本实验按既有协议（每轮最终 ckpt = replay ckpt）评估，口径一致、可复现

## 产物
- diagnostics 目录：/root/data/coin/results/diagnostics/ratio_0.1/
  （r3_task_imagenet/、r4_task_gqa/：merge.jsonl + Result.text + 日志）
- checkpoint tensor hash：正式验收已覆盖（8 ckpt 可加载、finite、task/replay 448/448 diff）
