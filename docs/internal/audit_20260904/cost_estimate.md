# 方案 D · 性能与费用估算（实测锚点，2026-09-04）

## 实测锚点（4×A100 SXM4 80G，zero3+CPU offload，LoRA r192，grad_ckpt）
| 段 | accum | 每真步耗时 | 实测来源 |
|---|---|---|---|
| task 训练 | 16 | 47.8-55.9 s/step | gate task_sqa（14 步，callback 47.79/55.88s/it）|
| replay 训练 | 1 | 4.3-8.0 s/step | gate r2_010（23 步 1:38-3:04，含 CPUAdam offload）|
| 每段固定开销 | - | ~4-6 min（模型加载 + cpu_adam 编译 42s + LoRA 适配器）| gate 时间线 |
| 显存 | - | 15G/卡级（gate 全程无 OOM；80G 余量极大）| gate 日志 |
| 每样本成本对比 | 16 | ~0.055 s/样本 | 52s/896 |
| | 1 | ~0.11 s/样本 | 6s/56（CPUAdam 固定开销所致，~2×）|

## 方案 D（replay gas1）正式 sweep 时长估算
0.1 sweep（4 任务 ×1 epoch task + round2-4 replay）：
- task 段：SQA 14 + TVQA 38 + ImageNet 144 + GQA 80 = 276 真步 × 52s ≈ 4.0h + 4 段加载 ~25min ≈ **4.4h**
- replay 段 gas1：r2 23 + r3 85 + r4 317 = 425 真步 × 6s ≈ 43min + 3 段加载 ~15min ≈ **1.0h**
- eval：10 任务-eval（每轮 1..j）× ~15min（4 卡 chunk 并行）≈ 2.5-3h（含模型加载/前后处理）
- **0.1 合计 ≈ 8.0-8.5h**

0.01 sweep：task 段同 4.4h + replay（3+9+32=44 步 ×6s + 3 加载 ≈ 20min）+ eval 2.6h ≈ **7.2-7.5h**

两 ratio 合计 ≈ **15.5-16h**（4 卡并行墙钟）

## 费用
ebcloud A100 80G 参考 ¥7-9/卡时（旧 presweep 估 ¥1,140/36-40h → ¥7.1/卡时 4 卡）：
- 两 ratio：15.5h × 4 卡 × ¥7-9 ≈ **¥435-560**
- 单 0.1：8h × 4 × ¥7-9 ≈ ¥225-290
（比旧估算 ¥1,140 省 ~60%——旧值基于 170s/step 线性外推；实测 48-56s/step 后正式时长应以本表为准）

## CPU offload 观察
- gas1 每步 CPUAdam 更新：总 step 时间 4-8s 中 fwd/bwd（1 micro, 56 样本）<2s、CPU 优化器更新主导
- 与 gas16（每 16 micro 一次 CPU 更新）相比，CPU 更新频率 ×16 → replay 段每样本成本 ~2×；
  对全 sweep 影响有限（replay 段仅占 0.1 总时长的 ~12%）
- 无 NaN/OOM/NCCL 错误（gate 三场景 + canary 全段实证）

## 正式跑命令（等用户批准后）
export REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1 GPUS=0,1,2,3
export PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
bash scripts/CoIN_Replay/run_sweep.sh 0.1        # 先 0.1（验收后 0.01 另行批准）
