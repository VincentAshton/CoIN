# CoIN+Replay 全链路审计 · 阶段二：跨层训练语义审计（谁是权威计数）

日期 2026-09-04 · 审计对象为云端 coin 环境锁定版本（requirements_coin.txt 实证）：
torch 2.0.1 / transformers 4.32.0 / deepspeed 0.14.0 / accelerate 0.21.0 / peft 0.4.0
源码审阅自 PyPI 同版本 sdist（pip 安装物同源）；云端实装 SHA256 补录待实例开机。

## 调用链（实际执行路径）
run_replay_exp.sh → deepspeed --include localhost:0,1,2,3 ... train_mem.py
→ train_mem.py（flash-attn monkey patch）→ train.py train()
→ LLaVATrainer(transformers.Trainer 薄子类) → trainer.train()（transformers 4.32 标准 loop）
→ accelerator.accumulate → training_step → accelerator.backward(loss)
→ [DS] DeepSpeedEngineWrapper.backward = engine.backward + engine.step()   ← 每 micro 一次

## 逐层证据（行号均指实际安装版源码）

### 1. shell 层（run_replay_exp.sh @ d7f9d1c）
- train_one L179-181: train-plan 计算用 --accum $ACCUM（16）
- train_one L214: --gradient_accumulation_steps "$ACCUM"（task 与 replay 共用同一函数同一值）
- run_round L316 task 段 / L333 replay 段均调 train_one，无区分 → 结构上无法 per-段覆盖（方案 D 改动点）
- zero3_offload.json L33: "gradient_accumulation_steps": "auto" → 由 HF args 决定 gas

### 2. CoIN 层（ETrain/Train/LLaVA/）
- train.py L115: training_args._frozen = False（FrozenInstanceError 规避，评审 C-4）
- train.py L206-211: trainer.train() → save_state() → save_trained_model()
  → train() 返回后无条件保存 adapter（config.json + adapter_model.bin + non_lora_trainables.bin）
  —— 0 真步也会「成功保存」，保存动作本身不是训练证据
- llava_trainer.py L377-478 create_optimizer: 3 参数组 decay(2e-4)/no_decay(2e-4)/mm_projector(2e-5)；
  DS zero3 下由 deepspeed.initialize 接管（zero_force_ds_cpu_optimizer → DeepSpeedCPUAdam）
- llava_trainer.py L496-521 _save_checkpoint/_save: tune_mm_mlp_adapter=False → super()（epoch 保存 checkpoint-<gs>）
- llava_trainer.py L362-376 _get_train_sampler: group_by_modality_length=True →
  LengthGroupedSampler(batch=14, world_size=4×16=64)（全局非分布式 sampler，长度=N，无 pad）
- LrLogCallback（L349-358）: 读 logs["learning_rate"] —— 该值来自 Trainer scheduler（phantom 可步进）

### 3. transformers 4.32.0（trainer.py）
- L841: get_train_dataloader → self.accelerator.prepare(DataLoader(sampler=LengthGroupedSampler))
  → 分布式分片实际由 accelerate BatchSamplerShard 完成（非 torch DistributedSampler！）
- L1580-1581: num_update_steps_per_epoch = max(len_dataloader // gas, 1)  ← ceil/兜底语义，高估
- L1851-1866: is_last_step_and_steps_less_than_grad_acc =
  (steps_in_epoch <= gas and step+1 == steps_in_epoch) → 强制 sync + 进 step 区块（对 DS 无效）
- L1908-1917: self.optimizer.step()（DS 下 = no-op wrapper）；optimizer_was_run = not
  accelerator.optimizer_step_was_skipped（DS 无 overflow 时恒 True → phantom）
- L1919-1925: optimizer_was_run → self.lr_scheduler.step() → self.state.global_step += 1
  —— HF global_step 递增不要求 DS 真实 apply

### 4. accelerate 0.21.0（utils/deepspeed.py + data_loader.py + accelerator.py）
- utils/deepspeed.py L165-179: DeepSpeedEngineWrapper.backward = engine.backward(loss) **+ engine.step()**
  （注释明示：DS 的 step 内含 accumulation 检查，非边界 no-op）
- utils/deepspeed.py L198-206: DeepSpeedOptimizerWrapper.step() = pass（真实 step 全在 engine）；
  step_was_skipped 仅查 overflow
- accelerator.py L2913 optimizer_step_was_skipped → wrapper.step_was_skipped → DS 无 overflow 恒 False
- data_loader.py L67-217 BatchSamplerShard（split_batches=False + even_batches=True）:
  - 主循环按 idx%world==process_index 持有 batch，组末（idx%world==world-1 且满长）yield
  - 尾部补齐：partial batch 用 initial_data 回绕补满，直到 yield 数为 world 整数倍
  - → 每 rank yield ≈ ceil(ceil(N/14)/4)，但**尾部可能 rank 间 ±1 不均**（N=1272: 22/23 分析），
    不均时训练同步行为（hang vs 静默少步）为静态不确定项，须 GPU 实测
- accelerator.py L891-921 accumulate(model)：sync 时 nullcontext，否则 no_sync——不影响 DS 计数

### 5. deepspeed 0.14.0（runtime/engine.py）
- L2016-2030: is_gradient_accumulation_boundary = (micro_steps+1) % gradient_accumulation_steps() == 0
- L2132-2173: engine.step() 开头 _step_applied=False（L2150）；仅 boundary 时 gas_boundary_ctr+=1 并
  _take_model_step；非 boundary 除计时外什么都不做（梯度保留，不 flush、不 zero_grad）
- L2066-2130: _take_model_step = optimizer.step() → _step_applied = not overflow（L2110）
  → lr_scheduler.step（engine 无 scheduler 时不动）→ global_steps += 1（L2129）
- L2235: micro_steps += 1（engine.step() 末尾，每 micro 一次）
- L1933-2014 backward: zero3 路径 self.optimizer.backward(loss)（L1976，zero stage3 内部管理梯度 partition）；
  gas>1 时 loss 按 gas 缩放（L1951-1952）
- zero/stage3.py: 梯度 partition 累积入 grad_buffer；boundary 时（is_gradient_accumulation_boundary L1454）
  触发完整 reduce+update 语义

## 权威计数结论

| 计数器 | 语义 | 可信度 | 证据 |
|---|---|---|---|
| HF planned steps（train-plan optimizer_steps） | max(ceil(micro/gas),1) | 高估（短段恒≥1） | tf L1580-81 |
| HF state.global_step | 每次进 step 区块 +1（含 phantom 尾部） | **不可信** | tf L1925 |
| callback/日志 LR | Trainer scheduler 随 phantom gs 步进 | **不可信** | tf L1922 |
| DS engine.global_steps | 每次真实 _take_model_step +1 | **权威**（无 overflow 时） | ds L2129 |
| DS engine._step_applied | 最近一次 engine.step 是否真 apply | 权威（布尔） | ds L2110/2150 |
| optimizer 内部 state['step'] | DS optimizer 自身更新计数 | 次级佐证 | — |
| tensor diff（changed/hash/L2） | 参数真实变化 | **最终铁证** | coin_lib ckpt-tensor-diff |

- 真步公式（DS 语义）：per-rank micro 数 // gas；per-rank micro = BatchSamplerShard yield 数
  （≈ ceil(ceil(N/batch)/world)，尾部 even 回绕补齐可能 ±1 → rank 间可能不均）
- 等价安全下界：floor(N / (batch×world×gas))（N=1771/gas16 时低估 1：M//gas=2 vs floor=1，
  见矩阵模拟）→ 计划字段须同时报告两种口径并标注实测确认
- 尾部未提交 micro 的梯度：保留在 grad buffer，训练结束即丢弃（无 flush 路径）→
  epoch 尾部样本「静默不贡献」且不报错

## 对阶段 II No-Go 的完整解释（0.01 r2 N=127, gas16, world4）
127 条 → LengthGroupedSampler 全局序 → BatchSampler(14) → 10 batch → BatchSamplerShard even 补齐
→ 每 rank 3 micro → DS boundary 需 (micro+1)%16==0：micro 1/2/3 全不中 → 0 真步；
HF 第 3 micro 命中 is_last_step 分支 → gs=1、scheduler 步进（phantom）→ 保存 adapter == task 段
（逐字节同，sha256 实证）✓ 与报告一致

## 待 GPU 实测确认项（阶段七）
1. N=1272 等非整除场景 per-rank micro 分布（22/23?）与是否 hang
2. gas=1 时 partial/回绕 batch 的 boundary 语义（accum1 下每 micro 即 boundary，应无丢弃）
3. 各场景 DS engine.global_steps 与 _step_applied 实测值 vs 计划表
4. optimizer state['step'] 与 tensor diff 一致性
