# CoIN+Replay 全链路审计 · 阶段八：隐藏故障审计（本地静态部分）

2026-09-04 · 逐项 PASS/FIXED/BLOCKED/NOT TESTED + 证据。BLOCKED 项 = 需云端实机确认（阶段七）。

## 训练/数据链
1. seed/data_seed —— PASS。run_replay_exp.sh L225 `--seed 1234 --data_seed 1234`；
   build_replay_data --seed 1234；LengthGroupedSampler 无显式 generator → 用 torch 全局 RNG
   （transformers set_seed 后确定）；worker 用 seed_worker（4.32 trainer L839）。DS 多 rank seed
   一致性依赖 transformers set_seed 每进程调用（4.32 标准行为）。实机复跑确定性比对为 NOT TESTED（eval 随机字段排除法已有先例）。
2. optimizer 参数组 / mm_projector_lr —— PASS。llava_trainer.py L377-478：decay(2e-4)/no_decay(2e-4)/
   mm_projector(2e-5) 三组；完整性断言（mm_lr 设置必有 mm 参数、总参非空）。DS 接管为 CPUAdam（zero_force_ds_cpu_optimizer）。
3. scheduler 真实类型/总 steps —— PASS。transformers 4.32 create_scheduler + get_scheduler(cosine,
   num_training_steps=max_steps)——总步数 HF 口径可能 ≠ DS 真步（阶段二结论：LR 轨迹随 HF gs 步进）。
4. warmup 对短 replay —— PASS（分析）。gas1 replay r2 T=3/r3 T=9 warm=int(0.03T)=0 → 首步 LR=2e-4 峰值；
   0.1 r4 gas1 T=317 warm=9。无 LR=0 或 NaN 风险；cosine 总步 ≥ 真步时 LR 曲线尾端截断（不越界）。
5. bf16/tf32 —— PASS。run_replay_exp.sh L209 `--bf16 True --tf32 True`；zero3_offload.json bf16
   （enabled auto）；tf32 由 torch 后端 flag（TrainingArguments.tf32 → torch.backends.cuda.matmul.allow_tf32）。
6. zero3 offload —— PASS。zero3_offload.json：stage3 + offload_optimizer(device cpu, pin_memory? 待云端
   复核) + offload_param cpu；gradient_accumulation_steps "auto"（跟随 HF args——方案 D gas1 语义已审）。
7. LoRA/non-LoRA 保存加载 —— PASS。save_trained_model（lora_adapter_model.bin + non_lora_trainables.bin +
   config）；load_model_from_previous_task（llava_trainer.py L261+）：non_lora strict=False + adapter
   set_peft_model_state_dict + key 归一化（base_model. 11 字符 + model. 6 字符剥离）+ missing/unexpected
   加载后断言（工单 4 加固）。
8. checkpoint tensor hash —— PASS。coin_lib ckpt-validate param_hash（bf16 字节 hash 修复 2026-09-02）；
   ckpt-tensor-diff changed/L2/max_abs_diff/规范化 hash；exit 0/1/2 语义 + 回归测试。
9. task→replay→下一轮链 —— PASS。run_round：roundj replay ckpt 为下一轮 prev（L309）；round1 无 replay
   用 task ckpt（L311）；缺 prev die（L312）；round≥2 replay 段用 task ckpt 起步（L333 prev=task_ckpt）。
10. 分离输出目录 —— PASS。CKPT_ROOT/RES_ROOT/REPLAY_DATA_DIR ratio 派生 + env 可覆盖（隔离 dry-run 法，
    测试 TestShellGuards）。task/replay ckpt 目录独立（roundj_task vs roundj_replay）。
11. resume/config hash —— PASS。run_manifest.json config_hash（CONFIG_FIELDS 快照，现含 replay_accum/
    replay_effective_batch/allow_single_step_replay）；manifest-resume-check 拒绝不一致；round manifest 补
    replay_plan（2026-09-04）。
12. .complete/round marker —— PASS。main：.complete 短路前先 resume 校验（L373-376）；缺 manifest 的
    .complete die；validate-round（ckpt_validate + round manifest + replay data）失败 → 删 marker 重跑。
13. eval chunk/PID —— PASS。eval 脚本按 chunk 分片（GPULIST[$IDX] 单卡串行/并行）+ CUDA_VISIBLE_DEVICES；
    随机字段（answer_id）排除法已有（4241 行双跑仅 answer_id 不同）。
14. stale result —— PASS。eval_one 临时目录 + verify-predictions + artifact-check 通过后原子 mv 换入
    （.stale_ 前缀旧目录删除）；失败非零退出不换入。
15. prediction count/ID/顺序 —— PASS。coin_lib verify-predictions：总数/唯一/集合/顺序四查。
16. ScienceQA text-only/image —— PASS。probe_logits 纯文本题支持（修复 2026-09-02，技能 references/
    probe-fix.md）；SQA test.json 全量。实机 eval 复核在阶段七。
17. TextVQA/GQA metadata —— PASS。preflight_data 校验 aux（pid_splits.json）缺则 FAIL（test_missing_aux）。
18. ImageNet layout —— PASS。cl_dataset/ImageNet_withlabel/{train/<synset>/,val/} + PREFLIGHT_ARGS
    layout-map（2026-09-03 实证，GQA 无条目默认正确）。
19. aggregate MAA/BWT —— PASS。aggregate_coin.py：A 矩阵（roundj 各任务 acc）→ MAA=(1/T)Σ_j(1/j)Σ_{i≤j}A_{j,i}
    + BWT=(1/T)Σ_i(A_{T,i}−A_{i,i})。round 缺失 → read_accuracy 抛错防静默（待复核代码细节——BLOCKED 云端
    对拍 old sweep 无，公式与 CoIN 论文一致）。

## 运行/运维
20. 磁盘/inode//dev/shm —— BLOCKED。云端 df/inode 实测（freeze 补录清单 E 项）。已知：/root/data 1T
    Lustre、/dev/shm outputs 纪律（TRACE 教训同源）。
21. CUDA_VISIBLE_DEVICES（train/eval）—— PASS。train：deepspeed --include localhost:$GPUS（L237）；
    eval：RESULT_DIR 临时 + CUDA_VISIBLE_DEVICES=$GPUS（L263）与 chunk 内 GPULIST 单卡。eval 与 train 同
    卡池分时（编排顺序执行，无重叠）——实机观察显存峰值（阶段七记录）。
22. 失败后错误继续下一 ratio —— PASS（FIXED 语义已具备）。run_sweep.sh set -euo pipefail fail-fast；
    run_replay_exp.sh set -euo pipefail；die 显式退出。0.1 失败 → sweep 退出不启动 0.01。
23. run_sweep 重复启动/防重入 —— NOT TESTED（无 lock 文件）。现有防护=技能纪律（tmux 唯一会话 +
    manifest resume 校验天然防双开同目录：第二个进程会看到 run_manifest 恢复校验，若同配置会并行跑
    同目录 → 竞态未防）。建议（不擅自实现）：正式启动用 tmux 唯一会话名 + 启动前 ps 检查同 RES_ROOT。
24. 自动关机写 checkpoint 时机 —— NOT TESTED。云端无自动关机策略（ebcloud 手动关）；技能记录关机前
    确认无训练进程。BLOCKED 实机确认无残留进程。

## 审计发现（新增/需报告）
- 旧 floor(N/896) 真步公式低估边界（N=895 实际 1 真步非 0；N=1771 gas16 模拟 2 非 1）——以
  per-rank micro // gas 为准，train_plan 已修复多口径报告（阶段五）
- gas1 下每 micro 即 boundary → optimizer step 频率 ×16、CPU offload 吞吐代价 → 阶段七实测
- run_replay_exp.sh DRY_RUN 分支无 train-one plan 文件（plan.json 由真实 train_one 写；DRY_RUN 也写
  ——train_one 内 plan 计算在 DRY_RUN 分支前 → plan.json 仍写 ✓ 无缺口）
- round≥2 eval 用 replay ckpt（eval_ckpt=replay_ckpt L344 起）→ round1 无 replay 用 task ckpt ✓ 语义正确
