# PRESWEEP GO/NO-GO 判定（2026-09-04，方案 D 全链路审计 + 实机验证后）

判定：**GO（有条件）——禁止自动启动正式 sweep，停等用户明确批准**

## 本次判定依据（全部实机证据）
1. 跨层语义审计（权威计数）：HF global_step/LR 不可信，DS engine.global_steps/_step_applied
   权威、tensor diff 铁证——审计报告 audit/audit_cross_layer.md（SHA256 见 env 清单）
2. 全矩阵静态审计（模拟器修正记录在内）：真步 = per-rank micro // gas；0.01 gas1 = 3/9/32、
   0.1 gas1 = 23/85/317——audit/audit_matrix_static.md
3. 代码锁定：commit 17cfa66（experiment/coin-replay-presweep-20260903，本地=GitHub=云端三端
   hash 一致、工作区干净）；run_tests Ran 89 OK（云端 coin env，73+16 无删改）
4. GPU 实机（4×A100）：replay gas1 N=127→3 真步、N=473→9、N=1272→23（gs=M 精确命中 +
   tensor 448/448 changed + hash differs + finite；无 NaN/OOM/NCCL）——
   logs/presweep/gas1_gate_20260904/single_step_summary.json（GATE_PASS）
5. Canary A–E 全 PASS（编排代码变更后完整重跑）——logs/canary_20260904/canary.log
6. 四任务 preflight RC=0（ImageNet_withlabel layout-map 正确）——
   logs/canary_20260904/preflight_4task.log；旧报告备份 logs/audit_freeze_20260904/old_preflight/
7. presweep dry-run 0.1 + 0.01 RC=0 + .complete（隔离目录，REPLAY_ACCUM=1 正式配置）
8. cross-manifest diff pass：0.1 vs 0.01 仅 ratio 差异；replay_accum=1 双端一致、
   effective_batch 896 / replay_effective_batch 56、allow_single_step_replay=0、
   round2 per_rank_micro [23×4] 与静态模拟一致
9. 性能/费用实测口径：0.1 ≈ 8-8.5h、0.01 ≈ 7.2-7.5h、两 ratio ≈ 15.5-16h ≈ ¥435-560

## 观察项（不阻塞 0.1；正式 0.01 前复查）
- 0.01 r4 (N=1771) gas16 真步模拟 = 2 vs 旧报告 1——0.01 正式跑时以实测 gs/plan 为准
- gas1 每样本成本 ~2×（CPUAdam offload）——replay 段仅占 0.1 时长 ~12%，影响可控

## 批准后执行命令（严禁带 0.01；0.1 完成验收停）
```
export REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1 GPUS=0,1,2,3
export PREFLIGHT_ARGS='--layout-map {"ImageNet":"ImageNet_withlabel"}'
bash scripts/CoIN_Replay/run_sweep.sh 0.1
```
约束：tmux 唯一会话 + pipefail；禁止 ALLOW_SINGLE_STEP_REPLAY=1；REPLAY_ACCUM 禁止按 ratio 变化。

## 历史
2026-09-03 PRESWEEP_GO_NOGO_20260903.md（sha256 404ab6aa…）→ 阶段 II No-Go → 方案 D
审计（本文件，2026-09-04）。旧报告/日志均保留（logs/ 时间戳目录，未覆盖）。
