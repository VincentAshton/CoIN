# CoIN+Replay 正式 sweep ratio=0.01 — 运行状态

- 状态：COMPLETE
- ratio：0.01
- commit：运行代码 = 17cfa66f009bd1fd1f5d360307f97d4249bf2c5c；云端 detached HEAD = 069b608（逐字节一致，git diff 0 行）
- 命令：bash scripts/CoIN_Replay/run_sweep.sh 0.01（唯一；0.1 未重跑）
- tmux 会话：coin_sweep_r001_20260905（启动前检查不存在）
- 主日志：/root/data/coin/logs/formal/ratio_0.01_20260905_035722/sweep.log
- 输出目录：/root/data/coin/project/results/CoIN_Replay/ratio_0.01
- checkpoint：/root/data/coin/project/checkpoints/CoIN_Replay/ratio_0.01
- replay 数据：/root/data/coin/project/playground/Replay/ratio_0.01
- manifest：/root/data/coin/project/results/CoIN_Replay/ratio_0.01/run_manifest.json
- task accum=16（eff 896）/ replay accum=1（eff 56）——方案 D
- 环境变量：REPLAY_ACCUM=1 ENFORCE_MIN_STEPS=1 GPUS=0,1,2,3（ALLOW_SINGLE_STEP_REPLAY 未设）

更新记录：
- STARTING：2026-09-05 03:57 启动目录创建，待 tmux 启动
- RUNNING：2026-09-05 04:33 CST 稳定性观察全通过（详情见当时记录）：preflight rc=0、
  manifest 正确、4 rank、round1 SQA task 14 steps（loss 0.92→0.32 finite、cosine LR 2e-4→0）、
  ckpt 落盘、GPU 17G/60-81%、系统盘不涨、无 OOM/NCCL/NaN、0.1 未被重跑
- STATUS: DONE (rc=0) 2026-09-05 11:03:31
- COMPLETE：2026-09-05 13:3x CST 全部验收通过：exit_code=0、.complete、4 round markers、
  7 ckpt 全部可加载+finite（r1-4 task + r2-4 replay，同 0.1 结构）、r2-4 tensor-diff
  448 changed（task≠replay 实证）、DS 真步 3/9/32 == manifest ds_expected（r2 N=127 / r3
  N=473 / r4 N=1771）、10 eval 单元齐全+merge 非空、prediction 数量/ID 10/10 PASS
  （4241/5000/5050/12578 与 0.1 逐单元行数一致）、coin_metrics 原子生成、独立重算
  CROSS-VALIDATION PASS（A/MAA/BWT 零差异）、结果 SHA256 清单 results_sha256.txt、
  无残留 GPU 进程。权威结果：MAA=60.4406、BWT=-13.6299、final avg=46.1925
  （0.1 对比：MAA=57.5057、BWT=+17.2306、final avg=55.7834）
