# CoIN+Replay 正式 sweep ratio=0.1 — 运行状态

- 状态：STARTING
- ratio：0.1
- commit：17cfa66f009bd1fd1f5d360307f97d4249bf2c5c（本地=GitHub=云端）
- 命令：bash scripts/CoIN_Replay/run_sweep.sh 0.1（唯一）
- tmux 会话：coin_sweep_r010_20260904（启动前检查不存在）
- 主日志：/root/data/coin/logs/formal/ratio_0.1_20260904_032926/sweep.log
- 输出目录：/root/data/coin/project/results/CoIN_Replay/ratio_0.1
- checkpoint：/root/data/coin/project/checkpoints/CoIN_Replay/ratio_0.1
- replay 数据：/root/data/coin/project/playground/Replay/ratio_0.1
- manifest：/root/data/coin/project/results/CoIN_Replay/ratio_0.1/run_manifest.json
- 开始时间：2026-09-04 03:29 CST（EST）
- 预计完成：开始后 ~8-8.5h（task 4.4h + replay 1.0h + eval 2.6h 实测口径）
- ratio=0.01：未启动（严禁）
- task accum=16（eff 896）/ replay accum=1（eff 56）

更新记录：
- STARTING：2026-09-04 03:30 启动目录创建，待 tmux 启动
