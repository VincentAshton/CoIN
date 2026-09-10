# 冻结现场 · 云端补录清单（实例开机后执行，30267）

数据持久于 /root/data（1T Lustre）；本清单为审计完整性所需补录项。

## A. commit / git（在 /root/data/coin/project）
- [ ] git rev-parse HEAD（期望 d7f9d1c27c5ed9eb42c1a3d2eabebdc48aeb16c8）
- [ ] git status --short（期望空）
- [ ] git log --oneline -5

## B. 源文件 SHA256（conda env /root/data/coin/conda_envs/coin）
- [ ] transformers 4.32.0 源文件 sha256（trainer.py / trainer_pt_utils.py / deepspeed.py）
- [ ] deepspeed 0.14.0 源文件 sha256（runtime/engine.py / runtime/zero/stage3.py）
- [ ] accelerate 0.21.0 源文件 sha256（utils/deepspeed.py / data_loader.py / accelerator.py）
- [ ] torch 2.0.1 版本核对（DistributedSampler 语义基线）
- [ ] python3.10 版本
- [ ] pip freeze 或 requirements 全量（conda env list / pip list > 快照）

## C. 日志/报告 SHA256 + 目录清单（logs/presweep/...）
- [ ] logs/presweep/single_step_replay/ 全部文件清单 + sha256
      （含 SINGLE_STEP_GATE_REPORT.md、gate_master.log、train_*.log）
- [ ] logs/presweep/PRESWEEP_GO_NOGO_20260903.md sha256
- [ ] logs/archive_final_20260902/ 清单（sha256_manifest_final.txt 复核仍一致）
- [ ] run_tests 日志（Ran 73 tests 证据）、canary v5 日志、四任务 preflight report

## D. 冻结：不覆盖任何旧报告/checkpoint/manifest（只读操作）

## E. 环境事实快照
- [ ] /root/data 磁盘余量、inode；/dev/shm 大小
- [ ] nvidia-smi（确认 4×A100 在）
- [ ] tmux/进程现场（无残留训练进程）
