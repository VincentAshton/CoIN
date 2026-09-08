# PRELAUNCH AUDIT —— codex/coin-replay-random-r001（正式 run_0001 启动前审计）

目标：把系统审计到「资源上线后可以先做短 canary」的状态。**本文件不含任何正式 run /
seed / 结果**（index.json runs=[] 必须保持）。

## 阶段 A（零 GPU 静态审计与加固）—— 2026-09-08

- 审计前 commit：22c6ad9f61386a49a65623a77484ad0711ce5ed4（工作树 clean；
  远端 codex 分支无后续提交）
- 审计 commit：见本目录 git 记录（AUDITED_COMMIT 在阶段 A 完成后回填）
- legacy 对照固定父提交：850db438d68ff59536a053076a22ee4ca727059a（防自比较）

### 加固清单（全部落实并测试）

| # | 项目 | 实现位置 | 状态 |
|---|------|---------|------|
| A1 | legacy 测试固定父提交 + 防自比较断言（源 SHA ≠ 当前实现） | tests/test_random_replay_selection.py | PASS |
| A2 | 启动 fail-fast 门（mode/ratio/seed 31-bit/REPLAY_ACCUM/run ID/三目录/port/GPU） | run_replay_exp.sh random_r001_gate | PASS |
| A3 | run ID 进 manifest/config hash/resume 校验 | coin_lib.compute_config + CONFIG_FIELDS | PASS |
| A4 | registry complete 禁 RUNNING；csv 原子写；A/C/D 协议强制 | tools/random_replay_registry.py | PASS |
| A5 | finalize staging + 原子换入；sampling_algorithm 常量门；源数据独立重建门；敏感扫描；--test-mode | tools/random_replay_finalize.py | PASS |
| A6 | manifest/env 记录 Python 完整版本 | coin_lib.env_versions | PASS |
| A7 | sidecar manifest 原子写；assert → 显式错误（-O 安全） | build_replay_data.py | PASS |
| A8 | 抽样语义未变（sha256("<seed>:<task>") + Random shuffle，算法名不变） | — | 保持 |

### 零 GPU 验证（2026-09-08，本地 WSL，python 3.10.12 + .venv pillow）

- run_tests.sh（bash -n 全部 .sh + py_compile + unittest + zero3/protobuf 检查）：**rc=0**
- unittest 全套：**run=120，PASS=111，FAIL=0，ERROR=0，SKIP=9**
  - 9 个 SKIP 全部为 test_ckpt_tensor_diff（torch 依赖）：
    test_bf16_hash_stable / test_cli_different_tensors_exit_0 /
    test_cli_identical_tensors_exit_1 / test_different_tensors_passes /
    test_identical_tensors_different_metadata_fails / test_missing_key_fails /
    test_nan_fails / test_shape_mismatch_fails / test_structural_missing_dir_exit_2
    —— 阶段 B 云端全量执行，任何相关 SKIP 判 NO-GO
- prefix 真 legacy（父提交 850db43 blob）逐字节对照：PASS（含防自比较断言）
- random 同 seed 重建/不同 seed/floor k/nested（seed·mode·SHA 失败路径）/resume mismatch：PASS
- registry 临时副本完整生命周期：register/complete/二次 complete 拒绝/seed 复用拒绝/
  complete RUNNING 拒绝且三文件逐字节不变：PASS（4 tests）
- finalize 成功（staging 原子换入）+ 失败路径（错误 seed/ckpt 路径/源 SHA/算法/已有 out
  无 force/敏感泄漏阻断）：PASS（9 tests，--test-mode 文件级降级显式标注）
- -O（python optimize）下 build 显式错误仍生效：PASS
- 敏感信息/大文件扫描：见提交前扫描记录（无命中）

### 未验证路径（阶段 B 必做）

- torch/GPU：上述 9 个 tensor 测试 + finalize 默认非 test-mode 的 torch 门
  （checkpoint 参数级 finite/hash、真实 tensor-diff）
- 真实数据 replay 构建（canary seed=1234，round2/3/4 真实数据 + 图片可读）
- 4×A100 训练/评估任何真实执行
- prefix 历史 manifest 对照（需要云端数据文件）

## 阶段 B（云端 canary 预启动）—— 待用户通知「ebcloud 30267 已启动」

> 以下由阶段 B 执行后回填（PASS/FAIL/命令/返回码/版本/脱敏 hash），GO/NO-GO 判定亦在此。

- [ ] 云端 clean worktree checkout 精确 AUDITED_COMMIT
- [ ] 4×A100 可见空闲；CUDA/NCCL/torch/DS/transformers/peft/flash-attn 版本；磁盘/inode/RAM/shm/MASTER_PORT
- [ ] 全套测试（torch 依赖 SKIP 必须实际执行；相关 SKIP = NO-GO）
- [ ] prefix 历史 manifest 对照（只允许预期字段差）
- [ ] 全新 PREFLIGHT_REPORT + 全量图片检查（--skip-pil 禁用）
- [ ] canary seed=1234 真实数据 replay 构建（round2/3/4，双重建一致 + 图片可读）
- [ ] 隔离 canary 目录 DRY_RUN 编排（禁正式目录）
- [ ] 分钟级 GPU/DS 验证（4 卡通信/加载/单步/ckpt 存取/tensor-diff/最小 eval）
- [ ] finalize + registry 临时副本生命周期（不污染真实 index）
- [ ] 输出 PRELAUNCH_AUDIT.json（逐项证据）
