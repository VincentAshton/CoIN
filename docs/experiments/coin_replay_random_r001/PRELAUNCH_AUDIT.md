# PRELAUNCH AUDIT —— codex/coin-replay-random-r001（正式 run_0001 启动前审计）

目标：把系统审计到「资源上线后可以先做短 canary」的状态。**本文件不含任何正式 run /
seed / 结果**（index.json runs=[] 必须保持）。

## 阶段 A（零 GPU 静态审计与加固）—— 2026-09-08

> **历史文档注记（2026-09-11，ratio 通用化）**：本文记录的是 r001 专用门（`random_r001_gate`）
> 在 commit 16f27d4 时期的审计快照，其 PASS 证据与日志措辞（`FAIL(random-r001 gate)`）保持原样不改写。
> 自分支 `codex/coin-replay-random` 起，该门已泛化为 **`random_replay_gate`**：ratio 属于
> {0.01, 0.10}（数值归一化判定）、ratio tag 由 ratio 派生（r001/r010）并与三个目录精确匹配，
> 报错前缀统一为 `FAIL(random-replay gate)`；上述 A2–A5 各项检查内容不变，仅解除 0.01 写死。
> 新实现的等价性与回归测试见 `tests/test_random_ratio_generalization.py` 与
> `tests/test_orchestrator_dryrun.py`（r001 与 r010 双路径）。

- 审计前 commit：22c6ad9f61386a49a65623a77484ad0711ce5ed4（工作树 clean；
  远端 codex 分支无后续提交）
- 审计 commit：见本目录 git 记录（AUDITED_COMMIT 在阶段 A 完成后回填）
- legacy 对照固定父提交：850db438d68ff59536a053076a22ee4ca727059a（防自比较）

### 加固清单（全部落实并测试）

| # | 项目 | 实现位置 | 状态 |
|---|------|---------|------|
| A1 | legacy 测试固定父提交 + 防自比较断言（源 SHA ≠ 当前实现） | tests/test_random_replay_selection.py | PASS |
| A2 | 启动 fail-fast 门（mode/ratio/seed 31-bit/REPLAY_ACCUM/run ID/三目录/port/GPU） | run_replay_exp.sh `random_r001_gate` | PASS（该实现名为历史名；2026-09-11 已泛化为 `random_replay_gate`，见下注） |
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

## 阶段 B（云端 canary 预启动）—— 2026-09-09 完成（实例 30267）

- 云端 checkout 精确 commit：**16f27d4469b07225bdcf820e43bac915809e4bbd**
  （阶段 A=605a39f；阶段 B 审计中发现 finalize 报告缺扫描门行 → 修复 16f27d4，两版全测）
- 环境：4×A100-80GB 空闲；torch 2.0.1+cu118 / transformers 4.32.0 / peft 0.4.0 /
  deepspeed 0.14.0 / flash-attn 2.5.6 / accelerate 0.21.0 / python 3.10.21 / pillow 10.3.0
  / NCCL True；/root/data 503G avail、inode 富余、RAM 1007G、shm 200G
- 全套测试（torch 实际执行）：**Ran 120, OK, skip=0**（@605a39f 与 @16f27d4 各一次）
- prefix manifest 对照：超参与任务序/ratio 全等；data_revision=bf6bd4ee…、
  model_config_hash=5fe5a4b3…、ds_config_hash=4e0c4eb1… **相等**；仅差允许项
  （sample_mode/replay_sample_seed/random_replay_run_id）与工作目录绝对路径前缀
- 全新 PREFLIGHT_REPORT + 全量图片检查（--layout-map、无 --skip-pil）：PASS（约 13min）
- 真实数据 replay canary（seed=1234，r2/3/4）：双重建 json 逐字节一致；独立重建
  task_seed/排列/k/idx/ids 全等；全部选中图 PIL 可读；SQA idx_sha 跨 round 相同
  （task seed 不含 round 实证）
- DRY_RUN 编排 canary：gate PASS → 4 rounds → .complete，rc=0（隔离目录 run_0000_seed_1234）
- GPU/DS：smoke_gpu rc=0（4 卡 NCCL + flash fwd/bwd + bf16）；smoke_ds rc=0（4 rank
  zero3_offload 真 step + 参数变化断言）；probe_logits 真实 task/replay ckpt 重载一致 +
  logits 不同；真实 tensor-diff r4 **448/448 changed** exit 0；8 题真实最小评估 +
  verify-predictions/artifact-check rc=0
- finalize canary：assemble rc=0（staging 原子换入 6 文件，报告含扫描 PASS 行）；
  registry 生命周期由云端全套测试覆盖（临时副本，真实 index 未动）
- 正式目录零污染：project & project_audit 下 CoIN_Replay_random/Replay_random 均不存在；
  prefix checkpoints/results 完好；**index.json runs=[]**
- **判定：GO**（正式启动 env 预览见 PRELAUNCH_AUDIT.json；未执行）

## 阶段 B 检查清单明细（PASS/FAIL/命令/证据）

见 **PRELAUNCH_AUDIT.json**（结构化，同目录）。
