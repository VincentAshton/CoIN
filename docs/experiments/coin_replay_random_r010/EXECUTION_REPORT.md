# random-0.10 `run_0001_seed_358341059` —— 执行报告（预启动审计 + 运行 + 首次失败与恢复）

本文件是 r010 首个正式 run 的**可独立核验**执行记录（脱敏：不含 checkpoint/权重/完整 replay JSON/
完整预测/原始日志/缓存，也不含任何凭据）。机器可读版：[PRELAUNCH_AUDIT.json](PRELAUNCH_AUDIT.json)。

## 1. 运行身份

| 项 | 值 |
|---|---|
| run id | `run_0001_seed_358341059` |
| ratio / ratio tag | 0.10 / **r010**（tag 由 ratio 数值派生，非手填） |
| replay sample seed | 358341059（复用 r001 已公开 seed → 同 seed 嵌套配对；**非**按结果选 seed） |
| 任务序 | ScienceQA → TextVQA → ImageNet → GQA |
| 抽样算法 | `sha256_task_seed_python_shuffle_v1` |
| 固定项 | `SAMPLE_MODE=random`、`SEED=1234`、`DATA_SEED=1234`、`REPLAY_ACCUM=1`、task 段 accum=16 / replay 段 accum=1 |
| 运行代码提交 | `41abc5af222ba5eed7c6d25edab4a2bb2d110bcc`（云端 checkout 精确一致） |
| config hash | `a6f4ba7c63ebd57a0139fb28e8afbb83177c1476b83a170501e6d231dfa60b35`（恢复前后一致） |
| 模型 / 数据 | `model_config_hash=5fe5a4b3…`、`data_revision=bf6bd4ee…`（与 r001 各 run 完全一致） |
| ds config hash | `4e0c4eb1…` |
| 提交链 | A `6419a686…` → C `1cdb09ae…` → D `3f6e13e8…` |
| 结果 | **MAA 60.122 / CoIN BWT +28.7287 / final_avg 66.402**（`validation_report.md` 20 项全 PASS） |

时间线（2026-09-11 CST）：

| 时刻 | 事件 |
|---|---|
| 11:22:40 | registry 登记 RUNNING（提交 A `6419a68`），`started_at` 记录此时间 |
| 11:50:08 | 首次启动（tmux `coin_r010_0001`，`run_replay_exp.sh 0.10`），启动门 PASS |
| 11:50–11:57 | preflight |
| 11:57–12:13 | round1（ScienceQA）训练成功：14 step、`train_runtime=711s`、loss 0.919 |
| **12:13** | **round1 评估崩溃**（见 §3），round 无完成标记 |
| 12:13–15:12 | GPU 空转（约 3.2 h，未被自动发现） |
| 15:12:47 | **非破坏性重启**：同 seed / 同 run id / 同目录 / 同配置（`config_hash` 一致） |
| 15:12–15:26 | 重启 preflight（含数据 preflight 缓存校验） |
| 15:26:14 | round1 训练**重新开始**（见 §3.3） |
| 22:45:15 | `.complete` 写出（run 完成，`[aggregate] MAA=60.1220 BWT=28.7287`） |
| 22:49:58 | 结果 summary `completed_at`；fully 验收后发布六件套（提交 C `1cdb09a`） |
| 22:53:21 | registry COMPLETE（提交 D `3f6e13e`） |

## 2. 预启动审计（云端 canary）

结论：**全部 PASS**，明细与命令级证据见 [PRELAUNCH_AUDIT.json](PRELAUNCH_AUDIT.json)（含审计脚本
sha256 与日志位置）。要点：

- 测试：云端完整依赖 **146 tests / 0 fail / 0 error / 0 skip**（370.5 s，@ `41abc5a`）；本地零依赖
  146 tests / 0 fail / **18 skip**（9 torch + 9 PIL，显式门控，不计为通过）。
- 数据：全量 preflight（无 skip-PIL）PASS，`data_sha256=30a878a2…`，四任务 missing=0 / corrupt=0。
- 样本量口径：N 取**真实发布产物/全量 preflight** 值（ScienceQA 12726、TextVQA 34602、ImageNet 129833、
  GQA 72140）；任务书早期文本中的 12721/129835 为旧值，勿沿用（k 与 round 总量不受影响）。
- 嵌套：0.01 与 0.10 分别从真实源数据独立重建，**A/B 逐字节一致**、**0.01 是 0.10 的严格前缀**，
  k = 1272 / 3460 / 12983，round 总量 1272 / 4732 / 17715（0.01 为 127 / 473 / 1771）。
- 硬件/训练栈：4×A100 NCCL + flash-attn + bf16 smoke、DS ZeRO-3 真实更新 smoke、最小真实评估
  与产物校验、§六.7 预启动八项 —— 全部 PASS。

## 3. 首次失败：round1 评估崩溃（相对路径 / 软链缺口）

### 3.1 现象
首次启动的 round1 训练正常完成（11:57–12:13），随后评估 chunk 进程失败：

```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5'
```

编排器整组退出，无 `.complete`、无 round 完成标记。

### 3.2 根因
`scripts/LLaVA/Eval/*.sh` 与 `ETrain/Eval/LLaVA/CoIN/eval_gqa.py` 依赖 **cwd 相对路径**
（`./checkpoints/...`、`./cl_dataset`、`./playground/...`、`./cl_dataset/GQA`）解析到运行时软链；
而本次 run 使用新 worktree `project_generic`，缺少这些软链。preflight 此前只检查显式传入的
模型/数据路径，**不检查评估阶段的相对路径**，因此缺口在训练数小时后才暴露。

### 3.3 恢复方式，以及 round1 是「复用」还是「重训」

- 恢复动作：为 worktree 补三个运行时软链（`checkpoints`、`cl_dataset`、`playground/Instructions_Original`）
  并加入 `.git/info/exclude`；用一条 8 题真实 eval 验证（rc=0、8/8）后，于 15:12:47 以
  **同 seed / 同 run id / 同目录 / 同配置** 重启（`config_hash` 与首轮一致，无任何语义改动）。
- **round1 的处理：重新训练并覆盖，未复用 checkpoint。** 依据：崩溃发生在 round1 的**评估**阶段，
  此前 round1 无任何完成标记（既无 `.round1_done`，也无当时尚不存在的训练段标记），编排器按设计
  重新执行 `train_one round1_ScienceQA_task`，输出目录与首轮相同
  （`checkpoints/CoIN_Replay_random/r010/run_0001_seed_358341059/round1_task_llava_lora`），**同名覆盖**；
  日志中 15:26:14 出现 `[train:round1_ScienceQA_task] 启动` 亦与之一致。
- 限制与说明：该结论基于当次编排日志与编排器语义；云端实例随后已释放（`2026-09-12` 复核端口不可达），
  无法再读取首轮 ckpt 的 mtime 佐证。**指标未受污染的独立证据**：最终 7 个 ckpt 通过参数级 finite
  校验、replay 段 tensor-diff `changed=448`、四个任务 10 个评估单元预测校验、独立重算
  （maxA=0、dMAA=7.6e-6、dBWT=2.8e-5），`validation_report.md` 20 项全 PASS。

## 4. 该门禁缺口的永久修复（2026-09-12，不改动任何已发布结果）

| 修复 | 内容 |
|---|---|
| eval 脚本显式路径 | 四个 `scripts/LLaVA/Eval/*.sh` 新增 `MODEL_BASE="${MODEL_BASE:-./checkpoints/...}"`、`IMAGE_FOLDER="${IMAGE_FOLDER:-./cl_dataset}"`，`--model-base` / `--image-folder` / 后处理参数（`--base-dir`、`--annotation-file`、`--test-file`）全部改为变量；**默认值保持旧行为**，不影响 prefix 既有流程 |
| python 侧硬编码 | `eval_gqa.py` 新增 `--data-root`（默认与旧行为一致），替换原硬编码 `os.path.join('./cl_dataset/GQA', ...)` |
| 训练前门禁 | `coin_lib.eval_path_audit` + CLI `eval-path-audit`；`run_replay_exp.sh` 在**训练前**调用：① 覆盖钩子仍在（有人改回硬编码即 FAIL）② 传入的模型/数据路径存在 ③ 文件里残余的相对路径必须可在 run 根解析，否则 FAIL |
| 分阶段完成标记 | 新增 `.round<j>_train_done`（训练段完成 + ckpt 校验通过后写入）；恢复时若标记存在且 `ckpt-validate` 通过 → **只重做评估，不重训 task 段**；标记存在但 ckpt 失效 → 删标记重训（不轻信标记） |
| 回归测试 | 新增 `tests/test_eval_path_gate.py`（含「把实踩 bug 改回去必须被拦下」用例、CLI 退出码契约）＋ `test_orchestrator_dryrun.py` 四类用例（门禁 PASS、fail-fast 早于训练、恢复不重训、ckpt 失效不轻信标记） |

修复后本地套件：**161 tests / 0 fail / 0 error / 18 skip**（零依赖环境；torch 与 PIL 用例须在云端执行）。
另修一处本次暴露的编排缺陷：门禁 PASS 输出用 `| head` 截断会在 `set -euo pipefail` 下触发 SIGPIPE
导致脚本静默退出（曾使 DRY_RUN 全链路用例失败），已改为不截断。

**本次 run 无需重跑**：最终 checkpoint 与评估产物全部通过验收，指标未受污染。

## 5. 存储与清理清单（只报告，不实际删除）

审计时（2026-09-11）：`<workspace-root>` 1.0T 卷 **532G used / 493G avail（52%）**，inode 24%，shm 200G tmpfs。

- 必须保留：`datasets` 44G、`models` 15G、`conda` 5.7G、r001 五个 run（各 6.6G）、r010 run 144G、导出六件套 36K、审计日志 3.5M。
- 可安全清理候选（需授权，未执行）：r010 run 下 7 个 `checkpoint-*` HF 单 epoch 存档（约 140G，不参与指标/加载链/验收，删除后失去中途恢复能力）；`<workspace-root>/project/checkpoints` 288G（prefix 历史探索产物，建议单独决定）；`<workspace-root>/tmp` 残留（<0.1G）。
- 状态不明、禁止自动处理：`tmp` 下未归类的历史脚本/JSON、本地两份历史笔记。

## 6. 统计勘误与口径说明

- 勘误：ratio 通用化提交 `9e25218` 的改动规模应为 **+2012 / −148，26 files**（此前一份叙述性报告写作
  +1974/−148，已在本文件与仓库文档中更正；提交内容与 hash 不变）。
- `summary.json` 中 `paired.this_result_commit = null` 属**正常**：提交无法预知自身 SHA；真实 C SHA
  记录在 registry `result_commit` 与本文件 §1 提交链中（已发布六件套**不回头修改**）。
- `repro` 字段列出的是同目录另外五个文件的 sha256（`summary.json` 无法自引用）。
- BWT 口径：BWT 由各任务 `final − diagonal` 合成（ScienceQA +3.3247、TextVQA +19.7300、
  ImageNet +91.8600、GQA 0），其中 ImageNet 在 round3 被强 replay 干扰至 4.67、round4 恢复到 96.53，
  对 BWT 贡献极大 —— **不应表述为「几乎没有遗忘」**；准确表述为「final_avg 与 BWT 更高、MAA 更低」
  的**稳定性—可塑性权衡**。
- 代码提交：r001 各 run = `16f27d4`，本 run = `41abc5a`；「固定项一致」严格指模型/数据/训练超参/
  训练 seed/replay sample seed/抽样算法一致，提交差异作为配对设计限制保留（向后兼容测试见仓库
  `PAIRED_REPORT.md` §4 与兼容性证据）。

## 7. 证据来源与脱敏

- 结论证据：云端审计日志与输出 `<log-root>/`（命令名 + 结论已入
  [PRELAUNCH_AUDIT.json](PRELAUNCH_AUDIT.json)，原始日志不入库）；审计脚本 sha256 同见该文件。
- 提交级证据：A/C/D 三个 commit 及六件套文件可在 GitHub 直接读取；`summary.json.repro` 内含五个
  发布文件的 sha256。
- 本文件与 JSON 均已脱敏：云主机/端口、工作区与日志绝对路径一律用 `<cloud-host>` / `<workspace-root>` / `<run-root>` / `<log-root>` 占位；无凭据、无 checkpoint/权重/完整 replay JSON/完整预测/原始日志/缓存。

## 8. 下一次正式实验前的云环境复验清单（PENDING，2026-09-12）

本节修复（提交 `bad0e3c`）目前**只有本地零依赖验证**（161 tests / 0 fail / 18 skip）。正式运行代码
`41abc5a` 的「146 tests / 0 skip」不能覆盖新代码，因此下次开实例后、正式启动前必须执行：

1. 完整云环境 `bash scripts/CoIN_Replay/run_tests.sh` —— **161 tests，torch 与 PIL 用例必须 0 skip**；
2. `python3 scripts/CoIN_Replay/coin_lib.py eval-path-audit --root <repo> --model-base <模型> --image-folder <图片根>`
   —— 门禁在真实路径下 PASS（并故意改坏一处验证 fail-fast）；
3. 四任务最小真实评估（8 题级）跑通并校验产物；
4. DRY_RUN 恢复测试：训练完成 + 评估失败 → 恢复只重做评估、不重训 task 段
   （`test_train_marker_skips_retrain_after_eval_failure`）。

自查工具：`tools/random_replay_sync_check.py`（registry/README/配对报告一致性 + 脱敏扫描），已接入
`run_tests.sh` 门禁，push 前应先跑。
