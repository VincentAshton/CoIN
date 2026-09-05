# CoIN+Replay ratio=0.10 — 完整性验收报告（2026-09-04；2026-09-05 复核修订）

运行：`bash scripts/CoIN_Replay/run_sweep.sh 0.1`（tmux coin_sweep_r010_20260904）
runtime commit `17cfa66f009bd1fd1f5d360307f97d4249bf2c5c`（本地=GitHub=云端，运行中未改动）
seed=data_seed=1234（单 seed）；10 个 eval 单元为「(round, 任务)」三角矩阵交集（非 10 次
独立重复）；对角项 A[i,i] 在该轮 task+replay 完成后的 checkpoint 上评估。

## 验收项（全部 PASS）
| 项 | 结果 | 证据 |
|---|---|---|
| 退出码 | 0 | logs/formal/…/exit_code.txt |
| .complete | 存在 | results/CoIN_Replay/ratio_0.1/.complete |
| round markers | 4/4 | .round1_done … .round4_done |
| checkpoint 可加载 | 7/7 | ckpt-validate（4 task：round1-4 + 3 replay：round2-4；逐路径 inventory 见 analysis/checkpoint_inventory.json）|
| 参数 finite | 7/7 | ckpt-validate finite=True |
| task/replay tensor diff | 448/448 changed ×3 | ckpt-tensor-diff（round2/3/4；448 = adapter safetensors 全部 tensor key，missing=0/unexpected=0）|
| DS 真步 = manifest | 23 / 85 / 317 | trainer_state.global_step == round manifest replay_plan.ds_expected_updates |
| eval 单元 | 10/10 | ScienceQA r1-4, TextVQA r2-4, ImageNet r3-4, GQA r4 |
| prediction 数量/ID | 10/10 | 每单元 merge.jsonl 行数/唯一 ID == 问题集（4241/5000/5050/12578；ID 级 hash 对比见 analysis/id_hash_comparison.json）|
| coin_metrics.json | 原子生成 | MAA/BWT 可解析 |
| manifest 复核 | OK | ratio 0.1 / grad_accum 16 / replay_accum 1 / git 17cfa66 |
| 无 NaN/Inf | OK | 全部 ckpt finite + 训练日志 loss 有限 |
| 磁盘 | 正常 | /root/data 647G 可用（无异常增长）|
| 残留进程 | 0 | 无 train_mem/model_vqa/run_sweep |

## 指标独立重算
- recompute_metrics.py（coin_replay/ 顶层双 ratio 版）从 acc_sources.json（10 单元聚合
  accuracy）独立构造 A 矩阵，计算 MAA/BWT/final avg，与 coin_metrics.json 交叉验证：
  A 矩阵最大差异 4.9e-7（acc_sources 六位舍入），MAA/BWT 差异 0（CROSS-VALIDATION PASS）
- 权威结果：MAA=57.5057、CoIN BWT=17.2306、final avg=55.7834

## 结果 SHA256
results_sha256.txt（coin_metrics.json、run_manifest.json、10 个 merge.jsonl）

## 运行记录
- 2026-09-04 03:30:23 启动（preflight PASS → round1-4 顺序完成）
- 2026-09-04 11:04:01 DONE（.complete 落盘）
- 全程无 NCCL/OOM/NaN/CUDA 错误；ratio=0.01 未启动
- 2026-09-05：ratio=0.01 运行完成前后复核本报告与 0.10 产物（SHA256 未变），
  修正 checkpoint 计数口径 8→7（实际为 4 task + 3 replay）
