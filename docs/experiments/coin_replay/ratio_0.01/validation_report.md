# CoIN+Replay ratio=0.01 — 完整性验收报告（2026-09-05）

运行：`bash scripts/CoIN_Replay/run_sweep.sh 0.01`（tmux coin_sweep_r001_20260905，唯一会话）
runtime commit：运行代码 `17cfa66f009bd1fd1f5d360307f97d4249bf2c5c`；云端 detached HEAD
`069b608`（与 17cfa66 逐字节一致，`git diff 17cfa66 HEAD -- scripts/ ETrain/ coin_lib/` = 0 行）
配置：task accum=16（eff 896）/ replay accum=1（eff 56）/ REPLAY_ACCUM=1 / ENFORCE_MIN_STEPS=1
seed=data_seed=1234（单一 seed，本报告为描述性证据，不构成统计显著性结论）
与 ratio=0.10 唯一配置差异 = ratio 及其派生项（replay 样本数 N、真实 optimizer steps、输出路径）

## 验收项（全部 PASS：verify_complete_r001.sh + 补充核对脚本）
| 项 | 结果 | 证据 |
|---|---|---|
| 退出码 | 0 | logs/formal/ratio_0.01_20260905_035722/exit_code.txt |
| .complete | 存在 | results/CoIN_Replay/ratio_0.01/.complete |
| round markers | 4/4 | .round1_done … .round4_done |
| checkpoint 可加载 | 7/7 | ckpt-validate（4 task：round1-4 + 3 replay：round2-4；逐路径 inventory 见 analysis/checkpoint_inventory.json）|
| 参数 finite | 7/7 | ckpt-validate finite=True |
| task/replay tensor diff | 448/448 changed ×3 | ckpt-tensor-diff（round2/3/4；448 = adapter safetensors 全部 tensor key，missing=0 / unexpected=0 / shape 全一致 / 全部 finite / 规范化 hash task≠replay）|
| DS 真步 = manifest | 3 / 9 / 32 | trainer_state.global_step == round manifest replay_plan.ds_expected_updates（r2 N=127 / r3 N=473 / r4 N=1771）|
| eval 单元 | 10/10 | 三角矩阵单元：ScienceQA r1-4, TextVQA r2-4, ImageNet r3-4, GQA r4（10 个「(round, task) 交集」评估，非 10 次独立重复运行）|
| prediction 数量/ID | 10/10 | merge.jsonl 行数==唯一 ID==问题集（4241/5000/5050/12578）；逐单元与 0.1 行数一致（ID hash 级对比见 analysis/id_hash_comparison.json）|
| coin_metrics.json | 原子生成 | A 矩阵/MAA/BWT 可解析 |
| manifest 复核 | OK | ratio 0.01 / grad_accum 16 / replay_accum 1 / git clean（sanitized 见 run_manifest.sanitized.json）|
| 无 NaN/Inf | OK | 全部 ckpt finite + 训练日志 loss 有限（round1 SQA 0.92→0.32 单调）|
| 磁盘 | 正常 | /root/data 可用 ≥500G |
| 残留进程 | 0 | 无 train_mem/model_vqa/run_sweep |
| 0.1 未被触碰 | OK | 0.1 产物时间戳未变；本运行唯一命令含 0.01（0.1 原结果 SHA256 复核见 analysis/ 与 ratio_0.10/results_sha256.txt）|

## 指标与独立重算
- recompute_metrics.py（公开包双 ratio 版）从 acc_sources.json（10 单元聚合 accuracy，SQA
  为 output_result.jsonl 原始 acc 与 correct/count，其余为 Result.text 原始两位值；读法与
  aggregate_coin.py 一致）独立构造 A 矩阵并计算 MAA/BWT/final avg，与 coin_metrics.json
  交叉验证：最大 A 差异 ≤5e-7（acc_sources 六位舍入），MAA/BWT 差异 0 → CROSS-VALIDATION PASS
- 权威结果：**MAA=60.4406、CoIN BWT=-13.6299、final avg=46.1925**（round4 全任务平均）
- 对角项（A[i,i]）在「该 round 完成 task 训练 + replay 微调后的 checkpoint」上评估——
  即刚学完即被本 round replay 干扰后的值（replay 不含当前任务），非纯净初学精度

## 结果 SHA256
results_sha256.txt（coin_metrics.json、run_manifest.json、10 个 merge.jsonl 的云端原始产物
SHA256；merge.jsonl 等原始预测不入库）

## 运行记录
- 2026-09-05 03:57:55 启动（preflight PASS 04:12 → round1-4 顺序完成）
- 2026-09-05 11:03:31 DONE（.complete 落盘，rc=0），全程 ~7.1h
- round1 SQA task 14 optimizer steps（49.4s/step）；replay 真步 3/9/32
- 全程无 NCCL/OOM/NaN/CUDA 错误；ratio=0.10 未重跑、未修改
