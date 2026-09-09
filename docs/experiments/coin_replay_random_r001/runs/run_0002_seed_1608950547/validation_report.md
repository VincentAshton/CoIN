# Validation Report — run_0002_seed_1608950547

- 生成: 2026-09-09T19:26:42+08:00（random_replay_finalize.py assemble）
- run_manifest engine run_id: coin_replay_r0.01_20260909_120116
- git commit: 16f27d4469b07225bdcf820e43bac915809e4bbd
- config_hash: 3fc0410240f49ba044c0e1a08135d3da6e383c36ce21b6c55fc0bc307ea47993
- test_mode: False（True=零 GPU 文件级降级，仅限 canary/单测）

## 检查清单

| 检查项 | 结果 | 证据 |
|---|---|---|
| .complete 存在 | PASS | .complete |
| 结果目录归属 run_id | PASS | run_0002_seed_1608950547 |
| run_manifest sample_mode==random | PASS | sample_mode=random |
| run_manifest replay_sample_seed==登记 | PASS | replay_sample_seed=1608950547 |
| run_manifest ratio==0.01 | PASS | ratio=0.01 |
| run_manifest replay_accum==1 | PASS | replay_accum=1 |
| coin_metrics.json 结构完整 | PASS | 缺 无 |
| round manifests 1..4 完整可解析 | PASS |  |
| replay json 与 sidecar SHA 一致 (r2..4) | PASS | r2 json-sha=True; r3 json-sha=True; r4 json-sha=True |
| sampling_algorithm==期望常量（全部 sidecar） | PASS | sha256_task_seed_python_shuffle_v1 |
| 源数据独立重建门（SHA/N/k/排列/ids/replay 顺序） | PASS | round2..4 × 全部历史任务 PASS |
| CKPT/RES/REPLAY 同属 random/r001/<run_id> 且互异 | PASS | ckpt=run_0002_seed_1608950547 res=run_0002_seed_1608950547 replay=run_0002_seed_1608950547 |
| checkpoint 校验（7 ckpt） | PASS | round1_task_llava_lora: finite=True hash=6fefa83ed921; round2_task_llava_lora: finite=True hash=2fe63a50d98b; round2_replay_llava_lora: finite=True hash=4e7ce25508ba; round3_task_llava_lora: finite=True hash=4a7cc4c41398; round3_replay_llava_lora: finite=True hash=55e5eb0c75a4; round4_task_llava_lora: finite=True hash=7c81b7da47ff; round4_replay_llava_lora: finite=True hash=a21415501ddf |
| replay tensor-diff | PASS | r2: changed=448 hash_differs=True finite=True; r3: changed=448 hash_differs=True finite=True; r4: changed=448 hash_differs=True finite=True |
| non_lora_trainables task vs replay 不同 (r2..4) | PASS | r2: sha_differ=True; r3: sha_differ=True; r4: sha_differ=True |
| A 矩阵单元完整（10 eval 单元 + 预测校验） | PASS | units=10 |
| MAA/BWT 独立重算 == coin_metrics.json | PASS | maxA=0.00e+00 dMAA=2.78e-05 dBWT=4.56e-05 |

## A 矩阵（10 单元全精度）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 72.907333 | — | — | — |
| round2 | 71.799104 | 54.040000 | — | — |
| round3 | 67.106814 | 49.620000 | 21.660000 | — |
| round4 | 70.832351 | 50.880000 | 94.120000 | 39.940000 |

## 指标（独立重算，全精度）

| 指标 | 重算值 | coin_metrics.json |
|---|---|---|
| MAA | 61.474728 | 61.4747 |
| CoIN BWT | 16.806254 | 16.8063 |
| Final Avg | 63.943088 | — |
| 终局旧任务均值 | 71.944117 | — |
| 对角项均值 | 47.136833 | — |
| row_averages | [72.9073, 62.9196, 46.1289, 63.9431] | — |

## replay 抽样一致性（replay_selection_summary.json）

- round2: mode=random sample_seed=1608950547 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=66c37d043619f7a0…
  - ScienceQA: N=12726 k=127 task_seed=265b5875b3282d01… ids_sha=3b54331f5dea3a4f…
- round3: mode=random sample_seed=1608950547 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=2d8ac3ef9c9a2b51…
  - ScienceQA: N=12726 k=127 task_seed=265b5875b3282d01… ids_sha=3b54331f5dea3a4f…
  - TextVQA: N=34602 k=346 task_seed=9ac69c6033af526a… ids_sha=e07c9b8b8b5183ea…
- round4: mode=random sample_seed=1608950547 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=ce80d0cc97ca4acc…
  - ScienceQA: N=12726 k=127 task_seed=265b5875b3282d01… ids_sha=3b54331f5dea3a4f…
  - TextVQA: N=34602 k=346 task_seed=9ac69c6033af526a… ids_sha=e07c9b8b8b5183ea…
  - ImageNet: N=129833 k=1298 task_seed=a57b06c38851f01e… ids_sha=b946b9f3b92af52f…

## 复现 hash（发布文件 sha256 全文见 summary.json repro）


结论: 全部验收项 PASS —— 可发布。

| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |
