# Validation Report — run_0003_seed_2089198544

- 生成: 2026-09-10T02:38:16+08:00（random_replay_finalize.py assemble）
- run_manifest engine run_id: coin_replay_r0.01_20260909_194715
- git commit: 16f27d4469b07225bdcf820e43bac915809e4bbd
- config_hash: 78b58ec20c3e917f0548ffae4c2245e72caf5ef622681f2ee46cfaa3ec8afb8c
- test_mode: False（True=零 GPU 文件级降级，仅限 canary/单测）

## 检查清单

| 检查项 | 结果 | 证据 |
|---|---|---|
| .complete 存在 | PASS | .complete |
| 结果目录归属 run_id | PASS | run_0003_seed_2089198544 |
| run_manifest sample_mode==random | PASS | sample_mode=random |
| run_manifest replay_sample_seed==登记 | PASS | replay_sample_seed=2089198544 |
| run_manifest ratio==0.01 | PASS | ratio=0.01 |
| run_manifest replay_accum==1 | PASS | replay_accum=1 |
| coin_metrics.json 结构完整 | PASS | 缺 无 |
| round manifests 1..4 完整可解析 | PASS |  |
| replay json 与 sidecar SHA 一致 (r2..4) | PASS | r2 json-sha=True; r3 json-sha=True; r4 json-sha=True |
| sampling_algorithm==期望常量（全部 sidecar） | PASS | sha256_task_seed_python_shuffle_v1 |
| 源数据独立重建门（SHA/N/k/排列/ids/replay 顺序） | PASS | round2..4 × 全部历史任务 PASS |
| CKPT/RES/REPLAY 同属 random/r001/<run_id> 且互异 | PASS | ckpt=run_0003_seed_2089198544 res=run_0003_seed_2089198544 replay=run_0003_seed_2089198544 |
| checkpoint 校验（7 ckpt） | PASS | round1_task_llava_lora: finite=True hash=a8ec1f914277; round2_task_llava_lora: finite=True hash=664151825bfc; round2_replay_llava_lora: finite=True hash=33e7a3fc7313; round3_task_llava_lora: finite=True hash=738853f90f62; round3_replay_llava_lora: finite=True hash=c0bd362ecff2; round4_task_llava_lora: finite=True hash=f53926bbda31; round4_replay_llava_lora: finite=True hash=8f057f266d73 |
| replay tensor-diff | PASS | r2: changed=448 hash_differs=True finite=True; r3: changed=448 hash_differs=True finite=True; r4: changed=448 hash_differs=True finite=True |
| non_lora_trainables task vs replay 不同 (r2..4) | PASS | r2: sha_differ=True; r3: sha_differ=True; r4: sha_differ=True |
| A 矩阵单元完整（10 eval 单元 + 预测校验） | PASS | units=10 |
| MAA/BWT 独立重算 == coin_metrics.json | PASS | maxA=0.00e+00 dMAA=1.75e-05 dBWT=3.82e-05 |

## A 矩阵（10 单元全精度）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.095968 | — | — | — |
| round2 | 68.568734 | 55.050000 | — | — |
| round3 | 65.644895 | 46.160000 | 37.070000 | — |
| round4 | 54.656921 | 50.740000 | 94.280000 | 40.760000 |

## 指标（独立重算，全精度）

| 指标 | 重算值 | coin_metrics.json |
|---|---|---|
| MAA | 61.159882 | 61.1599 |
| CoIN BWT | 8.615238 | 8.6152 |
| Final Avg | 60.109230 | — |
| 终局旧任务均值 | 66.558974 | — |
| 对角项均值 | 51.493992 | — |
| row_averages | [73.096, 61.8094, 49.625, 60.1092] | — |

## replay 抽样一致性（replay_selection_summary.json）

- round2: mode=random sample_seed=2089198544 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=8512cf730c91ad70…
  - ScienceQA: N=12726 k=127 task_seed=ff3f943686955cb1… ids_sha=01c3509c54a18533…
- round3: mode=random sample_seed=2089198544 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=4e13bf61b76498d4…
  - ScienceQA: N=12726 k=127 task_seed=ff3f943686955cb1… ids_sha=01c3509c54a18533…
  - TextVQA: N=34602 k=346 task_seed=51e2242b7b9b03c2… ids_sha=b7b81bcbe23158de…
- round4: mode=random sample_seed=2089198544 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=5907cbce45050b01…
  - ScienceQA: N=12726 k=127 task_seed=ff3f943686955cb1… ids_sha=01c3509c54a18533…
  - TextVQA: N=34602 k=346 task_seed=51e2242b7b9b03c2… ids_sha=b7b81bcbe23158de…
  - ImageNet: N=129833 k=1298 task_seed=9ddc88b632f97f34… ids_sha=1e19427104a2cfdc…

## 复现 hash（发布文件 sha256 全文见 summary.json repro）


结论: 全部验收项 PASS —— 可发布。

| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |
