# Validation Report — run_0005_seed_232675848

- 生成: 2026-09-10T21:44:28+08:00（random_replay_finalize.py assemble）
- run_manifest engine run_id: coin_replay_r0.01_20260910_142231
- git commit: 16f27d4469b07225bdcf820e43bac915809e4bbd
- config_hash: cc053788118d3d4089e79b900dfc000efc8191c1f8a5bc36fa76abfe8f4eb44e
- test_mode: False（True=零 GPU 文件级降级，仅限 canary/单测）

## 检查清单

| 检查项 | 结果 | 证据 |
|---|---|---|
| .complete 存在 | PASS | .complete |
| 结果目录归属 run_id | PASS | run_0005_seed_232675848 |
| run_manifest sample_mode==random | PASS | sample_mode=random |
| run_manifest replay_sample_seed==登记 | PASS | replay_sample_seed=232675848 |
| run_manifest ratio==0.01 | PASS | ratio=0.01 |
| run_manifest replay_accum==1 | PASS | replay_accum=1 |
| coin_metrics.json 结构完整 | PASS | 缺 无 |
| round manifests 1..4 完整可解析 | PASS |  |
| replay json 与 sidecar SHA 一致 (r2..4) | PASS | r2 json-sha=True; r3 json-sha=True; r4 json-sha=True |
| sampling_algorithm==期望常量（全部 sidecar） | PASS | sha256_task_seed_python_shuffle_v1 |
| 源数据独立重建门（SHA/N/k/排列/ids/replay 顺序） | PASS | round2..4 × 全部历史任务 PASS |
| CKPT/RES/REPLAY 同属 random/r001/<run_id> 且互异 | PASS | ckpt=run_0005_seed_232675848 res=run_0005_seed_232675848 replay=run_0005_seed_232675848 |
| checkpoint 校验（7 ckpt） | PASS | round1_task_llava_lora: finite=True hash=cfb0d1a66cc4; round2_task_llava_lora: finite=True hash=c35238bacf37; round2_replay_llava_lora: finite=True hash=a28ed234039a; round3_task_llava_lora: finite=True hash=2d5547a55742; round3_replay_llava_lora: finite=True hash=ec7cc1c52666; round4_task_llava_lora: finite=True hash=a237c8afc841; round4_replay_llava_lora: finite=True hash=e1718044357a |
| replay tensor-diff | PASS | r2: changed=448 hash_differs=True finite=True; r3: changed=448 hash_differs=True finite=True; r4: changed=448 hash_differs=True finite=True |
| non_lora_trainables task vs replay 不同 (r2..4) | PASS | r2: sha_differ=True; r3: sha_differ=True; r4: sha_differ=True |
| A 矩阵单元完整（10 eval 单元 + 预测校验） | PASS | units=10 |
| MAA/BWT 独立重算 == coin_metrics.json | PASS | maxA=0.00e+00 dMAA=3.64e-05 dBWT=3.65e-05 |

## A 矩阵（10 单元全精度）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 72.978071 | — | — | — |
| round2 | 67.319029 | 53.290000 | — | — |
| round3 | 70.384343 | 46.490000 | 19.500000 | — |
| round4 | 69.158217 | 50.420000 | 93.430000 | 39.390000 |

## 指标（独立重算，全精度）

| 指标 | 重算值 | coin_metrics.json |
|---|---|---|
| MAA | 60.460064 | 60.4601 |
| CoIN BWT | 16.810037 | 16.81 |
| Final Avg | 63.099554 | — |
| 终局旧任务均值 | 71.002739 | — |
| 对角项均值 | 46.289518 | — |
| row_averages | [72.9781, 60.3045, 45.4581, 63.0996] | — |

## replay 抽样一致性（replay_selection_summary.json）

- round2: mode=random sample_seed=232675848 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=55925b92c9ec11f2…
  - ScienceQA: N=12726 k=127 task_seed=83d9baa3a90c9972… ids_sha=821e5404a74a4c0c…
- round3: mode=random sample_seed=232675848 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=32d516437f674194…
  - ScienceQA: N=12726 k=127 task_seed=83d9baa3a90c9972… ids_sha=821e5404a74a4c0c…
  - TextVQA: N=34602 k=346 task_seed=96a99ab7eb0397ae… ids_sha=4faff25ab16d0038…
- round4: mode=random sample_seed=232675848 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=e49a068ef458c8fc…
  - ScienceQA: N=12726 k=127 task_seed=83d9baa3a90c9972… ids_sha=821e5404a74a4c0c…
  - TextVQA: N=34602 k=346 task_seed=96a99ab7eb0397ae… ids_sha=4faff25ab16d0038…
  - ImageNet: N=129833 k=1298 task_seed=6e398bd514be308a… ids_sha=5a8a077481a5350a…

## 复现 hash（发布文件 sha256 全文见 summary.json repro）


结论: 全部验收项 PASS —— 可发布。

| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |
