# Validation Report — run_0004_seed_1212218766

- 生成: 2026-09-10T14:07:53+08:00（random_replay_finalize.py assemble）
- run_manifest engine run_id: coin_replay_r0.01_20260910_025125
- git commit: 16f27d4469b07225bdcf820e43bac915809e4bbd
- config_hash: d2979e5ed045800ea2383ea1c39f4b44aecdadccb0f471212c14f9e9b24cccef
- test_mode: False（True=零 GPU 文件级降级，仅限 canary/单测）

## 检查清单

| 检查项 | 结果 | 证据 |
|---|---|---|
| .complete 存在 | PASS | .complete |
| 结果目录归属 run_id | PASS | run_0004_seed_1212218766 |
| run_manifest sample_mode==random | PASS | sample_mode=random |
| run_manifest replay_sample_seed==登记 | PASS | replay_sample_seed=1212218766 |
| run_manifest ratio==0.01 | PASS | ratio=0.01 |
| run_manifest replay_accum==1 | PASS | replay_accum=1 |
| coin_metrics.json 结构完整 | PASS | 缺 无 |
| round manifests 1..4 完整可解析 | PASS |  |
| replay json 与 sidecar SHA 一致 (r2..4) | PASS | r2 json-sha=True; r3 json-sha=True; r4 json-sha=True |
| sampling_algorithm==期望常量（全部 sidecar） | PASS | sha256_task_seed_python_shuffle_v1 |
| 源数据独立重建门（SHA/N/k/排列/ids/replay 顺序） | PASS | round2..4 × 全部历史任务 PASS |
| CKPT/RES/REPLAY 同属 random/r001/<run_id> 且互异 | PASS | ckpt=run_0004_seed_1212218766 res=run_0004_seed_1212218766 replay=run_0004_seed_1212218766 |
| checkpoint 校验（7 ckpt） | PASS | round1_task_llava_lora: finite=True hash=6b2891a3992d; round2_task_llava_lora: finite=True hash=51fc3711ae92; round2_replay_llava_lora: finite=True hash=b9e5bdcbd131; round3_task_llava_lora: finite=True hash=186eddfc5d42; round3_replay_llava_lora: finite=True hash=f3f12b40bb6b; round4_task_llava_lora: finite=True hash=5abf996fdd21; round4_replay_llava_lora: finite=True hash=f1d25079e485 |
| replay tensor-diff | PASS | r2: changed=448 hash_differs=True finite=True; r3: changed=448 hash_differs=True finite=True; r4: changed=448 hash_differs=True finite=True |
| non_lora_trainables task vs replay 不同 (r2..4) | PASS | r2: sha_differ=True; r3: sha_differ=True; r4: sha_differ=True |
| A 矩阵单元完整（10 eval 单元 + 预测校验） | PASS | units=10 |
| MAA/BWT 独立重算 == coin_metrics.json | PASS | maxA=0.00e+00 dMAA=6.89e-06 dBWT=4.94e-05 |

## A 矩阵（10 单元全精度）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.190285 | — | — | — |
| round2 | 70.360764 | 53.940000 | — | — |
| round3 | 69.959915 | 48.500000 | 54.510000 | — |
| round4 | 65.196888 | 49.350000 | 93.310000 | 39.290000 |

## 指标（独立重算，全精度）

| 指标 | 重算值 | coin_metrics.json |
|---|---|---|
| MAA | 63.696007 | 63.696 |
| CoIN BWT | 6.554151 | 6.5542 |
| Final Avg | 61.786722 | — |
| 终局旧任务均值 | 69.285629 | — |
| 对角项均值 | 55.232571 | — |
| row_averages | [73.1903, 62.1504, 57.6566, 61.7867] | — |

## replay 抽样一致性（replay_selection_summary.json）

- round2: mode=random sample_seed=1212218766 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=dc1cb9c9bb51f85f…
  - ScienceQA: N=12726 k=127 task_seed=6890819d44c71595… ids_sha=b111b2fe5ef5f299…
- round3: mode=random sample_seed=1212218766 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=f051384347d85e68…
  - ScienceQA: N=12726 k=127 task_seed=6890819d44c71595… ids_sha=b111b2fe5ef5f299…
  - TextVQA: N=34602 k=346 task_seed=c10102bd21877399… ids_sha=72980822b49ea6e0…
- round4: mode=random sample_seed=1212218766 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=b05f5a0163bb2f92…
  - ScienceQA: N=12726 k=127 task_seed=6890819d44c71595… ids_sha=b111b2fe5ef5f299…
  - TextVQA: N=34602 k=346 task_seed=c10102bd21877399… ids_sha=72980822b49ea6e0…
  - ImageNet: N=129833 k=1298 task_seed=36a2df4ad52e8fed… ids_sha=75d194d53644ec9e…

## 复现 hash（发布文件 sha256 全文见 summary.json repro）


结论: 全部验收项 PASS —— 可发布。

| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |
