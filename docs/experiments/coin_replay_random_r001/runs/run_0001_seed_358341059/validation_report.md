# Validation Report — run_0001_seed_358341059

- 生成: 2026-09-09T11:41:38+08:00（random_replay_finalize.py assemble）
- run_manifest engine run_id: coin_replay_r0.01_20260909_043021
- git commit: 16f27d4469b07225bdcf820e43bac915809e4bbd
- config_hash: d06b56169e4db95b7cab47c687f4ded43ee34a9417319f2731d44dcb5461b110
- test_mode: False（True=零 GPU 文件级降级，仅限 canary/单测）

## 检查清单

| 检查项 | 结果 | 证据 |
|---|---|---|
| .complete 存在 | PASS | .complete |
| 结果目录归属 run_id | PASS | run_0001_seed_358341059 |
| run_manifest sample_mode==random | PASS | sample_mode=random |
| run_manifest replay_sample_seed==登记 | PASS | replay_sample_seed=358341059 |
| run_manifest ratio==0.01 | PASS | ratio=0.01 |
| run_manifest replay_accum==1 | PASS | replay_accum=1 |
| coin_metrics.json 结构完整 | PASS | 缺 无 |
| round manifests 1..4 完整可解析 | PASS |  |
| replay json 与 sidecar SHA 一致 (r2..4) | PASS | r2 json-sha=True; r3 json-sha=True; r4 json-sha=True |
| sampling_algorithm==期望常量（全部 sidecar） | PASS | sha256_task_seed_python_shuffle_v1 |
| 源数据独立重建门（SHA/N/k/排列/ids/replay 顺序） | PASS | round2..4 × 全部历史任务 PASS |
| CKPT/RES/REPLAY 同属 random/r001/<run_id> 且互异 | PASS | ckpt=run_0001_seed_358341059 res=run_0001_seed_358341059 replay=run_0001_seed_358341059 |
| checkpoint 校验（7 ckpt） | PASS | round1_task_llava_lora: finite=True hash=dd459c6a4086; round2_task_llava_lora: finite=True hash=e0b814100c3f; round2_replay_llava_lora: finite=True hash=2195c9944271; round3_task_llava_lora: finite=True hash=5ca305402ec4; round3_replay_llava_lora: finite=True hash=d64a8d571054; round4_task_llava_lora: finite=True hash=39222724359e; round4_replay_llava_lora: finite=True hash=4392bbd72981 |
| replay tensor-diff | PASS | r2: changed=448 hash_differs=True finite=True; r3: changed=448 hash_differs=True finite=True; r4: changed=448 hash_differs=True finite=True |
| non_lora_trainables task vs replay 不同 (r2..4) | PASS | r2: sha_differ=True; r3: sha_differ=True; r4: sha_differ=True |
| A 矩阵单元完整（10 eval 单元 + 预测校验） | PASS | units=10 |
| MAA/BWT 独立重算 == coin_metrics.json | PASS | maxA=0.00e+00 dMAA=2.14e-05 dBWT=2.32e-05 |

## A 矩阵（10 单元全精度）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.166706 | — | — | — |
| round2 | 68.663051 | 56.340000 | — | — |
| round3 | 69.464749 | 47.050000 | 46.360000 | — |
| round4 | 61.188399 | 52.000000 | 94.420000 | 38.200000 |

## 指标（独立重算，全精度）

| 指标 | 重算值 | coin_metrics.json |
|---|---|---|
| MAA | 62.852979 | 62.853 |
| CoIN BWT | 7.935423 | 7.9354 |
| Final Avg | 61.452100 | — |
| 终局旧任务均值 | 69.202800 | — |
| 对角项均值 | 53.516676 | — |
| row_averages | [73.1667, 62.5015, 54.2916, 61.4521] | — |

## replay 抽样一致性（replay_selection_summary.json）

- round2: mode=random sample_seed=358341059 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=ff2d15556482c3c1…
  - ScienceQA: N=12726 k=127 task_seed=a8f172415563795f… ids_sha=928e415e9fbdf6f0…
- round3: mode=random sample_seed=358341059 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=21dceed8ff602f05…
  - ScienceQA: N=12726 k=127 task_seed=a8f172415563795f… ids_sha=928e415e9fbdf6f0…
  - TextVQA: N=34602 k=346 task_seed=33084a30aee13b7a… ids_sha=4586f12c1c6b69a6…
- round4: mode=random sample_seed=358341059 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=bec8c1326df2638f…
  - ScienceQA: N=12726 k=127 task_seed=a8f172415563795f… ids_sha=928e415e9fbdf6f0…
  - TextVQA: N=34602 k=346 task_seed=33084a30aee13b7a… ids_sha=4586f12c1c6b69a6…
  - ImageNet: N=129833 k=1298 task_seed=f38d05efaf95c676… ids_sha=642a720d8284d0fe…

## 复现 hash（发布文件 sha256 全文见 summary.json repro）


结论: 全部验收项 PASS —— 可发布。

| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |
