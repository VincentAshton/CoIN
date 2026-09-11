# Validation Report — run_0001_seed_358341059

- 生成: 2026-09-11T22:49:58+08:00（random_replay_finalize.py assemble）
- expected ratio: 0.10（ratio tag r010）
- run_manifest engine run_id: coin_replay_r0.1_20260911_115705
- git commit: 41abc5af222ba5eed7c6d25edab4a2bb2d110bcc
- config_hash: a6f4ba7c63ebd57a0139fb28e8afbb83177c1476b83a170501e6d231dfa60b35
- test_mode: False（True=零 GPU 文件级降级，仅限 canary/单测）

## 检查清单

| 检查项 | 结果 | 证据 |
|---|---|---|
| expected ratio 属于允许集合 | PASS | expected_ratio=0.10 → 归一化 0.10 / tag r010 |
| run_id 格式 run_NNNN_seed_<seed> 且内嵌 seed==登记 seed | PASS | run_0001_seed_358341059 |
| .complete 存在 | PASS | formal/r010_run_0001_seed_358341059/results/CoIN_Replay_random/r010/run_0001_seed_358341059/.complete |
| 三个目录 basename 均为 run_id | PASS | formal/r010_run_0001_seed_358341059/results/CoIN_Replay_random/r010/run_0001_seed_358341059 |
| run_manifest sample_mode==random | PASS | sample_mode=random |
| run_manifest replay_sample_seed==登记 | PASS | replay_sample_seed=358341059 |
| run_manifest ratio == expected ratio（数值比较） | PASS | manifest=0.1 expected=0.10 |
| run_manifest replay_accum==1 | PASS | replay_accum=1 |
| coin_metrics.json 结构完整 | PASS | 缺 无 |
| round manifests 1..4 完整可解析 | PASS |  |
| replay json 与 sidecar SHA 一致 (r2..4) | PASS | r2 json-sha=True; r3 json-sha=True; r4 json-sha=True |
| sampling_algorithm==期望常量（全部 sidecar） | PASS | sha256_task_seed_python_shuffle_v1 |
| 源数据独立重建门（SHA/N/k/排列/ids/replay 顺序） | PASS | round2..4 × 全部历史任务 PASS |
| CKPT/RES/REPLAY 同属 random/r010/<run_id> 且互异 | PASS | ckpt=formal/r010_run_0001_seed_358341059/checkpoints/CoIN_Replay_random/r010/run_0001_seed_358341059 res=formal/r010_run_0001_seed_358341059/results/CoIN_Replay_random/r010/run_0001_seed_358341059 replay=formal/r010_run_0001_seed_358341059/playground/Replay_random/r010/run_0001_seed_358341059 |
| checkpoint 校验（7 ckpt） | PASS | round1_task_llava_lora: finite=True hash=aa4155ad695f; round2_task_llava_lora: finite=True hash=14abde75861d; round2_replay_llava_lora: finite=True hash=5da5a7794107; round3_task_llava_lora: finite=True hash=360ed5268904; round3_replay_llava_lora: finite=True hash=d362010bd091; round4_task_llava_lora: finite=True hash=08a89d856df6; round4_replay_llava_lora: finite=True hash=e2ffee6d85c5 |
| replay tensor-diff | PASS | r2: changed=448 hash_differs=True finite=True; r3: changed=448 hash_differs=True finite=True; r4: changed=448 hash_differs=True finite=True |
| non_lora_trainables task vs replay 不同 (r2..4) | PASS | r2: sha_differ=True; r3: sha_differ=True; r4: sha_differ=True |
| A 矩阵单元完整（10 eval 单元 + 预测校验） | PASS | units=10 |
| MAA/BWT 独立重算 == coin_metrics.json | PASS | maxA=0.00e+00 dMAA=7.58e-06 dBWT=2.81e-05 |

## A 矩阵（10 单元全精度）

| round\task | ScienceQA | TextVQA | ImageNet | GQA |
|---|---|---|---|---|
| round1 | 73.143127 | — | — | — |
| round2 | 73.449658 | 36.550000 | — | — |
| round3 | 74.699363 | 58.460000 | 4.670000 | — |
| round4 | 76.467814 | 56.280000 | 96.530000 | 36.330000 |

## 指标（独立重算，全精度）

| 指标 | 重算值 | coin_metrics.json |
|---|---|---|
| MAA | 60.122008 | 60.122 |
| CoIN BWT | 28.728672 | 28.7287 |
| Final Avg | 66.401954 | — |
| 终局旧任务均值 | 76.425938 | — |
| 对角项均值 | 37.673282 | — |
| row_averages | [73.1431, 54.9998, 45.9431, 66.402] | — |

## replay 抽样一致性（replay_selection_summary.json）

- round2: mode=random sample_seed=358341059 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=b5a9804699c93299…
  - ScienceQA: N=12726 k=1272 task_seed=a8f172415563795f… ids_sha=98ea407746206f13…
- round3: mode=random sample_seed=358341059 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=a8c2b61d1852b3da…
  - ScienceQA: N=12726 k=1272 task_seed=a8f172415563795f… ids_sha=98ea407746206f13…
  - TextVQA: N=34602 k=3460 task_seed=33084a30aee13b7a… ids_sha=fe78c3164d06c65f…
- round4: mode=random sample_seed=358341059 algorithm=sha256_task_seed_python_shuffle_v1 output_sha=9941c84a29b9e8fa…
  - ScienceQA: N=12726 k=1272 task_seed=a8f172415563795f… ids_sha=98ea407746206f13…
  - TextVQA: N=34602 k=3460 task_seed=33084a30aee13b7a… ids_sha=fe78c3164d06c65f…
  - ImageNet: N=129833 k=12983 task_seed=f38d05efaf95c676… ids_sha=1735665a71d56b3a…

## 复现 hash（发布文件 sha256 全文见 summary.json repro）


结论: 全部验收项 PASS —— 可发布。

| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |
