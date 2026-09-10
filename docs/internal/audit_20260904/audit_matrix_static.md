# CoIN+Replay 全链路审计 · 阶段三：全矩阵步数静态审计

模拟器：精确移植 accelerate 0.21 BatchSamplerShard._iter_with_no_split（even_batches=True,
drop_last=False）+ transformers 4.32 步进语义 + deepspeed 0.14 boundary 语义。
代码 scripts/matrix_static_sim.py（可复跑）+ coin_lib.per_rank_micro_batches（同逻辑内联）；
全量 matrix_static.csv。

## ⚠ 2026-09-04 模拟修正记录（审计过程发现自身 bug，保留轨迹）
初版模拟器在 partial 尾时回绕起点取 idx=B（应为 B-1，源码 L204-206 仅满长尾 +1）→
伪造出 per-rank ±1 不均（如 [23,23,22,23]）→ 已修正。**修正后实证：even_batches=True 下
所有 N≥1 场景 4 rank 完全均匀**（accelerate 的设计目标）；N=1272 → [23,23,23,23] 等。
初版据此推的「gas1 尾部不均 hang 风险」为模拟 bug 假象，作废（真实均匀 → 无该风险）。
hang 风险复检结论：DS 齐步语义下 rank 等长 → 训练齐步，无尾部不均问题。

## 权威口径（修正后）
- per-rank micro M = 均匀值 = BatchSamplerShard yield 数（每 rank 相同）；
  精确 = 模拟器；近似 = ceil(ceil(N/batch)/world)，partial 尾与整除细节以模拟器为准
- DS 真步 = M // gas（rank 齐步；EPOCHS=1）。旧 floor(N/896) 公式**低估**某些场景：
  N=895 → M=16 → 真步 1（旧公式 0）；N=1771/gas16 → M=32 → 真步 2（旧报告 1）
- HF global_step = M//gas + phantom（仅 M ≤ gas 且余数非 0 时 +1，DS wrapper no-op 假象）
- 尾部未提交 micro = M - (M//gas)×gas（EPOCHS=1 时梯度保留后随训练结束丢弃）

## 任务段（accum16, batch14, world4, EPOCHS=1）
| 任务 | N | M/rank | 真步/epoch | HF gs | 尾部丢弃 micro | 尾部丢弃样本 |
|---|---|---|---|---|---|---|
| ScienceQA | 12726 | 228 | 14 | 14 | 4 | 224 |
| TextVQA | 34602 | 618 | 38 | 38 | 10 | 560 |
| ImageNet | 129833 | 2319 | 144 | 144 | 15 | 840 |
| GQA | 72140 | 1289 | 80 | 80 | 9 | 504 |

## Replay 段（M/rank 均匀；两 gas 对照）
| 场景 | N | M/rank | gas16 真步 | gas1 真步 |
|---|---|---|---|---|
| 0.01 r2 | 127 | 3 | 0（No-Go ✓） | 3（任务书 3 ✓） |
| 0.01 r3 | 473 | 9 | 0 | 9（✓） |
| 0.01 r4 | 1771 | 32 | 2（旧报告 1，待实测定论） | 32（✓） |
| 0.10 r2 | 1272 | 23 | 1（报告 1 ✓） | 23（✓） |
| 0.10 r3 | 4732 | 85 | 5（报告 5 ✓） | 85（✓） |
| 0.10 r4 | 17715 | 317 | 19（报告 19 ✓） | 317（✓） |

## 边界矩阵要点（全表 matrix_static.csv；M//gas 即 DS 真步）
- N≤56：M=1 → gas≥8 时 0 真步（HF phantom gs=1）；gas1 时 1 真步
- N=57..112：M=2 → gas1: 2；gas16: 0（2<16）
- N=127：M=3 → gas1: 3；gas16: 0
- N=895/896：M=16 → gas16: 1 真步（895 也 1——旧 floor 公式给 0，低估实证）
- N=897..1271：M=17..22 → gas16: 1
- N=1272：M=23 → gas8: 2（23//8）；gas16: 1
- warmup（T = HF 步数上限 ≈ max(1, M//gas) 及 ceil；warm=int(0.03T)）：
  gas1 replay: r2 T=3 warm=0 / r3 T=9 warm=0 / 0.1 r4 T=317 warm=9 → 短段无 warmup（LR 首步=峰值）
- LR 首真步：无 warmup 段 = 2e-4；0.1 r4 gas1（warm=9）首步 = 2e-4×(1/9)≈2.2e-5

## 对阶段 II 报告数字的交叉核对（M 口径）
0.01 r2 gas16 → 0 真步 ✓ / 0.1 r2 → 1 ✓ / r3 → 5 ✓ / r4 → 19 ✓ /
0.01 r4 → 模拟 2 vs 旧报告 1（旧为 floor(1771/896) 或错模拟 31//16 口径）——GPU 实测项

## GPU 实测必答问题（阶段七）
1. N=127/473/1272/4732/17715 gas1：DS engine.global_steps 是否 = M//1（3/9/23/85/317）
2. N=1771 gas16（若测）：真步 1 还是 2（本表预测 2）
3. N=1 退化：M=1 全 rank（纯回绕样本）→ gas1 1 真步可完成？
4. 各场景 _step_applied 与 tensor diff 与本表一致

