# CoIN+Replay 全链路审计 · 阶段四：方法学方案比较表

评审对象：replay 段 0 真实更新的修复方案（阶段 II No-Go 的处置）。

## A. ratio 专属 accum（0.01→1、0.10→8）——【拒绝】
- 两组 replay 有效 batch 不同（56 vs 448）、更新频率不同、梯度噪声不同
- ratio 成为 confound：0.01 与 0.10 的 replay 训练条件不可比，MAA/BWT 对比失去内部有效性
- 与任务书结论一致：造成新的实验混杂

## B. 保持 accum16 ——【拒绝】
- 0.01 全段真实零更新（r2 N=127 / r3 N=473：per-rank micro 3/9 < 16 → 0 真步，矩阵实证）
- 零更新 = 无 replay，实验无意义；评审 C-1 ≥2 真步不可能满足

## C. 改 ratio（0.01→0.05/0.02 使 replay N≥896）——【本轮不采用】
- 改变已批准的研究问题（ratio 集 0.10/0.01 是实验设计，含「极小回放比例」考察意图）
- 若评审将来要求极低比例有效性，另行立项

## D. 所有 replay 固定 accum=1（task accum16）——【推荐实现】
- task 段：accum16 冻结不变 → task checkpoint 链与已封存阶段 I 完全可比
- replay 段：accum1 → 每个 micro-batch（56 样本）即一个真实 optimizer step，
  无「尾部不提交」（每 micro 即 boundary）→ 0.01 与 0.10 的 replay 训练条件完全一致
- 两个 ratio 唯一差异 = replay N（与派生步数），cross-run manifest 可断言
- 真步预期（静态 min 口径）：0.01 r2=2/r3=8/r4=31；0.1 r2=22/r3=85/r4=316
  （rank0/ceil 口径 3/9/32 与 23/85/317——±1 差源于 accelerate even 补齐不均，GPU 实测定论）
- 代价/风险（如实记录）：
  1) replay 有效 batch 896→56：梯度噪声增大——replay 段本为小样本段，此为该设计固有语义，
     且两个 ratio 一致（可比性保留）
  2) 无 warmup（短段 int(0.03×T)=0）→ 首真步 LR=2e-4 峰值（原 accum16 计划同场景亦为 0 warmup，
     无新增偏差）
  3) CPU-offload optimizer step 频率 ×16（每 56 样本一次 update）→ 吞吐下降，实测估算（阶段七）
  4) per-rank micro 尾部不均（accelerate even 补齐，±1 micro）→ gas1 每 micro 均为同步边界，
     尾部不均可能 NCCL hang —— 实机验证必查项（N=127/1272/17715 gas1）
- 不丢 replay 尾部：任何 N 都产生 ≥2 真步（N≥112 时 min(M)≥2；N<112 的 replay 场景不存在——
  最小 0.01 r2 N=127 → 2 真步）

## E. 修改 Trainer/DeepSpeed 强制 flush partial accumulation ——【仅技术可行性分析，本轮不实现】
- 思路：transformers 4.32 在 dataloader 耗尽时对尾部未满 gas 的梯度强制 engine.step()
  （HF 侧 is_last_step 分支已尝试 sync_gradients=True，但 DS 不看 accelerate gradient_state）
  → 需在 DS engine 层支持「尾部 partial apply」或调用 set_gradient_accumulation_boundary(True)
  强制最后一个 micro 为 boundary
- 风险：
  1) 梯度归一化：DS backward 已按 gas 缩放 loss（_scale_loss_by_gas）——partial 段 apply 时
     缩放仍按完整 gas → 尾段梯度被低估 gas/gas_actual 倍 → 更新幅度失真（需额外 rescale）
  2) ZeRO-3 reduction：partial boundary 触发全量 reduce + optimizer step，通信模式非预期
  3) scheduler/global_step：需与 HF phantom gs 同步改造，双轨计数更复杂
  4) 需 patch accelerate 与/或 DS（site-packages 或 vendored）→ 偏离上游、维护面大
- 判定：比 D 复杂一个量级；即使实现也改变 replay 段训练语义（partial 更新权重）——
  除非能证明更简单且自动化测试完备，否则不采纳（本轮按任务书不实现）

## 结论
D 为唯一同时满足：两个 ratio 同构（A 不满足）、真实更新 ≥2（B 不满足）、
不改研究问题（C 不满足）、不动训练核心代码（E 不满足）的方案。
批准实现 D；实机验证回答其 4 项风险（hang/真步口径/吞吐/NaN）。
