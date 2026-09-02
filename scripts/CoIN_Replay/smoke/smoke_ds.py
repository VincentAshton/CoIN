#!/usr/bin/env python3
"""DeepSpeed 分布式最小任务冒烟（工单 9B 修复版，评审 2026-09-02 批准）。

修复背景：原版以单进程直接 deepspeed.initialize()，无 RANK/WORLD_SIZE 环境 →
deepspeed 0.14 走 MPI 探测路径（缺 mpi4py）。现由 canary.sh 以
`torchrun --standalone --nproc_per_node=4` 启动（与正式四卡分布式路径一致），
deepspeed 经 torchrun 环境变量走 env:// NCCL 初始化，不再依赖 MPI。

验证（每 rank 局部检查 + all_reduce 全局汇总）：
  1. WORLD_SIZE==4 断言；LOCAL_RANK 选择对应 GPU（每 rank 不同卡）
  2. 正式 ds config（zero3_offload.json）无 scheduler/optimizer 段
  3. engine.optimizer 非空；engine.scheduler 显式置空（trainer 侧 cosine 生效前提）
  4. 真实 forward/backward/optimizer step（AdamW lr=2e-4，bf16 与正式配置一致）
  5. loss finite；step 后全部参数（GatheredParameters 公共 API gather）finite
  6. step 后至少一个参数发生非零变化
  7. 全部 rank 成功才 exit 0；finally 清理 process group

用法: torchrun --standalone --nproc_per_node=4 scripts/CoIN_Replay/smoke/smoke_ds.py \\
      --ds-config scripts/zero3_offload.json
退出码: 0=通过；非 0=失败
"""
import argparse
import json
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ds-config", required=True)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world = int(os.environ.get("WORLD_SIZE", "0"))
    if rank < 0 or local_rank < 0:
        raise SystemExit("smoke_ds 必须由 torchrun --standalone 启动（缺 RANK/LOCAL_RANK）")
    if world != 4:
        raise SystemExit(f"smoke_ds 要求 WORLD_SIZE==4（当前 world={world}）——与正式 4 卡路径一致")

    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    ok = True
    try:
        import deepspeed
        from deepspeed.runtime.zero import GatheredParameters  # deepspeed 0.14 无 deepspeed.zero

        cfg = json.load(open(args.ds_config))
        if "scheduler" in cfg or "optimizer" in cfg:
            print(f"[smoke_ds:{rank}] FAIL: ds config 含 scheduler/optimizer 段"
                  f"（会替换 trainer 的 cosine 与分组优化器）", flush=True)
            ok = False

        print(f"[smoke_ds:{rank}] rank={rank} local_rank={local_rank} "
              f"gpu={torch.cuda.current_device()} name={torch.cuda.get_device_name(local_rank)} "
              f"world={world} deepspeed={deepspeed.__version__}", flush=True)

        torch.manual_seed(0)
        # 正式配置为 bf16+tf32：模型必须 bfloat16（fp16 会触发
        # "bfloat16 and fp16 modes cannot be simultaneously enabled" 断言）
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 16)).to(
            device=dev, dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

        ds_cfg = dict(cfg)
        # 正式路径由 HF trainer（--bf16 True）解析 fp16/bf16 的 "auto"：
        # fp16 off + bf16 on。直接 deepspeed.initialize(args=None) 时 auto 无从解析
        # （双双解析为 True -> "bfloat16 and fp16 modes cannot be simultaneously enabled"）。
        # 这里显式对齐 trainer 语义，config 文件本身不改。
        ds_cfg.setdefault("fp16", {}).update({"enabled": False})
        ds_cfg.setdefault("bf16", {}).update({"enabled": True})
        # 客户端自定义优化器 + ZeRO-Offload：HF trainer 集成也会置 False
        # （transformers/deepspeed.py 检测 trainer 自建 optimizer 时设置），对齐正式路径。
        ds_cfg["zero_force_ds_cpu_optimizer"] = False
        # gradient_clipping "auto" 由 HF trainer(max_grad_norm=1.0) 解析；args=None 下
        # 保持 "auto" 字符串 -> engine.step() 里 "auto" > 0.0 抛 TypeError。显式 1.0 对齐。
        ds_cfg["gradient_clipping"] = 1.0
        ds_cfg.update({
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 1,
        })
        engine, opt, _, _ = deepspeed.initialize(
            args=None, model=model, optimizer=optimizer,
            model_parameters=model.parameters(), config_params=ds_cfg)
        if opt is None:
            print(f"[smoke_ds:{rank}] FAIL: optimizer 为空", flush=True)
            ok = False
        engine.scheduler = None  # 显式置空：DeepSpeed 引擎用 trainer 侧 scheduler

        # 参数快照：ZeRO-3 下模型参数被分区/占位，用公共 API GatheredParameters
        # gather 到本 rank（与 zero_to_fp32/官方 recipe 同机制）
        params = list(model.parameters())

        def snapshot():
            with GatheredParameters(params):
                return [pp.detach().to("cpu", dtype=torch.float32).clone()
                        for pp in params]

        before = snapshot()

        x = torch.randn(8, 64, dtype=torch.bfloat16, device=dev)
        y = torch.randn(8, 16, dtype=torch.bfloat16, device=dev)
        criterion = nn.MSELoss()
        loss = criterion(engine(x), y)
        engine.backward(loss)
        engine.step()
        after = snapshot()

        loss_finite = bool(torch.isfinite(loss).item())
        params_finite = bool(after and all(torch.isfinite(a).all().item() for a in after))
        changed = bool(after and any(((b - a).abs().max() > 0).item()
                                     for b, a in zip(before, after)))
        print(f"[smoke_ds:{rank}] loss={loss.item():.6f} loss_finite={loss_finite} "
              f"params_finite={params_finite} n_params={len(after)} "
              f"param_changed={changed}", flush=True)
        if not (loss_finite and params_finite and changed):
            ok = False

        # 全局汇总：所有 rank 都通过才算 PASS
        ok_t = torch.tensor([1.0 if ok else 0.0], device=dev)
        dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
        ok = bool(ok_t.item() == 1.0)
        print(f"[smoke_ds:{rank}] 全局汇总: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
