#!/usr/bin/env python3
"""GPU 冒烟（工单 9B，云端 4×A100 上运行）：
1. NCCL all-reduce（4 rank）
2. flash-attn forward/backward（若可用）
3. 打印 torch/transformers/deepspeed/flash_attn 版本

用法: torchrun --nproc_per_node=4 scripts/CoIN_Replay/smoke/smoke_gpu.py
退出码: 0=通过；非 0=失败
"""
import os
import sys

import torch
import torch.distributed as dist

FAIL = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    print(f"[smoke_gpu] rank={rank} world={world} device={torch.cuda.get_device_name(rank)} "
          f"cap={torch.cuda.get_device_capability(rank)}", flush=True)

    # NCCL all-reduce
    dist.init_process_group(backend="nccl")
    t = torch.ones(256, device=dev)
    dist.all_reduce(t)
    check(torch.allclose(t, torch.full_like(t, world)), f"NCCL all-reduce OK (sum={t[0].item():.0f})")

    # flash-attn fwd/bwd
    try:
        import flash_attn
        from flash_attn import flash_attn_func
        q = torch.randn(2, 8, 64, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(2, 8, 64, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(2, 8, 64, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
        o = flash_attn_func(q, k, v)
        check(torch.isfinite(o).all().item(), f"flash-attn {flash_attn.__version__} fwd OK")
        o.sum().backward()
        # 梯度严格断言（评审 2026-09-02）：q/k/v grads 均存在、全部 finite、至少一个非零
        gq, gk, gv = q.grad, k.grad, v.grad
        grads_present = all(g is not None for g in (gq, gk, gv))
        grads_finite = grads_present and all(torch.isfinite(g).all().item() for g in (gq, gk, gv))
        nonzero = grads_present and any(bool((g != 0).any().item()) for g in (gq, gk, gv))
        check(grads_present, "q/k/v grads 均存在")
        check(grads_finite, "q/k/v grads 全部 finite")
        check(nonzero, "q/k/v 至少一个非零梯度")
        check(grads_present and grads_finite and nonzero,
              f"flash-attn {flash_attn.__version__} bwd OK")
    except Exception as e:
        check(False, f"flash-attn 不可用: {type(e).__name__}: {e}")

    # bf16 matmul sanity
    a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    b = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    check(torch.isfinite((a @ b)).all().item(), "bf16 matmul OK")

    dist.barrier()
    print(f"[smoke_gpu] rank={rank} 完成", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
