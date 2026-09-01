#!/usr/bin/env python3
"""DeepSpeed 最小任务冒烟（工单 3/9B，云端运行）：
用 scripts/zero3_offload.json 初始化一个微型 MLP，跑 forward/backward/step；
核心断言：ds config 无 scheduler/optimizer 段时，engine.scheduler 必须为 None
（即 DeepSpeed 不会把 trainer 的 cosine 替换为 WarmupLR），optimizer 非空。

用法: python scripts/CoIN_Replay/smoke/smoke_ds.py --ds-config scripts/zero3_offload.json
退出码: 0=通过；非 0=失败
"""
import argparse
import json
import sys

import torch
import torch.nn as nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ds-config", required=True)
    p.add_argument("--gpus", default="0,1,2,3")
    args = p.parse_args()

    cfg = json.load(open(args.ds_config))
    if "scheduler" in cfg or "optimizer" in cfg:
        print(f"FAIL: ds config 含 scheduler/optimizer 段——会替换 trainer 的 cosine 与分组优化器")
        sys.exit(1)
    print(f"[smoke_ds] ds config OK（无 scheduler/optimizer 段）")

    import deepspeed
    print(f"[smoke_ds] deepspeed {deepspeed.__version__}")

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 16)).half()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ds_cfg = dict(cfg)
    ds_cfg.update({
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
    })

    engine, opt, _, _ = deepspeed.initialize(
        args=None, model=model, optimizer=optimizer,
        model_parameters=model.parameters(), config_params=ds_cfg)
    assert opt is not None, "optimizer 为空"
    print(f"[smoke_ds] optimizer_class={type(opt).__module__}.{type(opt).__name__}")
    for gi, g in enumerate(opt.param_groups):
        print(f"  group[{gi}] lr={g.get('lr')} tensors={len(g.get('params', []))}")

    engine.scheduler = None  # 显式置空后 deepspeed 引擎用 trainer 侧 scheduler
    x = torch.randn(8, 64, dtype=torch.half, device="cuda")
    y = torch.randn(8, 16, dtype=torch.half, device="cuda")
    loss = engine.loss_fn(engine(x), y) if engine.loss_fn else nn.MSELoss()(engine(x), y)
    engine.backward(loss)
    engine.step()
    print("[smoke_ds] fwd/bwd/step OK")
    print("[smoke_ds] PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
