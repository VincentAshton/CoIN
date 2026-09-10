#!/usr/bin/env python3
"""matrix_static_sim.py — CoIN+Replay per-rank micro / HF gs / DS 真步 静态模拟器。
精确移植 accelerate 0.21 BatchSamplerShard._iter_with_no_split(even_batches=True,
drop_last=False) + transformers 4.32 步进语义 + deepspeed 0.14 boundary 语义。
用法: python3 matrix_static_sim.py [--csv out.csv]  （默认打印汇总 + 写 matrix_static.csv）
"""
import csv, sys, math

BATCH, WORLD = 14, 4

def batches(N):
    B = -(-N // BATCH)
    return [list(range(i * BATCH, min((i + 1) * BATCH, N))) for i in range(B)]

def per_rank_micro(N):
    bs = batches(N)
    counts = []
    for p in range(WORLD):
        cnt, initial_data, batch_to_yield = 0, [], []
        for idx, batch in enumerate(bs):
            if idx < WORLD:
                initial_data += batch
            if idx % WORLD == p:
                batch_to_yield = batch
            if idx % WORLD == WORLD - 1 and len(batch) == BATCH:
                cnt += 1
                batch_to_yield = []
        if len(initial_data) > 0:
            if len(batch_to_yield) == BATCH:
                cnt += 1
            last = bs[-1]
            batch = [] if len(last) == BATCH else list(last)
            idx2 = len(bs) if len(last) == BATCH else len(bs) - 1
            cycle = 0
            while idx2 % WORLD != 0 or len(batch) > 0:
                while cycle + (BATCH - len(batch)) > len(initial_data):
                    initial_data += initial_data
                need = BATCH - len(batch)
                batch += initial_data[cycle:cycle + need]
                if idx2 % WORLD == p:
                    cnt += 1
                cycle += need
                batch = []
                idx2 += 1
        counts.append(cnt)
    return counts

def hf_gs(M, gas):
    if M <= 0:
        return 0
    full = M // gas
    if M % gas == 0:
        return full
    if M <= gas:
        return full + 1  # phantom: is_last_step_and_steps_less_than_grad_acc
    return full

def lr_seq(gas, Mmin, lr=2e-4, warmup_ratio=0.03):
    T = max(1, -(-Mmin // gas))
    warm = int(warmup_ratio * T)
    seq = []
    for gs in range(1, Mmin // gas + 1):
        s = gs - 1
        if warm and s < warm:
            l = lr * (s + 1) / warm
        else:
            prog = (s - warm) / max(1, T - warm)
            l = lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        seq.append(round(l, 12))
    return seq, warm, T

def compute(N, gas, tag):
    micro = per_rank_micro(N)
    mn, mx = min(micro), max(micro)
    ds = mn // gas
    hf = max(hf_gs(m, gas) for m in micro)
    seq, warm, T = lr_seq(gas, mn)
    return dict(tag=tag, N=N, gas=gas, per_rank=str(micro), uneven=mx - mn,
                hf_gs=hf, ds_updates=ds, warmup=warm, hf_T=T,
                lr_first=seq[0] if seq else None, lr_last=seq[-1] if seq else None,
                discarded=max(0, mn - ds * gas))

def main():
    rows = []
    for name, N in [("task SQA", 12726), ("task TextVQA", 34602),
                    ("task ImageNet", 129833), ("task GQA", 72140)]:
        rows.append(compute(N, 16, name))
    for name, N in [("replay 0.01 r2", 127), ("replay 0.01 r3", 473),
                    ("replay 0.01 r4", 1771), ("replay 0.10 r2", 1272),
                    ("replay 0.10 r3", 4732), ("replay 0.10 r4", 17715)]:
        for gas in (16, 1):
            rows.append(compute(N, gas, f"{name} gas{gas}"))
    for N in (1, 55, 56, 57, 111, 112, 127, 473, 895, 896, 897, 1272, 4732, 17715):
        for gas in (1, 8, 16):
            rows.append(compute(N, gas, f"boundary N={N} gas{gas}"))
    out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else "matrix_static.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['tag']:<26} N={r['N']:>7} gas={r['gas']:<3} micro={r['per_rank']:<17} "
              f"unev={r['uneven']} hf_gs={r['hf_gs']:<3} ds_upd={r['ds_updates']:<3} "
              f"warm={r['warmup']} disc={r['discarded']} lr={r['lr_first']}..{r['lr_last']}")
    print(f"rows={len(rows)} -> {out}")

if __name__ == "__main__":
    main()
