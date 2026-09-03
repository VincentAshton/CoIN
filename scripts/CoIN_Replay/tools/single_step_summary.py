#!/usr/bin/env python3
"""阶段 II 汇总：single-step replay 门禁证据收集与判定（仓库版，需在 coin env 下运行，需 torch）。

用法: python single_step_summary.py <gate_dir> <log_dir>
gate_dir: 含 data/{r2,r3}.json.manifest.json 与 ckpt/{task_sqa,replay_r2,replay_r3}/
log_dir:  汇总 json 输出目录
"""
import json
import os
import sys

GATE, LDIR = sys.argv[1], sys.argv[2]
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # scripts/CoIN_Replay/
from coin_lib import ckpt_tensor_compare, ckpt_validate, per_rank_micro_batches  # noqa: E402

GAS = int(os.environ.get("REPLAY_ACCUM", "16"))  # 2026-09-04 方案 D：1 = 每 micro boundary
CKPTS = ["task_sqa", "replay_r2", "replay_r3"]
out = {"scenarios": {}, "ckpts": {}, "gas": GAS}


def read_ts(ck):
    p = os.path.join(GATE, "ckpt", ck, "trainer_state.json")
    if not os.path.isfile(p):
        return {"error": f"no trainer_state at {p}"}
    ts = json.load(open(p))
    lr_seq = [h.get("lr") for h in ts.get("log_history", []) if "lr" in h]
    loss_seq = [h.get("loss") for h in ts.get("log_history", []) if "loss" in h]
    return {"global_step": ts.get("global_step"),
            "log_history_lr": lr_seq, "log_history_loss": loss_seq,
            "epoch": ts.get("epoch")}


for ck in CKPTS:
    rep = ckpt_validate(os.path.join(GATE, "ckpt", ck))
    out["ckpts"][ck] = {"param_hash": rep.get("param_hash"),
                        "finite": rep.get("finite"),
                        "files": sorted(os.listdir(os.path.join(GATE, "ckpt", ck)))}
    out["ckpts"][ck]["trainer_state"] = read_ts(ck)


def diff(a, b):
    rep = ckpt_tensor_compare(os.path.join(GATE, "ckpt", a), os.path.join(GATE, "ckpt", b))
    th = rep.get("tensor_hash") or {}
    fin = rep.get("finite") or {}
    return {"pass": rep.get("pass"), "changed": rep.get("changed_tensor_count"),
            "l2": rep.get("l2_norm_diff"), "max_abs": rep.get("max_abs_diff"),
            "hash_task": th.get("task"), "hash_replay": th.get("replay"),
            "hash_differs": th.get("differs"),
            "finite_task": fin.get("task"), "finite_replay": fin.get("replay"),
            "structural_error": rep.get("structural_error")}


out["scenarios"]["r2_N127"] = diff("task_sqa", "replay_r2")
out["scenarios"]["r3_N473"] = diff("task_sqa", "replay_r3")

for sc, path in [("r2_N127", os.path.join(GATE, "data", "r2.json.manifest.json")),
                 ("r3_N473", os.path.join(GATE, "data", "r3.json.manifest.json"))]:
    m = json.load(open(path))
    out["scenarios"][sc]["replay_N"] = m["output"]["N"]
    out["scenarios"][sc]["mode"] = m.get("mode")
    out["scenarios"][sc]["ratio"] = m.get("ratio")

ok = True
for sc in ["r2_N127", "r3_N473"]:
    s = out["scenarios"][sc]
    ckpt = "replay_r2" if sc == "r2_N127" else "replay_r3"
    gs = out["ckpts"][ckpt]["trainer_state"].get("global_step")
    N = s.get("replay_N")
    M = per_rank_micro_batches(N, 14, 4)[0] if N else None
    if GAS == 1:
        # 方案 D：gas1 下每 micro 即 boundary → HF gs 每 micro +1（无 phantom）= M = DS 真步
        exp_gs = M
        note = f"(gas1: DS 真步 = M = {M})"
    else:
        # 旧 gas16 No-Go 复现：HF phantom gs=1、DS 0 真步
        exp_gs = 1
        note = "(gas16: HF phantom gs=1, DS 0 真步)"
    checks = [
        ("global_step", gs == exp_gs, f"{gs} vs exp {exp_gs} {note}"),
        ("real_update", s["pass"] is True and (s["changed"] or 0) > 0
         and (s["l2"] or 0) > 0 and s["hash_differs"] is True,
         {k: s.get(k) for k in ("changed", "l2", "max_abs", "hash_differs")}),
        ("finite", s.get("finite_task") is True and s.get("finite_replay") is True,
         {"task": s.get("finite_task"), "replay": s.get("finite_replay")}),
    ]
    for name, passed, val in checks:
        print(f"[check] {sc} {name}: {'PASS' if passed else 'FAIL'} ({val})")
        if not passed:
            ok = False

summary_path = os.path.join(LDIR, "single_step_summary.json")
json.dump(out, open(summary_path, "w"), indent=1, ensure_ascii=False, default=str)
print("SUMMARY ->", summary_path)
print("GATE_PASS" if ok else "GATE_FAIL")
sys.exit(0 if ok else 1)
