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
from coin_lib import ckpt_tensor_compare, ckpt_validate  # noqa: E402

CKPTS = ["task_sqa", "replay_r2", "replay_r3"]
out = {"scenarios": {}, "ckpts": {}}


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
                        "all_finite": rep.get("all_finite"),
                        "files": sorted(os.listdir(os.path.join(GATE, "ckpt", ck)))}
    out["ckpts"][ck]["trainer_state"] = read_ts(ck)


def diff(a, b):
    rep = ckpt_tensor_compare(os.path.join(GATE, "ckpt", a), os.path.join(GATE, "ckpt", b))
    return {"pass": rep.get("pass"), "changed": rep.get("changed_tensor_count"),
            "l2": rep.get("l2_norm_diff"), "max_abs": rep.get("max_abs_diff"),
            "hash_a": rep.get("hash_a") or rep.get("hash_task"),
            "hash_b": rep.get("hash_b") or rep.get("hash_replay"),
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
    checks = [
        ("global_step", gs == 1, gs),
        ("real_update", s["pass"] is True and (s["changed"] or 0) > 0
         and (s["l2"] or 0) > 0 and s["hash_a"] != s["hash_b"], s),
        ("finite", out["ckpts"][ckpt]["all_finite"] is True,
         out["ckpts"][ckpt]["all_finite"]),
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
