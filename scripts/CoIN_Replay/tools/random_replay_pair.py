#!/usr/bin/env python3
"""random-replay 同 seed 配对比较（ratio 通用；codex/coin-replay-random 分支）。

读两个（或多个）ratio 系列的注册表（index.json + runs/<run_id>/summary.json），
**按 replay sample seed 取交集**建立配对（绝不允许按 run_number 猜测配对），
生成：

  - paired_comparison.json —— 结构化配对结果（供机器读）
  - PAIRED_REPORT.md       —— 人读报告（含未配对 run 与限制说明）

配对差值统一定义为 **高 ratio − 低 ratio**（即 random-0.10 − random-0.01）：
  delta.MAA / delta.BWT / delta.final_avg = 高 ratio run − 低 ratio run

纪律（任务书五.10/五.11）：
  - 两个系列各自显示 n（n_total / COMPLETE / FAILED）；paired n 只算 seed 交集
  - 只在一个系列出现的 run 标记 UNPAIRED，绝不把未配对均值冒充配对比较
  - 全部 COMPLETE 与 FAILED run 都展示；禁止按指标隐藏/删除/重排 run
  - 任一 ratio 的 run 明确定位（run_id + seed + 两侧 result/code commit）

用法:
  python tools/random_replay_pair.py \
      --series docs/experiments/coin_replay_random_r001 \
      --series docs/experiments/coin_replay_random_r010 \
      --out-json docs/experiments/coin_replay_random/paired_comparison.json \
      --out-md   docs/experiments/coin_replay_random/PAIRED_REPORT.md
退出码: 0=生成成功（含"无配对"这种合法情形）；非零=结构性错误（缺 index/元数据非法）。
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from coin_lib import normalize_ratio, ratio_tag  # noqa: E402

METRICS = ("MAA", "BWT", "final_avg")
REQUIRED_META = ("ratio", "ratio_tag", "sample_mode", "result_directory_root")
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", "..", ".."))


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _rel(path, base):
    """证据字符串用相对路径：base 下 → relpath；否则 basename（禁绝对路径入库）。"""
    a, b = os.path.abspath(path), os.path.abspath(base)
    try:
        x = os.path.relpath(a, b)
    except ValueError:
        return os.path.basename(a)
    return x if not x.startswith("..") else os.path.basename(a)


def load_series(dirpath):
    """读一个系列：index.json 元数据 + runs + 每个 run 的 summary.json（可缺）。"""
    d = os.path.abspath(dirpath)
    ipath = os.path.join(d, "index.json")
    if not os.path.isfile(ipath):
        raise SystemExit(f"缺 {ipath}——不是系列注册表目录")
    idx = json.load(open(ipath, encoding="utf-8"))
    missing = [k for k in REQUIRED_META if idx.get(k) in (None, "")]
    if missing:
        raise SystemExit(f"{ipath} 缺系列元数据 {missing}")
    try:
        ratio = normalize_ratio(idx["ratio"])
    except ValueError as e:
        raise SystemExit(f"{ipath}: {e}")
    tag = ratio_tag(ratio)
    if idx["ratio_tag"] != tag:
        raise SystemExit(f"{ipath}: ratio_tag={idx['ratio_tag']!r} 与 ratio 派生的 "
                         f"{tag!r} 不一致")
    runs = []
    for r in idx["runs"]:  # 保持 index 内顺序（禁按指标重排）
        rid = r.get("run_id")
        s = None
        spath = os.path.join(d, "runs", str(rid), "summary.json")
        if os.path.isfile(spath):
            s = json.load(open(spath, encoding="utf-8"))
        runs.append({
            "series_dir": _rel(d, REPO_ROOT),
            "ratio": str(ratio),
            "ratio_tag": tag,
            "run_id": rid,
            "run_number": r.get("run_number"),
            "seed": r.get("replay_sample_seed"),
            "status": r.get("status"),
            "MAA": r.get("MAA"),
            "BWT": r.get("BWT"),
            "final_avg": (s or {}).get("final_avg"),
            "result_commit": r.get("result_commit"),
            "code_commit": r.get("code_commit") or (s or {}).get("code_commit"),
            "started_at": r.get("started_at"),
            "completed_at": r.get("completed_at"),
            "error_summary": r.get("error_summary"),
            "paired_with": r.get("paired_with"),
            "summary_path": _rel(spath, d) if s is not None else None,
        })
    complete = [r for r in runs if r["status"] == "COMPLETE"]
    failed = [r for r in runs if r["status"] == "FAILED"]
    meta = {
        "series_dir": _rel(d, REPO_ROOT),
        "index": os.path.join(_rel(d, REPO_ROOT), "index.json")
        if _rel(d, REPO_ROOT) != os.path.basename(d) else "index.json",
        "ratio": str(ratio),
        "ratio_tag": tag,
        "n_total": len(runs),
        "n_complete": len(complete),
        "n_failed": len(failed),
        "n_running": len([r for r in runs if r["status"] == "RUNNING"]),
        "runs": runs,
    }
    return meta


def metric_pair(hi, lo, key):
    a, b = hi.get(key), lo.get(key)
    if a is None or b is None:
        return None
    return round(float(a) - float(b), 4)


def build_pairing(series_list):
    """按 seed 交集配对；返回 (pairs, unpaired, ordering)。"""
    # 高 ratio 为被减数（方向 = 高 ratio − 低 ratio）
    ordered = sorted(series_list, key=lambda s: float(s["ratio"]))
    lo, hi = ordered[0], ordered[-1]
    if len(ordered) != 2:
        raise SystemExit(f"配对比较需要两个 ratio 系列，收到 {len(ordered)} 个")
    if float(lo["ratio"]) == float(hi["ratio"]):
        raise SystemExit("两个系列的 ratio 相同，无法做比例配对比较")

    hi_by_seed = {}
    for r in hi["runs"]:
        hi_by_seed.setdefault(r["seed"], []).append(r)
    lo_by_seed = {}
    for r in lo["runs"]:
        lo_by_seed.setdefault(r["seed"], []).append(r)

    pairs = []
    for seed in sorted(set(lo_by_seed) & set(hi_by_seed)):
        for a in lo_by_seed[seed]:
            for b in hi_by_seed[seed]:
                pairs.append({
                    "seed": seed,
                    "low_ratio": {"ratio": lo["ratio"], "ratio_tag": lo["ratio_tag"],
                                  "run_id": a["run_id"], "status": a["status"],
                                  "result_commit": a["result_commit"],
                                  "code_commit": a["code_commit"],
                                  "MAA": a["MAA"], "BWT": a["BWT"],
                                  "final_avg": a["final_avg"]},
                    "high_ratio": {"ratio": hi["ratio"], "ratio_tag": hi["ratio_tag"],
                                   "run_id": b["run_id"], "status": b["status"],
                                   "result_commit": b["result_commit"],
                                   "code_commit": b["code_commit"],
                                   "MAA": b["MAA"], "BWT": b["BWT"],
                                   "final_avg": b["final_avg"]},
                    "delta": {k: metric_pair(b, a, k) for k in METRICS},
                    "paired_complete": (a["status"] == "COMPLETE"
                                        and b["status"] == "COMPLETE"),
                })
    unpaired = []
    for s in series_list:
        other = hi if s is lo else lo
        other_seeds = {r["seed"] for r in other["runs"]}
        for r in s["runs"]:
            if r["seed"] not in other_seeds:
                unpaired.append({
                    "series": s["ratio_tag"], "ratio": s["ratio"],
                    "run_id": r["run_id"], "seed": r["seed"], "status": r["status"],
                    "MAA": r["MAA"], "BWT": r["BWT"], "final_avg": r["final_avg"],
                    "result_commit": r["result_commit"],
                    "code_commit": r["code_commit"],
                    "pairing_label": "UNPAIRED",
                })
    return lo, hi, pairs, unpaired


def paired_means(pairs, lo, hi):
    """仅用配对的 COMPLETE run 计算各 ratio 均值与差值（n_paired）。"""
    ok = [p for p in pairs if p["paired_complete"]]
    out = {"n_paired_complete": len(ok), "n_paired_all": len(pairs),
           "low_ratio_tag": lo["ratio_tag"], "high_ratio_tag": hi["ratio_tag"],
           "delta_definition": f"random-{float(hi['ratio']):.2f} − "
                               f"random-{float(lo['ratio']):.2f}",
           "metrics": {}}
    for k in METRICS:
        lo_vals = [p["low_ratio"][k] for p in ok if p["low_ratio"][k] is not None]
        hi_vals = [p["high_ratio"][k] for p in ok if p["high_ratio"][k] is not None]
        deltas = [p["delta"][k] for p in ok if p["delta"][k] is not None]
        if not deltas:
            out["metrics"][k] = {"low_ratio_mean": None, "high_ratio_mean": None,
                                 "mean_delta": None, "min_delta": None,
                                 "max_delta": None, "n": 0}
            continue
        out["metrics"][k] = {
            "low_ratio_mean": round(sum(lo_vals) / len(lo_vals), 4) if lo_vals else None,
            "high_ratio_mean": round(sum(hi_vals) / len(hi_vals), 4) if hi_vals else None,
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "min_delta": min(deltas), "max_delta": max(deltas),
            "n": len(deltas),
        }
    return out


def render_md(payload) -> str:
    p = payload
    lo, hi = p["series"][0], p["series"][1]
    lines = ["# Random Replay 同 seed 配对比较（ratio 对照组）", "",
             f"- 生成: {p['generated_at']}（`scripts/CoIN_Replay/tools/random_replay_pair.py`）",
             f"- 配对方式: **按 replay sample seed 取两个系列的交集**（不按 run_number 猜测配对）",
             f"- 配对差值定义: **{p['delta_definition']}**（高 ratio 为被减数）",
             f"- 配对 n（完整 COMPLETE 对）: **{p['paired_means']['n_paired_complete']}**；"
             f"seed 交集内 run 对（含未完成）: {p['paired_means']['n_paired_all']}",
             ""]
    lines += ["## 1. 两个 ratio 系列各自规模（不做配对合并）", "",
              "| ratio | ratio tag | 系列目录 | n_total | COMPLETE | FAILED | RUNNING |",
              "|---|---|---|---|---|---|---|"]
    for s in (lo, hi):
        lines.append(f"| {s['ratio']} | {s['ratio_tag']} | `{os.path.basename(s['series_dir'])}` | "
                     f"{s['n_total']} | {s['n_complete']} | {s['n_failed']} | {s['n_running']} |")
    lines += ["", "### 1.1 各系列全部 run（按注册顺序，含 FAILED / RUNNING，不做指标筛选）", "",
              "| ratio tag | run_id | seed | status | MAA | CoIN BWT | final_avg | result commit |",
              "|---|---|---|---|---|---|---|---|"]
    for s in (lo, hi):
        for r in s["runs"]:
            lines.append(f"| {s['ratio_tag']} | {r['run_id']} | {r['seed']} | {r['status']} | "
                         f"{_fmt(r['MAA'])} | {_fmt(r['BWT'])} | {_fmt(r['final_avg'])} | "
                         f"{_short(r['result_commit'])} |")
    lines += ["", "## 2. 同 seed 配对（seed 交集）", ""]
    if p["pairs"]:
        lines += ["| seed | " + f"{lo['ratio_tag']} run" + " | " + f"{hi['ratio_tag']} run"
                  + " | ΔMAA | ΔCoIN BWT | Δfinal_avg | 两边均 COMPLETE |",
                  "|---|---|---|---|---|---|---|"]
        for pr in p["pairs"]:
            lines.append(
                f"| {pr['seed']} | {pr['low_ratio']['run_id']} ({pr['low_ratio']['status']}) | "
                f"{pr['high_ratio']['run_id']} ({pr['high_ratio']['status']}) | "
                f"{_fmt(pr['delta']['MAA'])} | {_fmt(pr['delta']['BWT'])} | "
                f"{_fmt(pr['delta']['final_avg'])} | {pr['paired_complete']} |")
        lines += ["", "### 2.1 配对明细（run_id / commit / 指标）", "",
                  "| seed | 项 | run_id | status | MAA | CoIN BWT | final_avg | result commit | code commit |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for pr in p["pairs"]:
            for side in ("low_ratio", "high_ratio"):
                s = pr[side]
                lines.append(f"| {pr['seed']} | {s['ratio_tag']} | {s['run_id']} | {s['status']} | "
                             f"{_fmt(s['MAA'])} | {_fmt(s['BWT'])} | {_fmt(s['final_avg'])} | "
                             f"{_short(s['result_commit'])} | {_short(s['code_commit'])} |")
        lines += ["", "### 2.2 配对均值与差值（仅完整 COMPLETE 对）", "",
                  f"| 指标 | {lo['ratio_tag']} 均值 | {hi['ratio_tag']} 均值 | 均值差 | 差值范围 | n |",
                  "|---|---|---|---|---|---|"]
        for k in METRICS:
            m = p["paired_means"]["metrics"][k]
            if not m:
                lines.append(f"| {k} | — | — | — | — | 0 |")
                continue
            lines.append(f"| {k} | {m['low_ratio_mean']} | {m['high_ratio_mean']} | "
                         f"{m['mean_delta']} | [{m['min_delta']}, {m['max_delta']}] | {m['n']} |")
    else:
        lines += ["（当前无同 seed 配对）"]
    lines += ["", "## 3. 未配对 run（UNPAIRED）", ""]
    if p["unpaired"]:
        lines += ["| ratio tag | run_id | seed | status | MAA | CoIN BWT | final_avg |",
                  "|---|---|---|---|---|---|---|"]
        for u in p["unpaired"]:
            lines.append(f"| {u['series']} | {u['run_id']} | {u['seed']} | {u['status']} | "
                         f"{_fmt(u['MAA'])} | {_fmt(u['BWT'])} | {_fmt(u['final_avg'])} |")
        lines += ["", "未配对 run 只在单一 ratio 存在，其数值**不参与**配对均值/差值，"
                      "也不得当作配对结论使用。"]
    else:
        lines += ["（无未配对 run：两个系列 seed 完全重合）"]
    lines += ["", "## 4. 限制与口径", "",
              "1. 配对差值为 4 位显示精度；统计量由 `paired_comparison.json` 的逐对差值计算。",
              "2. n 为 seed 交集大小，属**描述性证据**，不作显著性推断。",
              "3. 固定训练 SEED 不等于逐 bit 确定性（CUDA / FlashAttention / TF32 / 分布式），"
              "差值不能唯一归因于 replay 比例。",
              "4. 同 seed 的 0.01 样本集是 0.10 的子集（嵌套），因此差值含「样本量」与"
              "「样本组成」两个无法完全分离的效应。",
              ""]
    return "\n".join(lines)


def _fmt(v):
    return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _short(h):
    return "—" if not h else str(h)[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", action="append", required=True,
                    help="系列目录（含 index.json），需给两个不同 ratio 的系列")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    series = [load_series(d) for d in args.series]
    lo, hi, pairs, unpaired = build_pairing(series)
    payload = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "pairing_rule": "seed_intersection",
        "delta_definition": f"random-{float(hi['ratio']):.2f} − random-{float(lo['ratio']):.2f}",
        "delta_definition_note": "高 ratio run − 低 ratio run；两 run 的 replay sample seed 相同",
        "series": [lo, hi],
        "pairs": pairs,
        "unpaired": unpaired,
        "paired_means": paired_means(pairs, lo, hi),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(render_md(payload))
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({"out_json": os.path.abspath(args.out_json),
                      "out_md": os.path.abspath(args.out_md),
                      "n_pairs": len(pairs), "n_unpaired": len(unpaired),
                      "paired_means": payload["paired_means"]["metrics"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
