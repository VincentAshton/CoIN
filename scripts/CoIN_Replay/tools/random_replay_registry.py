#!/usr/bin/env python3
"""random-replay 系列实验注册表维护工具（codex/coin-replay-random-r001 分支）。

维护三件套（同一目录）：
  - index.json    —— 权威结构化记录（append-only；RUNNING 可更新为 COMPLETE/FAILED）
  - index.csv     —— 同内容扁平表（供脚本/表格工具读取）
  - README.md     —— 两个标记块之间的状态表（<!-- registry-table:start/end -->）

纪律（任务书六）：
  - 记录只能追加或 RUNNING -> COMPLETE/FAILED；不得删除失败记录
  - 不得复用已用 seed / run_number（单调递增 run_0001, run_0002, ...）
  - 不得改已完成行的指标（complete 后再次 complete 即报错）

用法:
  python tools/random_replay_registry.py <docdir> register [--seed N] [--started-at ISO]
  python tools/random_replay_registry.py <docdir> complete --run-id ID --status COMPLETE|FAILED \\
        [--summary <runs/<run_id>/summary.json>] [--maa f] [--bwt f] [--error-summary TEXT] ...
  python tools/random_replay_registry.py <docdir> list
git add/commit/push 由调用方执行（工具只保证文件内容正确）。
"""
import argparse
import csv
import datetime
import json
import os
import secrets
import sys

INDEX_JSON = "index.json"
INDEX_CSV = "index.csv"
README_MD = "README.md"
TABLE_START = "<!-- registry-table:start -->"
TABLE_END = "<!-- registry-table:end -->"
CSV_FIELDS = ["run_number", "run_id", "replay_sample_seed", "status", "started_at",
              "completed_at", "MAA", "BWT", "result_directory", "code_commit",
              "data_revision", "model_config_hash", "config_hash", "error_summary",
              "registration_commit", "result_commit"]
ALLOWED_STATUS = ("RUNNING", "COMPLETE", "FAILED")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path, obj, indent=2):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_index(docdir):
    p = os.path.join(docdir, INDEX_JSON)
    if not os.path.isfile(p):
        raise SystemExit(f"缺 {p}——该目录不是本系列的注册表目录？")
    return json.load(open(p, encoding="utf-8"))


def write_index(docdir, idx):
    atomic_write(os.path.join(docdir, INDEX_JSON), idx)
    # CSV 同步重写
    with open(os.path.join(docdir, INDEX_CSV), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in idx["runs"]:
            w.writerow(r)
    # README 表格块重写
    rp = os.path.join(docdir, README_MD)
    text = open(rp, encoding="utf-8").read()
    if TABLE_START not in text or TABLE_END not in text:
        raise SystemExit(f"{rp} 缺少 registry-table 标记块，无法自动更新表格")
    head, tail = text.split(TABLE_START, 1)
    _, tail = tail.split(TABLE_END, 1)
    rows = ["| Run | Replay sample seed | MAA | BWT | Status | Result commit |",
            "|-----|--------------------|-----|-----|--------|---------------|"]
    for r in idx["runs"]:
        rows.append(f"| {r['run_id']} | {r['replay_sample_seed']} | "
                    f"{'' if r.get('MAA') is None else r['MAA']} | "
                    f"{'' if r.get('BWT') is None else r['BWT']} | {r['status']} | "
                    f"{r.get('result_commit') or ''} |")
    block = TABLE_START + "\n" + "\n".join(rows) + "\n" + TABLE_END
    atomic_write_text(rp, head + block + tail)


def cmd_register(docdir, args):
    idx = load_index(docdir)
    runs = idx["runs"]
    used_seeds = {r["replay_sample_seed"] for r in runs}
    run_number = (max((r["run_number"] for r in runs), default=0)) + 1
    seed = args.seed
    if seed is None:
        while True:
            cand = secrets.SystemRandom().randrange(1, 1 << 31)
            if cand not in used_seeds:
                seed = cand
                break
    if seed in used_seeds:
        raise SystemExit(f"seed {seed} 已在注册表中使用（禁止复用）")
    run_id = f"run_{run_number:04d}_seed_{seed}"
    if any(r["run_id"] == run_id for r in runs):
        raise SystemExit(f"{run_id} 已存在")
    entry = {
        "run_number": run_number,
        "run_id": run_id,
        "replay_sample_seed": seed,
        "status": "RUNNING",
        "started_at": args.started_at or now_iso(),
        "completed_at": None,
        "MAA": None,
        "BWT": None,
        "result_directory": f"results/CoIN_Replay_random/r001/{run_id}",
        "code_commit": None,
        "data_revision": None,
        "model_config_hash": None,
        "config_hash": None,
        "error_summary": None,
        "registration_commit": None,
        "result_commit": None,
    }
    runs.append(entry)
    write_index(docdir, idx)
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    print("推荐提交：")
    print(f"  git commit -m 'experiments(random-replay): register {run_id} "
          f"seed={seed}'")


def _fill_from_summary(entry, summary_path):
    s = json.load(open(summary_path, encoding="utf-8"))
    if s.get("run_id") != entry["run_id"]:
        raise SystemExit(f"summary run_id {s.get('run_id')} != 登记 {entry['run_id']}——"
                         "文件放错 run 目录？")
    if s.get("MAA") is None or s.get("BWT") is None:
        raise SystemExit("summary.json 缺 MAA/BWT（COMPLETE 必须先 assemble + 验收通过）")
    entry["MAA"] = s["MAA"]
    entry["BWT"] = s["BWT"]
    entry["result_directory"] = s.get("result_directory_rel") or entry["result_directory"]
    entry["code_commit"] = s.get("code_commit")
    entry["data_revision"] = s.get("data_revision")
    entry["model_config_hash"] = s.get("model_config_hash")
    entry["config_hash"] = s.get("config_hash")
    return s


def cmd_complete(docdir, args):
    idx = load_index(docdir)
    for r in idx["runs"]:
        if r["run_id"] != args.run_id:
            continue
        if r["status"] != "RUNNING":
            raise SystemExit(f"{args.run_id} 当前状态 {r['status']}——只有 RUNNING 可完结")
        r["completed_at"] = args.completed_at or now_iso()
        if args.status == "COMPLETE":
            if args.summary:
                _fill_from_summary(r, args.summary)
            if r["MAA"] is None:
                r["MAA"] = args.maa
            if r["BWT"] is None:
                r["BWT"] = args.bwt
            if r["MAA"] is None or r["BWT"] is None:
                raise SystemExit("COMPLETE 必须提供 MAA/BWT（--summary 或 --maa/--bwt）")
            r["error_summary"] = None
        else:  # FAILED
            if not (args.error_summary or "").strip():
                raise SystemExit("FAILED 必须提供 --error-summary（任务书十：保存脱敏错误摘要）")
            r["error_summary"] = args.error_summary
            r["MAA"] = None
            r["BWT"] = None
        r["registration_commit"] = args.registration_commit or r["registration_commit"]
        r["result_commit"] = args.result_commit or r["result_commit"]
        r["status"] = args.status
        if args.code_commit:
            r["code_commit"] = args.code_commit
        write_index(docdir, idx)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print("推荐提交：")
        print(f"  git commit -m 'results(random-replay): complete {args.run_id} "
              f"seed={r['replay_sample_seed']} MAA={r['MAA']} BWT={r['BWT']}'")
        return
    raise SystemExit(f"注册表无 {args.run_id}")


def cmd_list(docdir, args):
    idx = load_index(docdir)
    for r in idx["runs"]:
        print(json.dumps(r, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("docdir",
                    help="docs/experiments/coin_replay_random_r001 目录（含 index.json）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_reg = sub.add_parser("register")
    p_reg.add_argument("--seed", type=int, default=None, help="固定 seed（默认 SystemRandom 31-bit）")
    p_reg.add_argument("--started-at", default=None)
    p_reg.set_defaults(fn=cmd_register)
    p_com = sub.add_parser("complete")
    p_com.add_argument("--run-id", required=True)
    p_com.add_argument("--status", choices=ALLOWED_STATUS, required=True)
    p_com.add_argument("--summary", default=None)
    p_com.add_argument("--maa", type=float, default=None)
    p_com.add_argument("--bwt", type=float, default=None)
    p_com.add_argument("--error-summary", default=None)
    p_com.add_argument("--code-commit", default=None)
    p_com.add_argument("--registration-commit", default=None)
    p_com.add_argument("--result-commit", default=None)
    p_com.add_argument("--completed-at", default=None)
    p_com.set_defaults(fn=cmd_complete)
    p_ls = sub.add_parser("list")
    p_ls.set_defaults(fn=cmd_list)
    args = ap.parse_args()
    args.fn(os.path.abspath(args.docdir), args)


if __name__ == "__main__":
    main()
