#!/usr/bin/env python3
"""random replay 系列 一致性 + 脱敏 自查工具（push 前必须通过）。

设计依据 = 仓库文档权威层级（避免同一组数字在多处漂移）：

    权威机器数据   <series>/index.json（+ runs/<run_id>/summary.json）、
                   coin_replay_random/paired_comparison.json
    自动生成展示   <series>/README.md 状态行与 registry 表、PAIRED_REPORT.md
    人工说明       coin_replay_random/README.md（总入口）、SERIES_REPORT / EXECUTION_REPORT
    默认分支 README   只做稳定导航，不复制指标（漂移源头，禁止写数字）

检查项
  1. 系列 README 的「N 个 COMPLETE、M 个 RUNNING」== index.json 计数
  2. 系列 README 的 registry 表逐行 == index.json runs（id/seed/MAA/BWT/status/result commit）
  3. index.csv 行数与 run_id 集合 == index.json
  4. 总入口 README：各系列行的计数 / 最新 COMPLETE 指标 / 配对差值 / prefix 基线 == index.json
  5. paired_comparison.json == 由两个 index 重算（seed 交集；高 ratio − 低 ratio）
  6. PAIRED_REPORT.md 的配对 n 与差值 == paired_comparison.json
  7. 相对链接可解析（扫描到的 .md）
  8. 脱敏扫描（绝对路径 / 主机 / IP:port / 凭据词）——legacy 例外清单只警告，--strict 时失败

用法
  python3 tools/random_replay_sync_check.py                # 全查，打印报告
  python3 tools/random_replay_sync_check.py --json out.json
  python3 tools/random_replay_sync_check.py --strict        # legacy 例外也算失败
退出码：0 全通过 / 1 有 FAIL（FAIL 明细在报告的 fails 段）
"""
import argparse
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
import random_replay_pair as pair_tool  # noqa: E402

# legacy 例外：已发布的历史审核文档，原文保留（不改写历史记录）；--strict 时也报 FAIL
LEGACY_SANITIZE_EXEMPT = (
    "docs/experiments/coin_replay_random_r001/PRELAUNCH_AUDIT.md",
    "docs/experiments/coin_replay_random_r001/PRELAUNCH_AUDIT.json",
    "docs/experiments/coin_replay_random_r001/SERIES_REPORT.md",
)

SANITIZE_PATTERNS = (
    ("绝对云端路径 /root/", re.compile(r"/root/")),
    ("绝对路径 /home/ 或 /mnt/", re.compile(r"/(?:home|mnt)/")),
    ("Windows 盘符路径", re.compile(r"[A-Za-z]:\\\\")),
    ("云主机名/端口", re.compile(r"ssh-cn-|ebcloud")),
    ("IP 地址", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("凭据相关", re.compile(r"askpass|password|passwd|BEGIN [A-Z ]*PRIVATE KEY")),
)

COUNT_RE = re.compile(r"(\d+)\s*个\s*COMPLETE\s*[、，,]\s*(\d+)\s*个\s*RUNNING")
NUM = r"([+\-−]?\s*[\d.]+)"
ENTRY_METRIC_RE = re.compile(
    r"MAA\s+" + NUM + r"\s*/\s*CoIN\s+BWT\s+" + NUM + r"\s*/\s*final_avg\s+" + NUM)
ENTRY_PAIRED_RE = re.compile(
    r"ΔMAA\s+" + NUM + r"\s*/\s*ΔBWT\s+\*{0,2}" + NUM + r"\*{0,2}\s*/\s*Δfinal_avg\s+" + NUM)
BASELINE_RE = re.compile(
    r"prefix\s+ratio=(0\.\d+)[^\n]*?MAA\s+" + NUM + r"\s*/\s*CoIN\s+BWT\s+" + NUM +
    r"\s*/\s*final_avg\s+" + NUM)
PAIRED_N_RE = re.compile(r"配对\s*n（完整 COMPLETE 对）\s*[:：]\s*\*{0,2}(\d+)")


def num(s):
    return float(s.replace("−", "-").replace(" ", ""))


def close(a, b, tol):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


class Report:
    def __init__(self):
        self.checks = []
        self.fails = []
        self.warns = []

    def ok(self, name, detail=""):
        self.checks.append({"item": name, "result": "PASS", "detail": detail})

    def fail(self, name, detail):
        self.checks.append({"item": name, "result": "FAIL", "detail": detail})
        self.fails.append(f"{name}: {detail}")

    def warn(self, name, detail):
        self.checks.append({"item": name, "result": "WARN", "detail": detail})
        self.warns.append(f"{name}: {detail}")


def series_dirs(repo_root, given):
    if given:
        return [os.path.abspath(d) for d in given]
    base = os.path.join(repo_root, "docs", "experiments")
    return sorted(os.path.join(base, d) for d in os.listdir(base)
                  if d.startswith("coin_replay_random_") and
                  os.path.isfile(os.path.join(base, d, "index.json")))


# --------------------------------------------------------------------------- #
# 1-3. 系列目录内部一致性
# --------------------------------------------------------------------------- #
def check_series(sdir, repo_root, rep):
    tag = os.path.basename(sdir)
    idx = json.load(open(os.path.join(sdir, "index.json"), encoding="utf-8"))
    runs = idx.get("runs", [])
    n_comp = sum(1 for r in runs if r.get("status") == "COMPLETE")
    n_run = sum(1 for r in runs if r.get("status") == "RUNNING")
    readme_path = os.path.join(sdir, "README.md")
    readme = read(readme_path)

    m = COUNT_RE.search(readme)
    if not m:
        rep.fail(f"{tag} README 状态行", "未找到「N 个 COMPLETE、M 个 RUNNING」")
    elif (int(m.group(1)), int(m.group(2))) != (n_comp, n_run):
        rep.fail(f"{tag} README 状态行",
                 f"README={m.group(1)} COMPLETE/{m.group(2)} RUNNING，index={n_comp}/{n_run}")
    else:
        rep.ok(f"{tag} README 状态行", f"{n_comp} COMPLETE / {n_run} RUNNING")

    # registry 表
    rows = re.findall(r"^\|\s*(run_\d+_seed_\d+)\s*\|([^\n]*)$", readme, re.M)
    if len(rows) != len(runs):
        rep.fail(f"{tag} registry 表行数", f"README {len(rows)} 行 vs index {len(runs)} run")
    else:
        bad = []
        by_id = {r["run_id"]: r for r in runs}
        for rid, rest in rows:
            r = by_id.get(rid)
            if r is None:
                bad.append(f"{rid} 不在 index")
                continue
            cells = [c.strip() for c in rest.split("|")]
            if len(cells) < 4:
                bad.append(f"{rid} 表列过少: {cells}")
                continue
            seed_s, maa_s, bwt_s, status_s = cells[0], cells[1], cells[2], cells[3]
            commit_s = cells[4] if len(cells) > 4 else ""
            if str(r.get("replay_sample_seed")) != seed_s:
                bad.append(f"{rid} seed {seed_s}≠{r.get('replay_sample_seed')}")
            if not close(num(maa_s), r.get("MAA"), 5e-4):
                bad.append(f"{rid} MAA {maa_s}≠{r.get('MAA')}")
            if not close(num(bwt_s), r.get("BWT"), 5e-4):
                bad.append(f"{rid} BWT {bwt_s}≠{r.get('BWT')}")
            if status_s != r.get("status"):
                bad.append(f"{rid} status {status_s}≠{r.get('status')}")
            rc = r.get("result_commit") or ""
            if rc and commit_s and not rc.startswith(commit_s):
                bad.append(f"{rid} result_commit {commit_s}≠{rc}")
        if bad:
            rep.fail(f"{tag} registry 表内容", "; ".join(bad))
        else:
            rep.ok(f"{tag} registry 表内容", f"{len(rows)} 行与 index 一致")

    # csv
    csv_path = os.path.join(sdir, "index.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, encoding="utf-8", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        if len(csv_rows) != len(runs) or \
                {c["run_id"] for c in csv_rows} != {r["run_id"] for r in runs}:
            rep.fail(f"{tag} index.csv", f"{len(csv_rows)} 行 vs index {len(runs)} run")
        else:
            mism = [c["run_id"] for c, r in zip(csv_rows, runs)
                    if c["run_id"] != r["run_id"] or c["status"] != r["status"]
                    or not close(num(c["MAA"] or "nan"), r.get("MAA"), 5e-4)]
            if mism:
                rep.fail(f"{tag} index.csv 内容", f"与 index 不一致: {mism}")
            else:
                rep.ok(f"{tag} index.csv", f"{len(csv_rows)} 行与 index 一致")
    return idx


def latest_complete_final_avg(sdir, idx):
    """最新 COMPLETE run 的 final_avg（读 runs/<run_id>/summary.json；缺失返回 None）。"""
    for r in reversed(idx.get("runs", [])):
        if r.get("status") != "COMPLETE":
            continue
        p = os.path.join(sdir, "runs", r["run_id"], "summary.json")
        if os.path.isfile(p):
            return json.load(open(p, encoding="utf-8")), r
        return None, r
    return None, None


# --------------------------------------------------------------------------- #
# 4. 总入口 README
# --------------------------------------------------------------------------- #
def check_entry(entry_dir, repo_root, series, rep, strict=False):
    path = os.path.join(entry_dir, "README.md")
    text = read(path)
    idx_by_tag = {os.path.basename(d): json.load(open(os.path.join(d, "index.json"),
                                                      encoding="utf-8")) for d in series}
    # 各系列行
    for line in text.splitlines():
        if "coin_replay_random_r" not in line or not line.strip().startswith("|"):
            continue
        m_tag = re.search(r"coin_replay_random_(r\d+)", line)
        tag_short = m_tag.group(1) if m_tag else "?"
        tag_dir = f"coin_replay_random_{tag_short}" if m_tag else None
        idx = idx_by_tag.get(tag_dir) if tag_dir else None
        if not idx:
            continue
        runs = idx.get("runs", [])
        n_comp = sum(1 for r in runs if r.get("status") == "COMPLETE")
        n_run = sum(1 for r in runs if r.get("status") == "RUNNING")
        mc = COUNT_RE.search(line)
        if not mc:
            rep.fail(f"总入口 {tag_short} 行计数", "该行没有「N 个 COMPLETE、M 个 RUNNING」")
        elif (int(mc.group(1)), int(mc.group(2))) != (n_comp, n_run):
            rep.fail(f"总入口 {tag_short} 行计数",
                     f"README={mc.group(1)}/{mc.group(2)}，index={n_comp}/{n_run}")
        else:
            rep.ok(f"总入口 {tag_short} 行计数", f"{n_comp}/{n_run}")
        mm = ENTRY_METRIC_RE.search(line)
        if mm:
            summary, run = latest_complete_final_avg(
                os.path.join(repo_root, "docs", "experiments", tag_dir), idx)
            if run is None:
                rep.fail(f"总入口 {tag_short} 指标", "index 里没有 COMPLETE run")
                continue
            fa = (summary or {}).get("final_avg")
            bad = []
            if not close(num(mm.group(1)), run.get("MAA"), 5e-4):
                bad.append(f"MAA {mm.group(1)}≠{run.get('MAA')}")
            if not close(num(mm.group(2)), run.get("BWT"), 5e-4):
                bad.append(f"BWT {mm.group(2)}≠{run.get('BWT')}")
            if fa is not None and not close(num(mm.group(3)), fa, 5e-4):
                bad.append(f"final_avg {mm.group(3)}≠{fa}")
            if bad:
                rep.fail(f"总入口 {tag_short} 指标", "; ".join(bad))
            else:
                rep.ok(f"总入口 {tag_short} 指标",
                       f"与 index/{run['run_id']} 一致")

    # 配对行
    pj_path = os.path.join(entry_dir, "paired_comparison.json")
    if os.path.isfile(pj_path):
        pj = json.load(open(pj_path, encoding="utf-8"))
        means = pj["paired_means"]["metrics"]
        for line in text.splitlines():
            if "paired_comparison.json" not in line or "ΔMAA" not in line:
                continue
            mp = ENTRY_PAIRED_RE.search(line)
            if not mp:
                rep.fail("总入口 配对行差值", "行内未解析到 ΔMAA/ΔBWT/Δfinal_avg")
                continue
            bad = []
            for key, got in zip(("MAA", "BWT", "final_avg"), mp.groups()):
                want = means[key]["mean_delta"]
                if want is None or not close(num(got), want, 5e-4):
                    bad.append(f"Δ{key} {got}≠{want}")
            npair = re.search(r"当前\s*(\d+)\s*对", line)
            if npair and pj["paired_means"]["n_paired_complete"] is not None and \
                    int(npair.group(1)) != pj["paired_means"]["n_paired_complete"]:
                bad.append(f"对数 {npair.group(1)}≠{pj['paired_means']['n_paired_complete']}")
            if bad:
                rep.fail("总入口 配对行差值", "; ".join(bad))
            else:
                rep.ok("总入口 配对行差值", "与 paired_comparison.json 一致")
        # prefix 基线
        base_by_label = {}
        for idx in idx_by_tag.values():
            for k, b in (idx.get("baselines") or {}).items():
                base_by_label[b.get("label", k)] = b
        for m in BASELINE_RE.finditer(text):
            ratio = m.group(1)
            cand = [b for b in base_by_label.values() if f"ratio={ratio}" in (b.get("label") or "")]
            if not cand:
                rep.warn("总入口 prefix 基线", f"index 无 ratio={ratio} 基线，跳过比对")
                continue
            b = cand[0]
            bad = []
            if not close(num(m.group(2)), b.get("MAA"), 5e-4):
                bad.append(f"MAA {m.group(2)}≠{b.get('MAA')}")
            if not close(num(m.group(3)), b.get("BWT"), 5e-4):
                bad.append(f"BWT {m.group(3)}≠{b.get('BWT')}")
            if not close(num(m.group(4)), b.get("final_avg"), 5e-4):
                bad.append(f"final_avg {m.group(4)}≠{b.get('final_avg')}")
            if bad:
                rep.fail(f"总入口 prefix {ratio} 基线", "; ".join(bad))
            else:
                rep.ok(f"总入口 prefix {ratio} 基线", "与 index baselines 一致")


# --------------------------------------------------------------------------- #
# 5-6. 配对产物
# --------------------------------------------------------------------------- #
def check_pairing(entry_dir, series, rep):
    pj_path = os.path.join(entry_dir, "paired_comparison.json")
    md_path = os.path.join(entry_dir, "PAIRED_REPORT.md")
    if not (os.path.isfile(pj_path) and os.path.isfile(md_path)):
        rep.warn("配对产物", "缺少 paired_comparison.json / PAIRED_REPORT.md，跳过")
        return
    pj = json.load(open(pj_path, encoding="utf-8"))
    loaded = [pair_tool.load_series(d) for d in series]
    lo, hi, pairs, unpaired = pair_tool.build_pairing(loaded)
    means = pair_tool.paired_means(pairs, lo, hi)

    bad = []
    if len(pairs) != len(pj.get("pairs", [])):
        bad.append(f"配对数 {len(pairs)}≠{len(pj.get('pairs', []))}")
    pj_pairs = {p["seed"]: p for p in pj.get("pairs", [])}
    for pr in pairs:
        fp = pj_pairs.get(pr["seed"])
        if fp is None:
            bad.append(f"seed {pr['seed']} 缺失于 paired_comparison.json")
            continue
        for k in ("MAA", "BWT", "final_avg"):
            if not close(pr["delta"][k], fp["delta"].get(k), 5e-4):
                bad.append(f"seed {pr['seed']} Δ{k} {pr['delta'][k]}≠{fp['delta'].get(k)}")
    for k in ("MAA", "BWT", "final_avg"):
        want, got = means["metrics"][k]["mean_delta"], pj["paired_means"]["metrics"][k]["mean_delta"]
        if (want is None) != (got is None) or (want is not None and not close(got, want, 5e-4)):
            bad.append(f"配对均值 Δ{k} {got}≠{want}")
    if bad:
        rep.fail("paired_comparison.json 重算一致", "; ".join(bad))
    else:
        rep.ok("paired_comparison.json 重算一致",
               f"{len(pairs)} 对，未配对 {len(unpaired)}")

    md = read(md_path)
    m = PAIRED_N_RE.search(md)
    if not m:
        rep.fail("PAIRED_REPORT.md 配对数", "未解析到「配对 n（完整 COMPLETE 对）」")
    elif int(m.group(1)) != pj["paired_means"]["n_paired_complete"] or \
            int(m.group(1)) != len([p for p in pairs if p["paired_complete"]]):
        rep.fail("PAIRED_REPORT.md 配对数",
                 f"report={m.group(1)}，index 重算={len([p for p in pairs if p['paired_complete']])}")
    else:
        rep.ok("PAIRED_REPORT.md 配对数", f"{m.group(1)}")
    miss = []
    for key, glyph in (("MAA", "ΔMAA"), ("BWT", "ΔCoIN BWT"), ("final_avg", "Δfinal_avg")):
        for pr in pairs:
            v = pr["delta"][key]
            if v is None:
                continue
            want = f"{v:.4f}"
            if want not in md.replace("−", "-") and want.lstrip("-") not in md.replace("−", "-"):
                miss.append(f"{glyph}={want}")
    if miss:
        rep.fail("PAIRED_REPORT.md 差值", "未找到重算值: " + ", ".join(sorted(set(miss))[:6]))
    else:
        rep.ok("PAIRED_REPORT.md 差值", "覆盖全部重算差值")


# --------------------------------------------------------------------------- #
# 7-8. 链接 + 脱敏
# --------------------------------------------------------------------------- #
def iter_docs(dirs):
    for d in dirs:
        for root, _dirs, files in os.walk(d):
            if "/.git" in root:
                continue
            for f in files:
                if f.endswith((".md", ".json", ".csv")):
                    yield os.path.join(root, f)


def check_links_and_sanitize(paths, repo_root, rep, strict=False):
    link_re = re.compile(r"\]\(([^)\s]+)\)")
    n_links = n_scanned = 0
    bad_links, hits, legacy_hits = [], [], []
    for p in sorted(set(paths)):
        rel = os.path.relpath(p, repo_root)
        try:
            txt = read(p)
        except OSError:
            continue
        n_scanned += 1
        if p.endswith(".md"):
            base = os.path.dirname(p)
            for i, line in enumerate(txt.splitlines(), 1):
                for m in link_re.finditer(line):
                    tgt = m.group(1)
                    if tgt.startswith(("http://", "https://", "mailto:", "#")):
                        continue
                    t = tgt.split("#")[0]
                    if not t:
                        continue
                    n_links += 1
                    if not os.path.exists(os.path.normpath(os.path.join(base, t))):
                        bad_links.append(f"{rel}:{i} → {t}")
        for name, rx in SANITIZE_PATTERNS:
            if rx.search(txt):
                (legacy_hits if rel in LEGACY_SANITIZE_EXEMPT else hits).append(f"{rel} [{name}]")
    if bad_links:
        rep.fail("相对链接", "; ".join(bad_links[:8]))
    else:
        rep.ok("相对链接", f"{n_links} 个全部可解析")
    if hits:
        rep.fail("脱敏扫描（新增文档）", "; ".join(sorted(set(hits))[:8]))
    else:
        rep.ok("脱敏扫描（新增文档）", f"{n_scanned} 个文件干净")
    if legacy_hits:
        msg = "; ".join(sorted(set(legacy_hits))[:8])
        if strict:
            rep.fail("脱敏扫描（legacy 例外，--strict）", msg)
        else:
            rep.warn("脱敏扫描（legacy 例外，原文保留历史记录）", msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", default=REPO_ROOT)
    ap.add_argument("--series", action="append", help="系列目录（可多次）；默认自动发现")
    ap.add_argument("--entry", default="docs/experiments/coin_replay_random")
    ap.add_argument("--json", help="把报告写到此 JSON 文件")
    ap.add_argument("--strict", action="store_true", help="legacy 脱敏例外也算 FAIL")
    ap.add_argument("--skip-links", action="store_true")
    ap.add_argument("--skip-sanitize", action="store_true")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    series = series_dirs(repo_root, args.series)
    entry_dir = os.path.join(repo_root, args.entry) if not os.path.isabs(args.entry) else args.entry
    rep = Report()
    if not series:
        rep.fail("系列发现", "未找到任何 docs/experiments/coin_replay_random_* 系列目录")

    for sdir in series:
        check_series(sdir, repo_root, rep)
    if os.path.isdir(entry_dir):
        check_entry(entry_dir, repo_root, series, rep, strict=args.strict)
        check_pairing(entry_dir, series, rep)
    else:
        rep.fail("总入口目录", f"不存在: {args.entry}")

    scan = list(iter_docs(series + [entry_dir]))
    if not args.skip_links or not args.skip_sanitize:
        check_links_and_sanitize(scan, repo_root, rep, strict=args.strict)

    payload = {"repo_root": repo_root,
               "series": [os.path.relpath(d, repo_root) for d in series],
               "entry": os.path.relpath(entry_dir, repo_root),
               "n_pass": sum(1 for c in rep.checks if c["result"] == "PASS"),
               "n_fail": len(rep.fails), "n_warn": len(rep.warns),
               "checks": rep.checks, "fails": rep.fails, "warns": rep.warns}
    print("==== random 系列 一致性 + 脱敏 自查 ====")
    for c in rep.checks:
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "WARN": "warn"}[c["result"]]
        print(f"[{mark}] {c['item']}" + (f" —— {c['detail']}" if c["detail"] else ""))
    print(f"---- PASS {payload['n_pass']} / FAIL {payload['n_fail']} / WARN {payload['n_warn']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"报告已写入 {args.json}")
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
