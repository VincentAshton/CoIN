#!/usr/bin/env python3
"""random-replay 抽样可重建性 / 嵌套性验证（真实源数据；ratio 通用）。

对给定 seed，从**同一份真实源数据**分别构建 random-0.01 与 random-0.10 的 replay
数据（每个 ratio 独立构建两次 A/B），验证：

  1. 两边 sampling_algorithm 均为 sha256_task_seed_python_shuffle_v1
  2. 每个历史任务：source SHA256 相同、task_seed 相同、selected_indices 无重复且范围合法
  3. 0.01 的 selected_indices 是 0.10 的**严格前缀**（同排列取前 k，等价于子集；
     因 k01 < k10 且同 seed ⇒ 前缀关系）——并核对 selected_ids 对应一致
  4. 样本数量 k = floor(N×ratio)（可选 --expect-counts 对照任务书给定值）
  5. A/B 两次独立构建逐字节一致（manifest 仅允许 created_at / 绝对路径等非语义字段不同）

用法:
  python3 scripts/CoIN_Replay/tools/random_replay_nested_check.py \\
      --data-dir <DATA_DIR> --image-dir <IMG_DIR> \\
      --tasks ScienceQA TextVQA ImageNet GQA --seed 358341059 \\
      --tmp-dir <TMP_DIR>/nested_check --out-report <REPORT_JSON> \\
      --expect-counts '{"0.01": {...}, "0.10": {"ScienceQA": 1272, "TextVQA": 3460, "ImageNet": 12983, "round2": 1272, "round3": 4732, "round4": 17715}}'
退出码: 0=全部 PASS；1=存在 FAIL；2=结构性错误（缺数据/参数非法）。
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPLAY_DIR = os.path.dirname(HERE)
BUILD = os.path.join(REPLAY_DIR, "build_replay_data.py")
SAMPLING_ALGORITHM = "sha256_task_seed_python_shuffle_v1"
RATIOS = ("0.01", "0.10")
# manifest 中允许 A/B 不同的非语义字段
NON_SEMANTIC = ("created_at",)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sanitize_manifest(m):
    m = json.loads(json.dumps(m))
    for k in NON_SEMANTIC:
        m.pop(k, None)
    m.pop("nested_with", None)
    out = m.get("output", {})
    out.pop("path", None)
    for e in (m.get("sources") or {}).values():
        e.pop("path", None)
    return m


def build_once(ratio, round_j, seed, data_dir, image_dir, tasks, out_path):
    cmd = [sys.executable, BUILD, "--tasks", *tasks, "--data-dir", data_dir,
           "--image-dir", image_dir, "--round", str(round_j), "--ratio", str(ratio),
           "--sample-mode", "random", "--seed", str(seed), "--out", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build_replay_data 失败（ratio={ratio} round={round_j}）:\n"
                         f"{r.stdout}\n{r.stderr}")
    return json.load(open(out_path + ".manifest.json", encoding="utf-8"))


def check_counts(ratio, man, expect, row):
    """k == floor(N*ratio) 校验（逐任务）+ 可选期望值对照。"""
    ratio_f = float(ratio)
    bad = []
    for task, e in man["sources"].items():
        want = int(e["N"] * ratio_f)
        if e["k"] != want:
            bad.append(f"{task}: k={e['k']} != floor({e['N']}*{ratio})={want}")
    total = man["output"]["N"]
    if expect:
        for task, want in (expect.get("tasks") or {}).items():
            e = man["sources"].get(task)
            if e is None:
                bad.append(f"期望任务 {task} 不在 sidecar 中")
            elif e["k"] != want:
                bad.append(f"{task}: k={e['k']} != 期望 {want}")
        for key in ("round2", "round3", "round4"):
            if key in expect and str(man["round"]) == key[-1]:
                if total != expect[key]:
                    bad.append(f"{key} 总量 {total} != 期望 {expect[key]}")
    return total, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--rounds", nargs="+", type=int, default=[2, 3, 4])
    ap.add_argument("--expect-counts", default=None,
                    help='可选 JSON：{"0.01": {"tasks": {...}, "round2": n, ...}, "0.10": {...}}')
    args = ap.parse_args()
    if not (0 < args.seed < 1 << 31):
        raise SystemExit(f"seed 必须是正 31-bit 整数，收到 {args.seed}")
    expect = json.loads(args.expect_counts) if args.expect_counts else {}

    tmp = os.path.abspath(args.tmp_dir)
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    rows = []
    results = []
    for ratio in RATIOS:
        for j in args.rounds:
            a_path = os.path.join(tmp, f"r{ratio}_round{j}_A.json")
            b_path = os.path.join(tmp, f"r{ratio}_round{j}_B.json")
            man_a = build_once(ratio, j, args.seed, args.data_dir, args.image_dir,
                               args.tasks, a_path)
            man_b = build_once(ratio, j, args.seed, args.data_dir, args.image_dir,
                               args.tasks, b_path)
            row = {"ratio": ratio, "round": j,
                   "sampling_algorithm": man_a.get("sampling_algorithm"),
                   "mode": man_a.get("mode"), "sample_seed": man_a.get("sample_seed"),
                   "tasks": {}}
            fails = []
            if man_a.get("sampling_algorithm") != SAMPLING_ALGORITHM:
                fails.append(f"sampling_algorithm={man_a.get('sampling_algorithm')!r} "
                             f"!= {SAMPLING_ALGORITHM}")
            if man_a.get("mode") != "random":
                fails.append(f"mode={man_a.get('mode')!r} != random")
            if man_a.get("sample_seed") != args.seed:
                fails.append(f"sample_seed={man_a.get('sample_seed')} != {args.seed}")
            # A/B 逐字节一致（JSON）+ manifest 去非语义字段后一致
            if sha256_file(a_path) != sha256_file(b_path):
                fails.append("A/B replay JSON 逐字节不一致（重建不可复现）")
            if sanitize_manifest(man_a) != sanitize_manifest(man_b):
                fails.append("A/B manifest 语义字段不一致")
            total, bad = check_counts(ratio, man_a, expect.get(ratio), row)
            fails += bad
            row["output_N"] = total
            row["output_sha256"] = man_a["output"]["sha256"]
            row["ab_byte_identical"] = sha256_file(a_path) == sha256_file(b_path)
            for task, e in man_a["sources"].items():
                n, k = e["N"], e["k"]
                idx = e["selected_indices"]
                ok = (len(idx) == k and len(set(idx)) == k
                      and all(0 <= i < n for i in idx))
                row["tasks"][task] = {"N": n, "k": k, "task_seed": e["task_seed"],
                                      "source_sha256": e["sha256"],
                                      "indices_valid": ok,
                                      "selected_indices_sha256":
                                          hashlib.sha256(json.dumps(idx).encode()).hexdigest()}
                if not ok:
                    fails.append(f"{task}: selected_indices 长度/重复/越界异常")
            row["pass"] = not fails
            row["fails"] = fails
            rows.append(row)
            results.append((ratio, j, man_a))

    # 嵌套：0.01 ⊆ 0.10（同 seed 同任务：0.01 的 indices 必须是 0.10 的前 k01 个）
    nested = []
    for j in args.rounds:
        m01 = next(m for (r, jj, m) in results if r == "0.01" and jj == j)
        m10 = next(m for (r, jj, m) in results if r == "0.10" and jj == j)
        entry = {"round": j, "tasks": {}, "pass": True}
        for task in m01["sources"]:
            e01, e10 = m01["sources"][task], m10["sources"][task]
            i01, i10 = e01["selected_indices"], e10["selected_indices"]
            checks = {
                "same_source_sha256": e01["sha256"] == e10["sha256"],
                "same_task_seed": e01["task_seed"] == e10["task_seed"],
                "prefix_of_0.10": i01 == i10[:len(i01)],
                "subset_of_0.10": set(i01).issubset(set(i10)),
                "k01_lt_k10": e01["k"] < e10["k"],
            }
            checks["ids_prefix_of_0.10"] = e01["selected_ids"] == e10["selected_ids"][:e01["k"]]
            entry["tasks"][task] = {"k01": e01["k"], "k10": e10["k"], "checks": checks}
            if not all(checks.values()):
                entry["pass"] = False
        nested.append(entry)

    report = {
        "seed": args.seed,
        "tasks": args.tasks,
        "tmp_dir": tmp,
        "sampling_algorithm_expected": SAMPLING_ALGORITHM,
        "rounds": rows,
        "nested_check": nested,
        "pass": all(r["pass"] for r in rows) and all(n["pass"] for n in nested),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({"report": os.path.abspath(args.out_report),
                      "pass": report["pass"],
                      "rounds": [{"ratio": r["ratio"], "round": r["round"],
                                  "N": r["output_N"], "pass": r["pass"],
                                  "fails": r["fails"]} for r in rows],
                      "nested": [{"round": n["round"], "pass": n["pass"]} for n in nested]},
                     ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
