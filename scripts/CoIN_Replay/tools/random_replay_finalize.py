#!/usr/bin/env python3
"""random-replay 系列结果发布/清理脚本（codex/coin-replay-random-r001 分支，2026-09-08）。

两阶段：
  阶段 1 assemble —— 训练机（云端，真实产物 + torch）上运行。对 RES_ROOT 执行任务书十一
    的完成验证清单（全部满足才允许发布；任一失败 → 非零退出且不写发布目录），然后把
    脱敏/瘦身的 6 类轻量文件组装到 <out>/：
      coin_metrics.json / acc_sources.json / run_manifest.sanitized.json /
      replay_selection_summary.json / validation_report.md / summary.json
    完整 sidecar manifest / 预测 / checkpoint / 日志留在训练机（不进 Git）。
  阶段 2 fill-delta —— 本地发布目录上运行：读 index.json 的 prefix 基线，把与
    prefix-0.01 / prefix-0.10 的差值写进 summary.json（云端无基线，delta 为 null）。

用法:
  # 云端（cwd=项目根；--repo-root 用于 config 路径相对化）
  python3 scripts/CoIN_Replay/tools/random_replay_finalize.py assemble \\
      --res-root <RES_ROOT> --ckpt-root <CKPT_ROOT> --replay-data-dir <REPLAY_DATA_DIR> \\
      --data-dir <DATA_DIR> --run-id run_0001_seed_xxx --expected-seed <N> \\
      --repo-root . --out <export_dir>
  # 本地（发布目录已下载到 codex 分支 worktree）
  python3 scripts/CoIN_Replay/tools/random_replay_finalize.py fill-delta \\
      --summary docs/.../runs/<run_id>/summary.json \\
      --index  docs/experiments/coin_replay_random_r001/index.json
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "CoIN_Replay"))

from coin_lib import (  # noqa: E402
    ACC_TEXT_RE, artifact_check, ckpt_tensor_compare, ckpt_validate,
    json_load, sha256_file, verify_predictions,
)

TASKS = ["ScienceQA", "TextVQA", "ImageNet", "GQA"]
T = len(TASKS)
ALLOWED_PUB = ("coin_metrics.json", "acc_sources.json",
               "run_manifest.sanitized.json", "replay_selection_summary.json",
               "validation_report.md", "summary.json")
ENV_VER_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?[A-Za-z0-9+._-]*)")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def question_file(data_dir: str, task: str) -> str:
    name = "val.json" if task == "TextVQA" else "test.json"
    return os.path.join(data_dir, task, name)


def read_unit_acc(task: str, res_root: str, j: int) -> dict:
    """与 aggregate_coin.read_accuracy 同口径；返回 {accuracy, correct, count}。"""
    stage = os.path.join(res_root, task, f"round{j}")
    if task == "ScienceQA":
        f = os.path.join(stage, "output_result.jsonl")
        data = json_load(f)
        acc = float(data["acc"])
        return {"accuracy": acc, "correct": data.get("correct"),
                "count": data.get("count")}
    f = os.path.join(stage, "Result.text")
    m = ACC_TEXT_RE.search(open(f, encoding="utf-8").read())
    if not m:
        raise ValueError(f"{f} 无 Accuracy: xx.xx%")
    return {"accuracy": float(m.group(1)), "correct": None, "count": None}


def metrics_from_A(A):
    maa = sum(sum(A[j][:j + 1]) / (j + 1) for j in range(T)) / T
    bwt = sum(A[T - 1][i] - A[i][i] for i in range(T)) / T
    final_avg = sum(A[T - 1]) / T
    old_mean = sum(A[T - 1][i] for i in range(T - 1)) / (T - 1)
    diag_mean = sum(A[i][i] for i in range(T)) / T
    return {
        "MAA": maa, "BWT": bwt, "final_avg": final_avg,
        "final_old_task_mean": old_mean, "diagonal_mean": diag_mean,
        "row_averages": [round(sum(A[j][:j + 1]) / (j + 1), 4) for j in range(T)],
        "per_task_final_minus_diagonal": {
            t: A[T - 1][i] - A[i][i] for i, t in enumerate(TASKS)},
    }


# ---------------------------------------------------------------------------
# 阶段 1：assemble（云端验收 + 脱敏瘦身导出）
# ---------------------------------------------------------------------------

def sanitize_manifest(m: dict, repo_root: str) -> dict:
    """run_manifest 去敏：config 路径相对 repo 根；env 版本号正则清理；无凭据/主机。"""
    m = json.loads(json.dumps(m))  # 深拷贝
    root = os.path.abspath(repo_root) + os.sep
    for k in ("model_base", "vision_tower", "projector", "ds_config"):
        v = m.get("config", {}).get(k)
        if isinstance(v, str) and v.startswith(root):
            m["config"][k] = v[len(root):]
    clean = {}
    for k, v in m.get("env", {}).items():
        if not isinstance(v, str):
            clean[k] = v
            continue
        hits = ENV_VER_RE.findall(v)
        clean[k] = hits[-1] if hits else v.strip()
    m["env"] = clean
    m["sanitized_note"] = ("去敏（random_replay_finalize.py）：config 绝对路径已相对化；"
                           "env 版本号正则清理；不含账号/IP/端口/SSH/密码/token。"
                           "完整原始 manifest 保留在训练机结果目录。")
    return m


def build_replay_selection_summary(run_id: str, expected_seed: int, replay_dir: str) -> dict:
    out = {"run_id": run_id, "sample_seed": expected_seed, "rounds": {}}
    for j in (2, 3, 4):
        mp = os.path.join(replay_dir, f"round{j}_train.json.manifest.json")
        if not os.path.isfile(mp):
            raise ValueError(f"缺 replay sidecar manifest: {mp}")
        m = json_load(mp)
        if m.get("mode") != "random" or m.get("sample_seed") != expected_seed:
            raise ValueError(f"{mp} mode/sample_seed 与登记不一致"
                             f"（{m.get('mode')}/{m.get('sample_seed')}）")
        rj = {"mode": m["mode"], "sample_seed": m["sample_seed"],
              "sampling_algorithm": m.get("sampling_algorithm"),
              "output_sha256": m["output"]["sha256"], "tasks": {}}
        for task, e in m["sources"].items():
            ids_sha = sha256_text(json.dumps(e["selected_ids"], ensure_ascii=False))
            idx_sha = sha256_text(json.dumps(e["selected_indices"]))
            rj["tasks"][task] = {
                "N": e["N"], "k": e["k"], "task_seed": e["task_seed"],
                "source_sha256": e["sha256"],
                "selected_ids_sha256": ids_sha,
                "selected_indices_sha256": idx_sha,
                "example_ids": e["selected_ids"][:10],
            }
        out["rounds"][str(j)] = rj
    return out


def assemble(args) -> int:
    res_root = os.path.abspath(args.res_root)
    ckpt_root = os.path.abspath(args.ckpt_root)
    replay_dir = os.path.abspath(args.replay_data_dir)
    out_dir = os.path.abspath(args.out)
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.force:
        raise SystemExit(f"导出目录非空且未 --force: {out_dir}")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    results = []  # (检查项, PASS/FAIL, 证据)

    def check(name, ok, evidence=""):
        results.append((name, "PASS" if ok else "FAIL", evidence))
        return ok

    # 1) 权威完成标志与目录归属
    ok = check(".complete 存在", os.path.isfile(os.path.join(res_root, ".complete")),
               os.path.join(res_root, ".complete"))
    ok = check("结果目录归属 run_id",
               os.path.basename(res_root) == args.run_id, res_root) and ok
    # 2) run_manifest：配置与登记一致（sample_mode/seed/ratio/replay_accum）
    man = json_load(os.path.join(res_root, "run_manifest.json"))
    cfg = man["config"]
    check("run_manifest sample_mode==random", cfg.get("sample_mode") == "random",
          f"sample_mode={cfg.get('sample_mode')}")
    check("run_manifest replay_sample_seed==登记",
          cfg.get("replay_sample_seed") == args.expected_seed,
          f"replay_sample_seed={cfg.get('replay_sample_seed')}")
    check("run_manifest ratio==0.01", abs(float(cfg.get("ratio", -1)) - 0.01) < 1e-9,
          f"ratio={cfg.get('ratio')}")
    check("run_manifest replay_accum==1", cfg.get("replay_accum") == 1,
          f"replay_accum={cfg.get('replay_accum')}")
    # 3) coin_metrics 可解析 + 结构
    cm = json_load(os.path.join(res_root, "coin_metrics.json"))
    A0 = cm.get("A_matrix")
    need = {"ratio", "tasks", "T", "A_matrix", "MAA", "BWT"}
    check("coin_metrics.json 结构完整", need.issubset(cm) and
          isinstance(A0, list) and len(A0) == T and
          all(len(r) == T for r in A0),
          f"keys={sorted(need - set(cm)) or 'ok'}")
    # 4) round manifests 1..4
    rm_ok = True
    for j in range(1, T + 1):
        p = os.path.join(res_root, f"round{j}_manifest.json")
        if not os.path.isfile(p):
            rm_ok = False
            break
        json_load(p)
    check("round manifests 1..4 完整可解析", rm_ok)
    # 5) replay sidecar round2..4 + replay json sha 一致性
    rsum = build_replay_selection_summary(args.run_id, args.expected_seed, replay_dir)
    rp_ok = True
    for j in (2, 3, 4):
        rj = os.path.join(replay_dir, f"round{j}_train.json")
        m = json_load(rj + ".manifest.json")
        if sha256_file(rj) != m["output"]["sha256"]:
            rp_ok = False
    check("replay json 与 sidecar SHA 一致 (r2..4)", rp_ok)
    # 6) checkpoint 校验（7 个：round1 task + round2..4 task+replay）
    ckpt_paths = [os.path.join(ckpt_root, "round1_task_llava_lora")]
    for j in (2, 3, 4):
        ckpt_paths += [os.path.join(ckpt_root, f"round{j}_task_llava_lora"),
                       os.path.join(ckpt_root, f"round{j}_replay_llava_lora")]
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    ck_ok = True
    ck_detail = []
    for p in ckpt_paths:
        try:
            rep = ckpt_validate(p)
            okk = rep.get("finite") is not False
            if rep.get("note"):
                okk = False  # 降级说明=异常路径（真实 checkpoint 必须参数级校验）
            ck_ok = ck_ok and okk
            ck_detail.append(f"{os.path.basename(p)}: finite={rep.get('finite')} "
                             f"hash={str(rep.get('param_hash'))[:12] if rep.get('param_hash') else None}")
        except Exception as e:
            ck_ok = False
            ck_detail.append(f"{os.path.basename(p)}: ERROR {e}")
    check("checkpoint 校验（7 ckpt 参数级 PASS）", ck_ok and has_torch,
          "; ".join(ck_detail) + ("" if has_torch else "（torch 缺失=结构性失败）"))
    # 7) replay tensor-diff r2..4（task vs replay 真更新铁证）
    td_ok = True
    td_detail = []
    for j in (2, 3, 4):
        td = ckpt_tensor_compare(
            os.path.join(ckpt_root, f"round{j}_task_llava_lora"),
            os.path.join(ckpt_root, f"round{j}_replay_llava_lora"))
        td_ok = td_ok and td["pass"]
        td_detail.append(f"r{j}: changed={td['changed_tensor_count']} "
                         f"hash_differs={td['tensor_hash']['differs']} "
                         f"finite={td['finite']['task'] and td['finite']['replay']}")
        if not td["pass"]:
            td_detail.append(f"   keys missing={td['keys']['missing'][:3]} "
                             f"unexpected={td['keys']['unexpected'][:3]}")
    check("replay tensor-diff 通过（r2..4 真实更新）", td_ok, "; ".join(td_detail))
    # 8) non_lora_trainables 独立校验（mm_projector 在独立文件，需另验）
    nl_ok = True
    nl_detail = []
    for j in (2, 3, 4):
        a = os.path.join(ckpt_root, f"round{j}_task_llava_lora", "non_lora_trainables.bin")
        b = os.path.join(ckpt_root, f"round{j}_replay_llava_lora", "non_lora_trainables.bin")
        if not (os.path.isfile(a) and os.path.isfile(b)):
            nl_ok = False
            nl_detail.append(f"r{j}: 缺文件")
            continue
        diff = sha256_file(a) != sha256_file(b)
        nl_ok = nl_ok and diff
        nl_detail.append(f"r{j}: sha_differ={diff}")
    check("non_lora_trainables task vs replay 不同 (r2..4)", nl_ok, "; ".join(nl_detail))
    # 9) A 矩阵单元完整（10 单元）+ 预测校验 + 独立重算交叉验证
    units = []
    A = [[0.0] * T for _ in range(T)]
    ev_ok = True
    for j in range(1, T + 1):
        for i in range(1, j + 1):
            task = TASKS[i - 1]
            stage = os.path.join(res_root, task, f"round{j}")
            try:
                ac = artifact_check(task, stage)
                vp = verify_predictions(question_file(args.data_dir, task),
                                        os.path.join(stage, "merge.jsonl"),
                                        order_check=True)
                ru = read_unit_acc(task, res_root, j)
                A[j - 1][i - 1] = ru["accuracy"]
                units.append({"task": task, "round": j, "accuracy": ru["accuracy"],
                              "correct": ru["correct"], "count": ru["count"]})
            except Exception as e:
                ev_ok = False
                break
    check("A 矩阵单元完整（10 eval 单元 + 预测校验）", ev_ok,
          f"units={len(units)}")
    mtr = metrics_from_A(A)
    dA = max(abs(a - b) for ra, rb in zip(A, A0) for a, b in zip(ra, rb))
    dmaa = abs(mtr["MAA"] - float(cm["MAA"]))
    dbwt = abs(mtr["BWT"] - float(cm["BWT"]))
    cross_ok = ev_ok and dA < 1e-9 and dmaa < 5e-5 and dbwt < 5e-5
    check("MAA/BWT 独立重算 == coin_metrics.json", cross_ok,
          f"maxA={dA:.2e} dMAA={dmaa:.2e} dBWT={dbwt:.2e}")
    # 10) 无混用其他 run：eval/ckpt/replay 路径均在 per-run 目录
    mix_ok = (ckpt_root.endswith(os.path.join("CoIN_Replay_random", "r001", args.run_id))
              and replay_dir.endswith(os.path.join("CoIN_Replay_random", "r001", args.run_id)))
    check("无混用其他 run（ckpt/replay 路径归属 run_id）", mix_ok,
          f"ckpt_root={ckpt_root}\nreplay_dir={replay_dir}")

    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        print("=== 验收失败，禁止发布 ===")
        for name, st, ev in results:
            print(f"[{st}] {name} — {ev}")
        return 1
    print("=== 全部验收 PASS，组装发布目录 ===")

    # ---- 组装导出（仅轻量允许文件） ----
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(os.path.join(res_root, "coin_metrics.json"),
                 os.path.join(out_dir, "coin_metrics.json"))
    json.dump({"ratio": float(cm["ratio"]), "units": units},
              open(os.path.join(out_dir, "acc_sources.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(sanitize_manifest(man, args.repo_root),
              open(os.path.join(out_dir, "run_manifest.sanitized.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(rsum, open(os.path.join(out_dir, "replay_selection_summary.json"), "w"),
              indent=2, ensure_ascii=False)

    report = build_validation_report(args, results, A, mtr, cm, rsum, ck_detail,
                                     td_detail, units, man)
    open(os.path.join(out_dir, "validation_report.md"), "w", encoding="utf-8").write(report)

    summary = {
        "run_id": args.run_id,
        "engine_run_id": man.get("run_id"),
        "replay_sample_seed": args.expected_seed,
        "sample_mode": cfg.get("sample_mode"),
        "ratio": cfg.get("ratio"),
        "replay_accum": cfg.get("replay_accum"),
        "sampling_algorithm": rsum["rounds"]["2"]["sampling_algorithm"],
        "MAA": round(mtr["MAA"], 4),
        "BWT": round(mtr["BWT"], 4),
        "final_avg": round(mtr["final_avg"], 4),
        "final_old_task_mean": round(mtr["final_old_task_mean"], 4),
        "diagonal_mean": round(mtr["diagonal_mean"], 4),
        "row_averages": mtr["row_averages"],
        "per_task_final_minus_diagonal": {
            t: round(v, 4) for t, v in mtr["per_task_final_minus_diagonal"].items()},
        "result_directory_rel": os.path.join("results", "CoIN_Replay_random",
                                             "r001", args.run_id),
        "code_commit": man.get("git", {}).get("commit"),
        "config_hash": man.get("config_hash"),
        "model_config_hash": man.get("model_config_hash"),
        "data_revision": man.get("data_revision"),
        "ds_config_hash": man.get("ds_config_hash"),
        "completed_at": now_iso(),
        "deltas": None,  # 阶段 2 fill-delta 填 prefix 基线差值
        "repro": {},
    }
    for f in ALLOWED_PUB:
        p = os.path.join(out_dir, f)
        if os.path.isfile(p):
            summary["repro"][f] = sha256_file(p)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"),
              indent=2, ensure_ascii=False)
    # summary 自身 sha 附在报告尾部
    with open(os.path.join(out_dir, "validation_report.md"), "a", encoding="utf-8") as f:
        f.write(f"\nsummary.json sha256: {sha256_file(os.path.join(out_dir, 'summary.json'))}\n")

    print(f"导出目录: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}  {sha256_file(os.path.join(out_dir, f))[:16]}…")
    return 0


def build_validation_report(args, results, A, mtr, cm, rsum, ck_detail, td_detail,
                            units, man) -> str:
    lines = [f"# Validation Report — {args.run_id}", "",
             f"- 生成: {now_iso()}（random_replay_finalize.py assemble）",
             f"- run_manifest engine run_id: {man.get('run_id')}",
             f"- git commit: {man.get('git', {}).get('commit')}",
             f"- config_hash: {man.get('config_hash')}", "",
             "## 检查清单", "", "| 检查项 | 结果 | 证据 |", "|---|---|---|"]
    for name, st, ev in results:
        ev = ev.replace("\n", "<br>")
        lines.append(f"| {name} | {st} | {ev} |")
    lines += ["", "## A 矩阵（10 单元全精度）", "",
              "| round\\task | " + " | ".join(TASKS) + " |", "|---|" + "---|" * T]
    for j in range(T):
        lines.append(f"| round{j + 1} | " + " | ".join(
            f"{v:.6f}" if v else "—" for v in A[j]) + " |")
    lines += ["", "## 指标（独立重算，全精度）", "",
              "| 指标 | 重算值 | coin_metrics.json |",
              "|---|---|---|",
              f"| MAA | {mtr['MAA']:.6f} | {cm['MAA']} |",
              f"| CoIN BWT | {mtr['BWT']:.6f} | {cm['BWT']} |",
              f"| Final Avg | {mtr['final_avg']:.6f} | — |",
              f"| 终局旧任务均值 | {mtr['final_old_task_mean']:.6f} | — |",
              f"| 对角项均值 | {mtr['diagonal_mean']:.6f} | — |",
              f"| row_averages | {mtr['row_averages']} | — |", ""]
    lines += ["## replay 抽样一致性（replay_selection_summary.json）", ""]
    for j in (2, 3, 4):
        rj = rsum["rounds"][str(j)]
        lines.append(f"- round{j}: mode={rj['mode']} sample_seed={rj['sample_seed']} "
                     f"algorithm={rj['sampling_algorithm']} output_sha={rj['output_sha256'][:16]}…")
        for task, e in rj["tasks"].items():
            lines.append(f"  - {task}: N={e['N']} k={e['k']} "
                         f"task_seed={e['task_seed'][:16]}… "
                         f"ids_sha={e['selected_ids_sha256'][:16]}…")
    lines += ["", "## 复现 hash（发布文件 sha256 全文见 summary.json repro）", ""]
    lines += ["", "结论: 全部验收项 PASS —— 可发布。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 阶段 2：fill-delta（本地；读 index.json prefix 基线，写 summary.json deltas）
# ---------------------------------------------------------------------------

def fill_delta(args) -> int:
    s = json.load(open(args.summary, encoding="utf-8"))
    idx = json.load(open(args.index, encoding="utf-8"))
    if s.get("MAA") is None or s.get("BWT") is None:
        raise SystemExit("summary 缺 MAA/BWT——assemble 验收未过，禁止算 delta")
    base = idx.get("baselines", {})
    out = {}
    for key in ("prefix_0.10", "prefix_0.01"):
        b = base.get(key)
        if not b:
            raise SystemExit(f"index.json 缺基线 {key}")
        out[f"vs_{key}"] = {
            "MAA": round(s["MAA"] - b["MAA"], 4),
            "BWT": round(s["BWT"] - b["BWT"], 4),
            "final_avg": round(s["final_avg"] - b["final_avg"], 4),
        }
    s["deltas"] = out
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print("deltas 写入 summary.json:")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("assemble")
    p1.add_argument("--res-root", required=True)
    p1.add_argument("--ckpt-root", required=True)
    p1.add_argument("--replay-data-dir", required=True)
    p1.add_argument("--data-dir", required=True, help="问题文件根（TASK_QF 同布局）")
    p1.add_argument("--run-id", required=True)
    p1.add_argument("--expected-seed", type=int, required=True)
    p1.add_argument("--repo-root", default=".")
    p1.add_argument("--out", required=True)
    p1.add_argument("--force", action="store_true")
    p1.set_defaults(fn=assemble)
    p2 = sub.add_parser("fill-delta")
    p2.add_argument("--summary", required=True)
    p2.add_argument("--index", required=True)
    p2.set_defaults(fn=fill_delta)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
