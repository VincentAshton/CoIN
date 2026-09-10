#!/usr/bin/env python3
"""random-replay 系列结果发布/清理脚本（分支 codex/coin-replay-random；ratio 通用）。

两阶段：
  阶段 1 assemble —— 训练机（真实产物）上运行。对 RES_ROOT 执行任务书十一完成验证清单
    （全部满足才允许发布），随后把脱敏/瘦身的 6 类轻量文件组装到 sibling staging 目录，
    全部 PASS 后**原子换入** <out>/。验收失败或扫描不过 → 非零退出，既有发布目录分毫不动。
    完整 sidecar manifest / 预测 / checkpoint / 日志留在训练机（不进 Git）。
  阶段 2 fill-delta —— 本地发布目录上运行：读 index.json 的 prefix 基线，把与
    prefix-0.01 / prefix-0.10 的差值写进 summary.json；如给 --paired-summary 还写入与
    同 seed 另一 ratio run 的配对差值（方向恒为 高 ratio − 低 ratio，即 0.10 − 0.01）。

assemble 强制门（2026-09-11 ratio 通用化）：
  - expected ratio 显式给出且属于允许集合（0.01 / 0.10）；ratio tag 由 ratio 派生
    （coin_lib.ratio_tag：0.01→r001、0.10→r010），不接受调用方另传 tag
  - run_manifest ratio == expected ratio（数值比较）；sample_mode==random；
    replay_sample_seed==登记；replay_accum==1；run_id 内嵌 seed==登记 seed
  - RES_ROOT/CKPT_ROOT/REPLAY_DATA_DIR 分别属于 CoIN_Replay_random/<tag>/<run_id> 与
    Replay_random/<tag>/<run_id>，且互异
  - replay sidecar sampling_algorithm == 期望常量（build_replay_data.SAMPLING_ALGORITHM）
  - 用源数据（train.json）+ task seed 按 expected ratio 独立重建排列：索引数量=k、
    无重复、∈[0,N)、selected_ids 与 replay JSON 顺序一一对应；源文件 SHA 与 sidecar 一致
  - 发布 6 文件脱敏扫描：无绝对路径 / IP / 主机名 / 凭据 / token
  - torch 门禁（checkpoint 参数级 + tensor-diff）：默认必须真实 torch；
    --test-mode 仅供零 GPU 单测降级为文件级并显式标注（正式发布禁止）

用法:
  # 云端（cwd=项目根；--repo-root 用于路径相对化）
  python3 scripts/CoIN_Replay/tools/random_replay_finalize.py assemble \\
      --res-root <RES_ROOT> --ckpt-root <CKPT_ROOT> --replay-data-dir <REPLAY_DATA_DIR> \\
      --data-dir <DATA_DIR> --run-id run_0001_seed_xxx --expected-seed <N> \\
      --expected-ratio 0.10 --repo-root . --out <export_dir> [--force]
  # 本地
  python3 scripts/CoIN_Replay/tools/random_replay_finalize.py fill-delta \\
      --summary docs/.../runs/<run_id>/summary.json \\
      --index  docs/experiments/coin_replay_random_r010/index.json \\
      [--paired-summary docs/.../coin_replay_random_r001/runs/<run_id>/summary.json \\
       --result-commit <C> --paired-result-commit <C2>]
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import shutil
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "CoIN_Replay"))

from coin_lib import (  # noqa: E402
    ACC_TEXT_RE, artifact_check, ckpt_tensor_compare, ckpt_validate,
    json_load, normalize_ratio, random_ratio_layout, ratio_tag, sha256_file,
    verify_predictions,
)
from build_replay_data import SAMPLING_ALGORITHM, task_seed_hex  # noqa: E402

TASKS = ["ScienceQA", "TextVQA", "ImageNet", "GQA"]
T = len(TASKS)
ALLOWED_PUB = ("coin_metrics.json", "acc_sources.json",
               "run_manifest.sanitized.json", "replay_selection_summary.json",
               "validation_report.md", "summary.json")
ENV_VER_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?[A-Za-z0-9+._-]*)")
# 发布文件脱敏扫描（敏感信息防泄漏门；正式文件不得命中）
# 注意：禁止用泛化 `\.com` 之类规则——会误伤合法文本（如 ".complete"）。只列具体载体。
SENSITIVE_RES = [
    re.compile(r"/root/|/home/|/mnt/|C:\\"),
    re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b"),
    re.compile(r"ebcloud"),
    re.compile(r"BEGIN [A-Z0-9 ]*PRIVATE KEY"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"ssh-rsa |ssh-ed25519|ecdsa-sha2"),
]
RID_RE = re.compile(r"run_\d{4}_seed_(\d+)")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rel(path, repo_root):
    """证据字符串用：repo 根下 → 相对路径；否则 basename（防绝对路径泄漏）。"""
    a = os.path.abspath(path)
    r = os.path.abspath(repo_root)
    try:
        x = os.path.relpath(a, r)
    except ValueError:
        return os.path.basename(a)
    return x if not x.startswith("..") else os.path.basename(a)


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


def sanitize_manifest(m: dict, repo_root: str) -> dict:
    """run_manifest 去敏：config 路径相对 repo 根；env 版本号正则清理。"""
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


def build_replay_selection_summary(run_id: str, expected_seed: int,
                                   replay_dir: str, data_dir: str,
                                   ratio: float, tasks) -> dict:
    """sidecar -> 摘要；同时校验 seed/mode/sha。源数据校验在 rebuild 门中。"""
    out = {"run_id": run_id, "sample_seed": expected_seed, "rounds": {}}
    for j in (2, 3, 4):
        mp = os.path.join(replay_dir, f"round{j}_train.json.manifest.json")
        if not os.path.isfile(mp):
            raise ValueError(f"缺 replay sidecar manifest: {mp}")
        m = json_load(mp)
        if m.get("mode") != "random" or m.get("sample_seed") != expected_seed:
            raise ValueError(f"{mp} mode/sample_seed 与登记不一致"
                             f"（{m.get('mode')}/{m.get('sample_seed')}）")
        if m.get("sampling_algorithm") != SAMPLING_ALGORITHM:
            raise ValueError(f"{mp} sampling_algorithm 不一致: "
                             f"{m.get('sampling_algorithm')!r} != {SAMPLING_ALGORITHM!r}")
        # 源数据独立重建门：source SHA / N / k / 排列 / indices / ids
        rebuild_checks(m, data_dir, ratio, tasks)
        rj = {"mode": m["mode"], "sample_seed": m["sample_seed"],
              "sampling_algorithm": m["sampling_algorithm"],
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


def rebuild_checks(sidecar: dict, data_dir: str, ratio: float, tasks) -> None:
    """从源 train.json + task seed 独立重建随机选择，逐项对照 sidecar。

    违规即 raise（结构性失败）。检查：源 SHA、N、k=floor(N*ratio)、排列前 k、
    索引数量/无重复/∈[0,N)、ids 与 replay JSON 顺序对应（replay JSON 由调用方另查）。
    """
    j = sidecar["round"]
    prev = tasks[: j - 1]
    for task in prev:
        e = sidecar["sources"].get(task)
        if e is None:
            raise ValueError(f"round{j} sidecar 缺 {task} 源条目")
        src_path = os.path.join(data_dir, task, "train.json")
        if not os.path.isfile(src_path):
            raise ValueError(f"重建源文件缺失: {src_path}")
        if sha256_file(src_path) != e["sha256"]:
            raise ValueError(f"{task} round{j}: 源 train.json SHA 与 sidecar 不一致"
                             "（数据修订不同，样本集不可重建）")
        samples = json_load(src_path)
        n = len(samples)
        if n != e["N"]:
            raise ValueError(f"{task}: 源样本数 {n} != sidecar N={e['N']}")
        k = e["k"]
        if k != int(n * ratio):
            raise ValueError(f"{task}: sidecar k={k} != floor({n}*{ratio})={int(n * ratio)}")
        idx = e["selected_indices"]
        if len(idx) != k:
            raise ValueError(f"{task}: selected_indices 长度 {len(idx)} != k={k}")
        if len(set(idx)) != k:
            raise ValueError(f"{task}: selected_indices 含重复")
        if any(i < 0 or i >= n for i in idx):
            raise ValueError(f"{task}: selected_indices 越界 [0,{n})")
        rng = random.Random(int(e["task_seed"], 16))
        perm = list(range(n))
        rng.shuffle(perm)
        if perm[:k] != idx:
            raise ValueError(f"{task}: 独立重建排列前 k 与 sidecar selected_indices 不一致"
                             "（抽样不可复现？）")
        ids = [s.get("id", s.get("question_id")) for s in (samples[i] for i in idx)]
        if ids != e["selected_ids"]:
            raise ValueError(f"{task}: 重建 ids 与 sidecar selected_ids 不一致")
        # replay JSON 顺序对应：round<j> 的 replay = prev 任务顺序拼接
        rp_path = os.path.join(os.path.dirname(sidecar["output"]["path"]),
                               f"round{j}_train.json")
        if os.path.isfile(rp_path):
            rp = json_load(rp_path)
            rp_ids = [s.get("id", s.get("question_id")) for s in rp]
            expect = []
            for t in prev:
                e2 = sidecar["sources"][t]
                src2 = json_load(os.path.join(data_dir, t, "train.json"))
                expect += [s.get("id", s.get("question_id"))
                           for s in (src2[i] for i in e2["selected_indices"])]
            if rp_ids != expect:
                raise ValueError(f"round{j}: replay JSON id 顺序与 sidecar 选择不一致")


# ---------------------------------------------------------------------------
# 阶段 1：assemble（云端验收 + 脱敏瘦身导出；staging + 原子换入）
# ---------------------------------------------------------------------------

def _find_adapter(d):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def assemble(args) -> int:
    repo_root = os.path.abspath(args.repo_root)
    res_root = os.path.abspath(args.res_root)
    ckpt_root = os.path.abspath(args.ckpt_root)
    replay_dir = os.path.abspath(args.replay_data_dir)
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out)
    staging = out_dir + f".staging.{os.getpid()}"
    test_mode = bool(args.test_mode)

    results = []

    def check(name, ok, evidence=""):
        results.append((name, "PASS" if ok else "FAIL", evidence))
        return ok

    def R(p):
        return rel(p, repo_root)

    # ---- 门 0：expected ratio / tag / run_id（ratio 通用化；禁写死 0.01/r001） ----
    ratio_d, tag = None, None
    try:
        ratio_d = normalize_ratio(args.expected_ratio)
        tag = ratio_tag(ratio_d)
        check("expected ratio 属于允许集合", True,
              f"expected_ratio={args.expected_ratio} → 归一化 {ratio_d} / tag {tag}")
    except ValueError as e:
        check("expected ratio 属于允许集合", False, str(e))
    m_rid = re.fullmatch(r"run_(\d{4})_seed_(\d+)", args.run_id or "")
    check("run_id 格式 run_NNNN_seed_<seed> 且内嵌 seed==登记 seed",
          bool(m_rid) and int(m_rid.group(2)) == args.expected_seed, args.run_id or "<空>")
    # ---- 门 1-4：完成标志 / 归属 / manifest 配置 / coin_metrics ---------------
    check(".complete 存在", os.path.isfile(os.path.join(res_root, ".complete")),
          R(os.path.join(res_root, ".complete")))
    check("三个目录 basename 均为 run_id",
          os.path.basename(res_root) == args.run_id
          and os.path.basename(ckpt_root) == args.run_id
          and os.path.basename(replay_dir) == args.run_id,
          R(res_root))
    man = json_load(os.path.join(res_root, "run_manifest.json"))
    cfg = man["config"]
    check("run_manifest sample_mode==random", cfg.get("sample_mode") == "random",
          f"sample_mode={cfg.get('sample_mode')}")
    check("run_manifest replay_sample_seed==登记",
          cfg.get("replay_sample_seed") == args.expected_seed,
          f"replay_sample_seed={cfg.get('replay_sample_seed')}")
    check("run_manifest ratio == expected ratio（数值比较）",
          ratio_d is not None and abs(float(cfg.get("ratio", -1)) - float(ratio_d)) < 1e-9,
          f"manifest={cfg.get('ratio')} expected={args.expected_ratio}")
    check("run_manifest replay_accum==1", cfg.get("replay_accum") == 1,
          f"replay_accum={cfg.get('replay_accum')}")
    cm = json_load(os.path.join(res_root, "coin_metrics.json"))
    A0 = cm.get("A_matrix")
    need = {"ratio", "tasks", "T", "A_matrix", "MAA", "BWT"}
    check("coin_metrics.json 结构完整", need.issubset(cm) and isinstance(A0, list)
          and len(A0) == T and all(len(r) == T for r in A0),
          f"缺 {sorted(need - set(cm)) or '无'}")
    # ---- 门 5：round manifests 1..4 -------------------------------------------
    rm_ok = True
    for j in range(1, T + 1):
        p = os.path.join(res_root, f"round{j}_manifest.json")
        if not os.path.isfile(p):
            rm_ok = False
            break
        json_load(p)
    check("round manifests 1..4 完整可解析", rm_ok)
    # ---- 门 6-7：sidecar 一致性 + 独立重建（含 sampling_algorithm 常量门） ------
    _ratio_float = float(ratio_d) if ratio_d is not None else -1.0
    try:
        rsum = build_replay_selection_summary(args.run_id, args.expected_seed,
                                              replay_dir, data_dir,
                                              _ratio_float, TASKS)
        sidecar_ok = True
        side_ev = []
        for j in (2, 3, 4):
            rj = os.path.join(replay_dir, f"round{j}_train.json")
            mj = json_load(rj + ".manifest.json")
            same = sha256_file(rj) == mj["output"]["sha256"]
            side_ev.append(f"r{j} json-sha={same}")
            sidecar_ok = sidecar_ok and same
        check("replay json 与 sidecar SHA 一致 (r2..4)", sidecar_ok, "; ".join(side_ev))
        check("sampling_algorithm==期望常量（全部 sidecar）", True,
              SAMPLING_ALGORITHM)
        check("源数据独立重建门（SHA/N/k/排列/ids/replay 顺序）", True,
              "round2..4 × 全部历史任务 PASS")
    except Exception as e:
        check("sidecar/重建门", False, str(e))
    # ---- 门 8：目录同属与不混用（tag 由 expected ratio 派生） -------------------
    # 布局（任务书八.4）：checkpoints|results/CoIN_Replay_random/<tag>/<run_id>；
    # playground/Replay_random/<tag>/<run_id>（replay 数据目录无 CoIN_ 前缀）
    layout = random_ratio_layout(ratio_d, args.run_id) if tag else None
    if layout:
        run_dir = f"{tag}/{args.run_id}"
        ckpt_ok = ckpt_root.endswith(os.path.join("CoIN_Replay_random", run_dir))
        res_ok = res_root.endswith(os.path.join("CoIN_Replay_random", run_dir))
        replay_ok = replay_dir.endswith(os.path.join("Replay_random", run_dir))
        same_tree = ckpt_ok and res_ok and replay_ok
    else:
        same_tree = False
    distinct = len({ckpt_root, res_root, replay_dir}) == 3
    check(f"CKPT/RES/REPLAY 同属 random/{tag}/<run_id> 且互异",
          same_tree and distinct,
          f"ckpt={R(ckpt_root)} res={R(res_root)} replay={R(replay_dir)}")
    # ---- 门 9-11：torch 门（默认真实；--test-mode 降级并显式标注） ---------------
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    ckpt_paths = [os.path.join(ckpt_root, "round1_task_llava_lora")]
    for j in (2, 3, 4):
        ckpt_paths += [os.path.join(ckpt_root, f"round{j}_task_llava_lora"),
                       os.path.join(ckpt_root, f"round{j}_replay_llava_lora")]
    if not has_torch and not test_mode:
        check("checkpoint 参数级校验（7 ckpt）", False, "torch 缺失（结构性失败）")
        check("replay tensor-diff 真实更新（r2..4）", False, "torch 缺失（结构性失败）")
    else:
        ck_ok, ck_ev = True, []
        for p in ckpt_paths:
            try:
                rep = ckpt_validate(p)
                if test_mode:
                    okk = True  # 文件级校验即可（test-mode 显式降级）
                    ck_ev.append(f"{os.path.basename(p)}: file-level(test-mode) "
                                 f"note={rep.get('note')}")
                else:
                    okk = rep.get("finite") is not False and not rep.get("note")
                    ck_ev.append(f"{os.path.basename(p)}: finite={rep.get('finite')} "
                                 f"hash={str(rep.get('param_hash'))[:12] if rep.get('param_hash') else None}")
            except Exception as e:
                okk = False
                ck_ev.append(f"{os.path.basename(p)}: ERROR {e}")
            ck_ok = ck_ok and okk
        check("checkpoint 校验（7 ckpt）" + ("(file-level, test-mode)" if test_mode else ""),
              ck_ok, "; ".join(ck_ev))
        td_ok, td_ev = True, []
        for j in (2, 3, 4):
            tdir = os.path.join(ckpt_root, f"round{j}_task_llava_lora")
            rdir = os.path.join(ckpt_root, f"round{j}_replay_llava_lora")
            if test_mode:
                ta, ra = _find_adapter(tdir), _find_adapter(rdir)
                diff = bool(ta and ra) and sha256_file(ta) != sha256_file(ra)
                td_ok = td_ok and diff
                td_ev.append(f"r{j}: adapter_sha_diff={diff} (test-mode)")
            else:
                td = ckpt_tensor_compare(tdir, rdir)
                td_ok = td_ok and td["pass"]
                td_ev.append(f"r{j}: changed={td['changed_tensor_count']} "
                             f"hash_differs={td['tensor_hash']['differs']} "
                             f"finite={td['finite']['task'] and td['finite']['replay']}")
        check("replay tensor-diff" + ("(sha-level, test-mode)" if test_mode else ""),
              td_ok, "; ".join(td_ev))
    nl_ok, nl_ev = True, []
    for j in (2, 3, 4):
        a = os.path.join(ckpt_root, f"round{j}_task_llava_lora", "non_lora_trainables.bin")
        b = os.path.join(ckpt_root, f"round{j}_replay_llava_lora", "non_lora_trainables.bin")
        if not (os.path.isfile(a) and os.path.isfile(b)):
            nl_ok, nl_ev = False, nl_ev + [f"r{j}: 缺文件"]
            continue
        diff = sha256_file(a) != sha256_file(b)
        nl_ok = nl_ok and diff
        nl_ev.append(f"r{j}: sha_differ={diff}")
    check("non_lora_trainables task vs replay 不同 (r2..4)", nl_ok, "; ".join(nl_ev))
    # ---- 门 12-13：A 矩阵 10 单元 + 预测 + 独立重算交叉验证 ---------------------
    units, A = [], [[0.0] * T for _ in range(T)]
    ev_ok = True
    for j in range(1, T + 1):
        for i in range(1, j + 1):
            task = TASKS[i - 1]
            stage = os.path.join(res_root, task, f"round{j}")
            try:
                artifact_check(task, stage)
                verify_predictions(question_file(data_dir, task),
                                   os.path.join(stage, "merge.jsonl"),
                                   order_check=True)
                ru = read_unit_acc(task, res_root, j)
                A[j - 1][i - 1] = ru["accuracy"]
                units.append({"task": task, "round": j, "accuracy": ru["accuracy"],
                              "correct": ru["correct"], "count": ru["count"]})
            except Exception as e:
                ev_ok = False
                units.append({"task": task, "round": j, "error": str(e)})
                break
    check("A 矩阵单元完整（10 eval 单元 + 预测校验）", ev_ok, f"units={len(units)}")
    mtr = metrics_from_A(A) if ev_ok else None
    if ev_ok:
        dA = max(abs(a - b) for ra, rb in zip(A, A0) for a, b in zip(ra, rb))
        dmaa = abs(mtr["MAA"] - float(cm["MAA"]))
        dbwt = abs(mtr["BWT"] - float(cm["BWT"]))
        cross_ok = dA < 1e-9 and dmaa < 5e-5 and dbwt < 5e-5
    else:
        cross_ok = False
        dA = dmaa = dbwt = float("nan")
    check("MAA/BWT 独立重算 == coin_metrics.json", cross_ok,
          f"maxA={dA:.2e} dMAA={dmaa:.2e} dBWT={dbwt:.2e}")

    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        print("=== 验收失败，禁止发布（既有发布目录未动） ===")
        for name, st, ev in results:
            print(f"[{st}] {name} — {ev}")
        return 1

    # ---- 组装到 staging（全部 PASS 后才原子换入 out） ---------------------------
    layout = layout if layout else random_ratio_layout(ratio_d, args.run_id)
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.force:
        print(f"ERROR: 发布目录已存在且未 --force: {out_dir}（既有发布目录未动）")
        return 1
    os.makedirs(staging, exist_ok=True)
    try:
        shutil.copy2(os.path.join(res_root, "coin_metrics.json"),
                     os.path.join(staging, "coin_metrics.json"))
        json.dump({"ratio": float(cm["ratio"]), "units": units},
                  open(os.path.join(staging, "acc_sources.json"), "w"),
                  indent=2, ensure_ascii=False)
        json.dump(sanitize_manifest(man, repo_root),
                  open(os.path.join(staging, "run_manifest.sanitized.json"), "w"),
                  indent=2, ensure_ascii=False)
        json.dump(rsum, open(os.path.join(staging, "replay_selection_summary.json"), "w"),
                  indent=2, ensure_ascii=False)
        report = build_validation_report(args, results, A, mtr, cm, rsum, man,
                                         test_mode)
        with open(os.path.join(staging, "validation_report.md"), "w",
                  encoding="utf-8") as f:
            f.write(report)
        summary = {
            "run_id": args.run_id,
            "engine_run_id": man.get("run_id"),
            "replay_sample_seed": args.expected_seed,
            "sample_mode": cfg.get("sample_mode"),
            "ratio": cfg.get("ratio"),
            "ratio_tag": tag,
            "replay_accum": cfg.get("replay_accum"),
            "sampling_algorithm": SAMPLING_ALGORITHM,
            "MAA": round(mtr["MAA"], 4),
            "BWT": round(mtr["BWT"], 4),
            "final_avg": round(mtr["final_avg"], 4),
            "final_old_task_mean": round(mtr["final_old_task_mean"], 4),
            "diagonal_mean": round(mtr["diagonal_mean"], 4),
            "row_averages": mtr["row_averages"],
            "per_task_final_minus_diagonal": {
                t: round(v, 4) for t, v in mtr["per_task_final_minus_diagonal"].items()},
            "result_directory_rel": layout["res_root"],
            "ckpt_root_rel": layout["ckpt_root"],
            "replay_data_dir_rel": layout["replay_data_dir"],
            "code_commit": man.get("git", {}).get("commit"),
            "config_hash": man.get("config_hash"),
            "model_config_hash": man.get("model_config_hash"),
            "data_revision": man.get("data_revision"),
            "ds_config_hash": man.get("ds_config_hash"),
            "test_mode": test_mode,
            "completed_at": now_iso(),
            "deltas": None,  # 阶段 2 fill-delta 填 prefix 基线差值
            "repro": {},
        }
        for fname in ALLOWED_PUB:
            p = os.path.join(staging, fname)
            if os.path.isfile(p):
                summary["repro"][fname] = sha256_file(p)
        json.dump(summary, open(os.path.join(staging, "summary.json"), "w"),
                  indent=2, ensure_ascii=False)
        # 注：validation_report 不再追加 summary 行（避免 repro 哈希自指循环）；
        # 发布物完整性由 summary.json 的 repro 字段（覆盖其余五文件）保证
        # 敏感扫描门（六文件逐字扫；命中即 FAIL，绝不换入）。报告须含此门 → PASS 后回写
        hits = scan_export(staging)
        check("发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据）", not hits,
              "; ".join(hits) if hits else "clean")
        if hits:
            print("=== 敏感信息扫描未过，禁止发布（既有发布目录未动） ===\n")
            for h in hits:
                print(f"  HIT: {h}")
            return 1
        scan_row = "| 发布 6 文件脱敏扫描（无绝对路径/IP/主机名/凭据） | PASS | clean |\n"
        with open(os.path.join(staging, "validation_report.md"), "a",
                  encoding="utf-8") as f:
            f.write("\n" + scan_row)
        # 报告已含扫描 PASS 行 → 重算 repro（覆盖其余五文件），重写 summary
        summary["repro"] = {}
        for fname in ALLOWED_PUB:
            p = os.path.join(staging, fname)
            if os.path.isfile(p) and fname != "summary.json":
                summary["repro"][fname] = sha256_file(p)
        json.dump(summary, open(os.path.join(staging, "summary.json"), "w"),
                  indent=2, ensure_ascii=False)
        hits2 = scan_export(staging)  # summary 已更新，二次确认
        if hits2:
            print("=== 敏感信息二次扫描未过，禁止发布 ===\n")
            for h in hits2:
                print(f"  HIT: {h}")
            return 1
        # 原子换入
        if os.path.isdir(out_dir):
            stale = out_dir + f".stale.{os.getpid()}"
            os.replace(out_dir, stale)
            try:
                os.replace(staging, out_dir)
            except Exception:
                os.replace(stale, out_dir)  # 回滚
                raise
            shutil.rmtree(stale, ignore_errors=True)
        else:
            os.replace(staging, out_dir)
        print(f"导出目录（原子换入）: {out_dir}")
        for fname in sorted(os.listdir(out_dir)):
            print(f"  {fname}  {sha256_file(os.path.join(out_dir, fname))[:16]}…")
        return 0
    finally:
        if os.path.isdir(staging):  # 失败路径清理（成功时已被 rename，不触发）
            shutil.rmtree(staging, ignore_errors=True)


def scan_export(d):
    hits = []
    for fname in ALLOWED_PUB:
        p = os.path.join(d, fname)
        if not os.path.isfile(p):
            hits.append(f"{fname}: 缺失")
            continue
        try:
            text = open(p, encoding="utf-8").read()
        except Exception as e:
            hits.append(f"{fname}: 读取失败 {e}")
            continue
        for rx in SENSITIVE_RES:
            m = rx.search(text)
            if m:
                ln = text.count("\n", 0, m.start()) + 1
                hits.append(f"{fname}:{ln} 命中 {rx.pattern[:40]}… "
                            f"({m.group(0)[:40]!r})")
    return hits


def build_validation_report(args, results, A, mtr, cm, rsum, man, test_mode) -> str:
    try:
        ratio_line = (f"- expected ratio: {args.expected_ratio}"
                      f"（ratio tag {ratio_tag(args.expected_ratio)}）")
    except ValueError:
        ratio_line = f"- expected ratio: {args.expected_ratio}（非法，见检查清单）"
    lines = [f"# Validation Report — {args.run_id}", "",
             f"- 生成: {now_iso()}（random_replay_finalize.py assemble）",
             ratio_line,
             f"- run_manifest engine run_id: {man.get('run_id')}",
             f"- git commit: {man.get('git', {}).get('commit')}",
             f"- config_hash: {man.get('config_hash')}",
             f"- test_mode: {test_mode}（True=零 GPU 文件级降级，仅限 canary/单测）",
             "", "## 检查清单", "", "| 检查项 | 结果 | 证据 |", "|---|---|---|"]
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
                     f"algorithm={rj['sampling_algorithm']} "
                     f"output_sha={rj['output_sha256'][:16]}…")
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
    # 同 seed 配对差值（方向恒为 高 ratio − 低 ratio，即 random-0.10 − random-0.01）
    if args.paired_summary:
        p = json.load(open(args.paired_summary, encoding="utf-8"))
        for k in ("MAA", "BWT", "final_avg", "ratio", "replay_sample_seed"):
            if p.get(k) is None:
                raise SystemExit(f"配对 summary 缺 {k}: {args.paired_summary}")
        if p["replay_sample_seed"] != s.get("replay_sample_seed"):
            raise SystemExit(
                f"配对 summary 的 replay_sample_seed={p['replay_sample_seed']} 与本 run "
                f"{s.get('replay_sample_seed')} 不一致——配对必须同 seed（禁按 run_number 猜测）")
        if float(p["ratio"]) == float(s["ratio"]):
            raise SystemExit(f"配对 summary 与本 run 的 ratio 相同（{s['ratio']}）——"
                             "配对须为不同 ratio（0.01 vs 0.10）")
        hi, lo = ((s, p) if float(s["ratio"]) > float(p["ratio"]) else (p, s))
        direction = f"random-{float(hi['ratio']):.2f} − random-{float(lo['ratio']):.2f}"
        p_tag = p.get("ratio_tag") or ratio_tag(p["ratio"])
        out[f"vs_paired_{p_tag}"] = {
            "MAA": round(s["MAA"] - p["MAA"], 4),
            "BWT": round(s["BWT"] - p["BWT"], 4),
            "final_avg": round(s["final_avg"] - p["final_avg"], 4),
            "delta_definition": direction,
        }
        s["paired"] = {
            "paired_run_id": p.get("run_id"),
            "paired_ratio": p.get("ratio"),
            "paired_ratio_tag": p_tag,
            "paired_result_commit": args.paired_result_commit,
            "paired_code_commit": p.get("code_commit"),
            "this_run_id": s.get("run_id"),
            "this_ratio": s.get("ratio"),
            "this_ratio_tag": s.get("ratio_tag"),
            "this_result_commit": args.result_commit,
            "this_code_commit": s.get("code_commit"),
            "seed": s.get("replay_sample_seed"),
            "delta_definition": direction,
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
    p1.add_argument("--data-dir", required=True, help="源 train.json + 问题文件根")
    p1.add_argument("--run-id", required=True)
    p1.add_argument("--expected-seed", type=int, required=True)
    p1.add_argument("--expected-ratio", required=True,
                    help="本 run 的 replay 比例（random 系列只允许 0.01 / 0.10；"
                         "ratio tag 由该值派生，不接受另传 tag）")
    p1.add_argument("--repo-root", default=".")
    p1.add_argument("--out", required=True)
    p1.add_argument("--force", action="store_true")
    p1.add_argument("--test-mode", action="store_true",
                    help="零 GPU 单测用：torch 门降级为文件级并显式标注（正式发布禁止）")
    p1.set_defaults(fn=assemble)
    p2 = sub.add_parser("fill-delta")
    p2.add_argument("--summary", required=True)
    p2.add_argument("--index", required=True)
    p2.add_argument("--paired-summary", default=None,
                    help="同 seed 另一 ratio run 的 summary.json（配对差值 = 本 run − 该 run）")
    p2.add_argument("--result-commit", default=None, help="本 run 的结果提交 C（协议 C 真实 hash）")
    p2.add_argument("--paired-result-commit", default=None, help="配对 run 的结果提交 C 真实 hash")
    p2.set_defaults(fn=fill_delta)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
