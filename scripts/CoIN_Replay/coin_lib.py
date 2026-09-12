#!/usr/bin/env python3
"""CoIN+Replay 工程核心库：manifest / checkpoint 校验 / 预测校验 / 训练计划 / 轮次校验。

设计原则（TRACE 教训 + 任务规格）：
  - 所有校验失败 raise -> CLI 层非零退出，绝不吞错
  - 原子写（tmp + fsync + os.replace）
  - 零 GPU 可测：torch 为可选依赖，缺失时降级为文件级校验（显式标注）
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation

ACC_TEXT_RE = re.compile(r"Accuracy:\s*([\d.]+)%")

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def json_load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: str, obj, indent: int = 2) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git_commit_and_dirty(root: str):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        diff = subprocess.check_output(
            ["git", "diff", "--no-ext-diff"], cwd=root, stderr=subprocess.DEVNULL)
        dirty = sha256_text(diff.decode("utf-8", "replace")) if diff else "clean"
    except Exception:
        dirty = "unknown"
    return commit, dirty


def env_versions():
    out = {}
    for pkg in ("torch", "transformers", "peft", "deepspeed", "flash_attn", "vllm"):
        try:
            out[pkg] = subprocess.check_output(
                [sys.executable, "-c", f"import {pkg}; print({pkg}.__version__)"],
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            out[pkg] = "unavailable"
    try:
        import platform
        out["python"] = platform.python_version()
    except Exception:
        out["python"] = "unknown"
    out["python_full"] = sys.version.strip().replace("\n", " ")
    try:
        import torch
        out["cuda"] = torch.version.cuda or "unknown"
    except Exception:
        out["cuda"] = "unavailable"
    return out


# ---------------------------------------------------------------------------
# 训练分辨率报告（工单 8）
# ---------------------------------------------------------------------------

def per_rank_micro_batches(N: int, batch: int = 14, world: int = 4) -> list:
    """BatchSamplerShard（accelerate 0.21, split_batches=False, even_batches=True,
    drop_last=False）下每个 rank 实际 yield 的 micro-batch 数。

    语义来源：DataLoader(batch_size, sampler=LengthGroupedSampler) →
    accelerator.prepare → BatchSamplerShard._iter_with_no_split。even 补齐不保证
    rank 均匀（尾部 partial batch 的持有 rank 可能被跳过）→ 常见 ±1 不均
    （如 N=1272 → [23,23,22,23]）。返回 per-rank yield 计数。
    2026-09-04 审计移植（audit/matrix_static_sim.py 同逻辑，逐行对照源码）。
    """
    if N <= 0:
        return [0] * world
    B = -(-N // batch)  # 底层 BatchSampler 产出的 batch 数
    # batch 长度序列：前 B-1 个满长；最后 = N - batch*(B-1)（整除时也满长）
    blen = [batch] * (B - 1)
    blen.append(N - batch * (B - 1) if N % batch else batch)

    counts = []
    for p in range(world):
        cnt = 0
        bty = None  # 本进程最后持有的 batch 长度
        initial_n = sum(blen[:min(world, B)])  # initial_data 样本数（内容无关，仅数量）
        for idx, ln in enumerate(blen):
            if idx % world == p:
                bty = ln
            if idx % world == world - 1 and ln == batch:
                cnt += 1
                bty = None
        if initial_n > 0:
            if bty == batch:
                cnt += 1  # 尾部 a：满长 bty 补 yield
            # 尾部回绕补齐（源码 even_batches 分支）：
            # 源码: for 结束 idx=B-1 → 满长尾: batch=[]; idx+=1 → idx=B（从 B 起补）
            #       partial 尾: idx 保持 B-1（partial batch 续 initial_data 补满，从 B-1 起补）
            idx2 = B if blen[-1] == batch else B - 1
            carry = 0 if blen[-1] == batch else blen[-1]  # 最后 partial batch 长度
            while idx2 % world != 0 or carry > 0:
                if idx2 % world == p:
                    cnt += 1
                carry = 0
                idx2 += 1
        counts.append(cnt)
    return counts


def train_plan(data_json: str, batch: int, accum: int, world: int, lr: float,
               warmup_ratio: float, epochs: float, name: str,
               replay_k: int = None, max_len: int = None) -> dict:
    data = json_load(data_json)
    N = len(data)
    if N == 0:
        raise ValueError(f"train_plan: {data_json} 为空")
    total_batch = batch * accum * world
    micro = per_rank_micro_batches(N, batch, world)  # per-rank micro（BatchSamplerShard 语义）
    m_min, m_max = min(micro), max(micro)
    # HF 语义（每 rank 独立 len//gas，兜底 1）——rank0 口径
    hf_steps_per_epoch = max(1, (m_min * epochs) // accum) if epochs >= 1 else 0
    hf_steps = max(1, -(-(m_max * epochs) // accum))  # 旧 ceil 口径（多 rank 取 max）
    # DS 权威口径：rank 齐步共同完成的真步数 = floor(min_micro/gas)
    ds_updates = int((m_min * epochs) // accum)
    max_steps = hf_steps  # 保留旧语义（门禁 flag 兼容）
    warmup_steps = int(warmup_ratio * max_steps)
    if warmup_steps > 0:
        first_step_lr = lr * (1.0 / warmup_steps)  # 首个更新步的实际 LR（线性 warmup）
    else:
        first_step_lr = lr
    consumed = N * epochs
    yield_samples = sum(micro) * batch
    plan = {
        "name": name,
        "N": N,
        "replay_k": replay_k,
        "per_rank_microbatch": batch,
        "world_size": world,
        "grad_accum": accum,
        "total_train_batch_size": total_batch,
        "effective_batch": total_batch,
        "optimizer_steps": max_steps,   # 旧字段（HF ceil 口径，保留兼容门禁）
        "steps_per_epoch": hf_steps_per_epoch,
        # ---- 2026-09-04 审计新增：准确多口径报告（禁止单一 ceil 掩盖语义）----
        "per_rank_micro": micro,                         # BatchSamplerShard 实测语义
        "per_rank_micro_min": m_min,
        "per_rank_micro_max": m_max,
        "microbatch_remainder": (m_min * epochs) % accum,  # 尾部未满 accum 的 micro
        "discarded_or_uncommitted_microbatches": int(
            (m_min * epochs) - ds_updates * accum + (m_max - m_min) * epochs),
        "sampler_padding": max(0, yield_samples - N),      # even 补齐重复样本数
        "hf_planned_steps": max(1, hf_steps_per_epoch),    # HF 计划（rank0/min 口径）
        "ds_expected_updates": ds_updates,                 # DS 权威真步（rank 齐步下界）
        "warmup_steps": warmup_steps,
        "lr": lr,
        "first_step_lr": round(first_step_lr, 10),
        "last_step_lr": round(lr, 10),
        "consumed_samples": int(consumed),
        "max_len": max_len,
        "flag_replay_single_step": ds_updates <= 1,   # 改以 DS 真步判 single-step（原 HF ceil 高估）
        "flag_first_lr_zero": first_step_lr == 0.0,
        "flag_warmup_covers_all": warmup_steps >= max_steps,
    }
    return plan


# ---------------------------------------------------------------------------
# checkpoint 校验（工单 4）
# ---------------------------------------------------------------------------

CKPT_REQUIRED = ["adapter_config.json", "non_lora_trainables.bin", "config.json"]


def _load_adapter_tensors(adapter_path: str):
    """torch.load adapter 权重；仅接受全 tensor state dict（结构异常抛错，不静默降级）。"""
    import torch
    w = torch.load(adapter_path, map_location="cpu")
    if not isinstance(w, dict):
        raise ValueError(f"{adapter_path}: adapter 非 state dict（{type(w).__name__}）")
    bad = [k for k, v in w.items() if not isinstance(v, torch.Tensor)]
    if bad:
        raise ValueError(f"{adapter_path}: 含非 tensor 值 {len(bad)} 个: {bad[:5]}")
    return w


def tensor_bytes_sha256(state: dict) -> str:
    """规范化 tensor hash：sorted keys，key 名 + 张量原始字节。

    用 uint8 视图取字节（不改值、不依赖 numpy 对 bf16 的支持——原 numpy()
    在 bf16 上抛 TypeError，导致真实 checkpoint 的 param_hash 静默降级）。
    """
    import torch
    buf = hashlib.sha256()
    for k in sorted(state.keys()):
        buf.update(k.encode("utf-8"))
        t = state[k].detach().cpu().contiguous()
        buf.update(t.view(torch.uint8).numpy().tobytes())
    return buf.hexdigest()


def _adapter_state_report(adapter_path: str) -> dict:
    import torch
    w = _load_adapter_tensors(adapter_path)
    return {
        "n_tensors": len(w),
        "dtype": sorted({str(v.dtype) for v in w.values()}),
        "finite": all(bool(torch.isfinite(v).all().item()) for v in w.values()),
        "state": w,
        "hash": tensor_bytes_sha256(w),
    }


def ckpt_validate(ckpt_dir: str, require_torch: bool = True) -> dict:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"checkpoint 目录不存在: {ckpt_dir}")
    files = sorted(os.listdir(ckpt_dir))
    adapter = None
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if os.path.isfile(os.path.join(ckpt_dir, name)):
            adapter = name
            break
    if adapter is None:
        raise FileNotFoundError(f"{ckpt_dir} 缺少 adapter_model.(safetensors|bin)，实际文件: {files}")
    for req in CKPT_REQUIRED:
        p = os.path.join(ckpt_dir, req)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{ckpt_dir} 缺少 {req}")
        if os.path.getsize(p) == 0:
            raise ValueError(f"{ckpt_dir} 的 {req} 为空文件")
    report = {"ckpt_dir": ckpt_dir, "adapter": adapter,
              "files": {f: sha256_file(os.path.join(ckpt_dir, f))
                        for f in files if os.path.isfile(os.path.join(ckpt_dir, f))}}
    torch = None
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        report["param_hash"] = None
        report["finite"] = None
        report["note"] = "torch 不可用，仅文件级校验"
        return report
    # 参数级校验：加载 adapter 权重，检查 finite + 计算参数 hash
    adapter_path = os.path.join(ckpt_dir, adapter)
    # 尺寸守卫（评审 2026-09-02 方案 A）：真实 LoRA adapter（7B r=192 数百 MB、r=8 也 ≥1MB）
    # 不可能 <1MB；DRY_RUN 假文件（64B 随机字节）在 torch 存在时随机抛
    # ValueError(unsupported pickle protocol) 或 UnpicklingError，造成 DRY_RUN 测试 ~25% flake。
    # 此处对 <1MB 文件确定性降级为文件级校验（真实验证严格性零损失）。
    if os.path.getsize(adapter_path) < 1 << 20:
        report["param_hash"] = None
        report["finite"] = None
        report["note"] = (f"adapter 文件 {os.path.getsize(adapter_path)}B <1MB，"
                          f"视为非真实 checkpoint（DRY_RUN 假文件/占位），仅文件级校验")
        return report
    try:
        rep = _adapter_state_report(adapter_path)
        report["param_hash"] = rep["hash"]
        report["finite"] = rep["finite"]
        report["n_tensors"] = rep["n_tensors"]
        report["dtype"] = rep["dtype"]
        if not rep["finite"]:
            raise ValueError(f"{ckpt_dir} 的 adapter 权重含 NaN/Inf")
    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        report["param_hash"] = None
        report["finite"] = None
        report["note"] = f"权重解析失败({type(e).__name__})，已降级为文件级校验"
    return report


# ---------------------------------------------------------------------------
# tensor 级 checkpoint 比较（canary C 评审 2026-09-02：task vs replay）
# ---------------------------------------------------------------------------

def ckpt_tensor_compare(task_dir: str, replay_dir: str) -> dict:
    """tensor 级比较 task 与 replay adapter 权重。

    pass 仅当（全部满足，禁止用 metadata/目录名/mtime/JSON 差异替代）：
      - keys 完全一致（无 missing/unexpected）
      - 各 key shape 一致
      - 全部 tensor finite
      - changed_tensor_count >= 1（至少一个 tensor 的值不同）
      - replay 规范化 tensor hash != task 规范化 tensor hash
    结构性问题（缺文件/非 tensor/无 torch）raise（CLI exit 2）；
    结论性失败（tensor 相同/结构不符）返回 pass=False（CLI exit 1）。
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("ckpt-tensor-diff 需要 torch（canary 环境已安装）")

    def _adapter_file(d):
        for name in ("adapter_model.safetensors", "adapter_model.bin"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"{d} 缺少 adapter_model.(safetensors|bin)")

    tp, rp = _adapter_file(task_dir), _adapter_file(replay_dir)
    trep = _adapter_state_report(tp)
    rrep = _adapter_state_report(rp)
    tw, rw = trep.pop("state"), rrep.pop("state")
    tkeys, rkeys = set(tw), set(rw)
    missing = sorted(tkeys - rkeys)
    unexpected = sorted(rkeys - tkeys)
    shared = sorted(tkeys & rkeys)
    shape_mismatch = [k for k in shared if tuple(tw[k].shape) != tuple(rw[k].shape)]
    changed = []
    l2_sq = 0.0
    max_abs = 0.0
    for k in shared:
        if tuple(tw[k].shape) != tuple(rw[k].shape):
            continue
        if not torch.equal(tw[k], rw[k]):
            changed.append(k)
        d = (tw[k].float() - rw[k].float())
        l2_sq += float(d.pow(2).sum().item())
        m = float(d.abs().max().item())
        if m > max_abs:
            max_abs = m
    keys_ok = not missing and not unexpected and not shape_mismatch
    finite_ok = trep["finite"] and rrep["finite"]
    hash_diff = trep["hash"] != rrep["hash"]
    verdict_pass = bool(keys_ok and finite_ok and changed and hash_diff)
    return {
        "pass": verdict_pass,
        "task_dir": task_dir,
        "replay_dir": replay_dir,
        "keys": {
            "task_n": len(tkeys), "replay_n": len(rkeys),
            "missing": missing, "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
        },
        "finite": {"task": trep["finite"], "replay": rrep["finite"]},
        "dtype": {"task": trep["dtype"], "replay": rrep["dtype"]},
        "changed_tensor_count": len(changed),
        "changed_tensors": changed[:20],
        "l2_norm_diff": round(l2_sq ** 0.5, 6),
        "max_abs_diff": round(max_abs, 6),
        "tensor_hash": {"task": trep["hash"], "replay": rrep["hash"],
                        "differs": hash_diff},
    }



# ---------------------------------------------------------------------------
# 预测校验（工单 5）
# ---------------------------------------------------------------------------

def load_ids_from_questions(question_file: str):
    data = json_load(question_file)
    ids = []
    for line in data:
        if "question_id" in line:
            ids.append(line["question_id"])
        elif "id" in line:
            ids.append(line["id"])
        else:
            raise ValueError(f"{question_file} 条目缺少 question_id/id: {str(line)[:100]}")
    return ids


def load_ids_from_predictions(predictions_file: str):
    ids = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question_id" not in obj:
                raise ValueError(f"{predictions_file}:{ln} 缺少 question_id")
            ids.append(obj["question_id"])
    return ids


def verify_predictions(question_file: str, predictions_file: str, order_check: bool = True) -> dict:
    qids = load_ids_from_questions(question_file)
    pids = load_ids_from_predictions(predictions_file)
    report = {"question_count": len(qids), "prediction_count": len(pids),
              "unique_prediction_ids": len(set(pids))}
    if len(pids) != len(qids):
        raise ValueError(
            f"预测数 {len(pids)} != 问题数 {len(qids)}"
            f"（缺失 chunk 或重复？）question={question_file}")
    if len(set(pids)) != len(pids):
        raise ValueError(f"预测存在重复 question_id: {len(pids) - len(set(pids))} 个")
    if set(pids) != set(qids):
        missing = sorted(set(qids) - set(pids))
        extra = sorted(set(pids) - set(qids))
        raise ValueError(f"预测 ID 集合与问题不一致: 缺 {len(missing)} 多 {len(extra)}")
    if order_check and pids != qids:
        raise ValueError("预测顺序与问题文件顺序不一致（chunk 拼接乱序？）")
    return report


# ---------------------------------------------------------------------------
# 评估产物校验（工单 1/5）
# ---------------------------------------------------------------------------

def artifact_check(task: str, stage_dir: str) -> dict:
    if task == "ScienceQA":
        f = os.path.join(stage_dir, "output_result.jsonl")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺失 {f}")
        data = json_load(f)
        if "acc" not in data:
            raise ValueError(f"{f} 无 acc 字段")
        acc = float(data["acc"])
    else:
        f = os.path.join(stage_dir, "Result.text")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺失 {f}")
        text = open(f, encoding="utf-8").read()
        m = ACC_TEXT_RE.search(text)
        if not m:
            raise ValueError(f"{f} 无 Accuracy: xx.xx%")
        acc = float(m.group(1))
    if not (0.0 <= acc <= 100.0):
        raise ValueError(f"{task}@{stage_dir} 准确率越界: {acc}")
    return {"task": task, "stage": os.path.basename(stage_dir),
            "acc": acc, "artifact": f, "sha256": sha256_file(f)}


# ---------------------------------------------------------------------------
# random-replay ratio 通用化（2026-09-11）
#
# 单一来源：ratio 归一化 / ratio tag / 三目录布局只在本节实现。gate（run_replay_exp.sh）、
# finalize、registry、配对工具全部调用这里，禁止各自硬编码 "r001" 或 0.01——否则操作者
# 需要同时手填 ratio 与 tag，必然出现不一致。
# ---------------------------------------------------------------------------

# 正式 random 系列只允许这两个回放比例（数值集合；输入 "0.1" / "0.10" / 0.01 均合法）
ALLOWED_RANDOM_RATIOS = (Decimal("0.01"), Decimal("0.10"))


def normalize_ratio(value) -> Decimal:
    """把 ratio 表示（str/float/int，如 "0.1"/"0.10"/0.01）规范化为允许集合中的 Decimal。

    判定用数值比较（Decimal(str(v))），不做字符串比较——"0.1"、"0.10"、0.10 等价。
    不在允许集合 → ValueError（调用方必须非零退出）。
    """
    if isinstance(value, str):
        s = value.strip()
    elif isinstance(value, (int, float, Decimal)):
        s = str(value)
    else:
        raise ValueError(f"ratio 非法（类型 {type(value).__name__}）: {value!r}")
    try:
        d = Decimal(s)
    except (InvalidOperation, ArithmeticError, ValueError):
        raise ValueError(f"ratio 非法（无法解析为数值）: {value!r}")
    if not d.is_finite():
        raise ValueError(f"ratio 非法（非有限数值）: {value!r}")
    for allowed in ALLOWED_RANDOM_RATIOS:
        if d == allowed:
            return allowed
    raise ValueError(
        f"ratio {value!r} 不在 random 系列允许集合 "
        f"{[str(a) for a in ALLOWED_RANDOM_RATIOS]}（只允许 0.01 与 0.10）")


def ratio_tag(value) -> str:
    """由 ratio 数值派生 ratio tag：0.01→r001、0.10→r010（禁手工填 tag）。"""
    r = normalize_ratio(value)
    return "r%03d" % int((r * 100).to_integral_value())


def random_ratio_layout(value, run_id: str) -> dict:
    """三个正式目录的相对路径（tag 由 ratio 派生；不允许调用方另传 tag）。

    ratio=0.01: checkpoints|results/CoIN_Replay_random/r001/<run_id>、
                playground/Replay_random/r001/<run_id>
    ratio=0.10: 同布局，tag=r010
    """
    tag = ratio_tag(value)
    return {
        "ratio": str(normalize_ratio(value)),
        "ratio_tag": tag,
        "ckpt_root": f"checkpoints/CoIN_Replay_random/{tag}/{run_id}",
        "res_root": f"results/CoIN_Replay_random/{tag}/{run_id}",
        "replay_data_dir": f"playground/Replay_random/{tag}/{run_id}",
    }


# ---------------------------------------------------------------------------
# eval 路径门禁（2026-09-11）：训练前确认评估阶段用到的模型/数据路径可解析
#
# 背景（实踩）：四个 eval shell 与 eval_gqa.py 曾硬编码相对路径（./checkpoints、./cl_dataset、
# ./playground、./cl_dataset/GQA），新 worktree 缺这些软链时会在**数小时训练后**的评估阶段
# 才失败。现在：① 全部改为可覆盖（shell 用 ${VAR:-默认}，eval_gqa 增 --data-root）；
# ② 本门禁在训练前校验——暴露的覆盖钩子仍在、传入路径真实存在、文件里残余的相对路径
# 必须能在 run 根下解析（新增硬编码路径会被当场拦下）。
# ---------------------------------------------------------------------------

EVAL_SHELL_SCRIPTS = ("1_eval_sqa.sh", "2_eval_textqa.sh", "3_eval_ImageNet.sh", "4_eval_gqa.sh")
EVAL_PY_MODULES = ("eval_science_qa.py", "eval_textvqa.py", "eval_ImagetNet.py",
                   "eval_gqa.py", "convert_gqa_for_eval.py", "model_vqa.py",
                   "model_vqa_science.py", "model_text_vqa.py", "model_gqa.py")
REL_REF_RE = re.compile(r"\./[A-Za-z0-9_./@+-]+")
SHELL_DEFAULT_RE = re.compile(r"\$\{[A-Z_][A-Z0-9_]*:-")
# 形如 MODELPATH='./checkpoints/...'：位置参数（$2）覆盖的默认值分支，编排脚本必传 $2
SHELL_POSITIONAL_DEFAULT_RE = re.compile(r"^[A-Z_]+='?\./")
# python 侧：任何含 default= 的行视为「可覆盖默认值行」（add_argument 可能跨多行书写）；
# 注意：help 文本里不要出现字面相对路径（本门禁按原样扫描，出现即视为硬编码）
PY_DEFAULT_RE = re.compile(r"default\s*=")


def _strip_comment(line: str, is_python: bool) -> str:
    """去掉纯注释行与行尾注释（简化规则：'#' 之后一律视为注释；本门禁的路径不含 '#'）。"""
    s = line.strip()
    if s.startswith("#"):
        return ""
    i = line.find("#")
    return line[:i] if i >= 0 else line


def eval_path_audit(root: str, model_base: str, image_folder: str,
                    scripts_dir=None, py_dir=None) -> dict:
    """训练前门禁：eval 阶段的模型/数据路径必须可解析。

    检查项：
      1. 四个 eval shell 必须保留 MODEL_BASE / IMAGE_FOLDER 覆盖钩子
      2. eval_gqa.py 必须保留 --data-root 覆盖（原先硬编码 './cl_dataset/GQA'）
      3. model_base / image_folder 必须真实存在（目录）
      4. 文件里出现的相对路径：位于「可覆盖默认值行」（shell 的 ${VAR:-...}、python 的
         add_argument(default=...)）→ 视为由覆盖机制处理；其余相对路径必须在 root 下存在
    返回 JSON 报告（pass / 各项明细）；调用方非零退出即 fail-fast。
    """
    root = os.path.abspath(root)
    scripts_dir = scripts_dir or os.path.join(root, "scripts", "LLaVA", "Eval")
    py_dir = py_dir or os.path.join(root, "ETrain", "Eval", "LLaVA", "CoIN")
    rep = {"root": root, "model_base": model_base, "image_folder": image_folder,
           "hooks": {}, "missing_paths": [], "uncovered_relative_refs": [],
           "covered_relative_refs": [], "files_scanned": []}

    for name in EVAL_SHELL_SCRIPTS:
        p = os.path.join(scripts_dir, name)
        if not os.path.isfile(p):
            rep["hooks"][name] = {"exists": False, "model_base_hook": False,
                                  "image_folder_hook": False}
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        rep["hooks"][name] = {"exists": True,
                              "model_base_hook": "${MODEL_BASE:-" in txt,
                              "image_folder_hook": "${IMAGE_FOLDER:-" in txt,
                              # MODELPATH 由位置参数 $2 覆盖（编排脚本恒传 ckpt 路径）
                              "positional_modelpath_hook": "MODELPATH=$2" in txt}
        _scan_refs(p, root, rep, is_python=False)
        rep["files_scanned"].append(os.path.relpath(p, root))
    gqa = os.path.join(py_dir, "eval_gqa.py")
    if os.path.isfile(gqa):
        txt = open(gqa, encoding="utf-8", errors="replace").read()
        rep["hooks"]["eval_gqa.py"] = {"exists": True,
                                       # 精确匹配 add_argument('--data-root' / add_argument("--data-root"
                                       "data_root_arg": bool(re.search(
                                           r"add_argument\(\s*['\"]--data-root['\"]", txt)),
                                       "data_root_used": "args.data_root" in txt}
        _scan_refs(gqa, root, rep, is_python=True)
        rep["files_scanned"].append(os.path.relpath(gqa, root))
    for name in EVAL_PY_MODULES:
        p = os.path.join(py_dir, name)
        if os.path.isfile(p):
            _scan_refs(p, root, rep, is_python=True)
            rep["files_scanned"].append(os.path.relpath(p, root))

    for p in (model_base, image_folder):
        if not p or not os.path.isdir(p):
            rep["missing_paths"].append(p or "<空>")

    hooks_ok = all(v.get("model_base_hook") and v.get("image_folder_hook")
                   and v.get("positional_modelpath_hook")
                   for k, v in rep["hooks"].items() if k in EVAL_SHELL_SCRIPTS) and \
        all(v.get("exists") for v in rep["hooks"].values())
    gqa_ok = rep["hooks"].get("eval_gqa.py", {}).get("data_root_arg", False) and \
        rep["hooks"].get("eval_gqa.py", {}).get("data_root_used", False)
    rep["pass"] = bool(hooks_ok and gqa_ok and not rep["missing_paths"]
                       and not rep["uncovered_relative_refs"])
    rep["hooks_ok"] = bool(hooks_ok)
    rep["eval_gqa_data_root_ok"] = bool(gqa_ok)
    return rep


def _scan_refs(path: str, root: str, rep: dict, is_python: bool) -> None:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = _strip_comment(raw, is_python)
            if not line:
                continue
            is_default_line = (PY_DEFAULT_RE.search(line) if is_python
                               else (SHELL_DEFAULT_RE.search(line)
                                     or SHELL_POSITIONAL_DEFAULT_RE.search(line.strip())))
            for m in REL_REF_RE.finditer(line):
                ref = m.group(0)
                entry = {"file": os.path.relpath(path, root), "line": lineno, "ref": ref}
                if is_default_line:
                    rep["covered_relative_refs"].append(entry)
                elif os.path.exists(os.path.join(root, ref)):
                    rep["covered_relative_refs"].append(dict(entry, resolvable=True))
                else:
                    rep["uncovered_relative_refs"].append(entry)


# ---------------------------------------------------------------------------
# manifest（工单 6）
# ---------------------------------------------------------------------------

CONFIG_FIELDS = [
    "ratio", "tasks", "T", "model_base", "vision_tower", "projector",
    "lora_r", "lora_alpha", "lora_dropout", "lr", "mm_projector_lr",
    "epochs_per_task", "replay_epochs", "seed", "data_seed", "sample_mode",
    "replay_sample_seed", "random_replay_run_id",
    "per_device_batch", "grad_accum", "replay_accum", "world_size", "effective_batch",
    "replay_effective_batch", "allow_single_step_replay",
    "lr_scheduler_type", "warmup_ratio", "precision", "grad_ckpt",
    "ds_config", "gpus", "model_max_length", "temperature_eval",
]


def compute_config(env: dict) -> dict:
    def get(key, default=None, cast=None, required=False):
        v = env.get(key)
        if v is None or v == "":
            if required:
                raise KeyError(f"缺少必需环境变量 {key}")
            return default
        return cast(v) if cast else v

    world = len(str(get("GPUS", "0,1,2,3")).split(","))
    batch = int(get("BATCH", "14", required=True))
    accum = int(get("ACCUM", "16", required=True))
    replay_accum_raw = get("REPLAY_ACCUM")  # 空=未设置（replay 段继承 task accum）
    replay_accum = int(replay_accum_raw) if replay_accum_raw else None
    cfg = {
        "ratio": float(get("RATIO", required=True)),
        "tasks": json.loads(get("TASKS_JSON", '["ScienceQA","TextVQA","ImageNet","GQA"]')),
        "model_base": get("BASE_MODEL", required=True),
        "vision_tower": get("VISION_TOWER", required=True),
        "projector": get("PROJECTOR", required=True),
        "lora_r": int(get("LORA_R", "192")),
        "lora_alpha": int(get("LORA_ALPHA", "256")),
        "lora_dropout": float(get("LORA_DROPOUT", "0.05")),
        "lr": float(get("LR", "2e-4")),
        "mm_projector_lr": float(get("MM_PROJECTOR_LR", "2e-5")),
        "epochs_per_task": float(get("EPOCHS", "1")),
        "replay_epochs": float(get("REPLAY_EPOCHS", "1")),
        "seed": int(get("SEED", "1234")),
        "data_seed": int(get("DATA_SEED", "1234")),
        "sample_mode": get("SAMPLE_MODE", "prefix"),
        "replay_sample_seed": int(get("REPLAY_SAMPLE_SEED", "1234")),
        "per_device_batch": batch,
        "grad_accum": accum,
        "replay_accum": replay_accum,
        "world_size": world,
        "effective_batch": batch * accum * world,
        "replay_effective_batch": (batch * world * replay_accum) if replay_accum else None,
        "allow_single_step_replay": int(get("ALLOW_SINGLE_STEP_REPLAY", "0")),
        "lr_scheduler_type": get("LR_SCHEDULER_TYPE", "cosine"),
        "warmup_ratio": float(get("WARMUP_RATIO", "0.03")),
        "precision": get("PRECISION", "bf16+tf32"),
        "grad_ckpt": get("GRAD_CKPT", "true") == "true",
        "ds_config": get("DS_CONFIG", required=True),
        "gpus": get("GPUS", required=True),
        "model_max_length": int(get("MODEL_MAX_LENGTH", "2048")),
        "temperature_eval": float(get("EVAL_TEMPERATURE", "0")),
    }
    cfg["T"] = len(cfg["tasks"])
    # random 模式强制 run 身份：RANDOM_REPLAY_RUN_ID（run_NNNN_seed_<seed>）进 config hash，
    # 恢复运行/跨 run 混用校验自动覆盖（prefix 模式不加此键，保持旧兼容）
    if cfg["sample_mode"] == "random":
        rid = get("RANDOM_REPLAY_RUN_ID", required=True)
        m = re.fullmatch(r"run_\d{4}_seed_(\d+)", rid)
        if not m:
            raise ValueError(f"RANDOM_REPLAY_RUN_ID 格式非法: {rid!r}"
                             "（期望 run_NNNN_seed_<seed>，NNNN=4 位序号）")
        if int(m.group(1)) != cfg["replay_sample_seed"]:
            raise ValueError(f"RANDOM_REPLAY_RUN_ID 内嵌 seed ({m.group(1)}) 与 "
                             f"REPLAY_SAMPLE_SEED ({cfg['replay_sample_seed']}) 不一致")
        cfg["random_replay_run_id"] = rid
    return cfg


def config_hash(cfg: dict) -> str:
    canon = {k: cfg[k] for k in CONFIG_FIELDS if k in cfg}
    return sha256_text(json.dumps(canon, sort_keys=True, ensure_ascii=False))


def manifest_enrich(cfg: dict, root: str) -> dict:
    commit, dirty = git_commit_and_dirty(root)
    env = env_versions()
    m = {
        "run_id": f"coin_replay_r{cfg['ratio']}_{time.strftime('%Y%m%d_%H%M%S')}",
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git": {"commit": commit, "dirty_diff_hash": dirty},
        "env": env,
        "data_revision": os.environ.get("DATA_SHA256", "unknown"),
        "model_config_hash": os.environ.get("MODEL_CONFIG_HASH", "unknown"),
        "ds_config_hash": os.environ.get("DS_CONFIG_SHA256", "unknown"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return m


def manifest_write(res_root: str, cfg: dict, root: str, force: bool = False,
                   resume_ok: bool = False) -> str:
    path = os.path.join(res_root, "run_manifest.json")
    if os.path.exists(path):
        if resume_ok:
            # 恢复运行：校验 config hash，除运行态字段外不一致即失败
            existing = json_load(path)
            if existing.get("config_hash") != config_hash(cfg):
                old_cfg = existing.get("config", {})
                diffs = {k: {"existing": old_cfg.get(k), "new": cfg.get(k)}
                         for k in ("sample_mode", "replay_sample_seed", "ratio")
                         if old_cfg.get(k) != cfg.get(k)}
                msg = ("run_manifest.json 已存在且 config hash 不一致——恢复运行配置与首次运行"
                       "不同，禁止覆盖（sample mode 或 replay sample seed 不一致时禁止恢复到"
                       "同一结果目录）。首次配置见 manifest 的 config 字段。")
                if diffs:
                    msg += " 语义差异字段: " + json.dumps(diffs, ensure_ascii=False)
                raise ValueError(msg)
            return path
        if not force:
            raise FileExistsError(
                f"{path} 已存在；覆盖需 --force（配置快照不可被恢复运行覆盖）")
    m = manifest_enrich(cfg, root)
    atomic_write_json(path, m)
    return path


# ---------------------------------------------------------------------------
# 轮次校验 / round manifest（工单 6）
# ---------------------------------------------------------------------------

def validate_round(res_root: str, tasks: list, j: int, ckpt_dir: str,
                   replay_data: str = None) -> dict:
    """.round<j>_done 存在不等于成功：跳过前必须通过本校验。"""
    errors = []
    rm = os.path.join(res_root, f"round{j}_manifest.json")
    if not os.path.isfile(rm):
        errors.append(f"缺失 round manifest: {rm}")
    try:
        ckpt_validate(ckpt_dir)
    except Exception as e:
        errors.append(f"checkpoint 校验失败: {e}")
    for i in range(1, j + 1):
        task = tasks[i - 1]
        try:
            artifact_check(task, os.path.join(res_root, task, f"round{j}"))
        except Exception as e:
            errors.append(str(e))
    if j >= 2 and replay_data:
        if not os.path.isfile(replay_data):
            errors.append(f"缺失 replay 数据: {replay_data}")
        elif not os.path.isfile(replay_data + ".manifest.json"):
            errors.append(f"缺失 replay sidecar manifest: {replay_data}.manifest.json")
    if errors:
        raise ValueError("validate_round 失败:\n  " + "\n  ".join(errors))
    return {"round": j, "round_manifest": rm, "ckpt": ckpt_dir, "ok": True}


def round_manifest_write(res_root: str, j: int, info: dict) -> str:
    path = os.path.join(res_root, f"round{j}_manifest.json")
    atomic_write_json(path, {
        "round": j, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info,
    })
    return path


# ---------------------------------------------------------------------------
# cross-run manifest 校验（方案 D：0.1 与 0.01 除 ratio 外完全一致）
# ---------------------------------------------------------------------------

CROSS_ALLOWED_DIFF = {"ratio"}  # 唯一允许的 run_manifest config 差异字段


def manifest_cross_check(res_a: str, res_b: str) -> dict:
    """比较两个结果目录的 run_manifest.json config。断言：除 ratio（与派生输出路径）
    外所有 CONFIG_FIELDS 完全一致——即 0.1/0.01 的 replay_accum、batch、LR、
    scheduler、seed、模型、数据、代码 hash 相同。"""
    def load(p):
        path = os.path.join(p, "run_manifest.json")
        if not os.path.isfile(path):
            raise ValueError(f"缺 run_manifest.json: {path}")
        return json_load(path)

    ma, mb = load(res_a), load(res_b)
    ca, cb = ma["config"], mb["config"]
    diffs = {}
    for k in CONFIG_FIELDS:
        if k in CROSS_ALLOWED_DIFF:
            continue
        if ca.get(k) != cb.get(k):
            diffs[k] = {"a": ca.get(k), "b": cb.get(k)}
    common = set(ca) & set(cb)
    only_a = set(ca) - set(cb)
    only_b = set(cb) - set(ca)
    return {
        "pass": not diffs and not only_a and not only_b,
        "diffs": diffs,
        "config_keys_only_a": sorted(only_a),
        "config_keys_only_b": sorted(only_b),
        "ratio_a": ca.get("ratio"), "ratio_b": cb.get("ratio"),
        "replay_accum_a": ca.get("replay_accum"), "replay_accum_b": cb.get("replay_accum"),
        "replay_effective_batch_a": ca.get("replay_effective_batch"),
        "replay_effective_batch_b": cb.get("replay_effective_batch"),
        "config_hash_a": ma.get("config_hash"), "config_hash_b": mb.get("config_hash"),
    }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    cmd = sys.argv[1]
    if cmd == "train-plan":
        args = _kv(sys.argv[2:])
        plan = train_plan(
            args["--data-json"], int(args["--batch"]), int(args["--accum"]),
            int(args["--world"]), float(args["--lr"]), float(args["--warmup-ratio"]),
            float(args["--epochs"]), args.get("--name", "train"),
            replay_k=int(args["--replay-k"]) if args.get("--replay-k") else None)
        print(json.dumps(plan, ensure_ascii=False))
    elif cmd == "ckpt-validate":
        rep = ckpt_validate(sys.argv[2], require_torch=not _flag(sys.argv[3:], "--no-torch"))
        print(json.dumps(rep, ensure_ascii=False))
    elif cmd == "ckpt-tensor-diff":
        try:
            rep = ckpt_tensor_compare(sys.argv[2], sys.argv[3])
        except Exception as e:
            print(json.dumps({"pass": False, "structural_error": str(e),
                              "task_dir": sys.argv[2], "replay_dir": sys.argv[3]},
                             ensure_ascii=False))
            sys.exit(2)
        print(json.dumps(rep, ensure_ascii=False))
        sys.exit(0 if rep["pass"] else 1)
    elif cmd == "verify-predictions":
        a = _kv(sys.argv[2:])
        rep = verify_predictions(a["--questions"], a["--predictions"],
                                 order_check=a.get("--order-check", "1") != "0")
        print(json.dumps(rep, ensure_ascii=False))
    elif cmd == "artifact-check":
        a = _kv(sys.argv[2:])
        rep = artifact_check(a["--task"], a["--stage-dir"])
        print(json.dumps(rep, ensure_ascii=False))
    elif cmd == "config":
        cfg = compute_config(os.environ)
        print(json.dumps(cfg, ensure_ascii=False))
    elif cmd == "manifest-write":
        a = _kv(sys.argv[2:])
        cfg = compute_config(os.environ)
        path = manifest_write(a["--res-root"], cfg, a.get("--root", "."),
                              force=_flag(sys.argv[2:], "--force"),
                              resume_ok=_flag(sys.argv[2:], "--resume-ok"))
        print(json.dumps({"path": path, "config_hash": config_hash(cfg),
                          "effective_batch": cfg["effective_batch"]}))
    elif cmd == "manifest-resume-check":
        a = _kv(sys.argv[2:])
        cfg = compute_config(os.environ)
        manifest_write(a["--res-root"], cfg, a.get("--root", "."), resume_ok=True)
        print(json.dumps({"ok": True, "config_hash": config_hash(cfg)}))
    elif cmd == "validate-round":
        a = _kv(sys.argv[2:])
        rep = validate_round(a["--res-root"], json.loads(a["--tasks-json"]),
                             int(a["--round"]), a["--ckpt-dir"],
                             replay_data=a.get("--replay-data"))
        print(json.dumps(rep, ensure_ascii=False))
    elif cmd == "round-manifest-write":
        a = _kv(sys.argv[2:])
        info = json.loads(a["--info-json"])
        print(json.dumps({"path": round_manifest_write(a["--res-root"], int(a["--round"]), info)}))
    elif cmd == "manifest-cross-check":
        a = _kv(sys.argv[2:])
        rep = manifest_cross_check(a["--res-root-a"], a["--res-root-b"])
        print(json.dumps(rep, ensure_ascii=False))
        sys.exit(0 if rep["pass"] else 1)
    elif cmd == "eval-path-audit":
        # 训练前 eval 路径门禁（见 eval_path_audit 文档）；FAIL 时仍打印 JSON 并非零退出
        a = _kv(sys.argv[2:])
        rep = eval_path_audit(a.get("--root", "."), a.get("--model-base", ""),
                              a.get("--image-folder", ""),
                              scripts_dir=a.get("--scripts-dir"), py_dir=a.get("--py-dir"))
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        sys.exit(0 if rep["pass"] else 1)
    elif cmd == "ratio-tag":
        # random 系列 ratio → tag（唯一来源；非法 ratio 非零退出，供 gate fail-fast）
        try:
            print(ratio_tag(sys.argv[2]))
        except (ValueError, IndexError) as e:
            print(f"ERROR: {e if isinstance(e, ValueError) else '用法: coin_lib.py ratio-tag <ratio>'}", file=sys.stderr)
            sys.exit(2)
    elif cmd == "ratio-layout":
        # random 系列 ratio → 三目录布局 JSON（tag 由 ratio 派生，禁手工填）
        try:
            print(json.dumps(random_ratio_layout(sys.argv[2], sys.argv[3]), ensure_ascii=False))
        except IndexError:
            print("ERROR: 用法: coin_lib.py ratio-layout <ratio> <run_id>", file=sys.stderr)
            sys.exit(2)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        raise SystemExit(f"未知命令: {cmd}")


def _kv(argv):
    out = {}
    i = 0
    while i < len(argv):
        k = argv[i]
        if k.startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[k] = argv[i + 1]
                i += 2
            else:
                out[k] = "1"
                i += 1
        else:
            i += 1
    return out


def _flag(argv, name):
    return any(a == name for a in argv)


if __name__ == "__main__":
    main()
