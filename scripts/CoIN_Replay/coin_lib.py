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
# manifest（工单 6）
# ---------------------------------------------------------------------------

CONFIG_FIELDS = [
    "ratio", "tasks", "T", "model_base", "vision_tower", "projector",
    "lora_r", "lora_alpha", "lora_dropout", "lr", "mm_projector_lr",
    "epochs_per_task", "replay_epochs", "seed", "data_seed", "sample_mode",
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
                raise ValueError(
                    "run_manifest.json 已存在且 config hash 不一致——恢复运行配置与首次运行不同，"
                    "禁止覆盖。首次配置见 manifest 的 config 字段。")
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
