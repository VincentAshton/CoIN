#!/usr/bin/env python3
"""CoIN+Replay: 构建 replay 训练数据（LLaVA 指令格式，原样保留样本）。

规则（与 TRACE 一致）：
  - round j (2..T)：取前 j-1 个任务的 train.json，每个任务取 floor(N*ratio) 条，
    合并为一个 replay json。N*ratio < 1 时该任务贡献 0 条（记录 k=0）；若总数为 0 → 非零退出。
  - ScienceQA 允许样本无 image 字段（纯文本题）；若 image 字段存在，则必须能解析：
    路径不越界、文件存在、非空、PIL 可解码，否则非零退出。
  - conversations 必须合法：非空列表、每项为 dict 且含 "from"/"value" 字符串。
  - 输出 sidecar manifest（<out>.manifest.json）：各源文件 SHA256、N、k、选中 ID/索引、
    输出 SHA256，原子写。
  - 可选 --nested-with <sidecar manifest>：验证本组选中 ID 严格嵌套于给定组的对应任务
    （同 seed 下 0.01 ⊆ 0.10 验证；prefix 下天然成立，random 下由排列性质保证）。

抽样模式（--sample-mode）：
  - prefix（默认；与旧实现完全兼容）：
      k = floor(N*ratio)；picked = samples[:k]；selected_indices = list(range(k))。
      相同输入生成的 replay JSON 与旧版本一致（逐字节）；manifest 结构不变。
      --seed 不参与选择（接受但忽略选择语义）。
  - random（随机 replay 样本选择；本系列实验核心，--seed 即 REPLAY_SAMPLE_SEED）：
      k 仍严格等于 floor(N*ratio)。
      task seed = sha256("<sample_seed>:<task>")（UTF-8；禁止用 Python 的 hash(task)）。
      对完整 indices=list(range(N)) 用 random.Random(int(task_seed,16)).shuffle(indices)，
      selected_indices = shuffled_indices[:k]，picked = [samples[i] for i in selected_indices]。
      task seed 不含 round → 同一任务在一次完整实验的不同 round 使用同一随机排列；
      先产生完整排列再取前 k → 相同 seed 下 ratio=0.01 的集合 ⊆ ratio=0.10 的集合。
      manifest 记录 mode/sample_seed/sampling_algorithm/task_seed/N/k/selected_indices/
      selected_ids/源 SHA256/replay 输出 SHA256。

用法:
  python scripts/CoIN_Replay/build_replay_data.py \
      --tasks ScienceQA TextVQA ImageNet GQA \
      --data-dir playground/Instructions_Original \
      --image-dir cl_dataset \
      --round 3 --ratio 0.1 --sample-mode random --seed 1234 \
      --out playground/Replay/ratio_0.1/round3_train.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# 随机抽样算法版本标识（manifest 记录；算法/格式一旦变更必须换名，禁止静默改行为）
SAMPLING_ALGORITHM = "sha256_task_seed_python_shuffle_v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True,
                   help="cl_dataset 根目录（存在 image 字段时校验图片）")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--sample-mode", choices=("prefix", "random"), default="prefix",
                   help="prefix=取前 k 条（旧实现）；random=由 sample seed 派生 task seed 的"
                        "随机排列前 k 条")
    p.add_argument("--seed", type=int, default=1234,
                   help="random 模式 = REPLAY_SAMPLE_SEED（从它与 task name 派生 task seed）；"
                        "prefix 模式接受但不参与选择")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nested-with", type=Path, default=None,
                   help="断言本组选中 ID 严格嵌套于该 replay json 的对应任务子集"
                        "（random 模式额外校验 mode/sample_seed/源 SHA256 一致）")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_conversations(conv, sample_id) -> None:
    if not isinstance(conv, list) or len(conv) == 0:
        raise ValueError(f"{sample_id}: conversations 为空或不是列表")
    for i, turn in enumerate(conv):
        if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
            raise ValueError(f"{sample_id}: conversations[{i}] 缺少 from/value")
        if not isinstance(turn["from"], str) or not isinstance(turn["value"], str):
            raise ValueError(f"{sample_id}: conversations[{i}] from/value 非字符串")
        if turn["from"].lower() not in ("human", "gpt"):
            raise ValueError(f"{sample_id}: conversations[{i}] from 非法: {turn['from']!r}")
        if len(turn["value"].strip()) == 0:
            raise ValueError(f"{sample_id}: conversations[{i}] value 为空")


def validate_image(sample_id, image, image_dir: Path) -> None:
    """image 字段存在时必须校验图片；缺失字段合法（ScienceQA 纯文本题）。"""
    if image is None or image == "":
        return  # 无 image 字段：允许（ScienceQA）
    if not isinstance(image, str):
        raise ValueError(f"{sample_id}: image 字段非字符串: {image!r}")
    rel = Path(image)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"{sample_id}: image 路径越界: {image!r}")
    p = image_dir / rel
    if not p.is_file():
        raise FileNotFoundError(f"{sample_id}: 图片不存在（Linux 大小写敏感）: {image}")
    if p.stat().st_size == 0:
        raise ValueError(f"{sample_id}: 图片为空文件: {image}")
    try:
        from PIL import Image
        with Image.open(p) as im:
            im.verify()
    except ImportError:
        pass  # 无 PIL 环境跳过解码（云端预检会启用）
    except Exception as e:
        raise ValueError(f"{sample_id}: 图片损坏无法解码: {image} ({type(e).__name__})")


def task_seed_hex(sample_seed: int, task: str) -> str:
    """稳定 task seed：sha256("<sample_seed>:<task>")，UTF-8，hexdigest。

    不含 round（同一任务不同 round 用同一排列）；分隔符 ':' 固定，算法变更需换
    SAMPLING_ALGORITHM 版本号。禁止用 Python 内置 hash()（进程间/版本间不稳定）。
    """
    return hashlib.sha256(f"{sample_seed}:{task}".encode("utf-8")).hexdigest()


def sample_random(samples: list, k: int, task_seed: str):
    """对完整 indices 用 task seed 洗牌后取前 k。返回 (selected_indices, picked)。"""
    rng = random.Random(int(task_seed, 16))
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    selected = indices[:k]
    return selected, [samples[i] for i in selected]


def sample_id(s):
    return s.get("id", s.get("question_id"))


def main():
    args = parse_args()
    assert 0 < args.ratio <= 1.0, "ratio 必须在 (0,1]"
    assert args.round >= 2, "round 1 无需 replay（没有历史任务）"
    prev_tasks = args.tasks[: args.round - 1]
    if not prev_tasks:
        print(f"[build_replay] round {args.round}: 无历史任务，跳过")
        return

    replay = []
    sources = {}
    for task in prev_tasks:
        src = args.data_dir / task / "train.json"
        if not src.exists():
            print(f"[build_replay] ERROR: 缺少 {src}")
            sys.exit(1)
        samples = json.load(open(src, encoding="utf-8"))
        if not isinstance(samples, list):
            print(f"[build_replay] ERROR: {src} 不是 json 数组")
            sys.exit(1)
        n = len(samples)
        k = int(n * args.ratio)  # 两种模式统一：floor(N*ratio)
        if args.sample_mode == "random":
            tseed = task_seed_hex(args.seed, task)
            selected_indices, picked = sample_random(samples, k, tseed)
        else:  # prefix：与旧实现完全一致
            selected_indices, picked = list(range(k)), samples[:k]
            tseed = None
        for s in picked:
            sample_id_ = sample_id(s)
            if "conversations" not in s:
                raise ValueError(f"{src}: 样本 {sample_id_} 缺少 conversations（非 LLaVA 指令格式）")
            validate_conversations(s["conversations"], f"{task}/{sample_id_}")
            if "image" in s:
                validate_image(f"{task}/{sample_id_}", s.get("image"), args.image_dir)
        replay.extend(picked)
        entry = {
            "path": str(src),
            "sha256": sha256_file(src),
            "N": n,
            "k": k,
            "selected_ids": [sample_id(s) for s in picked],
            "selected_indices": selected_indices,
        }
        if args.sample_mode == "random":
            entry["task_seed"] = tseed
        sources[task] = entry
        print(f"[build_replay] {task}: {k}/{n} (floor({n}*{args.ratio})={k})")

    if not replay:
        print("[build_replay] ERROR: 所有任务 k=0，replay 数据为空（N*ratio 全部 <1），禁止继续")
        sys.exit(1)

    # 嵌套断言（同 seed 下 0.01 ⊆ 0.10）；--nested-with 指向外层组的 sidecar manifest
    if args.nested_with is not None:
        if not args.nested_with.exists():
            print(f"[build_replay] ERROR: --nested-with 目标不存在: {args.nested_with}")
            sys.exit(1)
        outer = json.load(open(args.nested_with, encoding="utf-8"))
        if args.sample_mode == "random":
            # random 模式兼容性前置校验（任一不一致即失败）：
            # ① 外层 mode 相同；② sample_seed 相同；③ 各源文件 SHA256 相同
            if outer.get("mode") != "random":
                print("[build_replay] ERROR: --nested-with 外层 manifest mode 不是 random"
                      f"（外层={outer.get('mode')!r}）——嵌套集合比较仅允许同 mode 同 seed 同源")
                sys.exit(1)
            if outer.get("sample_seed") != args.seed:
                print(f"[build_replay] ERROR: --nested-with sample_seed 不一致"
                      f"（外层={outer.get('sample_seed')!r} 本组={args.seed}）——禁止嵌套比较")
                sys.exit(1)
            for task, entry in sources.items():
                outer_src = outer.get("sources", {}).get(task, {})
                if outer_src.get("sha256") != entry["sha256"]:
                    print(f"[build_replay] ERROR: --nested-with 源文件 SHA256 不一致 ({task})"
                          f"：外层={outer_src.get('sha256')} 本组={entry['sha256']}——"
                          "数据修订不同，嵌套集合不可比")
                    sys.exit(1)
        outer_by_task = {t: set(e.get("selected_ids", []))
                         for t, e in outer.get("sources", {}).items()}
        for task, entry in sources.items():
            ids = set(entry["selected_ids"])
            outer_ids = outer_by_task.get(task, set())
            if not ids.issubset(outer_ids):
                missing = sorted(ids - outer_ids)
                raise ValueError(
                    f"嵌套断言失败: {task} 有 {len(missing)} 条不在外层组中"
                    f"（prefix 下应天然嵌套，检查数据/seed 是否一致）: {missing[:5]}")
        print("[build_replay] 嵌套断言通过: 本组样本 ⊆ 外层组")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False)
    os.replace(tmp, args.out)

    manifest = {
        "round": args.round,
        "ratio": args.ratio,
        "seed": args.seed,
        "mode": args.sample_mode,
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "sources": sources,
        "output": {
            "path": str(args.out),
            "N": len(replay),
            "sha256": sha256_file(args.out),
        },
        "nested_with": str(args.nested_with) if args.nested_with else None,
    }
    if args.sample_mode == "random":
        manifest["sample_seed"] = args.seed
        manifest["sampling_algorithm"] = SAMPLING_ALGORITHM
    mpath = args.out.with_suffix(".json.manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[build_replay] round {args.round} replay: {len(replay)} 条 -> {args.out}")
    print(f"[build_replay] sidecar manifest -> {mpath}")


if __name__ == "__main__":
    main()
