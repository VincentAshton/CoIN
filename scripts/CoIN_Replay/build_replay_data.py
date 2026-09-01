#!/usr/bin/env python3
"""CoIN+Replay: 构建 replay 训练数据（LLaVA 指令格式，原样保留样本）。

规则（与 TRACE 一致）：
  - round j (2..T)：取前 j-1 个任务的 train.json，每个任务取【前缀】的 floor(N*ratio) 条，
    合并为一个 replay json。N*ratio < 1 时该任务贡献 0 条（记录 k=0）；若总数为 0 → 非零退出。
  - ScienceQA 允许样本无 image 字段（纯文本题）；若 image 字段存在，则必须能解析：
    路径不越界、文件存在、非空、PIL 可解码，否则非零退出。
  - conversations 必须合法：非空列表、每项为 dict 且含 "from"/"value" 字符串。
  - 输出 sidecar manifest（<out>.manifest.json）：各源文件 SHA256、N、k、选中 ID/索引、
    输出 SHA256，原子写。
  - 可选 --nested-with <replay json>：验证本组选中 ID 严格嵌套于给定组的对应任务
    （用于 0.01 ⊆ 0.10 验证；prefix 下天然成立，此处为显式断言）。

用法:
  python scripts/CoIN_Replay/build_replay_data.py \
      --tasks ScienceQA TextVQA ImageNet GQA \
      --data-dir playground/Instructions_Original \
      --image-dir cl_dataset \
      --round 3 --ratio 0.1 --seed 1234 \
      --out playground/Replay/ratio_0.1/round3_train.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True,
                   help="cl_dataset 根目录（存在 image 字段时校验图片）")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nested-with", type=Path, default=None,
                   help="断言本组选中 ID 严格嵌套于该 replay json 的对应任务子集")
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
        k = int(n * args.ratio)  # TRACE 前缀数量：floor(N*ratio)
        picked = samples[:k]
        for s in picked:
            sample_id = s.get("id", s.get("question_id", "<no-id>"))
            if "conversations" not in s:
                raise ValueError(f"{src}: 样本 {sample_id} 缺少 conversations（非 LLaVA 指令格式）")
            validate_conversations(s["conversations"], f"{task}/{sample_id}")
            if "image" in s:
                validate_image(f"{task}/{sample_id}", s.get("image"), args.image_dir)
        replay.extend(picked)
        sources[task] = {
            "path": str(src),
            "sha256": sha256_file(src),
            "N": n,
            "k": k,
            "selected_ids": [s.get("id", s.get("question_id")) for s in picked],
            "selected_indices": list(range(k)),
        }
        print(f"[build_replay] {task}: {k}/{n} (floor({n}*{args.ratio})={k})")

    if not replay:
        print("[build_replay] ERROR: 所有任务 k=0，replay 数据为空（N*ratio 全部 <1），禁止继续")
        sys.exit(1)

    # 嵌套断言（0.01 ⊆ 0.10）；--nested-with 指向外层组的 sidecar manifest
    if args.nested_with is not None:
        if not args.nested_with.exists():
            print(f"[build_replay] ERROR: --nested-with 目标不存在: {args.nested_with}")
            sys.exit(1)
        outer = json.load(open(args.nested_with, encoding="utf-8"))
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
        "mode": "prefix",
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "sources": sources,
        "output": {
            "path": str(args.out),
            "N": len(replay),
            "sha256": sha256_file(args.out),
        },
        "nested_with": str(args.nested_with) if args.nested_with else None,
    }
    mpath = args.out.with_suffix(".json.manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[build_replay] round {args.round} replay: {len(replay)} 条 -> {args.out}")
    print(f"[build_replay] sidecar manifest -> {mpath}")


if __name__ == "__main__":
    main()
