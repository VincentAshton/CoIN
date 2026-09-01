#!/usr/bin/env python3
"""CoIN+Replay: 数据与图片 preflight（工单 7）。

检查项：
  1. 四任务 train/test(TextVQA=val) json 存在、非空、sha256、样本数
  2. 评估辅助元数据存在：
       ScienceQA: <image-dir>/ScienceQA/pid_splits.json + problems.json
       TextVQA  : <image-dir>/TextVQA/TextVQA_0.5.1_val.json
       GQA      : <image-dir>/GQA/testdev_balanced_questions.json
  3. 所有 json 实际引用的唯一图片：路径相对且不越界、文件存在（Linux 大小写敏感）、
     非零、PIL 可解码（--skip-pil 跳过）；输出每任务样本数/引用数/唯一数/缺失/损坏列表
  4. ImageNet（及所有任务）json 相对路径首段 == 任务名（布局完全匹配检查；
     可用 --layout-map 覆盖，如 '{"ImageNet":"ILSVRC2012"}')
  5. 报告含 json sha256（--hash-images 时含每张唯一图片 sha256），原子写

用法:
  python scripts/CoIN_Replay/preflight_data.py \
      --data-dir playground/Instructions_Original --image-dir cl_dataset \
      --out-report results/CoIN_Replay/preflight_report.json [--skip-pil] [--hash-images]
退出码: 0=全过; 1=任一检查失败（含缺失/损坏列表非空）
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

TASKS = ["ScienceQA", "TextVQA", "ImageNet", "GQA"]
AUX = {
    "ScienceQA": ["pid_splits.json", "problems.json"],
    "TextVQA": ["TextVQA_0.5.1_val.json"],
    "GQA": ["testdev_balanced_questions.json"],
    "ImageNet": [],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pil_ok(path: Path) -> bool:
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def collect_refs(samples: list):
    refs = []
    for s in samples:
        if isinstance(s, dict) and isinstance(s.get("image"), str) and s["image"]:
            refs.append(s["image"])
    return refs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--out-report", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--skip-pil", action="store_true")
    p.add_argument("--hash-images", action="store_true")
    p.add_argument("--layout-map", type=str, default=None,
                   help='JSON: {"任务名":"期望首段"}，默认期望首段==任务名')
    args = p.parse_args()

    layout_map = json.loads(args.layout_map) if args.layout_map else {}
    errors = []
    report = {"tasks": {}, "data_sha256": None, "ok": False}

    json_hashes = {}
    for task in args.tasks:
        data_dir = args.data_dir / task
        test_name = "val.json" if task == "TextVQA" else "test.json"
        files = {"train": data_dir / "train.json", "test": data_dir / test_name}
        task_report = {"files": {}, "samples": {}, "image_refs": 0, "unique_images": 0,
                       "missing": [], "corrupt": [], "layout_prefixes": set(),
                       "aux": {}}
        for role, fp in files.items():
            if not fp.is_file():
                errors.append(f"{task}/{role}: 缺失 {fp}")
                continue
            if fp.stat().st_size == 0:
                errors.append(f"{task}/{role}: 空文件 {fp}")
                continue
            data = json.load(open(fp, encoding="utf-8"))
            if not isinstance(data, list):
                errors.append(f"{task}/{role}: 不是 json 数组 {fp}")
                continue
            task_report["files"][role] = {"path": str(fp), "sha256": sha256_file(fp),
                                          "N": len(data)}
            task_report["samples"][role] = len(data)
            json_hashes[f"{task}/{role}"] = task_report["files"][role]["sha256"]
            for ref in collect_refs(data):
                task_report["image_refs"] += 1
                rel = Path(ref)
                if rel.is_absolute() or ".." in rel.parts:
                    errors.append(f"{task}/{role}: 图片路径越界: {ref!r}")
                    continue
                # 布局首段检查
                first = rel.parts[0] if rel.parts else ""
                task_report["layout_prefixes"].add(first)
                img = args.image_dir / rel
                if not img.is_file():
                    task_report["missing"].append(ref)
                    continue
                if img.stat().st_size == 0:
                    task_report["corrupt"].append(ref + " (空文件)")
                    continue
                if not args.skip_pil and not pil_ok(img):
                    task_report["corrupt"].append(ref)
                elif args.hash_images:
                    task_report.setdefault("image_sha256", {})[ref] = sha256_file(img)
        # 唯一图片数：从 train/test 两个 json 收集全部引用
        unique = set()
        for role in ("train", "test"):
            if role in task_report["files"]:
                fp = data_dir / ("train.json" if role == "train" else test_name)
                unique.update(collect_refs(json.load(open(fp, encoding="utf-8"))))
        task_report["unique_images"] = len(unique)
        # 布局匹配
        expect = layout_map.get(task, task)
        if task_report["layout_prefixes"] and task_report["layout_prefixes"] != {expect}:
            errors.append(
                f"{task}: json 图片路径首段 {sorted(task_report['layout_prefixes'])} "
                f"!= 期望 {expect!r}（布局不匹配，ImageNet 尤其要核对）")
        # 评估辅助元数据
        for aux in AUX.get(task, []):
            ap = args.image_dir / task / aux
            if not ap.is_file():
                errors.append(f"{task}: 缺评估辅助文件 {ap}")
            else:
                task_report["aux"][aux] = {"path": str(ap), "sha256": sha256_file(ap)}
        report["tasks"][task] = task_report
        task_report["layout_prefixes"] = sorted(task_report["layout_prefixes"])
        print(f"[preflight] {task}: train={task_report['samples'].get('train')} "
              f"test={task_report['samples'].get('test')} refs={task_report['image_refs']} "
              f"unique={task_report['unique_images']} missing={len(task_report['missing'])} "
              f"corrupt={len(task_report['corrupt'])} prefixes={sorted(task_report['layout_prefixes'])}")
        if task_report["missing"]:
            errors.append(f"{task}: {len(task_report['missing'])} 张图片缺失（前 10: "
                          f"{task_report['missing'][:10]}）")
        if task_report["corrupt"]:
            errors.append(f"{task}: {len(task_report['corrupt'])} 张图片损坏（前 10: "
                          f"{task_report['corrupt'][:10]}）")

    report["data_sha256"] = sha256_file_json(json_hashes)
    report["ok"] = not errors
    report["errors"] = errors
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(args.out_report) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, args.out_report)
    print(f"[preflight] data_sha256={report['data_sha256']}")
    print(f"[preflight] report -> {args.out_report}")
    if errors:
        print("[preflight] FAIL:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("[preflight] PASS")


def sha256_file_json(hashes: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(hashes):
        h.update(k.encode())
        h.update(hashes[k].encode())
    return h.hexdigest()


if __name__ == "__main__":
    main()
