#!/usr/bin/env python3
"""checkpoint 重载一致性探针（工单 9E，云端运行，真实 checkpoint）。

修复（2026-09-02，评审批准）：兼容 ScienceQA 纯文本样本（无 image 字段 / 值为 None），
与官方评估语义严格对齐（ETrain/Eval/LLaVA/CoIN/model_vqa_science.py）：
  - 无图题：images=None 传给模型，prompt 不含 <image> token（不注入空白图/伪造路径）；
  - 有图题：加载并预处理图片，prompt = "<image>\\n" + text；
  - text 统一 text.replace('<image>', '').strip()（与官方一致）；
  - image 字段声明但路径越界/缺失/空/损坏 → 立即非零退出（严格失败，不跳过）。

固定 probe 样本集（question_id，见 FIXED_PROBES）：重载验证与不同 ratio 必须使用
完全相同的集合；probe manifest 记录 question_id/has_image/图片相对路径/prompt hash/
input_ids hash/logits hash/数据文件 hash，原子写。

用法: python scripts/CoIN_Replay/smoke/probe_logits.py --ckpt <dir> --model-base <vicuna>
      --vision-tower <clip> --probe-json <指令 json> [--image-folder ./cl_dataset]
退出码: 0=通过；非 0=失败
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

IMAGE_TOKEN = "<image>"

# 固定 probe 样本（ScienceQA test.json 的 question_id；评审要求至少 1 无图 + 1 有图）
FIXED_PROBES = [
    {"question_id": "4", "expect_image": False},  # 纯文本题
    {"question_id": "5", "expect_image": True},   # 有图题
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_prompt(text, has_image, use_im_start_end=False,
                 im_start_token="", im_end_token=""):
    """与 model_vqa_science.py 完全一致地构造提问串（qs，尚未套 conv 模板）。"""
    qs = (text or "").replace(IMAGE_TOKEN, "").strip()
    if has_image:
        if use_im_start_end:
            qs = im_start_token + IMAGE_TOKEN + im_end_token + "\n" + qs
        else:
            qs = IMAGE_TOKEN + "\n" + qs
    return qs


def _validate_image(sample_id, rel, image_folder):
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"{sample_id}: 图片路径越界: {rel!r}")
    img = Path(image_folder) / p
    if not img.is_file():
        raise FileNotFoundError(f"{sample_id}: 图片不存在（Linux 大小写敏感）: {img}")
    if img.stat().st_size == 0:
        raise ValueError(f"{sample_id}: 图片为空文件: {img}")
    try:
        from PIL import Image
        with Image.open(img) as im:
            im.verify()
    except ImportError:
        pass  # 无 PIL 环境跳过解码（云端评估环境必有 PIL）
    except Exception as e:
        raise ValueError(f"{sample_id}: 图片损坏无法解码: {img} ({type(e).__name__})")


def build_probe_set(probe_json, image_folder, fixed=None, use_im_start_end=False):
    """构建并校验固定 probe 集合；任何不一致立即 raise（严格失败）。"""
    fixed = FIXED_PROBES if fixed is None else fixed
    with open(probe_json, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{probe_json}: 不是 json 数组")
    by_id = {}
    for s in data:
        qid = s.get("question_id")
        if qid is None:
            raise ValueError(f"{probe_json}: 条目缺少 question_id: {str(s)[:100]}")
        if qid in by_id:
            raise ValueError(f"{probe_json}: 重复 question_id: {qid}")
        by_id[qid] = s
    probes = []
    for spec in fixed:
        qid = spec["question_id"]
        s = by_id.get(qid)
        if s is None:
            raise ValueError(f"固定 probe id {qid!r} 不存在于 {probe_json}")
        raw_image = s.get("image")
        has_image = isinstance(raw_image, str) and raw_image != ""
        if has_image != spec["expect_image"]:
            raise ValueError(
                f"固定 probe id {qid!r}: 期望 expect_image={spec['expect_image']}，"
                f"实际 has_image={has_image}（数据漂移？）")
        if has_image:
            _validate_image(qid, raw_image, image_folder)
        text = s.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"固定 probe id {qid!r}: text 缺失或为空")
        probes.append({
            "question_id": qid,
            "has_image": has_image,
            "image": raw_image if has_image else None,
            "text": text,
            "prompt": build_prompt(text, has_image, use_im_start_end),
        })
    return probes, sha256_file(probe_json)


def atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_probe(ckpt, model_base, vision_tower, probes, image_folder):
    """加载一次模型，返回每条的 {prompt_hash, input_ids_hash, logits_hash, finite}。"""
    import torch
    from ETrain.Models.LLaVA.builder import load_pretrained_model
    from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token
    from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from ETrain.utils.LLaVA.conversation import conv_templates

    if DEFAULT_IMAGE_TOKEN != IMAGE_TOKEN:
        raise SystemExit(
            f"ETrain DEFAULT_IMAGE_TOKEN 与 probe 常量不一致: {DEFAULT_IMAGE_TOKEN!r}")
    model_name = ckpt.rstrip("/").split("/")[-1]
    tok, model, img_proc, _ = load_pretrained_model(ckpt, model_base, model_name)
    model.eval()
    model.to(torch.float16)
    if getattr(model.config, "mm_use_im_start_end", False):
        raise SystemExit("probe 仅支持 mm_use_im_start_end=False（LLaVA-1.5 语义）")
    out = []
    with torch.inference_mode():
        for q in probes:
            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], q["prompt"])
            conv.append_message(conv.roles[1], None)
            full = conv.get_prompt()
            input_ids = tokenizer_image_token(
                full, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            if q["has_image"]:
                from PIL import Image
                image = Image.open(os.path.join(image_folder, q["image"])).convert("RGB")
                img_t = img_proc.preprocess(image, return_tensors="pt")["pixel_values"][0].half().cuda()
                images = img_t.unsqueeze(0)
            else:
                images = None
            logits = model(input_ids, images=images).logits.detach().cpu()
            finite = bool(torch.isfinite(logits).all().item())
            out.append({
                "question_id": q["question_id"],
                "prompt": full,
                "prompt_hash": sha256_text(full),
                "input_ids_hash": hashlib.sha256(
                    input_ids.detach().cpu().numpy().tobytes()).hexdigest(),
                "logits_hash": hashlib.sha256(logits.numpy().tobytes()).hexdigest(),
                "finite": finite,
            })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-base", required=True)
    p.add_argument("--vision-tower", required=True)
    p.add_argument("--probe-json", required=True)
    p.add_argument("--image-folder", default="./cl_dataset")
    args = p.parse_args()

    probes, data_sha = build_probe_set(args.probe_json, args.image_folder)
    print(f"[probe] 固定样本集: {[q['question_id'] for q in probes]} "
          f"覆盖: text-only={sum(1 for q in probes if not q['has_image'])} "
          f"image={sum(1 for q in probes if q['has_image'])} data_sha256={data_sha}")

    sys.path.insert(0, ".")
    r1 = run_probe(args.ckpt, args.model_base, args.vision_tower, probes, args.image_folder)
    r2 = run_probe(args.ckpt, args.model_base, args.vision_tower, probes, args.image_folder)

    ok = True
    for a, b in zip(r1, r2):
        same = (a["prompt_hash"] == b["prompt_hash"]
                and a["input_ids_hash"] == b["input_ids_hash"]
                and a["logits_hash"] == b["logits_hash"])
        fin = a["finite"] and b["finite"]
        print(f"[probe:{a['question_id']}] finite={a['finite']}/{b['finite']} "
              f"two-load-identical={same} prompt_hash={a['prompt_hash'][:16]}")
        if not (same and fin):
            ok = False

    manifest = {
        "probe_json": os.path.abspath(args.probe_json),
        "data_sha256": data_sha,
        "fixed_ids": [q["question_id"] for q in probes],
        "coverage": {"text_only": sum(1 for q in probes if not q["has_image"]),
                     "image": sum(1 for q in probes if q["has_image"])},
        "image_folder": os.path.abspath(args.image_folder),
        "ckpt": os.path.abspath(args.ckpt),
        "loads": {"first": r1, "second": r2},
        "identical_across_loads": ok,
    }
    mp = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "probe_manifest.json")
    atomic_write_json(mp, manifest)
    print(f"[probe] manifest -> {mp}")
    print("[probe]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
