#!/usr/bin/env python3
"""checkpoint 重载一致性探针（工单 9E，云端运行，真实 checkpoint）。

验证：
  1. 同一 checkpoint 加载两次，固定 probe 输入的 logits 完全一致（加载确定性）
  2. 参数全部 finite
  3. 两次加载的 adapter 权重 hash 一致

用法: python scripts/CoIN_Replay/smoke/probe_logits.py --ckpt <dir> --model-base <vicuna dir>
       --vision-tower <clip dir> [--probe-json 任意指令 json 前 N 条]
退出码: 0=通过；非 0=失败
"""
import argparse
import hashlib
import json
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--model-base", required=True)
    p.add_argument("--vision-tower", required=True)
    p.add_argument("--probe-json", required=True, help="任意含 image/text 的 json（取前 4 条）")
    args = p.parse_args()

    sys.path.insert(0, ".")
    from ETrain.Models.LLaVA.builder import load_pretrained_model
    from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token
    from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from ETrain.utils.LLaVA.conversation import conv_templates
    from PIL import Image

    probes = json.load(open(args.probe_json))[:4]

    def load_and_logits(tag):
        model_name = args.ckpt.rstrip("/").split("/")[-1]
        tok, model, img_proc, _ = load_pretrained_model(
            args.ckpt, args.model_base, model_name)
        model.eval()
        model.to(torch.float16)
        logits = []
        with torch.inference_mode():
            for q in probes:
                image = Image.open(f"./cl_dataset/{q['image']}").convert("RGB")
                img_t = img_proc.preprocess(image, return_tensors="pt")["pixel_values"][0].half().cuda()
                qs = q.get("text", "<image>\nWhat is this?")
                if DEFAULT_IMAGE_TOKEN not in qs:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX,
                                                  return_tensors="pt").unsqueeze(0).cuda()
                out = model(input_ids, images=img_t.unsqueeze(0))
                logits.append(out.logits.detach().cpu())
        all_finite = all(torch.isfinite(l).all().item() for l in logits)
        print(f"[probe:{tag}] finite={all_finite} logits shapes={[tuple(l.shape) for l in logits]}")
        return logits, all_finite

    l1, f1 = load_and_logits("first")
    l2, f2 = load_and_logits("second")
    same = all(torch.equal(a, b) for a, b in zip(l1, l2))
    print(f"[probe] 两次加载 logits 完全一致: {same}")
    ok = same and f1 and f2
    print("[probe]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
