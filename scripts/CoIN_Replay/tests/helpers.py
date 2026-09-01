"""测试公共工具：合成数据 / 图片 / 子进程运行。"""
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPLAY_DIR = os.path.join(ROOT, "scripts", "CoIN_Replay")


def tiny_png() -> bytes:
    """1x1 RGB PNG（stdlib 生成，PIL 可解码）。"""
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def make_llava_sample(sid, image, has_image=True, conversations=None):
    conv = conversations if conversations is not None else [
        {"from": "human", "value": "<image>\nWhat is this?"},
        {"from": "gpt", "value": f"answer-{sid}"},
    ]
    s = {"id": sid, "conversations": conv}
    if has_image:
        s["image"] = image
    return s


def make_test_sample(qid, image, text="<image>\nWhat is this?", answer="42"):
    return {"question_id": qid, "image": image, "text": text, "answer": answer}


def build_synthetic(task_root, image_root, task, n=10, with_images=True):
    """建一个任务的 train/test json + 图片。返回 (train_path, test_path)。"""
    os.makedirs(os.path.join(task_root, task), exist_ok=True)
    os.makedirs(os.path.join(image_root, task, "img"), exist_ok=True)
    train, test = [], []
    for i in range(n):
        img = f"{task}/img/{i}.png"
        with open(os.path.join(image_root, task, "img", f"{i}.png"), "wb") as f:
            f.write(tiny_png())
        train.append(make_llava_sample(f"{task}_{i}", img, has_image=with_images))
        test.append(make_test_sample(f"{task}_q{i}", img))
    tp = os.path.join(task_root, task, "train.json")
    json.dump(train, open(tp, "w"))
    test_name = "val.json" if task == "TextVQA" else "test.json"
    jp = os.path.join(task_root, task, test_name)
    json.dump(test, open(jp, "w"))
    return tp, jp


def make_aux(image_root, task):
    """创建 preflight 需要的评估辅助文件（内容可为空结构）。"""
    os.makedirs(os.path.join(image_root, task), exist_ok=True)
    if task == "ScienceQA":
        for f in ("pid_splits.json", "problems.json"):
            json.dump({}, open(os.path.join(image_root, task, f), "w"))
    elif task == "TextVQA":
        json.dump({"data": []}, open(os.path.join(image_root, task, "TextVQA_0.5.1_val.json"), "w"))
    elif task == "GQA":
        json.dump({}, open(os.path.join(image_root, task, "testdev_balanced_questions.json"), "w"))


def run(cmd, cwd=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
    return r


def py(cmd, cwd=None, env_extra=None):
    return run([sys.executable] + cmd, cwd=cwd, env_extra=env_extra)
