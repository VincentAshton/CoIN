#!/usr/bin/env python3
"""CoIN ImageNet 子集提取：公共卷官方 ILSVRC2012 tar -> cl_dataset/ImageNet_withlabel.

外层 train tar 顺序流单遍扫描，仅解包 CoIN train.json 引用的 synset 类 tar（整类提取，
引用即类内全量/近全量，超集不影响 preflight）；val tar 按 test.json 引用的 5050 个文件名提取。
产物落 /root/data/coin/datasets/cl_dataset/ImageNet_withlabel/{train,val}/。
"""
import json
import os
import shutil
import sys
import tarfile
import time

TRAIN_TAR = "/public/huggingface-datasets/Imagenet2012/ILSVRC2012_img_train.tar"
VAL_TAR = "/public/huggingface-datasets/Imagenet2012/ILSVRC2012_img_val.tar"
BASE = "/root/data/coin/datasets/_downloads/imagenet-kaggle"
SYN_FILE = BASE + "/synsets.txt"
VAL_FILE = BASE + "/val_files.txt"
OUT = "/root/data/coin/datasets/cl_dataset/ImageNet_withlabel"
SUMMARY = BASE + "/extract_summary.json"


def log(msg):
    print(msg, flush=True)


def write_stream(src, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(src, f, 1 << 20)


def main():
    syns = set(open(SYN_FILE).read().split())
    vals = set(open(VAL_FILE).read().split())
    log(f"[start] synsets={len(syns)} valfiles={len(vals)}")
    t0 = time.time()
    train_dir = os.path.join(OUT, "train")
    os.makedirs(train_dir, exist_ok=True)
    n_train_files = 0
    n_train_classes = 0
    errs = []

    tf = tarfile.open(TRAIN_TAR, "r|")
    for m in tf:
        if not (m.isfile() and m.name.endswith(".tar")):
            continue
        syn = m.name[:-4]
        if syn not in syns:
            continue
        try:
            inner = tarfile.open(fileobj=tf.extractfile(m), mode="r|")
            cls_dir = os.path.join(train_dir, syn)
            os.makedirs(cls_dir, exist_ok=True)
            c = 0
            for im in inner:
                if im.isfile():
                    write_stream(inner.extractfile(im),
                                 os.path.join(cls_dir, os.path.basename(im.name)))
                    c += 1
            n_train_files += c
            n_train_classes += 1
            log(f"[train] {syn}: {c} files (total {n_train_files}, classes "
                f"{n_train_classes}, {time.time() - t0:.0f}s)")
        except Exception as e:
            errs.append(f"class {syn}: {e!r}")
            log(f"[train] ERROR {syn}: {e!r}")
    tf.close()

    val_dir = os.path.join(OUT, "val")
    os.makedirs(val_dir, exist_ok=True)
    n_val = 0
    tf = tarfile.open(VAL_TAR, "r|")
    for m in tf:
        if m.isfile() and m.name in vals:
            write_stream(tf.extractfile(m), os.path.join(val_dir, m.name))
            n_val += 1
    tf.close()

    summary = {"train_classes": n_train_classes, "train_files": n_train_files,
               "val_files": n_val, "seconds": round(time.time() - t0, 1),
               "errors": errs}
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    log(f"[done] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
