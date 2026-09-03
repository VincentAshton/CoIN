#!/usr/bin/env python3
"""生成 CoIN ImageNet 数据 manifest（阶段 7 交付物，数字取自 preflight 报告）。"""
import json
import subprocess
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))
FOUR = "/root/data/coin/logs/full_preflight/four_tasks_20260903_114629/preflight_report_four_tasks.json"
IMG = "/root/data/coin/logs/full_preflight/imagenet/preflight_report_imagenet.json"
OUT = "/root/data/coin/logs/full_preflight/manifest_20260903.json"

four = json.load(open(FOUR))
tasks = {}
for t, r in four["tasks"].items():
    tasks[t] = {
        "samples": r["samples"],
        "image_refs": r["image_refs"],
        "unique_images": r["unique_images"],
        "missing": len(r["missing"]),
        "corrupt": len(r["corrupt"]),
        "json_sha256": {k: v["sha256"] for k, v in r["files"].items()},
    }

commit = subprocess.run(["git", "-C", "/root/data/coin/project", "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()

manifest = {
    "manifest_version": 1,
    "created": datetime.now(CST).isoformat(timespec="seconds"),
    "scope": "CoIN+Replay 正式 sweep 前置：ImageNet 数据就位 + 四任务全量数据门禁",
    "repo": {"path": "/root/data/coin/project", "commit": commit, "clean_workspace": True},
    "host": {"instance": "coindl (CPU)", "cgroup": "8 vCPU / 32G RAM", "data_volume": "coinssd @ /root/data (lustre nvme)"},
    "imagenet_source": {
        "method": "公共卷官方 ILSVRC2012 tar 流式提取子集（用户决策 C：公共卷优先，失败回退 Kaggle）",
        "license": "ILSVRC2012 研究许可数据（官方 tar，经 ebcloud 公共镜像 /public 只读卷获取）",
        "tar_train": {"path": "/public/huggingface-datasets/Imagenet2012/ILSVRC2012_img_train.tar",
                      "bytes": 147897477120},
        "tar_val": {"path": "/public/huggingface-datasets/Imagenet2012/ILSVRC2012_img_val.tar",
                    "bytes": 6744924160},
        "kaggle_fallback": {"competition": "imagenet-object-localization-challenge",
                            "cli": "kaggle 1.7.4.5 @ /root/data/coin/tools/kaggle-download-venv",
                            "auth_verified": True, "rules_accepted": True, "used": False},
        "extract": {"train_files": 129833, "train_classes": 101, "val_files": 5050,
                    "errors": 0, "seconds": 523.3,
                    "summary": "/root/data/coin/datasets/_downloads/imagenet-kaggle/extract_summary.json"},
    },
    "final_layout": {
        "root": "/root/data/coin/datasets/cl_dataset/ImageNet_withlabel",
        "train": "train/<synset>/*.JPEG (101 synsets)",
        "val": "val/*.JPEG (5050)",
        "size_train": "14G", "size_val": "638M",
    },
    "tasks": tasks,
    "preflight": [
        {"scope": "ImageNet 专项", "ok": True, "exit": 0,
         "log": "/root/data/coin/logs/full_preflight/imagenet/preflight_imagenet.log",
         "report": "/root/data/coin/logs/full_preflight/imagenet/preflight_report_imagenet.json"},
        {"scope": "四任务全量门禁", "ok": True, "exit": 0,
         "log": "/root/data/coin/logs/full_preflight/four_tasks_20260903_114629/preflight_four_tasks.log",
         "report": "/root/data/coin/logs/full_preflight/four_tasks_20260903_114629/preflight_report_four_tasks.json",
         "layout_map": {"ImageNet": "ImageNet_withlabel"},
         "note": "GQA 无需映射条目：json 路径带 './' 前缀，preflight 用 Path.parts 折叠后首段='GQA'=默认期望；旧文档 GQA:\".\" 为未实测推测值，已废弃"},
    ],
    "data_sha256_four_tasks": four["data_sha256"],
    "disk": {"root_data_used": "98G", "root_data_avail": "927G"},
    "residual_procs": "none",
    "note_creds": "Kaggle 凭据已于 2026-09-03 从 CPU 下载实例删除（API token 建议用户在 Kaggle 轮换）",
}

json.dump(manifest, open(OUT, "w"), indent=1, ensure_ascii=False)
print("manifest ->", OUT)
print(json.dumps({"ok": four["ok"], "commit": commit,
                  "tasks": {t: (v["samples"], v["image_refs"], v["unique_images"],
                                v["missing"], v["corrupt"]) for t, v in tasks.items()}},
                 ensure_ascii=False, indent=1))
