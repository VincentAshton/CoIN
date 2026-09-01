"""工单 7 单测：preflight_data.py（缺失/损坏/越界/布局/辅助文件/data_sha256）。"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import REPLAY_DIR, build_synthetic, make_aux, run, tiny_png

PRE = [sys.executable, os.path.join(REPLAY_DIR, "preflight_data.py")]


class TestPreflightData(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_pre_")
        self.data_dir = os.path.join(self.tmp, "Instructions")
        self.img_dir = os.path.join(self.tmp, "images")
        os.makedirs(self.data_dir)
        os.makedirs(self.img_dir)
        self.report = os.path.join(self.tmp, "report.json")
        for task in ("ScienceQA", "TextVQA", "ImageNet", "GQA"):
            build_synthetic(self.data_dir, self.img_dir, task, n=5)
            make_aux(self.img_dir, task)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, extra=None):
        return run(PRE + [
            "--data-dir", self.data_dir, "--image-dir", self.img_dir,
            "--out-report", self.report,
            "--tasks", "ScienceQA", "TextVQA", "ImageNet", "GQA",
        ] + (extra or []))

    def test_ok(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.load(open(self.report))
        self.assertTrue(rep["ok"])
        self.assertTrue(rep["data_sha256"])
        self.assertEqual(rep["tasks"]["ScienceQA"]["samples"]["train"], 5)

    def test_missing_image(self):
        os.remove(os.path.join(self.img_dir, "TextVQA", "img", "1.png"))
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        rep = json.load(open(self.report))
        self.assertIn("TextVQA/img/1.png", rep["tasks"]["TextVQA"]["missing"])

    def test_corrupt_image(self):
        with open(os.path.join(self.img_dir, "GQA", "img", "2.png"), "wb") as f:
            f.write(b"not an image")
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        rep = json.load(open(self.report))
        self.assertIn("GQA/img/2.png", rep["tasks"]["GQA"]["corrupt"])

    def test_skip_pil_allows_corrupt(self):
        with open(os.path.join(self.img_dir, "GQA", "img", "2.png"), "wb") as f:
            f.write(b"not an image")
        r = self._run(["--skip-pil"])
        # --skip-pil 是显式跳过硬解码的选项：损坏但非空文件此时通过（文档化降级）
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_path_escape(self):
        tp = os.path.join(self.data_dir, "ImageNet", "train.json")
        data = json.load(open(tp))
        data.append({"id": "x", "image": "../outside.png",
                     "conversations": [{"from": "human", "value": "q"},
                                       {"from": "gpt", "value": "a"}]})
        json.dump(data, open(tp, "w"))
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("越界", r.stdout + r.stderr)

    def test_layout_mismatch(self):
        # 把 ImageNet 的 image 路径首段改成别的目录名
        tp = os.path.join(self.data_dir, "ImageNet", "train.json")
        data = json.load(open(tp))
        for s in data:
            s["image"] = s["image"].replace("ImageNet/", "OtherRoot/", 1)
        json.dump(data, open(tp, "w"))
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("布局", r.stdout + r.stderr)

    def test_missing_aux(self):
        os.remove(os.path.join(self.img_dir, "ScienceQA", "pid_splits.json"))
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("pid_splits.json", r.stdout + r.stderr)

    def test_data_sha256_changes_with_data(self):
        r1 = self._run()
        sha1 = json.load(open(self.report))["data_sha256"]
        tp = os.path.join(self.data_dir, "GQA", "train.json")
        data = json.load(open(tp))
        data.append({"id": "extra", "image": "GQA/img/0.png",
                     "conversations": [{"from": "human", "value": "q"},
                                       {"from": "gpt", "value": "a"}]})
        json.dump(data, open(tp, "w"))
        r2 = self._run()
        sha2 = json.load(open(self.report))["data_sha256"]
        self.assertNotEqual(sha1, sha2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
