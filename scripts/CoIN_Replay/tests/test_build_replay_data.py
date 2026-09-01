"""工单 2 单测：build_replay_data.py（floor 前缀 / 无图 / 缺图 / 损坏 / 空 conversations / sidecar / 嵌套）。"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import REPLAY_DIR, build_synthetic, make_llava_sample, run, tiny_png

BUILD = [sys.executable, os.path.join(REPLAY_DIR, "build_replay_data.py")]


class TestBuildReplayData(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_build_")
        self.data_dir = os.path.join(self.tmp, "Instructions")
        self.img_dir = os.path.join(self.tmp, "images")
        os.makedirs(self.data_dir)
        os.makedirs(self.img_dir)
        # ScienceQA 10 条（含 2 条无图）、TextVQA 20 条
        build_synthetic(self.data_dir, self.img_dir, "ScienceQA", 10)
        build_synthetic(self.data_dir, self.img_dir, "TextVQA", 20)
        # ScienceQA 追加无图样本
        tp = os.path.join(self.data_dir, "ScienceQA", "train.json")
        data = json.load(open(tp))
        data.append(make_llava_sample("ScienceQA_noid_0", None, has_image=False))
        json.dump(data, open(tp, "w"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, round_, ratio, out, extra=None, tasks=None):
        cmd = BUILD + [
            "--tasks", *(tasks or ["ScienceQA", "TextVQA"]),
            "--data-dir", self.data_dir, "--image-dir", self.img_dir,
            "--round", str(round_), "--ratio", str(ratio), "--seed", "1234",
            "--out", out,
        ] + (extra or [])
        return run(cmd)

    def test_floor_prefix_counts(self):
        out = os.path.join(self.tmp, "r3_01.json")
        r = self._build(3, 0.1, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(out + ".manifest.json"))
        # round3 源 = ScienceQA(10), TextVQA(20)；floor(10*0.1)=1, floor(20*0.1)=2
        self.assertEqual(m["sources"]["ScienceQA"]["k"], 1)
        self.assertEqual(m["sources"]["TextVQA"]["k"], 2)
        self.assertEqual(m["output"]["N"], 3)
        data = json.load(open(out))
        self.assertEqual([s["id"] for s in data], ["ScienceQA_0", "TextVQA_0", "TextVQA_1"])
        # 无图样本也在（ScienceQA_0 有图；这里验证 noid 样本没被误删）
        self.assertTrue(any(s["id"] == "ScienceQA_noid_0" for s in data) is False)  # 前缀只取前 k 条

    def test_floor_zero_contrib(self):
        # round3 ratio=0.05: floor(11*0.05)=0 (ScienceQA), floor(20*0.05)=1 (TextVQA)
        out = os.path.join(self.tmp, "r3_005.json")
        r = self._build(3, 0.05, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(out + ".manifest.json"))
        self.assertEqual(m["sources"]["ScienceQA"]["k"], 0)
        self.assertEqual(m["sources"]["TextVQA"]["k"], 1)
        self.assertEqual(m["output"]["N"], 1)

    def test_all_zero_fails(self):
        # round3 ratio=0.01: 两任务 k 全 0 → 禁止继续
        out = os.path.join(self.tmp, "r3_zero.json")
        r = self._build(3, 0.01, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("k=0", r.stdout + r.stderr)

    def test_missing_image_field_ok(self):
        # ScienceQA 无图样本被允许；构造只有无图样本的数据（只测 ScienceQA 单任务源）
        shutil.rmtree(self.data_dir, ignore_errors=True)
        os.makedirs(os.path.join(self.data_dir, "ScienceQA"))
        json.dump([make_llava_sample(f"t{i}", None, has_image=False) for i in range(10)],
                  open(os.path.join(self.data_dir, "ScienceQA", "train.json"), "w"))
        out = os.path.join(self.tmp, "r2_noimg.json")
        r = self._build(2, 0.5, out, tasks=["ScienceQA"])
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_missing_image_file_fails(self):
        tp = os.path.join(self.data_dir, "TextVQA", "train.json")
        data = json.load(open(tp))
        data.append(make_llava_sample("T_ghost", "TextVQA/img/ghost.png"))
        json.dump(data, open(tp, "w"))
        out = os.path.join(self.tmp, "r3_ghost.json")
        r = self._build(3, 1.0, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ghost.png", r.stdout + r.stderr)

    def test_corrupt_image_fails(self):
        with open(os.path.join(self.img_dir, "TextVQA", "img", "corrupt.png"), "wb") as f:
            f.write(b"this is not a png")
        tp = os.path.join(self.data_dir, "TextVQA", "train.json")
        data = json.load(open(tp))
        data.append(make_llava_sample("T_bad", "TextVQA/img/corrupt.png"))
        json.dump(data, open(tp, "w"))
        out = os.path.join(self.tmp, "r3_bad.json")
        r = self._build(3, 1.0, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("损坏", r.stdout + r.stderr)

    def test_empty_conversations_fails(self):
        tp = os.path.join(self.data_dir, "TextVQA", "train.json")
        data = json.load(open(tp))
        data.append(make_llava_sample("T_empty", "TextVQA/img/0.png", conversations=[]))
        json.dump(data, open(tp, "w"))
        out = os.path.join(self.tmp, "r3_empty.json")
        r = self._build(3, 1.0, out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("conversations", r.stdout + r.stderr)

    def test_sidecar_manifest_fields(self):
        out = os.path.join(self.tmp, "r3_01.json")
        self.assertEqual(self._build(3, 0.1, out).returncode, 0)
        m = json.load(open(out + ".manifest.json"))
        self.assertEqual(m["mode"], "prefix")
        self.assertEqual(m["round"], 3)
        self.assertAlmostEqual(m["ratio"], 0.1)
        for t in ("ScienceQA", "TextVQA"):
            src = m["sources"][t]
            for k in ("path", "sha256", "N", "k", "selected_ids", "selected_indices"):
                self.assertIn(k, src)
            self.assertEqual(len(src["selected_ids"]), src["k"])
            self.assertEqual(src["selected_indices"], list(range(src["k"])))
        import hashlib
        expect_sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
        self.assertEqual(m["output"]["sha256"], expect_sha)

    def test_nested_010_in_0100(self):
        # 用大一点的合成数据，保证 0.01 的 k>0（200*0.01=2, 300*0.01=3）
        build_synthetic(self.data_dir, self.img_dir, "ScienceQA", 200)
        build_synthetic(self.data_dir, self.img_dir, "TextVQA", 300)
        out10 = os.path.join(self.tmp, "r3_010.json")
        self.assertEqual(self._build(3, 0.1, out10).returncode, 0)
        out01 = os.path.join(self.tmp, "r3_001.json")
        r = self._build(3, 0.01, out01, extra=["--nested-with", out10 + ".manifest.json"])
        self.assertEqual(r.returncode, 0, r.stderr)
        d01 = {s["id"] for s in json.load(open(out01))}
        d10 = {s["id"] for s in json.load(open(out10))}
        self.assertTrue(d01.issubset(d10), f"{d01 - d10}")

    def test_nested_mismatch_fails(self):
        build_synthetic(self.data_dir, self.img_dir, "ScienceQA", 200)
        build_synthetic(self.data_dir, self.img_dir, "TextVQA", 300)
        out10 = os.path.join(self.tmp, "r3_010.json")
        self.assertEqual(self._build(3, 0.1, out10).returncode, 0)
        # 篡改外层 manifest 的 selected_ids 后断言失败
        mp = out10 + ".manifest.json"
        m = json.load(open(mp))
        m["sources"]["TextVQA"]["selected_ids"] = ["TextVQA_999"]
        json.dump(m, open(mp, "w"))
        out01 = os.path.join(self.tmp, "r3_001.json")
        r = self._build(3, 0.01, out01, extra=["--nested-with", mp])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("嵌套断言失败", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
