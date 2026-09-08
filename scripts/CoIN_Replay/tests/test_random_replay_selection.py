"""random-replay 系列单测：随机 replay 样本选择（验收点 1-8）。

覆盖：
  1. prefix 输出与旧实现（HEAD 处 build_replay_data.py）逐字节一致
  2. prefix 改变 --seed 输出不变（seed 不参与 prefix 选择）
  3. random 相同 seed 重建两次：IDs/indices/输出 SHA 相同（确定性）
  4. random 不同 seed → 集合不同
  5. k 始终等于 floor(N*ratio)（两种模式）
  6. random 相同 seed：0.01 集合 ⊆ 0.10 集合（完整排列取前 k 性质）
  7. nested-with：seed / mode / source SHA 任一不一致 → 失败（random 模式）
  8. replay manifest 字段完整（mode/sample_seed/sampling_algorithm/task_seed/N/k/…）
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, REPLAY_DIR, build_synthetic, run

BUILD = [sys.executable, os.path.join(REPLAY_DIR, "build_replay_data.py")]
SAMPLING_ALGORITHM = "sha256_task_seed_python_shuffle_v1"


class TestRandomReplaySelection(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_rnd_")
        self.data_dir = os.path.join(self.tmp, "Instructions")
        self.img_dir = os.path.join(self.tmp, "images")
        os.makedirs(self.data_dir)
        os.makedirs(self.img_dir)
        # 300/500 条：0.01 时 k=3/5 > 0，0.07 时 21/35，0.1 时 30/50
        build_synthetic(self.data_dir, self.img_dir, "ScienceQA", 300)
        build_synthetic(self.data_dir, self.img_dir, "TextVQA", 500)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, ratio, out, mode="prefix", seed=1234, extra=None, tasks=None):
        cmd = BUILD + [
            "--tasks", *(tasks or ["ScienceQA", "TextVQA"]),
            "--data-dir", self.data_dir, "--image-dir", self.img_dir,
            "--round", "3", "--ratio", str(ratio),
            "--sample-mode", mode, "--seed", str(seed),
            "--out", out,
        ] + (extra or [])
        return run(cmd)

    def _legacy_build(self, ratio, out):
        """旧实现（baseline HEAD 的 build_replay_data.py）——验收点 1 的对照物。"""
        old_py = os.path.join(self.tmp, "build_replay_data_legacy.py")
        with open(old_py, "wb") as f:
            f.write(subprocess.check_output(
                ["git", "show", "HEAD:scripts/CoIN_Replay/build_replay_data.py"],
                cwd=ROOT))
        cmd = [sys.executable, old_py,
               "--tasks", "ScienceQA", "TextVQA",
               "--data-dir", self.data_dir, "--image-dir", self.img_dir,
               "--round", "3", "--ratio", str(ratio), "--seed", "1234",
               "--out", out]
        return run(cmd)

    # ---- 验收点 1/2：prefix 完全兼容 ----------------------------------------
    def test_1_prefix_byte_identical_to_legacy(self):
        out_new = os.path.join(self.tmp, "new_prefix.json")
        out_old = os.path.join(self.tmp, "old_prefix.json")
        r_new = self._build(0.1, out_new, mode="prefix")
        r_old = self._legacy_build(0.1, out_old)
        self.assertEqual(r_new.returncode, 0, r_new.stderr)
        self.assertEqual(r_old.returncode, 0, r_old.stderr)
        b_new = open(out_new, "rb").read()
        b_old = open(out_old, "rb").read()
        self.assertEqual(b_new, b_old, "prefix replay JSON 必须与旧实现逐字节一致")
        m_new = json.load(open(out_new + ".manifest.json"))
        m_old = json.load(open(out_old + ".manifest.json"))
        self.assertEqual(m_new["mode"], "prefix")
        for k in ("round", "ratio", "seed", "sources"):
            self.assertEqual(m_new[k], m_old[k], f"manifest.{k} 应与旧实现一致")

    def test_2_prefix_seed_irrelevant(self):
        o1 = os.path.join(self.tmp, "p1.json")
        o2 = os.path.join(self.tmp, "p2.json")
        self.assertEqual(self._build(0.1, o1, mode="prefix", seed=1).returncode, 0)
        self.assertEqual(self._build(0.1, o2, mode="prefix", seed=987654321).returncode, 0)
        self.assertEqual(open(o1, "rb").read(), open(o2, "rb").read(),
                         "prefix 模式改 --seed 不得改变输出")

    # ---- 验收点 3/4：random 确定性与差异性 -----------------------------------
    def test_3_random_deterministic_same_seed(self):
        o1 = os.path.join(self.tmp, "r1.json")
        o2 = os.path.join(self.tmp, "r2.json")
        self.assertEqual(self._build(0.1, o1, mode="random", seed=4242).returncode, 0)
        self.assertEqual(self._build(0.1, o2, mode="random", seed=4242).returncode, 0)
        self.assertEqual(open(o1, "rb").read(), open(o2, "rb").read(),
                         "random 同 seed 两次重建必须逐字节一致")
        m1 = json.load(open(o1 + ".manifest.json"))
        m2 = json.load(open(o2 + ".manifest.json"))
        for t in ("ScienceQA", "TextVQA"):
            for k in ("task_seed", "N", "k", "selected_ids", "selected_indices"):
                self.assertEqual(m1["sources"][t][k], m2["sources"][t][k])
        self.assertEqual(m1["output"]["sha256"], m2["output"]["sha256"])
        self.assertEqual(
            hashlib.sha256(open(o1, "rb").read()).hexdigest(), m1["output"]["sha256"])

    def test_4_random_diff_seed_differs(self):
        o1 = os.path.join(self.tmp, "s42.json")
        o2 = os.path.join(self.tmp, "s43.json")
        self.assertEqual(self._build(0.2, o1, mode="random", seed=42).returncode, 0)
        self.assertEqual(self._build(0.2, o2, mode="random", seed=43).returncode, 0)
        self.assertNotEqual(open(o1, "rb").read(), open(o2, "rb").read())
        m1 = json.load(open(o1 + ".manifest.json"))
        m2 = json.load(open(o2 + ".manifest.json"))
        for t in ("ScienceQA", "TextVQA"):
            self.assertNotEqual(
                m1["sources"][t]["selected_ids"], m2["sources"][t]["selected_ids"],
                f"{t}: 不同 seed 必须得到不同集合")

    # ---- 验收点 5：k = floor(N*ratio) ----------------------------------------
    def test_5_k_floor_both_modes(self):
        import math
        for mode in ("prefix", "random"):
            out = os.path.join(self.tmp, f"k_{mode}.json")
            r = self._build(0.07, out, mode=mode, seed=99)
            self.assertEqual(r.returncode, 0, r.stderr)
            m = json.load(open(out + ".manifest.json"))
            expect = {"ScienceQA": math.floor(300 * 0.07),
                      "TextVQA": math.floor(500 * 0.07)}
            for t, k in expect.items():
                self.assertEqual(m["sources"][t]["k"], k)
                self.assertEqual(len(m["sources"][t]["selected_indices"]), k)
                self.assertEqual(len(m["sources"][t]["selected_ids"]), k)
            self.assertEqual(m["output"]["N"], sum(expect.values()))

    # ---- 验收点 6：同 seed 下 0.01 ⊆ 0.10 ------------------------------------
    def test_6_random_subset_same_seed(self):
        out10 = os.path.join(self.tmp, "s10.json")
        r = self._build(0.1, out10, mode="random", seed=2026)
        self.assertEqual(r.returncode, 0, r.stderr)
        out01 = os.path.join(self.tmp, "s01.json")
        r = self._build(0.01, out01, mode="random", seed=2026,
                        extra=["--nested-with", out10 + ".manifest.json"])
        self.assertEqual(r.returncode, 0, r.stderr)
        d01 = {s["id"] for s in json.load(open(out01))}
        d10 = {s["id"] for s in json.load(open(out10))}
        self.assertTrue(d01.issubset(d10), f"0.01 不在 0.10 内: {sorted(d01 - d10)[:5]}")

    # ---- 验收点 7：nested-with 一致性校验失败路径 -----------------------------
    def _outer_manifest(self, mode="random", seed=4242):
        out10 = os.path.join(self.tmp, "outer.json")
        r = self._build(0.1, out10, mode=mode, seed=seed)
        self.assertEqual(r.returncode, 0, r.stderr)
        return out10

    def test_7a_nested_seed_mismatch_fails(self):
        outer = self._outer_manifest()
        out01 = os.path.join(self.tmp, "inner_badseed.json")
        r = self._build(0.01, out01, mode="random", seed=999,
                        extra=["--nested-with", outer + ".manifest.json"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("sample_seed", r.stdout + r.stderr)

    def test_7b_nested_mode_mismatch_fails(self):
        outer = self._outer_manifest(mode="prefix")  # 外层 prefix
        out01 = os.path.join(self.tmp, "inner_vs_prefix.json")
        r = self._build(0.01, out01, mode="random", seed=4242,
                        extra=["--nested-with", outer + ".manifest.json"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mode", r.stdout + r.stderr)

    def test_7c_nested_source_sha_mismatch_fails(self):
        outer = self._outer_manifest()
        mp = outer + ".manifest.json"
        m = json.load(open(mp))
        m["sources"]["TextVQA"]["sha256"] = "0" * 64
        json.dump(m, open(mp, "w"))
        out01 = os.path.join(self.tmp, "inner_badsha.json")
        r = self._build(0.01, out01, mode="random", seed=4242,
                        extra=["--nested-with", mp])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("SHA256", r.stdout + r.stderr)

    def test_7d_nested_subset_mismatch_still_fails(self):
        outer = self._outer_manifest()
        mp = outer + ".manifest.json"
        m = json.load(open(mp))
        m["sources"]["TextVQA"]["selected_ids"] = []
        json.dump(m, open(mp, "w"))
        out01 = os.path.join(self.tmp, "inner_notsubset.json")
        r = self._build(0.01, out01, mode="random", seed=4242,
                        extra=["--nested-with", mp])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("嵌套断言失败", r.stdout + r.stderr)

    # ---- 验收点 8：manifest 字段完整 + 可重建 ---------------------------------
    def test_8_random_manifest_fields_and_rebuild(self):
        seed = 777
        out = os.path.join(self.tmp, "full.json")
        r = self._build(0.1, out, mode="random", seed=seed)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(out + ".manifest.json"))
        for k in ("round", "ratio", "seed", "mode", "sample_seed",
                  "sampling_algorithm", "created_at", "sources", "output",
                  "nested_with"):
            self.assertIn(k, m, f"manifest 缺 {k}")
        self.assertEqual(m["mode"], "random")
        self.assertEqual(m["sample_seed"], seed)
        self.assertEqual(m["sampling_algorithm"], SAMPLING_ALGORITHM)
        self.assertIsNone(m["nested_with"])
        srcs = {t: json.load(open(os.path.join(self.data_dir, t, "train.json")))
                for t in ("ScienceQA", "TextVQA")}
        for t, n in (("ScienceQA", 300), ("TextVQA", 500)):
            src = srcs[t]
            e = m["sources"][t]
            for k in ("path", "sha256", "N", "k", "task_seed",
                      "selected_ids", "selected_indices"):
                self.assertIn(k, e, f"{t} 缺 {k}")
            self.assertEqual(e["N"], n)
            self.assertEqual(e["k"], n // 10)  # floor(N*0.1)
            # task_seed = sha256("<seed>:<task>")
            expect_ts = hashlib.sha256(f"{seed}:{t}".encode()).hexdigest()
            self.assertEqual(e["task_seed"], expect_ts)
            # 确定性重建：同 task seed 洗牌全量索引，前 k 必须命中
            perm = list(range(n))
            rng = random.Random(int(e["task_seed"], 16))
            rng.shuffle(perm)
            self.assertEqual(e["selected_indices"], perm[: e["k"]])
            picked = [src[i]["id"] for i in e["selected_indices"]]
            self.assertEqual(e["selected_ids"], picked)
            # 源 SHA256 与真实文件一致
            self.assertEqual(
                e["sha256"],
                hashlib.sha256(open(os.path.join(self.data_dir, t, "train.json"),
                                    "rb").read()).hexdigest())
        self.assertEqual(m["output"]["N"], 30 + 50)
        self.assertEqual(m["output"]["sha256"],
                         hashlib.sha256(open(out, "rb").read()).hexdigest())

    def test_8b_prefix_manifest_schema_unchanged(self):
        out = os.path.join(self.tmp, "pre.json")
        r = self._build(0.1, out, mode="prefix", seed=1234)
        self.assertEqual(r.returncode, 0, r.stderr)
        m = json.load(open(out + ".manifest.json"))
        self.assertEqual(m["mode"], "prefix")
        for k in ("sample_seed", "sampling_algorithm"):
            self.assertNotIn(k, m, f"prefix manifest 不应新增 {k}")
        for t in ("ScienceQA", "TextVQA"):
            self.assertNotIn("task_seed", m["sources"][t],
                             "prefix source 不应新增 task_seed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
