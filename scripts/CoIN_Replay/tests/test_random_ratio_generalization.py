"""random-replay ratio 通用化单测（2026-09-11）——零 GPU、零依赖。

覆盖：
  - coin_lib ratio 归一化 / tag 派生 / 目录布局（数值比较；非法值拒绝）
  - registry：元数据驱动（不再写死 r001）、追加无上限（>5）、同 ratio seed 复用拒绝、
    跨 ratio 同 seed 配对允许且必须真同 seed
  - 配对比较工具：按 seed 取交集（不按 run_number）、UNPAIRED 展示、差值方向 0.10−0.01
  - 嵌套验证工具：真实构建器下 0.01 ⊆ 0.10（前缀关系）+ A/B 重建逐字节一致
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, build_synthetic

sys.path.insert(0, os.path.join(ROOT, "scripts", "CoIN_Replay"))
from coin_lib import (  # noqa: E402
    ALLOWED_RANDOM_RATIOS, normalize_ratio, random_ratio_layout, ratio_tag)

REG = os.path.join(ROOT, "scripts", "CoIN_Replay", "tools",
                   "random_replay_registry.py")
PAIR = os.path.join(ROOT, "scripts", "CoIN_Replay", "tools", "random_replay_pair.py")
NESTED = os.path.join(ROOT, "scripts", "CoIN_Replay", "tools",
                      "random_replay_nested_check.py")
DOC_R001 = os.path.join(ROOT, "docs", "experiments", "coin_replay_random_r001")
DOC_R010 = os.path.join(ROOT, "docs", "experiments", "coin_replay_random_r010")


def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


class TestRatioHelpers(unittest.TestCase):

    def test_normalize_and_tag_numeric(self):
        from decimal import Decimal
        for raw in ("0.01", "0.010", 0.01, Decimal("0.01")):
            self.assertEqual(normalize_ratio(raw), Decimal("0.01"))
            self.assertEqual(ratio_tag(raw), "r001")
        for raw in ("0.1", "0.10", 0.1, "0.100", Decimal("0.10")):
            self.assertEqual(normalize_ratio(raw), Decimal("0.10"))
            self.assertEqual(ratio_tag(raw), "r010")
        # 数值相等 ⇒ 同一 tag（不做字符串比较）
        self.assertEqual(ratio_tag("0.1"), ratio_tag("0.10"))

    def test_normalize_rejects_disallowed(self):
        from decimal import Decimal
        for bad in ("0.02", "0.5", "0.2", "-0.1", "abc", "", "NaN", Decimal("0"),
                    [0.1], None):
            with self.assertRaises(ValueError, msg=f"{bad!r} 必须被拒绝"):
                normalize_ratio(bad)

    def test_layout_paths(self):
        lay = random_ratio_layout("0.10", "run_0001_seed_1")
        self.assertEqual(lay["ratio_tag"], "r010")
        self.assertEqual(lay["ckpt_root"],
                         "checkpoints/CoIN_Replay_random/r010/run_0001_seed_1")
        self.assertEqual(lay["res_root"],
                         "results/CoIN_Replay_random/r010/run_0001_seed_1")
        self.assertEqual(lay["replay_data_dir"],
                         "playground/Replay_random/r010/run_0001_seed_1")
        lay01 = random_ratio_layout("0.01", "run_0002_seed_2")
        self.assertEqual(lay01["ratio_tag"], "r001")

    def test_cli_ratio_tag_and_layout(self):
        lib = os.path.join(ROOT, "scripts", "CoIN_Replay", "coin_lib.py")
        r = run([sys.executable, lib, "ratio-tag", "0.1"])
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), "r010")
        r = run([sys.executable, lib, "ratio-tag", "0.07"])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ERROR", r.stderr)
        r = run([sys.executable, lib, "ratio-layout", "0.01", "run_0003_seed_9"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["ratio_tag"], "r001")


class TestRegistryRatioGeneric(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_reg_g_")
        self.doc = os.path.join(self.tmp, "r001")
        shutil.copytree(DOC_R001, self.doc)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _reg(self, docdir, *args):
        return run([sys.executable, REG, docdir, *args])

    def test_unlimited_appends_beyond_five(self):
        """r001 已有 5 个 run，再追加 2 个（第 6、7 个）必须正常。"""
        idx0 = json.load(open(os.path.join(self.doc, "index.json")))
        self.assertEqual(len(idx0["runs"]), 5)
        for seed in (900001, 900002):
            r = self._reg(self.doc, "register", "--seed", str(seed))
            self.assertEqual(r.returncode, 0, r.stderr)
        idx = json.load(open(os.path.join(self.doc, "index.json")))
        self.assertEqual([x["run_number"] for x in idx["runs"]], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(idx["runs"][5]["run_id"], "run_0006_seed_900001")
        self.assertEqual(idx["runs"][6]["run_id"], "run_0007_seed_900002")
        # 三件套同步
        import csv as _csv
        rows = list(_csv.DictReader(open(os.path.join(self.doc, "index.csv"))))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[5]["ratio_tag"], "r001")
        readme = open(os.path.join(self.doc, "README.md")).read()
        self.assertIn("run_0007_seed_900002", readme)
        # 新 run 的结果目录来自元数据（不是写死的 r001 字符串常量）
        self.assertEqual(idx["runs"][5]["result_directory"],
                         "results/CoIN_Replay_random/r001/run_0006_seed_900001")

    def test_same_ratio_seed_reuse_rejected(self):
        used = json.load(open(os.path.join(self.doc, "index.json")))["runs"][0]
        r = self._reg(self.doc, "register", "--seed", str(used["replay_sample_seed"]))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("禁止复用", r.stderr)

    def test_missing_metadata_rejected(self):
        p = os.path.join(self.doc, "index.json")
        idx = json.load(open(p))
        del idx["result_directory_root"]
        json.dump(idx, open(p, "w"))
        r = self._reg(self.doc, "register", "--seed", "5")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("缺系列元数据", r.stderr)

    def test_inconsistent_ratio_tag_rejected(self):
        p = os.path.join(self.doc, "index.json")
        idx = json.load(open(p))
        idx["ratio_tag"] = "r010"  # 与 ratio 0.01 不符
        json.dump(idx, open(p, "w"))
        r = self._reg(self.doc, "register", "--seed", "5")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("不一致", r.stderr)

    def test_cross_ratio_same_seed_pairing(self):
        """跨 ratio 同 seed 允许，且必须指向真同 seed 的 run。"""
        self.assertTrue(os.path.isdir(DOC_R010), DOC_R010)
        r010 = os.path.join(self.tmp, "r010")
        shutil.copytree(DOC_R010, r010)
        r001_index = os.path.join(self.doc, "index.json")
        seed = 358341059
        r = self._reg(r010, "register", "--seed", str(seed),
                      "--paired-with", "run_0001_seed_358341059",
                      "--paired-index", r001_index)
        self.assertEqual(r.returncode, 0, r.stderr)
        idx = json.load(open(os.path.join(r010, "index.json")))
        row = idx["runs"][0]
        self.assertEqual(row["ratio_tag"], "r010")
        self.assertEqual(row["result_directory"],
                         "results/CoIN_Replay_random/r010/run_0001_seed_358341059")
        self.assertEqual(row["paired_with"]["run_id"], "run_0001_seed_358341059")
        self.assertEqual(row["paired_with"]["replay_sample_seed"], seed)
        # CSV 扁平字段
        import csv as _csv
        rows = list(_csv.DictReader(open(os.path.join(r010, "index.csv"))))
        self.assertEqual(rows[0]["paired_with_run_id"], "run_0001_seed_358341059")

    def test_cross_ratio_pairing_seed_mismatch_rejected(self):
        r010 = os.path.join(self.tmp, "r010b")
        shutil.copytree(DOC_R010, r010)
        r = self._reg(r010, "register", "--seed", "777",
                      "--paired-with", "run_0001_seed_358341059",
                      "--paired-index", os.path.join(self.doc, "index.json"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("配对必须同 seed", r.stderr)

    def test_cross_ratio_pairing_requires_index(self):
        r010 = os.path.join(self.tmp, "r010c")
        shutil.copytree(DOC_R010, r010)
        r = self._reg(r010, "register", "--seed", "778",
                      "--paired-with", "run_0001_seed_358341059")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--paired-index", r.stderr)


class TestPairTool(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_pair_")
        self.a = self._series("r001", "0.01", "r001", [
            # (run_number, seed, status, MAA, BWT, final_avg)
            (1, 111, "COMPLETE", 60.0, -10.0, 50.0),
            (2, 222, "COMPLETE", 55.0, -20.0, 45.0),
            (3, 333, "FAILED", None, None, None),
        ])
        self.b = self._series("r010", "0.1", "r010", [
            (1, 222, "RUNNING", None, None, None),   # run_number 相同但 seed 不同
            (2, 111, "COMPLETE", 62.0, -2.0, 58.0),
            (3, 444, "COMPLETE", 70.0, 5.0, 66.0),
        ])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _series(self, name, ratio, tag, rows):
        d = os.path.join(self.tmp, name)
        os.makedirs(os.path.join(d, "runs"), exist_ok=True)
        runs = []
        for rn, seed, status, maa, bwt, favg in rows:
            rid = f"run_{rn:04d}_seed_{seed}"
            runs.append({"run_number": rn, "run_id": rid, "replay_sample_seed": seed,
                         "status": status, "MAA": maa, "BWT": bwt,
                         "ratio": float(ratio), "ratio_tag": tag,
                         "result_directory": f"results/CoIN_Replay_random/{tag}/{rid}",
                         "result_commit": (f"{rn:040d}" if status == "COMPLETE" else None),
                         "code_commit": "f" * 40, "error_summary": None})
            if status == "COMPLETE":
                os.makedirs(os.path.join(d, "runs", rid), exist_ok=True)
                json.dump({"run_id": rid, "ratio": float(ratio), "ratio_tag": tag,
                           "MAA": maa, "BWT": bwt, "final_avg": favg,
                           "replay_sample_seed": seed, "code_commit": "f" * 40},
                          open(os.path.join(d, "runs", rid, "summary.json"), "w"))
        json.dump({"schema_version": 1, "ratio": float(ratio), "ratio_tag": tag,
                   "sample_mode": "random",
                   "result_directory_root": f"results/CoIN_Replay_random/{tag}",
                   "runs": runs}, open(os.path.join(d, "index.json"), "w"))
        return d

    def _pair(self):
        out_json = os.path.join(self.tmp, "paired_comparison.json")
        out_md = os.path.join(self.tmp, "PAIRED_REPORT.md")
        r = run([sys.executable, PAIR, "--series", self.a, "--series", self.b,
                 "--out-json", out_json, "--out-md", out_md])
        return r, out_json, out_md

    def test_seed_intersection_pairing_and_direction(self):
        r, out_json, out_md = self._pair()
        self.assertEqual(r.returncode, 0, r.stderr)
        p = json.load(open(out_json))
        # 配对按 seed 交集：seed 111 配对（忽略 run_number 差异）
        seeds = sorted(x["seed"] for x in p["pairs"])
        self.assertEqual(seeds, [111, 222])
        pair111 = [x for x in p["pairs"] if x["seed"] == 111][0]
        self.assertEqual(pair111["low_ratio"]["run_id"], "run_0001_seed_111")
        self.assertEqual(pair111["high_ratio"]["run_id"], "run_0002_seed_111")
        self.assertTrue(pair111["paired_complete"])
        self.assertEqual(p["delta_definition"], "random-0.10 − random-0.01")
        # 差值方向 = 0.10 − 0.01
        self.assertAlmostEqual(pair111["delta"]["MAA"], 2.0)
        self.assertAlmostEqual(pair111["delta"]["BWT"], 8.0)
        self.assertAlmostEqual(pair111["delta"]["final_avg"], 8.0)
        # 222 对里有一个 RUNNING → 不计入配对均值
        pair222 = [x for x in p["pairs"] if x["seed"] == 222][0]
        self.assertFalse(pair222["paired_complete"])
        self.assertIsNone(pair222["delta"]["MAA"])
        # 配对均值：仅 1 个完整对
        self.assertEqual(p["paired_means"]["n_paired_complete"], 1)
        self.assertAlmostEqual(p["paired_means"]["metrics"]["MAA"]["mean_delta"], 2.0)
        # 未配对 run：A 的 seed 333、B 的 seed 444
        unpaired = {(u["series"], u["seed"]) for u in p["unpaired"]}
        self.assertEqual(unpaired, {("r001", 333), ("r010", 444)})
        # 失败 run 必须展示（不隐藏）
        md = open(out_md).read()
        self.assertIn("run_0003_seed_333", md)
        self.assertIn("FAILED", md)
        self.assertIn("UNPAIRED", md)
        self.assertIn("random-0.10 − random-0.01", md)
        # 各自 n 分别显示，不合并
        self.assertIn("n_total", md)
        # 输出不得含绝对路径（发布物卫生）
        blob = json.dumps(p) + md
        for bad in ("/home/", "/tmp/", "/root/", "/mnt/"):
            self.assertNotIn(bad, blob, f"配对产物含绝对路径 {bad}")

    def test_series_with_single_missing_index_fails(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        r = run([sys.executable, PAIR, "--series", self.a, "--series", empty,
                 "--out-json", os.path.join(self.tmp, "x.json"),
                 "--out-md", os.path.join(self.tmp, "x.md")])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("不是系列注册表目录", r.stderr + r.stdout)


class TestNestedCheckTool(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_nested_")
        self.data = os.path.join(self.tmp, "data")
        self.img = os.path.join(self.tmp, "img")
        os.makedirs(self.data)
        os.makedirs(self.img)
        build_synthetic(self.data, self.img, "ScienceQA", 300)
        build_synthetic(self.data, self.img, "TextVQA", 500)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_nested_and_ab_rebuild(self):
        report = os.path.join(self.tmp, "nested_report.json")
        r = run([sys.executable, NESTED, "--data-dir", self.data,
                 "--image-dir", self.img, "--tasks", "ScienceQA", "TextVQA",
                 "--seed", "358341059", "--tmp-dir", os.path.join(self.tmp, "work"),
                 "--out-report", report,
                 "--rounds", "2", "3"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rep = json.load(open(report))
        self.assertTrue(rep["pass"])
        for row in rep["rounds"]:
            self.assertTrue(row["ab_byte_identical"])
            ratio = float(row["ratio"])
            for task, e in row["tasks"].items():
                n = e["N"]
                self.assertEqual(e["k"], int(n * ratio))
                self.assertTrue(e["indices_valid"])
        for n in rep["nested_check"]:
            self.assertTrue(n["pass"], n)
            for task, e in n["tasks"].items():
                self.assertTrue(all(e["checks"].values()), f"{task}: {e['checks']}")
                self.assertEqual(e["k01"] < e["k10"], True)

    def test_expect_counts_mismatch_fails(self):
        report = os.path.join(self.tmp, "bad_report.json")
        r = run([sys.executable, NESTED, "--data-dir", self.data,
                 "--image-dir", self.img, "--tasks", "ScienceQA",
                 "--seed", "1", "--tmp-dir", os.path.join(self.tmp, "work2"),
                 "--out-report", report, "--rounds", "2",
                 "--expect-counts", json.dumps({"0.10": {"tasks": {"ScienceQA": 999}}})])
        self.assertNotEqual(r.returncode, 0)
        rep = json.load(open(report))
        self.assertFalse(rep["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
