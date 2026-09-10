"""random-replay 注册表边界测试（2026-09-08 审计加固）。

覆盖：
  - register → RUNNING；complete COMPLETE/FAILED 生命周期；禁止二次 complete
  - complete --status RUNNING 必须非零退出，且 index.json/index.csv/README.md 三份文件
    逐字节不变（临时目录副本上运行，绝不碰真实 docs/ 注册表）
  - complete 缺 --registration-commit / COMPLETE 缺 --result-commit → 非零
  - seed 复用拒绝
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT

TOOL = os.path.join(ROOT, "scripts", "CoIN_Replay", "tools",
                    "random_replay_registry.py")
DOCSRC = os.path.join(ROOT, "docs", "experiments", "coin_replay_random_r001")


def sha3(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()


class TestRandomReplayRegistry(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_reg_")
        self.docdir = os.path.join(self.tmp, "registry")
        shutil.copytree(DOCSRC, self.docdir)
        # 空注册表夹具：清空 runs 并用 registry 自身重写三件套。测试断言 run_0001 起编号，
        # 不能依赖线上注册表当前累积的运行数（线上已有多轮 COMPLETE）。
        sys.path.insert(0, os.path.join(ROOT, "scripts", "CoIN_Replay", "tools"))
        import random_replay_registry as R
        idx = R.load_index(self.docdir)
        idx["runs"] = []
        R.write_index(self.docdir, idx)
        self.three = [os.path.join(self.docdir, "index.json"),
                      os.path.join(self.docdir, "index.csv"),
                      os.path.join(self.docdir, "README.md")]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _reg(self, *args):
        return subprocess.run([sys.executable, TOOL, self.docdir, *args],
                              capture_output=True, text=True)

    def _load_index(self):
        return json.load(open(os.path.join(self.docdir, "index.json")))

    def test_register_complete_lifecycle(self):
        r = self._reg("register", "--seed", "424242")
        self.assertEqual(r.returncode, 0, r.stderr)
        idx = self._load_index()
        self.assertEqual(len(idx["runs"]), 1)
        row = idx["runs"][0]
        self.assertEqual(row["run_id"], "run_0001_seed_424242")
        self.assertEqual(row["replay_sample_seed"], 424242)
        self.assertEqual(row["status"], "RUNNING")
        # csv 同步
        rows = list(csv.DictReader(open(os.path.join(self.docdir, "index.csv"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "run_0001_seed_424242")
        # README 表行
        readme = open(os.path.join(self.docdir, "README.md")).read()
        self.assertIn("| run_0001_seed_424242 | 424242 |  |  | RUNNING |  |", readme)
        # 二次 register 同 seed 拒绝
        r2 = self._reg("register", "--seed", "424242")
        self.assertNotEqual(r2.returncode, 0)
        self.assertIn("禁止复用", r2.stderr)
        # complete COMPLETE（协议 A/C hash 必填）
        r3 = self._reg("complete", "--run-id", "run_0001_seed_424242",
                       "--status", "COMPLETE", "--maa", "60.4406",
                       "--bwt", "-13.6299", "--registration-commit", "a" * 40,
                       "--result-commit", "c" * 40)
        self.assertEqual(r3.returncode, 0, r3.stderr)
        row = self._load_index()["runs"][0]
        self.assertEqual(row["status"], "COMPLETE")
        self.assertEqual(row["MAA"], 60.4406)
        self.assertEqual(row["BWT"], -13.6299)
        self.assertEqual(row["registration_commit"], "a" * 40)
        self.assertEqual(row["result_commit"], "c" * 40)
        # 二次 complete 拒绝（非 RUNNING）
        r4 = self._reg("complete", "--run-id", "run_0001_seed_424242",
                       "--status", "COMPLETE", "--maa", "1.0", "--bwt", "0.0",
                       "--registration-commit", "a" * 40, "--result-commit", "c" * 40)
        self.assertNotEqual(r4.returncode, 0)
        self.assertIn("只有 RUNNING 可完结", r4.stderr)

    def test_complete_status_running_rejected_files_unchanged(self):
        r = self._reg("register", "--seed", "7")
        self.assertEqual(r.returncode, 0, r.stderr)
        before = sha3(self.three)
        # argparse 层拒绝 RUNNING
        r1 = self._reg("complete", "--run-id", "run_0001_seed_7",
                       "--status", "RUNNING")
        self.assertNotEqual(r1.returncode, 0)
        self.assertIn("invalid choice", r1.stderr)
        self.assertEqual(sha3(self.three), before,
                         "complete RUNNING 被拒后三份注册文件必须逐字节不变")
        # 缺 registration_commit 也拒绝且不变
        r2 = self._reg("complete", "--run-id", "run_0001_seed_7",
                       "--status", "COMPLETE", "--maa", "1.0", "--bwt", "2.0")
        self.assertNotEqual(r2.returncode, 0)
        self.assertIn("--registration-commit", r2.stderr)
        self.assertEqual(sha3(self.three), before)
        # COMPLETE 缺 result_commit 拒绝且不变
        r3 = self._reg("complete", "--run-id", "run_0001_seed_7",
                       "--status", "COMPLETE", "--maa", "1.0", "--bwt", "2.0",
                       "--registration-commit", "a" * 40)
        self.assertNotEqual(r3.returncode, 0)
        self.assertIn("--result-commit", r3.stderr)
        self.assertEqual(sha3(self.three), before)
        # 状态仍 RUNNING
        self.assertEqual(self._load_index()["runs"][0]["status"], "RUNNING")

    def test_complete_failed_requires_error_summary(self):
        self.assertEqual(self._reg("register", "--seed", "99").returncode, 0)
        r1 = self._reg("complete", "--run-id", "run_0001_seed_99",
                       "--status", "FAILED", "--registration-commit", "a" * 40)
        self.assertNotEqual(r1.returncode, 0)
        self.assertIn("--error-summary", r1.stderr)
        r2 = self._reg("complete", "--run-id", "run_0001_seed_99",
                       "--status", "FAILED", "--error-summary", "脱敏：训练崩溃",
                       "--registration-commit", "a" * 40)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        row = self._load_index()["runs"][0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIsNone(row["MAA"])
        self.assertIsNone(row["BWT"])
        self.assertIn("训练崩溃", row["error_summary"])

    def test_register_auto_seed_and_run_number_monotonic(self):
        r1 = self._reg("register")
        self.assertEqual(r1.returncode, 0, r1.stderr)
        rid1 = json.loads(r1.stdout[r1.stdout.index("{"):r1.stdout.rindex("}") + 1])
        self.assertEqual(rid1["run_id"], "run_0001_seed_%d" % rid1["replay_sample_seed"])
        r2 = self._reg("register")
        rid2 = json.loads(r2.stdout[r2.stdout.index("{"):r2.stdout.rindex("}") + 1])
        self.assertEqual(rid2["run_number"], 2)
        self.assertEqual(rid2["run_id"], "run_0002_seed_%d" % rid2["replay_sample_seed"])
        self.assertNotEqual(rid1["replay_sample_seed"], rid2["replay_sample_seed"])
        idx = self._load_index()
        self.assertEqual([r["run_number"] for r in idx["runs"]], [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
