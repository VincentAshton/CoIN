"""sync/check 工具单测：README/registry 数字漂移与脱敏缺口必须被抓出。

用真实仓库的 docs/experiments 子树做夹具（复制到临时 repo-root 后变异），保证测试与
线上目录结构一致、变异点就是真实漂移点。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TOOL = os.path.join(ROOT, "scripts", "CoIN_Replay", "tools", "random_replay_sync_check.py")
DOCS_SRC = os.path.join(ROOT, "docs", "experiments")


class TestSyncCheck(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_sync_")
        self.root = os.path.join(self.tmp, "repo")
        dst = os.path.join(self.root, "docs", "experiments")
        os.makedirs(dst)
        for name in sorted(os.listdir(DOCS_SRC)):
            if name.startswith("coin_replay_random"):
                shutil.copytree(os.path.join(DOCS_SRC, name), os.path.join(dst, name))
        self.entry = os.path.join(dst, "coin_replay_random")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, root=None, extra=()):
        return subprocess.run([sys.executable, TOOL, "--repo-root", root or self.root, *extra],
                              capture_output=True, text=True)

    def _sub(self, rel, old, new, count=-1):
        p = os.path.join(self.root, rel)
        txt = open(p, encoding="utf-8").read()
        self.assertIn(old, txt, f"夹具缺待替换文本: {rel}")
        open(p, "w", encoding="utf-8").write(txt.replace(old, new, count))

    # ---- 基线：真实仓库与未变异副本都必须通过 -----------------------------

    def test_real_repo_passes(self):
        r = self._run(root=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("FAIL 0", r.stdout)

    def test_fixture_copy_passes(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    # ---- README/registry 漂移 -------------------------------------------------

    def test_series_count_mismatch_fails(self):
        self._sub("docs/experiments/coin_replay_random_r001/README.md",
                  "当前 5 个 COMPLETE", "当前 4 个 COMPLETE")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("状态行", r.stdout)

    def test_registry_table_metric_mismatch_fails(self):
        self._sub("docs/experiments/coin_replay_random_r010/README.md",
                  "| 60.122 | 28.7287 |", "| 60.999 | 28.7287 |")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("registry 表内容", r.stdout)

    def test_entry_metric_mismatch_fails(self):
        self._sub("docs/experiments/coin_replay_random/README.md",
                  "MAA 60.122 / CoIN BWT +28.7287", "MAA 60.121 / CoIN BWT +28.7287")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("总入口 r010 指标", r.stdout)

    def test_index_csv_mismatch_fails(self):
        p = os.path.join(self.root, "docs/experiments/coin_replay_random_r010/index.csv")
        txt = open(p, encoding="utf-8").read()
        open(p, "w", encoding="utf-8").write(txt.replace("60.122", "61.122"))
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("index.csv", r.stdout)

    # ---- 配对产物 -------------------------------------------------------------

    def test_paired_json_tamper_fails(self):
        p = os.path.join(self.entry, "paired_comparison.json")
        d = json.load(open(p, encoding="utf-8"))
        d["paired_means"]["metrics"]["BWT"]["mean_delta"] = 19.9999
        json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("paired_comparison.json 重算一致", r.stdout)

    def test_paired_report_n_mismatch_fails(self):
        self._sub("docs/experiments/coin_replay_random/PAIRED_REPORT.md",
                  "配对 n（完整 COMPLETE 对）: **1**", "配对 n（完整 COMPLETE 对）: **2**")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("PAIRED_REPORT.md 配对数", r.stdout)

    # ---- 脱敏 / 链接 ----------------------------------------------------------

    def test_new_doc_with_absolute_path_fails(self):
        p = os.path.join(self.entry, "NEW_NOTE.md")
        open(p, "w", encoding="utf-8").write("日志在 /root/data/coin/logs 下\n")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("脱敏扫描（新增文档）", r.stdout)

    def test_legacy_exempt_warns_then_strict_fails(self):
        r = self._run()
        self.assertEqual(r.returncode, 0)
        self.assertIn("legacy 例外", r.stdout)
        r2 = self._run(extra=("--strict",))
        self.assertEqual(r2.returncode, 1)
        self.assertIn("--strict", r2.stdout)

    def test_broken_relative_link_fails(self):
        self._sub("docs/experiments/coin_replay_random/README.md",
                  "](../coin_replay_random_r010/README.md)", "](../coin_replay_random_r010/NOPE.md)")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("相对链接", r.stdout)


if __name__ == "__main__":
    unittest.main()
