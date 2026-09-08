"""工单 1/4/6 端到端单测：run_replay_exp.sh DRY_RUN=1 全链路（零 GPU）。

覆盖：目录契约（results/<Task>/round<j>/）、checkpoint 链（task/replay 分离）、
manifest 不覆盖 + 恢复配置校验、validate_round 跳过、聚合 + .complete。
random-replay 系列（2026-09-08）追加：SAMPLE_MODE/REPLAY_SAMPLE_SEED 确实传入
构建器（sidecar manifest 断言）+ replay sample seed 不一致时禁止恢复。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, build_synthetic, make_aux

ORCH = os.path.join(ROOT, "scripts", "CoIN_Replay", "run_replay_exp.sh")


class TestOrchestratorDryRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_dry_")
        self.data_dir = os.path.join(self.tmp, "Instructions")
        self.img_dir = os.path.join(self.tmp, "images")
        self.ckpt = os.path.join(self.tmp, "ckpt")
        self.res = os.path.join(self.tmp, "res")
        os.makedirs(self.data_dir)
        os.makedirs(self.img_dir)
        for task in ("ScienceQA", "TextVQA"):
            build_synthetic(self.data_dir, self.img_dir, task, n=12)
            make_aux(self.img_dir, task)
        # 假模型路径（preflight 只查存在性）
        self.base = os.path.join(self.tmp, "base")
        self.vision = os.path.join(self.tmp, "vision")
        os.makedirs(self.base)
        os.makedirs(self.vision)
        json.dump({}, open(os.path.join(self.base, "config.json"), "w"))
        json.dump({}, open(os.path.join(self.vision, "config.json"), "w"))
        self.proj = os.path.join(self.tmp, "mm_projector.bin")
        with open(self.proj, "wb") as f:
            f.write(b"\x00\x01")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _env(self, **over):
        env = {
            "DRY_RUN": "1",
            "GPUS": "0",
            "BATCH": "2",
            "ACCUM": "1",
            "EPOCHS": "1",
            "REPLAY_EPOCHS": "1",
            "LR": "2e-4",
            "SEED": "1234",
            "DATA_SEED": "1234",
            "BASE_MODEL": self.base,
            "VISION_TOWER": self.vision,
            "PROJECTOR": self.proj,
            "DATA_DIR": self.data_dir,
            "IMG_DIR": self.img_dir,
            "CKPT_ROOT": self.ckpt,
            "RES_ROOT": self.res,
            "REPLAY_DATA_DIR": os.path.join(self.tmp, "replay"),
            "PREFLIGHT_ARGS": "--skip-pil",
            "TASKS_JSON": '["ScienceQA","TextVQA"]',
            "PREFLIGHT_REPORT": os.path.join(self.tmp, "preflight.json"),
            "PATH": os.environ.get("PATH", ""),
        }
        env.update(over)
        return env

    def _run(self, ratio="0.1", **over):
        env = dict(os.environ)
        env.update(self._env(**over))
        return subprocess.run(["bash", ORCH, ratio], capture_output=True,
                              text=True, cwd=ROOT, env=env, timeout=300)

    def test_full_dryrun_pipeline(self):
        r = self._run()
        self.assertEqual(r.returncode, 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        # 权威完成标志
        self.assertTrue(os.path.isfile(os.path.join(self.res, ".complete")))
        # 聚合产物
        m = json.load(open(os.path.join(self.res, "coin_metrics.json")))
        self.assertIn("MAA", m)
        self.assertIn("BWT", m)
        self.assertAlmostEqual(m["MAA"], 42.0)  # dry-run 产物 acc=42.0
        # manifest：显式配置 + effective_batch 含 world_size
        man = json.load(open(os.path.join(self.res, "run_manifest.json")))
        self.assertEqual(man["config"]["effective_batch"], 2 * 1 * 1)
        self.assertEqual(man["config"]["tasks"], ["ScienceQA", "TextVQA"])
        self.assertIn("config_hash", man)
        # 目录契约：results/<Task>/round<j>/
        for j in (1, 2):
            self.assertTrue(os.path.isfile(
                os.path.join(self.res, "round%d_manifest.json" % j)))
            for task in ("ScienceQA", "TextVQA"):
                if j >= 1 and task == "ScienceQA" or j == 2:
                    self.assertTrue(os.path.isdir(
                        os.path.join(self.res, task, "round%d" % j)))
        # checkpoint 链分离：task 与 replay 不同目录
        self.assertTrue(os.path.isdir(os.path.join(self.ckpt, "round1_task_llava_lora")))
        self.assertTrue(os.path.isdir(os.path.join(self.ckpt, "round2_task_llava_lora")))
        self.assertTrue(os.path.isdir(os.path.join(self.ckpt, "round2_replay_llava_lora")))
        # replay sidecar manifest 存在
        self.assertTrue(os.path.isfile(
            os.path.join(self.tmp, "replay", "round2_train.json.manifest.json")))
        # 无残留临时目录
        leftovers = [d for d in os.listdir(self.res) if d.startswith(".tmp_eval")]
        self.assertEqual(leftovers, [])

    def test_resume_same_config_skips(self):
        r1 = self._run()
        self.assertEqual(r1.returncode, 0, r1.stderr)
        r2 = self._run()
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("跳过", r2.stdout)
        # manifest 未被覆盖（创建时间不变）
        man_path = os.path.join(self.res, "run_manifest.json")
        st1 = os.stat(man_path)
        r3 = self._run()
        st2 = os.stat(man_path)
        self.assertEqual(st1.st_mtime_ns, st2.st_mtime_ns)

    def test_resume_config_mismatch_fails(self):
        r1 = self._run()
        self.assertEqual(r1.returncode, 0, r1.stderr)
        r2 = self._run(LR="3e-4")
        self.assertNotEqual(r2.returncode, 0)
        self.assertIn("config hash", r2.stdout + r2.stderr)

    def test_fault_injection_no_complete(self):
        # 在 round2 的 TextVQA 评估注入故障 → 整组失败，无 .complete，无新指标
        r = self._run(EVAL_FAULT_INJECT="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(os.path.isfile(os.path.join(self.res, ".complete")))
        # 旧结果不能被误用：round2 的 TextVQA 无产物
        self.assertFalse(os.path.isdir(os.path.join(self.res, "TextVQA", "round2")))

    # ---- random-replay 系列（2026-09-08）：编排层 plumbing --------------------

    def test_random_mode_plumbing_dryrun(self):
        """SAMPLE_MODE=random + REPLAY_SAMPLE_SEED 必须真正到达构建器。"""
        import hashlib
        r = self._run("0.1", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="424242")
        self.assertEqual(r.returncode, 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        # run manifest：config 记录 mode 与 sample seed（config hash 一部分）
        man = json.load(open(os.path.join(self.res, "run_manifest.json")))
        self.assertEqual(man["config"]["sample_mode"], "random")
        self.assertEqual(man["config"]["replay_sample_seed"], 424242)
        # replay sidecar manifest：构建器确实以 random+seed 运行（验收点 9 铁证）
        m = json.load(open(os.path.join(
            self.tmp, "replay", "round2_train.json.manifest.json")))
        self.assertEqual(m["mode"], "random")
        self.assertEqual(m["sample_seed"], 424242)
        self.assertEqual(m["sampling_algorithm"],
                         "sha256_task_seed_python_shuffle_v1")
        e = m["sources"]["ScienceQA"]
        self.assertEqual(e["task_seed"],
                         hashlib.sha256(b"424242:ScienceQA").hexdigest())
        # floor(12*0.1)=1：选中 1 条（具体哪条由 seed 决定）
        self.assertEqual(e["k"], 1)
        self.assertEqual(len(e["selected_indices"]), 1)
        data = json.load(open(os.path.join(self.tmp, "replay", "round2_train.json")))
        self.assertEqual(len(data), 1)

    def test_resume_replay_sample_seed_mismatch_fails(self):
        r1 = self._run("0.1", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="111")
        self.assertEqual(r1.returncode, 0, r1.stderr)
        r2 = self._run("0.1", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="222")
        self.assertNotEqual(r2.returncode, 0)
        out = r2.stdout + r2.stderr
        self.assertIn("config hash", out)
        self.assertIn("replay_sample_seed", out)  # 语义差异字段显式指出


if __name__ == "__main__":
    unittest.main(verbosity=2)
