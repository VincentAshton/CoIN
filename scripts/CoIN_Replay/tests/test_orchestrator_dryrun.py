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

    # ---- eval 路径门禁 + 分阶段完成标记（2026-09-11 修正）--------------------

    def test_eval_path_gate_passes_without_relative_symlinks(self):
        """门禁在训练前 PASS；且 ROOT 下没有 ./checkpoints、./cl_dataset

        （评估走显式 MODEL_BASE/IMAGE_FOLDER，不再依赖 worktree 软链——实踩缺口）。
        """
        self.assertFalse(os.path.exists(os.path.join(ROOT, "checkpoints")))
        self.assertFalse(os.path.exists(os.path.join(ROOT, "cl_dataset")))
        r = self._run()
        self.assertEqual(r.returncode, 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        self.assertIn("eval 路径门禁 PASS", r.stdout)

    def test_eval_path_gate_failfast_before_any_training(self):
        """eval 阶段路径不可解析 → 训练开始前就失败（不产生 ckpt / round 产物）。"""
        r = self._run(IMG_DIR=os.path.join(self.tmp, "nonexistent_dataset"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("eval 路径门禁 FAIL", r.stdout + r.stderr)
        self.assertFalse(os.path.isfile(os.path.join(self.res, ".round1_train_done")))
        self.assertFalse(os.path.isfile(os.path.join(self.res, "round1_manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(self.ckpt, "round1_task_llava_lora")))

    def test_train_marker_skips_retrain_after_eval_failure(self):
        """评估失败后恢复：训练段标记 + ckpt 校验通过 → 只重做评估，不重训 task 段。

        EVAL_FAULT_INJECT 在**每个**评估调用里注入（DRY_RUN 下最先触发的是 round1 的评估），
        因此 round1 训练完成 → 标记写入 → 评估失败；恢复时必须跳过 round1 训练。
        """
        r1 = self._run(EVAL_FAULT_INJECT="1")
        self.assertNotEqual(r1.returncode, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.res, ".round1_train_done")))
        self.assertFalse(os.path.isfile(os.path.join(self.res, ".round1_done")))
        task = os.path.join(self.ckpt, "round1_task_llava_lora", "adapter_model.bin")
        t_task = os.stat(task).st_mtime_ns
        r2 = self._run()  # 去掉故障注入 → 恢复
        self.assertEqual(r2.returncode, 0, f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
        self.assertIn("训练段已完成", r2.stdout)
        self.assertIn("跳过 task/replay 训练", r2.stdout)
        # 训练产物未被重写（mtime 不变 = 没有重训）
        self.assertEqual(os.stat(task).st_mtime_ns, t_task)
        self.assertTrue(os.path.isfile(os.path.join(self.res, ".complete")))

    def test_train_marker_not_trusted_when_ckpt_invalid(self):
        """标记存在但 ckpt 被破坏 → 不轻信标记，删标记重训后仍能完成。"""
        r1 = self._run(EVAL_FAULT_INJECT="1")
        self.assertNotEqual(r1.returncode, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.res, ".round1_train_done")))
        shutil.rmtree(os.path.join(self.ckpt, "round1_task_llava_lora"))
        r2 = self._run()
        self.assertIn("ckpt 校验失败", r2.stdout)
        self.assertEqual(r2.returncode, 0, f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
        self.assertTrue(os.path.isfile(os.path.join(self.res, ".complete")))

    # ---- random-replay 系列（ratio 通用）：编排层 plumbing + 启动门 ------------

    def _run_dirs(self, run_id, tag="r001"):
        """random gate 合规的 per-run 目录（pattern: <root>/CoIN_Replay_random/<tag>/<rid>）。"""
        base = os.path.join(self.tmp, "run_root", tag, run_id)
        return (os.path.join(base, "checkpoints", "CoIN_Replay_random", tag, run_id),
                os.path.join(base, "results", "CoIN_Replay_random", tag, run_id),
                os.path.join(base, "playground", "Replay_random", tag, run_id))

    def _big_data(self):
        """random 系列 ratio=0.01 需要 floor(N*0.01)>=1 → N>=100。"""
        build_synthetic(self.data_dir, self.img_dir, "ScienceQA", 120)
        build_synthetic(self.data_dir, self.img_dir, "TextVQA", 120)

    def _four_task_data(self, n=120):
        """random 0.10 四轮 DRY_RUN 用：四任务合成数据 + 评估辅助文件。"""
        for task in ("ScienceQA", "TextVQA", "ImageNet", "GQA"):
            build_synthetic(self.data_dir, self.img_dir, task, n)
            make_aux(self.img_dir, task)

    def test_random_mode_plumbing_dryrun(self):
        """SAMPLE_MODE=random + REPLAY_SAMPLE_SEED + RANDOM_REPLAY_RUN_ID 全链路（r001）。"""
        import hashlib
        self._big_data()
        rid = "run_0000_seed_424242"
        ckpt, res, replay = self._run_dirs(rid)
        r = self._run("0.01", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="424242",
                      RANDOM_REPLAY_RUN_ID=rid,
                      REPLAY_ACCUM="1",
                      CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertEqual(r.returncode, 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        self.assertIn("random-replay gate PASS", r.stdout)
        self.assertIn("tag=r001", r.stdout)
        # run manifest：config 记录 mode/seed/run id（config hash 一部分）
        man = json.load(open(os.path.join(res, "run_manifest.json")))
        self.assertEqual(man["config"]["sample_mode"], "random")
        self.assertEqual(man["config"]["replay_sample_seed"], 424242)
        self.assertEqual(man["config"]["random_replay_run_id"], rid)
        # replay sidecar manifest：构建器确实以 random+seed 运行（验收点 9 铁证）
        m = json.load(open(os.path.join(
            replay, "round2_train.json.manifest.json")))
        self.assertEqual(m["mode"], "random")
        self.assertEqual(m["sample_seed"], 424242)
        self.assertEqual(m["sampling_algorithm"],
                         "sha256_task_seed_python_shuffle_v1")
        e = m["sources"]["ScienceQA"]
        self.assertEqual(e["task_seed"],
                         hashlib.sha256(b"424242:ScienceQA").hexdigest())
        # floor(120*0.01)=1：选中 1 条（具体哪条由 seed 决定）
        self.assertEqual(e["k"], 1)
        self.assertEqual(len(e["selected_indices"]), 1)
        data = json.load(open(os.path.join(replay, "round2_train.json")))
        self.assertEqual(len(data), 1)

    def test_random_r010_four_round_dryrun(self):
        """ratio=0.10（tag r010）四任务四轮 DRY_RUN 全链路（新比例主路径）。"""
        self._four_task_data()
        rid = "run_0000_seed_424242"
        ckpt, res, replay = self._run_dirs(rid, tag="r010")
        r = self._run("0.1", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="424242",
                      RANDOM_REPLAY_RUN_ID=rid, REPLAY_ACCUM="1",
                      TASKS_JSON='["ScienceQA","TextVQA","ImageNet","GQA"]',
                      CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertEqual(r.returncode, 0, f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
        self.assertIn("random-replay gate PASS", r.stdout)
        self.assertIn("tag=r010", r.stdout)
        self.assertTrue(os.path.isfile(os.path.join(res, ".complete")))
        man = json.load(open(os.path.join(res, "run_manifest.json")))
        self.assertAlmostEqual(man["config"]["ratio"], 0.1)
        # 四轮 replay 数据 + sidecar（k = floor(120*0.1)=12/任务）
        for j in (2, 3, 4):
            rj = os.path.join(replay, f"round{j}_train.json")
            self.assertTrue(os.path.isfile(rj), rj)
            m = json.load(open(rj + ".manifest.json"))
            self.assertEqual(m["mode"], "random")
            self.assertEqual(len(m["sources"]), j - 1)
            for task, e in m["sources"].items():
                self.assertEqual(e["k"], 12, f"round{j}/{task}")
            self.assertEqual(m["output"]["N"], 12 * (j - 1))
        # 四轮 checkpoint 链
        for j in (1, 2, 3, 4):
            self.assertTrue(os.path.isdir(
                os.path.join(ckpt, f"round{j}_task_llava_lora")))
            if j > 1:
                self.assertTrue(os.path.isdir(
                    os.path.join(ckpt, f"round{j}_replay_llava_lora")))

    def test_random_gate_bad_seed_fails_fast(self):
        """启动门：seed 非法 → preflight/训练之前即失败（无 manifest、无 preflight 报告）。"""
        rid = "run_0000_seed_424242"
        ckpt, res, replay = self._run_dirs(rid)
        r = self._run("0.01", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="0",
                      RANDOM_REPLAY_RUN_ID=rid,
                      REPLAY_ACCUM="1",
                      CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertNotEqual(r.returncode, 0)
        out = r.stdout + r.stderr
        self.assertIn("FAIL(random-replay gate)", out)
        self.assertIn("REPLAY_SAMPLE_SEED", out)
        self.assertNotIn("运行数据 preflight", out, "门失败后不得进入 preflight")
        self.assertFalse(os.path.exists(os.path.join(res, "run_manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "preflight.json")))

    def test_random_gate_dir_mismatch_fails_fast(self):
        """启动门：目录属于别的 run → 失败。"""
        rid = "run_0000_seed_424242"
        other = "run_0000_seed_999999"
        ckpt, res, replay = self._run_dirs(other)  # 目录属于 other run
        r = self._run("0.01", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="424242",
                      RANDOM_REPLAY_RUN_ID=rid,
                      REPLAY_ACCUM="1",
                      CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertNotEqual(r.returncode, 0)
        out = r.stdout + r.stderr
        self.assertIn("FAIL(random-replay gate)", out)
        self.assertIn("目录必须属于", out)

    def test_random_gate_ratio_tag_mismatch_fails_fast(self):
        """启动门：ratio=0.10 却给了 r001 目录（tag 不一致）→ preflight 之前失败。"""
        self._four_task_data()
        rid = "run_0000_seed_424242"
        ckpt, res, replay = self._run_dirs(rid, tag="r001")  # 错误 tag
        r = self._run("0.1", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="424242",
                      RANDOM_REPLAY_RUN_ID=rid, REPLAY_ACCUM="1",
                      CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertNotEqual(r.returncode, 0)
        out = r.stdout + r.stderr
        self.assertIn("FAIL(random-replay gate)", out)
        self.assertIn("r010", out)
        self.assertIn("目录必须属于", out)
        self.assertNotIn("运行数据 preflight", out)
        self.assertFalse(os.path.exists(os.path.join(res, "run_manifest.json")))

    def test_random_gate_disallowed_ratio_fails_fast(self):
        """启动门：ratio 不在允许集合（0.01/0.10）→ 数值归一化判定失败。"""
        self._big_data()
        rid = "run_0000_seed_424242"
        ckpt, res, replay = self._run_dirs(rid)
        r = self._run("0.05", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="424242",
                      RANDOM_REPLAY_RUN_ID=rid, REPLAY_ACCUM="1",
                      CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertNotEqual(r.returncode, 0)
        out = r.stdout + r.stderr
        self.assertIn("RATIO 非法", out)
        self.assertIn("只允许 0.01 与 0.10", out)
        self.assertNotIn("运行数据 preflight", out)

    def test_resume_replay_sample_seed_mismatch_fails(self):
        self._big_data()
        rid1 = "run_0000_seed_111"
        ckpt, res, replay = self._run_dirs(rid1)
        r1 = self._run("0.01", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="111",
                       RANDOM_REPLAY_RUN_ID=rid1,
                       REPLAY_ACCUM="1",
                       CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        # 同一目录用不同 seed + 不同 run id 重跑 → resume config hash 校验拒绝
        rid2 = "run_0000_seed_222"
        r2 = self._run("0.01", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="222",
                       RANDOM_REPLAY_RUN_ID=rid2,
                       REPLAY_ACCUM="1",
                       CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertNotEqual(r2.returncode, 0)
        out = r2.stdout + r2.stderr
        self.assertIn("config hash", out)
        self.assertIn("replay_sample_seed", out)  # 语义差异字段显式指出

    def test_resume_r010_ratio_mismatch_rejected(self):
        """r010 run 的正式目录被 0.01 配置重跑 → 拒绝（manifest config hash 或 gate 拦下）。"""
        self._four_task_data()
        rid = "run_0000_seed_333"
        ckpt, res, replay = self._run_dirs(rid, tag="r010")
        r1 = self._run("0.1", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="333",
                       RANDOM_REPLAY_RUN_ID=rid, REPLAY_ACCUM="1",
                       TASKS_JSON='["ScienceQA","TextVQA","ImageNet","GQA"]',
                       CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        r2 = self._run("0.01", SAMPLE_MODE="random", REPLAY_SAMPLE_SEED="333",
                       RANDOM_REPLAY_RUN_ID=rid, REPLAY_ACCUM="1",
                       CKPT_ROOT=ckpt, RES_ROOT=res, REPLAY_DATA_DIR=replay)
        self.assertNotEqual(r2.returncode, 0)
        out = r2.stdout + r2.stderr
        self.assertIn("config hash", out)
        self.assertIn("ratio", out)   # 语义差异字段里指出 ratio（0.1 vs 0.01）
        self.assertIn("禁止覆盖", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
