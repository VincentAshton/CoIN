"""工单 1/4/6/8 单测：coin_lib（train_plan / manifest / ckpt / verify_predictions / artifact）。"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import coin_lib
from helpers import REPLAY_DIR, make_test_sample, py


class TestTrainPlan(unittest.TestCase):

    def test_steps_and_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d896.json")
            json.dump([{"id": i} for i in range(896)], open(p, "w"))
            plan = coin_lib.train_plan(
                data_json=p, batch=14, accum=16, world=4, lr=2e-4,
                warmup_ratio=0.03, epochs=1, name="t")
            self.assertEqual(plan["total_train_batch_size"], 896)
            self.assertEqual(plan["optimizer_steps"], 1)
            p1000 = os.path.join(tmp, "d1000.json")
            json.dump([{"id": i} for i in range(1000)], open(p1000, "w"))
            self.assertEqual(coin_lib.train_plan(
                p1000, 14, 16, 4, 2e-4, 0.03, 1, "t")["optimizer_steps"], 2)

    def test_plan_with_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            json.dump([{"id": i} for i in range(1000)], open(p, "w"))
            plan = coin_lib.train_plan(p, 14, 16, 4, 2e-4, 0.03, 1, "t", replay_k=120)
            self.assertEqual(plan["N"], 1000)
            self.assertEqual(plan["optimizer_steps"], 2)   # 旧 ceil 口径（兼容）
            self.assertEqual(plan["ds_expected_updates"], 1)  # DS 权威（18 micro //16）
            self.assertEqual(plan["replay_k"], 120)
            self.assertEqual(plan["consumed_samples"], 1000)
            # 2026-09-04 语义变更：flag 以 DS 真步判（≤1 拦截，评审 C-1 ≥2 real steps）
            self.assertTrue(plan["flag_replay_single_step"])
            # task 大段（真步≥2）不拦截
            pbig = os.path.join(tmp, "big.json")
            json.dump([{"id": i} for i in range(12726)], open(pbig, "w"))
            big = coin_lib.train_plan(pbig, 14, 16, 4, 2e-4, 0.03, 1, "t")
            self.assertFalse(big["flag_replay_single_step"])
            self.assertEqual(big["ds_expected_updates"], 14)

    def test_single_step_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            json.dump([{"id": i} for i in range(120)], open(p, "w"))
            plan = coin_lib.train_plan(p, 14, 16, 4, 2e-4, 0.03, 1, "replay")
            self.assertEqual(plan["optimizer_steps"], 1)
            self.assertTrue(plan["flag_replay_single_step"])

    def test_warmup_covers_all_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d3000.json")
            json.dump([{"id": i} for i in range(3000)], open(p, "w"))
            plan = coin_lib.train_plan(p, 14, 16, 4, 2e-4, 1.0, 1, "t")
            self.assertTrue(plan["flag_warmup_covers_all"])
            plan2 = coin_lib.train_plan(p, 14, 16, 4, 2e-4, 0.03, 1, "t")
            self.assertFalse(plan2["flag_warmup_covers_all"])


class TestConfigAndManifest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_manifest_")
        self.res = os.path.join(self.tmp, "res")
        os.makedirs(self.res)
        self.env = {
            "RATIO": "0.1", "BASE_MODEL": "/b", "VISION_TOWER": "/v",
            "PROJECTOR": "/p", "DS_CONFIG": "/ds.json", "GPUS": "0,1,2,3",
            "BATCH": "14", "ACCUM": "16", "LR": "2e-4", "MM_PROJECTOR_LR": "2e-5",
            "TASKS_JSON": '["ScienceQA","TextVQA","ImageNet","GQA"]',
            "SEED": "1234", "DATA_SEED": "1234", "LORA_R": "192", "LORA_ALPHA": "256",
            "EPOCHS": "1", "REPLAY_EPOCHS": "1", "SAMPLE_MODE": "prefix",
            "WARMUP_RATIO": "0.03", "LR_SCHEDULER_TYPE": "cosine",
            "MODEL_MAX_LENGTH": "2048", "EVAL_TEMPERATURE": "0",
            "PRECISION": "bf16+tf32", "GRAD_CKPT": "true",
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_compute_config_effective_batch(self):
        cfg = coin_lib.compute_config(self.env)
        self.assertEqual(cfg["world_size"], 4)
        self.assertEqual(cfg["effective_batch"], 14 * 16 * 4)

    def test_config_hash_stable(self):
        h1 = coin_lib.config_hash(coin_lib.compute_config(self.env))
        h2 = coin_lib.config_hash(coin_lib.compute_config(self.env))
        self.assertEqual(h1, h2)
        env2 = dict(self.env, LR="3e-4")
        self.assertNotEqual(h1, coin_lib.config_hash(coin_lib.compute_config(env2)))

    def test_manifest_no_overwrite_and_resume(self):
        cfg = coin_lib.compute_config(self.env)
        p1 = coin_lib.manifest_write(self.res, cfg, root=self.tmp)
        self.assertTrue(os.path.isfile(p1))
        # 恢复运行：同配置 OK
        coin_lib.manifest_write(self.res, cfg, root=self.tmp, resume_ok=True)
        # 覆盖被禁止
        with self.assertRaises(FileExistsError):
            coin_lib.manifest_write(self.res, cfg, root=self.tmp, force=False)
        # 配置不一致 → 恢复运行失败
        env2 = dict(self.env, LR="5e-4")
        with self.assertRaises(ValueError):
            coin_lib.manifest_write(self.res, coin_lib.compute_config(env2),
                                    root=self.tmp, resume_ok=True)

    def test_manifest_contains_expected_fields(self):
        cfg = coin_lib.compute_config(self.env)
        p1 = coin_lib.manifest_write(self.res, cfg, root=self.tmp)
        m = json.load(open(p1))
        self.assertEqual(m["config_hash"], coin_lib.config_hash(cfg))
        for k in ("git", "env", "data_revision", "model_config_hash", "ds_config_hash"):
            self.assertIn(k, m)


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_ckpt_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mk(self, name, with_adapter=True, empty_adapter=False):
        d = os.path.join(self.tmp, name)
        os.makedirs(d, exist_ok=True)
        json.dump({}, open(os.path.join(d, "config.json"), "w"))
        json.dump({}, open(os.path.join(d, "adapter_config.json"), "w"))
        with open(os.path.join(d, "non_lora_trainables.bin"), "wb") as f:
            f.write(b"\x00\x01")
        if with_adapter:
            with open(os.path.join(d, "adapter_model.bin"), "wb") as f:
                f.write(b"" if empty_adapter else b"\x00\x01\x02\x03")
        return d

    def test_ok(self):
        d = self._mk("ok")
        rep = coin_lib.ckpt_validate(d)
        self.assertIn("files", rep)
        self.assertIn("adapter_model.bin", rep["files"])

    def test_missing_adapter(self):
        d = self._mk("no_adapter", with_adapter=False)
        with self.assertRaises(FileNotFoundError):
            coin_lib.ckpt_validate(d)

    def test_empty_adapter(self):
        d = self._mk("empty_adapter", empty_adapter=True)
        # 空文件仍通过文件级校验（torch 不可用时无法解析权重；云端 will 解析并报错）
        coin_lib.ckpt_validate(d)

    def test_missing_config(self):
        d = os.path.join(self.tmp, "bad")
        os.makedirs(d)
        with open(os.path.join(d, "adapter_model.bin"), "wb") as f:
            f.write(b"x")
        with self.assertRaises(FileNotFoundError):
            coin_lib.ckpt_validate(d)


class TestVerifyPredictions(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_pred_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _files(self, qids, pids):
        qf = os.path.join(self.tmp, "q.json")
        pf = os.path.join(self.tmp, "p.jsonl")
        json.dump([make_test_sample(q, "i.png") for q in qids], open(qf, "w"))
        with open(pf, "w") as f:
            for p in pids:
                f.write(json.dumps({"question_id": p, "text": "a"}) + "\n")
        return qf, pf

    def test_ok(self):
        qf, pf = self._files(["a", "b", "c"], ["a", "b", "c"])
        rep = coin_lib.verify_predictions(qf, pf)
        self.assertEqual(rep["prediction_count"], 3)

    def test_count_mismatch(self):
        qf, pf = self._files(["a", "b", "c"], ["a", "b"])
        with self.assertRaises(ValueError):
            coin_lib.verify_predictions(qf, pf)

    def test_dup_id(self):
        qf, pf = self._files(["a", "b", "c"], ["a", "a", "b"])
        with self.assertRaises(ValueError):
            coin_lib.verify_predictions(qf, pf)

    def test_set_mismatch(self):
        qf, pf = self._files(["a", "b", "c"], ["a", "b", "d"])
        with self.assertRaises(ValueError):
            coin_lib.verify_predictions(qf, pf)

    def test_order_mismatch(self):
        qf, pf = self._files(["a", "b", "c"], ["b", "a", "c"])
        with self.assertRaises(ValueError):
            coin_lib.verify_predictions(qf, pf)
        # 关闭顺序检查则通过
        coin_lib.verify_predictions(qf, pf, order_check=False)


class TestArtifactCheck(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_art_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sqa_ok(self):
        d = os.path.join(self.tmp, "round1")
        os.makedirs(d)
        json.dump({"acc": 42.0}, open(os.path.join(d, "output_result.jsonl"), "w"))
        rep = coin_lib.artifact_check("ScienceQA", d)
        self.assertEqual(rep["acc"], 42.0)

    def test_text_ok(self):
        d = os.path.join(self.tmp, "round1")
        os.makedirs(d)
        open(os.path.join(d, "Result.text"), "w").write("Samples: 10\nAccuracy: 55.50%\n")
        rep = coin_lib.artifact_check("TextVQA", d)
        self.assertAlmostEqual(rep["acc"], 55.5)

    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            coin_lib.artifact_check("ScienceQA", self.tmp)

    def test_out_of_range(self):
        d = os.path.join(self.tmp, "round1")
        os.makedirs(d)
        open(os.path.join(d, "Result.text"), "w").write("Accuracy: 101.00%\n")
        with self.assertRaises(ValueError):
            coin_lib.artifact_check("GQA", d)


class TestPlanAudit20260904(unittest.TestCase):
    """2026-09-04 方案 D 审计：train_plan 多口径字段（任务书六 4/5/6）。"""

    def _plan(self, n, accum, batch=14, world=4):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "d.json")
            json.dump([{"id": i} for i in range(n)], open(p, "w"))
            return coin_lib.train_plan(p, batch, accum, world, 2e-4, 0.03, 1, "replay")

    def test_replay_gas1_expected_updates(self):
        # 任务书六.4：N=127/473 gas1 真步 ≈3/9（DS 权威口径 = M//gas，M 均匀）
        self.assertEqual(self._plan(127, 1)["ds_expected_updates"], 3)
        self.assertEqual(self._plan(473, 1)["ds_expected_updates"], 9)

    def test_replay_010_gas1_expected_updates(self):
        # 任务书六.5：0.1 计划 ≈23/85/317
        self.assertEqual(self._plan(1272, 1)["ds_expected_updates"], 23)
        self.assertEqual(self._plan(4732, 1)["ds_expected_updates"], 85)
        self.assertEqual(self._plan(17715, 1)["ds_expected_updates"], 317)

    def test_gas16_nogo_scenarios_zero_updates(self):
        # 阶段 II No-Go 复现：N=127/473 gas16 → 0 真步；flag 拦截（ds 口径）
        for n in (127, 473):
            p = self._plan(n, 16)
            self.assertEqual(p["ds_expected_updates"], 0)
            self.assertTrue(p["flag_replay_single_step"])  # 门禁以 DS 真步判（≤1 拦截）

    def test_hf_vs_ds_fields(self):
        # 任务书六.6：HF/DS/remainder/discarded/sampler_padding 字段正确
        p = self._plan(127, 16)   # M=3/rank
        self.assertEqual(p["per_rank_micro"], [3, 3, 3, 3])
        self.assertEqual(p["hf_planned_steps"], 1)          # HF phantom
        self.assertEqual(p["ds_expected_updates"], 0)       # DS 权威
        self.assertEqual(p["microbatch_remainder"], 3)
        self.assertEqual(p["discarded_or_uncommitted_microbatches"], 3)
        # sampler_padding: Σmicro×batch − N = 12×14−127=41? M=3 → 每 rank 3×14=42 样本 → 4×42=168 → pad=41
        self.assertEqual(p["sampler_padding"], 168 - 127)
        p2 = self._plan(1272, 16)  # M=23
        self.assertEqual(p2["per_rank_micro"], [23, 23, 23, 23])
        self.assertEqual(p2["ds_expected_updates"], 1)
        self.assertEqual(p2["microbatch_remainder"], 7)
        self.assertEqual(p2["discarded_or_uncommitted_microbatches"], 7)
        # effective_batch 语义
        self.assertEqual(p2["effective_batch"], 896)
        self.assertEqual(self._plan(127, 1)["effective_batch"], 56)

    def test_old_fields_compat(self):
        # 旧字段保留（门禁/上游调用兼容）
        p = self._plan(1000, 16)
        self.assertEqual(p["optimizer_steps"], 2)   # 旧 ceil 口径
        self.assertEqual(p["ds_expected_updates"], 1)  # 真实 62//16=3? M(1000)=18→18//16=1
        p2 = self._plan(896, 16)
        self.assertEqual(p2["optimizer_steps"], 1)
        self.assertEqual(p2["ds_expected_updates"], 1)


class TestReplayAccumConfig(unittest.TestCase):
    """2026-09-04 方案 D：REPLAY_ACCUM 配置与 manifest（任务书六 2/3/9/10/14）。"""

    BASE = {
        "RATIO": "0.1", "BASE_MODEL": "/b", "VISION_TOWER": "/v", "PROJECTOR": "/p",
        "DS_CONFIG": "/ds.json", "GPUS": "0,1,2,3", "BATCH": "14", "ACCUM": "16",
        "LR": "2e-4", "MM_PROJECTOR_LR": "2e-5",
        "TASKS_JSON": '["ScienceQA","TextVQA","ImageNet","GQA"]',
        "SEED": "1234", "DATA_SEED": "1234", "LORA_R": "192", "LORA_ALPHA": "256",
        "EPOCHS": "1", "REPLAY_EPOCHS": "1", "SAMPLE_MODE": "prefix",
        "WARMUP_RATIO": "0.03", "LR_SCHEDULER_TYPE": "cosine",
        "MODEL_MAX_LENGTH": "2048", "EVAL_TEMPERATURE": "0",
        "PRECISION": "bf16+tf32", "GRAD_CKPT": "true", "REPLAY_ACCUM": "1",
    }

    def test_task_accum16_replay_accum1(self):
        # 任务书六.2/3：manifest 记录 task_accum=16、replay_accum=1、两有效 batch
        cfg = coin_lib.compute_config(self.BASE)
        self.assertEqual(cfg["grad_accum"], 16)
        self.assertEqual(cfg["replay_accum"], 1)
        self.assertEqual(cfg["effective_batch"], 896)
        self.assertEqual(cfg["replay_effective_batch"], 56)

    def test_ratio_does_not_affect_replay_accum(self):
        # 任务书六.9：ratio 不得影响 replay_accum
        a = coin_lib.compute_config(dict(self.BASE, RATIO="0.1"))
        b = coin_lib.compute_config(dict(self.BASE, RATIO="0.01"))
        self.assertEqual(a["replay_accum"], b["replay_accum"])
        self.assertEqual(a["replay_effective_batch"], b["replay_effective_batch"])
        self.assertEqual(a["effective_batch"], b["effective_batch"])
        self.assertNotEqual(a["ratio"], b["ratio"])

    def test_unset_replay_accum_is_null(self):
        env = {k: v for k, v in self.BASE.items() if k != "REPLAY_ACCUM"}
        cfg = coin_lib.compute_config(env)
        self.assertIsNone(cfg["replay_accum"])
        self.assertIsNone(cfg["replay_effective_batch"])

    def test_resume_rejects_different_replay_accum(self):
        # 任务书六.10/14：旧 manifest（replay_accum 不同）不得被新配置恢复运行复用
        tmp = tempfile.mkdtemp(prefix="coin_ra_")
        try:
            res = os.path.join(tmp, "res")
            os.makedirs(res)
            env16 = dict(self.BASE, REPLAY_ACCUM="16")
            cfg16 = coin_lib.compute_config(env16)
            coin_lib.manifest_write(res, cfg16, root=tmp)
            # 换 REPLAY_ACCUM=1 恢复 → 拒绝
            cfg1 = coin_lib.compute_config(self.BASE)
            self.assertNotEqual(coin_lib.config_hash(cfg1), coin_lib.config_hash(cfg16))
            with self.assertRaises(ValueError):
                coin_lib.manifest_write(res, cfg1, root=tmp, resume_ok=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_allow_single_step_recorded(self):
        cfg = coin_lib.compute_config(self.BASE)
        self.assertEqual(cfg["allow_single_step_replay"], 0)
        cfg2 = coin_lib.compute_config(dict(self.BASE, ALLOW_SINGLE_STEP_REPLAY="1"))
        self.assertEqual(cfg2["allow_single_step_replay"], 1)


class TestCrossManifest(unittest.TestCase):
    """任务书六.11：cross-run manifest 校验（0.1 vs 0.01 除 ratio 外一致）。"""

    BASE = {
        "RATIO": "0.1", "BASE_MODEL": "/b", "VISION_TOWER": "/v", "PROJECTOR": "/p",
        "DS_CONFIG": "/ds.json", "GPUS": "0,1,2,3", "BATCH": "14", "ACCUM": "16",
        "REPLAY_ACCUM": "1", "LR": "2e-4", "MM_PROJECTOR_LR": "2e-5",
        "TASKS_JSON": '["ScienceQA","TextVQA","ImageNet","GQA"]',
        "SEED": "1234", "DATA_SEED": "1234", "LORA_R": "192", "LORA_ALPHA": "256",
        "EPOCHS": "1", "REPLAY_EPOCHS": "1", "SAMPLE_MODE": "prefix",
        "WARMUP_RATIO": "0.03", "LR_SCHEDULER_TYPE": "cosine",
        "MODEL_MAX_LENGTH": "2048", "EVAL_TEMPERATURE": "0",
        "PRECISION": "bf16+tf32", "GRAD_CKPT": "true",
    }

    def _mk(self, ratio, replay_accum="1", lr="2e-4"):
        tmp = tempfile.mkdtemp(prefix="coin_cross_")
        res = os.path.join(tmp, "res")
        os.makedirs(res)
        env = dict(self.BASE, RATIO=ratio, REPLAY_ACCUM=replay_accum, LR=lr)
        coin_lib.manifest_write(res, coin_lib.compute_config(env), root=tmp)
        return res

    def test_pass_same_replay_accum(self):
        a = self._mk("0.1")
        b = self._mk("0.01")
        rep = coin_lib.manifest_cross_check(a, b)
        self.assertTrue(rep["pass"], rep)
        self.assertEqual(rep["replay_accum_a"], rep["replay_accum_b"])

    def test_fail_different_replay_accum(self):
        a = self._mk("0.1", replay_accum="1")
        b = self._mk("0.01", replay_accum="16")
        rep = coin_lib.manifest_cross_check(a, b)
        self.assertFalse(rep["pass"])
        self.assertIn("replay_accum", rep["diffs"])
        self.assertIn("replay_effective_batch", rep["diffs"])

    def test_fail_other_config_diff(self):
        a = self._mk("0.1")
        b = self._mk("0.01", lr="3e-4")
        rep = coin_lib.manifest_cross_check(a, b)
        self.assertFalse(rep["pass"])
        self.assertIn("lr", rep["diffs"])


class TestShellGuards(unittest.TestCase):
    """任务书六.1/12/13：shell 层防隐式覆盖与输出隔离（静态断言，防回归）。"""

    ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def test_run_sweep_never_sets_replay_accum(self):
        # 任务书六.1/12：run_sweep 不得按 ratio 隐式设置 REPLAY_ACCUM
        text = open(os.path.join(self.ROOT, "scripts/CoIN_Replay/run_sweep.sh")).read()
        exec_lines = [l for l in text.splitlines()
                      if "REPLAY_ACCUM" in l and not l.strip().startswith("#")]
        self.assertEqual(exec_lines, [])

    def test_run_replay_exp_output_roots_isolatable(self):
        # 任务书六.13：正式输出与 Canary 输出隔离（CKPT_ROOT/RES_ROOT/REPLAY_DATA_DIR 可 env 覆盖）
        text = open(os.path.join(self.ROOT, "scripts/CoIN_Replay/run_replay_exp.sh")).read()
        for var in ("CKPT_ROOT", "RES_ROOT", "REPLAY_DATA_DIR"):
            self.assertIn(f'{var}="${{{var}:-', text)

    def test_train_one_accepts_accum_override(self):
        text = open(os.path.join(self.ROOT, "scripts/CoIN_Replay/run_replay_exp.sh")).read()
        self.assertIn('local accum="${7:-$ACCUM}"', text)
        self.assertIn('--gradient_accumulation_steps "$accum"', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
