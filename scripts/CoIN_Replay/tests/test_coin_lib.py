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
            self.assertEqual(plan["optimizer_steps"], 2)
            self.assertEqual(plan["replay_k"], 120)
            self.assertEqual(plan["consumed_samples"], 1000)
            self.assertFalse(plan["flag_replay_single_step"])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
