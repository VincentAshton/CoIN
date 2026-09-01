"""工单 1/5 单测：评估目录契约 + dry-run + 故障注入（零 GPU，直接跑仓库里的 eval 脚本）。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, build_synthetic, make_aux, run

EVALS = {
    "ScienceQA": ("1_eval_sqa.sh", "ScienceQA/test.json", "output_result.jsonl"),
    "TextVQA": ("2_eval_textqa.sh", "TextVQA/val.json", "Result.text"),
    "ImageNet": ("3_eval_ImageNet.sh", "ImageNet/test.json", "Result.text"),
    "GQA": ("4_eval_gqa.sh", "GQA/test.json", "Result.text"),
}


class TestEvalContract(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_eval_")
        self.data_dir = os.path.join(self.tmp, "Instructions")
        self.img_dir = os.path.join(self.tmp, "images")
        os.makedirs(self.data_dir)
        os.makedirs(self.img_dir)
        for task in EVALS:
            build_synthetic(self.data_dir, self.img_dir, task, n=5)
            make_aux(self.img_dir, task)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_eval(self, task, stage="round2", fault=0):
        script, qf, artifact = EVALS[task]
        res_dir = os.path.join(self.tmp, "results", task)  # 每个任务独立结果目录（契约：RES_ROOT/<Task>/）
        env = {
            "EVAL_DRY_RUN": "1",
            "EVAL_FAULT_INJECT": str(fault),
            "RESULT_DIR": res_dir,
            "CUDA_VISIBLE_DEVICES": "0",
            "QUESTION_FILE": os.path.join(self.data_dir, qf),
        }
        r = run(["bash", os.path.join(ROOT, "scripts", "LLaVA", "Eval", script),
                 stage, "/tmp/fake_ckpt_llava_lora"],
                cwd=ROOT, env_extra=env)
        return r, os.path.join(res_dir, stage)

    def test_all_tasks_contract(self):
        for task in EVALS:
            r, stage_dir = self._run_eval(task)
            self.assertEqual(r.returncode, 0, f"{task}: {r.stderr}")
            self.assertTrue(os.path.isfile(os.path.join(stage_dir, "merge.jsonl")),
                            f"{task}: merge.jsonl 缺失")
            art = EVALS[task][2]
            self.assertTrue(os.path.isfile(os.path.join(stage_dir, art)),
                            f"{task}: {art} 缺失")
            # merge.jsonl 与 question 文件 ID 一致（顺序+集合，verify-predictions 同款）
            qf = os.path.join(self.data_dir, EVALS[task][1])
            qids = [q["question_id"] for q in json.load(open(qf))]
            pids = [json.loads(l)["question_id"]
                    for l in open(os.path.join(stage_dir, "merge.jsonl"))]
            self.assertEqual(qids, pids, f"{task}: 顺序/集合不一致")

    def test_fault_injection_fails(self):
        for task in EVALS:
            r, stage_dir = self._run_eval(task, fault=1)
            self.assertNotEqual(r.returncode, 0, f"{task}: 故障注入应当非零退出")
            # 不产出准确性产物
            art = EVALS[task][2]
            self.assertFalse(os.path.isfile(os.path.join(stage_dir, art)),
                             f"{task}: 故障时不应有 {art}")

    def test_verify_predictions_catches_missing_chunk(self):
        """模拟：chunk 缺失但 merge 仍被拼接（旧版 bug 场景）→ verify 必须失败。"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import coin_lib
        qf = os.path.join(self.data_dir, "ScienceQA", "test.json")
        # 手工构造一个缺后半段的 merge
        qs = json.load(open(qf))
        pf = os.path.join(self.tmp, "partial.jsonl")
        with open(pf, "w") as f:
            for q in qs[: len(qs) // 2]:
                f.write(json.dumps({"question_id": q["question_id"], "text": "x"}) + "\n")
        with self.assertRaises(ValueError):
            coin_lib.verify_predictions(qf, pf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
