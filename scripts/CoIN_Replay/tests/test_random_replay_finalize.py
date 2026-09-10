"""random_replay_finalize.py assemble 测试（零 GPU --test-mode；ratio 通用）。

fixture = 完整 fake run：4 任务合成 train/问题文件、真实 build_replay_data 生成的
random sidecar（round2..4）、10 个 eval 单元产物、真实 aggregate_coin.py 生成的
coin_metrics.json、7 个假 checkpoint 目录、run_manifest.json。

覆盖：
  - 成功路径（ratio=0.01/r001 与 ratio=0.10/r010）：全部 PASS → staging 原子换入 out，
    六文件齐全，staging 无残留
  - expected ratio 非法 / 与 manifest 不符 → 非零，既有 out 分毫不动
  - 错误 seed / 错误路径 / 篡改源数据 SHA / 已有 out（无 --force）/ 敏感泄漏 → 非零，
    既有 out 分毫不动
  - fill-delta：prefix 基线差值 + 同 seed 跨 ratio 配对差值（方向 = 高 ratio − 低 ratio）
"""
import hashlib
import json
import os
import random as _rnd
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.dirname(__file__))
from helpers import ROOT, build_synthetic

SEED = 20260908
RID = "run_0001_seed_20260908"
TASKS = ["ScienceQA", "TextVQA", "ImageNet", "GQA"]
N = 140  # floor(140*0.01)=1 >= 1；floor(140*0.1)=14

TOOLS_DIR = os.path.join(ROOT, "scripts", "CoIN_Replay", "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, os.path.join(ROOT, "scripts", "CoIN_Replay"))
import random_replay_finalize as F  # noqa: E402
from build_replay_data import SAMPLING_ALGORITHM  # noqa: E402
from coin_lib import ratio_tag  # noqa: E402

AGG = os.path.join(ROOT, "scripts", "CoIN_Replay", "aggregate_coin.py")
BUILD = os.path.join(ROOT, "scripts", "CoIN_Replay", "build_replay_data.py")
PUB6 = ["coin_metrics.json", "acc_sources.json", "run_manifest.sanitized.json",
        "replay_selection_summary.json", "validation_report.md", "summary.json"]


class FinalizeFixture:
    """构建一个完整 fake random run 目录树（ratio 参数化）。"""

    def __init__(self, tmp, seed=SEED, rid=RID, leak=None, ratio="0.01"):
        run = os.path.join(tmp, "run")
        self.run = run
        self.rid = rid
        self.seed = seed
        self.ratio = ratio
        self.tag = ratio_tag(ratio)
        self.data = os.path.join(run, "data")
        self.img = os.path.join(run, "img")
        self.ckpt = os.path.join(run, "checkpoints", "CoIN_Replay_random",
                                 self.tag, rid)
        self.res = os.path.join(run, "results", "CoIN_Replay_random",
                                self.tag, rid)
        self.replay = os.path.join(run, "playground", "Replay_random",
                                   self.tag, rid)
        for d in (self.data, self.img, self.ckpt, self.res, self.replay):
            os.makedirs(d, exist_ok=True)
        # 1) 合成数据：无图样本（build 不校验 image 字段）；问题文件与 merge 对应
        for t in TASKS:
            os.makedirs(os.path.join(self.data, t), exist_ok=True)
            train = [{"id": f"{t}_{i}",
                      "conversations": [{"from": "human", "value": "q?"},
                                        {"from": "gpt", "value": "a"}]}
                     for i in range(N)]
            json.dump(train, open(os.path.join(self.data, t, "train.json"), "w"))
            qname = "val.json" if t == "TextVQA" else "test.json"
            qs = [{"question_id": f"{t}_q{i}"} for i in range(N)]
            json.dump(qs, open(os.path.join(self.data, t, qname), "w"))
        # 2) replay sidecar：真实 build_replay_data random 模式（round 2/3/4）
        for j in (2, 3, 4):
            out = os.path.join(self.replay, f"round{j}_train.json")
            r = subprocess.run(
                [sys.executable, BUILD, "--tasks", *TASKS,
                 "--data-dir", self.data, "--image-dir", self.img,
                 "--round", str(j), "--ratio", str(ratio),
                 "--sample-mode", "random", "--seed", str(seed),
                 "--out", out], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        # 3) 假 checkpoint（7 个；task vs replay 字节不同）
        for j in (1, 2, 3, 4):
            for kind in (["task"] if j == 1 else ["task", "replay"]):
                d = os.path.join(self.ckpt, f"round{j}_{kind}_llava_lora")
                os.makedirs(d, exist_ok=True)
                json.dump({}, open(os.path.join(d, "config.json"), "w"))
                json.dump({}, open(os.path.join(d, "adapter_config.json"), "w"))
                with open(os.path.join(d, "adapter_model.bin"), "wb") as f:
                    f.write(_rnd.randbytes(1024))
                with open(os.path.join(d, "non_lora_trainables.bin"), "wb") as f:
                    f.write(_rnd.randbytes(256))
        # 4) eval 产物（10 单元三角）+ round manifests + .complete
        for j in range(1, 5):
            for i in range(1, j + 1):
                task = TASKS[i - 1]
                stage = os.path.join(self.res, task, f"round{j}")
                os.makedirs(stage, exist_ok=True)
                acc = 40.0 + 10 * i + j  # 任意合法值
                qs = json.load(open(os.path.join(
                    self.data, task, "val.json" if task == "TextVQA" else "test.json")))
                with open(os.path.join(stage, "merge.jsonl"), "w") as f:
                    for q in qs:
                        f.write(json.dumps({"question_id": q["question_id"]}) + "\n")
                if task == "ScienceQA":
                    json.dump({"acc": acc, "correct": int(N * acc / 100),
                               "count": N},
                              open(os.path.join(stage, "output_result.jsonl"), "w"))
                else:
                    open(os.path.join(stage, "Result.text"), "w").write(
                        f"Accuracy: {acc:.2f}%\n")
            json.dump({"round": j}, open(
                os.path.join(self.res, f"round{j}_manifest.json"), "w"))
        open(os.path.join(self.res, ".complete"), "w").write("ok\n")
        # 5) run_manifest.json（无绝对路径；env 版本号格式干净）
        man = {
            "run_id": f"coin_replay_r0.01_{seed}",
            "config": {
                "ratio": float(ratio), "tasks": TASKS, "T": 4,
                "model_base": "checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5",
                "vision_tower": "checkpoints/LLaVA/clip-vit-large-patch14-336",
                "projector": "checkpoints/LLaVA/mm_projector.bin",
                "ds_config": "scripts/zero3_offload.json",
                "sample_mode": "random", "replay_sample_seed": seed,
                "random_replay_run_id": rid, "replay_accum": 1,
                "seed": 1234, "data_seed": 1234,
            },
            "config_hash": "fixture",
            "git": {"commit": "a" * 40, "dirty_diff_hash": "clean"},
            "env": {"torch": "2.0.1+cu118", "python": "3.10.12",
                    "python_full": "3.10.12 (main, Jan 1 2026)",
                    **({"leak": leak} if leak else {})},
            "data_revision": "d" * 64,
            "model_config_hash": "m" * 64,
            "ds_config_hash": "s" * 64,
        }
        json.dump(man, open(os.path.join(self.res, "run_manifest.json"), "w"))
        # 6) coin_metrics.json 用真实 aggregate 生成 → 交叉验证必过
        r = subprocess.run([sys.executable, AGG, "--results-dir", self.res,
                            "--tasks", *TASKS], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    def assemble_args(self, out, repo_root=None, seed=None, force=False,
                      test_mode=True, ckpt=None, expected_ratio=None):
        return Namespace(res_root=self.res,
                         ckpt_root=ckpt or self.ckpt,
                         replay_data_dir=self.replay,
                         data_dir=self.data,
                         run_id=self.rid,
                         expected_seed=self.seed if seed is None else seed,
                         expected_ratio=self.ratio if expected_ratio is None else expected_ratio,
                         repo_root=repo_root or self.run,
                         out=out, force=force, test_mode=test_mode)


class TestRandomReplayFinalize(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_fin_")
        self.fx = FinalizeFixture(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _out(self):
        return os.path.join(self.tmp, "export")

    def test_assemble_success_staging_swap(self):
        out = self._out()
        rc = F.assemble(self.fx.assemble_args(out))
        self.assertEqual(rc, 0)
        for f in PUB6:
            self.assertTrue(os.path.isfile(os.path.join(out, f)), f"缺 {f}")
        summary = json.load(open(os.path.join(out, "summary.json")))
        cm = json.load(open(os.path.join(out, "coin_metrics.json")))
        self.assertEqual(summary["MAA"], cm["MAA"])
        self.assertEqual(summary["run_id"], RID)
        self.assertEqual(summary["ratio_tag"], "r001")
        self.assertEqual(summary["result_directory_rel"],
                         f"results/CoIN_Replay_random/r001/{RID}")
        self.assertEqual(summary["sampling_algorithm"], SAMPLING_ALGORITHM)
        self.assertIsNone(summary["deltas"])
        self.assertTrue(summary["test_mode"])
        for f in PUB6:
            if f == "summary.json":
                continue  # repro 覆盖其余五文件（自指循环避免）
            self.assertEqual(summary["repro"][f],
                             hashlib.sha256(open(os.path.join(out, f), "rb")
                                            .read()).hexdigest())
        report = open(os.path.join(out, "validation_report.md")).read()
        self.assertIn("全部验收项 PASS", report)
        self.assertIn("| .complete 存在 | PASS |", report)
        # staging 无残留；无 .stale 残留；res/ckpt 未被改动
        leftovers = [x for x in os.listdir(self.tmp) if ".staging" in x or ".stale" in x]
        self.assertEqual(leftovers, [])
        self.assertTrue(os.path.isfile(os.path.join(self.fx.res, "run_manifest.json")))
        # replay_selection_summary 与 sidecar 一致（output sha 抽样）
        rs = json.load(open(os.path.join(out, "replay_selection_summary.json")))
        self.assertEqual(rs["rounds"]["2"]["tasks"]["ScienceQA"]["k"], 1)

    def test_assemble_wrong_expected_seed_preserves_existing_out(self):
        out = self._out()
        os.makedirs(out)
        marker = os.path.join(out, "KEEP.txt")
        open(marker, "w").write("keep")
        rc = F.assemble(self.fx.assemble_args(out, seed=SEED + 1))
        self.assertNotEqual(rc, 0)
        self.assertEqual(open(marker).read(), "keep", "验收失败不得碰既有 out")
        self.assertEqual(sorted(os.listdir(out)), ["KEEP.txt"])

    def test_assemble_wrong_ckpt_path_fails(self):
        wrong_ckpt = os.path.join(self.tmp, "run", "checkpoints",
                                  "CoIN_Replay_random", "r001", "run_9999_seed_x")
        rc = F.assemble(self.fx.assemble_args(self._out(), ckpt=wrong_ckpt))
        self.assertNotEqual(rc, 0)

    def test_assemble_tampered_source_fails(self):
        # 篡改源 train.json → rebuild 门源 SHA 不一致
        p = os.path.join(self.fx.data, "ScienceQA", "train.json")
        d = json.load(open(p))
        d.append({"id": "ScienceQA_tamper",
                  "conversations": [{"from": "human", "value": "q?"},
                                    {"from": "gpt", "value": "a"}]})
        json.dump(d, open(p, "w"))
        rc = F.assemble(self.fx.assemble_args(self._out()))
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.isdir(self._out()))

    def test_assemble_tampered_sampling_algorithm_fails(self):
        mp = os.path.join(self.fx.replay, "round2_train.json.manifest.json")
        m = json.load(open(mp))
        m["sampling_algorithm"] = "evil_algo_v2"
        json.dump(m, open(mp, "w"))
        rc = F.assemble(self.fx.assemble_args(self._out()))
        self.assertNotEqual(rc, 0)

    def test_assemble_existing_out_no_force_fails(self):
        out = self._out()
        rc1 = F.assemble(self.fx.assemble_args(out))
        self.assertEqual(rc1, 0)
        before = hashlib.sha256(open(os.path.join(out, "summary.json"), "rb")
                                .read()).hexdigest()
        rc2 = F.assemble(self.fx.assemble_args(out))  # 无 --force
        self.assertNotEqual(rc2, 0)
        after = hashlib.sha256(open(os.path.join(out, "summary.json"), "rb")
                               .read()).hexdigest()
        self.assertEqual(before, after, "无 --force 不得替换既有发布")

    def test_assemble_force_replaces_existing_out(self):
        out = self._out()
        self.assertEqual(F.assemble(self.fx.assemble_args(out)), 0)
        open(os.path.join(out, "stale_junk.txt"), "w").write("x")
        rc = F.assemble(self.fx.assemble_args(out, force=True))
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.isfile(os.path.join(out, "stale_junk.txt")))
        self.assertEqual(sorted(os.listdir(out)), sorted(PUB6))

    def test_sensitive_scan_blocks_leak(self):
        fx = FinalizeFixture(os.path.join(self.tmp, "leaky"), leak="/root/data/secret")
        rc = F.assemble(fx.assemble_args(self._out(), repo_root=fx.run))
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.isdir(self._out()),
                         "敏感扫描失败不得产出发布目录")
        # staging 已清理
        self.assertEqual([x for x in os.listdir(self.tmp) if ".staging" in x], [])

    # ---- ratio=0.10 / r010 通用化路径 -----------------------------------------
    def test_assemble_r010_success(self):
        fx = FinalizeFixture(os.path.join(self.tmp, "r010"), ratio="0.10")
        out = self._out()
        rc = F.assemble(fx.assemble_args(out))
        self.assertEqual(rc, 0)
        summary = json.load(open(os.path.join(out, "summary.json")))
        self.assertEqual(summary["ratio_tag"], "r010")
        self.assertEqual(float(summary["ratio"]), 0.1)
        self.assertEqual(summary["result_directory_rel"],
                         f"results/CoIN_Replay_random/r010/{RID}")
        self.assertEqual(summary["ckpt_root_rel"],
                         f"checkpoints/CoIN_Replay_random/r010/{RID}")
        self.assertEqual(summary["replay_data_dir_rel"],
                         f"playground/Replay_random/r010/{RID}")
        # r010 sidecar 抽样条数 = floor(140*0.1)=14
        rs = json.load(open(os.path.join(out, "replay_selection_summary.json")))
        self.assertEqual(rs["rounds"]["2"]["tasks"]["ScienceQA"]["k"], 14)

    def test_assemble_expected_ratio_mismatch_preserves_existing_out(self):
        # fixture 是 0.10，却声称 expected ratio=0.01 → 必须 FAIL（数值比较）；out 不动
        fx = FinalizeFixture(os.path.join(self.tmp, "r010b"), ratio="0.10")
        out = self._out()
        os.makedirs(out)
        marker = os.path.join(out, "KEEP.txt")
        open(marker, "w").write("keep")
        rc = F.assemble(fx.assemble_args(out, expected_ratio="0.01"))
        self.assertNotEqual(rc, 0)
        self.assertEqual(open(marker).read(), "keep")
        self.assertEqual(sorted(os.listdir(out)), ["KEEP.txt"])

    def test_assemble_invalid_expected_ratio_fails(self):
        for bad in ("0.02", "0.5", "abc", ""):
            rc = F.assemble(self.fx.assemble_args(self._out(), expected_ratio=bad))
            self.assertNotEqual(rc, 0, f"expected_ratio={bad!r} 必须失败")
            self.assertFalse(os.path.isdir(self._out()),
                             "非法 expected ratio 不得产出发布目录")

    def test_assemble_run_id_seed_mismatch_fails(self):
        # run_id 内嵌 seed 与 --expected-seed 不一致 → 门 0 失败
        rc = F.assemble(self.fx.assemble_args(self._out(), seed=SEED + 1))
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.isdir(self._out()))

    # ---- fill-delta：prefix 基线 + 同 seed 跨 ratio 配对 -----------------------
    def _fill_args(self, summary, index, paired=None, result_commit=None,
                   paired_result_commit=None):
        return Namespace(summary=summary, index=index, paired_summary=paired,
                         result_commit=result_commit,
                         paired_result_commit=paired_result_commit)

    def test_fill_delta(self):
        out = self._out()
        self.assertEqual(F.assemble(self.fx.assemble_args(out)), 0)
        summary = os.path.join(out, "summary.json")
        idx = os.path.join(ROOT, "docs", "experiments", "coin_replay_random_r001",
                           "index.json")
        rc = F.fill_delta(self._fill_args(summary, idx))
        self.assertEqual(rc, 0)
        s = json.load(open(summary))
        self.assertIn("vs_prefix_0.10", s["deltas"])
        self.assertIn("vs_prefix_0.01", s["deltas"])
        # delta = 本 run MAA - 基线 MAA（回读核对）
        base = json.load(open(idx))["baselines"]["prefix_0.01"]
        self.assertAlmostEqual(s["deltas"]["vs_prefix_0.01"]["MAA"],
                               round(s["MAA"] - base["MAA"], 4))

    def test_fill_delta_paired_same_seed(self):
        """同 seed 的 0.01 与 0.10 两个 run：配对差值方向 = 0.10 − 0.01。"""
        fx01 = FinalizeFixture(os.path.join(self.tmp, "pair01"), ratio="0.01")
        fx10 = FinalizeFixture(os.path.join(self.tmp, "pair10"), ratio="0.10")
        out01, out10 = self._out() + "_01", self._out() + "_10"
        self.assertEqual(F.assemble(fx01.assemble_args(out01)), 0)
        self.assertEqual(F.assemble(fx10.assemble_args(out10)), 0)
        idx = os.path.join(ROOT, "docs", "experiments", "coin_replay_random_r001",
                           "index.json")
        rc = F.fill_delta(self._fill_args(
            os.path.join(out10, "summary.json"), idx,
            paired=os.path.join(out01, "summary.json"),
            result_commit="c" * 40, paired_result_commit="b" * 40))
        self.assertEqual(rc, 0)
        s10 = json.load(open(os.path.join(out10, "summary.json")))
        s01 = json.load(open(os.path.join(out01, "summary.json")))
        d = s10["deltas"]["vs_paired_r001"]
        self.assertEqual(d["delta_definition"], "random-0.10 − random-0.01")
        self.assertAlmostEqual(d["MAA"], round(s10["MAA"] - s01["MAA"], 4))
        self.assertAlmostEqual(d["BWT"], round(s10["BWT"] - s01["BWT"], 4))
        self.assertAlmostEqual(d["final_avg"], round(s10["final_avg"] - s01["final_avg"], 4))
        p = s10["paired"]
        self.assertEqual(p["this_run_id"], fx10.rid)
        self.assertEqual(p["paired_run_id"], fx01.rid)
        self.assertEqual(p["seed"], SEED)
        self.assertEqual(p["this_result_commit"], "c" * 40)
        self.assertEqual(p["paired_result_commit"], "b" * 40)
        self.assertEqual(p["this_code_commit"], s10["code_commit"])
        self.assertEqual(p["paired_code_commit"], s01["code_commit"])

    def test_fill_delta_paired_seed_mismatch_rejected(self):
        fx01 = FinalizeFixture(os.path.join(self.tmp, "bad01"), ratio="0.01",
                               seed=SEED)
        fx10 = FinalizeFixture(os.path.join(self.tmp, "bad10"), ratio="0.10",
                               seed=SEED + 7, rid="run_0002_seed_%d" % (SEED + 7))
        out01, out10 = self._out() + "_b01", self._out() + "_b10"
        self.assertEqual(F.assemble(fx01.assemble_args(out01)), 0)
        self.assertEqual(F.assemble(fx10.assemble_args(out10)), 0)
        idx = os.path.join(ROOT, "docs", "experiments", "coin_replay_random_r001",
                           "index.json")
        with self.assertRaises(SystemExit):
            F.fill_delta(self._fill_args(os.path.join(out10, "summary.json"), idx,
                                         paired=os.path.join(out01, "summary.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
