"""ckpt-tensor-diff 回归测试（canary C 评审 2026-09-02）。

覆盖评审要求：
- task==replay tensors 时必须失败，即使 metadata/非 tensor 文件不同（假阳性回归）；
- 差异 tensor 时通过 + 报告 changed_count/L2/max_abs_diff/tensor hash；
- missing key / shape 不一致 / NaN 均不得通过；
- CLI 退出码：pass→0，结论性失败→1，结构性问题→2。
零 GPU：需要 torch，缺失时整体 skip（本地无 torch；云端 run_tests 执行）。
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    _TORCH_OK = True
except ImportError:
    torch = None
    _TORCH_OK = False

from coin_lib import ckpt_tensor_compare, tensor_bytes_sha256  # noqa: E402

COIN_LIB = os.path.join(os.path.dirname(__file__), "..", "coin_lib.py")


def _write_ckpt(root, tensors, adapter_config_extra=None):
    """写一个最小真实 ckpt 目录：adapter 权重 + 常规伴随文件（内容可不同）。"""
    os.makedirs(root, exist_ok=True)
    torch.save(tensors, os.path.join(root, "adapter_model.bin"))
    cfg = {"r": 192, "alpha": 256}
    if adapter_config_extra is not None:
        cfg.update(adapter_config_extra)
    with open(os.path.join(root, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump({"architectures": ["LlavaLlamaForCausalLM"]}, f)
    with open(os.path.join(root, "non_lora_trainables.bin"), "wb") as f:
        f.write(b"\x00\x01\x02")
    with open(os.path.join(root, "trainer_state.json"), "w") as f:
        json.dump({"global_step": 0}, f)


@unittest.skipUnless(_TORCH_OK, "torch 不可用（本机无 torch），云端执行")
class CkptTensorDiffTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ckptdiff_")
        self.task_dir = os.path.join(self._tmp, "round2_task")
        self.replay_dir = os.path.join(self._tmp, "round2_replay")
        self.tensors = {
            "base_model.model.model.layers.0.q_proj.lora_A.weight":
                torch.randn(8, 4096, dtype=torch.float32) * 0.1,
            "base_model.model.model.layers.0.q_proj.lora_B.weight":
                torch.randn(32, 8, dtype=torch.float32) * 0.1,
            "base_model.model.model.layers.1.v_proj.lora_A.weight":
                torch.randn(8, 4096, dtype=torch.float32) * 0.1,
        }

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_identical_tensors_different_metadata_fails(self):
        """假阳性回归：tensor 完全相同 + metadata（adapter_config/trainer_state）不同 → 必须失败。"""
        _write_ckpt(self.task_dir, self.tensors, adapter_config_extra={"a": 1})
        # replay 与 task tensor 逐字节相同，仅 metadata 不同
        same = {k: v.clone() for k, v in self.tensors.items()}
        _write_ckpt(self.replay_dir, same, adapter_config_extra={"a": 2})
        rep = ckpt_tensor_compare(self.task_dir, self.replay_dir)
        self.assertFalse(rep["pass"], "tensor 相同（metadata 不同）不得通过")
        self.assertEqual(rep["changed_tensor_count"], 0)
        self.assertFalse(rep["tensor_hash"]["differs"])

    def test_cli_identical_tensors_exit_1(self):
        """CLI 层：tensor 相同 → exit 1（结论性失败）。"""
        _write_ckpt(self.task_dir, self.tensors)
        _write_ckpt(self.replay_dir, {k: v.clone() for k, v in self.tensors.items()})
        r = subprocess_run([sys.executable, COIN_LIB, "ckpt-tensor-diff",
                            self.task_dir, self.replay_dir])
        self.assertEqual(r.returncode, 1)
        out = json.loads(r.stdout)
        self.assertFalse(out["pass"])

    def test_different_tensors_passes(self):
        """replay 权重相对 task 有真实变化 → 通过 + 报告差值统计。"""
        _write_ckpt(self.task_dir, self.tensors)
        replay = {k: v.clone() for k, v in self.tensors.items()}
        replay["base_model.model.model.layers.0.q_proj.lora_A.weight"] += 0.05
        _write_ckpt(self.replay_dir, replay)
        rep = ckpt_tensor_compare(self.task_dir, self.replay_dir)
        self.assertTrue(rep["pass"])
        self.assertGreaterEqual(rep["changed_tensor_count"], 1)
        self.assertGreater(rep["l2_norm_diff"], 0)
        self.assertGreater(rep["max_abs_diff"], 0)
        self.assertTrue(rep["tensor_hash"]["differs"])
        self.assertEqual(rep["keys"]["missing"], [])
        self.assertEqual(rep["keys"]["unexpected"], [])
        self.assertTrue(rep["finite"]["task"] and rep["finite"]["replay"])

    def test_cli_different_tensors_exit_0(self):
        _write_ckpt(self.task_dir, self.tensors)
        replay = {k: v.clone() for k, v in self.tensors.items()}
        replay["base_model.model.model.layers.1.v_proj.lora_A.weight"] += 1.0
        _write_ckpt(self.replay_dir, replay)
        r = subprocess_run([sys.executable, COIN_LIB, "ckpt-tensor-diff",
                            self.task_dir, self.replay_dir])
        self.assertEqual(r.returncode, 0)

    def test_missing_key_fails(self):
        _write_ckpt(self.task_dir, self.tensors)
        _write_ckpt(self.replay_dir,
                    {k: v for i, (k, v) in enumerate(self.tensors.items()) if i != 0})
        rep = ckpt_tensor_compare(self.task_dir, self.replay_dir)
        self.assertFalse(rep["pass"])
        self.assertEqual(len(rep["keys"]["missing"]), 1)

    def test_shape_mismatch_fails(self):
        _write_ckpt(self.task_dir, self.tensors)
        bad = dict(self.tensors)
        bad["base_model.model.model.layers.0.q_proj.lora_A.weight"] = torch.randn(4, 4096)
        _write_ckpt(self.replay_dir, bad)
        rep = ckpt_tensor_compare(self.task_dir, self.replay_dir)
        self.assertFalse(rep["pass"])
        self.assertEqual(len(rep["keys"]["shape_mismatch"]), 1)

    def test_nan_fails(self):
        _write_ckpt(self.task_dir, self.tensors)
        nan = dict(self.tensors)
        nan["base_model.model.model.layers.0.q_proj.lora_A.weight"] = \
            torch.full((8, 4096), float("nan"))
        _write_ckpt(self.replay_dir, nan)
        rep = ckpt_tensor_compare(self.task_dir, self.replay_dir)
        self.assertFalse(rep["pass"])
        self.assertFalse(rep["finite"]["replay"])

    def test_structural_missing_dir_exit_2(self):
        r = subprocess_run([sys.executable, COIN_LIB, "ckpt-tensor-diff",
                            os.path.join(self._tmp, "nope"), self.task_dir])
        self.assertEqual(r.returncode, 2)

    def test_bf16_hash_stable(self):
        """bf16 adapter（原 numpy() 抛 TypeError 的 dtype）hash 必须可用且稳定。"""
        t = torch.randn(64, dtype=torch.bfloat16)
        h1 = tensor_bytes_sha256({"a": t})
        h2 = tensor_bytes_sha256({"a": t.clone()})
        self.assertEqual(h1, h2)
        t2 = t.clone()
        t2[0] += 1
        self.assertNotEqual(h1, tensor_bytes_sha256({"a": t2}))


def subprocess_run(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
