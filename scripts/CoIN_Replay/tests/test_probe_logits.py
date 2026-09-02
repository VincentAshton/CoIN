"""probe_logits 修复单测（评审要求）：纯文本/有图 probe 构建、严格失败路径、固定 ID、hash 确定性。

零 GPU：probe_logits 模块顶层不 import torch/ETrain；仅 PIL 相关用例在无 PIL 时 skip。
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "smoke"))
import probe_logits as pl
from helpers import tiny_png


def _pil_ok():
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


TEXT_ONLY = {"question_id": "4", "text": "Which state is farthest north?\nA. West Virginia\nB. Louisiana",
             "answer": "A"}
WITH_IMAGE = {"question_id": "5", "text": "What does the image show?",
              "image": "ScienceQA/images/test/5/image.png", "answer": "X"}


def make_probe_json(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    return path


def make_image_dir(root, rel="ScienceQA/images/test/5/image.png"):
    img = os.path.join(root, rel)
    os.makedirs(os.path.dirname(img), exist_ok=True)
    with open(img, "wb") as f:
        f.write(tiny_png())
    return img


class TestBuildPrompt(unittest.TestCase):

    def test_text_only_no_image_token(self):
        qs = pl.build_prompt("What is X?\nA. a\nB. b", False)
        self.assertNotIn(pl.IMAGE_TOKEN, qs)
        self.assertEqual(qs, "What is X?\nA. a\nB. b")

    def test_image_prepends_token(self):
        qs = pl.build_prompt("What is X?", True)
        self.assertTrue(qs.startswith("<image>\n"))

    def test_strips_existing_image_token(self):
        self.assertEqual(pl.build_prompt("<image>\nWhat is X?", False), "What is X?")
        self.assertEqual(pl.build_prompt("<image>\nWhat is X?", True), "<image>\nWhat is X?")

    def test_im_start_end_variant(self):
        qs = pl.build_prompt("Q", True, use_im_start_end=True,
                             im_start_token="<s>", im_end_token="</s>")
        self.assertEqual(qs, "<s><image></s>\nQ")


class TestBuildProbeSet(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_probe_")
        self.img_dir = os.path.join(self.tmp, "cl_dataset")
        make_image_dir(self.img_dir)
        self.pj = make_probe_json(os.path.join(self.tmp, "test.json"),
                                  [TEXT_ONLY, WITH_IMAGE])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_text_only_entry(self):
        probes, sha = pl.build_probe_set(
            self.pj, self.img_dir, fixed=[{"question_id": "4", "expect_image": False}])
        self.assertEqual(len(probes), 1)
        q = probes[0]
        self.assertFalse(q["has_image"])
        self.assertIsNone(q["image"])
        self.assertNotIn(pl.IMAGE_TOKEN, q["prompt"])
        self.assertNotIn(pl.IMAGE_TOKEN, q["text"])
        self.assertEqual(q["prompt"], TEXT_ONLY["text"])

    def test_image_entry(self):
        probes, _ = pl.build_probe_set(
            self.pj, self.img_dir, fixed=[{"question_id": "5", "expect_image": True}])
        q = probes[0]
        self.assertTrue(q["has_image"])
        self.assertEqual(q["image"], WITH_IMAGE["image"])
        self.assertTrue(q["prompt"].startswith("<image>\n"))

    def test_default_fixed_covers_both_types(self):
        probes, _ = pl.build_probe_set(self.pj, self.img_dir)
        self.assertEqual([q["question_id"] for q in probes], ["4", "5"])
        self.assertEqual(sum(1 for q in probes if not q["has_image"]), 1)
        self.assertEqual(sum(1 for q in probes if q["has_image"]), 1)

    def test_data_sha256_matches_file(self):
        _, sha = pl.build_probe_set(self.pj, self.img_dir)
        self.assertEqual(sha, pl.sha256_file(self.pj))

    def test_declared_image_missing_fails(self):
        os.remove(os.path.join(self.img_dir, "ScienceQA/images/test/5/image.png"))
        with self.assertRaises(FileNotFoundError):
            pl.build_probe_set(self.pj, self.img_dir)

    @unittest.skipUnless(_pil_ok(), "PIL 不可用，跳过损坏图片解码用例")
    def test_corrupt_image_fails(self):
        img = os.path.join(self.img_dir, "ScienceQA/images/test/5/image.png")
        with open(img, "wb") as f:
            f.write(b"this is not an image at all" * 10)
        with self.assertRaises(ValueError):
            pl.build_probe_set(self.pj, self.img_dir)

    def test_empty_image_file_fails(self):
        img = os.path.join(self.img_dir, "ScienceQA/images/test/5/image.png")
        open(img, "wb").close()
        with self.assertRaises(ValueError):
            pl.build_probe_set(self.pj, self.img_dir)

    def test_fixed_id_missing_fails(self):
        with self.assertRaises(ValueError):
            pl.build_probe_set(self.pj, self.img_dir,
                               fixed=[{"question_id": "999", "expect_image": False}])

    def test_duplicate_id_fails(self):
        pj = make_probe_json(os.path.join(self.tmp, "dup.json"),
                             [dict(TEXT_ONLY), dict(TEXT_ONLY)])
        with self.assertRaises(ValueError):
            pl.build_probe_set(pj, self.img_dir)

    def test_expect_image_mismatch_fails(self):
        with self.assertRaises(ValueError):
            pl.build_probe_set(self.pj, self.img_dir,
                               fixed=[{"question_id": "4", "expect_image": True}])

    def test_missing_question_id_fails(self):
        pj = make_probe_json(os.path.join(self.tmp, "noid.json"),
                             [{"text": "no id here"}])
        with self.assertRaises(ValueError):
            pl.build_probe_set(pj, self.img_dir)

    def test_empty_text_fails(self):
        pj = make_probe_json(os.path.join(self.tmp, "empty.json"),
                             [{"question_id": "4", "text": "   "}])
        with self.assertRaises(ValueError):
            pl.build_probe_set(pj, self.img_dir)

    def test_prompt_hash_deterministic_across_calls(self):
        p1, _ = pl.build_probe_set(self.pj, self.img_dir)
        p2, _ = pl.build_probe_set(self.pj, self.img_dir)
        for a, b in zip(p1, p2):
            self.assertEqual(a["prompt"], b["prompt"])
            self.assertEqual(pl.sha256_text(a["prompt"]), pl.sha256_text(b["prompt"]))

    def test_traversal_image_path_fails(self):
        pj = make_probe_json(os.path.join(self.tmp, "trav.json"),
                             [{"question_id": "5", "text": "t",
                               "image": "ScienceQA/../../etc/passwd"}])
        with self.assertRaises(ValueError):
            pl.build_probe_set(pj, self.img_dir)


if __name__ == "__main__":
    unittest.main()
