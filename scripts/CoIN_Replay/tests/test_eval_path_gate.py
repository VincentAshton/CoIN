"""eval 路径门禁单测（2026-09-11 修正）。

背景（实踩）：四个 eval shell 与 ETrain/Eval/LLaVA/CoIN/eval_gqa.py 曾硬编码相对路径
（./checkpoints/LLaVA/...、./cl_dataset、./playground/...、eval_gqa 内的 ./cl_dataset/GQA），
新 worktree 缺这些软链时，评估阶段才失败——**训练数小时之后**才发现（单次损失 ~3.2h GPU）。
修复要求：① 路径全部可显式覆盖；② 训练前门禁 fail-fast；③ 本门禁自带回归测试。

本文件用**真实仓库文件**做模板（复制到临时树再变异），保证「有人改回硬编码」能被拦下。
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coin_lib import EVAL_SHELL_SCRIPTS, eval_path_audit  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COIN_LIB = os.path.join(ROOT, "scripts", "CoIN_Replay", "coin_lib.py")
REAL_EVAL_DIR = os.path.join(ROOT, "scripts", "LLaVA", "Eval")
REAL_PY_DIR = os.path.join(ROOT, "ETrain", "Eval", "LLaVA", "CoIN")
PY_MODULES = ("eval_science_qa.py", "eval_textvqa.py", "eval_ImagetNet.py", "eval_gqa.py",
              "convert_gqa_for_eval.py", "model_vqa.py", "model_vqa_science.py",
              "model_text_vqa.py", "model_gqa.py")


class TestEvalPathAudit(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="coin_evalpath_")
        self.root = os.path.join(self.tmp, "repo")
        self.sd = os.path.join(self.root, "scripts", "LLaVA", "Eval")
        self.pd = os.path.join(self.root, "ETrain", "Eval", "LLaVA", "CoIN")
        os.makedirs(self.sd)
        os.makedirs(self.pd)
        for name in EVAL_SHELL_SCRIPTS:
            shutil.copy2(os.path.join(REAL_EVAL_DIR, name), os.path.join(self.sd, name))
        for name in PY_MODULES:
            src = os.path.join(REAL_PY_DIR, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(self.pd, name))
        self.base = os.path.join(self.tmp, "model_base")
        self.img = os.path.join(self.tmp, "cl_dataset")
        os.makedirs(self.base)
        os.makedirs(self.img)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _audit(self, **over):
        kw = dict(root=self.root, model_base=self.base, image_folder=self.img)
        kw.update(over)
        return eval_path_audit(**kw)

    def _sub(self, path, old, new, count=1):
        with open(path) as fh:
            txt = fh.read()
        self.assertIn(old, txt, f"模板文件缺少待替换文本: {path}")
        with open(path, "w") as fh:
            fh.write(txt.replace(old, new, count))

    # ---- 正路径 ---------------------------------------------------------------

    def test_pass_on_real_repo_files(self):
        rep = self._audit()
        self.assertTrue(rep["pass"], json.dumps(rep, ensure_ascii=False)[:2000])
        self.assertEqual(rep["uncovered_relative_refs"], [])
        self.assertEqual(rep["missing_paths"], [])
        self.assertTrue(rep["hooks_ok"])
        self.assertTrue(rep["eval_gqa_data_root_ok"])

    def test_real_repo_root_passes(self):
        """真实仓库根 + 真实存在路径也能过（不依赖 ./checkpoints、./cl_dataset 软链）。"""
        rep = eval_path_audit(ROOT, model_base=self.base, image_folder=self.img)
        self.assertTrue(rep["pass"], json.dumps(rep["uncovered_relative_refs"])[:800])

    # ---- 回归：把实踩的 bug 改回去必须被拦下 ----------------------------------

    def test_fails_when_shell_hardcodes_model_base_again(self):
        """把 ${MODEL_BASE:-...} 覆盖改回硬编码 → 门禁 FAIL（钩子缺失）。"""
        self._sub(os.path.join(self.sd, "1_eval_sqa.sh"),
                  '--model-base "$MODEL_BASE"',
                  '--model-base ./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5')
        self._sub(os.path.join(self.sd, "1_eval_sqa.sh"),
                  'MODEL_BASE="${MODEL_BASE:-./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5}"',
                  "MODEL_BASE=./checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5")
        rep = self._audit()
        self.assertFalse(rep["pass"])
        self.assertFalse(rep["hooks"]["1_eval_sqa.sh"]["model_base_hook"])
        # 硬编码的相对路径在临时 root 下不存在 → 也要报出来
        self.assertTrue(any(r["file"].endswith("1_eval_sqa.sh")
                            for r in rep["uncovered_relative_refs"]))

    def test_fails_when_image_folder_hook_removed(self):
        self._sub(os.path.join(self.sd, "2_eval_textqa.sh"),
                  'IMAGE_FOLDER="${IMAGE_FOLDER:-./cl_dataset}"',
                  "IMAGE_FOLDER=./cl_dataset")
        rep = self._audit()
        self.assertFalse(rep["pass"])
        self.assertFalse(rep["hooks"]["2_eval_textqa.sh"]["image_folder_hook"])

    def test_fails_when_gqa_data_root_hardcoded_again(self):
        """eval_gqa.py 改回 os.path.join('./cl_dataset/GQA', ...) → 门禁 FAIL（实踩 bug 回归）。"""
        self._sub(os.path.join(self.pd, "eval_gqa.py"),
                  "os.path.join(args.data_root, 'GQA'",
                  "os.path.join('./cl_dataset/GQA', 'GQA'")
        rep = self._audit()
        self.assertFalse(rep["pass"])
        self.assertTrue(any(r["file"].endswith("eval_gqa.py")
                            for r in rep["uncovered_relative_refs"]),
                        json.dumps(rep["uncovered_relative_refs"], ensure_ascii=False))

    def test_fails_when_data_root_arg_removed(self):
        self._sub(os.path.join(self.pd, "eval_gqa.py"),
                  "parser.add_argument('--data-root'", "parser.add_argument('--data-root-disabled'")
        rep = self._audit()
        self.assertFalse(rep["pass"])
        self.assertFalse(rep["eval_gqa_data_root_ok"])

    def test_fails_when_new_hardcoded_relative_path_added(self):
        """新增任何无法解析的相对路径（例如 ./examples/foo）→ FAIL。"""
        self._sub(os.path.join(self.pd, "model_gqa.py"),
                  "import os", "import os\n_DEFAULT_WEIRD = './examples/definitely_missing'\n")
        rep = self._audit()
        self.assertFalse(rep["pass"])
        self.assertTrue(any(r["ref"] == "./examples/definitely_missing"
                            for r in rep["uncovered_relative_refs"]))

    # ---- 传入路径不存在 -------------------------------------------------------

    def test_fails_when_model_base_missing(self):
        rep = self._audit(model_base=os.path.join(self.tmp, "nope"))
        self.assertFalse(rep["pass"])
        self.assertTrue(rep["missing_paths"])

    def test_fails_when_image_folder_missing(self):
        rep = self._audit(image_folder=os.path.join(self.tmp, "nope_dataset"))
        self.assertFalse(rep["pass"])
        self.assertTrue(rep["missing_paths"])

    # ---- CLI 契约（编排脚本按退出码 fail-fast）--------------------------------

    def test_cli_exit_code_and_json(self):
        ok = subprocess.run([sys.executable, COIN_LIB, "eval-path-audit", "--root", self.root,
                             "--model-base", self.base, "--image-folder", self.img],
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertTrue(json.loads(ok.stdout)["pass"])
        bad = subprocess.run([sys.executable, COIN_LIB, "eval-path-audit", "--root", self.root,
                              "--model-base", self.base,
                              "--image-folder", os.path.join(self.tmp, "missing_dataset")],
                             capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertFalse(json.loads(bad.stdout)["pass"])

    def test_comment_lines_ignored(self):
        """注释里的相对路径不算硬编码（否则提示文本/maintenance 注释会误报）。"""
        self._sub(os.path.join(self.pd, "model_gqa.py"),
                  "import os", "# 说明：历史上使用过 ./cl_dataset/legacy\nimport os")
        rep = self._audit()
        self.assertTrue(rep["pass"], json.dumps(rep["uncovered_relative_refs"], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
