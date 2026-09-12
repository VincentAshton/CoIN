#!/bin/bash
# 门禁 A：bash -n / py_compile / 全部单测（零 GPU，可在本地或云端运行）。
# 用法: bash scripts/CoIN_Replay/run_tests.sh [PYTHON]
# 退出码: 0=全部通过；非 0=任一失败（不吞错，每个阶段独立报错）
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${1:-python3}"
FAIL=0
LOG="$(mktemp /tmp/coin_tests_XXXX.log)"

echo "==== [A1] bash -n 全部 shell 脚本 ===="
while IFS= read -r f; do
    if ! bash -n "$f"; then
        echo "FAIL bash -n: $f"; FAIL=1
    fi
done < <(find "$ROOT/scripts" -name '*.sh' | sort)
echo "bash -n done (fail=$FAIL)"

echo "==== [A2] py_compile 全部 python 脚本 ===="
if ! "$PY" -m py_compile "$ROOT"/scripts/CoIN_Replay/*.py "$ROOT"/scripts/CoIN_Replay/tools/*.py; then
    echo "FAIL py_compile"; FAIL=1
fi
echo "py_compile done (fail=$FAIL)"

echo "==== [A3] unittest 全套（零 GPU） ===="
"$PY" -m unittest discover -s "$ROOT/scripts/CoIN_Replay/tests" -v 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
    echo "FAIL unittest (rc=$RC)"; FAIL=1
fi

echo "==== [A4] zero3_offload.json 无 scheduler/optimizer 段（工单 3） ===="
if grep -qE '"scheduler"|"optimizer"|WarmupLR' "$ROOT/scripts/zero3_offload.json"; then
    echo "FAIL: zero3_offload.json 含 scheduler/optimizer 段（会替换 cosine/mm_projector_lr）"
    FAIL=1
else
    echo "zero3_offload.json OK（无 scheduler/optimizer 段）"
fi

echo "==== [A5] protobuf==4.25.3（llama tokenizer 前置；评审 2026-09-02） ===="
# protobuf 4.x 无顶层 `protobuf` 模块（import protobuf 必然失败），用 importlib.metadata 查版本
PBV=$("$PY" -c "import importlib.metadata as im; print(im.version('protobuf'))" 2>/dev/null)
if [ -n "$PBV" ]; then
    if [ "$PBV" = "4.25.3" ]; then
        echo "protobuf 4.25.3 OK"
    else
        echo "FAIL: protobuf 版本 $PBV != 4.25.3（新版 protobuf 使 sentencepiece_model_pb2 导入静默失败，tokenizer 会崩）"
        FAIL=1
    fi
else
    echo "protobuf 不可用（零依赖环境），跳过版本断言"
fi

echo "==== [A6] PIL 可用性（图片解码类用例前置；本地零依赖环境会 skip） ===="
PILV=$("$PY" -c "import PIL, importlib.metadata as im; print(im.version('Pillow'))" 2>/dev/null)
if [ -n "$PILV" ]; then
    echo "Pillow $PILV OK（图片解码类用例真实执行）"
else
    echo "PIL 不可用（零依赖环境）：图片解码类用例标 skip（见 helpers.PIL_SKIP_REASON）——"
    echo "  云端完整依赖环境必须真实执行这些用例，不得把 skip 计为通过"
fi

echo "==== [A7] random 系列文档一致性 + 脱敏扫描（push 前自查） ===="
# 权威数据 = index.json / paired_comparison.json；README 状态表与总入口数字必须与之同步。
# legacy 例外（r001 已发布审核文档）只 warn；--strict 可升级为 FAIL。
if ! "$PY" "$ROOT/scripts/CoIN_Replay/tools/random_replay_sync_check.py" \
        --repo-root "$ROOT" | tail -30; then
    echo "FAIL random_replay_sync_check（README/registry 漂移或新增文档未脱敏）"
    FAIL=1
fi

echo "=================================================="
if [ "$FAIL" -eq 0 ]; then
    echo "门禁 A 全部通过"
else
    echo "门禁 A 存在失败，详见上方输出（完整日志: $LOG）"
fi
exit "$FAIL"
