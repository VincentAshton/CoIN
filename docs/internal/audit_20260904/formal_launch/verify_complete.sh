#!/bin/bash
# 阶段 6 验收：ratio=0.1 正式结果深度检查（2026-09-04）
export PATH=/root/data/coin/conda_envs/coin/bin:$PATH
PY=/root/data/coin/conda_envs/coin/bin/python
R=/root/data/coin/project/results/CoIN_Replay/ratio_0.1
C=/root/data/coin/project/checkpoints/CoIN_Replay/ratio_0.1
P=/root/data/coin/project
FAIL=0
ck() { if [ "$1" = "0" ]; then echo "PASS  $2"; else echo "FAIL  $2"; FAIL=1; fi; }

# 1. 完整/标志
[ -f "$R/.complete" ] && ck 0 "1a .complete" || ck 1 "1a .complete"
N=$(ls "$R"/.round*_done 2>/dev/null | wc -l); [ "$N" = "4" ] && ck 0 "1b round markers (4)" || ck 1 "1b markers=$N"
[ -f "$R/coin_metrics.json" ] && ck 0 "1c coin_metrics.json" || ck 1 "1c coin_metrics"

# 2. checkpoint 可加载 + finite（8 个：r1-4 task + r2-4 replay）
for j in 1 2 3 4; do
  ckpt="$C/round${j}_task_llava_lora"
  rep=$($PY "$P/scripts/CoIN_Replay/coin_lib.py" ckpt-validate "$ckpt" 2>/dev/null)
  f=$(echo "$rep" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('finite'))")
  [ "$f" = "True" ] && ck 0 "2 task r$j 可加载+finite" || ck 1 "2 task r$j finite=$f"
done
for j in 2 3 4; do
  ckpt="$C/round${j}_replay_llava_lora"
  rep=$($PY "$P/scripts/CoIN_Replay/coin_lib.py" ckpt-validate "$ckpt" 2>/dev/null)
  f=$(echo "$rep" | $PY -c "import json,sys; d=json.load(sys.stdin); print(d.get('finite'))")
  [ "$f" = "True" ] && ck 0 "2 replay r$j 可加载+finite" || ck 1 "2 replay r$j finite=$f"
done

# 3. task/replay tensor diff（r2-4）
for j in 2 3 4; do
  if $PY "$P/scripts/CoIN_Replay/coin_lib.py" ckpt-tensor-diff "$C/round${j}_task_llava_lora" "$C/round${j}_replay_llava_lora" > /tmp/diff_$j.json 2>/dev/null; then
    ck 0 "3 r$j tensor-diff pass ($($PY -c "import json;d=json.load(open('/tmp/diff_$j.json'));print(d['changed_tensor_count'],'changed')" 2>/dev/null))"
  else
    ck 1 "3 r$j tensor-diff FAIL"
  fi
done

# 4. DS 真步 vs manifest（replay plan ds_expected_updates vs trainer_state gs）
$PY - "$R" "$C" <<'PYEOF'
import json, glob, os, sys
R, C = sys.argv[1], sys.argv[2]
ok = True
for j in (2, 3, 4):
    rm = os.path.join(R, f"round{j}_manifest.json")
    m = json.load(open(rm))
    exp = m.get("replay_plan", {}).get("ds_expected_updates")
    ts = os.path.join(C, f"round{j}_replay_llava_lora", "trainer_state.json")
    gs = json.load(open(ts)).get("global_step") if os.path.isfile(ts) else None
    N = m.get("replay_plan", {}).get("N")
    good = (exp == gs) and exp is not None and exp > 0
    print(f"{'PASS' if good else 'FAIL'}  4 r{j} replay N={N} ds_expected={exp} trainer_gs={gs}")
    ok = ok and good
sys.exit(0 if ok else 1)
PYEOF
ck $? "4 DS 真步与 manifest 一致"

# 5. eval 单元数量/产物/预测完整性
$PY - "$R" <<'PYEOF'
import json, os, sys
R = sys.argv[1]
units = []
for task in ("ScienceQA", "TextVQA", "ImageNet", "GQA"):
    td = os.path.join(R, task)
    if not os.path.isdir(td):
        continue
    for rd in sorted(os.listdir(td)):
        if rd.startswith("round"):
            units.append((task, rd))
expected = {1: ["ScienceQA"], 2: ["ScienceQA", "TextVQA"],
            3: ["ScienceQA", "TextVQA", "ImageNet"],
            4: ["ScienceQA", "TextVQA", "ImageNet", "GQA"]}
ok = True
got = {}
for t, rd in units:
    got.setdefault(int(rd[5:]), []).append(t)
for j, tasks in expected.items():
    if sorted(got.get(j, [])) != sorted(tasks):
        print(f"FAIL  5 round{j} 单元缺: {got.get(j)} vs {tasks}"); ok = False
# 每单元 merge.jsonl 行数
for t, rd in units:
    mf = os.path.join(R, t, rd, "merge.jsonl")
    n = sum(1 for _ in open(mf)) if os.path.isfile(mf) else -1
    if n <= 0:
        print(f"FAIL  5 {t}/{rd} merge 空"); ok = False
print(f"5 eval 单元: {len(units)} 个（期望 10）", "PASS" if len(units) == 10 and ok else "FAIL")
sys.exit(0 if len(units) == 10 and ok else 1)
PYEOF
ck $? "5 十单元齐全+merge 非空"

# 6. manifest 复核
$PY -c "import json; m=json.load(open('$R/run_manifest.json')); c=m['config']; print('ratio',c['ratio'],'replay_accum',c['replay_accum'],'grad_accum',c['grad_accum'],'git',m['git']['commit'])"
echo "=== 磁盘 ==="; df -h /root/data | tail -1
echo "=== 残留进程 ==="; ps aux | grep -cE "[t]rain_mem|[m]odel_vqa|[r]un_sweep"
echo "=== 最终结果 ==="
$PY -c "import json; m=json.load(open('$R/coin_metrics.json')); print('MAA=%.4f BWT=%.4f' % (m['MAA'], m['BWT']))"
exit $FAIL
