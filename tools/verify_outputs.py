"""쪽의 python 블록을 돌려 바로 아래 출력 블록과 대조한다.

    python3 tools/verify_outputs.py docs/ch14            장 전체 (디렉터리면 재귀)
    python3 tools/verify_outputs.py docs/ch14/limitations/limitations.md

한 쪽의 블록은 **하나의 프로세스에서 차례로** 실행하고 블록마다 stdout 을 따로
받는다. 쪽 안의 블록은 원래 앞 블록의 변수를 이어 쓰도록 쓰여 있으므로 이것이
의미도 맞고, 블록마다 앞 블록을 다시 돌리던 예전 방식의 O(n^2) 낭비도 없앤다
(ch11 f_test.md 가 20 분 넘게 걸리던 것이 이 때문이었다).

들여쓴 울타리는 **울타리의 들여쓰기만큼만** 벗긴다. textwrap.dedent 를 쓰면
머리글 줄의 선행 공백이 어긋나 거짓 불일치가 난다.

반드시 BLAS 스레드를 묶어 돌릴 것. 이 책의 모의실험은 배열이 작아 단일 스레드가
오히려 빠르고, 여럿을 함께 돌리면 8 코어에서 부하가 190 을 넘는다.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg \\
    python3 tools/verify_outputs.py docs/ch14

주의: 저장소 뿌리에서 돌리면 `savefig("x.png")` 처럼 상대경로로 저장하는 블록이
뿌리에 PNG 를 흘린다. 돌린 뒤 `git status` 로 확인하라.
"""
import json
import pathlib
import re
import subprocess
import sys
import tempfile

FENCE = re.compile(r"^(\s*)```(\w*)\s*$")

RUNNER = r'''
import contextlib, io, json, sys, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: None

_spec, _result = sys.argv[1], sys.argv[2]
_blocks = json.load(open(_spec, encoding="utf-8"))

# 쪽의 코드가 argparse 를 쓰면 우리 argv 를 자기 인자로 읽고 죽는다.
# (ch05 standard_error.md 가 바로 그 위험을 가르친다.) 깨끗이 비워 준다.
sys.argv = ["block"]

_g = {"__name__": "__main__"}
_out = []
for _code in _blocks:
    _buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_buf):
            exec(compile(_code, "block", "exec"), _g)
        _out.append({"ok": True, "stdout": _buf.getvalue()})
    except BaseException as _e:
        _out.append({"ok": False, "stdout": _buf.getvalue(),
                     "err": "%s: %s" % (type(_e).__name__, _e)})
    finally:
        _plt.close("all")
json.dump(_out, open(_result, "w", encoding="utf-8"))
'''


def blocks(path):
    """[(종류, 시작줄, 코드)] — 종류는 'python' 또는 출력('', 'text')."""
    lines = path.read_text(encoding="utf-8").split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        ind, lang = m.group(1), m.group(2)
        j = i + 1
        body = []
        while j < len(lines):
            mj = FENCE.match(lines[j])
            if mj and mj.group(2) == "" and len(mj.group(1)) == len(ind):
                break
            body.append(lines[j][len(ind):] if lines[j].startswith(ind) else lines[j].lstrip())
            j += 1
        out.append((lang, i + 1, "\n".join(body)))
        i = j + 1
    return out


def preamble(path):
    """쪽이 "import 를 마쳤다고 보고 적는다" 고 선언했으면 그 전제를 넣어 준다.

    결함이 아니라 그 쪽의 규약이다 (예: ch00/tools/numpy_basics.md 13 행).
    """
    head = path.read_text(encoding="utf-8")[:3000]
    pre = []
    for name, imp in (("numpy as np", "import numpy as np"),
                      ("pandas as pd", "import pandas as pd"),
                      ("matplotlib.pyplot as plt", "import matplotlib.pyplot as plt")):
        if f"`import {name}`" in head and "마쳤다고 보고" in head:
            pre.append(imp)
    return "\n".join(pre)


def check(path, timeout=1800):
    bs = blocks(path)
    codes, meta = [], []          # meta: (시작줄, 기대 출력 또는 None)
    pre = preamble(path)
    for k, (lang, ln, code) in enumerate(bs):
        if lang not in ("python", "py"):
            continue
        exp = None
        if k + 1 < len(bs) and bs[k + 1][0] in ("", "text"):
            exp = bs[k + 1][2]
        codes.append((pre + "\n" + code) if (pre and not codes) else code)
        meta.append((ln, exp))
    if not codes:
        return []

    with tempfile.TemporaryDirectory() as d:
        spec = pathlib.Path(d, "spec.json")
        spec.write_text(json.dumps(codes), encoding="utf-8")
        res = pathlib.Path(d, "res.json")
        run = pathlib.Path(d, "run.py")
        run.write_text(RUNNER, encoding="utf-8")
        try:
            subprocess.run([sys.executable, str(run), str(spec), str(res)],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(pathlib.Path.cwd()))
            got = json.loads(res.read_text(encoding="utf-8")) if res.exists() else None
        except subprocess.TimeoutExpired:
            got = None
    if got is None:
        return [("TIMEOUT", meta[0][0], f"{timeout}s 안에 끝나지 않았다")]

    rows = []
    for (ln, exp), r in zip(meta, got):
        if exp is None:
            if not r["ok"]:
                rows.append(("ERROR", ln, r["err"][:90]))
            continue
        if not r["ok"]:
            rows.append(("ERROR", ln, r["err"][:90]))
            continue
        a = [x.rstrip() for x in r["stdout"].strip("\n").split("\n")]
        b = [x.rstrip() for x in exp.strip("\n").split("\n")]
        if a != b:
            d1 = next((f"{i+1}행  코드[{a[i] if i < len(a) else '<없음>'}]  "
                       f"쪽[{b[i] if i < len(b) else '<없음>'}]"
                       for i in range(max(len(a), len(b)))
                       if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)), "")
            rows.append(("DIFF", ln, d1[:130]))
        else:
            rows.append(("OK", ln, f"{len(b)}행"))
    return rows


def main(argv):
    t = pathlib.Path(argv[0])
    files = sorted(t.rglob("*.md")) if t.is_dir() else [t]
    tally = {}
    for p in files:
        rows = check(p)
        if not rows:
            continue
        bad = [r for r in rows if r[0] != "OK"]
        for kind, _, _ in rows:
            tally[kind] = tally.get(kind, 0) + 1
        print(f"{'OK  ' if not bad else 'BAD '}{p}  ({len(rows)} 블록)", flush=True)
        for kind, ln, msg in bad:
            print(f"      {kind:7s}{ln:5d}  {msg}", flush=True)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
