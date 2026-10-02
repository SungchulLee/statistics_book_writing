"""쪽의 python 블록을 돌려 바로 아래 출력 블록과 대조한다.

    python3 verify_outputs.py docs/ch14            장 전체
    python3 verify_outputs.py docs/ch14/limitations/limitations.md

한 쪽 안의 블록은 앞 블록의 변수를 이어 쓰는 일이 있으므로, 대상 블록 앞의
python 블록들을 **출력을 삼킨 채** 먼저 실행한 뒤 대상 블록만 찍는다.

들여쓴 울타리는 **울타리의 들여쓰기만큼만** 벗긴다. textwrap.dedent 를 쓰면
머리글 줄의 선행 공백이 어긋나 거짓 불일치가 난다.
"""
import pathlib
import re
import subprocess
import sys
import tempfile

FENCE = re.compile(r"^(\s*)```(\w*)\s*$")
RUNNER = r'''
import contextlib, io, sys, warnings
warnings.filterwarnings("ignore")
_pre_path, _tgt_path = sys.argv[1], sys.argv[2]
# 쪽의 코드가 argparse 를 쓰면 우리 argv 를 자기 인자로 읽어 죽는다.
# (ch05 standard_error.md 가 바로 그 위험을 가르친다.) 깨끗이 비워 준다.
sys.argv = ["block"]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _p
_p.show = lambda *a, **k: None
_g = {"__name__": "__main__"}
_prefix = open(_pre_path, encoding="utf-8").read()
_target = open(_tgt_path, encoding="utf-8").read()
if _prefix.strip():
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(_prefix, "prefix", "exec"), _g)
        except Exception:
            pass
exec(compile(_target, "target", "exec"), _g)
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
            if FENCE.match(lines[j]) and FENCE.match(lines[j]).group(2) == "" \
                    and len(FENCE.match(lines[j]).group(1)) == len(ind):
                break
            body.append(lines[j][len(ind):] if lines[j].startswith(ind) else lines[j].lstrip())
            j += 1
        out.append((lang, i + 1, "\n".join(body)))
        i = j + 1
    return out


def check(path, timeout=900):
    bs = blocks(path)
    rows = []
    # 쪽이 "import 를 마쳤다고 보고 적는다" 라 선언한 경우 그 전제를 넣어 준다.
    # 이것은 결함이 아니라 그 쪽의 규약이다 (예: ch00/tools/numpy_basics.md 13 행).
    head = path.read_text(encoding="utf-8")[:3000]
    prefix = []
    for name, imp in (("numpy as np", "import numpy as np"),
                      ("pandas as pd", "import pandas as pd"),
                      ("matplotlib.pyplot as plt", "import matplotlib.pyplot as plt")):
        if f"`import {name}`" in head and "마쳤다고 보고" in head:
            prefix.append(imp)
    for k, (lang, ln, code) in enumerate(bs):
        if lang not in ("python", "py"):
            continue
        # 바로 다음 블록이 출력인가
        exp = None
        if k + 1 < len(bs) and bs[k + 1][0] in ("", "text"):
            exp = bs[k + 1][2]
        if exp is None:
            prefix.append(code)
            continue
        with tempfile.TemporaryDirectory() as d:
            pf = pathlib.Path(d, "p.py"); pf.write_text("\n".join(prefix), encoding="utf-8")
            tf = pathlib.Path(d, "t.py"); tf.write_text(code, encoding="utf-8")
            rf = pathlib.Path(d, "r.py"); rf.write_text(RUNNER, encoding="utf-8")
            try:
                r = subprocess.run([sys.executable, str(rf), str(pf), str(tf)],
                                   capture_output=True, text=True, timeout=timeout,
                                   cwd=str(pathlib.Path.cwd()))
                got, err, rc = r.stdout, r.stderr, r.returncode
            except subprocess.TimeoutExpired:
                got, err, rc = "", "TIMEOUT", -9
        a = [x.rstrip() for x in got.strip("\n").split("\n")]
        b = [x.rstrip() for x in exp.strip("\n").split("\n")]
        if rc != 0:
            rows.append(("ERROR", ln, err.strip().split("\n")[-1][:90]))
        elif a != b:
            d1 = next((f"{i+1}행  코드[{a[i] if i < len(a) else '<없음>'}]  쪽[{b[i] if i < len(b) else '<없음>'}]"
                       for i in range(max(len(a), len(b)))
                       if (a[i] if i < len(a) else None) != (b[i] if i < len(b) else None)), "")
            rows.append(("DIFF", ln, d1[:130]))
        else:
            rows.append(("OK", ln, f"{len(b)}행"))
        prefix.append(code)
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
        mark = "OK  " if not bad else "BAD "
        print(f"{mark}{p}  ({len(rows)} 블록)", flush=True)
        for kind, ln, msg in bad:
            print(f"      {kind:6s}{ln:5d}  {msg}", flush=True)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
