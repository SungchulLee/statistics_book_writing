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

출력이 **일부러** 재현되지 않는 블록은 코드 안에 `# verify: skip` 을 적어 두면
출력 비교를 건너뛴다(코드는 그대로 돌려 오류는 잡는다). 씨앗 없는 난수가 돌릴
때마다 달라지는 것을 보여 주는 보기가 그런 경우다. **결함을 덮는 데 쓰지 말 것** —
재현되지 않는 수를 실어 놓고 이 표시로 가리면 독자가 속는다.

애초에 **돌아갈 수 없는** 블록은 `# verify: noexec — <까닭>` 을 적는다. 바깥
꾸러미의 속을 그대로 옮겨 보이는 자리가 그렇다. 그 코드의 `X`, `Y` 는 옮겨 온
함수가 **인자로 받는** 것이라 이 자리에 있을 수가 없다. `skip` 으로는 안 된다 —
그것은 「돌리되 출력은 보지 않는다」이므로 NameError 가 그대로 남는다. noexec 는
돌리지 않으므로 뒤 블록이 이 블록의 변수를 쓰고 있으면 거기서 터진다. 이 표시는
**스스로 닫힌 인용**에만 쓸 것이다. 건너뛴 블록은 `SKIP` 으로 찍어 남기고, 쪽을
BAD 로 세지는 않는다 — 조용히 사라지면 결함을 덮는 데 쓰이기 때문이다.

시간제한은 **프로세스 무리째** 건다. `subprocess.run(timeout=)` 은 자식만 죽이므로
`n_jobs=-1` 이 띄운 joblib 일꾼은 살아남아 계속 돈다. 그러면 다음 쪽부터는 남의
일꾼과 코어를 다투게 된다. 자식을 `start_new_session` 으로 띄우고 `os.killpg` 로
무리째 죽이는 까닭이다.

시간제한은 **단조시계**로 재므로 맥이 잠든 동안은 흐르지 않는다. 밤에 걸어 두면
`ps` 의 etime 은 한 시간이 넘는데 시간제한은 아직 차지 않은 일이 생긴다. 그런
쪽을 만나면 멈춘 것인지 잠든 것인지 `pmset -g log` 로 먼저 가려라.
"""
import json
import os
import pathlib
import re
import signal
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


SKIP_OUT = re.compile(r"#\s*(verify|검증)\s*:\s*(skip|출력무시|출력\s*무시)")
# 돌아갈 수 없는 블록 — 바깥 꾸러미의 속을 그대로 옮긴 인용 따위. SKIP_OUT 과
# 달리 아예 돌리지 않는다. 까닭은 표시 뒤에 적게 하고 그대로 찍어 보인다.
NOEXEC = re.compile(r"#\s*(?:verify|검증)\s*:\s*(?:noexec|실행안함|돌리지\s*않음)"
                    r"\s*(?:[—–-]\s*(?P<why>.*))?")


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
    skipped = []                  # 돌리지 않은 블록 — (시작줄, 까닭)
    pre = preamble(path)
    for k, (lang, ln, code) in enumerate(bs):
        if lang not in ("python", "py"):
            continue
        mx = NOEXEC.search(code)
        if mx:
            # 돌리지 않으므로 codes/meta 양쪽에서 빠진다. 둘을 함께 빼야
            # 아래 zip(meta, got) 의 짝이 어긋나지 않는다.
            skipped.append((ln, (mx.group("why") or "").strip() or "까닭이 적혀 있지 않다"))
            continue
        exp = None
        if k + 1 < len(bs) and bs[k + 1][0] in ("", "text"):
            exp = bs[k + 1][2]
        # 쪽이 "이 블록의 출력은 재현되지 않는다" 고 스스로 선언할 수 있다.
        # 결함이 아니라 교육 목적인 경우가 있다 — 씨앗 없는 난수가 돌릴 때마다
        # 달라지는 것을 보여 주는 보기가 그렇다. 코드는 그대로 돌려 오류를
        # 잡되, 출력은 맞춰 보지 않는다.
        if SKIP_OUT.search(code):
            exp = None
        codes.append((pre + "\n" + code) if (pre and not codes) else code)
        meta.append((ln, exp))
    if not codes:
        return [("SKIP", ln, why) for ln, why in skipped]

    with tempfile.TemporaryDirectory() as d:
        spec = pathlib.Path(d, "spec.json")
        spec.write_text(json.dumps(codes), encoding="utf-8")
        res = pathlib.Path(d, "res.json")
        run = pathlib.Path(d, "run.py")
        run.write_text(RUNNER, encoding="utf-8")
        log = pathlib.Path(d, "stderr.txt")
        # 결과는 res.json 으로 받으므로 자식의 출력은 파이프가 아니라 파일로 뺀다.
        # 파일이면 죽은 뒤에도 읽을 수 있어, 결과를 못 남기고 죽은 쪽의 마지막
        # 줄을 알려 줄 수 있다.
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.Popen(
                [sys.executable, str(run), str(spec), str(res)],
                stdout=fh, stderr=subprocess.STDOUT,
                # 쪽의 예제가 savefig("a.png") 처럼 상대경로로 그림을
                # 저장하면 cwd 에 떨어진다. 저장소 뿌리에서 돌리면
                # 뿌리가 더럽혀지므로 임시 디렉터리에서 돌린다.
                # (쪽 코드가 읽는 자료는 모두 URL 이라 안전하다.)
                cwd=d, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
                died = None
            except subprocess.TimeoutExpired:
                # 자식만 죽이면 joblib/multiprocessing 일꾼이 살아남아 다음 쪽과
                # 코어를 다툰다. start_new_session 으로 따로 띄워 두었으므로
                # 무리째 죽인다.
                os.killpg(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                died = ("TIMEOUT", f"{timeout}s 안에 끝나지 않았다")
        got = json.loads(res.read_text(encoding="utf-8")) if res.exists() else None
        if got is None and died is None:
            tail = [x for x in log.read_text(encoding="utf-8",
                                             errors="replace").split("\n") if x.strip()]
            died = ("DEAD", "결과를 남기지 못하고 죽었다"
                            f" ({tail[-1][:70] if tail else '출력 없음'})")
    if got is None:
        return [(died[0], meta[0][0], died[1])] + \
               [("SKIP", ln, why) for ln, why in skipped]

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
    rows += [("SKIP", ln, why) for ln, why in skipped]
    rows.sort(key=lambda r: r[1])          # 쪽에 나오는 순서대로 되돌린다
    return rows


def main(argv):
    t = pathlib.Path(argv[0])
    files = sorted(t.rglob("*.md")) if t.is_dir() else [t]
    tally = {}
    for p in files:
        rows = check(p)
        if not rows:
            continue
        # SKIP 은 결함이 아니므로 쪽을 BAD 로 만들지 않되, 찍어서 남긴다.
        # 조용히 사라지면 결함을 덮는 표시가 되어 버린다.
        bad = [r for r in rows if r[0] not in ("OK", "SKIP")]
        for kind, _, _ in rows:
            tally[kind] = tally.get(kind, 0) + 1
        print(f"{'OK  ' if not bad else 'BAD '}{p}  ({len(rows)} 블록)", flush=True)
        for kind, ln, msg in rows:
            if kind != "OK":
                print(f"      {kind:7s}{ln:5d}  {msg}", flush=True)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
