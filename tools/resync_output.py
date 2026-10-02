"""한 쪽의 python 블록을 돌려 바로 아래 출력 블록을 실제 출력으로 갈아 준다.

    python3 resync_output.py <md> <python 울타리의 줄번호> [<줄번호> ...]

줄번호는 ```python 울타리가 있는 줄(1 기준)이다. 여러 개를 주면 **큰 번호부터**
처리해 앞쪽 줄번호가 밀리지 않게 한다.

앞 블록의 변수를 이어 쓰는 쪽이 있으므로, 대상 블록 앞의 python 블록을
출력을 삼킨 채 먼저 실행한다. 들여쓴 울타리는 울타리의 들여쓰기만큼만 벗긴다.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _p
_p.show = lambda *a, **k: None
_g = {"__name__": "__main__"}
_pre = open(sys.argv[1], encoding="utf-8").read()
_tgt = open(sys.argv[2], encoding="utf-8").read()
if _pre.strip():
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(_pre, "pre", "exec"), _g)
        except Exception:
            pass
exec(compile(_tgt, "tgt", "exec"), _g)
'''


def main(argv):
    path = pathlib.Path(argv[0])
    starts = sorted((int(a) for a in argv[1:]), reverse=True)
    lines = path.read_text(encoding="utf-8").split("\n")

    for start in starts:
        i = start - 1
        m = FENCE.match(lines[i])
        assert m and m.group(2) in ("python", "py"), f"{start} 행이 python 울타리가 아니다: {lines[i]!r}"
        ind = len(m.group(1))

        # 대상 블록 본문
        j = i + 1
        body = []
        while not (FENCE.match(lines[j]) and FENCE.match(lines[j]).group(2) == ""):
            body.append(lines[j][ind:] if lines[j].startswith(" " * ind) else lines[j].lstrip())
            j += 1
        code = "\n".join(body)

        # 그 앞의 python 블록들을 prefix 로 모은다
        pre = []
        k = 0
        while k < i:
            mm = FENCE.match(lines[k])
            if mm:
                ind2 = len(mm.group(1))
                e = k + 1
                b2 = []
                while e < len(lines) and not (FENCE.match(lines[e]) and FENCE.match(lines[e]).group(2) == ""):
                    b2.append(lines[e][ind2:] if lines[e].startswith(" " * ind2) else lines[e].lstrip())
                    e += 1
                if mm.group(2) in ("python", "py"):
                    pre.append("\n".join(b2))
                k = e + 1
                continue
            k += 1

        # 다음 출력 블록의 범위
        o = j + 1
        while not FENCE.match(lines[o]):
            o += 1
        o_open = o
        o_body_start = o + 1
        o_body_end = o_body_start
        while not FENCE.match(lines[o_body_end]):
            o_body_end += 1
        out_ind = len(FENCE.match(lines[o_open]).group(1))

        with tempfile.TemporaryDirectory() as d:
            pf = pathlib.Path(d, "p.py"); pf.write_text("\n".join(pre), encoding="utf-8")
            tf = pathlib.Path(d, "t.py"); tf.write_text(code, encoding="utf-8")
            rf = pathlib.Path(d, "r.py"); rf.write_text(RUNNER, encoding="utf-8")
            r = subprocess.run([sys.executable, str(rf), str(pf), str(tf)],
                               capture_output=True, text=True, timeout=1800,
                               cwd=str(pathlib.Path.cwd()))
        assert r.returncode == 0, f"{start} 행 블록 실행 실패:\n{r.stderr[-600:]}"

        actual = [(" " * out_ind + x).rstrip() for x in r.stdout.strip("\n").split("\n")]
        old = lines[o_body_start:o_body_end]
        if old == actual:
            print(f"  {start} 행: 이미 일치")
            continue
        print(f"  {start} 행: 출력 {len(old)} 줄 -> {len(actual)} 줄")
        for a, b in zip(old, actual):
            if a != b:
                print(f"      쪽  |{a}|")
                print(f"      코드|{b}|")
        if len(actual) > len(old):
            for b in actual[len(old):]:
                print(f"      코드에만 있던 줄|{b}|")
        elif len(old) > len(actual):
            for a in old[len(actual):]:
                print(f"      쪽에만 있던 줄|{a}|")
        lines[o_body_start:o_body_end] = actual

    path.write_text("\n".join(lines), encoding="utf-8")
    print("저장했다.")


if __name__ == "__main__":
    main(sys.argv[1:])
