"""쪽의 python 블록에서 **없는 이름**을 돌리지 않고 찾는다.

    python3 tools/lint_undefined.py                 책 전체 (몇 초)
    python3 tools/lint_undefined.py docs/ch06
    python3 tools/lint_undefined.py docs/ch06/mle/mle_bias.md

`verify_outputs.py` 는 코드를 **돌려서** NameError 를 잡으므로 한 장에 20~25 분이
든다(ch10 열두 쪽에 25 분). 그런데 이름이 없는 것은 돌리지 않고도 안다. 그래서
이 검사를 **먼저** 돌린다 — 책 전체가 몇 초다.

`verify_outputs` 와 **같은 규칙으로** 블록을 이어 붙인다(한 쪽 = 한 프로세스,
앞 블록의 변수를 뒤 블록이 이어 쓴다). `# verify: noexec` 가 붙은 블록은 양쪽
도구가 모두 건너뛴다.

## 나오는 것을 어떻게 읽는가

세 갈래로 갈리고 **셋째만 결함이다.**

1. **인용** — 꾸러미 안쪽이나 지금 이야기하는 함수의 두어 줄을 옮긴 것. 그 코드의
   `X`, `Y` 는 옮겨 온 함수가 인자로 받는 것이라 이 자리에 있을 수가 없다.
2. **골격** — 독자가 자기 모형·자기 자료를 끼우는 것(`generate`, `fit`, `x`).
3. **참 결함** — 돌 생각으로 적었는데 import 가 빠진 것. 2026-10-23 에 둘 찾았다:
   `ch06/mle/mle_bias.md` 는 `np` 와 `rng` 를 묶지 않고 썼고,
   `ch06/mle/parametric_vs_nonparametric.md` 는 numpy 를 265 행에서야 들이면서
   `np` 를 139 행 **모듈 수준**에서 썼다. 둘 다 첫 블록이 적힌 대로 돌지 않았다.

1 과 2 는 `# verify: noexec — <까닭>` 을 달아 재우고, 3 은 고친다. 까닭은 독자에게도
쓸모가 있다 — `# 출력 없음: <까닭>` 과 같은 꼴이다.

## 두 가지 함정

**모듈 수준의 선-사용만 잡는다.** 함수 몸 안의 `np` 는 모듈 수준에 그 이름이
(나중에라도) 묶이면 pyflakes 가 넘어간다. 그래서 위 쪽에서 139 행은 걸리고
107 행은 걸리지 않았다. **신호가 적다고 깨끗한 것이 아니다.**

**없는 꾸러미는 못 잡는다.** `import jax` 나 `from global_name_space import ARGS`
는 이름이 있는 import 이므로 여기서는 조용하다. 그것은 돌려 봐야 나온다
(`verify_outputs`). 이 검사는 `verify_outputs` 를 **대신하지 않고 앞세운다.**

필요: pyflakes (`pip install pyflakes`). 문서 빌드에는 필요하지 않다.
"""
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_outputs import NOEXEC, blocks, preamble

try:
    from pyflakes.api import check as pyflakes_check
    from pyflakes.reporter import Reporter
except ImportError:
    sys.exit("pyflakes 가 없다.  pip install pyflakes")


def undefined(path):
    """[(쪽의 줄번호, 알림)] — 없는 이름, 그리고 구문오류.

    구문오류를 **반드시 함께** 알려야 한다. 이어 붙인 것이 파싱되지 않으면
    pyflakes 는 구문오류 하나만 내고 **그 쪽의 없는 이름을 하나도 보지 않는다.**
    인용한 토막 하나(`elif condno > 1000:` 같은 statsmodels 소스 한 줄)가 그
    쪽 전체의 검사를 가려 버린다. 2026-10-23 에 ch13 에서 그런 자리를 찾았다.
    """
    bs = [(ln, c) for lang, ln, c in blocks(path)
          if lang in ("python", "py") and not NOEXEC.search(c)]
    if not bs:
        return []
    buf, lmap = [], {}        # lmap: 이어 붙인 줄 -> 쪽의 줄
    for line in preamble(path).split("\n"):
        if line:
            buf.append(line)
            lmap[len(buf)] = bs[0][0]
    for ln, code in bs:
        for off, line in enumerate(code.split("\n")):
            buf.append(line)
            lmap[len(buf)] = ln + 1 + off      # 울타리 줄 다음부터가 코드다
    out, err = io.StringIO(), io.StringIO()
    pyflakes_check("\n".join(buf), str(path), Reporter(out, err))

    def at(parts):
        try:
            return lmap.get(int(parts[1]), 0)
        except (ValueError, IndexError):
            return 0

    hits = []
    # 구문오류는 error 쪽으로 나오고, 나면 없는 이름 검사가 **아예 돌지 않는다**
    for line in err.getvalue().split("\n"):
        if ".md:" not in line:
            continue
        parts = line.split(":", 3)
        hits.append((at(parts), f"{parts[-1].strip()}"
                                "  ← 이 쪽의 없는 이름은 검사되지 않았다"))
    for line in out.getvalue().split("\n"):
        if "undefined name" in line:
            parts = line.split(":", 3)
            hits.append((at(parts), parts[-1].strip()))
    return hits


def main(argv):
    t = pathlib.Path(argv[0] if argv else "docs")
    pages = sorted(t.rglob("*.md")) if t.is_dir() else [t]
    total = 0
    for p in pages:
        hits = undefined(p)
        if not hits:
            continue
        print(f"── {p}")
        for ln, msg in hits:
            print(f"     {ln:>6}  {msg}")
            total += 1
    print(f"\n없는 이름 {total} 곳"
          + ("" if total else "  (인용과 골격은 noexec 로 재워 두었다)"))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
