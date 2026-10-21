"""찍는 python 블록인데 출력 블록이 없는 곳을 찾는다 — 돌리지 않고.

    python3 tools/lint_outputs.py            전체 요약 + 결함 목록
    python3 tools/lint_outputs.py ch06       그 장만

규칙은 「코드는 쪽 안에, 찍는 것이 있으면 출력도」다. 값이 맞는지는
tools/verify_outputs.py 가 돌려서 보고, 이 도구는 **출력 블록이 있는지만** 본다.
빌드는 둘 다 잡지 못한다.

세는 법에 함정이 셋 있어 그만큼을 걸러 낸다. 셋을 거르지 않으면 열셋이 잡히는데
그 가운데 여덟이 거짓이었다.

  1. **`def` 안의 `print`.** 정의만 하는 블록은 돌려도 아무것도 찍지 않는다.
     다음 블록이 그 함수를 부르고 그쪽에 출력이 붙는다. 최상위에 실행문이
     하나도 없으면 건너뛴다.
  2. **떨어져 있는 출력.** 문제 블록의 출력이 `??? success "풀이"` 안에 있는
     꼴이 있다. 닫는 울타리 뒤 여섯 줄만 보면 놓치므로 다음 코드 울타리를
     만날 때까지 훑는다.
  3. **주석으로 보인 값.** `print(a, b)   # 4.0  3.0` 처럼 한 줄짜리는 주석에
     값을 적어 둔다. 따로 세어 보고만 한다.

일부러 출력을 싣지 않은 블록은 두 가지뿐이어야 하고, 둘 다 쪽에 **까닭을
적어 두어야 한다.**

  - 독자가 자기 자료로 바꿔 끼우는 골격 (`x`, `generate`, `mle` 가 미정의).
    블록 안에 `# 출력 없음: <까닭>` 을 적는다. 독자에게도 알려 주는 줄이고
    이 도구는 그 줄이 있으면 넘어간다
  - 환경이 없어 확보하지 못한 출력 (`yfinance` 요청 한도, `pymc` 미설치).
    `!!! note` 로 사정과 맞춰 볼 값을 적고, 블록에도 같은 표시를 남긴다
"""
import pathlib
import re
import sys

FENCE = re.compile(r"^(\s*)```([\w-]*)\s*$")
# 최상위에서 '정의·수입·주석'이 아닌 실행문이 있는지 보는 데 쓴다
DEFONLY = re.compile(r"^(?:def\s|class\s|@|import\s|from\s|#|\s*$)")
# 일부러 출력을 싣지 않았다는 표시. 까닭을 반드시 뒤에 적게 했다
WAIVER = re.compile(r"#\s*출력 없음:\s*\S")
OUTLANG = ("", "text", "output", "console")


def blocks(lines):
    """(여는 울타리 줄번호, 들여쓰기, 본문줄, 닫는 울타리 줄번호, 언어)를 차례로 낸다."""
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        ind, lang = len(m.group(1)), m.group(2)
        j = i + 1
        while j < len(lines) and not (
            FENCE.match(lines[j]) and FENCE.match(lines[j]).group(2) == ""
        ):
            j += 1
        yield i, ind, lines[i + 1 : j], j, lang
        i = j + 1


def top_level_runs(body, ind):
    """최상위에 실행문이 하나라도 있는가. 없으면 정의만 하는 블록이다."""
    for line in body:
        s = line[ind:] if line.startswith(" " * ind) else line.lstrip()
        if s[:1] in (" ", "\t"):          # 들여쓴 줄 = 어떤 몸통 안
            continue
        if not DEFONLY.match(s):
            return True
    return False


def check(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    found = []
    for i, ind, body, j, lang in blocks(lines):
        if lang not in ("python", "py"):
            continue
        if not any("print(" in b for b in body):
            continue
        if any(WAIVER.search(b) for b in body):    # 쪽에 까닭을 적어 둔 블록
            continue
        if not top_level_runs(body, ind):      # 함정 1
            continue
        # 함정 2 — 다음 코드 울타리를 만날 때까지 출력 울타리를 찾는다
        kind, dist = None, None
        k = j + 1
        while k < len(lines) and k - j < 45:
            f = FENCE.match(lines[k])
            if f:
                if f.group(2) in OUTLANG:
                    kind, dist = "out", k - j
                break
            k += 1
        if kind == "out":
            found.append(("ok" if dist <= 6 else "far", i + 1, dist))
        elif any(re.search(r"print\(.*\)\s*#\s*\S", b) for b in body):
            found.append(("inline", i + 1, None))     # 함정 3
        else:
            found.append(("none", i + 1, None))
    return found


def main(argv):
    root = pathlib.Path("docs")
    pages = sorted(root.rglob("*.md")) if not argv else sorted(
        (root / argv[0]).rglob("*.md")
    )
    tally = {"ok": 0, "far": 0, "inline": 0, "none": 0}
    detail = {"far": [], "inline": [], "none": []}
    for p in pages:
        for kind, line, dist in check(p):
            tally[kind] += 1
            if kind != "ok":
                key = f"{p.relative_to(root)}:{line}"
                detail[kind].append(key + (f"  (+{dist}줄)" if dist else ""))

    print(f"출력 블록이 바로 아래      {tally['ok']}")
    print(f"출력이 떨어져 있다         {tally['far']}")
    print(f"주석으로 값을 보였다       {tally['inline']}")
    print(f"출력이 아예 없다           {tally['none']}")
    for kind, label in (
        ("none", "출력이 아예 없다 — 돌려서 채우거나, 까닭을 !!! note 로 적을 것"),
        ("far", "출력이 떨어져 있다 — 풀이 상자 안이면 정상"),
        ("inline", "주석으로 값을 보였다 — 한 줄짜리면 정상"),
    ):
        if detail[kind]:
            print(f"\n### {label}")
            for s in detail[kind]:
                print(f"    {s}")
    return 1 if tally["none"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
