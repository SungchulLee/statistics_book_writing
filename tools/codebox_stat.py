"""상자에 들어가지 않은 파이썬 코드블록이 장마다 얼마나 남았는지 센다.

    python3 tools/codebox_stat.py            장별 요약
    python3 tools/codebox_stat.py chNN       그 장의 문서별 내역

코드 시연은 파랑 예제 상자(`<div class="codebox">`)에 담는 것이 이 책의
규칙이다. 상자 밖에 남아 있는 블록이 아직 손대지 않은 것이다.

풀이·증명(`??? success`, `??? proof`) 안의 코드는 문항에 딸린 것이므로
세지 않는다. 들여쓴 펜스가 그에 해당한다.
"""
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "docs"
FENCE = re.compile(r"^```(\w*)")


def bare_blocks(path):
    """상자 밖에 있는 파이썬 코드블록의 (시작줄, 줄 수) 목록."""
    lines = path.read_text(encoding="utf-8").split("\n")
    out = []
    depth = 0
    i = 0
    while i < len(lines):
        x = lines[i]
        m = FENCE.match(x)          # 들여쓰지 않은 펜스만 본다
        if m:
            j = i + 1
            while j < len(lines) and not FENCE.match(lines[j]):
                j += 1
            if m.group(1) in ("python", "py") and depth == 0:
                out.append((i + 1, j - i - 1))
            i = j + 1
            continue
        if x.startswith("<div class="):
            depth += 1
        elif x.strip() == "</div>":
            depth = max(0, depth - 1)
        i += 1
    return out


def main(argv):
    only = argv[0] if argv else None
    per = collections.Counter()
    rows = []
    for p in sorted(ROOT.rglob("*.md")):
        rel = str(p.relative_to(ROOT))
        ch = rel.split("/")[0]
        if only and ch != only:
            continue
        n = len(bare_blocks(p))
        if n:
            per[ch] += n
            rows.append((n, rel))
    if only:
        for n, rel in sorted(rows, reverse=True):
            print(f"{n:4d}  {rel}")
    else:
        for ch in sorted(per):
            print(f"{ch:10s}{per[ch]:5d}")
    print(f"{'합계':10s}{sum(per.values()):5d}  상자 밖 파이썬 블록")


if __name__ == "__main__":
    main(sys.argv[1:])
