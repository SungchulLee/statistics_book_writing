"""보기·문제·연습문제에 난이도 점을 붙인다.

    python3 tools/diffmark.py dump <장>              판단할 블록을 훑어 낸다
    python3 tools/diffmark.py apply <장> <배정파일>   난이도를 적용한다
    python3 tools/diffmark.py relabel                 표시 문구만 다시 쓴다
    python3 tools/diffmark.py stat [장]               현재 분포를 센다

배정파일은 한 줄에 `<번호> <e|m|h>` 로 적는다. dump 가 매긴 번호와 같은
순서이며, 번호는 (문서, 그 문서 안 몇 번째 블록인지)로 정해져 편집에 안 밀린다.

붙는 모양:

    **연습문제 3.** <span class="diff med" title="중간"></span>
    본문…

점의 **색과 크기와 모양**은 이 파일이 아니라 `docs/stylesheets/extra.css` 의
`:root` 에 있는 `--diff-*` 값으로 정해진다. 거기만 고치면 3,000개 넘는 문서를
건드리지 않고 한꺼번에 바뀐다. 이 파일이 정하는 것은 **어느 문항이 어느
등급인가**와 **말풍선에 뜨는 문구**뿐이다.
"""
import pathlib
import re
import sys

# ── 바꿀 수 있는 값 ────────────────────────────────────────────────────
# LEVEL 의 문구를 고친 뒤 `relabel` 을 돌리면 이미 붙어 있는 표시도 따라 바뀐다.
# 클래스 이름(easy/med/hard)을 바꾸려면 extra.css 의 `.diff.easy` 들도 함께 고쳐야 한다.

LEVEL = {
    "e": ("easy", "쉬움"),
    "m": ("med", "중간"),
    "h": ("hard", "어려움"),
}

# 난이도를 붙일 블록 — (CSS 클래스, 도입어)
KINDS = [("exbox", "보기"), ("probox", "문제"), ("drillbox", "연습문제")]

# ───────────────────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).resolve().parent.parent / "docs"

CLASSES = "|".join(cls for cls, _ in LEVEL.values())
SPAN = re.compile(rf'\s*<span class="diff (?:{CLASSES})"[^>]*></span>')


def blocks(chapter):
    """(경로, 줄번호, 종류, 도입어줄, 본문미리보기) 를 문서 순서대로 낸다."""
    base = ROOT / chapter
    for p in sorted(base.rglob("*.md")) if base.is_dir() else []:
        lines = p.read_text(encoding="utf-8").split("\n")
        for i, l in enumerate(lines):
            for cls, word in KINDS:
                if not l.startswith(f'<div class="{cls}"'):
                    continue
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines) or not lines[j].lstrip().startswith(f"**{word}"):
                    continue
                # 본문 미리보기: 도입어 줄부터 상자가 닫힐 때까지
                k = j
                buf = []
                while k < len(lines) and lines[k].strip() != "</div>":
                    if lines[k].strip():
                        buf.append(lines[k].strip())
                    k += 1
                yield p, j, word, lines[j], " ".join(buf)


def cmd_dump(chapter, width=150):
    for n, (p, j, word, lead, body) in enumerate(blocks(chapter), 1):
        body = SPAN.sub("", body)
        body = re.sub(r"^\*\*[^*]+\*\*\s*", "", body)
        print(f"{n:4d} {word:<5} {str(p.relative_to(ROOT))}")
        print(f"     {body[:width]}")


def cmd_apply(chapter, path):
    want = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").split("\n"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        a, b = line.split()
        want[int(a)] = LEVEL[b.strip().lower()[0]]

    edits = {}
    for n, (p, j, word, lead, body) in enumerate(blocks(chapter), 1):
        if n not in want:
            continue
        cls, title = want[n]
        base = SPAN.sub("", lead).rstrip()
        m = re.match(r"^(?P<lead>\s*\*\*[^*]+\*\*)(?P<rest>.*)$", base)
        if not m:
            continue
        new = (f'{m.group("lead")} <span class="diff {cls}" title="{title}"></span>'
               f'{m.group("rest")}')
        edits.setdefault(p, {})[j] = new

    total = 0
    for p, per in edits.items():
        lines = p.read_text(encoding="utf-8").split("\n")
        for j, new in per.items():
            lines[j] = new
            total += 1
        p.write_text("\n".join(lines), encoding="utf-8")
    print(f"{total}건 적용")


def cmd_relabel():
    """이미 붙어 있는 표시의 문구를 LEVEL 의 현재 값으로 다시 쓴다.

    난이도 판정은 그대로 두고 말풍선 문구만 바꾼다. LEVEL 을 고친 뒤 한 번
    돌리면 책 전체가 따라온다.
    """
    byclass = {cls: title for cls, title in LEVEL.values()}
    pat = re.compile(r'<span class="diff (?P<c>\w+)" title="[^"]*"></span>')
    total = files = 0

    def sub(m):
        nonlocal total
        t = byclass.get(m.group("c"))
        if t is None:
            return m.group(0)
        new = f'<span class="diff {m.group("c")}" title="{t}"></span>'
        if new != m.group(0):
            total += 1
        return new

    for p in sorted(ROOT.rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        if 'class="diff ' not in text:
            continue
        new = pat.sub(sub, text)
        if new != text:
            p.write_text(new, encoding="utf-8")
            files += 1
    print(f"{total}건 문구 변경 / {files}개 문서")


def cmd_stat(chapter=None):
    import collections
    c = collections.Counter()
    per = collections.Counter()
    chapters = [chapter] if chapter else sorted(
        d.name for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("ch"))
    for ch in chapters:
        for p, j, word, lead, body in blocks(ch):
            m = re.search(r'class="diff (\w+)"', lead)
            c[m.group(1) if m else "없음"] += 1
            per[ch] += 0 if m else 1
    tot = sum(c.values())
    for k in ("easy", "med", "hard", "없음"):
        if c[k]:
            print(f"  {k:<5} {c[k]:5d}  ({c[k]/tot*100:.0f}%)")
    if not chapter:
        left = {k: v for k, v in per.items() if v}
        print("남은 장:", ", ".join(f"{k}({v})" for k, v in sorted(left.items())) or "없음")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dump":
        cmd_dump(sys.argv[2])
    elif cmd == "apply":
        cmd_apply(sys.argv[2], sys.argv[3])
    elif cmd == "relabel":
        cmd_relabel()
    elif cmd == "stat":
        cmd_stat(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
