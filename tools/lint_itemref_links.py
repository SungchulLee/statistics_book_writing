#!/usr/bin/env python3
r"""다른 쪽의 항목을 **링크 없이** 절 번호로만 가리키는 자리를 찾는다.

규약은 「다른 쪽의 항목을 가리킬 때는 쪽을 링크로 적는다」이고 까닭은 **한 절에 쪽이
여럿이라 번호만으로는 가리키는 곳이 하나가 아니라는 것**이다. 그래서 이 검사의 판정은
「번호가 맞나 틀리나」가 아니라 **「링크가 있나 없나」**다 — 기계로 끝난다.

왜 따로 필요한가. `lint_section_refs.py` 의 꼴 A~E 는 **절 번호**를 보고
`lint_item_refs.py` 는 **링크가 붙은** 항목 참조를 본다. 번호만 있고 링크가 없으면
둘 다 볼 것이 없는데, 이 꼴이 가장 오래 숨는다 — 2026-10-23 에 ch15 를 보던 세션이
「11.5절 보기 3」 셋을 찾았고 **11.5 는 쪽이 다섯이고 그중 넷에 보기 3 이 있었다.**
절 번호도 존재하고 항목 번호도 (네 쪽에) 존재하는데 가리키는 내용이 어디에도 없었다.
그 가운데 하나는 자기만 갖고 있는 수 둘($W = 2.0636$, $0.6888$)을 다른 쪽이 재어
두었다고 했다. 번호 검증을 전부 통과해도 남는 꼴이다.

CLAUDE.md 의 grep 은 이 가운데 **`연습문제` 하나만** 찾는다. 보기·정의·정리·그림·
상자를 가리키는 자리는 그 명령으로 0 이 나온다 — 「같은 계열의 다른 표기」 함정이다.

쓰기:  python3 tools/lint_itemref_links.py [chNN ...]
종료코드: 링크 없는 항목 참조가 있으면 1
"""
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

ITEM = "보기|연습문제|정의|정리|그림|상자"
# 「N.M절 …(사이에 쪽 이름이 올 수 있다)… 보기 3」, 단 「절」 바로 뒤에 링크는 없다
REF = re.compile(rf"(\d+\.\d+)\s*절\s*(?!\[)([^\n]{{0,24}}?)({ITEM})\s*(\d+)")
LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
# 문장이 끊기면 뒤의 항목 번호는 **이 쪽의** 것이다.
#   「…필요하다(19.3절 모형 평가). 연습문제 9에서 수치로 확인한다」
# 이 하나를 걸러내지 않으면 멀쩡한 자리를 링크로 바꾸게 된다.
BREAK = re.compile(r"[.)·,;] |\)\s*$|다\. ")


def main(argv: list) -> int:
    chapters = [a for a in argv if re.fullmatch(r"ch\d+", a)]
    rows, by_ch = [], Counter()
    for p in sorted(DOCS.rglob("*.md")):
        if p.name.endswith("_score.md"):
            continue
        rel = p.relative_to(DOCS)
        if chapters and rel.parts[0] not in chapters:
            continue
        text = p.read_text(encoding="utf-8")
        for m in REF.finditer(text):
            gap = m.group(2)
            if LINK.search(gap) or BREAK.search(gap):
                continue
            by_ch[rel.parts[0]] += 1
            rows.append((str(rel), text[:m.start()].count("\n") + 1,
                         re.sub(r"\s+", " ", m.group(0))))
    for f, line, txt in rows:
        print(f"  {f}:{line}  「{txt}」")
    if by_ch:
        print("  " + " · ".join(f"{c} {n}" for c, n in sorted(by_ch.items())))
    print(f"링크 없이 절 번호로만 가리킨 항목 참조 {len(rows)} 곳")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
