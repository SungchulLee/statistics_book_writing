#!/usr/bin/env python3
"""장 개요가 nav 의 쪽을 빠뜨린 자리를 찾는다.

`lint_index_links` 는 장 개요에 **적힌** 쪽 이름이 링크인지를 본다.
이 도구는 반대 방향이다 — nav 에 있는데 **개요가 아예 소개하지 않는** 쪽을 센다.
빌드는 할 말이 없다(nav 에 있으니 닿을 수 있다). 그런데 독자가 무엇을 읽을지
고르는 자리는 장 개요의 「장의 구성」이고, 거기 없는 쪽은 존재를 알 수 없다.

**꼴이 둘이라는 것이 이 검사의 핵심이다.** 어떤 장은 장의 구성을 쪽마다 한 줄씩
불릿으로 쓰고(ch01~ch04 등), 어떤 장은 **절마다 산문 한 단락**으로 쓴다(ch05 전체,
ch06 의 6.2·6.3, ch07 의 7.3). 산문 꼴에서는 쪽 링크가 없는 것이 정상이므로
그것까지 세면 82 가 나오고 그 가운데 54 가 거짓이다.

그래서 판정을 **절 단위**로 한다 — 그 절의 **다른 쪽이 개요에 링크로 있으면**
그 절은 불릿 꼴이고, 거기서 빠진 쪽은 진짜 누락이다.

    python3 tools/lint_index_coverage.py [chNN ...]
"""
import os
import re
import sys
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nav_sections():
    """nav 를 읽어 {쪽경로: 절제목} 과 {절제목: [쪽경로]} 를 돌려준다."""
    text = open(os.path.join(ROOT, "mkdocs.yml")).read().split("nav:", 1)[1]
    sec = None
    secof = {}
    insec = collections.defaultdict(list)
    for line in text.splitlines():
        m = re.match(r"^(\s+)- ([^:]+):\s*$", line)
        if m:
            title = m.group(2).strip()
            # 「12.3 사후비교」·「A.2 계산 도구」 꼴만 절로 센다
            if re.match(r"(\d+\.\d+|[A-Z]\.\d+) ", title):
                sec = title
        m2 = re.match(r"^\s+- (.+?):\s*(\S+\.md)\s*$", line)
        if m2 and sec:
            page = m2.group(2).strip()
            if page.endswith("/index.md"):
                continue            # 장 개요 자신은 세지 않는다
            secof[page] = sec
            insec[sec].append(page)
    return secof, insec


def linked_from(index_path, chapter):
    """장 개요가 링크한 쪽들을 docs/ 기준 경로의 집합으로."""
    out = set()
    try:
        s = open(index_path).read()
    except OSError:
        return out
    for m in re.finditer(r"\(([^)\s]+\.md)(?:#[^)]*)?\)", s):
        p = os.path.normpath(os.path.join(f"docs/{chapter}", m.group(1)))
        out.add(p.replace("docs/", "", 1))
    return out


def main(argv):
    want = tuple(argv) if argv else None
    secof, insec = nav_sections()
    chapters = sorted({p.split("/")[0] for p in secof})
    missing = 0
    prose = 0
    for ch in chapters:
        if want and ch not in want:
            continue
        idx = os.path.join(ROOT, "docs", ch, "index.md")
        if not os.path.exists(idx):
            continue
        linked = linked_from(idx, ch)
        gone = []
        for p in sorted(q for q in secof if q.startswith(ch + "/")):
            if p in linked:
                continue
            sibs = [q for q in insec[secof[p]] if q != p]
            if any(q in linked for q in sibs):
                gone.append((secof[p], p))      # 불릿 꼴 절에서 빠졌다
            else:
                prose += 1                       # 산문 꼴 절이거나 쪽이 하나뿐이다
        if gone:
            missing += len(gone)
            print(f"════ {ch}")
            for sec, p in gone:
                print(f"    [{sec}]  {p.split('/', 1)[1]}")
    print(f"\n장 개요가 소개하지 않는 쪽 — 불릿 꼴 절에서 빠진 것 {missing} · "
          f"산문 꼴이라 넘긴 것 {prose}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
