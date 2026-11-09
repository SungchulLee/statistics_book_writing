r"""`tools/terms.json` 으로 `docs/glossary.md` 를 다시 쓴다.

    python3 tools/gen_glossary.py            용어집을 다시 쓴다
    python3 tools/gen_glossary.py --check    달라지는 것이 있으면 1 로 끝낸다 (CI 용)

용어집은 **손으로 고치지 않는다.** 약어를 더하거나 번역을 고칠 일이 있으면
`tools/terms.json` 을 고치고 이 명령을 돌린다. 쪽과 사전이 갈리면 검사기
(`lint_term_glosses.py`)가 사전을 믿으므로, 손으로 고친 용어집은 다음 실행에서
조용히 덮어써진다.

약어 칸의 `<span id="...">` 가 본문이 거는 링크의 과녁이다. 머리글이 아니라
표의 칸이라 `toc` 가 앵커를 만들어 주지 않으므로 직접 박는다 — 상자에 `id` 를
주는 것과 같은 까닭이다(CLAUDE.md 의 「출처 줄」).
"""
import json
import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parent.parent
D = R / "docs"
OUT = D / "glossary.md"
NAV = re.compile(r'^\s*- "?(.+?)"?:\s*((?:ch|index)[\w/.\-]*\.md)\s*$')


def nav_titles():
    t = {}
    for line in (R / "mkdocs.yml").read_text(encoding="utf-8").split("\n"):
        m = NAV.match(line)
        if m:
            t.setdefault(m.group(2), m.group(1).strip("'\""))
    return t


def main(argv):
    terms = json.loads((R / "tools" / "terms.json").read_text(encoding="utf-8"))
    titles = nav_titles()
    L = []
    L.append("# 용어집")
    L.append("")
    L.append("이 책이 쓰는 약어와 전문 용어를 영어와 맞대어 적었다. 본문에서 약어가 처음")
    L.append("나오는 자리는 모두 이 표로 링크되어 있으므로, 낯선 약어를 만나면 눌러서")
    L.append("돌아오면 된다. 영어를 함께 적는 까닭은 이 책 밖의 문헌이 거의 다 영어이기")
    L.append("때문이다 — 「가족단위 오류율」로 검색해서 나오지 않는 것이 `familywise error")
    L.append("rate` 로는 나온다.")
    L.append("")
    L.append("## 1. 약어")
    L.append("")
    L.append("| 약어 | 우리말 | 영어 | 설명하는 쪽 |")
    L.append("|---|---|---|---|")
    for ab in sorted(terms["abbr"]):
        e = terms["abbr"][ab]
        tgt = e.get("def")
        if tgt and (D / tgt).exists():
            where = f"[{titles.get(tgt, tgt)}]({tgt})"
        else:
            where = "—"
        mark = "" if e.get("auto", True) else " ※"
        L.append(f'| <span id="{ab.lower()}"></span>**{ab}**{mark} | {e["ko"]} '
                 f'| {e["en"]} | {where} |')
    L.append("")
    L.append("※ 표시는 이 책 안에서 **뜻이 둘 이상인** 약어다. CV 는 변동계수이기도 하고")
    L.append("교차검증이기도 하며, VC 는 VC 차원이기도 하고 ToothGrowth 자료의 비타민 C")
    L.append("투여 방법이기도 하다. 그래서 본문에서 자동으로 링크를 걸지 않는다.")
    L.append("")
    L.append("## 2. 우리말–영어 대조표")
    L.append("")
    L.append("본문은 이 표의 용어가 **쪽마다 처음 나오는 자리**에 영어를 괄호로 병기한다.")
    L.append("")
    L.append("| 우리말 | 영어 |")
    L.append("|---|---|")
    for ko in sorted(terms["terms"]):
        L.append(f"| {ko} | {terms['terms'][ko]} |")
    L.append("")
    L.append("## 3. 기초어")
    L.append("")
    L.append("아래는 **병기하지 않는** 말이다. 둘 가운데 하나다 — 책 거의 모든 쪽에 나와")
    L.append("쪽마다 괄호를 달면 글이 괄호에 잠기거나(「분산」은 $474$ 쪽에 있다),")
    L.append("일상어와 겹쳐 기계가 뜻을 가릴 수 없거나(「대비」는 contrast 이기도 하고")
    L.append("「~에 대비하여」이기도 하다). 영어가 궁금하면 여기서 찾는다.")
    L.append("")
    L.append("| 우리말 | 영어 |")
    L.append("|---|---|")
    for ko in sorted(terms["basic"]):
        L.append(f"| {ko} | {terms['basic'][ko]} |")
    L.append("")
    new = "\n".join(L)
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if "--check" in argv:
        print("용어집이 사전과 " + ("어긋난다" if new != old else "맞는다"))
        return 1 if new != old else 0
    OUT.write_text(new, encoding="utf-8")
    print(f"{OUT.relative_to(R)} — 약어 {len(terms['abbr'])} · "
          f"용어 {len(terms['terms'])} · 기초어 {len(terms['basic'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
