#!/usr/bin/env python3
r"""본문의 「N.M절 [쪽](경로)」 참조에서 절 번호가 거짓인 곳을 찾는다.

링크는 맞고 번호만 틀린 자리는 `mkdocs build --strict` 가 잡지 못한다.
눌러 보면 제자리에 가고 빌드도 통과하므로, 틀린 것은 독자가 읽는 번호뿐이다.
장·절을 옮긴 뒤 본문 참조가 따라가지 않으면 이렇게 남는다.

진실의 원천은 `mkdocs.yml` 의 nav 다 — 쪽이 어느 절에 놓였는지는 거기에만 있다.

쓰기:  python3 tools/lint_section_refs.py [chNN ...]
종료코드: 틀린 참조가 있으면 1

꼴은 다섯이다. A~D 는 **다른 쪽**을 가리키는 참조고(번호 뒤에 링크나 쪽 이름이
따라온다), E 는 **같은 쪽 안**의 절 참조다(「§6」·「4절의 …」 — 뒤에 아무 이름도
따라오지 않는다). A~D 가 모두 이름이나 링크를 요구하므로 E 는 넷 다 놓친다.

거짓 양성 둘을 걸러낸다. 걸러내지 않으면 멀쩡한 곳을 고치게 된다.

1. **절이 주어인 꼴.** 「4.3절은 특히 [결합분포](../ch03/rv/joint.md)를 전제한다」는
   링크를 4.3 절이라 주장하는 것이 아니다. 「절」 바로 뒤에 조사가 오면 건너뛴다.
2. **번호가 링크 뒤 괄호에 오는 꼴.** 「[신뢰구간](ch08/...)(8.2절)과
   [검정](ch09/...)(9.2절)」에서 「(8.2절)과 [검정](ch09/...)」을 한 짝으로 읽으면
   멀쩡한 8.2 를 9.2 로 바꾸게 된다. 링크가 번호 **앞**에 오면 건너뛴다.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 「절」 뒤에 이 조사가 오면 절이 주어·목적어이고 뒤의 링크를 가리키는 것이 아니다
JOSA = re.compile(r"^\s*(은|는|이|가|을|를|도|만|까지|부터|에서|에는|에서는|의|과|와)")

# 꼴 A: 15.4절 [유도와 카이제곱 근사](derivation.md)
#       번호와 링크가 줄바꿈으로 나뉜 것도 받는다(본문을 줄여 쓰면 그렇게 된다)
A = re.compile(r"(\d+\.\d+)\s*절([^\[\n]{0,6}\n?[^\[\n]{0,6})\[[^\]]*\]\(([^)\s]+\.md)")
# 꼴 B: [13.5절 보기 1](../assumptions/checking_normality.md)
B = re.compile(r"\[[^\]]{0,24}?(\d+\.\d+)\s*절[^\]]*\]\(([^)\s]+\.md)")
# 꼴 C: [비율의 신뢰구간](../../ch08/one_sample_intervals/ci_p.md)(8.2절)
#       [평균의 신뢰구간](../../ch08/one_sample_intervals/ci_mu.md)은 8.2절에서
C = re.compile(r"\[[^\]]*\]\(([^)\s]+\.md)[^)]*\)\s*(?:\(\s*|(?:은|는|이|가)\s*)(\d+\.\d+)\s*절")

# 꼴 E: **같은 쪽 안**의 절 참조 — 「§6」·「$\S 6$」·「4절의 …」.
#       쪽 안에서 닫히므로 nav 지도가 필요 없다. 그 쪽의 `## N.` 머리글과 맞춘다.
#       꼴 A~D 는 모두 **번호 뒤에 이름이나 링크**를 요구하는데, 이 꼴은 둘 다 없다
#       (「4절의 Cornish–Fisher 전개로」). 그래서 넷 다 놓치고 `--strict` 도 볼 것이
#       없다. 쪽을 합치거나 절을 덜어 내면 똑같이 틀어지는데 틀어진 번호도 *존재*하는
#       일이 많아 「없는 번호」 검사에도 걸리지 않는다. 실제로 `financial_context` 가
#       §4 를 세 번 가리키고 있었고 그 쪽은 §3 까지다(내가 14.1↔14.7 을 합친 자국이다).
E = re.compile(r"(?<![\d.])(?:\$\\S *(\d{1,2})\$|§ *(\d{1,2})|(\d{1,2}) ?절)")
# 「2장 5절」 은 장을 넘는 참조다. 「1~3절」 은 범위이고 끝 번호만 걸린다
E_CHAP = re.compile(r"\d{1,2} ?장[^.]{0,3}$")
E_RANGE = re.compile(r"[~–-]\s*$")
# 「[쪽 이름](path.md) §6 의」 — 링크가 앞에 있으면 그 §6 은 **그 쪽**의 절이다
E_OTHER = re.compile(r"\.md\)[^(]{0,24}$")
# 「절반」·「절차」 처럼 「절」 로 시작하는 다른 낱말
E_WORD = re.compile(r"절[반차대감망정충]")

# 꼴 E 의 참 예외 — 책의 절이 아니라 **바깥 문헌의 절**을 가리키는 자리.
# 기계로 가릴 수 없으므로 판정을 적어 둔다(까닭 없이 늘리지 말 것).
E_ALLOW = {
    ("ch04/discrete_distributions/poisson.md", 12):
        "보르트키에비치 『작은 수의 법칙』(1898) 원서의 §12 다 — 이 쪽의 절이 아니다",
}

# 꼴 D: 링크가 **없이** 번호 뒤에 쪽 이름이나 절 이름이 따라오는 경우.
#       「5.3절 **비율의 표집분포**에서 다룬다」 — 링크가 없으니 A·B·C 가 모두 놓친다.
#       nav 의 절 제목과 쪽 제목, 그리고 각 쪽의 H1 을 모아 이름으로 맞춘다.
D_ANY = re.compile(r"(\d+\.\d+)\s*절")
# 번호가 링크에 덮여 있는 구간은 제외한다(A·B 가 이미 본다)
D_COV = (re.compile(r"\d+\.\d+\s*절[^\[\n]{0,8}\n?[^\[\n]{0,8}\[[^\]]*\]\([^)]+\)"),
         re.compile(r"\[[^\]]*\d+\.\d+\s*절[^\]]*\]\([^)]+\)"))


def nav_sections(yml: pathlib.Path) -> dict:
    """docs 상대경로 -> 절 번호."""
    out, cur = {}, None
    for line in yml.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\s*- (\d+\.\d+) ", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r":\s*(\S+\.md)\s*$", line)
        if m and cur:
            out[m.group(1)] = cur
    return out


def nav_titles(yml: pathlib.Path) -> dict:
    """제목(절 제목 · 쪽 제목 · 각 쪽의 H1) -> 그 제목이 속한 절 번호의 집합."""
    out, cur = {}, None
    for line in yml.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\s*- (\d+\.\d+) (.+?):\s*$", line)
        if m:
            cur = m.group(1)
            out.setdefault(m.group(2).strip().strip('"'), set()).add(cur)
            continue
        m = re.match(r"^\s*- (.+?):\s*(\S+\.md)\s*$", line)
        if m and cur:
            out.setdefault(m.group(1).strip().strip('"'), set()).add(cur)
            f = DOCS / m.group(2)
            if f.exists():
                h1 = next((x[2:].strip() for x in f.read_text(encoding="utf-8").split("\n")
                           if x.startswith("# ")), None)
                if h1:
                    out.setdefault(h1, set()).add(cur)
    return out


def main(argv: list) -> int:
    sec = nav_sections(ROOT / "mkdocs.yml")
    titles = nav_titles(ROOT / "mkdocs.yml")
    # 짧은 제목은 우연히 걸리므로 네 글자부터, 긴 것부터 맞춘다
    title_keys = sorted((x for x in titles if len(x) >= 4), key=len, reverse=True)
    chapters = [a for a in argv if re.fullmatch(r"ch\d+", a)]
    bad = []
    for p in sorted(DOCS.rglob("*.md")):
        if p.name.endswith("_score.md"):
            continue
        rel = p.relative_to(DOCS)
        if chapters and rel.parts[0] not in chapters:
            continue
        text = p.read_text(encoding="utf-8")
        for pat, form in ((A, "A"), (B, "B"), (C, "C")):
            for m in pat.finditer(text):
                if form == "C":
                    link, claim = m.group(1), m.group(2)
                elif form == "A":
                    claim, gap, link = m.group(1), m.group(2), m.group(3)
                    if JOSA.match(gap):
                        continue
                    # 「…](경로)(8.2절)과 [다음](…)」— 번호가 괄호에 싸여 있으면
                    # 그것은 앞 링크의 꼬리표이고 뒤 링크의 번호가 아니다
                    if gap.lstrip().startswith(")"):
                        continue
                else:
                    claim, link = m.group(1), m.group(2)
                try:
                    key = str((p.parent / link).resolve().relative_to(DOCS))
                except ValueError:
                    continue
                real = sec.get(key)
                if real and real != claim:
                    bad.append((str(rel), text[:m.start()].count("\n") + 1,
                                claim, real, key))
        # ── 꼴 D: 링크 없이 이름만 따라오는 참조를 제목으로 맞춘다
        covered = set()
        for pat in D_COV:
            for m in pat.finditer(text):
                covered.update(range(m.start(), m.end()))
        for m in D_ANY.finditer(text):
            if m.start() in covered:
                continue
            tail = re.sub(r"^[\s「『*`]+", "", text[m.end():m.end() + 44].replace("\n", " "))
            hit = next((x for x in title_keys if tail.startswith(x)), None)
            if hit is None:
                continue
            claim = m.group(1)
            if claim not in titles[hit]:
                bad.append((str(rel), text[:m.start()].count("\n") + 1,
                            claim, "·".join(sorted(titles[hit])), f"이름: {hit}"))
        # ── 꼴 E: 같은 쪽 안의 절 참조를 그 쪽의 `## N.` 머리글과 맞춘다
        heads = {int(x) for x in re.findall(r"^## (\d{1,2})\. ", text, re.M)}
        if heads:
            for m in E.finditer(text):
                num = int(next(g for g in m.groups() if g))
                pre = text[max(0, m.start() - 40):m.start()]
                if E_CHAP.search(pre) or E_RANGE.search(pre) or E_OTHER.search(pre):
                    continue
                if E_WORD.match(text[m.end() - 1:m.end() + 2]):
                    continue
                if E_ALLOW.get((str(rel), num)) or num in heads:
                    continue
                bad.append((str(rel), text[:m.start()].count("\n") + 1,
                            str(num), "·".join(str(h) for h in sorted(heads)),
                            f"같은 쪽: {m.group(0).strip()}"))
    seen, rows = set(), []
    for r in bad:
        if (r[0], r[1]) not in seen:
            seen.add((r[0], r[1]))
            rows.append(r)
    for f, line, claim, real, key in sorted(rows):
        print(f"  {f}:{line}  「{claim}절」 -> 실제 {real}   ({key})")
    print(f"절 번호가 거짓인 참조 {len(rows)} 곳")
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
