#!/usr/bin/env python3
r"""본문의 「N.M절 [쪽](경로)」 참조에서 절 번호가 거짓인 곳을 찾는다.

링크는 맞고 번호만 틀린 자리는 `mkdocs build --strict` 가 잡지 못한다.
눌러 보면 제자리에 가고 빌드도 통과하므로, 틀린 것은 독자가 읽는 번호뿐이다.
장·절을 옮긴 뒤 본문 참조가 따라가지 않으면 이렇게 남는다.

진실의 원천은 `mkdocs.yml` 의 nav 다 — 쪽이 어느 절에 놓였는지는 거기에만 있다.

쓰기:  python3 tools/lint_section_refs.py [chNN ...]
종료코드: 틀린 참조가 있으면 1

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


def main(argv: list) -> int:
    sec = nav_sections(ROOT / "mkdocs.yml")
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
