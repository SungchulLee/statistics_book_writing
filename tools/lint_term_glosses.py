r"""용어·약어 병기 규약을 검사하고 채워 넣는다.

    python3 tools/lint_term_glosses.py                전체 검사
    python3 tools/lint_term_glosses.py ch10           그 장만
    python3 tools/lint_term_glosses.py ch10 --apply   그 장에 병기를 넣는다
    python3 tools/lint_term_glosses.py ch10 --show    넣을 자리를 줄째로 보여 준다

규약 둘(`tools/terms.json` 이 사전이다):

1. **한글 전문 용어** — 쪽마다 **처음 나오는 자리**에 영어를 괄호로 병기한다.
   `등분산성(homoscedasticity)` · `가족단위 오류율(familywise error rate)`.
   그 쪽 어딘가에 이미 병기가 있으면 건드리지 않는다. 사전의 `basic` 은
   병기하지 않는다 — 「분산」은 $474$ 쪽에 있고 「대비」는 일상어와 겹친다.
2. **영어 약어** — 풀네임은 **용어집과 그 개념을 설명하는 쪽**에서만 펼치고,
   다른 쪽에서는 약어가 처음 나오는 자리를 용어집으로 링크한다
   `[FWER](../../glossary.md#fwer)`.

검사가 보는 곳은 **본문뿐이다.** 펜스(코드·출력)·수식·머리글·표·링크·대체텍스트·
HTML 태그·출처 줄은 가린다. 가리지 않으면 세 가지가 깨진다 — 코드의 `MSE` 에
링크를 걸어 `NameError` 를 만들고, `$\text{FWER}$` 안에 괄호를 넣어 수식을 깨고,
머리글에 영어를 넣어 오른쪽 목차를 어지럽힌다(CLAUDE.md 의 「머리글과 대체텍스트」).

**조사를 보고 자른다.** 「등분산성이」의 「이」는 조사이므로 용어가 거기서 끝나지만
「분산분석표」의 「표」는 아니다. 그래서 용어 뒤에 한글이 오면 **조사 목록에 있을 때만**
받는다. 앞에 한글이 붙어 있으면(「표본분산」 안의 「분산」) 받지 않는다.
"""
import json
import os
import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parent.parent
D = R / "docs"
HAN = re.compile(r"[가-힣]")
# 긴 것부터 — 「이다」가 「이」로 잘리면 「등분산성이다」가 「등분산성이」+「다」가 된다
JOSA = sorted(
    """이다 입니다 이라 이란 이고 이며 이면 이지만 이나 이란 으로 로서 로써 에서 에게
    에는 에도 에만 와의 과의 보다 처럼 마다 까지 부터 조차 밖에 라면 라서 이든 이라서
    의 은 는 이 가 을 를 와 과 도 만 로 에 라 란 나 야 든""".split(),
    key=len, reverse=True)


def mask(text):
    """본문이 아닌 구간을 공백으로 지운 사본을 돌려준다 (자리는 그대로 둔다)."""
    out = list(text)

    def blank(a, b):
        for i in range(a, min(b, len(out))):
            if out[i] != "\n":
                out[i] = " "

    # 펜스 — 들여쓴 것도 센다 (쪽지 안의 코드블록)
    pos, infence = 0, False
    for line in text.split("\n"):
        n = len(line)
        if re.match(r"\s*(```|~~~)", line):
            infence = not infence
            blank(pos, pos + n)
        elif infence:
            blank(pos, pos + n)
        elif re.match(r"\s{0,3}#{1,6}\s", line):          # 머리글
            blank(pos, pos + n)
        elif re.match(r"\s*\|", line):                     # 표
            blank(pos, pos + n)
        elif re.match(r"\s*(!!!|\?\?\?)", line):           # 쪽지 머리
            blank(pos, pos + n)
        elif re.match(r"\s*_확인한 곳", line):             # 출처 줄
            blank(pos, pos + n)
        pos += n + 1

    s = "".join(out)
    for m in re.finditer(r"\$\$.*?\$\$", s, re.S):         # 블록 수식 — 먼저 가린다
        blank(m.start(), m.end())
    s = "".join(out)
    # 줄을 넘는 인라인 수식 — 이 책에 84 줄 있고, 가리지 않으면 수식 **안쪽**에
    # 병기나 링크가 들어가 수식이 깨진다(`$SSB = SST -⏎SSW$` 가 그랬다).
    # 통화의 `\$` 는 건너뛰고, 닫는 `$` 를 400 자 안에서 못 찾으면 가리지 않는다.
    i = 0
    while i < len(s):
        if s[i] == "$" and (i == 0 or s[i - 1] != "\\"):
            j = i + 1
            while j < len(s) and j - i < 400:
                if s[j] == "$" and s[j - 1] != "\\":
                    break
                j += 1
            if j < len(s) and j - i < 400 and s[j] == "$":
                blank(i, j + 1)
                i = j + 1
                continue
        i += 1
    s = "".join(out)
    for rx in (r"`[^`\n]*`",                               # 인라인 코드
               r"!\[.*?\]\([^)]*\)",                       # 그림 (대체텍스트째)
               r"\[[^\]]*\]\([^)]*\)",                     # 링크 (글자째)
               r"<[^>\n]+>"):                              # HTML 태그
        for m in re.finditer(rx, s, re.S):
            blank(m.start(), m.end())
        s = "".join(out)
    return s


def tail_ok(text, end):
    """용어가 거기서 끝나는가 — 뒤에 한글이 오면 조사일 때만 받는다."""
    rest = text[end:]
    if not rest or not HAN.match(rest[0]):
        return True
    for j in JOSA:
        if rest.startswith(j) and (len(rest) == len(j) or not HAN.match(rest[len(j)])):
            return True
    return False


def pages(only):
    for p in sorted(D.rglob("*.md")):
        if p.name.endswith("_score.md") or p.name == "glossary.md":
            continue
        rel = p.relative_to(D).as_posix()
        if only and not (rel == only or rel.startswith(only.rstrip("/") + "/")):
            continue
        yield p


def main(argv):
    cfg = json.loads((R / "tools" / "terms.json").read_text(encoding="utf-8"))
    TERMS, ABBR = cfg["terms"], cfg["abbr"]
    apply_ = "--apply" in argv
    show = "--show" in argv
    only = next((a for a in argv if not a.startswith("--")), None)

    trx = re.compile("|".join(re.escape(k) for k in
                     sorted(TERMS, key=len, reverse=True)))
    auto = [a for a in ABBR if ABBR[a].get("auto", True)]
    arx = re.compile(r"(?<![A-Za-z0-9_\-\\])(" +
                     "|".join(re.escape(a) for a in
                              sorted(auto, key=len, reverse=True)) +
                     r")(?![A-Za-z0-9_\-])")

    need_term = need_link = need_full = done_t = done_a = byhand = 0
    for p in pages(only):
        raw = p.read_text(encoding="utf-8")
        body = mask(raw)
        rel = p.relative_to(D).as_posix()
        g = os.path.relpath(D / "glossary.md", p.parent)
        ins = []

        seen = set()
        for m in trx.finditer(body):
            ko = m.group(0)
            if ko in seen:
                continue
            if m.start() and HAN.match(body[m.start() - 1]):
                continue
            if not tail_ok(body, m.end()):
                continue
            seen.add(ko)
            # 「일원배치 분산분석」을 이미 풀었으면 그 안의 「분산분석」은 두 번 풀지 않는다
            if any(ko != k and ko in k for k in seen - {ko}):
                continue
            en = TERMS[ko]
            # 「가족단위 오류율(FWER)」도 병기다 — 그 FWER 이 용어집 링크면 `(` 뒤가 `[` 다
            if re.search(re.escape(ko) + r"\**[（(]\s*\[?[A-Za-z]", raw):
                done_t += 1
                continue                      # 그 쪽에 이미 병기가 있다
            e = m.end()
            if raw[e:e + 2] == "**":          # **등분산성** → 굵게 밖에 붙인다
                e += 2
            if raw[e:e + 1] in ("(", "（"):
                # 「제1종 오류(거짓 양성)」 — 괄호가 겹친다. 넣지 말고 **녹인다**
                # 「제1종 오류(type I error, 거짓 양성)」
                byhand += 1
                if show:
                    print(f"  녹임        {rel}  {ko}({en}, …) — 뒤에 괄호가 이미 있다")
                ins.append((e + 1, f"{en}, ", f"녹임 {ko}({en}, …)"))
                continue
            ins.append((e, f"({en})", f"용어 {ko}({en})"))
            need_term += 1

        seen = set()
        for m in arx.finditer(body):
            ab = m.group(1)
            if ab in seen:
                continue
            seen.add(ab)
            e = ABBR[ab]
            if e.get("def") == rel:
                # 쪽이 쉼표나 괄호로 끊어 적을 수 있으므로 첫 토막만 본다
                if re.split(r"[,(]", e["en"])[0].strip().lower() not in raw.lower():
                    need_full += 1
                    print(f"  풀네임 없음  {rel}  {ab} — "
                          f"{e['ko']}({e['en']}, {ab}) 로 펼칠 것")
                    # 설명하는 쪽에서는 풀네임을 펼친다. 「사분위범위(IQR)」 꼴이
                    # 있으면 그 괄호에 녹이고, 없으면 첫 출현 뒤에 붙인다
                    # 머리글·표·펜스에 든 「분산팽창인자(VIF)」를 잡으면 안 된다 —
                    # 머리글에 영어를 넣으면 오른쪽 목차가 어지러워진다. 가린 사본에서 찾는다
                    k = body.find(f"{e['ko']}(" + ab + ")")
                    if k >= 0:
                        ins.append((k + len(e["ko"]) + 1, f"{e['en']}, ",
                                    f"풀네임 {ab}"))
                    else:
                        ins.append((m.end(), f"({e['en']})", f"풀네임 {ab}"))
                continue
            if f"glossary.md#{ab.lower()}" in raw:
                done_a += 1
                continue
            ins.append((m.start(), m.end(),
                        f"[{ab}]({g}#{ab.lower()})", f"약어 {ab}"))
            need_link += 1

        if show and ins:
            print(f"\n--- {rel}")
            for it in ins:
                a = it[0]
                print(f"  {it[-1]:<34}…{raw[max(0, a - 38):a + 24]}…"
                      .replace("\n", "⏎"))
        if apply_ and ins:
            out = raw
            for it in sorted(ins, key=lambda x: -x[0]):
                if len(it) == 3:                       # 용어 병기 — 끼워 넣기
                    out = out[:it[0]] + it[1] + out[it[0]:]
                else:                                  # 약어 — 링크로 바꾸기
                    out = out[:it[0]] + it[2] + out[it[1]:]
            p.write_text(out, encoding="utf-8")

    what = "넣었다" if apply_ else "비어 있다"
    print(f"\n용어 병기 {need_term} 곳 {what} (이미 있던 것 {done_t})")
    print(f"약어 용어집 링크 {need_link} 곳 {what} (이미 있던 것 {done_a})")
    print(f"설명하는 쪽인데 풀네임이 없는 약어 {need_full} 곳")
    print(f"괄호가 겹쳐 손으로 녹일 자리 {byhand} 곳 (--show 로 본다)")
    return 0 if apply_ or not (need_term + need_link + need_full) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
