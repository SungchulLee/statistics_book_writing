#!/usr/bin/env python3
r"""수식 구간에만 치환한다 — 코드·출력·대체텍스트는 건드리지 않는다.

표기를 책 전체에서 통일할 때 쓴다. 2026-10-24 에 자유도 표기를 괄호에서 아래첨자로
옮기며(χ² 181 · t 218 · F 247 = 646 곳) 만든 것이다.

**이 도구의 요점은 치환이 아니라 「건드리면 안 되는 것」을 먼저 보여 주는 것이다.**
그때 바꾸지 않은 자리가 1141 곳이었고, 일괄 치환했으면 전부 틀린 수식이 되었다.

  F(x)·F(t)·F(T)        450  누적분포함수. 인자가 하나면 CDF 다
  F(x,y)·F(a,c)          20  **결합** 누적분포함수. 쉼표가 있어 인자 규칙으로 안 걸린다
  stats.t(5) · 출력      483  파이썬 호출과 코드가 찍은 문자열
  t(λ)·t(0)·t(VIF)       64  함수다(ch18 능형회귀의 t(λ))
  t(28)=2.1              19  APA 보고 형식. 여섯은 인용 블록 안이다
  χ²(c)·χ²(a)             5  절단점·칸도수의 함수
  대체텍스트·표 셀의 t(3)    6  수식이 아니라 꼬리표. alt 는 수식이 될 수 없다

쓰기:
  python3 tools/math_subst.py scan  '<정규식>' [chNN ...]
  python3 tools/math_subst.py apply '<정규식>' '<치환>' [chNN ...] [--exclude 경로 ...]

`scan` 은 맞은 자리를 **문맥별로 갈라** 세고 표본 위치를 찍는다. 그 표를 보고
무엇이 치환 대상이 아닌지 판정한 **뒤에** apply 를 쓴다. 치환군은 `\1` 꼴로 쓴다.

`apply` 는 **수식 구간 안에서만** 바꾸고(인라인 `$...$` 와 `$$` 블록), 쓰기 전에
두 가지를 지킨다.

  1. 홀수 `$` 줄의 수가 변하지 않았는가 — 변했으면 수식 구분자를 깨뜨린 것이다
  2. `{` 와 `}` 의 수가 줄마다 맞는가

하나라도 어긋나면 **아무것도 쓰지 않고** 멈춘다.

**함정 둘.** 줄을 넘는 인라인 수식(이 책에 84 줄 있다)은 건너뛴다 — 안전한 쪽으로
기운 것이고, 놓친 자리는 scan 의 「맨 본문」 칸에 남으므로 손으로 보면 된다. 그리고
파일 경로가 수식처럼 보이는 일이 있다(`student_t_299.png` 의 `t_299`). scan 이
「맨 본문」으로 분류하니 그 칸을 반드시 열어 볼 것.
"""
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

INLINE = re.compile(r"(?<!\$)\$(?!\$)([^$]{1,400})\$(?!\$)")
ALT = re.compile(r"!\[[^\]]*\]")


def pages(chapters):
    for p in sorted(DOCS.rglob("*.md")):
        if p.name.endswith("_score.md"):
            continue
        rel = p.relative_to(DOCS)
        if chapters and rel.parts[0] not in chapters:
            continue
        yield p, str(rel)


def odd_dollar_lines(text):
    """홀수 개의 `$` 를 가진 줄 수 — 치환 전후로 같아야 한다."""
    n, fence = 0, False
    for s in text.split("\n"):
        if s.strip().startswith("```"):
            fence = not fence
            continue
        if fence or s.strip() == "$$":
            continue
        if len(re.findall(r"(?<!\\)\$", s)) % 2:
            n += 1
    return n


def classify(pat, chapters):
    """맞은 자리를 문맥별로 센다."""
    buckets = defaultdict(Counter)
    sample = defaultdict(list)
    for p, rel in pages(chapters):
        fence, disp = None, False
        for i, s in enumerate(p.read_text(encoding="utf-8").split("\n")):
            st = s.strip()
            if st.startswith("```"):
                fence = None if fence else (st[3:].strip() or "출력")
                continue
            if st == "$$":
                disp = not disp
                continue
            if fence:
                where = "펜스:" + ("python" if fence == "python" else "출력")
                for m in pat.finditer(s):
                    buckets[where][m.group(0)] += 1
                    sample[(where, m.group(0))].append(f"{rel}:{i+1}")
                continue
            # 대체텍스트·표 셀은 수식이 아니다
            alt_spans = [(m.start(), m.end()) for m in ALT.finditer(s)]
            math_spans = ([(0, len(s))] if disp
                          else [(m.start(1), m.end(1)) for m in INLINE.finditer(s)])
            for m in pat.finditer(s):
                if any(a <= m.start() < b for a, b in alt_spans):
                    where = "대체텍스트"
                elif any(a <= m.start() < b for a, b in math_spans):
                    where = "수식"
                elif st.startswith("|"):
                    where = "표 셀"
                else:
                    where = "맨 본문"
                buckets[where][m.group(0)] += 1
                sample[(where, m.group(0))].append(f"{rel}:{i+1}")
    return buckets, sample


def cmd_scan(pat, chapters):
    buckets, sample = classify(pat, chapters)
    order = ["수식", "맨 본문", "표 셀", "대체텍스트", "펜스:python", "펜스:출력"]
    total = sum(sum(c.values()) for c in buckets.values())
    print(f"맞은 자리 모두 {total} 곳\n")
    for where in order + [w for w in buckets if w not in order]:
        c = buckets.get(where)
        if not c:
            continue
        note = {"수식": "  ← apply 가 바꾸는 칸",
                "맨 본문": "  ← 파일 경로가 섞인다. 반드시 열어 볼 것",
                "대체텍스트": "  ← 수식이 될 수 없다",
                "펜스:python": "  ← 코드. 바꾸면 출력과 어긋난다",
                "펜스:출력": "  ← 코드가 찍은 것. 코드와 함께 고쳐야 한다"}.get(where, "")
        print(f"— {where}: {sum(c.values())} 곳{note}")
        for txt, n in c.most_common(14):
            loc = sample[(where, txt)][0]
            print(f"    {n:>4d}  {txt[:38]:<38s} {loc}")
        print()
    return 0


def cmd_apply(pat, repl, chapters, exclude):
    # 수식 **안**을 고치는 치환이 달러를 더할 이유가 없다. 2026-10-24 에 이 도구를
    # 시험하다 't_5' -> 't_5\$' 가 서른다섯 곳을 망가뜨리고도 통과했다 — 넣은 것이
    # escape 된 `\$`(통화 기호)여서 아래의 구분자 검사가 일부러 건너뛴 것이다.
    # 구분자와 중괄호 검사는 **치환이 말이 되는지**는 보지 못한다.
    if "$" in repl:
        print("  멈춤: 치환에 `$` 가 들어 있다. 수식 안을 고치는 치환은 달러를 더하지 않는다")
        return 1

    # **치환 문자열의 역슬래시는 re 가 먹는다.** 2026-10-29 에 `z_{1-\alpha/2}` 를
    # 넣었더니 `\a` 가 BEL 제어문자로 바뀌어 218 곳에 `z_{1-<BEL>lpha/2}` 가 쓰였다.
    # 빌드도 통과하고 검사도 0 이었다 — 본문을 눈으로 봐야 보인다. LaTeX 치환은
    # 역슬래시가 거의 언제나 들어가므로 이 함정은 **매번** 밟는다.
    # 군 참조(\1 \g<1>)만 남기고 나머지 역슬래시를 글자 그대로 만든다.
    safe = re.sub(r"\\(?![0-9]|g<)", r"\\\\", repl)
    repl_fn = (lambda m: m.expand(safe))
    changed, total = [], 0
    for p, rel in pages(chapters):
        if rel in exclude:
            continue
        old = p.read_text(encoding="utf-8")
        L = old.split("\n")
        fence, disp, n = False, False, 0
        for i, s in enumerate(L):
            st = s.strip()
            if st.startswith("```"):
                fence = not fence
                continue
            if fence:
                continue
            if st == "$$":
                disp = not disp
                continue
            if disp:
                new, k = pat.subn(repl_fn, s)
            else:
                out, last, k = [], 0, 0
                for m in INLINE.finditer(s):
                    seg, kk = pat.subn(repl_fn, m.group(1))
                    k += kk
                    out.append(s[last:m.start(1)])
                    out.append(seg)
                    last = m.end(1)
                out.append(s[last:])
                new = "".join(out)
            if k:
                L[i] = new
                n += k
        if not n:
            continue
        new_text = "\n".join(L)
        # 안전장치 — 하나라도 어긋나면 아무것도 쓰지 않는다
        if odd_dollar_lines(new_text) != odd_dollar_lines(old):
            print(f"  멈춤: {rel} — 홀수 $ 줄 수가 바뀌었다. 아무것도 쓰지 않았다")
            return 1
        for i, s in enumerate(new_text.split("\n")):
            if "{" in s and s.count("{") != s.count("}") and \
                    old.split("\n")[i].count("{") == old.split("\n")[i].count("}"):
                print(f"  멈춤: {rel}:{i+1} — 중괄호 균형을 깨뜨렸다. 아무것도 쓰지 않았다")
                return 1
        changed.append((rel, n, new_text))
        total += n
    for rel, n, new_text in changed:
        (DOCS / rel).write_text(new_text, encoding="utf-8")
    print(f"바꾼 곳 {total} · 파일 {len(changed)}" + (f" · 제외 {len(exclude)}" if exclude else ""))
    for rel, n, _ in sorted(changed, key=lambda x: -x[1])[:10]:
        print(f"    {n:>4d}  {rel}")
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().split("쓰기:")[1].strip())
        return 2
    mode = argv[0]
    chapters = [a for a in argv if re.fullmatch(r"ch\d+", a)]
    exclude = set()
    if "--exclude" in argv:
        exclude = {a for a in argv[argv.index("--exclude") + 1:] if a.endswith(".md")}
    try:
        pat = re.compile(argv[1])
    except re.error as e:
        print(f"정규식이 깨졌다: {e}")
        return 2
    if mode == "scan":
        return cmd_scan(pat, chapters)
    if mode == "apply":
        if len(argv) < 3:
            print("apply 는 치환 문자열이 필요하다")
            return 2
        return cmd_apply(pat, argv[2], chapters, exclude)
    print(f"모르는 명령: {mode} (scan 또는 apply)")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
