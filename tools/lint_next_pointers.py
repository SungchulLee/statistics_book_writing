r"""「다음 절 **이름**」이 링크인지, 그리고 그 이름이 **정말 다음 쪽인지** 본다.

    python3 tools/lint_next_pointers.py            전체
    python3 tools/lint_next_pointers.py ch13       그 장만
    python3 tools/lint_next_pointers.py --apply    풀리는 것에 링크를 건다

쪽마다 마지막 줄이 다음 쪽을 **이름까지 적어 두고** 링크를 걸지 않는 일이 흔했다.
$2026$-$10$-$24$ 에 포인터 $240$ 개 가운데 링크가 있는 것이 **여덟**이었다. 빌드는
할 말이 없다 — 깨진 링크가 아니라 **없는** 링크다. 그 결과가 고아 쪽 $103$ 개였고,
링크를 걸자 $55$ 로, 완전 고립이 $38$ 에서 $13$ 으로 줄었다(`lint_orphans.py` 참조).

이 도구의 **진짜 값은 둘째 검사**다. 포인터의 이름을 nav 의 **바로 다음 쪽**과 맞추므로,
장·절을 옮긴 뒤 포인터가 따라가지 않은 자리가 드러난다. 그런 자리는 글자라서 어떤
빌드도 잡지 못하고, 더 나쁘게는 **가리키는 이름의 쪽이 책에 실제로 있는 일이 많아서**
「없는 쪽」 검사에도 걸리지 않는다. 쪽을 흡수해 없앤 뒤 남은 포인터가 전형이다.

이름이 안 맞는다고 모두 결함은 아니다. 두 가지가 섞여 나온다.

  표기 차이   가리키는 쪽은 맞고 이름만 줄이거나 달리 적었다
              (「BIC」 대 nav 의 「Bayes 정보기준」, 「ROC 와 AUC」 대 「ROC 곡선과 AUC」)
  진짜 어긋남  가리키는 쪽이 nav 의 다음 쪽이 아니다. 옮긴 자국이다

그래서 `--apply` 는 **이름이 맞는 것만** 건드리고 나머지는 보고만 한다. 손으로 보라.

맞추는 법. 제목을 비교하기 전에 양쪽을 같은 꼴로 눌러야 한다 — nav 는 유니코드를
쓰고(`σ²의 신뢰구간`) 본문은 LaTeX 를 쓰므로(`$\sigma^2$ 의 신뢰구간`) 날 문자열로는
하나도 맞지 않는다. `key()` 가 그리스 문자·아래위첨자·`\bar`·`\hat`·★·절 번호를
떼고 한글과 영숫자만 남긴다. 짧은 이름(「유도」·「비교」·「검정」)은 아무 데나 들어맞아
거짓으로 받아들이기 쉬우므로 **세 글자 미만이면 받지 않는다.**
"""
import os
import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parent.parent
D = R / "docs"
PAGE = re.compile(r'^\s*- "?(.+?)"?:\s*((?:ch|index)[\w/.\-]*\.md)\s*$')
POINTER = re.compile(r"(다음 절|다음 장)\s*\*\*(.+?)\*\*")
GREEK = ("sigma mu tau rho alpha beta lambda chi theta delta phi nu xi eta zeta "
         "psi omega gamma epsilon pi kappa iota upsilon").split()
UNI = {"σ": "sigma", "μ": "mu", "τ": "tau", "ρ": "rho", "α": "alpha", "β": "beta",
       "λ": "lambda", "χ": "chi", "θ": "theta", "δ": "delta", "φ": "phi", "ν": "nu",
       "ξ": "xi", "η": "eta", "ζ": "zeta", "ψ": "psi", "ω": "omega", "γ": "gamma",
       "ε": "epsilon", "π": "pi", "²": "2", "₁": "1", "₂": "2",
       "X̄": "xbar", "p̂": "phat", "̄": "bar", "̂": "hat"}


def key(s):
    s = s.replace("★", " ")
    s = re.sub(r"\\(?:bar|hat|operatorname|mathrm|text|left|right)", " ", s)
    for g in GREEK:
        s = s.replace("\\" + g, g)
    for u, v in UNI.items():
        s = s.replace(u, v)
    return re.sub(r"[^a-z0-9가-힣]", "", s.lower())


def main(argv):
    yml = (R / "mkdocs.yml").read_text(encoding="utf-8")
    order = [(m.group(1), m.group(2)) for m in map(PAGE.match, yml.split("\n"))
             if m and (D / m.group(2)).exists()]
    idx = {path: i for i, (_, path) in enumerate(order)}
    apply = "--apply" in argv
    only = next((a for a in argv if not a.startswith("--")), None)

    ok = already = 0
    bad, amb = [], []
    for _, rel in order:
        if only and not rel.startswith(only + "/"):
            continue
        p = D / rel
        t = p.read_text(encoding="utf-8")
        out, changed = t, 0
        for m in POINTER.finditer(t):
            kind, raw = m.group(1), m.group(2)
            if re.search(r"\]\([^)\s#]+\.md", t[max(0, m.start() - 50):m.end() + 400]):
                already += 1
                continue
            i = idx[rel]
            if i + 1 >= len(order):
                bad.append((rel, kind, raw, "nav 의 마지막 쪽"))
                continue
            ntitle, npath = order[i + 1]
            if kind == "다음 장":
                cur = rel.split("/")[0]
                nxt = next((pa for _, pa in order[i + 1:] if pa.split("/")[0] != cur), None)
                npath = f"{nxt.split('/')[0]}/index.md" if nxt else None
                if not npath or not (D / npath).exists():
                    bad.append((rel, kind, raw, "다음 장의 개요가 없다"))
                    continue
            else:
                kr, kn = key(raw), key(ntitle)
                if min(len(kr), len(kn)) < 3 or not (kr in kn or kn in kr):
                    bad.append((rel, kind, raw, f"nav 의 다음 쪽은 「{ntitle}」"))
                    continue
            old = f"{kind} **{raw}**"
            if out.count(old) != 1:
                amb.append((rel, kind, raw))
                continue
            link = os.path.relpath(D / npath, p.parent).replace(os.sep, "/")
            out = out.replace(old, f"{kind} **[{raw}]({link})**")
            changed += 1
            ok += 1
        if changed and apply:
            p.write_text(out, encoding="utf-8")

    print(f"포인터 — 링크 없고 이름은 맞음 {ok} · 이미 링크 {already} · "
          f"이름이 안 맞음 {len(bad)} · 본문에 둘 이상 {len(amb)}"
          + ("   (링크를 걸었다)" if apply else ""))
    if bad:
        print("\n### 이름이 nav 의 다음 쪽과 안 맞는다 — 표기 차이인가 옮긴 자국인가")
        for rel, kind, raw, why in bad:
            print(f"    {rel:<54s} {kind} **{raw}**  ← {why}")
    if amb:
        print("\n### 같은 문구가 본문에 둘 이상 — 손으로 걸 것")
        for rel, kind, raw in amb:
            print(f"    {rel:<54s} {kind} **{raw}**")
    return 1 if (ok or bad or amb) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
