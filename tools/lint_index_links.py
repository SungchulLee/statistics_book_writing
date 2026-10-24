r"""장 개요가 쪽 이름을 굵게 적고 링크를 걸지 않은 자리를 찾는다.

    python3 tools/lint_index_links.py            전체
    python3 tools/lint_index_links.py ch13       그 장만
    python3 tools/lint_index_links.py --apply    이름이 nav 와 맞는 것에 링크를 건다

장 개요의 「장의 구성」은 쪽마다 한 줄씩 `- **쪽 이름** --- 설명` 꼴로 적는다.
독자가 **무엇을 읽을지 고르는 자리**인데 2026-10-24 까지 그 줄 **331 개 가운데
링크가 걸린 것이 하나**였다. `--strict` 는 할 말이 없다 — 깨진 링크가 아니라
없는 링크다. `lint_orphans.py` 가 센 고아의 일부도 여기서 왔다.

맞추는 법은 `lint_next_pointers.py` 와 같다. 굵은 이름을 **그 장의** nav 제목과
맞추는데, 비교 전에 양쪽을 눌러야 한다 — nav 는 유니코드를 쓰고(`σ²의 신뢰구간`)
개요는 LaTeX 를 섞어 쓴다. 두 가지를 조심할 것:

- **개요에만 있는 줄이 있다.** 쪽이 아니라 **주제**를 묶어 적은 줄, 절 이름을 적은
  줄이 섞여 있다. 그런 줄은 맞는 쪽이 없으므로 **건드리지 않고 넘어간다**
- **짧은 이름은 아무 데나 들어맞는다.** 네 글자 미만은 느슨한 맞춤에서 받지 않는다
"""
import collections
import os
import pathlib
import re
import sys

R = pathlib.Path(__file__).resolve().parent.parent
D = R / "docs"
NAV = re.compile(r'^\s*- "?(.+?)"?:\s*((?:ch|index)[\w/.\-]*\.md)\s*$')
BULLET = re.compile(r"^(-\s+)\*\*(?!\[)(.+?)\*\*(\s*(?:---|—|--))", re.M)
GREEK = ("sigma mu tau rho alpha beta lambda chi theta delta phi nu xi eta zeta "
         "psi omega gamma epsilon pi kappa iota upsilon").split()
UNI = {"σ": "sigma", "μ": "mu", "τ": "tau", "ρ": "rho", "α": "alpha", "β": "beta",
       "λ": "lambda", "χ": "chi", "θ": "theta", "φ": "phi", "²": "2", "₁": "1",
       "₂": "2", "X̄": "xbar", "p̂": "phat", "̄": "bar", "̂": "hat"}


def key(s):
    s = s.replace("★", " ")
    s = re.sub(r"\\(?:bar|hat|operatorname|mathrm|text|left|right)", " ", s)
    for g in GREEK:
        s = s.replace("\\" + g, g)
    for u, v in UNI.items():
        s = s.replace(u, v)
    return re.sub(r"[^a-z0-9가-힣]", "", s.lower())


def main(argv):
    nav = [(m.group(1), m.group(2)) for m in
           map(NAV.match, (R / "mkdocs.yml").read_text(encoding="utf-8").split("\n"))
           if m and (D / m.group(2)).exists()]
    apply = "--apply" in argv
    only = next((a for a in argv if not a.startswith("--")), None)
    ok = already = skipped = 0
    amb = []
    byc = collections.Counter()
    for idxp in sorted(D.glob("ch*/index.md")):
        ch = idxp.parent.name
        if only and ch != only:
            continue
        pages = {}
        for title, path in nav:
            if path.startswith(ch + "/") and not path.endswith("index.md"):
                pages.setdefault(key(title), path)
        t = idxp.read_text(encoding="utf-8")
        already += len(re.findall(r"^-\s+\*\*\[", t, re.M))
        out, changed = t, 0
        for m in BULLET.finditer(t):
            name = m.group(2)
            k = key(name)
            tgt = pages.get(k)
            if not tgt:
                hits = [v for kk, v in pages.items()
                        if len(k) >= 4 and (k in kk or kk in k)]
                if len(hits) > 1:
                    amb.append((ch, name, hits))
                    continue
                tgt = hits[0] if hits else None
            if not tgt:
                skipped += 1          # 쪽이 아니라 주제·절을 적은 줄
                continue
            whole = m.group(0)
            if out.count(whole) != 1:
                amb.append((ch, name, [tgt]))
                continue
            link = os.path.relpath(D / tgt, idxp.parent).replace(os.sep, "/")
            out = out.replace(whole, f"{m.group(1)}**[{name}]({link})**{m.group(3)}")
            changed += 1
            ok += 1
            byc[ch] += 1
            if "--list" in argv:
                print(f"    {ch}  {name[:44]:<46s} -> {tgt}")
        if changed and apply:
            idxp.write_text(out, encoding="utf-8")

    print(f"장 개요의 쪽 이름 — 링크 없고 nav 와 맞음 {ok} · 이미 링크 {already} · "
          f"맞는 쪽이 없어 넘김 {skipped} · 애매함 {len(amb)}"
          + ("   (링크를 걸었다)" if apply else ""))
    if byc and not apply:
        print("\n### 장별")
        for c in sorted(byc):
            print(f"    {c}  {byc[c]}")
    if amb:
        print("\n### 애매함 — 손으로 볼 것")
        for ch, name, hits in amb:
            print(f"    {ch}  「{name}」  ->  {', '.join(hits)}")
    return 1 if (ok or amb) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
