r"""한 장 안에서 같은 일을 두 쪽이 나눠 맡고 있는 자리를 찾는다.

    python3 tools/lint_duplicate_pages.py            전 장
    python3 tools/lint_duplicate_pages.py ch09       그 장만

ch08 에서 이 훑기로 쪽 일곱을 줄였다. 찾아낸 것은 세 가지 꼴이었다.

  이론/모의실험 짝   ci_sigma2 절2 와 ci_var_sim 이 n=12, sigma=2, 100 개로
                    **같은 설정**을 돌리고 있었다. 셋이 그랬다
  이론/계산 짝       ci_mu 와 ci_mean_calc 이 같은 공식을 두 번 세웠다
  축소판 쪽          confidence_intervals.md 가 장 전체를 미리 했다

빌드는 이런 것을 하나도 잡지 못한다. 쪽이 둘이고 링크가 멀쩡하기 때문이다.

내는 신호 넷. **전부 후보일 뿐이고 사람이 읽어 판정해야 한다** — ch08 에서도
「해석」 절 일곱 가운데 버릴 것은 하나뿐이었고(나머지는 모수마다 달랐다),
confidence_intervals 는 지울 쪽이 아니라 미리보기였다.

  SAMESEC   두 쪽 이상에 같은 이름의 절
  SAMESET   두 쪽의 보기가 같은 수를 설정으로 쓴다 (n=12, sigma=2 꼴)
  TWIN      쪽 이름이 접미사만 다르다 (_sim, _calc, _examples, _demo …)
  SUBSET    한 쪽의 절 이름들이 다른 쪽의 절 이름을 거의 다 품는다 (축소판)
"""
import collections
import itertools
import pathlib
import re
import sys

H2 = re.compile(r"^## (?:\d+\.\s*)?(.+?)\s*$")
EX = re.compile(r"^\*\*보기 (\d+)\.\*\*(.*)$")
# 보기 제목에서 설정으로 쓰인 수. n = 12, sigma = 2.0, p = 0.2, rho = 0.6 …
SETUP = re.compile(
    r"(?:n_?\d?|sigma_?\d?|mu_?\d?|rho|p|alpha|lambda|\\sigma_?\d?|\\mu_?\d?|\\rho|\\alpha|\\lambda)"
    r"\s*=\s*(-?\d+(?:\.\d+)?)"
)
GENERIC = {"개요", "연습문제", "정리하며", "해석", "보기", "핵심 정리", "풀이 보기", "가정"}
SUFFIX = re.compile(r"_(sim|calc|examples?|demo|details|comparison|visualization|plot)$")


def scan_chapter(cdir):
    pages = {}
    for p in sorted(cdir.rglob("*.md")):
        if p.name == "index.md":
            continue
        t = p.read_text(encoding="utf-8")
        secs, exs = [], []
        for line in t.split("\n"):
            m = H2.match(line)
            if m:
                secs.append(m.group(1))
            m = EX.match(line)
            if m:
                exs.append((int(m.group(1)), tuple(sorted(SETUP.findall(m.group(2))))))
        pages[p] = (secs, exs)
    return pages


def report(cdir):
    pages = scan_chapter(cdir)
    if len(pages) < 2:
        return []
    out = []

    # SAMESEC
    bysec = collections.defaultdict(list)
    for p, (secs, _) in pages.items():
        for s in secs:
            if s not in GENERIC:
                bysec[s].append(p.name)
    for s, ps in sorted(bysec.items()):
        if len(ps) > 1:
            out.append(("SAMESEC", f"「{s}」 — {', '.join(sorted(set(ps)))}"))

    # SAMESET — 수 셋 이상이 겹치는 보기 짝
    for (p1, (_, e1)), (p2, (_, e2)) in itertools.combinations(pages.items(), 2):
        for (n1, s1), (n2, s2) in itertools.product(e1, e2):
            if len(s1) >= 3 and s1 == s2:
                out.append(("SAMESET",
                            f"{p1.name} 보기 {n1} ≡ {p2.name} 보기 {n2}  설정 {list(s1)}"))

    # TWIN
    stems = {p.stem: p for p in pages}
    for stem, p in sorted(stems.items()):
        m = SUFFIX.search(stem)
        if m and stem[: m.start()] in stems:
            out.append(("TWIN", f"{p.name}  ↔  {stems[stem[: m.start()]].name}"))

    # SUBSET — 작은 쪽의 절이 **한** 큰 쪽 안에 거의 다 들어 있다 (축소판/흡수 후보)
    #
    # 문턱을 느슨하게 잡으면 쓸모가 없다. 처음 「한 절이라도 겹치는 쪽이 셋 이상」으로
    # 두었더니 신호 341 개 가운데 330 개가 여기서 나왔는데, ch04 의 분포 쪽들이 모두
    # 「정의·적률·성질」을 갖는 것처럼 **책이 의도한 쪽 틀**에 걸린 것이었다.
    # 그래서 (1) 상대가 한 쪽이어야 하고 (2) 작은 쪽의 절이 70% 이상 들어 있어야 하고
    # (3) 큰 쪽이 절을 두 배 이상 가져야 한다로 좁혔다.
    for (p1, (s1, _)), (p2, (s2, _)) in itertools.permutations(pages.items(), 2):
        a = {s for s in s1 if s not in GENERIC}
        b = {s for s in s2 if s not in GENERIC}
        if len(a) < 3 or len(b) < 2 * len(a):
            continue
        if len(a & b) / len(a) >= 0.7:
            out.append(("SUBSET",
                        f"{p1.name} 의 절 {len(a)} 개 가운데 {len(a & b)} 개가 "
                        f"{p2.name}({len(b)} 절) 안에 있다"))
    return out


def main(argv):
    base = pathlib.Path("docs")
    chapters = [base / argv[0]] if argv else sorted(
        d for d in base.iterdir() if d.is_dir() and d.name.startswith(("ch", "appendix")))
    total = 0
    for c in chapters:
        if not c.is_dir():
            continue
        rows = report(c)
        if not rows:
            continue
        npages = len([p for p in c.rglob("*.md") if p.name != "index.md"])
        print(f"\n════ {c.name}  ({npages} 쪽)")
        for kind, msg in rows:
            print(f"  {kind:8} {msg}")
        total += len(rows)
    print(f"\n신호 {total} 개. 전부 후보이며 사람이 읽어 판정해야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
