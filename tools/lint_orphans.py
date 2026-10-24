r"""책 안에서 아무도 가리키지 않는 쪽을 찾는다.

    python3 tools/lint_orphans.py            전체 — 장별 요약과 완전 고립 목록
    python3 tools/lint_orphans.py ch14       그 장의 고아를 모두
    python3 tools/lint_orphans.py --all      책 전체의 고아를 모두

`mkdocs build --strict` 는 **깨진** 링크를 잡지만 **없는** 링크는 잡지 않는다. nav 에
실려 있으면 쪽은 닿을 수 있으니 빌드가 할 말이 없는 것이다. 그래서 천 줄짜리 쪽이
책의 어느 곳에서도 불리지 않은 채 조용히 서 있을 수 있다.

두 가지를 센다.

  고아        들어오는 링크가 0. nav 로만 닿는다
  완전 고립   들어오는 링크도 나가는 링크도 0. 책의 그물에서 완전히 떨어져 있다

**고아 전부가 결함은 아니다.** 순서대로 읽으면 되는 쪽, 장 개요가 이름으로만 소개하는
쪽은 들어오는 링크 없이도 제구실을 한다. 판정이 필요한 것은 **완전 고립** 쪽이다 —
나가는 링크조차 없다는 것은 그 쪽이 책의 다른 어디에도 기대지 않는다는 뜻이고, 그러면
대개 둘 중 하나다. 쓸 자리가 없거나(합치거나 흩을 후보), 쓸 자리가 있는데 잇지 않은 것.

세 가지를 조심할 것.

- **`index.md` 는 세지 않는다.** 장 개요는 nav 가 자리를 주는 쪽이다
- **nav 에 실린 쪽만 본다.** `docs/` 에는 nav 밖의 파일도 있다(`_score.md` 등)
- **nav 를 yaml 로 읽으면 안 된다.** `mkdocs.yml` 에 `!!python/object/apply:` 가
  섞여 있어 `yaml.safe_load` 가 터진다. 정규식으로 `.md` 경로만 긁는다
"""
import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
NAV_MD = re.compile(r":\s*((?:ch|index)[\w/.\-]*\.md)\s*$", re.M)
LINK = re.compile(r"\]\(([^)\s#]+\.md)")


def build():
    """(nav 쪽 목록, 들어오는 링크 수, 나가는 링크 수)"""
    pages = [p for p in NAV_MD.findall((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
             if (DOCS / p).exists()]
    inb, outb = collections.Counter(), {}
    for rel in pages:
        p = DOCS / rel
        links = LINK.findall(p.read_text(encoding="utf-8"))
        outb[rel] = len(links)
        for raw in links:
            try:
                tgt = str((p.parent / raw).resolve().relative_to(DOCS))
            except ValueError:
                continue                      # docs/ 밖을 가리키는 링크
            if tgt != rel:
                inb[tgt] += 1
    return pages, inb, outb


def lines(rel):
    return len((DOCS / rel).read_text(encoding="utf-8").split("\n"))


def main(argv):
    pages, inb, outb = build()
    body = [r for r in pages if not r.endswith("index.md")]
    orph = [r for r in body if inb[r] == 0]
    iso = [r for r in orph if outb[r] == 0]

    which = argv[0] if argv and not argv[0].startswith("--") else None
    print(f"nav {len(body)} 쪽 · 고아 {len(orph)} 개 · 완전 고립 {len(iso)} 개")

    if which:
        sel = sorted((r for r in orph if r.startswith(which + "/")), key=lambda r: -lines(r))
        print(f"\n### {which} 의 고아 {len(sel)} 개")
        for r in sel:
            print(f"    {lines(r):6d}줄  나가는 {outb[r]:2d}  {r}"
                  + ("   ← 완전 고립" if outb[r] == 0 else ""))
    else:
        ch = collections.Counter(r.split("/")[0] for r in orph)
        tot = collections.Counter(r.split("/")[0] for r in body)
        print("\n### 장별 — 고아 / nav 쪽수")
        for c in sorted(tot, key=lambda c: (-ch[c] / tot[c], c)):
            bar = "#" * round(20 * ch[c] / tot[c])
            print(f"    {c:>6}  {ch[c]:3d} / {tot[c]:3d}  {ch[c]/tot[c]:4.0%}  {bar}")
        print(f"\n### 완전 고립 — 들어오는 링크도 나가는 링크도 0 ({len(iso)} 개)")
        for r in sorted(iso, key=lambda r: -lines(r)):
            print(f"    {lines(r):6d}줄  {r}")
        print("\n    한 장의 고아를 모두 보려면:  python3 tools/lint_orphans.py chNN")

    if argv and "--all" in argv:
        print(f"\n### 고아 전부 ({len(orph)} 개)")
        for r in sorted(orph, key=lambda r: -lines(r)):
            print(f"    {lines(r):6d}줄  나가는 {outb[r]:2d}  {r}"
                  + ("   ← 완전 고립" if outb[r] == 0 else ""))
    return 1 if iso else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
