r"""가리키는 곳이 없는 그림과, 없는 그림을 가리키는 참조를 찾는다.

    python3 tools/lint_figures.py            전체
    python3 tools/lint_figures.py ch08       그 장만

`mkdocs build --strict` 는 **링크**를 검사하지만 PNG 가 커밋되어 있으면 아무 말도
하지 않는다. 그래서 두 방향이 모두 조용히 어긋난다.

  고아 그림   파일은 있는데 가리키는 쪽이 없다. 쪽에서 절을 걷어낼 때마다 생긴다
  없는 그림   쪽이 가리키는데 파일이 없다. 그림을 옮기거나 이름을 바꿀 때 생긴다

정규식 주의. 대체텍스트에 `]` 가 들어가면(`E[D]`, `100 t CIs | n=10`) 소박한
`!\[[^\]]*\]\(` 가 끊긴다. 비탐욕 `!\[.*?\]\(` 로 맞추고 `re.S` 를 켤 것.
그림이 `<figure markdown>` 안의 `<img src=...>` 로 들어간 자리도 있어 그쪽도 본다.
"""
import pathlib
import re
import sys

MD_IMG = re.compile(r"!\[.*?\]\(([^)\s]+)", re.S)
HTML_IMG = re.compile(r'(?:src|href)="([^"]+\.(?:png|jpe?g|gif|svg|webp))"')
EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def collect(root):
    """(그림 이름 -> 가리키는 쪽들),  (없는 그림을 가리키는 (쪽, 경로))"""
    refs, missing = {}, []
    for p in sorted(root.rglob("*.md")):
        t = p.read_text(encoding="utf-8")
        for m in list(MD_IMG.finditer(t)) + list(HTML_IMG.finditer(t)):
            raw = m.group(1).strip()
            if raw.startswith(("http://", "https://", "data:")):
                continue
            name = raw.split("/")[-1]
            refs.setdefault(name, []).append(p)
            # 상대경로가 실제로 가리키는 파일이 있는지
            target = (p.parent / raw).resolve()
            if not target.exists():
                missing.append((p, raw))
    return refs, missing


def main(argv):
    base = pathlib.Path("docs")
    root = base / argv[0] if argv else base
    refs, missing = collect(base)          # 참조는 책 전체에서 모은다
    files = [f for f in sorted(root.rglob("*")) if f.suffix.lower() in EXT]
    orphans = [f for f in files if f.name not in refs]

    print(f"그림 {len(files)} 개 · 고아 {len(orphans)} 개 · 없는 그림 참조 {len(missing)} 개")
    if orphans:
        print("\n### 고아 — 파일은 있는데 가리키는 쪽이 없다")
        for f in orphans:
            print(f"    {f.relative_to(base)}")
    if missing:
        print("\n### 없는 그림 — 쪽이 가리키는데 파일이 없다")
        for p, raw in missing:
            print(f"    {p.relative_to(base)}  ->  {raw}")
    return 1 if (orphans or missing) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
