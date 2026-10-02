"""풀이 상자의 구조 결함을 찾는다 — 빌드 없이.

찾는 것:
  UNFENCED  울타리 밖에 떠 있는 코드 줄 (`#` 주석이 제목으로 렌더된다)
  ODDFENCE  풀이 안 울타리 개수가 홀수 (닫히지 않았다)
  HR        들여쓴 `---` (상자 안에 <hr> 이 생긴다)
  HEADING   풀이 안의 `#` 제목
  BAREPIPE  표 셀 안 맨 `|` 수식
"""
import pathlib, re, sys

ROOT = pathlib.Path("docs")
ADMON = re.compile(r"^(\s*)\?\?\?\+?\s+\w+")
CODEY = re.compile(
    r"^\s*(?:import\s+\w|from\s+[\w.]+\s+import\b|def\s+\w+\s*\(|"
    r"(?:np|pd|plt|sns|sp|stats|sm|ax|fig)\.\w+\(|print\s*\(|"
    r"for\s+\w+\s+in\s+.*:\s*$|if\s+__name__)"
)

def check(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    bad = []
    i = 0
    while i < len(lines):
        m = ADMON.match(lines[i])
        if not m:
            i += 1
            continue
        base = len(m.group(1))
        head = lines[i]
        # 본문: 더 깊게 들여쓴 줄 + 빈 줄
        j = i + 1
        body = []
        while j < len(lines):
            s = lines[j]
            if not s.strip():
                body.append((j, s)); j += 1; continue
            ind = len(s) - len(s.lstrip())
            if ind <= base:
                break
            body.append((j, s)); j += 1
        infence = False
        nfence = 0
        for ln, s in body:
            t = s.strip()
            if t.startswith("```"):
                nfence += 1
                infence = not infence
                continue
            if infence:
                continue
            if CODEY.match(s):
                bad.append(("UNFENCED", ln + 1, t[:60]))
            if t == "---":
                bad.append(("HR", ln + 1, t))
            if re.match(r"^#{1,6}\s", t):
                bad.append(("HEADING", ln + 1, t[:60]))
        if nfence % 2:
            bad.append(("ODDFENCE", i + 1, head.strip()[:50]))
        i = j
    # 표 셀 안에서 수식이 맨 `|` 에 잘렸는가.
    #   `\$` 는 통화, `\|` 는 노름이므로 구분자가 아니다. 먼저 지운다.
    #   표 줄만 본다 (양끝이 `|`).
    for ln, s in enumerate(lines, 1):
        t = s.strip()
        if not (t.startswith("|") and t.endswith("|")):
            continue
        t = t.replace(r"\$", "").replace(r"\|", "")
        for cell in t[1:-1].split("|"):
            if cell.count("$") % 2:
                bad.append(("BAREPIPE", ln, s.strip()[:70]))
                break
    return bad

tot = 0
for p in sorted(ROOT.rglob("*.md")):
    b = check(p)
    if b:
        print(f"\n=== {p.relative_to(ROOT)}")
        for kind, ln, txt in b:
            print(f"  {kind:9s} {ln:5d}  {txt}")
        tot += len(b)
print(f"\n합계 {tot}")
