"""링크로 적은 **항목 번호**가 그 쪽에 실제로 있는지 본다.

    python3 tools/lint_item_refs.py            책 전체 (몇 초)
    python3 tools/lint_item_refs.py docs/ch15

CLAUDE.md 는 다른 쪽의 항목을 가리킬 때 **쪽을 링크로 적으라**고 한다. 까닭은
번호가 쪽마다 1 부터 다시 시작해서 「15.4절 연습문제 4」가 가리키는 곳이 넷이고
넷 다 존재하기 때문이다. 그 규약의 마지막 문장이 **「링크로 적어야 기계가
검증한다」**인데, 정작 검증하는 기계가 없었다. 이 도구가 그것이다.

`[제목](경로.md) … 보기 3` 꼴을 찾아 경로를 풀고, **그 쪽이 보기 3 을 정말
갖고 있는지** 본다. 없으면 독자가 눌러서 도착한 쪽에서 가리키는 것을 찾지 못한다.
`--strict` 는 링크가 살아 있으므로 아무 말도 하지 않고, `lint_section_refs` 는
절 번호만 보므로 이것을 못 잡는다.

2026-10-23 에 책 전체 383 곳을 검사해 넷을 찾았다. 셋이 ch15 였다 —
`bartlett_test.md` 가 민감도 쪽의 「보기 3」을 두 번 가리켰는데 그 쪽에는 보기가
하나뿐이었고(제 쪽의 보기 3 을 이야기하던 문단이라 번호가 옮아붙은 것으로 보인다),
`f_test_two_variances.md` 가 ch14 의 「연습문제 38」을 가리켰는데 그 쪽 연습문제는
열까지다(같은 장의 다른 두 쪽은 같은 자리를 「연습문제 8」로 바르게 적고 있었다).

## 참조도 두 가지 꼴이고, 둘 다 보아야 한다

    [등분산 점검](x.md) 보기 5      항목 번호가 링크 **뒤**
    [13.5절 보기 5](x.md)           항목 번호가 링크 **글자 안**

처음에는 앞의 꼴만 보고 있었고, 그래서 **규약대로 적은 자리 열둘이 통째로 검사
밖**에 있었다. 2026-10-23 에 다른 세션이 자기 도구에서 **같은 맹점을 반대
방향으로** 틀린 것을 알려 와(규약대로 적힌 열다섯을 결함으로 셌다) 이쪽을 재어
보고 찾았다. 한쪽은 거짓 양성, 한쪽은 거짓 음성이지만 원인은 하나다 — 「절」
뒤만 보고 **대괄호가 그보다 앞에 올 수 있다는 것**을 놓친 것이다.

## 항목을 세는 두 가지 꼴을 모두 보아야 한다

보기와 연습문제는 굵은 꼴(`**보기 1.**`)이고 정의와 정리는 **머리글 꼴**
(`### 정리 1. 이름 { .thm }`)이다. 굵은 꼴만 세면 정의·정리를 가리키는 참조가
**모두 거짓 양성으로 쏟아진다** — 처음 돌렸을 때 그렇게 일곱이 잘못 걸렸다.

## 한계

- 링크가 붙지 않은 참조(「15.4절 연습문제 4」)는 애초에 **검증할 수 없다.**
  그런 꼴을 찾는 것은 CLAUDE.md 점검 목록의 `grep -rnP` 쪽이다
- 번호가 **존재하기는 하는데 엉뚱한 것**을 가리키는 경우는 못 잡는다.
  쪽을 합쳐 번호가 밀렸을 때가 그렇다. 그것은 사람이 읽어야 한다
- 참조와 번호 사이를 30 자까지만 본다. 더 멀면 놓친다
"""
import collections
import glob
import os
import re
import sys

KINDS = r"(보기|연습문제|정의|정리)"
BOLD = re.compile(r"\*\*" + KINDS + r"\s*(\d+)[.,]")
HEAD = re.compile(r"^#{2,4}\s*" + KINDS + r"\s*(\d+)[.,]", re.M)
# 항목 번호가 **링크 뒤**에 오는 꼴:  [등분산 점검](x.md) 보기 5
AFTER = re.compile(r"\]\(([^)]+\.md)\)[^\n]{0,30}?" + KINDS + r"\s*(\d+)")
# 항목 번호가 **링크 글자 안**에 든 꼴:  [13.5절 보기 5](x.md)
# 이 꼴을 빠뜨리면 규약대로 적은 자리가 통째로 검사 밖에 남는다. 2026-10-23 에
# 다른 세션이 자기 도구에서 같은 꼴을 **반대 방향으로** 틀렸고(규약대로 적힌
# 열다섯을 결함으로 셌다), 그 이야기를 듣고 이쪽을 재어 보니 열둘이 검사되지
# 않고 있었다. 한쪽은 거짓 양성, 한쪽은 거짓 음성인 **같은 맹점**이다.
# 링크 글자의 길이를 묶어 둔다. `[^\]]` 는 개행을 막지 않으므로 묶지 않으면 위쪽의
# `[` 하나가 수백 줄 아래의 링크와 짝지어진다. 2026-10-24 에 그렇게 거짓 양성이
# 하나 났다 — fligner_killeen 의 「연습문제 16」이 390 줄 아래 comparison.md 링크의
# 글자로 읽혔다. 줄바꿈으로 나뉜 참조는 살려야 하므로 개행을 막는 대신 길이로 묶는다.
INSIDE = re.compile(r"\[[^\]]{0,120}?" + KINDS + r"\s*(\d+)[^\]]{0,120}\]\(([^)]+\.md)\)")


def items(path):
    """그 쪽이 가진 항목 번호 — 굵은 꼴과 머리글 꼴을 모두 센다."""
    text = open(path, encoding="utf-8").read()
    out = collections.defaultdict(set)
    for pat in (BOLD, HEAD):
        for kind, num in pat.findall(text):
            out[kind].add(int(num))
    return out


def main(argv):
    root = argv[0] if argv else "docs"
    cache, bad, checked = {}, [], 0
    for f in sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True)):
        d = os.path.dirname(f)
        text = open(f, encoding="utf-8").read()
        found = [(rel, kind, num) for rel, kind, num in AFTER.findall(text)]
        found += [(rel, kind, num) for kind, num, rel in INSIDE.findall(text)]
        for rel, kind, num in found:
            tgt = os.path.normpath(os.path.join(d, rel.split("#")[0]))
            if not os.path.exists(tgt):
                continue                      # 죽은 링크는 --strict 가 잡는다
            if tgt not in cache:
                cache[tgt] = items(tgt)
            checked += 1
            if int(num) not in cache[tgt][kind]:
                bad.append((f, rel, kind, int(num), sorted(cache[tgt][kind])))

    for f, rel, kind, num, have in bad:
        print(f"── {f.replace('docs/', '')}")
        print(f"     -> {rel}  {kind} {num}"
              f"   (그 쪽의 {kind}: {have if have else '없음'})")
    print(f"\n링크로 적은 항목 참조 {checked} 곳 · 그 쪽에 없는 번호 {len(bad)} 곳")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
