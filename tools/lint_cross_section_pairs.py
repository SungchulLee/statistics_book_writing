#!/usr/bin/env python3
r"""한 장 안에서 **절을 넘어** 같은 일을 하고 있는 쪽 쌍을 찾는다.

절 안의 중복은 눈에 띄지만 절을 넘는 중복은 그렇지 않다. nav 에서 멀리 떨어져 있어
훑을 때 함께 열리지 않는다. 2026-10-23 에 ch14 의 14.1 `financial_context` 와 14.7
`financial_returns` 가 「정형화된 사실」 넷을 **소제목까지 같게** 적고 있는 것을 뒤늦게
찾았다. 서로를 가리키는 링크도 없었다.

자: **같은 장 · 다른 절 두 쪽이 특징적 소제목을 둘 이상 공유**한다.
흔한 소제목(다섯 쪽 이상에 나오는 「해석」·「가설」·「검정통계량」 따위)은 뺀다.

쓰기:  python3 tools/lint_cross_section_pairs.py [chNN ...] [--all]
       --all 을 주면 판정이 끝난 쌍도 함께 보인다
종료코드: 판정이 안 된 쌍이 있으면 1

**이 검사는 후보만 낸다.** 소제목이 겹쳐도 세우는 것이 다르면 중복이 아니다 —
아홉 가운데 넷이 의도된 평행 구조였고, 하나는 틀이 달라서 양쪽이 다 있어야 했다.
판정은 열어서 한다. 자는 [[duplication-ruler]] 와 같다: **이 쪽이 세우는 것이 다른
쪽에 있는가.** 줄 수나 보기 수는 대리 지표일 뿐이다.

아래 SETTLED 는 판정을 끝낸 쌍이다. 같은 여덟을 다시 열지 않도록 까닭을 적어 둔다.
"""
import collections
import itertools
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# 판정이 끝난 쌍: (쪽 A, 쪽 B) -> 판정 까닭
SETTLED = {
    ("ch02/center/mean_median_mode.md", "ch02/spread/descriptive_stats_weed_prices.md"):
        "앞쪽이 정의와 정리를 세우고 뒤쪽이 실제 자료(대마 가격)에 적용한다. "
        "「중앙값」 헤딩 아래가 4줄 대 107줄로, 제목만 같고 하는 일이 다르다",
    ("ch05/mean/robust_means.md", "ch05/non_normal/heavy_tails.md"):
        "로버스트 통계의 두 틀이다 — robust_means 는 오염 모형(이상치를 섞어 붕괴점을 센다), "
        "heavy_tails 는 두꺼운 꼬리 모형(t_5 에서 효율을 본다). 두 틀에서 추정량 순위가 "
        "같지 않으므로 둘 다 있어야 한다. 2026-10-23 에 서로를 가리키는 링크를 달았다",
    ("ch13/distribution/degrees_of_freedom.md", "ch13/practice/validity_conditions.md"):
        "각 쪽이 세 검정(적합도·독립성·동질성)을 자기 관점에서 훑는다. "
        "「적합도 검정」 아래가 58줄 대 7줄로, 한쪽은 유도하고 한쪽은 조건만 적는다",
    ("ch12/anova_one_way/f_test.md", "ch12/anova_two_way/interaction.md"):
        "일원배치와 이원배치가 같은 단계 이름으로 각자의 절차를 적는다. "
        "가설이 다르고(한 요인 대 주효과+교호작용) 평균 계산도 다르다(집단 평균 대 셀 평균)",
    ("ch12/anova_one_way/one_way_model.md", "ch12/anova_two_way/two_way.md"):
        "「주요 차이」가 서로 다른 비교였다 — 한쪽은 t 검정과, 한쪽은 일원배치와. "
        "2026-10-23 에 둘 다 무엇과의 차이인지 말하는 이름으로 바꿨다",
    ("ch11/assumptions/checking_homoscedasticity.md", "ch11/diagnostics/residuals.md"):
        "「잔차-적합값 그림」은 용도가 다르다 — 한쪽은 선형성까지 읽고 한쪽은 등분산만 겨눈다. "
        "겹쳤던 「왜 제곱근을 쓰는가」(한쪽은 불릿 셋, 한쪽은 반정규의 왜도로 증명)는 "
        "2026-10-23 에 수로 답한 자리로 돌렸다",
    ("ch14/descriptive_statistics/jarque_bera.md", "ch14/formal_tests/anderson_darling.md"):
        "검정마다 자기 p 값 유도가 있어야 한다. 「섞인 자료에서 읽기」는 같은 자료를 "
        "두 검정으로 보는 의도된 평행 구조다",
    ("ch15/bartlett_test/limitations.md", "ch15/f_test/sensitivity.md"):
        "바틀렛과 F 가 각각 자기 검정의 1종 오류 팽창을 본다(0.3790 대 0.2665, "
        "F 는 균등분포에서 0.0053 으로 방향이 뒤집힌다). 평행 구조다. "
        "2026-10-23 에 「재현 코드」라는 이름만 수치를 말하는 것으로 바꿨다",
}

COMMON_AT = 5      # 이 쪽수 이상에 나오는 소제목은 지문이 못 된다
MIN_SHARED = 2     # 특징적 소제목을 이만큼 공유하면 후보


def norm(head: str) -> str:
    """소제목을 비교할 수 있게 다듬는다. 「두꺼운 꼬리(초과첨도)」와 「두꺼운 꼬리」를 맞춘다."""
    s = re.sub(r"^#{2,6}\s*", "", head)
    s = re.sub(r"\{[^}]*\}", "", s)            # { .dfn } 따위
    s = re.sub(r"\$[^$]*\$", "", s)            # 수식
    s = re.sub(r"^\d+[.)]\s*", "", s)          # 앞 번호
    s = re.sub(r"^(정의|정리|보조정리|따름정리|예제)\s*\d+\.?\s*", "", s)
    s = re.sub(r"[（(][^)）]*[)）]", "", s)      # 괄호
    s = re.sub(r"[\s·:,\-—]+", "", s)
    return s.strip()


def main(argv: list) -> int:
    show_all = "--all" in argv
    chapters = [a for a in argv if re.fullmatch(r"ch\d+", a)]
    heads, freq = {}, collections.Counter()
    for p in sorted(DOCS.rglob("*.md")):
        if p.name.endswith("_score.md") or p.name == "index.md":
            continue
        s = {norm(l) for l in p.read_text(encoding="utf-8").splitlines()
             if re.match(r"^#{2,4} ", l)}
        heads[p] = {x for x in s if len(x) >= 3}
        freq.update(heads[p])
    common = {k for k, v in freq.items() if v >= COMMON_AT}

    by_ch = collections.defaultdict(list)
    for p in heads:
        by_ch[p.relative_to(DOCS).parts[0]].append(p)

    fresh, settled = [], []
    for ch, ps in sorted(by_ch.items()):
        if chapters and ch not in chapters:
            continue
        for a, b in itertools.combinations(sorted(ps), 2):
            if a.parent == b.parent:
                continue
            shared = (heads[a] & heads[b]) - common
            if len(shared) < MIN_SHARED:
                continue
            key = (str(a.relative_to(DOCS)), str(b.relative_to(DOCS)))
            (settled if key in SETTLED else fresh).append((key, shared))

    for key, shared in fresh:
        print(f"  판정 필요  {key[0]}")
        print(f"             {key[1]}")
        print(f"             공유: {' · '.join(sorted(shared))[:88]}")
    if show_all:
        for key, shared in settled:
            print(f"  판정 완료  {key[0]}  ↔  {key[1]}")
            print(f"             {SETTLED[key][:88]}")
    # 더 이상 걸리지 않는 SETTLED 항목 — 소제목을 고쳐서 겹침이 풀린 쌍이다.
    # 되돌아가면 다시 걸리므로 기록은 남겨 둔다.
    if not chapters:
        gone = sorted(set(SETTLED) - {k for k, _ in settled})
        for key in gone:
            print(f"  이제 안 걸림  {key[0]}  ↔  {key[1]}")
    print(f"판정이 안 된 쌍 {len(fresh)} · 판정 완료 {len(settled)}"
          + (f" · 고쳐서 풀린 쌍 {len(gone)}" if not chapters else ""))
    return 1 if fresh else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
