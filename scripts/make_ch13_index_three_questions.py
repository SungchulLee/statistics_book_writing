"""13장 장 개요 들머리 — 한 학과에서 묻는 세 가지 물음.

왜 이 그림인가. 장 개요가 웰던 이야기를 열고 13.2 적합도 검정의 들머리가
같은 이야기를 또 열고 있었다. 웰던은 13.2 가 통째로 맡는 편이 낫다 —
초상도 출처도 거기 있다. 장 개요는 대신 **이 장이 답하는 물음 셋**을
세운다. 책 어디에도 셋을 나란히 세운 자리가 없다. homogeneity.md 가
동질성과 독립성의 차이를 네 겹으로 다루지만 그것은 세 쪽을 지나야 닿는다.

소재를 셋 다 성적으로 고정하는 것이 요점이다. 동질성과 독립성은 산술이
완전히 같다 — 같은 X², 같은 E = RC/n, 같은 자유도 (r-1)(c-1). 다른 것은
표집 설계 하나뿐이다. 소재가 주사위·의료·마케팅으로 갈리면 독자는 차이를
소재 탓으로 돌리고 정작 가르는 것은 못 본다.

그래서 그림이 그리는 것은 자료가 아니라 **설계가 무엇을 미리 정했는가**다.
  (가) 적합도   — 비교 대상이 밖에서 온다. 정책 20/30/30/20 은 자료가 아니다
  (나) 동질성   — 과목을 정해 놓고 따로 표집했다. 행 합이 설계가 정한 값
  (다) 독립성   — 기록을 무작위로 뽑아 둘로 쪼갰다. 정해진 것은 총계뿐

자유도가 셋 다 다르게 나오는 것이 덤이다 — 3, 6, 9.

판정은 일부러 다 같지 않게 두었다. 셋 다 기각하면 「카이제곱은 늘 기각한다」로
읽힌다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch13/img/"
GRADES = ["A", "B", "C", "D"]
POLICY = np.array([0.20, 0.30, 0.30, 0.20])     # 학교가 정한 비율
rng = np.random.default_rng(1302)


def chi2_gof(obs, p):
    exp = obs.sum() * p
    x2 = ((obs - exp) ** 2 / exp).sum()
    df = len(obs) - 1
    return x2, df, stats.chi2.sf(x2, df), exp


def pad(s, width):
    """한글은 고정폭 글꼴에서 두 칸을 차지하므로 글자 수로 세면 표가 어긋난다."""
    w = sum(2 if ord(ch) > 0x2000 else 1 for ch in s)
    return s + " " * max(0, width - w)


def chi2_table(tab):
    """동질성과 독립성이 공유하는 바로 그 셈. 설계는 이 함수가 알 수 없다."""
    r = tab.sum(1, keepdims=True)
    c = tab.sum(0, keepdims=True)
    exp = r * c / tab.sum()
    x2 = ((tab - exp) ** 2 / exp).sum()
    df = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return x2, df, stats.chi2.sf(x2, df), exp


# ════════════════════════════════════════════════════════════════════
# 물음 1 — 적합도. 한 과목이 정책을 따랐는가
#   설계가 정한 것: 비교 대상(정책)과 수강생 수. 표는 한 줄뿐이다.
# ════════════════════════════════════════════════════════════════════
N1 = 120
TRUE1 = np.array([0.32, 0.34, 0.22, 0.12])      # 이 강사는 학점을 후하게 준다
obs1 = rng.multinomial(N1, TRUE1)
x2_1, df1, p1, exp1 = chi2_gof(obs1, POLICY)

print("물음 1 — 적합도: 통계학개론 한 과목이 정책을 따랐는가")
print(f"  정책      {'  '.join(f'{g} {q:4.0%}' for g, q in zip(GRADES, POLICY))}")
print(f"  관측 도수  {'    '.join(f'{g} {o:3d}' for g, o in zip(GRADES, obs1))}"
      f"   (합 {obs1.sum()})")
print(f"  기대 도수  {'    '.join(f'{g} {e:3.0f}' for g, e in zip(GRADES, exp1))}")
print(f"  X2 = {x2_1:.3f}, df = {df1}, p = {p1:.5f}")

# ════════════════════════════════════════════════════════════════════
# 물음 2 — 동질성. 같은 학과 세 과목의 학점분포가 같은가
#   설계가 정한 것: 과목 셋과 과목마다의 인원. 행 합이 자료가 아니다.
# ════════════════════════════════════════════════════════════════════
COURSES = ["통계학개론", "미적분학", "선형대수"]
N2 = 120                                        # 과목마다 같은 수로 뽑는다
TRUE2 = np.array([
    [0.14, 0.26, 0.34, 0.26],
    [0.20, 0.31, 0.30, 0.19],
])
# 첫 줄은 물음 1 의 그 과목 그대로다 — 다시 뽑으면 같은 과목이 두 얼굴이 된다
tab2 = np.vstack([obs1] + [rng.multinomial(N2, q) for q in TRUE2])
x2_2, df2, p2, exp2 = chi2_table(tab2)

print("\n물음 2 — 동질성: 세 과목의 학점분포가 같은가")
print("                  A    B    C    D   합    ← 합은 설계가 정했다")
for name, row in zip(COURSES, tab2):
    print("  " + pad(name, 14) + "".join(f"{v:5d}" for v in row)
          + f"{row.sum():5d}")
print(f"  X2 = {x2_2:.3f}, df = {df2}, p = {p2:.5f}")

# ════════════════════════════════════════════════════════════════════
# 물음 3 — 독립성. 학과와 학점이 서로 얽혀 있는가
#   설계가 정한 것: 뽑는 건수 하나뿐. 학과별 건수도 학점별 건수도 자료가 정한다.
# ════════════════════════════════════════════════════════════════════
DEPTS = ["공학", "인문", "자연", "경영"]
N3 = 600
SHARE = np.array([0.31, 0.22, 0.26, 0.21])      # 학교 안 학과 비중
# 학점은 학과와 거의 무관하게 매겨진다고 두었다 — 여기서는 기각되지 않아야 한다
BASE = np.array([0.22, 0.31, 0.29, 0.18])
joint = SHARE[:, None] * BASE[None, :]
flat = rng.multinomial(N3, joint.ravel())
tab3 = flat.reshape(len(DEPTS), len(GRADES))
x2_3, df3, p3, exp3 = chi2_table(tab3)

print("\n물음 3 — 독립성: 학과와 학점이 독립인가")
print("                  A    B    C    D   합    ← 합까지 자료가 정했다")
for name, row in zip(DEPTS, tab3):
    print("  " + pad(name, 14) + "".join(f"{v:5d}" for v in row)
          + f"{row.sum():5d}")
print("  " + pad("합", 14) + "".join(f"{v:5d}" for v in tab3.sum(0))
      + f"{tab3.sum():5d}")
print(f"  X2 = {x2_3:.3f}, df = {df3}, p = {p3:.5f}")

print("\n자유도 셋:", df1, df2, df3)
print("판정:", " · ".join(f"{'기각' if p < 0.05 else '기각 못함'}"
                         for p in (p1, p2, p3)))

# ════════════════════════════════════════════════════════════════════
# 그림 — 세 칸. 그리는 것은 자료가 아니라 「설계가 무엇을 고정했는가」다
# ════════════════════════════════════════════════════════════════════
FIXED = "#c0392b"       # 미리 정해 둔 것
FREE = "#2c7fb8"        # 자료가 정한 것

fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.8))
x = np.arange(len(GRADES))

# 셋 다 「줄 안에서의 비율」로 그린다. 도수로 그리면 (다)에서 학과마다 건수가
# 달라 독립인데도 막대 높이가 갈리고, 독자가 그것을 「얽혀 있다」로 읽는다.
pct1 = obs1 / obs1.sum() * 100
pct2 = tab2 / tab2.sum(1, keepdims=True) * 100
pct3 = tab3 / tab3.sum(1, keepdims=True) * 100

# ── (가) 적합도 ──────────────────────────────────────────────────
ax = axes[0]
ax.bar(x, pct1, 0.62, color=FREE, label="이 과목에서 실제로 나온 학점")
ax.plot(x, POLICY * 100, "D--", color=FIXED, lw=2.4, ms=11,
        label="정책이 말하는 비율 (밖에서 왔다)")
for xi, (o, e) in enumerate(zip(obs1, exp1)):
    ax.text(xi, max(pct1[xi], POLICY[xi] * 100) + 1.6, f"{o - e:+.0f}명",
            ha="center", fontsize=10.6,
            color=FIXED if abs(o - e) > 8 else "0.45", fontweight="bold")
ax.set_ylim(0, 50)
ax.set_ylabel("비율 (%)", fontsize=11.5)
ax.legend(fontsize=10.0, loc="upper center", framealpha=0.95)
ax.set_title("(가) 이 과목은 정책을 따랐는가\n적합도 검정", fontsize=12.8)

# ── (나) 동질성 ──────────────────────────────────────────────────
ax = axes[1]
w = 0.26
for i, (name, row) in enumerate(zip(COURSES, pct2)):
    ax.bar(x + (i - 1) * w, row, w, label=f"{name} (120명)")
ax.set_ylim(0, 50)
ax.legend(fontsize=10.0, loc="upper center", ncol=3, framealpha=0.95,
          title="과목마다 따로 뽑았다", title_fontsize=10.0)
ax.set_title("(나) 세 과목의 분포가 서로 같은가\n동질성 검정", fontsize=12.8)

# ── (다) 독립성 ──────────────────────────────────────────────────
ax = axes[2]
w = 0.20
for i, (name, row) in enumerate(zip(DEPTS, pct3)):
    ax.bar(x + (i - 1.5) * w, row, w,
           label=f"{name} ({tab3[i].sum()}건)")
ax.set_ylim(0, 50)
ax.legend(fontsize=10.0, loc="upper center", ncol=2, framealpha=0.95,
          title="한 번에 600 건을 뽑았다", title_fontsize=10.0)
ax.set_title("(다) 학과와 학점이 얽혀 있는가\n독립성 검정", fontsize=12.8)

for ax in axes:
    ax.set_xticks(x, GRADES, fontsize=12)

# ── 칸마다 「미리 정해 둔 것」과 셈의 결과 ────────────────────────
def pstr(p):
    """p = 0.0000 으로 찍히면 0 과 구별되지 않는다."""
    return f"p = {p:.4f}" if p >= 0.0001 else f"p = {p:.1e}"


NOTE = [
    ("미리 정해 둔 것 — 정책 20·30·30·20 과 수강생 120 명",
     f"X² = {x2_1:.2f}   df = {df1}   {pstr(p1)}",
     "기각" if p1 < 0.05 else "기각 못함"),
    ("미리 정해 둔 것 — 과목 셋과 과목마다 120 명 (행 합)",
     f"X² = {x2_2:.2f}   df = {df2}   {pstr(p2)}",
     "기각" if p2 < 0.05 else "기각 못함"),
    ("미리 정해 둔 것 — 뽑는 건수 600 뿐 (행도 열도 자료가 정했다)",
     f"X² = {x2_3:.2f}   df = {df3}   {pstr(p3)}",
     "기각" if p3 < 0.05 else "기각 못함"),
]
for ax, (fixed, stat, verdict) in zip(axes, NOTE):
    ax.set_xlabel("학점", fontsize=11.5)
    ax.text(0.5, -0.185, fixed, transform=ax.transAxes, ha="center",
            fontsize=10.6, color=FIXED, fontweight="bold")
    ax.text(0.5, -0.255, f"{stat}    →  {verdict}", transform=ax.transAxes,
            ha="center", fontsize=11.0, color="0.15")
    ax.grid(alpha=0.22, axis="y")

fig.suptitle("같은 학과, 같은 학점 — 물음 셋은 자료가 아니라 "
             "「무엇을 미리 정했는가」로 갈린다", fontsize=14.6, y=0.995)
plt.tight_layout(rect=(0, 0.055, 1, 1))
path = OUT + "three_questions_one_setting.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("\nsaved", path)
