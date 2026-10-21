r"""$1747$ 년 린드의 두 집단 비교를 지금 눈금으로 재어 본다.

배 위에서 괴혈병 환자 $12$ 명을 여섯 쌍으로 갈라 여섯 가지 처방을 주었고,
감귤을 받은 쌍만 나았다. 치료 $2$ 명 가운데 $2$ 명, 나머지 $10$ 명 가운데 $0$ 명이다.

① 린드의 표.
② 그 표의 피셔 정확검정 — 대조군이 몇이어야 $0.05$ 를 넘는가.
③ 효과가 덜 극적이었다면 쌍 하나로는 아무것도 못 잡는다.

만드는 파일: docs/ch09/two_sample_tests/img/lind1747.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from math import comb  # noqa: E402
from scipy.stats import binom, fisher_exact  # noqa: E402

TREATS = [
    ("사과주 한 쿼트", 0),
    ("황산 엘릭시르", 0),
    ("식초 두 숟갈", 0),
    ("바닷물 반 파인트", 0),
    ("감귤 — 오렌지 둘, 레몬 하나", 2),
    ("약초 반죽과 보리물", 0),
]
N_PAIR = 2
CITRUS = 4   # 다섯째 줄

# === ② 완전분리일 때 얻을 수 있는 최선의 p ============================
# 치료군 k 명이 모두 낫고 대조군 m 명이 모두 낫지 않으면, 초기하분포에서
# 그보다 극단적인 표가 없으므로 단측 p = C(k,k)C(m,0)/C(k+m,k) 다.
K = 2
MS = np.arange(1, 21)
best_p = np.array([comb(K, K) * comb(m, 0) / comb(K + m, K) for m in MS],
                  dtype=float)
m_lind = 10
p_lind = comb(K, K) / comb(K + m_lind, K)
m_need = int(MS[np.argmax(best_p <= 0.05)])
print(f"② 치료 {K} 명이 모두 낫고 대조 m 명이 모두 안 나을 때의 단측 p")
for m in (1, 2, 4, 5, 6, 10, 20):
    print(f"   m = {m:>2}  p = {comb(K, K) / comb(K + m, K):.4f}")
print(f"   린드의 m = {m_lind} 이면 p = 1/{comb(K + m_lind, K)} = {p_lind:.4f}")
print(f"   0.05 를 넘으려면 대조군이 최소 {m_need} 명")

# 확인: 피셔 정확검정이 같은 값을 주는가
tbl = [[K, 0], [0, m_lind]]
_, p_fisher = fisher_exact(tbl, alternative="greater")
print(f"   scipy 피셔 정확검정(단측) = {p_fisher:.4f}  (손으로 센 값과 같은가: "
      f"{abs(p_fisher - p_lind) < 1e-12})")


# === ③ 효과가 덜 극적이었다면 — 정확 검정력 ==========================
def power_fisher(n, p1, p2, alpha=0.05):
    """두 집단이 각각 n 명일 때 피셔 정확검정(단측)의 정확한 검정력."""
    x = np.arange(n + 1)
    w1 = binom.pmf(x, n, p1)
    w2 = binom.pmf(x, n, p2)
    tot = 0.0
    for a in x:
        for b in x:
            if w1[a] * w2[b] < 1e-15:
                continue
            _, pv = fisher_exact([[a, n - a], [b, n - b]], alternative="greater")
            # p 가 정확히 alpha 인 표가 생긴다(예: 3 대 3 완전분리의 1/20).
            # 관례대로 p <= alpha 를 기각으로 세되 부동소수점 여유를 둔다.
            if pv <= alpha + 1e-12:
                tot += w1[a] * w2[b]
    return tot


NS = np.array([2, 3, 5, 8, 12, 20, 30])
SCEN = [("$0.9$ 대 $0.1$ — 린드만큼 극적", 0.9, 0.1, "#d62728"),
        ("$0.7$ 대 $0.3$ — 큰 효과", 0.7, 0.3, "#ff7f0e"),
        ("$0.5$ 대 $0.3$ — 보통 효과", 0.5, 0.3, "#1f77b4")]
POW = {}
for lab, p1, p2, c in SCEN:
    POW[lab] = np.array([power_fisher(int(n), p1, p2) for n in NS])
    print(f"③ {lab}: " + ", ".join(f"n={n} {v * 100:.0f}%"
                                   for n, v in zip(NS, POW[lab])))

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.9))

for i, (name, rec) in enumerate(TREATS):
    y = len(TREATS) - 1 - i
    for j in range(N_PAIR):
        ok = j < rec
        ax1.scatter([j], [y], s=330, marker="o",
                    color="#2ca02c" if ok else "#d9d9d9",
                    edgecolors="#555555", linewidths=1.0, zorder=3)
        ax1.text(j, y, "나음" if ok else "", ha="center", va="center",
                 fontsize=9, color="white", zorder=4)
    ax1.text(-0.42, y, name, ha="right", va="center",
             fontsize=10.5, fontweight="bold" if i == CITRUS else "normal",
             color="#2ca02c" if i == CITRUS else "#333333")
ax1.set_xlim(-3.3, 1.5)
ax1.set_ylim(-0.7, len(TREATS) - 0.3)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["환자 $1$", "환자 $2$"], fontsize=10)
ax1.set_yticks([])
ax1.set_title("① $1747$ 년 $5$ 월 $20$ 일, 군함 솔즈베리", fontsize=12)
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)

ax2.plot(MS, best_p, "o-", color="#1f77b4", lw=2.4, ms=6)
ax2.axhline(0.05, color="black", lw=1.5, ls="--")
ax2.text(20, 0.058, "$\\alpha = 0.05$", ha="right", fontsize=10)
ax2.plot([m_lind], [p_lind], "o", color="#d62728", ms=11, zorder=5)
ax2.annotate(f"린드의 대조군 ${m_lind}$ 명\n$p = 1/{comb(K + m_lind, K)} = {p_lind:.4f}$",
             xy=(m_lind, p_lind), xytext=(12.5, 0.24), fontsize=10,
             color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.plot([2], [best_p[1]], "o", color="#ff7f0e", ms=10, zorder=5)
ax2.annotate(f"쌍 하나하고만 견주면\n$p = 1/6 = {best_p[1]:.3f}$",
             xy=(2, best_p[1]), xytext=(3.4, 0.60), fontsize=10, color="#ff7f0e",
             arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.0))
ax2.axvline(m_need, color="#999999", lw=1.2, ls=":")
ax2.text(m_need + 0.4, 0.26, f"대조군 ${m_need}$ 명부터\n$0.05$ 아래", fontsize=9.5,
         color="#555555")
ax2.set_xlabel("대조군 인원 $m$ (치료군은 $2$ 명 고정)")
ax2.set_ylabel("완전분리일 때 얻을 수 있는 최선의 단측 $p$")
ax2.set_title("② 대조군의 크기가 검정력을 만든다", fontsize=12)
ax2.set_xticks([1, 5, 10, 15, 20])
ax2.set_ylim(0, 0.72)
ax2.grid(alpha=0.25)

for (lab, p1, p2, c) in SCEN:
    ax3.plot(NS, POW[lab] * 100, "o-", color=c, lw=2.4, ms=6, label=lab)
ax3.axhline(80, color="black", lw=1.3, ls="--")
ax3.text(30, 82, "검정력 $80\\%$", ha="right", fontsize=10)
ax3.axvline(2, color="#999999", lw=1.4, ls=":")
ax3.text(2.35, 60, "린드의\n집단 크기", fontsize=9.5, color="#555555")
ax3.set_xlabel("집단당 인원 $n$")
ax3.set_ylabel("$\\alpha = 0.05$ 단측에서 잡아낼 확률 ($\\%$)")
ax3.set_title("③ 효과가 덜 극적이었다면", fontsize=12)
ax3.set_xscale("log")
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${n}$" for n in NS])
ax3.minorticks_off()
ax3.set_ylim(0, 103)
ax3.legend(loc="upper left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch09/two_sample_tests/img/lind1747.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
