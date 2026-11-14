"""9.1절 신뢰구간과 검정의 쌍대성 §4 — 단측검정의 짝은 한쪽만 막은 구간이다.

보기 4 의 자료(x̄ = 52, σ = 10, n = 25, α = 0.05)를 그대로 쓴다. μ0 를 44 에서
60 까지 바꿔 가며 양측검정과 우측검정(H1: μ > μ0)을 돌리고, 기각하지 못한
μ0 를 모은다. 양측은 (48.08, 55.92), 우측은 (48.71, ∞) 가 된다.

들머리의 ruler_of_tests 는 양측 한 줄만 그린다. 여기서는 그 줄 아래에
우측검정의 줄을 놓아 하한이 올라오고 상한이 사라지는 것을 보인다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/foundations/img/"
XBAR, SIGMA, N, ALPHA = 52, 10, 25, 0.05
SE = SIGMA / np.sqrt(N)
Z2 = stats.norm.ppf(1 - ALPHA / 2)
Z1 = stats.norm.ppf(1 - ALPHA)
LO2, HI2 = XBAR - Z2 * SE, XBAR + Z2 * SE
LO1 = XBAR - Z1 * SE
print(f"양측 구간 ({LO2:.6f}, {HI2:.6f})   하한 신뢰한계 {LO1:.6f}")

grid = np.arange(44.0, 60.01, 0.5)
XMAX = 60.6

fig, ax = plt.subplots(figsize=(13.0, 4.6))
ROWS = [
    (1.0, "양측검정  $H_a$: $\\mu \\neq \\mu_0$", lambda m: abs((XBAR - m) / SE) > Z2,
     LO2, HI2, "tab:blue"),
    (0.0, "우측검정  $H_a$: $\\mu > \\mu_0$", lambda m: (XBAR - m) / SE > Z1,
     LO1, None, "tab:green"),
]
for y, name, rejects, lo, hi, col in ROWS:
    for m in grid:
        if rejects(m):
            ax.plot(m, y, "x", ms=8, mew=2.0, color="tab:red")
        else:
            ax.plot(m, y, "o", ms=8, mfc="white", mec=col, mew=2.0)
    yb = y - 0.3
    right = hi if hi is not None else XMAX - 0.25
    ax.plot([lo, right], [yb, yb], "-", lw=7, color=col, solid_capstyle="butt")
    ax.plot([lo, lo], [yb - 0.08, yb + 0.08], "-", lw=3, color=col)
    ax.text(lo, yb - 0.13, f"{lo:.2f}", ha="center", va="top", fontsize=11,
            color=col, fontweight="bold")
    if hi is not None:
        ax.plot([hi, hi], [yb - 0.08, yb + 0.08], "-", lw=3, color=col)
        ax.text(hi, yb - 0.13, f"{hi:.2f}", ha="center", va="top",
                fontsize=11, color=col, fontweight="bold")
        ax.text((lo + hi) / 2, yb + 0.07, "95% 신뢰구간", ha="center",
                va="bottom", fontsize=10.5, color=col)
    else:
        ax.annotate("", xy=(XMAX, yb), xytext=(XMAX - 0.6, yb),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
        ax.text((lo + XMAX) / 2, yb + 0.07, "95% 하한 신뢰한계 — 상한이 없다",
                ha="center", va="bottom", fontsize=10.5, color=col)
    ax.text(43.6, y, name, ha="right", va="center", fontsize=11.5,
            fontweight="bold")

ax.axvline(XBAR, color="0.4", lw=1.2, ls=":")
ax.text(XBAR, 1.33, f"표본평균 {XBAR}", ha="center", fontsize=10.5, color="0.3")
ax.axvspan(LO2, LO1, color="0.88", zorder=0)
ax.text((LO2 + LO1) / 2, 1.33, "양측은 기각 못함\n우측은 기각", ha="center",
        va="center", fontsize=9.5, color="0.2")
ax.plot([], [], "x", ms=8, mew=2.0, color="tab:red", label="기각한다")
ax.plot([], [], "o", ms=8, mfc="white", mec="0.3", mew=2.0, label="기각하지 못한다")
ax.legend(loc="upper right", fontsize=10, frameon=False, ncol=2,
          bbox_to_anchor=(1.0, 1.0))
ax.set_xlim(38.3, XMAX + 0.2)
ax.set_ylim(-0.7, 1.6)
ax.set_yticks([])
ax.set_xticks(np.arange(44, 61, 2))
ax.set_xlabel("검정이 묻는 값 $\\mu_0$", fontsize=11)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT + "one_sided_bound.png", dpi=150)
print("saved", OUT + "one_sided_bound.png")
