"""8.3절 p₁−p₂ 의 신뢰구간 — 같은 두 수, 두 개의 자.

들머리의 요점 하나를 그린다. 화이자 시험의 두 감염률은 0.0459% 와
0.9251% 다. 같은 두 수인데 **선형 자에 올리면 차가 보이고 비는 안 보이며,
로그 자에 올리면 비가 보이고 차는 안 보인다.** 그래서 하나는 0.88
퍼센트포인트이고 하나는 95% 다.

쪽의 absolute_vs_relative 는 1995 년 경구피임약 경보를 자료로 「비로 말할
때와 차로 말할 때」를 다루므로, 여기서는 그 일화를 건드리지 않고 백신
자료만 두 눈금에 올린다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/two_sample_intervals/img/"

N1, K1 = 17411, 8        # 백신군
N2, K2 = 17511, 162      # 위약군
Z = stats.norm.ppf(0.975)

p1, p2 = K1 / N1, K2 / N2
diff = 100 * (p1 - p2)
se = np.sqrt(p1 * (1 - p1) / N1 + p2 * (1 - p2) / N2)
dlo, dhi = diff - 100 * Z * se, diff + 100 * Z * se
ratio = p1 / p2
ve = 100 * (1 - ratio)

LAB = ["백신군", "위약군"]
VAL = [100 * p1, 100 * p2]
COL = ["tab:blue", "tab:red"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))

# ── (좌) 선형 자 — 차가 보인다 ─────────────────────────────────────
ax1.bar([0, 1], VAL, width=0.42, color=COL, alpha=0.85)
for x, v in zip([0, 1], VAL):
    ax1.text(x, v + 0.022, f"{v:.4f}%", ha="center", fontsize=11.5,
             fontweight="bold", color=COL[x])
ax1.annotate("", xy=(1.42, VAL[1]), xytext=(1.42, VAL[0]),
             arrowprops=dict(arrowstyle="<->", color="0.2", lw=2.0))
ax1.text(1.48, sum(VAL) / 2, f"차 {abs(diff):.4f} 퍼센트포인트\n"
         f"95% 구간 ({dlo:.4f}, {dhi:.4f})", ha="left", va="center",
         fontsize=11, fontweight="bold", color="0.2")
ax1.set_xticks([0, 1], LAB, fontsize=12)
ax1.set_xlim(-0.55, 2.55)
ax1.set_ylim(0, 1.18)
ax1.set_ylabel("감염된 비율 (%)", fontsize=11)
ax1.set_title("선형 자 — 차가 보이고 비는 안 보인다", fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)

# ── (우) 로그 자 — 비가 보인다 ─────────────────────────────────────
ax2.bar([0, 1], VAL, width=0.42, color=COL, alpha=0.85)
for x, v in zip([0, 1], VAL):
    ax2.text(x, v * 1.14, f"{v:.4f}%", ha="center", fontsize=11.5,
             fontweight="bold", color=COL[x])
ax2.set_yscale("log")
ax2.set_ylim(0.012, 4.0)
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에 없는 U+2212 를 부른다.
# 눈금을 직접 적어 막는다.
TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
ax2.set_yticks(TICKS, [f"{t:g}" for t in TICKS])
ax2.set_yticks([], minor=True)
ax2.annotate("", xy=(1.42, VAL[1]), xytext=(1.42, VAL[0]),
             arrowprops=dict(arrowstyle="<->", color="0.2", lw=2.0))
ax2.text(1.48, np.sqrt(VAL[0] * VAL[1]),
         f"비 {1/ratio:.2f} 배\n곧 효과 {ve:.2f}%", ha="left", va="center",
         fontsize=11, fontweight="bold", color="0.2")
ax2.set_xticks([0, 1], LAB, fontsize=12)
ax2.set_xlim(-0.55, 2.55)
ax2.set_ylabel("감염된 비율 (%, 로그 눈금)", fontsize=11)
ax2.set_title("로그 자 — 비가 보이고 차는 안 보인다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25, which="both")

fig.suptitle("같은 두 수인데 어느 자에 올리느냐가 크기를 정한다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "same_numbers_two_rulers.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n백신군 {K1}/{N1:,} = {100*p1:.6f}% · 위약군 {K2}/{N2:,} = "
      f"{100*p2:.6f}%")
print(f"  차 {diff:.6f} 퍼센트포인트 · 95% 구간 ({dlo:.4f}, {dhi:.4f})")
print(f"  비 {ratio:.6f} (= 1/{1/ratio:.4f}) · 효과 {ve:.4f}%")
