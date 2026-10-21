r"""같은 자료를 비로 말할 때와 차로 말할 때.

이 쪽이 가르치는 것은 두 비율의 **차**에 대한 신뢰구간이다. $1995$ 년 영국의
피임약 경보는 같은 자료를 **비**로 말해서 생긴 일이고, 차로 말했으면 아무 일도
없었을 것이다.

① 같은 자료, 두 가지 수 — 상대 $100\%$ 와 절대 $1/7000$.
② 기저위험이 낮을수록 같은 비가 더 작은 차가 된다 — 그리고 NNT 가 커진다.
③ 그런데 차를 재려면 표본이 아주 커야 한다. 연구가 비를 보고하는 까닭이다.

만드는 파일: docs/ch08/two_sample_intervals/img/absolute_vs_relative.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

Z = stats.norm.isf(0.025)
P1, P2 = 1 / 7000, 2 / 7000        # 2 세대 / 3 세대
RR = P2 / P1
DIFF = P2 - P1
NNT = 1 / DIFF
print(f"① 기저 {P1 * 100000:.1f}/10 만, 3 세대 {P2 * 100000:.1f}/10 만")
print(f"   상대위험 {RR:.0f} 배 (+{(RR - 1) * 100:.0f}%), 위험차 {DIFF:.3e} "
      f"= {DIFF * 100000:.1f}/10 만")
print(f"   한 건을 더 만들려면 {NNT:,.0f} 명이 바꿔야 한다")

# === ② 기저위험을 바꾸며 ============================================
BASE = np.logspace(-5, -1.0, 200)
d2 = BASE * (RR - 1)
MARKS = [(P1, "피임약 경보\n($1/7{,}000$)"), (1e-3, "$1/1{,}000$"),
         (1e-2, "$1/100$"), (5e-2, "$1/20$")]

# === ③ 차를 재려면 표본이 얼마나 ====================================
def n_needed(p1, rr=RR, frac=0.5):
    """위험차의 95% 구간 반너비가 차의 frac 배가 되게 하는 집단당 n."""
    p2 = p1 * rr
    d = p2 - p1
    v = p1 * (1 - p1) + p2 * (1 - p2)
    return Z**2 * v / (frac * d) ** 2


NS = np.array([n_needed(b) for b in BASE])
n_pill = n_needed(P1)
print(f"\n③ 기저 {P1:.3e} 에서 차의 95% 구간 반너비를 차의 절반으로 줄이려면 "
      f"집단당 {n_pill:,.0f} 명")
for b, lab in MARKS:
    print(f"   기저 {b:.1e} → 집단당 {n_needed(b):,.0f} 명")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

# ---- ① 두 가지 말하기 ----------------------------------------------
ax1.bar([0, 1], [P1 * 100000, P2 * 100000], 0.5,
        color=["#1f77b4", "#d62728"])
for i, v in enumerate([P1 * 100000, P2 * 100000]):
    ax1.text(i, v + 0.9, f"${v:.1f}$", ha="center", fontsize=12)
ax1.annotate("", xy=(1.42, P2 * 100000), xytext=(1.42, P1 * 100000),
             arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.6))
ax1.text(1.52, (P1 + P2) / 2 * 100000,
         f"위험차\n${DIFF * 100000:.1f}$ / $10$ 만\n$= 1/{NNT:,.0f}$".replace(",", "{,}"),
         va="center", fontsize=11)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["$2$ 세대 알약", "$3$ 세대 알약"], fontsize=11)
ax1.set_ylabel("해마다 혈전이 생기는 사람 ($10$ 만 명당)")
ax1.set_title("① 같은 자료, 두 가지 수", fontsize=12)
ax1.set_xlim(-0.6, 2.9)
ax1.set_ylim(0, P2 * 100000 * 1.55)
ax1.text(0.5, P2 * 100000 * 1.34,
         f"비로 말하면 — **${RR:.0f}$ 배, $+{(RR - 1) * 100:.0f}\\%$**".replace("**", ""),
         ha="center", fontsize=12.5, color="#d62728",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec="#d62728"))
ax1.grid(alpha=0.25, axis="y")

# ---- ② 기저위험에 따라 ---------------------------------------------
ax2.loglog(BASE, d2, color="#d62728", lw=2.8)
for b, lab in MARKS:
    ax2.plot([b], [b * (RR - 1)], "o", color="#d62728", ms=9, zorder=5)
ax2.annotate(MARKS[0][1], xy=(P1, P1 * (RR - 1)), xytext=(2.0e-5, 1.1e-2),
             fontsize=10, color="#d62728", ha="left",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
# 나머지 점에는 글자를 붙이지 않는다 — 두 곡선이 지나는 자리라 겹친다.
# 기저위험은 가로축이 이미 말해 준다.
ax2.set_xlabel("$2$ 세대의 위험 (기저위험)")
ax2.set_ylabel("위험차 — 상대위험은 어디서나 $2$ 배다")
ax2.set_title("② 같은 「$2$ 배」가 주는 차는 기저가 정한다", fontsize=12)
ax2.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
ax2.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                     "$10^{-1}$"])
ax2.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
ax2.set_yticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                     "$10^{-1}$"])
ax2.minorticks_off()
ax2.grid(alpha=0.25, which="major")

ax2b = ax2.twinx()
ax2b.loglog(BASE, 1 / d2, color="#1f77b4", lw=2.0, ls="--")
ax2b.set_ylabel("한 건을 더 만드는 데 필요한 사람 수", color="#1f77b4")
ax2b.tick_params(axis="y", labelcolor="#1f77b4")
ax2b.set_yticks([1e1, 1e3, 1e5])
ax2b.set_yticklabels(["$10$", "$1{,}000$", "$100{,}000$"])
ax2b.minorticks_off()
ax2.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])   # twinx 가 되돌린 것을 다시
ax2.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                     "$10^{-1}$"])
ax2.minorticks_off()

# ---- ③ 차를 재는 값 ------------------------------------------------
ax3.loglog(BASE, NS, color="#2ca02c", lw=2.8)
ax3.plot([P1], [n_pill], "o", color="#d62728", ms=10, zorder=5)
ax3.annotate(f"피임약 경보의 기저에서\n집단당 ${n_pill:,.0f}$ 명".replace(",", "{,}"),
             xy=(P1, n_pill), xytext=(3.0e-4, 2.5e5), fontsize=10,
             color="#d62728", ha="left",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax3.set_xlabel("$2$ 세대의 위험 (기저위험)")
ax3.set_ylabel("집단당 필요한 인원")
ax3.set_title("③ 차를 제대로 재려면 — 구간 반너비를 차의 절반으로",
              fontsize=12)
ax3.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
ax3.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                     "$10^{-1}$"])
ax3.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6])
ax3.set_yticklabels(["$100$", "$1{,}000$", "$10^4$", "$10^5$", "$10^6$"])
ax3.minorticks_off()
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch08/two_sample_intervals/img/absolute_vs_relative.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
