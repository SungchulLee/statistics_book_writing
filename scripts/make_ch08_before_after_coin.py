"""8.1절 해석과 흔한 오해 — 확률은 뽑기 전에 있고, 뽑은 뒤에는 0 과 1 만 남는다.

들머리의 동전 비유를 그대로 그린다. mu = 100, sigma = 15, n = 25 에서
95% 구간을 20 만 번 만든다.

왼쪽은 자료를 뽑기 **전**이다. 구간의 끝점이 아직 확률변수이므로 「구간이
mu 를 담는다」는 것이 사건이고 그 확률이 0.95 다. 오른쪽은 뽑은 **뒤**다.
참값을 아는 우리가 매긴 점수는 0 아니면 1 이고, 0.95 가 적힌 줄은 한 개도
없다 --- 0.95 는 채점표의 평균이지 어느 줄의 값이 아니다.

쪽의 1 절 그림(구간을 쌓아 보이는 것)과 겹치지 않게, 여기서는 구간을
하나도 그리지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/foundations/img/"

MU, SIG, N, R = 100.0, 15.0, 25, 200_000
SE = SIG / np.sqrt(N)
Z = stats.norm.ppf(0.975)
HALF = Z * SE

rng = np.random.default_rng(20261101)
xbar = rng.normal(MU, SE, R)
score = ((xbar - HALF <= MU) & (MU <= xbar + HALF)).astype(int)
n1, n0 = int(score.sum()), int(R - score.sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))

# ── (좌) 뽑기 전 — 확률이 있는 자리 ────────────────────────────────
g = np.linspace(MU - 4 * SE, MU + 4 * SE, 800)
d = stats.norm.pdf(g, MU, SE)
ax1.plot(g, d, color="0.25", lw=2.0)
m = np.abs(g - MU) <= HALF
ax1.fill_between(g[m], d[m], color="tab:blue", alpha=0.35,
                 label="구간이 μ 를 담는다")
ax1.fill_between(g[g < MU - HALF], d[g < MU - HALF], color="tab:red", alpha=0.45)
ax1.fill_between(g[g > MU + HALF], d[g > MU + HALF], color="tab:red", alpha=0.45,
                 label="담지 못한다")
ax1.axvline(MU, color="0.15", lw=1.8)
ax1.text(MU, d.max() * 1.06, "μ = 100", ha="center", fontsize=11,
         fontweight="bold", color="0.15")
ax1.text(MU, d.max() * 0.42, "0.95", ha="center", va="center", fontsize=19,
         fontweight="bold", color="tab:blue")
for sgn in (-1, 1):
    ax1.annotate("0.025", xy=(MU + sgn * (HALF + 0.9), 0.006),
                 xytext=(MU + sgn * (HALF + 2.6), 0.035), fontsize=10.5,
                 color="tab:red", fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.4))
ax1.set_xlabel("표본평균이 떨어질 자리", fontsize=11)
ax1.set_ylabel("밀도", fontsize=11)
ax1.set_title("자료를 뽑기 전 — 여기에는 확률이 있다", fontsize=12.5)
ax1.set_ylim(0, d.max() * 1.18)
ax1.legend(fontsize=9.8, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.2)

# ── (우) 뽑은 뒤 — 채점표에 적히는 수 ──────────────────────────────
ax2.bar([0, 1], [n0, n1], width=0.055, color=["tab:red", "tab:blue"],
        alpha=0.85)
ax2.text(0, n0 + R * 0.022, f"{n0:,} 개", ha="center", fontsize=11.5,
         fontweight="bold", color="tab:red")
ax2.text(1, n1 + R * 0.022, f"{n1:,} 개", ha="left", fontsize=11.5,
         fontweight="bold", color="tab:blue")
ax2.text(0, -R * 0.055, "못 담았다", ha="center", va="top", fontsize=11,
         color="tab:red")
ax2.text(1.03, -R * 0.055, "담았다", ha="left", va="top", fontsize=11,
         color="tab:blue")
mean = score.mean()
ax2.plot([mean], [0], "x", ms=13, mew=2.6, color="0.15", zorder=5)
ax2.text(mean - 0.012, -R * 0.055, f"{mean:.4f}", ha="right", va="top",
         fontsize=10, color="0.15", fontweight="bold")
ax2.annotate(f"채점표의 평균은 {mean:.4f} 다.\n그런데 {mean:.4f} 가 적힌 줄은\n"
             f"{R:,} 개 가운데 한 개도 없다.",
             xy=(mean, R * 0.015), xytext=(0.52, R * 0.60),
             fontsize=11.5, fontweight="bold", color="0.15", ha="center",
             arrowprops=dict(arrowstyle="->", color="0.15", lw=1.8,
                             connectionstyle="arc3,rad=-0.18"))
ax2.set_xlim(-0.16, 1.22)
ax2.set_ylim(-R * 0.13, R * 1.12)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax2.set_yticks([0, 50_000, 100_000, 150_000, 200_000],
               ["0", "5만", "10만", "15만", "20만"])
ax2.set_xlabel("참값을 아는 우리가 매긴 점수", fontsize=11)
ax2.set_ylabel("구간의 개수", fontsize=11)
ax2.set_title(f"자료를 뽑은 뒤 — 0 과 1 만 남는다 ({R:,} 번)", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("확률은 던지기 전에 있던 것이고, 덮고 난 뒤에 남는 것은 "
             "「모른다」는 사실뿐이다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "before_after_coin.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nmu={MU}, sigma={SIG}, n={N}, 되풀이 {R:,}")
print(f"  구간의 반폭 {HALF:.4f}  폭 {2*HALF:.4f} (모두 같다)")
print(f"  담았다 {n1:,} · 못 담았다 {n0:,} · 평균 {mean:.4f}")
print(f"  점수에 적힌 서로 다른 수 {sorted(set(score.tolist()))}")
print(f"  0.95 가 적힌 줄 {(score == 0.95).sum()} 개")
