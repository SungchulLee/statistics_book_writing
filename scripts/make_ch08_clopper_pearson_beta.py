r"""클로퍼–피어슨 구간이 자르는 **두 개의 베타분포**.

공식이 헷갈리는 까닭은 한 분포를 두 번 자르는 것처럼 보이기 때문이다.
실제로는 **서로 다른 두 베타**를 각각 **한 번씩** 자른다.

    하한 $p_L$ = Beta$(\alpha/2;\ k,\ n-k+1)$     ← 모수가 $(k,\ n-k+1)$
    상한 $p_U$ = Beta$(1-\alpha/2;\ k+1,\ n-k)$   ← 모수가 $(k+1,\ n-k)$

모수가 양쪽에서 **한 칸씩 어긋나 있는 것**이 요점이고, 그것이 이산성을
떠안는 대가다. 보기 5 와 같은 자료($n = 50$, $k = 12$)를 쓴다.

① 왼쪽 — Beta$(k,\ n-k+1)$ 의 **왼쪽** 꼬리 $\alpha/2$ 를 잘라 $p_L$ 을 얻는다.
② 오른쪽 — Beta$(k+1,\ n-k)$ 의 **오른쪽** 꼬리 $\alpha/2$ 를 잘라 $p_U$ 를 얻는다.
③ 아래 — 두 수가 만드는 구간을 $\hat p$ 와 함께 수직선에 놓는다.

만드는 파일: docs/ch08/one_sample_intervals/img/clopper_pearson_beta.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

OUT = "docs/ch08/one_sample_intervals/img/"

N, K, ALPHA = 50, 12, 0.05
PHAT = K / N

LOW = stats.beta(K, N - K + 1)          # 하한을 주는 분포
UPP = stats.beta(K + 1, N - K)          # 상한을 주는 분포
P_L = LOW.ppf(ALPHA / 2)
P_U = UPP.ppf(1 - ALPHA / 2)

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#4a5568"


def panel(ax, dist, cut, tail, title, a, b):
    """베타 밀도를 그리고 한쪽 꼬리를 칠한다. tail 은 'left' 또는 'right'."""
    x = np.linspace(0, 0.6, 1200)
    y = dist.pdf(x)
    ax.plot(x, y, color=GREY, lw=2)

    m = x <= cut if tail == "left" else x >= cut
    ax.fill_between(x[m], 0, y[m], color=RED, alpha=0.45)

    ax.axvline(cut, color=RED, lw=1.6, ls="--")
    ax.axvline(PHAT, color=BLUE, lw=1.4, ls=":")

    ymax = y.max()
    # 꼬리 넓이와 절단값을 한 덩이로 꼬리 쪽 여백에 적는다. 축 아래에 따로 적으면
    # 눈금 글자와 겹친다. mathtext 에서 맨 % 는 터지므로 「퍼센트」를 쓰지 않는다
    tx = cut - 0.085 if tail == "left" else cut + 0.085
    ax.annotate(f"꼬리 넓이 {ALPHA/2:.3f}\n$p$ = {cut:.4f}",
                xy=(cut, ymax * 0.10), xytext=(tx, ymax * 0.50),
                ha="center", fontsize=9.5, color=RED, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.text(PHAT, ymax * 1.02, f"$\\hat p$ = {PHAT:.2f}", ha="center",
            fontsize=9, color=BLUE)

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(f"Beta({a}, {b})", fontsize=10)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-ymax * 0.16, ymax * 1.16)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


fig = plt.figure(figsize=(11.5, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.0], hspace=0.62, wspace=0.18)

ax1 = fig.add_subplot(gs[0, 0])
panel(ax1, LOW, P_L, "left",
      "하한 — 왼쪽 꼬리를 자른다", K, N - K + 1)

ax2 = fig.add_subplot(gs[0, 1])
panel(ax2, UPP, P_U, "right",
      "상한 — 오른쪽 꼬리를 자른다", K + 1, N - K)

# ③ 두 수가 만드는 구간
ax3 = fig.add_subplot(gs[1, :])
ax3.hlines(0, 0, 0.6, color="#cbd5e0", lw=1)
ax3.hlines(0, P_L, P_U, color=BLUE, lw=5, alpha=0.75)
for v in (P_L, P_U):
    ax3.vlines(v, -0.22, 0.22, color=RED, lw=2)
ax3.plot([PHAT], [0], "o", color=BLUE, ms=8)
ax3.text(PHAT, 0.34, f"$\\hat p$ = {PHAT:.2f}", ha="center", fontsize=10, color=BLUE)
ax3.text(P_L, -0.46, f"$p_L$ = {P_L:.4f}", ha="center", va="top", fontsize=10, color=RED)
ax3.text(P_U, -0.46, f"$p_U$ = {P_U:.4f}", ha="center", va="top", fontsize=10, color=RED)
# 너비는 위쪽 줄에 따로 둔다 — 중점이 p̂ 과 가까워 같은 높이에 두면 겹친다
ax3.text((P_L + P_U) / 2, 1.05, f"너비 {P_U - P_L:.4f}",
         ha="center", fontsize=9.5, color=GREY)
ax3.set_xlim(0, 0.6)
ax3.set_ylim(-1.0, 1.35)
ax3.set_yticks([])
ax3.set_xlabel("$p$", fontsize=11)
for s in ("top", "right", "left"):
    ax3.spines[s].set_visible(False)

fig.suptitle(f"클로퍼–피어슨: 서로 다른 두 베타를 각각 한 번씩 자른다 "
             f"(n = {N}, k = {K}, 95 퍼센트)", fontsize=12.5, y=0.985)
fig.savefig(OUT + "clopper_pearson_beta.png", dpi=150, bbox_inches="tight")
print(f"p_L = {P_L:.6f}   p_U = {P_U:.6f}   너비 = {P_U - P_L:.6f}")
print(f"하한 분포 Beta({K}, {N-K+1})   상한 분포 Beta({K+1}, {N-K})")
print("saved", OUT + "clopper_pearson_beta.png")
