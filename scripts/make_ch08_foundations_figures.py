r"""8장 신뢰구간 기초 두 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch08/foundations/img/interval_moves_not_mu.png      움직이는 것은 끝점이지 모수가 아니다

실행:  python3 scripts/make_ch08_foundations_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch08/foundations/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
def fig_interval_moves_not_mu():
    rng = np.random.default_rng(42)
    mu, sigma, n = 50.0, 10.0, 30
    me = 1.96 * sigma / np.sqrt(n)          # 3.578 — 표본과 무관한 상수

    n_sim = 1000
    xbar = rng.normal(mu, sigma / np.sqrt(n), n_sim)
    covers = (xbar - me <= mu) & (mu <= xbar + me)
    print(f"  coverage {covers.sum()}/{n_sim} = {covers.mean():.3f}   me={me:.3f}")

    k = 32                                   # 위에서 k개만 그린다
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=False)

    # --- 왼쪽: mu 는 못 박힌 선, 구간이 움직인다 ---
    axL.axvline(mu, color=INK, lw=2.0, zorder=3)
    axL.text(mu + 0.4, k + 2.3, r"참값 $\mu=50$ (고정)", fontsize=11,
             color=INK, ha="left", va="center")
    miss_y = None
    for i in range(k):
        y = k - i
        c = BLUE if covers[i] else RED
        axL.plot([xbar[i] - me, xbar[i] + me], [y, y], color=c,
                 lw=(2.0 if not covers[i] else 1.6), zorder=(4 if not covers[i] else 2))
        axL.plot([xbar[i]], [y], marker="o", ms=3.4, color=c,
                 zorder=(5 if not covers[i] else 3))
        if not covers[i]:
            miss_y = y
    if miss_y is not None:
        axL.annotate(
            "참값을 놓친 구간", xy=(xbar[k - miss_y] + me + 0.15, miss_y),
            xytext=(mu + 2.4, -0.7),
            fontsize=10.5, color=RED, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2,
                            shrinkA=2, shrinkB=3),
        )
    axL.set_xlim(mu - 11.0, mu + 11.0)
    axL.set_ylim(-1.4, k + 3.2)
    axL.set_yticks([])
    axL.set_xlabel("값", fontsize=10.5, color=INK)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.spines["bottom"].set_color(MUTED)
    axL.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    axL.set_title(f"구간 {k}개를 그대로 쌓았다 — 흔들리는 것은 끝점이다",
                  fontsize=12, color=INK, pad=10)
    axL.text(mu - 10.6, k + 2.3,
             f"전체 1000개 중 {covers.sum()}개가 참값을 담았다",
             fontsize=10.5, color=MUTED, ha="left", va="center")

    # --- 오른쪽: 같은 자료를 구간 기준으로 다시 그린다 ---
    axR.axvspan(-me, me, color=BLUE_F, zorder=0)
    axR.axvline(0, color=MUTED, lw=1.1, ls=(0, (5, 4)), zorder=1)
    for e in (-me, me):
        axR.axvline(e, color=BLUE, lw=1.6, zorder=2)
    rel = mu - xbar                          # 구간 중심에서 본 mu 의 위치
    yj = rng.uniform(0.12, 0.92, n_sim)
    inside = np.abs(rel) <= me
    axR.scatter(rel[inside], yj[inside], s=9, color=BLUE, alpha=0.40,
                linewidths=0, zorder=3)
    axR.scatter(rel[~inside], yj[~inside], s=13, color=RED, alpha=0.85,
                linewidths=0, zorder=4)
    axR.text(0, 1.055, "구간의 중심", fontsize=10.5, color=MUTED,
             ha="center", va="center")
    axR.text(-me - 0.35, 1.055, r"$\bar X-3.58$", fontsize=10.5, color=BLUE,
             ha="right", va="center")
    axR.text(me + 0.35, 1.055, r"$\bar X+3.58$", fontsize=10.5, color=BLUE,
             ha="left", va="center")
    axR.text(-9.3, 1.16, "점 하나가 표본 하나", fontsize=10.5, color=INK,
             ha="left", va="center")
    axR.annotate(
        f"바깥의 빨간 점 {int((~inside).sum())}개",
        xy=(-5.4, 0.30), xytext=(-9.2, 0.13),
        fontsize=10.5, color=RED, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.2, shrinkA=2, shrinkB=3),
    )
    axR.set_xlim(-9.5, 9.5)
    axR.set_ylim(0, 1.22)
    axR.set_yticks([])
    axR.set_xlabel(r"구간의 중심에서 본 $\mu$ 의 위치", fontsize=10.5, color=INK)
    axR.spines[["top", "right", "left"]].set_visible(False)
    axR.spines["bottom"].set_color(MUTED)
    axR.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    axR.set_title("같은 1000개를 구간 기준으로 다시 그렸다",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.4)
    save(fig, "interval_moves_not_mu.png")


if __name__ == "__main__":
    fig_interval_moves_not_mu()
