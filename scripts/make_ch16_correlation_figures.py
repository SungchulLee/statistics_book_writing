r"""16.6 순위상관 두 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch16/correlation/img/spearman_rank_straightens.png  순위로 바꾸면 곡선이 펴진다
  ch16/correlation/img/kendall_pairs.png              tau 는 쌍의 기울기를 센다

실행:  python3 scripts/make_ch16_correlation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import itertools
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

OUT = "docs/ch16/correlation/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9.5, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 순위로 바꾸면 곡선이 펴진다  (spearman.md)
# ==================================================================
def spearman_rank_straightens():
    rng = np.random.default_rng(12)
    n = 30

    # (가) 단조이지만 선형이 아닌 관계
    x1 = np.sort(rng.uniform(0.2, 5.0, n))
    y1 = np.exp(2.2 * x1) + rng.normal(0, np.exp(2.2 * 2.5) * 0.04, n)

    # (나) 선형 관계에 이상치 하나
    x2 = rng.normal(0, 1, n)
    y2 = 1.4 * x2 + rng.normal(0, 0.6, n)
    x2 = np.append(x2, 6.0)
    y2 = np.append(y2, -6.0)

    cases = [("(가) 단조이지만 선형이 아닌 관계", x1, y1),
             ("(나) 선형 관계 + 이상치 하나", x2, y2)]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    for col, (title, x, y) in enumerate(cases):
        r_p = stats.pearsonr(x, y)[0]
        r_s = stats.spearmanr(x, y)[0]
        out = np.zeros(len(x), bool)
        if col == 1:
            out[-1] = True

        ax = axes[0, col]
        ax.plot(x[~out], y[~out], "o", ms=8, color=BLUE, mec="white", mew=0.9)
        if out.any():
            ax.plot(x[out], y[out], "o", ms=12, color=RED, mec="white",
                    mew=1.2)
            ax.annotate("이상치", xy=(x[out][0], y[out][0]),
                        xytext=(x[out][0] - 1.4, y[out][0] + 2.6),
                        fontsize=10.5, color=RED, ha="right", va="center",
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
        clean(ax)
        ax.set_xlabel("$X$", fontsize=10.5, color=INK)
        ax.set_ylabel("$Y$", fontsize=10.5, color=INK)
        ax.set_title(title + "\n원값", fontsize=11.5, color=INK, loc="left",
                     pad=8)
        ax.text(0.03, 0.97,
                f"Pearson $r$ = {r_p:+.3f}\nSpearman $r_s$ = {r_s:+.3f}",
                transform=ax.transAxes, fontsize=11, color=INK,
                ha="left", va="top")

        ax = axes[1, col]
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        ax.plot(rx[~out], ry[~out], "o", ms=8, color=GREEN, mec="white",
                mew=0.9)
        if out.any():
            ax.plot(rx[out], ry[out], "o", ms=12, color=RED, mec="white",
                    mew=1.2)
        lo, hi = 0.5, len(x) + 0.5
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        clean(ax)
        ax.set_xlabel("$X$ 의 순위", fontsize=10.5, color=INK)
        ax.set_ylabel("$Y$ 의 순위", fontsize=10.5, color=INK)
        ax.set_title("순위", fontsize=11.5, color=INK, loc="left", pad=8)
        ax.text(0.03, 0.97,
                f"순위의 Pearson 상관 = {stats.pearsonr(rx, ry)[0]:+.3f}"
                f"\n   ($= r_s$)",
                transform=ax.transAxes, fontsize=11, color=GREEN,
                ha="left", va="top")
        print(title, "pearson %+.3f spearman %+.3f" % (r_p, r_s))

    fig.suptitle("Spearman 상관은 원값을 순위로 옮겨 놓고 Pearson 상관을 구한 것이다",
                 fontsize=13.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "spearman_rank_straightens.png")


# ==================================================================
# 2. tau 는 쌍의 기울기를 센다  (kendall.md)
# ==================================================================
def kendall_pairs():
    R = np.array([1, 2, 3, 4, 5, 6], float)
    S = np.array([2, 1, 4, 3, 6, 5], float)
    names = list("ABCDEF")
    n = len(R)

    C = D = 0
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0),
                             gridspec_kw=dict(width_ratios=[1, 1.05]))

    ax = axes[0]
    for i, j in itertools.combinations(range(n), 2):
        conc = (R[i] - R[j]) * (S[i] - S[j]) > 0
        if conc:
            C += 1
            ax.plot([R[i], R[j]], [S[i], S[j]], color=GREEN, lw=1.5,
                    alpha=0.55, zorder=1)
        else:
            D += 1
            ax.plot([R[i], R[j]], [S[i], S[j]], color=RED, lw=2.4,
                    alpha=0.95, zorder=2)
    ax.plot(R, S, "o", ms=13, color=BLUE, mec="white", mew=1.4, zorder=3)
    for k in range(n):
        ax.text(R[k], S[k], names[k], fontsize=10, color="white",
                ha="center", va="center", zorder=4)
    tau, p_exact = stats.kendalltau(R, S)
    S_stat = C - D
    varS = n * (n - 1) * (2 * n + 5) / 18
    z = S_stat / np.sqrt(varS)
    p_norm = 2 * stats.norm.sf(abs(z))
    ax.set_xlim(0.3, 6.7)
    ax.set_ylim(0.3, 8.1)
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    clean(ax)
    ax.set_xlabel("심사위원 1 의 순위", fontsize=10.5, color=INK)
    ax.set_ylabel("심사위원 2 의 순위", fontsize=10.5, color=INK)
    ax.plot([], [], color=GREEN, lw=2.0, label=f"올라가는 선분 = 부합  $C$ = {C}")
    ax.plot([], [], color=RED, lw=2.4, label=f"내려가는 선분 = 비부합  $D$ = {D}")
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax.text(0.45, 6.45,
            f"$S = C - D$ = {S_stat},   "
            f"$\\tau_a = {S_stat}/{n*(n-1)//2}$ = {tau:.3f}\n"
            f"정규근사 $p$ = {p_norm:.4f},   정확 $p$ = {p_exact:.4f}",
            fontsize=11, color=INK, ha="left", va="top")
    ax.set_title("(가) 점을 둘씩 이어 선분 15개의 기울기 부호를 센다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 같은 자료에서 tau 와 r_s 의 관계
    ax = axes[1]
    rng = np.random.default_rng(4)
    taus, rss = [], []
    for rho in np.linspace(-0.98, 0.98, 60):
        cov = [[1.0, rho], [rho, 1.0]]
        for _ in range(4):
            z2 = rng.multivariate_normal([0, 0], cov, 60)
            taus.append(stats.kendalltau(z2[:, 0], z2[:, 1])[0])
            rss.append(stats.spearmanr(z2[:, 0], z2[:, 1])[0])
    taus, rss = np.array(taus), np.array(rss)
    ax.plot(taus, rss, "o", ms=4.5, color=PURPLE, alpha=0.40, mec="none")
    g = np.linspace(-1, 1, 200)
    ax.plot(g, g, color=MUTED, lw=1.4, ls=(0, (4, 3)), label="$r_s = \\tau$")
    gm = g[np.abs(g) <= 2 / 3]
    ax.plot(gm, 1.5 * gm, color=INK, lw=2.0, label="$r_s = 1.5\\,\\tau$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    clean(ax)
    ax.set_xlabel("Kendall $\\tau$", fontsize=10.5, color=INK)
    ax.set_ylabel("Spearman $r_s$", fontsize=10.5, color=INK)
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax.text(0.30, -0.72, "같은 자료에서 $|\\tau|$ 가\n거의 항상 더 작다",
            fontsize=11, color=PURPLE, ha="left", va="center")
    ax.set_title("(나) 이변량 정규 자료 240 개 표본 ($n = 60$)",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    fig.suptitle("$\\tau$ 와 $r_s$ 는 같은 것을 재지만 눈금이 다르다",
                 fontsize=13.5, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "kendall_pairs.png")
    print("C=%d D=%d S=%d tau=%.3f varS=%.2f z=%.3f p_norm=%.4f p_exact=%.4f"
          % (C, D, S_stat, tau, varS, z, p_norm, p_exact))
    ok = np.abs(taus) > 0.02
    print("median |r_s|/|tau| = %.3f"
          % np.median(np.abs(rss[ok]) / np.abs(taus[ok])))


if __name__ == "__main__":
    spearman_rank_straightens()
    kendall_pairs()
