r"""11장 일원배치 분산분석 세 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch11/anova_one_way/img/oneway_signal_noise.png    같은 평균 차이가 잡음에 따라 다르게 읽힌다
  ch11/anova_one_way/img/manual_ss_decomposition.png  제곱합은 SSE 가 크지만 평균제곱은 MST 가 크다
  ch11/anova_one_way/img/f_null_vs_alternative.png   귀무분포와 비중심분포, 그리고 기각률

실행:  python3 scripts/make_ch11_one_way_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch11/anova_one_way/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# =====================================================================
# 1. 신호 대 잡음 — 집단 평균은 그대로 두고 집단 내 흩어짐만 바꾼다
# =====================================================================
def fig_signal_noise():
    rng = np.random.default_rng(1105)
    mu = np.array([10.0, 12.0, 14.0])
    n = 20
    base = [rng.standard_normal(n) for _ in range(3)]
    for b in base:                      # 표본평균을 정확히 0으로 맞춘다
        b -= b.mean()
        b /= b.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for ax, sd, title in zip(axes, [1.2, 6.0],
                             ["집단 내 흩어짐이 작다", "집단 내 흩어짐이 크다"]):
        groups = [mu[i] + sd * base[i] for i in range(3)]
        allv = np.concatenate(groups)
        gm = allv.mean()
        N, k = len(allv), 3
        SST = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
        SSE = sum(((g - g.mean()) ** 2).sum() for g in groups)
        F = (SST / (k - 1)) / (SSE / (N - k))
        p = stats.f.sf(F, k - 1, N - k)

        xs = np.arange(3)
        jit = rng.uniform(-0.17, 0.17, (3, n))
        for i, g in enumerate(groups):
            ax.scatter(xs[i] + jit[i], g, s=26, color=BLUE,
                       alpha=0.55, edgecolor="none", zorder=3)
            ax.hlines(g.mean(), xs[i] - 0.33, xs[i] + 0.33,
                      color=ORANGE, lw=2.6, zorder=4)
        ax.axhline(gm, color=MUTED, lw=1.4, ls="--", zorder=2)
        ax.text(2.82, gm + 0.45, "전체평균", color=INK, fontsize=9,
                va="bottom", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels(["집단 1", "집단 2", "집단 3"], fontsize=10)
        ax.set_xlim(-0.6, 2.85)
        ax.set_title(f"{title}   ($\\sigma$ = {sd})", fontsize=11.5, color=INK)
        plab = "$p < 10^{-4}$" if p < 1e-4 else f"$p$ = {p:.4f}"
        ax.text(0.03, 0.96,
                f"$F$ = {F:.2f},  " + plab,
                transform=ax.transAxes, fontsize=11.5, color=RED,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=MUTED, lw=0.8))
        clean(ax)
        print(f"  sd={sd}: SST={SST:.3f} SSE={SSE:.3f} F={F:.3f} p={p:.5f}")

    axes[0].set_ylabel("측정값", fontsize=10.5, color=INK)
    axes[0].set_ylim(-8, 32)
    fig.suptitle("집단 평균은 10, 12, 14 로 두 그림이 똑같다",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "oneway_signal_noise.png")


# =====================================================================
# 2. 제곱합의 분해 — SSE 가 크지만 자유도로 나누면 뒤집힌다
# =====================================================================
PLANT = {
    "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
    "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
    "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
}


def fig_ss_decomposition():
    names = list(PLANT)
    allv = np.concatenate([PLANT[t] for t in names])
    gm = allv.mean()
    N, k = len(allv), len(names)
    SST = sum(len(PLANT[t]) * (PLANT[t].mean() - gm) ** 2 for t in names)
    SSE = sum(((PLANT[t] - PLANT[t].mean()) ** 2).sum() for t in names)
    MST, MSE = SST / (k - 1), SSE / (N - k)
    F = MST / MSE
    print(f"  SST={SST:.4f} SSE={SSE:.4f} MST={MST:.4f} MSE={MSE:.4f} F={F:.4f}")

    fig = plt.figure(figsize=(12.4, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.75, 1, 1], wspace=0.42)

    # --- (a) 관측값마다 두 조각으로 쪼갠다
    ax = fig.add_subplot(gs[0, 0])
    x0 = 0
    for i, t in enumerate(names):
        g = PLANT[t]
        m = g.mean()
        xs = np.arange(x0, x0 + len(g))
        ax.vlines(xs, m, g, color=ORANGE, lw=1.6, alpha=0.85, zorder=2)
        ax.vlines(xs, gm, m, color=BLUE, lw=3.2, alpha=0.9, zorder=1)
        ax.scatter(xs, g, s=22, color=INK, zorder=4)
        ax.hlines(m, x0 - 0.4, x0 + len(g) - 0.6, color=ORANGE,
                  lw=1.8, ls="-", zorder=3)
        ax.text(x0 + len(g) / 2 - 0.5, 6.62, t, ha="center",
                fontsize=10, color=INK)
        x0 += len(g)
    ax.axhline(gm, color=MUTED, lw=1.4, ls="--", zorder=0)
    ax.text(30.6, gm, "전체평균 5.073", fontsize=9.5, color=INK,
            ha="left", va="center")
    ax.set_ylim(3.2, 6.9)
    ax.set_xlim(-1, 36.5)
    ax.set_xticks([])
    ax.set_ylabel("식물 무게", fontsize=10.5, color=INK)
    ax.set_title("관측값 30 개를 두 조각으로 쪼갠다", fontsize=11.5, color=INK)
    ax.plot([], [], color=BLUE, lw=3.2, label="집단평균 - 전체평균 (집단 간)")
    ax.plot([], [], color=ORANGE, lw=1.6, label="관측값 - 집단평균 (집단 내)")
    ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=0.95)
    clean(ax)

    # --- (b) 제곱합
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar([0, 1], [SST, SSE], width=0.62,
                  color=[BLUE_F, ORANGE_F], edgecolor=[BLUE, ORANGE], lw=1.6)
    for b, v, d in zip(bars, [SST, SSE], [k - 1, N - k]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.3f}",
                ha="center", fontsize=11, color=INK)
        ax.text(b.get_x() + b.get_width() / 2, v * 0.5,
                f"df = {d}", ha="center", fontsize=10, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["집단 간\nSST", "집단 내\nSSE"], fontsize=10)
    ax.set_ylim(0, 13.2)
    ax.set_title("제곱합: SSE 가 2.8 배 크다", fontsize=11.5, color=INK)
    clean(ax)

    # --- (c) 평균제곱
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar([0, 1], [MST, MSE], width=0.62,
                  color=[BLUE_F, ORANGE_F], edgecolor=[BLUE, ORANGE], lw=1.6)
    for b, v in zip(bars, [MST, MSE]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.4f}",
                ha="center", fontsize=11, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["집단 간\nMST", "집단 내\nMSE"], fontsize=10)
    ax.set_ylim(0, 2.35)
    ax.set_title(f"평균제곱: MST 가 {F:.2f} 배 크다", fontsize=11.5, color=INK)
    ax.axhline(MSE, color=RED, lw=1.2, ls=":", zorder=0)
    ax.text(0.5, 2.15, f"$F$ = MST / MSE = {F:.4f}", ha="center",
            fontsize=12, color=RED)
    clean(ax)

    save(fig, "manual_ss_decomposition.png")


# =====================================================================
# 3. 귀무분포와 비중심분포
# =====================================================================
def fig_null_vs_alt():
    rng = np.random.default_rng(42)
    sizes = np.array([10, 20, 30])
    sd = 6.0
    k, N = 3, sizes.sum()
    df1, df2 = k - 1, N - k
    crit = stats.f.ppf(0.95, df1, df2)
    M = 20_000

    def sim(mu):
        out = np.empty(M)
        for i in range(M):
            g = [rng.normal(m, sd, n) for m, n in zip(mu, sizes)]
            out[i] = stats.f_oneway(*g).statistic
        return out

    F0 = sim([3, 3, 3])
    F1 = sim([3, 6, 9])
    mu1 = np.array([3.0, 6.0, 9.0])
    mubar = (sizes * mu1).sum() / N
    lam = (sizes * (mu1 - mubar) ** 2).sum() / sd ** 2
    rej0, rej1 = np.mean(F0 > crit), np.mean(F1 > crit)
    pow_th = stats.ncf.sf(crit, df1, df2, lam)
    print(f"  crit={crit:.4f} lambda={lam:.4f}")
    print(f"  size={rej0:.4f}  power_sim={rej1:.4f}  power_theory={pow_th:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    xs = np.linspace(0, 14, 600)

    ax = axes[0]
    ax.hist(F0, bins=np.linspace(0, 14, 71), density=True,
            color=BLUE_F, edgecolor=BLUE, lw=0.5, label="모의실험 20,000 회")
    ax.plot(xs, stats.f.pdf(xs, df1, df2), color=INK, lw=2,
            label=f"이론밀도 $F({df1}, {df2})$")
    ax.plot(xs, stats.ncf.pdf(xs, df1, df2, lam), color=GREEN, lw=2, ls="--",
            label=f"비중심 $F$, $\\lambda$ = {lam:.2f}")
    ax.axvline(crit, color=RED, lw=1.6)
    ax.text(crit + 0.25, 0.62, f"임계값 {crit:.3f}", color=RED, fontsize=10,
            rotation=90, va="top")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("$F$ 통계량", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("평균이 모두 같을 때: 이론과 정확히 맞는다",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9, loc="upper right")
    clean(ax)

    ax = axes[1]
    ax.axvspan(crit, 26, color="#FBE9E7", zorder=0)
    ax.hist(F0, bins=np.linspace(0, 30, 121), density=True,
            color=BLUE_F, edgecolor=BLUE, lw=0.4, zorder=2,
            label=f"$\\mu$ = (3, 3, 3) : 기각률 {rej0 * 100:.1f} %")
    ax.hist(F1, bins=np.linspace(0, 30, 121), density=True,
            color=GREEN_F, edgecolor=GREEN, lw=0.4, alpha=0.8, zorder=1,
            label=f"$\\mu$ = (3, 6, 9) : 기각률 {rej1 * 100:.1f} %")
    ax.axvline(crit, color=RED, lw=1.6, zorder=3)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 0.95)
    ax.text(crit + 0.35, 0.72, f"임계값 {crit:.3f}", color=RED, fontsize=10,
            rotation=90, va="top")
    ax.text(14.5, 0.06, "기각역", color=RED, fontsize=11, ha="center")
    ax.set_xlabel("$F$ 통계량", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("평균이 벌어지면 분포가 오른쪽으로 밀린다",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=10, loc="upper right")
    clean(ax)

    fig.tight_layout()
    save(fig, "f_null_vs_alternative.png")


if __name__ == "__main__":
    fig_signal_noise()
    fig_ss_decomposition()
    fig_null_vs_alt()
