r"""16.3 이표본 비모수 검정 다섯 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch16/two_sample_nonparametric/img/ranksum_null_enumeration.png  순위합의 정확 귀무분포
  ch16/two_sample_nonparametric/img/pairwise_wins_grid.png        U 는 쌍별 승수다
  ch16/two_sample_nonparametric/img/ks_ecdf_gap.png               KS 는 ECDF 간격을 본다
  ch16/two_sample_nonparametric/img/mw_tests_not_mean.png         MW 는 평균을 검정하지 않는다
  ch16/two_sample_nonparametric/img/mood_information_loss.png     한 비트로 뭉갤 때의 손실

실행:  python3 scripts/make_ch16_two_sample_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch16/two_sample_nonparametric/img/"
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
# 1. 순위합의 정확 귀무분포  (rank_sum.md)
# ==================================================================
def ranksum_null_enumeration():
    A = np.array([78, 64, 85, 72, 91, 80], float)
    B = np.array([55, 68, 61, 70, 66, 58], float)
    allv = np.concatenate([A, B])
    lab = np.array([0] * 6 + [1] * 6)
    order = np.argsort(allv)
    ranks = stats.rankdata(allv)
    W = ranks[lab == 0].sum()
    n1, n2, N = 6, 6, 12
    mu = n1 * (N + 1) / 2
    sd = np.sqrt(n1 * n2 * (N + 1) / 12)

    # 12개 순위 중 6개를 고르는 924가지를 모두 열거
    sums = np.array([sum(c) for c in itertools.combinations(range(1, N + 1), n1)])
    ks = np.arange(sums.min(), sums.max() + 1)
    pr = np.array([(sums == k).mean() for k in ks])
    p_exact = min(1.0, 2 * min((sums >= W).mean(), (sums <= W).mean()))
    p_approx = 2 * stats.norm.sf(abs((W - mu) / sd))

    fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.9),
                             gridspec_kw=dict(height_ratios=[0.52, 1]))

    # (가) 합친 순위 배치
    ax = axes[0]
    for pos, idx in enumerate(order):
        g = lab[idx]
        c, cf = (BLUE, BLUE_F) if g == 0 else (ORANGE, ORANGE_F)
        ax.add_patch(plt.Rectangle((pos, 0), 0.9, 1.0, facecolor=cf,
                                   edgecolor=c, lw=1.4))
        ax.text(pos + 0.45, 0.63, f"{allv[idx]:.0f}", fontsize=11, color=c,
                ha="center", va="center")
        ax.text(pos + 0.45, 0.22, f"{pos + 1}", fontsize=9.5, color=MUTED,
                ha="center", va="center")
    ax.text(-0.35, 0.63, "값", fontsize=10.5, color=INK, ha="right",
            va="center")
    ax.text(-0.35, 0.22, "순위", fontsize=10.5, color=INK, ha="right",
            va="center")
    ax.text(12.35, 0.75, f"집단 A 의 순위합  $W$ = {W:.0f}", fontsize=11.5,
            color=BLUE, ha="left", va="center")
    ax.text(12.35, 0.35, f"귀무 기댓값  $\\mu_W$ = {mu:.0f}", fontsize=11.5,
            color=INK, ha="left", va="center")
    ax.set_xlim(-2.0, 19.6)
    ax.set_ylim(-0.25, 1.45)
    ax.axis("off")
    ax.text(-2.0, 1.28, "(가) 두 표본을 합쳐 한 줄로 세우고 순위를 매긴다"
                        "   (파랑 = 집단 A, 주황 = 집단 B)",
            fontsize=11.5, color=INK, ha="left", va="center")

    # (나) 924가지 배정에서 나온 정확 귀무분포
    ax = axes[1]
    tail = ks >= W
    ax.bar(ks[~tail], pr[~tail], width=0.86, color=BLUE_F, edgecolor=BLUE,
           lw=0.9, label="정확 귀무분포 — 924 가지 순위 배정")
    ax.bar(ks[tail], pr[tail], width=0.86, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.2)
    grid = np.linspace(ks.min() - 1, ks.max() + 1, 500)
    ax.plot(grid, stats.norm.pdf(grid, mu, sd), color=INK, lw=2.0,
            label=f"정규근사 $\\mathcal{{N}}({mu:.0f},\\,{sd**2:.0f})$")
    ax.axvline(W, color=ORANGE, lw=1.4, ls=(0, (4, 3)))
    clean(ax)
    ax.set_xlabel("집단 A 의 순위합 $W$", fontsize=10.5, color=INK)
    ax.set_ylabel("확률 / 밀도", fontsize=10.5, color=INK)
    ax.set_ylim(0, pr.max() * 1.45)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.annotate(f"관측 $W$ = {W:.0f}\n정확 양측 $p$ = {p_exact:.5f}\n"
                f"정규근사 $p$ = {p_approx:.5f}",
                xy=(W + 0.4, pr[ks == W][0] + 0.002),
                xytext=(W + 4.5, pr.max() * 1.05), fontsize=10.5, color=INK,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    fig.suptitle("순위합의 귀무분포는 모집단이 아니라 '누가 어느 순위를 받았나'에서 나온다",
                 fontsize=13.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "ranksum_null_enumeration.png")
    print("W =", W, "exact %.5f" % p_exact, "approx %.5f" % p_approx,
          "scipy %.5f" % stats.mannwhitneyu(A, B, method="exact").pvalue)


# ==================================================================
# 2. U 는 쌍별 승수다  (mann_whitney.md)
# ==================================================================
def pairwise_wins_grid():
    A = np.sort(np.array([1.2, 0.5, -0.3, 2.1, 0.8, 1.5, 0.1]))
    B = np.sort(np.array([-0.5, 0.3, -1.0, 0.6, -0.2, 0.9]))
    n1, n2 = len(A), len(B)
    win = A[:, None] > B[None, :]
    U1 = int(win.sum())
    phat = U1 / (n1 * n2)
    p_exact = stats.mannwhitneyu(A, B, method="exact").pvalue

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    for i in range(n1):
        for j in range(n2):
            c = BLUE_F if win[i, j] else ORANGE_F
            e = BLUE if win[i, j] else ORANGE
            ax.add_patch(plt.Rectangle((j, n1 - 1 - i), 0.92, 0.92,
                                       facecolor=c, edgecolor=e, lw=1.2))
            ax.text(j + 0.46, n1 - 1 - i + 0.46,
                    "A 승" if win[i, j] else "B 승", fontsize=10,
                    color=e, ha="center", va="center")
    for i in range(n1):
        ax.text(-0.22, n1 - 1 - i + 0.46, f"{A[i]:+.1f}", fontsize=10.5,
                color=BLUE, ha="right", va="center")
    for j in range(n2):
        ax.text(j + 0.46, n1 + 0.14, f"{B[j]:+.1f}", fontsize=10.5,
                color=ORANGE, ha="center", va="bottom")
    ax.text(-0.22, n1 + 0.14, "주식 A  \\  주식 B", fontsize=10.5, color=INK,
            ha="right", va="bottom")
    ax.text(n2 + 0.35, n1 - 1.0,
            f"쌍의 개수  $n_1 n_2$ = {n1 * n2}\n"
            f"A 가 이긴 쌍  $U_1$ = {U1}\n"
            f"B 가 이긴 쌍  $U_2$ = {n1 * n2 - U1}\n\n"
            f"$\\hat{{P}}(X > Y) = \\dfrac{{{U1}}}{{{n1 * n2}}}$ = {phat:.3f}\n\n"
            f"정확 양측 $p$ = {p_exact:.4f}",
            fontsize=11.5, color=INK, ha="left", va="center")
    ax.set_xlim(-1.6, n2 + 4.6)
    ax.set_ylim(-0.4, n1 + 0.9)
    ax.axis("off")
    ax.set_title("$U$ 통계량은 '$A$ 가 $B$ 를 이긴 쌍이 몇 개인가'를 세는 것뿐이다",
                 fontsize=13, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, "pairwise_wins_grid.png")
    print("U1 =", U1, "phat %.4f" % phat, "exact p %.4f" % p_exact)


# ==================================================================
# 3. KS 는 ECDF 간격을 본다  (ks_two_sample.md)
# ==================================================================
def ecdf(x, grid):
    x = np.sort(x)
    return np.searchsorted(x, grid, side="right") / len(x)


def ks_ecdf_gap():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # (가) 본문 보기
    A = np.array([10.1, 10.3, 10.2, 10.5, 10.4])
    B = np.array([10.0, 10.2, 10.6, 10.8, 10.4])
    ax = axes[0]
    grid = np.linspace(9.92, 10.92, 1200)
    f1, f2 = ecdf(A, grid), ecdf(B, grid)
    ax.step(grid, f1, where="post", color=BLUE, lw=2.0, label="공정 A")
    ax.step(grid, f2, where="post", color=ORANGE, lw=2.0, label="공정 B")
    d = np.abs(f1 - f2)
    i = int(np.argmax(d))
    xstar = grid[i]
    ax.vlines(xstar, min(f1[i], f2[i]), max(f1[i], f2[i]), color=RED, lw=3.0)
    ax.annotate(f"$D$ = {d[i]:.1f}", xy=(xstar, (f1[i] + f2[i]) / 2),
                xytext=(xstar + 0.10, 0.45), fontsize=11.5, color=RED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ks = stats.ks_2samp(A, B, method="exact")
    ax.text(9.94, 0.97, f"정확 $p$ = {ks.pvalue:.4f}", fontsize=11, color=INK,
            ha="left", va="top")
    ax.set_ylim(-0.04, 1.14)
    clean(ax)
    ax.set_xlabel("지름 (mm)", fontsize=10.5, color=INK)
    ax.set_ylabel("경험적 누적분포 $\\hat{F}$", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    ax.set_title("(가) $D$ 는 두 계단 사이의 가장 큰 수직 간격이다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 중심은 같고 퍼짐만 다를 때
    rng = np.random.default_rng(7)
    n = 100
    X = rng.normal(0, 1, n)
    Y = rng.normal(0, 3, n)
    ax = axes[1]
    grid = np.linspace(-9, 9, 2000)
    f1, f2 = ecdf(X, grid), ecdf(Y, grid)
    ax.step(grid, f1, where="post", color=BLUE, lw=2.0,
            label="$X \\sim \\mathcal{N}(0,\\,1)$")
    ax.step(grid, f2, where="post", color=ORANGE, lw=2.0,
            label="$Y \\sim \\mathcal{N}(0,\\,9)$")
    d = np.abs(f1 - f2)
    i = int(np.argmax(d))
    xstar = grid[i]
    ax.vlines(xstar, min(f1[i], f2[i]), max(f1[i], f2[i]), color=RED, lw=3.0)
    p_ks = stats.ks_2samp(X, Y).pvalue
    p_mw = stats.mannwhitneyu(X, Y).pvalue
    ax.annotate(f"$D$ = {d[i]:.3f}", xy=(xstar, (f1[i] + f2[i]) / 2),
                xytext=(xstar - 0.6, 0.30), fontsize=11.5, color=RED,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.text(-8.7, 1.10, f"KS 검정 $p$ = {p_ks:.4f}\n"
                        f"Mann-Whitney $p$ = {p_mw:.4f}",
            fontsize=11, color=INK, ha="left", va="top")
    ax.set_ylim(-0.04, 1.22)
    clean(ax)
    ax.set_xlabel("값", fontsize=10.5, color=INK)
    ax.set_ylabel("경험적 누적분포 $\\hat{F}$", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    ax.set_title("(나) 중심이 같고 퍼짐만 달라도 KS 는 본다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    fig.suptitle("순위합검정은 중심을 보고 KS 검정은 분포 전체를 본다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "ks_ecdf_gap.png")
    print("scale case: D=%.3f  KS p=%.4f  MW p=%.4f" % (d[i], p_ks, p_mw))


# ==================================================================
# 4. Mann-Whitney 는 평균을 검정하지 않는다  (two_sample.md)
# ==================================================================
def mw_tests_not_mean():
    rng = np.random.default_rng(2024)
    mY = np.exp(0.5)          # exp(N(0,1)) 의 평균
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4),
                             gridspec_kw=dict(width_ratios=[1.12, 1]))

    # (가) 평균은 같고 중앙값은 다른 두 분포
    ax = axes[0]
    grid = np.linspace(-3.5, 4.5, 900)
    fx = stats.norm.pdf(grid, 0, 1)
    gpos = grid + mY
    fy = np.where(gpos > 0, stats.lognorm.pdf(np.maximum(gpos, 1e-9), 1.0), 0.0)
    ax.plot(grid, fx, color=BLUE, lw=2.1, label="$X$ (정규)")
    ax.fill_between(grid, 0, fx, color=BLUE_F, alpha=0.65)
    ax.plot(grid, fy, color=ORANGE, lw=2.1, label="$Y$ (치우친 분포)")
    ax.fill_between(grid, 0, fy, color=ORANGE_F, alpha=0.55)
    med_y = 1.0 - mY
    ax.vlines(0, 0, 0.42, color=INK, lw=1.6, ls=(0, (5, 3)))
    ax.vlines(med_y, 0, 0.62, color=ORANGE, lw=1.6, ls=(0, (3, 2)))
    ax.text(0.06, 0.45, "두 분포의 평균이\n모두 0 이다", fontsize=10.5,
            color=INK, ha="left", va="bottom")
    ax.annotate(f"$Y$ 의 중앙값 {med_y:.3f}", xy=(med_y, 0.64),
                xytext=(-3.4, 0.80), fontsize=10.5, color=ORANGE,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    xs = rng.normal(0, 1, 400_000)
    ys = rng.lognormal(0, 1, 400_000) - mY
    pxy = (xs > ys).mean()
    ax.text(1.35, 0.72, f"$P(X > Y)$ = {pxy:.3f}", fontsize=12, color=PURPLE,
            ha="left", va="center")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-3.5, 4.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    ax.set_xlabel("값", fontsize=10.5, color=INK)
    ax.set_title("(가) 평균은 같지만 $P(X>Y)$ 는 0.5 가 아니다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 표본크기별 기각률
    ax = axes[1]
    ns = [10, 20, 40, 80, 160]
    B = 3000
    rm, rt = [], []
    for n in ns:
        x = rng.normal(0, 1, (B, n))
        y = rng.lognormal(0, 1, (B, n)) - mY
        pm = np.array([stats.mannwhitneyu(x[i], y[i]).pvalue for i in range(B)])
        pt = stats.ttest_ind(x, y, axis=1, equal_var=False).pvalue
        rm.append((pm < 0.05).mean())
        rt.append((pt < 0.05).mean())
    ax.plot(ns, rm, "-o", color=PURPLE, lw=2.0, ms=6, label="Mann-Whitney $U$")
    ax.plot(ns, rt, "-s", color=INK, lw=2.0, ms=6, label="Welch $t$ 검정")
    ax.axhline(0.05, color=RED, lw=1.2, ls=(0, (5, 3)))
    ax.text(162, 0.010, "$\\alpha = 0.05$", fontsize=10, color=RED,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    clean(ax)
    ax.set_xlabel("집단당 표본크기 $n$", fontsize=10.5, color=INK)
    ax.set_ylabel("기각률", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="center left", frameon=False)
    ax.set_title("(나) 평균이 같은데도 $U$ 검정은 기각한다",
                 fontsize=11.5, color=INK, loc="left", pad=8)
    for n, v in zip(ns, rm):
        ax.annotate(f"{v:.3f}", xy=(n, v), xytext=(0, 9),
                    textcoords="offset points", fontsize=9, color=PURPLE,
                    ha="center")

    fig.suptitle("Mann-Whitney 검정의 귀무가설은 '평균이 같다'가 아니다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "mw_tests_not_mean.png")
    print("P(X>Y) = %.4f, median Y = %.4f" % (pxy, med_y))
    print("MW rej", np.round(rm, 3), " t rej", np.round(rt, 3))


# ==================================================================
# 5. 한 비트로 뭉갤 때의 손실  (two_sample_tests.md)
# ==================================================================
def mood_information_loss():
    d0 = np.array([10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43,
                   48, 49], float)
    d1 = np.array([28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92,
                   99], float)
    d2 = np.array([0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62,
                   67, 84], float)
    groups = [("집단 0", d0, BLUE), ("집단 1", d1, ORANGE),
              ("집단 2", d2, GREEN)]
    allv = np.concatenate([d0, d1, d2])
    gm = np.median(allv)
    ranks = stats.rankdata(allv)
    splits = np.cumsum([0, len(d0), len(d1), len(d2)])
    rk = [ranks[splits[i]:splits[i + 1]] for i in range(3)]

    h, p_kw = stats.kruskal(d0, d1, d2)
    mt = stats.median_test(d0, d1, d2)

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.6))

    # (가) 원값과 전체 중앙값
    ax = axes[0]
    for k, (name, v, c) in enumerate(groups):
        y = 2 - k
        above = v > gm
        ax.plot(v[~above], np.full((~above).sum(), y), "o", ms=9,
                color="white", mec=c, mew=1.8)
        ax.plot(v[above], np.full(above.sum(), y), "o", ms=9, color=c,
                mec="white", mew=0.8)
        ax.text(-7, y, name, fontsize=11, color=c, ha="right", va="center")
        ax.text(104, y, f"위 {int(above.sum())} / 아래 {int((~above).sum())}",
                fontsize=10.5, color=c, ha="left", va="center")
    ax.axvline(gm, color=RED, lw=1.8, ls=(0, (5, 3)))
    ax.text(gm + 1.5, 2.72, f"전체 중앙값 {gm:.0f}", fontsize=10.5, color=RED,
            ha="left", va="center")
    ax.set_xlim(-26, 132)
    ax.set_ylim(-0.7, 3.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("원값", fontsize=10.5, color=INK)
    ax.set_title("(가) Mood 중앙값검정이 보는 것 — 빨간 선의 왼쪽인가 오른쪽인가, 한 비트뿐",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 순위
    ax = axes[1]
    for k, (name, v, c) in enumerate(groups):
        y = 2 - k
        r = rk[k]
        ax.plot(r, np.full(len(r), y), "o", ms=9, color=c, mec="white",
                mew=0.8)
        ax.text(-3.2, y, name, fontsize=11, color=c, ha="right", va="center")
        ax.text(51.5, y, f"평균순위 {r.mean():.1f}", fontsize=10.5, color=c,
                ha="left", va="center")
    ax.set_xlim(-12.5, 65)
    ax.set_ylim(-0.7, 3.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("합친 표본에서의 순위", fontsize=10.5, color=INK)
    ax.set_title("(나) Kruskal-Wallis 가 보는 것 — 순위가 어디에 몰려 있는가",
                 fontsize=11.5, color=INK, loc="left", pad=8)
    ax.text(-12.0, 2.72,
            f"Kruskal-Wallis  $H$ = {h:.3f},  $p$ = {p_kw:.4f}"
            f"          Mood 중앙값검정  $\\chi^2$ = {mt.statistic:.2f}, "
            f" $p$ = {mt.pvalue:.4f}",
            fontsize=11, color=INK, ha="left", va="center")

    fig.suptitle("같은 자료에서 한쪽은 기각하고 한쪽은 기각하지 못한다",
                 fontsize=13.5, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "mood_information_loss.png")
    print("grand median", gm, "KW H=%.4f p=%.4f" % (h, p_kw),
          "Mood chi2=%.2f p=%.4f" % (mt.statistic, mt.pvalue))
    print("table\n", mt.table)
    print("mean ranks", [round(r.mean(), 2) for r in rk])


if __name__ == "__main__":
    ranksum_null_enumeration()
    pairwise_wins_grid()
    ks_ecdf_gap()
    mw_tests_not_mean()
    mood_information_loss()
