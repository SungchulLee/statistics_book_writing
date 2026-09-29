r"""13장 '계수 검정' 다섯 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/testing_coefficients/img/t_stat_and_vif.png     공선성이 t 를 깎는다
  ch13/testing_coefficients/img/pvalue_ci_duality.png  p 값과 신뢰구간은 한 몸이다
  ch13/testing_coefficients/img/stat_vs_practical.png  n 이 크면 티끌도 유의해진다
  ch13/testing_coefficients/img/regression_table_t.png 출력표의 t 열 읽기
  ch13/testing_coefficients/img/coefficient_plot.png   계수 그림으로 읽는 회귀 결과

실행:  python3 scripts/make_ch13_testing_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch13/testing_coefficients/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def bare(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# === 1. 공선성이 t 를 깎는다 ===
def fig_t_vif():
    b, se0, n, k = 2.5737, 0.332, 100, 3
    df = n - k
    tc = stats.t.ppf(0.975, df)
    vifs = [1, 4, 9, 16]
    ts = [b / (se0 * np.sqrt(v)) for v in vifs]
    cols = [GREEN, BLUE, PURPLE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    g = np.linspace(-4.5, 9, 900)
    d = stats.t.pdf(g, df)
    ax.plot(g, d, color=INK, lw=2.0)
    rej = g[g >= tc]
    ax.fill_between(rej, 0, stats.t.pdf(rej, df), color="#FFCDD2")
    rej2 = g[g <= -tc]
    ax.fill_between(rej2, 0, stats.t.pdf(rej2, df), color="#FFCDD2")
    ax.axvline(tc, color=RED, lw=1.5, ls="--")
    ax.text(tc - 0.38, 0.115, f"임계값 {tc:.3f}", fontsize=9.6, color=RED,
            ha="right")
    stem_h = {1: 0.20, 4: 0.34, 9: 0.47, 16: 0.60}
    for v, t, c in zip(vifs, ts, cols):
        h = stem_h[v]
        ax.plot([t, t], [0, h - 0.035], color=c, lw=1.8)
        ax.scatter([t], [0], s=46, color=c, zorder=6)
        ha, dx = ("right", -0.12) if v == 16 else ("center", 0.0)
        ax.text(t + dx, h - 0.028, f"VIF={v}\nt={t:.2f}", ha=ha, va="bottom",
                fontsize=9.6, color=c, linespacing=1.5)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel(r"$t = \hat\beta_j\,/\,\mathrm{SE}(\hat\beta_j)$",
                  fontsize=10.5, color=INK)
    ax.set_title(r"계수는 그대로, 표준오차만 $\sqrt{\mathrm{VIF}}$ 배",
                 fontsize=11.5, color=INK, pad=6)
    bare(ax)

    ax = axes[1]
    ys = np.arange(4)[::-1]
    for y, v, c in zip(ys, vifs, cols):
        se = se0 * np.sqrt(v)
        lo, hi = b - tc * se, b + tc * se
        ax.plot([lo, hi], [y, y], color=c, lw=3.2, solid_capstyle="butt")
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.12, y + 0.12], color=c, lw=3.2)
        ax.text(hi + 0.12, y, f"SE={se:.3f}", fontsize=9.8, color=c,
                va="center")
    ax.scatter([b] * 4, ys, s=48, color=INK, zorder=6)
    ax.axvline(0, color=RED, lw=1.8, ls="--")
    ax.text(0.08, 3.55, r"$\beta_j = 0$", fontsize=10.5, color=RED)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"VIF = {v}" for v in vifs], fontsize=10.5)
    ax.set_xlim(-2.6, 8.4)
    ax.set_ylim(-0.7, 3.9)
    ax.set_xlabel(r"$\beta_j$ 의 95% 신뢰구간", fontsize=10.5, color=INK)
    ax.set_title("같은 추정값인데 결론이 뒤집힌다", fontsize=11.5,
                 color=INK, pad=6)
    ax.text(b, -0.55, r"모두 $\hat\beta_j = 2.574$", fontsize=10, color=INK,
            ha="center")
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    fig.tight_layout()
    save(fig, "t_stat_and_vif.png")
    crit_vif = (b / se0 / tc) ** 2
    for v, t in zip(vifs, ts):
        se = se0 * np.sqrt(v)
        p = 2 * stats.t.sf(abs(t), df)
        print(f"  VIF={v:2d}  SE={se:.4f}  t={t:.3f}  p={p:.4f}  "
              f"CI=({b-tc*se:+.3f}, {b+tc*se:+.3f})")
    print(f"  임계 VIF = {crit_vif:.2f} (이보다 크면 유의성을 잃는다), "
          f"t* = {tc:.4f}")


# === 2. p 값과 신뢰구간은 한 몸이다 ===
def fig_pvalue_ci():
    b, se, df = 4.2, 1.5, 40
    t = b / se
    p = 2 * stats.t.sf(t, df)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    g = np.linspace(-4.5, 4.5, 900)
    d = stats.t.pdf(g, df)
    ax.plot(g, d, color=INK, lw=2.2)
    for s in (1, -1):
        tail = g[g * s >= t]
        ax.fill_between(tail, 0, stats.t.pdf(tail, df), color=ORANGE_F)
    ax.axvline(t, color=ORANGE, lw=1.8)
    ax.axvline(-t, color=ORANGE, lw=1.8, ls=":")
    ax.annotate(f"관측된 $t = {t:.2f}$", xy=(t, 0.045), xytext=(3.05, 0.20),
                fontsize=10, color=ORANGE, ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.text(0, 0.445, f"양쪽 꼬리 넓이의 합 = p = {p:.4f}", fontsize=10.5,
            color=INK, ha="center")
    ax.set_ylim(0, 0.50)
    ax.set_xlabel(r"$t_{40}$", fontsize=10.5, color=INK)
    ax.set_title("p 값은 꼬리의 넓이다", fontsize=11.5, color=INK, pad=6)
    bare(ax)

    ax = axes[1]
    lv = np.linspace(0.80, 0.9995, 900)
    tt = stats.t.ppf(1 - (1 - lv) / 2, df)
    lo, hi = b - tt * se, b + tt * se
    ax.fill_between(lv * 100, lo, hi, color=BLUE_F, alpha=0.9)
    ax.plot(lv * 100, lo, color=BLUE, lw=2.2)
    ax.plot(lv * 100, hi, color=BLUE, lw=2.2)
    ax.axhline(b, color=INK, lw=1.4, ls=":")
    ax.axhline(0, color=RED, lw=1.8, ls="--")
    cross = (1 - p) * 100
    ax.axvline(cross, color=ORANGE, lw=1.8)
    ax.scatter([cross], [0], s=60, color=RED, zorder=6)
    ax.annotate(f"{cross:.1f}% 에서 아래끝이 0 에 닿는다\n"
                f"곧 {cross:.1f} = 100(1 - p)",
                xy=(cross, 0), xytext=(81.2, -1.7), fontsize=9.8, color=RED,
                linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.text(81, b + 0.45, r"$\hat\beta_1 = 4.2$", fontsize=10, color=INK)
    ax.set_ylim(-2.2, 10.5)
    ax.set_xlim(80, 100)
    ax.set_xlabel("신뢰수준 (%)", fontsize=10.5, color=INK)
    ax.set_ylabel(r"구간의 양 끝", fontsize=10.5, color=INK)
    ax.set_title("신뢰수준을 올리면 언제 0 을 삼키는가", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    fig.tight_layout()
    save(fig, "pvalue_ci_duality.png")
    for L in (0.90, 0.95, 0.99, 1 - p, 0.995):
        tt = stats.t.ppf(1 - (1 - L) / 2, df)
        print(f"  {L:.4%} 구간 = ({b-tt*se:+.4f}, {b+tt*se:+.4f})")
    print(f"  t = {t:.4f}, p = {p:.6f}, 100(1-p) = {(1-p)*100:.3f}%")


# === 3. n 이 크면 티끌도 유의해진다 ===
def fig_stat_practical():
    beta, sig = 0.02, 1.0
    ns = np.logspace(2, 7, 300)
    se = sig / np.sqrt(ns)
    t = beta / se
    p = 2 * stats.norm.sf(t)
    ticks = [100, 1000, 10000, 100000, 1000000, 10000000]
    tlabs = ["100", "1000", "1만", "10만", "100만", "1000만"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    ax.plot(ns, p, color=BLUE, lw=2.6)
    ax.axhline(0.05, color=RED, lw=1.6, ls="--")
    n_cross = (stats.norm.ppf(0.975) * sig / beta) ** 2
    ax.axvline(n_cross, color=ORANGE, lw=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=9)
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 0.05, 1])
    ax.set_yticklabels(["1e-20", "1e-15", "1e-10", "1e-5", "0.05", "1"],
                       fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(1e-22, 3)
    ax.text(2.5e5, 0.12, "유의수준 0.05", fontsize=9.6, color=RED,
            ha="center")
    ax.annotate(f"n = {n_cross:,.0f} 을 넘으면\n유의해진다",
                xy=(n_cross, 0.05), xytext=(2.1e2, 2e-9), fontsize=9.8,
                color=ORANGE, linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel("p 값", fontsize=10.5, color=INK)
    ax.set_title(r"참 효과는 $\beta = 0.02$ 로 고정되어 있다", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    ax = axes[1]
    z = stats.norm.ppf(0.975)
    ax.fill_between(ns, beta - z * se, beta + z * se, color=BLUE_F,
                    alpha=0.95, label="95% 신뢰구간")
    ax.plot(ns, beta + z * se, color=BLUE, lw=1.8)
    ax.plot(ns, beta - z * se, color=BLUE, lw=1.8)
    ax.axhline(beta, color=INK, lw=1.8, ls=":")
    ax.axhline(0, color=RED, lw=1.4, ls="--")
    ax.axhspan(0.2, 0.6, color=GREEN_F, alpha=0.8)
    ax.text(3e4, 0.36, "실질적으로 의미 있는 크기", fontsize=10, color=GREEN,
            ha="center")
    ax.text(3e4, 0.30, r"($|\beta| \geq 0.2$ 라고 하자)", fontsize=9.6,
            color=GREEN, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(-0.25, 0.6)
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$\beta$", fontsize=11, color=INK)
    ax.set_title("구간은 좁아져도 효과는 커지지 않는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="lower right", frameon=False)
    ax.annotate(r"$\hat\beta = 0.02$ 는 어디서나 그대로", xy=(2e6, beta),
                xytext=(1.3e3, -0.17), fontsize=9.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    clean(ax)

    fig.tight_layout()
    save(fig, "stat_vs_practical.png")
    for nn in (1e3, 1e4, 1e5, 1e6, 1e7):
        s = sig / np.sqrt(nn)
        print(f"  n={nn:>10,.0f}  SE={s:.5f}  t={beta/s:7.2f}  "
              f"p={2*stats.norm.sf(beta/s):.3e}  "
              f"CI=({beta-z*s:+.5f}, {beta+z*s:+.5f})")
    print(f"  유의해지는 n = {n_cross:,.0f}")


# === 4. 출력표의 t 열 읽기 (Advertising 자료) ===
def fig_table_t():
    names = ["Intercept", "TV", "Radio", "Newspaper"]
    coef = np.array([3.0451, 0.0470, 0.1797, -0.0030])
    se = np.array([0.391, 0.002, 0.011, 0.007])
    tval = np.array([7.782, 27.653, 16.665, -0.428])
    pval = np.array([0.000, 0.000, 0.000, 0.669])
    n, k = 140, 4
    df = n - k
    tc = stats.t.ppf(0.975, df)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    g = np.linspace(-4.5, 4.5, 900)
    ax.plot(g, stats.t.pdf(g, df), color=INK, lw=2.2)
    for s in (1, -1):
        r = g[g * s >= tc]
        ax.fill_between(r, 0, stats.t.pdf(r, df), color="#FFCDD2")
    for v in (tc, -tc):
        ax.axvline(v, color=RED, lw=1.5, ls="--")
    ax.text(0, 0.475, f"기각역 경계 $\\pm{tc:.3f}$", fontsize=10.5, color=RED,
            ha="center")
    ax.plot([tval[3], tval[3]], [0, 0.30], color=GREEN, lw=2.6)
    ax.annotate(f"Newspaper\nt = {tval[3]:.3f},  p = {pval[3]:.3f}",
                xy=(tval[3], 0.30), xytext=(-4.4, 0.33), fontsize=9.8,
                color=GREEN, linespacing=1.6, ha="left",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.annotate("TV 는 t = 27.7\n(오른쪽으로 한참 밖)", xy=(4.45, 0.02),
                xytext=(2.2, 0.20), fontsize=9.8, color=BLUE,
                linespacing=1.6, ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.set_ylim(0, 0.53)
    ax.set_xlabel(r"$t_{136}$", fontsize=10.5, color=INK)
    ax.set_title("표의 네 행 가운데 하나만 안쪽에 있다", fontsize=11.5,
                 color=INK, pad=6)
    bare(ax)

    ax = axes[1]
    idx = np.arange(4)
    cols = [MUTED, BLUE, ORANGE, GREEN]
    bars = ax.bar(idx, np.abs(tval), color=cols, width=0.55)
    ax.axhline(tc, color=RED, lw=1.8, ls="--")
    ax.annotate(f"임계값 {tc:.3f}", xy=(3.2, tc), xytext=(3.45, 6.4),
                fontsize=9.8, color=RED, ha="right",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    for i, (t, p) in enumerate(zip(tval, pval)):
        ax.text(i, abs(t) + 0.8, f"{abs(t):.2f}", ha="center", fontsize=10.5,
                color=INK)
        ax.text(i, abs(t) + 2.6, f"p={p:.3f}", ha="center", fontsize=9.2,
                color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 33)
    ax.set_ylabel(r"$|t|$", fontsize=11, color=INK)
    ax.set_title("계수를 자기 표준오차로 나눈 값", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    fig.tight_layout()
    save(fig, "regression_table_t.png")
    for nm, c, s, t, p in zip(names, coef, se, tval, pval):
        lo, hi = c - tc * s, c + tc * s
        print(f"  {nm:10s} coef={c:+.4f} SE={s:.3f} t={t:+.3f} p={p:.3f} "
              f"CI=({lo:+.4f}, {hi:+.4f})")
    print(f"  df = {df}, t* = {tc:.4f}")


# === 5. 계수 그림으로 읽는 회귀 결과 ===
def fig_coef_plot():
    np.random.seed(42)                       # 본문 보기 3 과 같은 씨앗
    study = np.random.rand(100) * 10
    sleep = np.random.rand(100) * 8
    noise = np.random.randn(100)
    X = np.column_stack([np.ones(100), study, sleep])
    XtXi = np.linalg.inv(X.T @ X)
    truth = np.array([5.0, 2.5, -1.5])
    names = ["절편", "공부 시간", "수면 시간"]

    def fit(scale):
        y = X @ truth + noise * scale
        b = XtXi @ (X.T @ y)
        e = y - X @ b
        s2 = (e ** 2).sum() / (100 - 3)
        se = np.sqrt(s2 * np.diag(XtXi))
        tc = stats.t.ppf(0.975, 97)
        return b, se, b / se, 2 * stats.t.sf(np.abs(b / se), 97), tc

    small = fit(2.0)
    big = fit(16.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharex=True)
    for ax, (b, se, tv, pv, tc), ttl in [
            (axes[0], small, r"잡음 $\sigma = 2$"),
            (axes[1], big, r"잡음 $\sigma = 16$")]:
        ys = np.arange(3)[::-1]
        for y, j in zip(ys, range(3)):
            lo, hi = b[j] - tc * se[j], b[j] + tc * se[j]
            sig = pv[j] < 0.05
            c = BLUE if sig else RED
            ax.plot([lo, hi], [y, y], color=c, lw=3.2, solid_capstyle="butt")
            for e in (lo, hi):
                ax.plot([e, e], [y - 0.1, y + 0.1], color=c, lw=3.2)
            ax.scatter([b[j]], [y], s=50, color=c, zorder=6)
            ax.scatter([truth[j]], [y], s=95, color=GREEN, marker="D",
                       zorder=5, alpha=0.85)
            ax.text(25.0, y + 0.30,
                    ("유의함" if sig else "유의하지 않음") + f"  (p={pv[j]:.3g})",
                    fontsize=9.4, color=c, va="center", ha="right")
        ax.axvline(0, color=INK, lw=1.6, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlim(-22, 26)
        ax.set_ylim(-0.7, 2.75)
        ax.set_xlabel("계수", fontsize=10.5, color=INK)
        ax.set_title(ttl, fontsize=12, color=INK, pad=6)
        ax.tick_params(labelsize=9, colors=INK)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)

    axes[0].text(-21, 2.55, "초록 마름모 = 참값", fontsize=9.6, color=GREEN)
    fig.tight_layout()
    save(fig, "coefficient_plot.png")
    for tag, (b, se, tv, pv, tc) in [("sigma=2", small), ("sigma=16", big)]:
        print(f"  {tag}")
        for j, nm in enumerate(names):
            print(f"    {nm:6s} 참값={truth[j]:+.2f} 추정={b[j]:+.4f} "
                  f"SE={se[j]:.4f} t={tv[j]:+.2f} p={pv[j]:.3g} "
                  f"CI=({b[j]-tc*se[j]:+.3f}, {b[j]+tc*se[j]:+.3f})")


if __name__ == "__main__":
    fig_t_vif()
    fig_pvalue_ci()
    fig_stat_practical()
    fig_table_t()
    fig_coef_plot()
