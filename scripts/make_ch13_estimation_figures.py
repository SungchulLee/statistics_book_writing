r"""13장 '추정과 추론' 여섯 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/estimation_inference/img/mle_equals_ols.png        가능도 최대화 = 제곱 최소화
  ch13/estimation_inference/img/three_variance_pieces.png 예측 분산의 세 조각
  ch13/estimation_inference/img/ci_vs_pi_bands.png        신뢰띠는 줄고 예측띠는 안 준다
  ch13/estimation_inference/img/mc_beta_and_se.png        몬테카를로가 확인하는 표준오차
  ch13/estimation_inference/img/slope_ci_levels.png       신뢰수준을 올리면 결론이 바뀐다
  ch13/estimation_inference/img/joint_vs_marginal_ci.png  결합영역과 개별구간의 어긋남

실행:  python3 scripts/make_ch13_estimation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch13/estimation_inference/img/"
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


# === 1. 가능도 최대화와 제곱합 최소화는 같은 일이다 ===
def fig_mle_ols():
    rng = np.random.default_rng(3)
    n, sig = 12, 1.0
    x = np.sort(rng.uniform(0, 10, n))
    y = 1 + 0.9 * x + rng.normal(0, sig, n)

    X = np.column_stack([np.ones(n), x])
    bhat = np.linalg.lstsq(X, y, rcond=None)[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # 왼쪽: 후보 직선 하나와 각 점에서의 밀도 높이
    ax = axes[0]
    a, b = 2.6, 0.62                      # 일부러 어긋난 후보 직선
    gx = np.linspace(-0.4, 10.6, 100)
    ax.plot(gx, a + b * gx, color=ORANGE, lw=2.4, label="후보 직선")
    ax.plot(gx, bhat[0] + bhat[1] * gx, color=BLUE, lw=2.4, ls="--",
            label="최소제곱 직선")
    ax.scatter(x, y, s=34, color=INK, zorder=5, label="관측값")

    for i in (2, 10):
        mu = a + b * x[i]
        t = np.linspace(mu - 3.0, mu + 3.0, 200)
        d = stats.norm.pdf(t, mu, sig) * 2.2
        ax.plot(x[i] + d, t, color=PURPLE, lw=1.6)
        ax.plot([x[i], x[i]], [mu - 3.0, mu + 3.0], color=MUTED, lw=0.8)
        h = stats.norm.pdf(y[i], mu, sig) * 2.2
        ax.plot([x[i], x[i] + h], [y[i], y[i]], color=RED, lw=2.4)
        ax.scatter([x[i] + h], [y[i]], s=26, color=RED, zorder=6)

    ax.text(0.03, 0.96, "붉은 선분 = 그 점에서의 밀도 높이\n가능도는 이 높이들의 곱",
            transform=ax.transAxes, fontsize=9.6, color=RED, va="top",
            linespacing=1.6)
    ax.set_xlabel(r"$x$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$y$", fontsize=10.5, color=INK)
    ax.set_title("후보 직선 하나가 자료에 주는 가능도", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.2, loc="lower right", frameon=False)
    ax.set_xlim(-0.8, 13.2)
    clean(ax)

    # 오른쪽: (RSS, logL) 가 정확히 한 직선 위에 놓인다
    ax = axes[1]
    aa = rng.uniform(bhat[0] - 2.2, bhat[0] + 2.2, 2500)
    bb = rng.uniform(bhat[1] - 0.35, bhat[1] + 0.35, 2500)
    res = y[None, :] - aa[:, None] - bb[:, None] * x[None, :]
    rss = (res ** 2).sum(1)
    const = -n / 2 * np.log(2 * np.pi * sig ** 2)
    logL = const - rss / (2 * sig ** 2)

    ax.scatter(rss, logL, s=6, color=BLUE, alpha=0.35, edgecolor="none")
    rss_min = ((y - X @ bhat) ** 2).sum()
    ax.scatter([rss_min], [const - rss_min / (2 * sig ** 2)], s=90,
               color=RED, zorder=6, marker="*")
    ax.annotate("최소제곱 해 = 최대가능도 해",
                xy=(rss_min, const - rss_min / (2 * sig ** 2)),
                xytext=(rss_min + 22, const - 58), fontsize=9.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.text(0.97, 0.93,
            r"$\ln L = c - \dfrac{\mathrm{RSS}}{2\sigma^2}$" + "\n"
            + f"기울기 = {-1/(2*sig**2):.2f}",
            transform=ax.transAxes, fontsize=11, color=INK, ha="right",
            va="top", linespacing=1.8)
    ax.set_xlabel(r"잔차제곱합 $\mathrm{RSS}(\alpha,\beta)$", fontsize=10.5,
                  color=INK)
    ax.set_ylabel(r"로그가능도 $\ln L(\alpha,\beta)$", fontsize=10.5, color=INK)
    ax.set_title(r"후보 $(\alpha,\beta)$ 2500 개를 찍으면", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    fig.tight_layout()
    save(fig, "mle_equals_ols.png")
    print(f"  RSS 최소 = {rss_min:.2f}, 적합 직선 y = {bhat[0]:.2f} + {bhat[1]:.2f}x")
    print(f"  상관계수(RSS, logL) = {np.corrcoef(rss, logL)[0,1]:.6f}")


# === 2. 예측 분산의 세 조각 ===
def fig_three_pieces():
    n, sig = 25, 1.0
    x = np.linspace(0, 10, n)
    xbar = x.mean()
    Sxx = ((x - xbar) ** 2).sum()
    x0 = np.linspace(-3, 13, 400)
    piece_n = np.full_like(x0, 1 / n)
    piece_x = (x0 - xbar) ** 2 / Sxx
    piece_1 = np.ones_like(x0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    ax.stackplot(x0, piece_n, piece_x, piece_1,
                 colors=[BLUE_F, GREEN_F, ORANGE_F], edgecolor="none",
                 labels=[r"$1/n$ · 직선의 높이를 추정한 몫",
                         r"$(x_0-\bar x)^2/S_{xx}$ · 기울기를 추정한 몫",
                         r"$1$ · 줄일 수 없는 잡음"])
    ax.plot(x0, piece_n + piece_x, color=GREEN, lw=2.2)
    ax.plot(x0, piece_n + piece_x + piece_1, color=ORANGE, lw=2.2)
    ax.axvspan(0, 10, color=MUTED, alpha=0.12)
    ax.text(5, 0.62, "아래 두 조각만 쓰면 신뢰띠\n셋을 다 쓰면 예측띠",
            fontsize=10, color=INK, ha="center", linespacing=1.6)
    ax.set_ylim(0, 2.25)
    ax.set_xlim(-3, 13)
    ax.set_xlabel(r"$x_0$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$\mathrm{Var}/\sigma^2$", fontsize=10.5, color=INK)
    ax.set_title("예측오차 분산을 이루는 세 조각", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.0, loc="upper center", frameon=False, ncol=1,
              handlelength=1.4, labelspacing=0.35)
    clean(ax)

    ax = axes[1]
    se_mean = sig * np.sqrt(piece_n + piece_x)
    se_ind = sig * np.sqrt(piece_1 + piece_n + piece_x)
    ax.plot(x0, se_ind, color=ORANGE, lw=2.4, label="개별반응의 표준오차")
    ax.plot(x0, se_mean, color=BLUE, lw=2.4, label="평균반응의 표준오차")
    ax.axvline(xbar, color=MUTED, lw=1.2, ls=":")
    ax.axvspan(0, 10, color=MUTED, alpha=0.12)
    m0 = sig * np.sqrt(1 / n)
    i0 = sig * np.sqrt(1 + 1 / n)
    ax.scatter([xbar, xbar], [m0, i0], s=40, color=INK, zorder=6)
    ax.annotate(f"{i0:.3f}", xy=(xbar, i0), xytext=(xbar + 0.5, i0 + 0.16),
                fontsize=10, color=ORANGE)
    ax.annotate(f"{m0:.3f}", xy=(xbar, m0), xytext=(xbar - 1.6, m0 - 0.13),
                fontsize=10, color=BLUE)
    ax.text(xbar, 1.52, r"$x_0=\bar x$ 에서 5.10 배 차이", fontsize=9.8,
            color=INK, ha="center")
    ax.set_ylim(0, 2.2)
    ax.set_xlim(-3, 13)
    ax.set_xlabel(r"$x_0$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"표준오차 $(\sigma=1)$", fontsize=10.5, color=INK)
    ax.set_title("같은 t 분포, 다른 표준오차", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="upper left", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "three_variance_pieces.png")
    e12 = np.argmin(np.abs(x0 - 12))
    print(f"  n={n}, Sxx={Sxx:.1f}, xbar={xbar}")
    print(f"  x0=xbar: SE_mean={m0:.4f}, SE_ind={i0:.4f}, 비={i0/m0:.3f}")
    print(f"  x0=12  : SE_mean={se_mean[e12]:.4f}, SE_ind={se_ind[e12]:.4f}, "
          f"비={se_ind[e12]/se_mean[e12]:.3f}")
    print(f"  x0=12 의 기울기 조각 = {piece_x[e12]:.3f}")


# === 3. 신뢰띠는 줄고 예측띠는 줄지 않는다 ===
def fig_bands():
    rng = np.random.default_rng(0)
    n, sig = 100, 3.0
    x = rng.normal(0, 1, n)
    y = 1 + 2 * x + rng.normal(0, sig, n)
    xbar = x.mean()
    Sxx = ((x - xbar) ** 2).sum()
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s = np.sqrt(((y - X @ b) ** 2).sum() / (n - 2))
    tc = stats.t.ppf(0.975, n - 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    g = np.linspace(x.min(), x.max(), 300)
    yh = b[0] + b[1] * g
    core = 1 / n + (g - xbar) ** 2 / Sxx
    ci = tc * s * np.sqrt(core)
    pi = tc * s * np.sqrt(1 + core)
    ax.fill_between(g, yh - pi, yh + pi, color=ORANGE_F, alpha=0.75,
                    label="95% 예측띠")
    ax.fill_between(g, yh - ci, yh + ci, color=BLUE_F, label="95% 신뢰띠")
    ax.scatter(x, y, s=16, color=MUTED, alpha=0.9, edgecolor="none",
               label="관측값")
    ax.plot(g, yh, color=BLUE, lw=2.2, label="적합 직선")
    ax.set_xlabel(r"$x$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$y$", fontsize=10.5, color=INK)
    ax.set_title(r"$n=100$, $\sigma=3$ 에서의 두 띠", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.0, loc="upper left", frameon=False, ncol=2)
    ax.set_ylim(-12, 16)
    clean(ax)

    ax = axes[1]
    ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    ci_w, pi_w = [], []
    for m in ns:
        t = stats.t.ppf(0.975, m - 2)
        ci_w.append(t * sig * np.sqrt(1 / m))
        pi_w.append(t * sig * np.sqrt(1 + 1 / m))
    ax.plot(ns, pi_w, color=ORANGE, lw=2.4, marker="s", ms=6,
            label="예측띠 반폭")
    ax.plot(ns, ci_w, color=BLUE, lw=2.4, marker="o", ms=6,
            label="신뢰띠 반폭")
    ax.axhline(1.96 * sig, color=MUTED, lw=1.4, ls="--")
    ax.text(10000, 1.96 * sig + 0.22, r"$1.96\sigma = 5.88$", fontsize=9.8,
            color=INK, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(["10", "30", "100", "300", "1000", "3000", "10000"],
                       fontsize=8.6)
    ax.minorticks_off()
    ax.set_ylim(0, 8.2)
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$x_0=\bar x$ 에서의 반폭", fontsize=10.5, color=INK)
    ax.set_title("n 을 키우면 한쪽만 줄어든다", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="center right", frameon=False)
    ax.annotate(f"n=10000 에서 {ci_w[-1]:.3f}", xy=(10000, ci_w[-1]),
                xytext=(300, 1.1), fontsize=9.6, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    clean(ax)

    fig.tight_layout()
    save(fig, "ci_vs_pi_bands.png")
    for m, c, p in zip(ns, ci_w, pi_w):
        print(f"  n={m:6d}  CI 반폭={c:.4f}  PI 반폭={p:.4f}  비={p/c:.1f}")


# === 4. 몬테카를로가 확인하는 표준오차 ===
def fig_mc():
    rng = np.random.default_rng(11)
    n, sig, reps = 200, 2.0, 5000
    beta = np.array([2.0, 3.0, -1.0])
    X = np.column_stack([np.ones(n), rng.normal(0, 1, (n, 2))])
    XtXi = np.linalg.inv(X.T @ X)
    se_th = sig * np.sqrt(np.diag(XtXi))

    E = rng.normal(0, sig, (n, reps))
    Y = (X @ beta)[:, None] + E
    B = XtXi @ (X.T @ Y)                     # 3 x reps
    mc_mean = B.mean(1)
    mc_sd = B.std(1, ddof=1)

    res = Y - X @ B
    s2 = (res ** 2).sum(0) / (n - 3)
    se_hat = np.sqrt(np.outer(np.diag(XtXi), s2))
    tc = stats.t.ppf(0.975, n - 3)
    cover = np.mean(np.abs(B - beta[:, None]) <= tc * se_hat, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    cols = [BLUE, ORANGE, GREEN]
    names = [r"$\hat\beta_0$", r"$\hat\beta_1$", r"$\hat\beta_2$"]
    for j in range(3):
        ax.hist(B[j], bins=60, density=True, color=cols[j], alpha=0.35,
                edgecolor="none")
        g = np.linspace(beta[j] - 4 * se_th[j], beta[j] + 4 * se_th[j], 300)
        ax.plot(g, stats.norm.pdf(g, beta[j], se_th[j]), color=cols[j], lw=2.2)
        ax.axvline(beta[j], color=cols[j], lw=1.1, ls=":")
        ax.text(beta[j], 3.05, names[j] + f"\n참값 {beta[j]:.0f}", fontsize=9.8,
                color=cols[j], ha="center", linespacing=1.5)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel("추정값", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(r"5000 번 반복한 $\hat\beta$ 와 이론 곡선", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    ax = axes[1]
    idx = np.arange(3)
    w = 0.34
    ax.bar(idx - w / 2, mc_sd, w, color=INK, label="몬테카를로 표준편차")
    ax.bar(idx + w / 2, se_th, w, color=MUTED, label="이론 표준오차")
    for j in range(3):
        ax.text(j - w / 2, mc_sd[j] + 0.006, f"{mc_sd[j]:.4f}", ha="center",
                fontsize=9.4, color=INK)
        ax.text(j + w / 2, se_th[j] + 0.006, f"{se_th[j]:.4f}", ha="center",
                fontsize=9.4, color=INK)
    ax.text(0.5, 0.82,
            "95% 신뢰구간 포함률  "
            + " · ".join(f"{c:.3f}" for c in cover),
            transform=ax.transAxes, ha="center", fontsize=9.8, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, max(mc_sd.max(), se_th.max()) * 1.62)
    ax.set_ylabel("표준편차", fontsize=10.5, color=INK)
    ax.set_title("경험과 이론이 만나는 자리", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="upper center", frameon=False, ncol=2)
    clean(ax)

    fig.tight_layout()
    save(fig, "mc_beta_and_se.png")
    for j in range(3):
        print(f"  beta_{j}: 참값={beta[j]:+.1f} MC평균={mc_mean[j]:+.4f} "
              f"MC표준편차={mc_sd[j]:.4f} 이론SE={se_th[j]:.4f} "
              f"포함률={cover[j]:.4f}")
    print(f"  s^2 평균={s2.mean():.4f} (참값 {sig**2:.1f}), "
          f"RSS/n 평균={(s2*(n-3)/n).mean():.4f}")


# === 5. 신뢰수준을 올리면 결론이 뒤집힌다 ===
def fig_ci_levels():
    bhat, se, n = 0.164, 0.057, 20
    df = n - 2
    levels = [0.90, 0.95, 0.99]
    tstars = [stats.t.ppf(1 - (1 - L) / 2, df) for L in levels]
    margins = [t * se for t in tstars]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    ax = axes[0]
    g = np.linspace(-4.2, 4.2, 600)
    ax.plot(g, stats.norm.pdf(g), color=MUTED, lw=2.2, ls="--",
            label=r"표준정규 $N(0,1)$")
    ax.plot(g, stats.t.pdf(g, df), color=BLUE, lw=2.4,
            label=r"$t_{18}$")
    tail = g[g >= tstars[1]]
    ax.fill_between(tail, 0, stats.t.pdf(tail, df), color=BLUE_F)
    ax.axvline(1.96, color=MUTED, lw=1.6, ls=":", ymax=0.52)
    ax.axvline(tstars[1], color=BLUE, lw=1.6, ymax=0.52)
    ax.annotate(f"{tstars[1]:.4f}", xy=(tstars[1], 0.02),
                xytext=(2.85, 0.13), fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate("1.9600", xy=(1.96, 0.055), xytext=(0.55, 0.20),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(0.02, 0.60, r"오른쪽 꼬리 넓이 $0.025$", transform=ax.transAxes,
            fontsize=9.6, color=INK)
    ax.set_xlabel("표준화한 값", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(r"$\sigma$ 를 추정한 대가가 임계값에 붙는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper left", frameon=False)
    ax.set_ylim(0, 0.46)
    clean(ax)

    ax = axes[1]
    ys = [2, 1, 0]
    cols = [GREEN, BLUE, ORANGE]
    for yy, L, m, t, c in zip(ys, levels, margins, tstars, cols):
        lo, hi = bhat - m, bhat + m
        ax.plot([lo, hi], [yy, yy], color=c, lw=3.4, solid_capstyle="butt")
        for v in (lo, hi):
            ax.plot([v, v], [yy - 0.13, yy + 0.13], color=c, lw=3.4)
        ax.text(hi + 0.012, yy, f"{L:.0%}   ({lo:+.4f}, {hi:+.4f})",
                fontsize=10, color=c, va="center")
    ax.scatter([bhat] * 3, ys, s=50, color=INK, zorder=6)
    ax.axvline(0, color=RED, lw=1.8, ls="--")
    ax.text(0.004, 2.62, r"$\beta_1=0$", fontsize=10.5, color=RED)
    ax.text(bhat, -0.55, r"$\hat\beta_1=0.164$", fontsize=10.5, color=INK,
            ha="center")
    ax.annotate("99% 구간만 0 을 담는다", xy=(0.0, 0.10), xytext=(0.045, 0.56),
                fontsize=9.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlim(-0.06, 0.47)
    ax.set_ylim(-0.9, 2.95)
    ax.set_yticks([])
    ax.set_xlabel(r"기울기 $\beta_1$", fontsize=10.5, color=INK)
    ax.set_title("같은 자료, 세 가지 신뢰수준", fontsize=11.5, color=INK, pad=6)
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    fig.tight_layout()
    save(fig, "slope_ci_levels.png")
    for L, t, m in zip(levels, tstars, margins):
        print(f"  {L:.0%}: t*={t:.4f} margin={m:.4f} "
              f"({bhat-m:+.4f}, {bhat+m:+.4f})")
    print(f"  p-value = {2*stats.t.sf(bhat/se, df):.4f}")


# === 6. 결합 신뢰영역과 개별 신뢰구간 ===
def fig_joint():
    def build(rho, seed):
        rng = np.random.default_rng(seed)
        n = 60
        z1 = rng.normal(0, 1, n)
        z2 = rng.normal(0, 1, n)
        x1 = z1
        x2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        X = np.column_stack([np.ones(n), x1, x2])
        y = 1 + 0.6 * x1 + 0.6 * x2 + rng.normal(0, 1, n)
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ (X.T @ y)
        e = y - X @ b
        s2 = (e ** 2).sum() / (n - 3)
        V = s2 * XtXi[1:, 1:]
        se = np.sqrt(np.diag(V))
        tvals = b[1:] / se
        # 두 계수가 모두 0 이라는 결합가설의 F 검정
        Xr = X[:, :1]
        br = np.linalg.lstsq(Xr, y, rcond=None)[0]
        rss_r = ((y - Xr @ br) ** 2).sum()
        rss_f = (e ** 2).sum()
        F = ((rss_r - rss_f) / 2) / s2
        pF = stats.f.sf(F, 2, n - 3)
        return b[1:], V, se, tvals, F, pF, n

    # 공선인 설계에서 두 t 가 모두 유의하지 않은 표본을 고른다
    pick = None
    for sd in range(200):
        r = build(0.95, sd)
        if np.all(np.abs(r[3]) < 2.0) and r[5] < 0.01:
            pick = (sd, r)
            break
    sd_col, col = pick
    ind = build(0.0, sd_col)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, (b, V, se, tv, F, pF, n), ttl, rho in [
            (axes[0], col, "상관 0.95 인 두 설명변수", 0.95),
            (axes[1], ind, "무상관인 두 설명변수", 0.0)]:
        tc = stats.t.ppf(0.975, n - 3)
        fc = stats.f.ppf(0.95, 2, n - 3)
        w, v = np.linalg.eigh(V)
        ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
        k = np.sqrt(2 * fc)
        el = Ellipse(b, 2 * k * np.sqrt(w[-1]), 2 * k * np.sqrt(w[0]),
                     angle=ang, facecolor=BLUE_F, edgecolor=BLUE, lw=2.2,
                     alpha=0.85, zorder=2)
        ax.add_patch(el)
        rect = Rectangle((b[0] - tc * se[0], b[1] - tc * se[1]),
                         2 * tc * se[0], 2 * tc * se[1],
                         facecolor="none", edgecolor=ORANGE, lw=2.2,
                         ls="--", zorder=3)
        ax.add_patch(rect)
        ax.axhline(0, color=MUTED, lw=1.2)
        ax.axvline(0, color=MUTED, lw=1.2)
        ax.scatter([0], [0], s=70, color=RED, marker="X", zorder=6)
        ax.scatter([b[0]], [b[1]], s=45, color=INK, zorder=6)

        ax.set_xlim(-1.85, 1.85)
        ax.set_ylim(-1.85, 1.85)
        ax.set_xlabel(r"$\beta_1$", fontsize=11, color=INK)
        ax.set_ylabel(r"$\beta_2$", fontsize=11, color=INK)
        ax.set_title(ttl, fontsize=11.5, color=INK, pad=6)
        vif = 1 / (1 - rho ** 2)
        ax.text(0.03, 0.97,
                f"VIF = {vif:.2f}\n" + r"$t_1$ = " + f"{tv[0]:.2f},  "
                + r"$t_2$ = " + f"{tv[1]:.2f}\n"
                + r"$F$ = " + f"{F:.1f}  (p = {pF:.1e})",
                transform=ax.transAxes, fontsize=9.6, color=INK,
                va="top", linespacing=1.7)
        clean(ax)

    axes[0].text(0.97, 0.06, "파란 타원 = 결합 95% 영역\n"
                 "주황 점선 = 개별 95% 구간\n"
                 "붉은 X = 원점 (0, 0)",
                 transform=axes[0].transAxes, fontsize=9.4, color=INK,
                 ha="right", va="bottom", linespacing=1.6)
    fig.tight_layout()
    save(fig, "joint_vs_marginal_ci.png")
    for nm, r, rho in [("공선", col, 0.95), ("무상관", ind, 0.0)]:
        b, V, se, tv, F, pF, n = r
        print(f"  {nm}: beta={b.round(3)} se={se.round(3)} "
              f"t={tv.round(2)} F={F:.2f} p={pF:.2e} VIF={1/(1-rho**2):.2f}")


if __name__ == "__main__":
    fig_mle_ols()
    fig_three_pieces()
    fig_bands()
    fig_mc()
    fig_ci_levels()
    fig_joint()
