r"""13장 '선형회귀의 가정' 여섯 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/assumptions/img/line_four_violations.png        LINE 네 가정의 위배 신호 한눈에
  ch13/assumptions/img/linearity_curvature.png         곡선에 직선을 맞추면 남는 것
  ch13/assumptions/img/independence_se.png             자기상관은 표준오차를 망친다
  ch13/assumptions/img/homoscedasticity_se.png         부채꼴과 로버스트 표준오차
  ch13/assumptions/img/normality_clt_vs_pi.png         CLT 가 지켜 주는 것과 못 지키는 것
  ch13/assumptions/img/wls_efficiency.png              WLS 가 되찾는 효율

실행:  python3 scripts/make_ch13_assumptions_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch13/assumptions/img/"
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


# === 1. LINE 네 가정의 위배 신호 ===
def fig_four_violations():
    rng = np.random.default_rng(13)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))

    # (1) 선형성 위배 — 잔차에 곡선
    ax = axes[0, 0]
    x = np.linspace(0, 10, 120)
    y = 2 + 0.6 * x + 0.35 * (x - 5) ** 2 + rng.normal(0, 1.2, 120)
    X = np.column_stack([np.ones(120), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    fit, res = X @ b, y - X @ b
    ax.axhline(0, color=MUTED, lw=1)
    ax.scatter(fit, res, s=14, color=BLUE, alpha=0.75, edgecolor="none")
    o = np.argsort(fit)
    ax.plot(fit[o], np.poly1d(np.polyfit(fit, res, 2))(fit[o]), color=RED, lw=2)
    ax.set_title("선형성 위배 — 계수가 편향된다", fontsize=11, color=INK, pad=6)
    ax.set_xlabel(r"적합값 $\hat y$", fontsize=9.5, color=INK)
    ax.set_ylabel("잔차", fontsize=9.5, color=INK)
    ax.text(0.5, 0.92, "잔차에 곡선이 남는다", transform=ax.transAxes,
            fontsize=9, color=RED, ha="center")
    clean(ax)

    # (2) 독립성 위배 — 순서대로 물결
    ax = axes[0, 1]
    n = 120
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.85 * e[t - 1] + rng.normal(0, 1)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(np.arange(n), e, color=ORANGE, lw=1.4, marker="o", ms=3)
    ax.set_title("독립성 위배 — 표준오차가 작아진다", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("관측 순서", fontsize=9.5, color=INK)
    ax.set_ylabel("잔차", fontsize=9.5, color=INK)
    ax.text(0.03, 0.9, "부호가 뭉쳐 다닌다", transform=ax.transAxes,
            fontsize=9, color=ORANGE)
    clean(ax)

    # (3) 등분산성 위배 — 부채꼴
    ax = axes[1, 0]
    xf = rng.uniform(1, 10, 160)
    ef = rng.normal(0, 0.25 + 0.55 * xf)
    ax.axhline(0, color=MUTED, lw=1)
    ax.scatter(xf, ef, s=14, color=GREEN, alpha=0.8, edgecolor="none")
    gx = np.linspace(1, 10, 50)
    ax.plot(gx, 2 * (0.25 + 0.55 * gx), color=RED, lw=1.8, ls="--")
    ax.plot(gx, -2 * (0.25 + 0.55 * gx), color=RED, lw=1.8, ls="--")
    ax.set_title("등분산성 위배 — 표준오차가 틀린다", fontsize=11, color=INK, pad=6)
    ax.set_xlabel(r"설명변수 $x$", fontsize=9.5, color=INK)
    ax.set_ylabel("잔차", fontsize=9.5, color=INK)
    ax.text(0.03, 0.9, "퍼짐이 부채처럼 벌어진다", transform=ax.transAxes,
            fontsize=9, color=RED)
    clean(ax)

    # (4) 정규성 위배 — Q-Q 가 휜다
    ax = axes[1, 1]
    z = rng.chisquare(1, 200)
    z = (z - 1) / np.sqrt(2)
    q = np.sort(z)
    p = (np.arange(1, 201) - 0.5) / 200
    th = stats.norm.ppf(p)
    ax.plot([-3, 3], [-3, 3], color=MUTED, lw=1.4)
    ax.scatter(th, q, s=14, color=PURPLE, alpha=0.8, edgecolor="none")
    ax.set_xlim(-3.2, 3.2)
    ax.set_title("정규성 위배 — 구간이 흔들린다", fontsize=11, color=INK, pad=6)
    ax.set_xlabel("정규분포 분위수", fontsize=9.5, color=INK)
    ax.set_ylabel("잔차 분위수", fontsize=9.5, color=INK)
    ax.text(0.03, 0.9, "직선에서 휘어 나간다", transform=ax.transAxes,
            fontsize=9, color=PURPLE)
    clean(ax)

    fig.suptitle("가정마다 깨질 때의 얼굴이 다르고, 망가지는 것도 다르다",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "line_four_violations.png")


# === 2. 선형성 — 곡선에 직선을 맞추면 ===
def fig_linearity():
    rng = np.random.default_rng(7)
    n = 90
    x = np.linspace(0, 10, n)
    truth = 5 + 3.0 * x - 0.30 * x ** 2
    y = truth + rng.normal(0, 1.4, n)
    X = np.column_stack([np.ones(n), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    line = X @ b
    res = y - line

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

    ax = axes[0]
    ax.scatter(x, y, s=18, color=MUTED, alpha=0.85, edgecolor="none",
               label="관측값")
    ax.plot(x, truth, color=GREEN, lw=2.4, label="참 관계 (곡선)")
    ax.plot(x, line, color=BLUE, lw=2.4, label="적합한 직선")
    ax.fill_between(x, truth, line, where=truth > line, color=ORANGE_F, alpha=0.7)
    ax.fill_between(x, truth, line, where=truth <= line, color=BLUE_F, alpha=0.9)
    ax.annotate("가운데는 과소예측", xy=(5.2, 15.5), xytext=(3.9, 17.6),
                fontsize=9.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.annotate("양끝은 과대예측", xy=(9.5, 8.0), xytext=(6.4, 24.0),
                fontsize=9.5, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.set_xlabel(r"$x$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$y$", fontsize=10.5, color=INK)
    ax.set_title("직선은 어디서나 조금씩 틀린다", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9, loc="lower center", frameon=False)
    ax.set_ylim(0, 29)
    clean(ax)

    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.scatter(x, res, s=18, color=BLUE, alpha=0.85, edgecolor="none")
    sm = np.poly1d(np.polyfit(x, res, 2))(x)
    ax.plot(x, sm, color=RED, lw=2.4)
    ax.set_xlabel(r"$x$", fontsize=10.5, color=INK)
    ax.set_ylabel("잔차", fontsize=10.5, color=INK)
    ax.set_title("그 틀림이 잔차에 포물선으로 남는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.text(0.5, 0.06, "무작위가 아니다 — 남은 신호다",
            transform=ax.transAxes, fontsize=9.5, color=RED, ha="center")
    clean(ax)

    fig.tight_layout()
    save(fig, "linearity_curvature.png")

    lo = res[x < 2].mean()
    mid = res[(x > 4) & (x < 6)].mean()
    hi = res[x > 8].mean()
    print(f"  linearity: 잔차 평균  x<2: {lo:+.2f}, 4<x<6: {mid:+.2f}, x>8: {hi:+.2f}")
    print(f"  적합 직선: y = {b[0]:.2f} + {b[1]:.2f} x")


# === 3. 독립성 — 자기상관과 표준오차 ===
def fig_independence():
    rng = np.random.default_rng(2024)
    n, phi, reps = 60, 0.85, 4000
    x = np.linspace(0, 1, n)
    Sxx = ((x - x.mean()) ** 2).sum()

    def sim(ph):
        b, se = np.empty(reps), np.empty(reps)
        for r in range(reps):
            e = np.empty(n)
            e[0] = rng.normal(0, 1 / np.sqrt(1 - ph ** 2)) if ph else rng.normal()
            inn = rng.normal(0, 1, n)
            for t in range(1, n):
                e[t] = ph * e[t - 1] + inn[t]
            y = 1 + 2 * x + e
            bh = ((x - x.mean()) * y).sum() / Sxx
            ah = y.mean() - bh * x.mean()
            r2 = y - ah - bh * x
            s2 = (r2 ** 2).sum() / (n - 2)
            b[r], se[r] = bh, np.sqrt(s2 / Sxx)
        return b, se

    b0, se0 = sim(0.0)
    b1, se1 = sim(phi)
    tcrit = stats.t.ppf(0.975, n - 2)
    rej0 = np.mean(np.abs((b0 - 2) / se0) > tcrit)
    rej1 = np.mean(np.abs((b1 - 2) / se1) > tcrit)

    fig = plt.figure(figsize=(10.4, 4.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.62, wspace=0.28)

    # 잔차 순서 그림 두 개
    for k, (ph, col, lab) in enumerate([(0.0, MUTED, r"독립 오차 $\phi=0$"),
                                        (phi, ORANGE, r"자기상관 오차 $\phi=0.85$")]):
        ax = fig.add_subplot(gs[k, 0])
        e = np.zeros(n)
        inn = rng.normal(0, 1, n)
        for t in range(1, n):
            e[t] = ph * e[t - 1] + inn[t]
        ax.axhline(0, color=MUTED, lw=1)
        ax.plot(np.arange(n), e, color=col, lw=1.4, marker="o", ms=2.8)
        ax.set_title(lab, fontsize=10, color=INK, pad=4)
        ax.set_ylabel("잔차", fontsize=9, color=INK)
        if k == 1:
            ax.set_xlabel("관측 순서", fontsize=9, color=INK)
        clean(ax)

    # 기울기 표집분포 vs 보고된 표준오차
    ax = fig.add_subplot(gs[:, 1])
    ax.hist(b1, bins=55, density=True, color=ORANGE_F,
            edgecolor=ORANGE, lw=0.5, label="실제 표집분포")
    g = np.linspace(b1.min(), b1.max(), 400)
    ax.plot(g, stats.norm.pdf(g, 2, se1.mean()), color=BLUE, lw=2.4,
            label="보고된 SE 로 그린 종")
    ax.axvline(2, color=INK, lw=1.2, ls="--")
    ax.annotate(r"참값 $\beta_1=2$", xy=(2, 0.30), xytext=(4.6, 0.36),
                fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xlabel(r"기울기 추정값 $\hat\beta_1$", fontsize=10, color=INK)
    ax.set_ylabel("밀도", fontsize=10, color=INK)
    ax.set_title("중심은 맞는데 폭이 안 맞는다", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=8.8, loc="upper left", frameon=False)
    ax.text(0.98, 0.42,
            f"실제 SD = {b1.std():.3f}\n보고된 SE 평균 = {se1.mean():.3f}\n"
            f"5% 검정의 실제 기각률 = {rej1:.1%}",
            transform=ax.transAxes, fontsize=9.3, color=RED,
            ha="right", va="top", linespacing=1.6)
    clean(ax)

    save(fig, "independence_se.png")
    print(f"  독립: SD={b0.std():.3f} SE={se0.mean():.3f} 기각률={rej0:.1%}")
    print(f"  자기상관: SD={b1.std():.3f} SE={se1.mean():.3f} 기각률={rej1:.1%}")
    print(f"  평균 추정값: 독립 {b0.mean():.4f}, 자기상관 {b1.mean():.4f}")


# === 4. 등분산성 — 부채꼴과 로버스트 표준오차 ===
def fig_homoscedasticity():
    rng = np.random.default_rng(99)
    n, reps = 100, 4000
    x = np.linspace(1, 10, n)
    X = np.column_stack([np.ones(n), x])
    XtXi = np.linalg.inv(X.T @ X)
    sig = 0.3 + 0.22 * x ** 2                # 이분산 — x 가 커질수록 급히 벌어진다
    sig_const = np.full(n, np.sqrt((sig ** 2).mean()))   # 같은 평균분산의 등분산

    def run(s):
        bs, se_ols, se_hc = np.empty(reps), np.empty(reps), np.empty(reps)
        for r in range(reps):
            y = 3 + 2 * x + rng.normal(0, s)
            b = XtXi @ (X.T @ y)
            e = y - X @ b
            s2 = (e ** 2).sum() / (n - 2)
            h = np.einsum("ij,jk,ik->i", X, XtXi, X)
            om = (e / (1 - h)) ** 2           # HC3
            V = XtXi @ (X.T * om) @ X @ XtXi
            bs[r] = b[1]
            se_ols[r] = np.sqrt(s2 * XtXi[1, 1])
            se_hc[r] = np.sqrt(V[1, 1])
        return bs, se_ols, se_hc

    bh, se_o, se_h = run(sig)
    bc, se_oc, _ = run(sig_const)

    fig = plt.figure(figsize=(11.6, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.08], wspace=0.32)

    for k, (s, col, ttl) in enumerate([
            (sig_const, BLUE, "등분산 — 띠의 폭이 일정하다"),
            (sig, ORANGE, "이분산 — 부채꼴로 벌어진다")]):
        ax = fig.add_subplot(gs[0, k])
        y = 3 + 2 * x + rng.normal(0, s)
        b = XtXi @ (X.T @ y)
        e = y - X @ b
        ax.axhline(0, color=MUTED, lw=1.2)
        ax.scatter(X @ b, e, s=17, color=col, alpha=0.8, edgecolor="none")
        f = 3 + 2 * x
        ax.plot(f, 2 * s, color=RED, lw=1.7, ls="--")
        ax.plot(f, -2 * s, color=RED, lw=1.7, ls="--")
        ax.set_xlabel(r"적합값 $\hat y$", fontsize=10, color=INK)
        ax.set_ylabel("잔차", fontsize=10, color=INK)
        ax.set_title(ttl, fontsize=11, color=INK, pad=6)
        ax.set_ylim(-60, 60)
        clean(ax)

    ax = fig.add_subplot(gs[0, 2])
    vals = [bh.std(), se_o.mean(), se_h.mean()]
    labs = ["실제\n" + r"$\mathrm{SD}(\hat\beta_1)$", "OLS 가\n보고한 SE",
            "HC3\n로버스트 SE"]
    cols = [INK, RED, GREEN]
    bars = ax.bar(labs, vals, color=cols, width=0.58)
    top = max(vals) * 1.42
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + top * 0.05, f"{v:.3f}",
                ha="center", fontsize=10.5, color=INK)
    ax.set_ylim(0, top)
    ax.set_ylabel("기울기의 표준오차", fontsize=10, color=INK)
    ax.set_title("이분산 자료에서 어느 SE 가 맞나", fontsize=11, color=INK, pad=6)
    ax.axhline(vals[0], color=INK, lw=1, ls=":")
    clean(ax)

    save(fig, "homoscedasticity_se.png")
    tc = stats.t.ppf(0.975, n - 2)
    print(f"  이분산: 평균 beta1={bh.mean():.4f}, SD={bh.std():.4f}, "
          f"OLS SE={se_o.mean():.4f}, HC3 SE={se_h.mean():.4f}")
    print(f"  기각률 OLS SE={np.mean(np.abs((bh-2)/se_o)>tc):.1%}, "
          f"HC3={np.mean(np.abs((bh-2)/se_h)>tc):.1%}")
    print(f"  등분산 대조: SD={bc.std():.4f}, OLS SE={se_oc.mean():.4f}")


# === 5. 정규성 — CLT 가 지키는 것과 못 지키는 것 ===
def _skew_err(rng, size):
    """평균 0, 분산 1 로 맞춘 로그정규 오차. 오른쪽으로 심하게 치우쳐 있다."""
    m, s = np.exp(0.5), np.sqrt(np.exp(2) - np.exp(1))
    return (np.exp(rng.normal(0, 1, size)) - m) / s


def fig_normality():
    rng = np.random.default_rng(555)
    reps = 40000

    def slope_z(n):
        x = np.linspace(0, 1, n) ** 3      # 오른쪽으로 치우친 설계점
        xc = x - x.mean()
        Sxx = (xc ** 2).sum()
        y = 1 + 2 * x + _skew_err(rng, (reps, n))
        bh = (y * xc).sum(1) / Sxx
        return (bh - 2) / (bh - 2).std()

    z10, z200 = slope_z(10), slope_z(200)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    bins = np.linspace(-4, 6, 100)
    ax.hist(z10, bins=bins, density=True, histtype="stepfilled",
            color=ORANGE_F, edgecolor=ORANGE, lw=1.3,
            label=f"n = 10   (왜도 {stats.skew(z10):.2f})")
    ax.hist(z200, bins=bins, density=True, histtype="step",
            color=GREEN, lw=2.0,
            label=f"n = 200  (왜도 {stats.skew(z200):.2f})")
    g = np.linspace(-4, 6, 400)
    ax.plot(g, stats.norm.pdf(g), color=INK, lw=2.0, ls="--",
            label="표준정규")
    ax.set_xlabel(r"표준화한 기울기 $(\hat\beta_1-\beta_1)/\mathrm{SD}$",
                  fontsize=10, color=INK)
    ax.set_ylabel("밀도", fontsize=10, color=INK)
    ax.set_title("계수는 중심극한정리가 구해 준다", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.2, loc="upper right", frameon=False)
    ax.set_xlim(-4, 6)
    clean(ax)

    # 오른쪽: 예측구간은 개별 오차의 분포를 그대로 쓴다
    ax = axes[1]
    m, s = np.exp(0.5), np.sqrt(np.exp(2) - np.exp(1))
    g = np.linspace(-m / s + 1e-6, 6.0, 1500)
    dens = stats.lognorm.pdf(g * s + m, 1.0) * s
    dens = dens / dens.max() * 0.52
    ax.fill_between(g, 0, dens, color="#EDE1F4", zorder=1)
    ax.plot(g, dens, color=PURPLE, lw=2.0, zorder=2)
    ax.text(0.55, 0.30, "오차의 분포", fontsize=9.6, color=PURPLE)

    q_lo = (stats.lognorm.ppf(0.025, 1.0) - m) / s
    q_hi = (stats.lognorm.ppf(0.975, 1.0) - m) / s
    miss_hi = 1 - stats.lognorm.cdf(1.96 * s + m, 1.0)

    def band(y, lo, hi, col, lab):
        ax.plot([lo, hi], [y, y], color=col, lw=3.2, solid_capstyle="butt",
                zorder=3)
        for v in (lo, hi):
            ax.plot([v, v], [y - 0.045, y + 0.045], color=col, lw=3.2, zorder=3)
        ax.text((lo + hi) / 2, y + 0.085, lab, fontsize=9.8, color=col,
                ha="center")

    band(1.02, -1.96, 1.96, BLUE,
         r"정규성이 준 95% 예측구간  $\pm 1.96\,s$")
    band(0.74, q_lo, q_hi, GREEN,
         f"참 중앙 95% 구간  [{q_lo:.2f}, {q_hi:.2f}]")

    ax.annotate(f"위로 {miss_hi:.1%} 가 샌다", xy=(2.1, 1.02),
                xytext=(3.1, 1.30), fontsize=9.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.annotate("오차가 닿지도 못하는\n자리에 아래 한계가 있다",
                xy=(-1.96, 0.97), xytext=(-3.3, 0.47), fontsize=9.8,
                color=RED, ha="left", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlim(-3.4, 6)
    ax.set_ylim(0, 1.55)
    ax.set_yticks([])
    ax.set_xlabel(r"오차 $\varepsilon$ (평균 0, 분산 1 로 맞춤)",
                  fontsize=10, color=INK)
    ax.set_title("예측구간은 구해 주지 못한다", fontsize=11.5, color=INK, pad=6)
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    fig.suptitle("오차가 로그정규일 때 — 치우침은 표본을 키워도 사라지지 않는다",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "normality_clt_vs_pi.png")

    # 실제 예측구간 포함률을 표본크기별로 확인
    for n in (20, 100, 640):
        x = np.linspace(0, 1, n)
        xc = x - x.mean()
        Sxx = (xc ** 2).sum()
        y = 1 + 2 * x + _skew_err(rng, (8000, n))
        bh = (y * xc).sum(1) / Sxx
        ah = y.mean(1) - bh * x.mean()
        res = y - ah[:, None] - bh[:, None] * x
        sd = np.sqrt((res ** 2).sum(1) / (n - 2))
        tc = stats.t.ppf(0.975, n - 2)
        x0 = 0.5
        half = tc * sd * np.sqrt(1 + 1 / n + (x0 - x.mean()) ** 2 / Sxx)
        yh = ah + bh * x0
        ynew = 1 + 2 * x0 + _skew_err(rng, 8000)
        below = np.mean(ynew < yh - half)
        above = np.mean(ynew > yh + half)
        print(f"  n={n:4d}  PI 포함률={1-below-above:.3f}  "
              f"아래로 샘={below:.3f}  위로 샘={above:.3f}")
    print(f"  왜도: n=10 {stats.skew(z10):.3f}, n=200 {stats.skew(z200):.3f}")
    print(f"  참 중앙 95% 구간 = [{q_lo:.3f}, {q_hi:.3f}], "
          f"정규 구간 = [-1.96, 1.96]")


# === 6. WLS — 효율을 되찾는다 ===
def fig_wls():
    rng = np.random.default_rng(42)
    n, reps = 120, 4000
    x = rng.uniform(1, 10, n)
    x.sort()
    X = np.column_stack([np.ones(n), x])
    sig = 0.5 + 1.5 * x
    w = 1.0 / sig ** 2
    XtXi = np.linalg.inv(X.T @ X)
    XtWXi = np.linalg.inv((X.T * w) @ X)

    bo, bw = np.empty(reps), np.empty(reps)
    for r in range(reps):
        y = 3 + 2 * x + rng.normal(0, sig)
        bo[r] = (XtXi @ (X.T @ y))[1]
        bw[r] = (XtWXi @ ((X.T * w) @ y))[1]

    # OLS 가 눈에 띄게 빗나간 한 번의 표본을 골라 보인다 (흔한 일이다)
    for _ in range(3000):
        y1 = 3 + 2 * x + rng.normal(0, sig)
        b_o = XtXi @ (X.T @ y1)
        b_w = XtWXi @ ((X.T * w) @ y1)
        if abs(b_o[1] - 2) > 0.55 and abs(b_w[1] - 2) < 0.07:
            break
    print(f"  보인 표본: OLS 기울기={b_o[1]:.3f}, WLS 기울기={b_w[1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    ax = axes[0]
    sz = 8 + 340 * (w / w.max())
    ax.scatter(x, y1, s=sz, color=BLUE_F, edgecolor=BLUE, lw=0.8, alpha=0.9)
    ax.plot(x, 3 + 2 * x, color=INK, lw=2.2, ls="--", label="참 직선")
    ax.plot(x, X @ b_o, color=RED, lw=2.2, label="OLS")
    ax.plot(x, X @ b_w, color=GREEN, lw=2.2, label="WLS")
    ax.set_xlabel(r"$x$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"$y$", fontsize=10.5, color=INK)
    ax.set_title("점의 크기가 곧 무게 " + r"$w_i=1/\sigma_i^2$", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.text(0.97, 0.05, "왼쪽 점들이 직선을 붙잡는다",
            transform=ax.transAxes, fontsize=9.5, color=GREEN, ha="right")
    clean(ax)

    ax = axes[1]
    g = np.linspace(min(bo.min(), bw.min()), max(bo.max(), bw.max()), 500)
    ax.hist(bo, bins=60, density=True, color="#FFCDD2", edgecolor=RED,
            lw=0.4, alpha=0.85, label=f"OLS  (SD {bo.std():.3f})")
    ax.hist(bw, bins=60, density=True, color=GREEN_F, edgecolor=GREEN,
            lw=0.4, alpha=0.85, label=f"WLS  (SD {bw.std():.3f})")
    ax.axvline(2, color=INK, lw=1.4, ls="--")
    ax.set_xlabel(r"기울기 추정값 $\hat\beta_1$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("둘 다 참값 2 에 모이지만 폭이 다르다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.text(0.97, 0.55, f"분산비\n{(bo.std()/bw.std())**2:.2f} 배",
            transform=ax.transAxes, fontsize=10, color=INK, ha="right",
            va="top", linespacing=1.5)
    clean(ax)

    fig.tight_layout()
    save(fig, "wls_efficiency.png")
    print(f"  OLS  mean={bo.mean():.4f} SD={bo.std():.4f}")
    print(f"  WLS  mean={bw.mean():.4f} SD={bw.std():.4f}")
    print(f"  분산비 = {(bo.std()/bw.std())**2:.3f}")
    print(f"  무게 비: w(x=1)/w(x=10) = {(1/(0.5+1.5*1)**2)/(1/(0.5+1.5*10)**2):.1f}")


if __name__ == "__main__":
    fig_four_violations()
    fig_linearity()
    fig_independence()
    fig_homoscedasticity()
    fig_normality()
    fig_wls()
