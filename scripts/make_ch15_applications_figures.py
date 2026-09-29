r"""15장 응용 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch15/applications/img/pretest_strategies.png     세 전략의 실제 제1종 오류율
  ch15/applications/img/hetero_unbiased_se.png     이분산은 계수가 아니라 표준오차를 망가뜨린다
  ch15/applications/img/breusch_pagan_mechanism.png  BP 검정이 실제로 보는 것
  ch15/applications/img/volatility_estimation_noise.png  변동성이 변한 것처럼 보이는 까닭

실행:  python3 scripts/make_ch15_applications_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch15/applications/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 그림 1. 세 전략의 실제 제1종 오류율 ===
def fig_pretest():
    rows = ["$(5,5,5,5)$\n등분산", "$(5,5,5,10)$\n한 집단만 다름",
            "$(2,4,6,10)$\n크게 다름"]
    anova = [0.045, 0.063, 0.074]
    welch = [0.045, 0.044, 0.046]
    two = [0.046, 0.057, 0.047]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    x = np.arange(3)
    w = 0.26
    sets = [(anova, RED, "항상 표준 분산분석"),
            (two, ORANGE, "두 단계 (브라운–포사이드 사전검정)"),
            (welch, GREEN, "항상 웰치 분산분석")]
    for j, (vals, color, lab) in enumerate(sets):
        bars = ax.bar(x + (j - 1) * w, vals, w * 0.92, color=color,
                      alpha=0.85, label=lab, edgecolor="white", lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0016, f"{v:.3f}",
                    ha="center", fontsize=9.5, color=color,
                    fontweight="bold")
    ax.axhline(0.05, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.text(-0.52, 0.0515, "명목 0.05", fontsize=10.5, color=INK, ha="left")
    ax.annotate("사전검정이 가장 필요한 자리에서\n가장 도움이 되지 않는다",
                xy=(1.14, 0.0555), xytext=(0.62, 0.0775), fontsize=10.5,
                color=ORANGE, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.set_xticks(x)
    ax.set_xticklabels(rows, fontsize=10.5, color=INK)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 0.085)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11, color=INK)
    clean(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("집단 4개, 각 $n = 15$, 평균은 모두 같다 (반복 5000회)",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "pretest_strategies.png")


# === 그림 2. 이분산은 계수가 아니라 표준오차를 망가뜨린다 ===
def ols_fit(x, y):
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    s2 = (e @ e) / (n - k)
    se_ols = np.sqrt(np.diag(s2 * XtX_inv))
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / (1 - h)) ** 2               # HC3
    meat = X.T @ (X * w[:, None])
    se_hc3 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    return beta, se_ols, se_hc3, e, X @ beta, h


def fig_hetero_se():
    rng = np.random.default_rng(3)
    n, beta1 = 100, 1.0
    R = 4000
    b1 = np.empty(R)
    cov_ols = cov_hc3 = 0
    tcrit = stats.t.ppf(0.975, n - 2)
    for r in range(R):
        x = rng.uniform(0, 10, n)
        sd = 0.2 + 0.3 * x ** 2
        y = 2.0 + beta1 * x + rng.normal(0, sd)
        beta, se_o, se_h, *_ = ols_fit(x, y)
        b1[r] = beta[1]
        cov_ols += abs(beta[1] - beta1) <= tcrit * se_o[1]
        cov_hc3 += abs(beta[1] - beta1) <= tcrit * se_h[1]
    cov_ols /= R
    cov_hc3 /= R
    print(f"  beta1 평균 {b1.mean():.4f} (참값 {beta1}), 표준편차 "
          f"{b1.std(ddof=1):.4f}")
    print(f"  OLS 구간 포함률 {cov_ols:.4f}, HC3 구간 포함률 {cov_hc3:.4f}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.15, 1]})

    while True:                       # 전형적인 한 벌을 고른다
        x = rng.uniform(0, 10, n)
        y = 2.0 + beta1 * x + rng.normal(0, 0.2 + 0.3 * x ** 2)
        beta, se_o, se_h, e, fitted, _ = ols_fit(x, y)
        if abs(beta[1] - beta1) < 0.08:
            break
    xs = np.linspace(0, 10, 50)
    ax1.scatter(x, y, s=32, color=BLUE, alpha=0.55, edgecolors="white",
                lw=0.5)
    ax1.plot(xs, 2.0 + beta1 * xs, color=INK, lw=2.0, label="참 회귀직선")
    ax1.plot(xs, beta[0] + beta[1] * xs, color=RED, lw=2.0, ls=(0, (5, 3)),
             label=f"OLS 적합 (기울기 {beta[1]:.3f})")
    ax1.fill_between(xs, 2 + beta1 * xs - 2 * (0.2 + 0.3 * xs ** 2),
                     2 + beta1 * xs + 2 * (0.2 + 0.3 * xs ** 2),
                     color=MUTED, alpha=0.18)
    ax1.set_xlim(0, 10)
    ax1.set_xlabel("$x$", fontsize=11, color=INK)
    ax1.set_ylabel("$y$", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper left")
    ax1.set_title("오차 표준편차가 $x$ 와 함께 커진다 ($n = 100$)",
                  fontsize=11.5, color=INK, pad=9)

    ax2.hist(b1, bins=60, color=BLUE, alpha=0.40, density=True)
    ax2.axvline(beta1, color=INK, lw=2.0)
    ytop = ax2.get_ylim()[1]
    ax2.text(beta1 + 0.10, ytop * 0.97, "참값 $\\beta_1 = 1$", fontsize=10.5,
             color=INK, va="top")
    ax2.text(-1.45, ytop * 0.80,
             f"추정값 평균 {b1.mean():.4f}\n이분산이 있어도 편향되지 않는다",
             fontsize=10.5, color=BLUE, va="top")
    ax2.text(-1.45, ytop * 0.58,
             f"95% 구간이 참값을 덮은 비율\n"
             f"  OLS 표준오차 : {cov_ols:.3f}\n"
             f"  HC3 로버스트 : {cov_hc3:.3f}",
             fontsize=10.5, color=RED, va="top")
    ax2.set_xlabel("$\\hat\\beta_1$   (4000회 반복)", fontsize=11, color=INK)
    ax2.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax2)
    ax2.set_title("추정값은 맞고, 표준오차가 틀린다", fontsize=11.5,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "hetero_unbiased_se.png")


# === 그림 3. BP 검정이 실제로 보는 것 ===
def fig_bp():
    rng = np.random.default_rng(11)
    n = 120
    x = rng.uniform(0, 10, n)
    y_ho = 2 + x + rng.normal(0, 2.0, n)
    y_he = 2 + x + rng.normal(0, 0.4 + 0.5 * x)

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    for col, (y, tag, color) in enumerate(
            [(y_ho, "등분산 자료", GREEN), (y_he, "이분산 자료", RED)]):
        beta, se_o, se_h, e, fitted, _ = ols_fit(x, y)
        ax = axes[0][col]
        ax.scatter(fitted, e, s=30, color=color, alpha=0.55,
                   edgecolors="white", lw=0.5)
        ax.axhline(0, color=INK, lw=1.2)
        ax.set_xlabel("적합값 $\\hat{y}$", fontsize=11, color=INK)
        ax.set_ylabel("잔차 $e$", fontsize=11, color=INK)
        ax.set_ylim(-14, 14)
        clean(ax)
        ax.set_title(f"{tag} — 잔차 그림", fontsize=11.5, color=INK, pad=8)

        ax = axes[1][col]
        e2 = e ** 2
        X = np.column_stack([np.ones(n), x])
        g = np.linalg.lstsq(X, e2, rcond=None)[0]
        r2 = 1 - ((e2 - X @ g) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
        bp = n * r2
        p = stats.chi2.sf(bp, 1)
        ax.scatter(x, e2, s=30, color=color, alpha=0.5, edgecolors="white",
                   lw=0.5)
        xs = np.linspace(0, 10, 50)
        ax.plot(xs, g[0] + g[1] * xs, color=INK, lw=2.2)
        ax.set_xlabel("$x$", fontsize=11, color=INK)
        ax.set_ylabel("제곱잔차 $e^2$", fontsize=11, color=INK)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 150)
        clean(ax)
        ax.text(0.03, 0.95,
                f"보조회귀 기울기 {g[1]:.3f}\n$R^2$ = {r2:.4f}\n"
                f"BP = $n R^2$ = {bp:.3f},  $p$ = {p:.4f}",
                transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
        ax.set_title(f"{tag} — 보조회귀", fontsize=11.5, color=INK, pad=8)
        print(f"  {tag}: 기울기 {g[1]:.4f}, R2 {r2:.4f}, BP {bp:.3f}, "
              f"p {p:.4f}")

    fig.tight_layout()
    save(fig, "breusch_pagan_mechanism.png")


# === 그림 4. 변동성이 변한 것처럼 보이는 까닭 ===
def fig_vol_noise():
    rng = np.random.default_rng(5)
    T, win = 900, 60
    r = rng.standard_t(5, T) / np.sqrt(5 / 3) * 0.01      # 표준편차 1%
    roll = np.array([r[i - win:i].std(ddof=1)
                     for i in range(win, T)]) * 100
    true_sd = 1.0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})

    t = np.arange(win, T)
    ax1.plot(t, roll, color=BLUE, lw=1.5)
    ax1.axhline(true_sd, color=RED, lw=1.8, ls=(0, (5, 3)))
    ax1.text(T - 5, 0.90, "참 변동성 1.00%", fontsize=10.5,
             color=RED, ha="right")
    ax1.axhline(roll.max(), color=MUTED, lw=1.0, ls=":")
    ax1.axhline(roll.min(), color=MUTED, lw=1.0, ls=":")
    ax1.text(win + 5, roll.max() + 0.03, f"최대 {roll.max():.2f}%",
             fontsize=10, color=INK)
    ax1.text(win + 5, roll.min() - 0.10, f"최소 {roll.min():.2f}%",
             fontsize=10, color=INK)
    ax1.set_xlim(win, T)
    ax1.set_ylim(0.4, 2.0)
    ax1.set_xlabel("거래일", fontsize=11, color=INK)
    ax1.set_ylabel("60일 이동 표준편차 (%)", fontsize=11, color=INK)
    clean(ax1)
    ax1.set_title("변동성이 **일정한** $t_5$ 수익률에서 잰 이동 변동성"
                  .replace("**", ""), fontsize=11.5, color=INK, pad=9)
    print(f"  이동 변동성 최소 {roll.min():.3f}%, 최대 {roll.max():.3f}%, "
          f"비 {roll.max() / roll.min():.2f}")

    names = ["$F$ 검정", "브라운–포사이드", "플리그너–킬린"]
    vals = [0.195, 0.044, 0.045]
    cols = [RED, GREEN, PURPLE]
    ys = np.arange(3)
    ax2.barh(ys, vals, 0.5, color=cols, alpha=0.85)
    for y, v in zip(ys, vals):
        ax2.text(v + 0.014, y, f"{v:.3f}", va="center", fontsize=10.5,
                 color=INK)
    ax2.axvline(0.05, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax2.text(0.047, -0.45, "명목 0.05", fontsize=10, color=INK, ha="right")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(names, fontsize=10.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_ylim(2.8, -0.65)
    ax2.set_xlim(0, 0.25)
    ax2.set_xlabel("경험적 크기", fontsize=11, color=INK)
    clean(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("두 기간 각 60일, 참 변동성은 같다", fontsize=11.5,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "volatility_estimation_noise.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_pretest()
    fig_bp()
    fig_vol_noise()
    fig_hetero_se()
