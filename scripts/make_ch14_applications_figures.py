r"""14장 응용 절 세 쪽의 개념 그림을 만든다.

만드는 파일:
  ch14/applications/img/residual_not_predictor.png    정규성은 잔차에 요구된다
  ch14/applications/img/anova_pooled_vs_residuals.png 원자료가 아니라 잔차를 본다
  ch14/applications/img/var_tail_ratio.png            정규 VaR 는 언제부터 위험한가

실행:  python3 scripts/make_ch14_applications_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch14/applications/img/"
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


# ===================================================================
# 그림 1. 설명변수도 반응변수도 정규가 아닌데 잔차만 정규다
# ===================================================================
def fig_residual_not_predictor():
    rng = np.random.default_rng(1414)
    n = 200
    x = rng.exponential(3.0, size=n)          # 설명변수는 심하게 치우쳤다
    eps = rng.normal(0, 2.0, size=n)          # 오차만 정규다
    y = 2.0 + 3.0 * x + eps

    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta

    p_x = stats.shapiro(x).pvalue
    p_y = stats.shapiro(y).pvalue
    p_r = stats.shapiro(res).pvalue
    print(f"n={n}  Shapiro p:  X={p_x:.3g}  Y={p_y:.3g}  잔차={p_r:.3g}")
    print(f"  적합: y = {beta[0]:.2f} + {beta[1]:.2f} x")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))

    ax = axes[0]
    ax.scatter(x, y, s=14, color=MUTED, alpha=0.75, edgecolor="none")
    xx = np.linspace(0, x.max() * 1.02, 50)
    ax.plot(xx, beta[0] + beta[1] * xx, color=RED, lw=2)
    ax.set_xlabel("설명변수 $X$", fontsize=10.5, color=INK)
    ax.set_ylabel("반응변수 $Y$", fontsize=10.5, color=INK)
    ax.set_title("자료와 적합선", fontsize=11.5, color=INK)
    clean(ax)

    ax = axes[1]
    ax.hist(x, bins=28, density=True, color=ORANGE_F, edgecolor=ORANGE,
            linewidth=0.8,
            label="$X$  (Shapiro $p = 1.3 \\times 10^{-14}$)")
    ax.hist(y / y.std() * x.std(), bins=28, density=True, histtype="step",
            color=PURPLE, linewidth=1.8,
            label="$Y$ (척도 맞춤)  (Shapiro $p = 2.9 \\times 10^{-13}$)")
    ax.set_xlabel("값", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("둘 다 정규와 거리가 멀다", fontsize=11.5, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)
    ax.set_yticks([])

    ax = axes[2]
    m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    rs = np.sort(res)
    b, a = np.polyfit(m, rs, 1)
    ax.plot(m, a + b * m, color=MUTED, lw=1.5, ls="--")
    ax.scatter(m, rs, s=13, color=GREEN, alpha=0.85, edgecolor="none")
    ax.set_xlabel("정규 점수", fontsize=10.5, color=INK)
    ax.set_ylabel("정렬한 잔차", fontsize=10.5, color=INK)
    ax.set_title(f"잔차의 Q-Q 그림  (Shapiro $p = {p_r:.2f}$)",
                 fontsize=11.5, color=INK)
    clean(ax)

    fig.suptitle("$X$ 도 $Y$ 도 정규성을 압도적으로 기각하지만 잔차는 정규다",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "residual_not_predictor.png")


# ===================================================================
# 그림 2. 분산분석에서는 원자료가 아니라 잔차를 본다
# ===================================================================
def fig_anova_pooled_vs_residuals():
    rng = np.random.default_rng(1415)
    n_g = 60
    mus = [4.0, 9.0, 14.0]
    groups = [rng.normal(m, 1.4, size=n_g) for m in mus]
    pooled = np.concatenate(groups)
    res = np.concatenate([g - g.mean() for g in groups])

    p_pool = stats.shapiro(pooled).pvalue
    p_res = stats.shapiro(res).pvalue
    print(f"집단당 {n_g}개  합친 원자료 Shapiro p={p_pool:.3g}   "
          f"잔차 Shapiro p={p_res:.3f}")
    for i, g in enumerate(groups):
        print(f"   집단 {i+1} 자체의 Shapiro p={stats.shapiro(g).pvalue:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    for g, m, col in zip(groups, mus, [BLUE, ORANGE, GREEN]):
        ax.hist(g, bins=np.arange(0, 20, 0.6), color=col, alpha=0.55,
                edgecolor="none")
    ax.hist(pooled, bins=np.arange(0, 20, 0.6), histtype="step",
            color=INK, linewidth=1.8, label="합친 원자료")
    ax.set_xlim(0, 19)
    ax.set_xlabel("관측값", fontsize=10.5, color=INK)
    ax.set_ylabel("빈도", fontsize=10.5, color=INK)
    ax.set_title("합친 원자료  (Shapiro $p = 1.2 \\times 10^{-5}$)",
                 fontsize=11.5, color=INK)
    ax.text(0.03, 0.97,
            "집단마다 평균이 다르므로\n봉우리가 셋으로 갈린다",
            transform=ax.transAxes, fontsize=9.5, color=INK, va="top",
            ha="left", linespacing=1.6)
    clean(ax)

    ax = axes[1]
    ax.hist(res, bins=26, density=True, color="#ECEFF1", edgecolor=MUTED,
            linewidth=1.0)
    g = np.linspace(res.min() * 1.1, res.max() * 1.1, 300)
    ax.plot(g, stats.norm.pdf(g, 0, res.std(ddof=1)), color=RED, lw=2.2,
            label="같은 분산의 정규밀도")
    ax.set_xlabel("잔차 $X_{ij} - \\bar{X}_{i\\cdot}$", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(f"집단 평균을 뺀 잔차  (Shapiro $p = {p_res:.2f}$)",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)

    fig.suptitle("세 집단 모두 완벽한 정규분포에서 뽑았다 (집단당 60개)",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "anova_pooled_vs_residuals.png")


# ===================================================================
# 그림 3. 정규 VaR 는 어느 수준부터 위험을 과소평가하는가
# ===================================================================
def fig_var_tail_ratio():
    alphas = np.logspace(np.log10(0.25), np.log10(1e-4), 300)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    # --- 왼쪽: 분위수 비 ---
    ax = axes[0]
    z = stats.norm.ppf(alphas)
    for nu, col in [(4, PURPLE), (5, ORANGE), (8, BLUE)]:
        q = stats.t.ppf(alphas, nu) / np.sqrt(nu / (nu - 2))
        ratio = q / z
        ax.plot(alphas, ratio, color=col, lw=2.2, label=f"$t_{{{nu}}}$")
        if nu == 5:
            for a in (0.05, 0.01, 0.001):
                r = (stats.t.ppf(a, 5) / np.sqrt(5 / 3)) / stats.norm.ppf(a)
                ax.plot([a], [r], "o", color=col, ms=6.5, zorder=5)
                print(f"t5  alpha={a}: 표준화 분위수="
                      f"{stats.t.ppf(a,5)/np.sqrt(5/3):.3f}  "
                      f"정규={stats.norm.ppf(a):.3f}  비={r:.3f}")
            ax.annotate("$\\alpha = 0.05$ 에서는 비가 $0.95$ —\n"
                        "이쪽에서는 정규 쪽이 오히려 보수적이다",
                        xy=(0.05, 0.949), xytext=(0.022, 0.87),
                        fontsize=9.5, color=ORANGE, ha="left", va="center",
                        linespacing=1.5,
                        arrowprops=dict(arrowstyle="->", color=ORANGE,
                                        lw=1.2))
            ax.annotate("$\\alpha = 0.01$\n비 $1.12$", (0.01, 1.121),
                        textcoords="offset points", xytext=(-4, 18),
                        fontsize=9.5, color=ORANGE, ha="right",
                        linespacing=1.5)
            ax.annotate("$\\alpha = 0.001$\n비 $1.48$", (0.001, 1.477),
                        textcoords="offset points", xytext=(6, -4),
                        fontsize=9.5, color=ORANGE, ha="left",
                        linespacing=1.5)

    # 비가 1 이 되는 지점
    q5 = stats.t.ppf(alphas, 5) / np.sqrt(5 / 3)
    cross = alphas[np.argmin(np.abs(q5 / z - 1.0))]
    ax.plot([cross], [1.0], "*", color=RED, ms=15, zorder=6)
    print(f"t5 에서 비가 1 이 되는 alpha = {cross:.4f}")

    ax.axhline(1.0, color=MUTED, lw=1.4, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.25, 1e-4)                      # 오른쪽으로 갈수록 극단
    ax.set_xticks([0.1, 0.01, 0.001, 0.0001])
    ax.set_xticklabels(["0.1", "0.01", "0.001", "0.0001"])
    ax.minorticks_off()
    ax.set_ylim(0.80, 2.3)
    ax.set_xlabel("신뢰수준 $\\alpha$  (오른쪽으로 갈수록 극단)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("정규 VaR 대비 참 VaR 의 비", fontsize=10.5, color=INK)
    ax.set_title("분산을 맞추고 견준 꼬리 분위수", fontsize=11.5, color=INK)
    ax.text(0.23, 1.02, "이 선 위는 과소평가", fontsize=9.5, color=INK,
            ha="left", va="bottom")
    ax.legend(fontsize=10, frameon=False, loc="upper left", labelcolor=INK)
    clean(ax)

    # --- 오른쪽: 꼬리 확률 ---
    ax = axes[1]
    k = np.linspace(1.0, 5.0, 300)
    p_norm = stats.norm.cdf(-k)
    p_t = stats.t.cdf(-k * np.sqrt(5 / 3), 5)
    ax.plot(k, p_norm, color=BLUE, lw=2.4, label="정규분포")
    ax.plot(k, p_t, color=ORANGE, lw=2.4, label="분산을 맞춘 $t_5$")
    ax.set_yscale("log")
    ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    ax.set_yticklabels(["0.1", "0.01", "0.001", "0.0001", "0.00001",
                        "0.000001"])
    ax.minorticks_off()
    ax.set_ylim(2e-7, 0.3)
    ax.set_xlim(1.0, 5.0)
    for kk in (2.5, 3.5, 4.5):
        rn = stats.norm.cdf(-kk)
        rt = stats.t.cdf(-kk * np.sqrt(5 / 3), 5)
        print(f"k={kk}: 정규 {rn:.3e}  t5 {rt:.3e}  배수 {rt/rn:.1f}")
        ax.plot([kk, kk], [rn, rt], color=MUTED, lw=1.0, ls=":")
        ax.text(kk, rt * 1.7, f"{rt/rn:.0f}배" if rt / rn >= 10
                else f"{rt/rn:.1f}배", fontsize=9.5,
                color=RED, ha="center", va="bottom")
    ax.set_xlabel("평균에서 떨어진 표준편차 수 $k$", fontsize=10.5, color=INK)
    ax.set_ylabel("$k$ 표준편차보다 큰 손실이 날 확률 (로그 눈금)",
                  fontsize=10.5, color=INK)
    ax.set_title("같은 분산, 다른 꼬리", fontsize=11.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "var_tail_ratio.png")


if __name__ == "__main__":
    fig_residual_not_predictor()
    fig_anova_pooled_vs_residuals()
    fig_var_tail_ratio()
