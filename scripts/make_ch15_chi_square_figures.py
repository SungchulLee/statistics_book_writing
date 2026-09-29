r"""15장 카이제곱 분산 검정 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch15/chi_square_test/img/two_sided_critical_values.png  꼬리마다 alpha/2 를 써야 하는 이유
  ch15/chi_square_test/img/chi2_shift_with_sigma.png      sigma 가 커지면 T 가 오른쪽으로 밀린다
  ch15/chi_square_test/img/cochran_decomposition.png      자유도 하나가 어디로 갔는가
  ch15/chi_square_test/img/ci_width_by_n.png              분산 신뢰구간의 비대칭과 폭

실행:  python3 scripts/make_ch15_chi_square_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch15/chi_square_test/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def bare(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# === 그림 1. 양측검정의 임계값: alpha 씩인가 alpha/2 씩인가 ===
def fig_two_sided():
    df = 24
    x = np.linspace(0, 65, 1400)
    y = stats.chi2.pdf(x, df)

    wrong = (stats.chi2.ppf(0.05, df), stats.chi2.ppf(0.95, df))
    right = (stats.chi2.ppf(0.025, df), stats.chi2.ppf(0.975, df))
    obs = 30.0

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True)

    for ax, (lo, hi), color, size, head in [
        (axes[0], wrong, RED, 0.10, "흔한 실수: 양쪽에 5%씩"),
        (axes[1], right, GREEN, 0.05, "올바른 방법: 양쪽에 2.5%씩"),
    ]:
        ax.plot(x, y, color=INK, lw=1.7)
        mid = (x >= lo) & (x <= hi)
        ax.fill_between(x[mid], y[mid], color="#ECEFF1", alpha=0.9)
        ax.fill_between(x, y, where=x <= lo, color=color, alpha=0.55)
        ax.fill_between(x, y, where=x >= hi, color=color, alpha=0.55)
        for v in (lo, hi):
            ax.vlines(v, 0, 0.056, color=color, lw=1.5)
        ax.vlines(obs, 0, 0.063, color=BLUE, lw=1.8, ls=(0, (4, 3)))
        bare(ax)
        ax.set_xlim(0, 65)
        ax.set_ylim(0, 0.090)
        ax.text(0.012, 0.97, head, transform=ax.transAxes, fontsize=12,
                color=color, fontweight="bold", va="top")
        ax.text(0.012, 0.84,
                f"기각역 넓이 합 = {size:.2f}  →  실제로는 {size * 100:.0f}% 검정",
                transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
        ax.text(lo - 0.9, 0.0315, f"{lo:.3f}", fontsize=10, color=color,
                ha="right", va="center")
        ax.text(hi + 0.9, 0.0315, f"{hi:.3f}", fontsize=10, color=color,
                ha="left", va="center")

    axes[0].annotate("관측값 $\\chi^2 = 30$", xy=(obs, 0.056),
                     xytext=(obs + 6.5, 0.074), fontsize=10.5, color=BLUE,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    axes[1].set_xlabel("$\\chi^2$ 통계량 (자유도 24)", fontsize=11, color=INK)
    axes[0].set_title("같은 자료, 같은 $\\alpha = 0.05$ — 임계값을 어떻게 잡느냐만 다르다",
                      fontsize=12.5, color=INK, pad=10)
    fig.subplots_adjust(hspace=0.42)
    save(fig, "two_sided_critical_values.png")
    print(f"  wrong = {wrong[0]:.3f}, {wrong[1]:.3f}   "
          f"right = {right[0]:.3f}, {right[1]:.3f}")


# === 그림 2. sigma 가 커지면 T 가 밀린다 ===
def fig_shift_with_sigma():
    n, df = 100, 99
    x = np.linspace(50, 175, 1400)
    y = stats.chi2.pdf(x, df)
    lo = stats.chi2.ppf(0.025, df)
    hi = stats.chi2.ppf(0.975, df)

    obs = [(1.00, 101.58, 0.819), (1.05, 111.99, 0.351), (1.10, 122.92, 0.104),
           (1.15, 134.34, 0.021), (1.20, 146.28, 0.003)]
    cols = [MUTED, MUTED, ORANGE, RED, RED]

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    mid = (x >= lo) & (x <= hi)
    ax.fill_between(x[mid], y[mid], color=BLUE_F, alpha=0.95)
    ax.fill_between(x, y, where=x >= hi, color=RED, alpha=0.45)
    ax.fill_between(x, y, where=x <= lo, color=RED, alpha=0.45)
    ax.plot(x, y, color=INK, lw=1.8)
    top = stats.chi2.pdf(df - 2, df)
    for v in (lo, hi):
        ax.vlines(v, 0, top * 1.04, color=RED, lw=1.3, ls=(0, (5, 3)))

    for i, ((sig, t, p), c) in enumerate(zip(obs, cols)):
        yy = top * (1.68 - 0.145 * i)
        ax.plot([t], [yy], marker="o", ms=7, color=c, zorder=5)
        ax.plot([t, t], [0, yy], color=c, lw=1.0, alpha=0.45)
        ax.text(t + 2.2, yy, f"$\\sigma$ = {sig:.2f}   T = {t:.2f}   p = {p:.3f}",
                fontsize=10, color=c, va="center")

    ax.text(hi - 1.5, top * 0.30, f"상단 임계값 {hi:.2f}", fontsize=10,
            color=RED, ha="right", rotation=90, va="bottom")
    ax.text(lo + 1.5, top * 0.30, f"하단 임계값 {lo:.2f}", fontsize=10,
            color=RED, ha="left", rotation=90, va="bottom")
    ax.text(df, top * 0.42, "기각하지 못하는 영역", fontsize=10.5,
            color=BLUE, ha="center")
    ax.set_xlim(50, 185)
    ax.set_ylim(0, top * 1.95)
    ax.set_xlabel("$T = (n-1)S^2/\\sigma_0^2$", fontsize=11, color=INK)
    bare(ax)
    ax.set_title("$H_0\\colon \\sigma^2 = 1$ 아래의 기준분포 $\\chi^2_{99}$ 와 관측된 T ($n = 100$)",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "chi2_shift_with_sigma.png")
    print(f"  df=99 임계값 {lo:.3f}, {hi:.3f}")


# === 그림 3. 자유도 하나가 어디로 갔는가 ===
def fig_cochran():
    rng = np.random.default_rng(15)
    n, R = 6, 80000
    mu, sigma = 0.0, 1.0
    X = rng.normal(mu, sigma, size=(R, n))
    xbar = X.mean(axis=1)

    total = ((X - mu) ** 2).sum(axis=1) / sigma ** 2        # chi2_n
    within = ((X - xbar[:, None]) ** 2).sum(axis=1) / sigma ** 2   # chi2_{n-1}
    mean_part = n * (xbar - mu) ** 2 / sigma ** 2            # chi2_1

    r = np.corrcoef(within, mean_part)[0, 1]
    gap = np.abs(total - (within + mean_part)).max()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})

    grid = np.linspace(0.01, 22, 600)
    for s, dfree, color, label in [
        (total, n, INK, "$\\sum (X_i-\\mu)^2/\\sigma^2$"),
        (within, n - 1, BLUE, "$\\sum (X_i-\\bar{X})^2/\\sigma^2$"),
        (mean_part, 1, ORANGE, "$n(\\bar{X}-\\mu)^2/\\sigma^2$"),
    ]:
        ax1.hist(s, bins=np.linspace(0, 22, 90), density=True, color=color,
                 alpha=0.28)
        ax1.plot(grid, stats.chi2.pdf(grid, dfree), color=color, lw=2.0,
                 label=f"{label}  →  $\\chi^2_{{{dfree}}}$")
    ax1.set_xlim(0, 22)
    ax1.set_ylim(0, 0.32)
    ax1.set_xlabel("값", fontsize=11, color=INK)
    ax1.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper right")
    ax1.text(11.6, 0.205, "자유도가 더해진다:  6 = 5 + 1", fontsize=11,
             color=INK, ha="left")
    ax1.set_title(f"$n = {n}$, 정규 자료 {R // 1000}천 회 — 막대는 모의실험, 선은 이론",
                  fontsize=11.5, color=INK, pad=9)

    idx = rng.choice(R, 3000, replace=False)
    ax2.scatter(within[idx], mean_part[idx], s=6, color=PURPLE, alpha=0.28,
                edgecolors="none")
    ax2.set_xlim(0, 22)
    ax2.set_ylim(0, 11)
    ax2.set_xlabel("$\\sum (X_i-\\bar{X})^2/\\sigma^2$  (자유도 5)",
                   fontsize=10.5, color=INK)
    ax2.set_ylabel("$n(\\bar{X}-\\mu)^2/\\sigma^2$  (자유도 1)",
                   fontsize=10.5, color=INK)
    clean(ax2)
    ax2.text(0.97, 0.94, f"표본상관 r = {r:+.4f}", transform=ax2.transAxes,
             fontsize=11, color=PURPLE, ha="right", va="top",
             fontweight="bold")
    ax2.text(0.97, 0.85, "구름에 기울기가 없다 — 두 조각은 독립이다",
             transform=ax2.transAxes, fontsize=9.5, color=INK, ha="right",
             va="top")
    ax2.set_title("코크런 정리가 말하는 독립성", fontsize=11.5, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "cochran_decomposition.png")
    print(f"  r = {r:+.5f}, 항등식 최대오차 = {gap:.2e}")


# === 그림 4. 분산 신뢰구간의 비대칭과 폭 ===
def fig_ci_width():
    s2 = 120.0
    ns = [5, 10, 25, 50, 100, 500]
    rows = []
    for n in ns:
        df = n - 1
        lo = df * s2 / stats.chi2.ppf(0.975, df)
        hi = df * s2 / stats.chi2.ppf(0.025, df)
        rows.append((n, lo, hi, hi / lo))

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ypos = np.arange(len(ns))[::-1]
    for (n, lo, hi, ratio), yv in zip(rows, ypos):
        color = RED if n <= 10 else (ORANGE if n <= 25 else BLUE)
        ax.plot([lo, hi], [yv, yv], color=color, lw=7, alpha=0.75,
                solid_capstyle="butt")
        ax.plot([s2], [yv], marker="D", ms=7, color=INK, zorder=5)
        ax.text(lo - 6, yv, f"{lo:.1f}", ha="right", va="center", fontsize=9.5,
                color=color)
        ax.text(hi + 6, yv, f"{hi:.1f}", ha="left", va="center", fontsize=9.5,
                color=color)

    ax.axvline(s2, color=INK, lw=1.0, ls=":")
    ax.text(s2 + 30, len(ns) - 0.42, "점추정값 $S^2 = 120$", fontsize=10.5,
            color=INK, ha="left")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"n = {n}\n상한/하한 {r:.2f}" for (n, _, _, r) in rows],
                       fontsize=10.5, color=INK)
    ax.set_xlim(0, 1080)
    ax.set_ylim(-0.7, len(ns) - 0.05)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel("$\\sigma^2$  (시간$^2$)", fontsize=11, color=INK)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.set_title("표본분산을 120으로 고정했을 때의 95% 신뢰구간",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "ci_width_by_n.png")
    for n, lo, hi, ratio in rows:
        print(f"  n={n:3d}: ({lo:7.2f}, {hi:8.2f})  비 {ratio:5.2f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_two_sided()
    fig_shift_with_sigma()
    fig_cochran()
    fig_ci_width()
