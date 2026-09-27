r"""9장 분산 관련 두 쪽의 그림을 생성한다.

  ch09/one_sample_tests/img/chi2_rejection_asymmetry.png
      카이제곱 양측 기각역이 좌우 비대칭이라는 것. 자유도가 커지면
      대칭에 가까워지고, 작을 때는 "대칭 근사"가 음수까지 내려간다.

  ch09/two_sample_tests/img/var_ratio_skew.png
      두 분산이 같을 때 S1^2/S2^2 의 표본분포가 1 을 중심으로 비대칭이며,
      분산비의 구간은 덧셈이 아니라 곱셈 척도에서 대칭이라는 것.

실행:  python3 scripts/make_ch09_final_variance.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG 로 커밋되므로 CI 에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

ONE = "docs/ch09/one_sample_tests/img/"
TWO = "docs/ch09/two_sample_tests/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 카이제곱 기각역의 비대칭
# ==================================================================
def chi2_rejection_asymmetry():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # ---------- (a) 자유도 11 의 양측 기각역 ----------
    ax = axes[0]
    nu = 11
    lo = stats.chi2.ppf(0.025, nu)
    hi = stats.chi2.ppf(0.975, nu)
    x = np.linspace(0.01, 34, 900)
    y = stats.chi2.pdf(x, nu)

    ax.plot(x, y, color=BLUE, linewidth=2.1, zorder=4)
    xl = x[x <= lo]
    xr = x[x >= hi]
    ax.fill_between(xl, 0, stats.chi2.pdf(xl, nu), color=ORANGE_F, zorder=2)
    ax.fill_between(xr, 0, stats.chi2.pdf(xr, nu), color=ORANGE_F, zorder=2)
    ax.plot(xl, stats.chi2.pdf(xl, nu), color=ORANGE, linewidth=1.6, zorder=3)
    ax.plot(xr, stats.chi2.pdf(xr, nu), color=ORANGE, linewidth=1.6, zorder=3)

    for v in (lo, hi):
        ax.plot([v, v], [0, stats.chi2.pdf(v, nu)], color=ORANGE,
                linewidth=1.3, linestyle=(0, (4, 3)), zorder=5)
    ax.plot([nu, nu], [0, stats.chi2.pdf(nu, nu)], color=INK,
            linewidth=1.3, linestyle=(0, (2, 3)), zorder=5)

    ymax = y.max()
    ax.annotate("0.025", xy=(lo * 0.55, stats.chi2.pdf(lo, nu) * 0.35),
                xytext=(1.0, ymax * 0.55), fontsize=10.5, color=ORANGE,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate("0.025", xy=(hi * 1.10, stats.chi2.pdf(hi, nu) * 0.40),
                xytext=(27.0, ymax * 0.55), fontsize=10.5, color=ORANGE,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    # 평균까지의 거리 두 개를 이중화살표로
    yb = ymax * 1.06
    ax.annotate("", xy=(lo, yb), xytext=(nu, yb),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
    ax.annotate("", xy=(nu, yb), xytext=(hi, yb),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.5))
    ax.text((lo + nu) / 2, yb + ymax * 0.035, f"{nu - lo:.2f}", fontsize=11,
            color=GREEN, ha="center", va="bottom", fontweight="bold")
    ax.text((nu + hi) / 2, yb + ymax * 0.035, f"{hi - nu:.2f}", fontsize=11,
            color=PURPLE, ha="center", va="bottom", fontweight="bold")

    ax.set_xticks([lo, nu, hi])
    ax.set_xticklabels([f"{lo:.2f}", f"{nu}", f"{hi:.2f}"])
    bare_axis(ax)
    ax.set_xlim(0, 34)
    ax.set_ylim(0, ymax * 1.30)
    ax.set_xlabel(r"$\chi^2 = (n-1)S^2/\sigma_0^2$", fontsize=11, color=INK)
    ax.set_title("(a) 자유도 11 의 양측 기각역 — 두 꼬리의 임계값이 대칭이 아니다",
                 fontsize=11.5, color=INK, pad=14, loc="left")
    ax.text(0.99, 0.97,
            f"가운데(평균)에서 왼쪽으로 {nu - lo:.2f},\n"
            f"오른쪽으로 {hi - nu:.2f} — {(hi - nu) / (nu - lo):.2f} 배 차이",
            transform=ax.transAxes, fontsize=10, color=INK,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F5F7F8",
                      edgecolor=MUTED, linewidth=0.8))

    # ---------- (b) s^2/sigma0^2 눈금 위의 채택역 ----------
    ax = axes[1]
    ns = [5, 10, 20, 50, 200]
    ys = np.arange(len(ns))[::-1]

    ax.plot([1, 1], [-0.45, len(ns) - 0.55], color=INK, linewidth=1.2,
            linestyle=(0, (2, 3)), zorder=2)
    for yy, n in zip(ys, ns):
        nu = n - 1
        L = stats.chi2.ppf(0.025, nu) / nu
        U = stats.chi2.ppf(0.975, nu) / nu
        half = 1.96 * np.sqrt(2 / nu)

        # 대칭 근사 (정규): 아래쪽 얇은 회색 막대
        ax.plot([1 - half, 1 + half], [yy - 0.20, yy - 0.20], color=MUTED,
                linewidth=2.0, linestyle=(0, (4, 2)), zorder=3,
                solid_capstyle="butt")
        # 정확한 카이제곱 채택역
        ax.plot([L, U], [yy + 0.09, yy + 0.09], color=BLUE, linewidth=7.5,
                solid_capstyle="butt", zorder=4)
        ax.plot([1, 1], [yy - 0.02, yy + 0.20], color="white", linewidth=1.6,
                zorder=5)
        ax.text(L - 0.06, yy + 0.09, f"{L:.2f}", fontsize=9.5, color=BLUE,
                ha="right", va="center")
        ax.text(U + 0.06, yy + 0.09, f"{U:.2f}", fontsize=9.5, color=BLUE,
                ha="left", va="center")
        ax.text(3.50, yy + 0.02, f"{(U - 1) / (1 - L):.2f} 배",
                fontsize=10, color=PURPLE, ha="right", va="center")

    top = len(ns) - 1 + 0.75
    ax.text(3.50, top, "오른쪽 여유 ÷ 왼쪽 여유", fontsize=9.5,
            color=PURPLE, ha="right", va="center")
    ax.plot([-0.50, -0.14], [top, top], color=MUTED,
            linewidth=2.0, linestyle=(0, (4, 2)), solid_capstyle="butt")
    ax.text(-0.06, top, r"대칭 근사  $1 \pm 1.96\sqrt{2/(n-1)}$",
            fontsize=9.5, color=INK, ha="left", va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"n = {n}" for n in ns])
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10.5)
    ax.set_xlim(-0.62, 3.62)
    ax.set_ylim(-1.05, top + 0.28)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel(r"$s^2/\sigma_0^2$", fontsize=11, color=INK)
    ax.set_title("(b) 같은 채택역을 분산비 눈금에 옮기면 — n 이 커질수록 대칭에 가까워진다",
                 fontsize=11.5, color=INK, pad=14, loc="left")
    ax.text(-0.50, -0.80,
            "n = 5 에서는 대칭 근사의 왼쪽 끝이 음수다 — 분산비는 음수가 될 수 없다",
            fontsize=9.5, color=RED, ha="left", va="center")

    fig.tight_layout(w_pad=2.4)
    save(fig, ONE + "chi2_rejection_asymmetry.png")

    # 본문에 옮겨 적을 수치
    print("  [a] nu=11: lo=%.4f hi=%.4f  좌 %.4f 우 %.4f 비 %.4f"
          % (stats.chi2.ppf(0.025, 11), stats.chi2.ppf(0.975, 11),
             11 - stats.chi2.ppf(0.025, 11),
             stats.chi2.ppf(0.975, 11) - 11,
             (stats.chi2.ppf(0.975, 11) - 11) / (11 - stats.chi2.ppf(0.025, 11))))
    for n in ns:
        nu = n - 1
        L = stats.chi2.ppf(0.025, nu) / nu
        U = stats.chi2.ppf(0.975, nu) / nu
        print("  [b] n=%3d  L=%.4f U=%.4f  비=%.4f  대칭근사=(%.4f, %.4f)"
              % (n, L, U, (U - 1) / (1 - L),
                 1 - 1.96 * np.sqrt(2 / nu), 1 + 1.96 * np.sqrt(2 / nu)))


# ==================================================================
# 2. 분산비의 표본분포는 1 을 중심으로 비대칭
# ==================================================================
def var_ratio_skew():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # ---------- (a) 두 분산이 같을 때 S1^2/S2^2 의 분포 ----------
    ax = axes[0]
    cases = [(5, GREEN), (10, BLUE), (30, PURPLE)]
    x = np.linspace(0.005, 4.2, 1200)
    ymax = 0.0
    for n, c in cases:
        nu = n - 1
        d = stats.f.pdf(x, nu, nu)
        ymax = max(ymax, d.max())
        ax.plot(x, d, color=c, linewidth=2.1,
                label=f"$n_1 = n_2 = {n}$", zorder=4)

    nu = 9
    mode = (nu - 2) / nu * nu / (nu + 2)
    mean = nu / (nu - 2)
    xr = x[x >= 1]
    ax.fill_between(xr, 0, stats.f.pdf(xr, nu, nu), color=BLUE_F, zorder=2)
    ax.axvline(1.0, color=INK, linewidth=1.2, linestyle=(0, (2, 3)), zorder=3)

    ax.plot([mode], [stats.f.pdf(mode, nu, nu)], "o", color=ORANGE,
            markersize=6.5, zorder=6)
    ax.annotate(f"최빈값 {mode:.2f}", xy=(mode, stats.f.pdf(mode, nu, nu)),
                xytext=(0.06, ymax * 0.74), fontsize=10, color=ORANGE,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.plot([mean], [stats.f.pdf(mean, nu, nu)], "o", color=RED,
            markersize=6.5, zorder=6)
    ax.annotate(f"평균 {mean:.2f}", xy=(mean, stats.f.pdf(mean, nu, nu)),
                xytext=(1.95, ymax * 0.62), fontsize=10, color=RED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(2.42, ymax * 0.40,
            f"$n_1 = n_2 = 10$ 에서\n"
            f"최빈값 {mode:.2f} < 중앙값 1.00 < 평균 {mean:.2f}",
            fontsize=10, color=INK, ha="left", va="center", linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F5F7F8",
                      edgecolor=MUTED, linewidth=0.8))
    ax.text(1.30, ymax * 0.085, f"확률 {stats.f.sf(1, 9, 9):.2f}",
            fontsize=9.5, color=BLUE, ha="center", va="center")

    bare_axis(ax)
    ax.set_xlim(0, 4.2)
    ax.set_ylim(0, ymax * 1.10)
    ax.set_xticks([0, 0.5, 1, 2, 3, 4])
    ax.set_xlabel(r"$S_1^2/S_2^2$   (참값은 $\sigma_1^2/\sigma_2^2 = 1$)",
                  fontsize=11, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.99))
    ax.set_title("(a) 두 분산이 같아도 분산비는 1 을 중심으로 비대칭하게 흩어진다",
                 fontsize=11.5, color=INK, pad=14, loc="left")

    # ---------- (b) 95% 구간은 곱셈 척도에서 대칭 ----------
    ax = axes[1]
    ns = [5, 10, 20, 30, 50, 100]
    ys = np.arange(len(ns))[::-1]
    ax.plot([1, 1], [-0.55, len(ns) - 0.55], color=INK, linewidth=1.2,
            linestyle=(0, (2, 3)), zorder=2)

    for yy, n in zip(ys, ns):
        nu = n - 1
        U = stats.f.ppf(0.975, nu, nu)
        L = 1 / U
        ax.plot([L, U], [yy, yy], color=BLUE, linewidth=7.5,
                solid_capstyle="butt", zorder=4)
        ax.plot([1, 1], [yy - 0.11, yy + 0.11], color="white", linewidth=1.6,
                zorder=5)
        ax.text(L / 1.12, yy, f"{L:.2f}", fontsize=9.5, color=BLUE,
                ha="right", va="center")
        ax.text(U * 1.12, yy, f"{U:.2f}", fontsize=9.5, color=BLUE,
                ha="left", va="center")
        if n == 10:
            ax.text(1 / np.sqrt(U), yy + 0.30, f"÷ {U:.2f}", fontsize=10,
                    color=PURPLE, ha="center", va="bottom")
            ax.text(np.sqrt(U), yy + 0.30, f"× {U:.2f}", fontsize=10,
                    color=PURPLE, ha="center", va="bottom")
            ax.text(1 / np.sqrt(U), yy - 0.30,
                    f"1 - {L:.2f} = {1 - L:.2f}", fontsize=10,
                    color=RED, ha="center", va="top")
            ax.text(np.sqrt(U), yy - 0.30,
                    f"{U:.2f} - 1 = {U - 1:.2f}", fontsize=10,
                    color=RED, ha="center", va="top")

    ax.set_xscale("log")
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_yticks(ys)
    ax.set_yticklabels([f"n = {n}" for n in ns])
    clean_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10.5)
    ax.set_xlim(0.033, 24)
    ax.set_ylim(-1.55, len(ns) - 0.35)
    ax.set_xlabel(r"$S_1^2/S_2^2$   (로그 눈금)", fontsize=11, color=INK)
    ax.set_title("(b) 등분산일 때 분산비의 95% 범위 — 로그 눈금에서 비로소 대칭이다",
                 fontsize=11.5, color=INK, pad=14, loc="left")
    ax.text(0.9, -1.00,
            "막대는 1 을 기준으로 덧셈이 아니라 곱셈으로 반씩 나뉜다.\n"
            "분산비의 신뢰구간도 같은 이유로 점추정값의 좌우가 같지 않다.",
            fontsize=9.5, color=INK, ha="center", va="center", linespacing=1.6)

    fig.tight_layout(w_pad=2.4)
    save(fig, TWO + "var_ratio_skew.png")

    # 본문에 옮겨 적을 수치
    print("  [a] n=10: 최빈값 %.4f 중앙값 %.4f 평균 %.4f  P(R>1)=%.4f"
          % (mode, stats.f.ppf(0.5, 9, 9), mean, stats.f.sf(1, 9, 9)))
    for n in ns:
        nu = n - 1
        U = stats.f.ppf(0.975, nu, nu)
        print("  [b] n=%3d  (%.4f, %.4f)   폭(덧셈) 좌 %.4f 우 %.4f"
              % (n, 1 / U, U, 1 - 1 / U, U - 1))

    # 모의실험으로 확인
    rng = np.random.default_rng(20250927)
    M, n = 200_000, 10
    a = rng.standard_normal((M, n)).var(1, ddof=1)
    b = rng.standard_normal((M, n)).var(1, ddof=1)
    r = a / b
    print("  [MC] n=10, M=%d: P(R>1)=%.4f  중앙값=%.4f  평균=%.4f"
          % (M, np.mean(r > 1), np.median(r), r.mean()))
    print("  [MC] P(R>2)=%.4f  P(R<1/2)=%.4f  P(R>4)=%.4f  P(R<1/4)=%.4f"
          % (np.mean(r > 2), np.mean(r < 0.5),
             np.mean(r > 4), np.mean(r < 0.25)))
    print("  [MC] 2.5%%, 97.5%% 백분위 = %.4f, %.4f"
          % (np.percentile(r, 2.5), np.percentile(r, 97.5)))


if __name__ == "__main__":
    chi2_rejection_asymmetry()
    var_ratio_skew()
