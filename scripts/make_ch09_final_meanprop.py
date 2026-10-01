r"""9장 일표본 평균 검정과 일표본 비율 검정 두 쪽의 그림을 생성한다.

  ch09/one_sample_tests/img/power_curve_mean.png        검정력 곡선 — 기각 못 함 != 차이 없음
  ch09/one_sample_tests/img/discrete_level_sawtooth.png 이산성이 만드는 톱니 모양 실제 수준

앞 그림은 비중심 t 분포로 검정력을 정확히 계산하고, 뒤 그림은 이항분포의
확률질량을 그대로 더해 실제 제1종 오류율을 정확히 계산한다. 모의실험이 아니다.

실행:  python3 scripts/make_ch09_final_meanprop.py  (저장소 최상위)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
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


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 검정력 곡선 — 효과크기와 표본크기
#    양측 t 검정의 검정력은 비중심 t 분포로 정확히 계산된다.
# ==================================================================
def t_power(n, d, alpha=0.05, two_sided=True):
    """일표본 t 검정의 검정력. d = (mu - mu0)/sigma."""
    df = n - 1
    nc = d * np.sqrt(n)
    if two_sided:
        c = stats.t.ppf(1 - alpha / 2, df)
        return stats.nct.sf(c, df, nc) + stats.nct.cdf(-c, df, nc)
    c = stats.t.ppf(1 - alpha, df)
    return stats.nct.sf(c, df, nc)


def power_curve_mean():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # --- 왼쪽: 효과크기에 따른 검정력, 표본크기별 ---
    d = np.linspace(0.0, 1.3, 400)
    specs = [(10, BLUE), (25, ORANGE), (50, GREEN), (100, PURPLE)]
    for n, col in specs:
        axL.plot(d, [t_power(n, x) for x in d], color=col, lw=2.1,
                 label=f"$n={n}$")

    axL.axhline(0.80, color=MUTED, lw=1.1, ls="--")
    axL.text(1.44, 0.80, "0.80", ha="left", va="center",
             fontsize=9.5, color=INK)
    axL.axhline(0.05, color=RED, lw=1.1, ls=":")
    axL.text(1.44, 0.05, r"$\alpha=0.05$", ha="left", va="center",
             fontsize=9.5, color=RED)

    axL.set_xlim(0, 1.42)
    axL.set_ylim(0, 1.04)
    axL.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    axL.set_xlabel("효과크기  $d=(\\mu-\\mu_0)/\\sigma$", fontsize=11, color=INK)
    axL.set_ylabel("검정력", fontsize=11, color=INK)
    axL.set_title("양측 $t$ 검정의 검정력", fontsize=12, color=INK, pad=9)
    axL.legend(fontsize=9.5, loc="lower right", frameon=False,
               bbox_to_anchor=(1.0, 0.11))
    clean_axis(axL)

    # 작은 효과에서는 n=100 조차 버겁다는 것을 짚는다.
    p_small = t_power(100, 0.2)
    axL.plot([0.2], [p_small], "o", color=PURPLE, ms=5.5, zorder=5)
    axL.annotate(f"$d=0.2$ 에서는\n$n=100$ 도 {p_small:.2f}",
                 xy=(0.2, p_small), xytext=(0.02, 0.95),
                 fontsize=9.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))

    # --- 오른쪽: 이 쪽의 보기를 표본크기로 밀어 본다 ---
    d0 = 0.2 / 1.1                     # 보기 2: xbar-mu0 = 0.2, s = 1.1
    ns = np.arange(5, 401)
    pw = np.array([t_power(n, d0, two_sided=False) for n in ns])
    axR.plot(ns, pw, color=BLUE, lw=2.2)
    axR.fill_between(ns, 0, pw, color=BLUE_F, alpha=0.55)

    p25 = t_power(25, d0, two_sided=False)
    n80 = int(ns[np.argmax(pw >= 0.80)])
    axR.axhline(0.80, color=MUTED, lw=1.1, ls="--")
    axR.text(396, 0.815, "0.80", ha="right", va="bottom",
             fontsize=9.5, color=INK)

    axR.plot([25], [p25], "o", color=RED, ms=6.5, zorder=6)
    axR.annotate(f"보기의 $n=25$\n검정력 {p25:.2f}",
                 xy=(25, p25), xytext=(70, 0.30),
                 fontsize=10, color=RED, ha="left",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

    axR.plot([n80], [0.80], "o", color=GREEN, ms=6.5, zorder=6)
    axR.annotate(f"0.80 에 닿으려면\n$n={n80}$",
                 xy=(n80, 0.80), xytext=(212, 0.54),
                 fontsize=10, color=GREEN, ha="left",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))

    axR.set_xlim(0, 400)
    axR.set_ylim(0, 1.04)
    axR.set_xlabel("표본크기 $n$", fontsize=11, color=INK)
    axR.set_ylabel("검정력", fontsize=11, color=INK)
    axR.set_title("보기 2의 효과크기 $d=0.18$ 을 고정하고 (단측)",
                  fontsize=12, color=INK, pad=9)
    clean_axis(axR)

    fig.tight_layout(w_pad=3.0)
    save(fig, ONE + "power_curve_mean.png")

    print(f"  [평균] d0 = {d0:.4f}")
    print(f"  [평균] n=25 단측 검정력 = {p25:.4f}")
    print(f"  [평균] 0.80 에 필요한 n = {n80}")
    for n in (10, 25, 50, 100):
        print(f"  [평균] 양측 n={n:3d}: d=0.2 -> {t_power(n, 0.2):.4f}, "
              f"d=0.5 -> {t_power(n, 0.5):.4f}, "
              f"d=0.8 -> {t_power(n, 0.8):.4f}")


# ==================================================================
# 2. 이산성 때문에 실제 유의수준이 톱니처럼 오르내린다
#    이항의 표본공간은 유한하므로 실제 수준을 정확히 계산할 수 있다.
# ==================================================================
def score_level(n, p0, alpha=0.05):
    """양측 점수 z 검정의 실제 제1종 오류율 (정확 계산)."""
    k = np.arange(n + 1)
    z = (k / n - p0) / np.sqrt(p0 * (1 - p0) / n)
    rej = np.abs(z) > stats.norm.ppf(1 - alpha / 2)
    return stats.binom.pmf(k, n, p0)[rej].sum()


def exact_level(n, p0, alpha=0.05):
    """양측 정확 이항검정의 실제 제1종 오류율 (정확 계산)."""
    k = np.arange(n + 1)
    pv = np.array([stats.binomtest(int(j), n, p0).pvalue for j in k])
    return stats.binom.pmf(k, n, p0)[pv <= alpha].sum()


def discrete_level_sawtooth():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # --- 왼쪽: p0 = 0.5 를 고정하고 n 을 늘려 간다 ---
    ns = np.arange(10, 121)
    sc = np.array([score_level(n, 0.5) for n in ns])
    ex = np.array([exact_level(n, 0.5) for n in ns])

    axL.plot(ns, sc, color=BLUE, lw=1.5, marker="o", ms=2.6,
             label="점수 $z$ 검정")
    axL.plot(ns, ex, color=GREEN, lw=1.5, marker="o", ms=2.6,
             label="정확 이항검정")
    axL.axhline(0.05, color=RED, lw=1.3, ls="--")
    axL.text(124, 0.05, "명목 0.05", ha="left", va="center",
             fontsize=9.5, color=RED)

    axL.set_xlim(8, 123)
    axL.set_ylim(0, 0.088)
    axL.set_xticks([20, 40, 60, 80, 100, 120])
    axL.set_xlabel("표본크기 $n$   ($p_0=0.5$)", fontsize=11, color=INK)
    axL.set_ylabel("실제 제1종 오류율", fontsize=11, color=INK)
    axL.set_title("$n$ 을 하나 늘릴 때마다 위아래로 뛴다",
                  fontsize=12, color=INK, pad=9)
    axL.legend(fontsize=9.5, loc="lower right", frameon=False,
               bbox_to_anchor=(1.02, -0.01))
    clean_axis(axL)

    # --- 오른쪽: n = 50 을 고정하고 p0 을 훑는다 ---
    p0s = np.arange(0.05, 0.951, 0.002)
    lv = np.array([score_level(50, p) for p in p0s])
    axR.plot(p0s, lv, color=ORANGE, lw=1.3)
    axR.axhline(0.05, color=RED, lw=1.3, ls="--")
    axR.text(0.985, 0.05, "명목 0.05", ha="left", va="center",
             fontsize=9.5, color=RED)

    axR.set_xlim(0.03, 0.97)
    axR.set_ylim(0, 0.088)
    axR.set_xlabel("가설의 비율 $p_0$   ($n=50$)", fontsize=11, color=INK)
    axR.set_ylabel("실제 제1종 오류율", fontsize=11, color=INK)
    axR.set_title("$p_0$ 을 조금 옮기기만 해도 흔들린다",
                  fontsize=12, color=INK, pad=9)
    clean_axis(axR)

    i_hi, i_lo = int(np.argmax(lv)), int(np.argmin(lv))
    axR.plot([p0s[i_hi]], [lv[i_hi]], "o", color=RED, ms=6, zorder=6)
    axR.annotate(f"가장 높은 곳  $p_0={p0s[i_hi]:.3f}$ 에서 {lv[i_hi]:.3f}",
                 xy=(p0s[i_hi], lv[i_hi]), xytext=(0.30, 0.082),
                 fontsize=9.5, color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    axR.plot([p0s[i_lo]], [lv[i_lo]], "o", color=PURPLE, ms=6, zorder=6)
    axR.annotate(f"가장 낮은 곳  $p_0={p0s[i_lo]:.3f}$ 에서 {lv[i_lo]:.3f}",
                 xy=(p0s[i_lo], lv[i_lo]), xytext=(0.22, 0.011),
                 fontsize=9.5, color=PURPLE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))

    fig.tight_layout(w_pad=3.0)
    save(fig, ONE + "discrete_level_sawtooth.png")

    print(f"  [비율] p0=0.5, n=20..25 점수: "
          + ", ".join(f"n={n}:{score_level(n, 0.5):.4f}" for n in range(20, 26)))
    print(f"  [비율] p0=0.5 점수 최대 {sc.max():.4f} (n={ns[sc.argmax()]}), "
          f"최소 {sc.min():.4f} (n={ns[sc.argmin()]})")
    print(f"  [비율] p0=0.5 정확 최대 {ex.max():.4f}, 최소 {ex.min():.4f}, "
          f"0.05 초과 개수 {(ex > 0.05).sum()}")
    print(f"  [비율] n=50 점수: 최대 {lv[i_hi]:.4f} at p0={p0s[i_hi]:.3f}, "
          f"최소 {lv[i_lo]:.4f} at p0={p0s[i_lo]:.3f}")
    for p in (0.28, 0.30, 0.32, 0.34):
        print(f"  [비율] n=50, p0={p:.2f}: {score_level(50, p):.4f}")


if __name__ == "__main__":
    power_curve_mean()
    discrete_level_sawtooth()
