r"""9.4 대마초 가격 사례 연구 쪽의 마무리 그림 한 장을 생성한다.

  ch09/two_sample_tests/img/significance_vs_importance.png
      왼쪽  — 두 해의 품질 등급 비율 비교(차이가 최대 0.48%p)
      오른쪽 — 같은 비율 차이를 유지하고 표본만 줄였을 때 카이제곱과 판정

실행:  python3 scripts/make_ch09_final_weedprices.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.

자료 출처: 원자료를 새로 읽지 않는다. 이 스크립트의 모든 수치는
       docs/ch09/two_sample_tests/ttest_weed_prices.md 가 이미 본문과 출력에
       보고한 값을 그대로 옮긴 것이다.
         - 예제 4의 판매 도수와 비율, chi2 = 240.93
         - 연습문제 7의 N = 1,424,452, 코헨 w = 0.01301, 크라메르 V = 0.00920,
           2014 비율 [0.32046 0.48718 0.19236], 2015 비율 [0.32427 0.48821 0.18752],
           차이(%p) [+0.381 +0.103 -0.484],
           표본 축소 표(1,424,452 / 142,445 / 35,400 / 14,244)와
           유의해지는 최소 표본 약 35,400
       오른쪽 곡선은 그 쪽이 관찰한 관계 chi2 = N * w^2, 즉
       chi2(N) = 240.927 * N / 1,424,452 을 그린 것이다.
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

OUT = "docs/ch09/two_sample_tests/img/"

# === 그 쪽이 보고한 수치 ===
GRADES = ["고품질", "중품질", "저품질"]
P2014 = np.array([32.046, 48.718, 19.236])   # %
P2015 = np.array([32.427, 48.821, 18.752])   # %
DIFF = np.array([+0.381, +0.103, -0.484])    # %p

N_FULL = 1_424_452
CHI2_FULL = 240.927
CRAMER_V = 0.00920
N_CRIT = 35_400          # 그 쪽이 찾은 "유의해지는 최소 표본"
CHI2_CRIT_LINE = stats.chi2.ppf(0.95, 2)     # 5.991


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def chi2_of(n):
    """비율 차이는 그대로 두고 표본만 바꿀 때의 카이제곱. chi2 = N * w^2."""
    return CHI2_FULL * np.asarray(n, dtype=float) / N_FULL


# ==================================================================
# 왼쪽 — 등급 비율은 거의 그대로다
# ==================================================================
def panel_proportions(ax):
    x = np.arange(3)
    w = 0.34
    ax.bar(x - w / 2, P2014, w, color=BLUE_F, edgecolor=BLUE, linewidth=1.4,
           label="2014년 1월", zorder=3)
    ax.bar(x + w / 2, P2015, w, color=ORANGE_F, edgecolor=ORANGE, linewidth=1.4,
           label="2015년 1월", zorder=3)

    for xi, a, b, d in zip(x, P2014, P2015, DIFF):
        ax.text(xi - w / 2, a + 0.9, f"{a:.1f}%", ha="center", va="bottom",
                fontsize=9.5, color=BLUE)
        ax.text(xi + w / 2, b + 0.9, f"{b:.1f}%", ha="center", va="bottom",
                fontsize=9.5, color=ORANGE)
        ax.text(xi, max(a, b) + 5.0, f"{d:+.2f}%p", ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(GRADES, fontsize=11)
    ax.set_ylim(0, 76)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_ylabel("판매 비율 (%)", fontsize=10.5, color=INK)
    ax.set_title("등급 비율은 사실상 그대로다", fontsize=12.5, color=INK, pad=10)
    ax.legend(loc="upper right", fontsize=9.5, frameon=False)
    ax.grid(axis="y", color=MUTED, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)

    ax.text(0.02, 0.985,
            "$\\chi^2(2) = 240.93$,  $p < 10^{-50}$\n"
            "크라메르 $V = 0.0092$  (관례적 기준 0.1 = 작음)",
            transform=ax.transAxes, fontsize=10, color=INK, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=MUTED, alpha=0.95))


# ==================================================================
# 오른쪽 — 같은 차이인데 판정은 표본크기가 정한다
# ==================================================================
def panel_sample_size(ax):
    n = np.logspace(np.log10(3e3), np.log10(3e6), 400)
    c = chi2_of(n)

    ax.axhspan(CHI2_CRIT_LINE, 1e4, color=RED, alpha=0.05, zorder=0)
    ax.axhline(CHI2_CRIT_LINE, color=RED, lw=1.4, ls="--", zorder=2)
    ax.plot(n, c, color=PURPLE, lw=2.2, zorder=3)

    pts = [(14_244, "1.4만", 2.4, GREEN), (N_CRIT, "3.5만", 6.0, ORANGE),
           (142_445, "14만", 24.1, RED), (N_FULL, "142만", 240.9, RED)]
    for ni, lab, shown, col in pts:
        ci = float(chi2_of(ni))
        ax.plot([ni], [ci], "o", ms=7.5, zorder=5,
                color=col, mec="white", mew=1.1)
        if ni == 14_244:                      # 아래쪽으로 빼서 겹침을 피한다
            ax.annotate(f"{lab}\n$\\chi^2 = {shown:.1f}$",
                        xy=(ni, ci), xytext=(10, -2), textcoords="offset points",
                        ha="left", va="top", fontsize=9.3, color=col)
        else:
            ax.annotate(f"{lab}\n$\\chi^2 = {shown:.1f}$",
                        xy=(ni, ci), xytext=(-7, 8), textcoords="offset points",
                        ha="right", va="bottom", fontsize=9.3, color=col)

    ax.axvline(N_CRIT, color=INK, lw=1.0, ls=":", zorder=1)
    ax.annotate("유의해지는 최소 표본 ≈ 35,400",
                xy=(N_CRIT * 1.12, CHI2_CRIT_LINE * 0.92), xycoords="data",
                xytext=(6.0e5, 0.62), textcoords="data",
                ha="center", va="bottom", fontsize=9.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0,
                                shrinkA=2, shrinkB=2))

    ax.text(3.4e3, CHI2_CRIT_LINE * 1.5, "기각 ($p < 0.05$)",
            fontsize=9.8, color=RED, va="bottom")
    ax.text(3.4e3, CHI2_CRIT_LINE / 1.6, "기각 못 함", fontsize=9.8,
            color=GREEN, va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e3, 3e6)
    ax.set_ylim(0.4, 900)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"], fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("표본크기 $N$  (비율 차이는 고정, $\\chi^2 = N w^2$, 코헨 $w = 0.013$)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("카이제곱 통계량", fontsize=10.5, color=INK)
    ax.set_title("판정을 정하는 것은 표본크기다", fontsize=12.5, color=INK, pad=10)
    ax.grid(True, which="major", color=MUTED, alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def make_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    panel_proportions(axes[0])
    panel_sample_size(axes[1])
    fig.suptitle("유의성과 중요성은 다른 물음이다 — 등급 분포의 적합도 검정",
                 fontsize=13.5, color=INK, y=1.02)
    fig.text(0.5, -0.045,
             "이 쪽이 보고한 수치를 그대로 옮겨 그렸다. "
             "원자료를 다시 읽지 않았다.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout()
    save(fig, OUT + "significance_vs_importance.png")


if __name__ == "__main__":
    make_figure()
    print("saved:", OUT + "significance_vs_importance.png")
