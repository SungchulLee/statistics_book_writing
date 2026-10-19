r"""9.1절 「그림으로 보는 통계학」 — 같은 실험을 다시 하면.

2016 년 3 월, 미국통계학회가 통계 실무의 한 가지 주제에 대해 처음으로
공식 성명을 냈다. 주제가 p-값이었다. 여섯 조항 가운데 셋째가 이렇다.

  과학적 결론과 사업·정책 결정은 p-값이 어떤 문턱을 넘었는지만으로
  내려서는 안 된다.

왜 그런 성명이 필요했는가. 이 쪽 3 절은 p-값을 「귀무가설이 참일 때
이만큼 극단적인 결과가 나올 확률」로 정의한다. 옳은 정의다. 그런데
그 수가 **같은 실험을 다시 하면 얼마나 달라지는지**는 정의에 적혀
있지 않다.

참 효과를 고정해 놓고 실험을 되풀이하면

   검정력   p 중앙값     10% 분위     90% 분위
    0.30    0.1496     0.006575     0.7303
    0.50    0.0499     0.001183     0.4820
    0.80    0.0051     0.000044     0.1278
    0.90    0.0012     0.000006     0.0497

흔히 목표로 삼는 검정력 0.80 에서도 p 가 0.000044 에서 0.128 사이를
오간다. 거의 네 자릿수다. p-값은 추정량이고, 다른 추정량과 마찬가지로
표집분포를 갖는다 — 그런데 그 폭을 적어 주는 관행이 없다.

뒤집어 보면 이렇다. 처음 실험에서 p = 0.05 를 얻었고 그때 본 효과가
참값이라 치더라도, 다시 했을 때 p < 0.05 가 나올 확률이 **0.500** 이다.
그 가정 자체가 낙관적이니 실제로는 그보다 낮다.

만드는 파일:

  ch09/foundations/img/pvalue_dance.png

실행:  python3 scripts/make_ch09_pvalue_dance.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/foundations/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

ZC = 1.959964
REPS, SEED = 2_000_000, 2016
POWERS = ((0.30, RED), (0.50, AMBER), (0.80, TEAL))


def ncp(power):
    """두 자리 검정력을 내는 비중심모수."""
    return brentq(lambda L: norm.sf(ZC - L) + norm.cdf(-ZC - L) - power,
                  0, 12)


def main():
    rng = np.random.default_rng(SEED)
    print(f"{'검정력':>8}{'p 중앙값':>12}{'10% 분위':>13}{'90% 분위':>12}")
    draws = {}
    for pw, _ in POWERS:
        z = rng.normal(ncp(pw), 1, REPS)
        p = 2 * norm.sf(np.abs(z))
        draws[pw] = p
        q = np.quantile(p, [0.1, 0.5, 0.9])
        print(f"{pw:>8.2f}{q[1]:>12.4f}{q[0]:>13.6f}{q[2]:>12.4f}")

    print(f"\n  처음 본 효과가 참값이라 칠 때 재현에서 p < 0.05 일 확률")
    print(f"{'처음의 p':>12}{'재현 성공 확률':>16}")
    for p1 in (0.10, 0.05, 0.01, 0.005, 0.001, 0.0001):
        z1 = norm.isf(p1 / 2)
        print(f"{p1:>12.4f}{norm.cdf(z1 - ZC) + norm.cdf(-z1 - ZC):>16.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: p 는 한 수가 아니라 분포다 ──
    ax = axes[0]
    edges = np.linspace(-7, 0, 110)
    for pw, col in POWERS:
        lp = np.log10(np.maximum(draws[pw], 1e-300))
        ax.hist(lp, bins=edges, density=True, histtype="step", lw=3.0,
                color=col, label=f"검정력 ${pw:.2f}$")
        q = np.log10(np.quantile(draws[pw], [0.1, 0.9]))
        y = {0.30: 0.60, 0.50: 0.54, 0.80: 0.48}[pw]
        ax.plot(q, [y, y], lw=3.2, color=col, solid_capstyle="butt", zorder=5)
        ax.plot(q, [y, y], lw=0, marker="|", ms=14, mew=3, color=col, zorder=5)
    ax.axvline(np.log10(0.05), color=INK, lw=2.2, ls="--")
    ax.text(np.log10(0.05) + 0.12, 0.30, "$p = 0.05$", fontsize=10.8,
            color=INK)
    ax.annotate("가로 막대가 $10$~$90\\%$ 구간이다\n"
                "검정력 $0.80$ 에서도 네 자릿수를 오간다",
                xy=(np.log10(np.quantile(draws[0.80], 0.1)), 0.48),
                xytext=(-4.5, 0.22), fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xticks([-6, -4, -2, 0])
    ax.set_xticklabels(["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"])
    ax.set_xlabel("같은 실험을 되풀이했을 때의 $p$-값", fontsize=11.5)
    ax.set_ylabel("밀도 ($\\log_{10} p$ 에 대하여)", fontsize=11.5)
    ax.set_title("$p$ 는 한 수가 아니라 분포다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(-7, 0.3)
    ax.set_ylim(0, 0.68)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 처음의 p 와 재현의 확률 ──
    ax = axes[1]
    p1 = np.logspace(-5, np.log10(0.3), 400)
    z1 = norm.isf(p1 / 2)
    rep = norm.cdf(z1 - ZC) + norm.cdf(-z1 - ZC)
    ax.plot(p1, rep, lw=3.8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1.8, ls=":")
    for q, col in ((0.05, RED), (0.005, AMBER), (0.0001, TEAL)):
        v = norm.cdf(norm.isf(q / 2) - ZC) + norm.cdf(-norm.isf(q / 2) - ZC)
        ax.plot([q], [v], marker="o", ms=13, color=col, zorder=5,
                mec="white", mew=1.7)
    ax.annotate("처음에 $p = 0.05$ 를 얻었다면\n재현에서 유의할 확률이 $0.50$ 이다",
                xy=(0.05, 0.5), xytext=(0.016, 0.26), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("$p = 0.005$ 여야 $0.80$ 이 된다",
                xy=(0.005, 0.802), xytext=(2.5e-4, 0.55), fontsize=10.8,
                color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([0.1, 0.05, 0.01, 0.001, 0.0001, 1e-5])
    ax.set_xticklabels(["$0.1$", "$0.05$", "$0.01$", "$10^{-3}$",
                        "$10^{-4}$", "$10^{-5}$"])
    ax.minorticks_off()
    ax.set_xlabel("처음 실험에서 얻은 $p$-값", fontsize=11.5)
    ax.set_ylabel("다시 했을 때 $p < 0.05$ 일 확률", fontsize=11.5)
    ax.set_title("낙관적으로 잡아도 이만큼이다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("같은 실험을 다시 하면", fontsize=14.5)
    fig.text(0.5, -0.06,
             "오른쪽은 처음 실험에서 **본** 효과가 참값이라고 치고 잰 값이다. "
             "실제로는 유의하게 나온 연구의 효과가 부풀려져 있으므로 "
             "이 곡선은 윗한계에 가깝다.".replace("**", ""),
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "pvalue_dance.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
