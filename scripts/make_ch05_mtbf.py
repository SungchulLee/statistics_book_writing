r"""5.5절 「그림으로 보는 통계학」 — 다섯 번 고장 났을 때 할 수 있는 말.

이 쪽의 정리 1 은 지수모집단의 표본평균이 **정확히** 감마라고 말한다.
중심극한정리가 필요 없다는 뜻이다. 그것이 왜 중요한지는 고장을 세는
사람들의 일에서 드러난다.

신뢰성공학에서 재는 것은 평균고장간격(MTBF)이다. 그런데 시험할 수 있는
장비는 몇 대뿐이고 고장은 더 드물게 난다. 고장을 다섯 번 보고 「이 장비의
MTBF 는 적어도 얼마」라고 말해야 한다.

고장시간이 지수이면 r 번째 고장까지의 총 시간 T 에 대해

  2T / theta ~ chi2(2r)

이고, 여기서 90% 하한이 정확하게 나온다.

  theta_하한 = 2r xbar / chi2_{0.90}(2r)

정규근사를 쓰면 xbar(1 - 1.282/sqrt(r)) 이다. 둘을 견주면 이렇다.

   r      정확 하한      정규근사      정규근사의 실제 신뢰수준
   2     0.5142 xbar   0.0938 xbar        1.0000
   5     0.6255 xbar   0.4269 xbar        0.9907
  20     0.7721 xbar   0.7134 xbar        0.9528
 100     0.8849 xbar   0.8718 xbar        0.9246

정규근사는 틀린 쪽이 아니라 **버리는 쪽**으로 빗나간다. r = 5 에서
말할 수 있는 MTBF 를 32% 깎아먹고, r = 2 면 사실상 아무 말도 못 한다.
까닭은 감마분포가 오른쪽으로 치우쳐 있는데 정규근사는 대칭이기
때문이다. 군·항공 신뢰성 규격이 카이제곱 계수표를 싣는 이유가 이것이다.

만드는 파일:

  ch05/applications/img/mtbf.png

실행:  python3 scripts/make_ch05_mtbf.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/applications/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

CL = 0.90
Z = stats.norm.ppf(CL)
RS = np.array([2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400])
SHOW = (2, 5, 20)


def exact_lo(r):
    return 2 * r / stats.chi2.ppf(CL, 2 * r)


def norm_lo(r):
    return 1 - Z / np.sqrt(r)


def main():
    print(f"    지수 고장시간, 고장 {CL:.0%} 하한 (xbar 의 배수)")
    print(f"{'r':>5}{'정확':>10}{'정규근사':>12}{'정규의 실제 신뢰수준':>22}{'손해':>9}")
    for r in (1, 2, 3, 5, 10, 20, 50, 100, 300):
        e, nlo = exact_lo(r), norm_lo(r)
        cov = stats.chi2.cdf(2 * r / nlo, 2 * r) if nlo > 0 else 1.0
        print(f"{r:>5}{e:>10.4f}{nlo:>12.4f}{cov:>22.4f}{100*(1-nlo/e):>8.1f}%")

    print(f"\n    양쪽 95% 구간 (xbar 의 배수)")
    for r in (5, 20):
        el = 2 * r / stats.chi2.ppf(0.975, 2 * r)
        eh = 2 * r / stats.chi2.ppf(0.025, 2 * r)
        nl = 1 - 1.96 / np.sqrt(r)
        nh = 1 + 1.96 / np.sqrt(r)
        print(f"      r = {r:>3}   정확 [{el:.3f}, {eh:.3f}]   "
              f"정규 [{nl:.3f}, {nh:.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 정확한 분포는 치우쳐 있다 ──
    ax = axes[0]
    xs = np.linspace(0, 3.2, 700)
    for r, col in zip(SHOW, (RED, AMBER, TEAL)):
        ax.plot(xs, stats.gamma.pdf(xs, r, scale=1 / r), lw=2.8, color=col,
                label=f"정확한 분포 — 고장 {r} 번 (감마)")
        ax.plot(xs, stats.norm.pdf(xs, 1, 1 / np.sqrt(r)), lw=1.8, color=col,
                ls=":")
    ax.axvline(1, color=INK, lw=1.6, ls="--")
    ax.text(1.03, 2.45, "참값 $\\theta$", fontsize=11, color=INK)
    ax.annotate("점선은 같은 색의 정규근사다",
                xy=(0.30, 0.35), xytext=(1.75, 1.75), fontsize=10.8, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("고장이 적으면 꼬리가 오른쪽으로 길다 —\n"
                "대칭인 정규근사가 담지 못하는 모양이다",
                xy=(2.1, 0.09), xytext=(2.05, 1.15), fontsize=10.6, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("$\\bar X / \\theta$ — 관측한 평균고장간격을 참값으로 나눈 것",
                  fontsize=11.3)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title("표본평균의 분포를 정확히 안다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 2.75)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 말할 수 있는 MTBF ──
    ax = axes[1]
    e = np.array([exact_lo(r) for r in RS])
    n = np.array([norm_lo(r) for r in RS])
    ax.fill_between(RS, n, e, color=RED, alpha=0.14)
    ax.plot(RS, e, lw=3.0, color=TEAL, marker="o", ms=6,
            label="정확한 하한 — 카이제곱")
    ax.plot(RS, n, lw=2.8, color=BLUE, marker="s", ms=6, ls="--",
            label="정규근사 하한")
    ax.axhline(0, color=GREY, lw=1.4)
    for r in (5, 20):
        i = int(np.where(RS == r)[0][0])
        ax.annotate(f"고장 {r} 번\n{100*(1-n[i]/e[i]):.0f}% 를 버린다",
                    xy=(r, (e[i] + n[i]) / 2),
                    xytext=(r * (1.9 if r == 5 else 2.6), 0.33 if r == 5 else 0.52),
                    fontsize=10.6, color=RED, ha="center",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100, 400])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "400"])
    ax.minorticks_off()
    ax.set_xlabel("관측한 고장 횟수 $r$", fontsize=11.5)
    ax.set_ylabel("MTBF 의 $90\\%$ 하한 ($\\bar x$ 의 배수)", fontsize=11.5)
    ax.set_title("같은 자료로 말할 수 있는 양이 달라진다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(-0.05, 1.02)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("정확한 분포를 알면 같은 자료로 더 많이 말할 수 있다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             "두 하한 모두 참 MTBF 를 $90\\%$ 이상의 확률로 담는다. "
             "정규근사는 틀린 쪽이 아니라 버리는 쪽으로 빗나간다 — "
             "고장 $2$ 번이면 하한이 $0.09\\bar x$ 로 사실상 아무 말도 못 한다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "mtbf.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
