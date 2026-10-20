r"""6.2절 「그림으로 보는 통계학」 — 정칙조건은 잔글씨가 아니다.

크라메르-라오 하한에는 조건이 붙어 있다. 그 가운데 첫째가 「지지집합이
theta 에 의존하지 않는다」이고, 이 쪽도 정리 1 의 조건 1 로 적어 두었다.

조건이 깨지면 어떻게 되는가. 조금 나빠지는 정도가 아니라 **하한이
통째로 무너진다.**

세 모형을 견준다. 효율은 CRLB / Var 이고, 1 을 넘을 수 없어야 한다.

      n    베르누이   정규 S^2   균등(0,theta)
      5     1.000     0.800         7.0
     20     1.000     0.950        22.0
    100     1.000     0.990       102.0

베르누이의 p-hat 은 하한을 정확히 달성한다. 정규분포의 S^2 은 달성하지
못하지만 하한 아래로 내려가지도 않는다 — 효율이 (n-1)/n 이다.

균등분포는 다르다. 소박하게 계산한 하한보다 **n+2 배 아래**에 있다.
n = 100 이면 「넘을 수 없다」는 선보다 102 배 아래다.

까닭은 지지집합이 theta 에 달려 있어서다. 관측값 하나가 theta 보다
크면 그 theta 는 **즉시 배제된다.** 보통의 모형에서는 자료가 모수를
조금씩만 좁히는데, 여기서는 단칼에 자른다. 그래서 수렴 속도가
sqrt(n) 이 아니라 n 이다.

부등식 자체도 네 사람이 따로 얻었다 — 프레셰(1943, 프랑스),
다르무아(1945, 프랑스), 라오(1945, 인도), 크라메르(1946, 스웨덴).
전쟁으로 나라 사이의 연락이 끊겨 있던 네 해다. 라오는 스물다섯이었고,
수업에서 학생이 「추정량이 좋아지는 데 한계가 있느냐」고 물은 것이
계기였다고 한다.

만드는 파일:

  ch07/theory/img/crlb_regularity.png

실행:  python3 scripts/make_ch06_crlb_regularity.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch07/theory/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

NS = np.array([5, 8, 12, 20, 35, 60, 100, 200])
R, SEED = 200_000, 1945
SIM_NS = (5, 10, 20, 50, 100, 200)


def main():
    eff_b = np.ones_like(NS, float)
    eff_n = (NS - 1) / NS
    eff_u = NS + 2.0
    print(f"{'n':>5}{'베르누이':>12}{'정규 S^2':>12}{'균등':>12}")
    for n, a, b, c in zip(NS, eff_b, eff_n, eff_u):
        print(f"{n:>5}{a:>12.4f}{b:>12.4f}{c:>12.1f}")

    rng = np.random.default_rng(SEED)
    sim_n, sim_v = [], []
    print(f"\n  균등(0,1) 모의 — 불편추정량 ((n+1)/n)·X_(n)")
    for n in SIM_NS:
        X = rng.uniform(0, 1, (R, n)).max(1) * (n + 1) / n
        sim_n.append(n); sim_v.append(float(X.var()))
        print(f"    n={n:>4}  평균 {X.mean():.5f}   분산 {X.var():.4e}   "
              f"식 {1/(n*(n+2)):.4e}   소박한 CRLB {1/n:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 세 모형의 효율 ──
    ax = axes[0]
    ax.axhline(1.0, color=INK, lw=2.4, ls="--")
    ax.text(5.6, 1.18, "하한 — 넘을 수 없어야 한다", fontsize=11, color=INK)
    ax.plot(NS, eff_u, lw=3.2, color=RED, marker="o", ms=8,
            label="균등$(0,\\theta)$ — 지지집합이 $\\theta$ 에 달려 있다")
    ax.plot(NS, eff_b, lw=3.0, color=TEAL, marker="D", ms=7,
            label="베르누이 $\\hat p$ — 정확히 달성")
    ax.plot(NS, eff_n, lw=3.0, color=AMBER, marker="s", ms=7,
            label="정규 $S^2$ — 달성하지 못한다 $(n-1)/n$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_yticks([0.8, 1, 3, 10, 30, 100, 200])
    ax.set_yticklabels(["0.8", "1", "3", "10", "30", "100", "200"])
    ax.minorticks_off()
    ax.annotate(f"$n = 100$ 에서 하한보다\n${eff_u[NS.tolist().index(100)]:.0f}$ 배 아래다",
                xy=(100, eff_u[NS.tolist().index(100)]), xytext=(17, 48),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("효율 $=$ CRLB $/$ 분산", fontsize=11.5)
    ax.set_title("하한을 $102$ 배 넘어서는 추정량", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.7, 260)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="center left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 수렴 속도가 다르다 ──
    ax = axes[1]
    nn = np.logspace(np.log10(4), np.log10(260), 200)
    ax.plot(nn, 1 / nn, lw=3.0, color=INK, ls="--",
            label="소박하게 계산한 CRLB $= \\theta^2/n$")
    ax.plot(nn, 1 / (nn * (nn + 2)), lw=3.2, color=RED,
            label="실제 분산 $= \\theta^2/[n(n+2)]$")
    ax.plot(sim_n, sim_v, lw=0, marker="o", ms=11, mfc="white", mec=RED,
            mew=2.2, zorder=5, label=f"모의실험 {R//10000} 만 번")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                        "$10^{-1}$"])
    ax.minorticks_off()
    ax.annotate("기울기가 다르다 —\n하나는 $1/n$, 하나는 $1/n^2$\n"
                "그래서 격차가 계속 벌어진다",
                xy=(45, 1 / (45 * 47)), xytext=(12, 2.4e-5), fontsize=10.8,
                color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("분산 ($\\theta = 1$)", fontsize=11.5)
    ax.set_title("$\\sqrt{n}$ 이 아니라 $n$ 으로 줄어든다", fontsize=12.8,
                 color=INK, pad=8)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("정칙조건은 잔글씨가 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "균등$(0,\\theta)$ 에서 쓴 추정량은 $\\dfrac{n+1}{n}X_{(n)}$ 이고 "
             "불편이다. 「소박한 CRLB」는 정칙조건을 확인하지 않고 공식만 "
             "넣어 계산한 값이다 — 그래서 하한이 아니다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "crlb_regularity.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
