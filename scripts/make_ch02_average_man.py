r"""2.2절 「그림으로 보는 통계학」 — 평균인은 몇 명이나 되는가.

케틀레는 1835 년에 「평균인」을 내놓았다. 가슴둘레도 키도 정규곡선을
그리니, 그 가운데에 자연이 겨눈 표준적인 사람이 있고 나머지는 그것을
빗맞힌 오차라는 생각이다.

1952 년에 대니얼스가 그 생각을 공군 조종사 4,063 명으로 확인해 보았다.
열 가지 치수마다 「가운데 30%」를 평균 구간으로 잡고, 열 가지 모두에서
평균인 사람을 세었다. 답은 0 명이었다.

이 그림은 그것이 우연이 아니라 산수임을 보인다. 치수마다 30% 씩이면

  독립일 때   0.3^p  — 3 차원에서 2.7%, 10 차원에서 0.0006%
  상관이 있어도 크게 다르지 않다 (rho = 0.5 에서 3 차원 3.7%,
              10 차원 0.005% — 4,063 명 가운데 0.2 명)

만드는 파일:

  ch02/center/img/average_man.png

실행:  python3 scripts/make_ch02_average_man.py   (저장소 최상위에서)
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

OUT = "docs/ch02/center/img/"

INK = "#37474F"
MUTED = "#90A4AE"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, P, REPS, SEED = 4063, 10, 400, 1952
RHOS = (0.0, 0.3, 0.5, 0.7)
LO, HI = stats.norm.ppf([0.35, 0.65])        # 가운데 30%


def share(rho, rng):
    """치수 p 개 모두에서 가운데 30% 안에 드는 사람의 비율."""
    acc = np.zeros(P + 1)
    for _ in range(REPS):
        z = rng.normal(size=(N, 1))
        e = rng.normal(size=(N, P))
        x = np.sqrt(rho) * z + np.sqrt(1 - rho) * e
        inside = (x > LO) & (x < HI)
        for p in range(1, P + 1):
            acc[p] += inside[:, :p].all(1).mean()
    return acc / REPS


def main():
    rng = np.random.default_rng(SEED)
    curves = {}
    for rho in RHOS:
        c = share(rho, rng)
        curves[rho] = c
        print(f"    rho={rho:.1f}   1 차원 {c[1]:.4f}   3 차원 {c[3]:.4f}   "
              f"10 차원 {c[10]:.6f}   ({N} 명 중 {c[10]*N:.2f} 명)")
    print(f"    독립일 때 이론값  0.3^3 = {0.3**3:.4f}   "
          f"0.3^10 = {0.3**10:.3e}  ({N} 명 중 {0.3**10*N:.3f} 명)")

    ps = np.arange(1, P + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    ax = axes[0]
    for rho, col in zip(RHOS, (RED, AMBER, TEAL, BLUE)):
        ax.plot(ps, curves[rho][1:] * 100, marker="o", ms=6, lw=2.4, color=col,
                label=f"치수끼리의 상관 $\\rho = {rho}$")
    ax.set_yscale("log")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xticks(ps)
    ax.set_xlabel("함께 따지는 치수의 개수", fontsize=11.5)
    ax.set_ylabel("모든 치수에서 「평균」인 사람의 비율 (%)", fontsize=11.5)
    ax.set_title("치수를 하나 더할 때마다 자릿수가 떨어진다",
                 fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate(f"치수 하나면 {curves[0.5][1]*100:.0f}%",
                xy=(1, curves[0.5][1] * 100), xytext=(1.6, 55),
                fontsize=10.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))

    ax = axes[1]
    for rho, col in zip(RHOS, (RED, AMBER, TEAL, BLUE)):
        ax.plot(ps, curves[rho][1:] * N, marker="o", ms=6, lw=2.4, color=col,
                label=f"$\\rho = {rho}$")
    ax.axhline(1, color=INK, lw=2.2, ls="--")
    ax.text(9.8, 1.35, "한 명", ha="right", fontsize=10.8, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([1e-2, 1e-1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xticks(ps)
    ax.set_xlabel("함께 따지는 치수의 개수", fontsize=11.5)
    ax.set_ylabel(f"{N:,} 명 가운데 기대되는 사람 수", fontsize=11.5)
    ax.set_title("대니얼스가 센 것은 조종사 $4{,}063$ 명이었다",
                 fontsize=12.8, color=INK, pad=8)
    ax.annotate(f"열 치수 모두에서 평균이려면\n{N:,} 명 중 "
                f"{curves[0.5][10]*N:.1f} 명 — 그가 센 것은 $0$ 명",
                xy=(10, curves[0.5][10] * N), xytext=(6.4, 0.035),
                fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("평균인은 아무도 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "치수마다 「가운데 $30\\%$ 안」을 평균이라 불렀다. "
             "치수들이 서로 상관되어 있어도 사정은 크게 달라지지 않는다 — "
             "$\\rho = 0.7$ 로 세게 묶어도 열 치수에서 한두 명이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "average_man.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
