r"""4.1절 「그림으로 보는 통계학」 — 바나흐의 성냥갑.

주머니 양쪽에 성냥갑을 하나씩(각 n 개비) 넣고, 피울 때마다 한쪽을 반반의
확률로 골라 한 개비씩 꺼낸다. 어느 한쪽을 열었는데 비어 있었을 때 **다른
쪽에 몇 개비가 남아 있는가.**

P(K = k) = C(2n-k, n) * (1/2)^(2n-k)

음이항분포(정의 1)가 「r 번째 성공까지 걸린 횟수」를 세는 바로 그 구조다.
n = 40 에서 기댓값이 6.2 개이고, 비어 있을 확률이 0.089 다.

만드는 파일:

  ch04/discrete_distributions/img/banach_matchbox.png
"""

from math import comb

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch04/discrete_distributions/img/"
INK, TEAL, TEAL_F, RED = "#37474F", "#00838F", "#D5EBED", "#D32F2F"

N = 40


def main():
    k = np.arange(N + 1)
    pmf = np.array([comb(2 * N - kk, N) * 0.5 ** (2 * N - kk) for kk in k])
    mean = float((k * pmf).sum())

    fig, ax = plt.subplots(figsize=(9.8, 5.0), constrained_layout=True)
    ax.bar(k, pmf, color=TEAL_F, ec=TEAL, lw=1.1)
    ax.bar([0], [pmf[0]], color=RED, ec=RED, lw=1.1)

    ax.annotate(f"양쪽이 동시에 떨어질 확률\n{pmf[0]:.3f}",
                xy=(0, pmf[0]), xytext=(5.5, pmf[0] * 0.93),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.axvline(mean, color=INK, lw=2.2, ls="--")
    ax.text(mean + 0.6, pmf.max() * 0.72,
            f"평균 {mean:.1f} 개비", fontsize=11.5, color=INK)

    ax.set_xlabel(f"다른 갑에 남아 있는 개비 수  (각 갑 {N} 개비로 시작)", fontsize=11.5)
    ax.set_ylabel("확률", fontsize=11.5)
    ax.set_title("한쪽이 비었을 때 다른 쪽에 몇 개비가 남아 있는가",
                 fontsize=13.5, color=INK, pad=10)
    ax.set_xlim(-0.8, 28)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.04,
             f"절반 가까이({pmf[:6].sum():.0%})가 5 개비 이하로 남는다. "
             "두 갑을 반반으로 꺼냈는데도 한쪽이 먼저 바닥나는 일이 그만큼 흔하다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "banach_matchbox.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    합 {pmf.sum():.6f} · 평균 {mean:.4f} · P(0)={pmf[0]:.4f} · P(<=5)={pmf[:6].sum():.4f}")


if __name__ == "__main__":
    main()
