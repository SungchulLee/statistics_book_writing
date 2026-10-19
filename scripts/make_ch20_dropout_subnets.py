r"""20.2절 「AI로 보는 통계학」 — 망 하나를 학습시키면서 2^n 개를 학습시킨다.

드롭아웃은 학습할 때마다 은닉 단위를 저마다 독립으로 끈다. 그러면 한 번의
학습이 「단위 n 개 가운데 어느 것을 남길지」를 고르는 일이 되고, 고를 수
있는 꼴이 정확히 2^n 가지다.

  왼쪽   살아남은 단위가 k 개인 부분망은 C(n, k) 가지다. 막대가 파스칼
         삼각형의 한 줄이고 그 합이 2^n 이다.
  오른쪽 쓸 때는 2^n 개를 다 돌릴 수 없으므로, 단위를 모두 켜 놓고
         가중치에 유지확률을 곱한 **한 번**의 계산으로 그 평균을 대신한다.
         층이 선형이면 둘이 정확히 같고, 비선형이 끼면 벌어진다.

이 쪽의 벌점(L2)과 달리 드롭아웃은 목적함수에 항을 더하지 않는다. 그런데도
정칙화로 듣는 까닭이 이 평균내기에 있다.

만드는 파일:

  ch20/optimization/img/dropout_subnets.png

실행:  python3 scripts/make_ch20_dropout_subnets.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import itertools
import math
from math import comb

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch20/optimization/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"

N = 12
SEED = 20261005


def main():
    rng = np.random.default_rng(SEED)
    w = rng.normal(size=N).round(3)              # 출력 쪽 가중치
    h = rng.normal(size=N).round(3)              # 은닉 단위가 내놓는 값

    sizes = [0] * (N + 1)
    tot_lin = tot_nl = 0.0
    for m in itertools.product((0, 1), repeat=N):   # 2^12 가지를 전부
        m = np.array(m)
        sizes[m.sum()] += 1
        s = float(w @ (m * h))
        tot_lin += s
        tot_nl += math.tanh(s)

    cnt = 2 ** N
    full = float(w @ h)
    lin_avg, nl_avg = tot_lin / cnt, tot_nl / cnt
    lin_scaled, nl_scaled = 0.5 * full, math.tanh(0.5 * full)

    print(f"    부분망의 수 {cnt} = 2^{N}")
    print(f"    크기별 개수 {' '.join(str(c) for c in sizes)}  합 {sum(sizes)}")
    print(f"    선형   마스크 평균 {lin_avg:.6f}   가중치 스케일링 {lin_scaled:.6f}")
    print(f"    tanh   마스크 평균 {nl_avg:.6f}   스케일링 뒤    {nl_scaled:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    # ── 왼쪽: 크기별 부분망의 개수 ──
    ax = axes[0]
    ks = np.arange(N + 1)
    ax.bar(ks, sizes, color=TEAL, width=0.72)
    ax.set_xticks(ks)
    ax.set_xlabel("살아남은 단위의 수 k", fontsize=11.5)
    ax.set_ylabel("그런 부분망의 개수", fontsize=11.5)
    ax.set_title(f"파스칼 삼각형의 {N} 행 — 합이 $2^{{{N}}}$ = {cnt}",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, max(sizes) * 1.20)
    ax.text(N / 2, max(sizes) * 1.05, f"가운데가 $\\binom{{{N}}}{{6}}$ = {max(sizes)}",
            ha="center", fontsize=11, color=INK)
    ax.text(0.3, max(sizes) * 0.55,
            "단위 하나를 짚으면\n정확히 절반인 $2^{11}$ 개\n부분망에서만 살아 있다",
            fontsize=10.2, color=AMBER)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 평균 대신 스케일링 한 번 ──
    ax = axes[1]
    x = np.arange(2)
    ax.bar(x - 0.18, [lin_avg, nl_avg], width=0.34, color=TEAL,
           label=f"{cnt} 개를 다 돌린 평균")
    ax.bar(x + 0.18, [lin_scaled, nl_scaled], width=0.34, color=AMBER,
           label="가중치 스케일링 한 번")
    for xi, (a, b) in enumerate([(lin_avg, lin_scaled), (nl_avg, nl_scaled)]):
        ax.text(xi - 0.18, a + 0.05, f"{a:.4f}", ha="center", fontsize=10, color=TEAL)
        ax.text(xi + 0.18, b + 0.05, f"{b:.4f}", ha="center", fontsize=10, color=AMBER)
    ax.set_xticks(x)
    ax.set_xticklabels(["선형 층", "tanh 를 한 번 통과"], fontsize=11.5)
    ax.set_ylabel("출력", fontsize=11.5)
    ax.set_title("선형이면 겹치고 비선형이 끼면 벌어진다",
                 fontsize=12.8, color=INK, pad=8)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_ylim(0, max(lin_avg, lin_scaled) * 1.25)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    gap = abs(nl_avg - nl_scaled)
    ax.annotate(f"차 {gap:.3f}", xy=(1.0, (nl_avg + nl_scaled) / 2),
                xytext=(1.35, max(lin_avg, lin_scaled) * 0.75),
                fontsize=10.4, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

    fig.suptitle("단위를 끄는 일이 곧 부분망을 고르는 일이다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"은닉 단위 {N} 개, 유지확률 $1/2$ 로 둔 것이다. 왼쪽 막대의 합이 "
             f"부분망의 총수이고, 오른쪽은 그 전부를 평균 내는 대신 쓰는 한 번의 셈이다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "dropout_subnets.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
