r"""9.1절 「그림으로 보는 통계학」 — 찻잔 여덟 개로 세운 귀무가설.

피셔가 1935 년 『실험의 설계』 2 장에서 든 보기다. 우유를 먼저 넣었는지
차를 먼저 넣었는지 가려낼 수 있다고 한 사람에게 찻잔 여덟 개를 준다.
넷은 우유가 먼저, 넷은 차가 먼저이고, 그 사실을 미리 알려 준 뒤 순서만
무작위로 섞는다.

귀무가설은 「가려내지 못한다」, 곧 넷을 아무렇게나 고른다는 것이다.
그러면 맞힌 개수의 분포가 초기하분포로 **완전히 정해진다** — 정규성도
모집단 모형도 필요 없다. 섞은 방식이 분포를 만든다.

  맞힌 개수  0:1/70  1:16/70  2:36/70  3:16/70  4:1/70
  넷을 다 맞힐 확률 = 1/70 = 0.0143
  셋 이상 맞힐 확률 = 17/70 = 0.2429   <- 하나만 틀려도 유의하지 않다

잔의 수를 고르는 일이 곧 유의수준을 고르는 일이다.

  6 잔(3+3)  1/20  = 0.0500   <- 0.05 에 딱 걸린다
  8 잔(4+4)  1/70  = 0.0143
  10 잔(5+5) 1/252 = 0.0040

만드는 파일:

  ch09/foundations/img/lady_tasting_tea.png

실행:  python3 scripts/make_ch09_lady_tasting_tea.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

from math import comb

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/foundations/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N = 4                      # 한쪽 잔의 수 (전체 8 잔)
SIZES = (3, 4, 5, 6)       # 한쪽 잔의 수 — 전체는 그 두 배


def main():
    tot = comb(2 * N, N)
    pmf = np.array([comb(N, k) * comb(N, N - k) / tot for k in range(N + 1)])
    print(f"    잔 {2*N} 개 · 경우의 수 {tot}")
    for k in range(N + 1):
        print(f"      {k} 개 맞힘  {int(round(pmf[k]*tot)):2}/{tot} = {pmf[k]:.4f}")
    print(f"      셋 이상 맞힐 확률 {pmf[3:].sum():.4f}")
    print("    잔의 수에 따른 「모두 맞힐 확률」")
    for n in SIZES:
        print(f"      {2*n:2} 잔  1/{comb(2*n, n):3} = {1/comb(2*n, n):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 여덟 잔에서 맞힌 개수의 귀무분포 ──
    ax = axes[0]
    cols = [MUTED] * N + [RED]
    ax.bar(range(N + 1), pmf, color=cols, width=0.62)
    for k in range(N + 1):
        ax.text(k, pmf[k] + 0.012, f"{int(round(pmf[k]*tot))}/{tot}",
                ha="center", fontsize=10.6, color=INK)
    ax.axhline(0.05, color=TEAL, lw=2.2, ls="--")
    ax.text(0.1, 0.064, "유의수준 $5\\%$", fontsize=10.8, color=TEAL)
    ax.annotate(f"여기 하나뿐이다\n$1/{tot} = {pmf[N]:.4f}$",
                xy=(N, pmf[N]), xytext=(2.75, 0.30), fontsize=10.8, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(2.0, 0.44, f"셋 이상 맞혀도 $p = {pmf[3:].sum():.3f}$ —\n"
                       "하나만 틀리면 증거가 안 된다",
            ha="center", fontsize=10.4, color=INK)
    ax.set_xticks(range(N + 1))
    ax.set_xlabel("우유를 먼저 넣은 잔을 맞힌 개수", fontsize=11.5)
    ax.set_ylabel("「가려내지 못한다」 아래에서의 확률", fontsize=11.5)
    ax.set_title(f"잔 {2*N} 개 — 섞은 방식이 분포를 만든다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 0.60)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 잔을 몇 개 낼 것인가 ──
    ax = axes[1]
    ps = [1 / comb(2 * n, n) for n in SIZES]
    xs = [2 * n for n in SIZES]
    ax.bar(range(len(SIZES)), ps, color=[AMBER if n != N else RED for n in SIZES],
           width=0.56)
    ax.axhline(0.05, color=TEAL, lw=2.2, ls="--")
    ax.text(len(SIZES) - 0.55, 0.0535, "유의수준 $5\\%$", fontsize=10.8,
            color=TEAL, ha="right")
    for i, (n, pv) in enumerate(zip(SIZES, ps)):
        ax.text(i, pv + 0.0022, f"$1/{comb(2*n,n)}$\n{pv:.4f}", ha="center",
                fontsize=10.4, color=INK)
    ax.annotate("여섯 잔이면 다 맞혀도\n딱 $0.05$ — 여유가 없다",
                xy=(0, ps[0]), xytext=(1.35, 0.0415), fontsize=10.6, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("피셔가 고른 쪽", xy=(1, ps[1]), xytext=(1.0, 0.0245),
                fontsize=11, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([f"{v} 잔" for v in xs], fontsize=11.5)
    ax.set_xlabel("내어 놓은 잔의 수", fontsize=11.5)
    ax.set_ylabel("모두 맞힐 확률", fontsize=11.5)
    ax.set_title("잔의 수를 정하는 일이 곧 문턱을 정하는 일이다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 0.062)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("자료를 받기 전에 $p$ 값이 정해져 있다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "귀무가설은 「가려내지 못한다」 하나뿐이고, 그 아래의 분포는 "
             "잔을 어떻게 섞었는지만으로 정해진다. 정규성도 모집단 모형도 "
             "쓰이지 않았다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "lady_tasting_tea.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
