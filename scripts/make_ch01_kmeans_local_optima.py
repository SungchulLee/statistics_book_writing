r"""1.5절 「그림으로 보는 통계학」 — 어디서 시작했느냐가 답을 바꾼다.

스튜어트 로이드는 1957 년 벨 연구소에서 신호를 몇 단계로 끊어 보낼지를
정하는 문제를 풀다가 한 절차를 적었다. 중심을 잡고, 가까운 쪽에 점을
붙이고, 붙은 점들의 평균으로 중심을 옮기고, 다시 붙이고. 사내 보고서로
돌았을 뿐 활자로 나온 것은 1982 년이다. 그 사이 포지(1965)와
매퀸(1967)이 같은 것을 따로 적었고, 「k-평균」이라는 이름은 매퀸이
붙였다.

이 절차에는 보장이 하나 있다. 한 바퀴 돌 때마다 목적함수가 줄거나
그대로이고, 그래서 **반드시 멈춘다.** 그런데 멈춘 자리가 가장 좋은
자리라는 보장은 없다. 국소최적이다.

점 1500 개를 덩어리 K 개로 만들어 놓고, 시작점만 바꿔 120 번씩 돌리면

   K    무작위 시작: 최적 도달   중앙값/최적     k-means++: 도달   중앙값/최적
   4        0.583            1.000              1.000        1.000
   8        0.100            3.731              0.908        1.000
  12        0.025            2.299              0.558        1.000
  16        0.000            1.885              0.058        1.032
  20        0.000            1.917              0.017        1.084

K = 8 이면 무작위로 시작해 열 번에 한 번만 제일 좋은 자리에 닿고,
보통은 목적함수가 최적의 3.7 배인 데서 멈춘다. K = 16 이면 120 번을
돌려도 한 번도 닿지 못한다.

아서와 바실비츠키가 2007 년에 내놓은 k-means++ 는 첫 중심들을 서로
멀찍이 떨어지게 고르는 것뿐인데, K = 8 에서 도달률이 0.100 에서
0.908 로 간다.

만드는 파일:

  ch01/paradigms/img/kmeans_local_optima.png

실행:  python3 scripts/make_ch01_kmeans_local_optima.py   (저장소 최상위에서)
필요:  numpy, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/paradigms/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, STD, TRIES, SEED = 1500, 0.65, 120, 7
KS = (4, 6, 8, 10, 12, 16, 20)
KSHOW = 8


def blobs(k):
    return make_blobs(n_samples=N, centers=k, cluster_std=STD,
                      center_box=(-12, 12), random_state=SEED)[0]


def runs(X, k, init):
    out = []
    for s in range(TRIES):
        m = KMeans(k, init=init, n_init=1, random_state=s).fit(X)
        out.append((m.inertia_, m.labels_, m.cluster_centers_))
    return out


def main():
    rates = {"random": [], "k-means++": []}
    ratio = {"random": [], "k-means++": []}
    print(f"{'K':>4}{'무작위 도달':>12}{'중앙값/최적':>14}"
          f"{'++ 도달':>12}{'중앙값/최적':>14}")
    for k in KS:
        X = blobs(k)
        v = {i: np.array([r[0] for r in runs(X, k, i)])
             for i in ("random", "k-means++")}
        best = min(v["random"].min(), v["k-means++"].min())
        for i in v:
            rates[i].append(float((v[i] < best * 1.001).mean()))
            ratio[i].append(float(np.median(v[i]) / best))
        print(f"{k:>4}{rates['random'][-1]:>12.3f}{ratio['random'][-1]:>14.3f}"
              f"{rates['k-means++'][-1]:>12.3f}"
              f"{ratio['k-means++'][-1]:>14.3f}")

    # 왼쪽 그림에 쓸 두 판 — 가장 좋은 것과 전형적으로 나쁜 것
    X = blobs(KSHOW)
    rr = runs(X, KSHOW, "random")
    rr.sort(key=lambda t: t[0])
    good, bad = rr[0], rr[len(rr) // 2]
    print(f"\n  K = {KSHOW} 에서 보인 두 판: "
          f"최적 {good[0]:.1f},  중앙값 자리 {bad[0]:.1f} "
          f"({bad[0]/good[0]:.2f} 배)")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 같은 자료, 같은 알고리즘, 다른 시작점 ──
    ax = axes[0]
    off = 26.0
    cmap = plt.get_cmap("tab10")
    for j, (ine, lab, cen) in enumerate((good, bad)):
        dx = j * off
        for c in range(KSHOW):
            m = lab == c
            ax.plot(X[m, 0] + dx, X[m, 1], lw=0, marker="o", ms=3.4,
                    color=cmap(c % 10), alpha=0.75, zorder=2)
        ax.plot(cen[:, 0] + dx, cen[:, 1], lw=0, marker="X", ms=13,
                color=INK, mec="white", mew=1.6, zorder=4)
        ax.text(dx, 15.2, "가장 좋은 자리" if j == 0 else "흔히 멈추는 자리",
                fontsize=11.4, color=TEAL if j == 0 else RED, ha="center")
        ax.text(dx, 13.2, f"목적함수 ${ine:.0f}$"
                + ("" if j == 0 else f" — ${ine/good[0]:.1f}$ 배"),
                fontsize=10.6, color=INK, ha="center")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(-15, 17.5)
    ax.set_title(f"같은 자료, 같은 알고리즘 ($K = {KSHOW}$)", fontsize=12.8,
                 color=INK, pad=8)
    for sp in ax.spines.values():
        sp.set_color(MUTED)

    # ── 오른쪽: 시작점을 고르는 방법만 바꾸면 ──
    ax = axes[1]
    ax.plot(KS, rates["k-means++"], lw=3.6, color=TEAL, marker="o", ms=8,
            label="$k$-means++ 로 시작")
    ax.plot(KS, rates["random"], lw=3.4, color=RED, marker="D", ms=7, ls="--",
            label="아무 데서나 시작")
    i8 = KS.index(KSHOW)
    ax.annotate(f"$K = {KSHOW}$ 에서 ${rates['random'][i8]:.3f}$ 대 "
                f"${rates['k-means++'][i8]:.3f}$",
                xy=(KSHOW, (rates["random"][i8] + rates["k-means++"][i8]) / 2),
                xytext=(14.2, 0.74), fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate(f"$K = 16$ 부터는 둘 다 거의 못 닿는다",
                xy=(16.3, 0.055), xytext=(14.6, 0.29), fontsize=10.8,
                color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.set_xlabel("덩어리의 수 $K$", fontsize=11.5)
    ax.set_ylabel(f"${TRIES}$ 번 중 가장 좋은 자리에 닿은 비율", fontsize=11.5)
    ax.set_title("시작점을 고르는 방법만 바꾸면", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xticks(KS)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("어디서 시작했느냐가 답을 바꾼다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"점 ${N}$ 개를 덩어리 $K$ 개로 만들어 놓고 시작점만 바꿔 "
             f"${TRIES}$ 번씩 돌렸다. 「목적함수」는 각 점에서 제 중심까지의 "
             "거리제곱을 모두 더한 값이며, 작을수록 좋다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "kmeans_local_optima.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
