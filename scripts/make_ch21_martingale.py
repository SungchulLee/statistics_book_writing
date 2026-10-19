r"""21.2절 「그림으로 보는 통계학」 — 세던 것에서 기댓값을 빼면.

이 쪽은 넬슨-알렌 추정량을 세우고, 그 분산이 왜 옳은지는 「계수과정과
마팅게일 이론에서 나온다」는 한 줄로 넘긴다. 그 한 줄 뒤에 생존분석의
가장 큰 통합이 있다.

1970 년대 중반까지 이 분야의 큰 결과들 — 카플란-마이어, 넬슨-알렌,
로그순위, 콕스 — 은 각자의 대표본 논증을 따로 가지고 있었다. 절단이
들어가면 관측들이 독립이 아니게 되어 보통의 중심극한정리를 쓸 수 없고,
그때마다 그 상황에 맞는 손수 만든 논증이 필요했다.

오드 아알렌이 1975 년 버클리 박사논문에서 틀을 바꾸었다. 자료를
**계수과정**으로 보는 것이다.

  N(t) = 시각 t 까지 일어난 사건의 수
  Y(t) = 시각 t 직전에 위험집합에 있는 사람 수

그리고 센 것에서 그때그때의 기댓값을 빼면

  M(t) = N(t) - ∫ Y(s) h(s) ds

이것이 **마팅게일**이다. 어느 시점에서나 평균이 0 이고, 그 분산이
빼낸 양(보상자)의 기댓값과 같다. 이 사실 하나에서 모든 점근이론이
한꺼번에 나온다.

모의로 확인하면 (n=50, 지수 lam=0.5, 균등(0,3) 절단, 4 만 번)

    t      E[M(t)]   표준오차    Var[M(t)]   E[보상자]
  0.50      0.007    0.016       10.149     10.174
  1.50      0.029    0.023       20.382     20.594
  3.00      0.028    0.025       23.968     24.094

평균은 어느 시점에서나 표준오차 안이고 분산은 보상자와 맞는다.

이 틀의 값은 일반성이다. 절단, 시차 입장, 시간에 따라 변하는 공변량,
반복 사건, 경쟁 위험 — 모두 Y(t) 나 N(t) 의 정의만 바뀔 뿐이고 기계는
그대로다. 앤더슨과 길이 1982 년에 콕스 모형까지 같은 방식으로 끝냈다.

만드는 파일:

  ch21/nonparametric/img/martingale.png

실행:  python3 scripts/make_ch21_martingale.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/nonparametric/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, LAM, TMAX, R, SEED = 50, 0.5, 3.0, 40_000, 1975
NPATH = 60
GRID = np.linspace(0, TMAX, 121)


def main():
    rng = np.random.default_rng(SEED)
    T = rng.exponential(1 / LAM, (R, N))
    C = rng.uniform(0, TMAX, (R, N))
    t = np.minimum(T, C)
    d = T <= C

    M = np.empty((R, GRID.size))
    A = np.empty((R, GRID.size))
    for j, g in enumerate(GRID):
        M[:, j] = (d & (t <= g)).sum(1) - LAM * np.minimum(t, g).sum(1)
        A[:, j] = LAM * np.minimum(t, g).sum(1)

    print(f"  n={N}, 지수(lam={LAM}), 균등(0,{TMAX:g}) 절단, {R:,} 번")
    print(f"{'t':>6}{'E[M(t)]':>11}{'표준오차':>10}{'Var[M(t)]':>12}{'E[보상자]':>12}")
    for g in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        j = int(np.argmin(abs(GRID - g)))
        print(f"{g:>6.2f}{M[:, j].mean():>11.4f}{M[:, j].std()/np.sqrt(R):>10.4f}"
              f"{M[:, j].var():>12.3f}{A[:, j].mean():>12.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 0 둘레를 떠도는 길들 ──
    ax = axes[0]
    env = 2 * np.sqrt(A.mean(0))
    ax.fill_between(GRID, -env, env, color=TEAL, alpha=0.16,
                    label="보상자가 예측하는 $\\pm 2$ 표준편차")
    for r in range(NPATH):
        ax.step(GRID, M[r], where="post", lw=1.0, color=MUTED, alpha=0.55)
    ax.plot(GRID, M.mean(0), lw=3.2, color=RED,
            label=f"{R:,} 번의 평균".replace(",", ",") + " — $0$ 에 붙어 있다")
    ax.axhline(0, color=INK, lw=2.0, ls="--")
    ax.annotate("센 것에서 그때그때의 기댓값을 뺀 것이다.\n"
                "어느 시점에서 멈추어도 평균이 $0$ 이다.",
                xy=(2.2, 0.4), xytext=(1.45, 14.5), fontsize=10.8, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("시간 $t$", fontsize=11.5)
    ax.set_ylabel("$M(t) = N(t) - \\int_0^t Y(s)h(s)\\,ds$", fontsize=11.5)
    ax.set_title(f"길 {NPATH} 개를 겹쳐 그린 것", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0, TMAX)
    ax.set_ylim(-19, 19)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 분산이 보상자와 같다 ──
    ax = axes[1]
    ax.plot(GRID, A.mean(0), lw=3.4, color=TEAL,
            label="빼낸 양의 기댓값 — $E\\left[\\int_0^t Y h\\,ds\\right]$")
    k = np.arange(0, GRID.size, 6)
    ax.plot(GRID[k], M.var(0)[k], lw=0, marker="o", ms=9, mfc="white",
            mec=RED, mew=2.2, label="$M(t)$ 의 실제 분산")
    ax.annotate("두 곡선이 겹친다 —\n「분산을 따로 유도할 필요가 없다」는 뜻이다",
                xy=(2.1, float(A.mean(0)[int(2.1/TMAX*(GRID.size-1))])),
                xytext=(1.5, 7.0), fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("시간 $t$", fontsize=11.5)
    ax.set_ylabel("분산", fontsize=11.5)
    ax.set_title("분산이 저절로 따라 나온다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, TMAX)
    ax.set_ylim(0, 27)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("센 것에서 그때그때의 기댓값을 빼면 마팅게일이 된다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             f"사람 ${N}$ 명을 {R//10000} 만 번 만들었다. 절단이 있어 관측들이 "
             "서로 독립이 아닌데도, 이 한 가지 성질에서 넬슨–알렌·"
             "카플란–마이어·로그순위·콕스의 점근이론이 모두 따라 나온다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "martingale.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
