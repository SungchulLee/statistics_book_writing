r"""6.6절 「AI로 보는 통계학」 — U자 오른쪽에서 다시 내려간다.

이 쪽 5 절은 시험 MSE 가 U 자를 그린다고 적는다. 모수를 늘리면 편향이
줄고 분산이 늘어 어딘가에 최적점이 있다는 것이다. 옳은 그림이고, 모수
수가 표본 크기보다 적은 동안에는 거의 언제나 맞는다.

그 선을 넘어가면 달라진다.

모수 수 p 가 표본 크기 n 에 **닿는 순간** 시험오차가 치솟는다. 자료를
정확히 맞히는 해가 딱 하나뿐이라 잡음까지 그대로 외우기 때문이다.
그런데 p 를 더 늘리면 자료를 정확히 맞히는 해가 **여럿** 생기고, 그
가운데 노름이 가장 작은 것을 고르면 오차가 **다시 내려간다.**

이것을 이중 하강(double descent)이라 한다. 벨킨 등(2019)이 여러 모형
가족에서 같은 모양을 보이며 이름을 붙였다.

n = 40, 전체 특징 400, 잡음 sd 0.4 로 재면

  신호가 몇 개에 몰려 있을 때
    첫 바닥 p=8 에서 0.121   봉우리 p=40 에서 10.5   둘째 바닥 p=70 에서 0.672
    -> 고전적 최적점이 5.5 배 낫다

  신호가 넓게 퍼져 있을 때
    첫 바닥 p=2 에서 1.014   봉우리 p=40 에서 120.0   둘째 바닥 p=400 에서 0.923
    -> 과매개변수 쪽이 이긴다

이중 하강은 언제나 일어나지만, 어느 쪽이 이기느냐는 문제에 달렸다.

만드는 파일:

  ch06/estimator_quality/img/double_descent.png

실행:  python3 scripts/make_ch06_double_descent.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch06/estimator_quality/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, D, NT, SIG, REPS, SEED = 40, 400, 6000, 0.4, 50, 2019
PS = [1, 2, 3, 5, 8, 12, 18, 25, 32, 36, 39, 40, 41, 43, 48, 55, 70,
      100, 150, 250, 400]
#        감쇠  색   제목                      첫바닥 글   봉우리 글    둘째바닥 글
CASES = ((1.0, TEAL, "신호가 몇 개에 몰려 있을 때",
          (0.075, 0.085), (0.19, 20.0), (3.4, 0.15)),
         (0.1, RED, "신호가 넓게 퍼져 있을 때",
          (0.045, 0.22), (0.19, 14.0), (3.4, 0.22)))


def run(decay, rng):
    beta = 1.0 / np.arange(1, D + 1) ** decay
    beta /= np.linalg.norm(beta)
    out = []
    for p in PS:
        te = []
        for _ in range(REPS):
            Xtr = rng.normal(size=(N, D))
            Xte = rng.normal(size=(NT, D))
            ytr = Xtr @ beta + rng.normal(0, SIG, N)
            yte = Xte @ beta
            w = np.linalg.pinv(Xtr[:, :p]) @ ytr
            te.append(np.mean((Xte[:, :p] @ w - yte) ** 2))
        out.append(float(np.median(te)))
    return np.array(out)


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    for decay, col, lab, _, _, _ in CASES:
        v = run(decay, rng)
        res[decay] = v
        pre = [(p, x) for p, x in zip(PS, v) if p < N]
        post = [(p, x) for p, x in zip(PS, v) if p > N]
        p1, v1 = min(pre, key=lambda t: t[1])
        p2, v2 = min(post, key=lambda t: t[1])
        pk = max(x for p, x in zip(PS, v) if 39 <= p <= 41)
        print(f"  {lab}")
        print(f"    첫 바닥 p={p1} {v1:.4f}   봉우리 {pk:.1f}   "
              f"둘째 바닥 p={p2} {v2:.4f}   "
              f"{'과매개변수 승' if v2 < v1 else f'고전 승 ({v2/v1:.1f} 배)'}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)
    xs = np.array(PS) / N

    for ax, (decay, col, lab, t1, tp, t2) in zip(axes, CASES):
        v = res[decay]
        pre = [(p, x) for p, x in zip(PS, v) if p < N]
        post = [(p, x) for p, x in zip(PS, v) if p > N]
        p1, v1 = min(pre, key=lambda t: t[1])
        p2, v2 = min(post, key=lambda t: t[1])
        ax.axvspan(xs[0], 1.0, color=MUTED, alpha=0.12)
        ax.plot(xs, v, lw=3.4, color=col, marker="o", ms=5)
        ax.axvline(1.0, color=INK, lw=2.0, ls="--")
        ax.plot([p1 / N], [v1], marker="o", ms=13, color=INK, zorder=5,
                mec="white", mew=1.6)
        ax.plot([p2 / N], [v2], marker="D", ms=12, color=col, zorder=5,
                mec="white", mew=1.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([0.025, 0.1, 0.5, 1, 2, 10])
        ax.set_xticklabels(["0.025", "0.1", "0.5", "1", "2", "10"])
        ax.minorticks_off()
        ax.set_xlabel("모수 수 $\\div$ 표본 크기 $(p/n)$", fontsize=11.5)
        ax.set_title(lab, fontsize=12.8, color=col, pad=8)
        ax.set_ylim(0.07, 300)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.045, 190, "교과서의 $U$ 자가\n사는 구간", fontsize=10.6,
                color=GREY, ha="center")
        ax.annotate(f"첫 바닥 ${v1:.3f}$",
                    xy=(p1 / N, v1), xytext=t1,
                    fontsize=10.6, color=INK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        ax.annotate(f"$p = n$ 에서 ${v[PS.index(N)]:.0f}$ 로 치솟는다",
                    xy=(1.0, v[PS.index(N)]), xytext=tp, fontsize=10.6,
                    color=INK, ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
        verdict = ("과매개변수 쪽이 이긴다" if v2 < v1
                   else f"고전 쪽이 ${v2/v1:.1f}$ 배 낫다")
        ax.annotate(f"둘째 바닥 ${v2:.3f}$\n{verdict}",
                    xy=(p2 / N, v2), xytext=t2, fontsize=10.6,
                    color=col, ha="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))

    axes[0].set_ylabel("시험 MSE (중앙값)", fontsize=11.5)
    axes[0].set_yticks([0.1, 1, 10, 100])
    axes[0].set_yticklabels(["0.1", "1", "10", "100"])
    axes[1].set_yticks([0.1, 1, 10, 100])
    axes[1].set_yticklabels(["0.1", "1", "10", "100"])

    fig.suptitle("$U$ 자 오른쪽에서 다시 내려간다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개, 쓸 수 있는 특징 ${D}$ 개, 잡음 표준편차 ${SIG}$. "
             f"$p > n$ 이면 자료를 정확히 맞히는 해가 여럿이고, 그 가운데 "
             "노름이 가장 작은 것을 골랐다. 회색 구간이 교과서의 $U$ 자가 "
             "사는 곳이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "double_descent.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
