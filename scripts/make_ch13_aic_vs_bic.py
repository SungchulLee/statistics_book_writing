r"""13.6절 「그림으로 보는 통계학」 — 둘은 다른 물음에 답한다.

AIC 와 BIC 는 벌점만 2k 와 k ln n 으로 다른데, 그 한 글자가 둘을 서로
다른 목표로 갈라놓는다. 어느 쪽이 옳은지는 **무엇을 물었느냐**에 달렸고,
그림은 두 물음을 따로 재어 본다.

  왼쪽  참 모형이 후보 안에 있을 때 그것을 집어내는 비율.
        BIC 는 n 이 커지면 1 로 간다(일치성). AIC 는 0.73 언저리에서
        멈춘다 — 과적합할 확률이 사라지지 않는다.
  오른쪽 참 함수가 후보 안에 **없고** 계수가 서서히 줄어들 때의 예측오차.
        이번에는 AIC 가 이긴다. n 이 커질수록 BIC 가 너무 단순한 쪽을
        골라 뒤처진다(효율성).

작은 n 에서는 오른쪽에서도 BIC 가 낫다. 자료가 적을 때는 AIC 가 먼저
과적합하기 때문이다.

만드는 파일:

  ch13/model_selection/img/aic_vs_bic.png

실행:  python3 scripts/make_ch13_aic_vs_bic.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/model_selection/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

SIGMA, SEED = 1.0, 1978
NS_A = (30, 60, 120, 250, 500, 1000)
NS_B = (40, 80, 160, 320, 640)
TRUE_D = 3
REPS_A, REPS_B = 3000, 600
TAPER = np.array([1.6 / j ** 1.5 for j in range(1, 41)])


def scores(V, y):
    """(AIC, BIC) — 가우스 선형모형의 간단꼴. k 는 계수 + sigma^2 이다."""
    n = len(y)
    r = y - V @ np.linalg.lstsq(V, y, rcond=None)[0]
    rss = float(r @ r)
    k = V.shape[1] + 1
    base = n * np.log(rss / n)
    return base + 2 * k, base + k * np.log(n)


def cos_basis(x, d):
    return np.column_stack([np.ones_like(x)] +
                           [np.cos(j * np.pi * x) for j in range(1, d + 1)])


def regime_a(rng):
    """참 모형이 후보 안에 있다 — 몇 번이나 집어내는가."""
    beta = np.array([0.5, 1.2, -0.8, 1.5])
    hit_a, hit_b = [], []
    for n in NS_A:
        x = np.linspace(-1, 1, n)
        V = np.vander(x, TRUE_D + 1, increasing=True)
        ha = hb = 0
        for _ in range(REPS_A):
            y = V @ beta + rng.normal(0, SIGMA, n)
            s = [scores(np.vander(x, d + 1, increasing=True), y) + (d,)
                 for d in range(9)]
            ha += min(s, key=lambda z: z[0])[2] == TRUE_D
            hb += min(s, key=lambda z: z[1])[2] == TRUE_D
        hit_a.append(ha / REPS_A)
        hit_b.append(hb / REPS_A)
        print(f"    A  n={n:5}  AIC 적중 {ha/REPS_A:.4f}   BIC 적중 {hb/REPS_A:.4f}")
    return hit_a, hit_b


def regime_b(rng):
    """참 함수가 후보 밖이고 계수가 서서히 줄어든다 — 예측오차."""
    ratio, errs = [], []
    for n in NS_B:
        x = np.linspace(-1, 1, n)
        truth = cos_basis(x, 40)[:, 1:] @ TAPER
        ea = eb = 0.0
        for _ in range(REPS_B):
            y = truth + rng.normal(0, SIGMA, n)
            s = []
            for d in range(25):
                V = cos_basis(x, d)
                if V.shape[1] >= n:
                    break
                s.append(scores(V, y) + (d,))
            for which, key in (("a", 0), ("b", 1)):
                d = min(s, key=lambda z: z[key])[2]
                V = cos_basis(x, d)
                fit = V @ np.linalg.lstsq(V, y, rcond=None)[0]
                e = float(((fit - truth) ** 2).mean())
                if which == "a":
                    ea += e
                else:
                    eb += e
        ratio.append(eb / ea)
        errs.append((ea / REPS_B, eb / REPS_B))
        print(f"    B  n={n:5}  AIC 오차 {ea/REPS_B:.5f}   BIC 오차 {eb/REPS_B:.5f}"
              f"   BIC/AIC {eb/ea:.3f}")
    return ratio, errs


def main():
    rng = np.random.default_rng(SEED)
    hit_a, hit_b = regime_a(rng)
    ratio, _ = regime_b(rng)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    ax = axes[0]
    ax.plot(NS_A, np.array(hit_b) * 100, marker="s", ms=7, lw=2.6, color=TEAL,
            label="BIC")
    ax.plot(NS_A, np.array(hit_a) * 100, marker="o", ms=7, lw=2.6, color=AMBER,
            label="AIC")
    ax.axhline(100, color=MUTED, lw=1.6, ls="--")
    ax.annotate(f"{hit_a[-1]*100:.0f}% 에서 멈춘다",
                xy=(NS_A[-1], hit_a[-1] * 100), xytext=(150, hit_a[-1] * 100 - 16),
                fontsize=10.8, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks(NS_A)
    ax.set_xticklabels([str(v) for v in NS_A])
    ax.minorticks_off()
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("참 모형을 집어낸 비율 (%)", fontsize=11.5)
    ax.set_title("참 모형이 후보 안에 있을 때", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axhline(1.0, color=MUTED, lw=2.0, ls="--")
    ax.plot(NS_B, ratio, marker="D", ms=8, lw=2.8, color=RED)
    ax.fill_between(NS_B, 1.0, ratio, where=np.array(ratio) >= 1,
                    color=AMBER, alpha=0.18, interpolate=True)
    ax.fill_between(NS_B, 1.0, ratio, where=np.array(ratio) <= 1,
                    color=TEAL, alpha=0.18, interpolate=True)
    ax.text(NS_B[0] * 1.05, 0.76, "BIC 가 낫다", fontsize=11, color=TEAL)
    ax.text(NS_B[-1] * 0.99, 1.05, "AIC 가 낫다", fontsize=11, color=AMBER,
            ha="right")
    ax.annotate(f"BIC 의 오차가\n{ratio[-1]:.2f} 배다", xy=(NS_B[-1], ratio[-1]),
                xytext=(NS_B[-1] * 0.45, ratio[-1] - 0.02),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xscale("log")
    ax.set_xticks(NS_B)
    ax.set_xticklabels([str(v) for v in NS_B])
    ax.minorticks_off()
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("BIC 의 예측오차 ÷ AIC 의 예측오차", fontsize=11.5)
    ax.set_title("참 함수가 후보 안에 없을 때", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("벌점 한 글자가 목표를 갈라놓는다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "왼쪽은 $3$ 차 다항이 참인 자료, 오른쪽은 계수가 $1/j^{1.5}$ 로 "
             "줄어들어 어떤 유한 모형도 참이 아닌 자료다. "
             "같은 두 기준이 자리를 바꾼다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "aic_vs_bic.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
