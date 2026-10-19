r"""5.6절 「그림으로 보는 통계학」 — 잡음을 최적화에 넣으면.

이 쪽의 정리 1 은 Var(S^2) 이 4차 적률로 정해진다고 말하고, 균등분포에
넣어 보면 n = 5 에서 표준오차가 평균의 절반이다. 「분산 추정값이 꽤
흔들린다」는 말인데, 그래서 어떻다는 것인가.

흔들리는 추정값을 **최적화에 넣으면** 흔들림이 증폭된다. 가장 널리
알려진 보기가 포트폴리오다.

마코위츠의 평균-분산 최적화는 기대수익 벡터와 공분산행렬을 받아 가장
좋은 비중을 내놓는다. 참값을 알면 정말 좋다 — 아래 설정에서 접점
포트폴리오의 샤프비가 0.390 이고 모든 자산에 똑같이 넣는 1/N 은
0.327 이다. 19% 더 낫다.

문제는 참값을 모른다는 것이다. 표본에서 추정해 넣으면 이렇게 된다.

   자료 개월 T    1/N     표본 접점   접점 가중의 최대 |w|
         60     0.327      0.153          1.54
        120     0.327      0.229          0.52
        240     0.327      0.286          0.34
        600     0.327      0.339          0.24
       6000     0.327      0.384          0.16

**5 년치(60 개월)로 최적화하면 1/N 의 절반도 못 된다.** 20 년치로도
아직 진다. 1/N 을 이기려면 50 년치가 필요하다. 그 무렵이면 세상이
바뀌어 있다.

가운데 열이 까닭을 보여 준다. 추정오차가 들어가면 최적화가 한 자산에
154% 를 싣고 다른 자산을 공매도한다. 최적화는 추정값을 **믿는** 기계라
가장 운 좋게 추정된 자산에 몰아준다. 그래서 「오차 극대화기」라 불린다.

디미구엘·가를라피·우팔(2009)이 실제 자료로 같은 결론을 보였다.

만드는 파일:

  ch05/applications/img/naive_diversification.png

실행:  python3 scripts/make_ch05_naive_diversification.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

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

N, RHO, SEED, REPS = 25, 0.20, 2009, 2000
TS = (36, 60, 120, 240, 480, 600, 1200, 3000, 6000)
SHOW_T = 120            # 오른쪽 그림에서 보여 줄 자료 길이


def main():
    rng = np.random.default_rng(SEED)
    vol = rng.uniform(0.040, 0.070, N)                 # 월 변동성
    mu = 0.004 + 0.12 * vol + rng.normal(0, 0.0015, N)  # 월 기대수익
    S = RHO * np.outer(vol, vol)
    np.fill_diagonal(S, vol ** 2)
    Si = np.linalg.inv(S)
    one = np.ones(N)

    def sharpe(w):
        return (w @ mu) / np.sqrt(w @ S @ w)

    w_eq = one / N
    w_tan = Si @ mu / (one @ Si @ mu)
    s_eq, s_tan = sharpe(w_eq), sharpe(w_tan)
    print(f"    참값을 안다면   1/N {s_eq:.4f}   접점 {s_tan:.4f}   "
          f"({100*(s_tan/s_eq-1):.0f}% 더 낫다)")

    means, mx, sample_w = [], [], None
    print(f"\n{'자료 개월 T':>12}{'표본 접점 샤프':>15}{'최대 |w|':>11}{'1/N 대비':>11}")
    for T in TS:
        ss, mm = [], []
        for r in range(REPS):
            X = rng.multivariate_normal(mu, S, T)
            C = np.cov(X, rowvar=False)
            Ci = np.linalg.inv(C)
            w = Ci @ X.mean(0) / (one @ Ci @ X.mean(0))
            ss.append(sharpe(w))
            mm.append(np.abs(w).max())
            if T == SHOW_T and r == 0:
                sample_w = w
        means.append(np.mean(ss))
        mx.append(np.mean(mm))
        print(f"{T:>12,}{means[-1]:>15.4f}{mx[-1]:>11.2f}"
              f"{means[-1]/s_eq:>11.2f}")

    means = np.array(means)
    cross = next((T for T, m in zip(TS, means) if m > s_eq), None)
    print(f"\n    1/N 을 넘어서는 지점: 약 {cross} 개월 = {cross/12:.0f} 년")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 언제부터 최적화가 이기는가 ──
    ax = axes[0]
    ax.axhline(s_eq, color=TEAL, lw=3.0)
    ax.text(40, s_eq - 0.026, f"모두에게 똑같이 — $1/N$   {s_eq:.3f}",
            fontsize=11.2, color=TEAL)
    ax.axhline(s_tan, color=INK, lw=2.0, ls="--")
    ax.text(40, s_tan + 0.010,
            f"참값을 안다면 도달할 곳   {s_tan:.3f}", fontsize=11.2, color=INK)
    ax.plot(TS, means, lw=3.0, color=RED, marker="o", ms=7,
            label="표본으로 추정해 최적화한 포트폴리오")
    ax.fill_between(TS, means, s_eq, where=means < s_eq, color=RED, alpha=0.14)
    ax.axvline(cross, color=AMBER, lw=1.8, ls=":")
    ax.annotate(f"여기서야 $1/N$ 을 넘는다\n{cross} 개월 = {cross//12} 년치",
                xy=(cross, s_eq), xytext=(cross * 2.6, 0.175),
                fontsize=10.8, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.annotate(f"$5$ 년치로 최적화하면\n$1/N$ 의 {means[1]/s_eq*100:.0f}% 다",
                xy=(60, means[1]), xytext=(150, 0.105),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks([36, 60, 120, 240, 600, 1200, 6000])
    ax.set_xticklabels(["3 년", "5 년", "10 년", "20 년", "50 년", "100 년",
                        "500 년"])
    ax.minorticks_off()
    ax.set_xlabel("추정에 쓴 자료의 길이", fontsize=11.5)
    ax.set_ylabel("실제로 얻는 샤프비", fontsize=11.5)
    ax.set_title("추정값을 넣는 순간 최적화가 진다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.09, 0.475)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 최적화기가 내놓는 비중 ──
    ax = axes[1]
    idx = np.arange(N)
    ax.bar(idx - 0.21, w_tan, width=0.42, color=TEAL, alpha=0.85,
           label="참값을 알 때의 최적 비중")
    ax.bar(idx + 0.21, sample_w, width=0.42, color=RED, alpha=0.85,
           label=f"{SHOW_T//12} 년치로 추정해 얻은 비중")
    ax.axhline(0, color=INK, lw=1.4)
    ax.axhline(1 / N, color=GREY, lw=1.6, ls=":")
    ax.text(N - 0.4, 1 / N + 0.012, "$1/N$", fontsize=10.8, color=GREY,
            ha="right")
    j = int(np.argmax(np.abs(sample_w)))
    ax.annotate(f"이 자산 하나에 {sample_w[j]*100:.0f}% —\n"
                "모자란 돈은 다른 자산을 공매도해 만든다",
                xy=(j + 0.21, sample_w[j]),
                xytext=(N * 0.70, -0.44 if sample_w[j] < 0 else 0.44),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xticks([])
    ax.set_ylim(-0.58, 0.52)
    ax.set_xlabel(f"자산 ${N}$ 개", fontsize=11.5)
    ax.set_ylabel("포트폴리오 비중", fontsize=11.5)
    ax.set_title("잡음은 가장 운 좋은 자산으로 몰린다", fontsize=12.8,
                 color=INK, pad=8)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("추정값의 잡음을 최적화에 넣으면 증폭된다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"자산 ${N}$ 개, 월 변동성 $4$–$7\\%$, 쌍별 상관 ${RHO}$. "
             f"왼쪽은 각 길이마다 {REPS//1000} 천 번 돌린 평균이고, "
             "샤프비는 추정값이 아니라 참 모수로 계산했다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "naive_diversification.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
