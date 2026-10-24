r"""11.6절 「그림으로 보는 통계학」 — 2 는 어디서 왔는가.

아카이케 히로츠구는 1973 년 부다페스트에서 열린 제2회 국제 정보이론
심포지엄에서 「정보이론과 최대가능도 원리의 확장」을 발표했다. 거기에
지금 AIC 라 부르는 식이 들어 있다.

  AIC = -2 (로그가능도) + 2k

이름에 대해 본인이 적어 둔 것이 있다. A 는 아카이케의 A 가 아니라
「an」의 A 다 — 그냥 「어떤 정보기준」이라는 뜻이다. 2006 년에 교토상을
받았다.

벌점의 2 는 어디서 왔는가. 어림짐작이 아니라 **재어서 나온 수**다.

훈련자료로 맞춘 모형은 그 훈련자료에서 실제보다 잘 맞는다. 모수를
자료에 맞추느라 자료의 잡음까지 조금 빨아들였기 때문이다. 이 「낙관」의
크기를 정규선형모형(분산을 아는 경우)에서 적어 보면

  E[훈련 이탈도] = n - k        E[시험 이탈도] = n + k

이고 차이가 정확히 2k 다. AIC 가 더하는 수가 바로 이 낙관의 크기이고,
그래서 AIC 는 **시험 이탈도의 추정값**이 된다.

n = 60, 후보 25 개, sigma = 1, 4000 번으로 재면

     k      훈련      시험     차이     2k       AIC
     1    110.1     112.2      2.1      2     112.1
     5     66.0      76.2     10.2     10      76.0
     8     57.7      73.8     16.1     16      73.7
    10     53.9      74.0     20.1     20      73.9
    20     40.3      80.7     40.4     40      80.3
    25     34.9      85.1     50.2     50      84.9

시험 이탈도의 최소점이 k = 9 이고 AIC 가 고른 곳이 k = 8 이다.
차이와 2k 의 어긋남은 어디서도 0.41 을 넘지 않는다.

만드는 파일:

  ch11/model_selection/img/aic_optimism.png

실행:  python3 scripts/make_ch13_aic_optimism.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/model_selection/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, P, SIG, REPS, SEED = 60, 25, 1.0, 4000, 1973
KS = np.arange(1, P + 1)


def main():
    rng = np.random.default_rng(SEED)
    beta = 1.2 / KS.astype(float)              # 앞쪽이 세고 뒤로 갈수록 잦아든다
    tr = np.zeros(P)
    te = np.zeros(P)
    for _ in range(REPS):
        X = rng.normal(size=(N, P))
        mu = X @ beta
        y = mu + rng.normal(0, SIG, N)         # 훈련
        ys = mu + rng.normal(0, SIG, N)        # 같은 X 에서 새로 관측한 것
        for j, k in enumerate(KS):
            Xk = X[:, :k]
            fit = Xk @ np.linalg.solve(Xk.T @ Xk, Xk.T @ y)
            tr[j] += ((y - fit) ** 2).sum()
            te[j] += ((ys - fit) ** 2).sum()
    tr /= REPS * SIG ** 2
    te /= REPS * SIG ** 2
    aic = tr + 2 * KS

    print(f"  n = {N}, 후보 {P} 개, sigma = {SIG}, {REPS} 번")
    print(f"{'k':>5}{'훈련':>10}{'시험':>10}{'차이':>9}{'2k':>6}"
          f"{'AIC':>10}")
    for j, k in enumerate(KS):
        if k in (1, 3, 5, 8, 10, 15, 20, 25):
            print(f"{k:>5}{tr[j]:>10.1f}{te[j]:>10.1f}{te[j]-tr[j]:>9.1f}"
                  f"{2*k:>6}{aic[j]:>10.1f}")
    kt, ka = int(KS[te.argmin()]), int(KS[aic.argmin()])
    print(f"\n  시험 이탈도의 최소점 k = {kt},  AIC 의 최소점 k = {ka}")
    print(f"  차이 - 2k 의 최대 어긋남 {np.abs((te - tr) - 2 * KS).max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: AIC 는 시험 이탈도를 겨눈다 ──
    ax = axes[0]
    ax.plot(KS, te, lw=4.2, color=INK, label="시험 이탈도 — 알 수 없는 값")
    ax.plot(KS, aic, lw=3.0, color=RED, ls="--", marker="o", ms=5,
            label="AIC $=$ 훈련 이탈도 $+\\,2k$")
    ax.plot(KS, tr, lw=3.0, color=MUTED, label="훈련 이탈도 — 계속 내려간다")
    ax.plot([kt], [te[kt - 1]], marker="o", ms=13, color=INK, zorder=5,
            mec="white", mew=1.7)
    ax.annotate(f"시험 이탈도가 가장 작은 곳 $k = {kt}$\n"
                f"AIC 가 고른 곳도 $k = {ka}$",
                xy=(kt, te[kt - 1]), xytext=(16.5, 58.0), fontsize=10.8,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.annotate("훈련 이탈도만 보면\n크게 만들수록 좋다고 나온다",
                xy=(21.5, tr[21]), xytext=(9.5, 34.5), fontsize=10.8,
                color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.set_xlabel("모형에 넣은 설명변수의 수 $k$", fontsize=11.5)
    ax.set_ylabel("이탈도 (잔차제곱합 $\\div\\,\\sigma^2$)", fontsize=11.5)
    ax.set_title("AIC 는 시험 이탈도를 겨눈다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0, P + 1)
    ax.set_ylim(30, 118)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 낙관의 크기가 정확히 2k 다 ──
    ax = axes[1]
    ax.plot(KS, 2 * KS, lw=4.4, color=MUTED, label="$2k$ — 벌점이 더하는 수")
    ax.plot(KS, te - tr, lw=0, marker="o", ms=10, mfc="none", mec=RED,
            mew=2.4, label="재어 본 낙관 (시험 $-$ 훈련)")
    ax.annotate(f"어긋남이 어디서나 "
                f"${np.abs((te-tr)-2*KS).max():.2f}$ 를 넘지 않는다",
                xy=(18, 2 * 18), xytext=(7.4, 40.0), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(12.6, 4.0, "모수 하나를 더 넣을 때마다\n"
            "훈련에서 $1$ 만큼 덜 틀리고\n시험에서 $1$ 만큼 더 틀린다",
            fontsize=10.6, color=INK)
    ax.set_xlabel("모형에 넣은 설명변수의 수 $k$", fontsize=11.5)
    ax.set_ylabel("훈련이 시험보다 좋아 보이는 몫", fontsize=11.5)
    ax.set_title("낙관의 크기가 정확히 $2k$ 다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0, P + 1)
    ax.set_ylim(0, 54)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$2$ 는 어디서 왔는가", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"관측 ${N}$ 개, 후보 설명변수 ${P}$ 개, $\\sigma = {SIG:.0f}$ "
             "을 안다고 두었다. 시험 이탈도는 같은 $X$ 에서 $y$ 만 새로 "
             f"관측해 잰 것이고, 모든 곡선은 ${REPS//1000}{{,}}000$ 번의 "
             "평균이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "aic_optimism.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
