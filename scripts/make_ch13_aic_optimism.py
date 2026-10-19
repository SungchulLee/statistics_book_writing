r"""13.6절 「그림으로 보는 통계학」 — 벌점 2p 는 어림이 아니다.

훈련자료로 잰 적합도는 낙관적이다. 같은 자료로 계수를 고르고 그 자료로
점수를 매겼으니 당연하다. AIC 가 하는 일은 그 낙관의 크기를 빼는 것인데,
선형모형에서는 그 크기를 **정확히** 적을 수 있다.

  E[RSS_train] = (n - p) sigma^2 + 치우침^2
  E[RSS_test]  = (n + p) sigma^2 + 치우침^2     (같은 X, 새 y)

빼면 치우침이 약분되어 2 p sigma^2 만 남는다. 모형이 참이든 아니든
성립하고 근사도 아니다. sigma^2 으로 나누면 바로 2p 다.

그림은 차수 0..10 다항회귀에서 세 곡선을 그린다 — 훈련 이탈도, 새 자료
이탈도, 그리고 그 차. 셋째가 기울기 2 인 직선 위에 정확히 올라앉는다.

만드는 파일:

  ch13/model_selection/img/aic_optimism.png

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

OUT = "docs/ch13/model_selection/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N = 40
SIGMA = 0.5
DEGREES = range(11)
REPS = 20_000
SEED = 13


def main():
    rng = np.random.default_rng(SEED)
    x = np.linspace(-1, 1, N)
    truth = np.sin(2.1 * x) + 0.4 * x ** 3          # 어떤 차수로도 딱 맞지 않는다

    rows = []
    for d in DEGREES:
        V = np.vander(x, d + 1, increasing=True)     # 계수 p = d + 1 개
        H = V @ np.linalg.pinv(V)                    # 모자행렬 — 사영
        p = d + 1

        ytr = truth + rng.normal(0, SIGMA, (REPS, N))
        yte = truth + rng.normal(0, SIGMA, (REPS, N))   # 같은 X, 새 y
        fit = ytr @ H.T                              # 훈련자료로 적합한 값

        rss_tr = ((ytr - fit) ** 2).sum(axis=1).mean()
        rss_te = ((yte - fit) ** 2).sum(axis=1).mean()
        rows.append((p, rss_tr / SIGMA ** 2, rss_te / SIGMA ** 2))
        print(f"    p={p:2}  훈련 {rss_tr/SIGMA**2:8.3f}   새 자료 {rss_te/SIGMA**2:8.3f}"
              f"   차 {(rss_te-rss_tr)/SIGMA**2:7.3f}   2p = {2*p}")

    ps = np.array([r[0] for r in rows])
    tr = np.array([r[1] for r in rows])
    te = np.array([r[2] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 두 곡선 ──
    ax = axes[0]
    ax.plot(ps, tr, marker="o", ms=7, lw=2.6, color=TEAL, label="훈련자료로 잰 이탈도")
    ax.plot(ps, te, marker="s", ms=7, lw=2.6, color=AMBER, label="새 자료로 잰 이탈도")
    for p, a, b in zip(ps, tr, te):
        if p in (3, 6, 11):
            ax.annotate("", xy=(p, a), xytext=(p, b),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
            ax.text(p + 0.18, (a + b) / 2, f"{b-a:.1f}", fontsize=10, color=RED)
    best = int(ps[np.argmin(te)])
    ax.axvline(best, color=MUTED, lw=1.6, ls=":")
    # p = 1 은 170 을 넘어 눈금 밖으로 보낸다 — 이야기는 2 부터다
    ax.set_ylim(26, 56)
    ax.annotate(f"상수만 쓴 p = 1 은\n{tr[0]:.0f} 로 눈금 밖이다",
                xy=(1, 55), xytext=(2.45, 52.0), fontsize=10.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))
    ax.text(best + 0.18, 28.4, f"새 자료가 가장 좋은 곳 p = {best}",
            fontsize=10.2, color=MUTED)
    ax.set_xlabel("계수의 개수 p", fontsize=11.5)
    ax.set_ylabel("이탈도  RSS / σ²", fontsize=11.5)
    ax.set_title("훈련 쪽은 내려가기만 한다", fontsize=12.8, color=INK, pad=8)
    ax.set_xticks(ps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 차가 정확히 2p ──
    ax = axes[1]
    ax.plot(ps, 2 * ps, lw=3.2, color=RED, alpha=0.55, zorder=0,
            label="$2p$ — 이론이 적는 값")
    ax.scatter(ps, te - tr, s=55, color=TEAL, zorder=4, label="모의에서 잰 차")
    ax.set_xlabel("계수의 개수 p", fontsize=11.5)
    ax.set_ylabel("새 자료에서 뺀 훈련  (이탈도 차)", fontsize=11.5)
    ax.set_title("낙관의 크기가 곧 모수의 개수다", fontsize=12.8, color=INK, pad=8)
    ax.set_xticks(ps)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    err = np.abs((te - tr) - 2 * ps).max()
    ax.text(0.97, 0.06, f"열한 점 모두 어긋남 {err:.3f} 안",
            transform=ax.transAxes, ha="right", fontsize=10.4, color=INK)

    fig.suptitle("AIC 의 벌점은 고른 수가 아니라 잰 수다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"차수 $0$–$10$ 다항회귀, $n = {N}$, $\\sigma = {SIGMA}$, "
             f"{REPS//1000} 천 번 되풀이. 참 함수는 어떤 차수로도 딱 맞지 않게 잡았다 — "
             "그래도 두 곡선의 차에서 치우침이 약분되어 $2p$ 만 남는다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "aic_optimism.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    |차 - 2p| 최대 {err:.4f}")


if __name__ == "__main__":
    main()
