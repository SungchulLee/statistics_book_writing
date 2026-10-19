r"""7.3절 「그림으로 보는 통계학」 — 천 개 가운데 둘.

피셔는 흩어짐을 재는 데 표본표준편차 s 를 쓰라고 했다. 근거가 있었다.
자료가 정규분포에서 나왔다면 s 가 평균편차 d 보다 효율이 좋다 —
정확히 0.8760 배만큼. 평균편차로 같은 정밀도를 얻으려면 자료를
14% 쯤 더 모아야 한다는 뜻이다.

존 튜키가 1960 년에 「오염된 분포에서의 표본추출」이라는 글에서 물었다.
자료가 정규분포에서 「거의」 나왔다면 어떻게 되는가.

관측의 비율 e 가 N(0,1) 대신 N(0,3^2) 에서 나온다고 두고 효율을
다시 재면

        e          ARE(d 대 s)
   0.0000            0.8760      s 가 낫다
   0.0010            0.9483
   0.0018            1.0000      뒤집히는 자리
   0.0100            1.4390      d 가 낫다
   0.0500            2.0347

뒤집히는 자리가 e = 0.00175 다. **천 개 가운데 둘이 안 되는 수**다.
자료 천 개에 오염된 관측이 두 개 들어 있는 것을 눈으로 알아낼 방법은
없다. 그런데 그 둘이 두 추정량의 순위를 바꾼다.

「정규분포에 가깝다」는 말이 「정규분포다」와 같지 않다는 것을 수로
보인 글이고, 이것이 강건통계학의 출발점이 되었다.

만드는 파일:

  ch07/variance/img/tukey_contamination.png

실행:  python3 scripts/make_ch07_tukey_contamination.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch07/variance/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

K = 3.0                       # 오염 성분의 표준편차
C = np.sqrt(2 / np.pi)
N, REPS, EPS_SHOW, SEED = 1000, 30_000, 0.01, 1960


def are(e):
    """평균편차 d 와 표본표준편차 s 의 점근 상대효율. 1 보다 크면 d 가 낫다."""
    m2 = (1 - e) + e * K ** 2
    m4 = 3 * ((1 - e) + e * K ** 4)
    cv2_s = (m4 - m2 ** 2) / (4 * m2 ** 2)
    m1 = C * ((1 - e) + e * K)
    cv2_d = (m2 - m1 ** 2) / m1 ** 2
    return cv2_s / cv2_d


def sample(rng, n, e):
    x = rng.normal(size=n)
    bad = rng.random(n) < e
    x[bad] *= K
    return x


def main():
    cross = brentq(lambda e: are(e) - 1, 1e-9, 0.2)
    print(f"  오염 성분 N(0,{K:.0f}^2)")
    print(f"{'e':>10}{'ARE(d 대 s)':>14}")
    for e in (0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        print(f"{e:>10.4f}{are(e):>14.4f}")
    print(f"\n  뒤집히는 자리 e = {cross:.5f}  "
          f"(천 개 가운데 {cross*1000:.1f} 개)")

    rng = np.random.default_rng(SEED)
    ss = np.empty(REPS)
    dd = np.empty(REPS)
    for i in range(REPS):
        x = sample(rng, N, EPS_SHOW)
        ss[i] = x.std(ddof=1)
        dd[i] = np.abs(x - x.mean()).mean()
    cv_s = ss.std(ddof=1) / ss.mean()
    cv_d = dd.std(ddof=1) / dd.mean()
    print(f"\n  e = {EPS_SHOW}, n = {N}, {REPS} 번")
    print(f"    s 의 변동계수 {cv_s:.5f}")
    print(f"    d 의 변동계수 {cv_d:.5f}")
    print(f"    (s 의 변동계수 / d 의 변동계수)^2 = "
          f"{(cv_s/cv_d)**2:.4f}   이론 ARE {are(EPS_SHOW):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 뒤집히는 자리 ──
    ax = axes[0]
    es = np.logspace(-4, np.log10(0.2), 500)
    vals = np.array([are(e) for e in es])
    ax.axhline(1.0, color=INK, lw=2.4, ls="--")
    ax.fill_between(es, 1.0, vals, where=vals > 1, color=TEAL, alpha=0.14)
    ax.fill_between(es, vals, 1.0, where=vals < 1, color=RED, alpha=0.14)
    ax.plot(es, vals, lw=3.6, color=INK)
    ax.plot([cross], [1.0], marker="o", ms=14, color=RED, zorder=5,
            mec="white", mew=1.8)
    ax.plot([1e-4], [are(0)], marker="D", ms=11, color=RED, zorder=5,
            mec="white", mew=1.6)
    ax.annotate(f"오염이 없으면 ${are(0):.4f}$\n피셔가 $s$ 를 고른 근거다",
                xy=(1.3e-4, are(0)), xytext=(1.1e-3, 0.72), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"뒤집히는 자리 ${cross*100:.3f}\\%$\n"
                f"천 개 가운데 ${cross*1000:.1f}$ 개",
                xy=(cross, 1.0), xytext=(0.0115, 0.68), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(0.026, 1.26, "평균편차가 낫다", fontsize=11.2, color=TEAL,
            ha="center")
    ax.text(2.6e-4, 1.12, "$s$ 가 낫다", fontsize=11.2, color=RED)
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["$0.01\\%$", "$0.1\\%$", "$1\\%$", "$10\\%$"])
    ax.minorticks_off()
    ax.set_xlabel(f"$N(0,{K:.0f}^2)$ 에서 나온 관측의 비율", fontsize=11.5)
    ax.set_ylabel("평균편차 $\\div$ $s$ 의 상대효율", fontsize=11.5)
    ax.set_title("뒤집히는 자리는 생각보다 훨씬 가깝다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(0.6, 2.2)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 알아낼 수 없는 차이 ──
    ax = axes[1]
    for v, col, nm, cv in ((ss / ss.mean(), RED, "표본표준편차 $s$", cv_s),
                           (dd / dd.mean(), TEAL, "평균편차 $d$", cv_d)):
        ax.hist(v, bins=np.linspace(0.82, 1.22, 90), density=True,
                histtype="step", lw=3.0, color=col,
                label=f"{nm} — 변동계수 ${cv:.4f}$")
    ax.axvline(1.0, color=INK, lw=1.8, ls=":")
    ax.annotate(f"같은 자료인데 $s$ 쪽이 더 흔들린다\n"
                f"분산으로 ${(cv_s/cv_d)**2:.2f}$ 배",
                xy=(1.073, 1.35), xytext=(1.128, 8.6), fontsize=10.8,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xlabel("추정값 $\\div$ 그 추정량의 평균", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title(f"오염이 ${EPS_SHOW*100:.0f}\\%$ 일 때 "
                 f"($n = {N}$ 에 평균 ${int(N*EPS_SHOW)}$ 개)",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0.82, 1.22)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("천 개 가운데 둘", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"오염은 $N(0,1)$ 대신 $N(0,{K:.0f}^2)$ 에서 뽑는 것이다. "
             "왼쪽은 점근 상대효율을 식으로 구한 것이고, 오른쪽은 "
             f"${REPS//1000}{{,}}000$ 번 모의해 두 추정량의 흔들림을 "
             "직접 잰 것이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "tukey_contamination.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
