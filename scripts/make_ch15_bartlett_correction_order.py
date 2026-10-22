r"""15.4절 「그림으로 보는 통계학」 — 나눗수 하나가 고치는 차수.

바틀릿의 $M$ 은 $H_0$ 아래에서 점근적으로 $\chi^2_{k-1}$ 이지만 유한표본에서는
기댓값부터 어긋난다. 정규자료·등크기 집단에서는 그 기댓값이 **닫힌 꼴로** 적힌다.

  s_p^2 = sigma^2 * chi^2_{N-k}/(N-k),   s_i^2 = sigma^2 * chi^2_{d}/d

이고 $E[\ln \chi^2_d] = \psi(d/2) + \ln 2$ 이므로

  E[M] = (N-k)[psi((N-k)/2) - ln(N-k)] - sum_i d_i[psi(d_i/2) - ln d_i]

가 된다(ln 2 는 상쇄된다). 모의실험이 필요 없다 — 왼쪽 칸은 이 식을 그린 것이다.

  보정 전  |E[M]/(k-1) - 1|      ~ n^{-1}
  보정 후  |E[M/C]/(k-1) - 1|    ~ n^{-3}

오른쪽 칸은 그 차수가 실제 거짓 양성률에서 어떻게 보이는지다(여기는 모의).
집단이 많을수록 보정 전이 더 크게 어긋난다 — k = 10, n = 3 에서 명목 5% 가 10.4%.

롤리(1956)가 보인 것은 **평균만이 아니라 분포 전체**가 O(n^{-2}) 까지 맞는다는
것이고, 그래서 이 나눗수는 가능도비 검정 일반의 기법이 되었다.

만드는 파일:

  ch15/bartlett_test/img/correction_order.png

실행:  python3 scripts/make_ch15_bartlett_correction_order.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import special, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/bartlett_test/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 1937
REPS = 200_000
CHUNK = 50_000
K_MAIN = 4
NS_SIZE = [3, 4, 5, 8, 15, 30]


def EM(k, n):
    """E[M] — 정규자료·등크기에서 닫힌 꼴. 모의실험이 아니다."""
    d = n - 1.0
    Nk = k * d
    return (Nk * (special.digamma(Nk / 2) - np.log(Nk))
            - k * d * (special.digamma(d / 2) - np.log(d)))


def Cfac(k, n):
    """바틀릿 보정인자."""
    d = n - 1.0
    return 1 + (k / d - 1 / (k * d)) / (3 * (k - 1))


def size(k, n, rng):
    """명목 5% 에서의 실제 거짓 양성률 — 보정 전과 후."""
    d, c = n - 1, stats.chi2.ppf(0.95, k - 1)
    C = Cfac(k, n)
    hit, done = np.zeros(2), 0
    while done < REPS:
        m = min(CHUNK, REPS - done)
        x = rng.normal(size=(m, k, n))
        s2 = x.var(axis=2, ddof=1)
        sp2 = (d * s2).sum(axis=1) / (k * d)
        M = (k * d) * np.log(sp2) - (d * np.log(s2)).sum(axis=1)
        hit += [(M > c).sum(), (M / C > c).sum()]
        done += m
    return hit / REPS


def main():
    rng = np.random.default_rng(SEED)

    # ── 닫힌 꼴을 모의로 한 번 맞춰 본다 ──
    k, n = K_MAIN, 5
    d = n - 1
    x = rng.normal(size=(400_000, k, n))
    s2 = x.var(axis=2, ddof=1)
    sp2 = (d * s2).sum(axis=1) / (k * d)
    M = (k * d) * np.log(sp2) - (d * np.log(s2)).sum(axis=1)
    print(f"k={k}, n={n}:  닫힌 꼴 E[M] = {EM(k,n):.4f}   "
          f"모의 = {M.mean():.4f} ± {M.std(ddof=1)/np.sqrt(len(M)):.4f}")

    ns = np.unique(np.round(np.logspace(np.log10(3), 3, 80)).astype(int))
    before = np.abs(EM(K_MAIN, ns) / (K_MAIN - 1) - 1)
    after = np.abs(EM(K_MAIN, ns) / Cfac(K_MAIN, ns) / (K_MAIN - 1) - 1)
    fit = np.polyfit(np.log(ns[ns >= 40]), np.log(after[ns >= 40]), 1)[0]
    fit0 = np.polyfit(np.log(ns[ns >= 40]), np.log(before[ns >= 40]), 1)[0]
    print(f"기울기 — 보정 전 {fit0:.3f}, 보정 후 {fit:.3f}")

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("\n  k    n   보정 전   보정 후")
    sizes = {}
    for k in (K_MAIN, 10):
        sizes[k] = np.array([size(k, n, rng) for n in NS_SIZE])
        for n, (a, b) in zip(NS_SIZE, sizes[k]):
            print(f"{k:>3}{n:>5}{a:>9.4f}{b:>9.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── ① 차수 ──
    ax = axes[0]
    ax.loglog(ns, before, lw=2.9, color=AMBER, label="보정 전 — $M$")
    ax.loglog(ns, after, lw=2.9, color=TEAL, label="보정 후 — $M/C$")
    ax.loglog(ns, 0.4 / ns, lw=1.4, ls="--", color=MUTED)
    ax.loglog(ns, 1.2 / ns ** 3.0, lw=1.4, ls="--", color=MUTED)
    ax.text(300, 0.4 / 300 * 1.7, "기울기 $-1$", fontsize=10.6, color=INK)
    ax.text(26, 1.2 / 26 ** 3 * 2.4, "기울기 $-3$", fontsize=10.6, color=INK)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"집단당 관측값 수 $n$  (집단 ${K_MAIN}$ 개)", fontsize=11.5)
    ax.set_ylabel("기댓값의 상대오차", fontsize=11.5)
    ax.set_title("① 나눗수 하나가 차수를 바꾼다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10.5, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 실제 크기 ──
    ax = axes[1]
    for k, ls, mk in [(K_MAIN, "-", "o"), (10, "--", "s")]:
        ax.plot(NS_SIZE, sizes[k][:, 0] * 100, ls=ls, marker=mk, ms=7, lw=2.5,
                color=AMBER, label=f"보정 전 — 집단 ${k}$ 개")
        ax.plot(NS_SIZE, sizes[k][:, 1] * 100, ls=ls, marker=mk, ms=7, lw=2.5,
                color=TEAL, label=f"보정 후 — 집단 ${k}$ 개")
    ax.axhline(5, color=MUTED, lw=1.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(NS_SIZE)
    ax.set_xticklabels([str(v) for v in NS_SIZE], fontsize=10.5)
    ax.annotate(f"집단 $10$ 개 · $n = 3$ 에서 ${sizes[10][0,0]*100:.1f}\\%$",
                xy=(3.05, sizes[10][0, 0] * 100),
                xytext=(3.35, sizes[10][0, 0] * 100 - 1.15), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.annotate(f"같은 자리에서 ${sizes[10][0,1]*100:.1f}\\%$",
                xy=(3.05, sizes[10][0, 1] * 100),
                xytext=(4.2, sizes[10][0, 1] * 100 + 0.55), fontsize=11,
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("집단당 관측값 수 $n$  (정규 · 분산은 모두 같게)", fontsize=11.5)
    ax.set_ylabel("명목 $5\\%$ 의 실제 거짓 양성률 (%)", fontsize=11.5)
    ax.set_title("② 그 차수가 작은 표본에서 하는 일", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.6, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$1937$ 년에 끼워 넣은 나눗수 $C$ 가 하는 일", fontsize=14.5)
    fig.text(0.5, -0.045,
             "왼쪽은 모의실험이 아니라 정규자료에서 유도한 $E[M]$ 의 닫힌 꼴이다. "
             f"오른쪽만 모의이고 각 점이 ${REPS//10000}$ 만 번이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "correction_order.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
