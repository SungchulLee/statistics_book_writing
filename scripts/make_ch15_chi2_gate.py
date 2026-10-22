r"""15.2절 「AI로 보는 통계학」 — 마할라노비스 게이트의 문턱.

$x \sim N_p(\mu, \Sigma)$ 이고 $\mu, \Sigma$ 를 **안다면** 마할라노비스 거리

    D^2 = (x - mu)^T Sigma^{-1} (x - mu)

는 백색화하면 표준정규 $p$ 개의 제곱합이므로 정확히 $\chi^2(p)$ 다(정의 1).
그래서 「$99\%$ 게이트」는 `chi2.ppf(0.99, p)` 한 줄이 된다.

문제는 실제로는 $\mu, \Sigma$ 를 학습자료 $n$ 개로 **추정**한다는 것이다. 새 점
하나에 대해서는

    D^2 ~ [p(n^2-1) / (n(n-p))] * F(p, n-p)

가 되고(호텔링 $T^2$), 이것은 $n$ 이 $p$ 보다 압도적으로 크지 않으면
$\chi^2(p)$ 와 꽤 다르다. 카이제곱 문턱을 그대로 쓰면 경보가 명목보다 자주 울린다.

  왼쪽  — 명목 $1\%$ 로 맞춰 둔 게이트의 실제 경보율 (p = 2, 5, 20, 50)
  오른쪽 — p = 20 에서 카이제곱 문턱과 참 문턱의 벌어짐

만드는 파일:

  ch15/chi_square_test/img/chi2_gate.png

실행:  python3 scripts/make_ch15_chi2_gate.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/chi_square_test/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

ALPHA = 0.01
PS = [2, 5, 20, 50]
SEED = 15
MC_REPS = 20_000


def scale(n, p):
    """D^2 = scale * F(p, n-p) — 추정한 평균·공분산으로 잰 새 점의 거리."""
    return p * (n ** 2 - 1) / (n * (n - p))


def rate(n, p, alpha=ALPHA):
    """카이제곱 문턱을 쓸 때의 실제 경보율."""
    return stats.f.sf(stats.chi2.ppf(1 - alpha, p) / scale(n, p), p, n - p)


def mc_rate(n, p, rng, reps=MC_REPS, alpha=ALPHA):
    """같은 것을 모의로 — 학습 n 개로 잡고 새 점 하나를 잰다."""
    c = stats.chi2.ppf(1 - alpha, p)
    hit = 0
    for _ in range(reps):
        tr = rng.normal(size=(n, p))
        v = rng.normal(size=p) - tr.mean(axis=0)
        hit += v @ np.linalg.solve(np.cov(tr, rowvar=False), v) > c
    return hit / reps


def main():
    rng = np.random.default_rng(SEED)

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("   n    p   카이제곱 문턱   참 문턱   실제 경보율     모의")
    for n, p in [(200, 20), (100, 20), (1000, 20), (200, 50)]:
        c = stats.chi2.ppf(1 - ALPHA, p)
        print(f"{n:5d} {p:4d} {c:12.2f} {scale(n,p)*stats.f.ppf(1-ALPHA,p,n-p):10.2f}"
              f" {rate(n,p):12.4f} {mc_rate(n,p,rng):9.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── ① 실제 경보율 ──
    ax = axes[0]
    for p, col, mk in zip(PS, [BLUE, TEAL, AMBER, RED], ["o", "s", "D", "^"]):
        ns = np.unique(np.round(np.logspace(np.log10(p * 1.6), 4, 60)).astype(int))
        ax.plot(ns, [rate(n, p) * 100 for n in ns], lw=2.6, color=col,
                label=f"차원 $p = {p}$")
        for n0 in (200,):
            if n0 > p:
                ax.plot([n0], [rate(n0, p) * 100], marker=mk, ms=8.5, color=col)
    ax.axhline(ALPHA * 100, color=MUTED, lw=1.8, ls="--")
    ax.annotate("맞춰 둔 값 $1\\%$", xy=(3000, 1), xytext=(900, 1.7),
                fontsize=10.8, color=INK)
    ax.annotate(f"$n = 200$ · $p = 20$ 에서 ${rate(200,20)*100:.1f}\\%$",
                xy=(200, rate(200, 20) * 100), xytext=(330, 7.5),
                fontsize=11, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("평균과 공분산을 잡는 데 쓴 학습 표본 수 $n$", fontsize=11.5)
    ax.set_ylabel("실제 거짓 경보율 (%)", fontsize=11.5)
    ax.set_title("① 카이제곱 문턱을 그대로 쓰면", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 두 문턱 ──
    ax = axes[1]
    p = 20
    ns = np.unique(np.round(np.logspace(np.log10(26), 4, 120)).astype(int))
    c = stats.chi2.ppf(1 - ALPHA, p)
    true_c = np.array([scale(n, p) * stats.f.ppf(1 - ALPHA, p, n - p) for n in ns])
    ax.fill_between(ns, c, true_c, color=AMBER, alpha=0.18)
    ax.plot(ns, true_c, lw=2.8, color=AMBER, label="참 문턱 — $F$ 에서")
    ax.axhline(c, color=TEAL, lw=2.8, label=f"카이제곱 문턱 ${c:.2f}$")
    ax.set_xscale("log")
    ax.set_ylim(c - 8, 160)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for n0 in (50, 200, 1000):
        ax.annotate(f"$n={n0}$ 에서 ${scale(n0,p)*stats.f.ppf(1-ALPHA,p,n0-p):.1f}$",
                    xy=(n0, scale(n0, p) * stats.f.ppf(1 - ALPHA, p, n0 - p)),
                    xytext=(n0 * 1.25, scale(n0, p) * stats.f.ppf(1 - ALPHA, p, n0 - p) + 11),
                    fontsize=10.6, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
    ax.set_xlabel("학습 표본 수 $n$  (차원 $p = 20$)", fontsize=11.5)
    ax.set_ylabel("$99\\%$ 게이트의 문턱값", fontsize=11.5)
    ax.set_title("② 그 문턱이 있어야 할 자리", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$\\Sigma$ 를 추정하는 순간 문턱은 카이제곱이 아니다", fontsize=14.5)
    fig.text(0.5, -0.045,
             "두 칸 모두 자료가 정확히 정규일 때의 값이다. 곡선은 모의실험이 아니라 "
             "$D^2 \\sim \\frac{p(n^2-1)}{n(n-p)}F(p,\\,n-p)$ 에서 계산한 것이고, "
             "표의 「모의」 열로 맞춰 보았다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "chi2_gate.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
