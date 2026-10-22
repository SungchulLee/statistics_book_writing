r"""15.6절 「AI로 보는 통계학」 — 라이브러리 안에 들어 있는 역감마 사전분포.

`sklearn.linear_model.BayesianRidge` 의 `alpha_1`, `alpha_2` 는 **잡음 정밀도**
$\alpha = 1/\sigma^2$ 에 대한 감마 사전분포의 모양·비율 모수다. 정밀도의 감마는
분산의 **역감마**이므로, 그 둘은 이 쪽 절 $1$ 의 $(\alpha_0, \beta_0)$ 바로 그것이다.
기본값은 $10^{-6}$ 으로 사실상 무정보다.

그런데 적합이 끝나면 라이브러리가 내놓는 것은 `alpha_` 하나뿐이다. 사전분포를
적어 놓고도 **사후분포는 돌려주지 않는다.**

  왼쪽  — 같은 사전분포·같은 자료에서 나오는 사후분포와, 라이브러리가 내놓은 한 수
  오른쪽 — 자료를 200 번 새로 뽑았을 때 그 한 수가 흔들리는 폭

두 폭이 겹친다는 것이 요점이다. 사후구간은 장식이 아니라 그 점 하나가 다음 자료에서
어디로 갈지를 미리 그려 놓은 것이다.

만드는 파일:

  ch15/advanced_methods/img/noise_precision_prior.png

실행:  python3 scripts/make_ch15_noise_precision_prior.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/advanced_methods/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 15
N, P, SIGMA2 = 50, 5, 4.0
BETA = np.array([2.0, -1.5, 1.0, 0.0, 0.5])
ROUNDS = 200


def one(rng):
    """자료 한 벌 — 라이브러리의 한 수와 최소제곱 잔차제곱합."""
    X = rng.normal(size=(N, P))
    y = X @ BETA + rng.normal(0, np.sqrt(SIGMA2), N)
    br = BayesianRidge().fit(X, y)
    Xd = np.c_[np.ones(N), X]
    r = y - Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]
    return 1.0 / br.alpha_, float(r @ r)


def main():
    rng = np.random.default_rng(SEED)
    lib, rss = one(rng)                       # 쪽에 싣는 자료 한 벌
    df = N - P - 1
    a0 = b0 = 1e-6
    an, bn = a0 + df / 2, b0 + rss / 2
    lo, hi = stats.invgamma.ppf([0.025, 0.975], a=an, scale=bn)
    post_mean = bn / (an - 1)
    print(f"1/alpha_ = {lib:.3f}   RSS/(n-p-1) = {rss/df:.3f}   "
          f"사후평균 = {post_mean:.3f}   95% 신용구간 = ({lo:.3f}, {hi:.3f})")

    reps = np.array([one(rng)[0] for _ in range(ROUNDS)])
    print(f"자료를 {ROUNDS} 번 다시 뽑았을 때 1/alpha_ 의 "
          f"2.5%–97.5% = ({np.quantile(reps,0.025):.3f}, "
          f"{np.quantile(reps,0.975):.3f}), 평균 {reps.mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True,
                             sharex=True)
    xs = np.linspace(1.5, 9.5, 600)
    dens = stats.invgamma.pdf(xs, a=an, scale=bn)

    # ── ① 사후분포와 한 수 ──
    ax = axes[0]
    ax.plot(xs, dens, lw=2.8, color=TEAL)
    m = (xs >= lo) & (xs <= hi)
    ax.fill_between(xs[m], dens[m], color=TEAL, alpha=0.18)
    ax.axvline(lib, color=AMBER, lw=3.0)
    ax.axvline(SIGMA2, color=MUTED, lw=1.8, ls=":")
    ax.annotate(f"라이브러리가 내놓는 한 수\n$1/\\mathtt{{alpha}}\\_ = {lib:.2f}$",
                xy=(lib, dens.max() * 0.62), xytext=(5.6, dens.max() * 0.80),
                fontsize=11, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.annotate(f"$95\\%$ 신용구간\n$({lo:.2f},\\ {hi:.2f})$",
                xy=(hi * 0.97, dens.max() * 0.08), xytext=(6.4, dens.max() * 0.30),
                fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.annotate(f"참값 ${SIGMA2:.0f}$", xy=(SIGMA2, dens.max() * 1.02),
                xytext=(SIGMA2 - 1.75, dens.max() * 1.03), fontsize=10.8, color=INK)
    ax.set_xlabel("잡음 분산 $\\sigma^2$", fontsize=11.5)
    ax.set_ylabel("사후밀도", fontsize=11.5)
    ax.set_title("① 같은 사전분포가 말해 주는 것", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 그 한 수가 흔들리는 폭 ──
    ax = axes[1]
    ax.hist(reps, bins=26, color=AMBER, alpha=0.55, edgecolor="white")
    q = np.quantile(reps, [0.025, 0.975])
    ax.axvline(SIGMA2, color=MUTED, lw=1.8, ls=":")
    for v, c in [(lo, TEAL), (hi, TEAL)]:
        ax.axvline(v, color=c, lw=2.2, ls="--")
    ax.annotate(f"왼쪽 칸의 신용구간을\n그대로 옮겨 그린 두 선",
                xy=(hi, ax.get_ylim()[1] * 0.72), xytext=(6.6, ax.get_ylim()[1] * 0.74),
                fontsize=10.8, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("자료를 다시 뽑을 때마다의 $1/\\mathtt{alpha}\\_$", fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title(f"② 자료를 {ROUNDS} 번 새로 뽑으면 "
                 f"(${q[0]:.2f}$–${q[1]:.2f}$ 에 $95\\%$)",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("점 하나로 내놓지만, 점 하나로 알 수 있는 값이 아니다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             f"$n = {N}$ · 설명변수 ${P}$ 개 · 참 분산 ${SIGMA2:.0f}$ 인 자료. "
             "왼쪽 곡선은 모의실험이 아니라 역감마 사후분포의 밀도이고, "
             "오른쪽 막대만 모의다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "noise_precision_prior.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
