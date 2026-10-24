r"""11.2절 「AI로 보는 통계학」 — 실험 전 기간을 열 하나로 넣으면.

정리 1 은 $\operatorname{Var}(\hat{\boldsymbol\beta}) = \sigma^2(X^\top X)^{-1}$ 이라고
적는다. 이 식에서 줄일 수 있는 것은 $\sigma^2$ 과 $X^\top X$ 둘뿐이고, 온라인 실험은
그 가운데 **앞의 것**을 줄인다.

처치 지시변수 $T$ 하나만 넣은 모형과, 거기에 **실험 전 기간의 같은 지표** $X$ 를
열 하나로 더 넣은 모형을 견준다. 무작위 배정이므로 $T \perp X$ 이고 처치효과의
추정값은 양쪽 다 불편이다. 달라지는 것은 잔차분산뿐이다.

$$
\sigma^2_{Y \mid X} = \sigma^2_Y (1 - \rho^2)
\quad\Longrightarrow\quad
\frac{\mathrm{SE}_{\text{보정}}}{\mathrm{SE}_{\text{그냥}}} = \sqrt{1 - \rho^2}
$$

이 스크립트가 재는 값($n = 20{,}000$, 군마다 절반, 되풀이 $2000$ 번):

  rho   SE(그냥)  SE(보정)     비    sqrt(1-rho^2)  n 의 비   평균(보정)
  0.0    0.01394   0.01395   1.000      1.000       1.000     0.02018
  0.3    0.01391   0.01330   0.956      0.954       0.914     0.02017
  0.5    0.01389   0.01208   0.870      0.866       0.756     0.02015
  0.7    0.01386   0.00996   0.719      0.714       0.517     0.02013
  0.9    0.01383   0.00608   0.440      0.436       0.193     0.02008

곧 $\rho = 0.7$ 이면 **같은 검정력을 절반의 사람으로** 얻는다. 두 추정값의 평균은
어느 $\rho$ 에서도 참값 $0.02$ 에 그대로 있다.

만드는 파일:

  ch11/estimation_inference/img/pre_period_covariate.png

실행:  python3 scripts/make_ch13_pre_period_covariate.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 20_000          # 실험에 들어온 사람 수
TAU = 0.02          # 참 처치효과
SIGMA = 1.0         # 지표의 표준편차
RHOS = (0.0, 0.3, 0.5, 0.7, 0.9)
RHO_MAIN = 0.7
REPS = 2000
SEED = 21


def one(rho, i):
    """한 번의 실험 — 보정 없는 추정값과 보정한 추정값."""
    rng = np.random.default_rng([SEED, i])
    T = (np.arange(N) < N // 2).astype(float)      # 절반씩 무작위 배정
    rng.shuffle(T)
    X = rng.normal(0, SIGMA, N)                    # 실험 전 기간의 같은 지표
    Y = (0.5 + TAU * T + rho * X
         + np.sqrt(1 - rho ** 2) * rng.normal(0, SIGMA, N))
    A = np.column_stack([np.ones(N), T])
    B = np.column_stack([np.ones(N), T, X])
    return (np.linalg.lstsq(A, Y, rcond=None)[0][1],
            np.linalg.lstsq(B, Y, rcond=None)[0][1])


def main():
    rows = {}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"n = {N}, 참 처치효과 {TAU}, 되풀이 {REPS} 번")
    print("  rho   SE(그냥)  SE(보정)      비  sqrt(1-rho^2)   n 의 비   평균(보정)")
    for rho in RHOS:
        r = np.array([one(rho, i) for i in range(REPS)])
        rows[rho] = r
        sa, sb = r[:, 0].std(), r[:, 1].std()
        print(f"  {rho:3.1f} {sa:10.5f} {sb:9.5f} {sb / sa:8.3f} "
              f"{np.sqrt(1 - rho ** 2):12.3f} {(sb / sa) ** 2:10.3f} "
              f"{r[:, 1].mean():11.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    # ── ① 축소율은 sqrt(1 - rho^2) 다 ──
    ax = axes[0]
    g = np.linspace(0, 0.98, 200)
    ax.plot(g, np.sqrt(1 - g ** 2), lw=2.6, color=INK,
            label="$\\sqrt{1 - \\rho^2}$ — 정리 $1$ 이 말하는 값")
    ax.plot(RHOS, [rows[r][:, 1].std() / rows[r][:, 0].std() for r in RHOS],
            "o", ms=11, color=TEAL, mec="white", mew=1.4,
            label="모의실험으로 센 표준오차의 비")
    ax.plot(g, 1 - g ** 2, "--", lw=2.2, color=AMBER,
            label="$1 - \\rho^2$ — 같은 검정력에 드는 사람 수의 비")
    ax.axvline(RHO_MAIN, color=MUTED, lw=1.2, ls=":")
    ax.annotate(f"$\\rho = {RHO_MAIN}$ 이면 사람이\n절반이면 된다",
                xy=(RHO_MAIN, 1 - RHO_MAIN ** 2), xytext=(0.17, 0.26),
                fontsize=11.2, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.set_xlabel("실험 전 기간과 실험 기간 지표의 상관 $\\rho$", fontsize=11.5)
    ax.set_ylabel("보정하지 않았을 때를 $1$ 로 둔 비", fontsize=11.5)
    ax.set_ylim(0, 1.08)
    ax.set_title("① 열 하나가 지우는 것은 $\\sigma^2$ 이다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 같은 참값, 다른 폭 ──
    ax = axes[1]
    r = rows[RHO_MAIN]
    bins = np.linspace(TAU - 0.055, TAU + 0.055, 60)
    ax.hist(r[:, 0], bins=bins, color=MUTED, alpha=0.75,
            label=f"$T$ 만 넣었다 — SE ${r[:, 0].std():.5f}$")
    ax.hist(r[:, 1], bins=bins, color=TEAL, alpha=0.75,
            label=f"$T$ 와 실험 전 $X$ — SE ${r[:, 1].std():.5f}$")
    ax.axvline(TAU, color=RED, lw=2.2)
    top = ax.get_ylim()[1] * 1.3
    ax.set_ylim(0, top)
    ax.text(TAU + 0.0017, top * 0.47, f"참값 ${TAU}$",
            fontsize=11.2, color=RED, fontweight="bold")
    ax.text(TAU - 0.052, top * 0.74,
            f"두 평균 모두 ${TAU}$ 에 있다\n"
            f"(${r[:, 0].mean():.5f}$ 대 ${r[:, 1].mean():.5f}$)",
            fontsize=10.8, color=INK)
    ax.set_xlabel("추정한 처치효과", fontsize=11.5)
    ax.set_ylabel("되풀이 횟수", fontsize=11.5)
    ax.set_title(f"② $\\rho = {RHO_MAIN}$ 에서 — 치우침은 그대로, 폭만 준다",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무작위 배정인데도 열을 하나 더 넣는 까닭", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"사람 ${N // 1000}$ 천 명을 절반씩 나눈 실험을 ${REPS}$ 번 되풀이한 "
             f"것이다. 모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "pre_period_covariate.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
