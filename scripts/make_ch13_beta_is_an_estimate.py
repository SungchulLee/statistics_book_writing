r"""13.1절 「그림으로 보는 통계학」 — 베타는 수가 아니라 추정값이다.

이 쪽의 $4$ 절은 시장 기준지수에 대한 개별 자산의 회귀 기울기를 **시장 베타**라
부르고, 보기 $11$ 은 그것을 $100$ 일로 적합한다. 그 기울기에 표준오차가 붙는다는
사실은 적지 않는다. 얼마나 되는지 재어 본다.

시장 수익률의 일간 표준편차를 $\sigma_m$, 고유 잡음을 $\sigma_\varepsilon$ 로 두면
단순회귀의 표준오차는

    SE(beta_hat) = sigma_eps / (sigma_m * sqrt(n))

이다. 이 스크립트는 $\sigma_m = 1.2\%$, $\sigma_\varepsilon = 1.5\%$, 참 베타
$1.20$ 으로 두었다(그 조합이 주는 회귀의 결정계수가 $0.48$ 이다).

이 스크립트가 재는 값(되풀이 4000 번):

  창 길이    모의실험 SD    식이 말하는 SE    95% 구간의 폭(+-)
    60일       0.1618         0.1614            0.3163
   100일       0.1280         0.1250            0.2450
   250일       0.0802         0.0791            0.1550
   500일       0.0560         0.0559            0.1096
  1250일       0.0352         0.0354            0.0693

  100 일 창에서 참 베타 1.20 의 추정값이 1.00 아래로 내려간 비율  0.062
  업계의 조정 베타 (1/3) + (2/3) * beta_hat 을 씌우면
    평균 1.1322 (참값보다 0.0678 낮다),  SD 0.1280 -> 0.0853

만드는 파일:

  ch13/linear_regression/img/beta_is_an_estimate.png

실행:  python3 scripts/make_ch13_beta_is_an_estimate.py   (저장소 최상위에서)
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

OUT = "docs/ch13/linear_regression/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SIG_M = 0.012        # 시장 일간 수익률의 표준편차
SIG_E = 0.015        # 고유 잡음의 표준편차
BETA = 1.20          # 참 베타
WINDOWS = (60, 100, 250, 500, 1250)
MAIN = 100
REPS = 4000
SEED = 11


def betas(n, r0):
    """창 길이 n 으로 적합한 베타 추정값 REPS 개."""
    rng = np.random.default_rng([SEED, r0])
    m = rng.normal(0, SIG_M, (REPS, n))
    y = BETA * m + rng.normal(0, SIG_E, (REPS, n))
    mc = m - m.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    return (mc * yc).sum(axis=1) / (mc ** 2).sum(axis=1)


def main():
    r2 = (BETA * SIG_M) ** 2 / ((BETA * SIG_M) ** 2 + SIG_E ** 2)
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"참 베타 {BETA}, 시장 일간 SD {100 * SIG_M:.1f}%, "
          f"고유 잡음 {100 * SIG_E:.1f}%  (회귀의 R^2 = {r2:.2f})")
    print("  창 길이   모의실험 SD   식이 말하는 SE   95% 구간의 폭(±)")
    rows = []
    for k, n in enumerate(WINDOWS):
        b = betas(n, k)
        sd = float(b.std())
        se = SIG_E / (SIG_M * np.sqrt(n))
        rows.append((n, sd, se, b))
        print(f"  {n:5d}일 {sd:12.4f} {se:15.4f} {1.96 * se:17.4f}")

    main_b = [r[3] for r in rows if r[0] == MAIN][0]
    below = float((main_b < 1.0).mean())
    adj = 1 / 3 + 2 / 3 * main_b
    print(f"\n  {MAIN} 일 창에서 참 베타 {BETA} 의 추정값이 1.00 아래로 "
          f"내려간 비율  {below:.3f}")
    print(f"  업계의 조정 베타 (1/3) + (2/3)*beta_hat 을 씌우면")
    print(f"    평균 {adj.mean():.4f} (참값보다 {BETA - adj.mean():.4f} 낮다),  "
          f"SD {main_b.std():.4f} -> {adj.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 창 길이와 표준오차 ──
    ax = axes[0]
    g = np.linspace(40, 1300, 300)
    ax.plot(g, SIG_E / (SIG_M * np.sqrt(g)), lw=2.6, color=INK,
            label="$\\sigma_\\varepsilon / (\\sigma_m \\sqrt{n})$ — 식")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o", ms=12,
            color=AMBER, mec="white", mew=1.5, label="모의실험으로 잰 흩어짐")
    ax.axvline(MAIN, color=MUTED, lw=1.2, ls=":")
    r100 = [r for r in rows if r[0] == MAIN][0]
    ax.annotate(f"보기 $11$ 의 ${MAIN}$ 일 창\nSE $= {r100[2]:.3f}$",
                xy=(MAIN, r100[2]), xytext=(230, 0.142),
                fontsize=11.4, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks(list(WINDOWS))
    ax.set_xticklabels([f"${n}$" for n in WINDOWS])
    ax.minorticks_off()
    ax.set_xlabel("적합에 쓴 날 수 $n$ (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("$\\hat\\beta$ 의 표준오차", fontsize=11.5)
    ax.set_ylim(0, 0.185)
    ax.set_title("① 창을 네 배 늘려야 폭이 절반이 된다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 100 일 창의 베타와 조정 베타 ──
    ax = axes[1]
    bins = np.linspace(0.7, 1.75, 70)
    ax.hist(main_b, bins=bins, color=MUTED, alpha=0.85,
            label=f"${MAIN}$ 일로 잰 $\\hat\\beta$  (SD ${main_b.std():.3f}$)")
    ax.hist(adj, bins=bins, color=TEAL, alpha=0.75,
            label=f"조정 베타 $\\frac{{1}}{{3}} + \\frac{{2}}{{3}}\\hat\\beta$  "
                  f"(SD ${adj.std():.3f}$)")
    ax.axvline(BETA, color=RED, lw=2.4)
    top = ax.get_ylim()[1] * 1.22
    ax.set_ylim(0, top)
    ax.text(BETA + 0.015, top * 0.60, f"참 베타 ${BETA}$", fontsize=11.4,
            color=RED, fontweight="bold")
    ax.axvline(1.0, color=INK, lw=1.4, ls="--")
    ax.text(0.985, top * 0.60, "시장 $1.0$", fontsize=11.2, color=INK,
            ha="right")
    ax.annotate(f"$\\hat\\beta$ 가 $1.0$ 아래로 내려간 때가 ${100 * below:.1f}\\%$",
                xy=(0.95, top * 0.08), xytext=(0.72, top * 0.46),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("$\\hat\\beta$", fontsize=11.5)
    ax.set_ylabel("되풀이 횟수", fontsize=11.5)
    ax.set_title("② 같은 참 베타에서 나온 값들", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=9.8, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("한 수로 적히는 기울기에 붙어 있는 폭", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"시장 일간 표준편차 ${100 * SIG_M:.1f}\\%$, 고유 잡음 "
             f"${100 * SIG_E:.1f}\\%$, 참 베타 ${BETA}$ 로 둔 모의실험이다"
             f"(그 조합의 $R^2$ 이 ${r2:.2f}$ 다). 실제 시장 자료가 아니며, "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "beta_is_an_estimate.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
