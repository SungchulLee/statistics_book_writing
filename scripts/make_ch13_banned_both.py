r"""13.3절 「그림으로 보는 통계학」 — 둘 다 금지한 학술지.

$3$ 절의 쪽지는 「$p$ 값과 신뢰구간을 모두 보고하라」고 한다. $2015$ 년에 한 학술지가
정확히 반대를 했다 — **둘 다 싣지 못하게 했다.**

그러면 무엇이 남는가. 계수 추정값과 기술통계다. 그런데 **같은 추정값이 전혀 다른 두
연구에서 나올 수 있다.** 오차의 표준편차가 $\sigma = 6$ 이고 설명변수의 표준편차가
$1$ 인 같은 설계에서, 두 연구가 모두 $\hat\beta = 2.40$ 을 보고했다고 하자.

이 스크립트가 재는 값:

  연구     n       SE       95% 신뢰구간         p 값     0 이 닿는 신뢰수준
   가     20   1.3416  ( -0.419,  5.219)      0.0905         90.95%
   나   2000   0.1342  (  2.137,  2.663)    1.69e-66        100.00%

**기술통계만 적으면 두 연구의 결과 문장이 글자 그대로 같다.** 하나는 효과의 부호조차
정하지 못했고 다른 하나는 소수 둘째 자리까지 못 박았는데도 그렇다.

만드는 파일:

  ch13/testing_coefficients/img/banned_both.png

실행:  python3 scripts/make_ch13_banned_both.py   (저장소 최상위에서)
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

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/testing_coefficients/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

BETA = 2.40        # 두 연구가 똑같이 보고한 추정값
SIGMA = 6.0        # 오차의 표준편차
SD_X = 1.0         # 설명변수의 표준편차
STUDIES = (("가", 20), ("나", 2000))


def ptex(p):
    """mathtext 가 1.7e-66 의 e 를 기울여 찍으므로 10 의 거듭제곱으로 적는다."""
    if p >= 1e-4:
        return f"{p:.4f}"
    e = int(np.floor(np.log10(p)))
    return f"{p / 10 ** e:.1f} \\times 10^{{{e}}}"


def summarize(n):
    """표준오차, 95% 구간, p 값, 그리고 0 이 끝점이 되는 신뢰수준."""
    se = SIGMA / (SD_X * np.sqrt(n))
    d = n - 2
    tc = stats.t.ppf(0.975, d)
    lo, hi = BETA - tc * se, BETA + tc * se
    p = 2 * stats.t.sf(abs(BETA / se), d)
    return se, lo, hi, float(p), 100 * (1 - p)


def main():
    rows = {}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"두 연구가 똑같이 보고한 추정값 {BETA}  (sigma = {SIGMA:g})")
    print("  연구     n       SE       95% 신뢰구간         p 값     0 이 닿는 신뢰수준")
    for nm, n in STUDIES:
        se, lo, hi, p, lvl = summarize(n)
        rows[nm] = (n, se, lo, hi, p, lvl)
        print(f"   {nm}  {n:5d}  {se:7.4f}  ({lo:7.3f}, {hi:6.3f})  "
              f"{p:10.3g}       {lvl:7.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), constrained_layout=True)

    # ── ① 같은 점, 다른 구간 ──
    ax = axes[0]
    ys = {"가": 1.0, "나": 0.0}
    for nm, col in (("가", RED), ("나", TEAL)):
        n, se, lo, hi, p, lvl = rows[nm]
        y = ys[nm]
        ax.plot([lo, hi], [y, y], lw=5.0, color=col, alpha=0.85, solid_capstyle="butt")
        ax.plot([lo, lo], [y - 0.09, y + 0.09], lw=3.0, color=col)
        ax.plot([hi, hi], [y - 0.09, y + 0.09], lw=3.0, color=col)
        ax.plot([BETA], [y], "o", ms=13, color=INK, mec="white", mew=1.6,
                zorder=5)
        ax.text(hi + 0.18, y, f"$({lo:.2f},\\ {hi:.2f})$", va="center",
                fontsize=11.4, color=col, fontweight="bold")
        ax.text(-2.55, y + 0.26, f"연구 {nm} — 관측 ${n}$ 개", fontsize=11.6,
                color=col, fontweight="bold")
        ax.text(-2.55, y - 0.26, f"$p = {ptex(p)}$", fontsize=11, color=col)
    ax.axvline(0, color=MUTED, lw=1.8, ls="--")
    ax.text(0.08, 1.60, "$0$", fontsize=11.5, color=MUTED)
    ax.annotate(f"두 연구가 보고한 추정값은 똑같이 ${BETA}$ 다",
                xy=(BETA, 0.5), xytext=(2.9, 1.5), fontsize=11.4, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xlim(-2.7, 7.6)
    ax.set_ylim(-0.55, 1.85)
    ax.set_yticks([])
    ax.set_xlabel("계수", fontsize=11.5)
    ax.set_title("① 기술통계만 적으면 두 줄이 같은 줄이 된다", fontsize=12.6,
                 color=INK)
    ax.grid(alpha=0.22, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── ② 추정값은 가로축을 말해 주지 않는다 ──
    ax = axes[1]
    ns = np.logspace(np.log10(8), np.log10(5000), 200)
    half = stats.t.ppf(0.975, ns - 2) * SIGMA / (SD_X * np.sqrt(ns))
    ax.plot(ns, half, lw=2.8, color=INK, label="$95\\%$ 구간의 반폭")
    ax.axhline(BETA, color=MUTED, lw=1.6, ls=":")
    ax.text(4600, BETA * 1.12, f"반폭이 ${BETA}$ 를 넘으면 구간이 $0$ 을 품는다",
            ha="right", fontsize=10.6, color=MUTED)
    for nm, col in (("가", RED), ("나", TEAL)):
        n, se, lo, hi, p, lvl = rows[nm]
        h = (hi - lo) / 2
        ax.plot([n], [h], "o", ms=13, color=col, mec="white", mew=1.6,
                zorder=5)
        ax.annotate(f"연구 {nm}\n반폭 ${h:.2f}$", xy=(n, h),
                    xytext=(n * (2.4 if nm == "가" else 0.12),
                            h * (0.35 if nm == "가" else 5.0)),
                    fontsize=11.2, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("관측 수 $n$ (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("$95\\%$ 구간의 반폭 (로그 눈금)", fontsize=11.5)
    ax.set_title("② 추정값은 이 가로축을 말해 주지 않는다", fontsize=12.6,
                 color=INK)
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$p$ 값도 구간도 싣지 않으면", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"오차의 표준편차 $\\sigma = {SIGMA:g}$, 설명변수의 표준편차 "
             f"${SD_X:g}$ 인 같은 설계에서 두 연구가 모두 "
             f"$\\hat\\beta = {BETA}$ 를 보고했다고 둔 것이다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "banned_both.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
