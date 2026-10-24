r"""11.2절 「그림으로 보는 통계학」 — 아래 끝이 $0$ 에 닿은 구간.

카드와 크루거가 $1994$ 년 『American Economic Review』에 실은 표 $4$ 의 첫 줄이
**뉴저지 더미 $2.33$ $(1.19)$** 다. 가게 $357$ 곳, 종속변수는 FTE 고용의 변화이고
그 표준편차가 $8.825$ 라고 같은 표의 주석이 적는다.

그 둘로 이 쪽 $1$ 절의 식을 그대로 돌려 볼 수 있다. 설명변수가 $0/1$ 더미이면

    sum (x_i - xbar)^2 = n p (1 - p)        p = 처치군의 비율

이므로 $\mathrm{SE} = s/\sqrt{n\,p(1-p)}$ 다. 뉴저지 가게가 $331/410 = 0.807$ 이니

    SE = 8.825 / sqrt(357 * 0.807 * 0.193) = 1.18

으로 논문이 적은 $1.19$ 와 맞는다. **폭이 어디서 왔는지가 이것으로 드러난다** —
표본이 작아서가 아니라 **한쪽이 작아서**다. 반반이었다면 $0.93$ 으로 $21\%$ 좁고,
$80{:}20$ 인 채로 같은 폭을 얻으려면 가게가 $573$ 곳 필요하다.

그리는 것:

  ① 보고된 두 계수와 그 $95\%$ 구간. 아래 끝이 $0$ 에 닿는다
  ② 같은 $n$, 같은 $s$ 에서 처치군 비율만 바꿨을 때의 표준오차. 논문의 자리와
     반반인 자리를 찍는다

만드는 파일:

  ch11/estimation_inference/img/famous_coefficient.png

실행:  python3 scripts/make_ch13_famous_coefficient.py   (저장소 최상위에서)
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

OUT = "docs/ch11/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 논문 표 4 — 가게 357 곳, 종속변수(FTE 변화)의 표준편차 8.825
N = 357
S = 8.825
P_NJ = 331 / 410          # 1 차 조사에서 뉴저지 331 곳, 펜실베이니아 79 곳
EST = [("더미만 넣은 모형", 2.33, 1.19),
       ("체인·소유 형태를 통제", 2.30, 1.20)]


def se_of_dummy(p, n=N, s=S):
    """0/1 더미의 표준오차 — sum (x-xbar)^2 = n p (1-p) 이므로."""
    return s / np.sqrt(n * p * (1.0 - p))


def main():
    tcrit = stats.t.ppf(0.975, N - 2)
    print(f"t_{{{N - 2}}}(0.975) = {tcrit:.4f}")
    for name, b, se in EST:
        lo, hi = b - tcrit * se, b + tcrit * se
        print(f"  {name:22s} {b:5.2f} ({se:.2f})   95% 구간 ({lo:+.3f}, {hi:+.3f})")

    se_hat = se_of_dummy(P_NJ)
    se_bal = se_of_dummy(0.5)
    n_needed = N * (se_hat / se_bal) ** 2
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"\n뉴저지 비율 p = {P_NJ:.3f}")
    print(f"  식으로 구한 SE = {se_hat:.3f}   논문이 적은 SE = 1.19")
    print(f"  반반(p = 0.5)이면 SE = {se_bal:.3f}  "
          f"({100 * (1 - se_bal / se_hat):.0f}% 좁다)")
    print(f"  p = {P_NJ:.3f} 인 채로 같은 폭을 얻으려면 가게 {n_needed:.0f} 곳")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)

    # ── ① 보고된 계수와 그 구간 ──
    ax = axes[0]
    ax.axvline(0, color=INK, lw=2.0, ls="--")
    for i, (name, b, se) in enumerate(EST):
        y = len(EST) - 1 - i
        lo, hi = b - tcrit * se, b + tcrit * se
        ax.plot([lo, hi], [y, y], lw=7, color=TEAL, alpha=0.75,
                solid_capstyle="round")
        ax.plot([b], [y], "D", ms=12, color=TEAL, mec="white", mew=1.4)
        ax.text(b, y + 0.17, f"${b:.2f}$", ha="center", fontsize=12,
                color=TEAL, fontweight="bold")
        ax.text(lo - 0.12, y, f"${lo:.2f}$", ha="right", va="center",
                fontsize=10.6, color=INK)
        ax.text(hi + 0.12, y, f"${hi:.2f}$", ha="left", va="center",
                fontsize=10.6, color=INK)
        ax.text(-5.3, y, name, ha="left", va="center", fontsize=11.2, color=INK)
    ax.annotate("고전 이론이 가리킨 쪽", xy=(-1.6, 1.62), xytext=(-0.25, 1.62),
                ha="left", va="center", fontsize=11.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.text(2.33, -0.52, "구간의 아래 끝이 $0$ 에 닿는다", ha="center",
            fontsize=11.6, color=AMBER, fontweight="bold")
    ax.set_xlim(-5.4, 6.2)
    ax.set_ylim(-0.95, 2.05)
    ax.set_yticks([])
    ax.set_xlabel("FTE 고용 변화의 뉴저지 $-$ 펜실베이니아 차 (가게당 사람 수)",
                  fontsize=11.5)
    ax.set_title("① 그 계수와 $95\\%$ 구간", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── ② 폭은 어디서 왔는가 ──
    ax = axes[1]
    ps = np.linspace(0.04, 0.96, 400)
    ax.plot(ps, se_of_dummy(ps), color=INK, lw=2.8)
    ax.plot([P_NJ], [se_hat], "o", ms=13, color=RED, mec="white", mew=1.5,
            zorder=5)
    ax.plot([0.5], [se_bal], "o", ms=13, color=BLUE, mec="white", mew=1.5,
            zorder=5)
    ax.annotate(f"논문의 자리 — 뉴저지가 ${100 * P_NJ:.0f}\\%$\n"
                f"$\\mathrm{{SE}} = {se_hat:.2f}$  (논문이 적은 값 $1.19$)",
                xy=(P_NJ, se_hat), xytext=(0.47, 1.62),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate(f"반반으로 모았다면\n$\\mathrm{{SE}} = {se_bal:.2f}$ — "
                f"구간이 ${100 * (1 - se_bal / se_hat):.0f}\\%$ 좁다",
                xy=(0.5, se_bal), xytext=(0.13, 1.30),
                fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.78, 2.3)
    ax.set_xlabel("처치군의 비율 $p$  (여기서는 뉴저지 가게의 몫)", fontsize=11.5)
    ax.set_ylabel("$\\mathrm{SE}(\\hat\\beta)$", fontsize=11.5)
    ax.set_title("② 같은 가게 수, 다른 배분", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("표본이 작아서가 아니라 한쪽이 작아서 넓다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"두 판 모두 논문이 적은 세 수 — 계수 $2.33$ 과 $2.30$, 그 표준오차, "
             f"가게 ${N}$ 곳과 종속변수의 표준편차 ${S}$ — 에서 나온 것이다. "
             "오른쪽 곡선은 이 쪽 $1$ 절의 식을 더미에 쓴 것뿐이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "famous_coefficient.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
