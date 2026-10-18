r"""3.3절 「AI로 보는 통계학 1」 — 곱이 벼랑에서 떨어지는 자리.

정리 1(독립이면 곱이 곱으로 넘어간다)을 글자 그대로 계산하면 가능도는
밀도의 곱이다. 밀도값이 1 보다 작으므로 곱할수록 0 에 가까워지고, 어느
지점에서 부동소수점이 바닥에 닿아 **0 으로 주저앉는다.**

같은 양을 로그로 더하면 그 벼랑이 없다. 두 길을 한 그림에 겹쳐 그려,
교과서가 로그가능도를 먼저 꺼내는 까닭이 미분의 편의만이 아님을 보인다.

  파란 실선  곱을 그대로 계산한 것 — n = 234 에서 끊긴다
  주황 실선  로그로 더한 것      — 같은 직선을 그대로 이어 간다
  회색 점선  float64 가 표현할 수 있는 가장 작은 수

만드는 파일:

  ch03/rv/img/likelihood_underflow.png

실행:  python3 scripts/make_ch03_likelihood_underflow.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/rv/img/"

INK = "#37474F"
BLUE = "#1565C0"
AMBER = "#E65100"
MUTED = "#90A4AE"
RED = "#D32F2F"

MU, SIGMA, NMAX = 170.0, 6.0, 400
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    x = rng.normal(MU, SIGMA, NMAX)

    prod = np.cumprod(stats.norm.pdf(x, MU, SIGMA))      # 곱을 그대로
    logsum = np.cumsum(stats.norm.logpdf(x, MU, SIGMA))  # 로그로 더해서
    n = np.arange(1, NMAX + 1)

    with np.errstate(divide="ignore"):
        log10_prod = np.log10(prod)                      # 0 이 되면 -inf
    log10_sum = logsum / np.log(10)                      # 같은 눈금으로
    cliff = int(np.argmax(prod == 0.0)) + 1              # 곱이 0 이 되는 첫 n
    tiny = np.log10(np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(9.4, 5.2), constrained_layout=True)

    ax.axhline(tiny, color=MUTED, lw=1.5, ls="--")
    ax.text(NMAX, tiny + 9, "float64 가 담을 수 있는 가장 작은 수",
            ha="right", fontsize=10, color=MUTED)

    ax.plot(n, log10_sum, lw=3.0, color=AMBER, zorder=3,
            label="로그로 더한 것 — 끊기지 않는다")
    good = np.isfinite(log10_prod)
    ax.plot(n[good], log10_prod[good], lw=3.0, color=BLUE, zorder=4,
            label="곱을 그대로 계산한 것")

    ax.plot([cliff], [log10_sum[cliff - 1]], marker="v", ms=13, color=RED,
            zorder=6, clip_on=False)
    ax.annotate(f"n = {cliff} 에서 곱이 0 이 된다\n여기서부터 최대가능도는 길을 잃는다",
                xy=(cliff, log10_sum[cliff - 1]),
                xytext=(cliff + 26, log10_sum[cliff - 1] + 135),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

    ax.set_xlabel("관측값의 개수 n", fontsize=12)
    ax.set_ylabel("가능도의 크기 (10의 거듭제곱으로)", fontsize=12)
    ax.set_title("정리 1 을 글자 그대로 계산하면 234 개에서 벼랑을 만난다",
                 fontsize=13.5, color=INK, pad=10)
    ax.set_xlim(0, NMAX)
    ax.set_ylim(-600, 20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, loc="lower left", framealpha=0.92)

    path = OUT + "likelihood_underflow.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    곱이 0 이 되는 첫 n = {cliff}")
    print(f"    그 직전 곱 = {prod[cliff - 2]:.3e}")
    print(f"    n = {NMAX} 에서 로그합 = {logsum[-1]:.1f}  (10의 {log10_sum[-1]:.0f} 제곱)")


if __name__ == "__main__":
    main()
