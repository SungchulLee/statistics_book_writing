r"""1.3절 「그림으로 보는 통계학」 — 36 퍼센트라는 수의 산술.

2015 년 8 월, 『사이언스』에 심리학 논문 100 편을 다시 해 본 결과가
실렸다. 연구자 270 명 남짓이 몇 해에 걸쳐 한 일이다.

  원래 논문의 97% 가 유의했다
  다시 한 것의 36% 가 유의했다
  다시 한 쪽의 효과크기가 원래의 절반쯤이었다

세 수가 따로 노는 것처럼 보이지만 하나로 이어진다.

재현 연구들은 **원래 보고된 효과크기**에 대해 90% 검정력이 되도록
표본을 잡았다. 그런데 참 효과가 보고된 것의 절반이라면 그 설계의
검정력은

  lambda = 1.96 + 1.2816 = 3.2416     (보고된 효과에 대해 90%)
  참 효과가 절반이면 lambda' = 1.6208
  검정력 = Phi(1.6208 - 1.96) + Phi(-1.6208 - 1.96) = 0.367

이다. **0.367 대 관측된 0.36.** 셋째 줄이 둘째 줄을 설명한다.

그러면 왜 발표된 효과가 두 배로 부풀어 있었는가. 유의한 것만 실리기
때문이다. 참 효과 delta 를 검정력 beta 로 재서 유의할 때만 발표하면

  검정력 0.17 -> 발표된 효과가 참값의 2.49 배
  검정력 0.25 ->                      2.00 배
  검정력 0.52 ->                      1.39 배
  검정력 0.90 ->                      1.06 배

「절반」이 되려면 원래 연구들의 검정력이 0.25 쯤이었어야 한다.

두 토막을 한 줄로 이어 거꾸로 풀면, 재현 성공률 0.36 이 나오려면
원래 문헌의 검정력이 0.240 이어야 한다. 심리학 문헌의 검정력에 대해
코언 이래로 거듭 보고되어 온 수준이 그쯤이다.

만드는 파일:

  ch01/classical/img/replication_arithmetic.png

실행:  python3 scripts/make_ch01_replication_arithmetic.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/classical/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

ZC = 1.959964
OBS_REP = 0.36                 # 관측된 재현 성공률
TARGET = 0.90                  # 재현 연구가 노린 검정력


def power(lam):
    return norm.sf(ZC - lam) + norm.cdf(-ZC - lam)


def inflation(lam):
    """참 효과 lam 을 재서 유의할 때만 발표하면 발표값이 참값의 몇 배인가."""
    a = norm.pdf(ZC - lam) - norm.pdf(-ZC - lam)
    return (lam + a / power(lam)) / lam


def main():
    lam90 = brentq(lambda L: power(L) - TARGET, 0.1, 12)
    print(f"  보고된 효과에 대해 검정력 {TARGET}: lambda = {lam90:.4f}")
    print(f"\n{'참/보고 비율':>12}{'재현 검정력':>14}")
    for r in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3):
        print(f"{r:>12.2f}{power(lam90*r):>14.3f}")
    half = power(lam90 * 0.5)
    print(f"\n  절반이면 {half:.4f} — 관측된 {OBS_REP}")

    print(f"\n{'원래 연구의 검정력':>18}{'발표된 효과 / 참값':>20}")
    for pw in (0.17, 0.25, 0.35, 0.52, 0.80, 0.90):
        lam = brentq(lambda L: power(L) - pw, 0.01, 12)
        print(f"{pw:>18.2f}{inflation(lam):>20.3f}")
    lam2 = brentq(lambda L: inflation(L) - 2.0, 0.2, 6)
    print(f"\n  발표값이 참값의 두 배가 되는 검정력 = {power(lam2):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 원래 문헌의 검정력이 얼마였어야 하는가 ──
    ax = axes[0]
    pws = np.linspace(0.06, 0.95, 400)
    lams = np.array([brentq(lambda L: power(L) - p, 0.01, 12) for p in pws])
    comp = np.array([power(lam90 / inflation(l)) for l in lams])
    beta_star = float(np.interp(OBS_REP, comp, pws))
    print(f"\n  재현 성공률 {OBS_REP} 가 나오려면 원래 문헌의 검정력이 "
          f"{beta_star:.3f} 이어야 한다")
    ax.plot(pws, comp, lw=3.8, color=INK)
    ax.axhline(OBS_REP, color=RED, lw=2.2, ls="--")
    ax.axhline(TARGET, color=TEAL, lw=1.8, ls=":")
    ax.plot([beta_star], [OBS_REP], marker="o", ms=14, color=RED, zorder=5,
            mec="white", mew=1.8)
    ax.text(0.09, TARGET + 0.022, f"재현 연구들이 노린 ${TARGET:.2f}$",
            fontsize=10.6, color=TEAL)
    ax.annotate(f"관측된 ${OBS_REP}$ 가 나오려면\n"
                f"원래 문헌의 검정력이 ${beta_star:.2f}$ 이어야 한다",
                xy=(beta_star, OBS_REP), xytext=(0.58, 0.26), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("원래 연구가 튼튼했다면\n재현도 거의 다 성공한다",
                xy=(0.88, comp[np.argmin(abs(pws - 0.88))]),
                xytext=(0.62, 0.44), fontsize=10.8, color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.set_xlabel("원래 문헌의 검정력", fontsize=11.5)
    ax.set_ylabel("재현에서 유의할 확률", fontsize=11.5)
    ax.set_title("거꾸로 풀면 검정력이 나온다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0.05, 0.96)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 그러면 재현은 몇 번 성공하는가 ──
    ax = axes[1]
    rs = np.linspace(0.15, 1.15, 400)
    ax.plot(rs, [power(lam90 * r) for r in rs], lw=3.8, color=INK)
    ax.axhline(OBS_REP, color=RED, lw=2.2, ls="--")
    ax.axvline(0.5, color=RED, lw=1.8, ls=":")
    ax.plot([0.5], [half], marker="o", ms=14, color=RED, zorder=5,
            mec="white", mew=1.8)
    ax.plot([1.0], [TARGET], marker="D", ms=12, color=TEAL, zorder=5,
            mec="white", mew=1.6)
    ax.annotate(f"보고된 효과가 참값이라면 ${TARGET:.2f}$\n"
                f"— 재현 연구들이 노린 수다",
                xy=(1.0, TARGET), xytext=(0.78, 0.14), fontsize=10.8,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate(f"참값이 절반이면 ${half:.3f}$\n"
                f"실제로 관측된 수는 ${OBS_REP}$ 다",
                xy=(0.5, half), xytext=(0.33, 0.70), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("참 효과 $\\div$ 보고된 효과", fontsize=11.5)
    ax.set_ylabel("재현에서 유의할 확률", fontsize=11.5)
    ax.set_title("$36$ 퍼센트는 설명되는 수다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0.15, 1.15)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$36$ 퍼센트라는 수의 산술", fontsize=14.5)
    fig.text(0.5, -0.06,
             "둘 다 식으로 구한 값이고 모의실험이 아니다. 왼쪽은 "
             "「검정력이 낮다 → 발표된 효과가 부푼다 → 재현이 실패한다」를 "
             "한 줄로 이은 것이고(부푸는 폭은 이 쪽 정리 $2$ 가 준다), "
             "오른쪽은 그 가운데 토막이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "replication_arithmetic.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
