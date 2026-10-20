r"""1.1절 「그림으로 보는 통계학」 — 0.72 는 누구의 수인가.

2023 년 한국의 합계출산율이 0.72 였다. 1970 년 통계 작성 이래 가장
낮고, 2024 년에 0.75 로 조금 올랐다. OECD 에서 열두 해째 꼴찌이며
둘째인 폴란드(1.10)와도 멀다.

이 수를 흔히 「한국 여성이 평생 0.72 명을 낳는다」로 읽는다. 그런데
합계출산율은 그런 뜻이 아니다.

합계출산율은 **그해의 나이별 출산율을 한 여성이 평생 그대로 겪는다면**
몇을 낳겠는가를 더한 수다. 그런 여성은 실재하지 않는다. 열다섯 살의
올해 출산율과 마흔다섯 살의 올해 출산율은 서로 다른 세대의 것인데,
그것을 한 사람의 일생으로 이어 붙인 것이다. 인구학에서 **합성 코호트**
라 부른다.

그래서 이상한 일이 생긴다. 아무도 자녀 수를 바꾸지 않아도 이 수가
떨어진다. 낳는 나이를 뒤로 미루기만 하면 된다.

평생 자녀 수를 2.0 으로 **고정**해 놓고 출산 연령만 해마다 r 년씩
늦추면 그해의 합계출산율이

   r = 0.00  ->  2.000
   r = 0.05  ->  1.905
   r = 0.10  ->  1.818
   r = 0.20  ->  1.667

로 내려간다. 정확히 2.0/(1+r) 이다. 미루기를 멈추면 2.0 으로 그대로
돌아온다 — 떨어진 적이 없었던 것처럼.

한국의 하락이 이것만으로 설명되지는 않는다 — 0.72 는 그런 폭이
아니다. 요점은 다른 데 있다. **이 수는 실재하는 집단의 모수를 추정한
값이 아니다.** 무엇을 재고 있는지 알고 읽어야 하는 수다.

만드는 파일:

  ch01/classical/img/tfr_tempo.png

실행:  python3 scripts/make_ch01_tfr_tempo.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
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

C = 2.0                      # 평생 자녀 수 — 처음부터 끝까지 고정
M0, SD = 28.0, 4.5           # 처음의 평균 출산연령과 퍼짐
AGES = np.arange(15, 50)
R_SHOW = 0.10
START, STOP = 1990, 2015     # 미루기가 시작·끝나는 코호트 출생연도


def mean_age(c, r, start=START, stop=STOP):
    """코호트 c 의 평균 출산연령. start 와 stop 사이에만 해마다 r 년씩 늦어진다."""
    return M0 + r * np.clip(c - start, 0, stop - start)


def period_tfr(t, r):
    """해 t 의 합계출산율 — 그해 나이별 출산율을 모두 더한 값."""
    m = mean_age(t - AGES, r)
    return float((C * norm.pdf(AGES, m, SD)).sum())


def main():
    print(f"  평생 자녀 수를 {C} 로 고정하고 출산연령만 늦출 때")
    print(f"{'r (해마다 늦추는 햇수)':>22}{'그해 합계출산율':>18}{'C/(1+r)':>12}")
    for r in (0.0, 0.05, 0.10, 0.20, 0.30):
        t = START + 40
        print(f"{r:>22.2f}{period_tfr(t, r):>18.3f}{C/(1+r):>12.3f}")

    yrs = np.arange(1985, 2081)
    curves = {r: np.array([period_tfr(t, r) for t in yrs])
              for r in (0.05, 0.10, 0.20)}
    lo = curves[R_SHOW].min()
    hi = curves[R_SHOW][yrs > STOP].max()
    print(f"\n  r = {R_SHOW} 일 때 바닥 {lo:.3f}, 미루기를 멈춘 뒤 "
          f"꼭대기 {hi:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 아무도 자녀 수를 바꾸지 않았다 ──
    ax = axes[0]
    ax.axhspan(C - 0.004, C + 0.004, color=INK, alpha=0.5, zorder=4)
    ax.axvspan(START + 25, STOP + 25, color=AMBER, alpha=0.12, zorder=1)
    for r, col in ((0.05, TEAL), (0.10, RED), (0.20, AMBER)):
        ax.plot(yrs, curves[r], lw=3.2, color=col,
                label=f"해마다 ${r:.2f}$ 년씩 늦춘다")
    ax.text(START + 26, 2.055, "미루는 동안", fontsize=10.6, color=AMBER)
    ax.annotate(f"바닥 ${lo:.2f}$ — 자녀 수는 "
                f"내내 ${C:.1f}$ 로 고정이다",
                xy=(2030, lo), xytext=(1988, 1.56), fontsize=10.6,
                color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"미루기를 멈추면 ${hi:.2f}$ 로 돌아온다\n— 떨어진 적이 "
                f"없었던 것처럼",
                xy=(2050, hi - 0.004), xytext=(2052, 1.60),
                fontsize=10.6, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(1988, C + 0.015, f"평생 자녀 수 ${C:.1f}$ (내내 고정)",
            fontsize=10.6, color=INK)
    ax.set_xlabel("연도", fontsize=11.5)
    ax.set_ylabel("그해의 합계출산율", fontsize=11.5)
    ax.set_title("아무도 자녀 수를 바꾸지 않았다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(1985, 2080)
    ax.set_ylim(1.50, 2.12)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="center left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 얼마나 떨어지는가 ──
    ax = axes[1]
    rs = np.linspace(0, 0.4, 300)
    ax.plot(rs, C / (1 + rs), lw=3.8, color=INK, label="$C/(1+r)$ — 식으로")
    sim = [period_tfr(START + 40, r) for r in (0.05, 0.10, 0.20, 0.30)]
    ax.plot([0.05, 0.10, 0.20, 0.30], sim, lw=0, marker="o", ms=12,
            mfc="none", mec=RED, mew=2.4, label="모의로 재어 본 값")
    ax.axhline(C, color=MUTED, lw=1.8, ls=":")
    ax.annotate(f"해마다 ${R_SHOW:.2f}$ 년이면 ${C/(1+R_SHOW):.2f}$\n"
                f"— ${100*(1-1/(1+R_SHOW)):.0f}\\%$ 가 그냥 깎인다",
                xy=(R_SHOW, C / (1 + R_SHOW)), xytext=(0.245, 1.93),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("출산연령을 해마다 늦추는 햇수 $r$", fontsize=11.5)
    ax.set_ylabel("그해의 합계출산율", fontsize=11.5)
    ax.set_title(f"평생 자녀 수는 ${C:.1f}$ 로 고정인데", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(1.45, 2.08)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$0.72$ 는 누구의 수인가", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"평생 자녀 수를 ${C:.1f}$ 로 **고정**해 놓고 낳는 나이만 "
             "늦춘 모형이다. 자녀 수를 줄인 사람은 한 명도 없다."
             .replace("**", ""),
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "tfr_tempo.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
