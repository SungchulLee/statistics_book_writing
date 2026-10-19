r"""5.4절 「그림으로 보는 통계학」 — 0.2 에서 0.007 을 뗄 수 있는가.

이 쪽은 베르누이 표본평균이 곧 표본비율이고 그 표준오차가
sqrt(p(1-p)/n) 이라고 말한다. n 을 키우면 표준오차가 줄어든다는 당연한
식인데, 그 식이 1930 년대에 유명한 논쟁을 만들었다.

J. B. 라인은 듀크 대학에서 제너 카드로 투시 능력을 시험했다. 기호가
다섯 가지이므로 아무렇게나 찍으면 적중률이 0.2 다. 라인이 보고한 값은
0.207 언저리였다. 0.007 의 초과다.

표본이 작으면 아무 말도 못 할 차이다. 그런데 라인은 시행을 수십만 번
모았다. n = 100,000 이면 표준오차가 0.4/sqrt(n) = 0.00126 이므로
0.007 은 5.5 표준오차다. 「우연일 확률이 10^-7 보다 작다」가 된다.

1937 년에 수리통계학회 회장 버턴 캠프가 성명을 냈다. 라인의 **통계는
문제가 없으며, 반박하려면 수학이 아닌 다른 근거로 해야 한다**는 것이었다.

그 「다른 근거」가 요점이다. 적중률을 0.2 에서 0.007 만큼 밀어 올릴 수
있는 평범한 원인은 많다 — 카드 뒷면에 생긴 미세한 자국, 완전하지 않은
섞기, 실험자의 무의식적 신호, 기록 실수. 어느 것도 0.007 을 만들기에
부족하지 않다. 그리고 n 이 커지면 그 모두가 「압도적으로 유의」해진다.

  n           표준오차    0.007 의 z     탐지 가능한 최소 편차(검정력 0.8)
  1,000       0.01265      0.55            0.0354
  10,000      0.00400      1.75            0.0112
  100,000     0.00126      5.53            0.0035
  1,000,000   0.00040     17.50            0.0011

n 을 키우는 일은 **차이를 키우는 일이 아니라 자를 가늘게 만드는 일**이다.
가늘어진 자는 보고 싶은 것과 보고 싶지 않은 것을 가리지 않는다.

만드는 파일:

  ch05/applications/img/zener.png

실행:  python3 scripts/make_ch05_zener.py   (저장소 최상위에서)
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

OUT = "docs/ch05/applications/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

P0 = 0.2                 # 기호 다섯 가지 — 아무렇게나 찍으면 1/5
EXCESS = 0.007           # 라인이 보고한 초과 적중률 언저리
Z1, Z2 = stats.norm.ppf(0.975), stats.norm.ppf(0.80)


def main():
    ns = np.logspace(2.3, 6.3, 400)
    se = np.sqrt(P0 * (1 - P0) / ns)
    zz = EXCESS / se
    mde = (Z1 + Z2) * se

    print(f"{'n':>12}{'표준오차':>12}{'0.007 의 z':>13}"
          f"{'탐지 가능한 최소 편차':>22}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        s = np.sqrt(P0 * (1 - P0) / n)
        print(f"{n:>12,}{s:>12.5f}{EXCESS/s:>13.2f}{(Z1+Z2)*s:>22.4f}")
    n_star = ((Z1 + Z2) * np.sqrt(P0 * (1 - P0)) / EXCESS) ** 2
    print(f"\n    0.007 을 검정력 0.8 로 잡아내는 데 필요한 n = {n_star:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 자가 가늘어진다 ──
    ax = axes[0]
    ax.fill_between(ns, P0 - mde, P0 + mde, color=TEAL, alpha=0.18,
                    label="우연으로 설명되는 범위 (검정력 $0.8$ 기준)")
    ax.plot(ns, P0 + mde, lw=2.4, color=TEAL)
    ax.plot(ns, P0 - mde, lw=2.4, color=TEAL)
    ax.axhline(P0, color=INK, lw=2.0, ls="--")
    ax.text(230, P0 + 0.0015, "아무렇게나 찍으면 $0.2$", fontsize=11, color=INK)
    ax.axhline(P0 + EXCESS, color=RED, lw=2.4)
    ax.text(230, P0 + EXCESS + 0.0018,
            f"라인이 보고한 적중률 ${P0+EXCESS:.3f}$", fontsize=11, color=RED)
    ax.plot([n_star], [P0 + EXCESS], marker="o", ms=11, color=RED, zorder=5)
    ax.annotate(f"시행 {n_star:,.0f} 번을 넘기면\n이 선이 띠 밖으로 나간다",
                xy=(n_star, P0 + EXCESS), xytext=(2.4e5, 0.228),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(["1 천", "1 만", "10 만", "100 만"])
    ax.minorticks_off()
    ax.set_xlabel("시행 횟수 $n$", fontsize=11.5)
    ax.set_ylabel("적중률", fontsize=11.5)
    ax.set_title("$n$ 을 키우는 일은 자를 가늘게 만드는 일이다",
                 fontsize=12.6, color=INK, pad=8)
    ax.set_ylim(0.168, 0.238)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 평범한 원인들도 같은 크기다 ──
    ax = axes[1]
    ax.plot(ns, zz, lw=3.0, color=RED,
            label=f"초과 적중률 ${EXCESS:.3f}$ 의 $z$")
    for d, col, lab in ((0.003, AMBER, "뒷면의 미세한 자국"),
                        (0.0015, PURPLE, "섞기가 완전하지 않음"),
                        (0.0008, BLUE, "기록 실수 몇 건")):
        ax.plot(ns, d / se, lw=2.2, color=col, ls="--",
                label=f"{lab} — 치우침 ${d:.4f}$")
    for y, lab in ((1.96, "$z = 1.96$"), (5.0, "$z = 5$")):
        ax.axhline(y, color=GREY, lw=1.4, ls=":")
        ax.text(2.3e2, y + 0.6, lab, fontsize=10.6, color=GREY)
    ax.annotate("어느 원인이든 $n$ 만 키우면\n「압도적으로 유의」해진다",
                xy=(1.2e6, 0.0015 / np.sqrt(P0 * (1 - P0) / 1.2e6)),
                xytext=(8e3, 18.5), fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(["1 천", "1 만", "10 만", "100 만"])
    ax.minorticks_off()
    ax.set_xlabel("시행 횟수 $n$", fontsize=11.5)
    ax.set_ylabel("$0.2$ 에서 떨어진 정도 ($z$)", fontsize=11.5)
    ax.set_title("자는 보고 싶은 것만 재지 않는다", fontsize=12.6, color=INK,
                 pad=8)
    ax.set_ylim(0, 34)
    ax.legend(fontsize=10.0, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$0.2$ 에서 $0.007$ 을 떼어 내는 일", fontsize=14.5)
    fig.text(0.5, -0.05,
             "오른쪽의 점선 셋은 초능력과 무관한 평범한 원인들이 만들 법한 "
             "치우침이다. $z$ 가 $\\sqrt{n}$ 에 비례해 자라므로, 원인이 "
             "무엇이든 시행을 충분히 모으면 모두 유의해진다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "zener.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
