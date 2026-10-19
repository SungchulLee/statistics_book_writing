r"""5.8절 「그림으로 보는 통계학」 — 아무 일도 일어나지 않은 팔.

이 쪽은 p 가 작을 때 정규근사가 어디서 고장 나는지를 다룬다. 실무에서
그 고장이 가장 아프게 드러나는 자리가 약물 안전성이다.

새 약의 드문 부작용을 찾는 일을 생각해 보자. 참 사건률이 0.4%, 0.8%
언저리이고 시험에 각 팔 500 명씩 넣었다면, **한쪽 팔의 사건 수가 0 이
되는 일이 15% 의 확률로 일어난다.** 둘 다 0 인 경우도 있다.

사건이 0 이면 오즈비가 정의되지 않는다. 분모가 0 이다. 그래서 관례로
칸마다 0.5 를 더한다(홀데인·앤스콤). 그런데 그 0.5 는 자료에서 나온
수가 아니다.

  처치 5 건 / 대조 0 건, 각 1000 명
    보정 없음   정의 안 됨
    +0.5        오즈비 11.06
    +1.0        오즈비  6.03

**관례 하나가 답을 두 배 가까이 바꾼다.** 사건이 넉넉하면(8 대 2) 같은
보정이 4.02 를 3.42 로 바꿀 뿐이다. 보정이 문제가 되는 것은 0 이 있을
때뿐이고, 하필 드문 부작용에서는 0 이 흔하다.

2007 년 당뇨약 로시글리타존의 심근경색 위험을 두고 벌어진 논쟁이 이
자리에 있었다. 메타분석에 들어간 시험 가운데 상당수가 한쪽 또는 양쪽
팔의 사건 수가 0 이었고, 그 0 들을 어떻게 다루느냐가 결론의 세기를
좌우한다는 점이 공개적으로 다투어졌다.

만드는 파일:

  ch05/applications/img/zero_cell.png

실행:  python3 scripts/make_ch05_zero_cell.py   (저장소 최상위에서)
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

NTRIAL = 1000           # 오른쪽 그림에서 각 팔의 인원
CASES = [(5, 0, RED), (3, 0, AMBER), (1, 0, PURPLE), (8, 2, TEAL)]
RATES = [(0.002, 0.004, BLUE), (0.004, 0.008, TEAL), (0.01, 0.02, AMBER)]


def p_zero(n, p1, p2):
    """두 팔 가운데 적어도 하나의 사건 수가 0 일 확률 (정확)."""
    z1 = stats.binom.pmf(0, n, p1)
    z2 = stats.binom.pmf(0, n, p2)
    return z1 + z2 - z1 * z2


def odds(x1, x2, n, c):
    return ((x1 + c) / (n - x1 + c)) / ((x2 + c) / (n - x2 + c))


def main():
    ns = np.unique(np.round(np.logspace(np.log10(100), np.log10(20000), 160)
                            ).astype(int))
    print("    한쪽 팔이라도 사건 0 건일 확률")
    print(f"{'각 팔 인원':>10}" + "".join(f"{f'{a:.1%} vs {b:.1%}':>16}"
                                          for a, b, _ in RATES))
    for n in (250, 500, 1000, 2000, 5000):
        row = "".join(f"{p_zero(n, a, b):>16.4f}" for a, b, _ in RATES)
        print(f"{n:>10}{row}")

    print(f"\n    보정상수에 따라 오즈비가 어떻게 달라지는가 (각 팔 {NTRIAL} 명)")
    for x1, x2, _ in CASES:
        vals = "   ".join(f"c={c}: {odds(x1, x2, NTRIAL, c):6.2f}"
                          for c in (0.25, 0.5, 1.0, 2.0))
        base = "정의 안 됨" if x2 == 0 else f"{odds(x1, x2, NTRIAL, 0):.2f}"
        print(f"      처치 {x1} / 대조 {x2}   보정 없음 {base:>10}   {vals}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 0 건짜리 팔은 흔하다 ──
    ax = axes[0]
    for a, b, col in RATES:
        ax.plot(ns, [p_zero(n, a, b) for n in ns], lw=2.8, color=col,
                label=f"참 사건률 ${a*100:.1f}\\%$ 대 ${b*100:.1f}\\%$")
    ax.axhline(0.05, color=GREY, lw=1.4, ls=":")
    ax.text(110, 0.062, "$5\\%$", fontsize=10.6, color=GREY)
    a, b, _ = RATES[1]
    for n in (500, 2000):
        ax.plot([n], [p_zero(n, a, b)], marker="o", ms=10, color=TEAL, zorder=5)
    ax.annotate(f"각 팔 $500$ 명이면\n{p_zero(500, a, b)*100:.0f}% 의 확률로\n"
                "한쪽 팔이 $0$ 건이다",
                xy=(500, p_zero(500, a, b)), xytext=(1750, 0.62),
                fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([100, 300, 1000, 3000, 10000, 20000])
    ax.set_xticklabels(["100", "300", "1,000", "3,000", "1 만", "2 만"])
    ax.minorticks_off()
    ax.set_xlabel("각 팔의 인원", fontsize=11.5)
    ax.set_ylabel("한쪽 팔이라도 사건 $0$ 건일 확률", fontsize=11.5)
    ax.set_title("드문 부작용에서는 $0$ 이 흔하다", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 더하는 수가 답을 정한다 ──
    ax = axes[1]
    cs = np.logspace(np.log10(0.05), np.log10(3.0), 300)
    for x1, x2, col in CASES:
        lab = f"처치 {x1} 건 / 대조 {x2} 건"
        ax.plot(cs, [odds(x1, x2, NTRIAL, c) for c in cs], lw=2.8, color=col,
                label=lab)
    ax.axvline(0.5, color=INK, lw=2.2, ls="--")
    ax.text(0.53, 14.5, "관례로 쓰는 $0.5$", fontsize=11.2, color=INK)
    x1, x2, _ = CASES[0]
    ax.annotate(f"같은 자료인데\n$+0.5$ 면 {odds(x1,x2,NTRIAL,0.5):.1f}, "
                f"$+1.0$ 이면 {odds(x1,x2,NTRIAL,1.0):.1f}",
                xy=(1.0, odds(x1, x2, NTRIAL, 1.0)), xytext=(1.15, 20.5),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate("사건이 넉넉하면\n보정이 거의 듣지 않는다",
                xy=(2.0, odds(8, 2, NTRIAL, 2.0)), xytext=(2.05, 9.5),
                fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "3"])
    ax.minorticks_off()
    ax.set_xlabel(f"칸마다 더하는 수 $c$  (각 팔 ${NTRIAL:,}$ 명)".replace(",", "{,}"),
                  fontsize=11.5)
    ax.set_ylabel("그렇게 계산한 오즈비", fontsize=11.5)
    ax.set_title("자료에서 나오지 않은 수가 답을 정한다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("아무 일도 일어나지 않은 팔을 어떻게 셀 것인가", fontsize=14.5)
    fig.text(0.5, -0.05,
             "사건 수가 $0$ 이면 오즈비의 분모가 $0$ 이라 정의되지 않는다. "
             "그래서 칸마다 작은 수를 더하는데, 그 수는 자료에서 나온 것이 아니다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "zero_cell.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
