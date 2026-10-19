r"""20.2절 「그림으로 보는 통계학」 — 파란 버스를 한 대 들여놓으면.

소프트맥스에서 두 범주의 확률의 비는 둘의 점수 차이만으로 정해진다.
다른 범주가 몇 개 있든, 무엇이든 상관이 없다. 선택이론에서 이것을
「무관한 대안으로부터의 독립」(IIA)이라 부른다.

드브뢰(1960)가 루스의 선택공리에 던진 반례가 이렇다.

  출근길에 자동차와 빨간 버스가 있고 반반으로 갈린다.
  여기에 빨간 버스와 모든 것이 같고 색만 다른 파란 버스를 들여놓는다.
  소프트맥스는 셋의 점수가 같으니 1/3씩이라고 답한다.
  자동차가 50%에서 33%로 떨어진 것이다 — 버스를 칠해서.

둥지로짓은 비슷한 대안을 한 둥지에 묶고 비유사도 lambda 를 둔다.
자동차 몫이 1/(1 + 2^lambda) 이다.

  lambda = 1   두 버스가 전혀 안 닮았다      자동차 0.3333  (소프트맥스와 같다)
  lambda -> 0  두 버스가 완전히 같다          자동차 0.5000  (상식의 답)

소프트맥스는 이 가로축 전체에서 0.3333 한 줄이다.

그런데 같은 모형이 교통경제학에서 가장 유명한 예측을 해냈다.
맥패든은 BART 가 열리기 전의 통근 자료로 다항로짓을 맞추어 BART 의
통근 분담률을 6.3% 로 내다보았다. 1975 년의 실제 값이 6.2% 였다.
BART 측의 1973 년 공식 예측은 15% 였다. 맥패든은 2000 년 노벨
경제학상을 받았다.

만드는 파일:

  ch20/softmax_regression/img/iia_red_bus.png

실행:  python3 scripts/make_ch20_iia_red_bus.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch20/softmax_regression/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

BART = (("공식 예측\n(1973)", 15.0, GREY),
        ("맥패든의\n다항로짓", 6.3, TEAL),
        ("실제\n(1975)", 6.2, INK))


def main():
    lam = np.linspace(0.001, 1.0, 400)
    car = 1.0 / (1.0 + 2.0 ** lam)          # 둥지로짓에서 자동차의 몫
    bus = 2.0 ** (lam - 1) / (1.0 + 2.0 ** lam)   # 버스 한 대의 몫

    print("  둥지로짓 — 두 버스를 한 둥지에 묶었을 때")
    print(f"{'비유사도':>10}{'자동차':>10}{'빨간 버스':>12}{'파란 버스':>12}")
    for L in (0.001, 0.25, 0.5, 0.75, 1.0):
        c = 1 / (1 + 2 ** L)
        b = 2 ** (L - 1) / (1 + 2 ** L)
        print(f"{L:>10.3f}{c:>10.4f}{b:>12.4f}{b:>12.4f}")
    print("\n  소프트맥스는 이 줄 전체에서 자동차 0.3333 하나다.")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 버스를 칠하면 자동차가 줄어든다 ──
    ax = axes[0]
    ax.axhline(0.5, color=MUTED, lw=1.6, ls=":")
    ax.plot(lam, car, lw=3.6, color=RED,
            label="둥지로짓 — 자동차")
    ax.plot(lam, bus, lw=2.8, color=BLUE, ls="-.",
            label="둥지로짓 — 버스 한 대")
    ax.axhline(1 / 3, color=TEAL, lw=3.2, ls="--",
               label="소프트맥스 — 셋 모두 $1/3$")
    ax.plot([1.0], [1 / 3], marker="o", ms=13, color=TEAL, zorder=5,
            mec="white", mew=1.6)
    ax.plot([0.0], [0.5], marker="o", ms=13, color=RED, zorder=5,
            mec="white", mew=1.6, clip_on=False)
    ax.annotate("두 버스가 완전히 같다면\n자동차는 $0.5$ 그대로여야 한다",
                xy=(0.0, 0.5), xytext=(0.34, 0.383), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("소프트맥스는 닮았든 아니든\n언제나 여기 한 점이다",
                xy=(0.62, 1 / 3), xytext=(0.62, 0.252), fontsize=10.8,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("두 버스가 서로 안 닮은 정도 $\\lambda$", fontsize=11.5)
    ax.set_ylabel("출근길에 그것을 고를 확률", fontsize=11.5)
    ax.set_title("버스를 파랗게 칠하면 자동차가 줄어든다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.22, 0.56)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["$0$\n(똑같다)", "0.25", "0.5", "0.75",
                        "$1$\n(전혀 다르다)"])
    ax.set_yticks([0.25, 1 / 3, 0.4, 0.5])
    ax.set_yticklabels(["0.25", "0.333", "0.4", "0.5"])
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 그런데도 BART 를 맞혔다 ──
    ax = axes[1]
    xs = np.arange(len(BART))
    for x, (lab, v, col) in zip(xs, BART):
        ax.bar(x, v, width=0.56, color=col, alpha=0.9, zorder=3)
        ax.text(x, v + 0.42, f"${v}\\%$", ha="center", fontsize=13,
                color=col, fontweight="bold")
    ax.plot([0.62, 2.38], [6.2, 6.2], color=INK, lw=1.6, ls=":", zorder=4)
    ax.annotate("빗나간 폭이 $0.1$ 포인트다",
                xy=(1.5, 6.25), xytext=(1.62, 11.2), fontsize=11,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate("공식 예측은 두 배 반을 넘겼다",
                xy=(0.0, 15.0), xytext=(0.52, 16.9), fontsize=11,
                color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for lab, _, _ in BART], fontsize=11.2)
    ax.set_ylabel("BART 의 통근 분담률", fontsize=11.5)
    ax.set_title("그 결함 있는 모형이 해낸 예측", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0, 19.5)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_yticklabels(["$0\\%$", "$5\\%$", "$10\\%$", "$15\\%$"])
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("파란 버스를 한 대 들여놓으면", fontsize=14.5)
    fig.text(0.5, -0.07,
             "왼쪽 $\\lambda$ 는 둥지로짓의 비유사도다. $\\lambda = 1$ 이면 "
             "둥지로짓이 소프트맥스와 같아지고, $\\lambda \\to 0$ 이면 두 "
             "버스가 사실상 한 대가 된다. 오른쪽은 맥패든이 BART 개통 전 "
             "자료로 낸 예측과 $1975$ 년의 실제 값이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "iia_red_bus.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
