r"""3.3절 「그림으로 보는 통계학」 — 곱해도 되는지를 묻지 않은 대가.

정리 1 은 **독립일 때만** 곱이 곱으로 넘어간다고 했다. 한 집안에서 영아가
두 번 돌연사할 확률을 1/8543 의 제곱으로 적으면 7300 만분의 1 이 된다.
그러나 둘째가 첫째에 조금이라도 기대면 그 수가 통째로 무너진다.

가로축에 조건부확률 P(둘째 | 첫째) 를, 세로축에 결합확률을 로그로 둔다.
독립 가정은 그 축의 맨 왼쪽 끝 한 점일 뿐이고, 오른쪽으로 조금만 가도
결합확률이 수십·수백 배로 올라간다.

만드는 파일:

  ch03/rv/img/two_deaths_independence.png

실행:  python3 scripts/make_ch03_two_deaths.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/rv/img/"

INK = "#37474F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"
AMBER = "#B07C2C"

P1 = 1 / 8543          # 그런 집안에서 영아돌연사 한 번


def main():
    cond = np.logspace(np.log10(P1), np.log10(1 / 30), 400)   # P(둘째 | 첫째)
    joint = P1 * cond
    one_in = 1 / joint

    fig, ax = plt.subplots(figsize=(9.6, 5.3), constrained_layout=True)

    ax.plot(1 / cond, one_in, lw=3.0, color=BLUE, zorder=3)

    # 독립 가정 — 축의 맨 끝 한 점
    ax.plot([8543], [1 / P1**2], marker="o", ms=13, color=RED, zorder=6)
    ax.annotate("독립이라고 보면 여기 한 점\n7,300만분의 1",
                xy=(8543, 1 / P1**2), xytext=(1700, 1 / P1**2 * 0.42),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))

    for c, lab in [(1 / 100, "둘째가 100분의 1 이면"),
                   (1 / 50, "50분의 1 이면")]:
        ax.plot([1 / c], [1 / (P1 * c)], marker="s", ms=10, color=AMBER, zorder=6)
        ax.annotate(f"{lab}\n{1/(P1*c):,.0f}분의 1",
                    xy=(1 / c, 1 / (P1 * c)),
                    xytext=(1 / c * 1.5, 1 / (P1 * c) * 3.2),
                    fontsize=10.5, color=AMBER,
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([50, 100, 500, 1000, 8543])
    ax.set_xticklabels(["50", "100", "500", "1000", "8543"])
    ax.set_yticks([1e5, 1e6, 1e7, 1e8])
    ax.set_yticklabels(["10만", "100만", "1000만", "1억"])
    ax.minorticks_off()
    ax.invert_xaxis()

    ax.set_xlabel("둘째도 잃을 확률을 몇 분의 1 로 보는가  "
                  "(왼쪽 끝이 독립, 오른쪽으로 갈수록 두 죽음이 서로 얽힌다)", fontsize=11)
    ax.set_ylabel("두 번 모두 일어날 확률 — 몇 분의 1", fontsize=11.5)
    ax.set_title("곱해도 되는지를 묻지 않으면 분모가 백 배로 부풀어 오른다",
                 fontsize=13.5, color=INK, pad=10)
    ax.grid(alpha=0.25, which="major")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.045,
             "두 죽음이 서로 무관하다는 가정은 이 곡선의 왼쪽 끝 한 점일 뿐이다. "
             "한 집안이 유전·환경을 함께 나눈다면 참값은 오른쪽 어딘가에 있고, "
             "그만큼 「우연일 리 없다」는 말의 근거가 약해진다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "two_deaths_independence.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for c in (P1, 1 / 1000, 1 / 100, 1 / 50):
        print(f"    P(둘째|첫째)=1/{1/c:7.0f}  ->  1/{1/(P1*c):,.0f}"
              f"   독립가정의 {(P1*c)/P1**2:6.0f} 배")


if __name__ == "__main__":
    main()
