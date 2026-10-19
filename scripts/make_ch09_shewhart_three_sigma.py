r"""9.5절 「그림으로 보는 통계학」 — 왜 하필 3 인가.

1924 년 5 월 16 일, 벨 전화연구소의 월터 슈하트가 상사에게 한 장짜리
메모를 보냈다. 그 메모의 3 분의 1 이 그림 하나였다 — 공정에서 잰 값을
시간 순서로 찍고, 위아래에 선을 두 줄 그어 놓은 것. 관리도의 첫 모습이다.

슈하트가 한 일은 「틀렸다」를 두 가지로 나눈 것이다. 어느 공정에나 있는
**우연원인**의 흔들림과, 찾아내서 고칠 수 있는 **이상원인**. 앞의 것에
손대면 오히려 나빠진다. 그래서 선을 어디에 그을지가 문제가 된다.

그는 3 시그마를 골랐고, **그것이 확률에서 유도된 수가 아니라고 분명히
적었다.** 헛경보를 쫓아다니는 비용과 진짜 이동을 놓치는 비용 사이의
경제적 타협이라는 것이다. 값은 그가 몇 해 동안 공장에서 써 보고 정했다.

왜 3 이 그 자리인지는 평균 실행 길이(ARL)로 보인다.

  한계선 k    아무 일 없을 때 헛경보까지    1 시그마 이동을 잡기까지
      2.0              21.98 점                    6.25 점
      2.5              80.52                      14.92
      3.0             370.40                      43.89
      3.5            2149.34                     160.95
      4.0           15787.19                     740.64

k 를 2 로 내리면 아무 일이 없어도 스물두 점마다 공정을 멈추고 원인을
찾게 된다. 4 로 올리면 1 시그마 이동을 평균 740 점이나 지나쳐 보낸다.
3 은 그 사이다.

2 차 대전이 끝난 뒤 에드워즈 데밍이 이 방법을 일본에 가져갔다.
1950 년 여름 도쿄에서 한 강의가 시작이었다.

만드는 파일:

  ch09/one_sample_tests/img/shewhart_three_sigma.png

실행:  python3 scripts/make_ch09_shewhart_three_sigma.py   (저장소 최상위에서)
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

OUT = "docs/ch09/one_sample_tests/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

KS = ((2.0, RED, "$k = 2$"), (3.0, TEAL, "$k = 3$"),
      (4.0, AMBER, "$k = 4$"))


def arl(k, shift):
    """한계선 k 시그마, 평균이 shift 시그마 옮겨졌을 때의 평균 실행 길이."""
    p = norm.cdf(-k - shift) + norm.cdf(shift - k)
    return 1.0 / p


def main():
    print(f"{'k':>6}{'헛경보까지':>14}{'1σ 이동 탐지':>16}{'2σ':>12}{'3σ':>10}")
    for k in (2.0, 2.5, 3.0, 3.5, 4.0):
        print(f"{k:>6.1f}{arl(k,0):>14.2f}{arl(k,1):>16.2f}"
              f"{arl(k,2):>12.2f}{arl(k,3):>10.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 선을 넓힐수록 헛경보가 줄지만 ──
    ax = axes[0]
    ks = np.linspace(1.5, 4.2, 400)
    ax.plot(ks, [arl(k, 0) for k in ks], lw=3.8, color=INK,
            label="아무 일 없을 때 헛경보까지")
    ax.plot(ks, [arl(k, 1) for k in ks], lw=3.4, color=RED, ls="--",
            label="$1\\sigma$ 이동을 잡기까지")
    ax.axvline(3.0, color=TEAL, lw=2.4, ls=":")
    for y, col in ((arl(3, 0), INK), (arl(3, 1), RED)):
        ax.plot([3.0], [y], marker="o", ms=12, color=col, zorder=5,
                mec="white", mew=1.7)
    ax.annotate(f"$k = 3$ 에서 ${arl(3,0):.0f}$ 점",
                xy=(3.0, arl(3, 0)), xytext=(2.06, 2100), fontsize=10.8,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate(f"${arl(3,1):.0f}$ 점",
                xy=(3.0, arl(3, 1)), xytext=(3.62, 15.5), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_yscale("log")
    ax.set_yticks([10, 100, 1000, 10000])
    ax.set_yticklabels(["$10$", "$100$", "$1{,}000$", "$10{,}000$"])
    ax.minorticks_off()
    ax.set_xlabel("한계선을 몇 시그마에 긋는가", fontsize=11.5)
    ax.set_ylabel("평균 실행 길이 (점의 수)", fontsize=11.5)
    ax.set_title("넓히면 조용해지고, 느려진다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(1.5, 4.2)
    ax.set_ylim(5, 4e4)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 이동이 얼마나 커야 금방 잡히는가 ──
    ax = axes[1]
    sh = np.linspace(0.25, 3.5, 400)
    for k, col, lab in KS:
        ax.plot(sh, [arl(k, s) for s in sh], lw=3.4, color=col, label=lab)
    ax.axhline(1.0, color=MUTED, lw=1.6, ls=":")
    for s, col in ((1.0, TEAL), (2.0, TEAL)):
        ax.plot([s], [arl(3, s)], marker="o", ms=12, color=col, zorder=5,
                mec="white", mew=1.7)
    ax.annotate(f"$k=3$ 은 $1\\sigma$ 이동을\n"
                f"평균 ${arl(3,1):.0f}$ 점이나 지나쳐 보낸다",
                xy=(1.0, arl(3, 1)), xytext=(1.92, 160), fontsize=10.8,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate(f"$2\\sigma$ 면 ${arl(3,2):.1f}$ 점",
                xy=(2.0, arl(3, 2)), xytext=(2.76, 23), fontsize=10.8,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["$1$", "$10$", "$100$", "$1{,}000$"])
    ax.minorticks_off()
    ax.set_xlabel("평균이 옮겨 간 크기 ($\\sigma$ 단위)", fontsize=11.5)
    ax.set_ylabel("탐지까지의 평균 점 수", fontsize=11.5)
    ax.set_title("작은 이동에는 셋 다 느리다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0.25, 3.5)
    ax.set_ylim(0.9, 3e3)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("왜 하필 $3$ 인가", fontsize=14.5)
    fig.text(0.5, -0.06,
             "평균 실행 길이는 경보가 울릴 때까지 찍는 점의 수다. "
             "아무 일 없을 때는 길수록 좋고, 이동이 일어난 뒤에는 짧을수록 "
             "좋다. 한계선 하나가 두 값을 함께 정한다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "shewhart_three_sigma.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
