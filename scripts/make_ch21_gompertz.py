r"""21.2절 「그림으로 보는 통계학」 — 여덟 해마다 두 배.

이 쪽은 위험함수의 모양을 다룬다. 사람의 사망위험에는 이백 년 동안
버틴 모양이 하나 있다.

벤저민 곰페르츠(1779-1865)는 영국의 수학자이자 보험계리인이었다.
유대인이라 대학에 갈 수 없어 독학했고, 왕립학회 회원이 되었으며
얼라이언스 보험회사의 계리인으로 일했다. 1825 년에 왕립학회에 낸
논문에서 그는 사람의 사망위험이 나이에 따라 **기하급수로** 자란다고
적었다.

  h(t) = B exp(c t)

c 를 배가 시간으로 옮기면 ln2/c 이고, 사람에서 그 값이 대략 **여덟
해**다. 나라가 달라도 세기가 달라도 이 기울기는 놀랍도록 고정되어
있었다. 바뀐 것은 B — 선의 높이 — 뿐이다.

배가 시간이 여덟 해라는 말이 얼마나 센지 보자.

    30 세 연 0.00100      70 세  30 세의  32 배
    50 세 연 0.00566      90 세  30 세의 181 배

그리고 여기서 서늘한 결론이 나온다. 위험을 m 배로 줄이는 개입은
나이를 ln(1/m)/c 만큼 되돌리는 것과 같다. 배가 시간이 여덟 해이므로

  사망의 20% 를 없앤다   ->  30 세의 기대여명 +2.47 년
  사망의 30% 를 없앤다   ->  +3.95 년
  사망을 절반으로         ->  +7.72 년  (= 배가 시간 하나)
  사망을 10 분의 1 로     ->  +25.82 년

어떤 병 하나를 완전히 없애도 몇 해밖에 벌지 못한다는 널리 알려진
계산이 이 식에서 나온다. 지수로 자라는 벽 앞에서는 벽을 낮추는 일보다
**기울기를 바꾸는 일**이 어렵고, 아직 아무도 기울기를 바꾸지 못했다.

만드는 파일:

  ch21/introduction/img/gompertz.png

실행:  python3 scripts/make_ch21_gompertz.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/introduction/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

DT = 8.0                      # 사망위험의 배가 시간 (년)
C = np.log(2) / DT
H30 = 0.001                   # 30 세의 연간 사망위험
B = H30 / np.exp(C * 30)
AGES = np.linspace(30, 105, 600)
GRID = np.linspace(30, 125, 200_001)


def life_exp(mult):
    h = mult * B * np.exp(C * GRID)
    S = np.exp(-np.cumsum(h) * (GRID[1] - GRID[0]))
    return 30 + S.sum() * (GRID[1] - GRID[0])


def main():
    print(f"    배가 시간 {DT:g} 년  →  c = {C:.5f}")
    for a in (30, 50, 70, 90):
        print(f"      {a} 세  연 {B*np.exp(C*a):.5f}   "
              f"30 세의 {np.exp(C*(a-30)):.1f} 배")
    base = life_exp(1.0)
    print(f"\n    30 세의 기대여명 기준 {base:.2f} 세")
    cases = ((0.8, "사망의 $20\\%$ 를 없앤다"), (0.7, "사망의 $30\\%$ 를 없앤다"),
             (0.5, "사망을 절반으로"), (0.25, "사망을 $1/4$ 로"),
             (0.1, "사망을 $1/10$ 로"))
    gains = []
    for m, lab in cases:
        g = life_exp(m) - base
        gains.append(g)
        plain = lab.replace("$", "").replace("\\", "")
        print(f"      {plain:<22} +{g:5.2f} 년   (식 {np.log(1/m)/C:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 로그 위험이 직선이다 ──
    ax = axes[0]
    for m, col, lab in ((1.0, RED, "지금"), (0.5, AMBER, "사망을 절반으로"),
                        (0.25, TEAL, "사망을 $1/4$ 로")):
        ax.plot(AGES, m * B * np.exp(C * AGES), lw=3.0, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    for a in (70, 78, 86):
        ax.plot([a], [B * np.exp(C * a)], marker="o", ms=8, color=RED, zorder=5)
    ax.annotate("", xy=(78, B * np.exp(C * 70)), xytext=(70, B * np.exp(C * 70)),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
    ax.text(74, B * np.exp(C * 70) * 0.62, f"{DT:g} 년", fontsize=11, color=INK,
            ha="center")
    ax.annotate(f"$70$ 세의 위험이 $30$ 세의 "
                f"${np.exp(C*40):.0f}$ 배다",
                xy=(70, B * np.exp(C * 70)), xytext=(46, 0.22),
                fontsize=10.8, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(104, 0.00085, "위험을 절반으로 줄이는 일은\n"
            "여덟 해 젊어지는 일과 같다 —\n선이 내려가는 것이 아니라 옆으로 밀린다",
            fontsize=10.4, color=AMBER, ha="right", va="bottom")
    ax.set_xlabel("나이", fontsize=11.5)
    ax.set_ylabel("연간 사망위험", fontsize=11.5)
    ax.set_title("로그로 그리면 직선이다 — 기울기가 곧 배가 시간",
                 fontsize=12.3, color=INK, pad=8)
    ax.set_xlim(30, 105)
    ax.set_ylim(5e-4, 1.4)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 없앨수록 버는 햇수 ──
    ax = axes[1]
    fr = np.linspace(0.01, 0.92, 300)
    yrs = np.array([life_exp(1 - f) - base for f in fr])
    ax.plot(fr * 100, yrs, lw=3.2, color=TEAL)
    for (m, lab), g in zip(cases, gains):
        ax.plot([(1 - m) * 100], [g], marker="o", ms=10, color=AMBER, zorder=5)
    ax.annotate(f"사망의 $20\\%$ 를 통째로 없애도\n"
                f"$+{gains[0]:.1f}$ 년이다",
                xy=(20, gains[0]), xytext=(40, 1.1), fontsize=10.8,
                color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.annotate(f"절반으로 줄여야 $+{gains[2]:.1f}$ 년 —\n"
                f"배가 시간 하나만큼이다",
                xy=(50, gains[2]), xytext=(26, 17.5), fontsize=10.8,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("없앤 사망의 비율 (%)", fontsize=11.5)
    ax.set_ylabel("$30$ 세의 기대여명이 늘어난 햇수", fontsize=11.5)
    ax.set_title("지수로 자라는 벽 앞에서", fontsize=12.6, color=INK, pad=8)
    ax.set_xlim(0, 92)
    ax.set_ylim(0, 28)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("여덟 해마다 두 배", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"사망위험의 배가 시간을 {DT:g} 년으로, $30$ 세의 연간 위험을 "
             f"${H30}$ 으로 두었다. 기울기 $c$ 는 바꾸지 않고 높이 $B$ 만 "
             "낮추는 개입을 상상한 것이다 — 아직 아무도 기울기를 바꾸지 못했다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "gompertz.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
