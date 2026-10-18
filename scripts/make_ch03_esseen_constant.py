r"""3.5절 「그림으로 보는 통계학」 — 닫히지 않는 틈.

정리 1 의 상수 C 는 아직 아무도 모른다. 위에서 눌러 온 기록은 80 년 동안
7.59 에서 0.4690 까지 내려왔는데, 에세인이 1956 년에 세워 둔 바닥
(sqrt(10)+3)/(6 sqrt(2 pi)) = 0.409732... 에는 끝내 닿지 못했다.

값은 이 책 본문(연습문제 9)의 표를 그대로 쓴다.

  위에서 눌러 온 기록   1941 에세인 7.59 · 1972 판 베이크 0.7975
                       2001 시가노프 0.7056 · 2010 셰프초바 0.4748
                       2011 셰프초바 0.4690
  아래에서 받친 바닥   1956 에세인 0.409732

만드는 파일:

  ch03/limits/img/esseen_constant_gap.png

실행:  python3 scripts/make_ch03_esseen_constant.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/limits/img/"

INK = "#37474F"
RED = "#D32F2F"
TEAL = "#00838F"
MUTED = "#90A4AE"

UPPER = [(1941, 7.59, "에세인"), (1972, 0.7975, "판 베이크"),
         (1901 + 100, 0.7056, "시가노프"), (2010, 0.4748, "셰프초바"),
         (2011, 0.4690, "셰프초바")]
LOWER = (np.sqrt(10) + 3) / (6 * np.sqrt(2 * np.pi))
NOW = 2026


def main():
    yrs = [u[0] for u in UPPER] + [NOW]
    vals = [u[1] for u in UPPER] + [UPPER[-1][1]]

    fig, ax = plt.subplots(figsize=(9.8, 5.4), constrained_layout=True)
    TOP = 0.86                       # 7.59 는 눈금 밖으로 보낸다 — 이야기는 바닥 근처에 있다

    # 아직 아무도 들어가 보지 못한 띠
    ax.fill_between([1956, NOW + 2], LOWER, UPPER[-1][1],
                    color=RED, alpha=0.16, zorder=1)

    clipped = [(y, min(v, TOP)) for y, v, _ in UPPER] + [(NOW + 2, UPPER[-1][1])]
    ax.step([c[0] for c in clipped], [c[1] for c in clipped],
            where="post", lw=3.0, color=TEAL, zorder=4)

    for y, v, who in UPPER:
        if v > TOP:                  # 눈금 밖 — 화살표로만 알린다
            ax.annotate(f"{who} {y} 년에는 {v}\n(눈금 밖, 지금의 16 배)",
                        xy=(y, TOP), xytext=(y + 4, TOP * 0.80),
                        fontsize=10, color=INK,
                        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
            continue
        ax.plot([y], [v], marker="o", ms=9, color=TEAL, zorder=5)

    ax.annotate("판 베이크 1972\n0.7975", xy=(1972, 0.7975),
                xytext=(1974, 0.805), fontsize=9.8, color=INK)
    ax.annotate("시가노프 2001\n0.7056", xy=(2001, 0.7056),
                xytext=(1994, 0.735), fontsize=9.8, color=INK)
    ax.annotate("셰프초바 2010 년 0.4748\n       2011 년 0.4690",
                xy=(2011, 0.4690), xytext=(1972, 0.515),
                fontsize=9.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax.axhline(LOWER, color=RED, lw=2.6, zorder=3)
    ax.text(1937, LOWER - 0.013,
            f"에세인이 1956 년에 세운 바닥 — 어떤 증명도 이보다 작을 수 없다   {LOWER:.6f}",
            fontsize=10.2, color=RED, va="top")

    ax.annotate("", xy=(2024, UPPER[-1][1]), xytext=(2024, LOWER),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
    ax.text(2021, (LOWER + UPPER[-1][1]) / 2,
            f"여기 어딘가에 답이 있다\n너비 {UPPER[-1][1]-LOWER:.4f}\n70 년째 비어 있다",
            fontsize=10.2, color=RED, ha="right", va="center")

    ax.set_xlim(1935, NOW + 4)
    ax.set_ylim(0.37, TOP)
    ax.set_xlabel("연도", fontsize=11.5)
    ax.set_ylabel("상수 C 로 알려진 값", fontsize=11.5)
    ax.set_title("여든 해를 눌러 내려왔는데 바닥에는 닿지 못했다",
                 fontsize=13.5, color=INK, pad=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    path = OUT + "esseen_constant_gap.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    바닥 = {LOWER:.6f}")
    print(f"    가장 낮은 천장 = {UPPER[-1][1]}  (틈의 너비 {UPPER[-1][1]-LOWER:.4f})")
    print(f"    7.59 에서 0.4690 까지 {UPPER[-1][0]-UPPER[0][0]} 년")


if __name__ == "__main__":
    main()
