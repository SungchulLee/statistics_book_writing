r"""15.1절 「그림으로 보는 통계학」 — 이름에 시그마가 들어간 경영 운동.

이 쪽 §1 이 규격 이탈 확률을 2 Phi(-L/sigma) 로 적었다. 그 한 줄을
목표로 삼아 이름까지 거기서 따온 것이 1980 년대의 「식스 시그마」다.

  규격까지의 여유가 k 시그마일 때
    치우침이 없으면        2 Phi(-k)
    평균이 1.5 시그마 밀리면  Phi(-(k-1.5)) + Phi(-(k+1.5))

6 시그마에서 앞엣것은 십억에 둘이고 뒤엣것은 백만에 3.4 다. 널리 인용되는
「백만에 3.4」가 치우침을 1.5 시그마 허용한 쪽의 값이다.

만드는 파일:

  ch15/introduction/img/six_sigma.png

실행:  python3 scripts/make_ch15_six_sigma.py   (저장소 최상위에서)
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

OUT = "docs/ch15/introduction/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

SHIFT = 1.5
MARKS = (3.0, 4.0, 5.0, 6.0)


def dpmo_centered(k):
    return 2 * stats.norm.cdf(-k) * 1e6


def dpmo_shifted(k):
    return (stats.norm.cdf(-(k - SHIFT)) + stats.norm.cdf(-(k + SHIFT))) * 1e6


def main():
    print("    규격 여유   치우침 없음        1.5 시그마 밀림")
    for k in MARKS:
        print(f"    {k:g} 시그마   {dpmo_centered(k):12.4f} DPMO   "
              f"{dpmo_shifted(k):11.2f} DPMO")
    print(f"    6 시그마에서 두 값의 비 {dpmo_shifted(6)/dpmo_centered(6):.0f} 배")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    # ── 왼쪽: 1.5 시그마 밀림이 무엇인가 ──
    ax = axes[0]
    g = np.linspace(-7.2, 7.2, 1200)
    ax.plot(g, stats.norm.pdf(g), lw=2.6, color=MUTED, label="치우침 없는 공정")
    ax.plot(g, stats.norm.pdf(g, SHIFT), lw=2.6, color=TEAL,
            label=f"평균이 {SHIFT} 시그마 밀린 공정")
    for s in (-6, 6):
        ax.axvline(s, color=RED, lw=2.2)
    ax.text(6.15, 0.055, "규격 상한\n$+6\\sigma$", fontsize=10.4, color=RED)
    ax.text(-6.15, 0.055, "규격 하한\n$-6\\sigma$", fontsize=10.4, color=RED,
            ha="right")
    tail = g[g >= 6]
    ax.fill_between(tail, 0, stats.norm.pdf(tail, SHIFT), color=TEAL, alpha=0.5)
    ax.annotate(f"이 꼬리가 {dpmo_shifted(6):.1f} DPMO", xy=(6.2, 0.004),
                xytext=(1.2, 0.17), fontsize=10.8, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlim(-7.2, 7.5)
    ax.set_ylim(0, 0.46)
    ax.set_yticks([])
    ax.set_xlabel("$\\sigma$ 단위로 잰 자리", fontsize=11.5)
    ax.set_title("「밀려도 괜찮게」 여유를 잡는다", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── 오른쪽: 여유에 따른 불량률 ──
    ax = axes[1]
    ks = np.linspace(1.0, 6.4, 400)
    ax.semilogy(ks, dpmo_centered(ks), lw=2.8, color=MUTED,
                label="치우침 없음  $2\\Phi(-k)$")
    ax.semilogy(ks, dpmo_shifted(ks), lw=2.8, color=TEAL,
                label=f"{SHIFT} 시그마 밀림")
    for k in MARKS:
        ax.plot([k], [dpmo_shifted(k)], marker="o", ms=8, color=TEAL, zorder=5)
    ax.annotate(f"{dpmo_shifted(3):,.0f}", xy=(3, dpmo_shifted(3)),
                xytext=(2.3, dpmo_shifted(3) * 6), fontsize=10.8, color=TEAL)
    ax.annotate(f"{dpmo_shifted(6):.1f}\n백만에 {dpmo_shifted(6):.1f} 개",
                xy=(6, dpmo_shifted(6)), xytext=(4.5, dpmo_shifted(6) * 25),
                fontsize=10.8, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.annotate(f"{dpmo_centered(6):.4f}", xy=(6, dpmo_centered(6)),
                xytext=(4.4, dpmo_centered(6) * 4), fontsize=10.8, color=MUTED)
    ax.set_xlabel("규격까지의 여유 $k$ (시그마)", fontsize=11.5)
    ax.set_ylabel("백만 개당 불량 (DPMO)", fontsize=11.5)
    ax.set_title("여유를 두 배로 하면 불량은 자릿수로 준다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    # 로그 눈금 라벨을 평문으로 박는다 — mathtext 의 유니코드 마이너스가 두부가 된다
    yt = [1e-3, 1e-1, 1e1, 1e3, 1e5]
    ax.set_yticks(yt)
    ax.set_yticklabels(["0.001", "0.1", "10", "1,000", "100,000"])
    ax.minorticks_off()
    ax.set_ylim(5e-4, 1e6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$\\sigma$ 가 분모에 있으면 여유 한 칸이 자릿수를 바꾼다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             "두 곡선의 차가 $1.5$ 시그마 밀림을 허용한 값이다. "
             f"$6$ 시그마에서 {dpmo_centered(6):.4f} 와 {dpmo_shifted(6):.1f} 로 "
             f"{dpmo_shifted(6)/dpmo_centered(6):,.0f} 배 벌어진다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "six_sigma.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
