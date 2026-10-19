r"""5.3절 「그림으로 보는 통계학」 — 같은 자료, 세 가지 막대.

연습문제 8 이 수로 묻는 것을 눈으로 보인다. n = 25 인 같은 자료에 막대만
바꿔 그리면 SD · 95% CI · SE 의 반길이가 1 : 0.41 : 0.20 이다. 무엇을 그린
것인지 적지 않으면 읽는 사람은 어느 그림을 본 셈인지 알 수 없다.

두 집단을 나란히 두어, 막대 종류만 바꾸었을 뿐인데 「겹친다/안 겹친다」의
인상이 뒤집히는 것까지 보인다.

만드는 파일:

  ch05/applications/img/error_bar_three_ways.png

실행:  python3 scripts/make_ch05_error_bars.py   (저장소 최상위에서)
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
BARS = {"SD": "#1565C0", "95% 신뢰구간": "#B07C2C", "SE": "#D32F2F"}
MUTED = "#90A4AE"

N = 25
SEED = 5


def main():
    rng = np.random.default_rng(SEED)
    a = rng.normal(100.0, 10.0, N)
    b = rng.normal(106.0, 10.0, N)

    tc = stats.t.ppf(0.975, N - 1)
    half = {}
    for g, x in (("A", a), ("B", b)):
        s = x.std(ddof=1)
        half[g] = {"SD": s, "95% 신뢰구간": tc * s / np.sqrt(N), "SE": s / np.sqrt(N)}

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8), sharey=True,
                             constrained_layout=True)

    for ax, kind in zip(axes, BARS):
        for i, (g, x) in enumerate((("A", a), ("B", b))):
            ax.errorbar([i], [x.mean()], yerr=[half[g][kind]], fmt="o", ms=9,
                        lw=2.6, capsize=11, capthick=2.6, color=BARS[kind])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["A 집단", "B 집단"], fontsize=11)
        ratio = half["A"][kind] / half["A"]["SD"]
        ax.set_title(f"{kind} 막대\n반길이가 SD 의 {ratio:.2f} 배",
                     fontsize=12, color=BARS[kind], pad=8)
        ax.grid(alpha=0.22, axis="y")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("측정값", fontsize=11.5)
    fig.suptitle("같은 자료 · 같은 평균 — 막대만 바꿔 그렸다", fontsize=14.5)
    fig.text(0.5, -0.04,
             "무엇을 그린 막대인지 적지 않으면, 읽는 사람은 셋 중 어느 그림을 본 것인지 알 수 없다. "
             "가장 좁은 SE 막대를 고르면 같은 자료가 가장 정밀해 보인다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "error_bar_three_ways.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    n = {N},  t({N-1}, 0.975) = {tc:.4f}")
    for k in BARS:
        print(f"    {k:12} A 반길이 {half['A'][k]:6.3f}"
              f"   SD 대비 {half['A'][k]/half['A']['SD']:.3f}")


if __name__ == "__main__":
    main()
