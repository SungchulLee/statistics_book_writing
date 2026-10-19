r"""4.6절 「그림으로 보는 통계학」 — 타원은 그렸는데 선은 긋지 않았다.

브라베는 1846 년에 이변량 정규의 밀도를 적고 등고선 타원의 **주축**까지
구했다. 그러나 정리 1 이 말하는 선 — 「$a$ 를 알 때 $b$ 의 평균」 — 은
주축이 **아니다.** 두 선은 애초에 다른 물음에 답한다.

  주축          타원을 가장 길게 가로지르는 방향 (기울기 1)
  조건부 평균   각 세로 자름의 무게중심 (기울기 rho = 0.6)

오른쪽은 정리 1 의 다른 절반이다. 세 자름의 **폭이 모두 같다** —
조건부 분산이 a_0 에 의존하지 않는다는 것이 그림으로는 이렇게 보인다.

만드는 파일:

  ch04/bivariate_normal/img/bravais_axis_vs_regression.png

실행:  python3 scripts/make_ch04_bravais_axes.py   (저장소 최상위에서)
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

OUT = "docs/ch04/bivariate_normal/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
MUTED = "#90A4AE"

RHO = 0.6
CUTS = (-1.4, 0.0, 1.4)          # 세로로 자를 자리
LIM = 3.0


def main():
    g = np.linspace(-LIM, LIM, 400)
    U, V = np.meshgrid(g, g)
    cov = [[1.0, RHO], [RHO, 1.0]]
    dens = stats.multivariate_normal([0, 0], cov).pdf(np.dstack((U, V)))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.3), constrained_layout=True)

    # ── 왼쪽: 브라베의 타원 · 주축 · 조건부 평균선 ──
    ax = axes[0]
    ax.contour(U, V, dens, levels=6, colors=MUTED, linewidths=1.4)

    ax.plot([-LIM, LIM], [-LIM, LIM], lw=2.6, color=AMBER, zorder=4)
    ax.plot([-LIM, LIM], [-RHO * LIM, RHO * LIM], lw=2.8, color=TEAL, zorder=5)

    ax.annotate("타원의 주축 — 기울기 1\n브라베가 1846 년에 구한 것",
                xy=(2.2, 2.2), xytext=(-2.85, 2.35),
                fontsize=10.4, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.annotate(f"조건부 평균 — 기울기 ρ = {RHO}\n정리 1 이 말하는 선",
                xy=(1.4, RHO * 1.4), xytext=(2.9, -1.35),
                fontsize=10.4, color=TEAL, ha="right",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))

    for c in CUTS:
        ax.axvline(c, color=INK, lw=1.1, ls=":", zorder=3)
        ax.plot([c], [RHO * c], marker="o", ms=7, color=TEAL, zorder=6)

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xlabel("a", fontsize=11.5)
    ax.set_ylabel("b", fontsize=11.5)
    ax.set_title("같은 타원에 그을 수 있는 선이 둘이다",
                 fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 세로 자름 셋 ──
    ax = axes[1]
    sd = np.sqrt(1 - RHO**2)
    ys = np.linspace(-LIM, LIM, 500)
    for c, col in zip(CUTS, ("#90A4AE", TEAL, AMBER)):
        ax.plot(ys, stats.norm.pdf(ys, RHO * c, sd), lw=2.6, color=col,
                label=f"a = {c:+.1f} 에서 자른 것")
        ax.axvline(RHO * c, color=col, lw=1.3, ls=":")

    ax.annotate("", xy=(RHO * CUTS[0], 0.62), xytext=(RHO * CUTS[2], 0.62),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8))
    ax.text(0, 0.645, f"중심은 {RHO*(CUTS[2]-CUTS[0]):.2f} 만큼 옮겨 가고",
            fontsize=10.4, color=INK, ha="center")
    ax.text(0, 0.10, f"폭은 셋 다 {sd:.3f} 으로 같다",
            fontsize=11.0, color=INK, ha="center")

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(0, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("b", fontsize=11.5)
    ax.set_title("자름의 중심은 움직여도 폭은 움직이지 않는다",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10, framealpha=0.93)
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("밀도를 적는 일과 그 안에서 선을 찾아내는 일은 같지 않다",
                 fontsize=14.5)

    path = OUT + "bravais_axis_vs_regression.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    rho = {RHO}  ·  주축 기울기 1  ·  조건부 평균 기울기 {RHO}")
    print(f"    조건부 표준편차 sqrt(1-rho^2) = {sd:.6f}  (자르는 자리와 무관)")


if __name__ == "__main__":
    main()
