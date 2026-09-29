r"""3.3절 — 연속형 결합분포, 주변분포, 조건부분포를 한 그림에.

두 칸 모두 주변분포가 표준정규로 **똑같다.** 다른 것은 안쪽의 무게 배치뿐이다.

  왼쪽  rho = 0      등고선이 축에 나란한 원. 독립.
  오른쪽 rho = 0.8   등고선이 기울어진 타원. 종속.

각 칸에 x = 1 에서 세로로 자른 단면을 얹어 조건부분포를 보인다. 독립이면
단면이 주변분포와 포개지고, 종속이면 옮겨 가고 좁아진다.

  Y | X=1  ~  N(rho * 1,  1 - rho^2)
  rho = 0.8 이면 N(0.8, 0.6^2)

이 한 장이 세 가지를 동시에 말한다. 주변분포만으로는 결합분포를 되돌릴 수
없다는 것, 조건부분포가 결합분포를 세로로 자른 것이라는 것, 그리고 독립이란
그 단면이 어디서 자르든 같다는 것.

만드는 파일:

  ch03/rv/img/joint_continuous_conditional.png

실행:  python3 scripts/make_ch03_joint_continuous.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/rv/img/"

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
MUTED = "#90A4AE"
RED = "#D32F2F"

X0 = 1.0                       # 여기서 세로로 자른다
LIM = 3.2


def panel(fig, gs, col, rho, title, color, fill):
    """등고선 + 위아래 주변분포 + x = X0 단면(조건부분포)."""
    ax = fig.add_subplot(gs[1, col])
    ax_top = fig.add_subplot(gs[0, col], sharex=ax)
    ax_right = fig.add_subplot(gs[1, col + 1], sharey=ax)

    g = np.linspace(-LIM, LIM, 240)
    XX, YY = np.meshgrid(g, g)
    cov = [[1.0, rho], [rho, 1.0]]
    Z = stats.multivariate_normal([0, 0], cov).pdf(np.dstack([XX, YY]))

    ax.contourf(XX, YY, Z, levels=7, cmap="Blues" if rho == 0 else "Oranges",
                alpha=0.55)
    ax.contour(XX, YY, Z, levels=7, colors=color, linewidths=0.8)

    # 자르는 자리
    ax.axvline(X0, color=RED, lw=1.8, ls="--")
    ax.text(X0 + 0.12, LIM - 0.2, f"$x = {X0:g}$ 에서 자른다",
            color=RED, fontsize=10.5, rotation=90, va="top")

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel("$x$", fontsize=11)
    if col == 0:
        ax.set_ylabel("$y$", fontsize=11)
    ax.tick_params(labelsize=9)

    # 위: X 의 주변분포 — 두 칸이 같다
    ax_top.plot(g, stats.norm.pdf(g), color=MUTED, lw=2)
    ax_top.fill_between(g, stats.norm.pdf(g), color=MUTED, alpha=0.25)
    ax_top.set_ylim(0, 0.48)
    ax_top.set_title(title, fontsize=13, color=color, pad=8)
    ax_top.set_ylabel("$f_X$", fontsize=10, color=MUTED)
    ax_top.tick_params(labelbottom=False, labelleft=False, length=0)
    ax_top.spines[["top", "right", "left"]].set_visible(False)

    # 오른쪽: Y 의 주변분포(회색)와 조건부분포(색)
    ax_right.plot(stats.norm.pdf(g), g, color=MUTED, lw=2, label="$f_Y$ (주변)")
    ax_right.fill_betweenx(g, stats.norm.pdf(g), color=MUTED, alpha=0.25)

    m, s = rho * X0, np.sqrt(1 - rho ** 2)
    cond = stats.norm.pdf(g, m, s)
    # 겹칠 때 아래의 회색이 보이도록 파선으로 그린다.
    ax_right.fill_betweenx(g, cond, color=fill, alpha=0.6)
    ax_right.plot(cond, g, color=color, lw=2.6, ls="--")

    ax_right.set_xlim(0, 0.78)
    ax_right.tick_params(labelbottom=False, labelleft=False, length=0)
    ax_right.spines[["top", "right", "bottom"]].set_visible(False)
    # 범례 대신 곡선 옆에 직접 이름을 붙인다. 좁은 칸에서는 이쪽이 낫다.
    ax_right.text(0.40, LIM - 0.15, "$f_Y$", color=MUTED, fontsize=11,
                  ha="left", va="top")
    ax_right.text(0.40, LIM - 0.75, "$f_{Y|X=1}$", color=color, fontsize=11,
                  ha="left", va="top")

    note = ("단면이 주변분포와 포개진다\n(그래서 회색이 비쳐 보인다)" if rho == 0
            else f"단면이 ${m:.1f}$ 로 옮겨 가고\n폭이 ${s:.1f}$ 로 좁아진다")
    ax.text(-LIM + 0.15, -LIM + 0.15, note, fontsize=10.5, color=color,
            ha="left", va="bottom",
            bbox=dict(fc="white", ec=color, alpha=0.9, boxstyle="round,pad=0.35"))
    return ax


def main():
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = GridSpec(2, 5, figure=fig,
                  width_ratios=[4, 1.15, 0.5, 4, 1.15],
                  height_ratios=[1.15, 4],
                  hspace=0.06, wspace=0.06)

    panel(fig, gs, 0, 0.0, "$\\rho = 0$ — 독립", BLUE, BLUE_F)
    panel(fig, gs, 3, 0.8, "$\\rho = 0.8$ — 종속", ORANGE, ORANGE_F)

    fig.suptitle("주변분포가 같아도 결합분포는 다를 수 있다", fontsize=15, y=0.99)
    fig.text(0.5, 0.005,
             "위와 오른쪽의 회색 그림자는 두 칸이 완전히 같다. 안쪽 무게만 다르다. "
             "조건부분포는 결합분포를 세로로 자른 단면이며, 독립이면 어디서 자르든 주변분포와 같다.",
             ha="center", fontsize=11)

    path = OUT + "joint_continuous_conditional.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for rho in (0.0, 0.8):
        print(f"    rho={rho}:  Y|X=1 ~ N({rho * X0:.2f}, {1 - rho**2:.2f})  "
              f"sd={np.sqrt(1 - rho**2):.2f}")


if __name__ == "__main__":
    main()
