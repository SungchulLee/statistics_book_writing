r"""3.3절 — 결합분포와 주변분포, 그리고 확률변수의 독립.

동전을 두 번 던져 얻은 세 확률변수로 같은 그림을 두 번 그린다.

  X = 1(첫 번째가 앞면),  Y = 1(두 번째가 앞면),  Z = X + Y

  왼쪽 (X, Y)  칸의 무게가 가장자리 무게의 곱과 정확히 같다 -> 독립
  오른쪽 (X, Z) 곱이 맞지 않는 칸이 있다 -> 독립이 아니다

가장자리에 주변분포를 막대로 붙여, 주변분포가 "한쪽 축으로 눌러 모은
그림자"라는 것과 독립이 "안쪽 무게 = 두 그림자의 곱"이라는 것을 한눈에 보인다.

만드는 파일:

  ch03/rv/img/joint_marginal_coins.png

실행:  python3 scripts/make_ch03_joint_marginal.py   (저장소 최상위에서)
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
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

P = 0.6                      # 앞면 확률
Q = 1 - P


def coin_tables():
    """(X, Y) 와 (X, Z) 의 결합분포를 표로 만든다."""
    # Omega = {HH, HT, TH, TT}, 각각의 확률
    outcomes = {
        ("H", "H"): P * P,
        ("H", "T"): P * Q,
        ("T", "H"): Q * P,
        ("T", "T"): Q * Q,
    }
    xy = np.zeros((2, 2))        # xy[x, y]
    xz = np.zeros((2, 3))        # xz[x, z]
    for (a, b), prob in outcomes.items():
        x = 1 if a == "H" else 0
        y = 1 if b == "H" else 0
        xy[x, y] += prob
        xz[x, x + y] += prob
    return xy, xz


def panel(ax, joint, xlabels, ylabels, title, name_y, independent):
    """격자 위의 결합분포와 양쪽 가장자리의 주변분포를 한 칸에 그린다."""
    nx, ny = joint.shape
    px = joint.sum(axis=1)       # X 의 주변분포
    py = joint.sum(axis=0)       # 상대 변수의 주변분포

    face = BLUE_F if independent else ORANGE_F
    edge = BLUE if independent else ORANGE

    # 안쪽: 칸마다 무게에 비례하는 정사각형
    for i in range(nx):
        for j in range(ny):
            m = joint[i, j]
            if m == 0:
                # 무게가 0 인 칸. 독립이라면 여기에도 무게가 있어야 하므로
                # 이 칸이 곧 독립이 깨졌다는 가장 선명한 증거다.
                ax.add_patch(plt.Rectangle((i - 0.20, j - 0.20), 0.40, 0.40,
                                           fill=False, ec=MUTED, ls=":", lw=1.2))
                ax.text(i, j, "0", ha="center", va="center",
                        fontsize=10, color=MUTED)
                ax.text(i, j - 0.33, f"{px[i]:.1f}×{py[j]:.2f}={px[i] * py[j]:.2f}",
                        ha="center", va="top", fontsize=8.5, color=RED)
                continue
            s = 0.78 * np.sqrt(m / joint.max())
            ax.add_patch(plt.Rectangle((i - s / 2, j - s / 2), s, s,
                                       fc=face, ec=edge, lw=1.6))
            ax.text(i, j, f"{m:.2f}", ha="center", va="center",
                    fontsize=11, color=INK)
            # 곱이 맞는지 칸 아래에 적는다.
            prod = px[i] * py[j]
            ok = abs(prod - m) < 1e-12
            ax.text(i, j - s / 2 - 0.13,
                    f"{px[i]:.1f}×{py[j]:.2f}={prod:.2f}",
                    ha="center", va="top", fontsize=8.5,
                    color=GREEN if ok else RED)

    # 위쪽 가장자리: X 의 주변분포
    for i in range(nx):
        ax.add_patch(plt.Rectangle((i - 0.16, ny - 0.42), 0.32,
                                   0.42 * px[i] / max(px.max(), py.max()),
                                   fc=MUTED, ec=INK, lw=0.8))
        ax.text(i, ny - 0.50, f"{px[i]:.1f}", ha="center", va="top",
                fontsize=9.5, color=INK)

    # 오른쪽 가장자리: 상대 변수의 주변분포
    for j in range(ny):
        ax.add_patch(plt.Rectangle((nx - 0.42, j - 0.16),
                                   0.42 * py[j] / max(px.max(), py.max()),
                                   0.32, fc=MUTED, ec=INK, lw=0.8))
        ax.text(nx - 0.50, j, f"{py[j]:.2f}", ha="right", va="center",
                fontsize=9.5, color=INK)

    ax.set_xticks(range(nx))
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_yticks(range(ny))
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_xlabel("$X$ (첫 번째가 앞면이면 1)", fontsize=11)
    ax.set_ylabel(name_y, fontsize=11)
    ax.set_xlim(-0.75, nx - 0.05)
    ax.set_ylim(-0.75, ny - 0.05)
    ax.set_title(title, fontsize=13, color=edge, pad=10)
    ax.set_aspect("equal" if nx == ny else "auto")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)


def main():
    xy, xz = coin_tables()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2),
                             constrained_layout=True)

    panel(axes[0], xy, ["0", "1"], ["0", "1"],
          "$(X, Y)$ — 모든 칸에서 곱이 맞는다", "$Y$ (두 번째가 앞면이면 1)",
          independent=True)
    panel(axes[1], xz, ["0", "1"], ["0", "1", "2"],
          "$(X, Z)$ — 곱이 맞지 않는 칸이 있다", "$Z = X + Y$",
          independent=False)

    fig.suptitle(f"결합분포는 안쪽 무게, 주변분포는 가장자리 그림자 "
                 f"(앞면 확률 $p = {P}$)", fontsize=14.5)
    fig.text(0.5, -0.035,
             "가장자리 막대가 주변분포다. 독립이면 안쪽 칸의 무게가 두 가장자리 "
             "무게의 곱과 같고(왼쪽, 초록), 아니면 어긋난다(오른쪽, 빨강).",
             ha="center", fontsize=11)

    path = OUT + "joint_marginal_coins.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")

    print(f"\n  p = {P},  q = {Q}")
    print("  (X, Y) 결합분포")
    for i in range(2):
        for j in range(2):
            print(f"    P(X={i}, Y={j}) = {xy[i, j]:.2f}   "
                  f"P(X={i})P(Y={j}) = {xy.sum(1)[i] * xy.sum(0)[j]:.2f}")
    print("  (X, Z) 결합분포")
    for i in range(2):
        for j in range(3):
            print(f"    P(X={i}, Z={j}) = {xz[i, j]:.2f}   "
                  f"P(X={i})P(Z={j}) = {xz.sum(1)[i] * xz.sum(0)[j]:.3f}")


if __name__ == "__main__":
    main()
