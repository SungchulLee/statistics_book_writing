r"""11.6절 「그림으로 보는 통계학」 — 비어 있던 칸.

교호작용 항 $\beta_3 x_1 x_2$ 는 **두 변수가 모두 움직여야** 자료에서 보인다.
$2 \times 2$ 가운데 두 칸이 비면 계획행렬의 계수가 모자라고, 그 항은 추정되지
않은 채 절편에 섞여 들어간다.

$1989$ 년에 베일리의 연구진이 그 상황에 놓였다. 알코올이 펠로디핀에 영향을
주는지 보려고 알코올을 먹이고 안 먹이고를 갈랐는데, 알코올 맛을 가리려고
**모든 참가자에게** 자몽주스를 함께 주었다. 자몽은 움직이지 않았다 — $2 \times 2$
의 한 열만 채워진 것이다. 그래서 자몽의 효과는 「약이 문헌보다 세 배 세다」는
**모두에게 같은 어긋남**으로만 나타났다.

다음 날 베일리가 혼자 물과 자몽주스로 하루씩 먹어 비는 칸을 채웠고, 그제야
같은 $3$ 배가 **자몽의 효과**로 읽혔다.

이 스크립트가 그리는 것:

  ① 두 설계의 $2 \times 2$ 칸과 계획행렬의 계수. 넷 가운데 둘이 비면 계수가
     $4$ 가 아니라 $3$ 이고 교호작용 열이 다른 열들의 결합이 된다
  ② 같은 $3$ 배가 설계에 따라 「모두가 어긋난 것」으로도, 「자몽의 효과」로도
     보인다는 것

계수(rank)는 직접 계산한다. $3$ 배는 베일리 자신의 $2013$ 년 총설이 적은 값이다.

만드는 파일:

  ch11/interaction_polynomial/img/grapefruit_cell.png

실행:  python3 scripts/make_ch13_grapefruit_cell.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/interaction_polynomial/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

FOLD = 3.0        # 베일리 외(2013)가 적은 「물의 세 배」


def design(cells):
    """채워진 칸 목록으로 계획행렬 [1, x1, x2, x1*x2] 을 만든다."""
    rows = [[1.0, a, g, a * g] for a, g in cells]
    return np.array(rows)


def main():
    both_rows = [(0.0, 1.0), (1.0, 1.0)]          # 알코올만 갈랐다 (자몽은 늘 1)
    bailey = [(0.0, 0.0), (0.0, 1.0)]             # 자몽만 갈랐다 (알코올 없음)
    full = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for name, cells in [("1989 년의 설계", both_rows),
                        ("베일리의 하루 실험", bailey),
                        ("네 칸을 다 채웠다면", full)]:
        X = design(cells)
        r = np.linalg.matrix_rank(X)
        # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
        print(f"{name:20s} 칸 {len(cells)} 개   계획행렬 {X.shape}  계수 {r}  "
              f"(열 {X.shape[1]} 개 가운데 {X.shape[1] - r} 개가 남는다)")

    X = design(both_rows)
    print("\n1989 년의 설계에서 자몽 열과 교호작용 열은 절편·알코올 열의 결합이다:")
    print(f"  자몽 열      = 절편 열                     -> {np.allclose(X[:, 2], X[:, 0])}")
    print(f"  교호작용 열   = 알코올 열                   -> {np.allclose(X[:, 3], X[:, 1])}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    # ── ① 두 설계의 네 칸 ──
    ax = axes[0]
    ax.set_xlim(-0.35, 5.08)
    ax.set_ylim(-1.25, 2.55)
    ax.axis("off")

    def grid(x0, y0, filled, title, note, col):
        ax.text(x0 + 1.0, y0 + 2.18, title, ha="center", fontsize=12.2,
                color=INK, fontweight="bold")
        for i, g in enumerate([0, 1]):          # 가로: 자몽 없음/있음
            for j, a in enumerate([0, 1]):      # 세로: 아래가 없음, 위가 있음
                on = (a, g) in filled
                ax.add_patch(Rectangle((x0 + i, y0 + j), 0.96, 0.96,
                                       facecolor=col if on else "white",
                                       alpha=0.75 if on else 1.0,
                                       edgecolor=MUTED if not on else col, lw=1.8,
                                       hatch="" if on else "//"))
                ax.text(x0 + i + 0.48, y0 + j + 0.48, "잰 곳" if on else "비었다",
                        ha="center", va="center", fontsize=10.8,
                        color="white" if on else MUTED,
                        fontweight="bold" if on else "normal")
        ax.text(x0 + 0.48, y0 - 0.26, "자몽 없음", ha="center", fontsize=10.4,
                color=INK)
        ax.text(x0 + 1.48, y0 - 0.26, "자몽 있음", ha="center", fontsize=10.4,
                color=INK)
        ax.text(x0 - 0.1, y0 + 1.48, "알코올\n있음", ha="right", va="center",
                fontsize=10.4, color=INK)
        ax.text(x0 - 0.1, y0 + 0.48, "알코올\n없음", ha="right", va="center",
                fontsize=10.4, color=INK)
        ax.text(x0 + 1.0, y0 - 0.78, note, ha="center", fontsize=11, color=col,
                fontweight="bold")

    grid(0.45, 0.0, {(0, 1), (1, 1)}, "$1989$ 년의 설계",
         "계획행렬의 계수가 $2$ — 자몽 열은 절편 열과\n같고 교호작용 열은 알코올 열과 같다",
         RED)
    grid(2.95, 0.0, {(0, 0), (0, 1)}, "다음 날, 혼자",
         "자몽이 움직인다 — 그 차이가\n비로소 계수로 읽힌다", TEAL)
    ax.set_title("① 넷 가운데 둘이 비어 있었다", fontsize=12.8, color=INK, pad=14)

    # ── ② 같은 3 배, 다른 읽기 ──
    ax = axes[1]
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=":")
    ax.text(2.35, 0.52, "문헌이 말하던 농도", ha="center", fontsize=10.6,
            color=MUTED)

    ax.bar([0.6, 1.4], [FOLD, FOLD], width=0.55, color=RED, alpha=0.5,
           edgecolor=RED, lw=1.6)
    ax.text(1.0, FOLD + 0.22, "두 팔 모두 자몽주스를 마셨다", ha="center",
            fontsize=11.2, color=RED, fontweight="bold")
    ax.text(0.6, FOLD / 2, "알코올\n없음", ha="center", va="center", fontsize=10.6,
            color="white", fontweight="bold")
    ax.text(1.4, FOLD / 2, "알코올\n있음", ha="center", va="center", fontsize=10.6,
            color="white", fontweight="bold")
    ax.annotate("", xy=(0.25, FOLD), xytext=(0.25, 1.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(0.17, (1 + FOLD) / 2, f"${FOLD:.0f}$ 배", ha="right", va="center",
            fontsize=12, color=INK, fontweight="bold")

    ax.bar([3.0], [1.0], width=0.55, color=BLUE, alpha=0.8)
    ax.bar([3.8], [FOLD], width=0.55, color=TEAL, alpha=0.8)
    ax.text(3.0, 1.0 + 0.16, "물", ha="center", fontsize=11, color=BLUE,
            fontweight="bold")
    ax.text(3.8, FOLD + 0.16, "자몽주스", ha="center", fontsize=11, color=TEAL,
            fontweight="bold")
    ax.annotate("", xy=(4.35, FOLD), xytext=(4.35, 1.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(4.43, (1 + FOLD) / 2, f"${FOLD:.0f}$ 배", ha="left", va="center",
            fontsize=12, color=INK, fontweight="bold")

    ax.text(1.0, -0.42, "「약이 문헌보다 세 배 세다」", ha="center", fontsize=11.6,
            color=RED)
    ax.text(3.4, -0.42, "「자몽이 세 배로 올린다」", ha="center", fontsize=11.6,
            color=TEAL)
    ax.set_xlim(-0.45, 5.0)
    ax.set_ylim(-0.75, 4.1)
    ax.set_xticks([])
    ax.set_ylabel("펠로디핀의 전신 농도 (문헌 $= 1$)", fontsize=11.5)
    ax.set_title("② 같은 $3$ 배, 두 가지 읽기", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    fig.suptitle("모두에게 똑같이 준 것은 계수가 되지 않는다", fontsize=14.5)
    fig.text(0.5, -0.045,
             "왼쪽 계수는 계획행렬에서 직접 계산했다. $3$ 배는 베일리 외가 "
             "$2013$ 년 총설에 적은 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "grapefruit_cell.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
