r"""표본공간·사건·표본의 관계를 담은 그림을 생성한다.

docs/ch03/probability/sample_space.md 의 정리 1 에 실리는
img/sample_space_event.png 를 만든다. 계산이 아니라 정의를 그린 것이므로
페이지에는 코드 없이 그림만 싣는다.

  · 바깥 상자        = 표본공간 Ω
  · 안쪽 타원        = 사건 A ⊆ Ω
  · 점 ω            = 표본. 일부는 A 안에, 일부는 Ω \ A 에 놓인다.

실행:  python3 scripts/make_sample_space_event_figure.py   (저장소 최상위에서)
필요:  matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

# === Rendering ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

OMEGA_FACE, OMEGA_EDGE = "#F4F6F8", "#37474F"
A_FACE, A_EDGE = "#DCEBFB", "#1565C0"
DOT_IN, DOT_OUT = "#D32F2F", "#546E7A"

OUT = "docs/ch03/probability/img/sample_space_event.png"

# 타원 A: 중심과 반축. 점의 안/밖 판정이 눈으로 분명하도록 넉넉히 잡았다.
CX, CY, RX, RY = 3.5, 2.9, 2.0, 1.45

# (x, y, 라벨 위치 오프셋) — 겹치지 않도록 손으로 배치했다.
INSIDE = [(2.35, 3.60), (3.10, 2.50), (4.35, 3.30), (3.90, 2.15)]
OUTSIDE = [(6.60, 4.30), (8.40, 3.25), (7.10, 1.50), (1.40, 1.10)]


def main():
    fig, ax = plt.subplots(figsize=(9, 5.4))

    # 표본공간 Ω — 모든 결과를 담는 바깥 상자
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 9.0, 4.9,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                facecolor=OMEGA_FACE, edgecolor=OMEGA_EDGE,
                                linewidth=1.8, zorder=1))

    # 사건 A — Ω 의 부분집합
    ax.add_patch(Ellipse((CX, CY), 2 * RX, 2 * RY,
                         facecolor=A_FACE, edgecolor=A_EDGE,
                         linewidth=1.8, zorder=2))

    # 표본 ω — 개별 결과. A 안의 것과 밖의 것을 색으로 구별한다.
    for k, (x, y) in enumerate(INSIDE, start=1):
        ax.plot(x, y, "o", color=DOT_IN, markersize=7, zorder=4)
        ax.text(x + 0.16, y + 0.16, rf"$\omega_{{{k}}}$", fontsize=12,
                color=DOT_IN, ha="left", va="bottom", zorder=4)
    for k, (x, y) in enumerate(OUTSIDE, start=len(INSIDE) + 1):
        ax.plot(x, y, "o", color=DOT_OUT, markersize=7, zorder=4)
        ax.text(x + 0.16, y + 0.16, rf"$\omega_{{{k}}}$", fontsize=12,
                color=DOT_OUT, ha="left", va="bottom", zorder=4)

    # 이름 붙이기
    ax.text(0.95, 4.95, r"$\Omega$", fontsize=21, color=OMEGA_EDGE,
            ha="left", va="top", zorder=5)
    ax.text(1.55, 4.80, "표본공간 — 일어날 수 있는 모든 결과", fontsize=11.5,
            color=OMEGA_EDGE, ha="left", va="top", zorder=5)
    ax.text(CX, 3.95, r"$A$", fontsize=18, color=A_EDGE,
            ha="center", va="center", zorder=5)
    ax.text(CX, 3.58, "사건", fontsize=11.5, color=A_EDGE,
            ha="center", va="center", zorder=5)
    ax.text(8.15, 0.95, r"$\Omega \setminus A = A^{c}$", fontsize=13,
            color=DOT_OUT, ha="center", va="center", zorder=5)

    # 표본 하나를 가리켜 ω 가 무엇인지 못 박는다.
    ax.annotate("표본 $\\omega$ — 결과 하나", xy=(3.90, 2.15),
                xytext=(5.35, 0.95), fontsize=11.5, color=DOT_IN,
                ha="center", va="center", zorder=5,
                arrowprops=dict(arrowstyle="->", color=DOT_IN,
                                linewidth=1.3,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xlim(0, 10)
    ax.set_ylim(0.2, 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(r"표본공간 $\Omega$, 사건 $A \subseteq \Omega$, 표본 $\omega$",
                 fontsize=13.5, pad=2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
