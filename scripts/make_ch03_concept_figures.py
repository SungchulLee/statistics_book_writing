r"""3장(확률)의 개념 그림들을 생성한다.

계산 결과가 아니라 **정의를 그린** 그림들이므로 페이지에는 코드 없이 그림만
싣는다. 표본공간을 상자로, 사건을 그 안의 도형으로 그리는 방식을
docs/ch03/probability/img/sample_space_event.png 와 통일했다.

만드는 파일:

  ch03/probability/img/axioms_venn.png             공리의 따름정리 세 가지
  ch03/probability/img/conditional_narrowing.png   조건 = 표본공간 갈아 끼우기
  ch03/probability/img/total_probability.png       분할과 전확률의 법칙
  ch03/probability/img/bayes_base_rate.png         기저율의 지배
  ch03/probability/img/independent_vs_disjoint.png 독립과 배반
  ch03/probability/img/conditional_independence.png 공통 결과와 공통 원인
  ch03/rv/img/random_variable_map.png              확률변수 = Ω → R 함수
  ch03/rv/img/expectation_balance.png              기댓값 = 무게중심

실행:  python3 scripts/make_ch03_concept_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Rectangle)

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

OMEGA_FACE, OMEGA_EDGE = "#F4F6F8", "#37474F"
A_FACE, A_EDGE = "#DCEBFB", "#1565C0"          # 사건 A — 파랑
B_FACE, B_EDGE = "#FFE0B2", "#E65100"          # 사건 B — 주황
HL_FACE, HL_EDGE = "#C5E1A5", "#33691E"        # 강조(교집합 등) — 초록
MUTED = "#90A4AE"
RED = "#D32F2F"

PROB = "docs/ch03/probability/img/"
RV = "docs/ch03/rv/img/"


def omega_box(ax, x=0.4, y=0.4, w=9.2, h=5.2, label=True, alpha=1.0):
    """표본공간 Ω 를 나타내는 둥근 상자."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                facecolor=OMEGA_FACE, edgecolor=OMEGA_EDGE,
                                linewidth=1.6, alpha=alpha, zorder=1))
    if label:
        ax.text(x + 0.35, y + h - 0.3, r"$\Omega$", fontsize=16,
                color=OMEGA_EDGE, ha="left", va="top", zorder=6)


def lens_fill(ax, c1, w1, c2, w2, color, zorder=5):
    """두 타원이 겹치는 렌즈 모양 영역을 가로줄로 채운다.

    matplotlib 에는 도형의 교집합을 칠하는 기능이 없으므로, 높이마다
    두 타원의 가로 구간을 구해 겹치는 만큼만 선분을 그린다.
    """
    for y in np.linspace(c1[1] - w1[1] / 2, c1[1] + w1[1] / 2, 500):
        t1 = 1 - ((y - c1[1]) / (w1[1] / 2)) ** 2
        t2 = 1 - ((y - c2[1]) / (w2[1] / 2)) ** 2
        if t1 <= 0 or t2 <= 0:
            continue
        lo = max(c1[0] - (w1[0] / 2) * np.sqrt(t1),
                 c2[0] - (w2[0] / 2) * np.sqrt(t2))
        hi = min(c1[0] + (w1[0] / 2) * np.sqrt(t1),
                 c2[0] + (w2[0] / 2) * np.sqrt(t2))
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.6,
                    zorder=zorder)


def blank(ax, xlim=(0, 10), ylim=(0, 6)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ==================================================================
# 1. 공리의 따름정리 — 여집합·단조성·포함배제
# ==================================================================
def axioms_venn():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) 여집합
    ax = axes[0]
    omega_box(ax)
    ax.add_patch(Ellipse((3.6, 2.9), 3.6, 2.6, facecolor=A_FACE,
                         edgecolor=A_EDGE, linewidth=1.6, zorder=3))
    ax.text(3.6, 2.9, r"$A$", fontsize=17, color=A_EDGE,
            ha="center", va="center", zorder=5)
    ax.text(7.6, 2.9, r"$A^{c}$", fontsize=17, color=MUTED,
            ha="center", va="center", zorder=5)
    ax.set_title("여집합\n$P(A) + P(A^{c}) = 1$", fontsize=12.5, pad=8)
    blank(ax)

    # (b) 단조성
    ax = axes[1]
    omega_box(ax)
    ax.add_patch(Ellipse((4.4, 2.9), 5.6, 3.8, facecolor=B_FACE,
                         edgecolor=B_EDGE, linewidth=1.6, zorder=3))
    ax.add_patch(Ellipse((3.7, 2.9), 2.6, 2.0, facecolor=A_FACE,
                         edgecolor=A_EDGE, linewidth=1.6, zorder=4))
    ax.text(3.7, 2.9, r"$A$", fontsize=16, color=A_EDGE,
            ha="center", va="center", zorder=5)
    ax.text(6.3, 3.9, r"$B$", fontsize=16, color=B_EDGE,
            ha="center", va="center", zorder=5)
    ax.set_title("단조성\n$A \\subseteq B \\;\\Rightarrow\\; P(A) \\leq P(B)$",
                 fontsize=12.5, pad=8)
    blank(ax)

    # (c) 포함배제
    ax = axes[2]
    omega_box(ax)
    ax.add_patch(Ellipse((3.9, 2.9), 4.2, 3.0, facecolor=A_FACE,
                         edgecolor=A_EDGE, linewidth=1.6, alpha=0.9, zorder=3))
    ax.add_patch(Ellipse((6.1, 2.9), 4.2, 3.0, facecolor=B_FACE,
                         edgecolor=B_EDGE, linewidth=1.6, alpha=0.75, zorder=3))
    # 겹친 부분을 눈에 띄게 칠한다
    lens_fill(ax, (3.9, 2.9), (4.2, 3.0), (6.1, 2.9), (4.2, 3.0), "#C5CAE9",
              zorder=4)
    ax.add_patch(Ellipse((3.9, 2.9), 4.2, 3.0, facecolor="none",
                         edgecolor=A_EDGE, linewidth=1.6, zorder=5))
    ax.add_patch(Ellipse((6.1, 2.9), 4.2, 3.0, facecolor="none",
                         edgecolor=B_EDGE, linewidth=1.6, zorder=5))
    ax.text(2.8, 2.9, r"$A$", fontsize=16, color=A_EDGE,
            ha="center", va="center", zorder=6)
    ax.text(7.2, 2.9, r"$B$", fontsize=16, color=B_EDGE,
            ha="center", va="center", zorder=6)
    ax.annotate("두 번 세어진 $A \\cap B$", xy=(5.0, 2.9), xytext=(5.0, 0.85),
                fontsize=11, color=RED, ha="center", va="center", zorder=7,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))
    ax.set_title("포함배제\n$P(A \\cup B) = P(A) + P(B) - P(A \\cap B)$",
                 fontsize=12.5, pad=8)
    blank(ax)

    fig.tight_layout()
    save(fig, PROB + "axioms_venn.png")


# ==================================================================
# 2. 조건부확률 — 표본공간을 B 로 갈아 끼운다
# ==================================================================
def conditional_narrowing():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    AC, BC = (3.9, 2.9), (6.2, 2.9)
    AW, BW = (4.2, 3.0), (4.2, 3.0)

    # (a) 조건을 걸기 전
    ax = axes[0]
    omega_box(ax)
    ax.add_patch(Ellipse(AC, *AW, facecolor=A_FACE, edgecolor=A_EDGE,
                         linewidth=1.6, alpha=0.9, zorder=3))
    ax.add_patch(Ellipse(BC, *BW, facecolor=B_FACE, edgecolor=B_EDGE,
                         linewidth=1.6, alpha=0.7, zorder=3))
    ax.text(2.8, 2.9, r"$A$", fontsize=16, color=A_EDGE, ha="center",
            va="center", zorder=6)
    ax.text(7.3, 2.9, r"$B$", fontsize=16, color=B_EDGE, ha="center",
            va="center", zorder=6)
    ax.text(5.0, 0.95, "분모는 $\\Omega$ 전체 — $P(\\Omega) = 1$", fontsize=11,
            color=OMEGA_EDGE, ha="center", va="center", zorder=6)
    ax.set_title("조건을 걸기 전", fontsize=13, pad=8)
    blank(ax)

    # (b) B 가 일어났음을 알고 난 뒤
    ax = axes[1]
    omega_box(ax, alpha=0.45)
    # B 바깥은 흐리게, B 는 그대로 — "가능성이 B 안으로 좁혀졌다"
    ax.add_patch(Ellipse(AC, *AW, facecolor=A_FACE, edgecolor=A_EDGE,
                         linewidth=1.2, alpha=0.3, linestyle="--", zorder=3))
    ax.add_patch(Ellipse(BC, *BW, facecolor=B_FACE, edgecolor=B_EDGE,
                         linewidth=2.4, alpha=0.85, zorder=4))
    # 남은 가능성은 A ∩ B 뿐이라는 뜻에서 교집합을 초록으로 덮는다
    lens_fill(ax, AC, AW, BC, BW, HL_FACE, zorder=5)
    ax.text(2.7, 2.9, r"$A$", fontsize=16, color=A_EDGE, alpha=0.45,
            ha="center", va="center", zorder=6)
    ax.text(5.05, 2.9, r"$A \cap B$", fontsize=13, color=HL_EDGE,
            ha="center", va="center", zorder=7)
    ax.text(7.4, 2.9, r"$B$", fontsize=16, color=B_EDGE, ha="center",
            va="center", zorder=7)
    ax.text(5.0, 0.95,
            "분모가 $B$ 로 바뀐다 — $P(A \\mid B) = P(A \\cap B) / P(B)$",
            fontsize=11, color=HL_EDGE, ha="center", va="center", zorder=7)
    ax.set_title("$B$ 가 일어났음을 알고 난 뒤", fontsize=13, pad=8)
    blank(ax)

    fig.suptitle("조건을 건다는 것은 표본공간을 갈아 끼우는 일이다",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()
    save(fig, PROB + "conditional_narrowing.png")


# ==================================================================
# 3. 전확률의 법칙 — 분할 위의 합
# ==================================================================
def total_probability():
    fig, ax = plt.subplots(figsize=(10, 5.2))

    x0, y0, w, h = 0.5, 0.7, 9.0, 4.6
    edges = [x0, x0 + 2.7, x0 + 4.6, x0 + 6.6, x0 + w]
    shades = ["#E3F2FD", "#FFF3E0", "#E8F5E9", "#F3E5F5"]
    names = [r"$B_1$", r"$B_2$", r"$B_3$", r"$B_4$"]

    frame = FancyBboxPatch((x0, y0), w, h,
                           boxstyle="round,pad=0,rounding_size=0.25",
                           facecolor="white", edgecolor=OMEGA_EDGE,
                           linewidth=1.6, zorder=1)
    ax.add_patch(frame)
    for i in range(4):
        strip = Rectangle((edges[i], y0), edges[i + 1] - edges[i], h,
                          facecolor=shades[i], edgecolor="none", zorder=2)
        ax.add_patch(strip)
        strip.set_clip_path(frame)      # 둥근 모서리 밖으로 삐져나가지 않게
        if i:
            ax.plot([edges[i], edges[i]], [y0, y0 + h], color=OMEGA_EDGE,
                    linewidth=1.1, zorder=4)
        ax.text((edges[i] + edges[i + 1]) / 2, y0 + h - 0.45, names[i],
                fontsize=15, color=OMEGA_EDGE, ha="center", va="center",
                zorder=6)
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor="none", edgecolor=OMEGA_EDGE,
                                linewidth=1.8, zorder=5))

    # A 는 네 조각을 가로지르는 띠
    cy = y0 + 1.9
    ax.add_patch(Ellipse((x0 + w / 2, cy), 8.0, 2.0, facecolor=A_FACE,
                         edgecolor=A_EDGE, linewidth=1.8, alpha=0.62, zorder=7))
    ax.text(x0 + 1.4, cy, r"$A$", fontsize=17, color=A_EDGE,
            ha="center", va="center", zorder=9)
    for i in range(1, 4):
        ax.plot([edges[i], edges[i]],
                [cy - 1.0 * np.sqrt(max(0.0, 1 - ((edges[i] - (x0 + w / 2))
                                                  / 4.0) ** 2)),
                 cy + 1.0 * np.sqrt(max(0.0, 1 - ((edges[i] - (x0 + w / 2))
                                                  / 4.0) ** 2))],
                color=A_EDGE, linewidth=1.0, linestyle="--", zorder=9)

    ax.annotate(r"$A \cap B_2$", xy=(edges[1] + 0.9, cy - 0.55),
                xytext=(edges[1] + 0.5, y0 - 0.45), fontsize=11.5,
                color=RED, ha="center", va="center", zorder=10,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))
    ax.text(x0 + w / 2 + 2.2, y0 - 0.45,
            "$P(A) = \\sum_i P(A \\mid B_i)\\,P(B_i)$",
            fontsize=12.5, color=OMEGA_EDGE, ha="center", va="center")

    ax.set_title("전확률의 법칙 — $\\Omega$ 를 분할하고 조각마다 센다",
                 fontsize=13.5, pad=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.9, 5.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save(fig, PROB + "total_probability.png")


# ==================================================================
# 4. 기저율의 지배 — 넓이로 본 의학 검사
# ==================================================================
def bayes_base_rate():
    prev, sens, fpr = 0.01, 0.95, 0.10
    tp, fp = sens * prev, fpr * (1 - prev)          # 0.0095, 0.099
    ppv = tp / (tp + fp)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) 10,000 명을 실제 사람 수로 갈라 보는 나무
    ax = axes[0]

    def box(xy, lines, edge, face, w=2.15, h=0.92, fs=11):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.6, zorder=4))
        ax.text(x, y, lines, fontsize=fs, color=edge, ha="center",
                va="center", zorder=5, linespacing=1.35)

    def edge_arrow(p, q, label, color, dx=0.0, dy=0.22):
        # 나무의 가지는 선으로 충분하다. 화살표 머리는 상자에 가려 보이지 않는다.
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, linewidth=1.3,
                zorder=3)
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label,
                fontsize=10, color=color, ha="center", va="center", zorder=5)

    root = (1.35, 3.0)
    sick, well = (4.9, 5.0), (4.9, 1.2)
    sp, sn = (8.6, 5.9), (8.6, 4.1)
    wp, wn = (8.6, 2.1), (8.6, 0.3)

    box(root, "검사받은\n10,000명", OMEGA_EDGE, "white", w=2.0)
    box(sick, "질병\n100명", RED, "#FFEBEE", w=1.9)
    box(well, "건강\n9,900명", "#607D8B", "#ECEFF1", w=1.9)
    box(sp, "양성 95명", RED, "#EF9A9A", w=2.1, h=0.8)
    box(sn, "음성 5명", "#B0BEC5", "white", w=2.1, h=0.8)
    box(wp, "양성 990명", "#BF360C", "#FFCC80", w=2.1, h=0.8)
    box(wn, "음성 8,910명", "#B0BEC5", "white", w=2.1, h=0.8)

    edge_arrow(root, sick, "1%", RED, dx=-0.15, dy=0.3)
    edge_arrow(root, well, "99%", "#607D8B", dx=-0.15, dy=-0.35)
    edge_arrow(sick, sp, "민감도 95%", RED, dy=0.3)
    edge_arrow(sick, sn, "5%", "#90A4AE", dy=-0.32)
    edge_arrow(well, wp, "위양성 10%", "#BF360C", dy=0.32)
    edge_arrow(well, wn, "90%", "#90A4AE", dy=-0.3)

    # 두 양성 상자를 묶어 분모를 만든다
    ax.plot([9.75, 10.1, 10.1, 9.75], [5.9, 5.9, 2.1, 2.1], color=OMEGA_EDGE,
            linewidth=1.2, zorder=3)
    ax.text(10.3, 4.0, "양성\n1,085명", fontsize=11, color=OMEGA_EDGE,
            ha="left", va="center", zorder=5, linespacing=1.35)

    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.4, 6.6)
    ax.axis("off")
    ax.set_title("사람 수로 세어 보면 — 10,000명의 갈림길",
                 fontsize=12.5, pad=8)

    # (b) 양성 판정을 받은 사람들의 구성
    ax = axes[1]
    ax.barh([0], [tp / (tp + fp)], color=RED, edgecolor="white")
    ax.barh([0], [fp / (tp + fp)], left=[tp / (tp + fp)], color="#FFB74D",
            edgecolor="white")
    ax.text(ppv / 2, 0, f"{ppv:.1%}", fontsize=12, color="white",
            ha="center", va="center", fontweight="bold")
    ax.text(ppv + (1 - ppv) / 2, 0, f"{1 - ppv:.1%}", fontsize=13,
            color="#5D4037", ha="center", va="center", fontweight="bold")
    ax.text(ppv / 2, 0.62, "실제 질병\n95명", fontsize=10.5, color=RED,
            ha="center", va="center", linespacing=1.3)
    ax.text(ppv + (1 - ppv) / 2, 0.62, "건강한데 양성 990명", fontsize=11,
            color="#BF360C", ha="center", va="center")
    ax.text(0.5, -0.75, "P(질병 | 양성) = 95 / 1,085 = 8.8%",
            fontsize=12.5, color=OMEGA_EDGE, ha="center", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, 1.1)
    ax.axis("off")
    ax.set_title("양성 판정을 받은 1,085명의 구성", fontsize=12.5, pad=8)

    fig.suptitle("기저율의 지배 — 민감도 95%, 위양성률 10%, 유병률 1%",
                 fontsize=13.5, y=1.01)
    fig.tight_layout()
    save(fig, PROB + "bayes_base_rate.png")


# ==================================================================
# 5. 독립과 배반
# ==================================================================
def independent_vs_disjoint():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # (a) 배반
    ax = axes[0]
    omega_box(ax)
    ax.add_patch(Ellipse((3.0, 2.9), 3.0, 2.6, facecolor=A_FACE,
                         edgecolor=A_EDGE, linewidth=1.7, zorder=3))
    ax.add_patch(Ellipse((6.9, 2.9), 3.0, 2.6, facecolor=B_FACE,
                         edgecolor=B_EDGE, linewidth=1.7, zorder=3))
    ax.text(3.0, 2.9, r"$A$", fontsize=16, color=A_EDGE, ha="center",
            va="center", zorder=5)
    ax.text(6.9, 2.9, r"$B$", fontsize=16, color=B_EDGE, ha="center",
            va="center", zorder=5)
    ax.text(5.0, 4.6, "$A \\cap B = \\emptyset$", fontsize=12.5,
            color=OMEGA_EDGE, ha="center", va="center", zorder=5)
    ax.text(5.0, 0.95, "$A$ 를 알면 $B$ 는 일어나지 않았다 — 최대로 종속",
            fontsize=11, color=RED, ha="center", va="center", zorder=5)
    ax.set_title("배반 $P(A \\cap B) = 0$", fontsize=13, pad=8)
    blank(ax)

    # (b) 독립 — 단위정사각형의 띠
    ax = axes[1]
    pa, pb = 0.5, 0.4
    x0, y0, s = 2.7, 0.95, 4.65         # 정사각형의 왼쪽 아래 모서리와 한 변
    ax.add_patch(Rectangle((x0, y0), s, s, facecolor=OMEGA_FACE,
                           edgecolor=OMEGA_EDGE, linewidth=1.6, zorder=1))
    ax.add_patch(Rectangle((x0, y0), s * pa, s, facecolor=A_FACE,
                           edgecolor="none", alpha=0.85, zorder=2))
    ax.add_patch(Rectangle((x0, y0), s, s * pb, facecolor=B_FACE,
                           edgecolor="none", alpha=0.55, zorder=3))
    ax.add_patch(Rectangle((x0, y0), s * pa, s * pb, facecolor=HL_FACE,
                           edgecolor=HL_EDGE, linewidth=1.6, zorder=4))
    ax.add_patch(Rectangle((x0, y0), s, s, facecolor="none",
                           edgecolor=OMEGA_EDGE, linewidth=1.6, zorder=5))
    ax.text(x0 + s * pa / 2, y0 + s + 0.22, "$A$ : 가로 $0.5$", fontsize=11.5,
            color=A_EDGE, ha="center", va="bottom", zorder=6)
    ax.text(x0 - 0.35, y0 + s * pb / 2, "$B$ : 세로 $0.4$", fontsize=11.5,
            color=B_EDGE, ha="center", va="center", rotation=90, zorder=6)
    ax.text(x0 + s - 0.3, y0 + s - 0.3, r"$\Omega$", fontsize=15,
            color=OMEGA_EDGE, ha="right", va="top", zorder=6)
    ax.text(x0 + s * pa / 2, y0 + s * pb / 2, "$0.5 \\times 0.4$",
            fontsize=12, color=HL_EDGE, ha="center", va="center", zorder=6)
    ax.text(x0 + s / 2, y0 - 0.55,
            "$B$ 안에서 $A$ 가 차지하는 비율이 $\\Omega$ 에서와 같다",
            fontsize=11, color=HL_EDGE, ha="center", va="center", zorder=6)
    ax.set_title("독립 $P(A \\cap B) = P(A)P(B)$", fontsize=13, pad=8)
    blank(ax)

    fig.suptitle("독립과 배반은 닮은 말이 아니라 정반대다",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()
    save(fig, PROB + "independent_vs_disjoint.png")


# ==================================================================
# 6. 공통 결과와 공통 원인
# ==================================================================
def conditional_independence():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    def node(ax, xy, label, color, face):
        ax.add_patch(Circle(xy, 0.58, facecolor=face, edgecolor=color,
                            linewidth=1.8, zorder=4))
        ax.text(xy[0], xy[1], label, fontsize=15, color=color,
                ha="center", va="center", zorder=5)

    def arrow(ax, p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=17,
                                     color=OMEGA_EDGE, linewidth=1.6,
                                     shrinkA=20, shrinkB=20, zorder=3))

    # (a) 공통 결과 — 충돌자
    ax = axes[0]
    A, B, C = (2.0, 4.0), (5.6, 4.0), (3.8, 1.5)
    arrow(ax, A, C)
    arrow(ax, B, C)
    node(ax, A, r"$A$", A_EDGE, A_FACE)
    node(ax, B, r"$B$", B_EDGE, B_FACE)
    node(ax, C, r"$C$", HL_EDGE, HL_FACE)
    ax.text(0.9, 4.9, "화재", fontsize=10.5, color=A_EDGE, ha="center")
    ax.text(6.7, 4.9, "탄 토스트", fontsize=10.5, color=B_EDGE, ha="center")
    ax.text(3.8, 0.55, "경보", fontsize=10.5, color=HL_EDGE, ha="center")
    ax.text(3.8, 5.95, "조건을 걸지 않으면  독립", fontsize=11.5,
            color=OMEGA_EDGE, ha="center")
    ax.text(3.8, -0.35, "$C$ 로 조건을 걸면  종속  (해명 효과)",
            fontsize=11.5, color=RED, ha="center")
    ax.set_title("공통 결과 — 조건을 걸면 얽힌다", fontsize=13, pad=10)
    blank(ax, xlim=(-0.4, 8.0), ylim=(-1.0, 6.6))

    # (b) 공통 원인 — 갈래
    ax = axes[1]
    C, A, B = (3.8, 4.3), (2.0, 1.7), (5.6, 1.7)
    arrow(ax, C, A)
    arrow(ax, C, B)
    node(ax, C, r"$C$", HL_EDGE, HL_FACE)
    node(ax, A, r"$A$", A_EDGE, A_FACE)
    node(ax, B, r"$B$", B_EDGE, B_FACE)
    ax.text(3.8, 5.25, "어느 동전인가", fontsize=10.5, color=HL_EDGE,
            ha="center")
    ax.text(0.9, 0.85, "첫 던지기", fontsize=10.5, color=A_EDGE, ha="center")
    ax.text(6.7, 0.85, "둘째 던지기", fontsize=10.5, color=B_EDGE, ha="center")
    ax.text(3.8, 5.95, "조건을 걸지 않으면  종속", fontsize=11.5,
            color=RED, ha="center")
    ax.text(3.8, -0.35, "$C$ 로 조건을 걸면  독립  (혼합)",
            fontsize=11.5, color=HL_EDGE, ha="center")
    ax.set_title("공통 원인 — 조건을 걸면 풀린다", fontsize=13, pad=10)
    blank(ax, xlim=(-0.4, 8.0), ylim=(-1.0, 6.6))

    fig.suptitle("조건을 거는 것은 종속을 만들 수도, 없앨 수도 있다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, PROB + "conditional_independence.png")


# ==================================================================
# 7. 확률변수 = Ω 에서 실직선으로 가는 함수
# ==================================================================
def random_variable_map():
    fig, ax = plt.subplots(figsize=(12, 5.6))

    outcomes = ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
    counts = [w.count("H") for w in outcomes]

    # 왼쪽: 표본공간
    bx, by, bw, bh = 0.3, 0.6, 3.6, 5.0
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh,
                                boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=OMEGA_FACE, edgecolor=OMEGA_EDGE,
                                linewidth=1.6, zorder=1))
    ax.text(bx + 0.3, by + bh - 0.25, r"$\Omega$", fontsize=16,
            color=OMEGA_EDGE, ha="left", va="top", zorder=5)

    pos = {}
    for i, w in enumerate(outcomes):
        px = bx + 0.85 + 1.9 * (i % 2)
        py = by + bh - 1.05 - 0.95 * (i // 2)
        pos[w] = (px, py)
        color = RED if counts[i] == 1 else OMEGA_EDGE
        ax.plot(px, py, "o", color=color, markersize=6, zorder=4)
        ax.text(px + 0.14, py, w, fontsize=10.5, color=color,
                ha="left", va="center", zorder=4)

    # 오른쪽: 실직선과 그 위에 쌓인 벽돌
    line_y = 1.1
    tick_x = {k: 6.2 + 1.35 * k for k in range(4)}
    ax.annotate("", xy=(11.4, line_y), xytext=(5.4, line_y),
                arrowprops=dict(arrowstyle="->", color=OMEGA_EDGE,
                                linewidth=1.6))
    ax.text(11.5, line_y, r"$\mathbb{R}$", fontsize=14, color=OMEGA_EDGE,
            ha="left", va="center")

    stack = {k: 0 for k in range(4)}
    bw_, bh_ = 0.78, 0.42
    for w, k in zip(outcomes, counts):
        x = tick_x[k]
        y = line_y + 0.14 + stack[k] * (bh_ + 0.07)
        color = "#FFCDD2" if k == 1 else "#CFD8DC"
        edge = RED if k == 1 else OMEGA_EDGE
        ax.add_patch(Rectangle((x - bw_ / 2, y), bw_, bh_, facecolor=color,
                               edgecolor=edge, linewidth=1.1, zorder=4))
        ax.text(x, y + bh_ / 2, w, fontsize=8, color=edge, ha="center",
                va="center", zorder=5)
        stack[k] += 1

    for k in range(4):
        ax.plot([tick_x[k], tick_x[k]], [line_y - 0.13, line_y + 0.13],
                color=OMEGA_EDGE, linewidth=1.4, zorder=5)
        ax.text(tick_x[k], line_y - 0.32, f"${k}$", fontsize=12,
                color=OMEGA_EDGE, ha="center", va="top")
        ax.text(tick_x[k], line_y - 0.95, f"$\\frac{{{stack[k]}}}{{8}}$",
                fontsize=12, color=RED if k == 1 else OMEGA_EDGE,
                ha="center", va="center")
    ax.text(5.3, line_y - 0.95, "$P(X = k)$", fontsize=11.5,
            color=OMEGA_EDGE, ha="right", va="center")

    # Ω 에서 실직선으로 가는 화살표
    for w, k in zip(outcomes, counts):
        px, py = pos[w]
        highlight = (k == 1)
        ax.add_patch(FancyArrowPatch((px + 0.62, py), (tick_x[k], line_y + 0.1),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=RED if highlight else MUTED,
                                     alpha=0.95 if highlight else 0.45,
                                     linewidth=1.3 if highlight else 1.0,
                                     connectionstyle="arc3,rad=0.07",
                                     shrinkA=2, shrinkB=2, zorder=2))

    ax.text(4.9, 5.25, r"$X : \Omega \longrightarrow \mathbb{R}$", fontsize=15,
            color=OMEGA_EDGE, ha="center", va="center")
    ax.text(4.9, 4.7, "앞면의 개수", fontsize=11, color=OMEGA_EDGE,
            ha="center", va="center")
    ax.text(8.9, 4.0, "$HTT,\\; THT,\\; TTH$ 세 결과가\n같은 자리 $1$ 에 쌓인다",
            fontsize=11, color=RED, ha="center", va="center")

    ax.set_xlim(0, 12.2)
    ax.set_ylim(-0.4, 5.9)
    ax.axis("off")
    ax.set_title("확률변수는 결과를 수로 옮기고, 벽돌은 실직선 위에 다시 쌓인다",
                 fontsize=13.5, pad=6)
    fig.tight_layout()
    save(fig, RV + "random_variable_map.png")


# ==================================================================
# 8. 기댓값 = 무게중심
# ==================================================================
def expectation_balance():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))

    cases = [
        (np.arange(1, 7), np.full(6, 1 / 6), "공정한 주사위"),
        (np.array([1, 2, 3, 10]), np.array([0.4, 0.3, 0.2, 0.1]),
         "한쪽으로 치우친 분포"),
    ]

    for ax, (vals, probs, name) in zip(axes, cases):
        mu = float(np.sum(vals * probs))
        lo, hi = vals.min() - 1.1, vals.max() + 1.1

        ax.plot([lo, hi], [0, 0], color=OMEGA_EDGE, linewidth=2.2, zorder=3)
        for v, p in zip(vals, probs):
            ax.add_patch(Rectangle((v - 0.28, 0), 0.56, p * 3.2,
                                   facecolor=A_FACE, edgecolor=A_EDGE,
                                   linewidth=1.3, zorder=4))
            ax.text(v, p * 3.2 + 0.07, f"${p:.2f}$".rstrip("0").rstrip("."),
                    fontsize=9.5, color=A_EDGE, ha="center", va="bottom")
            ax.text(v, -0.16, f"${v}$", fontsize=11, color=OMEGA_EDGE,
                    ha="center", va="top")

        # 받침대를 무게중심에 놓는다
        ax.add_patch(Polygon([[mu, -0.04], [mu - 0.32, -0.62],
                              [mu + 0.32, -0.62]], closed=True,
                             facecolor=RED, edgecolor=RED, zorder=5))
        ax.text(mu, -0.82, f"$E[X] = {mu:g}$", fontsize=12, color=RED,
                ha="center", va="top")

        ax.set_xlim(lo - 0.3, hi + 0.3)
        ax.set_ylim(-1.5, 1.6)
        ax.axis("off")
        ax.set_title(name, fontsize=12.5, pad=6)

    axes[1].annotate("멀리 있는 값 하나가\n받침대를 끌어당긴다", xy=(9.6, 0.2),
                     xytext=(6.6, 1.05), fontsize=10.5, color=RED,
                     ha="center", va="center", linespacing=1.35,
                     arrowprops=dict(arrowstyle="->", color=RED,
                                     linewidth=1.2))

    fig.suptitle("기댓값은 확률이라는 무게가 균형을 이루는 점이다",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()
    save(fig, RV + "expectation_balance.png")


if __name__ == "__main__":
    axioms_venn()
    conditional_narrowing()
    total_probability()
    bayes_base_rate()
    independent_vs_disjoint()
    conditional_independence()
    random_variable_map()
    expectation_balance()
