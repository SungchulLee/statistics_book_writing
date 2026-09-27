r"""5장에서 그림이 없던 네 쪽의 그림을 생성한다.

  ch05/distributions/img/normal_standardization.png   표준화가 단위를 지운다
  ch05/distributions/img/f_construction.png           F 의 조립도와 두 겹의 정규성
  ch05/applications/img/diff_means_tree.png           다섯 갈래의 지도
  ch05/applications/img/diff_props_two_worlds.png     구간의 세계와 귀무가설의 세계

실행:  python3 scripts/make_ch05_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# === 공통 설정 — 3장 개념 그림과 같은 팔레트 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
MUTED = "#90A4AE"
RED = "#D32F2F"

DIST = "docs/ch05/distributions/img/"
APP = "docs/ch05/applications/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def phi(z):
    return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)


# ==================================================================
# 1. 표준화는 단위를 지운다
# ==================================================================
def normal_standardization():
    # 단위도 자릿수도 전혀 다른 세 정규모집단
    cases = [
        ("성인 몸무게", 70.0, 12.0, "kg", "70", "12", BLUE),
        ("같은 몸무게를 파운드로", 154.0, 26.5, "lb", "154", "26.5", ORANGE),
        ("베어링 지름", 0.0030, 0.00004, "m", "0.00300", "0.00004", GREEN),
    ]

    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.25], hspace=0.95,
                          wspace=0.22)

    for i, (name, mu, sd, unit, mu_s, sd_s, color) in enumerate(cases):
        ax = fig.add_subplot(gs[i, 0])
        x = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
        y = phi((x - mu) / sd) / sd
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, y, color=color, alpha=0.16)
        for k in (-1.96, 1.96):
            ax.axvline(mu + k * sd, color=color, linewidth=1.0,
                       linestyle="--", alpha=0.9)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=9, colors=INK, length=3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(f"{name}  —  $\\mu = {mu_s}$, $\\sigma = {sd_s}$ ({unit})",
                     fontsize=10.5, color=color, pad=4, loc="left")

    # 오른쪽: 표준화하면 셋이 한 분포로 포개진다
    ax = fig.add_subplot(gs[:, 1])
    z = np.linspace(-4, 4, 600)
    ax.plot(z, phi(z), color=INK, linewidth=1.0, alpha=0.35, zorder=5)
    ax.fill_between(z, phi(z), where=np.abs(z) >= 1.96, color=RED, alpha=0.25,
                    zorder=3)
    ax.fill_between(z, phi(z), where=np.abs(z) <= 1.96, color=INK, alpha=0.07,
                    zorder=2)

    # 세 분포를 표준화한 결과는 완전히 포개진다. 파선의 위상을 어긋나게 주어
    # 한 곡선 위에 세 색이 번갈아 나타나도록 그린다.
    for j, case in enumerate(cases):
        ax.plot(z, phi(z), color=case[-1], linewidth=2.6, alpha=0.95,
                linestyle=(j * 7, (7, 14)), solid_capstyle="butt",
                zorder=6 + j)

    # 임계선은 곡선까지만 그린다. 아래 설명글과 겹치지 않게 하려는 것이다.
    for k in (-1.96, 1.96):
        ax.plot([k, k], [0, 0.40], color=RED, linewidth=1.2, linestyle="--",
                zorder=7)
    ax.text(1.96, 0.415, "$+1.96$", fontsize=11, color=RED, ha="left",
            va="center")
    ax.text(-1.96, 0.415, "$-1.96$", fontsize=11, color=RED, ha="right",
            va="center")
    ax.text(0, 0.17, "가운데 $95\\%$", fontsize=12, color=INK, ha="center")
    ax.text(2.75, 0.035, "$2.5\\%$", fontsize=10.5, color=RED, ha="center")
    ax.text(-2.75, 0.035, "$2.5\\%$", fontsize=10.5, color=RED, ha="center")
    ax.text(0, -0.085, "세 분포를 표준화한 곡선이 완전히 포개져 있다", fontsize=11,
            color=INK, ha="center")

    ax.set_ylim(-0.11, 0.46)
    ax.set_xlim(-4.2, 4.2)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set_title("표준화한 뒤 — $Z \\sim N(0, 1)$ 하나", fontsize=12.5,
                 color=INK, pad=8)

    fig.suptitle("단위도 자릿수도 다른 세 분포가 표준화 뒤에는 같은 자 위에 놓인다",
                 fontsize=13.5, y=1.02)
    save(fig, DIST + "normal_standardization.png")


# ==================================================================
# 2. F 의 조립도 — 정규성이 두 번 쓰인다
# ==================================================================
def f_construction():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))

    def box(xy, text, edge, face, w=2.5, h=1.0, fs=11.5):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.7, zorder=4))
        ax.text(x, y, text, fontsize=fs, color=edge, ha="center", va="center",
                zorder=5, linespacing=1.4)

    def arrow(p, q, color=MUTED):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                                     color=color, linewidth=1.5,
                                     shrinkA=4, shrinkB=4, zorder=3))

    rows = [(4.55, 1, BLUE, BLUE_F), (1.35, 2, ORANGE, ORANGE_F)]
    for y, i, color, face in rows:
        box((1.5, y), f"정규모집단 {i}\n$N(\\mu_{i}, \\sigma_{i}^2)$", color,
            face, w=2.6)
        box((5.2, y), f"표본분산 $S_{i}^2$", color, face, w=2.4, h=0.85)
        box((9.1, y), f"$\\dfrac{{(n_{i}-1)S_{i}^2}}{{\\sigma_{i}^2}}"
                      f" \\sim \\chi^2_{{n_{i}-1}}$", color, face, w=3.3,
            h=1.15, fs=12)
        arrow((2.8, y), (4.0, y), color)
        arrow((6.4, y), (7.45, y), color)
        # 빨간 세로선 왼쪽에 붙여 쓴다. 선 위에 놓으면 글자가 지워진 것처럼
        # 보이고, 오른쪽에 놓으면 카이제곱 상자에 가린다.
        ax.text(6.84, y + 0.34, "정규성", fontsize=10, color=RED,
                ha="right", va="bottom", zorder=6)

    # 두 카이제곱을 모아 비를 만든다
    arrow((10.75, 4.55), (11.9, 3.35), MUTED)
    arrow((10.75, 1.35), (11.9, 2.55), MUTED)
    box((13.5, 2.95), "$\\dfrac{S_1^2 / \\sigma_1^2}{S_2^2 / \\sigma_2^2}"
                      " \\sim F_{n_1-1,\\, n_2-1}$", GREEN, GREEN_F,
        w=3.4, h=1.3, fs=12)
    ax.text(13.5, 1.95, "$H_0: \\sigma_1^2 = \\sigma_2^2$ 이면\n"
                        "$\\sigma$ 가 약분되어 $S_1^2/S_2^2$ 만 남는다",
            fontsize=10.5, color=GREEN, ha="center", va="top",
            linespacing=1.45)

    # 두 겹의 정규성 의존을 묶어 표시한다
    ax.plot([6.92, 6.92], [4.97, 5.55], color=RED, linewidth=1.1)
    ax.plot([6.92, 6.92], [0.93, 0.35], color=RED, linewidth=1.1)
    ax.plot([6.92, 6.92], [5.55, 5.55], color=RED, linewidth=1.1)
    ax.annotate("", xy=(6.92, 5.55), xytext=(6.92, 0.35),
                arrowprops=dict(arrowstyle="-", color=RED, linewidth=1.1,
                                linestyle=":"))
    ax.text(6.92, 5.75, "같은 가정을 두 번 빌려 쓴다 — 어긋남은 상쇄되지 않고 더해진다",
            fontsize=11.5, color=RED, ha="center", va="bottom")

    ax.text(7.6, -0.35, "분자도 분모도 2차 적률이다. 중심극한정리가 손댈 자리가 없다.",
            fontsize=11.5, color=INK, ha="center", va="center")

    ax.set_xlim(-0.2, 15.4)
    ax.set_ylim(-0.9, 6.3)
    ax.axis("off")
    ax.set_title("$F$ 는 독립인 두 카이제곱의 비로 조립된다", fontsize=13.5,
                 pad=14)
    fig.tight_layout()
    save(fig, DIST + "f_construction.png")


# ==================================================================
# 3. 두 평균 차 — 다섯 갈래의 지도
# ==================================================================
def diff_means_tree():
    fig, ax = plt.subplots(figsize=(13, 6.0))

    def diamond(xy, text, w=2.9, h=1.5):
        x, y = xy
        ax.add_patch(Polygon([[x - w / 2, y], [x, y + h / 2],
                              [x + w / 2, y], [x, y - h / 2]], closed=True,
                             facecolor="white", edgecolor=INK, linewidth=1.6,
                             zorder=4))
        ax.text(x, y, text, fontsize=10.5, color=INK, ha="center",
                va="center", zorder=5, linespacing=1.4)

    def leaf(xy, tag, text, edge, face, w=4.0, h=1.15):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.7, zorder=4))
        ax.text(x - w / 2 + 0.28, y, tag, fontsize=12, color=edge,
                ha="left", va="center", zorder=5, fontweight="bold")
        ax.text(x + 0.35, y, text, fontsize=10.5, color=edge, ha="center",
                va="center", zorder=5, linespacing=1.4)

    def link(p, q, label, dx=0.0, dy=0.18, t=0.5):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, linewidth=1.4,
                zorder=2)
        # t 로 라벨을 선 위 어디에 놓을지 고른다. 마름모 꼭짓점과 겹치지 않게.
        ax.text(p[0] + t * (q[0] - p[0]) + dx,
                p[1] + t * (q[1] - p[1]) + dy, label, fontsize=10,
                color=INK, ha="center", va="center", zorder=5)

    q1 = (2.2, 5.6)
    q2 = (6.2, 3.9)
    q3 = (10.0, 2.2)
    diamond(q1, "$\\sigma_1^2,\\ \\sigma_2^2$ 를\n아는가", w=3.0)
    diamond(q2, "두 표본이\n충분히 큰가", w=3.0)
    diamond(q3, "두 분산이\n같다고 볼 수 있는가", w=3.9, h=1.6)

    leaf((6.6, 6.9), "A", "$Z \\sim N(0,1)$\n정규모집단이면 정확하다",
         BLUE, BLUE_F)
    leaf((10.4, 5.3), "B", "$Z \\approx N(0,1)$\n$S_i^2$ 로 갈아 끼운다",
         BLUE, BLUE_F)
    leaf((14.6, 3.5), "C", "합동 $t$\n$\\mathrm{df} = n_1 + n_2 - 2$",
         GREEN, GREEN_F)
    leaf((14.6, 1.6), "D", "Welch $t$\nSatterthwaite 자유도",
         ORANGE, ORANGE_F)
    leaf((14.6, -0.3), "E", "보수적 $t$\n$\\mathrm{df} = \\min(n_1-1, n_2-1)$",
         ORANGE, ORANGE_F)

    link(q1, (4.6, 6.9), "예")
    link(q1, q2, "아니오", dy=-0.3)
    link(q2, (8.4, 5.3), "예")
    link(q2, q3, "아니오 — 정규모집단 가정", dx=-0.55, dy=-0.42, t=0.4)
    link(q3, (12.6, 3.5), "예", dy=0.22, t=0.62)
    link(q3, (12.6, 1.6), "아니오", dx=0.45, dy=-0.26, t=0.6)
    ax.plot([12.6, 12.0, 12.0, 12.6], [1.6, 1.6, -0.3, -0.3], color=MUTED,
            linewidth=1.4, zorder=2)
    ax.text(11.55, 0.62, "손으로\n계산할 때", fontsize=9.5, color=INK,
            ha="center", va="center", linespacing=1.35)

    ax.add_patch(FancyBboxPatch((0.5, -0.95), 8.6, 1.0,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="#FFF3E0", edgecolor=ORANGE,
                                linewidth=1.5, zorder=4))
    ax.text(4.8, -0.45, "실무 권고 — 조건을 모르면 처음부터 D(Welch)를 쓴다",
            fontsize=11.5, color=ORANGE, ha="center", va="center", zorder=5)

    ax.set_xlim(0, 17.0)
    ax.set_ylim(-1.5, 7.9)
    ax.axis("off")
    ax.set_title("두 평균의 차 — 무엇을 아는가에 따라 기준분포가 갈린다",
                 fontsize=13.5, pad=10)
    fig.tight_layout()
    save(fig, APP + "diff_means_tree.png")


# ==================================================================
# 4. 두 비율 차 — 구간의 세계와 귀무가설의 세계
# ==================================================================
def diff_props_two_worlds():
    n1 = n2 = 100
    p1_hat, p2_hat = 0.70, 0.30
    d_hat = p1_hat - p2_hat

    se_un = np.sqrt(p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2)
    p_pool = (p1_hat * n1 + p2_hat * n2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    x = np.linspace(-0.32, 0.72, 800)

    y0 = phi((x - 0) / se_pool) / se_pool
    yd = phi((x - d_hat) / se_un) / se_un

    ax.plot(x, y0, color=ORANGE, linewidth=2.0, zorder=5)
    ax.fill_between(x, y0, color=ORANGE, alpha=0.18, zorder=3)
    ax.plot(x, yd, color=BLUE, linewidth=2.0, zorder=5)
    ax.fill_between(x, yd, color=BLUE, alpha=0.18, zorder=3)

    top = float(max(y0.max(), yd.max()))

    # 관측된 차
    ax.plot([d_hat, d_hat], [0, top * 1.02], color=INK, linewidth=1.6,
            linestyle="--", zorder=6)
    ax.plot([d_hat], [0], marker="o", color=INK, markersize=7, zorder=7)
    ax.text(d_hat, top * 1.06, "관측된 차  $\\hat p_1 - \\hat p_2 = 0.40$",
            fontsize=11.5, color=INK, ha="center", va="bottom")

    # 신뢰구간
    lo, hi = d_hat - 1.96 * se_un, d_hat + 1.96 * se_un
    ybar = -top * 0.13
    ax.plot([lo, hi], [ybar, ybar], color=BLUE, linewidth=3.0,
            solid_capstyle="butt", zorder=6)
    for b in (lo, hi):
        ax.plot([b, b], [ybar - top * 0.035, ybar + top * 0.035], color=BLUE,
                linewidth=2.0, zorder=6)
    ax.text(d_hat, ybar - top * 0.10,
            f"95% 신뢰구간  [{lo:.3f},  {hi:.3f}]", fontsize=11,
            color=BLUE, ha="center", va="top")

    # 귀무가설 쪽 꼬리
    zobs = d_hat / se_pool
    ax.annotate(f"$H_0$ 의 세계에서 보면  $z = {zobs:.2f}$",
                xy=(d_hat, top * 0.055), xytext=(0.03, top * 0.62),
                fontsize=11.5, color=ORANGE, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.4,
                                connectionstyle="arc3,rad=-0.18"))

    ax.text(0.0, top * 1.04, "검정의 세계  —  중심 $0$, 합동 표준오차 "
                             f"${se_pool:.4f}$",
            fontsize=11.5, color=ORANGE, ha="center", va="bottom")
    ax.text(0.615, top * 0.52, "구간의 세계\n중심 $\\hat p_1 - \\hat p_2$\n"
                               f"비합동 표준오차 ${se_un:.4f}$",
            fontsize=11, color=BLUE, ha="center", va="center",
            linespacing=1.45)

    ax.set_xlim(-0.34, 0.78)
    ax.set_ylim(-top * 0.33, top * 1.22)
    ax.set_yticks([])
    ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set_title("같은 자료, 두 물음 — $n_1 = n_2 = 100$, "
                 "$\\hat p_1 = 0.70$, $\\hat p_2 = 0.30$",
                 fontsize=13, pad=12)
    fig.tight_layout()
    save(fig, APP + "diff_props_two_worlds.png")


if __name__ == "__main__":
    normal_standardization()
    f_construction()
    diff_means_tree()
    diff_props_two_worlds()
