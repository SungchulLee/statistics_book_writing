r"""10장 카이제곱 분포 절의 그림을 생성한다.

만드는 파일:
  ch10/distribution/img/df_is_k_minus_one.png   피어슨 통계량은 왜 chi2_{k-1} 인가
  ch10/distribution/img/df_counting.png         자유로운 칸을 세어 자유도를 얻는다

실행:  python3 scripts/make_ch10_distribution_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch10/distribution/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# =====================================================================
# 1. 피어슨 통계량은 chi2_k 가 아니라 chi2_{k-1} 이다
# =====================================================================
def fig_df_is_k_minus_one():
    rng = np.random.default_rng(10)
    n, k = 200, 4
    p = np.array([0.4, 0.3, 0.2, 0.1])
    E = n * p
    O = rng.multinomial(n, p, size=200_000)

    pearson = ((O - E) ** 2 / E).sum(1)            # 분모 = E_i
    naive = ((O - E) ** 2 / (n * p * (1 - p))).sum(1)  # 분모 = 참된 표준편차

    x = np.linspace(0, 20, 600)
    d3, d4 = stats.chi2.pdf(x, 3), stats.chi2.pdf(x, 4)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # --- (가) 피어슨 통계량 ---
    ax = axes[0]
    ax.hist(pearson, bins=np.linspace(0, 20, 81), density=True,
            color=BLUE_F, edgecolor="white", linewidth=0.4,
            label="모의실험 20만 회")
    ax.plot(x, d3, color=BLUE, lw=2.2, label=r"$\chi^2_3$  (자유도 $k-1$)")
    ax.plot(x, d4, color=ORANGE, lw=2.0, ls="--", label=r"$\chi^2_4$  (자유도 $k$)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.265)
    clean_axis(ax)
    ax.set_title(r"(가) 분모를 $E_i$ 로 쓴 피어슨 통계량", fontsize=12,
                 color=INK, pad=9)
    ax.set_xlabel(r"$\sum (O_i-E_i)^2 / E_i$", fontsize=11, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.02, 1.0))
    ax.text(0.30, 0.52,
            "평균 3.003, 분산 6.001\n" r"$\chi^2_3$ 은 평균 3, 분산 6",
            transform=ax.transAxes, fontsize=10, color=BLUE, va="top")

    # --- (나) 참된 표준편차로 나눈 합 ---
    ax = axes[1]
    ax.hist(naive, bins=np.linspace(0, 20, 81), density=True,
            color=ORANGE_F, edgecolor="white", linewidth=0.4,
            label="모의실험 20만 회")
    ax.plot(x, d4, color=ORANGE, lw=2.2, label=r"$\chi^2_4$  (자유도 $k$)")
    ax.plot(x, d3, color=BLUE, lw=2.0, ls="--", label=r"$\chi^2_3$  (자유도 $k-1$)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.265)
    clean_axis(ax)
    ax.set_title(r"(나) 분모를 $\sqrt{n p_i(1-p_i)}$ 로 쓴 합", fontsize=12,
                 color=INK, pad=9)
    ax.set_xlabel(r"$\sum (O_i-E_i)^2 / \{n p_i(1-p_i)\}$", fontsize=11, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.02, 1.0))
    ax.text(0.30, 0.52,
            "평균 4.002, 분산 10.80\n" r"$\chi^2_4$ 는 평균 4, 분산 8",
            transform=ax.transAxes, fontsize=10, color=ORANGE, va="top")

    fig.suptitle("범주 $k=4$, 표본 $n=200$, "
                 r"$p=(0.4,\,0.3,\,0.2,\,0.1)$ 에서 두 표준화를 비교한다",
                 fontsize=12.5, color=INK, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "df_is_k_minus_one.png")


# =====================================================================
# 2. 자유로운 칸 세기: k-1 과 (r-1)(c-1)
# =====================================================================
def draw_cell(ax, x, y, w, h, fc, ec, text, tcolor, fs=11, lw=1.6):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tcolor, zorder=3)


def fig_df_counting():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # --- (가) 적합도 검정: k 개 범주, k-1 개가 자유 ---
    ax = axes[0]
    k = 5
    w, h = 1.0, 1.0
    for i in range(k):
        free = i < k - 1
        draw_cell(ax, i * w, 0, w, h,
                  BLUE_F if free else ORANGE_F,
                  BLUE if free else ORANGE,
                  "자유" if free else "결정",
                  BLUE if free else ORANGE, fs=10.5)
        ax.text(i * w + w / 2, h + 0.16, f"$O_{i+1}$", ha="center",
                va="bottom", fontsize=11, color=INK)
    ax.annotate("", xy=(k * w - 0.5 * w, -0.28), xytext=(0.5 * w, -0.28),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))
    ax.text(k * w / 2, -0.62, r"합이 $n$ 이어야 한다  $\Rightarrow$  제약 1개",
            ha="center", va="top", fontsize=11, color=INK)
    ax.text(k * w / 2, -1.12, r"$\text{df} = k - 1 = 4$", ha="center",
            va="top", fontsize=13, color=BLUE)
    ax.set_xlim(-0.35, k * w + 0.35)
    ax.set_ylim(-1.75, h + 0.75)
    ax.axis("off")
    ax.set_title("(가) 적합도 검정: 범주 5개", fontsize=12, color=INK)

    # --- (나) 3x4 분할표: (r-1)(c-1) 개가 자유 ---
    ax = axes[1]
    r, c = 3, 4
    w, h = 1.0, 0.85
    for i in range(r):
        for j in range(c):
            free = (i < r - 1) and (j < c - 1)
            draw_cell(ax, j * w, -(i + 1) * h, w, h,
                      BLUE_F if free else ORANGE_F,
                      BLUE if free else ORANGE,
                      "자유" if free else "결정",
                      BLUE if free else ORANGE, fs=10)
    # 주변합 칸
    for i in range(r):
        draw_cell(ax, c * w + 0.22, -(i + 1) * h, w, h, "white", MUTED,
                  "행 합", INK, fs=10, lw=1.2)
    for j in range(c):
        draw_cell(ax, j * w, -(r + 1) * h - 0.22, w, h, "white", MUTED,
                  "열 합", INK, fs=10, lw=1.2)
    draw_cell(ax, c * w + 0.22, -(r + 1) * h - 0.22, w, h, "white", MUTED,
              "$n$", INK, fs=11, lw=1.2)

    ax.text(0.0, 0.30,
            r"주변합을 고정하면 파란 칸 $(r-1)(c-1)=2\times 3=6$ 개만 자유롭다.",
            fontsize=11, color=INK, va="bottom")
    ax.text(0.0, -(r + 2) * h - 0.60,
            r"$\text{df} = rc - 1 - (r-1) - (c-1) = 12 - 1 - 2 - 3 = 6$",
            fontsize=12.5, color=BLUE, va="top")
    ax.set_xlim(-0.35, c * w + w + 0.75)
    ax.set_ylim(-(r + 2) * h - 1.35, 1.05)
    ax.axis("off")
    ax.set_title("(나) 독립성·동질성 검정: 3행 4열", fontsize=12, color=INK)

    fig.tight_layout()
    save(fig, OUT + "df_counting.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_df_is_k_minus_one()
    fig_df_counting()
