r"""9.4 두 분산에 대한 F-검정 쪽의 그림을 생성한다.

  ch09/two_sample_tests/img/f_two_sided_asymmetry.png
      F 분포는 1 을 중심으로 대칭이 아니다 — 양측 기각역과 역수 관계

  (a) F(9,11) 의 양측 5% 기각역. 1 에서 위쪽 임계값까지의 거리가
      아래쪽까지의 거리보다 몇 배 먼지 재서 적는다.
  (b) 가로축을 로그로 바꾸면 F(9,11) 과 F(11,9) 가 서로 거울상이 된다.
      이것이 F_{1-a}(d1,d2) = 1 / F_a(d2,d1) 이라는 역수 관계다.
  (c) 역수를 취할 때 자유도를 바꾸지 않으면 실제 유의수준이 얼마나
      어긋나는지 자유도 조합마다 계산한다.

실행:  python3 scripts/make_ch09_final_ftest.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch09/two_sample_tests/img/"
ALPHA = 0.05


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 측정 ===
D1, D2 = 9, 11                      # 본문 수치 예제와 같은 자유도
LO = stats.f.ppf(ALPHA / 2, D1, D2)         # 아래쪽 2.5% 점
HI = stats.f.ppf(1 - ALPHA / 2, D1, D2)     # 위쪽 2.5% 점
HI_SWAP = stats.f.ppf(1 - ALPHA / 2, D2, D1)

# 자유도를 바꾸지 않고 역수만 취했을 때의 실제 양측 수준
PAIRS = [(20, 4), (40, 9), (19, 19), (9, 11), (15, 20), (9, 40), (4, 20)]


def naive_level(d1, d2, alpha=ALPHA):
    """아래쪽 임계값을 1/F_{1-a/2}(d1,d2) 로 잘못 잡았을 때의 실제 수준."""
    hi = stats.f.ppf(1 - alpha / 2, d1, d2)
    return stats.f.cdf(1 / hi, d1, d2) + stats.f.sf(hi, d1, d2)


# ==================================================================
# F 분포의 비대칭성과 양측 기각역
# ==================================================================
def f_two_sided_asymmetry():
    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85],
                          hspace=0.42, wspace=0.16)

    # ---------- (a) 양측 기각역 ----------
    ax = fig.add_subplot(gs[0, 0])
    x = np.linspace(0.001, 6.0, 900)
    y = stats.f.pdf(x, D1, D2)
    ax.plot(x, y, color=BLUE, linewidth=2.6, zorder=6)

    xl = x[x <= LO]
    ax.fill_between(xl, stats.f.pdf(xl, D1, D2), color=ORANGE_F,
                    edgecolor=ORANGE, linewidth=1.0, zorder=4)
    xr = x[x >= HI]
    ax.fill_between(xr, stats.f.pdf(xr, D1, D2), color=ORANGE_F,
                    edgecolor=ORANGE, linewidth=1.0, zorder=4)

    ax.plot([1, 1], [0, 0.84], color=MUTED, linewidth=1.3,
            linestyle=":", zorder=3)
    for c in (LO, HI):
        ax.plot([c, c], [-0.045, stats.f.pdf(c, D1, D2) + 0.02], color=ORANGE,
                linewidth=1.5, linestyle="--", zorder=5)

    ax.set_ylim(-0.26, 0.95)
    ax.set_xlim(0, 6)
    ax.spines["bottom"].set_position(("data", 0.0))
    ax.text(1.0, 0.86, "$F=1$", fontsize=11, color=INK, ha="center", va="bottom")

    ax.text(0.34, -0.075, f"${LO:.3f}$", fontsize=11, color=ORANGE,
            ha="center", va="top")
    ax.text(HI, -0.075, f"${HI:.3f}$", fontsize=11, color=ORANGE,
            ha="center", va="top")
    ax.text(3.0, -0.165, "두 꼬리에 각각 2.5% 를 둔다", fontsize=10.5,
            color=ORANGE, ha="center", va="top")

    ax.text(5.95, 0.93,
            f"1 에서 아래쪽 임계값까지   {1 - LO:.3f}\n"
            f"1 에서 위쪽 임계값까지     {HI - 1:.3f}\n"
            f"위쪽이 {(HI - 1) / (1 - LO):.1f} 배 멀다",
            fontsize=10.5, color=INK, ha="right", va="top", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.42", fc="#F5F7F9", ec=MUTED, lw=0.8))

    ax.set_xlabel("$F = S_1^2 / S_2^2$", fontsize=12, color=INK, labelpad=34)
    bare_axis(ax)
    ax.set_title(f"(a)  $F_{{{D1},{D2}}}$ 의 양측 5% 기각역",
                 fontsize=12.5, pad=10, color=INK)

    # ---------- (b) 로그 축에서의 거울상 ----------
    ax = fig.add_subplot(gs[0, 1])
    e1, e2 = 4, 20                       # 자유도 차가 커서 거울상이 잘 보인다
    lo_b = stats.f.ppf(ALPHA / 2, e1, e2)
    hi_b = stats.f.ppf(1 - ALPHA / 2, e2, e1)

    u = np.linspace(np.log(0.012), np.log(85), 900)
    f_u = np.exp(u)
    ax.plot(u, stats.f.pdf(f_u, e1, e2) * f_u, color=BLUE, linewidth=2.6,
            label=f"$F_{{{e1},{e2}}}$", zorder=6)
    ax.plot(u, stats.f.pdf(f_u, e2, e1) * f_u, color=ORANGE, linewidth=2.6,
            linestyle="--", label=f"$F_{{{e2},{e1}}}$", zorder=6)

    ax.plot([0, 0], [0, 0.50], color=MUTED, linewidth=1.3, linestyle=":",
            zorder=3)
    ax.plot([np.log(lo_b)] * 2, [0, 0.30], color=BLUE, linewidth=1.5,
            linestyle="--", zorder=5)
    ax.plot([np.log(hi_b)] * 2, [0, 0.30], color=ORANGE, linewidth=1.5,
            linestyle="--", zorder=5)

    ticks = [1 / 64, 1 / 16, 1 / 4, 1, 4, 16, 64]
    ax.set_xticks([np.log(t) for t in ticks])
    ax.set_xticklabels(["1/64", "1/16", "1/4", "1", "4", "16", "64"])
    ax.set_xlim(np.log(0.012), np.log(85))
    ax.set_ylim(0, 0.56)

    ax.text(np.log(lo_b) - 0.15, 0.205, f"${lo_b:.4f}$", fontsize=11,
            color=BLUE, ha="right", va="center")
    ax.text(np.log(lo_b) - 0.15, 0.125, "아래쪽 2.5% 점", fontsize=10,
            color=BLUE, ha="right", va="center")
    ax.text(np.log(hi_b) + 0.15, 0.205, f"${hi_b:.4f}$", fontsize=11,
            color=ORANGE, ha="left", va="center")
    ax.text(np.log(hi_b) + 0.15, 0.125, "위쪽 2.5% 점", fontsize=10,
            color=ORANGE, ha="left", va="center")

    ax.text(np.log(30), 0.43,
            f"${lo_b:.4f} = 1 / {hi_b:.4f}$", fontsize=11.5, color=PURPLE,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PURPLE, lw=0.9))
    ax.text(np.log(30), 0.355, "자유도를 맞바꾸면\n서로 거울상이다",
            fontsize=10.5, color=INK, ha="center", va="top", linespacing=1.5)

    ax.set_xlabel("$F$ (배수 척도)", fontsize=12, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("(b)  배수 척도에서는 대칭이 드러난다",
                 fontsize=12.5, pad=10, color=INK)

    # ---------- (c) 자유도를 안 바꾸면 ----------
    ax = fig.add_subplot(gs[1, :])
    levels = [naive_level(d1, d2) for d1, d2 in PAIRS]
    labels = [f"$({d1},{d2})$" for d1, d2 in PAIRS]
    pos = np.arange(len(PAIRS))

    colors, edges = [], []
    for (d1, d2), lv in zip(PAIRS, levels):
        if d1 == d2:
            colors.append(GREEN_F); edges.append(GREEN)
        elif lv > ALPHA:
            colors.append(ORANGE_F); edges.append(ORANGE)
        else:
            colors.append(BLUE_F); edges.append(BLUE)

    ax.bar(pos, levels, width=0.58, color=colors, edgecolor=edges,
           linewidth=1.4, zorder=4)
    for p, lv in zip(pos, levels):
        ax.text(p, lv + 0.004, f"{lv:.4f}", fontsize=10.5, color=INK,
                ha="center", va="bottom")

    ax.axhline(ALPHA, color=RED, linewidth=1.5, linestyle="--", zorder=5)
    ax.text(-0.52, ALPHA + 0.004, "명목 수준 0.05", fontsize=10.5,
            color=RED, ha="left", va="bottom")

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.6, len(PAIRS) - 0.4)
    ax.set_ylim(0, 0.165)
    ax.set_xlabel("자유도 $(d_1, d_2)$", fontsize=12, color=INK)
    ax.set_ylabel("실제 기각률", fontsize=12, color=INK)
    clean_axis(ax)
    ax.annotate("$d_1 = d_2$ 이면 오류가 드러나지 않는다",
                xy=(2.25, 0.0495), xytext=(2.55, 0.105), fontsize=10.5,
                color=GREEN, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.set_title("(c)  아래쪽 임계값을 $1/F_{0.975}(d_1,d_2)$ 로 잘못 잡았을 때의 "
                 "실제 양측 수준",
                 fontsize=12.5, pad=10, color=INK)

    fig.suptitle("$F$ 분포는 1 을 중심으로 대칭이 아니다 — 양측 기각역과 역수 관계",
                 fontsize=14, y=0.975, color=INK)
    save(fig, OUT + "f_two_sided_asymmetry.png")


if __name__ == "__main__":
    print(f"F({D1},{D2}):  아래쪽 2.5% 점 {LO:.4f}   위쪽 2.5% 점 {HI:.4f}")
    print(f"F({D2},{D1}):  위쪽 2.5% 점 {HI_SWAP:.4f}   1/{HI_SWAP:.4f} = "
          f"{1 / HI_SWAP:.4f}")
    print(f"1 로부터의 거리 비  {(HI - 1) / (1 - LO):.3f}")
    for d1, d2 in PAIRS:
        print(f"  ({d1:2d},{d2:2d})  잘못 잡은 아래쪽 임계값 "
              f"{1 / stats.f.ppf(0.975, d1, d2):.4f}   실제 수준 "
              f"{naive_level(d1, d2):.4f}")
    f_two_sided_asymmetry()
