r"""5.1절 반복추출 — LaTeX 배열로 그려 두었던 두 그림을 제대로 그린다.

  repeated_sampling_loop.png    같은 모집단에서 되풀이해 뽑으면 추정값이 쌓이고,
                                쌓인 것이 표본분포로 드러난다. k 를 키워 가며 보인다.
  three_distributions.png       이름이 닮은 세 분포를 나란히. 이름만 적지 않고
                                각 분포의 생김새와 퍼진 정도까지 함께 그린다.

모집단은 평균 170, 표준편차 6 인 정규분포로 두고 표본크기는 n = 25 다.
따라서 한 표본 안의 퍼짐은 sigma = 6 이고 표본평균의 퍼짐은 6/5 = 1.2 다.

만드는 파일:

  ch05/foundations/img/repeated_sampling_loop.png
  ch05/foundations/img/three_distributions.png

실행:  python3 scripts/make_ch05_repeated_sampling_diagrams.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/foundations/img/"

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
MUTED = "#90A4AE"
RED = "#D32F2F"

MU, SIGMA, N = 170.0, 6.0, 25
SEED = 11


def box(ax, x, y, w, h, text, fc, ec, fs=11.5, tc=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.10", fc=fc, ec=ec, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), mutation_scale=14,
                                 arrowstyle="-|>", color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- 그림 1
def loop_figure():
    rng = np.random.default_rng(SEED)
    est = rng.normal(MU, SIGMA / np.sqrt(N), 20000)

    fig = plt.figure(figsize=(14, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.35)

    ax = fig.add_subplot(gs[:, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # 모집단 하나에서 몇 번이고 다시 뽑는다
    ax.add_patch(plt.Circle((1.3, 5.0), 1.15, fc=BLUE_F, ec=BLUE, lw=2))
    ax.text(1.3, 5.35, "모집단", ha="center", fontsize=12, color=BLUE)
    ax.text(1.3, 4.55, f"$\\mu = {MU:.0f}$", ha="center", fontsize=12, color=RED)

    rows = [(8.5, "1"), (6.7, "2"), (4.3, "k"), (2.5, "k+1")]
    for y, lab in rows:
        arrow(ax, 2.6, 5.0, 3.9, y, rad=0.12)
        box(ax, 5.0, y, 2.0, 1.0,
            f"표본 $\\mathbf{{x}}_{{{lab}}}$\n$n = {N}$", "white", MUTED, 10.5)
        arrow(ax, 6.1, y, 7.0, y, color=GREEN)
        ax.text(8.2, y, f"$\\hat\\theta(\\mathbf{{x}}_{{{lab}}})$",
                ha="center", va="center", fontsize=12.5, color=GREEN)
    ax.text(5.0, 5.5, "$\\vdots$", ha="center", fontsize=15, color=MUTED)
    ax.text(5.0, 1.5, "$\\vdots$", ha="center", fontsize=15, color=MUTED)
    ax.text(8.2, 1.5, "$\\vdots$", ha="center", fontsize=15, color=GREEN)

    # 추정값을 한데 묶는 큰 중괄호
    ax.annotate("", xy=(9.3, 1.2), xytext=(9.3, 9.0),
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=2.2,
                                connectionstyle="bar,fraction=0.10"))
    ax.set_title("같은 모집단에서 되풀이해 뽑으면 추정값이 쌓인다",
                 fontsize=13.5, color=INK)

    # 오른쪽: k 가 커지면 드러나는 분포
    for r, k in enumerate((10, 200, 20000)):
        a = fig.add_subplot(gs[r, 1])
        a.hist(est[:k], bins=np.linspace(165.5, 174.5, 46), density=True,
               color=GREEN_F, ec=GREEN, lw=0.5)
        a.axvline(MU, color=RED, lw=1.8)
        a.set_xlim(165.5, 174.5); a.set_yticks([])
        a.spines[["top", "right", "left"]].set_visible(False)
        a.text(0.02, 0.86, f"표본을 $k = {k:,}$번 뽑았을 때",
               transform=a.transAxes, fontsize=11, color=GREEN)
        if r == 0:
            a.set_title("쌓인 값들이 이루는 분포 — 이것이 표본분포다",
                        fontsize=13.5, color=INK, pad=10)
        if r < 2:
            a.set_xticklabels([])
        else:
            a.set_xlabel("$\\hat\\theta$ 의 값", fontsize=11)

    fig.suptitle("표본분포는 뽑기를 되풀이해야 비로소 보이는 분포다", fontsize=15)
    path = OUT + "repeated_sampling_loop.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------- 그림 2
def three_figure():
    rng = np.random.default_rng(SEED)
    one = rng.normal(MU, SIGMA, N)
    est = rng.normal(MU, SIGMA / np.sqrt(N), 20000)
    g = np.linspace(150, 190, 400)

    fig = plt.figure(figsize=(14, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], hspace=0.05)

    top = fig.add_subplot(gs[0, :])
    top.set_xlim(0, 12); top.set_ylim(0, 3); top.axis("off")
    xs = [2.0, 6.0, 10.0]
    labels = ["모집단", "표본 $\\mathbf{x}$", "추정값 $\\hat\\theta(\\mathbf{x})$"]
    cols = [BLUE, ORANGE, GREEN]
    fills = [BLUE_F, ORANGE_F, GREEN_F]
    for x, lab, c, fc in zip(xs, labels, cols, fills):
        box(top, x, 2.0, 2.7, 0.95, lab, fc, c, 13)
        top.add_patch(FancyArrowPatch((x, 1.45), (x, 0.55), mutation_scale=14,
                                      arrowstyle="-|>", color=c, lw=1.8))
    arrow(top, 3.45, 2.0, 4.6, 2.0, color=MUTED, lw=1.8)
    arrow(top, 7.45, 2.0, 8.6, 2.0, color=MUTED, lw=1.8)
    top.text(4.0, 2.35, "뽑는다", ha="center", fontsize=10.5, color=MUTED)
    top.text(8.0, 2.35, "계산한다", ha="center", fontsize=10.5, color=MUTED)

    # 아래: 세 분포의 생김새
    panels = [
        ("모집단분포", "모집단 전체가 이루는 분포\n관측할 수 없다",
         BLUE, BLUE_F, f"퍼짐 $\\sigma = {SIGMA:.0f}$"),
        ("한 표본의 분포", f"뽑아 온 값 $n = {N}$개의 분포\n**유일하게 손에 쥔 것**",
         ORANGE, ORANGE_F, f"퍼짐 $\\approx \\sigma = {SIGMA:.0f}$"),
        ("표본분포", "무한히 되풀이해 얻은\n추정값들의 분포",
         GREEN, GREEN_F, f"퍼짐 $\\sigma/\\sqrt{{n}} = {SIGMA/np.sqrt(N):.1f}$"),
    ]
    for k, (title, sub, c, fc, width) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        if k == 0:
            a.plot(g, np.exp(-((g - MU) ** 2) / (2 * SIGMA ** 2)), color=c, lw=2)
            a.fill_between(g, np.exp(-((g - MU) ** 2) / (2 * SIGMA ** 2)),
                           color=fc, alpha=0.9)
        elif k == 1:
            a.hist(one, bins=np.linspace(150, 190, 22), color=fc, ec=c, lw=1.0)
            a.plot(one, np.full(N, -0.35), "|", color=c, ms=10, clip_on=False)
        else:
            a.hist(est, bins=np.linspace(150, 190, 160), density=True,
                   color=fc, ec=c, lw=0.3)
        a.set_xlim(150, 190); a.set_yticks([])
        a.spines[["top", "right", "left"]].set_visible(False)
        a.set_xlabel("키 (cm)" if k < 2 else "$\\bar x$ (cm)", fontsize=10.5)
        a.set_title(title, fontsize=13, color=c, pad=8)
        a.text(0.5, -0.30, sub.replace("**", ""), transform=a.transAxes,
               ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.4)
        a.text(0.5, -0.47, width, transform=a.transAxes, ha="center", va="top",
               fontsize=11, color=c)

    fig.suptitle("이름이 닮은 세 분포 — 가리키는 대상이 전혀 다르다", fontsize=15)
    fig.text(0.5, -0.10,
             "관측되는 것은 가운데뿐이고, 알고 싶은 것은 왼쪽이며, 그 사이를 이어 주는 것이 오른쪽이다. "
             f"같은 자료인데 퍼진 정도가 {SIGMA:.0f}과 {SIGMA/np.sqrt(N):.1f}로 다르다.",
             ha="center", fontsize=11.5)
    path = OUT + "three_distributions.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    한 표본의 표준편차 = {one.std(ddof=1):.2f}  (sigma = {SIGMA})")
    print(f"    추정값들의 표준편차 = {est.std():.3f}  (이론 {SIGMA/np.sqrt(N):.1f})")


if __name__ == "__main__":
    loop_figure()
    three_figure()
