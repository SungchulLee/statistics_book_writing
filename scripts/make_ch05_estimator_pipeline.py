r"""5.1절 — 모수 · 추정량 · 추정값이 서로 어떻게 다른지 한 그림에.

왼쪽   모집단에 고정된 모수 mu 가 있고, 표본을 뽑을 때마다 추정값이 하나씩
       나온다. 같은 규칙(추정량)을 썼는데도 값이 매번 다르다.
오른쪽 그 값들을 모으면 추정량의 표본분포가 된다. 중심이 모수와 맞으면
       불편이고, 그 분포의 표준편차가 표준오차다.

20세 남자 키를 mu = 170, sigma = 6 인 정규모집단으로 두고 n = 25 로 뽑는다.
이론적으로 표준오차는 6/sqrt(25) = 1.2 다.

만드는 파일:

  ch05/foundations/img/estimator_pipeline.png

실행:  python3 scripts/make_ch05_estimator_pipeline.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/foundations/img/"

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

MU, SIGMA, N = 170.0, 6.0, 25
SEED = 2026
B = 20000


def main():
    rng = np.random.default_rng(SEED)
    shown = rng.normal(MU, SIGMA, (3, N))        # 눈으로 보여 줄 표본 셋
    xbar_shown = shown.mean(axis=1)
    xbar_all = rng.normal(MU, SIGMA, (B, N)).mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6),
                             gridspec_kw={"width_ratios": [1.05, 1]},
                             constrained_layout=True)

    # ---------------- 왼쪽: 모집단에서 표본으로, 표본에서 추정값으로 ----------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 모집단
    ax.add_patch(plt.Circle((1.9, 5.0), 1.55, fc=BLUE_F, ec=BLUE, lw=2))
    ax.text(1.9, 5.9, "모집단", ha="center", fontsize=12.5, color=BLUE)
    ax.text(1.9, 5.0, f"$\\mu = {MU:.0f}$", ha="center", va="center",
            fontsize=15, color=RED)
    ax.text(1.9, 4.2, "고정, 그러나\n알 수 없다", ha="center", va="center",
            fontsize=9.5, color=RED)
    ax.text(1.9, 2.9, "모수 (parameter)", ha="center", fontsize=11, color=RED)

    ys = [7.6, 5.0, 2.4]
    for k, (y, xb) in enumerate(zip(ys, xbar_shown)):
        ax.add_patch(FancyArrowPatch((3.5, 5.0), (4.9, y), mutation_scale=15,
                                     arrowstyle="-|>", color=MUTED, lw=1.5,
                                     connectionstyle="arc3,rad=0.16"))
        # 표본
        ax.add_patch(plt.Rectangle((5.0, y - 0.62), 1.5, 1.24,
                                   fc="white", ec=MUTED, lw=1.4))
        ax.text(5.75, y + 0.22, f"표본 {k + 1}", ha="center", fontsize=10,
                color=INK)
        ax.text(5.75, y - 0.28, f"$n = {N}$", ha="center", fontsize=9.5,
                color=MUTED)
        # 추정값
        ax.add_patch(FancyArrowPatch((6.6, y), (7.7, y), mutation_scale=15,
                                     arrowstyle="-|>", color=GREEN, lw=1.6))
        ax.text(8.6, y, f"$\\bar x = {xb:.2f}$", ha="center", va="center",
                fontsize=13, color=GREEN)

    ax.text(7.15, 8.9, "$\\bar X(\\cdot)$", ha="center", fontsize=13,
            color=GREEN)
    ax.text(7.15, 8.3, "같은 규칙", ha="center", fontsize=9.5, color=GREEN)
    ax.text(7.15, 0.95, "추정량 (estimator) — 함수", ha="center", fontsize=11,
            color=GREEN)
    ax.text(8.6, 0.35, "추정값 (estimate) — 수 하나", ha="center", fontsize=11,
            color=GREEN)
    ax.set_title("같은 규칙을 써도 표본이 바뀌면 값이 바뀐다",
                 fontsize=13, color=INK)

    # ---------------- 오른쪽: 추정값을 모으면 표본분포 ----------------
    ax = axes[1]
    se = SIGMA / np.sqrt(N)
    ax.hist(xbar_all, bins=60, density=True, color=BLUE_F, ec="white", lw=0.3)
    ax.axvline(MU, color=RED, lw=2.4)
    ax.text(MU, ax.get_ylim()[1] * 0.97, f"  모수 $\\mu = {MU:.0f}$",
            color=RED, fontsize=11.5, va="top")

    # 표준오차를 폭으로 표시
    top = ax.get_ylim()[1] * 0.52
    ax.annotate("", xy=(MU, top), xytext=(MU + se, top),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=2.2))
    ax.text(MU + se / 2, top * 1.06, "표준오차", ha="center", fontsize=11.5,
            color=ORANGE)
    ax.text(MU + se / 2, top * 0.90,
            f"$\\sigma/\\sqrt{{n}} = {se:.1f}$", ha="center", fontsize=10.5,
            color=ORANGE)

    for xb in xbar_shown:
        ax.plot([xb], [ax.get_ylim()[1] * 0.045], marker="v", ms=9,
                color=GREEN, clip_on=False)
    ax.text(xbar_shown.min() - 0.15, ax.get_ylim()[1] * 0.10,
            "왼쪽의 추정값 셋", fontsize=10, color=GREEN, ha="right")

    ax.set_xlabel("$\\bar x$", fontsize=12)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"추정값을 {B:,}번 모으면 — 추정량의 표본분포",
                 fontsize=13, color=INK)

    fig.suptitle("모수는 하나, 추정값은 표본마다, 그 전부의 분포가 표본분포다",
                 fontsize=14.5)
    fig.text(0.5, -0.035,
             f"모은 값들의 평균은 {xbar_all.mean():.3f} 로 모수 {MU:.0f} 과 맞고"
             f"(불편), 표준편차는 {xbar_all.std():.3f} 로 이론값 {se:.1f} 과 맞는다.",
             ha="center", fontsize=11)

    path = OUT + "estimator_pipeline.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    보여 준 추정값 셋 = {np.round(xbar_shown, 2)}")
    print(f"    {B}개의 평균 = {xbar_all.mean():.4f}  (모수 {MU})")
    print(f"    {B}개의 표준편차 = {xbar_all.std():.4f}  (이론 {se:.4f})")


if __name__ == "__main__":
    main()
