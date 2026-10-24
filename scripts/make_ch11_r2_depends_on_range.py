r"""11.7절 「그림으로 보는 통계학」 — R^2 은 모형이 아니라 설계를 잰다.

참 모형도 같고 오차도 같은데, 설명변수를 어디서 얼마나 넓게 뽑았느냐만
바꾸면 R^2 이 0.1 에서 0.99 까지 움직인다. 기울기 추정값은 셋 다 참값
근처에 그대로 있다.

식으로 적으면 숨을 곳이 없다.

  R^2 -> beta^2 Var(x) / (beta^2 Var(x) + sigma^2)

분자에 Var(x) 가 있다. **자료를 모은 사람이 정한 양**이 R^2 을 정한다.

만드는 파일:

  ch11/performance/img/r2_depends_on_range.png

실행:  python3 scripts/make_ch13_r2_depends_on_range.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/performance/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

BETA, SIGMA, N = 2.0, 1.0, 60
SPREADS = (0.2, 0.7, 2.5)
SEED = 21


def main():
    rng = np.random.default_rng(SEED)
    fig = plt.figure(figsize=(12.6, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)

    print(f"    참 기울기 {BETA}, 오차 표준편차 {SIGMA}, n = {N}")
    seen = []
    for j, sd in enumerate(SPREADS):
        x = rng.normal(0, sd, N)
        y = BETA * x + rng.normal(0, SIGMA, N)
        b = float((x - x.mean()) @ (y - y.mean()) / ((x - x.mean()) ** 2).sum())
        a = y.mean() - b * x.mean()
        resid = y - (a + b * x)
        r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        seen.append(r2)
        print(f"      sd(x) = {sd:4.1f}   기울기 추정 {b:+.3f}   R^2 = {r2:.3f}"
              f"   (이론 {BETA**2*sd**2/(BETA**2*sd**2+SIGMA**2):.3f})")

        ax = fig.add_subplot(gs[0, j])
        ax.scatter(x, y, s=24, color=MUTED, alpha=0.8, edgecolors="none")
        g = np.linspace(-8, 8, 20)
        ax.plot(g, a + b * g, lw=2.8, color=TEAL)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-17, 17)
        ax.set_title(f"$R^2 = {r2:.3f}$", fontsize=13.5, color=AMBER, pad=6)
        ax.set_xlabel(f"$x$ 의 표준편차 {sd}", fontsize=11)
        ax.text(0.5, 0.04, f"기울기 추정 {b:+.2f}", transform=ax.transAxes,
                ha="center", fontsize=10.6, color=TEAL)
        if j == 0:
            ax.set_ylabel("$y$", fontsize=12)
        else:
            ax.set_yticklabels([])
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽 넷째 칸: 곡선 ──
    ax = fig.add_subplot(gs[0, 3])
    sds = np.linspace(0.05, 4, 300)
    curve = BETA ** 2 * sds ** 2 / (BETA ** 2 * sds ** 2 + SIGMA ** 2)
    ax.plot(sds, curve, lw=3.0, color=RED)
    for sd in SPREADS:
        v = BETA ** 2 * sd ** 2 / (BETA ** 2 * sd ** 2 + SIGMA ** 2)
        ax.plot([sd], [v], marker="o", ms=9, color=AMBER, zorder=5)
        ax.annotate(f"{v:.2f}", xy=(sd, v), xytext=(sd + 0.18, v - 0.07),
                    fontsize=10.4, color=AMBER)
    ax.set_xlabel("$x$ 의 표준편차", fontsize=11.5)
    ax.set_ylabel("$R^2$ 의 극한값", fontsize=11.5)
    ax.set_title("모형은 그대로다", fontsize=12.8, color=INK, pad=6)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"참 모형도 오차도 같은데 $R^2$ 만 ${seen[0]:.2f}$ 에서 "
                 f"${seen[-1]:.2f}$ 로 움직인다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"셋 다 $y = {BETA:g}x + \\varepsilon$, $\\varepsilon \\sim N(0, {SIGMA:g}^2)$, "
             f"$n = {N}$ 이다. 바꾼 것은 $x$ 를 얼마나 넓게 뽑았는지 하나뿐이고, "
             "기울기 추정값은 셋 다 참값 둘레에 있다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "r2_depends_on_range.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
