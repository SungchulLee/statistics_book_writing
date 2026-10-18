r"""3.4절 「AI로 보는 통계학 1」 — 상관계수로 변수를 고르면 버려지는 것.

같은 관계 y = x^2 를 두 모집단에서 그린다.

  왼쪽  x ~ Uniform(-1, 1)  대칭이라 양쪽 기여가 상쇄되어 r = 0.003
  오른쪽 x ~ Uniform(0, 1)   대칭이 깨져 같은 관계에서 r = 0.968

둘 다 y 가 x 로 **완전히** 결정되는데 상관계수만 다르다. 상관이 작다는 것은
변수의 쓸모가 아니라 자료의 대칭성에 대한 말이라는 것이 요점이다.

구간별 조건부평균 E[y|x] 를 함께 얹는다. 독립이면 이 점들이 수평선이어야
하는데, 왼쪽에서도 0.02 에서 0.77 까지 오르내린다 — 상관은 0 인데 종속이다.

만드는 파일:

  ch03/rv/img/corr_misses_curve.png

실행:  python3 scripts/make_ch03_corr_misses_curve.py   (저장소 최상위에서)
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
RED = "#D32F2F"
MUTED = "#90A4AE"

N = 10_000
NBIN = 8
SEED = 1


# === 한 칸 그리기 ==================================================
def panel(ax, x, title, note):
    """산점도 + 구간별 조건부평균. r 과 E[y|x] 의 흔들림을 함께 적는다."""
    y = x**2
    r = np.corrcoef(x, y)[0, 1]

    ax.plot(x, y, linestyle="none", marker="o", ms=2.0, alpha=0.18,
            color=BLUE, mec="none")

    lo, hi = x.min(), x.max()
    edges = np.linspace(lo, hi, NBIN + 1)
    cell = np.digitize(x, edges[1:-1])
    cx = 0.5 * (edges[:-1] + edges[1:])
    cm = np.array([y[cell == k].mean() for k in range(NBIN)])

    ax.plot(cx, cm, marker="s", ms=8, lw=2.2, color=RED, mec="white", mew=1.2,
            zorder=5, label="구간별 $E[y \\mid x]$")
    ax.axhline(y.mean(), color=MUTED, lw=1.4, ls="--", zorder=1,
               label="전체 평균 $E[y]$")

    ax.set_title(title, fontsize=12.5, color=INK, pad=8)
    ax.set_xlabel("$x$", fontsize=11.5)
    ax.set_ylabel("$y = x^2$", fontsize=11.5)
    ax.set_ylim(-0.06, 1.06)
    ax.spines[["top", "right"]].set_visible(False)

    ax.text(0.5, 0.965, f"$r = {r:+.3f}$", transform=ax.transAxes,
            ha="center", va="top", fontsize=15, color=RED,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED, lw=1.3))
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center",
            fontsize=10.5, color=INK)
    return r, cm


# === 그림 =========================================================
def main():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), constrained_layout=True)

    r1, cm1 = panel(
        axes[0], rng.uniform(-1, 1, N),
        "$x \\sim \\mathrm{Uniform}(-1,\\,1)$ — 대칭",
        "왼쪽 가지와 오른쪽 가지가 서로 상쇄되어 직선 성분이 남지 않는다.\n"
        "그런데 붉은 점은 수평이 아니다 — 무상관인데 종속이다.")
    r2, cm2 = panel(
        axes[1], rng.uniform(0, 1, N),
        "$x \\sim \\mathrm{Uniform}(0,\\,1)$ — 대칭이 깨짐",
        "관계식은 똑같이 $y = x^2$ 인데 상관계수만 올라갔다.\n"
        "작은 $r$ 는 변수의 쓸모가 아니라 자료의 대칭성에 대한 말이다.")

    axes[0].legend(fontsize=10, loc="upper left", framealpha=0.9)
    fig.suptitle("같은 관계, 다른 상관계수 — 상관은 직선 성분만 잰다",
                 fontsize=14.5)

    path = OUT + "corr_misses_curve.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    왼쪽  r = {r1:+.5f}   E[y|x] 범위 {cm1.min():.3f} ~ {cm1.max():.3f}")
    print(f"    오른쪽 r = {r2:+.5f}   E[y|x] 범위 {cm2.min():.3f} ~ {cm2.max():.3f}")


if __name__ == "__main__":
    main()
