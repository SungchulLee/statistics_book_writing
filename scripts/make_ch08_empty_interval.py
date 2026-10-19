r"""8.1절 「그림으로 보는 통계학」 — 아무 값도 담지 않은 신뢰구간.

이 쪽 6 절은 「신뢰수준은 구간을 만드는 **절차**가 갖는 성질이다」라고
적는다. 그 문장이 가장 사납게 드러난 자리가 입자물리학이다.

재려는 양이 음수가 될 수 없는 경우가 많다 — 질량, 사건 발생률, 붕괴
확률. 그런데 **측정값**은 배경 잡음 때문에 음수가 될 수 있다. 재는 값이
x ~ N(mu, 1) 이고 mu >= 0 이라 하자. 고전적 중심 90% 구간은

  [x - 1.645, x + 1.645]

이다. x = -2 가 나오면 구간은 [-3.65, -0.35] 이고, 가능한 값이 **하나도
들어 있지 않다.** 1990 년대에 이런 일이 거듭 일어났다.

중요한 것은 이것이 고장이 아니라는 점이다. 이 절차의 포함확률은 어느
mu 에서나 정확히 0.90 이다. 참값이 경계에 있을 때 구간이 통째로 음수
쪽으로 빠질 확률은 Phi(-1.645) = 0.0500 — 아래로 빗나가도 되는 바로 그
5% 다. **절차는 약속을 지켰고, 그날의 구간은 쓸모가 없다.**

물리학자들이 실제로 한 일이 더 나빴다. 자료를 보고 나서 「유의하지
않으니 상한만 보고하자」와 「유의하니 양쪽 구간을 보고하자」를 골랐다.
펠드먼과 쿠즌스(1998)는 이 「플립플롭」의 포함확률이 0.90 이 아니라
**0.85** 로 떨어짐을 보였다 — 고르는 데 자료를 쓰고, 고른 뒤의 절차는
그 사실을 모른다.

그들의 답은 우도비로 순서를 매겨 받아들임 영역을 만드는 것이었다. 그러면
구간이 비는 일이 없고, x 가 크면 고전적 중심구간과 정확히 같아진다
(x = 6 에서 둘 다 [4.36, 7.64]).

만드는 파일:

  ch08/foundations/img/empty_interval.png

실행:  python3 scripts/make_ch08_empty_interval.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch08/foundations/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

CL = 0.90
Z95 = stats.norm.ppf(0.95)      # 1.645 — 중심 90% 구간
Z90 = stats.norm.ppf(0.90)      # 1.282 — 한쪽 상한
FLIP = 3.0                      # 「3 시그마 넘으면 양쪽 구간」

XG = np.linspace(-10, 24, 34001)
MU = np.linspace(0, 14, 2801)


def fc_band():
    """펠드먼-쿠즌스 받아들임 영역의 양 끝을 mu 마다 구한다."""
    dx = XG[1] - XG[0]
    ref = stats.norm.pdf(XG, np.maximum(XG, 0.0), 1)
    lo, hi = np.empty(MU.size), np.empty(MU.size)
    for i, mu in enumerate(MU):
        f = stats.norm.pdf(XG, mu, 1)
        o = np.argsort(-(f / ref))
        k = np.searchsorted(np.cumsum(f[o]) * dx, CL) + 1
        xs = XG[np.sort(o[:k])]
        lo[i], hi[i] = xs.min(), xs.max()
    return lo, hi


def main():
    lo, hi = fc_band()

    def fc(x0):
        ok = (lo <= x0) & (x0 <= hi)
        return (MU[ok].min(), MU[ok].max()) if ok.any() else (np.nan, np.nan)

    print("    펠드먼-쿠즌스 90% 구간 (x ~ N(mu,1), mu >= 0)")
    for x0 in (-3, -2, -1, 0, 1, 2, 3, 6):
        a, b = fc(x0)
        c0, c1 = x0 - Z95, x0 + Z95
        state = "비어 있음" if c1 < 0 else f"[{max(c0,0):.2f}, {c1:.2f}]"
        print(f"      x = {x0:4.1f}   FC [{a:.2f}, {b:.2f}]   "
              f"고전 중심 [{c0:6.2f}, {c1:5.2f}] -> {state}")
    print(f"\n    고전 중심구간이 통째로 음수가 되는 조건: x < {-Z95:.3f}")
    print(f"    참 mu = 0 일 때 그 확률 = {stats.norm.cdf(-Z95):.4f}")

    xs = np.linspace(-3.2, 5.0, 300)
    fcl = np.array([fc(x)[0] for x in xs])
    fch = np.array([fc(x)[1] for x in xs])

    def cov_flip(mu):
        a = max(stats.norm.cdf(FLIP, mu, 1) - stats.norm.cdf(mu - Z90, mu, 1), 0)
        b = max(stats.norm.cdf(mu + Z95, mu, 1)
                - stats.norm.cdf(max(FLIP, mu - Z95), mu, 1), 0)
        return a + b

    mm = np.linspace(0, 7, 701)
    cf = np.array([cov_flip(m) for m in mm])
    print(f"\n    플립플롭 정책의 포함확률 최솟값 {cf.min():.4f} "
          f"(mu = {mm[cf.argmin()]:.2f})")
    flat = mm[np.isclose(cf, 0.85, atol=5e-4)]
    print(f"    0.85 로 주저앉는 구간: mu = {flat.min():.2f} ~ {flat.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), constrained_layout=True)

    # ── 왼쪽: 관측값마다 두 구간 ──
    ax = axes[0]
    ax.axhspan(-4.2, 0, color=RED, alpha=0.10, zorder=0)
    ax.text(4.85, -3.55, "물리적으로 불가능한 값", ha="right", fontsize=11,
            color=RED)
    ax.axhline(0, color=RED, lw=1.8)
    ax.fill_between(xs, xs - Z95, xs + Z95, color=BLUE, alpha=0.22)
    ax.plot(xs, xs - Z95, lw=2.0, color=BLUE)
    ax.plot(xs, xs + Z95, lw=2.0, color=BLUE)
    ax.fill_between(xs, fcl, fch, color=TEAL, alpha=0.30)
    ax.plot(xs, fcl, lw=2.4, color=TEAL)
    ax.plot(xs, fch, lw=2.4, color=TEAL)
    ax.axvline(-Z95, color=INK, lw=1.5, ls=":")
    ax.annotate(f"$x < {-Z95:.3f}$ 이면\n고전 구간이 통째로 음수 —\n"
                "담긴 값이 하나도 없다",
                xy=(-2.0, -2.0), xytext=(1.1, -2.75), fontsize=10.6, color=BLUE,
                ha="center", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    a2, b2 = fc(-2.0)
    ax.annotate(f"같은 관측에서 FC 는\n$[{a2:.2f},\\ {b2:.2f}]$",
                xy=(-2.0, (a2 + b2) / 2), xytext=(-0.3, 3.6), fontsize=10.6,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.legend(handles=[
        Patch(facecolor=BLUE, alpha=0.4, label="고전적 중심 $90\\%$ 구간"),
        Patch(facecolor=TEAL, alpha=0.5, label="펠드먼–쿠즌스 $90\\%$ 구간")],
        fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.set_xlabel("측정값 $x$  (배경 잡음 때문에 음수도 나온다)", fontsize=11.5)
    ax.set_ylabel("참값 $\\mu \\geq 0$ 에 대한 구간", fontsize=11.5)
    ax.set_title("같은 자료, 한쪽은 빈 구간", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(-3.2, 5.0)
    ax.set_ylim(-4.2, 7.0)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 자료를 보고 고르면 약속이 깨진다 ──
    ax = axes[1]
    ax.fill_between(mm, cf, CL, where=cf < CL, color=RED, alpha=0.18)
    ax.plot(mm, cf, lw=3.0, color=RED, label="플립플롭 — 자료를 보고 고른다")
    ax.plot(mm, np.full_like(mm, CL), lw=2.8, color=BLUE,
            label="고전적 중심구간 — 어느 $\\mu$ 에서나 $0.90$")
    ax.plot(mm, np.full_like(mm, CL), lw=2.8, color=TEAL, ls="--",
            label="펠드먼–쿠즌스 — 어느 $\\mu$ 에서나 $0.90$")
    ax.axhline(CL, color=GREY, lw=1.2, ls=":")
    ax.annotate(f"약속은 $0.90$ 인데 받은 것은 ${cf.min():.2f}$\n"
                f"$\\mu = {flat.min():.1f}$ 부터 ${flat.max():.1f}$ 까지 줄곧",
                xy=(2.6, 0.85), xytext=(3.5, 0.873), fontsize=10.8, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("참값 $\\mu$", fontsize=11.5)
    ax.set_ylabel("구간이 참값을 담을 확률", fontsize=11.5)
    ax.set_title("고르는 데 자료를 쓰면 약속이 깨진다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0.838, 0.912)
    ax.set_xlim(0, 7)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("약속을 지킨 절차가 쓸모없는 구간을 내놓을 수 있다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "재려는 값이 음수가 될 수 없고 측정값은 음수가 될 수 있는 상황이다.\n"
             "왼쪽 고전 구간은 포함확률이 정확히 $0.90$ 이면서도 "
             f"참값이 경계에 있을 때 ${stats.norm.cdf(-Z95):.2f}$ 의 확률로 빈 구간을 낸다 — "
             "아래로 빗나가도 되는 바로 그 몫이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "empty_interval.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
