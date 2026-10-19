r"""21.1절 「그림으로 보는 통계학」 — 함께 죽는 일.

보험계리에는 오래된 자료가 있다. 배우자를 잃은 사람이 그 뒤 얼마나
사는가 — 이른바 「상심 증후군」 자료다. 두 사람의 사망 시각이 서로
독립이 아니라는 것, 그리고 그 의존을 어떻게 적을 것인가가 문제였다.

데이비드 X. 리는 2000 년에 『Journal of Fixed Income』에 그 도구를
회사의 부도에 옮겨 적었다. 회사 수천 개가 언제 부도 나는지를 각각의
생존함수와 **코퓰러** 하나로 잇는 것이다. 가우스 코퓰러를 쓰면 의존
구조 전체가 상관계수 하나로 줄어든다. 값을 매길 수 없던 CDO 조각들에
값이 매겨졌고, 업계와 신용평가사가 이 식을 받아들였다.

가우스 코퓰러에는 성질이 하나 있다. **꼬리의존이 0 이다.**

  lim_{u->1} P(V > u | U > u) = 0     (상관이 1 이 아닌 한)

극단이 함께 일어날 확률이 극단으로 갈수록 0 으로 간다는 뜻이다.
자유도 nu 인 t 코퓰러는 그 극한이 양수다.

  rho = 0.1, nu = 4 이면 꼬리의존 계수가 0.0990

회사 100 개, 각각의 5 년 부도확률 0.05, rho = 0.1 로 두고
200 만 번 모의하면

       한꺼번에 부도 나는 수      가우스        t(4)     배수
                   20 개 이상   0.008000   0.059933    7.5
                   25 개 이상   0.001891   0.035333   18.7
                   30 개 이상   0.000430   0.020601   47.9
                   35 개 이상   0.000098   0.011798  120.4

평균 부도 수는 양쪽 다 5.0 으로 같다. 다른 것은 **함께 일어나느냐**뿐이고,
선순위 조각이 값을 잃는 것은 바로 그 자리에서다.

만드는 파일:

  ch21/introduction/img/copula_tail.png

실행:  python3 scripts/make_ch21_copula_tail.py   (저장소 최상위에서)
       200 만 번 모의라 몇 분 걸린다.
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import norm, t as tdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/introduction/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

M, PD, RHO, NU = 100, 0.05, 0.10, 4
REPS, NSHOW, SEED = 2_000_000, 4000, 2000
KS = np.arange(5, 41)
CORNER = 0.02                      # 양쪽 다 하위 2% 안에 드는가


def draw(rng, n, m, kind):
    """공통인자 하나로 상관을 주고, t 쪽은 공통 척도를 하나 더 곱한다."""
    z = (rng.normal(size=(n, 1)) * np.sqrt(RHO)
         + rng.normal(size=(n, m)) * np.sqrt(1 - RHO))
    if kind == "t":
        w = np.sqrt(NU / rng.chisquare(NU, size=(n, 1)))
        return tdist.cdf(z * w, NU)
    return norm.cdf(z)


def main():
    lam = 2 * tdist.sf(np.sqrt((NU + 1) * (1 - RHO) / (1 + RHO)), NU + 1)
    print(f"  t({NU}) 의 꼬리의존 계수 {lam:.4f},  가우스는 0")

    curves = {}
    for kind in ("gauss", "t"):
        rng = np.random.default_rng(SEED)
        d = (draw(rng, REPS, M, kind) < PD).sum(1)
        curves[kind] = np.array([(d >= k).mean() for k in KS])
        print(f"  {kind:>6}  부도 수 평균 {d.mean():.2f}")
    print(f"\n{'한꺼번에':>10}{'가우스':>12}{'t(4)':>12}{'배수':>9}")
    for k in (10, 15, 20, 25, 30, 35):
        i = int(np.where(KS == k)[0][0])
        g, t = curves["gauss"][i], curves["t"][i]
        print(f"{k:>8} 개{g:>12.6f}{t:>12.6f}{t/max(g,1e-12):>9.1f}")

    fig = plt.figure(figsize=(13.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4])

    # ── 왼쪽 두 장: 같은 상관, 다른 모서리 ──
    rs = np.random.default_rng(7)
    q = norm.ppf(CORNER)
    for row, (kind, nm, col) in enumerate((("gauss", "가우스 코퓰러", TEAL),
                                           ("t", f"$t$ 코퓰러 ($\\nu = {NU}$)",
                                            RED))):
        ax = fig.add_subplot(gs[row, 0])
        u = draw(rs, NSHOW, 2, kind)
        x, y = norm.ppf(u[:, 0]), norm.ppf(u[:, 1])
        hit = (x < q) & (y < q)
        ax.add_patch(plt.Rectangle((-4.2, -4.2), q + 4.2, q + 4.2,
                                   color=col, alpha=0.14, zorder=1))
        ax.plot(x[~hit], y[~hit], lw=0, marker="o", ms=2.6, color=MUTED,
                alpha=0.75, zorder=2)
        ax.plot(x[hit], y[hit], lw=0, marker="o", ms=5, color=col, zorder=3)
        ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{nm} — 모서리에 {hit.sum()} 개", fontsize=11.4,
                     color=col, pad=5)
        for sp in ax.spines.values():
            sp.set_color(MUTED)

    # ── 오른쪽: 함께 부도 나는 수 ──
    ax = fig.add_subplot(gs[:, 1])
    ax.plot(KS, curves["t"], lw=3.6, color=RED,
            label=f"$t$ 코퓰러 ($\\nu = {NU}$)")
    ax.plot(KS, curves["gauss"], lw=3.4, color=TEAL, ls="--",
            label="가우스 코퓰러")
    for k, col in ((30, INK),):
        i = int(np.where(KS == k)[0][0])
        ax.plot([k, k], [curves["gauss"][i], curves["t"][i]], lw=2.4,
                color=col, zorder=4)
        ax.plot([k, k], [curves["gauss"][i], curves["t"][i]], lw=0,
                marker="o", ms=9, color=col, zorder=5, mec="white", mew=1.4)
        ax.annotate(f"${k}$ 개가 한꺼번에 부도 날 확률이\n"
                    f"${curves['t'][i]/curves['gauss'][i]:.0f}$ 배 갈린다",
                    xy=(k, np.sqrt(curves["gauss"][i] * curves["t"][i])),
                    xytext=(23.5, 2.2e-4), fontsize=10.8, color=INK,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_yscale("log")
    ax.set_xlabel(f"한꺼번에 부도 나는 회사의 수 (전체 ${M}$ 개)",
                  fontsize=11.5)
    ax.set_ylabel("그만큼 이상 부도 날 확률", fontsize=11.5)
    ax.set_title("평균은 같고 꼬리만 다르다", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(5e-5, 1.0)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$",
                        "$1$"])
    ax.minorticks_off()
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("함께 죽는 일", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"회사 ${M}$ 개, 각각의 부도확률 ${PD}$, 상관 ${RHO}$ 로 두고 "
             f"${REPS//1000000}$ 백만 번 모의했다. 두 모형의 부도확률도 "
             "상관도 똑같다 — 다른 것은 극단이 함께 일어나는 방식뿐이다. "
             f"왼쪽은 양쪽 모두 하위 ${CORNER*100:.0f}\\%$ 안에 든 점을 "
             "표시한 것이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "copula_tail.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
