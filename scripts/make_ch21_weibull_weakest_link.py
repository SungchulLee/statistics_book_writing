r"""21.4절 「그림으로 보는 통계학」 — 강철과 목화와 콩에 같은 곡선이 맞는 까닭.

발로디 와이블(1887-1979)은 스웨덴의 공학자다. 재료의 강도와 베어링을
다루다가 1951 년에 「폭넓게 쓸 수 있는 통계분포함수」라는 논문을 냈다.

그 논문이 한 일은 분포 하나를 서로 아무 상관 없는 자료 일곱 벌에 맞춘
것이다 — 보포스 강철의 강도, 인도 목화 섬유의 강도, 강재의 피로수명,
영국 성인 남성의 키, 그리고 강낭콩의 폭.

자연스러운 반론이 있다. **무엇에나 맞는 분포는 아무것도 설명하지
못한다.** 자유로운 모수가 둘이면 어지간한 모양은 다 흉내 낼 수 있다.

그런데 와이블에는 반론을 견디는 근거가 있다. **최약 고리**다.

물건이 가장 약한 부품에서 끊어진다면, 그 물건의 강도는 부품 강도의
**최솟값**이다. 그리고 부품이 많으면 최솟값의 분포는 부품이 무엇이든
와이블로 간다 — 남는 것은 부품 분포의 **왼쪽 꼬리 지수** 하나뿐이다.

  F(x) ~ c x^a   (x -> 0+)   이면   최솟값은 모양 a 인 와이블

모의로 확인하면 이렇다 (부품 n 개 중 가장 약한 것에 와이블을 적합).

  부품 분포        이론 a    n=5     n=20    n=100   n=1000
  균등(0,1)          1      1.124   1.028   1.008   0.995
  감마(모양 2)        2      1.688   1.825   1.916   1.974
  Beta(3,1)         3      3.367   3.077   3.033   2.986
  |정규|             1      1.091   1.036   1.012   1.007

부품 분포가 균등이든 감마든 반정규든 상관없다. 가운데가 어떻게 생겼든
**0 근처에서 어떻게 생겼는지만** 살아남는다. 강철과 목화와 콩에 같은
곡선이 맞는 까닭이 이것이다.

만드는 파일:

  ch21/parametric/img/weibull_weakest_link.png

실행:  python3 scripts/make_ch21_weibull_weakest_link.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/parametric/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

R, SEED = 40_000, 1951
NS = (5, 20, 100, 1000)
SHOW_N = 1000


def families(rng):
    return [
        ("균등$(0,1)$", lambda s: rng.uniform(0, 1, s), 1, BLUE),
        ("$|$정규$|$", lambda s: np.abs(rng.normal(0, 1, s)), 1, TEAL),
        ("감마(모양 $2$)", lambda s: rng.gamma(2, 1, s), 2, AMBER),
        ("Beta$(2,1)$", lambda s: rng.beta(2, 1, s), 2, RED),
        ("Beta$(3,1)$", lambda s: rng.beta(3, 1, s), 3, PURPLE),
    ]


def main():
    rng = np.random.default_rng(SEED)
    fams = families(rng)

    print(f"{'부품 분포':>18}{'이론 a':>8}" + "".join(f"{f'n={n}':>10}" for n in NS))
    curves, shown = {}, {}
    for name, f, a, col in fams:
        row, ks = "", []
        for n in NS:
            m = f((R, n)).min(1)
            k, _, sc = stats.weibull_min.fit(m, floc=0)
            ks.append(k)
            row += f"{k:>10.3f}"
            if n == SHOW_N:
                shown[name] = (m, k, sc, a, col)
        curves[name] = (ks, a, col)
        plain = name.replace("$", "").replace("\\", "")
        print(f"{plain:>18}{a:>8}{row}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 와이블 확률지 위에서 직선이 된다 ──
    ax = axes[0]
    for name in ("균등$(0,1)$", "감마(모양 $2$)", "Beta$(3,1)$"):
        m, k, sc, a, col = shown[name]
        q = np.quantile(m, np.linspace(0.02, 0.98, 60))
        emp = np.array([(m <= v).mean() for v in q])
        ax.plot(np.log(q), np.log(-np.log(1 - emp)), lw=0, marker="o", ms=5,
                color=col, alpha=0.8,
                label=f"{name} — 기울기 {k:.2f} (이론 {a})")
        xs = np.array([np.log(q[0]), np.log(q[-1])])
        ax.plot(xs, k * (xs - np.log(sc)), lw=2.2, color=col, alpha=0.75)
    ax.set_xlabel("$\\log$ (가장 약한 부품의 강도)", fontsize=11.5)
    ax.set_ylabel("$\\log(-\\log S)$", fontsize=11.5)
    ax.set_title(f"부품 ${SHOW_N:,}$ 개 중 최솟값 — 와이블 확률지".replace(",", "{,}"),
                 fontsize=12.6, color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.04, "직선이 되면 와이블이다.\n기울기가 곧 모양 $k$ 다.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10.6,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GREY, alpha=0.95))

    # ── 오른쪽: 부품 수가 늘수록 이론값으로 ──
    ax = axes[1]
    for a, ls in ((1, "-"), (2, "-"), (3, "-")):
        ax.axhline(a, color=GREY, lw=1.4, ls=":")
        ax.text(4.5, a + 0.05, f"$a = {a}$", fontsize=10.8, color=GREY,
                ha="left")
    for name, (ks, a, col) in curves.items():
        ax.plot(NS, ks, lw=2.6, color=col, marker="o", ms=6, label=name)
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([f"{n:,}".replace(",", ",") for n in NS])
    ax.minorticks_off()
    ax.annotate("부품이 많아질수록\n부품 분포의 모양은 잊히고\n"
                "왼쪽 꼬리 지수만 남는다",
                xy=(1000, 2.0), xytext=(40, 2.62), fontsize=10.8, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("이어 붙인 부품의 수 $n$", fontsize=11.5)
    ax.set_ylabel("적합한 와이블 모양 $k$", fontsize=11.5)
    ax.set_title("서로 다른 다섯 부품 분포", fontsize=12.6, color=INK, pad=8)
    ax.set_xlim(4, 1500)
    ax.set_ylim(0.75, 3.6)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right", ncol=2)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무엇에나 맞는 까닭이 있다 — 가장 약한 고리", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"각 조건 {R//10000} 만 번. 부품 강도의 분포가 $0$ 근처에서 "
             "$F(x) \\approx c\\,x^{a}$ 이면, 부품을 많이 이어 붙인 물건의 강도는 "
             "모양 $a$ 인 와이블로 간다 — 부품 분포의 나머지 부분은 잊힌다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "weibull_weakest_link.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
