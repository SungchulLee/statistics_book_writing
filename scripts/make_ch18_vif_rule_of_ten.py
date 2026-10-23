r"""18.1절 「그림으로 보는 통계학」 — $10$ 이라는 수와, 이름이 없어 걱정하지 않는 것.

계수의 분산은 네 가지가 함께 정한다.

$$
\operatorname{Var}(\hat\beta_j)
  = \frac{\sigma^2}{n\,s_j^2\,(1 - R_j^2)}
  = \underbrace{\frac{1}{1 - R_j^2}}_{\mathrm{VIF}_j}
    \cdot \frac{\sigma^2}{n\,s_j^2}
$$

$\mathrm{VIF}$ 는 그 가운데 **한 조각**이고, 분모의 $n$ 과 정확히 **같은 자리에서
곱해진다.** 그래서 $\mathrm{VIF} = 10$ 은 표본을 $10$ 배로 늘리면 그대로 상쇄된다.

이 스크립트가 그리는 것:

  ① $\mathrm{VIF}$ 를 $1,\,4,\,10,\,40$ 으로 두고 $\mathrm{SE}(\hat\beta_j)$ 를 $n$ 에
     대해 그린다. 같은 $\mathrm{SE}$ 에 닿는 $n$ 이 $\mathrm{VIF}$ 에 비례한다
  ② 세 분석을 나란히 — 오브라이언이 든 기준선($\mathrm{VIF} = 1.25$, $n = 100$)과,
     「심각」이라 불리는 $\mathrm{VIF} = 10$ 을 표본 $100$ 명·$800$ 명에서 본 것

만드는 파일:

  ch18/motivation/img/vif_rule_of_ten.png

실행:  python3 scripts/make_ch18_vif_rule_of_ten.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch18/motivation/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

VIFS = (1.0, 4.0, 10.0, 40.0)
BASE_VIF, BASE_N = 1.25, 100          # 오브라이언이 든 기준선


def se(vif, n, sigma=1.0, sj=1.0):
    """SE(beta_hat_j) — 분산식의 제곱근. sigma 와 s_j 는 1 로 고정한다."""
    return sigma / (sj * np.sqrt(n / vif))


def main():
    base = se(BASE_VIF, BASE_N)
    # 기준선과 같은 SE 를 주는 n 은 VIF 에 비례한다
    n_eq = {v: BASE_N * v / BASE_VIF for v in VIFS}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"기준선 — VIF {BASE_VIF}, n = {BASE_N}  ->  SE = {base:.5f}")
    print(" VIF   같은 SE 에 닿는 n   표본 배수")
    for v in VIFS:
        print(f"{v:5.2f} {n_eq[v]:14.0f} {n_eq[v]/BASE_N:11.1f} 배")

    cases = [("기준선\nVIF $1.25$ · $n=100$", BASE_VIF, BASE_N, MUTED),
             ("「심각」\nVIF $10$ · $n=100$", 10.0, 100, RED),
             ("같은 VIF, 표본만 늘려\nVIF $10$ · $n=800$", 10.0, 800, TEAL)]
    print("\n세 분석의 SE")
    for name, v, n, _ in cases:
        print(f"  VIF {v:5.2f}  n {n:4d}   SE {se(v, n):.5f}   "
              f"기준선의 {se(v, n)/base:.2f} 배")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① VIF 와 n 은 같은 자리에서 곱해진다 ──
    ax = axes[0]
    ns = np.linspace(20, 2000, 800)
    for v, col in zip(VIFS, (MUTED, BLUE, RED, AMBER)):
        ax.plot(ns, se(v, ns), lw=2.4, color=col, label=f"$\\mathrm{{VIF}} = {v:g}$")
    ax.axhline(base, color=INK, lw=1.6, ls="--")
    ax.text(1980, base * 1.06, f"기준선의 $\\mathrm{{SE}} = {base:.3f}$", ha="right",
            fontsize=10.6, color=INK)
    for v, col in zip(VIFS, (MUTED, BLUE, RED, AMBER)):
        if n_eq[v] <= 2000:
            ax.plot([n_eq[v]], [base], "o", ms=10, color=col, mec="white", mew=1.4,
                    zorder=5)
            ax.annotate(f"$n = {n_eq[v]:.0f}$", xy=(n_eq[v], base),
                        xytext=(n_eq[v], base * (1.55 if v == 10 else 1.3)),
                        ha="center", fontsize=10.4, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.set_xlim(0, 2050)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("$\\mathrm{SE}(\\hat\\beta_j)$   ($\\sigma = s_j = 1$)",
                  fontsize=11.5)
    ax.set_title("① $\\mathrm{VIF}$ 를 $10$ 배 하면 $n$ 을 $10$ 배 해서 지운다",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 세 분석 ──
    ax = axes[1]
    xs = np.arange(3)
    vals = [se(v, n) for _, v, n, _ in cases]
    cols = [c for *_, c in cases]
    ax.bar(xs, vals, width=0.56, color=cols, alpha=0.85)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.006, f"${v:.3f}$", ha="center", fontsize=12,
                color=INK, fontweight="bold")
    ax.axhline(base, color=INK, lw=1.4, ls="--")
    ax.text(2.45, base * 1.05, "기준선", ha="right", fontsize=10.6, color=INK)
    ax.annotate(f"${vals[1]/base:.2f}$ 배", xy=(1, vals[1]), xytext=(1, vals[1] * 0.55),
                ha="center", fontsize=12.5, color="white", fontweight="bold")
    ax.annotate(f"${vals[2]/base:.2f}$ 배", xy=(2, vals[2]), xytext=(2, vals[2] * 0.5),
                ha="center", fontsize=12.5, color="white", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, *_ in cases], fontsize=10.8)
    ax.set_ylim(0, 0.42)
    ax.set_ylabel("$\\mathrm{SE}(\\hat\\beta_j)$", fontsize=11.5)
    ax.set_title("② 같은 $\\mathrm{VIF}$ 가 두 가지 뜻을 가진다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$\\mathrm{VIF}$ 는 네 조각 가운데 하나다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "$\\operatorname{Var}(\\hat\\beta_j) = \\sigma^2 / [n\\,s_j^2\\,"
             "(1-R_j^2)]$ 에서 $\\sigma$ 와 $s_j$ 를 $1$ 로 두고 그린 것이다. "
             "모든 수는 이 식에서 직접 계산했다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "vif_rule_of_ten.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
