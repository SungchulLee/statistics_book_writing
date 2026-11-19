r"""4장 감마분포 쪽과 장 개요의 그림.

만드는 파일:
  docs/ch04/continuous_distributions/img/gamma_density.png
      λ = 2 로 두고 형상 α = 1..5 의 밀도. α = 1 이 지수분포다.
  docs/ch04/img/distribution_relations.png
      4장 분포들의 관계도 — 합, 극한, 같은 분포, 제곱·나누기.

실행:  python3 scripts/make_ch04_gamma_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
DISC, DISC_F = "#1565C0", "#DCEBFB"
CONT, CONT_F = "#E65100", "#FFE0B2"
INFER, INFER_F = "#2E7D32", "#DCEDC8"

OUT_GAMMA = "docs/ch04/continuous_distributions/img/"
OUT_INDEX = "docs/ch04/img/"


def fig_gamma_density():
    lam = 2.0
    x = np.linspace(0, 6, 1201)
    colors = ["#1565C0", "#C62828", "#6A1B9A", "#37474F", "#00838F"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for a, c in zip(range(1, 6), colors):
        y = stats.gamma(a=a, scale=1 / lam).pdf(x)
        lab = f"α = {a}" + ("  (지수분포 Exp(2))" if a == 1 else "")
        ax.plot(x, y, color=c, lw=2, label=lab)
        ax.plot([a / lam], [0], marker="^", color=c, ms=8, clip_on=False)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.05)
    ax.set_xlabel("x", color=INK)
    ax.set_ylabel("f(x)", color=INK)
    ax.set_title("감마분포 Gam(α, 2) 의 밀도 — 삼각형은 평균 α/λ", color=INK, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT_GAMMA + "gamma_density.png", dpi=170, facecolor="white")
    plt.close(fig)


NODES = {
    "Ber": (0.0, 4.0, "Ber(p)", "d"),
    "B": (3.0, 4.0, "B(n, p)", "d"),
    "Geo": (0.0, 2.4, "Geo(p)", "d"),
    "NB": (3.0, 2.4, "NB(r, p)", "d"),
    "Poi": (6.4, 2.4, "Poi(λ)", "d"),
    "Exp": (0.0, 0.8, "Exp(λ)", "c"),
    "Gam": (3.0, 0.8, "Gam(α, λ)", "c"),
    "Z": (6.4, 0.8, "Z ~ N(0, 1)", "c"),
    "chi": (3.0, -0.8, r"$\chi^2_d$", "i"),
    "t": (6.4, -0.8, r"$t_d$", "i"),
    "F": (3.0, -2.3, r"$F_{d_1,\,d_2}$", "i"),
}
STYLE = {"d": (DISC, DISC_F), "c": (CONT, CONT_F), "i": (INFER, INFER_F)}
W, H = 1.55, 0.56


def clip(x0, y0, x1, y1, gap=0.07):
    """(x0, y0) 상자의 가장자리에서 (x1, y1) 쪽으로 나가는 점."""
    dx, dy = x1 - x0, y1 - y0
    hw, hh = W / 2 + gap, H / 2 + gap
    t = min(hw / abs(dx) if dx else np.inf, hh / abs(dy) if dy else np.inf)
    return x0 + t * dx, y0 + t * dy


def edge(ax, a, b, kind, label, lpos=0.5, off=(0, 0), rad=0.0, ha="center"):
    xa, ya = NODES[a][:2]
    xb, yb = NODES[b][:2]
    ls = {"sum": "-", "limit": (0, (5, 3)), "same": (0, (1, 2.5))}[kind]
    col = {"sum": INK, "limit": "#C62828", "same": "#7B1FA2"}[kind]
    arrow = "-|>" if kind != "same" else "<|-|>"
    pa, pb = clip(xa, ya, xb, yb), clip(xb, yb, xa, ya)
    p = FancyArrowPatch(pa, pb, arrowstyle=arrow, mutation_scale=14,
                        lw=1.6, color=col, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(p)
    mx = xa + (xb - xa) * lpos + off[0]
    my = ya + (yb - ya) * lpos + off[1]
    ax.text(mx, my, label, fontsize=9.6, color=col, ha=ha, va="center",
            bbox=dict(fc="white", ec="none", pad=1.2), zorder=5)


def fig_relations():
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.set_xlim(-1.4, 8.4)
    ax.set_ylim(-2.9, 4.7)
    ax.axis("off")
    for key, (x, y, lab, k) in NODES.items():
        ec, fc = STYLE[k]
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.8, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=12.5, color=INK,
                zorder=4, weight="bold")

    edge(ax, "Ber", "B", "sum", "n 개를 더한다", off=(0, 0.22))
    edge(ax, "Ber", "Geo", "sum", "첫 성공까지\n기다린다", off=(-0.55, 0), ha="center")
    edge(ax, "Geo", "NB", "sum", "r 개를 더한다", off=(0, 0.22))
    edge(ax, "B", "Poi", "limit", "n → ∞, np = λ", lpos=0.55, off=(0.55, 0.15))
    edge(ax, "Geo", "Exp", "limit", "÷ n,  p = λ/n\nn → ∞", off=(-0.62, 0))
    edge(ax, "NB", "Gam", "limit", "÷ n,  p = λ/n\nn → ∞", off=(-0.62, 0))
    edge(ax, "Exp", "Gam", "sum", "α 개를 더한다", off=(0, 0.22))
    edge(ax, "Poi", "Gam", "same", "세기 ↔ 기다리기\n" + r"$P(N_t \geq r) = P(S_r \leq t)$",
         lpos=0.5, off=(0.25, 0.32))
    edge(ax, "Gam", "Z", "limit", "α → ∞ 표준화\n(중심극한정리)", off=(0, 0.3))
    edge(ax, "Gam", "chi", "same", r"$\chi^2_d$ = Gam(d/2, 1/2)", off=(-1.05, 0))
    edge(ax, "Z", "chi", "sum", "d 개를\n제곱해 더한다", lpos=0.5, off=(0.25, 0.3))
    edge(ax, "Z", "t", "sum", r"$Z\,/\,\sqrt{\chi^2_d/d}$", off=(0.85, 0))
    edge(ax, "chi", "t", "sum", "", off=(0, 0))
    edge(ax, "chi", "F", "sum", r"$(\chi^2_{d_1}/d_1)\,/\,(\chi^2_{d_2}/d_2)$", off=(-1.35, 0))
    edge(ax, "t", "F", "same", r"$t_d^{\,2} = F_{1,\,d}$", lpos=0.5, off=(0.75, -0.1))

    # 범례
    lx, ly = 5.2, -2.05
    items = [("-", INK, "만든다 (합·제곱·나누기)"),
             ((0, (5, 3)), "#C62828", "극한으로 간다"),
             ((0, (1, 2.5)), "#7B1FA2", "같은 분포 · 같은 사건")]
    for k, (ls, col, lab) in enumerate(items):
        y = ly - 0.32 * k
        ax.plot([lx, lx + 0.5], [y, y], ls=ls, color=col, lw=1.8)
        ax.text(lx + 0.62, y, lab, va="center", fontsize=9.8, color=INK)
    for k, (kk, lab) in enumerate([("d", "이산 (4.1절)"), ("c", "연속 (4.2절)"),
                                   ("i", "추론의 세 분포 (4.2절)")]):
        ec, fc = STYLE[kk]
        y = 4.45 - 0.0 * k
        x = -1.2 + 2.6 * k
        ax.add_patch(FancyBboxPatch((x, y - 0.12), 0.38, 0.24,
                                    boxstyle="round,pad=0.01,rounding_size=0.05",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + 0.5, y, lab, va="center", fontsize=10, color=INK)
    fig.savefig(OUT_INDEX + "distribution_relations.png", dpi=170,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_INDEX, exist_ok=True)
    fig_gamma_density()
    fig_relations()
    print("saved", OUT_GAMMA + "gamma_density.png", OUT_INDEX + "distribution_relations.png")
