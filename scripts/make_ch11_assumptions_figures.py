r"""11.5 가정 개관의 개념 그림을 생성한다.

분산검정 쪽 넷(바틀렛·카이제곱·F·레빈)의 그림은 그 쪽들이 15장으로
정리되면서 함께 지웠다.

만드는 파일:
  ch11/assumptions/img/assumption_costs.png      가정마다 위반의 대가가 다르다

실행:  python3 scripts/make_ch11_assumptions_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch11/assumptions/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def welch_p(groups):
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    k = len(n)
    w = n / v
    W = w.sum()
    m_t = (w * m).sum() / W
    tmp = np.sum((1 - w / W) ** 2 / (n - 1))
    F = ((w * (m - m_t) ** 2).sum() / (k - 1)) / (
        1 + 2 * (k - 2) / (k * k - 1) * tmp)
    return stats.f.sf(F, k - 1, (k * k - 1) / (3 * tmp))


# =====================================================================
# 1. 가정마다 대가가 다르다 — assumptions_overview.md
# =====================================================================
def fig_assumption_costs():
    rng = np.random.default_rng(1104)
    M = 8_000

    def clustered(n_cluster, m, icc):
        sb, sw = np.sqrt(icc), np.sqrt(1 - icc)
        cl = rng.normal(0, sb, n_cluster)
        return np.concatenate([cl[j] + rng.normal(0, sw, m)
                               for j in range(n_cluster)])

    rows = []

    # (1) 가정이 모두 성립
    a = b = 0
    for _ in range(M):
        g = [rng.standard_normal(10) for _ in range(3)]
        a += stats.f_oneway(*g).pvalue < 0.05
        b += stats.kruskal(*g).pvalue < 0.05
    rows.append(("가정 모두 성립\n$\\sigma$ = (1, 1, 1), $n$ = (10, 10, 10)",
                 a / M, b / M, "크러스컬–월리스"))

    # (2) 정규성 위반
    a = b = 0
    for _ in range(M):
        g = [rng.lognormal(0, 1, 10) for _ in range(3)]
        a += stats.f_oneway(*g).pvalue < 0.05
        b += stats.kruskal(*g).pvalue < 0.05
    rows.append(("정규성 위반\n대수정규, $n$ = (10, 10, 10)",
                 a / M, b / M, "크러스컬–월리스"))

    # (3) 등분산 위반 (본문 예시)
    a = b = 0
    for _ in range(M):
        g = [rng.normal(0, s, 10) for s in [1, 1, 3]]
        a += stats.f_oneway(*g).pvalue < 0.05
        b += welch_p(g) < 0.05
    rows.append(("등분산 위반 (균형)\n$\\sigma^2$ = (1, 1, 9), $n$ = (10, 10, 10)",
                 a / M, b / M, "웰치 분산분석"))

    # (4) 등분산 위반 + 불균형
    a = b = 0
    for _ in range(M):
        g = [rng.normal(0, s, n) for s, n in zip([1, 1, 3], [20, 20, 5])]
        a += stats.f_oneway(*g).pvalue < 0.05
        b += welch_p(g) < 0.05
    rows.append(("등분산 위반 + 불균형\n$\\sigma^2$ = (1, 1, 9), $n$ = (20, 20, 5)",
                 a / M, b / M, "웰치 분산분석"))

    # (5) 독립성 위반
    a = b = 0
    nc, mm, icc = 6, 5, 0.3
    for _ in range(M):
        g = [clustered(nc, mm, icc) for _ in range(3)]
        a += stats.f_oneway(*g).pvalue < 0.05
        cm = [np.array([x[j * mm:(j + 1) * mm].mean() for j in range(nc)])
              for x in g]
        b += stats.f_oneway(*cm).pvalue < 0.05
    rows.append(("독립성 위반\n군집 6 개 $\\times$ 5 명, ICC = 0.3",
                 a / M, b / M, "군집평균 분산분석"))

    for r in rows:
        print(f"  {r[0][:14]:16s} 고전 {r[1]:.4f}  처방 {r[2]:.4f}  ({r[3]})")

    # 검정력: 정규성 위반의 진짜 대가
    gens = {"정규": lambda n, d: rng.standard_normal(n) + d,
            "지수": lambda n, d: rng.exponential(1, n) + d,
            "대수정규": lambda n, d: rng.lognormal(0, 1, n) + d,
            "코시": lambda n, d: rng.standard_cauchy(n) + d}
    pw = []
    for lb, gen in gens.items():
        a = b = 0
        for _ in range(4_000):
            g = [gen(15, d) for d in (0, 0.8, 1.6)]
            a += stats.f_oneway(*g).pvalue < 0.05
            b += stats.kruskal(*g).pvalue < 0.05
        pw.append((lb, a / 4_000, b / 4_000))
        print(f"  검정력 {lb}: F {a / 4000:.4f}  KW {b / 4000:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8),
                             gridspec_kw=dict(width_ratios=[1.55, 1]))

    ax = axes[0]
    ys = np.arange(len(rows))[::-1]
    cls = [r[1] for r in rows]
    fix = [r[2] for r in rows]
    ax.barh(ys + 0.19, cls, height=0.36, color=BLUE_F, edgecolor=BLUE, lw=1.6,
            label="고전 $F$ 검정")
    ax.barh(ys - 0.19, fix, height=0.36, color=GREEN_F, edgecolor=GREEN,
            lw=1.6, label="가정에 맞춘 처방")
    for i, y in enumerate(ys):
        ax.text(cls[i] + 0.006, y + 0.19, f"{cls[i]:.3f}", va="center",
                fontsize=10, color=BLUE)
        ax.text(fix[i] + 0.006, y - 0.19, f"{fix[i]:.3f}  {rows[i][3]}",
                va="center", fontsize=9.5, color=GREEN)
    ax.axvline(0.05, color=RED, lw=1.5, ls="--")
    ax.text(0.057, ys[0] + 0.62, "명목 0.05", color=RED, fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 0.42)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xlabel("평균이 모두 같을 때의 실제 기각률", fontsize=10.5, color=INK)
    ax.set_title("같은 '위반'이라도 대가가 다르다", fontsize=12, color=INK)
    ax.legend(fontsize=10, loc="lower right")
    clean(ax)

    ax = axes[1]
    xs = np.arange(4)
    fv = [p[1] for p in pw]
    kv = [p[2] for p in pw]
    ax.bar(xs - 0.19, fv, width=0.36, color=BLUE_F, edgecolor=BLUE, lw=1.6,
           label="고전 $F$ 검정")
    ax.bar(xs + 0.19, kv, width=0.36, color=GREEN_F, edgecolor=GREEN, lw=1.6,
           label="크러스컬–월리스")
    for i in range(4):
        ax.text(xs[i] - 0.19, fv[i] + 0.015, f"{fv[i]:.2f}", ha="center",
                fontsize=9.5, color=BLUE)
        ax.text(xs[i] + 0.19, kv[i] + 0.015, f"{kv[i]:.2f}", ha="center",
                fontsize=9.5, color=GREEN)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[0] for p in pw], fontsize=10)
    ax.set_ylim(0, 1.32)
    ax.set_ylabel("검정력 ($\\mu$ = 0, 0.8, 1.6)", fontsize=10.5, color=INK)
    ax.set_title("정규성 위반의 진짜 대가는 검정력", fontsize=12, color=INK)
    ax.legend(fontsize=10, loc="upper center", ncol=2)
    clean(ax)

    fig.tight_layout()
    save(fig, "assumption_costs.png")


# =====================================================================
# 2. 바틀렛의 붕괴 — bartlett_test.md


if __name__ == "__main__":
    fig_assumption_costs()
