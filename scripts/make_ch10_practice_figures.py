r"""10장 실무 절(타당성 조건·Fisher 정확검정·효과크기)의 그림을 생성한다.

만드는 파일:
  ch10/practice/img/validity_type1.png    기대도수가 작을 때 실제 제1종 오류율
  ch10/practice/img/fisher_enumeration.png  주변합이 고정된 표를 모두 나열한다
  ch10/practice/img/chisq_vs_cramersv.png   카이제곱은 n 을 따라 커지고 V 는 그대로다

실행:  python3 scripts/make_ch10_practice_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  첫 번째 그림은 2x2 표를 모두 열거하므로 30초쯤 걸린다.
"""
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch10/practice/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# =====================================================================
# 1. 작은 기대도수에서 실제 제1종 오류율 (2x2, 두 집단 크기 m, 공통 p=0.3)
# =====================================================================
def exact_type1(m, p=0.3, alpha=0.05):
    """두 개의 독립 이항표본에서 세 검정의 실제 제1종 오류율을 정확히 계산한다."""
    x = np.arange(m + 1)
    pb = stats.binom.pmf(x, m, p)
    P = np.outer(pb, pb)
    X, Y = np.meshgrid(x, x, indexing="ij")
    a, b, c, d = X, m - X, Y, m - Y
    n = 2 * m
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    ok = (c1 > 0) & (c2 > 0)
    denom = np.where(ok, r1 * r2 * np.maximum(c1, 1) * np.maximum(c2, 1), 1)
    num = (a * d - b * c) ** 2
    chi = n * num / denom
    numy = np.clip(np.abs(a * d - b * c) - n / 2, 0, None) ** 2
    chiy = n * numy / denom

    p_plain = np.where(ok, stats.chi2.sf(chi, 1), 1.0)
    p_yates = np.where(ok, stats.chi2.sf(chiy, 1), 1.0)

    p_fisher = np.ones_like(P)
    cache = {}
    for i in range(m + 1):
        for j in range(m + 1):
            if not ok[i, j]:
                continue
            key = (int(a[i, j]), int(c1[i, j]))
            if key not in cache:
                cache[key] = stats.fisher_exact(
                    [[a[i, j], b[i, j]], [c[i, j], d[i, j]]])[1]
            p_fisher[i, j] = cache[key]

    return (float((P * (p_plain < alpha)).sum()),
            float((P * (p_yates < alpha)).sum()),
            float((P * (p_fisher < alpha)).sum()))


def fig_validity_type1():
    p = 0.3
    ms = np.arange(5, 51)
    plain, yates, fisher = [], [], []
    for m in ms:
        a, b, c = exact_type1(int(m), p)
        plain.append(a)
        yates.append(b)
        fisher.append(c)
    plain, yates, fisher = map(np.array, (plain, yates, fisher))
    emin = ms * p   # 가장 작은 기대도수

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- (가) 세 검정의 실제 유의수준 ---
    ax = axes[0]
    ax.axhline(0.05, color=MUTED, lw=1.4, ls="--", label="명목 수준 0.05")
    ax.axvline(5, color=GREEN, lw=1.3, ls=":")
    ax.text(5.15, 0.0655, "경험 법칙\n기대도수 5", fontsize=9.5, color=GREEN,
            va="top")
    ax.plot(emin, plain, color=BLUE, lw=2.0, marker="o", ms=3.0,
            label="카이제곱 (보정 없음)")
    ax.plot(emin, yates, color=ORANGE, lw=2.0, marker="s", ms=3.0,
            label="Yates 보정")
    ax.plot(emin, fisher, color=PURPLE, lw=1.8, ls="--",
            label="Fisher 정확검정")
    ax.set_ylim(0, 0.072)
    ax.set_xlim(1.2, 15.3)
    clean_axis(ax)
    ax.set_xlabel("가장 작은 기대 칸 도수", fontsize=11, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11, color=INK)
    ax.set_title(r"(가) 두 집단 $m$ 명씩, 공통 성공확률 0.3 에서 정확히 계산",
                 fontsize=12, color=INK, pad=9)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")

    # --- (나) m=8 에서 검정통계량의 실제 꼬리확률 대 카이제곱 근사 ---
    m = 8
    x = np.arange(m + 1)
    pb = stats.binom.pmf(x, m, p)
    P = np.outer(pb, pb)
    X, Y = np.meshgrid(x, x, indexing="ij")
    a, b, c, d = X, m - X, Y, m - Y
    n = 2 * m
    r1, r2, c1, c2 = a + b, c + d, a + c, b + d
    ok = (c1 > 0) & (c2 > 0)
    denom = np.where(ok, r1 * r2 * np.maximum(c1, 1) * np.maximum(c2, 1), 1)
    chi = np.where(ok, n * (a * d - b * c) ** 2 / denom, 0.0)

    vals = np.unique(np.round(chi[ok], 10))
    w = np.array([P[ok & (np.round(chi, 10) == v)].sum() for v in vals])
    surv = np.array([w[vals >= v].sum() for v in vals])

    tail_at_crit = float(w[vals >= 3.841].sum())

    ax = axes[1]
    grid = np.linspace(0, 9, 400)
    ax.plot(grid, stats.chi2.sf(grid, 1), color=ORANGE, lw=2.2,
            label=r"$\chi^2_1$ 근사")
    ax.step(np.concatenate([[0], vals, [9]]),
            np.concatenate([[1.0], surv, [0.0]]),
            where="post", color=BLUE, lw=2.0, label="실제 꼬리확률 (계단)")
    ax.axhline(0.05, color=MUTED, lw=1.3, ls="--")
    ax.axvline(3.841, color=GREEN, lw=1.3, ls=":")
    ax.text(3.95, 0.72, "임계값 3.841", fontsize=9.5, color=GREEN)
    ax.text(8.9, 0.062, "0.05", ha="right", fontsize=10, color=INK)
    ax.text(4.35, 0.20, f"임계값에서 실제 꼬리확률은\n{tail_at_crit:.4f} 이다",
            fontsize=10, color=BLUE, va="top")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    ax.set_xlabel(r"카이제곱 통계량 $x$", fontsize=11, color=INK)
    ax.set_ylabel(r"$P(\chi^2 \geq x)$", fontsize=11, color=INK)
    ax.set_title(r"(나) $m=8$ 일 때 통계량이 실제로 갖는 값은 몇 개뿐이다",
                 fontsize=12, color=INK, pad=9)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, OUT + "validity_type1.png")


# =====================================================================
# 2. Fisher 정확검정: 주변합이 고정된 표를 모두 나열한다
# =====================================================================
def fig_fisher_enumeration():
    # 관측 표: [[1,5],[8,2]]  -> 행합 6, 10 / 열합 9, 7 / n=16
    r1, r2, c1, n = 6, 10, 9, 16
    lo, hi = max(0, c1 - r2), min(r1, c1)
    aa = np.arange(lo, hi + 1)
    pmf = stats.hypergeom.pmf(aa, n, r1, c1)
    a_obs = 1
    p_obs = stats.hypergeom.pmf(a_obs, n, r1, c1)
    extreme = pmf <= p_obs + 1e-12
    pval = pmf[extreme].sum()

    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    colors = [ORANGE if e else BLUE_F for e in extreme]
    edges = [ORANGE if e else BLUE for e in extreme]
    ax.bar(aa, pmf, width=0.62, color=colors, edgecolor=edges, linewidth=1.4)
    for x, y, e in zip(aa, pmf, extreme):
        ax.text(x, y + 0.008, f"{y:.4f}", ha="center", va="bottom",
                fontsize=9.5, color=ORANGE if e else INK)
    # 각 표의 모양을 x 축 아래에 적는다
    for x in aa:
        ax.text(x, -0.030, f"{x}  {r1 - x}", ha="center", va="top",
                fontsize=9.5, color=INK, family="monospace")
        ax.text(x, -0.062, f"{c1 - x}  {r2 - c1 + x}", ha="center", va="top",
                fontsize=9.5, color=INK, family="monospace")
    ax.text(lo - 0.95, -0.030, "1행:", ha="left", va="top", fontsize=9.5,
            color=MUTED)
    ax.text(lo - 0.95, -0.062, "2행:", ha="left", va="top", fontsize=9.5,
            color=MUTED)

    ax.annotate("관측된 표", xy=(0.72, 0.014), xytext=(-0.95, 0.105),
                fontsize=11, color=RED, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(0.03, 0.96,
            "주변합 (6, 10 | 9, 7) 을 고정하면 가능한 표는 7개뿐이다.\n"
            "관측된 표보다 확률이 크지 않은 표(주황)를 모두 더한 것이 p-값이다.\n"
            f"p = {pval:.4f}",
            transform=ax.transAxes, fontsize=11, color=INK, va="top")

    ax.set_xticks(aa)
    ax.set_xlim(lo - 1.1, hi + 0.7)
    ax.set_ylim(-0.085, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left"]].set_color(MUTED)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.tick_params(labelsize=10, colors=INK)
    ax.set_xlabel(r"왼쪽 위 칸 $a$ (나머지는 주변합으로 결정된다)",
                  fontsize=11, color=INK, labelpad=34)
    ax.set_ylabel("초기하분포 확률", fontsize=11, color=INK)
    ax.set_title("주변합이 고정된 모든 표와 그 확률", fontsize=12.5,
                 color=INK, pad=10)
    fig.tight_layout()
    save(fig, OUT + "fisher_enumeration.png")


# =====================================================================
# 3. 카이제곱은 n 에 비례해 커지고 Cramér의 V 는 그대로다
# =====================================================================
def fig_chisq_vs_cramersv():
    table = np.array([[934, 1070], [113, 92], [20, 8]], dtype=float)
    n0 = table.sum()
    chi0 = stats.chi2_contingency(table)[0]
    q = min(table.shape) - 1
    v0 = np.sqrt(chi0 / (n0 * q))

    ns = np.linspace(100, 20000, 400)
    chis = chi0 * ns / n0                    # 비율이 같으면 정확히 비례한다
    pvals = stats.chi2.sf(chis, 2)
    crit = stats.chi2.ppf(0.95, 2)
    n_cross = crit / (chi0 / n0)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    # --- (가) 카이제곱과 p-값 ---
    ax = axes[0]
    ax.plot(ns, chis, color=BLUE, lw=2.3, label=r"$\chi^2$ 통계량")
    ax.axhline(crit, color=MUTED, lw=1.4, ls="--")
    ax.text(19800, crit + 3, "5% 임계값 5.991", ha="right", va="bottom",
            fontsize=10, color=INK)
    ax.axvline(n_cross, color=GREEN, lw=1.3, ls=":")
    ax.text(n_cross + 400, 78, f"n = {n_cross:.0f} 에서\n유의해진다",
            fontsize=10, color=GREEN, va="top")
    ax.plot([n0], [chi0], "o", color=RED, ms=7, zorder=5)
    ax.annotate(f"자료: n = {n0:.0f}\n" + r"$\chi^2$ = " + f"{chi0:.2f}, p = 0.0027",
                xy=(n0, chi0), xytext=(4200, 55), fontsize=10, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 110)
    clean_axis(ax)
    ax.set_xlabel(r"표본크기 $n$ (칸 비율은 그대로 둔 채 확대·축소)",
                  fontsize=11, color=INK)
    ax.set_ylabel(r"$\chi^2$ 통계량", fontsize=11, color=INK)
    ax.set_title(r"(가) $\chi^2$ 은 $n$ 에 비례해 끝없이 커진다", fontsize=12,
                 color=INK, pad=9)

    # --- (나) Cramér의 V ---
    ax = axes[1]
    ax.axhspan(0, 0.10, color="#ECEFF1")
    ax.axhspan(0.10, 0.30, color=BLUE_F, alpha=0.55)
    ax.axhspan(0.30, 0.50, color=ORANGE_F, alpha=0.55)
    ax.axhspan(0.50, 0.62, color=GREEN_F, alpha=0.55)
    for y, lab in [(0.05, "무시할 만함"), (0.20, "작음"),
                   (0.40, "중간"), (0.56, "큼")]:
        ax.text(19700, y, lab, ha="right", va="center", fontsize=10,
                color=INK)
    ax.plot(ns, np.full_like(ns, v0), color=PURPLE, lw=2.6)
    ax.plot([n0], [v0], "o", color=RED, ms=7, zorder=5)
    ax.text(2500, v0 + 0.035, f"V = {v0:.4f} 로 $n$ 과 무관하게 일정하다",
            fontsize=11, color=PURPLE)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0, 0.62)
    clean_axis(ax)
    ax.set_xlabel(r"표본크기 $n$", fontsize=11, color=INK)
    ax.set_ylabel("Cramér의 V", fontsize=11, color=INK)
    ax.set_title("(나) 효과크기는 표본을 키워도 움직이지 않는다",
                 fontsize=12, color=INK, pad=9)

    fig.tight_layout()
    save(fig, OUT + "chisq_vs_cramersv.png")
    print(f"  n0={n0} chi0={chi0:.4f} v0={v0:.4f} n_cross={n_cross:.1f}")
    for nn in [300, 600, 2237, 20000]:
        cc = chi0 * nn / n0
        print(f"  n={nn}: chi2={cc:.3f} p={stats.chi2.sf(cc, 2):.3g}")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_fisher_enumeration()
    fig_chisq_vs_cramersv()
    fig_validity_type1()
