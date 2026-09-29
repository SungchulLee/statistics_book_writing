r"""11장 분산분석 가정 다섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch11/assumptions/img/assumption_costs.png      가정마다 위반의 대가가 다르다
  ch11/assumptions/img/bartlett_breakdown.png    바틀렛은 비정규에서 무너진다
  ch11/assumptions/img/levene_transform.png      절대편차로 바꾸면 분산 문제가 평균 문제가 된다
  ch11/assumptions/img/chi2_variance_power.png   분산 검정의 귀무분포와 검정력
  ch11/assumptions/img/ftest_normality.png       분산비 F 검정은 정규성에 목을 맨다

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
# =====================================================================
def fig_bartlett():
    rng = np.random.default_rng(1504)
    M = 8_000
    dists = [("균등", lambda n: rng.uniform(-1, 1, n)),
             ("정규", lambda n: rng.standard_normal(n)),
             ("$t_5$", lambda n: rng.standard_t(5, n)),
             ("지수", lambda n: rng.exponential(1, n)),
             ("대수정규", lambda n: rng.lognormal(0, 1, n))]
    res = []
    for lb, gen in dists:
        a = b = 0
        for _ in range(M):
            g = [gen(20) for _ in range(3)]     # 세 집단의 모분산이 정확히 같다
            a += stats.bartlett(*g).pvalue < 0.05
            b += stats.levene(*g, center="median").pvalue < 0.05
        res.append((lb, a / M, b / M))
        print(f"  {lb}: 바틀렛 {a / M:.4f}  브라운–포사이스 {b / M:.4f}")

    # 정규 자료에서의 검정력: sigma 비 1.2 (분산비 1.44)
    ns = [25, 50, 100, 200, 400, 800]
    pw = []
    for n in ns:
        hit = 0
        for _ in range(3_000):
            x = rng.normal(0, 1.0, n)
            y = rng.normal(0, 1.2, n)
            hit += stats.bartlett(x, y).pvalue < 0.05
        pw.append(hit / 3_000)
        print(f"  n = {n}: 검정력 {hit / 3000:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    xs = np.arange(len(res))
    bv = [r[1] for r in res]
    lv = [r[2] for r in res]
    ax.bar(xs - 0.19, bv, width=0.36, color=ORANGE_F, edgecolor=ORANGE, lw=1.6,
           label="바틀렛 검정")
    ax.bar(xs + 0.19, lv, width=0.36, color=GREEN_F, edgecolor=GREEN, lw=1.6,
           label="브라운–포사이스 (중앙값 레빈)")
    for i in range(len(res)):
        ax.text(xs[i] - 0.19, bv[i] + 0.018, f"{bv[i]:.3f}", ha="center",
                fontsize=9.5, color=ORANGE)
        ax.text(xs[i] + 0.19, lv[i] + 0.018, f"{lv[i]:.3f}", ha="center",
                fontsize=9.5, color=GREEN)
    ax.axhline(0.05, color=RED, lw=1.5, ls="--")
    ax.text(-0.45, 0.075, "명목 0.05", color=RED, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in res], fontsize=10.5)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.set_title("모분산이 정확히 같은 세 집단, 각 $n$ = 20",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper left")
    clean(ax)

    ax = axes[1]
    ax.plot(ns, pw, "o-", color=ORANGE, lw=2.2, ms=8)
    for x, y in zip(ns, pw):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(2, 12), ha="center", fontsize=9.5, color=ORANGE)
    ax.axhline(0.8, color=RED, lw=1.4, ls="--")
    ax.text(30, 0.83, "검정력 0.80", color=RED, fontsize=10)
    ax.axvline(100, color=MUTED, lw=1.2, ls=":")
    ax.text(112, 0.12, "본문 예제의 $n$ = 100", fontsize=10, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("집단당 표본크기 (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("기각률", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_title("정규 자료, $\\sigma$ 비 1.2 (분산비 1.44)",
                 fontsize=11.5, color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "bartlett_breakdown.png")


# =====================================================================
# 3. 레빈의 변환 — levene_test.md
# =====================================================================
def fig_levene():
    rng = np.random.default_rng(606)
    g1 = stats.norm(0, 1.0).rvs(50, random_state=0)
    g2 = stats.norm(0, 1.5).rvs(50, random_state=1)
    W, p = stats.levene(g1, g2, center="median")
    z1 = np.abs(g1 - np.median(g1))
    z2 = np.abs(g2 - np.median(g2))
    Fz, pz = stats.f_oneway(z1, z2)
    print(f"  levene W = {W:.4f} p = {p:.4f} / |편차|에 대한 분산분석 "
          f"F = {Fz:.4f} p = {pz:.4f}")
    print(f"  Z 평균: {z1.mean():.4f} vs {z2.mean():.4f}")

    # 중심의 선택
    M = 6_000
    dists = [("정규", lambda n: rng.standard_normal(n)),
             ("지수", lambda n: rng.exponential(1, n)),
             ("대수정규", lambda n: rng.lognormal(0, 1, n)),
             ("$t_3$", lambda n: rng.standard_t(3, n))]
    res = []
    for lb, gen in dists:
        a = b = 0
        for _ in range(M):
            g = [gen(20) for _ in range(3)]
            a += stats.levene(*g, center="mean").pvalue < 0.05
            b += stats.levene(*g, center="median").pvalue < 0.05
        res.append((lb, a / M, b / M))
        print(f"  {lb}: 평균 중심 {a / M:.4f}  중앙값 중심 {b / M:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.1]))

    jit = rng.uniform(-0.16, 0.16, (2, 50))
    ax = axes[0]
    for i, (g, c) in enumerate([(g1, BLUE), (g2, ORANGE)]):
        ax.scatter(i + jit[i], g, s=26, color=c, alpha=0.6, edgecolor="none")
        ax.hlines(np.median(g), i - 0.3, i + 0.3, color=INK, lw=2.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["집단 1  ($\\sigma$ = 1.0)", "집단 2  ($\\sigma$ = 1.5)"],
                       fontsize=10)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("원자료 $y$", fontsize=10.5, color=INK)
    ax.set_title("모평균은 같고 흩어짐만 다르다", fontsize=11.5, color=INK)
    clean(ax)

    ax = axes[1]
    for i, (z, c) in enumerate([(z1, BLUE), (z2, ORANGE)]):
        ax.scatter(i + jit[i], z, s=26, color=c, alpha=0.6, edgecolor="none")
        ax.hlines(z.mean(), i - 0.32, i + 0.32, color=RED, lw=2.6)
        ax.text(i, 3.95, f"$\\bar{{Z}}$ = {z.mean():.3f}",
                ha="center", fontsize=11, color=RED)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["집단 1", "집단 2"], fontsize=10.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.25, 4.4)
    ax.set_ylabel("$Z = |y - $ 중앙값 $|$", fontsize=10.5, color=INK)
    ax.set_title(f"이제 평균 차이 문제다  ($F$ = {Fz:.3f}, $p$ = {pz:.3f})",
                 fontsize=11, color=INK)
    clean(ax)

    ax = axes[2]
    xs = np.arange(len(res))
    mv = [r[1] for r in res]
    dv = [r[2] for r in res]
    ax.bar(xs - 0.19, mv, width=0.36, color=BLUE_F, edgecolor=BLUE, lw=1.6,
           label="center='mean' (원래 레빈)")
    ax.bar(xs + 0.19, dv, width=0.36, color=GREEN_F, edgecolor=GREEN, lw=1.6,
           label="center='median' (브라운–포사이스)")
    for i in range(len(res)):
        ax.text(xs[i] - 0.19, mv[i] + 0.006, f"{mv[i]:.3f}", ha="center",
                fontsize=9.5, color=BLUE)
        ax.text(xs[i] + 0.19, dv[i] + 0.006, f"{dv[i]:.3f}", ha="center",
                fontsize=9.5, color=GREEN)
    ax.axhline(0.05, color=RED, lw=1.5, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in res], fontsize=10.5)
    ax.set_ylim(0, 0.37)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.set_title("중심을 무엇으로 잡느냐가 전부다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9, loc="upper left")
    clean(ax)

    fig.tight_layout()
    save(fig, "levene_transform.png")


# =====================================================================
# 4. 분산에 대한 카이제곱 검정 — chi2_test_for_variance.md
# =====================================================================
def fig_chi2_variance():
    n = 100
    df = n - 1
    lo, hi = stats.chi2.ppf(0.025, df), stats.chi2.ppf(0.975, df)
    obs = [(1.00, 78.35), (1.05, 86.38), (1.10, 94.80),
           (1.15, 103.62), (1.20, 112.82)]
    print(f"  임계값 {lo:.3f}, {hi:.3f}")

    sigmas = np.linspace(1.0, 1.7, 60)
    power = []
    for s in sigmas:
        # T = (n-1)S^2/1 이고 참 분포는 sigma^2 * chi2(df)
        power.append(stats.chi2.sf(hi / s ** 2, df)
                     + stats.chi2.cdf(lo / s ** 2, df))
    power = np.array(power)
    s80 = sigmas[power >= 0.8][0]
    print(f"  검정력 0.8 이 되는 sigma = {s80:.4f} (분산비 {s80 ** 2:.3f})")
    print(f"  sigma = 1.2 에서 검정력 {np.interp(1.2, sigmas, power):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    xs = np.linspace(50, 165, 600)
    ax.plot(xs, stats.chi2.pdf(xs, df), color=INK, lw=2.2)
    inside = (xs >= lo) & (xs <= hi)
    ax.fill_between(xs[inside], stats.chi2.pdf(xs[inside], df), color=BLUE_F)
    for side in [xs < lo, xs > hi]:
        ax.fill_between(xs[side], stats.chi2.pdf(xs[side], df), color="#FBE9E7")
    ax.axvline(lo, color=RED, lw=1.5)
    ax.axvline(hi, color=RED, lw=1.5)
    ax.text(lo - 2, 0.0295, f"{lo:.2f}", color=RED, fontsize=10, ha="right")
    ax.text(hi + 2, 0.0295, f"{hi:.2f}", color=RED, fontsize=10, ha="left")
    for s, t in obs:
        ax.plot([t, t], [0, 0.0042], color=ORANGE, lw=2.4)
        ax.text(t, 0.0052, f"{s:.2f}", rotation=90, ha="center", va="bottom",
                fontsize=9, color=ORANGE)
    ax.text(141, 0.0205, "$\\sigma_Y$ 를 1.00 에서\n1.20 까지 키워도\n"
            "다섯 값이 모두\n채택역 한가운데\n머문다", ha="center",
            fontsize=10, color=INK)
    ax.set_xlim(50, 165)
    ax.set_ylim(0, 0.032)
    ax.set_xlabel("$T = (n-1)S^2/\\sigma_0^2$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(f"귀무분포 $\\chi^2({df})$ 와 본문의 다섯 통계량",
                 fontsize=11.5, color=INK)
    clean(ax)

    ax = axes[1]
    ax.plot(sigmas, power, color=PURPLE, lw=2.4)
    ax.axhline(0.8, color=RED, lw=1.4, ls="--")
    ax.text(1.005, 0.83, "검정력 0.80", color=RED, fontsize=10)
    ax.axhline(0.05, color=MUTED, lw=1.2, ls=":")
    ax.text(1.68, 0.075, "명목 0.05", color=MUTED, fontsize=10, ha="right")
    p12 = np.interp(1.2, sigmas, power)
    for xv, yv, c in [(1.2, p12, ORANGE), (s80, 0.8, RED)]:
        ax.plot([xv, xv], [0, yv], color=c, lw=1.1, ls=":")
        ax.scatter([xv], [yv], s=85, color=c, zorder=5, edgecolor="white")
    ax.text(1.40, 0.30,
            f"$\\sigma_Y$ = 1.20  →  검정력 {p12:.3f}\n"
            f"검정력 0.80  →  $\\sigma_Y$ = {s80:.2f}",
            fontsize=11, color=INK, ha="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=MUTED, lw=0.9))
    ax.set_xlabel("참 표준편차 $\\sigma_Y$  ($H_0$ 은 $\\sigma$ = 1)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("기각률", fontsize=10.5, color=INK)
    ax.set_xlim(1.0, 1.7)
    ax.set_ylim(0, 1.08)
    ax.set_title("$n$ = 100 에서의 이론 검정력", fontsize=11.5, color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "chi2_variance_power.png")


# =====================================================================
# 5. 분산비 F 검정과 정규성 — f_test_equality_of_variances.md
# =====================================================================
def fig_ftest_normality():
    rng = np.random.default_rng(1503)
    n = 100
    d1 = d2 = n - 1
    lo, hi = stats.f.ppf(0.025, d1, d2), stats.f.ppf(0.975, d1, d2)
    obs = [(1.00, 1.000), (1.05, 0.907), (1.10, 0.826),
           (1.15, 0.756), (1.20, 0.694)]
    print(f"  임계값 {lo:.4f}, {hi:.4f}")

    M = 8_000
    dists = [("정규", lambda m: rng.standard_normal(m)),
             ("균등", lambda m: rng.uniform(-1, 1, m)),
             ("$t_5$", lambda m: rng.standard_t(5, m)),
             ("지수", lambda m: rng.exponential(1, m)),
             ("대수정규", lambda m: rng.lognormal(0, 1, m))]
    res = []
    for lb, gen in dists:
        a = b = 0
        for _ in range(M):
            x, y = gen(30), gen(30)
            F = x.var(ddof=1) / y.var(ddof=1)
            p = 2 * min(stats.f.cdf(F, 29, 29), stats.f.sf(F, 29, 29))
            a += p < 0.05
            b += stats.levene(x, y, center="median").pvalue < 0.05
        res.append((lb, a / M, b / M))
        print(f"  {lb}: F 검정 {a / M:.4f}  레빈 {b / M:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    xs = np.linspace(0.35, 2.6, 600)
    ax.plot(xs, stats.f.pdf(xs, d1, d2), color=INK, lw=2.2)
    inside = (xs >= lo) & (xs <= hi)
    ax.fill_between(xs[inside], stats.f.pdf(xs[inside], d1, d2), color=BLUE_F)
    for side in [xs < lo, xs > hi]:
        ax.fill_between(xs[side], stats.f.pdf(xs[side], d1, d2),
                        color="#FBE9E7")
    ax.axvline(lo, color=RED, lw=1.5)
    ax.axvline(hi, color=RED, lw=1.5)
    ax.text(lo - 0.03, 2.05, f"{lo:.3f}", color=RED, fontsize=10, ha="right")
    ax.text(hi + 0.03, 2.05, f"{hi:.3f}", color=RED, fontsize=10, ha="left")
    for idx, (s, f) in enumerate(obs):
        h = 0.30 if idx % 2 == 0 else 0.62
        ax.plot([f, f], [0, h], color=ORANGE, lw=2.4)
        ax.text(f, h + 0.05, f"{s:.2f}", rotation=90, ha="center",
                va="bottom", fontsize=9, color=ORANGE)
    ax.annotate("$\\sigma_Y$ = 1.20 에서도 $F$ = 0.694 로\n"
                "임계값 0.673 을 넘지 못한다  ($p$ = 0.071)",
                (0.694, 0.32), textcoords="offset points", xytext=(74, 128),
                fontsize=10, color=ORANGE, ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.set_xlim(0.35, 2.6)
    ax.set_ylim(0, 2.3)
    ax.set_xlabel("$F = S_X^2 / S_Y^2$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(f"귀무분포 $F({d1}, {d2})$ 와 본문의 다섯 통계량",
                 fontsize=11.5, color=INK)
    clean(ax)

    ax = axes[1]
    xs = np.arange(len(res))
    fv = [r[1] for r in res]
    lv = [r[2] for r in res]
    ax.bar(xs - 0.19, fv, width=0.36, color=ORANGE_F, edgecolor=ORANGE, lw=1.6,
           label="분산비 $F$ 검정")
    ax.bar(xs + 0.19, lv, width=0.36, color=GREEN_F, edgecolor=GREEN, lw=1.6,
           label="레빈 (중앙값)")
    for i in range(len(res)):
        ax.text(xs[i] - 0.19, fv[i] + 0.012, f"{fv[i]:.3f}", ha="center",
                fontsize=9.5, color=ORANGE)
        ax.text(xs[i] + 0.19, lv[i] + 0.012, f"{lv[i]:.3f}", ha="center",
                fontsize=9.5, color=GREEN)
    ax.axhline(0.05, color=RED, lw=1.5, ls="--")
    ax.text(2.5, 0.078, "명목 0.05", color=RED, fontsize=10, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in res], fontsize=10.5)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.set_title("두 집단의 모분산이 정확히 같다, 각 $n$ = 30",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper left")
    clean(ax)

    fig.tight_layout()
    save(fig, "ftest_normality.png")


if __name__ == "__main__":
    fig_assumption_costs()
    fig_bartlett()
    fig_levene()
    fig_chi2_variance()
    fig_ftest_normality()
