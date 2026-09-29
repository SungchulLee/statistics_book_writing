r"""12장 상관 검정 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch12/correlation_test/img/significance_vs_effect.png   유의성과 크기는 다른 것이다
  ch12/correlation_test/img/spearman_exact_and_power.png 정확분포와 검정력
  ch12/correlation_test/img/kendall_exact_vs_normal.png  작은 n 에서 정규근사의 실패
  ch12/correlation_test/img/comparing_power.png          두 상관을 견주려면 표본이 얼마나

실행:  python3 scripts/make_ch12_correlation_test_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os
from itertools import permutations

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

OUT = "docs/ch12/correlation_test/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def headroom(ax, frac=0.28):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def bivariate(gen, rho, n):
    z1 = gen.normal(0, 1, n)
    z2 = gen.normal(0, 1, n)
    return z1, rho * z1 + np.sqrt(1 - rho ** 2) * z2


# === 그림 1. 유의성과 효과크기 ===
def fig_significance_vs_effect():
    ns = np.unique(np.round(np.logspace(np.log10(5), 4, 90)).astype(int))
    tc = stats.t.ppf(0.975, ns - 2)
    r_crit = tc / np.sqrt(tc ** 2 + ns - 2)

    marks = [10, 25, 100, 1000, 10000]
    mvals = []
    for n in marks:
        t = stats.t.ppf(0.975, n - 2)
        mvals.append(t / np.sqrt(t ** 2 + n - 2))

    # 큰 표본, 작은 상관
    g = np.random.default_rng(5)
    while True:
        xb, yb = bivariate(g, 0.08, 1000)
        rb, pb = stats.pearsonr(xb, yb)
        if 0.075 < rb < 0.085:
            break
    # 작은 표본, 큰 상관
    g2 = np.random.default_rng(3)
    while True:
        xc, yc = bivariate(g2, 0.6, 10)
        rc, pc = stats.pearsonr(xc, yc)
        if 0.62 < rc < 0.66 and 0.03 < pc < 0.05:
            break

    fig = plt.figure(figsize=(13.2, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    ax.plot(ns, r_crit, color=BLUE, lw=2.4)
    ax.fill_between(ns, 0, r_crit, color=BLUE_F, alpha=0.6)
    ax.set_xscale("log")
    ticks = [5, 10, 100, 1000, 10000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["5", "10", "100", "1000", "10000"], fontsize=9.5,
                       color=INK)
    ax.scatter(marks, mvals, s=46, color=RED, zorder=3)
    for n, v in zip(marks, mvals):
        ax.annotate(f"{v:.3f}", xy=(n, v), xytext=(n * 1.25, v + 0.045),
                    fontsize=10, color=RED)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel("유의해지는 최소 |r|", fontsize=10.5, color=INK)
    ax.set_title("파란 영역은 " + r"$p > 0.05$" + " 인 곳이다", fontsize=12.5,
                 color=INK, pad=8)

    for k, (ax_i, x, y, r, p) in enumerate(
            [(1, xb, yb, rb, pb), (2, xc, yc, rc, pc)]):
        ax = fig.add_subplot(gs[0, ax_i])
        clean(ax)
        ax.scatter(x, y, s=10 if len(x) > 100 else 55, color=BLUE,
                   alpha=0.35 if len(x) > 100 else 0.85, edgecolor="none")
        s, b = np.polyfit(x, y, 1)
        t = np.linspace(x.min(), x.max(), 10)
        ax.plot(t, b + s * t, color=ORANGE, lw=2.0)
        headroom(ax, 0.30)
        ax.text(0.03, 0.97,
                f"n = {len(x)}\nr = {r:.3f}\np = {p:.3f}\n"
                + r"$r^2$" + f" = {100 * r ** 2:.1f}%",
                transform=ax.transAxes, fontsize=10.2, color=INK, va="top",
                linespacing=1.45)
        ax.set_xlabel("X", fontsize=10.5, color=INK)
        ax.set_ylabel("Y", fontsize=10.5, color=INK)
        ax.set_title("큰 표본, 작은 상관" if k == 0 else "작은 표본, 큰 상관",
                     fontsize=12.5, color=INK, pad=8)

    save(fig, "significance_vs_effect.png")
    for n, v in zip(marks, mvals):
        print(f"  n={n:6d}  최소 |r| = {v:.4f}")
    print(f"  큰 표본: n=1000 r={rb:.4f} p={pb:.4f}")
    print(f"  작은 표본: n=10  r={rc:.4f} p={pc:.4f}")


# === 그림 2. Spearman 의 정확분포와 검정력 ===
def fig_spearman():
    n = 8
    base = np.arange(1, n + 1)
    denom = n * (n ** 2 - 1)
    rs_all = np.empty(40320)
    for i, perm in enumerate(permutations(base)):
        d = base - np.array(perm)
        rs_all[i] = 1 - 6 * np.sum(d ** 2) / denom

    t_c = stats.t.ppf(0.975, n - 2)
    r_t = np.sqrt(t_c ** 2 / (t_c ** 2 + n - 2))       # t 근사 임계값
    size_t = float((np.abs(rs_all) >= r_t - 1e-12).mean())

    vals = np.unique(np.round(np.abs(rs_all), 10))
    r_exact = None
    for v in vals[::-1]:
        if (np.abs(rs_all) >= v - 1e-12).mean() > 0.05:
            break
        r_exact = v
    size_exact = float((np.abs(rs_all) >= r_exact - 1e-12).mean())

    # 검정력 비교
    reps = 4000
    g = np.random.default_rng(20)
    m = 30
    scen = ["이변량 정규", "단조 비선형", "이상점 10% 오염"]
    pw = {"Pearson": [], "Spearman": []}
    for s in scen:
        hit_p = hit_s = 0
        for _ in range(reps):
            x, y = bivariate(g, 0.45, m)
            if s == "단조 비선형":
                y = np.exp(2.6 * y)
            elif s == "이상점 10% 오염":
                k = max(1, int(0.10 * m))
                idx = g.choice(m, k, replace=False)
                x[idx] = g.normal(0, 7, k)
                y[idx] = g.normal(0, 7, k)
            rp, pp = stats.pearsonr(x, y)
            rsp, psp = stats.spearmanr(x, y)
            hit_p += (pp < 0.05) and (rp > 0)
            hit_s += (psp < 0.05) and (rsp > 0)
        pw["Pearson"].append(hit_p / reps)
        pw["Spearman"].append(hit_s / reps)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax = axes[0]
    clean(ax)
    atoms, counts = np.unique(np.round(rs_all, 10), return_counts=True)
    probs = counts / counts.sum()
    step = 12.0 / denom          # 이웃한 r_s 값 사이의 간격
    ax.bar(atoms, probs, width=step * 0.72, color=BLUE, alpha=0.85,
           edgecolor="none", label="정확분포")
    grid = np.linspace(-0.995, 0.995, 400)
    tt = grid * np.sqrt((n - 2) / (1 - grid ** 2))
    dens = stats.t.pdf(tt, n - 2) * np.sqrt(n - 2) / (1 - grid ** 2) ** 1.5
    ax.plot(grid, dens * step, color=RED, lw=2.2, label="t 근사")
    top = probs.max()
    ax.vlines([-r_exact, r_exact], 0, top * 1.12, color=GREEN, lw=1.8,
              linestyles="--")
    ax.vlines([-r_t, r_t], 0, top * 1.12, color=ORANGE, lw=1.8,
              linestyles=":")
    ax.set_ylim(0, top * 1.62)
    ax.text(0.02, 0.985,
            f"정확 임계값  {r_exact:.3f}  (실제 크기 {size_exact:.3f})\n"
            f"t 근사 임계값 {r_t:.3f}  (실제 크기 {size_t:.3f})",
            transform=ax.transAxes, fontsize=10.2, color=INK, va="top",
            linespacing=1.6)
    ax.set_xlabel(r"$H_0$" + " 아래에서의 " + r"$r_s$" + "  (n = 8)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("확률", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("귀무분포는 띄엄띄엄한 값만 가진다", fontsize=12.5,
                 color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    xs = np.arange(3)
    w = 0.34
    ax.bar(xs - w / 2, pw["Pearson"], width=w, color=ORANGE, alpha=0.9,
           label="Pearson")
    ax.bar(xs + w / 2, pw["Spearman"], width=w, color=BLUE, alpha=0.9,
           label="Spearman")
    for i in range(3):
        ax.text(xs[i] - w / 2, pw["Pearson"][i] + 0.018,
                f"{pw['Pearson'][i]:.2f}", ha="center", fontsize=10,
                color=ORANGE)
        ax.text(xs[i] + w / 2, pw["Spearman"][i] + 0.018,
                f"{pw['Spearman'][i]:.2f}", ha="center", fontsize=10,
                color=BLUE)
    ax.set_xticks(xs)
    ax.set_xticklabels(scen, fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("검정력 (n = 30, " + r"$\alpha = 0.05$" + ")",
                  fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("정규성이 깨지면 순위가 이긴다", fontsize=12.5, color=INK,
                 pad=8)

    save(fig, "spearman_exact_and_power.png")
    print(f"  n=8 정확 임계값={r_exact:.4f} (크기 {size_exact:.4f}), "
          f"t 근사={r_t:.4f} (크기 {size_t:.4f})")
    for i, s in enumerate(scen):
        print(f"  {s}: Pearson {pw['Pearson'][i]:.3f}, "
              f"Spearman {pw['Spearman'][i]:.3f}")


# === 그림 3. Kendall 의 정확분포와 정규근사 ===
def inversion_pmf(n):
    """순열의 역위 수 분포를 동적계획법으로 정확히 구한다."""
    pmf = np.array([1.0])
    for k in range(2, n + 1):
        new = np.zeros(len(pmf) + k - 1)
        csum = np.concatenate([[0.0], np.cumsum(pmf)])
        for j in range(len(new)):
            lo = max(0, j - k + 1)
            new[j] = csum[min(j, len(pmf) - 1) + 1] - csum[lo]
        pmf = new
    return pmf / pmf.sum()


def fig_kendall():
    n = 5
    pmf = inversion_pmf(n)
    M = n * (n - 1) // 2
    inv = np.arange(len(pmf))
    S = M - 2 * inv
    var_S = n * (n - 1) * (2 * n + 5) / 18
    s_obs = -6
    p_exact = float(pmf[np.abs(S) >= abs(s_obs)].sum())
    z_obs = s_obs / np.sqrt(var_S)
    p_norm = 2 * stats.norm.cdf(-abs(z_obs))

    # 여러 n 에 대한 정규근사 검정의 실제 크기
    ns = np.arange(5, 41)
    sizes = []
    for m in ns:
        pm = inversion_pmf(m)
        Mm = m * (m - 1) // 2
        Sm = Mm - 2 * np.arange(len(pm))
        vm = m * (m - 1) * (2 * m + 5) / 18
        sizes.append(float(pm[np.abs(Sm / np.sqrt(vm)) > 1.959963985].sum()))
    sizes = np.array(sizes)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax = axes[0]
    clean(ax)
    cols = [RED if abs(s) >= abs(s_obs) else BLUE for s in S]
    ax.bar(S, pmf, width=1.4, color=cols, alpha=0.8, edgecolor="none")
    grid = np.linspace(-M - 1, M + 1, 400)
    ax.plot(grid, stats.norm.pdf(grid, 0, np.sqrt(var_S)) * 2,
            color=INK, lw=2.0, label="정규근사")
    headroom(ax, 0.40)
    ax.text(0.02, 0.97,
            f"관측  S = {s_obs}\n"
            f"정확 p-값    {p_exact:.3f}\n"
            f"정규근사 p-값 {p_norm:.3f}",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
            linespacing=1.55)
    ax.set_xlabel(r"$H_0$" + " 아래에서의 S  (n = 5)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("확률", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("S 는 열한 개의 값만 가진다", fontsize=12.5, color=INK,
                 pad=8)

    ax = axes[1]
    clean(ax)
    ax.plot(ns, sizes, "o-", color=BLUE, lw=1.8, ms=4)
    ax.axhline(0.05, color=RED, lw=1.6, ls="--")
    ax.text(38.5, 0.0515, "명목 수준 0.05", fontsize=10, color=RED,
            ha="right", va="bottom")
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel("정규근사 검정의 실제 제1종 오류율", fontsize=10.5,
                  color=INK)
    ax.set_title("근사는 톱니를 그리며 수렴한다", fontsize=12.5, color=INK,
                 pad=8)

    save(fig, "kendall_exact_vs_normal.png")
    print(f"  n=5: Var(S)={var_S:.2f}, 정확 p={p_exact:.4f}, "
          f"정규 p={p_norm:.4f}")
    for m, sz in zip(ns, sizes):
        if m in (5, 6, 8, 10, 15, 20, 30, 40):
            print(f"  n={m:3d} 실제 크기 {sz:.4f}")


# === 그림 4. 두 상관의 비교 ===
def fig_comparing():
    r1, n1 = 0.65, 50
    r2, n2 = 0.40, 60
    z1, z2 = np.arctanh(r1), np.arctanh(r2)

    reps = 20000
    g = np.random.default_rng(88)
    s1 = np.empty(reps)
    s2 = np.empty(reps)
    for i in range(reps):
        x, y = bivariate(g, r1, n1)
        s1[i] = stats.pearsonr(x, y)[0]
        x, y = bivariate(g, r2, n2)
        s2[i] = stats.pearsonr(x, y)[0]
    flip = float((s2 > s1).mean())

    ns = np.arange(10, 601)
    zc = 1.959963985
    pow_single = (stats.norm.cdf(np.arctanh(r2) * np.sqrt(ns - 3) - zc)
                  + stats.norm.cdf(-np.arctanh(r2) * np.sqrt(ns - 3) - zc))
    d = z1 - z2
    pow_diff = (stats.norm.cdf(d * np.sqrt((ns - 3) / 2) - zc)
                + stats.norm.cdf(-d * np.sqrt((ns - 3) / 2) - zc))
    n_single = int(ns[np.argmax(pow_single >= 0.8)])
    n_diff = int(ns[np.argmax(pow_diff >= 0.8)])

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))

    ax = axes[0]
    clean(ax)
    bins = np.linspace(-0.1, 1.0, 70)
    ax.hist(s2, bins=bins, color=BLUE, alpha=0.7, edgecolor="none",
            label=r"$\rho = 0.40$" + f",  n = {n2}")
    ax.hist(s1, bins=bins, color=ORANGE, alpha=0.65, edgecolor="none",
            label=r"$\rho = 0.65$" + f",  n = {n1}")
    ax.axvline(r2, color=BLUE, lw=1.8, ls="--")
    ax.axvline(r1, color=ORANGE, lw=1.8, ls="--")
    headroom(ax, 0.30)
    ax.text(0.02, 0.97,
            f"두 표본의 r 이 뒤집히는 경우\n{100 * flip:.1f}%",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
            linespacing=1.5)
    ax.set_xlabel("표본상관 r", fontsize=10.5, color=INK)
    ax.set_ylabel("모의실험 횟수", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("참 상관이 달라도 표본분포는 겹친다", fontsize=12.5,
                 color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    ax.plot(ns, pow_single, color=GREEN, lw=2.3,
            label=r"$H_0\!: \rho = 0$" + " 검정 (" + r"$\rho = 0.40$" + ")")
    ax.plot(ns, pow_diff, color=PURPLE, lw=2.3,
            label=r"$H_0\!: \rho_1 = \rho_2$" + " 검정 (0.65 대 0.40)")
    ax.axhline(0.8, color=MUTED, lw=1.3, ls=":")
    for nn, col in [(n_single, GREEN), (n_diff, PURPLE)]:
        ax.plot([nn, nn], [0, 0.8], color=col, lw=1.2, ls=":")
        ax.scatter([nn], [0.8], s=45, color=col, zorder=3)
        ax.annotate(f"n = {nn}", xy=(nn, 0.8), xytext=(nn + 12, 0.63),
                    fontsize=10.5, color=col)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("집단당 표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel("검정력", fontsize=10.5, color=INK)
    ax.legend(fontsize=9.8, frameon=False, loc="lower right")
    ax.set_title("차이를 보려면 세 배 가까운 표본이 필요하다", fontsize=12.5,
                 color=INK, pad=8)

    save(fig, "comparing_power.png")
    print(f"  z1={z1:.4f} z2={z2:.4f} 뒤집힘={flip:.4f}")
    print(f"  80% 검정력: 단일 상관 n={n_single}, 차이 검정 n={n_diff}")


if __name__ == "__main__":
    fig_significance_vs_effect()
    fig_spearman()
    fig_kendall()
    fig_comparing()
