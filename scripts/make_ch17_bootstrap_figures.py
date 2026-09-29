r"""17장 붓스트랩 절(bootstrap/) 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch17/bootstrap/img/bootstrap_world_analogy.png   두 세계의 유비와 중심화한 두 분포
  ch17/bootstrap/img/four_ci_one_distribution.png  한 붓스트랩 분포에서 읽는 네 구간
  ch17/bootstrap/img/resample_pairs_vs_separate.png  쌍을 지키는 재표집과 깨는 재표집
  ch17/bootstrap/img/parametric_vs_nonparametric.png  무엇에서 다시 뽑는가

실행:  python3 scripts/make_ch17_bootstrap_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch17/bootstrap/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def tidy(ax, left=False):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_color(MUTED)


# ===========================================================================
# 그림 1. 두 세계의 유비 (principle.md)
# ===========================================================================
def fig_analogy():
    rng = np.random.default_rng(2)
    n = 30
    true_med = np.log(2.0)

    x = rng.exponential(1.0, n)
    obs_med = np.median(x)

    # 현실 세계: F 에서 반복 표집한 중앙값
    real = np.median(rng.exponential(1.0, (40000, n)), axis=1) - true_med
    # 붓스트랩 세계: 관측 표본 하나에서 복원추출한 중앙값
    boot = np.median(x[rng.integers(0, n, (40000, n))], axis=1) - obs_med

    print("[analogy] obs_med=%.4f  real_sd=%.4f  boot_sd=%.4f"
          % (obs_med, real.std(ddof=1), boot.std(ddof=1)))
    print("[analogy] real 2.5/97.5 = %.3f %.3f | boot = %.3f %.3f"
          % (*np.percentile(real, [2.5, 97.5]), *np.percentile(boot, [2.5, 97.5])))

    grid = np.linspace(-0.55, 0.85, 1200)
    F_real = np.searchsorted(np.sort(real), grid, side="right") / real.size
    F_boot = np.searchsorted(np.sort(boot), grid, side="right") / boot.size
    k = int(np.argmax(np.abs(F_real - F_boot)))
    sup = abs(F_real[k] - F_boot[k])
    print("[analogy] sup |G_boot - G_real| = %.4f  at t = %.3f" % (sup, grid[k]))

    fig = plt.figure(figsize=(11.2, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], hspace=0.52, wspace=0.18)

    # --- (a) 현실 세계 ---
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 4.2, 400)
    ax.fill_between(t, 0.55 * np.exp(-t), color=BLUE_F, zorder=1)
    ax.plot(t, 0.55 * np.exp(-t), color=BLUE, lw=2, zorder=2)
    ax.plot(x, np.full(n, -0.10), "|", color=INK, ms=11, mew=1.4, zorder=3)
    ax.axvline(true_med, color=PURPLE, lw=1.6, ls="--", zorder=4)
    ax.text(true_med + 0.08, 0.70, r"$\theta = \ln 2$", color=PURPLE, fontsize=10.5)
    ax.text(1.9, 1.02, "모집단 $F$\n(알 수 없다)", color=BLUE, fontsize=10.5,
            ha="left", va="top")
    ax.text(4.15, -0.27, "손에 쥔 표본 하나 $(n = 30)$", color=INK, fontsize=9.5,
            ha="right")
    ax.set_ylim(-0.34, 1.10)
    ax.set_xlim(-0.12, 4.2)
    ax.set_title("(a) 현실 세계 — 표집을 되풀이할 수 없다", fontsize=11, color=INK)
    tidy(ax)

    # --- (b) 붓스트랩 세계 ---
    ax = fig.add_subplot(gs[0, 1])
    ax.vlines(x, 0, 0.55, color=ORANGE, lw=1.8, zorder=2)
    ax.plot(x, np.full(n, 0.55), "o", color=ORANGE, ms=4.0, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(obs_med, color=PURPLE, lw=1.6, ls="--", zorder=4)
    ax.text(obs_med + 0.08, 0.70, r"$\hat\theta = %.3f$" % obs_med,
            color=PURPLE, fontsize=10.5)
    ax.text(1.9, 1.02, r"$\hat F_n$ — 각 점에 질량 $1/30$" + "\n(완전히 안다)",
            color=ORANGE, fontsize=10.5, ha="left", va="top")
    ax.text(4.15, -0.27, "여기서 원하는 만큼 다시 뽑는다", color=INK, fontsize=9.5,
            ha="right")
    ax.set_ylim(-0.34, 1.10)
    ax.set_xlim(-0.12, 4.2)
    ax.set_title("(b) 붓스트랩 세계 — 몇 번이든 되풀이한다", fontsize=11, color=INK)
    tidy(ax)

    # --- (c) 중심화한 두 분포함수 ---
    ax = fig.add_subplot(gs[1, :])
    ax.plot(grid, F_real, color=BLUE, lw=2.4,
            label=r"현실 세계 $G$: $\hat\theta - \theta$ (모집단에서 40000번 표집)")
    ax.plot(grid, F_boot, color=ORANGE, lw=2.0, ls="--",
            label=r"붓스트랩 세계 $\hat G$: $\hat\theta^* - \hat\theta$ (표본 하나, $B = 40000$)")
    ax.vlines(grid[k], min(F_real[k], F_boot[k]), max(F_real[k], F_boot[k]),
              color=PURPLE, lw=2.4, zorder=5)
    ax.annotate(r"가장 벌어진 곳 $\sup_t |\hat G - G| = %.3f$" % sup,
                xy=(grid[k], (F_real[k] + F_boot[k]) / 2),
                xytext=(grid[k] + 0.14, 0.34), fontsize=10, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
    ax.axvline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlim(-0.55, 0.85)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("(c) 중심을 맞추면 두 분포함수가 포개진다 — 붓스트랩이 사는 이유",
                 fontsize=11, color=INK)
    ax.set_xlabel("중심에서 벗어난 양", fontsize=10, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.text(-0.52, 0.97,
            "표준편차\n현실 %.3f\n붓스트랩 %.3f" % (real.std(ddof=1), boot.std(ddof=1)),
            fontsize=9.5, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, lw=0.8))
    tidy(ax, left=True)

    save(fig, "bootstrap_world_analogy.png")


# ===========================================================================
# 그림 2. 한 붓스트랩 분포에서 읽는 네 개의 신뢰구간 (bootstrap.md)
# ===========================================================================
def fig_four_ci():
    rng = np.random.default_rng(4)
    n, B = 30, 20000
    true = np.log(2.0)

    x = rng.exponential(1.0, n)
    th = np.median(x)
    bs = np.median(x[rng.integers(0, n, (B, n))], axis=1)

    se = bs.std(ddof=1)
    zl, zu = -1.959964, 1.959964
    lo_p, hi_p = np.percentile(bs, [2.5, 97.5])

    z0 = stats.norm.ppf(np.clip((bs < th).mean(), 1e-6, 1 - 1e-6))
    jk = np.array([np.median(np.delete(x, i)) for i in range(n)])
    d = jk.mean() - jk
    a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)
    a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))
    lo_b, hi_b = np.percentile(bs, [100 * a1, 100 * a2])

    rows = [
        ("BCa", lo_b, hi_b, GREEN, "0.945"),
        ("기본 (추축)", 2 * th - hi_p, 2 * th - lo_p, RED, "0.821"),
        ("백분위수", lo_p, hi_p, BLUE, "0.943"),
        (r"정규 ($\hat\theta \pm 1.96\,\widehat{SE}$)", th - 1.96 * se, th + 1.96 * se,
         MUTED, "0.933"),
    ]
    print("[four_ci] th=%.4f se=%.4f z0=%.4f a=%.4f a1=%.4f a2=%.4f"
          % (th, se, z0, a, a1, a2))
    for name, lo, hi, _, cov in rows:
        print("   %-22s [%.3f, %.3f]  폭 %.3f  포함확률 %s"
              % (name, lo, hi, hi - lo, cov))
    print("[four_ci] skew=%.3f" % stats.skew(bs))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.4, 6.2), sharex=True,
        gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.16))

    bins = np.linspace(0.15, 1.38, 36)
    ax1.hist(bs, bins=bins, density=True, color=BLUE_F, edgecolor=BLUE, lw=0.9)
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(0, ymax * 1.30)
    ax1.axvline(th, color=PURPLE, lw=2.0)
    ax1.text(th + 0.02, ymax * 1.20,
             r"관측된 중앙값 $\hat\theta = %.3f$" % th,
             color=PURPLE, fontsize=10.5, va="center")
    ax1.axvline(true, color=INK, lw=1.5, ls="--")
    ax1.text(true - 0.02, ymax * 0.62,
             r"참값 $\ln 2 = 0.693$", color=INK, fontsize=10.5, ha="right",
             va="center")
    ax1.set_title(r"표본 하나에서 얻은 붓스트랩 중앙값 $B = 20000$ 개 (왜도 %.2f)"
                  % stats.skew(bs), fontsize=11.5, color=INK)
    tidy(ax1)

    for i, (name, lo, hi, col, cov) in enumerate(rows):
        y = i + 0.5
        ax2.plot([lo, hi], [y, y], color=col, lw=6, solid_capstyle="butt",
                 alpha=0.85, zorder=2)
        ax2.plot([lo, hi], [y, y], "|", color=col, ms=14, mew=2, zorder=3)
        ax2.text(lo - 0.02, y, "%.3f" % lo, ha="right", va="center",
                 fontsize=9, color=col)
        ax2.text(hi + 0.02, y, "%.3f" % hi, ha="left", va="center",
                 fontsize=9, color=col)
        ax2.text(0.17, y + 0.30, name, fontsize=10, color=col, va="bottom")
        ax2.text(1.33, y + 0.30, "포함확률 " + cov, fontsize=9.5, color=col,
                 va="bottom", ha="right")
    ax2.axvline(th, color=PURPLE, lw=2.0, zorder=1)
    ax2.axvline(true, color=INK, lw=1.5, ls="--", zorder=1)
    ax2.set_ylim(0, len(rows) + 0.05)
    ax2.set_xlim(0.12, 1.38)
    ax2.set_xlabel("중앙값", fontsize=10, color=INK)
    tidy(ax2)

    save(fig, "four_ci_one_distribution.png")


# ===========================================================================
# 그림 3. 쌍을 지키는 재표집과 깨는 재표집 (nonparametric.md)
# ===========================================================================
def fig_pairs():
    rng = np.random.default_rng(5)
    n, B = 40, 20000
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, 0.7, n)
    r_obs = stats.pearsonr(x, y).statistic

    def corr_rows(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))

    idx = rng.integers(0, n, (B, n))
    r_pair = corr_rows(x[idx], y[idx])

    ix = rng.integers(0, n, (B, n))
    iy = rng.integers(0, n, (B, n))
    r_sep = corr_rows(x[ix], y[iy])

    print("[pairs] r_obs=%.4f" % r_obs)
    print("   쌍 재표집  mean=%.4f sd=%.4f  95%%=[%.3f, %.3f]"
          % (r_pair.mean(), r_pair.std(ddof=1), *np.percentile(r_pair, [2.5, 97.5])))
    print("   따로 재표집 mean=%.4f sd=%.4f  95%%=[%.3f, %.3f]"
          % (r_sep.mean(), r_sep.std(ddof=1), *np.percentile(r_sep, [2.5, 97.5])))

    fig = plt.figure(figsize=(11.4, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.48, wspace=0.22)

    one = np.random.default_rng(11)
    j = one.integers(0, n, n)
    kx, ky = one.integers(0, n, n), one.integers(0, n, n)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x, y, s=26, color=MUTED, alpha=0.55, zorder=2,
               label="원자료 40쌍")
    ax.scatter(x[j], y[j], s=52, facecolor="none", edgecolor=BLUE, lw=1.4,
               zorder=3, label="붓스트랩 표본")
    ax.set_title("(a) 쌍을 통째로 뽑는다 — 관계가 남는다", fontsize=11, color=INK)
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)
    ax.text(0.03, 0.94, r"$r^* = %.3f$" % stats.pearsonr(x[j], y[j]).statistic,
            transform=ax.transAxes, fontsize=11, color=BLUE, va="top")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    tidy(ax, left=True)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(x, y, s=26, color=MUTED, alpha=0.55, zorder=2)
    ax.scatter(x[kx], y[ky], s=52, facecolor="none", edgecolor=ORANGE, lw=1.4,
               zorder=3)
    ax.set_title("(b) $x$ 와 $y$ 를 따로 뽑는다 — 관계가 사라진다",
                 fontsize=11, color=INK)
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)
    ax.text(0.03, 0.94, r"$r^* = %.3f$" % stats.pearsonr(x[kx], y[ky]).statistic,
            transform=ax.transAxes, fontsize=11, color=ORANGE, va="top")
    tidy(ax, left=True)

    ax = fig.add_subplot(gs[1, :])
    bins = np.linspace(-0.6, 0.95, 110)
    h_sep, _ = np.histogram(r_sep, bins=bins, density=True)
    h_pair, _ = np.histogram(r_pair, bins=bins, density=True)
    top = max(h_sep.max(), h_pair.max())

    ax.hist(r_sep, bins=bins, density=True, color=ORANGE_F, edgecolor=ORANGE,
            lw=0.9)
    ax.hist(r_pair, bins=bins, density=True, color=BLUE_F, edgecolor=BLUE,
            lw=0.9)
    ax.axvline(r_obs, color=PURPLE, lw=2.0)
    ax.text(0.0, h_sep.max() * 1.10,
            "따로 재표집 = 검정용 분포\n"
            r"평균 $%.3f$, 표준편차 $%.3f$" % (r_sep.mean(), r_sep.std(ddof=1)),
            color=ORANGE, fontsize=10, ha="center", va="bottom")
    ax.text(0.70, h_pair.max() * 1.04,
            "쌍 재표집 = 추정용 분포\n"
            r"평균 $%.3f$, 표준편차 $%.3f$" % (r_pair.mean(), r_pair.std(ddof=1)),
            color=BLUE, fontsize=10, ha="left", va="bottom")
    ax.text(r_obs - 0.025, top * 1.24, r"관측 $r = %.3f$" % r_obs,
            color=PURPLE, fontsize=10.5, ha="right", va="center")
    ax.set_xlim(-0.62, 1.22)
    ax.set_ylim(0, top * 1.32)
    ax.set_title("(c) 같은 도구가 서로 다른 두 분포를 만든다", fontsize=11, color=INK)
    ax.set_xlabel(r"붓스트랩 상관계수 $r^*$", fontsize=10, color=INK)
    tidy(ax)

    save(fig, "resample_pairs_vs_separate.png")


# ===========================================================================
# 그림 4. 무엇에서 다시 뽑는가 (parametric.md)
# ===========================================================================
def fig_parametric():
    rng = np.random.default_rng(7)
    n, B = 30, 20000
    target = -np.log(0.1)

    x = rng.exponential(1.0, n)
    mu, sd = x.mean(), x.std(ddof=1)

    q_np = np.percentile(x[rng.integers(0, n, (B, n))], 90, axis=1)
    q_exp = np.percentile(rng.exponential(mu, (B, n)), 90, axis=1)
    q_nrm = np.percentile(rng.normal(mu, sd, (B, n)), 90, axis=1)

    print("[param] n=%d  xbar=%.4f  s=%.4f  obs q90=%.4f"
          % (n, mu, sd, np.percentile(x, 90)))
    for name, q in (("비모수", q_np), ("Exp 모형", q_exp), ("정규 모형", q_nrm)):
        lo, hi = np.percentile(q, [2.5, 97.5])
        print("   %-9s [%.3f, %.3f]  폭 %.3f" % (name, lo, hi, hi - lo))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                   gridspec_kw=dict(wspace=0.16))

    # --- 왼쪽: 재표집의 원천 세 가지 ---
    t = np.linspace(-1.4, 5.2, 500)
    ax1.vlines(x, 0, 0.46, color=MUTED, lw=1.8, zorder=2)
    ax1.plot(x, np.full(n, 0.46), "o", color=MUTED, ms=4, zorder=3,
             label=r"$\hat F_n$ — 관측 30점 (비모수)")
    ax1.plot(t, stats.expon.pdf(t, scale=mu), color=GREEN, lw=2.2, zorder=4,
             label=r"적합된 $\mathrm{Exp}(1/\bar x)$ — 옳은 모형")
    ax1.plot(t, stats.norm.pdf(t, mu, sd), color=RED, lw=2.2, ls="--", zorder=4,
             label=r"적합된 $\mathcal{N}(\bar x, s^2)$ — 틀린 모형")
    ax1.fill_between(t[t <= 0], 0, stats.norm.pdf(t[t <= 0], mu, sd),
                     color=RED, alpha=0.18, zorder=1)
    ax1.axvline(0, color=INK, lw=0.9, ls=":")
    ax1.text(-1.35, 1.02, "정규모형은\n음수 대기시간을\n만든다",
             fontsize=9.5, color=RED, va="top", ha="left")
    ax1.set_xlim(-1.4, 5.2)
    ax1.set_ylim(0, 1.22)
    ax1.set_title("무엇에서 다시 뽑는가", fontsize=11.5, color=INK)
    ax1.set_xlabel("대기시간", fontsize=10, color=INK)
    ax1.legend(fontsize=9, frameon=False, loc="upper right")
    tidy(ax1)

    # --- 오른쪽: 90 백분위수의 붓스트랩 분포 ---
    bins = np.linspace(0.6, 6.0, 110)
    tops = [np.histogram(q, bins=bins, density=True)[0].max()
            for q in (q_np, q_exp, q_nrm)]
    ymax = max(tops)
    ax2.hist(q_np, bins=bins, density=True, color=MUTED, alpha=0.45,
             label="비모수 — 포함확률 0.855, 폭 1.77")
    ax2.hist(q_exp, bins=bins, density=True, histtype="step", color=GREEN,
             lw=2.0, label="Exp 모형 — 포함확률 0.967, 폭 1.93")
    ax2.hist(q_nrm, bins=bins, density=True, histtype="step", color=RED,
             lw=2.0, ls="--", label="정규 모형 — 포함확률 0.737, 폭 1.10")
    ax2.plot([target, target], [0, ymax * 1.04], color=PURPLE, lw=2.0)
    ax2.text(target + 0.10, ymax * 1.10, r"참값 $q_{0.9} = 2.303$",
             color=PURPLE, fontsize=10.5, va="center", ha="left")
    ax2.set_xlim(0.6, 6.0)
    ax2.set_ylim(0, ymax * 1.55)
    ax2.set_title(r"90 백분위수 $q_{0.9}^*$ 의 붓스트랩 분포", fontsize=11.5,
                  color=INK)
    ax2.set_xlabel(r"$q_{0.9}^*$", fontsize=10, color=INK)
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    tidy(ax2)

    save(fig, "parametric_vs_nonparametric.png")


if __name__ == "__main__":
    fig_analogy()
    fig_four_ci()
    fig_pairs()
    fig_parametric()
