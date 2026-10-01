r"""17장 붓스트랩 가설검정 절(bootstrap_testing/) 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch17/bootstrap_testing/img/null_by_shifting.png     귀무분포는 자료를 옮겨서 만든다
  ch17/bootstrap_testing/img/studentize_pivot.png     스튜던트화가 폭의 흔들림을 없앤다
  ch17/bootstrap_testing/img/pool_vs_center.png       합치기와 중심화
  ch17/bootstrap_testing/img/correlation_boundary.png 경계에 눌린 상관계수

실행:  python3 scripts/make_ch17_bootstrap_testing_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch17/bootstrap_testing/img/"
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


def step_hist(ax, v, bins, col, colf, lw=1.3):
    h, e = np.histogram(v, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax.fill_between(c, h, step="mid", color=colf, zorder=1)
    ax.step(c, h, where="mid", color=col, lw=lw, zorder=2)
    return h, c


# ===========================================================================
# 그림 1. 귀무분포는 자료를 옮겨서 만든다 (bootstrap_tests.md)
# ===========================================================================
def fig_shift():
    rng_data = np.random.default_rng(1)
    rng = np.random.default_rng(7)      # 본문 보기와 같은 난수열
    rng2 = np.random.default_rng(99)    # 비교용(옮기지 않은) 분포에만 쓴다
    B = 10000

    # --- 일표본 ---
    x = rng_data.exponential(5.0, 50) + 2.0
    mu0 = 5.0
    obs = x.mean()
    xc = x - obs + mu0
    nullb = xc[rng.integers(0, 50, (B, 50))].mean(axis=1)
    raw = x[rng2.integers(0, 50, (4 * B, 50))].mean(axis=1)
    p1 = ((np.abs(nullb - mu0) >= abs(obs - mu0)).sum() + 1) / (B + 1)
    print("[shift] x_bar=%.4f  이동량=%.4f  p=%.4f" % (obs, mu0 - obs, p1))

    # --- 이표본 ---
    y1 = rng_data.normal(52, 10, 40)
    y2 = rng_data.normal(48, 10, 40)
    d_obs = y1.mean() - y2.mean()
    pool = np.concatenate([y1, y2])
    P = pool[rng.integers(0, 80, (B, 80))]
    dnull = P[:, :40].mean(axis=1) - P[:, 40:].mean(axis=1)
    p2 = ((np.abs(dnull) >= abs(d_obs)).sum() + 1) / (B + 1)
    print("[shift] d_obs=%.4f  p=%.4f" % (d_obs, p2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                   gridspec_kw=dict(wspace=0.16))

    # (a) 일표본
    bins = np.linspace(1.4, 11.8, 76)
    h2, c2 = step_hist(ax1, raw, bins, MUTED, "#EEF1F3", lw=1.1)
    h, c = step_hist(ax1, nullb, bins, BLUE, BLUE_F)
    lo, hi = mu0 - abs(obs - mu0), mu0 + abs(obs - mu0)
    for msk in (c <= lo, c >= hi):
        ax1.fill_between(c, h, where=msk, step="mid", color=RED, alpha=0.75,
                         zorder=3)
    ymax = max(h.max(), h2.max())
    ax1.set_ylim(-0.34 * ymax, ymax * 1.62)
    ax1.plot(x, np.full(50, -0.10 * ymax), "|", color=INK, ms=9, mew=1.2)
    ax1.plot(xc, np.full(50, -0.24 * ymax), "|", color=BLUE, ms=9, mew=1.2)
    ax1.text(11.7, -0.10 * ymax, "원자료", color=INK, fontsize=9, ha="right",
             va="center",
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax1.text(11.7, -0.24 * ymax, "옮긴 자료", color=BLUE, fontsize=9,
             ha="right", va="center",
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    ax1.annotate("", xy=(mu0, -0.17 * ymax), xytext=(obs, -0.17 * ymax),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
    ax1.text((mu0 + obs) / 2, -0.055 * ymax, r"$-%.2f$ 만큼 옮긴다" % (obs - mu0),
             color=BLUE, fontsize=9.5, ha="center", va="bottom")
    ax1.plot([mu0, mu0], [0, ymax * 1.10], color=BLUE, lw=1.6, ls="--")
    ax1.plot([obs, obs], [0, ymax * 1.10], color=PURPLE, lw=2.0)
    ax1.text(mu0 - 0.15, ymax * 1.20, r"$\mu_0 = 5$", color=BLUE, fontsize=10.5,
             ha="right", va="center")
    ax1.text(obs + 0.15, ymax * 1.20, r"$\bar x = %.3f$" % obs, color=PURPLE,
             fontsize=10.5, va="center")
    ax1.text(11.7, ymax * 1.56,
             "회색: 옮기지 않고 재표집한 분포\n(검정에는 쓸 수 없다)",
             color=MUTED, fontsize=9, va="top", ha="right")
    ax1.text(11.7, ymax * 0.95, "붉은 꼬리 = $p$ 값\n$p = %.4f$" % p1,
             color=RED, fontsize=10.5, va="top", ha="right")
    ax1.set_xlim(1.4, 11.8)
    ax1.set_title("(a) 일표본: 자료를 옮겨 귀무가설을 강제한다", fontsize=11.5,
                  color=INK)
    ax1.set_xlabel("평균", fontsize=10, color=INK)
    tidy(ax1)

    # (b) 이표본
    bins = np.linspace(-9, 9, 70)
    h, c = step_hist(ax2, dnull, bins, GREEN, GREEN_F)
    for msk in (c <= -abs(d_obs), c >= abs(d_obs)):
        ax2.fill_between(c, h, where=msk, step="mid", color=RED, alpha=0.75,
                         zorder=3)
    ymax = h.max()
    ax2.set_ylim(-0.34 * ymax, ymax * 1.42)
    ax2.plot(y1, np.full(40, -0.10 * ymax) + 0, "|", color=MUTED, ms=0)
    ax2.plot([d_obs, d_obs], [0, ymax * 1.10], color=PURPLE, lw=2.0)
    ax2.plot([-d_obs, -d_obs], [0, ymax * 1.10], color=PURPLE, lw=1.0, ls=":")
    ax2.text(d_obs + 0.25, ymax * 1.20,
             r"$\bar x - \bar y = %.3f$" % d_obs, color=PURPLE, fontsize=10.5,
             va="center")
    ax2.text(-8.6, ymax * 1.05,
             "두 표본을 합친 뒤\n다시 40개씩 나누어 뽑는다", color=GREEN,
             fontsize=9.5, va="top")
    ax2.text(-8.6, ymax * 0.58, "붉은 꼬리 = $p$ 값\n$p = %.4f$" % p2,
             color=RED, fontsize=10.5, va="top")
    ax2.set_xlim(-9, 9)
    ax2.set_title("(b) 이표본: 라벨을 지우고 합친다", fontsize=11.5, color=INK)
    ax2.set_xlabel("평균차", fontsize=10, color=INK)
    tidy(ax2)

    save(fig, "null_by_shifting.png")


# ===========================================================================
# 그림 2. 스튜던트화가 폭의 흔들림을 없앤다 (single_mean.md)
# ===========================================================================
def fig_studentize():
    rng = np.random.default_rng(3)
    n, B = 10, 40000
    mu0 = 1.0

    # s 가 작은 표본과 큰 표본을 고른다
    small = large = None
    while small is None or large is None:
        x = rng.exponential(1.0, n)
        s = x.std(ddof=1)
        if small is None and s < 0.55:
            small = x
        if large is None and s > 1.6:
            large = x
    print("[stud] s(작은 표본)=%.3f  s(큰 표본)=%.3f"
          % (small.std(ddof=1), large.std(ddof=1)))

    curves = []
    for x, col, colf, lab in ((small, BLUE, BLUE_F, "$s$ 가 작은 표본"),
                              (large, ORANGE, ORANGE_F, "$s$ 가 큰 표본")):
        xc = x - x.mean() + mu0
        xb = xc[rng.integers(0, n, (B, n))]
        u = xb.mean(axis=1) - mu0
        t = u / (xb.std(axis=1, ddof=1) / np.sqrt(n))
        curves.append((u, t, col, colf, lab, x.std(ddof=1)))
        print("   %s: 비스튜던트화 sd=%.3f, 스튜던트화 2.5/97.5=%.2f/%.2f"
              % (lab, u.std(), *np.percentile(t, [2.5, 97.5])))

    ns = [10, 20, 50, 200]
    rates = {
        r"고전 $t$": ([0.104, 0.070, 0.069, 0.053], MUTED, "o"),
        "스튜던트화 붓스트랩": ([0.073, 0.056, 0.059, 0.049], GREEN, "s"),
        "비스튜던트화 붓스트랩": ([0.143, 0.091, 0.074, 0.054], RED, "^"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                             gridspec_kw=dict(wspace=0.22,
                                              width_ratios=[1, 1, 1.05]))

    bins = np.linspace(-1.6, 1.6, 120)
    for u, t, col, colf, lab, sd in curves:
        step_hist(axes[0], u, bins, col, colf)
    axes[0].set_xlim(-1.6, 1.6)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax * 1.30)
    for i, (u, t, col, colf, lab, sd) in enumerate(curves):
        axes[0].text(-1.55, ymax * (1.22 - 0.16 * i),
                     "%s ($s = %.2f$)" % (lab, sd), color=col, fontsize=9.5,
                     va="center")
    axes[0].set_title(r"(a) 비스튜던트화 $\bar x^* - \mu_0$", fontsize=11,
                      color=INK)
    axes[0].set_xlabel("귀무분포", fontsize=10, color=INK)
    tidy(axes[0])

    bins = np.linspace(-7, 4, 120)
    for u, t, col, colf, lab, sd in curves:
        step_hist(axes[1], t, bins, col, colf)
    axes[1].set_xlim(-7, 4)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax * 1.30)
    axes[1].text(-6.9, ymax * 1.20, "폭이 거의 겹친다", color=INK,
                 fontsize=10, va="center")
    axes[1].set_title(r"(b) 스튜던트화 $(\bar x^* - \mu_0)/(s^*/\sqrt{n})$",
                      fontsize=11, color=INK)
    axes[1].set_xlabel("귀무분포", fontsize=10, color=INK)
    tidy(axes[1])

    ax = axes[2]
    for lab, (v, col, mk) in rates.items():
        ax.plot(ns, v, mk + "-", color=col, lw=2, ms=7, label=lab)
    ax.axhline(0.05, color=PURPLE, lw=1.4, ls="--")
    ax.text(10.4, 0.0455, "명목 0.05", color=PURPLE, fontsize=9.5, ha="left",
            va="top")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_ylim(0.03, 0.16)
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10, color=INK)
    ax.set_ylabel("제1종 오류율", fontsize=10, color=INK)
    ax.set_title(r"(c) $\mathrm{Exp}(1)$ 에서의 실제 크기", fontsize=11,
                 color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    tidy(ax, left=True)

    save(fig, "studentize_pivot.png")


# ===========================================================================
# 그림 3. 합치기와 중심화 (two_means.md)
# ===========================================================================
def fig_pool_center():
    rng = np.random.default_rng(12)
    m, n, B = 30, 10, 40000
    x = rng.normal(0, 1, m)
    y = rng.normal(0, 3, n)
    se = np.sqrt(x.var(ddof=1) / m + y.var(ddof=1) / n)
    tobs = (x.mean() - y.mean()) / se

    z = np.concatenate([x, y])
    N = m + n
    xb = z[rng.integers(0, N, (B, m))]
    yb = z[rng.integers(0, N, (B, n))]
    t_pool = (xb.mean(1) - yb.mean(1)) / np.sqrt(
        xb.var(1, ddof=1) / m + yb.var(1, ddof=1) / n)

    xc, yc = x - x.mean(), y - y.mean()
    xb2 = xc[rng.integers(0, m, (B, m))]
    yb2 = yc[rng.integers(0, n, (B, n))]
    t_cent = (xb2.mean(1) - yb2.mean(1)) / np.sqrt(
        xb2.var(1, ddof=1) / m + yb2.var(1, ddof=1) / n)

    print("[pool] s_x=%.3f s_y=%.3f  t_obs=%.3f" % (x.std(ddof=1),
                                                    y.std(ddof=1), tobs))
    print("   합치기 귀무분포 2.5/97.5 = %.2f / %.2f"
          % (*np.percentile(t_pool, [2.5, 97.5]),))
    print("   중심화 귀무분포 2.5/97.5 = %.2f / %.2f"
          % (*np.percentile(t_cent, [2.5, 97.5]),))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw=dict(wspace=0.22,
                                                    width_ratios=[1.1, 1.0]))

    bins = np.linspace(-6, 6, 130)
    h1, c1 = step_hist(ax1, t_pool, bins, ORANGE, ORANGE_F)
    h2, c2 = step_hist(ax1, t_cent, bins, GREEN, "none")
    ax1.step(c2, h2, where="mid", color=GREEN, lw=2.2, zorder=4)
    ymax = max(h1.max(), h2.max())
    ax1.set_ylim(0, ymax * 1.34)
    qa = np.percentile(t_pool, 97.5)
    qb = np.percentile(t_cent, 97.5)
    ax1.plot([qa], [0.02 * ymax], "^", color=ORANGE, ms=9)
    ax1.plot([qb], [0.02 * ymax], "^", color=GREEN, ms=9)
    ax1.text(-5.8, ymax * 1.26,
             "합친 붓스트랩 — 97.5 백분위수 %.2f" % qa, color=ORANGE,
             fontsize=9.5, va="center")
    ax1.text(-5.8, ymax * 1.14,
             "중심화 붓스트랩 — 97.5 백분위수 %.2f" % qb, color=GREEN,
             fontsize=9.5, va="center")
    ax1.axvspan(qa, qb, color=PURPLE, alpha=0.12, zorder=0)
    ax1.text(-5.8, ymax * 0.86,
             "합치면 두 집단이 모두\n같은 분산을 갖게 되어\n귀무분포의 꼬리가 짧아진다",
             color=ORANGE, fontsize=9.5, va="top")
    ax1.annotate("이 띠에 떨어진 $t_{obs}$ 는\n합치기에서만 기각된다",
                 xy=((qa + qb) / 2, ymax * 0.18),
                 xytext=(4.3, ymax * 0.62), fontsize=9, color=PURPLE,
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.1))
    ax1.set_xlim(-6, 6)
    ax1.set_title(r"$m = 30\ (s_x \approx 1)$, $n = 10\ (s_y \approx 3)$ 에서의 귀무분포",
                  fontsize=11.5, color=INK)
    ax1.set_xlabel(r"스튜던트화 통계량 $t^*$", fontsize=10, color=INK)
    tidy(ax1)

    labels = ["합친\n붓스트랩", "중심화\n붓스트랩", "Welch $t$", "합동 $t$"]
    cols = [ORANGE, GREEN, MUTED, RED]
    data = {
        "$m=10$, $n=30$": [0.041, 0.052, 0.049, 0.003],
        "$m=30$, $n=10$": [0.064, 0.047, 0.055, 0.204],
        "$m=n=20$": [0.055, 0.049, 0.049, 0.053],
    }
    w = 0.24
    xs = np.arange(4)
    for k, (lab, vals) in enumerate(data.items()):
        ax2.bar(xs + (k - 1) * w, vals, width=w * 0.92,
                color=[cols[i] for i in range(4)],
                alpha=[0.45, 0.72, 1.0][k], edgecolor="white", lw=0.6)
    for k, (lab, vals) in enumerate(data.items()):
        ax2.text(3 + (k - 1) * w, min(vals[3] + 0.004, 0.215),
                 "%.3f" % vals[3], ha="center", va="bottom", fontsize=8.5,
                 color=RED)
    ax2.axhline(0.05, color=PURPLE, lw=1.4, ls="--")
    ax2.text(-0.45, 0.054, "명목 0.05", color=PURPLE, fontsize=9.5, ha="left",
             va="bottom")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylim(0, 0.235)
    ax2.set_ylabel("제1종 오류율", fontsize=10, color=INK)
    handles = [matplotlib.patches.Patch(facecolor=INK, alpha=al, label=lab)
               for al, lab in zip([0.45, 0.72, 1.0], data.keys())]
    ax2.legend(handles=handles, fontsize=9, frameon=False, loc="upper left",
               bbox_to_anchor=(0.02, 0.98))
    ax2.set_title("세 가지 배치에서의 실제 크기", fontsize=11.5, color=INK)
    tidy(ax2, left=True)

    save(fig, "pool_vs_center.png")


# ===========================================================================
# 그림 4. 경계에 눌린 상관계수 (correlation.md)
# ===========================================================================
def fig_corr_boundary():
    n, B = 50, 40000
    dat = np.random.default_rng(0)
    rng = np.random.default_rng(21)

    def gen(rho):
        v = dat.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        dat.integers(0, n, (5000, n))   # 본문 코드가 소비하는 난수를 맞춘다
        return v

    def boot_r(v):
        a = v[:, 0]
        b = v[:, 1]
        idx = rng.integers(0, n, (B, n))
        aa, bb = a[idx], b[idx]
        aa = aa - aa.mean(1, keepdims=True)
        bb = bb - bb.mean(1, keepdims=True)
        return (aa * bb).sum(1) / np.sqrt((aa * aa).sum(1) * (bb * bb).sum(1))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.6),
                             gridspec_kw=dict(hspace=0.48, wspace=0.16))

    for row, rho in enumerate((0.6, 0.95)):
        v = gen(rho)
        r = stats.pearsonr(v[:, 0], v[:, 1]).statistic
        rb = boot_r(v)
        zb = np.arctanh(rb)
        pct = np.percentile(rb, [2.5, 97.5])
        zse = 1 / np.sqrt(n - 3)
        fis = np.tanh([np.arctanh(r) - 1.96 * zse, np.arctanh(r) + 1.96 * zse])
        sk = stats.skew(rb)
        print("[corr] rho=%.2f r=%.4f 왜도=%.3f" % (rho, r, sk))
        print("   백분위수 [%.4f, %.4f] 폭 %.4f / Fisher [%.4f, %.4f] 폭 %.4f"
              % (*pct, pct[1] - pct[0], *fis, fis[1] - fis[0]))

        ax = axes[row, 0]
        lo = r - 4.6 * rb.std()
        hi = min(1.0, r + 4.6 * rb.std())
        bins = np.linspace(lo, min(1.0, hi + 0.02), 110)
        h, c = step_hist(ax, rb, bins, BLUE, BLUE_F)
        ymax = h.max()
        ax.set_ylim(-0.40 * ymax, ymax * 1.20)
        ax.plot([r, r], [0, ymax * 1.05], color=PURPLE, lw=2.0)
        ax.text(r, ymax * 1.14, r"$r = %.4f$" % r, color=PURPLE, fontsize=10,
                ha="center", va="center")
        for (a, b), yy, col, lab in ((pct, -0.13 * ymax, BLUE, "백분위수"),
                                     (fis, -0.28 * ymax, MUTED, "Fisher $z$")):
            ax.plot([a, b], [yy, yy], color=col, lw=5.5, solid_capstyle="butt")
            ax.plot([a, b], [yy, yy], "|", color=col, ms=11, mew=1.8)
            ax.text(a - 0.004 * (hi - lo) * 12, yy, lab, ha="right",
                    va="center", fontsize=9, color=col)
            ax.text(b + 0.004 * (hi - lo) * 12, yy, "폭 %.4f" % (b - a),
                    ha="left", va="center", fontsize=8.5, color=col)
        if hi >= 0.999:
            ax.axvline(1.0, color=RED, lw=1.6, ls=":")
            ax.text(1.0, ymax * 0.55, " 경계 $r = 1$", color=RED, fontsize=9.5,
                    va="center")
        ax.set_title(r"$\rho = %.2f$ — $r$ 척도, 왜도 $%.3f$" % (rho, sk),
                     fontsize=11, color=INK)
        ax.set_xlabel(r"$r^*$", fontsize=10, color=INK)
        tidy(ax)

        ax = axes[row, 1]
        h, c = step_hist(ax, zb, np.linspace(zb.min(), zb.max(), 110),
                         GREEN, GREEN_F)
        ymax = h.max()
        ax.set_ylim(0, ymax * 1.20)
        ax.plot([np.arctanh(r)] * 2, [0, ymax * 1.05], color=PURPLE, lw=2.0)
        ax.text(np.arctanh(r), ymax * 1.12,
                r"$\mathrm{arctanh}(r) = %.3f$" % np.arctanh(r), color=PURPLE,
                fontsize=10, ha="center", va="center")
        ax.set_title(r"$\rho = %.2f$ — $z = \mathrm{arctanh}(r)$ 척도, 왜도 $%.3f$"
                     % (rho, stats.skew(zb)), fontsize=11, color=INK)
        ax.set_xlabel(r"$z^*$", fontsize=10, color=INK)
        tidy(ax)

    save(fig, "correlation_boundary.png")


if __name__ == "__main__":
    fig_shift()
    fig_studentize()
    fig_pool_center()
    fig_corr_boundary()
