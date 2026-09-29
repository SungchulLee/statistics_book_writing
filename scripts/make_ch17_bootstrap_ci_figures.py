r"""17장 붓스트랩 신뢰구간 절(bootstrap_ci/) 일곱 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch17/bootstrap_ci/img/basic_is_a_mirror.png        기본 구간은 반사된 분포의 백분위수
  ch17/bootstrap_ci/img/percentile_invariance.png    백분위수 구간의 변환 불변성
  ch17/bootstrap_ci/img/bca_dial.png                 BCa 는 읽는 자리를 옮긴다
  ch17/bootstrap_ci/img/studentized_pivot.png        스튜던트화가 추축성을 만든다
  ch17/bootstrap_ci/img/median_bootstrap_atoms.png   중앙값 붓스트랩 분포의 이산성
  ch17/bootstrap_ci/img/coverage_vs_width.png        포함확률과 폭은 함께 보아야 한다
  ch17/bootstrap_ci/img/levels_and_coverage.png      신뢰수준의 폭과 장기 포함확률

실행:  python3 scripts/make_ch17_bootstrap_ci_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch17/bootstrap_ci/img/"
os.makedirs(OUT, exist_ok=True)

ZL, ZU = -1.959964, 1.959964


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


def bca_levels(bs, th, jk):
    z0 = stats.norm.ppf(np.clip((bs < th).mean(), 1e-6, 1 - 1e-6))
    d = jk.mean() - jk
    den = ((d ** 2).sum()) ** 1.5
    a = (d ** 3).sum() / (6 * den) if den > 0 else 0.0
    a1 = stats.norm.cdf(z0 + (z0 + ZL) / (1 - a * (z0 + ZL)))
    a2 = stats.norm.cdf(z0 + (z0 + ZU) / (1 - a * (z0 + ZU)))
    return z0, a, a1, a2


# ===========================================================================
# 그림 1. 기본 구간은 반사된 붓스트랩 분포의 백분위수 (bootstrap_ci.md)
# ===========================================================================
def fig_mirror():
    rng = np.random.default_rng(0)
    n, B = 30, 40000
    x = rng.exponential(1.0, n)
    th = x.mean()
    bs = x[rng.integers(0, n, (B, n))].mean(axis=1)
    ref = 2 * th - bs

    lo_p, hi_p = np.percentile(bs, [2.5, 97.5])
    lo_b, hi_b = np.percentile(ref, [2.5, 97.5])
    shift = 2 * (bs.mean() - th)
    print("[mirror] th=%.4f  boot mean=%.4f  shift=%.4f" % (th, bs.mean(), shift))
    print("   백분위수 [%.4f, %.4f] 폭 %.4f" % (lo_p, hi_p, hi_p - lo_p))
    print("   기본     [%.4f, %.4f] 폭 %.4f" % (lo_b, hi_b, hi_b - lo_b))

    mid_p = 0.5 * (lo_p + hi_p)
    mid_b = 0.5 * (lo_b + hi_b)
    print("   백분위수 구간의 중점 %.4f (theta_hat 보다 %+.4f)"
          % (mid_p, mid_p - th))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True,
                                   gridspec_kw=dict(hspace=0.34))
    bins = np.linspace(0.55, 1.95, 90)

    for ax, v, lo, hi, mid, col, colf, title in (
        (ax1, bs, lo_p, hi_p, mid_p, BLUE, BLUE_F,
         r"붓스트랩 분포 $\hat\theta^*$ — 여기서 읽으면 백분위수 구간"),
        (ax2, ref, lo_b, hi_b, mid_b, ORANGE, ORANGE_F,
         r"뒤집은 분포 $2\hat\theta - \hat\theta^*$ — 여기서 읽으면 기본 구간"),
    ):
        h, e = np.histogram(v, bins=bins, density=True)
        c = 0.5 * (e[1:] + e[:-1])
        ax.fill_between(c, h, step="mid", color=colf, zorder=1)
        ax.step(c, h, where="mid", color=col, lw=1.4, zorder=2)
        for msk in (c <= lo, c >= hi):
            ax.fill_between(c, h, where=msk, step="mid", color=col,
                            alpha=0.55, zorder=3)
        ymax = h.max()
        ax.plot([th, th], [-0.16 * ymax, ymax * 1.30], color=PURPLE, lw=2.0,
                zorder=4)
        ax.set_ylim(-0.46 * ymax, ymax * 1.52)
        ax.plot([lo, hi], [-0.24 * ymax] * 2, color=col, lw=6,
                solid_capstyle="butt", zorder=5)
        ax.plot([lo, hi], [-0.24 * ymax] * 2, "|", color=col, ms=13, mew=2,
                zorder=6)
        ax.plot([mid], [-0.24 * ymax], "v", color=INK, ms=8, zorder=7)
        ax.text(lo - 0.02, -0.24 * ymax, "%.3f" % lo, ha="right", va="center",
                fontsize=9.5, color=col)
        ax.text(hi + 0.02, -0.24 * ymax, "%.3f" % hi, ha="left", va="center",
                fontsize=9.5, color=col)
        ax.text(0.57, ymax * 1.46, title, fontsize=10.5, color=col, va="top")
        ax.text(1.93, -0.24 * ymax, "폭 %.4f" % (hi - lo), ha="right",
                va="center", fontsize=9.5, color=INK)
        ax.text(mid, -0.40 * ymax, "중점 %.3f" % mid, ha="center", va="center",
                fontsize=9, color=INK)
        tidy(ax)

    y1 = ax1.get_ylim()[1]
    ax1.text(th - 0.02, y1 * 0.80, r"$\hat\theta = %.4f$" % th, color=PURPLE,
             fontsize=10.5, ha="right", va="center")
    ax1.annotate("", xy=(mid_p, y1 * 0.74), xytext=(th, y1 * 0.74),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax1.text(mid_p + 0.03, y1 * 0.74,
             r"구간의 중점이 $\hat\theta$ 보다 $%.3f$ 오른쪽" % (mid_p - th),
             fontsize=9.5, color=INK, va="center")
    y2 = ax2.get_ylim()[1]
    ax2.annotate("", xy=(mid_b, y2 * 0.74), xytext=(th, y2 * 0.74),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax2.text(mid_b - 0.03, y2 * 0.74,
             r"그만큼 왼쪽으로 — 폭은 그대로", fontsize=9.5, color=INK,
             va="center", ha="right")
    ax2.set_xlabel("평균", fontsize=10, color=INK)
    ax2.set_xlim(0.55, 1.95)

    save(fig, "basic_is_a_mirror.png")


# ===========================================================================
# 그림 2. 백분위수 구간의 변환 불변성 (percentile.md)
# ===========================================================================
def fig_invariance():
    rng = np.random.default_rng(0)
    n, B = 40, 40000
    x = rng.lognormal(0, 1, n)
    m = x[rng.integers(0, n, (B, n))].mean(axis=1)
    lm = np.log(m)
    th = x.mean()

    p_raw = np.percentile(m, [2.5, 97.5])
    p_log = np.exp(np.percentile(lm, [2.5, 97.5]))
    se = m.std(ddof=1)
    n_raw = np.array([th - 1.96 * se, th + 1.96 * se])
    lse = lm.std(ddof=1)
    n_log = np.exp([np.log(th) - 1.96 * lse, np.log(th) + 1.96 * lse])
    print("[invariance] 백분위수 원척도 [%.4f, %.4f] / 로그척도 [%.4f, %.4f]"
          % (*p_raw, *p_log))
    print("[invariance] 정규근사 원척도 [%.4f, %.4f] / 로그척도 [%.4f, %.4f]"
          % (*n_raw, *n_log))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0),
                                   gridspec_kw=dict(wspace=0.14))

    # --- 왼쪽: 로그 척도 ---
    ax1.hist(lm, bins=90, density=True, color=GREEN_F, edgecolor=GREEN, lw=0.7)
    ax1.axvline(np.log(th), color=PURPLE, lw=1.8)
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(-0.30 * ymax, ymax * 1.10)
    ax1.plot(np.log(p_log), [-0.12 * ymax] * 2, color=BLUE, lw=6,
             solid_capstyle="butt")
    ax1.plot(np.log(n_log), [-0.23 * ymax] * 2, color=MUTED, lw=6,
             solid_capstyle="butt")
    ax1.text(np.log(p_log[0]) - 0.02, -0.12 * ymax, "백분위수", ha="right",
             va="center", fontsize=9.5, color=BLUE)
    ax1.text(np.log(n_log[0]) - 0.02, -0.23 * ymax, "정규근사", ha="right",
             va="center", fontsize=9.5, color=MUTED)
    ax1.text(np.log(th) + 0.02, ymax * 1.02, r"$\log \bar x$", color=PURPLE,
             fontsize=10, va="top")
    ax1.set_title(r"로그 척도 $\log \bar{x}^*$ — 거의 대칭", fontsize=11.5,
                  color=INK)
    ax1.set_xlabel(r"$\log \bar{x}^*$", fontsize=10, color=INK)
    tidy(ax1)

    # --- 오른쪽: 원 척도 ---
    ax2.hist(m, bins=90, density=True, color=ORANGE_F, edgecolor=ORANGE, lw=0.7)
    ax2.axvline(th, color=PURPLE, lw=1.8)
    ymax = ax2.get_ylim()[1]
    ax2.set_ylim(-0.46 * ymax, ymax * 1.10)
    rows = [
        (p_raw, -0.10 * ymax, BLUE, "백분위수 (원 척도)"),
        (p_log, -0.19 * ymax, BLUE, "백분위수 (로그 척도에서 되돌림)"),
        (n_raw, -0.30 * ymax, MUTED, "정규근사 (원 척도)"),
        (n_log, -0.39 * ymax, MUTED, "정규근사 (로그 척도에서 되돌림)"),
    ]
    for (lo, hi), y, col, lab in rows:
        ax2.plot([lo, hi], [y, y], color=col, lw=5.5, solid_capstyle="butt")
        ax2.plot([lo, hi], [y, y], "|", color=col, ms=11, mew=1.8)
        ax2.text(hi + 0.03, y, "%.4f   %s" % (hi, lab), fontsize=9, color=col,
                 va="center")
        ax2.text(lo - 0.03, y, "%.4f" % lo, fontsize=8.5, color=col,
                 va="center", ha="right")
    ax2.set_xlim(0.80, 2.72)
    ax2.set_title(r"원 척도 $\bar{x}^*$ — 오른쪽으로 치우쳐 있다",
                  fontsize=11.5, color=INK)
    ax2.set_xlabel(r"$\bar{x}^*$", fontsize=10, color=INK)
    ax2.text(1.78, ymax * 0.92,
             "파란 두 막대는\n끝점이 소수점까지 같다", fontsize=10, color=BLUE,
             va="top")
    ax2.text(th + 0.03, ymax * 1.02, r"$\bar x = %.3f$" % th, color=PURPLE,
             fontsize=10, va="top")
    tidy(ax2)

    save(fig, "percentile_invariance.png")


# ===========================================================================
# 그림 3. BCa 는 읽는 자리를 옮긴다 (bca.md)
# ===========================================================================
def fig_bca_dial():
    rng = np.random.default_rng(4)
    n, B = 20, 40000
    x = rng.exponential(1.0, n)
    th = x.var(ddof=1)
    bs = x[rng.integers(0, n, (B, n))].var(axis=1, ddof=1)
    jk = np.array([np.delete(x, i).var(ddof=1) for i in range(n)])
    z0, a, a1, a2 = bca_levels(bs, th, jk)
    pct = np.percentile(bs, [2.5, 97.5])
    bca = np.percentile(bs, [100 * a1, 100 * a2])
    print("[bca] th=%.4f z0=%.4f a=%.4f -> a1=%.4f a2=%.4f" % (th, z0, a, a1, a2))
    print("   백분위수 [%.3f, %.3f] / BCa [%.3f, %.3f]" % (*pct, *bca))

    fig = plt.figure(figsize=(12.4, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.30,
                          wspace=0.26)
    axu = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[1, 0], sharex=axu)
    ax2 = fig.add_subplot(gs[:, 1])

    # --- 왼쪽: 조정된 백분위수 수준 (위 = 상한, 아래 = 하한) ---
    aa = np.linspace(0, 0.20, 300)
    for z, ls in ((0.0, ":"), (z0, "-")):
        l1 = stats.norm.cdf(z + (z + ZL) / (1 - aa * (z + ZL)))
        l2 = stats.norm.cdf(z + (z + ZU) / (1 - aa * (z + ZU)))
        axl.plot(aa, 100 * l1, color=BLUE, ls=ls, lw=2.0)
        axu.plot(aa, 100 * l2, color=ORANGE, ls=ls, lw=2.0)

    axu.axhline(97.5, color=MUTED, lw=1, ls="--")
    axu.plot([a], [100 * a2], "o", color=ORANGE, ms=8, zorder=5)
    axu.text(a - 0.010, 100 * a2 + 0.10, "%.1f%%" % (100 * a2), color=ORANGE,
             fontsize=10, va="center", ha="right")
    axu.text(0.198, 97.3, "백분위수법: 97.5%", color=MUTED, fontsize=9,
             ha="right", va="top")
    axu.set_ylim(96.8, 100.15)
    axu.set_yticks([97.5, 98.5, 99.5])
    axu.set_yticklabels(["97.5", "98.5", "99.5"])
    axu.set_ylabel(r"상한 $\alpha_2$ (%)", fontsize=10, color=ORANGE)
    axu.set_title("BCa 는 읽어 내는 자리를 옮긴다", fontsize=11.5, color=INK)
    axu.tick_params(labelbottom=False)
    tidy(axu, left=True)

    axl.axhline(2.5, color=MUTED, lw=1, ls="--")
    axl.plot([a], [100 * a1], "o", color=BLUE, ms=8, zorder=5)
    axl.text(a - 0.010, 100 * a1 + 0.35, "%.1f%%" % (100 * a1), color=BLUE,
             fontsize=10, va="center", ha="right")
    axl.text(0.198, 2.2, "백분위수법: 2.5%", color=MUTED, fontsize=9,
             ha="right", va="top")
    axl.text(0.005, 11.6, "실선: 이 표본의 $\\hat z_0 = %.3f$" % z0,
             color=INK, fontsize=9)
    axl.text(0.005, 9.8, "점선: 편향보정이 없다면", color=MUTED, fontsize=9)
    axl.set_ylim(0, 13.2)
    axl.set_yticks([2.5, 6, 10])
    axl.set_yticklabels(["2.5", "6", "10"])
    axl.set_ylabel(r"하한 $\alpha_1$ (%)", fontsize=10, color=BLUE)
    axl.set_xlabel(r"가속 $\hat a$", fontsize=10, color=INK)
    axl.set_xlim(0, 0.20)
    tidy(axl, left=True)

    # --- 오른쪽: 같은 붓스트랩 분포, 다른 절단점 ---
    bins = np.linspace(0, 4.2, 100)
    h, e = np.histogram(bs, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax2.fill_between(c, h, step="mid", color=BLUE_F, zorder=1)
    ax2.step(c, h, where="mid", color=BLUE, lw=1.2, zorder=2)
    ymax = h.max()
    ax2.set_ylim(-0.42 * ymax, ymax * 1.24)
    ax2.plot([1, 1], [0, ymax * 1.02], color=PURPLE, lw=2.0, zorder=4)
    ax2.text(0.94, ymax * 1.12, "참값 $\\sigma^2 = 1$", color=PURPLE,
             fontsize=10.5, va="center", ha="right")
    ax2.plot([th, th], [0, ymax * 1.02], color=INK, lw=1.5, ls="--", zorder=4)
    ax2.text(th + 0.08, ymax * 1.12, r"$s^2 = %.3f$" % th, color=INK,
             fontsize=10.5, va="center")
    for (lo, hi), y, col, lab in (
            (pct, -0.14 * ymax, MUTED, "백분위수 2.5 / 97.5"),
            (bca, -0.30 * ymax, GREEN,
             "BCa %.1f / %.1f" % (100 * a1, 100 * a2))):
        ax2.plot([lo, hi], [y, y], color=col, lw=6, solid_capstyle="butt")
        ax2.plot([lo, hi], [y, y], "|", color=col, ms=13, mew=2)
        ax2.text(lo - 0.05, y, "%.2f" % lo, ha="right", va="center",
                 fontsize=9, color=col)
        ax2.text(hi + 0.05, y, "%.2f  %s" % (hi, lab), ha="left", va="center",
                 fontsize=9, color=col)
    ax2.set_xlim(-0.05, 4.9)
    ax2.set_xlabel(r"붓스트랩 분산 $s^{2*}$", fontsize=10, color=INK)
    ax2.set_title(r"$\mathrm{Exp}(1)$, $n = 20$ 의 분산 — 같은 분포, 다른 절단점",
                  fontsize=11.5, color=INK)
    tidy(ax2)

    save(fig, "bca_dial.png")


# ===========================================================================
# 그림 4. 스튜던트화가 추축성을 만든다 (bootstrap_t.md)
# ===========================================================================
def fig_studentized():
    rng = np.random.default_rng(2)
    n, B, M = 15, 40000, 200000

    # 참 추축량 분포: 모집단에서 반복 표집
    real = rng.exponential(1.0, (M, n))
    T_true = (real.mean(1) - 1.0) / (real.std(1, ddof=1) / np.sqrt(n))

    # 표본 하나에서의 붓스트랩
    x = rng.exponential(1.0, n)
    th, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    xb = x[rng.integers(0, n, (B, n))]
    bs = xb.mean(1)
    seb = xb.std(1, ddof=1) / np.sqrt(n)
    tstar = (bs - th) / seb

    q_true = np.percentile(T_true, [2.5, 97.5])
    q_star = np.percentile(tstar, [2.5, 97.5])
    q_t = stats.t.ppf([0.025, 0.975], n - 1)
    print("[stud] x_bar=%.4f se=%.4f" % (th, se))
    print("   참 추축량 2.5/97.5 = %.3f %.3f" % (*q_true,))
    print("   붓스트랩 t*        = %.3f %.3f" % (*q_star,))
    print("   t(14)              = %.3f %.3f" % (*q_t,))

    ci_t = (th - q_t[1] * se, th + q_t[1] * se)
    ci_bt = (th - q_star[1] * se, th - q_star[0] * se)
    ci_p = tuple(np.percentile(bs, [2.5, 97.5]))
    print("   고전 t %s / 붓스트랩-t %s / 백분위수 %s"
          % (np.round(ci_t, 3), np.round(ci_bt, 3), np.round(ci_p, 3)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 6.6),
                                   gridspec_kw=dict(height_ratios=[1.5, 1.0],
                                                    hspace=0.42))

    bins = np.linspace(-7, 3.2, 170)
    ax1.hist(T_true, bins=bins, density=True, color=BLUE_F, edgecolor=BLUE,
             lw=0.6, label=r"참 추축량 $(\bar x - \mu)/\widehat{se}$ (모집단에서 20만 번)")
    h, e = np.histogram(tstar, bins=bins, density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax1.step(c, h, where="mid", color=ORANGE, lw=1.8,
             label=r"붓스트랩 $t^*$ (표본 하나에서 $B = 40000$)")
    tt = np.linspace(-7, 3.2, 400)
    ax1.plot(tt, stats.t.pdf(tt, n - 1), color=RED, lw=2.0, ls="--",
             label=r"$t(14)$ 밀도 — 대칭을 가정한다")
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(-0.26 * ymax, ymax * 1.04)
    for q, col, dx, ha in ((q_true[0], BLUE, 0.10, "left"),
                           (q_star[0], ORANGE, -0.10, "right"),
                           (q_t[0], RED, 0.0, "center")):
        ax1.plot([q, q], [-0.085 * ymax, 0], color=col, lw=1.2, ls=":")
        ax1.plot([q], [-0.085 * ymax], "^", color=col, ms=9)
        ax1.text(q + dx, -0.135 * ymax, "%.2f" % q, color=col, fontsize=9.5,
                 ha=ha, va="top")
    ax1.text(-6.9, -0.135 * ymax, "각자의 2.5 백분위수", color=INK,
             fontsize=9.5, ha="left", va="top")
    ax1.set_xlim(-7, 3.2)
    ax1.set_title(r"$\mathrm{Exp}(1)$, $n = 15$ — 추축량의 왼쪽 꼬리는 $t$ 분포보다 훨씬 길다",
                  fontsize=11.5, color=INK)
    ax1.set_xlabel(r"$t$ 값", fontsize=10, color=INK)
    ax1.legend(fontsize=9.5, frameon=False, loc="upper left")
    tidy(ax1)

    rows = [
        ("붓스트랩-$t$  (포함확률 0.944)", ci_bt, GREEN),
        ("백분위수  (포함확률 0.896)", ci_p, BLUE),
        ("고전 $t$ 구간  (포함확률 0.903)", ci_t, MUTED),
    ]
    for i, (lab, (lo, hi), col) in enumerate(rows):
        y = i + 0.5
        ax2.plot([lo, hi], [y, y], color=col, lw=6, solid_capstyle="butt")
        ax2.plot([lo, hi], [y, y], "|", color=col, ms=13, mew=2)
        ax2.text(lo - 0.02, y, "%.3f" % lo, ha="right", va="center",
                 fontsize=9, color=col)
        ax2.text(hi + 0.02, y, "%.3f" % hi, ha="left", va="center",
                 fontsize=9, color=col)
        ax2.text(0.27, y + 0.28, lab, fontsize=10, color=col, va="bottom")
    ax2.axvline(1.0, color=PURPLE, lw=2.0)
    ax2.text(0.98, 3.25, r"참 평균 $\mu = 1$", color=PURPLE, fontsize=10.5,
             va="center", ha="right")
    ax2.axvline(th, color=INK, lw=1.2, ls="--")
    ax2.text(th + 0.02, 3.25, r"$\bar x = %.3f$" % th, color=INK, fontsize=10,
             ha="left", va="center")
    ax2.set_xlim(0.25, 2.72)
    ax2.set_ylim(0, 3.55)
    ax2.set_xlabel("평균", fontsize=10, color=INK)
    ax2.set_title("같은 표본에서 만든 세 구간", fontsize=11.5, color=INK)
    tidy(ax2)

    save(fig, "studentized_pivot.png")


# ===========================================================================
# 그림 5. 중앙값 붓스트랩 분포의 이산성 (bootstrap_median.md)
# ===========================================================================
def fig_median_atoms():
    rng = np.random.default_rng(7)
    n, B = 51, 40000
    x = np.sort(rng.lognormal(10.5, 0.8, n)) / 1000.0
    idx = rng.integers(0, n, (B, n))
    bm = np.median(x[idx], axis=1)
    bmean = x[idx].mean(axis=1)

    vals, cnt = np.unique(np.round(bm, 9), return_counts=True)
    q = np.percentile(bm, [2.5, 97.5])
    inner = ((vals >= q[0]) & (vals <= q[1])).sum()
    print("[median] n=%d  중앙값=%.1f  평균=%.1f" % (n, np.median(x), x.mean()))
    print("   붓스트랩 중앙값의 고유값 %d 개 / 가운데 95%% 안에는 %d 개"
          % (len(vals), inner))
    print("   SE(중앙값)=%.2f  SE(평균)=%.2f"
          % (bm.std(ddof=1), bmean.std(ddof=1)))

    counts = []
    for m in (15, 25, 51, 101, 201, 401):
        z = rng.lognormal(10.5, 0.8, m) / 1000.0
        v = np.median(z[rng.integers(0, m, (20000, m))], axis=1)
        u = np.unique(np.round(v, 9))
        qq = np.percentile(v, [2.5, 97.5])
        counts.append((m, int(((u >= qq[0]) & (u <= qq[1])).sum())))
    print("   가운데 95%% 안의 고유값 개수:", counts)

    fig = plt.figure(figsize=(12.8, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.92], wspace=0.26)

    lo, hi = 18, 62
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(bmean, bins=np.linspace(lo, hi, 80), density=True, color=BLUE_F,
            edgecolor=BLUE, lw=0.7)
    ax.axvline(x.mean(), color=PURPLE, lw=1.8)
    ax.set_xlim(lo, hi)
    ax.set_title("(a) 평균 — 매끄럽다", fontsize=11, color=INK)
    ax.set_xlabel("소득 (천 단위)", fontsize=10, color=INK)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.vlines(vals, 0, cnt / B, color=ORANGE, lw=2.4)
    ax.plot(vals, cnt / B, "o", color=ORANGE, ms=4)
    ax.axvline(np.median(x), color=PURPLE, lw=1.8)
    msk = (x > lo) & (x < hi)
    top = (cnt / B).max()
    ax.plot(x[msk], np.full(msk.sum(), -0.035 * top), "|", color=MUTED, ms=8,
            mew=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.155 * top, top * 1.30)
    ax.text(lo + 1, top * 1.24,
            "가운데 95%% 안에 값이 %d 개뿐이다" % inner, fontsize=10,
            color=ORANGE, va="top")
    ax.text(hi - 1, -0.105 * top, "회색 눈금: 원자료 51점", fontsize=8.5,
            color=MUTED, ha="right", va="center")
    ax.set_title("(b) 중앙값 — 듬성듬성하다", fontsize=11, color=INK)
    ax.set_xlabel("소득 (천 단위)", fontsize=10, color=INK)
    tidy(ax)

    ax = fig.add_subplot(gs[0, 2])
    ms = np.array([c[0] for c in counts], float)
    ks = np.array([c[1] for c in counts], float)
    ax.plot(ms, ks, "o-", color=ORANGE, lw=2, ms=6, label="실제 고유값 개수")
    ref = ks[2] / np.sqrt(ms[2]) * np.sqrt(ms)
    ax.plot(ms, ref, ls="--", color=MUTED, lw=1.6, label=r"$c\sqrt{n}$ 기준선")
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_ylim(0, max(ks.max(), ref.max()) * 1.35)
    ax.set_xlabel("표본크기 $n$", fontsize=10, color=INK)
    ax.set_ylabel("고유값 개수", fontsize=10, color=INK)
    ax.set_title(r"(c) 개수는 $\sqrt{n}$ 을 따른다", fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    tidy(ax, left=True)

    save(fig, "median_bootstrap_atoms.png")


# ===========================================================================
# 그림 6. 포함확률과 폭을 함께 본다 (comparison.md)
# ===========================================================================
def fig_coverage_width():
    easy = [
        ("정규", 0.940, 0.9494, MUTED),
        ("백분위수", 0.942, 0.9458, BLUE),
        ("기본", 0.934, 0.9458, ORANGE),
        ("BCa", 0.946, 0.9573, GREEN),
        (r"붓스트랩-$t$", 0.947, 0.9856, PURPLE),
    ]
    hard = [
        ("백분위수", 0.677, 1.59, BLUE),
        ("BCa", 0.736, 1.88, GREEN),
        (r"붓스트랩-$t$", 0.878, 14.15, PURPLE),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8),
                                   gridspec_kw=dict(wspace=0.22))

    for name, cov, wid, col in easy:
        ax1.plot([wid], [cov], "o", color=col, ms=10)
    ax1.text(0.9458, 0.942 + 0.0018, "백분위수", color=BLUE, fontsize=10,
             ha="center", va="bottom")
    ax1.text(0.9458, 0.934 - 0.0020, "기본", color=ORANGE, fontsize=10,
             ha="center", va="top")
    ax1.text(0.9494, 0.940 - 0.0020, "정규", color=MUTED, fontsize=10,
             ha="center", va="top")
    ax1.text(0.9573 + 0.0015, 0.946, "BCa", color=GREEN, fontsize=10,
             va="center")
    ax1.text(0.9856 - 0.0015, 0.947, r"붓스트랩-$t$", color=PURPLE, fontsize=10,
             ha="right", va="center")
    ax1.axhline(0.95, color=RED, lw=1.4, ls="--")
    ax1.text(0.9885, 0.9505, "명목 0.95", color=RED, fontsize=9.5, ha="right")
    ax1.set_xlim(0.938, 0.994)
    ax1.set_ylim(0.928, 0.953)
    ax1.set_xlabel("평균 구간 폭", fontsize=10, color=INK)
    ax1.set_ylabel("실제 포함확률", fontsize=10, color=INK)
    ax1.set_title(r"쉬운 문제: $\chi^2_3$ 의 평균, $n = 100$", fontsize=11.5,
                  color=INK)
    tidy(ax1, left=True)

    for name, cov, wid, col in hard:
        ax2.plot([wid], [cov], "o", color=col, ms=10)
    ax2.text(1.59, 0.677 - 0.012, "백분위수", color=BLUE, fontsize=10,
             ha="center", va="top")
    ax2.text(1.88, 0.736 + 0.012, "BCa", color=GREEN, fontsize=10,
             ha="center", va="bottom")
    ax2.text(14.15 * 0.88, 0.878, r"붓스트랩-$t$" + "\n폭 14.15", color=PURPLE,
             fontsize=10, ha="right", va="center")
    ax2.axhline(0.95, color=RED, lw=1.4, ls="--")
    ax2.text(1.15, 0.955, "명목 0.95", color=RED, fontsize=9.5)
    ax2.set_xscale("log")
    ax2.set_xlim(1.1, 30)
    ax2.set_ylim(0.62, 0.99)
    ax2.set_xticks([1.5, 2, 3, 5, 10, 20])
    ax2.set_xticklabels(["1.5", "2", "3", "5", "10", "20"])
    ax2.minorticks_off()
    ax2.set_xlabel("평균 구간 폭 (로그 눈금)", fontsize=10, color=INK)
    ax2.set_ylabel("실제 포함확률", fontsize=10, color=INK)
    ax2.set_title(r"어려운 문제: $\mathrm{Exp}(1)$ 의 분산, $n = 15$",
                  fontsize=11.5, color=INK)
    ax2.text(2.6, 0.66, "참값이 1 인데 폭이 14 라면\n아무것도 말해 주지 않는다",
             fontsize=9.5, color=INK)
    tidy(ax2, left=True)

    save(fig, "coverage_vs_width.png")


# ===========================================================================
# 그림 7. 신뢰수준의 폭과 장기 포함확률 (bootstrap_ci_visualization.md)
# ===========================================================================
def fig_levels_coverage():
    pop = np.random.default_rng(3).exponential(50_000, 5000) + 20_000
    mu = pop.mean() / 1000.0
    rng = np.random.default_rng(303)

    s = rng.choice(pop, 20, replace=False) / 1000.0
    bm = s[rng.integers(0, 20, (40000, 20))].mean(axis=1)
    lv = {}
    for lab, q in (("90%", [5, 95]), ("95%", [2.5, 97.5]), ("99%", [0.5, 99.5])):
        lv[lab] = np.percentile(bm, q)
    print("[levels] 모평균=%.1f  표본평균=%.1f" % (mu, s.mean()))
    for k, (lo, hi) in lv.items():
        print("   %s [%.1f, %.1f]  폭 %.1f" % (k, lo, hi, hi - lo))
    print("   99/90 폭의 비 = %.3f"
          % ((lv["99%"][1] - lv["99%"][0]) / (lv["90%"][1] - lv["90%"][0])))

    N, B = 2000, 1000
    los = np.empty(N)
    his = np.empty(N)
    for i in range(N):
        t = rng.choice(pop, 20, replace=False) / 1000.0
        b = t[rng.integers(0, 20, (B, 20))].mean(axis=1)
        los[i], his[i] = np.percentile(b, [2.5, 97.5])
    cov = float(((los <= mu) & (mu <= his)).mean())
    print("   95%% 백분위수 구간의 실제 포함확률 = %.3f" % cov)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                                   gridspec_kw=dict(wspace=0.20,
                                                    width_ratios=[1.0, 1.25]))

    ax1.hist(bm, bins=90, density=True, color=BLUE_F, edgecolor=BLUE, lw=0.7)
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(-0.44 * ymax, ymax * 1.22)
    ax1.plot([s.mean()] * 2, [0, ymax * 1.02], color=INK, lw=1.4, ls="--")
    ax1.plot([mu] * 2, [0, ymax * 1.02], color=PURPLE, lw=2.0)
    ax1.text(mu - 0.8, ymax * 1.10, "참 평균 %.1f" % mu, color=PURPLE,
             fontsize=10, va="center", ha="right")
    ax1.text(s.mean() + 0.8, ymax * 1.10, "표본평균 %.1f" % s.mean(),
             color=INK, fontsize=10, va="center")
    for (lab, (lo, hi)), y, col in zip(lv.items(),
                                       (-0.10, -0.22, -0.34),
                                       (GREEN, BLUE, ORANGE)):
        ax1.plot([lo, hi], [y * ymax] * 2, color=col, lw=6,
                 solid_capstyle="butt")
        ax1.plot([lo, hi], [y * ymax] * 2, "|", color=col, ms=12, mew=2)
        ax1.text(lo - 1.2, y * ymax, lab, ha="right", va="center",
                 fontsize=10, color=col)
        ax1.text(hi + 1.2, y * ymax, "폭 %.1f" % (hi - lo), ha="left",
                 va="center", fontsize=9, color=col)
    ax1.set_xlim(38, 124)
    ax1.set_xlabel("붓스트랩 표본평균 (천 단위)", fontsize=10, color=INK)
    ax1.set_title("표본 하나: 신뢰수준을 올리면 폭이 커진다", fontsize=11.5,
                  color=INK)
    tidy(ax1)

    K = 70
    hit = (los[:K] <= mu) & (mu <= his[:K])
    for i in range(K):
        col = BLUE if hit[i] else RED
        ax2.plot([los[i], his[i]], [i, i], color=col, lw=1.8,
                 alpha=0.9 if hit[i] else 1.0)
    ax2.plot([mu, mu], [-1.5, K + 0.5], color=PURPLE, lw=2.0, zorder=5)
    ax2.set_ylim(-6, K + 12)
    ax2.set_xlim(25, 148)
    ax2.text(mu + 1.5, -3.0, "참 평균 %.1f" % mu, color=PURPLE,
             fontsize=10.5, va="center")
    ax2.text(27, K + 8,
             "처음 70개 구간 중 %d 개가 빗나갔다" % int((~hit).sum()),
             color=RED, fontsize=10, va="center")
    ax2.text(27, K + 3.5,
             "2000번 반복한 실제 포함확률 %.3f  (약속은 0.95)" % cov,
             color=INK, fontsize=10.5, va="center")
    ax2.set_xlabel("95% 백분위수 붓스트랩 구간 (천 단위)", fontsize=10, color=INK)
    ax2.set_title("같은 절차를 되풀이하면", fontsize=11.5, color=INK)
    tidy(ax2)

    save(fig, "levels_and_coverage.png")


if __name__ == "__main__":
    fig_mirror()
    fig_invariance()
    fig_bca_dial()
    fig_studentized()
    fig_median_atoms()
    fig_coverage_width()
    fig_levels_coverage()
