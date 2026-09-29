r"""17장 순열검정 절(permutation/) 여섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch17/permutation/img/label_shuffle.png        라벨을 섞어 만드는 정확 귀무분포
  ch17/permutation/img/binary_permutation.png   이진 자료의 순열분포는 초기하분포다
  ch17/permutation/img/sign_flip_exact.png      부호 뒤집기의 정확분포
  ch17/permutation/img/statistic_picks.png      통계량이 대립가설을 정한다
  ch17/permutation/img/group_variance_test.png  집단평균의 분산을 통계량으로
  ch17/permutation/img/pairing_gains_power.png  짝을 살리면 얻는 것

실행:  python3 scripts/make_ch17_permutation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import itertools
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

OUT = "docs/ch17/permutation/img/"
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
# 그림 1. 라벨을 섞어 만드는 정확 귀무분포 (foundations.md)
# ===========================================================================
def fig_label_shuffle():
    times = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
                      173, 181, 182, 170, 169, 177, 168, 183, 169, 164])
    lab = np.array([0] * 10 + [1] * 10)          # 0 = A, 1 = B
    obs = times[lab == 1].mean() - times[lab == 0].mean()

    combs = np.array(list(itertools.combinations(range(20), 10)))
    s = times[combs].sum(1)
    tot = times.sum()
    d = s / 10 - (tot - s) / 10                  # 뽑힌 쪽이 B 일 때의 차이
    ge = (np.abs(d) >= abs(obs) - 1e-9).mean()
    gt = (np.abs(d) > abs(obs) + 1e-9).mean()
    print("[label] obs=%.2f  정확 p(>=)=%.4f  p(>)=%.4f  배열 수=%d"
          % (obs, ge, gt, len(d)))

    rng = np.random.default_rng(4)
    shuf = rng.permutation(lab)
    d1 = times[shuf == 1].mean() - times[shuf == 0].mean()
    print("   한 번 섞었을 때의 차이 %.2f" % d1)

    fig = plt.figure(figsize=(12.4, 6.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.15], hspace=0.42)

    ax = fig.add_subplot(gs[0])
    for r, (lv, name) in enumerate(((lab, "원자료"), (shuf, "라벨을 한 번 섞으면"))):
        yy = -r * 1.0
        for j, (t, g) in enumerate(zip(times, lv)):
            col = BLUE if g == 0 else ORANGE
            ax.plot([t], [yy], "o", color=col, ms=9, alpha=0.9)
        ma = times[lv == 0].mean()
        mb = times[lv == 1].mean()
        ax.plot([ma, ma], [yy - 0.28, yy + 0.28], color=BLUE, lw=2.4)
        ax.plot([mb, mb], [yy - 0.28, yy + 0.28], color=ORANGE, lw=2.4)
        ax.text(139, yy, name, ha="right", va="center", fontsize=10.5,
                color=INK)
        ax.text(192, yy, r"$\bar x_B - \bar x_A = %.2f$" % (mb - ma),
                ha="left", va="center", fontsize=10, color=PURPLE)
    ax.text(139, 0.62, "파랑 = 페이지 A,  주황 = 페이지 B", ha="right",
            fontsize=9.5, color=INK)
    ax.text(192, 0.62, "값은 그대로, 색만 다시 칠한다", ha="left",
            fontsize=9.5, color=INK)
    ax.set_xlim(120, 215)
    ax.set_ylim(-1.7, 0.95)
    ax.set_yticks([])
    ax.set_xticks([140, 150, 160, 170, 180, 190])
    ax.set_xlabel("세션 시간 (초)", fontsize=10, color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.set_title("(a) 귀무가설이 참이면 라벨은 임의적이다", fontsize=11.5,
                 color=INK)

    ax = fig.add_subplot(gs[1])
    vals, cnt = np.unique(np.round(d, 6), return_counts=True)
    pr = cnt / cnt.sum()
    inner = np.abs(vals) < abs(obs) - 1e-9
    ax.vlines(vals[inner], 0, pr[inner], color=GREEN, lw=1.6)
    ax.vlines(vals[~inner], 0, pr[~inner], color=RED, lw=1.6)
    top = pr.max()
    ax.set_ylim(0, top * 1.40)
    ax.plot([obs, obs], [0, top * 1.14], color=PURPLE, lw=2.0)
    ax.text(obs + 0.4, top * 1.22, r"관측 $%.2f$" % obs, color=PURPLE,
            fontsize=10.5, va="center")
    ax.text(-19.5, top * 1.30,
            r"$\binom{20}{10} = 184{,}756$ 가지 배열의 정확분포", color=GREEN,
            fontsize=10.5, va="center")
    ax.text(-19.5, top * 1.14,
            "붉은 막대의 합 = $p = %.4f$" % ge, color=RED, fontsize=10.5,
            va="center")
    ax.annotate("관측값과 정확히 같은 배열까지 세면 %.4f,\n"
                "빼면 %.4f — 여기서만 %.3f 차이가 난다" % (ge, gt, ge - gt),
                xy=(abs(obs), pr[np.argmin(np.abs(vals - abs(obs)))] * 1.05),
                xytext=(11.5, top * 0.72), fontsize=9.5, color=INK,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xlim(-20, 20)
    ax.set_title("(b) 그렇게 만든 귀무분포에서 $p$ 값을 센다", fontsize=11.5,
                 color=INK)
    ax.set_xlabel(r"$\bar x_B - \bar x_A$", fontsize=10, color=INK)
    tidy(ax)

    save(fig, "label_shuffle.png")


# ===========================================================================
# 그림 2. 이진 자료의 순열분포는 초기하분포다 (ab_testing_permutation.md)
# ===========================================================================
def fig_binary():
    n0, n1 = 23739, 22588
    c0, c1 = 200, 182
    N, K = n0 + n1, c0 + c1
    obs = c1 / n1 - c0 / n0

    rng = np.random.default_rng(0)
    B = 30000
    k = rng.hypergeometric(K, N - K, n1, B)      # 처치군의 전환 수
    d = k / n1 - (K - k) / n0
    p_perm = ((np.abs(d) >= abs(obs) - 1e-15).sum() + 1) / (B + 1)

    ks = np.arange(K + 1)
    pmf = stats.hypergeom.pmf(ks, N, K, n1)
    ds = ks / n1 - (K - ks) / n0
    p_exact = pmf[np.abs(ds) >= abs(obs) - 1e-15].sum()
    fisher = stats.fisher_exact([[c1, n1 - c1], [c0, n0 - c0]])[1]
    chi2 = stats.chi2_contingency([[c1, n1 - c1], [c0, n0 - c0]])[1]
    print("[binary] obs=%.6f  순열 p=%.4f  초기하 정확 p=%.4f"
          % (obs, p_perm, p_exact))
    print("   Fisher=%.4f  카이제곱=%.4f" % (fisher, chi2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw=dict(wspace=0.22,
                                                    width_ratios=[1.35, 1.0]))

    sc = 1000.0
    edges = ((np.arange(K + 2) - 0.5) / n1
             - (K - np.arange(K + 2) + 0.5) / n0) * sc
    step = edges[1] - edges[0]
    sel = (edges >= -1.65) & (edges <= 1.65)
    ax1.hist(d * sc, bins=edges[sel], density=True, color=ORANGE_F,
             edgecolor=ORANGE, lw=0.8, label="라벨을 3만 번 섞은 결과")
    m = np.abs(ds) < 1.7e-3
    ax1.plot(ds[m] * sc, pmf[m] / step, color=GREEN, lw=2.2,
             label="초기하분포 (재표집 없이 계산)")
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(0, ymax * 1.42)
    for v, ls in ((obs, "-"), (-obs, ":")):
        ax1.plot([v * sc, v * sc], [0, ymax * 1.10], color=PURPLE, lw=2.0,
                 ls=ls)
    ax1.text(obs * sc - 0.04, ymax * 1.20, r"관측 $-0.368$", color=PURPLE,
             fontsize=10.5, ha="right", va="center")
    ax1.text(-obs * sc + 0.04, ymax * 1.20, "반대쪽 꼬리", color=PURPLE,
             fontsize=9.5, ha="left", va="center")
    ax1.set_xlim(-1.65, 1.65)
    ax1.set_xlabel("전환율 차이 (1000분의 1 단위)", fontsize=10, color=INK)
    ax1.set_title(r"$n = 46{,}327$ 의 이진 A/B 검정", fontsize=11.5, color=INK)
    ax1.legend(fontsize=9.5, frameon=False, loc="upper left")
    tidy(ax1)

    rows = [("순열검정 (3만 번)", p_perm, ORANGE),
            ("초기하 정확", p_exact, GREEN),
            ("Fisher 정확검정", fisher, MUTED),
            ("카이제곱 (연속성 보정)", chi2, BLUE)]
    for i, (lab, v, col) in enumerate(rows):
        y = len(rows) - i
        ax2.plot([v], [y], "o", color=col, ms=11)
        ax2.text(v, y + 0.30, "%.4f" % v, ha="center", va="bottom",
                 fontsize=9.5, color=col)
        ax2.text(v - 0.025, y, lab, ha="right", va="center", fontsize=10,
                 color=col)
    ax2.axvline(0.05, color=RED, lw=1.2, ls="--")
    ax2.text(0.055, 0.55, r"$\alpha = 0.05$", color=RED, fontsize=9.5)
    ax2.set_xlim(0.0, 0.80)
    ax2.set_ylim(0.2, 4.9)
    ax2.set_xlabel("$p$ 값", fontsize=10, color=INK)
    ax2.set_title("네 방법이 같은 답을 준다", fontsize=11.5, color=INK)
    tidy(ax2)

    save(fig, "binary_permutation.png")


# ===========================================================================
# 그림 3. 부호 뒤집기의 정확분포 (paired.md)
# ===========================================================================
def fig_sign_flip():
    d = np.array([-8, -3, 2, -12, -5, -1, 4, -7, -6, -2, -9, -4])
    n = len(d)
    obs = d.mean()
    S = np.array(list(itertools.product([1, -1], repeat=n)))
    m = (S * d).mean(1)
    cnt = (np.abs(m) >= abs(obs) - 1e-12).sum()
    p_exact = cnt / len(m)
    sd = d.std(ddof=1)
    tp = stats.ttest_1samp(d, 0).pvalue
    bound = np.abs(d).mean()
    print("[sign] d_bar=%.2f s=%.3f  정확 p=%d/%d=%.5f  t 검정 p=%.5f"
          % (obs, sd, cnt, len(m), p_exact, tp))
    print("   순열분포의 지지집합은 [%.2f, %.2f] 로 유계" % (-bound, bound))

    fig = plt.figure(figsize=(12.4, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.46)

    ax = fig.add_subplot(gs[0])
    rng = np.random.default_rng(1)
    rows = [(np.ones(n, int), "관측된 부호")]
    for k in range(2):
        rows.append((rng.choice([-1, 1], n), "뒤집은 예 %d" % (k + 1)))
    for r, (s, name) in enumerate(rows):
        yy = -r * 1.0
        v = s * d
        for j, val in enumerate(v):
            col = BLUE if val < 0 else ORANGE
            ax.plot([j], [yy], "o", color=col, ms=8)
            ax.plot([j, j], [yy, yy + 0.0], color=col)
        ax.text(-0.9, yy, name, ha="right", va="center", fontsize=10,
                color=INK)
        ax.text(12.4, yy, r"$\bar d = %.2f$" % v.mean(), ha="left",
                va="center", fontsize=10, color=PURPLE)
    ax.text(-0.9, 0.7, "파랑 = 음수,  주황 = 양수", ha="right", fontsize=9.5,
            color=INK)
    ax.text(12.4, 0.7, r"$|d_i|$ 는 언제나 그대로다", ha="left", fontsize=9.5,
            color=INK)
    ax.set_xlim(-5.5, 17)
    ax.set_ylim(-2.7, 1.0)
    ax.axis("off")
    ax.set_title(r"(a) 부호만 뒤집는다 — $2^{12} = 4096$ 가지", fontsize=11.5,
                 color=INK)

    ax = fig.add_subplot(gs[1])
    vals, c2 = np.unique(np.round(m, 8), return_counts=True)
    pr = c2 / c2.sum()
    inner = np.abs(vals) < abs(obs) - 1e-12
    w = 0.16
    ax.bar(vals[inner], pr[inner], width=w, color=GREEN)
    ax.bar(vals[~inner], pr[~inner], width=w, color=RED)
    top = pr.max()
    ax.set_ylim(0, top * 1.46)
    ax.plot([obs, obs], [0, top * 1.16], color=PURPLE, lw=2.0)
    ax.text(obs - 0.25, top * 0.80, r"관측 $\bar d = -4.25$", color=PURPLE,
            fontsize=10.5, ha="right", va="center")
    tt = np.linspace(-9, 9, 500)
    gap = float(np.median(np.diff(vals)))
    ax.plot(tt, stats.t.pdf(tt / (sd / np.sqrt(n)), n - 1)
            / (sd / np.sqrt(n)) * gap, color=MUTED, lw=2.0, ls="--")
    ax.text(7.4, top * 0.62,
            r"$t_{11}$ 밀도 — 폭 $s_d/\sqrt{12} = %.2f$" % (sd / np.sqrt(n))
            + "\n꼬리가 무한히 뻗는다", color=MUTED, fontsize=9.5,
            ha="center", va="center")
    ax.text(-8.8, top * 1.04,
            r"순열분포의 폭은 $\sqrt{\sum d_i^2}/12 = %.2f$ 로 더 넓다"
            % (np.sqrt((d ** 2).sum()) / n), color=GREEN, fontsize=10.5,
            va="center")
    for b in (-bound, bound):
        ax.plot([b, b], [0, top * 0.70], color=INK, lw=1.2, ls=":")
    ax.text(bound, top * 0.76, r"$\pm\,\overline{|d|} = \pm 5.25$", color=INK,
            fontsize=9.5, ha="center")
    ax.text(-8.8, top * 1.34,
            "붉은 막대 %d 개의 합 = $p = %d/4096 = %.5f$" % (int(cnt), int(cnt),
                                                          p_exact),
            color=RED, fontsize=10.5, va="center")
    ax.text(-8.8, top * 1.19,
            r"대응 $t$ 검정은 $p = %.4f$ 로 더 작다" % tp, color=MUTED,
            fontsize=10.5, va="center")
    ax.set_xlim(-9, 9)
    ax.set_title(r"(b) 4096 가지 배정이 만드는 정확 귀무분포", fontsize=11.5,
                 color=INK)
    ax.set_xlabel(r"$\bar d^{*}$", fontsize=10, color=INK)
    tidy(ax)

    save(fig, "sign_flip_exact.png")


# ===========================================================================
# 그림 4. 통계량이 대립가설을 정한다 (correlation.md)
# ===========================================================================
def fig_statistic_picks():
    rng = np.random.default_rng(14)
    n = 30
    x = rng.normal(0, 1, n)
    y = x ** 2 + rng.normal(0, 0.5, n)
    a = np.abs(x - np.median(x))
    r = np.corrcoef(x, y)[0, 1]
    rs = stats.spearmanr(x, y).statistic
    q = np.corrcoef(a, y)[0, 1]
    print("[pick] r=%.3f  spearman=%.3f  corr(|x-med|, y)=%.3f" % (r, rs, q))

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3),
                             gridspec_kw=dict(wspace=0.26,
                                              width_ratios=[1, 1, 0.9]))

    axes[0].scatter(x, y, s=34, color=BLUE, alpha=0.8)
    axes[0].set_xlabel("$x$", fontsize=10, color=INK)
    axes[0].set_ylabel("$y$", fontsize=10, color=INK)
    axes[0].set_title(r"(a) $y = x^2 + \varepsilon$ — Pearson $r = %.3f$" % r,
                      fontsize=11, color=INK)
    tidy(axes[0], left=True)

    axes[1].scatter(a, y, s=34, color=GREEN, alpha=0.8)
    cf = np.polyfit(a, y, 1)
    xs = np.linspace(a.min(), a.max(), 10)
    axes[1].plot(xs, np.polyval(cf, xs), color=GREEN, lw=1.6, ls="--")
    axes[1].set_xlabel(r"$|x - \mathrm{med}(x)|$", fontsize=10, color=INK)
    axes[1].set_ylabel("$y$", fontsize=10, color=INK)
    axes[1].set_title(r"(b) 축을 바꾸면 상관 $%.3f$" % q, fontsize=11,
                      color=INK)
    tidy(axes[1], left=True)

    ax = axes[2]
    labs = [r"Pearson $r$", r"Spearman $\rho$",
            r"$\mathrm{corr}(|x-\mathrm{med}|,\,y)$"]
    vals = [0.331, 0.117, 0.999]
    cols = [BLUE, MUTED, GREEN]
    ax.barh(np.arange(3), vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, "%.3f" % v, va="center", fontsize=10,
                color=cols[i])
    ax.axvline(0.05, color=RED, lw=1.2, ls="--")
    ax.text(0.07, -0.62, r"크기 $\alpha = 0.05$ (셋 모두 정확)", color=RED,
            fontsize=9)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labs, fontsize=9.5)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.9, 2.6)
    ax.set_xlabel(r"검정력 ($n = 30$)", fontsize=10, color=INK)
    ax.set_title("(c) 같은 순열, 다른 통계량", fontsize=11, color=INK)
    tidy(ax, left=True)

    save(fig, "statistic_picks.png")


# ===========================================================================
# 그림 5. 집단평균의 분산을 통계량으로 (permutation_tests.md)
# ===========================================================================
def fig_group_variance():
    rng_data = np.random.default_rng(11)
    _ = rng_data.normal(120, 30, 36)
    _ = rng_data.normal(135, 30, 40)
    groups = [rng_data.normal(mu, 25, 30) for mu in [160, 170, 155, 180]]
    means = np.array([g.mean() for g in groups])
    obs = np.var(means)
    pooled = np.concatenate(groups)
    k, m = 4, 30
    N = k * m

    rng = np.random.default_rng(3)
    B = 40000
    P = np.array([rng.permutation(pooled) for _ in range(B)])
    gm = P.reshape(B, k, m).mean(2)
    pv = gm.var(1)
    p = ((pv >= obs).sum() + 1) / (B + 1)

    ssb = m * ((gm - pooled.mean()) ** 2).sum(1)
    sst = ((pooled - pooled.mean()) ** 2).sum()
    F = (ssb / (k - 1)) / ((sst - ssb) / (N - k))
    F_obs = (m * ((means - pooled.mean()) ** 2).sum() / (k - 1)) / (
        (sst - m * ((means - pooled.mean()) ** 2).sum()) / (N - k))
    p_F = ((F >= F_obs).sum() + 1) / (B + 1)
    print("[group] 집단평균 %s  Var=%.3f  p=%.4f" % (np.round(means, 2), obs, p))
    print("   같은 순열의 F 로 세면 p=%.4f, 고전 ANOVA p=%.4f"
          % (p_F, stats.f_oneway(*groups).pvalue))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                             gridspec_kw=dict(wspace=0.26,
                                              width_ratios=[0.95, 1.2, 0.9]))

    ax = axes[0]
    jr = np.random.default_rng(2)
    for i, g in enumerate(groups):
        ax.plot(i + jr.uniform(-0.16, 0.16, m), g, "o", color=MUTED, ms=4,
                alpha=0.7)
        ax.plot([i - 0.3, i + 0.3], [g.mean()] * 2, color=BLUE, lw=3)
        ax.text(i, 232, "%.1f" % g.mean(), ha="center", fontsize=9.5,
                color=BLUE)
    ax.axhline(pooled.mean(), color=PURPLE, lw=1.4, ls="--")
    ax.text(-0.46, pooled.mean() + 4, "전체 평균", color=PURPLE, fontsize=9,
            ha="left")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_ylim(85, 245)
    ax.set_xlabel("처치군", fontsize=10, color=INK)
    ax.set_title("(a) 네 집단, 각 $n = 30$", fontsize=11, color=INK)
    tidy(ax, left=True)

    ax = axes[1]
    h, c = step_hist(ax, pv, np.linspace(0, 120, 90), GREEN, GREEN_F)
    ax.fill_between(c, h, where=c >= obs, step="mid", color=RED, alpha=0.75)
    ymax = h.max()
    ax.set_ylim(0, ymax * 1.34)
    ax.plot([obs, obs], [0, ymax * 1.08], color=PURPLE, lw=2.0)
    ax.text(obs + 2, ymax * 1.18, r"관측 $T = %.1f$" % obs, color=PURPLE,
            fontsize=10.5, va="center")
    ax.text(118, ymax * 0.72,
            "오른쪽 꼬리만 센다\n$p = %.4f$" % p, color=RED, fontsize=10.5,
            ha="right", va="top")
    ax.set_xlim(0, 120)
    ax.set_xlabel(r"$T = \mathrm{Var}(\bar x_1, \ldots, \bar x_4)$",
                  fontsize=10, color=INK)
    ax.set_title("(b) 단측 귀무분포", fontsize=11, color=INK)
    tidy(ax)

    ax = axes[2]
    sel = np.random.default_rng(0).choice(B, 3000, replace=False)
    ax.plot(pv[sel], F[sel], ".", color=MUTED, ms=3, alpha=0.6)
    ax.plot([obs], [F_obs], "o", color=PURPLE, ms=9)
    ax.text(obs - 4, F_obs + 0.6, "관측", color=PURPLE, fontsize=9.5,
            ha="right")
    ax.set_xlim(0, 120)
    ax.set_xlabel(r"$T = \mathrm{Var}(\bar x_j)$", fontsize=10, color=INK)
    ax.set_ylabel(r"분산분석 $F$", fontsize=10, color=INK)
    ax.set_title("(c) 둘은 같은 순서를 준다", fontsize=11, color=INK)
    tidy(ax, left=True)

    save(fig, "group_variance_test.png")


# ===========================================================================
# 그림 6. 짝을 살리면 얻는 것 (resampling_shoe_sales.md)
# ===========================================================================
def fig_pairing():
    before = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
    after = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27])
    d = after - before
    obs = d.mean()
    r = np.corrcoef(before, after)[0, 1]

    # 비대응 순열
    rng = np.random.default_rng(42)
    comb = np.concatenate([before, after])
    B = 200000
    P = np.array([rng.permutation(comb) for _ in range(B)])
    du = P[:, 12:].mean(1) - P[:, :12].mean(1)
    p_un1 = ((du >= obs - 1e-12).sum() + 1) / (B + 1)
    p_un2 = ((np.abs(du) >= obs - 1e-12).sum() + 1) / (B + 1)

    # 대응(부호 뒤집기) 정확분포
    S = np.array(list(itertools.product([1, -1], repeat=12)))
    dp = (S * d).mean(1)
    p_pa1 = (dp >= obs - 1e-12).mean()
    p_pa2 = (np.abs(dp) >= obs - 1e-12).mean()
    print("[pair] d_bar=%.2f  r=%.3f" % (obs, r))
    print("   비대응 순열 단측 %.4f / 양측 %.4f" % (p_un1, p_un2))
    print("   대응 정확   단측 %.4f / 양측 %.4f" % (p_pa1, p_pa2))
    print("   sd: 비대응 %.3f / 대응 %.3f"
          % (du.std(), dp.std()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                   gridspec_kw=dict(wspace=0.24,
                                                    width_ratios=[0.95, 1.25]))

    for i, (b0, a0) in enumerate(zip(before, after)):
        col = GREEN if a0 >= b0 else RED
        ax1.plot([0, 1], [b0, a0], "-o", color=col, lw=1.4, ms=5, alpha=0.85)
    ax1.plot([0, 1], [before.mean(), after.mean()], "-", color=PURPLE, lw=3.4)
    ax1.text(-0.06, before.mean(), "%.2f" % before.mean(), ha="right",
             va="center", fontsize=10, color=PURPLE)
    ax1.text(1.06, after.mean(), "%.2f" % after.mean(), ha="left",
             va="center", fontsize=10, color=PURPLE)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["전", "후"], fontsize=11)
    ax1.set_xlim(-0.42, 1.42)
    ax1.set_ylabel("주간 판매량 (켤레)", fontsize=10, color=INK)
    ax1.set_ylim(13.5, 38.5)
    ax1.text(0.5, 37.0, r"주별 상관 $r = %.3f$" % r, ha="center", fontsize=10.5,
             color=INK)
    ax1.text(0.5, 14.6, "초록 = 늘어난 주,  빨강 = 줄어든 주", ha="center",
             fontsize=9.5, color=INK)
    ax1.set_title("(a) 같은 주를 두 번 잰 자료", fontsize=11.5, color=INK)
    tidy(ax1, left=True)

    bins = np.linspace(-7, 7, 100)
    step_hist(ax2, du, bins, MUTED, "#EEF1F3")
    vals, c2 = np.unique(np.round(dp, 8), return_counts=True)
    ax2.bar(vals, c2 / c2.sum() / (vals[1] - vals[0]), width=0.13,
            color=GREEN)
    ymax = ax2.get_ylim()[1]
    ax2.set_ylim(0, ymax * 1.36)
    ax2.plot([obs, obs], [0, ymax * 1.08], color=PURPLE, lw=2.0)
    ax2.text(obs + 0.2, ymax * 1.18, r"관측 $+2.25$", color=PURPLE,
             fontsize=10.5, va="center")
    ax2.text(-6.8, ymax * 1.26,
             "비대응 순열 (표준편차 %.2f) — 단측 $p = %.3f$"
             % (du.std(), p_un1), color=MUTED, fontsize=10, va="center")
    ax2.text(-6.8, ymax * 1.12,
             "대응 부호뒤집기 (표준편차 %.2f) — 단측 $p = %.4f$"
             % (dp.std(), p_pa1), color=GREEN, fontsize=10, va="center")
    ax2.set_xlim(-7, 7)
    ax2.set_xlabel("평균차 (켤레)", fontsize=10, color=INK)
    ax2.set_title("(b) 짝을 살리면 귀무분포가 좁아진다", fontsize=11.5,
                  color=INK)
    tidy(ax2)

    save(fig, "pairing_gains_power.png")


if __name__ == "__main__":
    fig_label_shuffle()
    fig_binary()
    fig_sign_flip()
    fig_statistic_picks()
    fig_group_variance()
    fig_pairing()
