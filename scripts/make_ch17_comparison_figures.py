r"""17장 비교 절(comparison/) 여섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch17/comparison/img/two_questions.png        무엇을 재배열하고 무엇을 근사하는가
  ch17/comparison/img/small_sample_gap.png     n = 5 에서 벌어지는 틈
  ch17/comparison/img/monte_carlo_convergence.png  B 를 얼마나 키워야 하는가
  ch17/comparison/img/cv_flat_curve.png        교차검증 곡선은 평평하다
  ch17/comparison/img/dependence_breaks_bootstrap.png  종속자료에서의 실패
  ch17/comparison/img/statistic_choice.png     순열검정은 통계량 선택에 달렸다

실행:  python3 scripts/make_ch17_comparison_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch17/comparison/img/"
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
# 그림 1. 무엇을 재배열하고 무엇을 근사하는가 (bootstrap_vs_permutation.md)
# ===========================================================================
def fig_two_questions():
    rng = np.random.default_rng(11)
    m = n = 12
    x = np.round(rng.normal(52, 6, m)).astype(int)
    y = np.round(rng.normal(46, 6, n)).astype(int)
    obs = x.mean() - y.mean()
    pooled = np.concatenate([x, y])

    B = 40000
    perm = np.array([rng.permutation(pooled) for _ in range(B)])
    dperm = perm[:, :m].mean(1) - perm[:, m:].mean(1)
    p = ((np.abs(dperm) >= abs(obs)).sum() + 1) / (B + 1)

    bx = x[rng.integers(0, m, (B, m))].mean(1)
    by = y[rng.integers(0, n, (B, n))].mean(1)
    dboot = bx - by
    ci = np.percentile(dboot, [2.5, 97.5])
    print("[two] obs=%.3f  순열 p=%.4f  붓스트랩 95%% CI=[%.3f, %.3f]"
          % (obs, p, *ci))

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.30)

    # --- (a) 재표본의 생김새 ---
    ax = fig.add_subplot(gs[0])
    rows = [("원자료", pooled, np.arange(24), None)]
    pr = np.random.default_rng(5)
    for k in range(1):
        pi = pr.permutation(24)
        rows.append(("순열 재표본", pooled[pi], pi, "perm"))
    for k in range(1):
        bi = np.concatenate([pr.integers(0, m, m), m + pr.integers(0, n, n)])
        rows.append(("붓스트랩 재표본", pooled[bi], bi, "boot"))

    for r, (lab, vals, idx, kind) in enumerate(rows):
        yy = -r * 1.0
        for j, v in enumerate(vals):
            col = BLUE if j < m else ORANGE
            colf = BLUE_F if j < m else ORANGE_F
            ax.add_patch(plt.Rectangle((j * 1.0, yy - 0.34), 0.9, 0.68,
                                       fc=colf, ec=col, lw=1.2))
            ax.text(j * 1.0 + 0.45, yy, str(v), ha="center", va="center",
                    fontsize=9, color=INK)
        ax.text(-0.4, yy, lab, ha="right", va="center", fontsize=10.5,
                color=INK)
        if kind == "boot":
            cnt = np.bincount(idx, minlength=24)
            miss = np.where(cnt == 0)[0]
            ax.text(24.3, yy, "%d 개 관측이 빠지고\n어떤 값은 여러 번 나온다"
                    % len(miss), fontsize=9, color=ORANGE, va="center")
        elif kind == "perm":
            ax.text(24.3, yy, "값은 그대로, 라벨만 섞인다", fontsize=9,
                    color=BLUE, va="center")
        else:
            ax.text(24.3, yy, "파랑 = 집단 X,  주황 = 집단 Y", fontsize=9,
                    color=INK, va="center")
    ax.set_xlim(-4.2, 33)
    ax.set_ylim(-2.6, 0.7)
    ax.axis("off")
    ax.set_title("(a) 재표본이 어떻게 생겼는가", fontsize=11.5, color=INK)

    # --- (b) 두 분포 ---
    ax = fig.add_subplot(gs[1])
    bins = np.linspace(-11, 17, 130)
    step_hist(ax, dperm, bins, GREEN, GREEN_F)
    step_hist(ax, dboot, bins, BLUE, BLUE_F)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(-0.30 * ymax, ymax * 1.30)
    ax.plot([obs, obs], [0, ymax * 1.06], color=PURPLE, lw=2.0)
    ax.text(obs, ymax * 1.14, r"관측 $\bar x - \bar y = %.2f$" % obs,
            color=PURPLE, fontsize=10.5, ha="center", va="center")
    ax.plot([0, 0], [0, ymax * 0.98], color=MUTED, lw=1.2, ls=":")
    ax.text(-10.7, ymax * 1.14,
            "순열: 귀무분포 — $p = %.3f$" % p, color=GREEN, fontsize=10.5,
            va="center")
    ax.text(-10.7, ymax * 0.98,
            "붓스트랩: 표집분포 — 95%% 구간 [%.2f, %.2f]" % (ci[0], ci[1]),
            color=BLUE, fontsize=10.5, va="center")
    ax.plot(ci, [-0.16 * ymax] * 2, color=BLUE, lw=6, solid_capstyle="butt")
    ax.plot(ci, [-0.16 * ymax] * 2, "|", color=BLUE, ms=13, mew=2)
    ax.text(ci[1] + 0.4, -0.16 * ymax, "0 을 포함하지 않는다", fontsize=9,
            color=BLUE, va="center")
    ax.set_xlim(-11, 17)
    ax.set_title("(b) 같은 자료, 서로 다른 두 분포", fontsize=11.5, color=INK)
    ax.set_xlabel("평균차", fontsize=10, color=INK)
    tidy(ax)

    save(fig, "two_questions.png")


# ===========================================================================
# 그림 2. n = 5 에서 벌어지는 틈 (comparison.md)
# ===========================================================================
def fig_small_gap():
    a = np.array([8, 7, 9, 10, 6])
    b = np.array([5, 6, 4, 3, 7])
    z = np.concatenate([a, b])
    obs = a.mean() - b.mean()

    dperm = np.array([z[list(c)].mean() - np.delete(z, list(c)).mean()
                      for c in itertools.combinations(range(10), 5)])
    p_exact = (np.abs(dperm) >= abs(obs) - 1e-12).mean()

    rng = np.random.default_rng(55)
    B = 200000
    A = a[rng.integers(0, 5, (B, 5))]
    Bm = b[rng.integers(0, 5, (B, 5))]
    dboot = A.mean(1) - Bm.mean(1)
    ci = np.percentile(dboot, [2.5, 97.5])
    p_boot = (dboot <= 0).mean() * 2
    se = np.sqrt(a.var(ddof=1) / 5 + b.var(ddof=1) / 5)
    tw = stats.ttest_ind(a, b, equal_var=False)
    df = se ** 4 / ((a.var(ddof=1) / 5) ** 2 / 4 + (b.var(ddof=1) / 5) ** 2 / 4)
    tcrit = stats.t.ppf(0.975, df)
    welch = (obs - tcrit * se, obs + tcrit * se)
    print("[gap] obs=%.2f  정확 순열 p=%.4f  붓스트랩 꼬리확률=%.5f"
          % (obs, p_exact, p_boot))
    print("   붓스트랩 [%.2f, %.2f] / Welch [%.2f, %.2f] / 순열역변환 [0.50, 5.50]"
          % (*ci, *welch))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 6.2),
                                   gridspec_kw=dict(height_ratios=[1.35, 1.0],
                                                    hspace=0.34))

    vals, cnt = np.unique(dperm, return_counts=True)
    ax1.vlines(vals, 0, cnt / len(dperm), color=GREEN, lw=3)
    ax1.plot(vals, cnt / len(dperm), "o", color=GREEN, ms=4)
    hi = (cnt / len(dperm)).max()
    ax1.set_ylim(0, hi * 1.42)
    ax1.plot([obs, obs], [0, hi * 1.16], color=PURPLE, lw=2.0)
    ax1.text(obs + 0.12, hi * 1.24, r"관측 $\bar a - \bar b = 3.0$",
             color=PURPLE, fontsize=10.5, va="center")
    for v, c in zip(vals, cnt):
        if abs(v) >= abs(obs) - 1e-12:
            ax1.vlines(v, 0, c / len(dperm), color=RED, lw=3)
    ax1.text(-4.6, hi * 1.24,
             "정확 순열분포 — 252가지 배열을 모두 센다", color=GREEN,
             fontsize=10.5, va="center")
    ax1.text(-4.6, hi * 1.04,
             "붉은 막대의 합 = $p = %.4f$" % p_exact, color=RED,
             fontsize=10.5, va="center")
    ax1.set_xlim(-4.8, 4.8)
    ax1.set_xlabel("평균차", fontsize=10, color=INK)
    tidy(ax1)

    rows = [
        ("순열검정 역변환", (0.50, 5.50), GREEN, "폭 5.00"),
        (r"Welch $t$", welch, MUTED, "폭 %.2f" % (welch[1] - welch[0])),
        ("붓스트랩 백분위수", tuple(ci), BLUE, "폭 %.2f" % (ci[1] - ci[0])),
    ]
    for i, (lab, (lo, hi2), col, wtxt) in enumerate(rows):
        yy = i + 0.5
        ax2.plot([lo, hi2], [yy, yy], color=col, lw=6, solid_capstyle="butt")
        ax2.plot([lo, hi2], [yy, yy], "|", color=col, ms=13, mew=2)
        ax2.text(lo - 0.12, yy, "%.2f" % lo, ha="right", va="center",
                 fontsize=9, color=col)
        ax2.text(hi2 + 0.12, yy, "%.2f" % hi2, ha="left", va="center",
                 fontsize=9, color=col)
        ax2.text(-1.3, yy, lab, ha="left", va="center", fontsize=10, color=col)
        ax2.text(7.3, yy, wtxt, ha="right", va="center", fontsize=9.5,
                 color=col)
    ax2.plot([0, 0], [0.1, 3.4], color=PURPLE, lw=1.6, ls="--")
    ax2.text(-0.15, 3.25, "0", color=PURPLE, fontsize=10, ha="right",
             va="center")
    ax2.set_xlim(-1.4, 7.4)
    ax2.set_ylim(0, 3.7)
    ax2.set_xlabel(r"$\mu_A - \mu_B$ 의 95% 구간", fontsize=10, color=INK)
    tidy(ax2)

    save(fig, "small_sample_gap.png")


# ===========================================================================
# 그림 3. B 를 얼마나 키워야 하는가 (convergence.md)
# ===========================================================================
def fig_convergence():
    rng = np.random.default_rng(77)
    n = 40
    x = rng.normal(10, 2, n)
    print("[conv] x_bar=%.4f s=%.4f" % (x.mean(), x.std(ddof=1)))

    # 누적 추정값
    Bmax = 20000
    tracks_se, tracks_lo = [], []
    for k in range(4):
        r = np.random.default_rng(100 + k)
        b = x[r.integers(0, n, (Bmax, n))].mean(1)
        grid = np.unique(np.round(np.logspace(1.3, np.log10(Bmax), 120)
                                  ).astype(int))
        se = [b[:g].std(ddof=1) for g in grid]
        lo = [np.percentile(b[:g], 2.5) for g in grid]
        tracks_se.append((grid, se))
        tracks_lo.append((grid, lo))

    Bs = [200, 1000, 5000, 20000]
    sd_lo, sd_se = [], []
    for B in Bs:
        L, S = [], []
        for _ in range(300):
            b = x[rng.integers(0, n, (B, n))].mean(1)
            L.append(np.percentile(b, 2.5))
            S.append(b.std(ddof=1))
        sd_lo.append(np.std(L, ddof=1))
        sd_se.append(np.std(S, ddof=1))
    print("[conv] sd(하한)=%s" % np.round(sd_lo, 5))
    print("[conv] sd(SE)  =%s" % np.round(sd_se, 5))
    print("[conv] 비 =%s" % np.round(np.array(sd_lo) / np.array(sd_se), 2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                                   gridspec_kw=dict(wspace=0.24))

    for (g, se), (g2, lo) in zip(tracks_se, tracks_lo):
        ax1.plot(g, se, color=BLUE, lw=1.3, alpha=0.8)
        ax1.plot(g2, lo - (np.mean([np.mean(l[-10:]) for _, l in tracks_lo])
                           - np.mean(se[-10:])), color=ORANGE, lw=1.3,
                 alpha=0.0)
    ax1.set_xscale("log")
    ax1.set_xticks([20, 100, 1000, 10000])
    ax1.set_xticklabels(["20", "100", "1000", "10000"])
    ax1.minorticks_off()
    ax1.axhline(np.mean([s[-1] for _, s in tracks_se]), color=MUTED, lw=1.2,
                ls="--")
    ax1.set_ylim(0.20, 0.50)
    ax1.set_xlabel("재표집 횟수 $b$ (로그 눈금)", fontsize=10, color=INK)
    ax1.set_ylabel(r"누적 $\widehat{SE}_{boot}$", fontsize=10, color=INK)
    ax1.set_title("(a) 씨앗 네 개의 누적 추정값", fontsize=11.5, color=INK)
    ax1.text(22, 0.475, "$B$ 가 작을 때는 씨앗마다 답이 다르다", fontsize=9.5,
             color=INK)
    tidy(ax1, left=True)

    ax2.plot(Bs, sd_lo, "o-", color=ORANGE, lw=2, ms=7,
             label="95% 구간의 하한")
    ax2.plot(Bs, sd_se, "s-", color=BLUE, lw=2, ms=7,
             label=r"$\widehat{SE}_{boot}$")
    ref = sd_lo[0] * np.sqrt(Bs[0] / np.array(Bs, float))
    ax2.plot(Bs, ref, ls=":", color=MUTED, lw=1.6, label=r"$1/\sqrt{B}$ 기준선")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(Bs)
    ax2.set_xticklabels(["200", "1000", "5000", "20000"])
    ax2.set_yticks([0.002, 0.005, 0.01, 0.02, 0.05])
    ax2.set_yticklabels(["0.002", "0.005", "0.01", "0.02", "0.05"])
    ax2.minorticks_off()
    for bb, l, s in zip(Bs, sd_lo, sd_se):
        ax2.plot([bb, bb], [s, l], color=MUTED, lw=0.9, ls="-")
    ax2.text(2600, 0.0082, "세로 간격 약 %.1f 배"
             % np.mean(np.array(sd_lo) / np.array(sd_se)),
             fontsize=10, color=INK, ha="center")
    ax2.set_xlabel("재표집 횟수 $B$ (로그 눈금)", fontsize=10, color=INK)
    ax2.set_ylabel("몬테카를로 표준편차 (로그 눈금)", fontsize=10, color=INK)
    ax2.set_title("(b) 분위수는 표준오차보다 훨씬 느리게 안정된다",
                  fontsize=11.5, color=INK)
    ax2.legend(fontsize=9, frameon=False, loc="lower left")
    tidy(ax2, left=True)

    save(fig, "monte_carlo_convergence.png")


# ===========================================================================
# 그림 4. 교차검증 곡선은 평평하다 (cv_methods_comparison.md)
# ===========================================================================
def fig_cv_flat():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(-3, 3, n)
    y = np.sin(x) + 0.3 * x + rng.normal(0, 0.5, n)
    degrees = np.arange(1, 11)

    def mse_fit(xt, yt, xv, yv, d):
        c = np.polyfit(xt, yt, d)
        return np.mean((yv - np.polyval(c, xv)) ** 2)

    loo = []
    for d in degrees:
        e = [mse_fit(np.delete(x, i), np.delete(y, i), x[i:i + 1], y[i:i + 1], d)
             for i in range(n)]
        loo.append(np.mean(e))
    loo = np.array(loo)

    def kfold(k, seed=42):
        r = np.random.default_rng(seed)
        perm = r.permutation(n)
        folds = np.array_split(perm, k)
        out = []
        for d in degrees:
            m = [mse_fit(np.delete(x, f), np.delete(y, f), x[f], y[f], d)
                 for f in folds]
            out.append(np.mean(m))
        return np.array(out)

    # 본문 표의 sklearn KFold 결과(분할이 다르면 최솟값 위치가 흔들린다)
    k5 = np.array([0.4463, 0.4513, 0.2715, 0.2697, 0.2641,
                   0.2628, 0.2716, 0.2742, 0.2751, 0.2817])
    k10 = np.array([0.4418, 0.4506, 0.2664, 0.2673, 0.2627,
                    0.2625, 0.2686, 0.2761, 0.2761, 0.2816])
    _ = kfold

    r = np.random.default_rng(7)
    splits = []
    for _ in range(10):
        perm = r.permutation(n)
        tr, te = perm[:n // 2], perm[n // 2:]
        splits.append(np.array([mse_fit(x[tr], y[tr], x[te], y[te], d)
                                for d in degrees]))
    splits = np.array(splits)
    picks = degrees[splits.argmin(axis=1)]
    print("[cv] LOOCV 최적차수 %d, 5겹 %d, 10겹 %d"
          % (degrees[loo.argmin()], degrees[k5.argmin()], degrees[k10.argmin()]))
    print("[cv] LOOCV MSE d=3..10: %s" % np.round(loo[2:], 4))
    print("[cv] 검증집합 10회의 선택: %s  (표준편차 %.2f)"
          % (picks.tolist(), picks.std(ddof=1)))
    print("[cv] 검증집합 평균 MSE: %s" % np.round(splits.mean(0), 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                                   gridspec_kw=dict(wspace=0.22))

    ax1.axhspan(loo[2:].min(), loo[2:].max(), color=PURPLE, alpha=0.10,
                zorder=0)
    for v, col, mk, lab in ((loo, BLUE, "o", "LOOCV"),
                            (k5, GREEN, "s", "5-겹"),
                            (k10, ORANGE, "^", "10-겹"),
                            (splits.mean(0), MUTED, "D", "검증집합(10회 평균)")):
        ax1.plot(degrees, v, mk + "-", color=col, lw=1.8, ms=6, label=lab)
    ax1.plot([degrees[loo.argmin()]], [loo.min()], "o", color=PURPLE, ms=13,
             mfc="none", mew=2)
    ax1.text(6.2, loo.min() - 0.012,
             "네 방법 모두 $d = %d$" % degrees[loo.argmin()], fontsize=10,
             color=PURPLE)
    ax1.text(9.9, 0.352,
             "띠 안(LOOCV, $d = 3 \\sim 10$)의 최댓값과\n최솟값 차이가 %.1f%% 에 불과하다"
             % (100 * (loo[2:].max() / loo[2:].min() - 1)),
             fontsize=9.5, color=PURPLE, ha="right", va="top")
    ax1.set_xticks(degrees)
    ax1.set_ylim(0.24, 0.50)
    ax1.set_xlabel("다항식 차수 $d$", fontsize=10, color=INK)
    ax1.set_ylabel("교차검증 MSE", fontsize=10, color=INK)
    ax1.set_title("(a) 네 방법의 MSE 곡선", fontsize=11.5, color=INK)
    ax1.legend(fontsize=9, frameon=False, loc="upper right")
    tidy(ax1, left=True)

    for i, sp in enumerate(splits):
        ax2.plot(degrees[2:], sp[2:], color=MUTED, lw=1.1, alpha=0.8)
        ax2.plot([picks[i]], [sp.min()], "o", color=RED, ms=6)
    ax2.plot(degrees[2:], loo[2:], "-", color=BLUE, lw=2.6, label="LOOCV")
    ax2.set_xticks(degrees[2:])
    ax2.set_xlim(2.6, 10.4)
    ax2.set_ylim(0.238, 0.375)
    ax2.text(2.75, 0.370,
             "검증집합 10회가 고른 차수\n%s  (표준편차 %.2f)"
             % (", ".join(map(str, picks)), picks.std(ddof=1)),
             fontsize=9.5, color=RED, va="top")
    ax2.text(2.75, 0.3465, "파란 굵은 선 = LOOCV", fontsize=9.5, color=BLUE,
             ha="left")
    ax2.set_xlabel("다항식 차수 $d$", fontsize=10, color=INK)
    ax2.set_ylabel("검정 MSE", fontsize=10, color=INK)
    ax2.set_title(r"(b) 검증집합 방법 10회 ($d \geq 3$ 만 확대)", fontsize=11.5,
                  color=INK)
    tidy(ax2, left=True)

    save(fig, "cv_flat_curve.png")


# ===========================================================================
# 그림 5. 종속자료에서의 실패 (limitations.md)
# ===========================================================================
def fig_dependence():
    rng = np.random.default_rng(101)
    phi, n = 0.7, 200

    def ar1(nn):
        e = rng.normal(0, 1, nn + 200)
        v = np.zeros(nn + 200)
        for t in range(1, nn + 200):
            v[t] = phi * v[t - 1] + e[t]
        return v[200:]

    ser = ar1(n)
    res = ser[rng.integers(0, n, n)]

    def acf(v, K=20):
        v = v - v.mean()
        d = (v * v).sum()
        return np.array([(v[:n - k] * v[k:]).sum() / d for k in range(K + 1)])

    sd_marg = np.sqrt(1 / (1 - phi ** 2))
    se_true = np.sqrt((1 + phi) / (1 - phi)) * sd_marg / np.sqrt(n)
    se_naive = sd_marg / np.sqrt(n)
    print("[dep] 참 SE=%.4f  독립 가정 SE=%.4f  비=%.2f"
          % (se_true, se_naive, se_naive / se_true))

    fig = plt.figure(figsize=(12.8, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.95, 1.05],
                          hspace=0.55, wspace=0.28)

    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0], sharey=axa)
    axa.plot(ser, color=BLUE, lw=1.0)
    axa.set_title("원 계열 — 이웃한 값이 함께 움직인다", fontsize=10,
                  color=BLUE)
    axb.plot(res, color=ORANGE, lw=1.0)
    axb.set_title("iid 붓스트랩 재표본 — 시간 구조가 사라졌다", fontsize=10,
                  color=ORANGE)
    for ax in (axa, axb):
        ax.set_xlim(0, n)
        ax.set_ylim(-4.5, 4.5)
        tidy(ax, left=True)
        ax.set_yticks([-3, 0, 3])
    axb.set_xlabel("시점", fontsize=10, color=INK)

    ax = fig.add_subplot(gs[:, 1])
    K = 15
    lags = np.arange(K + 1)
    ax.bar(lags - 0.2, acf(ser, K), width=0.4, color=BLUE, label="원 계열")
    ax.bar(lags + 0.2, acf(res, K), width=0.4, color=ORANGE,
           label="재표본")
    ax.plot(lags, phi ** lags, color=PURPLE, lw=1.6, ls="--",
            label=r"이론값 $0.7^k$")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("시차 $k$", fontsize=10, color=INK)
    ax.set_ylabel("자기상관", fontsize=10, color=INK)
    ax.set_title("자기상관이 지워진다", fontsize=10.5, color=INK)
    ax.legend(fontsize=9, frameon=False)
    tidy(ax, left=True)

    ax = fig.add_subplot(gs[:, 2])
    ells = [5, 10, 20, 40]
    cov = [0.827, 0.865, 0.863, 0.805]
    wid = [0.650, 0.742, 0.770, 0.720]
    ax.plot(ells, cov, "o-", color=GREEN, lw=2, ms=7, label="이동 블록")
    ax.plot([1], [0.572], "s", color=RED, ms=9)
    ax.text(3.2, 0.572, "iid 붓스트랩 0.572", fontsize=9.5, color=RED,
            va="center")
    ax.axhline(0.95, color=PURPLE, lw=1.4, ls="--")
    ax.text(41, 0.955, "명목 0.95", color=PURPLE, fontsize=9.5, ha="right")
    for e, c, w in zip(ells, cov, wid):
        ax.text(e, c - 0.022, "폭 %.2f" % w, fontsize=8.5, color=GREEN,
                ha="center", va="top")
    ax.set_xlim(0, 44)
    ax.set_ylim(0.50, 1.0)
    ax.set_xlabel(r"블록 길이 $\ell$", fontsize=10, color=INK)
    ax.set_ylabel("포함확률", fontsize=10, color=INK)
    ax.set_title("블록도 완전하지는 않다", fontsize=10.5, color=INK)
    tidy(ax, left=True)

    save(fig, "dependence_breaks_bootstrap.png")


# ===========================================================================
# 그림 6. 순열검정은 통계량 선택에 달렸다 (resampling_methods.md)
# ===========================================================================
def fig_statistic_choice():
    rng = np.random.default_rng(9)
    nx, ny, B, M = 20, 50, 20000, 40000

    # 참 표집분포 (H0 가 참일 때)
    xs = rng.normal(5, 1, (M, nx))
    ys = rng.normal(5, 3, (M, ny))
    d_true = xs.mean(1) - ys.mean(1)
    t_true = d_true / np.sqrt(xs.var(1, ddof=1) / nx + ys.var(1, ddof=1) / ny)

    # 표본 하나의 순열분포
    x = rng.normal(5, 1, nx)
    y = rng.normal(5, 3, ny)
    z = np.concatenate([x, y])
    P = np.array([rng.permutation(z) for _ in range(B)])
    A, C = P[:, :nx], P[:, nx:]
    d_perm = A.mean(1) - C.mean(1)
    t_perm = d_perm / np.sqrt(A.var(1, ddof=1) / nx + C.var(1, ddof=1) / ny)

    print("[stat] 참 sd(평균차)=%.3f  순열 sd=%.3f  비=%.2f"
          % (d_true.std(), d_perm.std(), d_perm.std() / d_true.std()))
    print("   참 t 2.5/97.5=%.2f/%.2f  순열 t=%.2f/%.2f"
          % (*np.percentile(t_true, [2.5, 97.5]),
             *np.percentile(t_perm, [2.5, 97.5])))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                             gridspec_kw=dict(wspace=0.22,
                                              width_ratios=[1, 1, 0.85]))

    bins = np.linspace(-2.6, 2.6, 110)
    step_hist(axes[0], d_perm, bins, ORANGE, ORANGE_F)
    h, e = np.histogram(d_true, bins=bins, density=True)
    axes[0].step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=BLUE, lw=2.2)
    ymax = axes[0].get_ylim()[1]
    axes[0].set_ylim(0, ymax * 1.36)
    axes[0].text(-2.5, ymax * 1.28, "참 표집분포 (sd %.2f)" % d_true.std(),
                 color=BLUE, fontsize=9.5, va="center")
    axes[0].text(-2.5, ymax * 1.14, "순열 귀무분포 (sd %.2f)" % d_perm.std(),
                 color=ORANGE, fontsize=9.5, va="center")
    axes[0].set_xlim(-2.6, 2.6)
    axes[0].set_title(r"(a) 통계량 = $\bar x - \bar y$", fontsize=11,
                      color=INK)
    axes[0].set_xlabel("평균차", fontsize=10, color=INK)
    tidy(axes[0])

    bins = np.linspace(-4.5, 4.5, 110)
    step_hist(axes[1], t_perm, bins, ORANGE, ORANGE_F)
    h, e = np.histogram(t_true, bins=bins, density=True)
    axes[1].step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=BLUE, lw=2.2)
    ymax = axes[1].get_ylim()[1]
    axes[1].set_ylim(0, ymax * 1.36)
    axes[1].text(-4.4, ymax * 1.28,
                 "참 97.5 백분위수 %.2f" % np.percentile(t_true, 97.5),
                 color=BLUE, fontsize=9.5, va="center")
    axes[1].text(-4.4, ymax * 1.14,
                 "순열 97.5 백분위수 %.2f" % np.percentile(t_perm, 97.5),
                 color=ORANGE, fontsize=9.5, va="center")
    axes[1].set_xlim(-4.5, 4.5)
    axes[1].set_title(r"(b) 통계량 = Welch $t$", fontsize=11, color=INK)
    axes[1].set_xlabel("$t$ 값", fontsize=10, color=INK)
    tidy(axes[1])

    ax = axes[2]
    xs2 = np.arange(2)
    ax.bar(xs2 - 0.18, [0.003, 0.591], width=0.34, color=ORANGE,
           label=r"$\bar x - \bar y$")
    ax.bar(xs2 + 0.18, [0.053, 0.857], width=0.34, color=GREEN,
           label=r"Welch $t$")
    for xx, v in zip([-0.18, 0.18, 0.82, 1.18], [0.003, 0.053, 0.591, 0.857]):
        ax.text(xx, v + 0.02, "%.3f" % v, ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.axhline(0.05, color=PURPLE, lw=1.2, ls="--")
    ax.text(0.5, 0.085, "명목 0.05", color=PURPLE, fontsize=9, ha="center")
    ax.set_xticks(xs2)
    ax.set_xticklabels(["제1종 오류율", "검정력 (이동 1.5)"], fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("(c) 결과", fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    tidy(ax, left=True)

    save(fig, "statistic_choice.png")


if __name__ == "__main__":
    fig_two_questions()
    fig_small_gap()
    fig_convergence()
    fig_cv_flat()
    fig_dependence()
    fig_statistic_choice()
