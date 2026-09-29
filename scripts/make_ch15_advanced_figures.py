r"""15장 고급 방법(붓스트랩·베이즈·가능도비) 여섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch15/advanced_methods/img/bootstrap_pool_vs_within.png  합쳐서 뽑아야 귀무분포가 된다
  ch15/advanced_methods/img/bootstrap_vs_f_size.png       붓스트랩은 비정규에서도 크기를 지킨다
  ch15/advanced_methods/img/bootstrap_null_reference.png  p 값은 귀무값과 견주어야 한다
  ch15/advanced_methods/img/bayes_prior_weight.png        사전분포는 가상 관측값 2*alpha0 개다
  ch15/advanced_methods/img/bayes_vs_frequentist_ci.png   신용구간과 신뢰구간의 차이
  ch15/advanced_methods/img/lrt_g_function.png            r - 1 - ln r 의 비대칭

실행:  python3 scripts/make_ch15_advanced_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats
from scipy.optimize import brentq

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

OUT = "docs/ch15/advanced_methods/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 그림 1. 합쳐서 뽑기 vs 집단 안에서 뽑기 ===
def fig_pool_vs_within():
    rng = np.random.default_rng(0)
    s1 = np.array([10, 12, 14, 16, 18], float)
    s2 = np.array([22, 24, 26, 30, 40], float)
    obs = s1.var(ddof=1) - s2.var(ddof=1)
    B = 20000

    pool = np.concatenate([s1 - s1.mean(), s2 - s2.mean()])
    idx = rng.integers(0, pool.size, size=(B, 10))
    d_pool = pool[idx][:, :5].var(axis=1, ddof=1) - \
        pool[idx][:, 5:].var(axis=1, ddof=1)
    p_pool = float(np.mean(np.abs(d_pool) >= abs(obs)))

    i1 = rng.integers(0, 5, size=(B, 5))
    i2 = rng.integers(0, 5, size=(B, 5))
    d_within = s1[i1].var(axis=1, ddof=1) - s2[i2].var(axis=1, ddof=1)
    p_within = float(np.mean(np.abs(d_within) >= abs(obs)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5), sharex=True)

    bins = np.linspace(-140, 100, 90)
    for ax, d, p, title, color in [
        (ax1, d_within, p_within, "잘못된 방법: 집단 안에서 따로 재표집", RED),
        (ax2, d_pool, p_pool, "올바른 방법: 중심화해 합친 풀에서 재표집", GREEN),
    ]:
        ax.hist(d, bins=bins, color=color, alpha=0.45)
        ax.axvline(obs, color=BLUE, lw=2.0, ls=(0, (4, 3)))
        ax.axvline(float(np.median(d)), color=color, lw=1.6)
        ax.set_xlim(-140, 100)
        ax.set_ylim(0, 1800)
        ax.set_xlabel("붓스트랩 분산차 $s_1^{*2} - s_2^{*2}$", fontsize=11,
                      color=INK)
        clean(ax)
        ax.set_title(title, fontsize=11.5, color=INK, pad=9)
        ax.text(0.03, 0.95,
                f"붓스트랩 분포의 중앙값 {np.median(d):.1f}\n"
                f"관측값 {obs:.1f}\n$p$ = {p:.4f}",
                transform=ax.transAxes, fontsize=10.5, color=INK, va="top")
    ax1.set_ylabel("빈도", fontsize=11, color=INK)
    ax1.annotate("관측값이 분포 한가운데 있다", xy=(obs, 780),
                 xytext=(-138, 1180), fontsize=10.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax2.annotate("관측값이 꼬리에 있다", xy=(obs, 330),
                 xytext=(-138, 900), fontsize=10.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))

    fig.tight_layout()
    save(fig, "bootstrap_pool_vs_within.png")
    print(f"  관측 차이 {obs:.2f}, 잘못된 p {p_within:.4f}, "
          f"올바른 p {p_pool:.4f}")


# === 그림 2. 붓스트랩은 비정규에서도 크기를 지킨다 ===
def fig_bootstrap_size():
    rng = np.random.default_rng(1)
    n, B, R = 50, 399, 2000
    res = {}
    for name, gen in [("정규", lambda: rng.normal(0, 1, n)),
                      ("지수", lambda: rng.exponential(1, n))]:
        f_rej = b_rej = 0
        keep = None
        for r in range(R):
            x, y = gen(), gen()
            F = x.var(ddof=1) / y.var(ddof=1)
            p = 2 * min(stats.f.cdf(F, n - 1, n - 1),
                        stats.f.sf(F, n - 1, n - 1))
            f_rej += p < 0.05
            pool = np.concatenate([x - x.mean(), y - y.mean()])
            idx = rng.integers(0, 2 * n, size=(B, 2 * n))
            s = pool[idx]
            lf = np.abs(np.log(s[:, :n].var(axis=1, ddof=1)
                               / s[:, n:].var(axis=1, ddof=1)))
            b_rej += float(np.mean(lf >= abs(np.log(F)))) < 0.05
            if r == 0 and name == "지수":
                big = pool[rng.integers(0, 2 * n, size=(20000, 2 * n))]
                keep = (np.log(big[:, :n].var(axis=1, ddof=1)
                               / big[:, n:].var(axis=1, ddof=1)),
                        np.log(F))
        res[name] = (f_rej / R, b_rej / R)
        print(f"  {name}: F 검정 {res[name][0]:.4f}, "
              f"붓스트랩 {res[name][1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    boot, obs = keep
    grid = np.linspace(-1.6, 1.6, 400)
    ax1.hist(boot, bins=np.linspace(-1.6, 1.6, 90), density=True,
             color=GREEN, alpha=0.40,
             label="붓스트랩 귀무분포 (지수 자료, $B = 20{,}000$)")
    fdens = stats.f.pdf(np.exp(grid), n - 1, n - 1) * np.exp(grid)
    ax1.plot(grid, fdens, color=RED, lw=2.2,
             label="$F$ 검정이 쓰는 기준분포")
    q = np.percentile(boot, [2.5, 97.5])
    fq = np.log([stats.f.ppf(0.025, n - 1, n - 1),
                 stats.f.ppf(0.975, n - 1, n - 1)])
    for v in q:
        ax1.vlines(v, 0, 1.05, color=GREEN, lw=1.5, ls=(0, (5, 3)))
    for v in fq:
        ax1.vlines(v, 0, 1.05, color=RED, lw=1.5, ls=(0, (5, 3)))
    ax1.text(q[1] + 0.03, 1.10, f"붓스트랩 2.5% 한계 ±{q[1]:.3f}",
             fontsize=9.5, color=GREEN, ha="left")
    ax1.text(fq[1] + 0.03, 0.72, f"$F$ 기준 ±{fq[1]:.3f}", fontsize=9.5,
             color=RED, ha="left")
    ax1.set_xlim(-1.6, 1.6)
    ax1.set_ylim(0, 1.55)
    ax1.set_xlabel("$\\ln(S_1^2/S_2^2)$", fontsize=11, color=INK)
    ax1.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper left")
    ax1.set_title(f"한 자료에서 본 두 귀무분포 ($n_1 = n_2 = {n}$)",
                  fontsize=11.5, color=INK, pad=9)

    labels = ["정규 자료", "지수 자료"]
    xs = np.arange(2)
    w = 0.33
    fvals = [res["정규"][0], res["지수"][0]]
    bvals = [res["정규"][1], res["지수"][1]]
    b1 = ax2.bar(xs - w / 2, fvals, w, color=RED, alpha=0.85, label="$F$ 검정")
    b2 = ax2.bar(xs + w / 2, bvals, w, color=GREEN, alpha=0.85,
                 label="붓스트랩 검정")
    for bars, vals in [(b1, fvals), (b2, bvals)]:
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.4f}",
                     ha="center", fontsize=10, color=INK)
    ax2.axhline(0.05, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax2.text(-0.52, 0.060, "명목 0.05", fontsize=10, color=INK, ha="left")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=11, color=INK)
    ax2.set_xlim(-0.55, 1.55)
    ax2.set_ylim(0, 0.35)
    ax2.set_ylabel("실제 제1종 오류율", fontsize=11, color=INK)
    clean(ax2)
    ax2.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax2.set_title(f"{R}회 반복, 붓스트랩 $B = {B}$", fontsize=11.5,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "bootstrap_vs_f_size.png")


# === 그림 3. p 값은 귀무값과 견주어야 한다 ===
def fig_null_reference():
    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], float)
    rng = np.random.default_rng(42)
    B = 10000
    obs = np.log(x1.var(ddof=1) / x2.var(ddof=1))
    boot = np.empty(B)          # 본문 코드와 같은 난수 소비 순서
    for b in range(B):
        r1 = rng.choice(x1, size=8, replace=True)
        r2 = rng.choice(x2, size=8, replace=True)
        boot[b] = np.log(r1.var(ddof=1) / r2.var(ddof=1))
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_right = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    p_wrong = 2 * min(float(np.mean(boot <= obs)), float(np.mean(boot >= obs)))

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.hist(boot, bins=np.linspace(-4.2, 3.0, 110), color=PURPLE, alpha=0.35)
    ax.axvline(0.0, color=GREEN, lw=2.2)
    ax.axvline(obs, color=RED, lw=2.2, ls=(0, (4, 3)))
    ax.axvline(lo, color=INK, lw=1.3, ls=":")
    ax.axvline(hi, color=INK, lw=1.3, ls=":")
    top = 470
    ax.annotate(f"귀무값 $\\ln 1 = 0$\n$p$ = {p_right:.4f}  ← 올바름",
                xy=(0.0, top * 0.80), xytext=(0.55, top * 0.93),
                fontsize=11, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    ax.annotate(f"관측값 $\\ln \\hat\\theta$ = {obs:.4f}\n"
                f"$p$ = {p_wrong:.4f}  ← 틀림",
                xy=(obs, top * 0.62), xytext=(-4.1, top * 0.86),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.text(lo - 0.07, top * 0.30, f"2.5% 백분위\n{np.exp(lo):.4f}",
            fontsize=9.5, color=INK, ha="right", va="top")
    ax.text(hi + 0.07, top * 0.30, f"97.5% 백분위\n{np.exp(hi):.4f}",
            fontsize=9.5, color=INK, ha="left", va="top")
    ax.set_xlim(-4.2, 3.0)
    ax.set_ylim(0, top)
    ax.set_xlabel("$\\ln \\hat\\theta^{*}$   (붓스트랩 분산비의 로그)",
                  fontsize=11, color=INK)
    ax.set_ylabel("빈도", fontsize=11, color=INK)
    clean(ax)
    ax.set_title("집단 안에서 재표집한 붓스트랩 분포 ($B = 10{,}000$)",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "bootstrap_null_reference.png")
    print(f"  obs log ratio {obs:.4f}, CI ({np.exp(lo):.4f}, "
          f"{np.exp(hi):.4f}), p 올바름 {p_right:.4f}, p 틀림 {p_wrong:.4f}")


# === 그림 4. 사전분포는 가상 관측값 2*alpha0 개다 ===
def fig_prior_weight():
    n, SS = 20, 180.0
    s2 = SS / (n - 1)
    prior_mean = 4.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    x = np.linspace(2, 32, 800)
    for a0, color, lab in [(0.001, GREEN, "$\\alpha_0 = 0.001$  (확산)"),
                           (9.5, ORANGE, "$\\alpha_0 = 9.5$  (자료와 대등)"),
                           (40.0, RED, "$\\alpha_0 = 40$  (강한 사전분포)")]:
        b0 = a0 * (prior_mean - 0.0) if a0 > 1 else 0.001
        an = a0 + (n - 1) / 2
        bn = b0 + SS / 2
        ax1.plot(x, stats.invgamma.pdf(x, a=an, scale=bn), color=color,
                 lw=2.2, label=lab)
        print(f"  a0={a0}: 사후평균 {bn / (an - 1):.4f}")
    ax1.axvline(s2, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax1.text(s2 + 0.5, 0.40, f"표본분산 $s^2$ = {s2:.3f}", fontsize=10.5,
             color=INK)
    ax1.axvline(prior_mean, color=MUTED, lw=1.4, ls=":")
    ax1.text(prior_mean - 0.5, 0.47, "사전평균 4", fontsize=10.5,
             color=MUTED, ha="right")
    ax1.set_xlim(2, 32)
    ax1.set_ylim(0, 0.62)
    ax1.set_xlabel("$\\sigma^2$", fontsize=11, color=INK)
    ax1.set_ylabel("사후밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper right")
    ax1.set_title(f"$n = {n}$, $\\sum (x_i-\\bar{{x}})^2 = {SS:.0f}$ 인 같은 자료",
                  fontsize=11.5, color=INK, pad=9)

    a0s = np.linspace(0.001, 60, 400)
    means = []
    for a0 in a0s:
        b0 = a0 * prior_mean
        an, bn = a0 + (n - 1) / 2, b0 + SS / 2
        means.append(bn / (an - 1))
    ax2.plot(2 * a0s, means, color=BLUE, lw=2.4)
    ax2.axhline(s2, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax2.text(118, s2 + 0.25, f"표본분산 {s2:.3f}", fontsize=10, color=INK,
             ha="right")
    ax2.axhline(prior_mean, color=MUTED, lw=1.3, ls=":")
    ax2.text(118, prior_mean + 0.25, "사전평균 4", fontsize=10, color=MUTED,
             ha="right")
    ax2.axvline(n - 1, color=PURPLE, lw=1.5, ls=(0, (4, 3)))
    ax2.annotate(f"가상 관측값 {n - 1}개 = 자료의 자유도\n여기서 둘의 무게가 같다",
                 xy=(n - 1, 7.25), xytext=(44, 8.3), fontsize=10.5,
                 color=PURPLE,
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax2.set_xlim(0, 120)
    ax2.set_ylim(3.5, 11)
    ax2.set_xlabel("사전분포가 담은 가상 관측값 수  $2\\alpha_0$", fontsize=11,
                   color=INK)
    ax2.set_ylabel("사후평균", fontsize=11, color=INK)
    clean(ax2)
    ax2.set_title("사전평균 4 로 고정하고 강도만 키울 때", fontsize=11.5,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "bayes_prior_weight.png")


# === 그림 5. 신용구간과 신뢰구간 ===
def fig_credible_vs_ci():
    n, SS = 20, 180.0
    a0 = b0 = 0.001
    an, bn = a0 + (n - 1) / 2, b0 + SS / 2
    cred = stats.invgamma.ppf([0.025, 0.975], a=an, scale=bn)
    freq = (SS / stats.chi2.ppf(0.975, n - 1),
            SS / stats.chi2.ppf(0.025, n - 1))

    an2 = a0 + n / 2
    bn2 = b0 + SS / 2
    cred2 = stats.invgamma.ppf([0.025, 0.975], a=an2, scale=bn2)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})

    x = np.linspace(2, 32, 800)
    y = stats.invgamma.pdf(x, a=an, scale=bn)
    ax1.fill_between(x, y, where=(x >= cred[0]) & (x <= cred[1]),
                     color=BLUE_F, alpha=0.95)
    ax1.plot(x, y, color=BLUE, lw=2.3)
    for v in cred:
        ax1.vlines(v, 0, stats.invgamma.pdf(v, a=an, scale=bn), color=BLUE,
                   lw=1.5)
    ax1.text((cred[0] + cred[1]) / 2, 0.022, "95% 신용구간", fontsize=11,
             color=BLUE, ha="center")
    ax1.text(cred[0], 0.148, f"{cred[0]:.3f}", fontsize=10, color=BLUE,
             ha="center")
    ax1.text(cred[1], 0.148, f"{cred[1]:.3f}", fontsize=10, color=BLUE,
             ha="center")
    ax1.text(19.5, 0.10,
             f"신용구간 ({cred[0]:.3f}, {cred[1]:.3f})\n"
             f"신뢰구간 ({freq[0]:.3f}, {freq[1]:.3f})\n"
             f"소수 셋째 자리까지 같다",
             fontsize=10.5, color=INK, va="top")
    ax1.set_xlim(2, 32)
    ax1.set_ylim(0, 0.165)
    ax1.set_xlabel("$\\sigma^2$", fontsize=11, color=INK)
    ax1.set_ylabel("사후밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.set_title("확산 사전분포 $\\text{Inv-Gamma}(0.001, 0.001)$, "
                  f"$n = {n}$", fontsize=11.5, color=INK, pad=9)

    labels = [f"$\\alpha_n = \\alpha_0 + (n-1)/2$\n(평균을 추정한 경우)",
              f"$\\alpha_n = \\alpha_0 + n/2$\n(평균을 아는 경우)",
              "빈도주의 신뢰구간"]
    rows = [cred, cred2, freq]
    cols = [BLUE, RED, ORANGE]
    for i, (iv, c) in enumerate(zip(rows, cols)):
        y = 2 - i
        ax2.plot([iv[0], iv[1]], [y, y], color=c, lw=7, alpha=0.8,
                 solid_capstyle="butt")
        ax2.text(iv[0] - 0.4, y, f"{iv[0]:.2f}", ha="right", va="center",
                 fontsize=10, color=c)
        ax2.text(iv[1] + 0.4, y, f"{iv[1]:.2f}", ha="left", va="center",
                 fontsize=10, color=c)
    ax2.set_yticks([2, 1, 0])
    ax2.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax2.set_xlim(2, 26)
    ax2.set_ylim(-0.7, 2.7)
    ax2.set_xlabel("$\\sigma^2$", fontsize=11, color=INK)
    clean(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("자유도를 잘못 쓰면 구간이 어긋난다", fontsize=11.5,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "bayes_vs_frequentist_ci.png")
    print(f"  신용 ({cred[0]:.4f}, {cred[1]:.4f})  "
          f"신뢰 ({freq[0]:.4f}, {freq[1]:.4f})  "
          f"n/2 판 ({cred2[0]:.4f}, {cred2[1]:.4f})")


# === 그림 6. r - 1 - ln r 의 비대칭 ===
def fig_lrt_g():
    r = np.linspace(0.08, 3.2, 800)
    g = r - 1 - np.log(r)
    bad = r - 1 + np.log(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    ax1.axhline(0, color=MUTED, lw=1.0)
    ax1.plot(r, g, color=GREEN, lw=2.4, label="$r - 1 - \\ln r$  (올바름)")
    ax1.plot(r, bad, color=RED, lw=2.4, ls=(0, (5, 3)),
             label="$r - 1 + \\ln r$  (틀림)")
    ax1.fill_between(r, bad, 0, where=bad < 0, color=RED, alpha=0.20)
    for v in (0.5, 1.0, 2.0):
        ax1.plot([v], [v - 1 - np.log(v)], marker="o", ms=7, color=GREEN,
                 zorder=5)
        ax1.plot([v], [v - 1 + np.log(v)], marker="o", ms=7, color=RED,
                 zorder=5)
    ax1.text(0.52, 0.193 + 0.12, "0.193", fontsize=10, color=GREEN)
    ax1.text(2.0, 0.307 + 0.12, "0.307", fontsize=10, color=GREEN,
             ha="center")
    ax1.text(0.44, -1.193, "$-1.193$", fontsize=10, color=RED,
             ha="right", va="center")
    ax1.text(2.0, 1.693 + 0.10, "1.693", fontsize=10, color=RED,
             ha="center")
    ax1.text(1.55, -1.45, "$-2\\ln\\Lambda$ 는 음수가 될 수 없다",
             fontsize=10.5, color=RED, ha="center")
    ax1.set_xlim(0, 3.2)
    ax1.set_ylim(-1.8, 2.1)
    ax1.set_xlabel("$r = \\hat\\sigma^2/\\sigma_0^2$", fontsize=11, color=INK)
    ax1.set_ylabel("함수값", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax1.set_title("부호 하나가 검정을 무너뜨린다", fontsize=11.5, color=INK,
                  pad=9)

    n = 25
    crit = stats.chi2.ppf(0.95, 1)
    stat = n * (r - 1 - np.log(r))
    ax2.plot(r, stat, color=BLUE, lw=2.4)
    ax2.fill_between(r, 0, stat, color=BLUE_F, alpha=0.8)
    ax2.axhline(crit, color=RED, lw=1.4, ls=(0, (5, 3)))
    ax2.text(1.10, crit + 0.45, f"$\\chi^2_{{0.95,1}}$ = {crit:.3f}",
             fontsize=10.5, color=RED, ha="center")
    f = lambda v: n * (v - 1 - np.log(v)) - crit
    r_lo = brentq(f, 0.05, 0.999)
    r_hi = brentq(f, 1.001, 10.0)
    for v in (r_lo, r_hi):
        ax2.vlines(v, 0, crit, color=RED, lw=1.4)
        ax2.plot([v], [crit], marker="o", ms=7, color=RED, zorder=5)
    ax2.text(r_lo - 0.04, crit * 0.55, f"{r_lo:.3f}", fontsize=10.5,
             color=RED, ha="right")
    ax2.text(r_hi + 0.04, crit * 0.55, f"{r_hi:.3f}", fontsize=10.5,
             color=RED, ha="left")
    ax2.text(1.0, 9.0,
             f"기각하지 않는 구간\n$1 - {1 - r_lo:.3f}$  vs  "
             f"$1 + {r_hi - 1:.3f}$\n1 을 중심으로 대칭이 아니다",
             fontsize=10.5, color=INK, ha="center", va="top")
    ax2.set_xlim(0, 3.2)
    ax2.set_ylim(0, 12)
    ax2.set_xlabel("$r = \\hat\\sigma^2/\\sigma_0^2$", fontsize=11, color=INK)
    ax2.set_ylabel(f"$-2\\ln\\Lambda = n(r-1-\\ln r)$", fontsize=11,
                   color=INK)
    clean(ax2)
    ax2.set_title(f"$n = {n}$ 에서의 기각역", fontsize=11.5, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "lrt_g_function.png")
    print(f"  g(0.5)={0.5 - 1 - np.log(0.5):.4f}, "
          f"g(2)={2 - 1 - np.log(2):.4f}, "
          f"기각역 경계 {r_lo:.4f} / {r_hi:.4f}")


# === 그림 7. 역감마 사후분포를 무엇으로 요약할 것인가 ===
def fig_posterior_summary():
    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], float)
    n = x1.size
    s2 = x1.var(ddof=1)
    a0 = b0 = 0.01
    an = a0 + n / 2
    bn = b0 + 0.5 * ((x1 - x1.mean()) ** 2).sum()
    mode = bn / (an + 1)
    mean = bn / (an - 1)
    med = float(stats.invgamma.ppf(0.5, a=an, scale=bn))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})

    x = np.linspace(0.3, 16, 900)
    y = stats.invgamma.pdf(x, a=an, scale=bn)
    ax1.fill_between(x, y, color=BLUE_F, alpha=0.9)
    ax1.plot(x, y, color=BLUE, lw=2.3)
    ymax = float(y.max())
    for v, c, lab, dy in [(mode, PURPLE, f"최빈값 {mode:.3f}", 0.96 * ymax),
                          (med, GREEN, f"중앙값 {med:.3f}", 0.78 * ymax),
                          (s2, INK, f"표본분산 {s2:.3f}", 0.60 * ymax),
                          (mean, RED, f"사후평균 {mean:.3f}", 0.42 * ymax)]:
        ax1.vlines(v, 0, stats.invgamma.pdf(v, a=an, scale=bn), color=c,
                   lw=1.8)
        ax1.annotate(lab, xy=(v, stats.invgamma.pdf(v, a=an, scale=bn)),
                     xytext=(7.5, dy), fontsize=10.5, color=c,
                     arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, float(y.max()) * 1.15)
    ax1.set_xlabel("$\\sigma_1^2$", fontsize=11, color=INK)
    ax1.set_ylabel("사후밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.set_title(f"$\\text{{Inv-Gamma}}({an:.2f}, {bn:.3f})$ — 집단 1, $n = {n}$",
                  fontsize=11.5, color=INK, pad=9)

    ns = np.arange(4, 61)
    ratio = [(a0 + m / 2 - 1) for m in ns]
    infl = [(m - 1) / 2 / (a0 + m / 2 - 1) for m in ns]
    ax2.plot(ns, infl, color=RED, lw=2.4, label="사후평균 / 표본분산")
    infl_mode = [(m - 1) / 2 / (a0 + m / 2 + 1) for m in ns]
    ax2.plot(ns, infl_mode, color=PURPLE, lw=2.4, label="사후최빈값 / 표본분산")
    ax2.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax2.plot([n], [(n - 1) / 2 / (a0 + n / 2 - 1)], marker="o", ms=8,
             color=RED, zorder=5)
    ax2.annotate(f"$n = {n}$ 에서 {100 * ((n - 1) / 2 / (a0 + n / 2 - 1) - 1):.0f}% 크다",
                 xy=(n, (n - 1) / 2 / (a0 + n / 2 - 1)), xytext=(17, 1.22),
                 fontsize=10.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax2.set_xlim(2, 62)
    ax2.set_ylim(0.55, 1.35)
    ax2.set_xlabel("표본크기 $n$", fontsize=11, color=INK)
    ax2.set_ylabel("표본분산 대비 비", fontsize=11, color=INK)
    clean(ax2)
    ax2.legend(fontsize=10, frameon=False, loc="lower right")
    ax2.set_title("치우침은 $n$ 과 함께 줄어든다", fontsize=11.5, color=INK,
                  pad=9)

    fig.tight_layout()
    save(fig, "posterior_summary_skew.png")
    print(f"  an={an:.3f}, bn={bn:.4f}, 최빈 {mode:.4f}, 중앙 {med:.4f}, "
          f"평균 {mean:.4f}, s2 {s2:.4f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_pool_vs_within()
    fig_null_reference()
    fig_prior_weight()
    fig_credible_vs_ci()
    fig_lrt_g()
    fig_posterior_summary()
    fig_bootstrap_size()
