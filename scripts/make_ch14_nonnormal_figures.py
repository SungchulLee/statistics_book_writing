r"""14장 비정규 자료 절 두 쪽의 개념 그림을 만든다.

만드는 파일:
  ch14/non_normal_data/img/bootstrap_shape.png      붓스트랩은 모양에 적응한다
  ch14/non_normal_data/img/rank_test_power.png      순위검정이 잃는 것과 얻는 것

실행:  python3 scripts/make_ch14_nonnormal_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch14/non_normal_data/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ===================================================================
# 그림 1. 붓스트랩 분포는 표집분포의 모양을 따라간다
# ===================================================================
def fig_bootstrap_shape():
    rng = np.random.default_rng(1420)
    n = 20
    sigma = 0.9                       # 대수정규 모양모수
    mu_true = np.exp(sigma**2 / 2)

    # (1) 진짜 표집분포 — 현실에서는 알 수 없지만 모의실험으로는 볼 수 있다
    true_means = rng.lognormal(0.0, sigma, size=(200_000, n)).mean(axis=1)

    # (2) 표본 하나에서 만든 붓스트랩 분포
    x = rng.lognormal(0.0, sigma, size=n)
    idx = rng.integers(0, n, size=(20_000, n))
    boot_means = x[idx].mean(axis=1)

    # (3) 그 표본의 t 근사 — 대칭일 수밖에 없다
    xbar, s = x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)

    print(f"모집단 왜도 = {float(stats.lognorm.stats(sigma, moments='s')):.3f}")
    print(f"진짜 표집분포의 왜도 = {stats.skew(true_means):.3f}")
    print(f"붓스트랩 분포의 왜도 = {stats.skew(boot_means):.3f}")
    print(f"표본: xbar={xbar:.3f}  s={s:.3f}  참평균={mu_true:.3f}")
    lo_b, hi_b = np.quantile(boot_means, [0.025, 0.975])
    print(f"백분위수 붓스트랩 구간 = [{lo_b:.3f}, {hi_b:.3f}]")
    print(f"t 구간 = [{xbar - tcrit*se:.3f}, {xbar + tcrit*se:.3f}]")

    # (4) 두 구간의 실제 포함률
    ns = [10, 20, 40, 80, 160]
    B_out, B_in = 1500, 800
    cov_t, cov_b = [], []
    for nn in ns:
        ct = cb = 0
        for _ in range(B_out):
            xs = rng.lognormal(0.0, sigma, size=nn)
            m, sd = xs.mean(), xs.std(ddof=1)
            tc = stats.t.ppf(0.975, nn - 1) * sd / np.sqrt(nn)
            if abs(m - mu_true) < tc:
                ct += 1
            ii = rng.integers(0, nn, size=(B_in, nn))
            bm = xs[ii].mean(axis=1)
            lo, hi = np.quantile(bm, [0.025, 0.975])
            if lo < mu_true < hi:
                cb += 1
        cov_t.append(ct / B_out)
        cov_b.append(cb / B_out)
        print(f"n={nn:4d}  t 구간 포함률={cov_t[-1]:.3f}   "
              f"백분위수 붓스트랩={cov_b[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    ax = axes[0]
    lo, hi = 0.6, 3.6
    ax.hist(true_means, bins=200, range=(lo, hi), density=True,
            color="#ECEFF1", edgecolor=MUTED, linewidth=0.2,
            label=r"$\bar{X}$ 의 진짜 표집분포")
    ax.hist(boot_means - xbar + mu_true, bins=110, range=(lo, hi),
            density=True, histtype="step", color=ORANGE, linewidth=2.0,
            label="표본 하나에서 만든 붓스트랩 분포\n(중심을 맞춰 겹쳤다)")
    g = np.linspace(lo, hi, 400)
    ax.plot(g, stats.norm.pdf(g, mu_true, se), color=BLUE, lw=2.2, ls="--",
            label=r"$t$ 근사가 강요하는 대칭 종모양")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"표본평균 $\bar{X}$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(f"대수정규 모집단, $n = {n}$  (모집단 왜도 $4.75$)",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)
    ax.set_yticks([])

    ax = axes[1]
    ax.axhline(0.95, color=MUTED, lw=1.3, ls="--", label="약속한 0.95")
    ax.plot(ns, cov_t, "s-", color=BLUE, lw=2, ms=6, label=r"$t$ 구간")
    ax.plot(ns, cov_b, "o-", color=ORANGE, lw=2, ms=6,
            label="백분위수 붓스트랩 구간")
    for xx, v in zip(ns, cov_t):
        ax.annotate(f"{v:.3f}", (xx, v), textcoords="offset points",
                    xytext=(0, 7), fontsize=8.5, color=BLUE, ha="center")
    for xx, v in zip(ns, cov_b):
        ax.annotate(f"{v:.3f}", (xx, v), textcoords="offset points",
                    xytext=(0, -14), fontsize=8.5, color=ORANGE, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_xlim(8.5, 200)
    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("95% 구간의 실제 포함률", fontsize=10.5, color=INK)
    ax.set_title(f"같은 모집단에서 잰 두 구간의 성능 ({B_out}회 반복)",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right", labelcolor=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "bootstrap_shape.png")


# ===================================================================
# 그림 2. 순위검정이 잃는 것과 얻는 것
# ===================================================================
def fig_rank_test_power():
    rng = np.random.default_rng(1421)
    n, B, alpha = 25, 8000, 0.05

    def draw(kind, shift):
        if kind == "normal":
            a = rng.normal(0, 1, size=(B, n))
            b = rng.normal(0, 1, size=(B, n)) + shift
        elif kind == "exp":
            a = rng.exponential(1.0, size=(B, n))
            b = rng.exponential(1.0, size=(B, n)) + shift
        elif kind == "t3":
            a = rng.standard_t(3, size=(B, n)) / np.sqrt(3.0)
            b = rng.standard_t(3, size=(B, n)) / np.sqrt(3.0) + shift
        else:
            raise ValueError(kind)
        return a, b

    kinds = [("normal", "정규분포", 0.80),
             ("exp", "지수분포\n(치우침)", 0.80),
             ("t3", "$t_3$\n(두꺼운 꼬리)", 0.80)]
    labels, p_t, p_u, p_size_t, p_size_u = [], [], [], [], []
    for kind, lab, shift in kinds:
        a, b = draw(kind, shift)
        p_t.append(np.mean(stats.ttest_ind(a, b, axis=1).pvalue < alpha))
        p_u.append(np.mean(stats.mannwhitneyu(a, b, axis=1).pvalue < alpha))
        a0, b0 = draw(kind, 0.0)
        p_size_t.append(np.mean(stats.ttest_ind(a0, b0, axis=1).pvalue < alpha))
        p_size_u.append(np.mean(
            stats.mannwhitneyu(a0, b0, axis=1).pvalue < alpha))
        labels.append(lab)
        print(f"{kind:8s} 이동 {shift}: t 검정력={p_t[-1]:.3f}  "
              f"Mann-Whitney={p_u[-1]:.3f}   "
              f"(크기: t={p_size_t[-1]:.3f}  MW={p_size_u[-1]:.3f})")

    p_t, p_u = np.array(p_t), np.array(p_u)
    ratio = p_u / p_t

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    xs = np.arange(3)
    w = 0.32
    ax.bar(xs - w / 2, p_t, width=w * 0.95, color=BLUE_F, edgecolor=BLUE,
           linewidth=1.3, label=r"이표본 $t$ 검정")
    ax.bar(xs + w / 2, p_u, width=w * 0.95, color=ORANGE_F, edgecolor=ORANGE,
           linewidth=1.3, label="Mann-Whitney $U$ 검정")
    for xx, v in zip(xs - w / 2, p_t):
        ax.annotate(f"{v:.3f}", (xx, v), textcoords="offset points",
                    xytext=(0, 4), fontsize=10, color=BLUE, ha="center")
    for xx, v in zip(xs + w / 2, p_u):
        ax.annotate(f"{v:.3f}", (xx, v), textcoords="offset points",
                    xytext=(0, 4), fontsize=10, color=ORANGE, ha="center")
    for xx, r in zip(xs, ratio):
        ax.text(xx, 1.10, f"순위검정 / $t$ 검정 = {r:.2f}", fontsize=10,
                color=RED if r > 1 else INK, ha="center")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("검정력", fontsize=10.5, color=INK)
    ax.set_title(f"두 집단을 $0.80$ 만큼 옮긴 대립가설 "
                 f"(집단당 $n = {n}$, {B}회 반복)", fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), ncol=2, labelcolor=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "rank_test_power.png")


if __name__ == "__main__":
    fig_bootstrap_shape()
    fig_rank_test_power()
