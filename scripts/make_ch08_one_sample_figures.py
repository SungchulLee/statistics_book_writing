r"""8장 일표본 신뢰구간 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch08/one_sample_intervals/img/t_vs_z_cost.png         sigma 를 모르는 대가
  ch08/one_sample_intervals/img/width_decomposition.png 너비 차이를 둘로 쪼갠다
  ch08/one_sample_intervals/img/wald_vs_wilson_n20.png  Wald 가 무너지는 자리
  ch08/one_sample_intervals/img/four_methods_tradeoff.png 보장과 너비의 맞바꿈

실행:  python3 scripts/make_ch08_one_sample_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os
import math

import numpy as np
from scipy import stats
from scipy.stats import norm, beta, binom

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

OUT = "docs/ch08/one_sample_intervals/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 그림 1. sigma 를 모르는 대가 (ci_mu.md) =====================
def fig_t_vs_z_cost():
    z = norm.ppf(0.975)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.6, 4.6), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    # --- 왼쪽: 꼬리가 두꺼우면 임계값이 밀린다 ---
    x = np.linspace(-4.2, 4.2, 900)
    axL.plot(x, norm.pdf(x), color=MUTED, lw=2.0, label="표준정규")
    axL.plot(x, stats.t.pdf(x, 24), color=BLUE, lw=2.0,
             label=r"$t_{24}$ ($n=25$)")
    axL.plot(x, stats.t.pdf(x, 4), color=ORANGE, lw=2.0,
             label=r"$t_4$ ($n=5$)")

    xt = np.linspace(z, 4.2, 300)
    axL.fill_between(xt, 0, norm.pdf(xt), color=MUTED, alpha=0.35, zorder=0)
    for v, c, dy in ((z, INK, 0.0), (stats.t.ppf(0.975, 24), BLUE, 0.0),
                     (stats.t.ppf(0.975, 4), ORANGE, 0.0)):
        axL.plot([v, v], [0, 0.055 + dy], color=c, lw=1.4, ls=(0, (4, 3)))

    axL.annotate("1.960", xy=(z, 0.055), xytext=(1.05, 0.135),
                 fontsize=10.5, color=INK, ha="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.1,
                                 shrinkA=2, shrinkB=2))
    axL.annotate("2.064", xy=(stats.t.ppf(0.975, 24), 0.055),
                 xytext=(2.55, 0.175),
                 fontsize=10.5, color=BLUE, ha="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1,
                                 shrinkA=2, shrinkB=2))
    axL.annotate("2.776", xy=(stats.t.ppf(0.975, 4), 0.055),
                 xytext=(3.72, 0.225),
                 fontsize=10.5, color=ORANGE, ha="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1,
                                 shrinkA=2, shrinkB=2))
    axL.text(3.12, 0.048, "오른쪽 꼬리 2.5%", fontsize=10, color=INK,
             ha="left", va="center")

    axL.set_xlim(-4.2, 4.2)
    axL.set_ylim(0, 0.44)
    axL.set_yticks([])
    axL.set_xlabel("표준화된 값", fontsize=10.5, color=INK)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.spines["bottom"].set_color(MUTED)
    axL.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    axL.legend(fontsize=10, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 1.02))
    axL.set_title("꼬리가 두꺼우면 2.5%를 남기는 자리가 밀린다",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 대가는 n 이 커지면 급히 사라진다 ---
    ns = np.arange(3, 121)
    ratio = stats.t.ppf(0.975, ns - 1) / z
    axR.plot(ns, ratio, color=BLUE, lw=2.2)
    axR.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (5, 4)))
    axR.plot([30, 30], [0.97, 1.16], color=GREEN, lw=1.4, ls=(0, (4, 3)))

    marks = [5, 10, 25, 30, 100]
    for m in marks:
        r = stats.t.ppf(0.975, m - 1) / z
        axR.plot([m], [r], marker="o", ms=6, color=BLUE, zorder=4)
    axR.annotate(f"$n=5$ : {stats.t.ppf(0.975, 4)/z:.3f}배",
                 xy=(5, stats.t.ppf(0.975, 4) / z), xytext=(17, 1.40),
                 fontsize=10.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1,
                                 shrinkA=2, shrinkB=4))
    axR.annotate(f"$n=10$ : {stats.t.ppf(0.975, 9)/z:.3f}배",
                 xy=(10, stats.t.ppf(0.975, 9) / z), xytext=(35, 1.27),
                 fontsize=10.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1,
                                 shrinkA=2, shrinkB=4))
    axR.annotate(f"$n=25$ : {stats.t.ppf(0.975, 24)/z:.3f}배",
                 xy=(25, stats.t.ppf(0.975, 24) / z), xytext=(52, 1.132),
                 fontsize=10.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1,
                                 shrinkA=2, shrinkB=4))
    axR.annotate(f"$n=100$ : {stats.t.ppf(0.975, 99)/z:.3f}배",
                 xy=(100, stats.t.ppf(0.975, 99) / z), xytext=(62, 1.075),
                 fontsize=10.5, color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1,
                                 shrinkA=2, shrinkB=4))
    axR.text(31.5, 1.185, r"관행적 문턱 $n=30$", fontsize=10.5, color=GREEN,
             ha="left", va="center")

    axR.set_xlim(2, 122)
    axR.set_ylim(0.97, 1.52)
    axR.set_xlabel("표본크기 $n$", fontsize=10.5, color=INK)
    axR.set_ylabel(r"$t_{0.025,\,n-1}\,/\,z_{0.025}$", fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title("임계값이 얼마나 더 큰가 — 대가는 빠르게 사라진다",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.4)
    save(fig, "t_vs_z_cost.png")


# === 그림 2. 너비 차이를 둘로 쪼갠다 (ci_mean_calc.md) ===========
def fig_width_decomposition():
    n, xbar = 25, 3.2
    s, sigma = 1.1, 1.0
    z = norm.ppf(0.975)
    t = stats.t.ppf(0.975, n - 1)

    m_z = z * sigma / math.sqrt(n)            # 0.3920
    m_mid = t * sigma / math.sqrt(n)          # 0.4128  임계값만 바꿈
    m_t = t * s / math.sqrt(n)                # 0.4541  산포까지 바꿈
    print(f"  moe z={m_z:.4f} mid={m_mid:.4f} t={m_t:.4f}")
    print(f"  임계값 기여 {m_mid - m_z:.4f}, 산포 기여 {m_t - m_mid:.4f}")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.6, 4.4), gridspec_kw={"width_ratios": [1.0, 1.05]}
    )

    # --- 왼쪽: 두 구간을 수직선 위에 ---
    rows = [
        ("z-구간", xbar - m_z, xbar + m_z, GREEN),
        ("t-구간", xbar - m_t, xbar + m_t, BLUE),
    ]
    for i, (lab, lo, hi, col) in enumerate(rows):
        y = 1 - i
        axL.plot([lo, hi], [y, y], color=col, lw=2.0)
        for e in (lo, hi):
            axL.plot([e, e], [y - 0.09, y + 0.09], color=col, lw=2.0)
        axL.plot([xbar], [y], marker="o", ms=6, color=col)
        axL.text(2.67, y, lab, fontsize=11, color=col, ha="left", va="center")
        axL.text(lo, y + 0.17, f"{lo:.4f}", fontsize=10, color=col,
                 ha="center", va="bottom")
        axL.text(hi, y + 0.17, f"{hi:.4f}", fontsize=10, color=col,
                 ha="center", va="bottom")
    axL.plot([xbar, xbar], [-0.22, 1.42], color=MUTED, lw=1.1,
             ls=(0, (5, 4)), zorder=0)
    axL.text(xbar, -0.42, r"$\bar x=3.2$", fontsize=10.5, color=MUTED,
             ha="center", va="center")
    axL.set_xlim(2.64, 3.76)
    axL.set_ylim(-0.62, 1.52)
    axL.set_yticks([])
    axL.set_xticks([2.8, 3.0, 3.2, 3.4, 3.6])
    axL.set_xlabel(r"$\mu$", fontsize=10.5, color=INK)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.spines["bottom"].set_color(MUTED)
    axL.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    axL.set_title(r"$n=25,\ \bar x=3.2,\ s=1.1,\ \sigma=1.0$",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 오차한계를 세 칸으로 ---
    bars = [
        ("z-구간\n$1.960\\times1.0/5$", m_z, GREEN),
        ("임계값만 교체\n$2.064\\times1.0/5$", m_mid, MUTED),
        ("t-구간\n$2.064\\times1.1/5$", m_t, BLUE),
    ]
    xs = np.arange(3)
    for i, (lab, v, col) in enumerate(bars):
        axR.bar(i, v, width=0.52, color=col, alpha=0.32, edgecolor=col, lw=1.6)
        axR.text(i, v + 0.006, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=11, color=col)
    axR.annotate("", xy=(0.74, m_mid), xytext=(0.26, m_z),
                 arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.6))
    axR.text(0.5, m_mid + 0.018, f"임계값 몫\n+{m_mid - m_z:.4f}",
             fontsize=10.5, color=PURPLE, ha="center", va="bottom",
             linespacing=1.5)
    axR.annotate("", xy=(1.74, m_t), xytext=(1.26, m_mid),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    axR.text(1.5, m_t + 0.018, f"산포 몫\n+{m_t - m_mid:.4f}",
             fontsize=10.5, color=ORANGE, ha="center", va="bottom",
             linespacing=1.5)

    axR.set_xticks(xs)
    axR.set_xticklabels([b[0] for b in bars], fontsize=10.5, color=INK)
    axR.set_ylim(0, 0.60)
    axR.set_ylabel("오차한계", fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title("넓어진 몫의 3분의 2는 임계값이 아니라 산포다",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.4)
    save(fig, "width_decomposition.png")


# === 비율 구간 네 가지 ===========================================
def ci_wald(k, n, z):
    p = k / n
    se = math.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def ci_wilson(k, n, z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ci_ac(k, n, z):
    nt = n + z * z
    pt = (k + 0.5 * z * z) / nt
    h = z * math.sqrt(pt * (1 - pt) / nt)
    return pt - h, pt + h


def ci_cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def coverage_curve(n, maker, ps):
    """자르지 않은 끝점 그대로 정확 포함확률을 계산한다."""
    ks = np.arange(n + 1)
    los = np.array([maker(k, n)[0] for k in ks])
    his = np.array([maker(k, n)[1] for k in ks])
    out = np.empty_like(ps)
    for i, p in enumerate(ps):
        ok = (los <= p) & (p <= his)
        out[i] = binom.pmf(ks[ok], n, p).sum()
    return out, los, his


# === 그림 3. Wald 가 무너지는 자리 (ci_p.md) =====================
def fig_wald_vs_wilson():
    n = 20
    z = norm.ppf(0.975)
    ks = np.arange(n + 1)
    phat = ks / n

    w_lo = np.array([ci_wald(k, n, z)[0] for k in ks])
    w_hi = np.array([ci_wald(k, n, z)[1] for k in ks])
    s_lo = np.array([ci_wilson(k, n, z)[0] for k in ks])
    s_hi = np.array([ci_wilson(k, n, z)[1] for k in ks])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.6))

    # --- 왼쪽: k 마다의 구간 ---
    axL.axhspan(0, 1, color="#F5F7F8", zorder=0)
    axL.axhline(0, color=MUTED, lw=1.0)
    axL.axhline(1, color=MUTED, lw=1.0)
    for k in ks:
        axL.plot([k - 0.16, k - 0.16], [w_lo[k], w_hi[k]], color=ORANGE,
                 lw=2.6, solid_capstyle="butt", zorder=3)
        axL.plot([k + 0.16, k + 0.16], [s_lo[k], s_hi[k]], color=BLUE,
                 lw=2.6, solid_capstyle="butt", zorder=3)
    axL.plot(ks, phat, color=INK, lw=1.0, ls=(0, (3, 3)), zorder=4)

    axL.plot([], [], color=ORANGE, lw=2.6, label="Wald")
    axL.plot([], [], color=BLUE, lw=2.6, label="Wilson")
    axL.plot([], [], color=INK, lw=1.0, ls=(0, (3, 3)), label=r"$\hat p=k/n$")
    axL.legend(fontsize=10, frameon=False, loc="upper left",
               bbox_to_anchor=(0.01, 1.0))

    axL.annotate(f"Wald: $k=0$ 이면 너비 0인 한 점",
                 xy=(-0.16, 0.0), xytext=(2.4, -0.13),
                 fontsize=10.5, color=ORANGE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                 shrinkA=2, shrinkB=4))
    axL.annotate(f"Wilson: 그래도 $[0,\\ {s_hi[0]:.3f}]$ 를 남긴다",
                 xy=(0.16, s_hi[0] * 0.5), xytext=(2.4, -0.26),
                 fontsize=10.5, color=BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2,
                                 shrinkA=2, shrinkB=4))
    axL.annotate("Wald 는 1을 넘어선다",
                 xy=(18.84, w_hi[19]), xytext=(11.3, 1.17),
                 fontsize=10.5, color=ORANGE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                 shrinkA=2, shrinkB=4))
    axL.set_xlim(-1.0, 21.6)
    axL.set_ylim(-0.36, 1.26)
    axL.set_xticks([0, 5, 10, 15, 20])
    axL.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xlabel("성공 횟수 $k$", fontsize=10.5, color=INK)
    axL.set_ylabel("$p$", fontsize=10.5, color=INK)
    clean_axis(axL)
    axL.set_title(r"$n=20$ 에서 관측값마다의 95% 구간",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 정확 포함확률 ---
    ps = np.linspace(0.001, 0.999, 1400)
    cov_w, _, _ = coverage_curve(n, lambda k, nn: ci_wald(k, nn, z), ps)
    cov_s, _, _ = coverage_curve(n, lambda k, nn: ci_wilson(k, nn, z), ps)

    axR.plot(ps, cov_w, color=ORANGE, lw=1.4, label="Wald")
    axR.plot(ps, cov_s, color=BLUE, lw=1.4, label="Wilson")
    axR.axhline(0.95, color=INK, lw=1.4, ls=(0, (5, 4)))
    axR.text(0.5, 0.962, "명목 95%", fontsize=10.5, color=INK,
             ha="center", va="bottom")

    c10 = float(np.interp(0.1, ps, cov_w))
    axR.plot([0.1, 0.1], [0.625, c10], color=ORANGE, lw=1.2, ls=(0, (3, 3)),
             zorder=2)
    axR.plot([0.1], [c10], marker="o", ms=6, color=ORANGE, zorder=5)
    axR.text(0.118, 0.645, f"$p=0.1$ 에서 Wald 는 {c10:.3f}",
             fontsize=10.5, color=ORANGE, ha="left", va="center")
    axR.legend(fontsize=10, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=2)
    axR.set_xlim(0, 1)
    axR.set_ylim(0.60, 1.03)
    axR.set_xlabel("참 비율 $p$", fontsize=10.5, color=INK)
    axR.set_ylabel("실제 포함확률", fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title("약속한 95%를 실제로 지키는가",
                  fontsize=12, color=INK, pad=10)

    print(f"  n=20 Wald coverage at p=0.1: {np.interp(0.1, ps, cov_w):.4f}")
    print(f"  n=20 Wilson coverage at p=0.1: {np.interp(0.1, ps, cov_s):.4f}")
    print(f"  n=20 Wald 최소 {cov_w[(ps > 0.02) & (ps < 0.98)].min():.4f}")

    fig.tight_layout(w_pad=2.6)
    save(fig, "wald_vs_wilson_n20.png")


# === 그림 4. 보장과 너비의 맞바꿈 (ci_prop_calc.md) ==============
def fig_four_methods_tradeoff():
    n, k = 50, 12
    z = norm.ppf(0.975)
    methods = [
        ("Wald", lambda kk, nn: ci_wald(kk, nn, z), ORANGE),
        ("Wilson", lambda kk, nn: ci_wilson(kk, nn, z), BLUE),
        ("Agresti–Coull", lambda kk, nn: ci_ac(kk, nn, z), GREEN),
        ("Clopper–Pearson", lambda kk, nn: ci_cp(kk, nn), PURPLE),
    ]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.8, 4.6), gridspec_kw={"width_ratios": [1.0, 1.05]}
    )

    # --- 왼쪽: k=12, n=50 의 네 구간 ---
    for i, (name, f, col) in enumerate(methods):
        lo, hi = f(k, n)
        y = len(methods) - 1 - i
        axL.plot([lo, hi], [y, y], color=col, lw=2.2)
        for e in (lo, hi):
            axL.plot([e, e], [y - 0.13, y + 0.13], color=col, lw=2.2)
        axL.text(0.008, y, name, fontsize=10.5, color=col, ha="left",
                 va="center")
        axL.text(lo, y + 0.17, f"{lo:.4f}", fontsize=10, color=col,
                 ha="center", va="bottom")
        axL.text(hi, y + 0.17, f"{hi:.4f}", fontsize=10, color=col,
                 ha="center", va="bottom")
        axL.text(hi + 0.012, y, f"너비 {hi - lo:.4f}", fontsize=10, color=col,
                 ha="left", va="center")
        print(f"  {name:16s} ({lo:.4f}, {hi:.4f})  너비 {hi - lo:.4f}")
    axL.plot([k / n, k / n], [-0.30, 3.40], color=MUTED, lw=1.1,
             ls=(0, (5, 4)), zorder=0)
    axL.text(k / n, 3.60, r"$\hat p=0.24$", fontsize=10.5, color=MUTED,
             ha="center", va="center")
    axL.set_xlim(0.0, 0.50)
    axL.set_ylim(-0.7, 3.95)
    axL.set_yticks([])
    axL.set_xticks([0.10, 0.20, 0.30, 0.40])
    axL.set_xlabel("$p$", fontsize=10.5, color=INK)
    axL.spines[["top", "right", "left"]].set_visible(False)
    axL.spines["bottom"].set_color(MUTED)
    axL.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    axL.set_title(r"$n=50,\ k=12$ 의 95% 구간 네 가지",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 최소 포함확률 대 평균 너비 ---
    ps = np.linspace(0.05, 0.95, 901)
    ks = np.arange(n + 1)
    pts = []
    for name, f, col in methods:
        cov, los, his = coverage_curve(n, f, ps)
        widths = np.clip(his, 0, 1) - np.clip(los, 0, 1)
        # p 를 균등하게 본 평균 너비
        mean_w = np.mean([np.dot(binom.pmf(ks, n, p), widths) for p in ps])
        mincov = cov.min()
        pts.append((name, mean_w, mincov, col))
        axR.plot([mean_w], [mincov], marker="o", ms=11, color=col, zorder=4)
        print(f"  {name:16s} 최소포함 {mincov:.4f}  평균너비 {mean_w:.4f}")

    offsets = {"Wald": (0.0012, 0.0, "left", "center"),
               "Wilson": (-0.0012, 0.0, "right", "center"),
               "Agresti–Coull": (0.0012, 0.0, "left", "center"),
               "Clopper–Pearson": (0.0, 0.007, "center", "bottom")}
    for name, mw, mc, col in pts:
        dx, dy, ha, va = offsets[name]
        axR.text(mw + dx, mc + dy, f"{name}\n({mc:.3f})", fontsize=10.5,
                 color=col, ha=ha, va=va, linespacing=1.5)

    axR.axhline(0.95, color=INK, lw=1.4, ls=(0, (5, 4)))
    axR.text(0.2142, 0.9525, "명목 95%", fontsize=10.5, color=INK,
             ha="left", va="bottom")

    axR.set_xlim(0.2135, 0.2500)
    axR.set_ylim(0.775, 0.982)
    axR.set_xlabel(r"평균 너비 ($n=50$, $0.05\leq p\leq0.95$)",
                   fontsize=10.5, color=INK)
    axR.set_ylabel(r"최소 포함확률 ($0.05\leq p\leq0.95$)",
                   fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title("오른쪽 아래로 갈수록 나쁘다 — 넓고도 못 지킨다면",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.6)
    save(fig, "four_methods_tradeoff.png")


if __name__ == "__main__":
    fig_t_vs_z_cost()
    fig_width_decomposition()
    fig_wald_vs_wilson()
    fig_four_methods_tradeoff()
