r"""15장 F 검정 여섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch15/f_test/img/f_size_by_shape.png      모집단 모양이 바뀌면 F 검정의 실제 크기가 무너진다
  ch15/f_test/img/f_order_matters.png      자유도의 순서와 역수 성질
  ch15/f_test/img/f_rejection_region.png   1 을 중심으로 대칭이 아닌 기각역
  ch15/f_test/img/shapiro_power.png        작은 표본에서 정규성 검정은 무력하다
  ch15/f_test/img/f_ratio_dullness.png     분산이 44% 달라도 F 검정은 기각하지 못한다
  ch15/f_test/img/f_required_n.png         검정력 0.8 에 필요한 표본크기

실행:  python3 scripts/make_ch15_f_test_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch15/f_test/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def bare(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# === 그림 1. 모집단 모양과 F 검정의 실제 크기 ===
def fig_size_by_shape():
    """본문 재현 코드와 같은 시드·같은 순서로 돌려 같은 숫자를 얻는다."""
    rng = np.random.default_rng(1)
    n, R, alpha = 20, 20000, 0.05
    df = n - 1
    cases = [
        ("정규", 0.0, lambda: rng.normal(0, 1, n), MUTED),
        ("t 자유도 10", 1.0, lambda: rng.standard_t(10, n), None),
        ("t 자유도 5", 6.0, lambda: rng.standard_t(5, n), ORANGE),
        ("지수", 6.0, lambda: rng.exponential(1, n), RED),
        ("균등", -1.2, lambda: rng.uniform(0, 1, n), GREEN),
        ("라플라스", 3.0, lambda: rng.laplace(0, 1, n), None),
    ]
    sizes, keep = [], {}
    for name, g2, gen, color in cases:
        rej, fs = 0, []
        for _ in range(R):
            F = gen().var(ddof=1) / gen().var(ddof=1)
            p = 2 * min(stats.f.cdf(F, df, df), stats.f.sf(F, df, df))
            rej += (p < alpha)
            fs.append(F)
        sizes.append((name, g2, rej / R))
        if color is not None:
            keep[name] = (np.array(fs), color)

    lo = stats.f.ppf(0.025, df, df)
    hi = stats.f.ppf(0.975, df, df)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})

    x = np.linspace(0.02, 4.2, 800)
    ax1.fill_between(x, stats.f.pdf(x, df, df), color=BLUE_F, alpha=0.9)
    ax1.plot(x, stats.f.pdf(x, df, df), color=INK, lw=1.8,
             label="이론 귀무분포 $F_{19,19}$")
    for name, (fs, color) in keep.items():
        kde = stats.gaussian_kde(fs[fs < 12], bw_method=0.25)
        ax1.plot(x, kde(x), color=color, lw=2.0, label=f"{name} 자료의 실제 분포")
    for v in (lo, hi):
        ax1.vlines(v, 0, 1.44, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax1.text(lo, 1.47, f"{lo:.3f}", fontsize=9.5, color=INK, ha="center")
    ax1.text(hi, 1.47, f"{hi:.3f}", fontsize=9.5, color=INK, ha="center")
    ax1.text((lo + hi) / 2, 1.50, "명목 5% 임계값", fontsize=10, color=INK,
             ha="center")
    ax1.set_xlim(0, 4.2)
    ax1.set_ylim(0, 1.62)
    ax1.set_xlabel("$F = S_1^2/S_2^2$", fontsize=11, color=INK)
    ax1.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=9.5, frameon=False, loc=(0.53, 0.45))
    ax1.set_title("두 표본을 같은 분포에서 뽑았다 ($n_1 = n_2 = 20$)",
                  fontsize=12, color=INK, pad=9)

    order = sorted(sizes, key=lambda r: r[2])
    ys = np.arange(len(order))
    cols = [RED if s > 0.08 else (GREEN if s < 0.03 else MUTED)
            for _, _, s in order]
    ax2.barh(ys, [s for _, _, s in order], 0.55, color=cols, alpha=0.85)
    for y, (name, g2, s) in zip(ys, order):
        ax2.text(s + 0.006, y, f"{s:.4f}", va="center", fontsize=10,
                 color=INK)
    ax2.axvline(0.05, color=BLUE, lw=1.5, ls=(0, (5, 3)))
    ax2.text(0.05, len(order) - 0.35, "명목 0.05", fontsize=10, color=BLUE,
             ha="center")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([f"{name}  (초과첨도 {g2:g})" for name, g2, _ in order],
                        fontsize=10, color=INK)
    ax2.set_xlim(0, 0.33)
    ax2.set_ylim(-0.7, len(order) - 0.05)
    ax2.set_xlabel("실제 제1종 오류율", fontsize=11, color=INK)
    clean(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("명목 5% 검정이 실제로 기각하는 비율", fontsize=12,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "f_size_by_shape.png")
    for name, g2, s in sizes:
        print(f"  {name}: {s:.4f}")
    print(f"  임계값 {lo:.4f}, {hi:.4f}")


# === 그림 2. 자유도의 순서와 역수 성질 ===
def fig_order_matters():
    a1, a2 = 5, 30
    x = np.linspace(0.005, 5.0, 1000)
    hi_a = stats.f.ppf(0.95, a1, a2)
    hi_b = stats.f.ppf(0.95, a2, a1)
    lo_a = stats.f.ppf(0.05, a1, a2)

    d1, d2 = 15, 20
    xb = np.linspace(0.005, 3.6, 900)
    mean_b = d2 / (d2 - 2)
    mode_b = (d1 - 2) / d1 * d2 / (d2 + 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax1.fill_between(x, stats.f.pdf(x, a1, a2), color=BLUE_F, alpha=0.8)
    ax1.plot(x, stats.f.pdf(x, a1, a2), color=BLUE, lw=2.3,
             label="$F_{5,30}$  (분자 자유도 5)")
    ax1.plot(x, stats.f.pdf(x, a2, a1), color=ORANGE, lw=2.3,
             label="$F_{30,5}$  (분자 자유도 30)")
    ax1.vlines(hi_a, 0, 0.55, color=BLUE, lw=1.4, ls=(0, (4, 3)))
    ax1.vlines(hi_b, 0, 0.55, color=ORANGE, lw=1.4, ls=(0, (4, 3)))
    ax1.text(hi_a, 0.58, f"상단 5%\n{hi_a:.3f}", fontsize=9.5, color=BLUE,
             ha="center")
    ax1.text(hi_b, 0.58, f"상단 5%\n{hi_b:.3f}", fontsize=9.5, color=ORANGE,
             ha="center")
    ax1.text(2.55, 0.34,
             f"역수 성질\n$F_{{0.05,\\,5,\\,30}} = 1/{hi_b:.3f} = {lo_a:.3f}$",
             fontsize=10, color=INK, va="top")
    ax1.set_xlim(0, 5.0)
    ax1.set_ylim(0, 0.95)
    ax1.set_xlabel("$F$", fontsize=11, color=INK)
    ax1.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax1.set_title("같은 두 숫자, 순서만 바꿨다", fontsize=12, color=INK, pad=9)

    y = stats.f.pdf(xb, d1, d2)
    ax2.fill_between(xb, y, color=BLUE_F, alpha=0.8)
    ax2.plot(xb, y, color=BLUE, lw=2.3)
    ax2.vlines(mode_b, 0, stats.f.pdf(mode_b, d1, d2), color=PURPLE, lw=1.6)
    ax2.vlines(1.0, 0, 0.90, color=MUTED, lw=1.2, ls=":")
    ax2.vlines(mean_b, 0, 0.78, color=RED, lw=1.6, ls=(0, (4, 3)))
    ax2.annotate(f"최빈값 {mode_b:.3f}",
                 xy=(mode_b, stats.f.pdf(mode_b, d1, d2) * 0.55),
                 xytext=(0.08, 0.72), fontsize=10.5, color=PURPLE,
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax2.text(1.0, 0.93, "1", fontsize=10.5, color=MUTED, ha="center")
    ax2.annotate(f"평균 {mean_b:.3f}", xy=(mean_b, 0.78), xytext=(1.75, 0.86),
                 fontsize=10.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax2.text(1.55, 0.50, "최빈값 < 1 < 평균\n오른쪽으로 치우쳐 있기 때문이다",
             fontsize=10, color=INK, va="top")
    ax2.set_xlim(0, 3.6)
    ax2.set_ylim(0, 1.03)
    ax2.set_xlabel("$F$", fontsize=11, color=INK)
    ax2.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax2)
    ax2.set_title("$F_{15,20}$ — 귀무가설 아래에서도 평균은 1 이 아니다",
                  fontsize=12, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "f_order_matters.png")
    print(f"  F(5,30) 상단5% {hi_a:.4f}, F(30,5) 상단5% {hi_b:.4f}, "
          f"F(5,30) 하단5% {lo_a:.4f}")
    print(f"  F(15,20) 평균 {mean_b:.4f}, 최빈값 {mode_b:.4f}")


# === 그림 3. 1 을 중심으로 대칭이 아닌 기각역 ===
def fig_rejection_region():
    d1, d2 = 14, 19
    lo = stats.f.ppf(0.025, d1, d2)
    hi = stats.f.ppf(0.975, d1, d2)
    obs = 25 / 16
    p = 2 * min(stats.f.cdf(obs, d1, d2), stats.f.sf(obs, d1, d2))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})

    x = np.linspace(0.01, 4.4, 900)
    y = stats.f.pdf(x, d1, d2)
    mid = (x >= lo) & (x <= hi)
    ax1.fill_between(x[mid], y[mid], color=BLUE_F, alpha=0.95)
    ax1.fill_between(x, y, where=x <= lo, color=RED, alpha=0.5)
    ax1.fill_between(x, y, where=x >= hi, color=RED, alpha=0.5)
    ax1.plot(x, y, color=INK, lw=1.8)
    for v in (lo, hi):
        ax1.vlines(v, 0, 0.80, color=RED, lw=1.4)
    ax1.vlines(1.0, 0, 0.92, color=MUTED, lw=1.0, ls=":")
    ax1.text(1.0, 0.94, "1", fontsize=10, color=MUTED, ha="center")
    ax1.text(lo, 0.83, f"{lo:.3f}", fontsize=10, color=RED, ha="center")
    ax1.text(hi, 0.83, f"{hi:.3f}", fontsize=10, color=RED, ha="center")
    ax1.vlines(obs, 0, 0.62, color=GREEN, lw=2.0, ls=(0, (4, 3)))
    ax1.annotate(f"관측 $F = {obs:.4f}$   ($p = {p:.3f}$)", xy=(obs, 0.62),
                 xytext=(2.05, 0.70), fontsize=10.5, color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
    ax1.text(2.95, 0.46,
             f"1 에서 왼쪽으로 {1 - lo:.3f}\n1 에서 오른쪽으로 {hi - 1:.3f}",
             fontsize=10, color=INK, va="top")
    ax1.set_xlim(0, 4.4)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("$F = s_1^2/s_2^2$   (자유도 14, 19)", fontsize=11,
                   color=INK)
    bare(ax1)
    ax1.set_title("기각역은 1 을 중심으로 대칭이 아니다", fontsize=12,
                  color=INK, pad=9)

    ns = np.arange(5, 101)
    need = np.array([stats.f.ppf(0.975, m - 1, m - 1) for m in ns])
    ax2.plot(ns, need, color=BLUE, lw=2.2)
    ax2.fill_between(ns, 1, need, color=BLUE_F, alpha=0.8)
    ax2.axhline(1, color=MUTED, lw=1.0, ls=":")
    for m in (5, 15, 30, 60, 100):
        v = stats.f.ppf(0.975, m - 1, m - 1)
        ax2.plot([m], [v], marker="o", ms=6, color=ORANGE, zorder=5)
        if m >= 100:
            ax2.text(m + 3.0, v + 1.15, f"n = {m}:\n{v:.2f} 배",
                     fontsize=9.5, color=ORANGE, ha="right")
        else:
            ax2.text(m + 2.5, v + 0.25, f"n = {m}:  {v:.2f} 배",
                     fontsize=9.5, color=ORANGE)
    ax2.set_xlim(0, 108)
    ax2.set_ylim(0, 10.5)
    ax2.set_xlabel("각 집단의 표본크기 $n$", fontsize=11, color=INK)
    ax2.set_ylabel("기각에 필요한 분산비", fontsize=11, color=INK)
    clean(ax2)
    ax2.set_title("5% 양측검정이 기각하려면 분산이 몇 배 달라야 하는가",
                  fontsize=11.5, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "f_rejection_region.png")
    print(f"  임계값 {lo:.4f}, {hi:.4f}, 관측 {obs:.4f}, p = {p:.4f}")
    for m in (5, 15, 30, 60, 100):
        print(f"  n={m}: {stats.f.ppf(0.975, m - 1, m - 1):.3f}")


# === 그림 4. 작은 표본에서 정규성 검정은 무력하다 ===
def fig_shapiro_power():
    rng = np.random.default_rng(3)
    ns = [8, 10, 15, 20, 30, 50, 80, 120]
    R = 3000
    pops = [
        ("지수분포", lambda m: rng.exponential(1, m), RED),
        ("대수정규(0, 0.5)", lambda m: rng.lognormal(0, 0.5, m), ORANGE),
        ("t 자유도 5", lambda m: rng.standard_t(5, m), PURPLE),
        ("정규 ($H_0$ 이 참)", lambda m: rng.normal(0, 1, m), MUTED),
    ]
    res = {}
    for name, gen, color in pops:
        ps = []
        for m in ns:
            hit = sum(stats.shapiro(gen(m))[1] < 0.05 for _ in range(R))
            ps.append(hit / R)
        res[name] = (ps, color)
        print(f"  {name}: " + ", ".join(f"n={m}:{v:.3f}"
                                        for m, v in zip(ns, ps)))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for name, (ps, color) in res.items():
        ax.plot(ns, ps, marker="o", ms=5, lw=2.0, color=color, label=name)
    ax.axhline(0.05, color=MUTED, lw=1.0, ls=":")
    ax.axhline(0.8, color=GREEN, lw=1.2, ls=(0, (5, 3)))
    ax.text(121, 0.81, "검정력 0.8", fontsize=10, color=GREEN, ha="right")
    ax.vlines(8, 0, 1.0, color=INK, lw=1.2, ls=(0, (2, 2)))
    ax.annotate("본문 예제의 $n = 8$", xy=(8, 0.62), xytext=(17, 0.70),
                fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.set_xlim(0, 126)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("표본크기 $n$", fontsize=11, color=INK)
    ax.set_ylabel("샤피로–윌크가 비정규성을 잡아낼 확률", fontsize=11,
                  color=INK)
    clean(ax)
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.set_title("정규성 검정의 검정력 ($\\alpha = 0.05$, 3000회 반복)",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "shapiro_power.png")


# === 그림 5. F 검정은 둔하다 ===
def fig_ratio_dullness():
    n, df = 100, 99
    lo = stats.f.ppf(0.025, df, df)
    hi = stats.f.ppf(0.975, df, df)
    obs = [(1.00, 1.0000, 1.000), (1.05, 0.9070, 0.628), (1.10, 0.8264, 0.345),
           (1.15, 0.7561, 0.166), (1.20, 0.6944, 0.071)]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.linspace(0.35, 2.2, 900)
    y = stats.f.pdf(x, df, df)
    mid = (x >= lo) & (x <= hi)
    ax.fill_between(x[mid], y[mid], color=BLUE_F, alpha=0.95)
    ax.fill_between(x, y, where=x <= lo, color=RED, alpha=0.45)
    ax.fill_between(x, y, where=x >= hi, color=RED, alpha=0.45)
    ax.plot(x, y, color=INK, lw=1.8)
    top = stats.f.pdf(1.0, df, df)
    for v in (lo, hi):
        ax.vlines(v, 0, top * 1.03, color=RED, lw=1.3, ls=(0, (5, 3)))
    ax.text(lo, top * 1.07, f"{lo:.3f}", fontsize=10, color=RED, ha="center")
    ax.text(hi, top * 1.07, f"{hi:.3f}", fontsize=10, color=RED, ha="center")

    for i, (sig, f_obs, p) in enumerate(obs):
        c = RED if p < 0.10 else MUTED
        yy = top * (1.22 + 0.135 * i)
        ax.plot([f_obs], [yy], marker="o", ms=7, color=c, zorder=5)
        ax.plot([f_obs, f_obs], [0, yy], color=c, lw=1.0, alpha=0.40)
        ax.text(f_obs + 0.035, yy,
                f"$\\sigma_y$ = {sig:.2f}   F = {f_obs:.4f}   p = {p:.3f}",
                fontsize=10, color=c, va="center", ha="left")

    ax.text(1.12, top * 0.30, "기각하지 못하는 영역", fontsize=10.5,
            color=BLUE, ha="center")
    ax.set_xlim(0.28, 2.2)
    ax.set_ylim(0, top * 2.00)
    ax.set_xlabel("$F = s_x^2/s_y^2$   ($n_1 = n_2 = 100$)", fontsize=11,
                  color=INK)
    bare(ax)
    ax.set_title("귀무분포 $F_{99,99}$ 와 본문 예제의 다섯 관측값",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "f_ratio_dullness.png")
    print(f"  임계값 {lo:.4f}, {hi:.4f}")


# === 그림 6. 검정력 0.8 에 필요한 표본크기 ===
def power_equal_n(ratio, n, alpha=0.05):
    """참 표준편차비 = ratio 일 때 양측 F 검정의 정확한 검정력."""
    df = n - 1
    lo = stats.f.ppf(alpha / 2, df, df)
    hi = stats.f.ppf(1 - alpha / 2, df, df)
    k = ratio ** 2                      # sigma2^2 / sigma1^2
    return stats.f.cdf(lo * k, df, df) + stats.f.sf(hi * k, df, df)


def fig_required_n():
    ratios = np.linspace(1.15, 3.0, 220)
    need = []
    for r in ratios:
        f = lambda m: power_equal_n(r, m) - 0.8
        need.append(brentq(f, 4.0, 4000.0))
    need = np.array(need)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})

    ax1.plot(ratios, need, color=BLUE, lw=2.4)
    ax1.fill_between(ratios, 0, need, color=BLUE_F, alpha=0.8)
    marks = [1.25, 1.5, 2.0, 2.5]
    for r in marks:
        m = brentq(lambda v: power_equal_n(r, v) - 0.8, 4.0, 4000.0)
        ax1.plot([r], [m], marker="o", ms=6.5, color=ORANGE, zorder=5)
        ax1.text(r + 0.04, m + 12, f"비 {r}배 →  n = {m:.0f}", fontsize=10,
                 color=ORANGE)
        print(f"  ratio {r}: n = {m:.1f}")
    ax1.set_xlim(1.1, 3.05)
    ax1.set_ylim(0, 330)
    ax1.set_xlabel("참 표준편차비 $\\sigma_2/\\sigma_1$", fontsize=11,
                   color=INK)
    ax1.set_ylabel("집단당 필요한 표본크기", fontsize=11, color=INK)
    clean(ax1)
    ax1.set_title("검정력 0.8 을 얻으려면 ($\\alpha = 0.05$, 정확 계산)",
                  fontsize=12, color=INK, pad=9)

    nn = np.arange(5, 121)
    for r, color, lab in [(2.0, RED, "표준편차 2배 (분산 4배)"),
                          (1.5, ORANGE, "표준편차 1.5배 (분산 2.25배)"),
                          (1.25, MUTED, "표준편차 1.25배 (분산 1.56배)")]:
        ax2.plot(nn, [power_equal_n(r, m) for m in nn], color=color, lw=2.2,
                 label=lab)
    ax2.axhline(0.8, color=GREEN, lw=1.2, ls=(0, (5, 3)))
    ax2.text(118, 0.82, "0.8", fontsize=10, color=GREEN, ha="right")
    p10 = power_equal_n(2.0, 10)
    ax2.plot([10], [p10], marker="o", ms=7, color=RED, zorder=5)
    ax2.annotate(f"본문 모의실험: $n = 10$, 검정력 {p10:.3f}",
                 xy=(10, p10), xytext=(26, 0.34), fontsize=10, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax2.set_xlim(0, 125)
    ax2.set_ylim(0, 1.04)
    ax2.set_xlabel("집단당 표본크기 $n$", fontsize=11, color=INK)
    ax2.set_ylabel("검정력", fontsize=11, color=INK)
    clean(ax2)
    ax2.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax2.set_title("표본크기에 따른 검정력", fontsize=12, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "f_required_n.png")
    print(f"  n=10, 비 2배 검정력 = {p10:.4f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_order_matters()
    fig_rejection_region()
    fig_ratio_dullness()
    fig_required_n()
    fig_shapiro_power()
    fig_size_by_shape()
