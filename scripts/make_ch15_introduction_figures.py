r"""15장 도입부 세 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch15/introduction/img/mean_vs_variance_quality.png  평균이 같아도 분산이 다르면 불량률이 다르다
  ch15/introduction/img/size_by_distribution.png      분포가 바뀌면 검정마다 실제 크기가 갈린다
  ch15/introduction/img/chisq_reference_wrong.png     비정규 자료에서는 기준분포의 척도가 틀린다

실행:  python3 scripts/make_ch15_introduction_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
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

OUT = "docs/ch15/introduction/img/"


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


# === 그림 1. 평균은 같고 분산만 다른 두 충전기 ===
def fig_mean_vs_variance():
    lo, hi = 495.0, 505.0
    x = np.linspace(480, 520, 1200)
    a = stats.norm(500.1, 2.0)
    b = stats.norm(499.9, 6.0)
    out_a = a.cdf(lo) + (1 - a.cdf(hi))
    out_b = b.cdf(lo) + (1 - b.cdf(hi))

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True)

    for ax, dist, color, fill, title, out in [
        (axes[0], a, BLUE, BLUE_F, "기계 A", out_a),
        (axes[1], b, ORANGE, ORANGE_F, "기계 B", out_b),
    ]:
        y = dist.pdf(x)
        ax.plot(x, y, color=color, lw=2.0)
        inside = (x >= lo) & (x <= hi)
        ax.fill_between(x[inside], y[inside], color=fill, alpha=0.9)
        ax.fill_between(x, y, where=~inside, color=RED, alpha=0.55)
        for v in (lo, hi):
            ax.axvline(v, color=INK, lw=1.2, ls=(0, (5, 3)))
        ax.axvline(500, color=MUTED, lw=1.0, ls=":")
        bare(ax)
        ax.set_xlim(480, 520)
        ax.set_ylim(0, dist.pdf(dist.mean()) * 1.42)
        ax.text(0.015, 0.90,
                f"{title}:  평균 {dist.mean():.1f} g,  표준편차 {dist.std():.0f} g",
                transform=ax.transAxes, fontsize=11.5, color=color,
                fontweight="bold", va="top")
        ax.text(0.015, 0.68, f"규격을 벗어나는 비율 {out * 100:.2f}%",
                transform=ax.transAxes, fontsize=11, color=RED, va="top")

    top = axes[0].get_ylim()[1]
    axes[0].annotate("규격 하한 495 g", xy=(lo, top * 0.24),
                     xytext=(481.5, top * 0.30), fontsize=9.5, color=INK,
                     va="center",
                     arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    axes[0].annotate("규격 상한 505 g", xy=(hi, top * 0.24),
                     xytext=(506.5, top * 0.30), fontsize=9.5, color=INK,
                     va="center",
                     arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    axes[1].set_xlabel("상자 중량 (g)", fontsize=11, color=INK)
    axes[0].set_title("평균은 사실상 같지만 규격을 벗어나는 비율은 30배 이상 다르다",
                      fontsize=12.5, color=INK, pad=10)
    save(fig, "mean_vs_variance_quality.png")
    return out_a, out_b


# === 그림 2. 분포별 경험적 제1종 오류율 ===
def fig_size_by_distribution():
    tests = ["바틀렛", "레빈(평균)", "브라운–포사이드", "플리그너–킬린"]
    rows = [
        ("정규 N(0,1)", [0.045, 0.048, 0.039, 0.038], GREEN),
        ("대수정규(0, 0.8)", [0.620, 0.189, 0.039, 0.087], ORANGE),
        ("t 자유도 5", [0.267, 0.054, 0.042, 0.040], PURPLE),
    ]
    xpos = np.arange(len(tests))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for j, (label, vals, color) in enumerate(rows):
        off = (j - 1) * width
        bars = ax.bar(xpos + off, vals, width * 0.92, color=color,
                      alpha=0.88, label=label, edgecolor="white", lw=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.axhline(0.05, color=RED, lw=1.4, ls=(0, (5, 3)))
    ax.annotate("명목 수준 0.05", xy=(2.50, 0.05), xytext=(2.50, 0.26),
                fontsize=10.5, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.set_xticks(xpos)
    ax.set_xticklabels(tests, fontsize=11, color=INK)
    ax.set_ylabel("경험적 제1종 오류율", fontsize=11, color=INK)
    ax.set_ylim(0, 0.72)
    ax.set_xlim(-0.5, 3.5)
    clean(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    ax.set_title("$H_0$ 이 참인데 기각한 비율 — 집단 3개, 각 $n = 40$, 2000회 반복",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "size_by_distribution.png")


# === 그림 3. 기준분포의 척도가 틀린다 ===
def fig_reference_wrong():
    rng = np.random.default_rng(15)
    n, R = 25, 60000
    df = n - 1
    crit = stats.chi2.ppf(0.95, df)

    def stat(sample):
        return df * sample.var(axis=1, ddof=1)

    cases = []
    # 정규: 분산 1
    z = rng.normal(0, 1, size=(R, n))
    cases.append(("정규", stat(z), 0.0, BLUE))
    # t6 을 분산 1로 표준화 (Var = 6/4 = 1.5)
    t = rng.standard_t(6, size=(R, n)) / np.sqrt(1.5)
    cases.append(("t 자유도 6", stat(t), 3.0, ORANGE))
    # 지수분포: 분산 1
    e = rng.exponential(1.0, size=(R, n))
    cases.append(("지수", stat(e), 6.0, PURPLE))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.65, 1]})

    xs = np.linspace(0, 80, 700)
    ax1.fill_between(xs, stats.chi2.pdf(xs, df), color=BLUE_F, alpha=0.85)
    ax1.plot(xs, stats.chi2.pdf(xs, df), color=INK, lw=1.6,
             label="기준분포 $\\chi^2_{24}$")

    grid = np.linspace(0, 80, 260)
    rates = []
    for label, s, g2, color in cases:
        kde = stats.gaussian_kde(s[s < 150], bw_method=0.28)
        ax1.plot(grid, kde(grid), color=color, lw=2.0,
                 label=f"{label}  (초과첨도 {g2:.0f})")
        rates.append((label, np.mean(s > crit), g2, color))

    ax1.axvline(crit, color=RED, lw=1.3, ls=(0, (5, 3)))
    ax1.text(crit + 1.6, stats.chi2.pdf(df - 2, df) * 0.96,
             "상단 5% 임계값 36.4", fontsize=9.5, color=RED)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(0, 0.062)
    ax1.set_xlabel("$(n-1)S^2/\\sigma^2$", fontsize=11, color=INK)
    ax1.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=9.5, frameon=False, loc="upper right")
    ax1.set_title("$n = 25$, 모분산은 세 경우 모두 1로 맞추었다",
                  fontsize=12, color=INK, pad=9)

    names = [r[0] for r in rates]
    vals = [r[1] for r in rates]
    colors = [r[3] for r in rates]
    ax2.barh(np.arange(3), vals, 0.5, color=colors, alpha=0.88)
    for i, v in enumerate(vals):
        ax2.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10.5,
                 color=colors[i], fontweight="bold")
    ax2.axvline(0.05, color=RED, lw=1.4, ls=(0, (5, 3)))
    ax2.text(0.05, 2.52, "0.05", fontsize=10, color=RED, ha="center")
    ax2.set_yticks(np.arange(3))
    ax2.set_yticklabels(names, fontsize=10.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(vals) * 1.26)
    ax2.set_ylim(2.75, -0.6)
    ax2.set_xlabel("상단 단측 기각률", fontsize=11, color=INK)
    clean(ax2)
    ax2.set_title("명목 5% 검정의 실제 오류율", fontsize=12, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "chisq_reference_wrong.png")
    for label, r, g2, _ in rates:
        print(f"  {label}: 기각률 {r:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    oa, ob = fig_mean_vs_variance()
    print(f"  기계 A 규격이탈 {oa:.4f}, 기계 B {ob:.4f}")
    fig_size_by_distribution()
    fig_reference_wrong()
