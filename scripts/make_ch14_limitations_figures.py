r"""14장 한계와 함정 절 네 쪽의 개념 그림을 만든다.

만드는 파일:
  ch14/limitations/img/power_paradox.png          검정력이 있는 구간과 필요한 구간
  ch14/limitations/img/practical_vs_statistical.png  기각했다고 망가진 것은 아니다
  ch14/limitations/img/test_choice_flow.png       검정 고르기 흐름도
  ch14/limitations/img/pvalue_vs_effectsize.png   n 이 바꾸는 것과 바꾸지 않는 것

실행:  python3 scripts/make_ch14_limitations_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch14/limitations/img/"
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
# 그림 1. 검정력이 생기는 구간과 정규성이 필요한 구간은 어긋나 있다
# ===================================================================
def fig_power_paradox():
    rng = np.random.default_rng(1401)
    ns = np.array([10, 20, 30, 50, 100, 200, 500, 1000, 2000, 5000])
    B, alpha = 3000, 0.05
    curves = {}
    for nu in (5, 10, 30):
        pw = []
        for n in ns:
            x = rng.standard_t(nu, size=(B, n))
            pw.append(np.mean(stats.shapiro(x, axis=1).pvalue < alpha))
        curves[nu] = np.array(pw)
        print(f"nu={nu:3d} 초과첨도={6/(nu-4):.3f} : "
              + " ".join(f"{v:.3f}" for v in curves[nu]))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.axvspan(9, 30, color="#FBE9E7", zorder=0)
    ax.axvspan(500, 6000, color="#E8F5E9", zorder=0)
    ax.text(16, 1.05, "정규성이\n가장 필요한 구간", fontsize=9.5, color=RED,
            ha="center", va="center", linespacing=1.5, zorder=5)
    ax.text(1700, 1.05, "CLT 가 대신 지켜 주는 구간", fontsize=9.5,
            color=GREEN, ha="center", va="center", zorder=5)

    ax.axhline(0.8, color=MUTED, lw=1.2, ls="--")
    ax.text(10.5, 0.82, "검정력 0.8", fontsize=9, color=INK, ha="left")
    for nu, col, mk in [(5, ORANGE, "o"), (10, PURPLE, "s"), (30, BLUE, "^")]:
        ax.plot(ns, curves[nu], mk + "-", color=col, lw=2, ms=5.5,
                label=f"$t_{{{nu}}}$  (초과첨도 {6/(nu-4):.2f})", zorder=4)

    ax.annotate(f"$n=50$ 에서 {curves[5][3]:.2f}", (50, curves[5][3]),
                textcoords="offset points", xytext=(4, -16), fontsize=9,
                color=ORANGE, ha="left")
    ax.annotate(f"$n=5000$ 에서도 {curves[30][-1]:.2f}", (5000, curves[30][-1]),
                textcoords="offset points", xytext=(-6, 10), fontsize=9,
                color=BLUE, ha="right")

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_xlim(9, 6000)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("Shapiro-Wilk 검정의 기각률", fontsize=10.5, color=INK)
    ax.set_title("정규성 검정이 힘을 내는 구간은 정규성이 덜 필요한 구간이다",
                 fontsize=12.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="center left", labelcolor=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "power_paradox.png")


# ===================================================================
# 그림 2. 기각했다는 것과 망가졌다는 것은 다르다
# ===================================================================
def fig_practical_vs_statistical():
    rng = np.random.default_rng(1402)
    ns = np.array([15, 20, 30, 50, 100, 200, 500, 1000])
    B, alpha = 6000, 0.05

    cases = [(0.2, "가벼운 이탈:  대수정규 $\\sigma = 0.2$"),
             (0.8, "심한 이탈:  대수정규 $\\sigma = 0.8$")]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    for ax, (s, title) in zip(axes, cases):
        mu_true = np.exp(s**2 / 2)
        skew_true = float(stats.lognorm.stats(s, moments="s"))
        rej, cov = [], []
        for n in ns:
            x = rng.lognormal(0.0, s, size=(B, n))
            rej.append(np.mean(stats.shapiro(x, axis=1).pvalue < alpha))
            xbar = x.mean(axis=1)
            se = x.std(axis=1, ddof=1) / np.sqrt(n)
            t = stats.t.ppf(0.975, n - 1)
            cov.append(np.mean(np.abs(xbar - mu_true) < t * se))
        rej, cov = np.array(rej), np.array(cov)
        print(f"sigma={s} 왜도={skew_true:.3f}")
        for n, r, c in zip(ns, rej, cov):
            print(f"   n={n:5d}  기각률={r:.3f}  95% t 구간 포함률={c:.3f}")

        ax.axhline(0.95, color=MUTED, lw=1.2, ls="--",
                   label="약속한 포함률 0.95")
        ax.plot(ns, rej, "o-", color=ORANGE, lw=2, ms=5.5,
                label="Shapiro-Wilk 가 정규성을 기각한 비율")
        ax.plot(ns, cov, "s-", color=BLUE, lw=2, ms=5.5,
                label="95% $t$ 신뢰구간의 실제 포함률")
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(v) for v in ns])
        ax.minorticks_off()
        ax.set_xlim(13, 1300)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
        ax.set_title(f"{title}  (왜도 {skew_true:.2f})", fontsize=11.5,
                     color=INK)
        ax.annotate(f"{cov[-1]:.3f}", (ns[-1], cov[-1]),
                    textcoords="offset points", xytext=(-4, -15), fontsize=9,
                    color=BLUE, ha="right")
        clean(ax)
    axes[0].set_ylabel("비율", fontsize=10.5, color=INK)
    axes[0].legend(fontsize=9.5, frameon=False, loc="lower right",
                   labelcolor=INK)

    fig.suptitle("같은 자료에서 정규성 검정의 기각률과 $t$ 구간의 포함률",
                 fontsize=12.5, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "practical_vs_statistical.png")


# ===================================================================
# 그림 3. 어떤 검정을 고를 것인가 — 흐름도
# ===================================================================
def _box(ax, x, y, w, h, text, face, edge, fs=10.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=face, edgecolor=edge, lw=1.6,
                                zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            linespacing=1.55, zorder=3, fontweight=weight)


def _arrow(ax, p0, p1, color=MUTED):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.5, zorder=1,
                                 shrinkA=2, shrinkB=2))


def fig_test_choice_flow():
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    ax.set_xlim(-2.3, 12)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    _box(ax, 6.0, 6.85, 5.6, 0.72, "정규성을 확인해야 한다",
         "#ECEFF1", INK, fs=12, weight="bold")

    # --- 1단계: 표본크기 ---
    ys = 5.55
    _box(ax, 2.0, ys, 3.2, 0.78, "$n < 30$", "#FBE9E7", RED, fs=12)
    _box(ax, 6.0, ys, 3.2, 0.78, "$30 \\leq n \\leq 5000$", BLUE_F, BLUE,
         fs=12)
    _box(ax, 10.0, ys, 3.2, 0.78, "$n > 5000$", "#E8F5E9", GREEN, fs=12)
    for x in (2.0, 6.0, 10.0):
        _arrow(ax, (6.0, 6.49), (x, ys + 0.39))
    ax.text(-2.2, ys, "1단계\n표본크기를 본다", fontsize=10, color=MUTED,
            ha="left", va="center", linespacing=1.5)

    # --- 2단계: 의심되는 이탈 (가운데 줄기만 갈라진다) ---
    yq = 4.05
    _box(ax, 3.6, yq, 2.6, 0.72, "무엇이 의심되는지\n모른다", "white", MUTED,
         fs=10)
    _box(ax, 6.3, yq, 2.4, 0.72, "두꺼운 꼬리\n(금융 자료 등)", "white", MUTED,
         fs=10)
    _box(ax, 9.0, yq, 2.4, 0.72, "치우침 · 첨도", "white", MUTED, fs=10)
    for x in (3.6, 6.3, 9.0):
        _arrow(ax, (6.0, ys - 0.39), (x, yq + 0.36), color=BLUE)
    ax.text(-2.2, yq, "2단계\n무엇이 의심되는가", fontsize=10, color=MUTED,
            ha="left", va="center", linespacing=1.5)

    # --- 결론 ---
    yr = 2.45
    _box(ax, 2.0, yr, 3.2, 1.30,
         "Shapiro-Wilk\n+ 반드시 Q-Q 그림\n검정력이 낮아 기각 못 해도\n정규성의 증거가 아니다",
         "#FBE9E7", RED, fs=9.5)
    _box(ax, 6.0, yr, 4.4, 1.30,
         "무엇이든 모르면  Shapiro-Wilk\n꼬리가 걱정이면  Anderson-Darling\n"
         "적률이 궁금하면  D'Agostino $K^2$",
         BLUE_F, BLUE, fs=9.5)
    _box(ax, 10.2, yr, 3.1, 1.30,
         "Anderson-Darling\n또는 Jarque-Bera\n$p$ 값 대신 $g_1$, $g_2$ 를\n효과 크기로 보고한다",
         "#E8F5E9", GREEN, fs=9.5)
    _arrow(ax, (2.0, ys - 0.39), (2.0, yr + 0.65), color=RED)
    for x in (3.6, 6.3, 9.0):
        _arrow(ax, (x, yq - 0.36), (6.0 if x != 9.0 else 6.0, yr + 0.65),
               color=BLUE)
    _arrow(ax, (10.0, ys - 0.39), (10.2, yr + 0.65), color=GREEN)

    # --- 바닥 띠 ---
    _box(ax, 6.0, 0.95, 10.6, 0.80,
         "어느 경로로 가든 Q-Q 그림을 함께 본다.  검정은 “아니다”만 말하고, 그림이 “어떻게 아닌지”를 말한다.",
         "#ECEFF1", INK, fs=11)
    _arrow(ax, (6.0, yr - 0.65), (6.0, 1.37), color=INK)

    save(fig, "test_choice_flow.png")


# ===================================================================
# 그림 4. n 이 바꾸는 것과 바꾸지 않는 것
# ===================================================================
def fig_pvalue_vs_effectsize():
    rng = np.random.default_rng(1404)
    ns = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
    B = 3000
    nu = 30
    true_g2 = 6.0 / (nu - 4)

    p_med, p_lo, p_hi = [], [], []
    g_med, g_lo, g_hi = [], [], []
    for n in ns:
        x = rng.standard_t(nu, size=(B, n))
        p = stats.shapiro(x, axis=1).pvalue
        g = stats.kurtosis(x, axis=1, bias=False)
        p_med.append(np.median(p))
        p_lo.append(np.quantile(p, 0.25))
        p_hi.append(np.quantile(p, 0.75))
        g_med.append(np.median(g))
        g_lo.append(np.quantile(g, 0.25))
        g_hi.append(np.quantile(g, 0.75))
        print(f"n={n:5d}  p 중앙값={p_med[-1]:.2e}  "
              f"g2 중앙값={g_med[-1]:+.3f}  (참값 {true_g2:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    ax = axes[0]
    ax.fill_between(ns, p_lo, p_hi, color=ORANGE_F, alpha=0.8,
                    label="가운데 50% 범위")
    ax.plot(ns, p_med, "o-", color=ORANGE, lw=2, ms=5.5, label="중앙값")
    ax.axhline(0.05, color=RED, lw=1.4, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_ylim(8e-4, 3)
    ax.set_xlim(18, 6000)
    ax.text(19, 0.062, "유의수준 0.05", fontsize=9, color=RED, ha="left")
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("Shapiro-Wilk 의 $p$ 값 (로그 눈금)", fontsize=10.5,
                  color=INK)
    ax.set_title("$p$ 값은 $n$ 과 함께 무너진다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="lower left", labelcolor=INK)
    clean(ax)

    ax = axes[1]
    ax.fill_between(ns, g_lo, g_hi, color=BLUE_F, alpha=0.9,
                    label="가운데 50% 범위")
    ax.plot(ns, g_med, "s-", color=BLUE, lw=2, ms=5.5, label="중앙값")
    ax.axhline(true_g2, color=INK, lw=1.4, ls="--")
    ax.text(5800, true_g2 + 0.04, f"모집단의 참값 {true_g2:.3f}", fontsize=9.5,
            color=INK, ha="right", va="bottom")
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_xlim(18, 6000)
    ax.set_ylim(-0.75, 1.15)
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("표본 초과첨도 $g_2$", fontsize=10.5, color=INK)
    ax.set_title("이탈의 크기는 참값에서 멈춘다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)

    fig.suptitle(f"모집단을 $t_{{{nu}}}$ 로 고정하고 $n$ 만 키웠다 (각 {B}회 반복)",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "pvalue_vs_effectsize.png")


if __name__ == "__main__":
    fig_power_paradox()
    fig_practical_vs_statistical()
    fig_test_choice_flow()
    fig_pvalue_vs_effectsize()
