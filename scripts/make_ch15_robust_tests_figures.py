r"""15장 로버스트 분산 검정 쪽들의 개념 그림을 생성한다.

만드는 파일:
  ch15/robust_tests/img/levene_transform.png      절대편차 변환으로 분산 비교가 평균 비교가 된다
  ch15/robust_tests/img/levene_power_cost.png     정규 자료에서 로버스트 검정이 치르는 비용
  ch15/robust_tests/img/trimmed_size_disaster.png center='trimmed' 의 크기 팽창
  ch15/robust_tests/img/fk_outlier_immunity.png   순위·정규점수가 이상점을 막는다
  ch15/robust_tests/img/size_power_tradeoff.png   크기와 검정력을 함께 보아야 한다

실행:  python3 scripts/make_ch15_robust_tests_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch15/robust_tests/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 그림 1. 절대편차 변환 ===
def fig_levene_transform():
    groups = [np.array([10, 12, 14, 11, 13], float),
              np.array([20, 28, 22, 35, 25], float),
              np.array([15, 16, 14, 17, 15], float)]
    means = [g.mean() for g in groups]
    zs = [np.abs(g - m) for g, m in zip(groups, means)]
    zbar = [z.mean() for z in zs]
    W, p = stats.levene(*groups, center="mean")
    cols = [BLUE, ORANGE, GREEN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4))

    for i, (g, m, c) in enumerate(zip(groups, means, cols)):
        ax1.scatter(np.full(g.size, i + 1), g, s=70, color=c, alpha=0.85,
                    zorder=4, edgecolors="white", lw=0.8)
        ax1.hlines(m, i + 0.72, i + 1.28, color=c, lw=2.4)
        for v in g:
            ax1.plot([i + 1, i + 1], [m, v], color=c, lw=1.0, alpha=0.45,
                     zorder=2)
        ax1.text(i + 1, 38.0, f"평균 {m:.1f}\n표본분산 {g.var(ddof=1):.1f}",
                 fontsize=10, color=c, ha="center", va="top")
    ax1.set_xlim(0.45, 3.55)
    ax1.set_ylim(5, 40)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["집단 1", "집단 2", "집단 3"], fontsize=11, color=INK)
    ax1.set_ylabel("원자료 $X_{ij}$", fontsize=11, color=INK)
    clean(ax1)
    ax1.set_title("원자료 — 묻는 것은 산포의 차이", fontsize=12, color=INK,
                  pad=9)

    for i, (z, zb, c) in enumerate(zip(zs, zbar, cols)):
        ax2.scatter(np.full(z.size, i + 1), z, s=70, color=c, alpha=0.85,
                    zorder=4, edgecolors="white", lw=0.8)
        ax2.hlines(zb, i + 0.72, i + 1.28, color=c, lw=2.4)
        ax2.text(i + 1, 10.4, f"$\\bar{{Z}}_{i + 1}$ = {zb:.2f}", fontsize=11,
                 color=c, ha="center", va="top")
    ax2.set_xlim(0.45, 3.55)
    ax2.set_ylim(-0.4, 11)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["집단 1", "집단 2", "집단 3"], fontsize=11, color=INK)
    ax2.set_ylabel("$Z_{ij} = |X_{ij} - \\bar{X}_i|$", fontsize=11, color=INK)
    clean(ax2)
    ax2.text(0.60, 7.2, f"이 세 평균이 같은가?\n분산분석 $W$ = {W:.3f},  "
                        f"$p$ = {p:.4f}", fontsize=10.5, color=INK, va="top")
    ax2.set_title("절대편차 — 묻는 것이 평균의 차이로 바뀐다", fontsize=12,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "levene_transform.png")
    print(f"  평균 {means}, Zbar {[round(v, 3) for v in zbar]}, "
          f"W = {W:.4f}, p = {p:.4f}")


# === 그림 2. 정규 자료에서 로버스트 검정이 치르는 비용 ===
def fig_power_cost():
    scales = [1.00, 1.05, 1.10, 1.15, 1.20]
    f_p = [1.000, 0.628, 0.345, 0.166, 0.071]
    lev_p = [1.000, 0.652, 0.379, 0.198, 0.095]

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    ax.plot(scales, f_p, marker="o", ms=7, lw=2.2, color=BLUE,
            label="$F$ 검정 = 바틀렛 (정규성 가정)")
    ax.plot(scales, lev_p, marker="s", ms=7, lw=2.2, color=ORANGE,
            label="레빈 (중앙값 중심, 로버스트)")
    ax.axhline(0.05, color=RED, lw=1.3, ls=(0, (5, 3)))
    ax.text(1.008, 0.075, "$\\alpha = 0.05$", fontsize=10.5, color=RED)
    for s, a, b in zip(scales, f_p, lev_p):
        if s == 1.0:
            continue
        ax.text(s, a - 0.055, f"{a:.3f}", fontsize=9.5, color=BLUE,
                ha="center")
        ax.text(s, b + 0.035, f"{b:.3f}", fontsize=9.5, color=ORANGE,
                ha="center")
    ax.text(1.003, 0.34,
            "같은 자료, 같은 표본크기.\n레빈의 $p$ 값이 늘 조금 더 크다.\n"
            "이 간격이 로버스트성의 값이다.",
            fontsize=10.5, color=INK, va="top")
    ax.set_xlim(0.985, 1.215)
    ax.set_ylim(0, 1.06)
    ax.set_xticks(scales)
    ax.set_xlabel("두 번째 집단의 참 표준편차 $\\sigma_y$  ($n_1 = n_2 = 100$)",
                  fontsize=11, color=INK)
    ax.set_ylabel("$p$ 값", fontsize=11, color=INK)
    clean(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("정규 자료에서는 고전적 검정이 조금 더 예민하다",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "levene_power_cost.png")


# === 그림 3. center='trimmed' 의 크기 팽창 ===
def fig_trimmed():
    ns = [15, 25, 50, 100]
    rows = [("center='mean'", [0.060, 0.057, 0.045, 0.055], BLUE),
            ("center='median'  (브라운–포사이드)",
             [0.030, 0.036, 0.038, 0.051], GREEN),
            ("center='trimmed'  (10% 절사)",
             [0.120, 0.147, 0.161, 0.191], RED)]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(ns))
    width = 0.26
    for j, (label, vals, color) in enumerate(rows):
        off = (j - 1) * width
        bars = ax.bar(x + off, vals, width * 0.9, color=color, alpha=0.85,
                      label=label, edgecolor="white", lw=0.6)
        for b, v in zip(bars, vals):
            dy = 0.011 if 0.042 <= v <= 0.053 else 0.004
            ax.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.3f}",
                    ha="center", fontsize=9.5, color=color, fontweight="bold")
    ax.axhline(0.05, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax.annotate("명목 0.05", xy=(2.5, 0.05), xytext=(2.5, 0.105),
                fontsize=10.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {v}" for v in ns], fontsize=11, color=INK)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 0.225)
    ax.set_ylabel("경험적 제1종 오류율", fontsize=11, color=INK)
    clean(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_title("완전한 정규 자료 — 세 집단, 반복 4000회",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "trimmed_size_disaster.png")


# === 그림 4. 정규점수가 이상점을 막는다 ===
def fig_fk_outlier():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    for N, color in [(10, RED), (30, ORANGE), (60, BLUE)]:
        r = np.arange(1, N + 1)
        a = stats.norm.ppf((1 + r / (N + 1)) / 2)
        ax1.plot(r / N, a, marker="o", ms=4.5, lw=1.8, color=color,
                 label=f"N = {N}  (최대 점수 {a[-1]:.3f})")
    ax1.set_xlim(0, 1.06)
    ax1.set_ylim(0, 2.75)
    ax1.set_xlabel("전체에서의 상대 순위  (순위 / N)", fontsize=11, color=INK)
    ax1.set_ylabel("정규점수 $a_{ij}$", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper left")
    ax1.text(0.50, 0.52,
             "가장 큰 편차가 받는 점수에 상한이 있다.\n"
             "편차가 15 이든 15,000 이든 점수는 같다.",
             fontsize=10, color=INK, va="top")
    ax1.set_title("편차를 순위로, 다시 정규점수로", fontsize=12, color=INK,
                  pad=9)

    rng = np.random.default_rng(21)
    n = 12
    base = [rng.normal(0, 1, n) for _ in range(3)]
    sizes = np.linspace(0, 14, 40)
    curves = {"바틀렛": ([], RED), "레빈 (평균)": ([], ORANGE),
              "브라운–포사이드": ([], GREEN), "플리그너–킬린": ([], PURPLE)}
    for s in sizes:
        g = [base[0].copy(), base[1].copy(), base[2].copy()]
        g[0][0] = base[0][0] + s
        curves["바틀렛"][0].append(stats.bartlett(*g)[1])
        curves["레빈 (평균)"][0].append(stats.levene(*g, center="mean")[1])
        curves["브라운–포사이드"][0].append(
            stats.levene(*g, center="median")[1])
        curves["플리그너–킬린"][0].append(stats.fligner(*g)[1])

    for label, (vals, color) in curves.items():
        ax2.plot(sizes, vals, lw=2.2, color=color, label=label)
        print(f"  {label}: s=0 → p={vals[0]:.3f},  s=14 → p={vals[-1]:.4f}")
    ax2.axhline(0.05, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax2.text(13.8, 0.075, "$\\alpha = 0.05$", fontsize=10, color=INK,
             ha="right")
    ax2.set_xlim(0, 14.3)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("집단 1 의 한 관측값을 얼마나 밀어냈는가", fontsize=11,
                   color=INK)
    ax2.set_ylabel("$p$ 값", fontsize=11, color=INK)
    clean(ax2)
    ax2.legend(fontsize=10, frameon=False, loc="upper right")
    ax2.set_title("세 집단 모두 $\\mathcal{N}(0,1)$, $n = 12$ — 점 하나만 옮긴다",
                  fontsize=12, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "fk_outlier_immunity.png")


# === 그림 5. 크기와 검정력을 함께 보기 ===
def fig_size_power():
    tests = ["바틀렛", "레빈 (평균)", "브라운–포사이드", "플리그너–킬린"]
    cols = [RED, ORANGE, GREEN, PURPLE]
    normal = [(0.052, 0.692), (0.058, 0.628), (0.039, 0.560), (0.036, 0.497)]
    t5 = [(0.218, 0.693), (0.064, 0.508), (0.041, 0.440), (0.039, 0.406)]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.axvspan(0, 0.05, color=GREEN_F, alpha=0.35)
    ax.axvline(0.05, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax.text(0.004, 0.762, "초록 띠 = 타당한 영역 (실제 크기 ≤ 명목 0.05)",
            fontsize=10.5, color=GREEN, ha="left", va="center")

    for (name, c, a, b) in zip(tests, cols, normal, t5):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6,
                                    alpha=0.75))
        ax.plot(*a, marker="o", ms=10, color=c, zorder=5)
        ax.plot(*b, marker="D", ms=9, color=c, zorder=5,
                markerfacecolor="white", markeredgewidth=2)
        if name in ("바틀렛", "레빈 (평균)"):
            ax.text(a[0], a[1] + 0.022, name, fontsize=10.5, color=c,
                    ha="center")
        else:
            ax.text(a[0] - 0.005, a[1], name, fontsize=10.5, color=c,
                    ha="right", va="center")
    ax.text(0.218, 0.693 - 0.045, "$t_5$ 자료에서\n크기가 0.218", fontsize=10,
            color=RED, ha="center", va="top")

    ax.plot([], [], marker="o", ms=9, color=MUTED, lw=0, label="정규 자료")
    ax.plot([], [], marker="D", ms=8, color=MUTED, lw=0,
            markerfacecolor="white", markeredgewidth=2, label="$t_5$ 자료")
    ax.set_xlim(0, 0.245)
    ax.set_ylim(0.33, 0.78)
    ax.set_xlabel("실제 제1종 오류율 (크기)", fontsize=11, color=INK)
    ax.set_ylabel("분산비 3 을 탐지한 비율", fontsize=11, color=INK)
    clean(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")
    ax.set_title("검정력은 크기를 지킬 때만 의미가 있다 (집단 3개, $n_i = 20$)",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "size_power_tradeoff.png")


# === 그림 6. 중앙값 중심화의 기계적 이점 ===
def fig_center_shift():
    clean_g = np.array([12, 15, 14, 10, 13, 14, 12, 11], float)
    dirty = clean_g.copy()
    dirty[3] = 34.0          # 10 -> 34, 이상점 하나

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})

    for row, (data, tag) in enumerate([(clean_g, "원자료"),
                                       (dirty, "한 값만 34 로 바꾼 자료")]):
        y = 1 - row
        ax1.scatter(data, np.full(data.size, y), s=90, color=MUTED,
                    alpha=0.9, zorder=4, edgecolors="white", lw=0.8)
        m, med = data.mean(), np.median(data)
        ax1.plot([m], [y], marker="v", ms=13, color=ORANGE, zorder=6)
        ax1.plot([med], [y], marker="^", ms=13, color=GREEN, zorder=6)
        ax1.text(8.0, y + 0.30, tag, fontsize=11, color=INK)
        ax1.text(8.0, y + 0.17,
                 f"평균 {m:.2f}   중앙값 {med:.1f}", fontsize=10, color=INK)
    ax1.plot([], [], marker="v", ms=10, color=ORANGE, lw=0, label="평균")
    ax1.plot([], [], marker="^", ms=10, color=GREEN, lw=0, label="중앙값")
    ax1.set_xlim(7.5, 36)
    ax1.set_ylim(-0.6, 1.62)
    ax1.set_yticks([])
    ax1.set_xlabel("관측값", fontsize=11, color=INK)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.spines["bottom"].set_color(MUTED)
    ax1.tick_params(axis="x", labelsize=10, colors=INK)
    ax1.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax1.set_title("이상점 하나가 중심을 어디로 옮기는가", fontsize=12,
                  color=INK, pad=9)

    z_mean = np.abs(dirty - dirty.mean())
    z_med = np.abs(dirty - np.median(dirty))
    idx = np.arange(dirty.size)
    ax2.bar(idx - 0.19, z_mean, 0.36, color=ORANGE, alpha=0.85,
            label=f"평균 중심  ($\\bar{{Z}}$ = {z_mean.mean():.2f})")
    ax2.bar(idx + 0.19, z_med, 0.36, color=GREEN, alpha=0.85,
            label=f"중앙값 중심  ($\\bar{{Z}}$ = {z_med.mean():.2f})")
    ax2.set_xticks(idx)
    ax2.set_xticklabels([f"{v:.0f}" for v in dirty], fontsize=9.5, color=INK)
    ax2.set_xlabel("관측값", fontsize=11, color=INK)
    ax2.set_ylabel("절대편차 $Z_{ij}$", fontsize=11, color=INK)
    ax2.set_ylim(0, 23)
    clean(ax2)
    ax2.legend(fontsize=10, frameon=False, loc="upper left")
    ax2.set_title("이상점이 아닌 값들의 편차까지 부풀려진다", fontsize=12,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "median_centering_mechanism.png")
    print(f"  평균중심 Z = {np.round(z_mean, 2)}  (평균 {z_mean.mean():.3f})")
    print(f"  중앙값중심 Z = {np.round(z_med, 2)}  (평균 {z_med.mean():.3f})")

    g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], float)
    g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], float)
    g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31], float)
    for tag, a in [("원자료", g1), ("오염", dirty)]:
        wm = stats.levene(a, g2, g3, center="mean")
        wd = stats.levene(a, g2, g3, center="median")
        print(f"  {tag}: 평균중심 W={wm[0]:.4f} p={wm[1]:.4f} | "
              f"중앙값중심 W={wd[0]:.4f} p={wd[1]:.4f}")


# === 그림 7. 순위·정규점수 변환 ===
def fig_rank_scores():
    g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], float)
    g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], float)
    g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31], float)
    groups = [g1, g2, g3]
    cols = [BLUE, ORANGE, GREEN]
    z = np.concatenate([np.abs(g - np.median(g)) for g in groups])
    gid = np.concatenate([np.full(g.size, i) for i, g in enumerate(groups)])
    N = z.size
    r = stats.rankdata(z)
    a = stats.norm.ppf((1 + r / (N + 1)) / 2)
    X2, p = stats.fligner(g1, g2, g3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))

    rng = np.random.default_rng(4)
    jitter = rng.uniform(-0.12, 0.12, N)
    for i, c in enumerate(cols):
        m = gid == i
        ax1.scatter(z[m], gid[m] + jitter[m], s=80, color=c, alpha=0.85,
                    edgecolors="white", lw=0.8)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["집단 1", "집단 2", "집단 3"], fontsize=11, color=INK)
    ax1.set_xlabel("$Z_{ij} = |X_{ij} - \\tilde{X}_i|$", fontsize=11,
                   color=INK)
    ax1.set_xlim(-0.3, 4.2)
    ax1.set_ylim(-0.6, 2.6)
    clean(ax1)
    ax1.spines["left"].set_visible(False)
    ax1.set_title("1단계 — 중앙값으로부터의 절대편차", fontsize=12, color=INK,
                  pad=9)

    marks = ["o", "s", "^"]
    offs = [-0.55, 0.0, 0.55]
    for i, (c, mk, off) in enumerate(zip(cols, marks, offs)):
        m = gid == i
        ax2.scatter(r[m] + off, a[m], s=85, color=c, alpha=0.8, marker=mk,
                    edgecolors="white", lw=0.8, zorder=4,
                    label=f"집단 {i + 1}")
    uniq = np.unique(r)
    ax2.plot(uniq, stats.norm.ppf((1 + uniq / (N + 1)) / 2), color=MUTED,
             lw=1.3, alpha=0.8, zorder=2)
    a1, a2 = a[gid == 0].mean(), a[gid == 1].mean()
    ax2.hlines(a1, 0, N + 1, color=GREEN, lw=1.6, ls=(0, (5, 3)), alpha=0.9)
    ax2.hlines(a2, 0, N + 1, color=ORANGE, lw=1.6, ls=(0, (5, 3)), alpha=0.9)
    ax2.text(1.0, a1 + 0.05,
             f"$\\bar{{a}}_1 = \\bar{{a}}_3$ = {a1:.3f}", fontsize=10.5,
             color=GREEN)
    ax2.text(1.0, a2 + 0.05, f"$\\bar{{a}}_2$ = {a2:.3f}", fontsize=10.5,
             color=ORANGE)
    ax2.set_xlim(0, N + 1)
    ax2.set_ylim(0, 2.25)
    ax2.set_xlabel("전체 24개 편차 중의 순위 $R_{ij}$  (동점은 평균순위)",
                   fontsize=11, color=INK)
    ax2.set_ylabel("정규점수 $a_{ij}$", fontsize=11, color=INK)
    clean(ax2)
    ax2.legend(fontsize=10, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 0.88))
    ax2.text(12.0, 2.12,
             f"$X^2$ = {X2:.4f},  $p$ = {p:.4f}", fontsize=11, color=INK,
             ha="center")
    ax2.set_title("2–3단계 — 순위를 정규점수로", fontsize=12, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "fk_rank_scores.png")
    print(f"  X2 = {X2:.4f}, p = {p:.4f}, "
          f"abar = {[round(a[gid == i].mean(), 4) for i in range(3)]}")


# === 그림 8. n 이 커질 때 두 검정이 갈라진다 ===
def fig_size_vs_n():
    bart_n = [20, 30, 100]
    bart = [0.675, 0.732, 0.813]
    bf_n = [10, 20, 50, 100]
    bf = [0.0378, 0.0382, 0.0420, 0.0462]

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(bart_n, bart, marker="o", ms=8, lw=2.4, color=RED,
            label="바틀렛 — 모형 오설정")
    ax.plot(bf_n, bf, marker="s", ms=8, lw=2.4, color=GREEN,
            label="브라운–포사이드 — 유한표본 근사 오차")
    ax.axhline(0.05, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax.text(103, 0.075, "명목 0.05", fontsize=10.5, color=INK, ha="right")
    for x, v in zip(bart_n, bart):
        ax.text(x, v + 0.035, f"{v:.3f}", fontsize=10, color=RED,
                ha="center")
    for x, v in zip(bf_n, bf):
        ax.text(x, v - 0.055, f"{v:.4f}", fontsize=10, color=GREEN,
                ha="center")
    ax.annotate("표본을 키우면 더 나빠진다", xy=(60, 0.775), xytext=(24, 0.62),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.annotate("표본을 키우면 0.05 로 다가간다", xy=(75, 0.043),
                xytext=(34, 0.20), fontsize=11, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    ax.set_xlim(0, 110)
    ax.set_ylim(-0.02, 0.92)
    ax.set_xlabel("집단당 표본크기 $n$", fontsize=11, color=INK)
    ax.set_ylabel("제1종 오류율", fontsize=11, color=INK)
    clean(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("$\\text{Lognormal}(0,1)$ 자료, 세 집단 — 같은 $H_0$, 반대 방향",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "size_vs_n_divergence.png")


# === 그림 9. 대수정규에서 평균과 중앙값의 안정성 ===
def fig_lognormal_center():
    rng = np.random.default_rng(42)
    R, n = 4000, 30
    x = rng.lognormal(0, 1, size=(R, n))
    means = x.mean(axis=1)
    meds = np.median(x, axis=1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

    bins = np.linspace(0.5, 4.5, 90)
    ax1.hist(means, bins=bins, color=ORANGE, alpha=0.55,
             label=f"표본평균  (표준편차 {means.std(ddof=1):.3f})")
    ax1.hist(meds, bins=bins, color=GREEN, alpha=0.60,
             label=f"표본중앙값  (표준편차 {meds.std(ddof=1):.3f})")
    ax1.axvline(np.exp(0.5), color=ORANGE, lw=1.4, ls=(0, (5, 3)))
    ax1.axvline(1.0, color=GREEN, lw=1.4, ls=(0, (5, 3)))
    ax1.text(np.exp(0.5) + 0.06, 330, "모평균 1.649", fontsize=9.5,
             color=ORANGE)
    ax1.text(1.04, 430, "모중앙값 1", fontsize=9.5, color=GREEN)
    ax1.set_xlim(0.5, 4.2)
    ax1.set_ylim(0, 500)
    ax1.set_xlabel("집단 중심의 추정값", fontsize=11, color=INK)
    ax1.set_ylabel("빈도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper right")
    ax1.set_title("$\\text{Lognormal}(0,1)$ 에서 $n = 30$, 4000회",
                  fontsize=12, color=INK, pad=9)

    names = ["바틀렛", "레빈 (평균)", "플리그너–킬린", "브라운–포사이드"]
    vals = [0.7320, 0.2605, 0.1140, 0.0295]
    cols = [RED, RED, ORANGE, GREEN]
    ys = np.arange(4)
    ax2.barh(ys, vals, 0.55, color=cols, alpha=0.85)
    for y, v in zip(ys, vals):
        ax2.text(v + 0.012, y, f"{v:.4f}", va="center", fontsize=10.5,
                 color=INK)
    ax2.axvline(0.05, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax2.text(0.05, -0.44, "명목 0.05", fontsize=10, color=INK, ha="center")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(names, fontsize=10.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_ylim(3.8, -0.6)
    ax2.set_xlim(0, 0.92)
    ax2.set_xlabel("거짓 양성률", fontsize=11, color=INK)
    clean(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("같은 자료에서 네 검정의 성적", fontsize=12, color=INK,
                  pad=9)

    fig.tight_layout()
    save(fig, "lognormal_center_stability.png")
    print(f"  평균 sd {means.std(ddof=1):.4f}, 중앙값 sd {meds.std(ddof=1):.4f}, "
          f"비 {means.std(ddof=1) / meds.std(ddof=1):.2f}")


# === 그림 10. 하나의 자료, 네 개의 렌즈 ===
def fig_four_lenses():
    rng = np.random.default_rng(9)
    n = 12
    g = [rng.normal(0, 1, n), rng.normal(0, 1, n), rng.normal(0, 1, n)]
    g[0][0] = 4.6                     # 집단 1 에 이상점 하나
    cols = [BLUE, ORANGE, GREEN]
    gid = np.repeat([0, 1, 2], n)
    allx = np.concatenate(g)

    sq = np.concatenate([(a - a.mean()) ** 2 for a in g])
    lev = np.concatenate([np.abs(a - a.mean()) for a in g])
    bf = np.concatenate([np.abs(a - np.median(a)) for a in g])
    N = bf.size
    r = stats.rankdata(bf)
    fk = stats.norm.ppf((1 + r / (N + 1)) / 2)

    p_bart = stats.bartlett(*g)[1]
    p_lev = stats.levene(*g, center="mean")[1]
    p_bf = stats.levene(*g, center="median")[1]
    p_fk = stats.fligner(*g)[1]

    rows = [
        ("원자료 $X_{ij}$", allx, None, None),
        ("바틀렛:  $(X_{ij}-\\bar{X}_i)^2$", sq, None, f"$p$ = {p_bart:.4f}"),
        ("레빈:  $|X_{ij}-\\bar{X}_i|$", lev, None, f"$p$ = {p_lev:.4f}"),
        ("브라운–포사이드:  $|X_{ij}-\\tilde{X}_i|$", bf, None,
         f"$p$ = {p_bf:.4f}"),
        ("플리그너–킬린:  편차 순위의 정규점수", fk, None,
         f"$p$ = {p_fk:.4f}"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(9.6, 7.4))
    jit = rng.uniform(-0.16, 0.16, N)
    for ax, (label, vals, flat, ptxt) in zip(axes, rows):
        for i, c in enumerate(cols):
            m = gid == i
            ax.scatter(vals[m], jit[m] + (0 if flat else 0), s=52,
                       color=(flat or c), alpha=0.8, edgecolors="white",
                       lw=0.6, zorder=4)
            if flat is None:
                ax.plot([vals[m].mean()], [0], marker="|", ms=26, mew=2.6,
                        color=c, zorder=5)
        ax.set_yticks([])
        ax.set_ylim(-0.45, 0.45)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
        ax.set_title(label, fontsize=11.5, color=INK, loc="left", pad=6)
        if ptxt:
            ax.text(1.0, 1.02, ptxt, transform=ax.transAxes, fontsize=11,
                    color=INK, ha="right", va="bottom")

    axes[0].text(1.0, 1.02, "세 집단 모두 $\\mathcal{N}(0,1)$, $n = 12$"
                            " — 집단 1 에만 이상점 4.6",
                 transform=axes[0].transAxes, fontsize=10, color=INK,
                 ha="right", va="bottom")
    for i, c in enumerate(cols):
        axes[0].scatter([], [], s=52, color=c, label=f"집단 {i + 1}")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c,
                          label=f"집단 {i + 1}") for i, c in enumerate(cols)]
    handles.append(plt.Line2D([], [], marker="|", ls="", color=INK, ms=12,
                              mew=2.2, label="집단평균"))
    axes[4].legend(handles=handles, fontsize=10, frameon=False, ncol=4,
                   loc="upper center", bbox_to_anchor=(0.5, -0.45))
    fig.subplots_adjust(hspace=0.95)
    save(fig, "four_lenses.png")
    print(f"  p: 바틀렛 {p_bart:.4f}, 레빈 {p_lev:.4f}, BF {p_bf:.4f}, "
          f"FK {p_fk:.4f}")
    print(f"  집단별 표본분산 {[round(a.var(ddof=1), 3) for a in g]}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_four_lenses()
    fig_levene_transform()
    fig_power_cost()
    fig_trimmed()
    fig_fk_outlier()
    fig_size_power()
    fig_center_shift()
    fig_rank_scores()
    fig_size_vs_n()
    fig_lognormal_center()
