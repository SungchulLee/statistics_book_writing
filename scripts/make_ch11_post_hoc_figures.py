r"""11장 사후비교 다섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch11/post_hoc/img/posthoc_thresholds.png    네 방법의 문턱이 k 에 따라 어떻게 벌어지는가
  ch11/post_hoc/img/tukey_studentized_range.png  투키의 임계값은 최대차의 귀무분포에서 나온다
  ch11/post_hoc/img/scheffe_and_bonferroni.png    모든 대비 위의 최댓값이 (k-1)F 다
  ch11/post_hoc/img/dunnett_critical.png      대조군 비교의 상관을 쓰면 문턱이 낮아진다
  ch11/post_hoc/img/games_howell_pairs.png    쌍마다 표준오차와 자유도를 따로 쓴다

실행:  python3 scripts/make_ch11_post_hoc_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, statsmodels — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os
from itertools import combinations

import numpy as np
from scipy import stats
from statsmodels.stats.libqsturng import qsturng

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

OUT = "docs/ch11/post_hoc/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# =====================================================================
# 1. 네 방법의 문턱 — post_hoc_comparisons.md
# =====================================================================
def fig_thresholds():
    n = 15
    ks = np.arange(2, 11)
    lsd, tuk, bon, sch = [], [], [], []
    for k in ks:
        nu = k * n - k
        m = k * (k - 1) // 2
        lsd.append(stats.t.ppf(0.975, nu))
        tuk.append(qsturng(0.95, k, nu) / np.sqrt(2))
        bon.append(stats.t.ppf(1 - 0.025 / m, nu))
        sch.append(np.sqrt((k - 1) * stats.f.ppf(0.95, k - 1, nu)))
    for i, k in enumerate(ks):
        print(f"  k={k}: LSD {lsd[i]:.3f}  투키 {tuk[i]:.3f}  "
              f"본페로니 {bon[i]:.3f}  셰페 {sch[i]:.3f}")

    # 본문 보기 자료
    rng = np.random.default_rng(42)
    g = [rng.normal(10.0, 3.5, 15), rng.normal(12.0, 3.5, 15),
         rng.normal(15.0, 3.5, 15)]
    names = ["A", "B", "C"]
    N, k = 45, 3
    MSW = sum(((x - x.mean()) ** 2).sum() for x in g) / (N - k)
    se = np.sqrt(MSW * 2 / 15)
    pairs = [(0, 1), (0, 2), (1, 2)]
    diffs = [abs(g[j].mean() - g[i].mean()) for i, j in pairs]
    th = {"피셔 LSD": stats.t.ppf(0.975, N - k) * se,
          "투키 HSD": qsturng(0.95, k, N - k) / np.sqrt(2) * se,
          "본페로니": stats.t.ppf(1 - 0.025 / 3, N - k) * se,
          "셰페": np.sqrt(2 * stats.f.ppf(0.95, 2, N - k)) * se}
    print(f"  MSW = {MSW:.4f}, SE = {se:.4f}")
    print(f"  차이 = {np.round(diffs, 4).tolist()}")
    for kk, vv in th.items():
        print(f"    {kk}: 문턱 {vv:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))

    ax = axes[0]
    for vals, c, lb, ls in [(lsd, MUTED, "보정 없음 (피셔 LSD)", "--"),
                            (tuk, GREEN, "투키 HSD", "-"),
                            (bon, BLUE, "본페로니 (모든 쌍)", "-"),
                            (sch, PURPLE, "셰페", "-")]:
        ax.plot(ks, vals, "o" + ls, color=c, lw=2.2, ms=6, label=lb)
    ax.set_xlabel("집단 수 $k$   (집단당 $n$ = 15)", fontsize=10.5, color=INK)
    ax.set_ylabel("문턱 / 표준오차", fontsize=10.5, color=INK)
    ax.set_xticks(ks)
    ax.set_ylim(1.8, 4.6)
    ax.set_title("쌍별 차이를 유의하다고 하려면 몇 배가 필요한가",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper left")
    clean(ax)

    ax = axes[1]
    ys = np.arange(3)[::-1]
    ax.barh(ys, diffs, height=0.42, color=BLUE_F, edgecolor=BLUE, lw=1.6)
    for lb, c, ls in [("피셔 LSD", MUTED, "--"), ("투키 HSD", GREEN, "-"),
                      ("본페로니", BLUE, "-"), ("셰페", PURPLE, "-")]:
        ax.axvline(th[lb], color=c, lw=2, ls=ls,
                   label=f"{lb} 문턱  {th[lb]:.3f}")
    for i, (a, b) in enumerate(pairs):
        ax.text(0.12, ys[i] + 0.29, f"{names[a]} – {names[b]}", va="bottom",
                fontsize=11, color=INK)
        ax.text(diffs[i] - 0.14, ys[i], f"{diffs[i]:.3f}", va="center",
                ha="right", fontsize=10.5, color=BLUE)
    ax.annotate("보정하면 문턱 아래로 내려간다", (2.36, 2.0),
                textcoords="offset points", xytext=(60, 36), fontsize=10,
                color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_yticks([])
    ax.set_xlim(0, 6.9)
    ax.set_ylim(-0.75, 2.85)
    ax.set_xlabel("평균 차이", fontsize=10.5, color=INK)
    ax.set_title("$k$ = 3, $n$ = 15 인 본문 보기", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="lower right")
    clean(ax)

    fig.tight_layout()
    save(fig, "posthoc_thresholds.png")


# =====================================================================
# 2. 스튜던트화 범위 — tukey.md
# =====================================================================
PLANT = {
    "ctrl": np.array([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]),
    "trt1": np.array([4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]),
    "trt2": np.array([6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]),
}


def fig_studentized_range():
    rng = np.random.default_rng(2024)
    k, n = 3, 10
    N = k * n
    nu = N - k
    M = 60_000
    mx = np.empty(M)
    for i in range(M):
        g = [rng.standard_normal(n) for _ in range(k)]
        MSW = sum(((x - x.mean()) ** 2).sum() for x in g) / nu
        se = np.sqrt(MSW * 2 / n)
        mx[i] = max(abs(g[a].mean() - g[b].mean())
                    for a, b in combinations(range(k), 2)) / se

    c_lsd = stats.t.ppf(0.975, nu)
    c_tuk = qsturng(0.95, k, nu) / np.sqrt(2)
    c_bon = stats.t.ppf(1 - 0.025 / 3, nu)
    fw = {"피셔 LSD": np.mean(mx > c_lsd), "투키 HSD": np.mean(mx > c_tuk),
          "본페로니": np.mean(mx > c_bon)}
    print(f"  임계값  LSD {c_lsd:.4f}  투키 {c_tuk:.4f}  본페로니 {c_bon:.4f}")
    for a, b in fw.items():
        print(f"    {a}: FWER {b:.4f}")

    # PlantGrowth 의 보정 p 값
    names = list(PLANT)
    allv = np.concatenate([PLANT[t] for t in names])
    MSW = sum(((PLANT[t] - PLANT[t].mean()) ** 2).sum() for t in names) / 27
    se = np.sqrt(MSW * 2 / 10)
    rows = []
    for a, b in combinations(range(3), 2):
        d = PLANT[names[a]].mean() - PLANT[names[b]].mean()
        t = abs(d) / se
        p_raw = 2 * stats.t.sf(t, 27)
        p_tuk = 1 - stats.studentized_range.cdf(t * np.sqrt(2), 3, 27)
        p_bon = min(1.0, 3 * p_raw)
        rows.append((f"{names[a]}–{names[b]}", p_raw, p_tuk, p_bon))
        print(f"  {names[a]}-{names[b]}: raw {p_raw:.4f} tukey {p_tuk:.4f} "
              f"bonf {p_bon:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))

    ax = axes[0]
    ax.hist(mx, bins=np.linspace(0, 6, 160), density=True, color=GREEN_F,
            edgecolor=GREEN, lw=0.4)
    for c, col, lb in [(c_lsd, MUTED, "피셔 LSD"), (c_tuk, GREEN, "투키 HSD"),
                       (c_bon, BLUE, "본페로니")]:
        ax.axvline(c, color=col, lw=2)
    ax.annotate(f"피셔 LSD  {c_lsd:.3f}\nFWER {fw['피셔 LSD']:.3f}",
                (c_lsd, 0.55), textcoords="offset points", xytext=(-108, 24),
                fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.annotate(f"투키 HSD  {c_tuk:.3f}\nFWER {fw['투키 HSD']:.3f}",
                (c_tuk, 0.32), textcoords="offset points", xytext=(28, 58),
                fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.annotate(f"본페로니  {c_bon:.3f}\nFWER {fw['본페로니']:.3f}",
                (c_bon, 0.14), textcoords="offset points", xytext=(34, 34),
                fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("세 쌍 가운데 가장 큰 $|t|$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("$H_0$ 아래 최대 $|t|$ 의 분포 ($k$ = 3, $n$ = 10, 6만 회)",
                 fontsize=11, color=INK)
    clean(ax)

    ax = axes[1]
    xs = np.arange(3)
    for off, key, c, lb in [(-0.26, 1, MUTED, "보정 없음"),
                            (0.0, 2, GREEN, "투키 HSD"),
                            (0.26, 3, BLUE, "본페로니")]:
        vals = [r[key] for r in rows]
        ax.bar(xs + off, vals, width=0.24, color=c + "33", edgecolor=c,
               lw=1.6, label=lb)
        for i, v in enumerate(vals):
            ax.text(xs[i] + off, v + 0.012, f"{v:.4f}", ha="center",
                    fontsize=8.5, color=c, rotation=90, va="bottom")
    ax.axhline(0.05, color=RED, lw=1.5, ls="--")
    ax.text(2.42, 0.062, "0.05", color=RED, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("조정 $p$ 값", fontsize=10.5, color=INK)
    ax.set_title("PlantGrowth 자료의 세 쌍", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper left")
    clean(ax)

    fig.tight_layout()
    save(fig, "tukey_studentized_range.png")


# =====================================================================
# 3. 본페로니가 버리는 상관 — bonferroni_scheffe.md
# =====================================================================
def fig_bonferroni_correlation():
    """셰페의 기하: 모든 대비 위에서 F_L 의 최댓값이 (k-1)F 다."""
    m = np.array([PLANT[t].mean() for t in PLANT])
    k, n = 3, 10
    N = k * n
    nu = N - k
    MSW = sum(((PLANT[t] - PLANT[t].mean()) ** 2).sum() for t in PLANT) / nu
    F_om = (n * ((m - m.mean()) ** 2).sum() / (k - 1)) / MSW
    print(f"  평균 {np.round(m, 4).tolist()}, MSW {MSW:.4f}, "
          f"옴니버스 F {F_om:.4f}")

    u1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    u2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    th = np.linspace(0, np.pi, 721)
    FL = np.array([((np.cos(t) * u1 + np.sin(t) * u2) @ m) ** 2 / (MSW / n)
                   for t in th])
    crit_s = (k - 1) * stats.f.ppf(0.95, k - 1, nu)
    crit_1 = stats.f.ppf(0.95, 1, nu)
    print(f"  max F_L {FL.max():.4f}  (k-1)F {(k - 1) * F_om:.4f}  "
          f"셰페 임계 {crit_s:.4f}  보정없음 임계 {crit_1:.4f}")

    marks = [("ctrl - trt1", 0.0), ("ctrl - trt2", np.pi / 3),
             ("trt1 - trt2", 2 * np.pi / 3)]
    for lb, t in marks:
        c = np.cos(t) * u1 + np.sin(t) * u2
        print(f"  {lb}: theta {np.degrees(t):.0f}도  F_L "
              f"{((c @ m) ** 2 / (MSW / n)):.4f}")
    t_best = th[np.argmax(FL)]
    c_best = np.cos(t_best) * u1 + np.sin(t_best) * u2
    print(f"  최적 대비 theta {np.degrees(t_best):.1f}도, "
          f"c = {np.round(c_best / np.abs(c_best).max(), 3).tolist()}")

    # m 에 따른 본페로니 대 셰페 (k = 4, n = 8)
    k2, n2 = 4, 8
    nu2 = k2 * n2 - k2
    ms = np.arange(1, 21)
    bon = stats.t.ppf(1 - 0.025 / ms, nu2)
    sch = np.sqrt((k2 - 1) * stats.f.ppf(0.95, k2 - 1, nu2))
    tuk2 = qsturng(0.95, k2, nu2) / np.sqrt(2)
    cross = ms[bon > sch][0]
    print(f"  k=4, nu={nu2}: 셰페 {sch:.4f}, 투키 {tuk2:.4f}, "
          f"본페로니가 셰페를 넘는 m = {cross}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    ax = axes[0]
    deg = np.degrees(th)
    ax.plot(deg, FL, color=PURPLE, lw=2.4)
    ax.axhline(crit_s, color=ORANGE, lw=2,
               label=f"셰페 임계값 $(k-1)F_{{0.05}}$ = {crit_s:.3f}")
    ax.axhline(crit_1, color=MUTED, lw=1.8, ls="--",
               label=f"보정 없는 임계값 $F_{{0.05,1,27}}$ = {crit_1:.3f}")
    ax.scatter([np.degrees(t_best)], [FL.max()], s=90, color=RED, zorder=5,
               edgecolor="white")
    ax.annotate(f"최댓값 {FL.max():.3f} = $(k-1)F$ = 2 $\\times$ {F_om:.3f}",
                (np.degrees(t_best), FL.max()), textcoords="offset points",
                xytext=(-46, 22), fontsize=10, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    for lb, t in marks:
        c = np.cos(t) * u1 + np.sin(t) * u2
        f = (c @ m) ** 2 / (MSW / n)
        ax.scatter([np.degrees(t)], [f], s=60, color=BLUE, zorder=4,
                   edgecolor="white")
        dy, va = (0.45, "bottom") if lb != "ctrl - trt2" else (-0.5, "top")
        ax.text(np.degrees(t), f + dy, lb, ha="center", va=va,
                fontsize=9.5, color=BLUE)
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_xlabel("대비의 방향 (모든 대비를 한 바퀴 돈다)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("대비의 $F_L$", fontsize=10.5, color=INK)
    ax.set_xlim(-6, 186)
    ax.set_ylim(-2.6, 12.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.set_title("$k$ = 3 의 대비 공간을 한 바퀴 돌면", fontsize=11.5,
                 color=INK)
    ax.legend(fontsize=9.5, loc="lower center")
    clean(ax)

    ax = axes[1]
    ax.plot(ms, bon, "o-", color=PURPLE, lw=2.2, ms=6, label="본페로니")
    ax.axhline(sch, color=ORANGE, lw=2.2,
               label=f"셰페 (모든 대비)  {sch:.3f}")
    ax.axhline(tuk2, color=GREEN, lw=2.2, ls="--",
               label=f"투키 (쌍별 6 개)  {tuk2:.3f}")
    ax.axvline(6, color=MUTED, lw=1.2, ls=":")
    ax.text(6.25, 2.12, "쌍별 전부\n$m$ = 6", fontsize=9.5, color=INK)
    ax.scatter([cross], [bon[cross - 1]], s=90, color=RED, zorder=5,
               edgecolor="white")
    ax.annotate(f"$m$ = {cross} 부터는 셰페가 더 낫다",
                (cross, bon[cross - 1]), textcoords="offset points",
                xytext=(-16, 40), fontsize=10, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("계획한 비교의 수 $m$   ($k$ = 4, $n$ = 8)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("문턱 / 표준오차", fontsize=10.5, color=INK)
    ax.set_xticks([1, 3, 5, 6, 10, 15, 20])
    ax.set_ylim(1.9, 3.6)
    ax.set_title("가족을 좁히면 이기고 넓히면 진다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="lower right")
    clean(ax)

    fig.tight_layout()
    save(fig, "scheffe_and_bonferroni.png")


# =====================================================================
# 4. 더넷 — dunnett.md
# =====================================================================
def fig_dunnett():
    rng = np.random.default_rng(1955)

    def dunnett_crit(k, n, M=80_000):
        """균형 설계에서 max |t_i| 의 95 백분위수를 모의실험으로 구한다."""
        nu = k * n - k
        out = np.empty(M)
        for i in range(M):
            g = [rng.standard_normal(n) for _ in range(k)]
            MSW = sum(((x - x.mean()) ** 2).sum() for x in g) / nu
            se = np.sqrt(MSW * 2 / n)
            out[i] = max(abs(g[j].mean() - g[0].mean()) / se
                         for j in range(1, k))
        return np.percentile(out, 95)

    n = 6
    ks = [3, 4, 5, 6, 7]
    dun, bon, tuk = [], [], []
    for k in ks:
        nu = k * n - k
        dun.append(dunnett_crit(k, n, 40_000))
        bon.append(stats.t.ppf(1 - 0.025 / (k - 1), nu))
        tuk.append(qsturng(0.95, k, nu) / np.sqrt(2))
        print(f"  k={k}: 더넷 {dun[-1]:.4f}  본페로니 {bon[-1]:.4f}  "
              f"투키 {tuk[-1]:.4f}")

    # 본문 보기 1
    ts = [3.49, 1.65, 4.16]
    labels = ["약 A", "약 B", "약 C"]
    i4 = ks.index(4)
    d4, b4, t4 = dun[i4], bon[i4], tuk[i4]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    ax = axes[0]
    ax.plot(ks, dun, "o-", color=GREEN, lw=2.2, ms=8, label="더넷 (대조군 비교)")
    ax.plot(ks, bon, "s-", color=PURPLE, lw=2.2, ms=7,
            label="본페로니 ($k-1$ 개)")
    ax.plot(ks, tuk, "^--", color=MUTED, lw=2.0, ms=7,
            label="투키 (쌍별 전부)")
    for i, k in enumerate(ks):
        ax.text(k, dun[i] - 0.14, f"{dun[i]:.2f}", ha="center", fontsize=9,
                color=GREEN)
    ax.set_xticks(ks)
    ax.set_xlabel("집단 수 $k$ (대조군 1 + 처치 $k-1$),  집단당 $n$ = 6",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("문턱 / 표준오차", fontsize=10.5, color=INK)
    ax.set_ylim(2.1, 3.6)
    ax.set_title("상관을 쓰면 문턱이 내려간다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper left")
    clean(ax)

    ax = axes[1]
    ys = np.arange(3)[::-1]
    ax.barh(ys, ts, height=0.45, color=BLUE_F, edgecolor=BLUE, lw=1.6)
    for i, (y, t) in enumerate(zip(ys, ts)):
        ax.text(0.1, y, labels[i], va="center", fontsize=11, color=INK)
        ax.text(t + 0.08, y, f"$t$ = {t}", va="center", fontsize=10,
                color=BLUE)
    for c, col, lb in [(d4, GREEN, "더넷"), (b4, PURPLE, "본페로니"),
                       (t4, MUTED, "투키")]:
        ax.axvline(c, color=col, lw=2.2, label=f"{lb} 문턱  {c:.3f}")
    ax.set_yticks([])
    ax.set_xlim(0, 5.6)
    ax.set_ylim(-0.6, 3.1)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_xlabel("$|t_i|$", fontsize=11, color=INK)
    ax.set_title("본문 보기 1 — 위약 대비 세 약 ($k$ = 4, $n$ = 6)",
                 fontsize=11.5, color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "dunnett_critical.png")


# =====================================================================
# 5. 게임스–하월 — games_howell.md
# =====================================================================
def fig_games_howell():
    names = ["A 기준", "B 소셜", "C 이메일", "D 인플루언서"]
    n = np.array([15.0, 10.0, 20.0, 8.0])
    m = np.array([12.0, 16.5, 13.2, 18.0])
    v = np.array([4.0, 12.0, 3.5, 15.0])
    k = 4
    N = n.sum()
    MSW = ((n - 1) * v).sum() / (N - k)
    q_t = qsturng(0.95, k, N - k) / np.sqrt(2)
    print(f"  합동 MSW = {MSW:.4f}, df = {N - k:.0f}, 투키 q/√2 = {q_t:.4f}")

    rows = []
    for i, j in combinations(range(k), 2):
        d = abs(m[i] - m[j])
        se_t = np.sqrt(MSW * (1 / n[i] + 1 / n[j]))
        a, b = v[i] / n[i], v[j] / n[j]
        se_g = np.sqrt(a + b)
        nu = (a + b) ** 2 / (a ** 2 / (n[i] - 1) + b ** 2 / (n[j] - 1))
        th_t = q_t * se_t
        th_g = qsturng(0.95, k, nu) / np.sqrt(2) * se_g
        rows.append((f"{names[i][0]}–{names[j][0]}", d, th_t, th_g, nu,
                     se_t, se_g))
        print(f"  {names[i][0]}-{names[j][0]}: 차이 {d:.2f}  "
              f"투키문턱 {th_t:.3f}  GH문턱 {th_g:.3f}  nu {nu:.2f}  "
              f"판정 {'투키O' if d > th_t else '투키X'}/"
              f"{'GH O' if d > th_g else 'GH X'}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))

    ax = axes[0]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r[1] for r in rows], height=0.5, color=BLUE_F,
            edgecolor=BLUE, lw=1.6, label="관측된 평균 차이")
    ax.plot([r[2] for r in rows], ys, "s", color=MUTED, ms=9,
            label="투키–크레이머 문턱 (합동 $\\text{MS}_W$)")
    ax.plot([r[3] for r in rows], ys, "o", color=ORANGE, ms=9,
            label="게임스–하월 문턱 (쌍별 분산)")
    for i, r in enumerate(rows):
        ax.text(0.1, ys[i], r[0], va="center", fontsize=11, color=INK)
        flip = (r[1] > r[2]) != (r[1] > r[3])
        if flip:
            ax.text(max(r[1], r[3]) + 0.3, ys[i], "판정이 갈린다",
                    va="center", fontsize=10, color=RED)
    ax.set_yticks([])
    ax.set_xlim(0, 8.4)
    ax.set_ylim(-0.7, 6.0)
    ax.set_xlabel("평균 차이", fontsize=10.5, color=INK)
    ax.set_title("여섯 쌍의 차이와 두 방법의 문턱", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="lower right")
    clean(ax)

    ax = axes[1]
    ax.barh(ys, [r[4] for r in rows], height=0.5, color=ORANGE_F,
            edgecolor=ORANGE, lw=1.6, label="게임스–하월의 쌍별 $\\nu$")
    ax.axvline(N - k, color=MUTED, lw=2.2,
               label=f"투키의 공통 자유도  {N - k:.0f}")
    for i, r in enumerate(rows):
        ax.text(r[4] + 0.6, ys[i], f"{r[4]:.1f}", va="center", fontsize=10,
                color=ORANGE)
        ax.text(0.8, ys[i], r[0], va="center", fontsize=10.5, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.7, 6.0)
    ax.set_xlabel("분모 자유도", fontsize=10.5, color=INK)
    ax.set_title("쌍마다 자유도가 다르다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper right")
    clean(ax)

    fig.tight_layout()
    save(fig, "games_howell_pairs.png")


if __name__ == "__main__":
    fig_thresholds()
    fig_studentized_range()
    fig_bonferroni_correlation()
    fig_dunnett()
    fig_games_howell()
