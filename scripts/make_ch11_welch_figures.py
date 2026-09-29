r"""11장 웰치 분산분석 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch11/anova_welch/img/welch_weights.png        분산의 역수로 가중하면 발언권이 바뀐다
  ch11/anova_welch/img/welch_size_direction.png 고전 F 의 수준은 불균형의 방향에 달려 있다
  ch11/anova_welch/img/welch_twoway_df.png      자유도가 작으면 유의성 문턱이 폭발한다
  ch11/anova_welch/img/welch_hc3_vs_classic.png 칸 분산이 다르면 고전 F 가 부풀려진다

실행:  python3 scripts/make_ch11_welch_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, statsmodels — 문서 빌드에는 필요하지 않다.
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

OUT = "docs/ch11/anova_welch/img/"
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


def welch_from_summary(n, m, v):
    """표본크기·평균·분산 요약값에서 웰치 통계량과 자유도를 계산한다."""
    n, m, v = map(lambda a: np.asarray(a, float), (n, m, v))
    k = len(n)
    w = n / v
    W = w.sum()
    m_t = (w * m).sum() / W
    tmp = np.sum((1 - w / W) ** 2 / (n - 1))
    A = (w * (m - m_t) ** 2).sum() / (k - 1)
    B = 1 + 2 * (k - 2) / (k * k - 1) * tmp
    F = A / B
    df2 = (k * k - 1) / (3 * tmp)
    return w, W, m_t, F, df2, stats.f.sf(F, k - 1, df2)


# =====================================================================
# 1. 가중치 — welch_one_way.md
# =====================================================================
def fig_weights():
    labels = ["모멘텀", "가치", "인덱스"]
    n = np.array([36.0, 24.0, 48.0])
    m = np.array([1.8, 1.2, 1.0])
    v = np.array([12.5, 3.1, 5.8])
    k, N = 3, n.sum()

    w, W, m_t, F_w, df2_w, p_w = welch_from_summary(n, m, v)
    gm = (n * m).sum() / N
    MSW = ((n - 1) * v).sum() / (N - k)
    SSB = (n * (m - gm) ** 2).sum()
    F_c = (SSB / (k - 1)) / MSW
    p_c = stats.f.sf(F_c, k - 1, N - k)
    se = np.sqrt(v / n)
    print(f"  w = {np.round(w, 4).tolist()},  W = {W:.4f}")
    print(f"  고전 전체평균 {gm:.4f} / 웰치 가중평균 {m_t:.4f}")
    print(f"  고전 F = {F_c:.4f} (df2 = {N - k:.0f}), p = {p_c:.4f}")
    print(f"  웰치 F = {F_w:.4f} (df2 = {df2_w:.4f}), p = {p_w:.4f}")
    print(f"  SE = {np.round(se, 4).tolist()}")

    fig = plt.figure(figsize=(12.6, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.4)

    # (a) 집단 평균과 표준오차
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(3)
    ax.errorbar(xs, m, yerr=se, fmt="o", color=INK, ms=9, lw=0,
                ecolor=MUTED, elinewidth=2.4, capsize=8, capthick=2.4)
    for i in range(3):
        ax.text(xs[i] + 0.16, m[i], f"$n$ = {n[i]:.0f}\n$s^2$ = {v[i]}",
                fontsize=9.5, color=INK, va="center", ha="left")
    ax.axhline(gm, color=BLUE, lw=1.6, ls="--")
    ax.axhline(m_t, color=ORANGE, lw=1.6, ls="-")
    ax.text(-0.42, gm + 0.035, f"고전 전체평균 {gm:.3f}", fontsize=9.5,
            color=BLUE, va="bottom")
    ax.text(-0.42, m_t - 0.035, f"웰치 가중평균 {m_t:.3f}", fontsize=9.5,
            color=ORANGE, va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_xlim(-0.5, 2.85)
    ax.set_ylim(0.3, 2.6)
    ax.set_ylabel("월 수익률 (%)", fontsize=10.5, color=INK)
    ax.set_title("평균 $\\pm$ 표준오차", fontsize=11.5, color=INK)
    clean(ax)

    # (b) 발언권 비교
    ax = fig.add_subplot(gs[0, 1])
    share_c = n / N
    share_w = w / W
    xs = np.arange(3)
    ax.bar(xs - 0.19, share_c, width=0.36, color=BLUE_F, edgecolor=BLUE,
           lw=1.6, label="고전 분산분석 ($\\propto n_i$)")
    ax.bar(xs + 0.19, share_w, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.6, label="웰치 ($\\propto n_i / s_i^2$)")
    for i in range(3):
        ax.text(xs[i] - 0.19, share_c[i] + 0.012, f"{share_c[i] * 100:.1f}%",
                ha="center", fontsize=9.5, color=BLUE)
        ax.text(xs[i] + 0.19, share_w[i] + 0.012, f"{share_w[i] * 100:.1f}%",
                ha="center", fontsize=9.5, color=ORANGE)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("전체 가중치에서 차지하는 몫", fontsize=10.5, color=INK)
    ax.set_title("분산이 크면 발언권을 잃는다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper center", ncol=1)
    clean(ax)

    # (c) 자유도
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar([0, 1], [N - k, df2_w], width=0.55,
                  color=[BLUE_F, ORANGE_F], edgecolor=[BLUE, ORANGE], lw=1.6)
    for b, val, col in zip(bars, [N - k, df2_w], [BLUE, ORANGE]):
        ax.text(b.get_x() + b.get_width() / 2, val + 3, f"{val:.1f}",
                ha="center", fontsize=12, color=col)
    ax.text(0, 42, f"$F$ = {F_c:.3f}\n$p$ = {p_c:.3f}", ha="center",
            fontsize=10.5, color=BLUE)
    ax.text(1, 22, f"$F$ = {F_w:.3f}\n$p$ = {p_w:.3f}", ha="center",
            fontsize=10.5, color=ORANGE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"고전 ($N-k$)", "웰치 (Satterthwaite)"], fontsize=10)
    ax.set_ylim(0, 128)
    ax.set_ylabel("분모 자유도", fontsize=10.5, color=INK)
    ax.set_title("이분산은 자유도로 대가를 치른다", fontsize=11.5, color=INK)
    clean(ax)

    save(fig, "welch_weights.png")


# =====================================================================
# 2. 불균형의 방향 — welch_simulation.md
# =====================================================================
def fig_size_direction():
    rng = np.random.default_rng(11)
    sig = np.array([1.0, 3.0, 6.0])
    configs = [("(18, 10, 7)\n큰 $\\sigma$ ↔ 작은 $n$", (18, 10, 7)),
               ("(10, 18, 7)\n본문 설계", (10, 18, 7)),
               ("(12, 12, 11)\n거의 균형", (12, 12, 11)),
               ("(7, 10, 18)\n큰 $\\sigma$ ↔ 큰 $n$", (7, 10, 18))]
    M = 20_000
    res = []
    for label, ns in configs:
        a = b = 0
        for _ in range(M):
            g = [rng.normal(10.0, s, nn) for s, nn in zip(sig, ns)]
            a += stats.f_oneway(*g).pvalue < 0.05
            nn_ = [len(x) for x in g]
            mm = [x.mean() for x in g]
            vv = [x.var(ddof=1) for x in g]
            b += welch_from_summary(nn_, mm, vv)[5] < 0.05
        res.append((label, a / M, b / M))
        print(f"  {ns}: 고전 {a / M:.4f}  웰치 {b / M:.4f}")

    # 검정력: 본문 설계에서 G3 의 표본만 키운다
    n3s = [7, 12, 20, 35, 60, 100, 150]
    pw = []
    Mp = 6_000
    for n3 in n3s:
        ns = (10, 18, n3)
        hit = 0
        for _ in range(Mp):
            g = [rng.normal(mu, s, nn) for mu, s, nn
                 in zip([10.0, 10.0, 12.0], sig, ns)]
            hit += welch_from_summary([len(x) for x in g],
                                      [x.mean() for x in g],
                                      [x.var(ddof=1) for x in g])[5] < 0.05
        pw.append(hit / Mp)
        print(f"  n3 = {n3}: 웰치 검정력 {hit / Mp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    xs = np.arange(4)
    cls = [r[1] for r in res]
    wel = [r[2] for r in res]
    ax.bar(xs - 0.19, cls, width=0.36, color=BLUE_F, edgecolor=BLUE, lw=1.6,
           label="고전 $F$ 검정")
    ax.bar(xs + 0.19, wel, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.6, label="웰치 분산분석")
    for i in range(4):
        ax.text(xs[i] - 0.19, cls[i] + 0.006, f"{cls[i]:.3f}", ha="center",
                fontsize=9.5, color=BLUE)
        ax.text(xs[i] + 0.19, wel[i] + 0.006, f"{wel[i]:.3f}", ha="center",
                fontsize=9.5, color=ORANGE)
    ax.axhline(0.05, color=RED, lw=1.5, ls="--")
    ax.text(-0.5, 0.056, "명목 0.05", color=RED, fontsize=10, va="bottom",
            ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in res], fontsize=9)
    ax.set_ylim(0, 0.245)
    ax.set_xlim(-0.55, 3.6)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.set_title("$\\sigma$ = (1, 3, 6) 고정, 표본만 바꾼다",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=10, loc="upper right")
    clean(ax)

    ax = axes[1]
    ax.plot(n3s, pw, "o-", color=GREEN, lw=2.2, ms=8)
    for x, y in zip(n3s, pw):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9.5, color=GREEN)
    ax.axhline(0.8, color=RED, lw=1.4, ls="--")
    ax.text(4, 0.818, "검정력 0.80", color=RED, fontsize=10, ha="left")
    ax.set_xlabel("$G_3$ 의 표본크기 $n_3$   ($n_1 = 10$, $n_2 = 18$ 고정)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("웰치 분산분석의 검정력", fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 162)
    ax.set_title("$\\Delta$ = 2, $\\sigma_3$ = 6 일 때의 검정력",
                 fontsize=11.5, color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "welch_size_direction.png")


# =====================================================================
# 3. 작은 자유도 — welch_two_way.md
# =====================================================================
def fig_twoway_df():
    from statsmodels.stats.libqsturng import qsturng

    nus = np.linspace(2.2, 30, 400)
    crit = np.array([qsturng(0.95, 3, nu) for nu in nus]) / np.sqrt(2)

    obs = [("High - Low", 4.0000, 1.8708),
           ("High - Medium", 3.4483, 1.2649),
           ("Low - Medium", 3.4483, 3.4785)]
    for lb, nu, t in obs:
        print(f"  {lb}: nu = {nu}, |T| = {t}, 문턱 = "
              f"{qsturng(0.95, 3, nu) / np.sqrt(2):.4f}")

    # 반복을 늘리면 문턱(필요 평균차)이 어떻게 내려가는가
    v_low, v_med = 7.0 / 3, 1.0          # Low, Medium 의 표본분산
    ns = np.arange(3, 25)
    need = []
    for n in ns:
        a, b = v_low / n, v_med / n
        nu = (a + b) ** 2 / (a ** 2 / (n - 1) + b ** 2 / (n - 1))
        se = np.sqrt(a + b)
        need.append(qsturng(0.95, 3, nu) / np.sqrt(2) * se)
    need = np.array(need)
    print(f"  n = 3 에서 필요한 평균차 {need[0]:.4f}, "
          f"n = 6 에서 {need[3]:.4f}, n = 10 에서 {need[7]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    ax.plot(nus, crit, color=INK, lw=2.4)
    ax.fill_between(nus, crit, 6.5, color="#FBE9E7", zorder=0)
    ax.text(21, 4.6, "유의", fontsize=12, color=RED, ha="center")
    ax.text(21, 1.2, "유의하지 않음", fontsize=12, color=MUTED, ha="center")
    for lb, nu, t in obs:
        ax.scatter([nu], [t], s=70, color=BLUE, zorder=4, edgecolor="white")
    ax.annotate("Low - Medium\n$|T|$ = 3.479", (3.4483, 3.4785),
                textcoords="offset points", xytext=(30, 10), fontsize=9.5,
                color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate("High - Low\n$|T|$ = 1.871", (4.0, 1.8708),
                textcoords="offset points", xytext=(34, -6), fontsize=9.5,
                color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate("High - Medium\n$|T|$ = 1.265", (3.4483, 1.2649),
                textcoords="offset points", xytext=(36, -24), fontsize=9.5,
                color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_xlabel("웰치–새터스웨이트 자유도 $\\nu$", fontsize=10.5, color=INK)
    ax.set_ylabel("게임스–하월의 유의성 문턱  $|T|$", fontsize=10.5, color=INK)
    ax.set_xlim(2, 30)
    ax.set_ylim(0, 6.5)
    ax.set_title("자유도가 4 이하면 문턱이 폭발한다", fontsize=11.5, color=INK)
    clean(ax)

    ax = axes[1]
    ax.plot(ns, need, "o-", color=ORANGE, lw=2.2, ms=6)
    ax.axhline(3.6667, color=GREEN, lw=1.8, ls="--")
    ax.text(24.3, 3.82, "관측된 평균차 3.667", color=GREEN, fontsize=10,
            ha="right")
    ax.scatter([3], [need[0]], s=90, color=RED, zorder=5, edgecolor="white")
    ax.annotate(f"수준당 3 개:\n{need[0]:.2f} 은 되어야 한다",
                (3, need[0]), textcoords="offset points", xytext=(22, -8),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    cross = ns[need < 3.6667][0]
    ax.axvline(cross, color=MUTED, lw=1.2, ls=":")
    ax.text(cross + 0.4, 5.3, f"{cross} 개부터 유의", fontsize=10, color=INK)
    ax.set_xlabel("온도 수준당 관측값 수", fontsize=10.5, color=INK)
    ax.set_ylabel("유의해지는 데 필요한 평균차", fontsize=10.5, color=INK)
    ax.set_xlim(2, 25)
    ax.set_ylim(0, 6.2)
    ax.set_title("분산은 그대로 두고 표본만 늘린다면", fontsize=11.5, color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "welch_twoway_df.png")


# =====================================================================
# 4. HC3 대 고전 — welch_twoway_robust.md
# =====================================================================
ROBUST = {
    "High": {"A": [12, 13], "B": [15, 18], "C": [14, 15]},
    "Low": {"A": [10, 9], "B": [13, 12], "C": [11, 13]},
    "Medium": {"A": [14, 16], "B": [16, 21], "C": [15, 17]},
}


def fig_hc3():
    import pandas as pd
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    rows = []
    for t, d in ROBUST.items():
        for f, vals in d.items():
            for v in vals:
                rows.append({"Temperature": t, "Fertilizer": f, "Growth": v})
    df = pd.DataFrame(rows)

    model = ols("Growth ~ C(Temperature) * C(Fertilizer)", data=df).fit()
    rob = model.get_robustcov_results(cov_type="HC3")
    pnames = model.params.index.tolist()

    def wald(pref, inter=False):
        if inter:
            terms = [p for p in pnames if ":" in p]
        else:
            terms = [p for p in pnames if p.startswith(pref) and ":" not in p]
        r = rob.f_test(", ".join(f"{t} = 0" for t in terms))
        return float(r.fvalue), float(r.pvalue)

    F_t, p_t = wald("C(Temperature)[")
    F_f, p_f = wald("C(Fertilizer)[")
    F_i, p_i = wald("", inter=True)
    tab = sm.stats.anova_lm(model, typ=2)
    cF = [tab.loc["C(Temperature)", "F"], tab.loc["C(Fertilizer)", "F"],
          tab.loc["C(Temperature):C(Fertilizer)", "F"]]
    cp = [tab.loc["C(Temperature)", "PR(>F)"], tab.loc["C(Fertilizer)", "PR(>F)"],
          tab.loc["C(Temperature):C(Fertilizer)", "PR(>F)"]]
    rF, rp = [F_t, F_f, F_i], [p_t, p_f, p_i]
    print(f"  고전 F {np.round(cF, 4).tolist()}  p {np.round(cp, 4).tolist()}")
    print(f"  HC3  F {np.round(rF, 4).tolist()}  p {np.round(rp, 4).tolist()}")

    temps = ["Low", "High", "Medium"]
    ferts = ["A", "B", "C"]
    cvars, clabels = [], []
    for f in ferts:
        for t in temps:
            cvars.append(np.var(ROBUST[t][f], ddof=1))
            clabels.append(f"{f} · {t}")
    print(f"  칸 분산 최소 {min(cvars)} 최대 {max(cvars)}")

    fig = plt.figure(figsize=(12.8, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.2, 1], wspace=0.4)

    # (a) 칸별 관측값
    ax = fig.add_subplot(gs[0, 0])
    cols = {"A": BLUE, "B": RED, "C": GREEN}
    for j, t in enumerate(temps):
        for i, f in enumerate(ferts):
            x = j + (i - 1) * 0.24
            vals = ROBUST[t][f]
            ax.plot([x, x], vals, color=cols[f], lw=2.2, alpha=0.75)
            ax.scatter([x, x], vals, s=34, color=cols[f], zorder=3)
    for f in ferts:
        ax.plot([], [], color=cols[f], lw=2.4, label=f"비료 {f}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(temps, fontsize=10.5)
    ax.set_ylim(7, 23)
    ax.set_ylabel("성장량", fontsize=10.5, color=INK)
    ax.set_title("칸마다 반복 2 개 — 세로 길이가 칸 내 흩어짐",
                 fontsize=11, color=INK)
    ax.legend(fontsize=9.5, loc="upper left")
    clean(ax)

    # (b) 칸 분산
    ax = fig.add_subplot(gs[0, 1])
    colors = [cols[lb.split(" · ")[0]] for lb in clabels]
    ax.bar(range(9), cvars, width=0.68, color=[c + "33" for c in colors],
           edgecolor=colors, lw=1.6)
    for i, v in enumerate(cvars):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(9))
    ax.set_xticklabels(clabels, fontsize=8.5, rotation=45, ha="right")
    ax.set_ylim(0, 15)
    ax.set_ylabel("칸 내 표본분산", fontsize=10.5, color=INK)
    ax.set_title("최소 0.5, 최대 12.5 — 25 배 차이", fontsize=11.5, color=INK)
    clean(ax)

    # (c) F 비교
    ax = fig.add_subplot(gs[0, 2])
    xs = np.arange(3)
    ax.bar(xs - 0.19, cF, width=0.36, color=BLUE_F, edgecolor=BLUE, lw=1.6,
           label="고전 분산분석")
    ax.bar(xs + 0.19, rF, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.6, label="HC3 발트 검정")
    for i in range(3):
        ax.text(xs[i] - 0.19, cF[i] + 0.4, f"{cF[i]:.2f}", ha="center",
                fontsize=9.5, color=BLUE)
        ax.text(xs[i] + 0.19, rF[i] + 0.4, f"{rF[i]:.2f}", ha="center",
                fontsize=9.5, color=ORANGE)
    for i, dfn in enumerate([2, 2, 4]):
        c = stats.f.ppf(0.95, dfn, 9)
        ax.hlines(c, xs[i] - 0.42, xs[i] + 0.42, color=RED, lw=1.8)
    ax.text(2.45, 4.6, "5 % 임계값", color=RED, fontsize=9.5, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(["온도", "비료", "교호작용"], fontsize=10.5)
    ax.set_ylim(0, 19.5)
    ax.set_ylabel("$F$ 통계량", fontsize=10.5, color=INK)
    ax.set_title("로버스트 쪽이 일관되게 작다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, loc="upper right")
    clean(ax)

    save(fig, "welch_hc3_vs_classic.png")


if __name__ == "__main__":
    fig_weights()
    fig_size_direction()
    fig_twoway_df()
    fig_hc3()
