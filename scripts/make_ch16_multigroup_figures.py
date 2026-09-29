r"""16.5 다집단 비모수 검정 네 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch16/multi_group_nonparametric/img/kw_rank_spread.png       H 는 평균순위의 흩어짐이다
  ch16/multi_group_nonparametric/img/dunn_fwer.png            보정하지 않으면 FWER 이 샌다
  ch16/multi_group_nonparametric/img/friedman_blocks.png      블록 안에서 순위를 매긴다
  ch16/multi_group_nonparametric/img/mood_vs_kw_power.png     한 비트만 쓸 때의 검정력 값

실행:  python3 scripts/make_ch16_multigroup_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch16/multi_group_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9.5, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. H 는 평균순위의 흩어짐이다  (kruskal_wallis.md)
# ==================================================================
def kw_rank_spread():
    A = np.array([12, 15, 14, 10, 13], float)
    B = np.array([20, 18, 22, 17, 19], float)
    C = np.array([8, 11, 9, 7, 10], float)
    groups = [("비료 A", A, BLUE), ("비료 B", B, ORANGE), ("비료 C", C, GREEN)]
    allv = np.concatenate([A, B, C])
    N = len(allv)
    ranks = stats.rankdata(allv)
    rk = [ranks[0:5], ranks[5:10], ranks[10:15]]
    grand = (N + 1) / 2
    H, p = stats.kruskal(A, B, C)

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    for i, ((name, v, c), r) in enumerate(zip(groups, rk)):
        y = 2 - i
        mr = r.mean()
        ax.plot(r, np.full(len(r), y), "o", ms=11, color=c, mec="white",
                mew=1.2, zorder=3)
        for rv, vv in zip(r, v):
            ax.text(rv, y + 0.24, f"{vv:.0f}", fontsize=9, color=c,
                    ha="center", va="bottom")
        # 평균순위에서 전체평균까지의 편차를 막대로
        ax.plot([mr, mr], [y - 0.30, y + 0.30], color=c, lw=3.0, zorder=4)
        ax.annotate("", xy=(grand, y - 0.42), xytext=(mr, y - 0.42),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.4))
        if abs(mr - grand) < 1.5:
            lx, ly, la = grand + 1.3, y - 0.42, "left"
        else:
            lx, ly, la = (mr + grand) / 2, y - 0.62, "center"
        ax.text(lx, ly, f"$\\bar{{R}}_i - {grand:.0f}$ = {mr - grand:+.1f}",
                fontsize=10, color=c, ha=la, va="center")
        ax.text(-1.2, y, name, fontsize=11.5, color=c, ha="right",
                va="center")
        ax.text(16.6, y, f"$\\bar{{R}}_i$ = {mr:.1f}", fontsize=11, color=c,
                ha="left", va="center")

    ax.vlines(grand, -0.20, 2.85, color=INK, lw=1.8, ls=(0, (5, 3)))
    ax.text(grand + 0.18, 2.92, f"전체 평균순위 $(N+1)/2$ = {grand:.0f}",
            fontsize=11, color=INK, ha="left", va="center")
    ax.set_xlim(-4.6, 20.4)
    ax.set_ylim(-0.95, 3.15)
    ax.set_yticks([])
    ax.set_xticks(np.arange(1, 16, 2))
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("합친 표본에서의 순위", fontsize=10.5, color=INK)
    ax.text(-4.6, -0.78,
            f"$H = \\frac{{12}}{{N(N+1)}}\\sum n_i (\\bar{{R}}_i - 8)^2 "
            f"= \\frac{{12}}{{240}} \\times 5 \\times "
            f"[(-0.3)^2 + 5.0^2 + (-4.7)^2] = {H:.3f}$"
            f"        $p = {p:.4f}$",
            fontsize=11.5, color=INK, ha="left", va="center")
    ax.set_title("$H$ 는 각 집단의 평균순위가 전체 평균순위에서 얼마나 멀어졌는지를 잰다",
                 fontsize=13, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, "kw_rank_spread.png")
    print("mean ranks", [round(r.mean(), 2) for r in rk], "H %.4f p %.5f"
          % (H, p))


# ==================================================================
# 2. 보정하지 않으면 FWER 이 샌다  (dunn.md)
# ==================================================================
def dunn_pvalues(groups):
    """Dunn 검정의 보정 전 쌍별 p값."""
    allv = np.concatenate(groups)
    N = len(allv)
    ranks = stats.rankdata(allv)
    idx = np.cumsum([0] + [len(g) for g in groups])
    mr = [ranks[idx[i]:idx[i + 1]].mean() for i in range(len(groups))]
    ns = [len(g) for g in groups]
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            se = np.sqrt(N * (N + 1) / 12 * (1 / ns[i] + 1 / ns[j]))
            z = (mr[i] - mr[j]) / se
            out.append(2 * stats.norm.sf(abs(z)))
    return np.array(out)


def holm(pv):
    m = len(pv)
    order = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(run, 1.0)
    return adj


def dunn_fwer():
    rng = np.random.default_rng(5)
    ks = [3, 4, 5, 6, 8]
    n, B = 12, 4000
    raw, bon, hol, indep = [], [], [], []
    for k in ks:
        m = k * (k - 1) // 2
        cr = cb = ch = 0
        for _ in range(B):
            gs = [rng.normal(0, 1, n) for _ in range(k)]
            pv = dunn_pvalues(gs)
            cr += (pv < 0.05).any()
            cb += (np.minimum(pv * m, 1.0) < 0.05).any()
            ch += (holm(pv) < 0.05).any()
        raw.append(cr / B)
        bon.append(cb / B)
        hol.append(ch / B)
        indep.append(1 - (1 - 0.05) ** m)
        print(f"k={k} m={m} raw={cr/B:.3f} bonf={cb/B:.3f} holm={ch/B:.3f} "
              f"indep={indep[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(ks))
    w = 0.26
    ax.bar(x - w, raw, width=w, color=ORANGE_F, edgecolor=ORANGE, lw=1.3,
           label="보정하지 않음")
    ax.bar(x, bon, width=w, color=BLUE_F, edgecolor=BLUE, lw=1.3,
           label="Bonferroni 보정")
    ax.bar(x + w, hol, width=w, color=GREEN_F, edgecolor=GREEN, lw=1.3,
           label="Holm 보정")
    ax.plot(x, indep, "D--", color=PURPLE, lw=1.6, ms=6,
            label="검정이 독립이라면 $1-(1-\\alpha)^m$")
    ax.axhline(0.05, color=RED, lw=1.5, ls=(0, (5, 3)))
    ax.text(len(ks) - 0.55, 0.062, "명목 $\\alpha = 0.05$", fontsize=10.5,
            color=RED, ha="right", va="bottom")
    for xi, v in zip(x - w, raw):
        ax.text(xi, v + 0.012, f"{v:.3f}", fontsize=9, color=ORANGE,
                ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$k$ = {k}\n($m$ = {k*(k-1)//2} 쌍)" for k in ks])
    ax.set_ylim(0, 0.85)
    clean(ax)
    ax.set_ylabel("집단별 오류율 (FWER)", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.text(1.45, 0.60,
            "Bonferroni 와 Holm 은 '적어도 하나 기각'의 조건이 같아\n"
            "FWER 이 정확히 일치한다. Holm 이 더 강력한 것은\n"
            "두 번째 이후 쌍을 판정할 때다.",
            fontsize=10.5, color=INK, ha="left", va="top")
    ax.set_title("모든 집단이 동일한 자료에서 '유의한 쌍'을 하나라도 찾을 확률",
                 fontsize=13, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, "dunn_fwer.png")


# ==================================================================
# 3. 블록 안에서 순위를 매긴다  (friedman.md)
# ==================================================================
def friedman_blocks():
    d = np.array([[30, 45, 60, 50], [25, 40, 55, 35], [35, 50, 65, 45],
                  [20, 30, 50, 40], [40, 55, 70, 60]], float)
    drugs = ["약 A", "약 B", "약 C", "약 D"]
    cols = [BLUE, ORANGE, GREEN, PURPLE]
    b, k = d.shape
    R = np.apply_along_axis(stats.rankdata, 1, d)
    chi, p_f = stats.friedmanchisquare(*d.T)
    h, p_kw = stats.kruskal(*d.T.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    # (가) 원점수
    ax = axes[0]
    for j in range(b):
        ax.plot(np.arange(k), d[j], "-", color=MUTED, lw=1.2, zorder=1)
        ax.text(-0.18, d[j, 0], f"환자 {j + 1}", fontsize=9, color=MUTED,
                ha="right", va="center")
    for i in range(k):
        ax.plot(np.full(b, i), d[:, i], "o", ms=9, color=cols[i],
                mec="white", mew=1.0, zorder=3)
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(drugs, fontsize=10.5)
    ax.set_xlim(-1.05, k - 0.55)
    ax.set_ylim(14, 78)
    clean(ax)
    ax.set_ylabel("통증 완화 점수", fontsize=10.5, color=INK)
    ax.text(-1.0, 75, "환자마다 기준선이 달라\n점수 구간이 크게 겹친다",
            fontsize=10.5, color=INK, ha="left", va="top")
    ax.set_title("(가) 원점수 — 블록 간 변동이 섞여 있다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 블록 내 순위
    ax = axes[1]
    jit = np.linspace(-0.16, 0.16, b)
    for j in range(b):
        ax.plot(np.arange(k) + jit[j], R[j], "-", color=MUTED, lw=1.1,
                alpha=0.8, zorder=1)
    for i in range(k):
        ax.plot(i + jit, R[:, i], "o", ms=8.5, color=cols[i],
                mec="white", mew=1.0, zorder=3)
        ax.text(i, 4.75, f"$R_i$ = {R[:, i].sum():.0f}", fontsize=10.5,
                color=cols[i], ha="center", va="center")
    ax.set_xticks(np.arange(k))
    ax.set_xticklabels(drugs, fontsize=10.5)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlim(-0.55, k - 0.45)
    ax.set_ylim(0.5, 5.25)
    clean(ax)
    ax.set_ylabel("환자 안에서의 순위", fontsize=10.5, color=INK)
    ax.text(-0.45, 4.25,
            f"Friedman $\\chi^2_F$ = {chi:.2f},  정확 $p$ = 0.000139\n"
            f"Kruskal-Wallis $H$ = {h:.2f},  $p$ = {p_kw:.4f}",
            fontsize=10.5, color=INK, ha="left", va="top")
    ax.set_title("(나) 블록 내 순위 — 다섯 환자가 같은 이야기를 한다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    fig.suptitle("블록 안에서 순위를 매기는 순간 환자별 기준선 차이가 사라진다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "friedman_blocks.png")
    print("R", R.sum(0), "chi %.3f p_f %.5f  H %.3f p_kw %.5f"
          % (chi, p_f, h, p_kw))


# ==================================================================
# 4. 한 비트만 쓸 때의 검정력 값  (median_test.md)
# ==================================================================
def mood_vs_kw_power():
    A = np.array([3.2, 4.5, 2.8, 5.1, 3.9, 4.2])
    Bd = np.array([1.5, 2.0, 3.0, 2.5, 1.8, 2.2])
    C = np.array([4.0, 5.5, 3.5, 6.0, 4.8, 5.0])
    groups = [("식단 A", A, BLUE), ("식단 B", Bd, ORANGE), ("식단 C", C, GREEN)]
    gm = np.median(np.concatenate([A, Bd, C]))
    mt = stats.median_test(A, Bd, C)
    h, p_kw = stats.kruskal(A, Bd, C)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (가) 자료와 전체 중앙값
    ax = axes[0]
    for i, (name, v, c) in enumerate(groups):
        y = 2 - i
        above = v > gm
        ax.plot(v[~above], np.full((~above).sum(), y), "o", ms=11,
                color="white", mec=c, mew=2.0)
        ax.plot(v[above], np.full(above.sum(), y), "o", ms=11, color=c,
                mec="white", mew=1.0)
        ax.text(0.9, y, name, fontsize=11.5, color=c, ha="right", va="center")
        ax.text(6.35, y, f"위 {int(above.sum())} / 아래 {int((~above).sum())}",
                fontsize=11, color=c, ha="left", va="center")
    ax.axvline(gm, color=RED, lw=1.8, ls=(0, (5, 3)))
    ax.text(gm + 0.08, 2.78, f"전체 중앙값 {gm}", fontsize=11, color=RED,
            ha="left", va="center")
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-0.95, 3.0)
    ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("체중 감량 (kg)", fontsize=10.5, color=INK)
    ax.text(-0.6, -0.62,
            f"Mood $\\chi^2$ = {mt.statistic:.2f},  $p$ = {mt.pvalue:.4f}"
            f"        Kruskal-Wallis $H$ = {h:.2f},  $p$ = {p_kw:.4f}",
            fontsize=11, color=INK, ha="left", va="center")
    ax.set_title("(가) 속이 빈 점은 중앙값 아래, 찬 점은 위 — 검정이 보는 전부다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 위치이동 대립가설에서의 검정력
    ax = axes[1]
    rng = np.random.default_rng(3)
    deltas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    n, Bn = 15, 2500
    pk, pm = [], []
    for dl in deltas:
        ck = cm = 0
        for _ in range(Bn):
            g1 = rng.normal(0, 1, n)
            g2 = rng.normal(dl, 1, n)
            g3 = rng.normal(2 * dl, 1, n)
            ck += stats.kruskal(g1, g2, g3).pvalue < 0.05
            cm += stats.median_test(g1, g2, g3)[1] < 0.05
        pk.append(ck / Bn)
        pm.append(cm / Bn)
    ax.plot(deltas, pk, "-o", color=BLUE, lw=2.0, ms=6,
            label="Kruskal-Wallis")
    ax.plot(deltas, pm, "-s", color=ORANGE, lw=2.0, ms=6, label="Mood 중앙값")
    ax.axhline(0.05, color=RED, lw=1.2, ls=(0, (5, 3)))
    ax.text(1.21, 0.072, "$\\alpha = 0.05$", fontsize=10, color=RED,
            ha="right", va="bottom")
    ax.set_ylim(0, 1.05)
    clean(ax)
    ax.set_xlabel("집단 간 위치 간격 $\\delta$  (평균 $0,\\ \\delta,\\ 2\\delta$)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("기각률", fontsize=10.5, color=INK)
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax.set_title("(나) 집단당 $n = 15$ 에서의 검정력",
                 fontsize=11.5, color=INK, loc="left", pad=8)
    for dd, a, bb in zip(deltas, pk, pm):
        if abs(dd - 0.6) < 1e-9:
            ax.annotate(f"{a:.3f}", xy=(dd, a), xytext=(0, 10),
                        textcoords="offset points", fontsize=10, color=BLUE,
                        ha="center")
            ax.annotate(f"{bb:.3f}", xy=(dd, bb), xytext=(0, -18),
                        textcoords="offset points", fontsize=10, color=ORANGE,
                        ha="center")
            print("delta=0.6 KW %.3f Mood %.3f" % (a, bb))

    fig.suptitle("Mood 중앙값검정은 단순한 대신 검정력을 내준다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "mood_vs_kw_power.png")
    print("grand median", gm, "table\n", mt.table,
          "mood chi2 %.4f p %.5f  KW H %.4f p %.5f"
          % (mt.statistic, mt.pvalue, h, p_kw))
    print("KW power", np.round(pk, 3), "Mood power", np.round(pm, 3))


if __name__ == "__main__":
    kw_rank_spread()
    dunn_fwer()
    friedman_blocks()
    mood_vs_kw_power()
