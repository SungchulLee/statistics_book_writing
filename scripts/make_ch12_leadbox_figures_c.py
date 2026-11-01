"""12장 들머리 그림 일곱 장 (가정 넷 · 웰치 셋).

만드는 파일:
  ch12/assumptions/img/what_you_can_still_fix.png   받은 뒤에 고칠 수 있는 것
  ch12/assumptions/img/what_w_does_not_see.png      W 가 보지 않는 것
  ch12/assumptions/img/s_itself_wobbles.png         s 자체가 흔들린다
  ch12/assumptions/img/against_what_you_plot.png    무엇에 대해 그리는가
  ch12/anova_welch/img/where_the_decimal_comes.png  소수점은 어디서 오나
  ch12/anova_welch/img/pairing_sets_direction.png   방향을 정하는 것은 짝이다
  ch12/anova_welch/img/pooling_cuts_both_ways.png   합동은 양쪽으로 틀어진다

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

ASM = "docs/ch12/assumptions/img/"
WEL = "docs/ch12/anova_welch/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def classic_f(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups])
    m = np.array([g.mean() for g in groups])
    gm = np.concatenate(groups).mean()
    ssb = (n * (m - gm) ** 2).sum()
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df1, df2 = k - 1, n.sum() - k
    f = (ssb / df1) / (ssw / df2)
    return stats.f.sf(f, df1, df2)


def welch_f(groups):
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    mw = (w * m).sum() / w.sum()
    a = ((w * (m - mw) ** 2).sum() / (k - 1))
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    f = a / b
    df2 = (k ** 2 - 1) / (3 * lam)
    return stats.f.sf(f, k - 1, df2), df2


# ════════════════════════════════════════════════════════════════════
def fig_what_you_can_still_fix():
    """12.4 가정 개요 — 자료를 받은 뒤에 고칠 수 있는 것과 없는 것.

    쪽의 assumption_costs 는 네 위반의 기각률을, pseudoreplication 상자는
    수조를 세는 이야기를 다룬다. 여기서는 들머리의 넷째 문단 ---
    **독립성은 설계가 정하고 나면 되돌릴 수 없다** --- 만 본다. 군집이
    섞인 자료에서 군집 이름표를 가지고 있을 때와 없을 때를 가른다.
    """
    K, N, REPS = 3, 20, 4000
    rng = np.random.default_rng(1240)

    def heavy(gen, shape):
        return gen.standard_t(3, shape) / np.sqrt(3)

    def make(kind, gen):
        if kind == "정규성":
            return [heavy(gen, N) for _ in range(K)], None
        if kind == "등분산":
            sds = [1.0, 2.0, 4.0]
            ns = [30, 20, 10]
            return [gen.normal(0, s, n) for s, n in zip(sds, ns)], None
        # 군집 --- 집단마다 군집 다섯씩, 군집 안이 닮았다
        g, cl = [], []
        for _ in range(K):
            c = gen.normal(0, 0.9, 5)
            y = np.repeat(c, 4) + gen.normal(0, 0.4, 20)
            g.append(y)
            cl.append(np.repeat(np.arange(5), 4))
        return g, cl

    def cluster_p(groups, cl):
        """군집 평균을 관측으로 보고 다시 검정한다."""
        gg = [np.array([g[c == j].mean() for j in np.unique(c)])
              for g, c in zip(groups, cl)]
        return classic_f(gg)

    rows = []
    for kind in ("정규성", "등분산", "군집"):
        raw = fixed = 0
        for _ in range(REPS):
            g, cl = make(kind, rng)
            raw += classic_f(g) < 0.05
            if kind == "정규성":
                fixed += stats.kruskal(*g).pvalue < 0.05
            elif kind == "등분산":
                fixed += welch_f(g)[0] < 0.05
            else:
                fixed += cluster_p(g, cl) < 0.05
        rows.append((kind, raw / REPS, fixed / REPS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    xs = np.arange(len(rows))
    w = 0.3
    ax1.bar(xs - w / 2, [100 * r[1] for r in rows], width=w * 0.92,
            color="tab:red", alpha=0.9, label="고전 $F$ 를 그대로")
    ax1.bar(xs + w / 2, [100 * r[2] for r in rows], width=w * 0.92,
            color="tab:blue", alpha=0.9, label="그 위반에 맞는 처방")
    for x_, r in zip(xs, rows):
        ax1.text(x_ - w / 2, 100 * r[1] + 0.6, f"{100 * r[1]:.1f}",
                 ha="center", va="bottom", fontsize=11, color="tab:red",
                 fontweight="bold")
        ax1.text(x_ + w / 2, 100 * r[2] + 0.6, f"{100 * r[2]:.1f}",
                 ha="center", va="bottom", fontsize=11, color="tab:blue",
                 fontweight="bold")
    ax1.axhline(5, color="0.2", ls="--", lw=1.8)
    ax1.set_xticks(xs, ["정규성이 깨졌다\n(꼬리가 두껍다)",
                        "등분산이 깨졌다\n(불균형이 겹쳤다)",
                        "독립성이 깨졌다\n(군집이 섞였다)"], fontsize=10.5)
    ax1.set_ylim(0, max(100 * r[1] for r in rows) * 1.3)
    ax1.set_ylabel("참인 귀무가설의 기각률 (%)", fontsize=11)
    ax1.set_title(f"평균이 모두 같은 자료 ({REPS:,} 번)", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.5, 0.04, "정규성은 안전한 쪽으로, 나머지 둘은 "
             "거짓 양성 쪽으로 기운다", transform=ax1.transAxes,
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    cl_raw, cl_fix = rows[2][1], rows[2][2]
    ys = np.arange(3)
    vals = [cl_raw, cl_raw, cl_fix]
    cols = ["tab:red", "tab:red", "tab:blue"]
    labs = ["군집을 적어 두지 않았다\n— 고칠 길이 없다",
            "뒤늦게 알아챘다\n— 그래도 고칠 길이 없다",
            "군집 이름표가 있다\n— 모형에 적어 넣는다"]
    ax2.barh(ys[::-1], [100 * v for v in vals], height=0.5, color=cols,
             alpha=0.9)
    for y_, v in zip(ys[::-1], vals):
        ax2.text(100 * v + 0.6, y_, f"{100 * v:.1f}%", va="center",
                 ha="left", fontsize=12, fontweight="bold", color="0.15")
    ax2.axvline(5, color="0.2", ls="--", lw=1.8)
    ax2.set_yticks(ys[::-1], labs, fontsize=10.5)
    ax2.set_xlim(0, max(100 * cl_raw * 1.35, 12))
    ax2.set_xlabel("기각률 (%)", fontsize=11)
    ax2.set_title("독립성만은 설계가 정하고 나면 되돌릴 수 없다",
                  fontsize=12.5)
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("같은 「가정 위반」 아래에서 틀어지는 방향이 서로 반대다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, ASM + "what_you_can_still_fix.png")

    print(f"\n[가정 개요] 각 {REPS:,} 번")
    for kind, a, b in rows:
        print(f"  {kind:<6s} 고전 {100*a:5.1f}%  처방 뒤 {100*b:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_what_w_does_not_see():
    """12.4 정규성 — 같은 잔차를 섞어도 W 는 꿈쩍하지 않는다.

    쪽의 quantization_tails 는 눈금에 걸린 꼬리를 다룬다. 여기서는
    들머리가 드는 대조 --- **잔차를 1000 번 섞으면 W 는 10^-15 규모로
    한 값인데 더빈–왓슨은 1.16 에서 2.69 까지 움직인다** --- 와, 이
    검정이 꼬리를 잡아내는 비율만 본다.
    """
    N, PERM = 60, 1000
    rng = np.random.default_rng(1241)
    res = rng.normal(0, 1, N)
    res = res - res.mean()

    ws, dws = [], []
    for _ in range(PERM):
        p = rng.permutation(res)
        ws.append(stats.shapiro(p).statistic)
        d = np.diff(p)
        dws.append((d ** 2).sum() / (p ** 2).sum())
    ws, dws = np.array(ws), np.array(dws)

    NS = [20, 40, 60, 100, 200, 400]
    REPS = 3000
    pw = []
    for n in NS:
        hit = 0
        for _ in range(REPS):
            y = rng.standard_t(10, n)
            hit += stats.shapiro(y).pvalue < 0.05
        pw.append(hit / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.scatter(dws, ws, s=16, color="0.6", alpha=0.6, edgecolors="none")
    ax1.axhline(ws.mean(), color="tab:blue", lw=2.4)
    ax1.set_xlabel("더빈–왓슨 통계량", fontsize=11)
    ax1.set_ylabel("샤피로–윌크 $W$", fontsize=11)
    ax1.set_ylim(ws.mean() - 2e-15, ws.mean() + 2e-15)
    ax1.set_yticks([ws.mean() - 1e-15, ws.mean(), ws.mean() + 1e-15],
                   [f"{ws.mean():.12f}\n(빼기 1e-15)", f"{ws.mean():.12f}",
                    f"{ws.mean():.12f}\n(더하기 1e-15)"], fontsize=8.5)
    ax1.set_title(f"같은 잔차 {N} 개를 {PERM:,} 번 섞었다", fontsize=12.5)
    ax1.grid(alpha=0.25)
    ax1.text(0.5, 0.92, f"$W$ 는 {ws.max() - ws.min():.1e} 규모로 한 값인데\n"
             f"더빈–왓슨은 {dws.min():.4f} 에서 {dws.max():.4f} 까지 움직인다",
             transform=ax1.transAxes, ha="center", va="top", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(NS, [100 * v for v in pw], "o-", ms=9, lw=2.8, color="tab:red")
    for n, v in zip(NS, pw):
        ax2.text(n, 100 * v + 2.5, f"{100 * v:.1f}%", ha="center",
                 va="bottom", fontsize=10.5, color="tab:red",
                 fontweight="bold")
    i60 = NS.index(60)
    ax2.plot([60], [100 * pw[i60]], "o", ms=14, mfc="none", mec="tab:red",
             mew=2.4)
    ax2.set_xscale("log")
    ax2.set_xticks(NS, [str(v) for v in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("표본 크기 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("꼬리가 두꺼운 자료를 잡아낼 비율 (%)", fontsize=11)
    ax2.set_title(f"자유도 $10$ 인 $t$ 자료를 {REPS:,} 번 재면",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("통과한 $p$ 값은 무엇을 확인해 준 것인가", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, ASM + "what_w_does_not_see.png")

    print(f"\n[정규성] 잔차 {N} 개를 {PERM:,} 번 섞으면")
    print(f"  W 의 폭 {ws.max() - ws.min():.3e} · 더빈–왓슨 "
          f"{dws.min():.4f}~{dws.max():.4f}")
    for n, v in zip(NS, pw):
        print(f"  n = {n:>4d}  꼬리를 잡아낼 비율 {100 * v:.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_s_itself_wobbles():
    """12.4 등분산 — s 자체가 크게 흔들려 구별할 힘이 없다.

    쪽의 robust_design 은 로버스트 설계를 다룬다. 여기서는 들머리 ---
    **참 비 1.6 이 표본에서 1.3158 까지 눌린다** --- 만 본다. 같은
    설정에서 표본 표준편차의 비가 어떻게 흩어지는지 세고, 르빈이
    기각할 수 있는 표본 크기를 잰다.
    """
    SD = np.array([1.0, 1.3, 1.6])
    N_PAGE, REPS = 20, 20000
    rng = np.random.default_rng(1242)

    ratio = np.array([
        (lambda g: max(g) / min(g))(
            [x.std(ddof=1) for x in rng.normal(0, SD[:, None], (3, N_PAGE))])
        for _ in range(REPS)])

    NS = [20, 40, 80, 160, 320, 640]
    pw = []
    for n in NS:
        hit = 0
        for _ in range(2000):
            g = [rng.normal(0, s, n) for s in SD]
            hit += stats.levene(*g, center="median").pvalue < 0.05
        pw.append(hit / 2000)

    OBS = 1.145 / 0.870
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.hist(ratio, bins=100, color="0.65", alpha=0.9)
    ax1.axvline(SD.max() / SD.min(), color="tab:blue", lw=2.6)
    ax1.text(SD.max() / SD.min() + 0.03, ax1.get_ylim()[1] * 0.95,
             f"참 비 {SD.max() / SD.min():.1f}", ha="left", va="top",
             fontsize=11, color="tab:blue", fontweight="bold")
    ax1.axvline(OBS, color="tab:red", lw=2.6)
    ax1.text(OBS - 0.03, ax1.get_ylim()[1] * 0.80,
             f"이 쪽의 표본 {OBS:.4f}", ha="right", va="top", fontsize=11,
             color="tab:red", fontweight="bold")
    ax1.set_xlim(1, 4)
    ax1.set_xlabel("표본표준편차의 최대 ÷ 최소", fontsize=11)
    ax1.set_ylabel("횟수", fontsize=11)
    ax1.set_title(f"참 표준편차 {tuple(SD)} · 집단당 {N_PAGE} ({REPS:,} 번)",
                  fontsize=12)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.97, 0.55, f"참 비보다 작게 나올 확률 "
             f"{100 * (ratio < SD.max() / SD.min()).mean():.1f}%\n"
             f"중앙값 {np.median(ratio):.3f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=11.5, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(NS, [100 * v for v in pw], "o-", ms=9, lw=2.8,
             color="tab:purple")
    for n, v in zip(NS, pw):
        ax2.text(n, 100 * v + 2.5, f"{100 * v:.0f}%", ha="center",
                 va="bottom", fontsize=10.5, color="tab:purple",
                 fontweight="bold")
    ax2.axhline(80, color="0.3", ls="--", lw=1.8)
    ax2.text(640, 82, "열에 여덟", ha="right", va="bottom", fontsize=10.5,
             color="0.3", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xticks(NS, [str(v) for v in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(0, 108)
    ax2.set_xlabel("집단당 표본 크기 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("르빈이 등분산을 기각하는 비율 (%)", fontsize=11)
    ax2.set_title("이 정도 차이를 잡아내려면", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("검정이 틀린 것이 아니라 $s$ 자체가 크게 흔들리는 것이다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, ASM + "s_itself_wobbles.png")

    print(f"\n[등분산] 참 표준편차 {SD.tolist()} · 집단당 {N_PAGE}")
    print(f"  표본 비의 중앙값 {np.median(ratio):.4f} · 참 비보다 작을 확률 "
          f"{100 * (ratio < SD.max() / SD.min()).mean():.2f}%")
    for n, v in zip(NS, pw):
        print(f"  집단당 {n:>4d}  르빈의 기각률 {100 * v:.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_against_what_you_plot():
    """12.4 선형성 — 잔차를 무엇에 대해 그리는가.

    쪽의 linear_probe 는 선형성 탐침을 다룬다. 여기서는 들머리의 둘째
    문단 --- **공변량에 대해 보면 이차항이 p = 0.0004 인데 적합값에
    대해 보면 0.0245 로 흐려진다** --- 만 본다. 집단 효과가 섞인
    적합값에 대해 그리면 휘어짐이 집단 사이에 흩어져 묻힌다.
    """
    K, N, REPS = 3, 40, 2000
    CURVE = np.linspace(0, 0.9, 10)
    rng = np.random.default_rng(1243)
    EFF = np.array([0.0, 2.5, 5.0])

    def one(c, gen):
        g = np.repeat(np.arange(K), N)
        x = gen.uniform(-2, 2, K * N)
        y = EFF[g] + 1.2 * x + c * x ** 2 + gen.normal(0, 1.2, K * N)
        M = np.column_stack([np.ones(K * N)] +
                            [(g == j).astype(float) for j in (1, 2)] + [x])
        b = np.linalg.lstsq(M, y, rcond=None)[0]
        r = y - M @ b
        fit = M @ b

        def quad_p(v):
            A = np.column_stack([np.ones(len(v)), v, v ** 2])
            bb = np.linalg.lstsq(A, r, rcond=None)[0]
            res = r - A @ bb
            s2 = (res ** 2).sum() / (len(v) - 3)
            se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[2, 2])
            return 2 * stats.t.sf(abs(bb[2] / se), len(v) - 3)
        return quad_p(x), quad_p(fit)

    pw_x, pw_f = [], []
    for c in CURVE:
        rows = np.array([one(c, rng) for _ in range(REPS)])
        pw_x.append((rows[:, 0] < 0.05).mean())
        pw_f.append((rows[:, 1] < 0.05).mean())

    C_SHOW = 0.5
    g = np.repeat(np.arange(K), N)
    x = rng.uniform(-2, 2, K * N)
    y = EFF[g] + 1.2 * x + C_SHOW * x ** 2 + rng.normal(0, 1.2, K * N)
    M = np.column_stack([np.ones(K * N)] +
                        [(g == j).astype(float) for j in (1, 2)] + [x])
    b = np.linalg.lstsq(M, y, rcond=None)[0]
    r = y - M @ b
    fit = M @ b

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    COLS = ["tab:blue", "tab:orange", "tab:green"]
    for ax, v, nm in ((axes[0], x, "공변량 $x$ 에 대해"),
                      (axes[1], fit, "적합값에 대해")):
        for j in range(K):
            ax.scatter(v[g == j], r[g == j], s=16, color=COLS[j], alpha=0.55,
                       edgecolors="none")
        o = np.argsort(v)
        p = np.polyfit(v, r, 2)
        ax.plot(v[o], np.polyval(p, v[o]), "-", lw=3.0, color="0.15")
        ax.axhline(0, color="0.5", lw=1.2)
        ax.set_xlabel(nm.replace("에 대해", ""), fontsize=11)
        ax.set_title(nm, fontsize=12.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("잔차", fontsize=11)
    axes[1].text(0.5, 0.04, "집단 효과가 섞여 들어와\n휘어짐이 흩어진다",
                 transform=axes[1].transAxes, ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="0.15",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           alpha=0.94, edgecolor="0.5", lw=1.0))

    ax = axes[2]
    ax.plot(CURVE, [100 * v for v in pw_x], "o-", ms=8, lw=2.6,
            color="tab:blue", label="공변량에 대해 그린다")
    ax.plot(CURVE, [100 * v for v in pw_f], "s-", ms=8, lw=2.6,
            color="tab:red", label="적합값에 대해 그린다")
    ax.axhline(5, color="0.3", ls=":", lw=1.6)
    ax.set_xlabel("참 관계의 휘어짐 (이차항의 계수)", fontsize=11)
    ax.set_ylabel("이차항을 잡아낼 비율 (%)", fontsize=11)
    ax.set_title(f"같은 자료, 가로축만 바꿨다 (각 {REPS:,} 번)",
                 fontsize=12.5)
    ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.suptitle("선형성은 무엇에 대해 그리느냐가 정한다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, ASM + "against_what_you_plot.png")

    print(f"\n[선형성] 각 {REPS:,} 번")
    for c, a, b_ in zip(CURVE, pw_x, pw_f):
        print(f"  휘어짐 {c:.2f}  공변량 {100*a:5.1f}%  적합값 {100*b_:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_where_the_decimal_comes():
    """12.5 웰치 일원배치 — 깎인 자유도와 그 소수점.

    쪽의 welch_weights 는 가중치를, seed_variance 는 자료를 다룬다.
    여기서는 들머리의 물음 --- **깎인 자유도는 무엇을 치른 값이고 그
    소수점은 어디서 오는가** --- 에 답한다. 웰치의 분모 자유도는
    분산이 고르면 N-k 에 닿고 한 집단이 전부를 차지하면 그 집단의
    n-1 로 내려간다.
    """
    NS = np.array([4, 3, 3])          # 관측 10 개를 셋으로
    K = len(NS)
    ts = np.logspace(0, 2, 300)       # 분산의 흩어진 정도

    def welch_df(v, n):
        w = n / v
        lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
        return (K ** 2 - 1) / (3 * lam)

    dfs = np.array([welch_df(np.array([1.0, t, t ** 2]), NS.astype(float))
                    for t in ts])
    DF_PAGE = 4.142737

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(ts, dfs, "-", lw=2.8, color="tab:purple")
    ax1.axhline(NS.sum() - K, color="tab:blue", ls="--", lw=2.2)
    ax1.text(100, NS.sum() - K + 0.1, f"분산이 고를 때의 위 끝  "
             f"$N - k = {NS.sum() - K}$", ha="right", va="bottom",
             fontsize=10.5, color="tab:blue", fontweight="bold")
    ax1.axhline(NS.min() - 1, color="tab:red", ls="--", lw=2.2)
    ax1.text(100, NS.min() - 1 + 0.08, f"한 집단이 전부를 차지할 때의 "
             f"아래 끝  ${NS.min() - 1}$", ha="right", va="bottom",
             fontsize=10.5, color="tab:red", fontweight="bold")
    ax1.axhline(DF_PAGE, color="0.25", ls=":", lw=2.0)
    ax1.text(1.05, DF_PAGE + 0.08, f"이 쪽의 자료 {DF_PAGE:.6f}",
             ha="left", va="bottom", fontsize=11, color="0.2",
             fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax1.set_xticks([], minor=True)
    ax1.set_ylim(1.5, 7.5)
    ax1.set_xlabel("분산이 흩어진 정도 (1 이면 모두 같다, 로그 눈금)",
                   fontsize=11)
    ax1.set_ylabel("웰치의 분모 자유도", fontsize=11)
    ax1.set_title(f"관측 {NS.sum()} 개를 {tuple(NS)} 로 나눈 설계",
                  fontsize=12.5)
    ax1.grid(alpha=0.25)

    nn = np.arange(3, 41)
    curve = []
    for n in nn:
        ns = np.full(K, n, float)
        curve.append(welch_df(np.array([1.0, 9.0, 36.0]), ns) /
                     (K * n - K))
    ax2.plot(nn, 100 * np.array(curve), "-", lw=2.8, color="tab:purple")
    ax2.set_xlabel("집단당 표본 크기", fontsize=11)
    ax2.set_ylabel("웰치의 자유도 ÷ 고전의 자유도 (%)", fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.set_title("분산비를 $1:9:36$ 으로 못 박고 표본만 키우면",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.06, "표본이 커져도 비는 거의 그대로다\n"
             "— 깎이는 몫은 분산의 흩어짐이 정한다",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=11,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("자료의 개수는 그대로인데 분산이 다르다는 사실이 "
                 "자유도를 깎는다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, WEL + "where_the_decimal_comes.png")

    print(f"\n[웰치 일원배치] 설계 {NS.tolist()} · 위 끝 {NS.sum() - K} · "
          f"아래 끝 {NS.min() - 1}")
    for t in (1, 2, 4, 8, 20, 100):
        print(f"  흩어진 정도 {t:>4}  자유도 "
              f"{welch_df(np.array([1.0, t, t ** 2]), NS.astype(float)):.6f}")


# ════════════════════════════════════════════════════════════════════
def fig_pairing_sets_direction():
    """12.5 웰치 모의실험 — 방향을 정하는 것은 분산과 표본의 짝이다.

    쪽의 welch_size_direction 은 네 배치를 나란히 놓는다. 여기서는 그
    넷 사이를 **이어서** 본다. 분산이 큰 집단에 표본을 얼마나 주느냐를
    연속으로 바꾸면 고전 F 의 실제 유의수준이 0.05 를 가로지른다.
    """
    SD = np.array([1.0, 3.0, 6.0])
    TOTAL, REPS = 60, 4000
    rng = np.random.default_rng(1244)
    TILTS = np.linspace(-1, 1, 9)

    def alloc(t):
        w = np.exp(t * np.array([-1.0, 0.0, 1.0]))
        n = np.maximum(np.round(TOTAL * w / w.sum()), 5).astype(int)
        return n

    lev_c, lev_w = [], []
    for t in TILTS:
        n = alloc(t)
        c = w = 0
        for _ in range(REPS):
            g = [rng.normal(0, s, m) for s, m in zip(SD, n)]
            c += classic_f(g) < 0.05
            w += welch_f(g)[0] < 0.05
        lev_c.append(c / REPS)
        lev_w.append(w / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    ax1.plot(TILTS, [100 * v for v in lev_c], "o-", ms=9, lw=2.8,
             color="tab:red", label="고전 $F$")
    ax1.plot(TILTS, [100 * v for v in lev_w], "D-", ms=8, lw=2.8,
             color="tab:blue", label="웰치")
    ax1.axhline(5, color="0.2", ls="--", lw=1.8)
    ax1.axvline(0, color="0.7", lw=1.2)
    ax1.set_xlabel("분산이 큰 집단에 표본을 더 주는 정도", fontsize=11)
    ax1.set_ylabel("참인 귀무가설의 기각률 (%)", fontsize=11)
    ax1.set_title(f"표준편차 {tuple(SD.astype(int))} · 모두 {TOTAL} 개 "
                  f"(각 {REPS:,} 번)", fontsize=12)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(-0.95, 2.0, "분산이 큰 집단에\n표본이 적게 간다",
             ha="left", va="bottom", fontsize=10.5, color="tab:red",
             fontweight="bold")
    ax1.text(0.95, 2.0, "많이 간다", ha="right", va="bottom", fontsize=10.5,
             color="0.3", fontweight="bold")

    i0, i1 = 0, len(TILTS) - 1
    xs = np.arange(2)
    ax2.bar(xs - 0.17, [100 * lev_c[i0], 100 * lev_c[i1]], width=0.32,
            color="tab:red", alpha=0.9, label="고전 $F$")
    ax2.bar(xs + 0.17, [100 * lev_w[i0], 100 * lev_w[i1]], width=0.32,
            color="tab:blue", alpha=0.9, label="웰치")
    for x_, a, b in zip(xs, (lev_c[i0], lev_c[i1]), (lev_w[i0], lev_w[i1])):
        ax2.text(x_ - 0.17, 100 * a + 0.4, f"{100 * a:.1f}", ha="center",
                 va="bottom", fontsize=11.5, color="tab:red",
                 fontweight="bold")
        ax2.text(x_ + 0.17, 100 * b + 0.4, f"{100 * b:.1f}", ha="center",
                 va="bottom", fontsize=11.5, color="tab:blue",
                 fontweight="bold")
    ax2.axhline(5, color="0.2", ls="--", lw=1.8)
    ax2.set_xticks(xs, [f"큰 분산에 적게\n{tuple(alloc(TILTS[i0]))}",
                        f"큰 분산에 많이\n{tuple(alloc(TILTS[i1]))}"],
                   fontsize=11)
    ax2.set_ylim(0, max(100 * max(lev_c), 25) * 1.2)
    ax2.set_ylabel("기각률 (%)", fontsize=11)
    ax2.set_title("두 끝만 떼어 보면", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("「부풀려진다」는 말에 방향이 없다 — 양쪽으로 틀어진다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, WEL + "pairing_sets_direction.png")

    print(f"\n[웰치 모의실험] 표준편차 {SD.tolist()} · 모두 {TOTAL} 개")
    for t, a, b in zip(TILTS, lev_c, lev_w):
        print(f"  기울기 {t:+.2f}  배치 {tuple(alloc(t))}  고전 "
              f"{a:.3f}  웰치 {b:.3f}")


# ════════════════════════════════════════════════════════════════════
def fig_pooling_cuts_both_ways():
    """12.5 이원배치 로버스트 — 합동이 늘 부풀리는 쪽은 아니다.

    쪽의 welch_hc3_vs_classic 은 한 자료에서 고전과 HC3 가 갈리는 모습을
    그린다. 여기서는 들머리의 넷째 문단 --- **분산이 작은 칸이 많이
    쓰이는 항에서는 반대로 분모가 커져 효과를 놓친다** --- 를 두 설계로
    갈라 보인다.
    """
    A, B, REP, REPS = 3, 3, 2, 3000
    rng = np.random.default_rng(1245)
    N = A * B * REP
    EFF = np.array([0.0, 1.0, 2.0])

    def one(where, gen):
        """수준 1 과 2 의 대비를 잰다. 분산이 큰 칸을 그 대비 **안**에
        둘 때와 **밖**(수준 3)에 둘 때를 가른다."""
        a = np.repeat(np.arange(A), B * REP)
        b = np.tile(np.repeat(np.arange(B), REP), A)
        hot = 0 if where == "in" else 2
        sd = np.where(a == hot, 4.0, 0.7)
        y = EFF[a] + gen.normal(0, sd, N)
        M = np.column_stack([np.ones(N)] +
                            [(a == j).astype(float) for j in (1, 2)] +
                            [(b == j).astype(float) for j in (1, 2)])
        beta = np.linalg.lstsq(M, y, rcond=None)[0]
        res = y - M @ beta
        p = M.shape[1]
        s2 = (res ** 2).sum() / (N - p)
        XtXi = np.linalg.inv(M.T @ M)
        h = np.einsum("ij,jk,ik->i", M, XtXi, M)
        om = (res / (1 - h)) ** 2
        V_hc3 = XtXi @ (M.T * om) @ M @ XtXi
        R = np.zeros((1, p))
        R[0, 1] = 1.0                      # 수준 1 대 수준 2
        def wald(V):
            d = R @ beta
            return float(d @ np.linalg.solve(R @ V @ R.T, d))
        return wald(s2 * XtXi), wald(V_hc3)

    out = {}
    for where in ("in", "out"):
        rows = np.array([one(where, rng) for _ in range(REPS)])
        out[where] = rows.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(2)
    w = 0.3
    cls = [out["in"][0], out["out"][0]]
    hc3 = [out["in"][1], out["out"][1]]
    ax1.bar(xs - w / 2, cls, width=w * 0.92, color="tab:red", alpha=0.9,
            label="고전 (합동분산 하나)")
    ax1.bar(xs + w / 2, hc3, width=w * 0.92, color="tab:blue", alpha=0.9,
            label="HC3 로버스트")
    for x_, a, b in zip(xs, cls, hc3):
        ax1.text(x_ - w / 2, a + 0.1, f"{a:.2f}", ha="center", va="bottom",
                 fontsize=11.5, color="tab:red", fontweight="bold")
        ax1.text(x_ + w / 2, b + 0.1, f"{b:.2f}", ha="center", va="bottom",
                 fontsize=11.5, color="tab:blue", fontweight="bold")
    ax1.set_xticks(xs, ["대비 안에\n분산이 큰 칸이 있다",
                        "대비 밖에\n분산이 큰 칸이 있다"], fontsize=11)
    ax1.set_ylabel("수준 $1$ 대 $2$ 를 재는 Wald 통계량의 평균",
                   fontsize=11)
    ax1.set_ylim(0, max(max(cls), max(hc3)) * 1.42)
    ax1.set_title(f"칸 표준편차 $0.7$ 과 $4.0$ (각 {REPS:,} 번)",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper center", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    r = [cls[0] / hc3[0], cls[1] / hc3[1]]
    cols = ["tab:red" if v > 1 else "tab:blue" for v in r]
    ax2.bar(xs, r, width=0.46, color=cols, alpha=0.9)
    ax2.axhline(1, color="0.15", lw=2.0)
    for x_, v in zip(xs, r):
        ax2.text(x_, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=13, fontweight="bold", color="0.15")
    ax2.set_xticks(xs, ["분산이 큰 칸이\n많이 쓰인다", "적게 쓰인다"],
                   fontsize=11.5)
    ax2.set_ylim(0, max(r) * 1.3)
    ax2.set_ylabel("고전 ÷ 로버스트", fontsize=11)
    ax2.set_title("$1$ 보다 위면 합동이 부풀린 것이다", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.97, 0.95, "「로버스트로 바꾸면 보수적이 된다」는\n"
             "어림짐작은 맞지 않는다", transform=ax2.transAxes,
             ha="right", va="top", fontsize=11.5, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("합동이 늘 부풀리는 쪽인 것은 아니다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, WEL + "pooling_cuts_both_ways.png")

    print(f"\n[이원배치 로버스트] 각 {REPS:,} 번")
    for nm, v in out.items():
        print(f"  {nm:<6s} 고전 {v[0]:.4f}  HC3 {v[1]:.4f}  "
              f"비 {v[0] / v[1]:.4f}")


fig_what_you_can_still_fix()
fig_what_w_does_not_see()
fig_s_itself_wobbles()
fig_against_what_you_plot()
fig_where_the_decimal_comes()
fig_pairing_sets_direction()
fig_pooling_cuts_both_ways()
