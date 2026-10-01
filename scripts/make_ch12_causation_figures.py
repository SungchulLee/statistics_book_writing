r"""12장 인과 여섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch12/causation/img/potential_outcomes_gap.png        잠재결과의 절반은 영원히 비어 있다
  ch12/causation/img/four_structures_same_r.png        같은 r, 네 가지 인과 구조
  ch12/causation/img/simpson_and_partial.png           합치면 뒤집히고, 통제하면 사라진다
  ch12/causation/img/three_structures_conditioning.png 사슬·갈래·충돌자와 조건화
  ch12/causation/img/randomization_balance.png         무작위화는 못 본 변수까지 맞춘다
  ch12/causation/img/iv_vs_ols_strength.png            도구가 약하면 IV 가 OLS 보다 나쁘다

실행:  python3 scripts/make_ch12_causation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch12/causation/img/"
os.makedirs(OUT, exist_ok=True)


# === 공통 도우미 ===
def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def node(ax, xy, label, color=INK, fc="white", r=0.115):
    circ = plt.Circle(xy, r, facecolor=fc, edgecolor=color, lw=1.6, zorder=3)
    ax.add_patch(circ)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=12, color=color, zorder=4)


def arrow(ax, a, b, color=INK, r=0.115, style="-|>", ls="-"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.hypot(*d)
    u = d / L
    ax.add_patch(FancyArrowPatch(a + u * r, b - u * r, arrowstyle=style,
                                 mutation_scale=13, lw=1.5, color=color,
                                 linestyle=ls, zorder=2))


# === 그림 1. 잠재결과의 빈 칸 ===
def fig_potential_outcomes():
    rng = np.random.default_rng(12)
    N = 5000
    y0 = rng.normal(50, 9, N)
    tau = rng.normal(5, 7, N)           # 개인 인과효과: 평균 5, 편차 7
    y1 = y0 + tau
    x = rng.integers(0, 2, N)

    ate = tau.mean()
    frac_neg = float((tau < 0).mean())
    naive = y1[x == 1].mean() - y0[x == 0].mean()

    fig = plt.figure(figsize=(11.2, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)

    # --- 왼쪽: 관측표 ---
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.set_xlim(0, 4.6)
    ax.set_ylim(-0.4, 6.5)

    head = ["개인", "처치", "Y(0)", "Y(1)", "차이"]
    xs = [0.45, 1.35, 2.3, 3.2, 4.1]
    for xc, h in zip(xs, head):
        ax.text(xc, 5.95, h, ha="center", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
    ax.plot([0.05, 4.55], [5.62, 5.62], color=MUTED, lw=1.1)

    for k in range(5):
        yrow = 5.0 - k
        obs_t = int(x[k])
        ax.text(xs[0], yrow, f"{k + 1}", ha="center", va="center",
                fontsize=10.5, color=INK)
        ax.text(xs[1], yrow, "받음" if obs_t else "안 받음", ha="center",
                va="center", fontsize=10.5,
                color=ORANGE if obs_t else BLUE)
        for j, (val, is_obs) in enumerate([(y0[k], obs_t == 0),
                                           (y1[k], obs_t == 1)]):
            xc = xs[2 + j]
            if is_obs:
                ax.add_patch(Rectangle((xc - 0.38, yrow - 0.3), 0.76, 0.6,
                                       facecolor=BLUE_F if j == 0 else ORANGE_F,
                                       edgecolor="none", zorder=1))
                ax.text(xc, yrow, f"{val:.1f}", ha="center", va="center",
                        fontsize=10.5, color=INK, zorder=2)
            else:
                ax.add_patch(Rectangle((xc - 0.38, yrow - 0.3), 0.76, 0.6,
                                       facecolor="#ECEFF1", edgecolor="none",
                                       zorder=1))
                ax.text(xc, yrow, "?", ha="center", va="center", fontsize=12,
                        color=MUTED, zorder=2)
        ax.text(xs[4], yrow, "?", ha="center", va="center", fontsize=12,
                color=MUTED)

    ax.text(2.3, -0.05,
            "한 사람에게서 두 잠재결과를 함께 볼 길은 없다",
            ha="center", va="center", fontsize=10.5, color=RED)
    ax.set_title("개인 인과효과는 관측되지 않는다", fontsize=12.5, color=INK,
                 pad=10)

    # --- 오른쪽: tau 의 분포 ---
    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    bins = np.linspace(tau.min(), tau.max(), 46)
    cnt, edges = np.histogram(tau, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cols = [RED if c < 0 else GREEN for c in centers]
    ax.bar(centers, cnt, width=np.diff(edges) * 0.95, color=cols,
           alpha=0.55, edgecolor="none")
    ax.axvline(0, color=MUTED, lw=1.2, ls=":")
    ax.axvline(ate, color=PURPLE, lw=2.0)
    ymax = cnt.max()
    ax.annotate(f"ATE = {ate:.2f}", xy=(ate, ymax * 0.97),
                xytext=(ate + 5.5, ymax * 1.0), fontsize=11, color=PURPLE,
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.3))
    ax.text(-13.5, ymax * 0.62,
            f"효과가 음수인 사람\n{100 * frac_neg:.1f}%",
            fontsize=10.5, color=RED, ha="center", va="center")
    ax.set_xlabel("개인 인과효과", fontsize=10.5, color=INK)
    ax.set_ylabel("사람 수", fontsize=10.5, color=INK)
    ax.set_title(f"평균은 양수인데 {100 * frac_neg:.0f}% 는 오히려 손해를 본다",
                 fontsize=12.5, color=INK, pad=10)
    ax.set_ylim(0, ymax * 1.22)

    save(fig, "potential_outcomes_gap.png")
    print(f"  ATE={ate:.3f}  tau<0 비율={frac_neg:.3f}  단순차={naive:.3f}")


# === 그림 2. 같은 r, 네 가지 구조 ===
def fig_four_structures():
    rng = np.random.default_rng(7)
    n = 300

    # (a) X -> Y
    xa = rng.normal(0, 1, n)
    ya = 0.98 * xa + rng.normal(0, 1, n)

    # (b) Y -> X
    yb = rng.normal(0, 1, n)
    xb = 0.98 * yb + rng.normal(0, 1, n)

    # (c) Z -> X, Z -> Y
    z = rng.normal(0, 1, n)
    xc = 1.528 * z + rng.normal(0, 1, n)
    yc = 1.528 * z + rng.normal(0, 1, n)

    # (d) 우연 — 독립인 두 변수, n = 12
    for _ in range(4000):
        xd = rng.normal(0, 1, 12)
        yd = rng.normal(0, 1, 12)
        rr, pp = stats.pearsonr(xd, yd)
        if 0.68 < rr < 0.72:
            break

    sets = [("직접 인과", xa, ya, [("X", (0.25, 0.5)), ("Y", (0.75, 0.5))],
             [(0, 1)]),
            ("역인과", xb, yb, [("X", (0.25, 0.5)), ("Y", (0.75, 0.5))],
             [(1, 0)]),
            ("공통원인", xc, yc,
             [("X", (0.2, 0.32)), ("Y", (0.8, 0.32)), ("Z", (0.5, 0.78))],
             [(2, 0), (2, 1)]),
            ("우연", xd, yd, [("X", (0.25, 0.5)), ("Y", (0.75, 0.5))], [])]

    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.52, 1.0], hspace=0.32,
                          wspace=0.26)

    for j, (name, xv, yv, nodes, edges) in enumerate(sets):
        axd = fig.add_subplot(gs[0, j])
        blank(axd)
        axd.set_xlim(0, 1)
        axd.set_ylim(0.05, 1.0)
        axd.set_aspect("auto")
        pos = [p for _, p in nodes]
        for lab, p in nodes:
            node(axd, p, lab, color=INK, r=0.115)
        for a, b in edges:
            arrow(axd, pos[a], pos[b], color=INK, r=0.135)
        if not edges:
            axd.text(0.5, 0.17, "화살표 없음", ha="center", va="center",
                     fontsize=10, color=MUTED)
        axd.set_title(name, fontsize=12.5, color=INK, pad=6)

        r, p = stats.pearsonr(xv, yv)
        ax = fig.add_subplot(gs[1, j])
        clean(ax)
        ax.scatter(xv, yv, s=16 if len(xv) > 50 else 46, color=BLUE,
                   alpha=0.5 if len(xv) > 50 else 0.85,
                   edgecolor="none")
        b, a0 = np.polyfit(xv, yv, 1)
        gx = np.linspace(xv.min(), xv.max(), 20)
        ax.plot(gx, a0 + b * gx, color=ORANGE, lw=1.8)
        ax.set_xlabel("X", fontsize=10, color=INK)
        if j == 0:
            ax.set_ylabel("Y", fontsize=10, color=INK)
        ax.text(0.04, 0.95, f"r = {r:.2f}\nn = {len(xv)}",
                transform=ax.transAxes, fontsize=10.5, color=INK,
                va="top", ha="left")
        print(f"  {name}: r={r:.3f}, n={len(xv)}, p={p:.4f}")

    fig.suptitle("네 구조가 남기는 산점도는 구별되지 않는다",
                 fontsize=13.5, color=INK, y=1.03)
    save(fig, "four_structures_same_r.png")


# === 그림 3. 심슨의 역설과 부분상관 ===
def fig_simpson_and_partial():
    # 본문 보기 3 과 똑같은 난수 순서를 재현한다.
    np.random.seed(42)
    groups = {"A": (50, 0.2, 2, -0.5),
              "B": (50, 0.5, 5, -0.5),
              "C": (50, 0.8, 8, -0.5)}
    gx, gy, names = [], [], []
    for name, (ng, xm, yb, slope) in groups.items():
        xg = np.random.normal(xm, 0.15, ng)
        yg = yb + slope * xg + np.random.normal(0, 0.3, ng)
        gx.append(xg)
        gy.append(yg)
        names.append(name)
    all_x = np.concatenate(gx)
    all_y = np.concatenate(gy)
    m_all, b_all = np.polyfit(all_x, all_y, 1)
    r_all, _ = stats.pearsonr(all_x, all_y)

    # 본문 보기 4 와 똑같은 난수 순서
    np.random.seed(42)
    n = 200
    Z = np.random.normal(0, 1, n)
    X = 0.7 * Z + np.random.normal(0, 0.5, n)
    Y = 0.6 * Z + np.random.normal(0, 0.5, n)
    r_xy, _ = stats.pearsonr(X, Y)
    r_xz, _ = stats.pearsonr(X, Z)
    r_yz, _ = stats.pearsonr(Y, Z)
    r_part = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    eX = X - np.polyval(np.polyfit(Z, X, 1), Z)
    eY = Y - np.polyval(np.polyfit(Z, Y, 1), Z)
    r_res, _ = stats.pearsonr(eX, eY)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    cols = [BLUE, GREEN, PURPLE]

    ax = axes[0]
    clean(ax)
    for xg, yg, c, nm in zip(gx, gy, cols, names):
        ax.scatter(xg, yg, s=20, color=c, alpha=0.65, edgecolor="none")
        mg, bg = np.polyfit(xg, yg, 1)
        t = np.linspace(xg.min(), xg.max(), 12)
        ax.plot(t, bg + mg * t, color=c, lw=2.0)
        ax.text(xg.max() + 0.04, bg + mg * xg.max(), f"집단 {nm}", fontsize=10,
                color=c, ha="left", va="center")
    t = np.linspace(all_x.min(), all_x.max(), 20)
    ax.plot(t, b_all + m_all * t, color=RED, lw=2.4, ls="--")
    ax.set_xlim(-0.15, 1.85)
    ax.text(1.02, 2.9, f"합친 기울기\n+{m_all:.2f}", fontsize=11, color=RED,
            ha="left", va="center")
    ax.text(0.04, 0.96, f"합친 r = {r_all:.3f}\n집단 안 기울기는 모두 " + r"$-0.5$",
            transform=ax.transAxes, fontsize=10, color=INK, va="top")
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("합치면 부호가 뒤집힌다", fontsize=12.5, color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    ax.scatter(X, Y, s=18, color=MUTED, alpha=0.7, edgecolor="none")
    mx, bx = np.polyfit(X, Y, 1)
    t = np.linspace(X.min(), X.max(), 12)
    ax.plot(t, bx + mx * t, color=RED, lw=2.2)
    ax.text(0.04, 0.96, f"r(X, Y) = {r_xy:.3f}", transform=ax.transAxes,
            fontsize=11, color=RED, va="top")
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("통제하기 전", fontsize=12.5, color=INK, pad=8)

    ax = axes[2]
    clean(ax)
    ax.scatter(eX, eY, s=18, color=GREEN, alpha=0.7, edgecolor="none")
    me, be = np.polyfit(eX, eY, 1)
    t = np.linspace(eX.min(), eX.max(), 12)
    ax.plot(t, be + me * t, color=GREEN, lw=2.2)
    ax.text(0.04, 0.96, f"r = {r_res:.3f}", transform=ax.transAxes,
            fontsize=11, color=GREEN, va="top")
    ax.set_xlabel("X 의 잔차", fontsize=10.5, color=INK)
    ax.set_ylabel("Y 의 잔차", fontsize=10.5, color=INK)
    ax.set_title("Z 를 통제한 뒤", fontsize=12.5, color=INK, pad=8)

    save(fig, "simpson_and_partial.png")
    print(f"  합친 기울기={m_all:.3f}, 합친 r={r_all:.4f}")
    print(f"  r_xy={r_xy:.4f}, 부분상관={r_part:.4f}, 잔차상관={r_res:.4f}")


# === 그림 4. 사슬·갈래·충돌자 ===
def fig_three_structures():
    rng = np.random.default_rng(3)
    n = 900

    # 사슬 A -> B -> C
    A1 = rng.normal(0, 1, n)
    B1 = A1 + rng.normal(0, 0.7, n)
    C1 = B1 + rng.normal(0, 0.7, n)

    # 갈래 A <- B -> C
    B2 = rng.normal(0, 1, n)
    A2 = B2 + rng.normal(0, 0.7, n)
    C2 = B2 + rng.normal(0, 0.7, n)

    # 충돌자 A -> B <- C
    A3 = rng.normal(0, 1, n)
    C3 = rng.normal(0, 1, n)
    B3 = A3 + C3 + rng.normal(0, 0.5, n)

    def strata(b, k=4):
        qs = np.quantile(b, np.linspace(0, 1, k + 1))
        labels = np.zeros(len(b), dtype=int)
        for i in range(k):
            lo, hi = qs[i], qs[i + 1]
            m = (b >= lo) & (b <= hi) if i == k - 1 else (b >= lo) & (b < hi)
            labels[m] = i
        return labels

    def partial(a, c, b):
        """B 를 통제한 뒤의 A 와 C 의 상관 = 두 잔차의 상관."""
        ea = a - np.polyval(np.polyfit(b, a, 1), b)
        ec = c - np.polyval(np.polyfit(b, c, 1), b)
        return stats.pearsonr(ea, ec)[0]

    specs = [
        ("사슬 (매개)", A1, C1, B1,
         [("A", (0.14, 0.5)), ("B", (0.5, 0.5)), ("C", (0.86, 0.5))],
         [(0, 1), (1, 2)]),
        ("갈래 (공통원인)", A2, C2, B2,
         [("A", (0.14, 0.3)), ("B", (0.5, 0.82)), ("C", (0.86, 0.3))],
         [(1, 0), (1, 2)]),
        ("충돌자 (공통결과)", A3, C3, B3,
         [("A", (0.14, 0.82)), ("B", (0.5, 0.3)), ("C", (0.86, 0.82))],
         [(0, 1), (2, 1)]),
    ]

    fig = plt.figure(figsize=(11.6, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1.0], hspace=0.34,
                          wspace=0.24)
    strat_cols = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]

    for j, (name, a, c, b, nodes, edges) in enumerate(specs):
        axd = fig.add_subplot(gs[0, j])
        blank(axd)
        axd.set_xlim(0, 1)
        axd.set_ylim(0.08, 1.02)
        pos = [p for _, p in nodes]
        for lab, p in nodes:
            node(axd, p, lab, color=PURPLE if lab == "B" else INK,
                 fc="#F3E5F5" if lab == "B" else "white", r=0.115)
        for u, v in edges:
            arrow(axd, pos[u], pos[v], color=INK, r=0.135)
        axd.set_title(name, fontsize=12.5, color=INK, pad=6)

        r_marg, _ = stats.pearsonr(a, c)
        r_in = partial(a, c, b)
        labels = strata(b)

        ax = fig.add_subplot(gs[1, j])
        clean(ax)
        for i in range(4):
            m = labels == i
            ax.scatter(a[m], c[m], s=13, color=strat_cols[i], alpha=0.8,
                       edgecolor="none")
            mm, bb = np.polyfit(a[m], c[m], 1)
            t = np.linspace(np.quantile(a[m], 0.03), np.quantile(a[m], 0.97), 8)
            ax.plot(t, bb + mm * t, color=strat_cols[i], lw=1.6)
        mm, bb = np.polyfit(a, c, 1)
        t = np.linspace(a.min(), a.max(), 10)
        ax.plot(t, bb + mm * t, color=RED, lw=2.2, ls="--")
        ax.set_xlabel("A", fontsize=10.5, color=INK)
        if j == 0:
            ax.set_ylabel("C", fontsize=10.5, color=INK)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.30 * (hi_ - lo_))
        ax.text(0.03, 0.97,
                f"B 를 무시하면  r = {r_marg:+.2f}\n"
                f"B 를 통제하면  r = {r_in:+.2f}",
                transform=ax.transAxes, fontsize=10.2, color=INK, va="top")
        print(f"  {name}: 주변 r={r_marg:+.3f}, 부분상관={r_in:+.3f}")

    fig.suptitle("붉은 파선은 B 를 무시한 관계, 파란 직선은 B 의 층 안에서의 관계",
                 fontsize=12.5, color=INK, y=1.02)
    save(fig, "three_structures_conditioning.png")


# === 그림 5. 무작위화와 균형 ===
def fig_randomization_balance():
    rng = np.random.default_rng(2024)
    names = ["나이", "기저 혈압", "체중", "운동량", "유전 위험 (미측정)"]
    K = 5
    beta_y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    TAU = 5.0

    def one_trial(n, design, gen):
        Zc = gen.normal(0, 1, (n, K))
        if design == "self":
            logit = 0.75 * Zc.sum(axis=1) / np.sqrt(K) * 2.0
            p = 1 / (1 + np.exp(-logit))
            X = (gen.random(n) < p).astype(int)
        else:
            X = (gen.random(n) < 0.5).astype(int)
        Y = 50 + Zc @ beta_y + TAU * X + gen.normal(0, 5, n)
        return Zc, X, Y

    def smd(Zc, X):
        out = []
        for k in range(K):
            a, b = Zc[X == 1, k], Zc[X == 0, k]
            s = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
            out.append((a.mean() - b.mean()) / s)
        return np.array(out)

    n = 400
    Zs, Xs, Ys = one_trial(n, "self", np.random.default_rng(5))
    Zr, Xr, Yr = one_trial(n, "rand", np.random.default_rng(5))
    smd_s, smd_r = smd(Zs, Xs), smd(Zr, Xr)

    reps = 3000
    est_s = np.empty(reps)
    est_r = np.empty(reps)
    g = np.random.default_rng(99)
    for i in range(reps):
        Zc, X, Y = one_trial(n, "self", g)
        est_s[i] = Y[X == 1].mean() - Y[X == 0].mean()
        Zc, X, Y = one_trial(n, "rand", g)
        est_r[i] = Y[X == 1].mean() - Y[X == 0].mean()

    fig = plt.figure(figsize=(12.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    ypos = np.arange(K)[::-1]
    ax.vlines(0, -0.35, K - 0.6, color=MUTED, lw=1.1)
    ax.vlines([-0.1, 0.1], -0.35, K - 0.6, color=GREEN, lw=1.1,
              linestyles=":")
    ax.scatter(smd_s, ypos, s=90, color=RED, zorder=3, label="자기선택")
    ax.scatter(smd_r, ypos, s=90, color=BLUE, zorder=3, marker="D",
               label="무작위배정")
    for k in range(K):
        ax.plot([smd_r[k], smd_s[k]], [ypos[k], ypos[k]], color=MUTED,
                lw=1.0, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10.5, color=INK)
    ax.set_xlabel("표준화 평균차 (처치군 - 대조군)", fontsize=10.5, color=INK)
    ax.set_xlim(-0.35, 0.95)
    ax.set_ylim(-0.85, K - 0.45)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("처치 전 특성의 균형", fontsize=12.5, color=INK, pad=8)
    ax.text(-0.3, -0.62, "초록 점선은 균형의 통상적 기준 " + r"$\pm 0.1$",
            fontsize=9.5, color=GREEN, va="center", ha="left")

    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    lo = min(est_s.min(), est_r.min())
    hi = max(est_s.max(), est_r.max())
    bins = np.linspace(lo, hi, 55)
    ax.hist(est_r, bins=bins, color=BLUE, alpha=0.65, edgecolor="none",
            label="무작위배정")
    ax.hist(est_s, bins=bins, color=RED, alpha=0.55, edgecolor="none",
            label="자기선택")
    ax.axvline(TAU, color=INK, lw=2.0)
    ax.text(TAU - 0.35, ax.get_ylim()[1] * 0.93, f"참 ATE = {TAU:.0f}",
            fontsize=10.5, color=INK, ha="right", va="top")
    ax.set_xlabel("집단 평균의 단순 차", fontsize=10.5, color=INK)
    ax.set_ylabel("모의실험 횟수", fontsize=10.5, color=INK, labelpad=8)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title(f"{reps:,}번 되풀이한 추정값의 분포", fontsize=12.5,
                 color=INK, pad=8)

    save(fig, "randomization_balance.png")
    print("  SMD 자기선택:", np.round(smd_s, 3))
    print("  SMD 무작위  :", np.round(smd_r, 3))
    print(f"  자기선택 평균={est_s.mean():.3f} sd={est_s.std():.3f}")
    print(f"  무작위   평균={est_r.mean():.3f} sd={est_r.std():.3f}")


# === 그림 6. IV 대 OLS ===
def fig_iv_vs_ols():
    BETA = 1.0
    n = 1000

    def draw(pi, gen, n=n):
        U = gen.normal(0, 1, n)
        Z = gen.normal(0, 1, n)
        X = pi * Z + U + gen.normal(0, 1, n)
        Y = BETA * X + U + gen.normal(0, 1, n)
        return Z, X, Y

    def ols(X, Y):
        return np.cov(X, Y, ddof=1)[0, 1] / np.var(X, ddof=1)

    def iv(Z, X, Y):
        return np.cov(Z, Y, ddof=1)[0, 1] / np.cov(Z, X, ddof=1)[0, 1]

    def first_F(Z, X):
        b = np.cov(Z, X, ddof=1)[0, 1] / np.var(Z, ddof=1)
        a = X.mean() - b * Z.mean()
        res = X - (a + b * Z)
        sse = np.sum(res ** 2)
        ssr = np.sum(((a + b * Z) - X.mean()) ** 2)
        return ssr / (sse / (len(X) - 2))

    reps = 3000
    g = np.random.default_rng(31)
    ols_s, iv_s = np.empty(reps), np.empty(reps)
    for i in range(reps):
        Z, X, Y = draw(1.0, g)
        ols_s[i] = ols(X, Y)
        iv_s[i] = iv(Z, X, Y)

    pis = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.2, 0.3,
                    0.45, 0.65, 0.9, 1.2])
    R = 1500
    rmse_iv, rmse_ols, Fbar = [], [], []
    g2 = np.random.default_rng(77)
    for pi in pis:
        bi = np.empty(R)
        bo = np.empty(R)
        ff = np.empty(R)
        for i in range(R):
            Z, X, Y = draw(pi, g2)
            bi[i] = iv(Z, X, Y)
            bo[i] = ols(X, Y)
            ff[i] = first_F(Z, X)
        # 꼬리가 두꺼워 평균제곱오차가 발산하므로 중앙절대오차를 쓴다.
        rmse_iv.append(np.median(np.abs(bi - BETA)))
        rmse_ols.append(np.median(np.abs(bo - BETA)))
        Fbar.append(ff.mean())
    rmse_iv = np.array(rmse_iv)
    rmse_ols = np.array(rmse_ols)
    Fbar = np.array(Fbar)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    ax = axes[0]
    clean(ax)
    bins = np.linspace(0.6, 1.8, 60)
    ax.hist(np.clip(iv_s, bins[0], bins[-1]), bins=bins, color=BLUE,
            alpha=0.7, edgecolor="none", label="IV (2SLS)")
    ax.hist(np.clip(ols_s, bins[0], bins[-1]), bins=bins, color=RED,
            alpha=0.55, edgecolor="none", label="OLS")
    ax.axvline(BETA, color=INK, lw=2.0)
    ax.text(BETA - 0.02, ax.get_ylim()[1] * 0.96, "참 " + r"$\beta = 1$",
            fontsize=10.5, color=INK, ha="right", va="top")
    ax.set_xlabel("추정값", fontsize=10.5, color=INK)
    ax.set_ylabel("모의실험 횟수", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("강한 도구에서는 IV 만 참값을 맞힌다", fontsize=12.5,
                 color=INK, pad=8)
    print(f"  강한 도구: OLS 평균={ols_s.mean():.3f}, IV 중앙값={np.median(iv_s):.3f}")

    ax = axes[1]
    clean(ax)
    ax.plot(Fbar, rmse_iv, "o-", color=BLUE, lw=2.0, ms=5, label="IV (2SLS)")
    ax.plot(Fbar, rmse_ols, "s-", color=RED, lw=2.0, ms=5, label="OLS")
    ax.set_xscale("log")
    ticks = [1, 3, 10, 30, 100, 300, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9.5, color=INK)
    ax.axvline(10, color=GREEN, lw=1.4, ls="--")
    ax.text(11, max(rmse_iv) * 0.72, "경험 법칙\nF = 10", fontsize=10,
            color=GREEN, va="center")
    ax.set_xlabel("1단계 F-통계량 (평균)", fontsize=10.5, color=INK)
    ax.set_ylabel("중앙절대오차", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("도구가 약해지면 IV 의 오차가 커진다", fontsize=12.5,
                 color=INK, pad=8)
    ax.text(1.12, 0.10,
            r"$F \approx 3$" + " 아래에서는\nIV 가 더 부정확하다",
            fontsize=10, color=INK, ha="left", va="center")

    save(fig, "iv_vs_ols_strength.png")
    for pi, f, a, b in zip(pis, Fbar, rmse_iv, rmse_ols):
        print(f"  pi={pi:.2f} F={f:7.1f} IV MAE={a:6.3f} OLS MAE={b:6.3f}")


if __name__ == "__main__":
    fig_potential_outcomes()
    fig_four_structures()
    fig_simpson_and_partial()
    fig_three_structures()
    fig_randomization_balance()
    fig_iv_vs_ols()
