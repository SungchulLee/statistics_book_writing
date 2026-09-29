r"""10장 검정 절 여섯 쪽의 그림을 생성한다.

만드는 파일:
  ch10/test/img/mcnemar_discordant.png    일치 쌍은 식에 등장하지 않는다
  ch10/test/img/cochran_informative.png   상수 행은 Q 에 아무 기여도 하지 않는다
  ch10/test/img/gof_scipy_null.png        n=24 의 귀무분포는 계단이다
  ch10/test/img/homogeneity_profiles.png  세 모집단의 프로파일과 칸별 기여도
  ch10/test/img/yates_shift.png           예이츠 보정은 통계량을 거의 일정량 깎는다
  ch10/test/img/titanic_permutation.png   우연이 만들 수 있는 표의 범위

실행:  python3 scripts/make_ch10_test_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import itertools

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch10/test/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# =====================================================================
# 1. McNemar: 일치 쌍은 식에 등장하지 않는다
# =====================================================================
def fig_mcnemar():
    a, b, c, d = 101, 121, 59, 33
    m = b + c
    chi_corr = (abs(b - c) - 1) ** 2 / m
    p_chi = stats.chi2.sf(chi_corr, 1)
    p_exact = 2 * stats.binom.sf(b - 1, m, 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    # --- (가) 대응표 ---
    ax = axes[0]
    w, h = 1.0, 0.8
    cells = [((0, 0), a, False), ((1, 0), b, True),
             ((0, 1), c, True), ((1, 1), d, False)]
    for (j, i), val, disc in cells:
        x, y = j * w, -i * h
        ax.add_patch(Rectangle((x, y), w, h,
                               facecolor=ORANGE_F if disc else "#ECEFF1",
                               edgecolor=ORANGE if disc else MUTED,
                               linewidth=2.0 if disc else 1.2))
        ax.text(x + w / 2, y + h / 2 + 0.10, f"{val}", ha="center",
                va="center", fontsize=17,
                color=ORANGE if disc else MUTED)
        ax.text(x + w / 2, y + h / 2 - 0.19,
                "바뀐 쌍" if disc else "그대로인 쌍",
                ha="center", va="center", fontsize=10,
                color=ORANGE if disc else MUTED)
    ax.text(0.5 * w, h + 0.10, "치료후 +", ha="center", va="bottom",
            fontsize=11, color=INK)
    ax.text(1.5 * w, h + 0.10, "치료후 −", ha="center", va="bottom",
            fontsize=11, color=INK)
    ax.text(-0.12, h / 2, "치료전 +", ha="right", va="center",
            fontsize=11, color=INK)
    ax.text(-0.12, -h + h / 2, "치료전 −", ha="right", va="center",
            fontsize=11, color=INK)
    ax.text(1.0, -2 * h + 0.18,
            "101 과 33 은 검정통계량에 등장조차 하지 않는다.\n"
            "두 수를 1000 으로 바꿔도 결과는 그대로다.",
            ha="center", va="top", fontsize=11, color=INK)
    ax.text(1.0, -2 * h - 0.34,
            r"$\chi^2 = (|121-59|-1)^2 / 180 = " + f"{chi_corr:.2f}$",
            ha="center", va="top", fontsize=13, color=ORANGE)
    ax.set_xlim(-1.05, 2.15)
    ax.set_ylim(-2 * h - 0.85, h + 0.45)
    ax.axis("off")
    ax.set_title("(가) 314쌍 중 실제로 쓰이는 것은 180쌍뿐이다",
                 fontsize=12, color=INK)

    # --- (나) 이항 귀무분포 ---
    ax = axes[1]
    xs = np.arange(55, 131)
    pmf = stats.binom.pmf(xs, m, 0.5)
    cols = [RED if (x >= b or x <= c) else BLUE_F for x in xs]
    edges = [RED if (x >= b or x <= c) else BLUE for x in xs]
    ax.bar(xs, pmf, width=0.9, color=cols, edgecolor=edges, linewidth=0.5)
    ax.axvline(b, color=RED, lw=1.6, ls="--")
    ax.annotate(f"관측 {b}", xy=(b, 0.004), xytext=(108, 0.030),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"반대쪽 꼬리 {c} 이하", xy=(c, 0.003), xytext=(56, 0.030),
                fontsize=10.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.03, 0.97,
            "불일치 180쌍이 어느 방향으로 갈지는\n"
            r"$H_0$ 아래에서 동전 던지기와 같다."
            f"\n평균 {m / 2:.0f}, 표준편차 {np.sqrt(m) / 2:.2f}",
            transform=ax.transAxes, fontsize=11, color=INK, va="top")
    ax.text(0.03, 0.62,
            f"정확한 양측 p = {p_exact:.2e}\n"
            f"카이제곱 근사 p = {p_chi:.2e}",
            transform=ax.transAxes, fontsize=11, color=BLUE, va="top")
    ax.set_xlim(54, 131)
    ax.set_ylim(0, 0.064)
    clean_axis(ax)
    ax.set_xlabel("호전된 쌍의 수 (180쌍 중)", fontsize=11, color=INK)
    ax.set_ylabel("확률", fontsize=11, color=INK)
    ax.set_title(r"(나) $H_0$ 아래 이항분포 $B(180,\ 0.5)$", fontsize=12,
                 color=INK)

    fig.tight_layout()
    save(fig, OUT + "mcnemar_discordant.png")


# =====================================================================
# 2. Cochran Q: 상수 행은 정보가 없다
# =====================================================================
TASKS = np.array([
    [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0],
    [1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 0],
    [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
])


def cochran_q(data):
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    Tj = data.sum(axis=0)
    Li = data.sum(axis=1)
    T = Tj.sum()
    return (k - 1) * (k * (Tj ** 2).sum() - T ** 2) / (k * T - (Li ** 2).sum())


def fig_cochran():
    n, k = TASKS.shape
    L = TASKS.sum(1)
    informative = (L > 0) & (L < k)
    q0 = cochran_q(TASKS)
    q_dropped = cochran_q(TASKS[informative])
    p_chi = stats.chi2.sf(q0, k - 1)

    # 행 안 순열의 정확한 귀무분포 (3^9 = 19683 가지)
    rows = []
    for l in L:
        opts = sorted(set(itertools.permutations([1] * l + [0] * (k - l))))
        rows.append([np.array(o) for o in opts])
    denom = k * L.sum() - (L ** 2).sum()
    counts = {}
    for combo in itertools.product(*rows):
        Tj = np.array(combo).sum(0)
        T = Tj.sum()
        q = round((k - 1) * (k * (Tj ** 2).sum() - T ** 2) / denom, 9)
        counts[q] = counts.get(q, 0) + 1
    total = sum(counts.values())
    qv = np.array(sorted(counts))
    qw = np.array([counts[x] for x in qv]) / total
    surv = np.array([qw[qv >= v].sum() for v in qv])
    p_perm = float(qw[qv >= q0 - 1e-6].sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # --- (가) 자료 행렬 ---
    ax = axes[0]
    w, h = 1.0, 0.62
    for i in range(n):
        alive = informative[i]
        for j in range(k):
            val = TASKS[i, j]
            fc = (BLUE if val else "white") if alive else ("#B0BEC5" if val else "white")
            ec = BLUE if alive else MUTED
            ax.add_patch(Rectangle((j * w, -i * h), w * 0.9, h * 0.82,
                                   facecolor=fc, edgecolor=ec, linewidth=1.2))
            ax.text(j * w + w * 0.45, -i * h + h * 0.41, str(val),
                    ha="center", va="center", fontsize=10,
                    color=("white" if val else (BLUE if alive else MUTED)))
        if not alive:
            ax.text(k * w + 0.18, -i * h + h * 0.41, "정보 없음",
                    ha="left", va="center", fontsize=10, color=MUTED)
    for j, t in enumerate(TASKS.sum(0)):
        ax.text(j * w + w * 0.45, h * 0.95, f"과제 {j + 1}", ha="center",
                va="bottom", fontsize=10.5, color=INK)
        ax.text(j * w + w * 0.45, -n * h + h * 0.12, f"{t}", ha="center",
                va="top", fontsize=12, color=BLUE)
    ax.text(-0.22, -n * h + h * 0.12, "성공 수", ha="right", va="top",
            fontsize=10.5, color=INK)
    ax.text(-0.22, -3 * h + h * 0.41, "피험자\n12명", ha="right",
            va="center", fontsize=10.5, color=INK)
    ax.text(k * w / 2 - 0.3, -n * h - 0.42,
            f"상수 행 3개를 지워도  Q = {q_dropped:.4f} 로 똑같다",
            ha="center", va="top", fontsize=11.5, color=ORANGE)
    ax.set_xlim(-1.5, k * w + 1.45)
    ax.set_ylim(-n * h - 0.95, h * 1.55)
    ax.axis("off")
    ax.set_title("(가) 전부 0 이거나 전부 1 인 행은 버려도 된다",
                 fontsize=12, color=INK)

    # --- (나) 정확한 순열분포 대 카이제곱 근사 ---
    ax = axes[1]
    grid = np.linspace(0, 16, 500)
    ax.plot(grid, stats.chi2.sf(grid, k - 1), color=ORANGE, lw=2.2,
            label=r"$\chi^2_2$ 근사")
    ax.step(np.concatenate([[0], qv, [16]]),
            np.concatenate([[1.0], surv, [0.0]]),
            where="post", color=BLUE, lw=2.0,
            label="정확한 순열분포 (19683가지)")
    ax.axvline(q0, color=RED, lw=1.5, ls="--")
    ax.text(q0 + 0.25, 0.80, f"관측 Q = {q0:.3f}", fontsize=11, color=RED)
    ax.axhline(0.05, color=MUTED, lw=1.2, ls=":")
    ax.text(15.7, 0.058, "0.05", ha="right", fontsize=10, color=INK)
    ax.text(0.19, 0.44,
            f"순열 p = {p_perm:.4f}\n"
            f"카이제곱 p = {p_chi:.4f}",
            transform=ax.transAxes, fontsize=11.5, color=INK, va="top")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    ax.set_xlabel(r"$Q$", fontsize=11, color=INK)
    ax.set_ylabel(r"$P(Q \geq q)$", fontsize=11, color=INK)
    ax.set_title("(나) Q 가 가질 수 있는 값은 11개뿐이다", fontsize=12,
                 color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, OUT + "cochran_informative.png")


# =====================================================================
# 3. 적합도 검정 (n = 24): 귀무분포가 계단이다
# =====================================================================
def fig_gof_scipy():
    obs = np.array([4, 13, 7], dtype=float)
    n = int(obs.sum())
    E = n / 3
    contrib = (obs - E) ** 2 / E
    chi_obs = contrib.sum()
    p_approx = stats.chi2.sf(chi_obs, 2)

    counts = {}
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            O = np.array([a, b, c])
            key = round(((O - E) ** 2 / E).sum(), 9)
            counts[key] = counts.get(key, 0.0) + stats.multinomial.pmf(
                O, n, [1 / 3] * 3)
    kv = np.array(sorted(counts))
    kw = np.array([counts[x] for x in kv])
    surv = np.array([kw[kv >= v].sum() for v in kv])
    p_exact = float(kw[kv >= chi_obs - 1e-9].sum())
    crit = stats.chi2.ppf(0.95, 2)
    a_true = float(kw[kv >= crit].sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    # --- (가) 관측 대 기대 ---
    ax = axes[0]
    labels = ["승", "패", "무"]
    x = np.arange(3)
    ax.bar(x - 0.19, obs, width=0.36, color=BLUE_F, edgecolor=BLUE,
           linewidth=1.6, label="관측도수")
    ax.bar(x + 0.19, [E] * 3, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           linewidth=1.6, label="기대도수 8")
    for i in range(3):
        ax.text(i, max(obs[i], E) + 0.55,
                r"$(O-E)^2/E$ = " + f"{contrib[i]:.3f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.text(0.5, 0.96, f"합하면 " + r"$\chi^2$ = " + f"{chi_obs:.2f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=12,
            color=PURPLE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 17.5)
    clean_axis(ax)
    ax.set_ylabel("경기 수 (전체 24)", fontsize=11, color=INK)
    ax.set_title("(가) 24경기에서 4승 13패 7무", fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper left")

    # --- (나) 정확한 귀무분포 ---
    ax = axes[1]
    grid = np.linspace(0, 14, 500)
    ax.plot(grid, stats.chi2.sf(grid, 2), color=ORANGE, lw=2.2,
            label=r"$\chi^2_2$ 근사")
    ax.step(np.concatenate([[0], kv, [14]]),
            np.concatenate([[1.0], surv, [0.0]]),
            where="post", color=BLUE, lw=1.8,
            label="정확한 다항분포 (44개 값)")
    ax.axvline(chi_obs, color=RED, lw=1.5, ls="--")
    ax.text(chi_obs + 0.25, 0.80, r"관측 $\chi^2$ = 5.25", fontsize=11,
            color=RED)
    ax.axhline(0.05, color=MUTED, lw=1.2, ls=":")
    ax.text(13.8, 0.058, "0.05", ha="right", fontsize=10, color=INK)
    ax.text(0.40, 0.46,
            f"정확한 p = {p_exact:.4f}\n"
            f"카이제곱 p = {p_approx:.4f}",
            transform=ax.transAxes, fontsize=11.5, color=INK, va="top")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 1.02)
    clean_axis(ax)
    ax.set_xlabel(r"$\chi^2$", fontsize=11, color=INK)
    ax.set_ylabel(r"$P(\chi^2 \geq x)$", fontsize=11, color=INK)
    ax.set_title(r"(나) $n=24$ 에서 통계량이 갖는 값은 44개뿐이다",
                 fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, OUT + "gof_scipy_null.png")
    print(f"  gof: exact p={p_exact:.4f}, approx p={p_approx:.4f}, "
          f"true alpha at 5.991 = {a_true:.4f}")


# =====================================================================
# 4. 동질성: 프로파일과 칸별 기여도
# =====================================================================
def fig_homogeneity():
    obs = np.array([[25, 30, 20, 25],
                    [18, 22, 35, 25],
                    [30, 25, 15, 30]], dtype=float)
    chi2, p, df, exp = stats.chi2_contingency(obs, correction=False)
    prop = obs / obs.sum(1, keepdims=True)
    pooled = obs.sum(0) / obs.sum()
    contrib = (obs - exp) ** 2 / exp

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    cats = [f"범주 {j + 1}" for j in range(4)]
    x = np.arange(4)
    pop_colors = [BLUE, ORANGE, GREEN]

    # --- (가) 프로파일 ---
    ax = axes[0]
    ax.plot(x, pooled, color=INK, lw=2.4, ls="--", marker="D", ms=7,
            label=r"합동 비율 ($H_0$)", zorder=5)
    for i in range(3):
        ax.plot(x, prop[i], color=pop_colors[i], lw=2.0, marker="o", ms=6,
                label=f"모집단 {i + 1}")
    ax.annotate("모집단 2 의 범주 3\n35% 대 23.3%",
                xy=(2, prop[1, 2]), xytext=(2.25, 0.335), fontsize=10.5,
                color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10.5)
    ax.set_ylim(0.10, 0.40)
    ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_yticklabels(["10%", "15%", "20%", "25%", "30%", "35%", "40%"])
    clean_axis(ax)
    ax.set_ylabel("모집단 안에서의 비율", fontsize=11, color=INK)
    ax.set_title(r"(가) $H_0$ 는 세 선이 검은 선에 겹친다는 말이다",
                 fontsize=12, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left", ncol=2)

    # --- (나) 칸별 기여도 ---
    ax = axes[1]
    width = 0.26
    for i in range(3):
        ax.bar(x + (i - 1) * width, contrib[i], width=width,
               color=pop_colors[i], alpha=0.85, label=f"모집단 {i + 1}")
    ax.text(2.0, contrib[1, 2] + 0.12, f"{contrib[1, 2]:.2f}",
            ha="center", va="bottom", fontsize=11, color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10.5)
    ax.set_ylim(0, 7.0)
    clean_axis(ax)
    ax.set_ylabel(r"$(O-E)^2/E$", fontsize=11, color=INK)
    ax.set_title(f"(나) 12칸의 기여도를 더하면 " + r"$\chi^2$ = "
                 + f"{chi2:.2f}", fontsize=12, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.text(0.97, 0.94, f"df = {df},  p = {p:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11.5, color=PURPLE)

    fig.tight_layout()
    save(fig, OUT + "homogeneity_profiles.png")


# =====================================================================
# 5. 예이츠 보정은 통계량을 거의 일정량 깎는다
# =====================================================================
def fig_yates_shift():
    base = np.array([[10, 5], [3, 12]], dtype=float)
    plain = stats.chi2_contingency(base, correction=False)
    yates = stats.chi2_contingency(base, correction=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # --- (가) 두 통계량의 꼬리넓이 ---
    ax = axes[0]
    xg = np.linspace(3.0, 12, 700)
    dens = stats.chi2.pdf(xg, 1)
    ax.plot(xg, dens, color=INK, lw=2.0)
    m1 = xg >= plain[0]
    ax.fill_between(xg[m1], 0, dens[m1], color=BLUE_F, edgecolor=BLUE,
                    linewidth=1.2)
    m2 = (xg >= yates[0]) & (xg < plain[0])
    ax.fill_between(xg[m2], 0, dens[m2], color=ORANGE_F, edgecolor=ORANGE,
                    linewidth=1.2)
    ax.plot([yates[0]] * 2, [0, stats.chi2.pdf(yates[0], 1)],
            color=ORANGE, lw=1.8)
    ax.plot([plain[0]] * 2, [0, stats.chi2.pdf(plain[0], 1)],
            color=BLUE, lw=1.8)
    ax.text(yates[0] - 0.14, 0.0555,
            f"보정 있음 {yates[0]:.2f}\np = {yates[1]:.4f}",
            ha="right", va="top", fontsize=11, color=ORANGE)
    ax.text(plain[0] + 0.14, 0.0555,
            f"보정 없음 {plain[0]:.2f}\np = {plain[1]:.4f}",
            ha="left", va="top", fontsize=11, color=BLUE)
    ax.annotate("파란 꼬리가 0.0099", xy=(7.4, 0.005), xytext=(9.0, 0.025),
                fontsize=10.5, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.annotate("주황 띠를 더하면 0.0271", xy=(5.6, 0.014),
                xytext=(8.6, 0.040), fontsize=10.5, color=ORANGE,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.set_xlim(3, 12)
    ax.set_ylim(0, 0.058)
    clean_axis(ax)
    ax.set_xlabel(r"$\chi^2$", fontsize=11, color=INK)
    ax.set_ylabel(r"$\chi^2_1$ 밀도", fontsize=11, color=INK)
    ax.set_title("(가) 같은 표, 다른 두 결론", fontsize=12, color=INK)

    # --- (나) 표를 키워도 깎이는 양은 그대로다 ---
    ax = axes[1]
    ts = np.arange(1, 9)
    ns = 30 * ts
    cp = np.array([stats.chi2_contingency(base * t, correction=False)[0]
                   for t in ts])
    cy = np.array([stats.chi2_contingency(base * t, correction=True)[0]
                   for t in ts])
    ax.plot(ns, cp, color=BLUE, lw=2.0, marker="o", ms=5,
            label="보정 없음")
    ax.plot(ns, cy, color=ORANGE, lw=2.0, marker="s", ms=5,
            label="예이츠 보정")
    ax.axhline(3.841, color=GREEN, lw=1.3, ls="--")
    ax.text(238, 4.3, "5% 임계값 3.841", ha="right", fontsize=10, color=GREEN)
    ax.axhline(6.635, color=PURPLE, lw=1.3, ls=":")
    ax.text(238, 7.1, "1% 임계값 6.635", ha="right", fontsize=10,
            color=PURPLE)
    for t, n, a, b in zip(ts, ns, cp, cy):
        if t in (1, 4, 8):
            ax.annotate("", xy=(n, a), xytext=(n, b),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.3))
            ax.text(n + 4, (a + b) / 2, f"{a - b:.2f}", fontsize=10,
                    color=RED, va="center")
    ax.text(0.04, 0.95,
            "칸 비율을 그대로 둔 채 표를 키웠다.\n"
            "깎이는 양은 1.8 근처에 머문다.",
            transform=ax.transAxes, fontsize=11, color=INK, va="top")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 62)
    clean_axis(ax)
    ax.set_xlabel(r"표본크기 $n$", fontsize=11, color=INK)
    ax.set_ylabel(r"$\chi^2$ 통계량", fontsize=11, color=INK)
    ax.set_title("(나) 보정이 결론을 바꾸는 곳은 문턱 근처뿐이다",
                 fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="center right")

    fig.tight_layout()
    save(fig, OUT + "yates_shift.png")
    print(f"  yates: plain {plain[0]:.4f} p {plain[1]:.4f} | "
          f"yates {yates[0]:.4f} p {yates[1]:.4f} | gap {cp - cy}")


# =====================================================================
# 6. 타이타닉: 우연이 만들 수 있는 표의 범위
# =====================================================================
def fig_titanic():
    n, n_f, n_s = 891, 314, 342

    def chi2_2x2(a):
        b = n_f - a
        c = n_s - a
        d = n - n_f - c
        return n * (a * d - b * c) ** 2 / (n_f * (n - n_f) * n_s * (n - n_s))

    obs_a = 233
    obs_chi = chi2_2x2(obs_a)
    lo = max(0, n_s - (n - n_f))
    hi = min(n_f, n_s)
    av = np.arange(lo, hi + 1)
    pm = stats.hypergeom.pmf(av, n, n_f, n_s)   # H0 아래 정확한 분포
    mu = n_f * n_s / n
    sd = np.sqrt(n_f * n_s * (n - n_f) * (n - n_s) / (n ** 2 * (n - 1)))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # --- (가) 여성 생존자 수 ---
    ax = axes[0]
    ax.bar(av, pm, width=1.0, color=BLUE_F, edgecolor=BLUE, linewidth=0.4)
    ax.axvline(obs_a, color=RED, lw=2.0)
    ax.annotate(f"실제 {obs_a}명", xy=(obs_a - 1.5, 0.012),
                xytext=(178, 0.020), fontsize=12, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.text(0.36, 0.97,
            r"$H_0$ 아래 여성 생존자 수의 분포" "\n"
            f"평균 {mu:.2f}, 표준편차 {sd:.2f}\n"
            f"1만 번 뒤섞어도 97 ~ 146명",
            transform=ax.transAxes, fontsize=11, color=INK, va="top")
    ax.text(0.36, 0.66,
            f"관측값은 평균에서\n{(obs_a - mu) / sd:.1f} 표준편차 떨어져 있다",
            transform=ax.transAxes, fontsize=11, color=RED, va="top")
    ax.set_xlim(85, 250)
    ax.set_ylim(0, 0.062)
    clean_axis(ax)
    ax.set_xlabel("여성 생존자 수", fontsize=11, color=INK)
    ax.set_ylabel("확률", fontsize=11, color=INK)
    ax.set_title("(가) 우연이 만들 수 있는 표는 여기까지다", fontsize=12,
                 color=INK)

    # --- (나) 카이제곱 통계량 ---
    ax = axes[1]
    chis = chi2_2x2(av)
    order = np.argsort(chis)
    cs, ws = chis[order], pm[order]
    grid = np.linspace(0.05, 16, 500)
    ax.plot(grid, stats.chi2.pdf(grid, 1), color=ORANGE, lw=2.2,
            label=r"$\chi^2_1$ 밀도")
    bins = np.linspace(0, 16, 41)
    ax.hist(cs, bins=bins, weights=ws, density=True, color=BLUE_F,
            edgecolor=BLUE, linewidth=0.5, label=r"$H_0$ 아래 실제 분포")
    ax.axvline(13.494, color=GREEN, lw=1.5, ls="--")
    ax.text(13.3, 0.62, "1만 번 중 최댓값 13.49", rotation=90, fontsize=10,
            color=GREEN, va="center", ha="right")
    ax.annotate("관측 " + r"$\chi^2$ = 263.05" + "\n이 축의 16배 바깥이다",
                xy=(15.9, 0.10), xytext=(7.3, 0.40), fontsize=12, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.05)
    clean_axis(ax)
    ax.set_xlabel(r"$\chi^2$", fontsize=11, color=INK)
    ax.set_ylabel("밀도", fontsize=11, color=INK)
    ax.set_title(r"(나) $H_0$ 의 세계는 $\chi^2_1$ 을 그대로 따른다",
                 fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    fig.tight_layout()
    save(fig, OUT + "titanic_permutation.png")
    print(f"  titanic: mu={mu:.2f} sd={sd:.3f} z={(obs_a - mu) / sd:.2f} "
          f"chi={obs_chi:.4f}")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_mcnemar()
    fig_cochran()
    fig_gof_scipy()
    fig_homogeneity()
    fig_yates_shift()
    fig_titanic()
