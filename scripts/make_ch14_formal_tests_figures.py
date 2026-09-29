r"""14장 형식적 검정 절 여덟 쪽의 개념 그림을 만든다.

만드는 파일:
  ch14/formal_tests/img/test_power_by_alternative.png   검정마다 잘 잡는 이탈이 다르다
  ch14/formal_tests/img/shapiro_w_as_qq_fit.png         W 는 Q-Q 그림을 숫자로 옮긴 것
  ch14/formal_tests/img/shapiro_null_scale.png          W=0.975 의 뜻은 n 에 따라 다르다
  ch14/formal_tests/img/ks_ecdf_sup.png                 D 는 세로 틈 하나다
  ch14/formal_tests/img/ks_estimated_params_shrink.png  모수를 추정하면 D 가 줄어든다
  ch14/formal_tests/img/lilliefors_null_shift.png       기준 분포 자체가 달라진다
  ch14/formal_tests/img/ad_weight_function.png          AD 는 꼬리에 무게를 싣는다
  ch14/formal_tests/img/ad_vs_ks_tail_power.png         그 무게가 사는 검정력

실행:  python3 scripts/make_ch14_formal_tests_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch14/formal_tests/img/"
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
# 공통 도구: 벡터화한 KS 통계량과 AD 통계량
# ===================================================================
def ks_D(x, fitted=True):
    """행마다 KS 통계량. fitted=True 면 모수를 그 행에서 추정한다."""
    n = x.shape[1]
    xs = np.sort(x, axis=1)
    if fitted:
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, ddof=1, keepdims=True)
    else:
        mu, sd = 0.0, 1.0
    u = stats.norm.cdf((xs - mu) / sd)
    i = np.arange(1, n + 1)
    return np.maximum(i / n - u, u - (i - 1) / n).max(axis=1)


def ad_A2(x):
    """행마다 Anderson-Darling 통계량(모수는 그 행에서 추정)."""
    n = x.shape[1]
    xs = np.sort(x, axis=1)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=1, keepdims=True)
    u = np.clip(stats.norm.cdf((xs - mu) / sd), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - ((2 * i - 1) * (np.log(u) + np.log(1 - u[:, ::-1]))).sum(axis=1) / n


def ad_crit5(n):
    """scipy 가 쓰는 5% 임계값(표본크기 보정 포함)."""
    return 0.787 / (1.0 + 4.0 / n - 25.0 / n**2)


def lilliefors_crit(n, alpha=0.05, B=60000, seed=99):
    rng = np.random.default_rng(seed)
    return float(np.quantile(ks_D(rng.normal(size=(B, n))), 1 - alpha))


# ===================================================================
# 그림 1. 검정마다 잘 잡는 이탈이 다르다
# ===================================================================
def fig_power_by_alternative():
    n, B, alpha = 50, 6000, 0.05
    rng = np.random.default_rng(314)

    def draw(kind):
        if kind == "normal":
            return rng.normal(0, 1, size=(B, n))
        if kind == "lognormal":
            return rng.lognormal(0, 0.6, size=(B, n))
        if kind == "t3":
            return rng.standard_t(3, size=(B, n))
        if kind == "uniform":
            return rng.uniform(-1, 1, size=(B, n))
        if kind == "bimodal":
            z = rng.normal(0, 1, size=(B, n))
            s = rng.integers(0, 2, size=(B, n)) * 2 - 1
            return z + 1.6 * s
        raise ValueError(kind)

    alts = ["normal", "lognormal", "t3", "uniform", "bimodal"]
    alt_labels = ["정규\n(크기 확인)", "대수정규\n(오른쪽 치우침)",
                  r"$t_3$" + "\n(두꺼운 꼬리)", "균등\n(얇은 꼬리)",
                  "이봉 혼합\n(봉우리 둘)"]
    test_labels = ["Shapiro-Wilk", r"D'Agostino $K^2$", "Jarque-Bera",
                   "Lilliefors (KS)", "Anderson-Darling"]

    d_crit = lilliefors_crit(n)
    a_crit = ad_crit5(n)
    print(f"n={n}  Lilliefors 5% 임계값 D={d_crit:.4f}   AD 5% 임계값 A2={a_crit:.4f}")

    table = np.zeros((5, len(alts)))
    for j, kind in enumerate(alts):
        x = draw(kind)
        table[0, j] = np.mean(stats.shapiro(x, axis=1).pvalue < alpha)
        table[1, j] = np.mean(stats.normaltest(x, axis=1).pvalue < alpha)
        table[2, j] = np.mean(stats.jarque_bera(x, axis=1).pvalue < alpha)
        table[3, j] = np.mean(ks_D(x) > d_crit)
        table[4, j] = np.mean(ad_A2(x) > a_crit)
        print(f"  {kind:10s}", " ".join(f"{v:.3f}" for v in table[:, j]))

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    im = ax.imshow(table, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for r in range(5):
        for c in range(len(alts)):
            v = table[r, c]
            ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=11,
                    color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(alts)))
    ax.set_xticklabels(alt_labels, fontsize=9.5, color=INK)
    ax.set_yticks(range(5))
    ax.set_yticklabels(test_labels, fontsize=10, color=INK)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(f"모집단별 기각률  ($n = {n}$, $\\alpha = 0.05$, 6000회 반복)",
                 fontsize=12.5, color=INK, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("기각률", fontsize=10, color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK)
    cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, "test_power_by_alternative.png")


# ===================================================================
# 그림 2. W 는 Q-Q 그림을 숫자 하나로 줄인 것
# ===================================================================
def fig_w_as_qq_fit():
    rng = np.random.default_rng(5)
    n = 60
    sets = [
        ("정규 $N(0,1)$", rng.normal(0, 1, n), BLUE),
        ("대수정규 (치우침)", rng.lognormal(0, 0.6, n), ORANGE),
        (r"$t_3$ (두꺼운 꼬리)", rng.standard_t(3, n), PURPLE),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2))
    for ax, (name, x, col) in zip(axes, sets):
        xs = np.sort(x)
        m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        W, p = stats.shapiro(x)
        r = np.corrcoef(m, xs)[0, 1]
        print(f"{name}:  W={W:.4f}  p={p:.3g}  r^2={r**2:.4f}")

        b, a = np.polyfit(m, xs, 1)
        ax.plot(m, a + b * m, color=MUTED, lw=1.5, ls="--", zorder=1)
        ax.scatter(m, xs, s=18, color=col, alpha=0.85, zorder=2,
                   edgecolor="none")
        ax.set_title(name, fontsize=11.5, color=INK)
        ax.set_xlabel("정규 점수 $m_i$", fontsize=10, color=INK)
        ax.text(0.04, 0.96, f"$W = {W:.4f}$\n$r^2 = {r**2:.4f}$\n$p = {p:.3g}$",
                transform=ax.transAxes, fontsize=10.5, color=col,
                va="top", ha="left", linespacing=1.6)
        clean(ax)
    axes[0].set_ylabel("정렬한 자료 $X_{(i)}$", fontsize=10, color=INK)

    fig.suptitle("$W$ 는 Q-Q 그림의 직선성을 숫자 하나로 요약한다  ($n = 60$)",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "shapiro_w_as_qq_fit.png")


# ===================================================================
# 그림 3. 같은 W=0.975 도 n 에 따라 뜻이 다르다
# ===================================================================
def fig_shapiro_null_scale():
    rng = np.random.default_rng(21)
    B = 30000
    w_obs = 0.9750
    out = {}
    for n in (30, 300):
        W = stats.shapiro(rng.normal(size=(B, n)), axis=1).statistic
        q05 = np.quantile(W, 0.05)
        pct = np.mean(W < w_obs)
        out[n] = (W, q05, pct)
        print(f"n={n}: W 중앙값={np.median(W):.4f}  5백분위={q05:.4f}  "
              f"P(W<0.975)={pct:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=False)
    for ax, n in zip(axes, (30, 300)):
        W, q05, pct = out[n]
        ax.hist(W, bins=120, range=(0.80, 1.0), density=True,
                color=BLUE_F, edgecolor=BLUE, linewidth=0.3)
        ax.axvline(q05, color=MUTED, lw=1.5, ls="--")
        ax.axvline(w_obs, color=RED, lw=2.2)
        ax.set_xlim(0.85, 1.0)
        ax.set_title(f"정규자료에서 $W$ 의 귀무분포  ($n = {n}$)",
                     fontsize=11.5, color=INK)
        ax.set_xlabel("$W$", fontsize=10.5, color=INK)
        clean(ax)
        ax.set_yticks([])
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.18)
        # 라벨은 선 옆이 아니라 비어 있는 왼쪽 위에 모아 둔다
        ax.text(0.03, 0.97, f"귀무분포의 5백분위 (회색 파선)  {q05:.4f}",
                transform=ax.transAxes, fontsize=9.5, color=INK,
                ha="left", va="top")
        ax.text(0.03, 0.88, f"빨간 선 $W = 0.975$ 는 아래쪽 {pct*100:.1f}%",
                transform=ax.transAxes, fontsize=9.5, color=RED,
                ha="left", va="top")
    axes[0].set_ylabel("밀도", fontsize=10.5, color=INK)

    fig.suptitle("같은 $W = 0.975$ 도 $n$ 이 달라지면 전혀 다른 증거가 된다",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "shapiro_null_scale.png")


# ===================================================================
# 그림 4. D 는 경험분포와 이론분포 사이의 세로 틈 하나다
# ===================================================================
def _ecdf_sup(x, mu=0.0, sd=1.0):
    n = x.size
    xs = np.sort(x)
    u = stats.norm.cdf((xs - mu) / sd)
    i = np.arange(1, n + 1)
    up = i / n - u
    lo = u - (i - 1) / n
    D = max(up.max(), lo.max())
    if up.max() >= lo.max():
        k = int(np.argmax(up))
        return D, xs[k], u[k], (k + 1) / n
    k = int(np.argmax(lo))
    return D, xs[k], u[k], k / n


def fig_ks_ecdf_sup():
    n = 40
    x_norm = np.random.default_rng(3).normal(0, 1, n)
    # 평균 0, 분산 1 로 맞춘 지수분포: 모수는 N(0,1) 과 같고 모양만 다르다
    x_skew = np.random.default_rng(8).exponential(1.0, n) - 1.0

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, x, name, col in [
        (axes[0], x_norm, "정규 표본", BLUE),
        (axes[1], x_skew, "치우친 표본 (평균 0, 분산 1 인 지수분포)", ORANGE),
    ]:
        D, xstar, u_at, f_at = _ecdf_sup(x)
        p = stats.kstest(x, "norm").pvalue
        print(f"{name}: D={D:.4f}  p={p:.4f}  최대 틈의 위치 x={xstar:.3f}")

        grid = np.linspace(-3.4, 3.4, 400)
        ax.plot(grid, stats.norm.cdf(grid), color=INK, lw=2,
                label="이론 분포 $\\Phi(x)$")
        xs = np.sort(x)
        ax.step(np.concatenate([[-3.4], xs, [3.4]]),
                np.concatenate([[0], np.arange(1, n + 1) / n, [1]]),
                where="post", color=col, lw=1.6, label="경험분포함수 $F_n(x)$")

        lo, hi = sorted([u_at, f_at])
        ax.add_patch(FancyArrowPatch((xstar, lo), (xstar, hi),
                                     arrowstyle="<->", mutation_scale=12,
                                     color=RED, lw=2.0, zorder=5))
        ax.plot([xstar, xstar], [0, lo], color=RED, lw=0.8, ls=":", zorder=1)
        ax.text(xstar + 0.14, (lo + hi) / 2, f"$D = {D:.3f}$", fontsize=11.5,
                color=RED, va="center", ha="left")

        ax.set_xlim(-3.4, 3.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{name}  ($p = {p:.3f}$)", fontsize=11.5, color=INK)
        ax.set_xlabel("$x$", fontsize=10.5, color=INK)
        ax.legend(fontsize=9.5, frameon=False, loc="upper left", labelcolor=INK)
        clean(ax)
    axes[0].set_ylabel("누적확률", fontsize=10.5, color=INK)

    fig.suptitle(r"KS 통계량은 두 곡선 사이의 가장 큰 세로 거리 하나다  ($n = 40$)",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "ks_ecdf_sup.png")


# ===================================================================
# 그림 5. 모수를 자료에서 추정하면 D 가 줄어든다
# ===================================================================
def fig_ks_shrink():
    rng = np.random.default_rng(58)
    n = 25
    x = rng.normal(0, 1, n)
    mu_hat, sd_hat = x.mean(), x.std(ddof=1)

    D_fix, xs_f, u_f, f_f = _ecdf_sup(x, 0.0, 1.0)
    D_est, xs_e, u_e, f_e = _ecdf_sup(x, mu_hat, sd_hat)
    print(f"한 표본: mu_hat={mu_hat:.4f} sd_hat={sd_hat:.4f} "
          f"D(고정)={D_fix:.4f}  D(추정)={D_est:.4f}")

    # 같은 일이 평균적으로 얼마나 일어나는지도 재 둔다
    Z = rng.normal(size=(40000, n))
    print(f"40000회 평균: D(고정)={ks_D(Z, fitted=False).mean():.4f}  "
          f"D(추정)={ks_D(Z, fitted=True).mean():.4f}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    grid = np.linspace(-3.0, 3.0, 400)
    ax.plot(grid, stats.norm.cdf(grid), color=BLUE, lw=2,
            label="못박은 $N(0,1)$ 의 분포함수")
    ax.plot(grid, stats.norm.cdf((grid - mu_hat) / sd_hat), color=ORANGE, lw=2,
            label=f"자료에서 적합한 $N({mu_hat:.2f}, {sd_hat:.2f}^2)$ 의 분포함수")
    xs = np.sort(x)
    ax.step(np.concatenate([[-3.0], xs, [3.0]]),
            np.concatenate([[0], np.arange(1, n + 1) / n, [1]]),
            where="post", color=INK, lw=1.6, label="경험분포함수 $F_n(x)$")

    for D, xstar, u_at, f_at, col, dx, lab in [
        (D_fix, xs_f, u_f, f_f, BLUE, -1, "못박은 모수"),
        (D_est, xs_e, u_e, f_e, ORANGE, +1, "추정한 모수"),
    ]:
        lo, hi = sorted([u_at, f_at])
        ax.add_patch(FancyArrowPatch((xstar, lo), (xstar, hi),
                                     arrowstyle="<->", mutation_scale=11,
                                     color=col, lw=2.2, zorder=5))
        # 한글은 mathtext 바깥에 둔다
        ax.annotate(f"{lab}:  $D = {D:.3f}$", (xstar, (lo + hi) / 2),
                    textcoords="offset points",
                    xytext=(24 * dx, 0),
                    fontsize=11.5, color=col, va="center",
                    ha="left" if dx > 0 else "right")

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("$x$", fontsize=10.5, color=INK)
    ax.set_ylabel("누적확률", fontsize=10.5, color=INK)
    ax.set_title(f"같은 표본, 다른 기준선: 적합한 곡선은 자료 쪽으로 끌려온다  ($n = {n}$)",
                 fontsize=12, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left", labelcolor=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "ks_estimated_params_shrink.png")


# ===================================================================
# 그림 6. 기준으로 삼아야 할 귀무분포 자체가 다르다
# ===================================================================
def fig_lilliefors_null_shift():
    rng = np.random.default_rng(77)
    n, B = 30, 80000
    Z = rng.normal(size=(B, n))
    D_fix = ks_D(Z, fitted=False)
    D_est = ks_D(Z, fitted=True)

    q_fix = np.quantile(D_fix, 0.95)
    q_est = np.quantile(D_est, 0.95)
    size_naive = float(np.mean(D_est > q_fix))
    print(f"n={n}: 95백분위 고정={q_fix:.4f}  추정={q_est:.4f}  "
          f"순진한 KS 의 실제 크기={size_naive:.4f}")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bins = np.linspace(0.02, 0.36, 130)
    ax.hist(D_est, bins=bins, density=True, color=ORANGE_F,
            edgecolor=ORANGE, linewidth=0.3,
            label=r"모수를 표본에서 추정할 때의 $D$")
    ax.hist(D_fix, bins=bins, density=True, histtype="step",
            color=BLUE, linewidth=2.0,
            label=r"모수를 못박았을 때의 $D$")

    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.24)
    ax.axvline(q_est, color=ORANGE, lw=1.6, ls="--")
    ax.axvline(q_fix, color=BLUE, lw=1.6, ls="--")
    ax.text(q_est - 0.006, top * 1.20, f"Lilliefors 5% 임계값\n{q_est:.3f}",
            fontsize=9.5, color=ORANGE, ha="right", va="top", linespacing=1.4)
    ax.text(q_fix - 0.006, top * 1.20, f"KS 5% 임계값\n{q_fix:.3f}",
            fontsize=9.5, color=BLUE, ha="right", va="top", linespacing=1.4)
    ax.annotate(f"KS 임계값을 쓰면\n실제 유의수준은 {size_naive:.5f}",
                xy=(q_fix, 0.6), xytext=(0.30, top * 0.55),
                fontsize=10, color=INK, ha="center", va="center",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

    ax.set_xlim(0.02, 0.36)
    ax.set_xlabel("$D$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(f"정규자료에서 $D$ 의 귀무분포  ($n = {n}$, 80000회 반복)",
                 fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.86), labelcolor=INK)
    clean(ax)
    ax.set_yticks([])
    fig.tight_layout()
    save(fig, "lilliefors_null_shift.png")


# ===================================================================
# 그림 7. AD 의 가중함수가 꼬리를 키운다
# ===================================================================
def fig_ad_weight():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- 왼쪽: 가중함수 ---
    ax = axes[0]
    u = np.linspace(0.004, 0.996, 1200)
    w = 1.0 / (u * (1 - u))
    ax.plot(u, w, color=ORANGE, lw=2.4, label=r"AD 의 가중 $1/[F_0(1-F_0)]$")
    ax.axhline(1.0, color=BLUE, lw=2.0, ls="--",
               label="KS·CvM 의 가중 (모든 곳에서 같다)")
    ax.set_yscale("log")
    ax.set_ylim(0.6, 400)
    ax.set_yticks([1, 4, 10, 100])
    ax.set_yticklabels(["1", "4", "10", "100"])   # 로그축 라벨은 평문으로
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    ax.set_xticklabels(["0", "0.05", "0.25", "0.5", "0.75", "0.95", "1"])
    for uu in (0.5, 0.05, 0.01):
        ax.plot([uu], [1 / (uu * (1 - uu))], "o", color=ORANGE, ms=6)
    ax.annotate("가운데 $F_0 = 0.5$\n가중 4.0", (0.5, 4.0),
                textcoords="offset points", xytext=(0, -46), fontsize=9.5,
                color=INK, ha="center", linespacing=1.4)
    ax.annotate("$F_0 = 0.05$\n가중 21.1", (0.05, 21.1),
                textcoords="offset points", xytext=(26, 6), fontsize=9.5,
                color=INK, ha="left", linespacing=1.4)
    ax.annotate("$F_0 = 0.01$\n가중 101", (0.01, 101.0),
                textcoords="offset points", xytext=(26, 12), fontsize=9.5,
                color=INK, ha="left", linespacing=1.4)
    ax.set_xlabel("$F_0(x)$ — 분포의 어느 지점인가", fontsize=10.5, color=INK)
    ax.set_ylabel("같은 크기의 어긋남이 받는 무게 (로그 눈금)",
                  fontsize=10.5, color=INK)
    ax.set_title("AD 는 꼬리의 어긋남을 증폭한다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper center", labelcolor=INK)
    clean(ax)

    # --- 오른쪽: 실제 표본 하나에서 어느 구역이 통계량을 만드는가 ---
    ax = axes[1]
    rng = np.random.default_rng(31)
    n = 200
    mask = rng.random(n) < 0.12
    x = rng.normal(0, 1, n) * np.where(mask, 3.0, 1.0)   # 12% 꼬리 오염

    xs = np.sort(x)
    mu, sd = x.mean(), x.std(ddof=1)
    u = np.clip(stats.norm.cdf((xs - mu) / sd), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    d2 = ((i - 0.5) / n - u) ** 2
    raw = d2
    wtd = d2 / (u * (1 - u))
    A2 = -n - ((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))).sum() / n
    D = max((i / n - u).max(), (u - (i - 1) / n).max())
    print(f"오른쪽 칸 표본: n={n}  D={D:.4f}  A2={A2:.4f}")

    zones = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.75), (0.75, 0.95), (0.95, 1.0)]
    zone_labels = ["아래 5%", "5–25%", "가운데 50%", "75–95%", "위 5%"]
    share_raw, share_wtd = [], []
    for lo, hi in zones:
        m = (u >= lo) & (u < hi)
        share_raw.append(raw[m].sum() / raw.sum() * 100)
        share_wtd.append(wtd[m].sum() / wtd.sum() * 100)
    for lab, a, b in zip(zone_labels, share_raw, share_wtd):
        print(f"  {lab:>8s}  무가중 {a:5.1f}%   AD 가중 {b:5.1f}%")

    xsb = np.arange(len(zones))
    ax.bar(xsb - 0.19, share_raw, width=0.36, color=BLUE_F, edgecolor=BLUE,
           linewidth=1.2, label="무가중 (KS·CvM 의 시선)")
    ax.bar(xsb + 0.19, share_wtd, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           linewidth=1.2, label="AD 가중")
    for xx, v in zip(xsb - 0.19, share_raw):
        ax.annotate(f"{v:.1f}", (xx, v), textcoords="offset points",
                    xytext=(0, 3), fontsize=9, color=BLUE, ha="center")
    for xx, v in zip(xsb + 0.19, share_wtd):
        ax.annotate(f"{v:.1f}", (xx, v), textcoords="offset points",
                    xytext=(0, 3), fontsize=9, color=ORANGE, ha="center")
    ax.set_xticks(xsb)
    ax.set_xticklabels(zone_labels, fontsize=9.5, color=INK)
    ax.set_ylim(0, 78)
    ax.set_ylabel("통계량에 기여하는 비중 (%)", fontsize=10.5, color=INK)
    ax.set_xlabel("관측값이 분포의 어느 구역에 있는가", fontsize=10.5, color=INK)
    ax.set_title(f"꼬리를 12% 오염시킨 표본 하나  ($n = {n}$, $A^2 = {A2:.2f}$)",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "ad_weight_function.png")


# ===================================================================
# 그림 8. 꼬리 오염에 대한 검정력
# ===================================================================
def fig_ad_vs_ks_power():
    rng = np.random.default_rng(808)
    n, B, alpha = 100, 6000, 0.05
    eps = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30])

    d_crit = lilliefors_crit(n)
    a_crit = ad_crit5(n)
    pw = {"ad": [], "sw": [], "ks": []}
    for e in eps:
        mask = rng.random((B, n)) < e
        x = rng.normal(0, 1, size=(B, n)) * np.where(mask, 3.0, 1.0)
        pw["sw"].append(np.mean(stats.shapiro(x, axis=1).pvalue < alpha))
        pw["ad"].append(np.mean(ad_A2(x) > a_crit))
        pw["ks"].append(np.mean(ks_D(x) > d_crit))
        print(f"eps={e:.2f}  AD={pw['ad'][-1]:.3f}  SW={pw['sw'][-1]:.3f}  "
              f"KS={pw['ks'][-1]:.3f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.axhline(alpha, color=MUTED, lw=1.2, ls="--")
    ax.text(0.20, alpha + 0.018, "유의수준 0.05", fontsize=9.5, color=INK,
            ha="center", va="bottom")
    for key, col, mk, lab in [("ad", ORANGE, "o", "Anderson-Darling"),
                              ("sw", GREEN, "s", "Shapiro-Wilk"),
                              ("ks", BLUE, "^", "Lilliefors (KS)")]:
        ax.plot(eps, pw[key], mk + "-", color=col, lw=2, ms=5.5, label=lab)

    ax.set_xlim(-0.01, 0.315)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"오염 비율 $\varepsilon$ — 표준편차가 3배인 관측값의 비중",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("검정력", fontsize=10.5, color=INK)
    ax.set_title(f"꼬리 오염 $(1-\\varepsilon)N(0,1) + \\varepsilon N(0,3^2)$ 탐지력"
                 f"  ($n = {n}$, 6000회 반복)", fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="lower right", labelcolor=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "ad_vs_ks_tail_power.png")


if __name__ == "__main__":
    fig_power_by_alternative()
    fig_w_as_qq_fit()
    fig_shapiro_null_scale()
    fig_ks_ecdf_sup()
    fig_ks_shrink()
    fig_lilliefors_null_shift()
    fig_ad_weight()
    fig_ad_vs_ks_power()
