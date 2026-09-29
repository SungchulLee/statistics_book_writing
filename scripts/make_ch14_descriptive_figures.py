r"""14장 기술통계량 절 여덟 쪽의 개념 그림을 만든다.

만드는 파일:
  ch14/descriptive_statistics/img/moment_weights_tails.png    3·4차 적률은 꼬리가 만든다
  ch14/descriptive_statistics/img/skew_vs_kurt_blindspot.png  두 검정은 서로의 맹점이다
  ch14/descriptive_statistics/img/dagostino_simple_vs_scipy_z.png  변환이 하는 일
  ch14/descriptive_statistics/img/jb_chi2_calibration.png     JB 의 카이제곱 근사
  ch14/descriptive_statistics/img/skew_cutoff_vs_n.png        큰 왜도의 기준은 n 이 정한다
  ch14/descriptive_statistics/img/kurt_g2_null_skew.png       g2 의 귀무분포는 치우쳐 있다
  ch14/descriptive_statistics/img/dagostino_z1_z2_plane.png   K제곱은 평면 위의 거리다
  ch14/descriptive_statistics/img/jb_decomposition.png        JB 를 두 항으로 쪼개면

실행:  python3 scripts/make_ch14_descriptive_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats, integrate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch14/descriptive_statistics/img/"
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
# 그림 1. 3차·4차 적률은 꼬리의 몇 안 되는 점이 만든다
# ===================================================================
def fig_moment_weights():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- 왼쪽: |z|^k ---
    ax = axes[0]
    z = np.linspace(0.05, 3.6, 500)
    for k, col, lab in [(2, BLUE, r"$z^2$  (분산)"),
                        (3, GREEN, r"$|z|^3$  (왜도)"),
                        (4, ORANGE, r"$z^4$  (첨도)")]:
        ax.plot(z, z**k, color=col, lw=2.4, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(0.002, 400)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])   # 평문 라벨
    ax.set_xlim(0, 3.6)
    ax.axvline(1.0, color=MUTED, lw=1.0, ls=":")
    ax.axvline(3.0, color=MUTED, lw=1.0, ls=":")
    ax.text(1.02, 0.0035, "$|z| = 1$", fontsize=9.5, color=INK, ha="left")
    ax.text(3.02, 0.0035, "$|z| = 3$", fontsize=9.5, color=INK, ha="left")
    ax.annotate("$3^4 = 81$", (3.0, 81), textcoords="offset points",
                xytext=(-8, 4), fontsize=9.5, color=ORANGE, ha="right")
    ax.annotate("$3^2 = 9$", (3.0, 9), textcoords="offset points",
                xytext=(-8, -14), fontsize=9.5, color=BLUE, ha="right")
    ax.set_xlabel("표준화한 관측값 $|z|$", fontsize=10.5, color=INK)
    ax.set_ylabel("그 관측값이 적률 합에 보태는 양 (로그 눈금)",
                  fontsize=10.5, color=INK)
    ax.set_title("차수가 높아지면 꼬리만 남는다", fontsize=11.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper left", labelcolor=INK)
    clean(ax)

    # --- 오른쪽: 정규분포에서 구역별 몫 ---
    ax = axes[1]
    edges = [0, 1, 2, 3, np.inf]
    labels = ["$|z| < 1$", "$1 \\leq |z| < 2$", "$2 \\leq |z| < 3$",
              "$|z| \\geq 3$"]
    mass, share4 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi_ = 12.0 if np.isinf(hi) else hi
        mass.append(2 * (stats.norm.cdf(hi_) - stats.norm.cdf(lo)))
        val, _ = integrate.quad(lambda t: 2 * t**4 * stats.norm.pdf(t), lo, hi_)
        share4.append(val / 3.0)
    mass = np.array(mass) * 100
    share4 = np.array(share4) * 100
    for lab, m, s in zip(labels, mass, share4):
        print(f"  {lab:>16s}  관측 비율 {m:5.2f}%   4차적률 기여 {s:5.2f}%")

    xs = np.arange(4)
    ax.bar(xs - 0.19, mass, width=0.36, color=BLUE_F, edgecolor=BLUE,
           linewidth=1.2, label="전체 관측값 중 비율")
    ax.bar(xs + 0.19, share4, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           linewidth=1.2, label=r"$\sum z^4$ 에 대한 기여")
    for xx, v in zip(xs - 0.19, mass):
        ax.annotate(f"{v:.1f}%", (xx, v), textcoords="offset points",
                    xytext=(0, 3), fontsize=9, color=BLUE, ha="center")
    for xx, v in zip(xs + 0.19, share4):
        ax.annotate(f"{v:.1f}%", (xx, v), textcoords="offset points",
                    xytext=(0, 3), fontsize=9, color=ORANGE, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_ylim(0, 85)
    ax.set_ylabel("백분율", fontsize=10.5, color=INK)
    ax.set_title("정규분포에서 각 구역이 차지하는 몫", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "moment_weights_tails.png")


# ===================================================================
# 그림 2. 왜도 검정과 첨도 검정은 서로의 맹점을 덮는다
# ===================================================================
def fig_blindspot():
    rng = np.random.default_rng(101)
    n, B, alpha = 100, 8000, 0.05

    a = 5.0
    sn_m, sn_s = stats.skewnorm.stats(a, moments="mv")
    sn_s = np.sqrt(sn_s)
    print("skewnorm(5): 왜도", stats.skewnorm.stats(a, moments="s"),
          " 초과첨도", stats.skewnorm.stats(a, moments="k"))
    print("t5: 초과첨도", stats.t.stats(5, moments="k"))

    def draw(kind):
        if kind == "normal":
            return rng.normal(0, 1, size=(B, n))
        if kind == "skew":
            return (stats.skewnorm.rvs(a, size=(B, n),
                                       random_state=rng) - sn_m) / sn_s
        if kind == "thin":
            return rng.uniform(-1, 1, size=(B, n))
        if kind == "heavy":
            return rng.standard_t(5, size=(B, n)) / np.sqrt(5 / 3)
        raise ValueError(kind)

    kinds = ["normal", "skew", "thin", "heavy"]
    kind_labels = ["정규\n(크기 확인)", "치우친 자료\n왜도 $0.85$",
                   "균등분포\n초과첨도 $-1.2$",
                   "$t_5$\n초과첨도 $+6$"]
    table = np.zeros((3, 4))
    for j, kd in enumerate(kinds):
        x = draw(kd)
        table[0, j] = np.mean(stats.skewtest(x, axis=1).pvalue < alpha)
        table[1, j] = np.mean(stats.kurtosistest(x, axis=1).pvalue < alpha)
        table[2, j] = np.mean(stats.normaltest(x, axis=1).pvalue < alpha)
        print(f"  {kd:8s} skewtest={table[0,j]:.3f}  "
              f"kurtosistest={table[1,j]:.3f}  K2={table[2,j]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    # --- 왼쪽: 네 모집단의 밀도 ---
    ax = axes[0]
    g = np.linspace(-4.2, 4.2, 600)
    ax.plot(g, stats.norm.pdf(g), color=MUTED, lw=2.4, ls="--",
            label="정규 $N(0,1)$")
    ax.plot(g, stats.skewnorm.pdf(g * sn_s + sn_m, a) * sn_s, color=BLUE,
            lw=2.2, label="치우친 자료 (왜도 $0.85$)")
    ax.plot(g, np.where(np.abs(g) <= np.sqrt(3), 1 / (2 * np.sqrt(3)), 0),
            color=GREEN, lw=2.2, label="균등분포 (얇은 꼬리)")
    ax.plot(g, stats.t.pdf(g * np.sqrt(5 / 3), 5) * np.sqrt(5 / 3),
            color=ORANGE, lw=2.2, label="$t_5$ (두꺼운 꼬리)")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(0, 0.56)
    ax.set_xlabel("표준화한 값", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("평균과 분산은 같고 모양만 다른 네 분포",
                 fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left", labelcolor=INK)
    clean(ax)

    # --- 오른쪽: 검정력 막대 ---
    ax = axes[1]
    xs = np.arange(4)
    w = 0.26
    for r, (col, colf, lab) in enumerate([
            (BLUE, BLUE_F, "왜도 검정"),
            (ORANGE, ORANGE_F, "첨도 검정"),
            (PURPLE, "#E1D3EC", r"D'Agostino $K^2$")]):
        ax.bar(xs + (r - 1) * w, table[r], width=w * 0.92, color=colf,
               edgecolor=col, linewidth=1.2, label=lab)
        for xx, v in zip(xs + (r - 1) * w, table[r]):
            ax.annotate(f"{v:.2f}", (xx, v), textcoords="offset points",
                        xytext=(0, 3), fontsize=8.5, color=col, ha="center")
    ax.axhline(alpha, color=MUTED, lw=1.2, ls="--")
    ax.text(0.0, alpha + 0.035, "유의수준 0.05", fontsize=9, color=INK,
            ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(kind_labels, fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("기각률", fontsize=10.5, color=INK)
    ax.set_title(f"$n = {n}$, 8000회 반복", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left", labelcolor=INK,
              ncol=1)
    clean(ax)

    fig.tight_layout()
    save(fig, "skew_vs_kurt_blindspot.png")


# ===================================================================
# 그림 3. 점근 z 점수와 scipy 의 변환된 z 점수
# ===================================================================
def simple_z(x):
    """이 쪽에 적힌 단순 표준화 z 점수(점근 근사)."""
    n = x.shape[1]
    S = stats.skew(x, axis=1)
    K = stats.kurtosis(x, axis=1)     # 이미 초과첨도
    se_s = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    se_k = np.sqrt(24 * n * (n - 1) ** 2 /
                   ((n - 3) * (n - 2) * (n + 3) * (n + 5)))
    return S / se_s, K / se_k


def fig_simple_vs_scipy_z():
    rng = np.random.default_rng(2)
    n, B = 30, 80000
    x = rng.normal(size=(B, n))
    zs_simple, zk_simple = simple_z(x)
    zs_scipy = stats.skewtest(x, axis=1).statistic
    zk_scipy = stats.kurtosistest(x, axis=1).statistic

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    g = np.linspace(-4.5, 4.5, 500)
    for ax, simple, scip, name in [
            (axes[0], zs_simple, zs_scipy, "왜도"),
            (axes[1], zk_simple, zk_scipy, "첨도")]:
        s1 = float(np.mean(np.abs(simple) > 1.96))
        s2 = float(np.mean(np.abs(scip) > 1.96))
        print(f"{name}: 단순 z 의 실제 크기={s1:.4f}   scipy z={s2:.4f}")
        ax.hist(simple, bins=170, range=(-4.5, 4.5), density=True,
                color=ORANGE_F, edgecolor=ORANGE, linewidth=0.25,
                label="이 쪽의 단순 $z$ 점수")
        ax.hist(scip, bins=170, range=(-4.5, 4.5), density=True,
                histtype="step", color=BLUE, linewidth=1.8,
                label="scipy 의 변환된 $z$ 점수")
        ax.plot(g, stats.norm.pdf(g), color=INK, lw=2.0, ls="--",
                label="$N(0,1)$")
        ax.set_xlim(-4.5, 4.5)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.35)
        ax.text(0.03, 0.97,
                f"$|z| > 1.96$ 인 비율\n단순 {s1:.4f}   scipy {s2:.4f}",
                transform=ax.transAxes, fontsize=9.5, color=INK,
                va="top", ha="left", linespacing=1.6)
        ax.set_title(f"{name} $z$ 점수의 귀무분포  ($n = {n}$)",
                     fontsize=11.5, color=INK)
        ax.set_xlabel("$z$", fontsize=10.5, color=INK)
        ax.legend(fontsize=9, frameon=False, loc="upper right",
                  labelcolor=INK)
        clean(ax)
        ax.set_yticks([])
    axes[0].set_ylabel("밀도", fontsize=10.5, color=INK)

    fig.suptitle("정규자료 80000개 표본: 변환은 유한표본에서 분포를 바로잡는다",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "dagostino_simple_vs_scipy_z.png")


# ===================================================================
# 그림 4. JB 의 카이제곱 근사는 얼마나 맞는가
# ===================================================================
def jb_stat(x):
    n = x.shape[1]
    g1 = stats.skew(x, axis=1)
    g2 = stats.kurtosis(x, axis=1)
    return n / 6 * (g1**2 + g2**2 / 4)


def fig_jb_calibration():
    rng = np.random.default_rng(404)
    n = 50
    chunks, per = 12, 200_000
    vals = []
    for _ in range(chunks):
        vals.append(jb_stat(rng.normal(size=(per, n))))
    jb = np.concatenate(vals)
    B = jb.size
    q95 = float(np.quantile(jb, 0.95))
    print(f"n={n}  B={B}  JB 의 95백분위={q95:.4f}  (chi2_2 의 값 5.991)")
    print(f"  5.991 을 넘는 비율 = {np.mean(jb > stats.chi2.ppf(0.95, 2)):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # --- 왼쪽: 귀무밀도 비교 ---
    ax = axes[0]
    g = np.linspace(0.01, 12, 500)
    ax.hist(jb, bins=240, range=(0, 12), density=True, color=ORANGE_F,
            edgecolor=ORANGE, linewidth=0.2,
            label=f"$n = {n}$ 에서 $JB$ 의 실제 분포")
    ax.plot(g, stats.chi2.pdf(g, 2), color=BLUE, lw=2.2,
            label=r"가정한 $\chi^2_2$")
    ax.axvline(5.991, color=BLUE, lw=1.5, ls="--")
    ax.axvline(q95, color=ORANGE, lw=1.5, ls="--")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 0.62)
    ax.text(5.991 + 0.25, 0.30, "$\\chi^2_2$ 의 95% 값\n5.991", fontsize=9.5,
            color=BLUE, ha="left", va="top", linespacing=1.5)
    ax.text(q95 - 0.25, 0.30, f"실제 95백분위\n{q95:.2f}", fontsize=9.5,
            color=ORANGE, ha="right", va="top", linespacing=1.5)
    ax.set_xlabel("$JB$", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("귀무분포가 가정보다 왼쪽에 있다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)

    # --- 오른쪽: p-값 눈금 맞춤 ---
    ax = axes[1]
    nominal = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002,
                        0.001, 0.0005])
    thr = stats.chi2.ppf(1 - nominal, 2)
    actual = np.array([np.mean(jb > t) for t in thr])
    for a, b in zip(nominal, actual):
        print(f"  chi2 가 말하는 p={a:<8g} 실제 p={b:.5f}")
    lim = (3e-4, 0.8)
    ax.plot(lim, lim, color=MUTED, lw=1.4, ls="--", label="둘이 같다면")
    ax.plot(nominal, actual, "o-", color=ORANGE, lw=2, ms=6,
            label=f"$n = {n}$ 의 $JB$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ticks = [0.001, 0.01, 0.1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.001", "0.01", "0.1"])     # 로그축 평문 라벨
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.001", "0.01", "0.1"])
    ax.minorticks_off()
    ax.set_xlabel(r"$\chi^2_2$ 가 말하는 $p$ 값", fontsize=10.5, color=INK)
    ax.set_ylabel("실제 $p$ 값 (모의실험)", fontsize=10.5, color=INK)
    ax.set_title("작은 $p$ 값일수록 더 크게 어긋난다", fontsize=11.5, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper left", labelcolor=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "jb_chi2_calibration.png")


# ===================================================================
# 그림 5. "큰 왜도"의 기준은 n 이 정한다
# ===================================================================
def fig_skew_cutoff():
    rng = np.random.default_rng(55)
    ns = np.array([10, 20, 30, 50, 100, 200, 500, 1000, 2000])
    B = 40000
    lo, hi = [], []
    for n in ns:
        g1 = stats.skew(rng.normal(size=(B, n)), axis=1, bias=False)
        lo.append(np.quantile(g1, 0.025))
        hi.append(np.quantile(g1, 0.975))
        print(f"n={n:5d}  g1 의 95% 범위 [{lo[-1]:+.3f}, {hi[-1]:+.3f}]")
    lo, hi = np.array(lo), np.array(hi)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.fill_between(ns, lo, hi, color=BLUE_F, edgecolor=BLUE, linewidth=1.4,
                    label="정규자료에서 $g_1$ 의 95% 범위")
    ax.axhline(0, color=MUTED, lw=1.2, ls="--")
    ax.plot(ns, 1.96 * np.sqrt(6 / ns), color=MUTED, lw=1.6, ls=":",
            label=r"교과서 어림값 $\pm 1.96\sqrt{6/n}$")
    ax.plot(ns, -1.96 * np.sqrt(6 / ns), color=MUTED, lw=1.6, ls=":")

    ax.plot([300], [2.2452], "o", color=RED, ms=8, zorder=5)
    ax.annotate("예제의 대수정규 자료\n$n = 300$, $g_1 = 2.245$",
                (300, 2.2452), textcoords="offset points", xytext=(0, -16),
                fontsize=10, color=RED, ha="center", va="top",
                linespacing=1.5)

    ax.annotate(f"$n = 20$ 에서는\n$|g_1| < {hi[1]:.2f}$ 가 평범하다",
                (20, hi[1]), textcoords="offset points", xytext=(14, 16),
                fontsize=9.5, color=BLUE, ha="left", linespacing=1.5)
    ax.annotate(f"$n = 1000$ 에서는\n$|g_1| > {hi[7]:.2f}$ 면 유의하다",
                (1000, hi[7]), textcoords="offset points", xytext=(-6, 30),
                fontsize=9.5, color=BLUE, ha="right", linespacing=1.5)

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.minorticks_off()
    ax.set_xlim(9, 2400)
    ax.set_ylim(-1.6, 2.7)
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("표본왜도 $g_1$", fontsize=10.5, color=INK)
    ax.set_title("같은 왜도 값도 $n$ 에 따라 전혀 다른 증거가 된다",
                 fontsize=12, color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "skew_cutoff_vs_n.png")


# ===================================================================
# 그림 6. 표본 초과첨도의 귀무분포는 표본이 커져도 오래 치우쳐 있다
# ===================================================================
def fig_g2_null_skew():
    rng = np.random.default_rng(606)
    ns = [50, 200, 1000]
    B = 60000
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=False)

    for ax, n in zip(axes, ns):
        g2 = stats.kurtosis(rng.normal(size=(B, n)), axis=1, bias=False)
        med = float(np.median(g2))
        q_lo = float(np.quantile(g2, 0.025))
        q_hi = float(np.quantile(g2, 0.975))
        sk = float(stats.skew(g2))
        print(f"n={n:5d}  중앙값={med:+.3f}  95% 범위 [{q_lo:+.3f}, {q_hi:+.3f}]"
              f"  귀무분포의 왜도={sk:.3f}")

        span = max(abs(q_lo), abs(q_hi)) * 1.9
        ax.hist(g2, bins=180, range=(-span, span), density=True,
                color=ORANGE_F, edgecolor=ORANGE, linewidth=0.25)
        ax.axvline(0, color=INK, lw=1.4, ymax=0.76)
        ax.axvline(q_lo, color=BLUE, lw=1.5, ls="--", ymax=0.70)
        ax.axvline(q_hi, color=BLUE, lw=1.5, ls="--", ymax=0.70)
        ax.set_xlim(-span, span)
        top = ax.get_ylim()[1]
        ax.set_ylim(0, top * 1.32)
        ax.text(0.03, 0.97,
                f"95% 범위\n[{q_lo:+.2f},  {q_hi:+.2f}]\n"
                f"이 분포 자체의 왜도  {sk:.2f}",
                transform=ax.transAxes, fontsize=9.5, color=INK,
                va="top", ha="left", linespacing=1.6)
        ax.set_title(f"$n = {n}$", fontsize=12, color=INK)
        ax.set_xlabel("표본 초과첨도 $g_2$", fontsize=10.5, color=INK)
        clean(ax)
        ax.set_yticks([])
    axes[0].set_ylabel("밀도", fontsize=10.5, color=INK)

    fig.suptitle("정규자료에서 $g_2$ 의 귀무분포: 왼쪽은 막혀 있고 오른쪽만 길다",
                 fontsize=12.5, color=INK, y=1.03)
    fig.tight_layout()
    save(fig, "kurt_g2_null_skew.png")


# ===================================================================
# 그림 7. K제곱은 (Z1, Z2) 평면 위의 거리다
# ===================================================================
def fig_z1z2_plane():
    rng = np.random.default_rng(77)
    n, B = 50, 900

    groups = []
    groups.append((rng.normal(0, 1, size=(B, n)), MUTED, "정규 자료"))
    groups.append((stats.skewnorm.rvs(4, size=(B, n), random_state=rng),
                   BLUE, "치우친 자료 (왜도 $0.78$)"))
    groups.append((rng.standard_t(6, size=(B, n)), ORANGE,
                   "두꺼운 꼬리 ($t_6$)"))
    groups.append((rng.uniform(-1, 1, size=(B, n)), GREEN,
                   "얇은 꼬리 (균등분포)"))

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    r = np.sqrt(stats.chi2.ppf(0.95, 2))
    ax.add_patch(Circle((0, 0), r, facecolor="none", edgecolor=PURPLE,
                        lw=2.2, ls="-", zorder=4,
                        label=r"$K^2 = 5.99$ — $K^2$ 검정의 5% 경계"))
    ax.add_patch(Rectangle((-1.96, -1.96), 3.92, 3.92, facecolor="none",
                           edgecolor=MUTED, lw=1.5, ls="--", zorder=3,
                           label="각 검정을 따로 볼 때의 5% 경계"))

    for x, col, lab in groups:
        z1 = stats.skewtest(x, axis=1).statistic
        z2 = stats.kurtosistest(x, axis=1).statistic
        rej = np.mean(stats.normaltest(x, axis=1).pvalue < 0.05)
        print(f"{lab}: Z1 중앙값={np.median(z1):.2f}  Z2 중앙값={np.median(z2):.2f}"
              f"  K2 기각률={rej:.3f}")
        ax.scatter(z1, z2, s=9, alpha=0.40, color=col, edgecolor="none",
                   label=lab, zorder=2)

    ax.plot([1.8], [1.8], "*", color=RED, ms=17, zorder=7)
    ax.annotate("$Z_1 = Z_2 = 1.8$\n각각은 유의하지 않지만\n$K^2 = 6.48$ 은 기각",
                xy=(1.8, 1.8), xytext=(2.3, -4.9),
                fontsize=9.5, color=RED, ha="left", va="center",
                linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-8.0, 5.0)
    ax.set_aspect("equal")
    ax.set_xlabel("$Z_1$  (왜도 성분)", fontsize=11, color=INK)
    ax.set_ylabel("$Z_2$  (첨도 성분)", fontsize=11, color=INK)
    ax.set_title(f"$K^2$ 은 원점에서 잰 거리의 제곱이다  ($n = {n}$)",
                 fontsize=12.5, color=INK)
    leg = ax.legend(fontsize=9.5, frameon=False, loc="lower left",
                    labelcolor=INK, markerscale=2.4)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    clean(ax)
    fig.tight_layout()
    save(fig, "dagostino_z1_z2_plane.png")


# ===================================================================
# 그림 8. JB 를 왜도 항과 첨도 항으로 쪼개면
# ===================================================================
def fig_jb_decomposition():
    n = 300
    sets = []
    rng = np.random.default_rng(0)
    sets.append(("정규 $N(0,1)$", rng.normal(0, 1, size=n)))
    rng = np.random.default_rng(0)
    sets.append(("정규 240 + 대수정규 60",
                 np.concatenate([rng.normal(0, 1, size=240),
                                 rng.lognormal(0, 0.6, size=60)])))
    rng = np.random.default_rng(6)
    sets.append((r"$t_5$ (대칭, 두꺼운 꼬리)", rng.standard_t(5, size=n)))
    rng = np.random.default_rng(4)
    sets.append(("균등분포 (대칭, 얇은 꼬리)", rng.uniform(-1, 1, size=n)))

    names, sk_share, ku_share, totals, pvals = [], [], [], [], []
    for name, x in sets:
        g1 = stats.skew(x)
        g2 = stats.kurtosis(x)
        t_sk = n / 6 * g1**2
        t_ku = n / 24 * g2**2
        tot = t_sk + t_ku
        p = stats.chi2.sf(tot, 2)
        names.append(name)
        sk_share.append(t_sk / tot * 100)
        ku_share.append(t_ku / tot * 100)
        totals.append(tot)
        pvals.append(p)
        print(f"{name}: g1={g1:+.4f} g2={g2:+.4f}  왜도항={t_sk:.2f} "
              f"첨도항={t_ku:.2f}  JB={tot:.2f}  p={p:.3g}")

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, sk_share, height=0.5, color=BLUE_F, edgecolor=BLUE,
            linewidth=1.2, label=r"왜도 항  $\frac{n}{6}g_1^2$")
    ax.barh(ys, ku_share, left=sk_share, height=0.5, color=ORANGE_F,
            edgecolor=ORANGE, linewidth=1.2,
            label=r"첨도 항  $\frac{n}{24}g_2^2$")
    for y, a, b in zip(ys, sk_share, ku_share):
        if a > 8:
            ax.text(a / 2, y, f"{a:.0f}%", ha="center", va="center",
                    fontsize=10, color=BLUE)
        if b > 8:
            ax.text(a + b / 2, y, f"{b:.0f}%", ha="center", va="center",
                    fontsize=10, color=ORANGE)
    for y, tot, p in zip(ys, totals, pvals):
        txt = f"$JB = {tot:.2f}$,   $p = {p:.3g}$" if p >= 1e-4 else \
              f"$JB = {tot:.2f}$,   $p < 0.0001$"
        ax.text(103, y, txt, ha="left", va="center", fontsize=10, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("$JB$ 에서 각 항이 차지하는 비중 (%)", fontsize=10.5,
                  color=INK)
    ax.set_title(f"같은 $JB$ 라도 어느 항이 만들었는지는 자료마다 다르다  ($n = {n}$)",
                 fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.42), labelcolor=INK, ncol=2)
    clean(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "jb_decomposition.png")


if __name__ == "__main__":
    fig_moment_weights()
    fig_blindspot()
    fig_simple_vs_scipy_z()
    fig_jb_calibration()
    fig_skew_cutoff()
    fig_g2_null_skew()
    fig_z1z2_plane()
    fig_jb_decomposition()
