r"""9장 대응표본 세 쪽과 일표본 검정 네 쪽의 그림을 생성한다.

  ch09/paired_sample_tests/img/paired_vs_unpaired.png   같은 자료, 두 분석
  ch09/paired_sample_tests/img/paired_reduction.png     차이 하나로 줄이기
  ch09/paired_sample_tests/img/pairing_efficiency.png   짝짓기가 언제 이득인가
  ch09/one_sample_tests/img/z_rejection_two_scales.png  두 눈금 위의 기각역
  ch09/one_sample_tests/img/t_vs_z_critical.png         t 의 두꺼운 꼬리
  ch09/one_sample_tests/img/binomial_normal_condition.png  np >= 10 의 뜻
  ch09/one_sample_tests/img/chi2_variance_nonrobust.png 정규성이 깨지면

실행:  python3 scripts/make_ch09_paired_onesample_figures.py  (저장소 최상위)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

PAIR = "docs/ch09/paired_sample_tests/img/"
ONE = "docs/ch09/one_sample_tests/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 같은 자료를 짝지어 보면 / 안 짝지어 보면
# ==================================================================
def paired_vs_unpaired():
    rng = np.random.default_rng(4)
    n = 10
    # 사람마다 체지방률이 크게 다르고(개인차), 프로그램 효과는 일정하다.
    subject = rng.normal(28.0, 5.0, n)
    before = subject + rng.normal(0, 0.8, n)
    after = subject - 2.0 + rng.normal(0, 0.8, n)
    d = before - after

    t_pair = stats.ttest_rel(before, after)
    t_ind = stats.ttest_ind(before, after)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) 원자료 — 개인차가 효과를 덮는다
    ax = axes[0]
    for i in range(n):
        ax.plot([0, 1], [before[i], after[i]], color=MUTED, linewidth=1.3,
                alpha=0.8, zorder=3)
    ax.plot(np.zeros(n), before, "o", color=BLUE, markersize=9, zorder=5)
    ax.plot(np.ones(n), after, "o", color=GREEN, markersize=9, zorder=5)
    for x, v, c in [(0, before.mean(), BLUE), (1, after.mean(), GREEN)]:
        ax.plot([x - 0.14, x + 0.14], [v, v], color=c, linewidth=3.0,
                zorder=6)
    ax.text(0, before.max() + 1.6, f"평균 {before.mean():.1f}", fontsize=11,
            color=BLUE, ha="center")
    ax.text(1, before.max() + 1.6, f"평균 {after.mean():.1f}", fontsize=11,
            color=GREEN, ha="center")
    ax.text(0.5, before.min() - 3.6,
            f"짝을 무시하면  $t = {t_ind.statistic:.2f}$,  "
            f"$p = {t_ind.pvalue:.3f}$",
            fontsize=12, color=RED, ha="center", va="center")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(before.min() - 5.2, before.max() + 3.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["전", "후"], fontsize=12)
    ax.set_ylabel("체지방률 (%)", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.set_title("원자료 — 두 무리가 거의 겹친다", fontsize=13, pad=10)

    # (b) 차이만 보면
    ax = axes[1]
    ax.plot(d, np.arange(1, n + 1), "o", color=PURPLE, markersize=9,
            zorder=5)
    for i, v in enumerate(d, start=1):
        ax.plot([0, v], [i, i], color=PURPLE, linewidth=1.2, alpha=0.5,
                zorder=3)
    ax.axvline(0, color=INK, linewidth=1.8, zorder=6)
    se = d.std(ddof=1) / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    ax.plot([d.mean() - tc * se, d.mean() + tc * se], [n + 1.6] * 2,
            color=PURPLE, linewidth=3.0, solid_capstyle="butt", zorder=6)
    ax.plot([d.mean()], [n + 1.6], "o", color=PURPLE, markersize=8, zorder=7)
    ax.text(d.mean(), n + 2.3,
            f"평균 차 {d.mean():.2f},  95% 구간 "
            f"[{d.mean()-tc*se:.2f}, {d.mean()+tc*se:.2f}]",
            fontsize=10.5, color=PURPLE, ha="center", va="bottom")
    ax.text(d.mean(), -1.3,
            f"짝을 살리면  $t = {t_pair.statistic:.2f}$,  "
            f"$p = {t_pair.pvalue:.5f}$",
            fontsize=12, color=GREEN, ha="center", va="center")
    ax.set_xlim(-1.2, max(d) + 1.4)
    ax.set_ylim(-2.4, n + 3.4)
    ax.set_xlabel("차이 $d_i = $ 전 $-$ 후", fontsize=11.5, color=INK)
    ax.set_ylabel("참가자", fontsize=11.5, color=INK)
    ax.set_yticks([1, 5, 10])
    clean_axis(ax)
    ax.set_title("차이만 보면 — 흩어짐이 훨씬 작다", fontsize=13, pad=10)

    fig.suptitle("같은 자료, 같은 열 개의 쌍 — 짝을 살리느냐가 결론을 바꾼다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, PAIR + "paired_vs_unpaired.png")


# ==================================================================
# 2. 대응 t 검정은 일표본 t 검정이다
# ==================================================================
def paired_reduction():
    rng = np.random.default_rng(9)
    n = 12
    d = rng.normal(1.6, 2.2, n)
    dbar, s = d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n)
    t_obs = dbar / se
    p = 2 * (1 - stats.t.cdf(abs(t_obs), n - 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(d, np.zeros(n), "o", color=PURPLE, markersize=10, alpha=0.85,
            zorder=5)
    ax.axvline(0, color=INK, linewidth=1.8, zorder=4)
    ax.text(0, 0.42, "$H_0:\\ \\mu_d = 0$", fontsize=12, color=INK,
            ha="center", va="bottom")
    ax.plot([dbar], [0], "D", color=RED, markersize=11, zorder=7)
    ax.plot([dbar - se, dbar + se], [-0.22] * 2, color=RED, linewidth=3.0,
            solid_capstyle="butt", zorder=6)
    ax.text(dbar, -0.34, f"$\\bar d = {dbar:.2f}$\n$\\pm$ 표준오차 {se:.2f}",
            fontsize=11, color=RED, ha="center", va="top", linespacing=1.4)
    ax.set_xlim(min(d.min(), -0.6) - 0.8, d.max() + 0.8)
    ax.set_ylim(-1.0, 0.75)
    ax.set_xlabel("차이 $d_i$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.set_title(f"$n = {n}$ 개의 차이 — 여기서부터는 일표본 문제다",
                 fontsize=13, pad=10)

    ax = axes[1]
    x = np.linspace(-4.5, 4.5, 700)
    ax.plot(x, stats.t.pdf(x, n - 1), color=INK, linewidth=2.2, zorder=5)
    for side in (1, -1):
        m = (x * side) >= abs(t_obs)
        ax.fill_between(x[m], stats.t.pdf(x[m], n - 1), color=RED, alpha=0.4,
                        zorder=4)
    ax.plot([t_obs, t_obs], [0, stats.t.pdf(t_obs, n - 1)], color=RED,
            linewidth=1.8, zorder=6)
    ax.text(t_obs, stats.t.pdf(t_obs, n - 1) + 0.02,
            f"$t = {t_obs:.2f}$", fontsize=11.5, color=RED, ha="center",
            va="bottom")
    tc = stats.t.ppf(0.975, n - 1)
    for s_ in (-tc, tc):
        ax.plot([s_, s_], [0, stats.t.pdf(s_, n - 1)], color=MUTED,
                linewidth=1.3, linestyle="--", zorder=5)
    ax.text(tc, -0.022, f"$\\pm{tc:.3f}$", fontsize=10.5, color=MUTED,
            ha="center", va="top")
    ax.text(-3.4, 0.28, f"$p = {p:.4f}$", fontsize=13, color=INK,
            ha="center", va="center")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-0.012, 0.42)
    ax.set_xlabel(f"$t_{{{n-1}}}$", fontsize=12, color=INK, labelpad=16)
    bare_axis(ax)
    ax.set_title("귀무분포 위의 관측값", fontsize=13, pad=10)

    fig.suptitle("대응 t 검정 = 차이에 대한 일표본 t 검정", fontsize=13.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, PAIR + "paired_reduction.png")


# ==================================================================
# 3. 짝짓기는 언제 이득인가
# ==================================================================
def pairing_efficiency():
    rho = np.linspace(-0.3, 0.95, 400)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

    # (a) 표준오차의 비
    ax = axes[0]
    ax.plot(rho, np.sqrt(1 - rho), color=BLUE, linewidth=2.8)
    ax.axhline(1.0, color=MUTED, linewidth=1.4, linestyle="--")
    ax.axvline(0, color=MUTED, linewidth=1.2, linestyle=":")
    ax.fill_between(rho, np.sqrt(1 - rho), 1.0,
                    where=np.sqrt(1 - rho) <= 1, color=GREEN_F, alpha=0.6)
    for r in (0.3, 0.6, 0.9):
        ax.plot([r], [np.sqrt(1 - r)], "o", color=BLUE, markersize=8,
                zorder=6)
        ax.text(r, np.sqrt(1 - r) - 0.05, f"$\\rho={r}$\n"
                f"{np.sqrt(1-r):.2f}배", fontsize=10, color=BLUE,
                ha="center", va="top", linespacing=1.4)
    ax.text(0.45, 1.06, "짝짓기가 이득인 영역", fontsize=11, color=GREEN,
            ha="center", va="bottom")
    ax.set_xlim(-0.3, 0.95)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("쌍 안의 상관 $\\rho$", fontsize=11.5, color=INK)
    ax.set_ylabel("표준오차의 비  $\\sqrt{1-\\rho}$", fontsize=11.5,
                  color=INK)
    clean_axis(ax)
    ax.set_title("상관이 높을수록 차이가 조용해진다", fontsize=12.5, pad=10)

    # (b) 자유도까지 셈에 넣으면
    ax = axes[1]
    for n, color in [(5, "#90CAF9"), (10, BLUE), (30, "#0D47A1")]:
        ratio = (stats.t.ppf(0.975, n - 1) / stats.t.ppf(0.975, 2 * n - 2)
                 * np.sqrt(1 - rho))
        ax.plot(rho, ratio, color=color, linewidth=2.4, label=f"$n = {n}$")
        # 손익분기
        be = 1 - (stats.t.ppf(0.975, 2 * n - 2)
                  / stats.t.ppf(0.975, n - 1)) ** 2
        ax.plot([be], [1.0], "o", color=color, markersize=8, zorder=6)
        ax.text(be, 1.04, f"{be:.2f}", fontsize=10, color=color,
                ha="center", va="bottom")
    ax.axhline(1.0, color=RED, linewidth=1.6, linestyle="--")
    ax.text(0.93, 1.02, "여기보다 아래면 짝짓기가 낫다", fontsize=10.5,
            color=RED, ha="right", va="bottom")
    ax.text(-0.25, 0.45,
            "쌍이 적을수록 자유도를 잃는 대가가 커서\n"
            "손익분기 $\\rho$ 가 오른쪽으로 밀린다",
            fontsize=10.5, color=INK, ha="left", va="center",
            linespacing=1.5)
    ax.set_xlim(-0.3, 0.95)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("쌍 안의 상관 $\\rho$", fontsize=11.5, color=INK)
    ax.set_ylabel("신뢰구간 폭의 비", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="lower left")
    ax.set_title("자유도 손실까지 넣으면 손익분기가 생긴다", fontsize=12.5,
                 pad=10)

    fig.suptitle("짝지을 수 있다고 언제나 짝지어야 하는 것은 아니다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, PAIR + "pairing_efficiency.png")


# ==================================================================
# 4. 같은 기각역을 두 눈금에서
# ==================================================================
def z_rejection_two_scales():
    mu0, sigma, n = 50.0, 10.0, 25
    se = sigma / np.sqrt(n)
    xbar = 53.5                       # 쪽의 풀이 예제와 같은 숫자
    z_obs = (xbar - mu0) / se

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.6), sharex=False)

    for ax, scale in zip(axes, ["xbar", "z"]):
        if scale == "xbar":
            x = np.linspace(mu0 - 4 * se, mu0 + 4 * se, 700)
            dens = stats.norm.pdf(x, mu0, se)
            crit = (mu0 - 1.96 * se, mu0 + 1.96 * se)
            obs = xbar
            lab = "$\\bar x$ 의 눈금"
            ticks = [mu0 - 2 * se, mu0, mu0 + 2 * se]
            tlabs = [f"{t:.0f}" for t in ticks]
        else:
            x = np.linspace(-4, 4, 700)
            dens = stats.norm.pdf(x)
            crit = (-1.96, 1.96)
            obs = z_obs
            lab = "$z$ 의 눈금"
            ticks = [-2, 0, 2]
            tlabs = ["$-2$", "$0$", "$2$"]

        ax.plot(x, dens, color=INK, linewidth=2.2, zorder=5)
        ax.fill_between(x, dens, where=x <= crit[0], color=RED, alpha=0.35)
        ax.fill_between(x, dens, where=x >= crit[1], color=RED, alpha=0.35)
        for c in crit:
            ax.plot([c, c], [0, np.interp(c, x, dens)], color=RED,
                    linewidth=1.5, linestyle="--", zorder=6)
            ax.text(c, -dens.max() * 0.10,
                    f"{c:.2f}" if scale == "xbar" else f"${c}$",
                    fontsize=10.5, color=RED, ha="center", va="top")
        ax.plot([obs], [0], "o", color=BLUE, markersize=10, zorder=7)
        ax.text(obs, dens.max() * 0.22,
                (f"$\\bar x = {obs}$" if scale == "xbar"
                 else f"$z = {obs:.2f}$"),
                fontsize=12, color=BLUE, ha="center", va="bottom")
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(-dens.max() * 0.30, dens.max() * 1.15)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlabs, fontsize=11)
        ax.set_ylabel(lab, fontsize=11.5, color=INK)
        bare_axis(ax)

    axes[0].set_title(f"$H_0: \\mu = {mu0:g}$,  $\\sigma = {sigma:g}$,  "
                      f"$n = {n}$  →  표준오차 ${se:g}$", fontsize=12.5,
                      pad=10)
    axes[1].text(0, -0.30, "표준화는 눈금을 바꿀 뿐 그림을 바꾸지 않는다. "
                           "두 줄의 붉은 넓이는 같은 $5\\%$ 다.",
                 fontsize=11, color=INK, ha="center", va="top")
    fig.suptitle("기각역은 자료의 눈금으로도, $z$ 의 눈금으로도 적을 수 있다",
                 fontsize=13.5, y=1.0)
    fig.tight_layout()
    save(fig, ONE + "z_rejection_two_scales.png")


# ==================================================================
# 5. t 는 z 보다 꼬리가 두껍다
# ==================================================================
def t_vs_z_critical():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    x = np.linspace(-4.5, 4.5, 800)
    ax.plot(x, stats.norm.pdf(x), color=INK, linewidth=2.6,
            label="$N(0,1)$")
    for df, color in [(2, ORANGE), (5, GREEN), (30, BLUE)]:
        ax.plot(x, stats.t.pdf(x, df), color=color, linewidth=2.0,
                label=f"$t_{{{df}}}$")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(0, 0.43)
    ax.set_xlabel("통계량", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("자유도가 작을수록 꼬리가 두껍다", fontsize=13, pad=10)

    # 꼬리 확대
    axins = ax.inset_axes([0.56, 0.42, 0.42, 0.48])
    xx = np.linspace(1.8, 4.2, 300)
    axins.plot(xx, stats.norm.pdf(xx), color=INK, linewidth=2.2)
    for df, color in [(2, ORANGE), (5, GREEN), (30, BLUE)]:
        axins.plot(xx, stats.t.pdf(xx, df), color=color, linewidth=1.8)
    axins.set_xlim(1.8, 4.2)
    axins.set_ylim(0, 0.075)
    axins.tick_params(labelsize=8, colors=INK)
    axins.set_yticks([])
    axins.spines[["top", "right", "left"]].set_visible(False)
    axins.spines["bottom"].set_color(MUTED)
    axins.set_title("오른쪽 꼬리 확대", fontsize=9.5, color=INK, pad=3)

    ax = axes[1]
    dfs = np.arange(1, 61)
    crit = stats.t.ppf(0.975, dfs)
    ax.plot(dfs, crit, color=BLUE, linewidth=2.6, zorder=5)
    ax.axhline(1.96, color=RED, linewidth=1.6, linestyle="--", zorder=4)
    ax.text(60, 2.02, "$z_{0.025} = 1.96$", fontsize=11.5, color=RED,
            ha="right", va="bottom")
    for df in (2, 5, 10, 30):
        v = stats.t.ppf(0.975, df)
        ax.plot([df], [v], "o", color=BLUE, markersize=8, zorder=6)
        ax.text(df + 1.5, v, f"$n-1={df}$ : {v:.2f}", fontsize=10.5,
                color=INK, ha="left", va="center")
    ax.set_xlim(0, 62)
    ax.set_ylim(1.8, 4.6)
    ax.set_xlabel("자유도 $n - 1$", fontsize=11.5, color=INK)
    ax.set_ylabel("양측 $5\\%$ 임계값", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.set_title("$\\sigma$ 를 $S$ 로 바꾼 값", fontsize=13, pad=10)

    fig.suptitle("$\\sigma$ 를 모르면 문턱이 높아진다 — 그 대가가 $t$ 분포다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, ONE + "t_vs_z_critical.png")


# ==================================================================
# 6. np >= 10 이 무엇을 요구하는가
# ==================================================================
def binomial_normal_condition():
    cases = [(50, 0.05), (100, 0.05), (400, 0.05)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    for ax, (n, p) in zip(axes, cases):
        mu, sd = n * p, np.sqrt(n * p * (1 - p))
        ks = np.arange(max(0, int(mu - 4 * sd)), int(mu + 4 * sd) + 2)
        ax.bar(ks, stats.binom.pmf(ks, n, p), width=0.85, color=BLUE_F,
               edgecolor=BLUE, linewidth=0.9, label="이항분포")
        xs = np.linspace(ks[0] - 0.5, ks[-1] + 0.5, 400)
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=RED, linewidth=2.2,
                label="정규근사")
        ax.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
        ax.set_xlabel("성공 횟수", fontsize=11, color=INK)
        bare_axis(ax)
        ax.set_title(f"$n = {n}$, $p_0 = {p}$\n$np_0 = {n*p:.0f}$",
                     fontsize=12, color=INK, pad=8,
                     linespacing=1.4)
        if n == 50:
            ax.legend(fontsize=10, frameon=False, loc="upper right")

    fig.suptitle("$np_0$ 가 작으면 봉우리가 치우치고 계단이 성기다",
                 fontsize=13.5, y=1.04)
    fig.text(0.5, -0.03,
             "왼쪽은 $np_0 = 2.5$ 로 오른쪽으로 치우쳐 있고, 가운데는 $5$, "
             "오른쪽은 $20$ 이다. 문턱 $10$ 은 가운데와 오른쪽 사이에 있다.",
             fontsize=11, color=INK, ha="center")
    fig.tight_layout()
    save(fig, ONE + "binomial_normal_condition.png")


# ==================================================================
# 7. 분산 검정은 정규성에 기댄다
# ==================================================================
def chi2_variance_nonrobust():
    rng = np.random.default_rng(0)
    B = 20000
    ns = [10, 30, 100, 300]
    pops = [("정규", lambda s: rng.normal(0, 1, s), BLUE),
            ("균등", lambda s: rng.uniform(-1.7321, 1.7321, s), GREEN),
            ("지수", lambda s: rng.exponential(1.0, s), ORANGE),
            ("로그정규", lambda s: rng.lognormal(0, 1, s), RED)]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    for name, gen, color in pops:
        rates = []
        for n in ns:
            x = gen((B, n))
            s2 = x.var(axis=1, ddof=1)
            sigma2 = x.var()                      # 참 분산을 그대로 쓴다
            chi = (n - 1) * s2 / sigma2
            lo = stats.chi2.ppf(0.025, n - 1)
            hi = stats.chi2.ppf(0.975, n - 1)
            rates.append(np.mean((chi < lo) | (chi > hi)))
        ax.plot(ns, rates, "o-", color=color, linewidth=2.4, markersize=7,
                label=name)
        ax.text(ns[-1] * 1.04, rates[-1], f"{rates[-1]:.2f}", fontsize=10.5,
                color=color, ha="left", va="center")

    ax.axhline(0.05, color=INK, linewidth=1.6, linestyle="--")
    ax.text(9.5, 0.062, "명목 $\\alpha = 0.05$", fontsize=11, color=INK,
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlim(8, 480)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=11.5, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="center right",
              bbox_to_anchor=(0.99, 0.56), title="모집단")
    ax.set_title("$\\sigma^2$ 에 대한 카이제곱 검정의 실제 오류율",
                 fontsize=13.5, pad=10)
    fig.text(0.5, -0.02,
             "정규모집단에서만 명목 수준이 지켜진다. 표본을 키워도 나머지는 "
             "제자리로 돌아오지 않는다.", fontsize=11, color=INK,
             ha="center")
    fig.tight_layout()
    save(fig, ONE + "chi2_variance_nonrobust.png")


if __name__ == "__main__":
    paired_vs_unpaired()
    paired_reduction()
    pairing_efficiency()
    z_rejection_two_scales()
    t_vs_z_critical()
    binomial_normal_condition()
    chi2_variance_nonrobust()
