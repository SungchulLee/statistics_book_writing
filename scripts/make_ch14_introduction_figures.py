r"""14장 입문 절 세 쪽의 개념 그림을 만든다.

만드는 파일:
  ch14/introduction/img/clt_mean_vs_variance.png   CLT 는 평균만 구해 준다
  ch14/introduction/img/type1_vs_n.png             n 을 늘리면 무엇이 낫는가
  ch14/introduction/img/aggregation_kurtosis.png   집계는 꼬리를 천천히 지운다

실행:  python3 scripts/make_ch14_introduction_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch14/introduction/img/"
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
# 그림 1. CLT 는 평균을 구해 주지만 분산을 구해 주지 않는다
# ===================================================================
def fig_clt_mean_vs_variance():
    rng = np.random.default_rng(14)
    sigma_ln = 0.6                      # 대수정규 모양모수: 왜도 2.26
    mu_pop = np.exp(sigma_ln**2 / 2)
    var_pop = (np.exp(sigma_ln**2) - 1) * np.exp(sigma_ln**2)
    sd_pop = np.sqrt(var_pop)

    B = 40000
    ns = [30, 200]
    stats_out = {}

    for n in ns:
        x = rng.lognormal(0.0, sigma_ln, size=(B, n))
        xbar = x.mean(axis=1)
        s2 = x.var(axis=1, ddof=1)
        t = (xbar - mu_pop) / np.sqrt(s2 / n)
        q = (n - 1) * s2 / var_pop

        tcrit = stats.t.ppf(0.975, n - 1)
        cov_t = np.mean(np.abs(t) < tcrit)
        lo = stats.chi2.ppf(0.025, n - 1)
        hi = stats.chi2.ppf(0.975, n - 1)
        cov_chi = np.mean((q > lo) & (q < hi))
        stats_out[n] = (cov_t, cov_chi, t, q)
        print(f"n={n:4d}  t구간 포함률={cov_t:.3f}   카이제곱구간 포함률={cov_chi:.3f}")

    print("모집단 왜도", stats.lognorm.stats(sigma_ln, moments="s"))

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))

    for j, n in enumerate(ns):
        cov_t, cov_chi, t, q = stats_out[n]

        # --- 위: 표준화한 평균 ---
        ax = axes[0, j]
        grid = np.linspace(-5, 4, 400)
        ax.hist(t, bins=110, range=(-5, 4), density=True,
                color=BLUE_F, edgecolor=BLUE, linewidth=0.3)
        ax.plot(grid, stats.t.pdf(grid, n - 1), color=INK, lw=1.8)
        ax.set_xlim(-5, 4)
        ax.set_title(f"표준화한 평균  ($n = {n}$)", fontsize=11, color=INK)
        ax.text(0.03, 0.93, f"95% $t$ 구간 포함률  {cov_t:.3f}",
                transform=ax.transAxes, fontsize=9.5, color=BLUE,
                va="top", ha="left")
        ax.set_xlabel(r"$(\bar{X}-\mu)\,/\,(S/\sqrt{n})$", fontsize=10, color=INK)
        clean(ax)
        ax.set_yticks([])

    for j, n in enumerate(ns):
        cov_t, cov_chi, t, q = stats_out[n]

        # --- 아래: 분산 피벗 ---
        ax = axes[1, j]
        hi_x = stats.chi2.ppf(0.9995, n - 1) * 2.4
        grid = np.linspace(0, hi_x, 500)
        ax.hist(q, bins=130, range=(0, hi_x), density=True,
                color=ORANGE_F, edgecolor=ORANGE, linewidth=0.3)
        ax.plot(grid, stats.chi2.pdf(grid, n - 1), color=INK, lw=1.8)
        ax.set_xlim(0, hi_x)
        ax.set_title(f"분산 피벗  ($n = {n}$)", fontsize=11, color=INK)
        ax.text(0.97, 0.93, f"95% $\\chi^2$ 구간 포함률  {cov_chi:.3f}",
                transform=ax.transAxes, fontsize=9.5, color=ORANGE,
                va="top", ha="right")
        ax.set_xlabel(r"$(n-1)S^2/\sigma^2$", fontsize=10, color=INK)
        clean(ax)
        ax.set_yticks([])

    # 검은 실선이 무엇인지 한 번만 밝힌다
    axes[0, 0].plot([], [], color=INK, lw=1.8, label="정규성 아래의 이론 분포")
    axes[0, 0].legend(fontsize=9, frameon=False, loc="upper right",
                      labelcolor=INK)

    fig.suptitle("대수정규 모집단에서 뽑은 표본: 평균은 이론에 수렴하지만 분산은 그렇지 않다",
                 fontsize=12.5, color=INK, y=1.00)
    fig.tight_layout()
    save(fig, "clt_mean_vs_variance.png")


# ===================================================================
# 그림 2. 표본을 늘리면 평균 검정은 낫지만 분산 검정은 그대로다
# ===================================================================
def fig_type1_vs_n():
    rng = np.random.default_rng(2024)
    ns = np.array([10, 15, 20, 30, 50, 80, 120, 200, 400, 800])
    B = 40000
    alpha = 0.05

    err_t = []
    err_chi = []
    for n in ns:
        x = rng.exponential(1.0, size=(B, n))     # 평균 1, 분산 1
        xbar = x.mean(axis=1)
        s2 = x.var(axis=1, ddof=1)

        t = (xbar - 1.0) / np.sqrt(s2 / n)
        err_t.append(np.mean(np.abs(t) > stats.t.ppf(1 - alpha / 2, n - 1)))

        q = (n - 1) * s2 / 1.0
        lo = stats.chi2.ppf(alpha / 2, n - 1)
        hi = stats.chi2.ppf(1 - alpha / 2, n - 1)
        err_chi.append(np.mean((q < lo) | (q > hi)))
        print(f"n={n:4d}  t검정 크기={err_t[-1]:.4f}   카이제곱검정 크기={err_chi[-1]:.4f}")

    err_t = np.array(err_t)
    err_chi = np.array(err_chi)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(alpha, color=MUTED, lw=1.2, ls="--")
    ax.text(ns[-1], alpha - 0.012, "약속한 유의수준 0.05", fontsize=9.5,
            color=INK, ha="right", va="top")

    ax.plot(ns, err_chi, "o-", color=ORANGE, lw=2, ms=5,
            label=r"분산에 대한 $\chi^2$ 검정")
    ax.plot(ns, err_t, "s-", color=BLUE, lw=2, ms=5,
            label=r"평균에 대한 $t$ 검정")

    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])      # 로그축 라벨은 평문으로
    ax.minorticks_off()
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.set_ylim(0, 0.38)
    ax.set_title("지수분포 모집단, 귀무가설은 참: 40000회 반복", fontsize=12, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="center right", labelcolor=INK)

    ax.annotate(f"{err_t[0]:.3f}", (ns[0], err_t[0]), textcoords="offset points",
                xytext=(0, -14), fontsize=9, color=BLUE, ha="center")
    ax.annotate(f"{err_t[-1]:.3f}", (ns[-1], err_t[-1]), textcoords="offset points",
                xytext=(-2, 9), fontsize=9, color=BLUE, ha="right")
    ax.annotate(f"{err_chi[0]:.3f}", (ns[0], err_chi[0]), textcoords="offset points",
                xytext=(0, -15), fontsize=9, color=ORANGE, ha="center")
    ax.annotate(f"{err_chi[-1]:.3f}", (ns[-1], err_chi[-1]),
                textcoords="offset points", xytext=(-2, 9), fontsize=9,
                color=ORANGE, ha="right")
    clean(ax)
    fig.tight_layout()
    save(fig, "type1_vs_n.png")


# ===================================================================
# 그림 3. 집계는 꼬리를 지우지만 아주 천천히 지운다
# ===================================================================
def fig_aggregation_kurtosis():
    rng = np.random.default_rng(7)
    # GARCH(1,1) + t(7) 혁신: 변동성 군집과 두꺼운 꼬리를 함께 갖는다.
    # 400년 대신 4000년치를 만들어야 연간 수익률 4000개로 첨도를 잴 수 있다.
    T = 252 * 4000
    a, b = 0.12, 0.80
    omega = 2e-6 * (1 - a - b) / 0.02      # 무조건분산을 1% 일간변동성 근처로
    nu = 7.0
    z = stats.t.rvs(df=nu, size=T, random_state=11) / np.sqrt(nu / (nu - 2))
    h = np.empty(T)
    r = np.empty(T)
    h[0] = omega / (1 - a - b)
    for i in range(T):
        if i > 0:
            h[i] = omega + a * r[i - 1] ** 2 + b * h[i - 1]
        r[i] = np.sqrt(h[i]) * z[i]

    horizons = [1, 5, 21, 63, 252]
    names = ["1일\n(일간)", "5일\n(주간)", "21일\n(월간)", "63일\n(분기)", "252일\n(연간)"]
    kurt, tail = [], []
    for hz in horizons:
        m = (T // hz) * hz
        agg = r[:m].reshape(-1, hz).sum(axis=1)
        k = stats.kurtosis(agg)
        sd = agg.std(ddof=1)
        p3 = np.mean(np.abs(agg - agg.mean()) > 3 * sd)
        kurt.append(k)
        tail.append(p3)
        print(f"h={hz:4d}  개수={len(agg):6d}  초과첨도={k:6.3f}  P(|R|>3s)={p3:.4f}")

    normal_tail = 2 * (1 - stats.norm.cdf(3))
    print("정규 꼬리확률", normal_tail)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    ax = axes[0]
    xs = np.arange(len(horizons))
    ax.axhline(0, color=MUTED, lw=1.2, ls="--")
    ax.plot(xs, kurt, "o-", color=PURPLE, lw=2, ms=6)
    for x, k in zip(xs, kurt):
        ax.annotate(f"{k:.2f}", (x, k), textcoords="offset points",
                    xytext=(0, 9), fontsize=9.5, color=PURPLE, ha="center")
    ax.text(len(horizons) - 1, 0.30, "정규분포의 초과첨도 0", fontsize=9.5,
            color=INK, ha="right", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("초과첨도", fontsize=10.5, color=INK)
    ax.set_ylim(-0.6, max(kurt) * 1.25)
    ax.set_title("집계 기간이 길어지면 첨도가 줄지만", fontsize=11.5, color=INK)
    clean(ax)

    ax = axes[1]
    ax.bar(xs, tail, color=ORANGE_F, edgecolor=ORANGE, linewidth=1.2, width=0.62)
    ax.axhline(normal_tail, color=BLUE, lw=1.6, ls="--",
               label=f"정규분포가 예측하는 값  {normal_tail:.4f}")
    ax.legend(fontsize=9.5, frameon=False, loc="upper right", labelcolor=BLUE)
    for x, p in zip(xs, tail):
        ax.annotate(f"{p:.4f}", (x, p), textcoords="offset points",
                    xytext=(0, 4), fontsize=9.5, color=ORANGE, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("$3$ 표준편차를 넘는 비율", fontsize=10.5, color=INK)
    ax.set_ylim(0, max(tail) * 1.28)
    ax.set_title("3 표준편차 사건은 여전히 잦다", fontsize=11.5, color=INK)
    clean(ax)

    fig.suptitle("GARCH(1,1) + $t_7$ 로 만든 4000년치 수익률을 기간별로 더해 본 결과",
                 fontsize=12.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "aggregation_kurtosis.png")


if __name__ == "__main__":
    fig_clt_mean_vs_variance()
    fig_type1_vs_n()
    fig_aggregation_kurtosis()
