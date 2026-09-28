r"""3.5절 — 약한 큰수의 법칙과 중심극한정리를 한 그림에 담는다.

같은 모의실험을 두 배율로 본다.

  윗줄 (약한 큰수의 법칙)  표본평균 X̄_n 의 분포. n 이 커지면 μ 둘레로
                           오그라들고, 제목의 P(|X̄_n - μ| > ε) 가 0 으로 간다.
  아랫줄 (중심극한정리)    표준화한 Z_n = √n (X̄_n - μ)/σ 의 분포를 N(0,1) 과
                           견준다.

만드는 파일:

  ch03/limits/img/wlln_exponential.png      윗줄만 (lln.md 용)
  ch03/limits/img/lln_clt_uniform.png       두 줄 (clt.md 용)
  ch03/limits/img/lln_clt_exponential.png
  ch03/limits/img/lln_clt_lognormal.png
  ch03/limits/img/lln_clt_bernoulli.png
  ch03/limits/img/two_rulers_exponential.png      한 그림에 눈금 두 벌 (clt.md)
  ch03/limits/img/rate_trichotomy_exponential.png 배율 세 가지 (clt.md)

실행:  python3 scripts/make_ch03_lln_clt.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/limits/img/"
INK = "#37474F"

N_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
M = 10_000                    # 되풀이 횟수
EPS = 0.2                     # ε (σ 단위)
SEED = 2026

# 이름 -> (표본추출기, 평균 μ, 표준편차 σ, 라벨, 이산인가)
DISTRIBUTIONS = {
    "uniform": (
        lambda rng, s: rng.uniform(0, 1, s),
        0.5, np.sqrt(1 / 12), "Uniform(0,1)", False),
    "exponential": (
        lambda rng, s: rng.exponential(1.0, s),
        1.0, 1.0, "Exponential(1)", False),
    "bernoulli": (
        lambda rng, s: rng.binomial(1, 0.3, s).astype(float),
        0.3, np.sqrt(0.3 * 0.7), "Bernoulli(0.3)", True),
    "poisson": (
        lambda rng, s: rng.poisson(2.0, s).astype(float),
        2.0, np.sqrt(2.0), "Poisson(2)", True),
    "chi2": (
        lambda rng, s: rng.chisquare(1, s),
        1.0, np.sqrt(2.0), r"$\chi^2_1$", False),
    "gamma": (
        lambda rng, s: rng.gamma(0.5, 2.0, s),
        1.0, np.sqrt(0.5) * 2.0, "Gamma(0.5, 2)", False),
    "lognormal": (
        lambda rng, s: rng.lognormal(0, 0.75, s),
        np.exp(0.75 ** 2 / 2),
        np.sqrt((np.exp(0.75 ** 2) - 1) * np.exp(0.75 ** 2)),
        "LogNormal(0, 0.75)", False),
    "beta": (
        lambda rng, s: rng.beta(0.5, 0.5, s),
        0.5, np.sqrt(0.25 / 2), "Beta(0.5, 0.5)", False),
}


def draw(dist, rows=2):
    """rows=2 면 큰수의 법칙과 중심극한정리를 함께, rows=1 이면 윗줄만 그린다."""
    sampler, mu, sigma, label, discrete = DISTRIBUTIONS[dist]
    rng = np.random.default_rng(SEED)
    eps = EPS * sigma

    # 가장 큰 n 으로 한 번만 뽑고 앞쪽 n 개 열을 각 칸에 쓴다.
    # 표본이 중첩되므로 칸끼리 다른 것은 오직 n 뿐이다.
    X = sampler(rng, (M, max(N_LIST)))

    height = 7.5 if rows == 2 else 4.0
    fig, axes = plt.subplots(rows, 10, figsize=(22, height),
                             constrained_layout=True, squeeze=False)
    head = ("Weak LLN (top) and CLT (bottom)" if rows == 2
            else "Weak LLN: distribution of the sample mean")
    fig.suptitle(f"{head} for {label}:  "
                 f"$\\mu$ = {mu:.3f},  $\\sigma$ = {sigma:.3f},  "
                 f"Monte Carlo size = {M:,}", fontsize=15)

    # 윗줄의 가로 범위는 가장 넓은 경우(가장 작은 n)에 맞춘다.
    means_min_n = X[:, :min(N_LIST)].mean(axis=1)
    lo, hi = np.percentile(means_min_n, [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    lln_xlim = (lo - pad, hi + pad)
    lln_bins = np.linspace(*lln_xlim, 60)

    z_grid = np.linspace(-4, 4, 400)
    z_bins = np.linspace(-4, 4, 50)

    report = []
    for j, n in enumerate(N_LIST):
        xbar = X[:, :n].mean(axis=1)

        # 정수값 분포에서는 합 S_n 이 정수이므로 X̄_n 이 격자 k/n 위에만 있다.
        # 격자점마다 막대 하나를 두어야 밀도로 읽을 수 있다.
        if discrete:
            s = np.rint(X[:, :n].sum(axis=1))
            edges = (np.arange(s.min(), s.max() + 2) - 0.5) / n
            lln_bins = edges
            z_bins = np.sqrt(n) * (edges - mu) / sigma

        # ---- 윗줄: 약한 큰수의 법칙 ----
        ax = axes[0, j]
        ax.hist(xbar, bins=lln_bins, density=True,
                color="tab:blue", alpha=0.6, edgecolor="white", linewidth=0.3)
        ax.axvline(mu, color="red", lw=2, label=r"$\mu$")
        ax.axvspan(mu - eps, mu + eps, color="orange", alpha=0.18,
                   label=r"$\mu\pm\varepsilon$")
        p_out = float(np.mean(np.abs(xbar - mu) > eps))
        ax.set_title(f"n = {n}\n"
                     rf"$\hat P(|\bar X_n-\mu|>\varepsilon)$ = {p_out:.3f}",
                     fontsize=11)
        ax.set_xlim(lln_xlim)
        ax.set_xlabel(r"$\bar X_n$")
        if j == 0:
            ax.set_ylabel("Weak LLN\ndensity of $\\bar X_n$", fontsize=12)
            ax.legend(fontsize=9, loc="upper right")
        ax.set_ylim(bottom=0)

        # ---- 아랫줄: 중심극한정리 ----
        z = np.sqrt(n) * (xbar - mu) / sigma
        ks = float(stats.kstest(z, "norm").statistic)
        skew = float(stats.skew(z))
        report.append((n, p_out, ks, skew))

        if rows == 2:
            ax = axes[1, j]
            ax.hist(z, bins=z_bins, density=True,
                    color="tab:green", alpha=0.6, edgecolor="white",
                    linewidth=0.3)
            ax.plot(z_grid, stats.norm.pdf(z_grid), "k-", lw=2, label="N(0,1)")
            ax.set_title(f"n = {n}\nKS = {ks:.3f},  skew = {skew:.2f}",
                         fontsize=11)
            ax.set_xlim(-4, 4)
            ax.set_xlabel(r"$Z_n=\sqrt{n}(\bar X_n-\mu)/\sigma$")
            if j == 0:
                ax.set_ylabel("CLT\ndensity of $Z_n$", fontsize=12)
                ax.legend(fontsize=9, loc="upper right")

    if discrete:
        fig.text(0.5, -0.04,
                 "Note: discrete distribution — $\\bar X_n$ lives on the "
                 "lattice $k/n$; one histogram bar per lattice point.",
                 ha="center", fontsize=10, style="italic")

    name = f"lln_clt_{dist}.png" if rows == 2 else f"wlln_{dist}.png"
    fig.savefig(OUT + name, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}{name}")
    for n, p_out, ks, skew in report:
        print(f"    n={n:>3}  P(|Xbar-mu|>eps)={p_out:.4f}  "
              f"KS={ks:.4f}  skew={skew:+.3f}")


def two_rulers(dist="exponential", ns=(5, 50)):
    """같은 히스토그램에 눈금 두 벌을 달아 '한 동전의 양면'을 보인다.

    z 눈금으로 그리면 두 칸의 폭이 같다. 같은 그림을 x̄ 눈금으로 읽으면
    n 이 큰 쪽이 √n 배 좁다. 두 정리는 이 두 눈금에 각각 붙은 이름이다.
    """
    sampler, mu, sigma, label, _ = DISTRIBUTIONS[dist]
    rng = np.random.default_rng(SEED)
    X = sampler(rng, (M, max(ns)))

    fig, axes = plt.subplots(1, len(ns), figsize=(12.5, 4.6),
                             constrained_layout=True)
    z_bins = np.linspace(-4, 4, 46)
    z_grid = np.linspace(-4, 4, 400)

    for ax, n in zip(axes, ns):
        xbar = X[:, :n].mean(axis=1)
        z = np.sqrt(n) * (xbar - mu) / sigma
        half = 4 * sigma / np.sqrt(n)          # x̄ 눈금에서의 반폭

        ax.hist(z, bins=z_bins, density=True, color="tab:blue", alpha=0.55,
                edgecolor="white", linewidth=0.4)
        ax.plot(z_grid, stats.norm.pdf(z_grid), "k-", lw=2)
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.52)
        ax.set_xlabel(r"아래 눈금:  $Z_n=\sqrt{n}\,(\bar X_n-\mu)/\sigma$",
                      fontsize=11)
        ax.set_title(f"n = {n}", fontsize=13, pad=34)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        # 같은 그림에 x̄ 눈금을 하나 더 단다.
        top = ax.secondary_xaxis(
            "top", functions=(lambda t, n=n: mu + t * sigma / np.sqrt(n),
                              lambda v, n=n: (v - mu) * np.sqrt(n) / sigma))
        top.set_xlabel(rf"위 눈금:  $\bar X_n$   (폭 $\pm${half:.2f})",
                       fontsize=11, color="tab:red")
        top.tick_params(labelsize=10, colors="tab:red")

    axes[0].set_ylabel("같은 히스토그램", fontsize=12)
    fig.suptitle(f"한 히스토그램, 두 눈금 — {label}", fontsize=14)
    fig.text(0.5, -0.04,
             "두 칸의 히스토그램은 폭이 같다. 위의 붉은 눈금으로 읽으면 "
             f"{ns[1]} 쪽이 {np.sqrt(ns[1] / ns[0]):.1f}배 좁다.",
             ha="center", fontsize=11)
    fig.savefig(OUT + f"two_rulers_{dist}.png", dpi=170, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}two_rulers_{dist}.png")
    for n in ns:
        print(f"    n={n:>3}  x̄ 눈금 반폭 = {4 * sigma / np.sqrt(n):.3f}")


def rate_trichotomy(dist="exponential", ns=(5, 20, 100, 500)):
    """배율을 바꿔 가며 n^a (X̄-μ)/σ 를 그린다.

    a < 1/2 이면 0 으로 뭉개지고, a > 1/2 이면 퍼져 나간다. a = 1/2 에서만
    아무 데로도 가지 않고 모양이 남는다. 이것이 "수렴 속도가 1/√n" 의 뜻이다.
    """
    sampler, mu, sigma, label, _ = DISTRIBUTIONS[dist]
    rng = np.random.default_rng(SEED)
    X = sampler(rng, (M, max(ns)))

    exps = [(0.25, r"$n^{1/4}(\bar X_n-\mu)/\sigma$", "너무 약한 배율",
             "한 점으로 뭉개진다", "tab:orange"),
            (0.50, r"$n^{1/2}(\bar X_n-\mu)/\sigma$", "꼭 맞는 배율",
             "모양이 남는다", "tab:green"),
            (1.00, r"$n^{1}(\bar X_n-\mu)/\sigma$", "너무 센 배율",
             "그림 밖으로 나간다", "tab:purple")]

    # 표준편차가 줄마다 100배 넘게 차이 나므로 칸 폭도 줄마다 맞춰 준다.
    # 고정 폭을 쓰면 좁은 분포가 들쭉날쭉한 막대 몇 개로 뭉개진다.
    scaled = {(i, n): n ** a * (X[:, :n].mean(axis=1) - mu) / sigma
              for i, (a, *_ ) in enumerate(exps) for n in ns}

    fig, axes = plt.subplots(3, len(ns), figsize=(13.5, 7.6),
                             constrained_layout=True, sharex=True)
    grid = np.linspace(-4, 4, 400)

    for i, (a, formula, note, fate, color) in enumerate(exps):
        # 한 줄 안에서는 세로 눈금을 하나로 묶어야 폭의 변화가 읽힌다.
        peaks = []
        for n in ns:
            t = scaled[(i, n)]
            w = float(np.clip(t.std() / 10, 8 / 400, 8 / 18))
            h, _ = np.histogram(t, bins=np.arange(-4, 4 + w, w), density=True)
            peaks.append(h.max())
        row_top = 1.15 * max(peaks)

        for j, n in enumerate(ns):
            ax = axes[i, j]
            t = scaled[(i, n)]
            w = float(np.clip(t.std() / 10, 8 / 400, 8 / 18))
            ax.hist(t, bins=np.arange(-4, 4 + w, w), density=True,
                    color=color, alpha=0.65, edgecolor="none")
            if abs(a - 0.5) < 1e-9:
                ax.plot(grid, stats.norm.pdf(grid), "k-", lw=1.8)
            ax.set_xlim(-4, 4)
            ax.set_ylim(0, row_top)
            ax.set_yticks([])
            ax.tick_params(labelsize=9)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.text(0.03, 0.93, f"표준편차 {t.std():.2f}", transform=ax.transAxes,
                    fontsize=10, color=INK, va="top")
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=12.5)
        axes[i, 0].set_ylabel(f"{formula}\n{note} — {fate}", fontsize=11,
                              color=color, rotation=0, ha="right", va="center",
                              labelpad=12)

    fig.suptitle(f"$\\sqrt{{n}}$ 만이 꼭 맞는 배율이다 — {label}", fontsize=14.5)
    fig.text(0.5, -0.03,
             "네 칸 모두 가로 범위가 같다. 줄마다 세로 눈금은 그 줄에 맞추었으므로 "
             "읽어야 할 것은 높이가 아니라 폭이다.",
             ha="center", fontsize=11)
    fig.savefig(OUT + f"rate_trichotomy_{dist}.png", dpi=170,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}rate_trichotomy_{dist}.png")
    for i, (a, _, note, _, _) in enumerate(exps):
        sds = [float(scaled[(i, n)].std()) for n in ns]
        print(f"    a={a:.2f} ({note}): sd = "
              + ", ".join(f"{v:.2f}" for v in sds))


if __name__ == "__main__":
    draw("exponential", rows=1)          # lln.md 용
    for d in ("uniform", "exponential", "lognormal", "bernoulli"):
        draw(d, rows=2)                  # clt.md 용
    two_rulers("exponential")            # clt.md 정리 2 바로 뒤
    rate_trichotomy("exponential")       # clt.md 배율 이야기
