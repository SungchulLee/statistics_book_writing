r"""7.4 분산추정 네 쪽의 그림을 생성한다.

네 쪽 모두 그림이 없던 곳이다. 쪽마다 그 쪽의 핵심 주장 하나를 그렸다.

만드는 파일:
  ch07/variance/img/bessels_dof.png          편차 하나는 자유롭지 않다 — 자유도 n-1
  ch07/variance/img/naive_always_short.png   소박한 추정량은 언제나, 얼마나 모자라는가
  ch07/variance/img/mse_divisor_curve.png    나누는 수에 따른 MSE — 바닥은 n+1
  ch07/variance/img/robust_scale_breakdown.png  오염 비율과 붕괴점

실행:  python3 scripts/make_ch07_variance_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import numpy as np
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

OUT = "docs/ch07/variance/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def chi2_pdf(x, k):
    """카이제곱 밀도 — scipy 없이 쓰려고 로그감마를 직접 부른다."""
    from math import lgamma
    return np.exp((k / 2 - 1) * np.log(x) - x / 2
                  - lgamma(k / 2) - (k / 2) * np.log(2))


# ==================================================================
# 1. 자유도 — 편차 n 개 중 자유로운 것은 n-1 개다
# ==================================================================
def bessels_dof():
    rng = np.random.default_rng(12)
    n = 5
    x = rng.normal(10.0, 2.0, size=n)
    d = x - x.mean()
    print(f"  편차 {np.round(d, 3)}  합 {d.sum():.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # --- (a) 마지막 편차는 앞의 넷이 정한다 ---
    ax = axes[0]
    idx = np.arange(1, n + 1)
    colors = [BLUE] * (n - 1) + [RED]
    for i, (v, c) in enumerate(zip(d, colors), start=1):
        if i < n:
            ax.bar(i, v, 0.55, color=c, alpha=0.85, edgecolor="white",
                   zorder=5)
        else:
            ax.bar(i, v, 0.55, facecolor="white", edgecolor=c, linewidth=2.0,
                   hatch="///", zorder=5)
        ax.text(i, v + (0.12 if v >= 0 else -0.12), f"{v:+.2f}",
                fontsize=11, color=c, ha="center",
                va="bottom" if v >= 0 else "top")

    ax.axhline(0, color=INK, linewidth=1.3, zorder=6)
    ax.annotate("앞의 넷이 정해지면\n이 값은 선택의 여지가 없다",
                xy=(n, d[-1] * 0.55), xytext=(3.15, -2.55), fontsize=11,
                color=RED, ha="center", va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))
    ax.text(0.55, 2.55,
            f"편차 {n} 개의 합은 언제나 $0$ 이다\n"
            r"$\sum_i (x_i - \bar x) = 0$",
            fontsize=11.5, color=INK, ha="left", va="center",
            linespacing=1.7)

    ax.set_xlim(0.4, n + 0.75)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xticks(idx)
    ax.set_xlabel("관측값 번호", fontsize=12, color=INK)
    ax.set_ylabel(r"편차 $x_i - \bar x$", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title(f"관측값은 {n} 개, 자유로운 편차는 {n - 1} 개", fontsize=13,
                 pad=10)

    # --- (b) 그래서 Q/σ² 의 평균이 n-1 이다 ---
    ax = axes[1]
    k = n - 1
    q = np.linspace(0.02, 16, 800)
    ax.plot(q, chi2_pdf(q, k), color=BLUE, linewidth=2.4, zorder=5)
    ax.fill_between(q, chi2_pdf(q, k), color=BLUE, alpha=0.14, zorder=3)

    top = 0.20
    for val, c, name in [(k, GREEN, f"평균 $= n - 1 = {k}$"),
                         (n, MUTED, f"$n = {n}$")]:
        ax.plot([val, val], [0, top], color=c, linewidth=1.8,
                linestyle="--" if c is GREEN else ":", zorder=6)
    ax.text(k - 0.25, top + 0.004, f"평균 $= n - 1 = {k}$", fontsize=11.5,
            color=GREEN, ha="right", va="bottom")
    ax.text(n + 0.25, top + 0.004, f"$n = {n}$ 이 아니다", fontsize=11.5,
            color=MUTED, ha="left", va="bottom")

    ax.text(7.2, 0.135,
            r"$Q/\sigma^2 \sim \chi^2_{n-1}$" + "\n\n"
            r"$E[Q] = (n-1)\,\sigma^2$" + "\n"
            "이므로 $n$ 이 아니라 $n-1$ 로\n나누어야 평균이 제자리에 온다",
            fontsize=11.5, color=INK, ha="left", va="top", linespacing=1.6)

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 0.235)
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    bare_axis(ax)
    ax.set_xlabel(r"$Q/\sigma^2 = \sum_i (X_i - \bar X)^2 / \sigma^2$",
                  fontsize=12, color=INK)
    ax.set_title(f"편차제곱합의 분포  ($n = {n}$)", fontsize=13, pad=10)

    fig.suptitle("평균을 자료에서 가져다 쓰면 자유도 하나를 내놓아야 한다",
                 fontsize=14.5, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "bessels_dof.png")


# ==================================================================
# 2. 소박한 추정량 — 언제나 모자라고, n 이 작을수록 심하다
# ==================================================================
def naive_always_short():
    rng = np.random.default_rng(2)
    n, m = 5, 25
    x = rng.normal(0.0, 1.0, size=(m, n))
    naive = ((x - x.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    known_mu = (x ** 2).mean(axis=1)          # 참 평균 0 을 쓴 추정값
    order = np.argsort(known_mu)
    naive, known_mu = naive[order], known_mu[order]
    print(f"  보여 준 {m} 개 표본: 평균(μ 기준) {known_mu.mean():.4f}, "
          f"평균(x̄ 기준) {naive.mean():.4f}, "
          f"주황이 아래인 횟수 {np.sum(naive < known_mu)}/{m}")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # --- (a) 표본마다 두 값을 짝지어 본다 ---
    ax = axes[0]
    idx = np.arange(1, m + 1)
    for i, (a, b) in zip(idx, zip(known_mu, naive)):
        ax.plot([i, i], [b, a], color=MUTED, linewidth=1.2, zorder=3)
    ax.plot(idx, known_mu, "o", color=BLUE, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5,
            label=r"참 평균 기준  $\frac{1}{n}\sum (x_i - \mu)^2$")
    ax.plot(idx, naive, "o", color=ORANGE, markersize=7,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5,
            label=r"표본평균 기준  $\tilde S^2$")

    ax.axhline(1.0, color=RED, linewidth=1.5, zorder=4)
    ax.text(m + 0.4, 1.0, r"$\sigma^2 = 1$", fontsize=11.5, color=RED,
            ha="left", va="center")

    ax.text(0.6, 3.35,
            f"표본 {m} 개 모두에서 주황이 파랑보다 아래에 있다 — "
            "예외는 있을 수 없다",
            fontsize=11.5, color=INK, ha="left", va="center")

    ax.set_xlim(0, m + 2.6)
    ax.set_ylim(0, 3.65)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlabel("표본 번호", fontsize=12, color=INK)
    ax.set_ylabel("분산 추정값", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 0.93))
    ax.set_title(f"같은 표본을 두 방식으로 재 보면  ($n = {n}$, "
                 f"$\\sigma^2 = 1$)", fontsize=13, pad=10)

    # --- (b) 얼마나 모자라는가 ---
    ax = axes[1]
    nn = np.arange(2, 61)
    ax.plot(nn, (nn - 1) / nn, color=BLUE, linewidth=2.5, zorder=5)
    ax.axhline(1.0, color=MUTED, linewidth=1.3, linestyle=":", zorder=3)
    ax.text(60, 1.012, "편향이 없다면 여기", fontsize=11, color=MUTED,
            ha="right", va="bottom")

    for k, c in [(2, RED), (5, ORANGE), (10, GREEN), (30, PURPLE)]:
        v = (k - 1) / k
        ax.plot([k], [v], "o", color=c, markersize=8.5, zorder=7)
        ax.text(k + 1.6, v - 0.012, f"$n = {k}$ :  {v:.0%}", fontsize=11,
                color=c, ha="left", va="top")
        print(f"    n={k:3d}  E[naive]/σ² = {v:.4f}")

    ax.set_xlim(0, 62)
    ax.set_ylim(0.42, 1.06)
    ax.set_xlabel("표본크기 $n$", fontsize=12, color=INK)
    ax.set_ylabel(r"$E[\tilde S^2] \,/\, \sigma^2 = (n-1)/n$", fontsize=12,
                  color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("모자라는 정도는 $n$ 이 정한다", fontsize=13, pad=10)

    fig.suptitle("표본평균에서 재면 매번 모자란다 — 평균적으로가 아니라 매번",
                 fontsize=14.5, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "naive_always_short.png")


# ==================================================================
# 3. 나누는 수에 따른 MSE — 바닥은 n-1 도 n 도 아닌 n+1
# ==================================================================
def mse_divisor_curve():
    n = 10
    d = np.linspace(5.5, 18.0, 600)
    var = 2 * (n - 1) / d ** 2
    bias2 = ((n - 1) / d - 1) ** 2
    mse = var + bias2

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- (a) n = 10 에서 분해해 보기 ---
    ax = axes[0]
    ax.plot(d, var, color=GREEN, linewidth=2.0,
            label=r"분산  $2(n-1)/d^2$")
    ax.plot(d, bias2, color=ORANGE, linewidth=2.0,
            label=r"편향제곱  $\left[(n-1)/d - 1\right]^2$")
    ax.plot(d, mse, color=BLUE, linewidth=2.8, label="MSE  (둘의 합)")

    pts = [(n - 1, GREEN, "Bessel"), (n, PURPLE, "MLE"),
           (n + 1, RED, "MSE 최적")]
    for dd, c, name in pts:
        v = 2 * (n - 1) / dd ** 2 + ((n - 1) / dd - 1) ** 2
        ax.plot([dd], [v], "o", color=c, markersize=9, zorder=8)
        print(f"  n={n}  d={dd}: MSE={v:.4f}")

    ax.annotate("Bessel  $d = n-1 = 9$\nMSE $= 0.222$",
                xy=(9, 0.2222), xytext=(7.9, 0.375), fontsize=11,
                color=GREEN, ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.2))
    ax.annotate("MLE  $d = n = 10$\nMSE $= 0.190$",
                xy=(10, 0.1900), xytext=(12.3, 0.300), fontsize=11,
                color=PURPLE, ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=1.2))
    ax.annotate("최적  $d = n+1 = 11$\nMSE $= 0.182$",
                xy=(11, 0.1818), xytext=(12.9, 0.108), fontsize=11,
                color=RED, ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2))

    ax.set_xlim(5.5, 18.0)
    ax.set_ylim(0, 0.42)
    ax.set_xticks([6, 8, 9, 10, 11, 12, 14, 16, 18])
    ax.set_xlabel("나누는 수 $d$", fontsize=12, color=INK)
    ax.set_ylabel(r"$\sigma^4$ 을 $1$ 로 둔 값", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.set_title(f"$n = {n}$ 에서 $d$ 를 바꿔 가며 본 MSE", fontsize=13,
                 pad=10)

    # --- (b) 차이가 문제되는 것은 작은 표본에서뿐 ---
    ax = axes[1]
    nn = np.arange(3, 61)
    best = 2 / (nn + 1)
    r_bessel = (2 / (nn - 1)) / best
    r_mle = ((2 * nn - 1) / nn ** 2) / best
    ax.plot(nn, r_bessel, color=GREEN, linewidth=2.4,
            label="Bessel  ($n-1$ 로 나눔)")
    ax.plot(nn, r_mle, color=PURPLE, linewidth=2.4,
            label="MLE  ($n$ 으로 나눔)")
    ax.axhline(1.0, color=RED, linewidth=1.6, zorder=4)
    # 기준선 라벨을 선 바로 위에 두면 초록 곡선이 글자를 가로지른다.
    ax.annotate("기준선 — 최적 추정량 ($n+1$)", xy=(50, 1.0),
                xytext=(36, 1.20), fontsize=11, color=RED, ha="center",
                va="center",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2))

    for k, c in [(5, GREEN), (10, GREEN), (30, GREEN)]:
        v = (2 / (k - 1)) / (2 / (k + 1))
        ax.plot([k], [v], "o", color=c, markersize=8, zorder=7)
        print(f"    n={k:3d}  Bessel/최적 = {v:.3f}  "
              f"MLE/최적 = {((2 * k - 1) / k ** 2) / (2 / (k + 1)):.3f}")
    ax.text(5.9, 1.503, "$n = 5$ 에서 $1.50$ 배", fontsize=11, color=GREEN,
            ha="left", va="center")
    ax.text(11.2, 1.222, "$n = 10$ 에서 $1.22$ 배", fontsize=11, color=GREEN,
            ha="left", va="center")
    ax.text(31.5, 1.112, "$n = 30$ 에서 $1.07$ 배", fontsize=11, color=GREEN,
            ha="left", va="center")

    ax.set_xlim(2.5, 61)
    ax.set_ylim(0.97, 1.75)
    ax.set_xlabel("표본크기 $n$", fontsize=12, color=INK)
    ax.set_ylabel("최적 추정량 대비 MSE 배수", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.set_title("분모의 선택이 실제로 중요한 구간", fontsize=13, pad=10)

    fig.suptitle("불편성은 공짜가 아니다 — 나누는 수를 바꿔 가며 본 값",
                 fontsize=14.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "mse_divisor_curve.png")


# ==================================================================
# 4. 붕괴점 — 오염 비율을 올려 가며 본 세 척도추정량
# ==================================================================
def robust_scale_breakdown():
    rng = np.random.default_rng(99)
    n = 200
    base = rng.standard_normal(n)
    outlier = 50.0                      # 오염값은 한 자리에 크게 둔다

    fracs = np.arange(0, 0.56, 0.005)
    out = {"s": [], "mad": [], "iqr": []}
    for f in fracs:
        k = int(round(f * n))
        x = base.copy()
        x[:k] = outlier
        med = np.median(x)
        out["s"].append(x.std(ddof=1))
        out["mad"].append(1.4826 * np.median(np.abs(x - med)))
        q1, q3 = np.percentile(x, [25, 75])
        out["iqr"].append((q3 - q1) / 1.349)
    for key in out:
        out[key] = np.array(out[key])

    for f in (0.02, 0.10, 0.24, 0.30, 0.49, 0.52):
        j = int(np.argmin(np.abs(fracs - f)))
        print(f"  오염 {fracs[j]:.0%}:  s={out['s'][j]:8.2f}  "
              f"MAD={out['mad'][j]:6.2f}  IQR={out['iqr'][j]:8.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # --- (a) 오염 비율을 올려 가며 ---
    ax = axes[0]
    top = 8.0
    for key, c, name in [("s", RED, "표본표준편차 $s$"),
                         ("iqr", ORANGE, r"IQR 기반  $\hat\sigma_{\mathrm{IQR}}$"),
                         ("mad", BLUE, r"MAD 기반  $\hat\sigma_{\mathrm{MAD}}$")]:
        y = np.clip(out[key], 0, top * 1.4)
        ax.plot(fracs * 100, y, color=c, linewidth=2.5, zorder=5, label=name)

    for xb, c, tag in [(25, ORANGE, "붕괴점 25%"), (50, BLUE, "붕괴점 50%")]:
        ax.plot([xb, xb], [0, top], color=c, linewidth=1.3, linestyle="--",
                zorder=4)
        ax.text(xb - 0.8, top * 0.985, tag, fontsize=11, color=c,
                ha="right", va="top")

    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=":", zorder=3)
    ax.text(34, 1.10, "참값 $\\sigma = 1$", fontsize=11, color=MUTED,
            ha="left", va="bottom")
    # MAD 는 50% 를 넘으면 폭발이 아니라 0 으로 주저앉는다 — 짚어 준다.
    ax.annotate("오염값이 과반이 되면\nMAD 는 $0$ 으로 주저앉는다",
                xy=(51.6, 0.12), xytext=(28.5, 0.48), fontsize=11,
                color=BLUE, ha="left", va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.3))
    ax.annotate("오염 2% 만으로 이미\n화면 밖으로 나간다",
                xy=(2.6, 7.6), xytext=(9.5, 6.25), fontsize=11, color=RED,
                ha="left", va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))

    ax.set_xlim(0, 56)
    ax.set_ylim(0, top)
    ax.set_xticks([0, 10, 20, 25, 30, 40, 50])
    ax.set_xlabel("오염된 관측값의 비율 (%)", fontsize=12, color=INK)
    ax.set_ylabel(r"$\sigma$ 의 추정값", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper left",
              bbox_to_anchor=(0.30, 0.62))
    ax.set_title(f"깨끗한 관측값 {n} 개 중 일부를 $50$ 으로 바꿔 가며",
                 fontsize=13, pad=10)

    # --- (b) 깨끗할 때 치르는 값 ---
    ax = axes[1]
    m, reps = 50, 60_000
    z = rng.standard_normal((reps, m))
    med = np.median(z, axis=1)
    est = {
        "s": z.std(axis=1, ddof=1),
        "mad": 1.4826 * np.median(np.abs(z - med[:, None]), axis=1),
    }
    q1, q3 = np.percentile(z, [25, 75], axis=1)
    est["iqr"] = (q3 - q1) / 1.349

    grid = np.linspace(0.45, 1.65, 700)
    for key, c, name in [("s", RED, "표본표준편차 $s$"),
                         ("iqr", ORANGE, "IQR 기반"),
                         ("mad", BLUE, "MAD 기반")]:
        v = est[key]
        d = (grid[:, None] - v[None, :40000]) / 0.028
        y = np.exp(-d ** 2 / 2).sum(axis=1) / (40000 * 0.028
                                               * np.sqrt(2 * np.pi))
        are = est["s"].var() / v.var()
        ax.plot(grid, y, color=c, linewidth=2.3, zorder=5,
                label=f"{name}   표준편차 {v.std():.3f}")
        ax.fill_between(grid, y, color=c, alpha=0.10, zorder=3)
        print(f"  깨끗한 자료 n={m}: {key:4s} 평균={v.mean():.4f}  "
              f"SD={v.std():.4f}  ARE={are:.3f}")

    ax.plot([1, 1], [0, 4.0], color=MUTED, linewidth=1.3, linestyle=":",
            zorder=4)
    ax.text(1.02, 4.05, "참값 $\\sigma = 1$", fontsize=11, color=MUTED,
            ha="left", va="bottom")
    ax.text(1.20, 3.15,
            "정규에서의 ARE\nMAD 기반 $0.39$\nIQR 기반 $0.40$",
            fontsize=11, color=INK, ha="left", va="top", linespacing=1.7)

    ax.set_xlim(0.45, 1.65)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel(f"추정값  (깨끗한 $N(0,1)$, $n = {m}$)", fontsize=12,
                  color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("오염이 없을 때의 표본분포", fontsize=13, pad=10)

    fig.suptitle("붕괴점은 버티는 한계이고, ARE 는 그 대가다", fontsize=14.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, OUT + "robust_scale_breakdown.png")


if __name__ == "__main__":
    bessels_dof()
    naive_always_short()
    mse_divisor_curve()
    robust_scale_breakdown()
