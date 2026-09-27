r"""6.2 추정 방법 절 일곱 쪽의 그림을 생성한다.

일곱 쪽 모두 그림이 없던 곳이다.

  ch06/estimation_methods/img/moments_procedure.png       적률 맞추기 절차
  ch06/estimation_methods/img/sample_moments_converge.png 왜 통하는가 — 대수의 법칙
  ch06/estimation_methods/img/mom_three_families.png      세 분포에 적용
  ch06/estimation_methods/img/mom_vs_mle_efficiency.png   두 방법의 표본분포와 ARE
  ch06/estimation_methods/img/gmm_overidentification.png  적률조건이 남을 때
  ch06/estimation_methods/img/loglik_additivity.png       로그가능도는 더해진다
  ch06/estimation_methods/img/variance_estimator_mse.png  분산추정량 셋

감마 형상모수의 ARE 는 이론값을 직접 계산해 모의실험과 맞추어 두었다.
  nVar(MoM) = 4,  nVar(MLE) = 1/(psi'(1) - 1) = 1.5505,  ARE = 0.388

실행:  python3 scripts/make_ch06_methods_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats
from scipy.special import digamma, polygamma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

OUT = "docs/ch06/estimation_methods/img/"


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


def gamma_shape_mle(x):
    """감마 형상모수의 MLE. log a - psi(a) = log(mean) - mean(log) 를 푼다."""
    s = np.log(x.mean(-1)) - np.log(x).mean(-1)
    a = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(10):
        a = a - (np.log(a) - digamma(a) - s) / (1 / a - polygamma(1, a))
    return a


# ==================================================================
# 1. 적률 맞추기 절차
# ==================================================================
def moments_procedure():
    fig, ax = plt.subplots(figsize=(13, 4.4))

    def box(xy, title, sub, edge, face, w=3.1, h=1.5):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.14",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.8, zorder=4))
        ax.text(x, y + 0.3, title, fontsize=12, color=edge, ha="center",
                va="center", zorder=5)
        ax.text(x, y - 0.32, sub, fontsize=10.5, color=edge, ha="center",
                va="center", zorder=5, linespacing=1.4)

    box((1.9, 3.2), "자료", "$x_1, \\ldots, x_n$", INK, "#ECEFF1")
    box((6.0, 3.2), "표본 적률", "$m_1 = \\bar x$,\n$m_2 = \\frac{1}{n}\\sum x_i^2$",
        GREEN, GREEN_F, h=1.7)
    box((10.3, 3.2), "모집단 적률", "$\\mu_1(\\theta)$,  $\\mu_2(\\theta)$",
        ORANGE, ORANGE_F)
    box((14.4, 3.2), "추정값", "$\\hat\\theta$", BLUE, BLUE_F, w=2.4)

    for x0, x1 in [(3.5, 4.4), (11.9, 13.1)]:
        ax.add_patch(FancyArrowPatch((x0, 3.2), (x1, 3.2), arrowstyle="-|>",
                                     mutation_scale=16, color=MUTED,
                                     linewidth=1.6, zorder=3))
    # 가운데는 "같다고 놓는다"
    ax.add_patch(FancyArrowPatch((7.6, 3.2), (8.7, 3.2), arrowstyle="<|-|>",
                                 mutation_scale=16, color=RED, linewidth=1.8,
                                 zorder=3))
    ax.text(8.15, 4.15, "같다고 놓는다", fontsize=12, color=RED, ha="center",
            va="bottom")
    ax.text(8.15, 2.15, "$m_k = \\mu_k(\\theta)$", fontsize=12.5, color=RED,
            ha="center", va="top")
    ax.text(12.5, 3.75, "연립방정식을\n푼다", fontsize=10.5, color=MUTED,
            ha="center", va="bottom", linespacing=1.4)

    ax.text(8.15, 0.95,
            "감마분포의 예 —  $\\mu_1 = \\alpha/\\beta$,  "
            "$\\mu_2 - \\mu_1^2 = \\alpha/\\beta^2$ 이므로  "
            "$\\hat\\alpha = \\bar x^2 / s^2$,  $\\hat\\beta = \\bar x / s^2$",
            fontsize=11.5, color=INK, ha="center", va="center")

    ax.set_xlim(0, 15.9)
    ax.set_ylim(0.55, 4.75)
    ax.axis("off")
    ax.set_title("적률법 — 모르는 것을 아는 것에 맞춘다", fontsize=13.5, pad=10)
    fig.tight_layout()
    save(fig, OUT + "moments_procedure.png")


# ==================================================================
# 2. 왜 통하는가 — 표본 적률이 모집단 적률로 간다
# ==================================================================
def sample_moments_converge():
    rng = np.random.default_rng(4)
    alpha, beta = 2.0, 1.0            # Gamma(2, 1)
    mu1, mu2 = alpha / beta, alpha * (alpha + 1) / beta ** 2
    N, paths = 2000, 6
    x = rng.gamma(alpha, 1 / beta, size=(paths, N))
    ns = np.arange(1, N + 1)
    m1 = np.cumsum(x, axis=1) / ns
    m2 = np.cumsum(x ** 2, axis=1) / ns

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharex=True)
    for ax, m, true, name, color in [
            (axes[0], m1, mu1, "$m_1 = \\bar x$", BLUE),
            (axes[1], m2, mu2, "$m_2 = \\frac{1}{n}\\sum x_i^2$", GREEN)]:
        for row in m:
            ax.plot(ns, row, color=color, linewidth=1.0, alpha=0.5)
        ax.axhline(true, color=RED, linewidth=1.8, zorder=6)
        ax.text(N * 0.99, true * 1.035, f"모집단 적률 ${true:g}$",
                fontsize=11, color=RED, ha="right", va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=11.5, color=INK)
        ax.set_ylabel(name, fontsize=12, color=color)
        ax.set_ylim(true * 0.35, true * 1.75)
        clean_axis(ax)

    fig.suptitle("적률법이 통하는 이유 — 표본 적률이 모집단 적률로 수렴한다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "sample_moments_converge.png")


# ==================================================================
# 3. 세 분포에 같은 조리법을
# ==================================================================
def mom_three_families():
    rng = np.random.default_rng(11)
    n = 400

    cases = []
    # 정규 — 평균과 분산을 그대로
    z = rng.normal(8.0, 2.0, n)
    cases.append(("정규분포", z, stats.norm(z.mean(), z.std(ddof=0)),
                  "$\\hat\\mu = \\bar x$,   $\\hat\\sigma^2 = s^2$", BLUE))
    # 감마 — 두 식을 풀어야 한다
    g = rng.gamma(3.0, 1 / 1.5, n)
    a_h = g.mean() ** 2 / g.var(ddof=0)
    b_h = g.mean() / g.var(ddof=0)
    cases.append(("감마분포", g, stats.gamma(a_h, scale=1 / b_h),
                  f"$\\hat\\alpha = {a_h:.2f}$,   $\\hat\\beta = {b_h:.2f}$",
                  GREEN))
    # 베타 — 모수가 둘 다 비선형으로 얽힌다
    b = rng.beta(2.0, 5.0, n)
    mb, vb = b.mean(), b.var(ddof=0)
    c = mb * (1 - mb) / vb - 1
    a_b, b_b = mb * c, (1 - mb) * c
    cases.append(("베타분포", b, stats.beta(a_b, b_b),
                  f"$\\hat\\alpha = {a_b:.2f}$,   $\\hat\\beta = {b_b:.2f}$",
                  ORANGE))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, (name, data, fit, label, color) in zip(axes, cases):
        ax.hist(data, bins=26, density=True, color="#ECEFF1",
                edgecolor=MUTED, linewidth=0.6)
        xs = np.linspace(data.min(), data.max(), 400)
        ax.plot(xs, fit.pdf(xs), color=color, linewidth=2.4)
        ax.set_title(name, fontsize=12.5, color=color, pad=8)
        ax.text(0.5, -0.24, label, transform=ax.transAxes, fontsize=11,
                color=INK, ha="center", va="center")
        bare_axis(ax)

    fig.suptitle(f"같은 조리법, 다른 대수 — 표본 {n}개에 적률법으로 맞춘 곡선",
                 fontsize=13.5, y=1.03)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, OUT + "mom_three_families.png")


# ==================================================================
# 4. 적률법과 MLE의 효율 비교
# ==================================================================
def mom_vs_mle_efficiency():
    rng = np.random.default_rng(1)
    B, n = 40000, 50
    x = rng.gamma(1.0, 1.0, size=(B, n))        # Gamma(alpha = 1)
    mom = x.mean(1) ** 2 / x.var(1, ddof=1)
    mle = gamma_shape_mle(x)

    # 이론값
    v_mle = 1 / (polygamma(1, 1.0) - 1)          # 1.5505
    v_mom = 4.0
    are = v_mle / v_mom                          # 0.3876

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) 표본분포
    ax = axes[0]
    bins = np.linspace(0.3, 2.3, 90)
    ax.hist(mom, bins=bins, density=True, color=ORANGE, alpha=0.35,
            label=f"적률법   IQR 폭 {np.subtract(*np.percentile(mom, [75, 25])):.3f}")
    ax.hist(mle, bins=bins, density=True, color=BLUE, alpha=0.45,
            label=f"MLE       IQR 폭 {np.subtract(*np.percentile(mle, [75, 25])):.3f}")
    ax.axvline(1.0, color=RED, linewidth=1.8, zorder=6)
    ax.text(1.0, ax.get_ylim()[1] * 0.97, "  참값 $\\alpha = 1$", fontsize=11,
            color=RED, ha="left", va="top")
    ax.set_xlim(0.3, 2.3)
    ax.set_xlabel("형상모수 추정값 $\\hat\\alpha$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title(f"$\\mathrm{{Gamma}}(1, 1)$ 에서 $n = {n}$, "
                 f"{B // 10000}만 번 되풀이", fontsize=12.5, pad=10)

    # (b) n 을 키우며 본 분산 비
    ax = axes[1]
    ns = [25, 50, 100, 200, 400, 800]
    rm, rl = [], []
    for m in ns:
        xx = rng.gamma(1.0, 1.0, size=(30000, m))
        rm.append(m * (xx.mean(1) ** 2 / xx.var(1, ddof=1)).var())
        rl.append(m * gamma_shape_mle(xx).var())
    ax.plot(ns, rm, "o-", color=ORANGE, linewidth=2.0, markersize=6,
            label="적률법  $n\\,\\mathrm{Var}$")
    ax.plot(ns, rl, "o-", color=BLUE, linewidth=2.0, markersize=6,
            label="MLE  $n\\,\\mathrm{Var}$")
    ax.axhline(v_mom, color=ORANGE, linewidth=1.3, linestyle="--")
    ax.axhline(v_mle, color=BLUE, linewidth=1.3, linestyle="--")
    ax.text(ns[-1], v_mom + 0.12, f"이론값 ${v_mom:g}$", fontsize=10.5,
            color=ORANGE, ha="right", va="bottom")
    ax.text(ns[-1], v_mle + 0.12, f"이론값 ${v_mle:.3f}$", fontsize=10.5,
            color=BLUE, ha="right", va="bottom")
    ax.text(ns[0] * 1.35, 2.95,
            f"$\\mathrm{{ARE}} = {v_mle:.3f} / {v_mom:g} = {are:.2f}$\n"
            f"같은 정밀도를 얻으려면\n적률법은 관측값이 "
            f"${1 / are:.1f}$ 배 필요하다",
            fontsize=11, color=INK, ha="left", va="center", linespacing=1.5)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(m) for m in ns])
    ax.set_ylim(0, 5.2)
    ax.set_xlabel("표본크기 $n$ (로그 눈금)", fontsize=11.5, color=INK)
    ax.set_ylabel("$n \\times$ 분산", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("$n$ 을 키우면 이론값으로 다가간다", fontsize=12.5, pad=10)

    fig.suptitle("감마 형상모수 — 적률법은 MLE 만큼 정밀하지 못하다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "mom_vs_mle_efficiency.png")


# ==================================================================
# 5. 과대식별 — 적률조건이 모수보다 많을 때
# ==================================================================
def gmm_overidentification():
    # n 이 크면 두 근이 거의 겹쳐 그림에서 구별되지 않는다. 과대식별의 요점은
    # "유한표본에서 두 조건이 서로 다른 답을 가리킨다"이므로 작은 표본을 쓴다.
    rng = np.random.default_rng(40)
    n = 25
    x = rng.exponential(1.0, n)           # 평균 theta = 1 인 지수분포
    m1, m2 = x.mean(), (x ** 2).mean()

    th = np.linspace(0.75, 1.45, 600)
    g1 = m1 - th                          # E[X] = theta
    g2 = m2 - 2 * th ** 2                 # E[X^2] = 2 theta^2
    r1 = m1
    r2 = np.sqrt(m2 / 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) 두 조건이 한 점에서 만나지 않는다
    ax = axes[0]
    ax.axhline(0, color=MUTED, linewidth=1.2)
    ax.plot(th, g1, color=BLUE, linewidth=2.3, label="$g_1(\\theta) = m_1 - \\theta$")
    ax.plot(th, g2, color=ORANGE, linewidth=2.3,
            label="$g_2(\\theta) = m_2 - 2\\theta^2$")
    for r, c in [(r1, BLUE), (r2, ORANGE)]:
        ax.plot([r], [0], "o", color=c, markersize=8, zorder=6)
        ax.plot([r, r], [0, -0.42], color=c, linewidth=1.3, linestyle="--")
    ax.text(r1 - 0.012, -0.47, f"$g_1 = 0$\n${r1:.3f}$", fontsize=10.5,
            color=BLUE, ha="right", va="top", linespacing=1.4)
    ax.text(r2 + 0.012, -0.47, f"$g_2 = 0$\n${r2:.3f}$", fontsize=10.5,
            color=ORANGE, ha="left", va="top", linespacing=1.4)
    ax.annotate("", xy=(r1, 0.30), xytext=(r2, 0.30),
                arrowprops=dict(arrowstyle="<|-|>", color=RED, linewidth=1.5))
    ax.text((r1 + r2) / 2, 0.36, "두 근이 다르다", fontsize=11, color=RED,
            ha="center", va="bottom")
    ax.set_xlim(0.75, 1.45)
    ax.set_ylim(-1.0, 0.72)
    ax.set_xlabel("모수 $\\theta$", fontsize=11.5, color=INK, labelpad=16)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("적률조건 둘, 모수 하나 — 둘을 동시에 0 으로 만들 수 없다",
                 fontsize=12.5, pad=10)

    # (b) 그래서 가중해 더한 것을 최소화한다
    ax = axes[1]
    G = np.column_stack([g1, g2])
    h = np.column_stack([x - m1, x ** 2 - m2])
    S = h.T @ h / n
    corr = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])

    def quad(W):
        return np.einsum("ij,ki,kj->k", W, G, G)

    q_eye = quad(np.eye(2))
    q_diag = quad(np.diag(1 / np.diag(S)))
    q_opt = quad(np.linalg.inv(S))
    th_opt = th[int(np.argmin(q_opt))]

    for q, c, name in [(q_eye, MUTED, "가중 없음  $W = I$"),
                       (q_diag, PURPLE, "분산으로 표준화")]:
        ax.plot(th, q / q.min(), color=c, linewidth=2.4, label=name)
        i = int(np.argmin(q))
        ax.plot([th[i]], [1.0], "o", color=c, markersize=8, zorder=6)
        # 값은 표시점 위에 적는다. 축 아래에 두면 눈금 숫자와 겹친다.
        ax.text(th[i], 1.7, f"${th[i]:.3f}$", fontsize=10.5, color=c,
                ha="center", va="bottom")
    for r, c in [(r1, BLUE), (r2, ORANGE)]:
        ax.axvline(r, color=c, linewidth=1.1, linestyle=":", alpha=0.9)

    ax.text(0.78, 19.5,
            "$W = I$ 는 눈금이 큰 $g_2$ 에 끌려가고,\n"
            "분산으로 나누면 두 조건이 대등해진다",
            fontsize=10.5, color=INK, ha="left", va="top", linespacing=1.5)
    ax.text(0.78, 12.0,
            f"최적 가중 $W = S^{{-1}}$ 의 답은 ${th_opt:.3f}$ 이다.\n"
            f"두 조건의 상관이 ${corr:.2f}$ 로 높아\n두 근 바깥으로도 갈 수 있다.",
            fontsize=10, color=RED, ha="left", va="top", linespacing=1.5)

    ax.set_xlim(0.75, 1.45)
    ax.set_ylim(0, 22)
    ax.set_xlabel("모수 $\\theta$", fontsize=11.5, color=INK, labelpad=16)
    ax.set_ylabel("$Q(\\theta)$ (최솟값 $= 1$ 로 맞춤)", fontsize=11, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("가중을 어떻게 주느냐가 답을 바꾼다", fontsize=12.5, pad=10)

    fig.suptitle(f"지수분포 표본 $n = {n}$ — $E[X] = \\theta$ 와 "
                 "$E[X^2] = 2\\theta^2$ 를 함께 쓸 때", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "gmm_overidentification.png")


# ==================================================================
# 6. 로그가능도는 관측값마다 더해진다
# ==================================================================
def loglik_additivity():
    rng = np.random.default_rng(5)
    mu_true = 3.0
    data5 = rng.normal(mu_true, 1.0, 5)
    data50 = np.concatenate([data5, rng.normal(mu_true, 1.0, 45)])
    th = np.linspace(0.2, 5.8, 600)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) 관측값 하나하나의 기여와 그 합
    ax = axes[0]
    total = np.zeros_like(th)
    for xi in data5:
        c = -0.5 * np.log(2 * np.pi) - (xi - th) ** 2 / 2
        total += c
        ax.plot(th, c, color=MUTED, linewidth=1.3, alpha=0.9)
        ax.plot([xi], [-0.5 * np.log(2 * np.pi)], "|", color=INK,
                markersize=11, markeredgewidth=1.6)
    ax.plot(th, total, color=BLUE, linewidth=2.8, zorder=5)
    i = int(np.argmax(total))
    ax.plot([th[i]], [total[i]], "o", color=RED, markersize=8, zorder=6)
    ax.text(th[i], total[i] + 0.7, f"$\\hat\\mu = \\bar x = {data5.mean():.2f}$",
            fontsize=11.5, color=RED, ha="center", va="bottom")
    ax.text(5.6, -16.8, "가는 곡선 — 관측값 하나의 기여\n굵은 곡선 — 그 합",
            fontsize=10.5, color=INK, ha="right", va="bottom",
            linespacing=1.5)
    ax.set_xlim(0.2, 5.8)
    ax.set_ylim(-18, 2.2)
    ax.set_xlabel("모수 $\\mu$", fontsize=11.5, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("$n = 5$ — 로그가능도는 기여의 합이다", fontsize=12.5, pad=10)

    # (b) 관측값이 늘면 봉우리가 뾰족해진다
    ax = axes[1]
    for data, color, name in [(data5, "#90CAF9", "$n = 5$"),
                              (data50, BLUE, "$n = 50$")]:
        ll = np.array([np.sum(-0.5 * np.log(2 * np.pi) - (data - t) ** 2 / 2)
                       for t in th])
        ax.plot(th, ll - ll.max(), color=color, linewidth=2.6, label=name)
    ax.axhline(-2, color=MUTED, linewidth=1.1, linestyle=":")
    ax.text(5.72, -2, "꼭대기에서 $2$ 내려온 높이", fontsize=10, color=MUTED,
            ha="right", va="bottom")
    ax.set_xlim(0.2, 5.8)
    ax.set_ylim(-22, 3)
    ax.set_xlabel("모수 $\\mu$", fontsize=11.5, color=INK)
    ax.set_ylabel("$\\ell(\\mu) - \\ell(\\hat\\mu)$", fontsize=11.5, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.set_title("관측값이 쌓이면 봉우리가 좁아진다", fontsize=12.5, pad=10)

    fig.suptitle("곱이 합이 되므로 관측값을 더하고 빼기 쉽다", fontsize=13.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, OUT + "loglik_additivity.png")


# ==================================================================
# 7. 분산추정량 셋 — 불편이 최선은 아니다
# ==================================================================
def variance_estimator_mse():
    sigma2 = 4.0
    s4 = sigma2 ** 2
    n = 10
    names = ["MLE\n$(n)$", "Bessel\n$(n-1)$", "MSE 최적\n$(n+1)$"]
    bias = [-sigma2 / n, 0.0, -2 * sigma2 / (n + 1)]
    mse = [(2 * n - 1) / n ** 2 * s4, 2 / (n - 1) * s4, 2 / (n + 1) * s4]
    var = [m - b ** 2 for m, b in zip(mse, bias)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) n = 10 에서 MSE 를 두 조각으로
    ax = axes[0]
    xs = np.arange(3)
    ax.bar(xs, var, width=0.55, color=BLUE_F, edgecolor=BLUE, linewidth=1.6,
           label="분산")
    ax.bar(xs, [b ** 2 for b in bias], width=0.55, bottom=var,
           color=ORANGE_F, edgecolor=ORANGE, linewidth=1.6, label="편향제곱")
    for i, (m, b) in enumerate(zip(mse, bias)):
        ax.text(i, m + 0.12, f"MSE $= {m:.2f}$", fontsize=11, color=INK,
                ha="center", va="bottom")
        ax.text(i, 0.16, f"편향 ${b:+.2f}$", fontsize=10, color=INK,
                ha="center", va="bottom")
    ax.plot([1], [mse[1]], "o", color=RED, markersize=9, zorder=6)
    ax.annotate("불편인데도 MSE 가 가장 크다", xy=(1, mse[1]),
                xytext=(1.05, mse[1] + 1.25), fontsize=11, color=RED,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 4.9)
    ax.set_ylabel("$\\sigma^2 = 4$, $n = 10$ 에서의 값", fontsize=11,
                  color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("MSE = 분산 + 편향제곱", fontsize=12.5, pad=10)

    # (b) n 을 바꿔 가며
    ax = axes[1]
    nn = np.arange(3, 41)
    curves = [((2 * nn - 1) / nn ** 2 * s4, BLUE, "MLE $(n)$"),
              (2 / (nn - 1) * s4, GREEN, "Bessel $(n-1)$"),
              (2 / (nn + 1) * s4, ORANGE, "MSE 최적 $(n+1)$")]
    for y, c, name in curves:
        ax.plot(nn, y, color=c, linewidth=2.3, label=name)
    ax.set_xlim(3, 40)
    ax.set_ylim(0, 8.5)
    ax.set_xlabel("표본크기 $n$", fontsize=11.5, color=INK)
    ax.set_ylabel("MSE", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.text(16, 4.6, "$n$ 이 커지면 셋의 차이가 사라진다", fontsize=11,
            color=MUTED, ha="left", va="center")
    ax.set_title("순서는 모든 $n$ 에서 같다", fontsize=12.5, pad=10)

    fig.suptitle("정규모집단에서 $\\sigma^2$ 을 재는 세 가지 방법",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "variance_estimator_mse.png")


if __name__ == "__main__":
    moments_procedure()
    sample_moments_converge()
    mom_three_families()
    mom_vs_mle_efficiency()
    gmm_overidentification()
    loglik_additivity()
    variance_estimator_mse()
