r"""6.4 최대가능도 절에 남아 있던 여섯 쪽의 그림을 생성한다.

  ch06/mle/img/normal_loglik_surface.png      두 모수의 로그가능도 등고선
  ch06/mle/img/exponential_invariance.png     재모수화해도 답은 같은 점
  ch06/mle/img/poisson_mle_fit.png            계수 자료와 맞춘 분포
  ch06/mle/img/optimization_paths.png         Newton 반복과 봉우리가 둘일 때
  ch06/mle/img/score_variance_information.png 점수의 분산이 정보량이다
  ch06/mle/img/capture_recapture_likelihood.png  N 에 대한 가능도

실행:  python3 scripts/make_ch06_mle_examples_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import special, stats
from scipy.special import digamma, polygamma

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

OUT = "docs/ch06/mle/img/"


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
# 1. 정규분포 — 모수가 둘이면 지형이 된다
# ==================================================================
def normal_loglik_surface():
    rng = np.random.default_rng(2)
    n = 40
    x = rng.normal(5.0, 2.0, n)
    mu_hat = x.mean()
    sd_hat = x.std(ddof=0)          # MLE 는 n 으로 나눈다

    mus = np.linspace(mu_hat - 1.8, mu_hat + 1.8, 260)
    sds = np.linspace(sd_hat * 0.55, sd_hat * 1.75, 260)
    M, S = np.meshgrid(mus, sds)
    ll = (-n * np.log(S) - n / 2 * np.log(2 * np.pi)
          - ((x[:, None, None] - M) ** 2).sum(0) / (2 * S ** 2))
    ll_max = ll.max()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) 등고선
    ax = axes[0]
    levels = ll_max - np.array([40, 25, 15, 9, 5, 2, 0.5])
    cs = ax.contour(M, S, ll, levels=levels, colors=BLUE, linewidths=1.3)
    ax.contourf(M, S, ll, levels=np.append(levels, ll_max), colors=[
        "#FFFFFF", "#F4F8FD", "#E9F1FB", "#DCEBFB", "#C8DFF7", "#B2D2F3"],
        alpha=0.9)
    ax.plot([mu_hat], [sd_hat], "o", color=RED, markersize=9, zorder=6)
    ax.plot([mu_hat, mu_hat], [sds[0], sd_hat], color=RED, linewidth=1.2,
            linestyle="--", zorder=5)
    ax.plot([mus[0], mu_hat], [sd_hat, sd_hat], color=RED, linewidth=1.2,
            linestyle="--", zorder=5)
    ax.text(mu_hat + 0.08, sd_hat + 0.08,
            f"$(\\hat\\mu, \\hat\\sigma) = ({mu_hat:.2f}, {sd_hat:.2f})$",
            fontsize=11.5, color=RED, ha="left", va="bottom")
    ax.set_xlabel("평균 $\\mu$", fontsize=11.5, color=INK)
    ax.set_ylabel("표준편차 $\\sigma$", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.set_title(f"로그가능도 $\\ell(\\mu, \\sigma)$   ($n = {n}$)",
                 fontsize=13, pad=10)

    # (b) 자료와 맞춘 곡선
    ax = axes[1]
    ax.hist(x, bins=12, density=True, color="#ECEFF1", edgecolor=MUTED,
            linewidth=0.7)
    xs = np.linspace(x.min() - 1.5, x.max() + 1.5, 400)
    ax.plot(xs, stats.norm.pdf(xs, mu_hat, sd_hat), color=BLUE,
            linewidth=2.6, zorder=5)
    ax.plot(x, np.full(n, -0.012), "|", color=INK, markersize=10,
            markeredgewidth=1.2)
    ax.plot([mu_hat, mu_hat], [0, stats.norm.pdf(mu_hat, mu_hat, sd_hat)],
            color=RED, linewidth=1.5, linestyle="--", zorder=6)
    ax.text(mu_hat, stats.norm.pdf(mu_hat, mu_hat, sd_hat) * 1.04,
            "$\\hat\\mu$", fontsize=12.5, color=RED, ha="center", va="bottom")
    ax.set_ylim(-0.03, 0.26)
    ax.set_xlabel("$x$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.set_title("그 꼭대기가 고른 분포", fontsize=13, pad=10)

    fig.suptitle("모수가 둘이면 봉우리가 아니라 지형을 찾는 일이 된다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "normal_loglik_surface.png")


# ==================================================================
# 2. 지수분포 — 재모수화해도 답은 따라 옮겨 간다
# ==================================================================
def exponential_invariance():
    rng = np.random.default_rng(6)
    n = 30
    x = rng.exponential(1 / 1.5, n)      # 비율 lambda = 1.5
    lam_hat = 1 / x.mean()
    th_hat = x.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    lams = np.linspace(0.6, 3.2, 500)
    ll_lam = n * np.log(lams) - lams * x.sum()
    ths = np.linspace(1 / 3.2, 1 / 0.6, 500)
    ll_th = -n * np.log(ths) - x.sum() / ths

    for ax, t, ll, hat, name, sym, color in [
            (axes[0], lams, ll_lam, lam_hat, "비율로 적으면", "\\lambda", BLUE),
            (axes[1], ths, ll_th, th_hat, "평균으로 적으면", "\\theta = 1/\\lambda",
             GREEN)]:
        ax.plot(t, ll - ll.max(), color=color, linewidth=2.6, zorder=5)
        ax.plot([hat, hat], [-14, 0], color=RED, linewidth=1.5,
                linestyle="--", zorder=6)
        ax.plot([hat], [0], "o", color=RED, markersize=8, zorder=7)
        ax.text(hat, 0.5, f"${hat:.3f}$", fontsize=12, color=RED,
                ha="center", va="bottom")
        ax.set_xlabel(f"${sym}$", fontsize=12.5, color=color)
        ax.set_ylim(-14, 2.2)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(name, fontsize=13, pad=10)

    axes[0].text(2.55, -10.0, f"$\\hat\\lambda = 1/\\bar x$", fontsize=12,
                 color=INK, ha="center", va="center")
    axes[1].text(1.05, -10.0, f"$\\hat\\theta = \\bar x = 1/\\hat\\lambda$",
                 fontsize=12, color=INK, ha="center", va="center")
    axes[1].text(1.05, -12.0, f"${th_hat:.3f} = 1 / {lam_hat:.3f}$",
                 fontsize=11, color=RED, ha="center", va="center")

    fig.suptitle("함수적 불변성 — 모수를 바꿔 적어도 MLE 는 그대로 따라간다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "exponential_invariance.png")


# ==================================================================
# 3. 포아송 — 계수 자료에 맞추기
# ==================================================================
def poisson_mle_fit():
    rng = np.random.default_rng(8)
    n, lam_true = 60, 3.0
    x = rng.poisson(lam_true, n)
    lam_hat = x.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # (a) 관측된 계수와 맞춘 확률질량함수
    ax = axes[0]
    ks = np.arange(0, x.max() + 3)
    obs = np.array([(x == k).mean() for k in ks])
    ax.bar(ks - 0.17, obs, width=0.34, color="#CFD8DC", edgecolor=MUTED,
           linewidth=0.8, label="관측된 비율")
    ax.bar(ks + 0.17, stats.poisson.pmf(ks, lam_hat), width=0.34,
           color=BLUE_F, edgecolor=BLUE, linewidth=1.2,
           label=f"$\\mathrm{{Poisson}}({lam_hat:.2f})$")
    ax.set_xticks(ks)
    ax.set_xlabel("계수 $k$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title(f"관측 {n}개와 맞춘 분포", fontsize=13, pad=10)

    # (b) 로그가능도
    ax = axes[1]
    lams = np.linspace(1.6, 5.2, 500)
    ll = x.sum() * np.log(lams) - n * lams
    ax.plot(lams, ll - ll.max(), color=BLUE, linewidth=2.6, zorder=5)
    ax.plot([lam_hat, lam_hat], [-16, 0], color=RED, linewidth=1.5,
            linestyle="--", zorder=6)
    ax.plot([lam_hat], [0], "o", color=RED, markersize=8, zorder=7)
    ax.text(lam_hat - 0.08, -3.4,
            f"$\\hat\\lambda = \\bar x = {lam_hat:.2f}$", fontsize=12,
            color=RED, ha="right", va="top")
    # 세로 점선은 곡선이 -1.92 를 지나는 자리, 곧 가능도비 구간의 양 끝이다.
    inside = lams[(ll - ll.max()) >= -1.92]
    for b in (inside[0], inside[-1]):
        ax.plot([b, b], [-16, -1.92], color=MUTED, linewidth=1.2,
                linestyle=":")
    ax.text(inside[0] - 0.04, -15.2,
            f"$[{inside[0]:.2f},\\; {inside[-1]:.2f}]$", fontsize=11,
            color=MUTED, ha="right", va="bottom")
    ax.axhline(-1.92, color=MUTED, linewidth=1.1, linestyle=":")
    ax.text(5.15, -1.92, "꼭대기에서 $1.92$ 내려온 높이\n(가능도비 $95\\%$ 구간)",
            fontsize=10, color=MUTED, ha="right", va="bottom",
            linespacing=1.4)
    ax.set_xlabel("$\\lambda$", fontsize=12.5, color=INK)
    ax.set_ylim(-16, 2.4)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("로그가능도와 그 봉우리", fontsize=13, pad=10)

    fig.suptitle("포아송 MLE — 표본평균이 곧 답이다", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "poisson_mle_fit.png")


# ==================================================================
# 4. 최적화 — 반복으로 오르기, 그리고 봉우리가 둘일 때
# ==================================================================
def optimization_paths():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))

    # (a) Newton-Raphson 으로 감마 형상모수 찾기
    ax = axes[0]
    rng = np.random.default_rng(3)
    n = 80
    x = rng.gamma(4.0, 1.0, n)
    s = np.log(x.mean()) - np.log(x).mean()

    def ell(a):                       # 척도를 프로파일로 소거한 로그가능도
        return n * ((a - 1) * np.log(x).mean() - a * np.log(x.mean() / a)
                    - a - special.gammaln(a))

    def d1(a):
        return n * (np.log(a) - digamma(a) - s)

    def d2(a):
        return n * (1 / a - polygamma(1, a))

    a_grid = np.linspace(1.2, 9.0, 500)
    ax.plot(a_grid, ell(a_grid) - ell(a_grid).max(), color=BLUE,
            linewidth=2.4, zorder=4)

    a = 1.5
    path = [a]
    for _ in range(5):
        a = a - d1(a) / d2(a)
        path.append(a)
    path = np.array(path)
    ax.plot(path, ell(path) - ell(a_grid).max(), "o-", color=RED,
            markersize=7, linewidth=1.4, zorder=6)
    for i, (ai, li) in enumerate(zip(path[:4],
                                     ell(path[:4]) - ell(a_grid).max())):
        ax.text(ai, li - 1.1, f"${i}$", fontsize=11, color=RED,
                ha="center", va="top")
    # 곡선 오른쪽 위 빈 곳에 적는다.
    ax.text(8.9, -2.2,
            f"출발 $\\alpha^{{(0)}} = 1.5$\n"
            f"네 걸음 만에 ${path[4]:.4f}$\n참값은 $4$",
            fontsize=11, color=INK, ha="right", va="top", linespacing=1.5)
    ax.set_xlabel("형상모수 $\\alpha$", fontsize=11.5, color=INK)
    ax.set_ylim(-18, 2.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("닫힌 형태가 없으면 반복해 오른다 (Newton–Raphson)",
                 fontsize=12.5, pad=10)

    # (b) 봉우리가 둘인 혼합모형 — 쪽의 예제와 같은 자료
    ax = axes[1]
    rng = np.random.default_rng(42)
    m = 200
    z = rng.binomial(1, 0.4, m)
    data = np.where(z, rng.normal(0, 1, m), rng.normal(4, 1, m))

    g = np.linspace(-2.5, 6.5, 200)
    M1, M2 = np.meshgrid(g, g)
    comp = (0.4 * stats.norm.pdf(data[:, None, None], M1, 1.0)
            + 0.6 * stats.norm.pdf(data[:, None, None], M2, 1.0))
    L = np.log(comp).sum(0)
    ax.contourf(M1, M2, L, levels=np.linspace(L.max() - 120, L.max(), 14),
                cmap="Blues", alpha=0.9)
    ax.contour(M1, M2, L, levels=np.linspace(L.max() - 120, L.max(), 14),
               colors=BLUE, linewidths=0.6, alpha=0.6)

    # 두 꼭대기 — 성분의 이름을 맞바꾼 것이라 값이 같다
    for pt, name in [((0.0, 4.0), "봉우리 A"), ((4.0, 0.0), "봉우리 B")]:
        ax.plot(*pt, "o", color=RED, markersize=9, zorder=6)
        # 짙은 등고선 위이므로 흰 글씨가 읽힌다.
        ax.text(pt[0] + 0.28, pt[1] + 0.28, name, fontsize=10.5,
                color="white", ha="left", va="bottom", fontweight="bold")
    ax.plot([-2.5, 6.5], [-2.5, 6.5], color=MUTED, linewidth=1.1,
            linestyle="--")
    ax.text(6.35, 3.4, "$\\mu_1 = \\mu_2$\n(성분이 하나로 무너지는 선)",
            fontsize=9.5, color=MUTED, ha="right", va="top",
            linespacing=1.4)
    ax.set_xlabel("$\\mu_1$", fontsize=12.5, color=INK)
    ax.set_ylabel("$\\mu_2$", fontsize=12.5, color=INK)
    clean_axis(ax)
    ax.set_title("봉우리가 둘이면 출발값이 답을 고른다", fontsize=12.5, pad=10)

    fig.suptitle("최대화는 언제나 같은 일이지만, 지형이 늘 순하지는 않다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "optimization_paths.png")


# ==================================================================
# 5. 점수의 분산이 정보량이다
# ==================================================================
def score_variance_information():
    rng = np.random.default_rng(42)
    N = 200_000
    sigma = 2.0
    mu = 5.0
    data = rng.normal(mu, sigma, N)
    score = (data - mu) / sigma ** 2          # 정규분포 평균의 점수함수

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) 점수의 분포
    ax = axes[0]
    ax.hist(score, bins=90, density=True, color=BLUE_F, edgecolor=BLUE,
            linewidth=0.5)
    ax.axvline(0, color=RED, linewidth=1.8, zorder=6)
    ax.text(0.02, 0.92, f"평균 ${score.mean():.4f}$  (이론값 $0$)\n"
                        f"분산 ${score.var():.4f}$  (이론값 ${1/sigma**2:.2f}$)",
            transform=ax.transAxes, fontsize=11.5, color=INK, ha="left",
            va="top", linespacing=1.5)
    ax.text(0.02, 0.62, "점수가 크게 흔들릴수록\n자료가 모수를 잘 짚어낸다",
            transform=ax.transAxes, fontsize=10.5, color=MUTED, ha="left",
            va="top", linespacing=1.5)
    ax.set_xlabel("점수 $s(X; \\mu) = (X - \\mu)/\\sigma^2$", fontsize=11.5,
                  color=INK)
    bare_axis(ax)
    ax.set_title(f"$N({mu:g}, {sigma**2:g})$ 에서 뽑은 점수 {N//10000}만 개",
                 fontsize=12.5, pad=10)

    # (b) 네 분포에서 해석값과 수치값
    ax = axes[1]
    rows = [("정규 $\\mu$", 1 / sigma ** 2,
             ((rng.normal(mu, sigma, N) - mu) / sigma ** 2).var()),
            ("베르누이 $p$", 1 / (0.3 * 0.7),
             ((rng.binomial(1, 0.3, N) - 0.3) / (0.3 * 0.7)).var()),
            ("포아송 $\\lambda$", 1 / 3.0,
             ((rng.poisson(3.0, N) - 3.0) / 3.0).var()),
            ("지수 $\\lambda$", 1 / 2.0 ** 2,
             (1 / 2.0 - rng.exponential(1 / 2.0, N)).var())]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys + 0.18, [r[1] for r in rows], height=0.34, color=ORANGE_F,
            edgecolor=ORANGE, linewidth=1.4, label="해석값")
    ax.barh(ys - 0.18, [r[2] for r in rows], height=0.34, color=BLUE_F,
            edgecolor=BLUE, linewidth=1.4, label="점수의 표본분산")
    for y, r in zip(ys, rows):
        ax.text(max(r[1], r[2]) + 0.12, y, f"${r[1]:.3f}$", fontsize=10.5,
                color=INK, ha="left", va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5)
    ax.set_xlim(0, 6.6)
    ax.set_xlabel("$I(\\theta)$", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")
    ax.set_title("네 분포에서 둘이 일치한다", fontsize=12.5, pad=10)

    fig.suptitle("Fisher 정보량은 점수함수의 분산이다", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "score_variance_information.png")


# ==================================================================
# 6. 포획–재포획 — N 에 대한 가능도
# ==================================================================
def capture_recapture_likelihood():
    def like(N, c, r, t):
        return (special.comb(N - c, r - t) * special.comb(c, t)
                / special.comb(N, r))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, (c, r, t) in zip(axes, [(10, 10, 3), (5, 6, 2)]):
        lo = c + r - t
        Ns = np.arange(lo, lo + 110)
        L = np.array([like(N, c, r, t) for N in Ns])
        # cr/t 가 정수면 두 값이 함께 최대가 된다. 그대로 표시한다.
        tops = Ns[L >= L.max() - 1e-12]

        ax.bar(Ns, L, width=0.9, color=BLUE_F, edgecolor=BLUE, linewidth=0.5)
        for h in tops:
            ax.plot([h, h], [0, L.max()], color=RED, linewidth=1.8, zorder=6)
        label = (f"$\\hat N = {tops[0]}$" if len(tops) == 1
                 else f"$\\hat N = {tops[0]}$ 와 ${tops[-1]}$ 가 함께 최대")
        ax.text(tops[0], L.max() * 1.04, label, fontsize=12, color=RED,
                ha="center", va="bottom")
        note = (f"$cr/t = {c}\\times{r}/{t} = {c*r/t:.1f}$"
                + ("" if len(tops) == 1 else " — 정수라서 동점"))
        # 막대 테두리가 글자 사이로 비치므로 흰 바탕을 깔아 준다.
        ax.text(tops[-1] + 3.0, L.max() * 0.62, note, fontsize=10.5,
                color=INK, ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=2.0))

        # 가능도가 꼭대기의 절반 이상인 구간 — 오른쪽으로 길다
        keep = Ns[L >= L.max() / 2]
        ax.plot([keep[0], keep[-1]], [-L.max() * 0.07] * 2, color=MUTED,
                linewidth=3.0, solid_capstyle="butt")
        ax.text((keep[0] + keep[-1]) / 2, -L.max() * 0.12,
                f"가능도가 절반 이상인 구간  ${keep[0]}$–${keep[-1]}$",
                fontsize=10, color=MUTED, ha="center", va="top")

        ax.set_xlim(lo - 2, lo + 92)
        ax.set_ylim(-L.max() * 0.22, L.max() * 1.2)
        ax.set_xlabel("개체군 크기 $N$", fontsize=11.5, color=INK)
        bare_axis(ax)
        ax.set_title(f"표지 $c = {c}$, 재포획 $r = {r}$, 그중 표지 $t = {t}$",
                     fontsize=12.5, pad=10)

    fig.suptitle("포획–재포획 — 봉우리는 뾰족하지만 오른쪽 꼬리가 길다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "capture_recapture_likelihood.png")


if __name__ == "__main__":
    normal_loglik_surface()
    exponential_invariance()
    poisson_mle_fit()
    optimization_paths()
    score_variance_information()
    capture_recapture_likelihood()
