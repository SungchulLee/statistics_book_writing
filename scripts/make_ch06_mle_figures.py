r"""6.4 최대가능도 절의 개념 그림을 생성한다.

그림이 없던 쪽 가운데 개념을 떠받치는 다섯 쪽을 골랐다. log_likelihood 쪽은
"시각화"라는 제목을 달고도 그림이 없어 두 장을 넣는다. 3·5장 그림과 같은
팔레트를 쓴다.

  ch06/mle/img/probability_vs_likelihood.png   확률과 가능도 — 무엇을 고정하는가
  ch06/mle/img/mle_candidates.png              후보 분포 가운데 고르기
  ch06/mle/img/bernoulli_loglik_curve.png      보기 1 의 로그가능도 곡선
  ch06/mle/img/likelihood_vs_loglik_scale.png  보기 3 의 두 축
  ch06/mle/img/fisher_curvature_se.png         곡률에서 표준오차로
  ch06/mle/img/mle_asymptotics.png             n 이 커지며 정규로

실행:  python3 scripts/make_ch06_mle_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

from math import comb

import numpy as np

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
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch06/mle/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def normal_pdf(x, mu, sd):
    return np.exp(-((x - mu) / sd) ** 2 / 2) / (sd * np.sqrt(2 * np.pi))


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# ==================================================================
# 1. 확률과 가능도 — 같은 식, 무엇을 고정하느냐만 다르다
# ==================================================================
def probability_vs_likelihood():
    n, k = 10, 7                 # 쪽의 보기와 같은 숫자
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) p 를 고정하고 자료를 본다
    ax = axes[0]
    xs = np.arange(n + 1)
    pmf = np.array([comb(n, x) * 0.5 ** n for x in xs])
    colors = [RED if x == k else "#B0BEC5" for x in xs]
    ax.bar(xs, pmf, width=0.68, color=colors, edgecolor="white", linewidth=1.0)
    ax.text(k, pmf[k] + 0.012, "관측된 자료\n$k = 7$", fontsize=11, color=RED,
            ha="center", va="bottom", linespacing=1.4)
    ax.text(0.2, 0.245, "$p = 0.5$ 로 **고정**".replace("**", ""), fontsize=12,
            color=INK, ha="left", va="top")
    ax.text(0.2, 0.215, "막대의 합 $= 1$", fontsize=11, color=MUTED,
            ha="left", va="top")
    ax.set_xticks(xs)
    ax.set_xlabel("앞면의 개수 $k$", fontsize=11.5, color=INK)
    ax.set_ylim(0, 0.30)
    bare_axis(ax)
    ax.set_title("확률 — 모수를 고정하고 자료를 묻는다", fontsize=13, pad=10)

    # (b) 자료를 고정하고 p 를 본다
    ax = axes[1]
    p = np.linspace(0.001, 0.999, 600)
    lik = p ** k * (1 - p) ** (n - k)
    ax.plot(p, lik, color=BLUE, linewidth=2.4, zorder=5)
    ax.fill_between(p, lik, color=BLUE, alpha=0.14, zorder=3)

    for pv, color in [(0.5, MUTED), (0.7, RED)]:
        lv = pv ** k * (1 - pv) ** (n - k)
        ax.plot([pv, pv], [0, lv], color=color, linewidth=1.6,
                linestyle="--", zorder=6)
        ax.plot([pv], [lv], "o", color=color, markersize=7, zorder=7)
        ax.text(pv, lv + 0.00012, f"$L({pv}) = {lv:.4f}$", fontsize=10.5,
                color=color, ha="center", va="bottom")

    ax.text(0.04, 0.00215, "자료 $k = 7$ 을 고정", fontsize=12, color=INK,
            ha="left", va="top")
    ax.text(0.04, 0.00195, "곡선 아래 넓이는 $1$ 이 아니다\n"
                           "— 가능도는 확률이 아니다",
            fontsize=11, color=MUTED, ha="left", va="top", linespacing=1.45)
    ax.text(0.7, -0.00018, "$\\hat p = 0.7$", fontsize=12, color=RED,
            ha="center", va="top")
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.0024)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bare_axis(ax)
    ax.set_title("가능도 — 자료를 고정하고 모수를 묻는다", fontsize=13, pad=10)

    fig.suptitle("$L(p) = p^7(1-p)^3$ — 같은 식을 어느 쪽에서 보느냐의 차이",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "probability_vs_likelihood.png")


# ==================================================================
# 2. 후보 가운데 고르기
# ==================================================================
def mle_candidates():
    rng = np.random.default_rng(3)
    mu_true, sd = 5.0, 1.0
    data = rng.normal(mu_true, sd, 12)
    mle = data.mean()

    def loglik(m):
        return np.sum(-0.5 * np.log(2 * np.pi) - (data - m) ** 2 / 2)

    # 두 후보를 비대칭으로 둔다. 대칭이면 로그가능도 값이 똑같이 나와
    # 범례가 오기처럼 보인다.
    cands = [(mle - 1.5, ORANGE, "너무 작다"),
             (mle, GREEN, "MLE"),
             (mle + 2.4, BLUE, "너무 크다")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) 후보 분포 셋을 자료 위에 얹는다
    ax = axes[0]
    x = np.linspace(mle - 5.2, mle + 5.2, 600)
    for m, color, name in cands:
        ax.plot(x, normal_pdf(x, m, sd), color=color, linewidth=2.2,
                label=f"{name}   $\\ell = {loglik(m):.1f}$")
    ax.plot(data, np.full_like(data, -0.022), "|", color=INK, markersize=14,
            markeredgewidth=1.8)
    ax.text(mle, -0.055, "관측된 자료 12개", fontsize=11, color=INK,
            ha="center", va="top")
    ax.set_ylim(-0.085, 0.47)
    ax.set_xlim(mle - 5.2, mle + 5.2)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("후보 분포를 자료 위에 얹어 본다", fontsize=13, pad=10)

    # (b) 로그가능도 곡선 위의 같은 세 점
    ax = axes[1]
    ms = np.linspace(mle - 3.4, mle + 3.4, 500)
    lls = np.array([loglik(m) for m in ms])
    ax.plot(ms, lls, color=INK, linewidth=2.2, zorder=4)
    for m, color, name in cands:
        ax.plot([m], [loglik(m)], "o", color=color, markersize=9, zorder=6)
    ax.plot([mle, mle], [lls.min(), loglik(mle)], color=GREEN, linewidth=1.4,
            linestyle="--", zorder=5)
    ax.text(mle, loglik(mle) + 1.2, "꼭대기가 MLE", fontsize=11.5,
            color=GREEN, ha="center", va="bottom")
    ax.set_xlabel("모수 $\\mu$", fontsize=11.5, color=INK)
    ax.set_ylabel("로그가능도 $\\ell(\\mu)$", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("같은 세 후보를 곡선 위에서 보면", fontsize=13, pad=10)

    fig.suptitle("MLE 는 자료를 가장 그럴듯하게 만드는 후보를 고르는 일이다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "mle_candidates.png")


# ==================================================================
# 3·4. 쪽의 보기와 같은 자료로 그린 로그가능도
# ==================================================================
def _page_coins():
    """보기 1·3 과 같은 자료. 같은 시드이므로 k = 67 이 그대로 나온다."""
    rng = np.random.default_rng(1)
    coins = rng.binomial(n=1, p=0.7, size=100)
    return coins.sum(), len(coins)


def bernoulli_loglik_curve():
    k, n = _page_coins()
    ps = np.linspace(0.01, 0.99, 200)
    ll = k * np.log(ps) + (n - k) * np.log(1 - ps)
    mle = k / n
    ll_max = k * np.log(mle) + (n - k) * np.log(1 - mle)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(ps, ll, color=BLUE, linewidth=2.4, zorder=5)
    ax.plot([mle, mle], [-260, ll_max], color=RED, linewidth=1.6,
            linestyle="--", zorder=6)
    ax.plot([mle], [ll_max], "o", color=RED, markersize=9, zorder=7)
    ax.text(mle + 0.015, ll_max - 6,
            f"$\\hat p = k/n = {mle:.2f}$\n$\\ell(\\hat p) = {ll_max:.2f}$",
            fontsize=11.5, color=RED, ha="left", va="top", linespacing=1.45)

    # 봉우리 부근이 평평하다는 것을 보이는 보조선
    ax.axhline(ll_max - 2, color=MUTED, linewidth=1.1, linestyle=":",
               zorder=4)
    ax.text(0.03, ll_max - 2, "꼭대기에서 $2$ 만큼 내려온 높이", fontsize=10,
            color=MUTED, ha="left", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(-260, ll_max + 28)
    ax.set_xlabel("모수 $p$", fontsize=12, color=INK)
    ax.set_ylabel("로그가능도 $\\ell(p)$", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title(f"앞면 {k}회 / {n}회 던지기의 로그가능도", fontsize=13.5,
                 pad=12)
    fig.tight_layout()
    save(fig, OUT + "bernoulli_loglik_curve.png")


def likelihood_vs_loglik_scale():
    k, n = _page_coins()
    ps = np.linspace(0.01, 0.99, 600)
    mle = k / n
    ll = k * np.log(ps) + (n - k) * np.log(1 - ps)
    lik = np.exp(ll)
    ll_max = k * np.log(mle) + (n - k) * np.log(1 - mle)
    lik_at_07 = np.exp(k * np.log(0.7) + (n - k) * np.log(0.3))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    ax.plot(ps, lik, color=ORANGE, linewidth=2.4, zorder=5)
    ax.fill_between(ps, lik, color=ORANGE, alpha=0.15, zorder=3)
    ax.plot([mle, mle], [0, np.exp(ll_max)], color=RED, linewidth=1.5,
            linestyle="--", zorder=6)
    mant, expo = f"{lik_at_07:.2e}".split("e")
    ax.text(0.03, np.exp(ll_max) * 0.92,
            f"$p = 0.7$ 에서  $L = {mant} \\times 10^{{{int(expo)}}}$",
            fontsize=11, color=INK, ha="left", va="top")
    ax.text(0.03, np.exp(ll_max) * 0.62,
            "$n$ 이 조금만 커져도\n0 으로 언더플로된다", fontsize=11,
            color=MUTED, ha="left", va="top", linespacing=1.45)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.exp(ll_max) * 1.12)
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    ax.set_ylabel("가능도 $L(p)$", fontsize=11.5, color=INK)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("가능도 — 세로축이 $10^{-28}$ 자리", fontsize=13, pad=10)

    ax = axes[1]
    ax.plot(ps, ll, color=BLUE, linewidth=2.4, zorder=5)
    ax.plot([mle, mle], [-260, ll_max], color=RED, linewidth=1.5,
            linestyle="--", zorder=6)
    ax.text(0.03, -30, f"$\\ell(0.7) = -63.63$\n$\\ell(\\hat p) = {ll_max:.2f}$",
            fontsize=11, color=INK, ha="left", va="top", linespacing=1.45)
    ax.set_xlim(0, 1)
    ax.set_ylim(-260, ll_max + 28)
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    ax.set_ylabel("로그가능도 $\\ell(p)$", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("로그가능도 — 다루기 좋은 음수", fontsize=13, pad=10)

    for ax in axes:
        # 파선 오른쪽으로 띄운다. 선 위에 얹으면 글자가 그어진 것처럼 보인다.
        ax.text(mle + 0.025, ax.get_ylim()[0], f"$\\hat p = {mle:.2f}$",
                fontsize=11, color=RED, ha="left", va="bottom")

    fig.suptitle("축만 바뀌었을 뿐 봉우리의 자리는 같다", fontsize=13.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, OUT + "likelihood_vs_loglik_scale.png")


# ==================================================================
# 5. 곡률에서 표준오차로
# ==================================================================
def fisher_curvature_se():
    k, n = _page_coins()
    mle = k / n
    se = np.sqrt(mle * (1 - mle) / n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

    # (a) 로그가능도와 그 2차 근사
    ax = axes[0]
    ps = np.linspace(mle - 0.22, mle + 0.22, 600)
    ll = k * np.log(ps) + (n - k) * np.log(1 - ps)
    ll_max = k * np.log(mle) + (n - k) * np.log(1 - mle)
    curv = n / (mle * (1 - mle))                 # -ℓ''(p̂) = n I(p̂)
    quad = ll_max - curv * (ps - mle) ** 2 / 2

    ax.plot(ps, ll, color=BLUE, linewidth=2.4, zorder=5, label="로그가능도")
    ax.plot(ps, quad, color=GREEN, linewidth=2.0, linestyle="--", zorder=6,
            label="꼭대기에서의 2차 근사")
    ax.plot([mle], [ll_max], "o", color=RED, markersize=8, zorder=7)
    # 점 오른쪽에 붙인다. 위에 두면 제목과 부딪힌다.
    ax.text(mle + 0.012, ll_max, "$\\hat p$", fontsize=12.5, color=RED,
            ha="left", va="center")
    ax.text(mle - 0.215, ll_max - 15.2,
            f"$-\\ell''(\\hat p) = n\\,I(\\hat p) = {curv:.0f}$",
            fontsize=12, color=GREEN, ha="left", va="center")
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    ax.set_ylabel("로그가능도", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="lower center")
    ax.set_title("꼭대기 부근은 포물선으로 보인다", fontsize=13, pad=10)

    # (b) 그 곡률이 정하는 표본분포
    ax = axes[1]
    x = np.linspace(mle - 4.2 * se, mle + 4.2 * se, 600)
    y = normal_pdf(x, mle, se)
    ax.plot(x, y, color=BLUE, linewidth=2.3, zorder=5)
    ax.fill_between(x, y, where=np.abs(x - mle) <= 1.96 * se, color=BLUE,
                    alpha=0.18, zorder=3)
    top = y.max()
    for s in (-1.96, 1.96):
        ax.plot([mle + s * se, mle + s * se], [0, normal_pdf(mle + s * se,
                                                             mle, se)],
                color=RED, linewidth=1.4, linestyle="--", zorder=6)
    ax.text(mle, top * 0.42, "$\\hat p \\pm 1.96\\,\\widehat{SE}$",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(mle, top * 1.06,
            f"$\\widehat{{SE}} = 1/\\sqrt{{n I(\\hat p)}} = {se:.3f}$",
            fontsize=12, color=BLUE, ha="center", va="bottom")
    ax.set_xlabel("$\\hat p$ 의 표본분포", fontsize=11.5, color=INK)
    ax.set_ylim(0, top * 1.3)
    bare_axis(ax)
    ax.set_title("곡률이 크면 분포가 좁다", fontsize=13, pad=10)

    fig.suptitle("휘어짐을 재면 표준오차가 나온다", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "fisher_curvature_se.png")


# ==================================================================
# 6. n 이 커지면 MLE 는 정규로 간다
# ==================================================================
def mle_asymptotics():
    rng = np.random.default_rng(0)
    lam, B = 1.0, 200_000
    ns = [5, 20, 100]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=False)
    for ax, n in zip(axes, ns):
        # 지수분포의 MLE 는 1/X̄ 이다. 유한 n 에서는 위로 치우쳐 있다.
        xbar = rng.gamma(shape=n, scale=1 / (lam * n), size=B)
        lam_hat = 1 / xbar
        se = lam / np.sqrt(n)                 # 1/sqrt(n I), I(λ) = 1/λ²

        lo, hi = lam - 3.6 * se, lam + 3.6 * se
        ax.hist(lam_hat, bins=np.linspace(lo, hi, 70), density=True,
                color=BLUE_F, edgecolor=BLUE, linewidth=0.6)
        x = np.linspace(lo, hi, 400)
        ax.plot(x, normal_pdf(x, lam, se), color=INK, linewidth=2.0,
                zorder=5, label="극한 정규분포")
        ax.axvline(lam, color=RED, linewidth=1.5, zorder=6)
        ax.axvline(lam_hat.mean(), color=GREEN, linewidth=1.5,
                   linestyle="--", zorder=6)

        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_title(f"$n = {n}$     평균 $= {lam_hat.mean():.3f}$",
                     fontsize=12.5, pad=8)

    axes[0].legend(fontsize=10, frameon=False, loc="upper right")
    # n 이 작을 때 1/X̄ 의 오른쪽 꼬리는 그림 밖까지 이어진다. 잘라 놓고
    # 말하지 않으면 분포를 실제보다 얌전해 보이게 만든다.
    axes[0].text(0.98, 0.42, "오른쪽 꼬리는\n그림 밖까지 이어진다",
                 transform=axes[0].transAxes, fontsize=10, color=MUTED,
                 ha="right", va="top", linespacing=1.45)
    axes[0].text(0.02, 0.86, "붉은 선이 참값,\n초록 파선이 $\\hat\\lambda$ 의 평균",
                 transform=axes[0].transAxes, fontsize=10.5, color=INK,
                 ha="left", va="top", linespacing=1.45)
    fig.suptitle("지수분포 $\\hat\\lambda = 1/\\bar X$ — 편향이 사라지고 "
                 "모양이 정규로 간다", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "mle_asymptotics.png")


if __name__ == "__main__":
    probability_vs_likelihood()
    mle_candidates()
    bernoulli_loglik_curve()
    likelihood_vs_loglik_scale()
    fisher_curvature_se()
    mle_asymptotics()
