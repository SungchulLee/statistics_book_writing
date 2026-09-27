r"""6.5 베이즈 절에서 그림이 없던 네 쪽의 그림을 생성한다.

같은 절의 bayesian_beta_conjugate 쪽에는 이미 그림이 있다. 그 그림은 n 을
고정하고 사전분포를 바꿔 가며 민감도를 보이므로, 여기서는 그것과 겹치지
않는 네 가지를 그린다.

  ch06/bayesian/img/prior_likelihood_posterior.png  세 요소와 점추정 셋
  ch06/bayesian/img/conjugate_updating.png          갱신 규칙과 그 예
  ch06/bayesian/img/map_vs_mle_penalty.png          MAP = MLE + 벌점
  ch06/bayesian/img/posterior_weighted_average.png  정밀도로 가중한 절충

실행:  python3 scripts/make_ch06_bayes_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

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

OUT = "docs/ch06/bayesian/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# ==================================================================
# 1. 사전분포 × 가능도 ∝ 사후분포, 그리고 점추정 셋
# ==================================================================
def prior_likelihood_posterior():
    a0, b0 = 2, 2            # 사전분포 Beta(2, 2)
    k, n = 8, 10             # 관측
    a1, b1 = a0 + k, b0 + n - k      # 사후분포 Beta(10, 4)

    p = np.linspace(0, 1, 800)
    prior = stats.beta.pdf(p, a0, b0)
    # 가능도는 p 의 밀도가 아니다. 같은 축에 얹기 위해 넓이가 1 이 되도록
    # 눈금만 맞춘다(= Beta(k+1, n-k+1) 밀도).
    lik = stats.beta.pdf(p, k + 1, n - k + 1)
    post = stats.beta.pdf(p, a1, b1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) 세 곡선
    ax = axes[0]
    ax.plot(p, prior, color=ORANGE, linewidth=2.2, label=f"사전분포  Beta({a0}, {b0})")
    ax.fill_between(p, prior, color=ORANGE, alpha=0.12)
    ax.plot(p, lik, color=GREEN, linewidth=2.2, linestyle="--",
            label=f"가능도  $k = {k}$, $n = {n}$")
    ax.plot(p, post, color=BLUE, linewidth=2.6,
            label=f"사후분포  Beta({a1}, {b1})")
    ax.fill_between(p, post, color=BLUE, alpha=0.14)

    ax.annotate("", xy=(a1 / (a1 + b1), 0.38), xytext=(0.5, 0.38),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.5))
    # 화살표 위에 적는다. 아래에 두면 x 축 눈금과 겹친다.
    ax.text(0.5 + (a1 / (a1 + b1) - 0.5) / 2, 0.5,
            "사전분포가 자료 쪽으로 끌려간다", fontsize=10.5, color=INK,
            ha="center", va="bottom")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.7)
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("사전분포 $\\times$ 가능도 $\\propto$ 사후분포", fontsize=13,
                 pad=10)

    # (b) 사후분포 하나에서 나오는 세 가지 점추정
    ax = axes[1]
    mean = a1 / (a1 + b1)
    mode = (a1 - 1) / (a1 + b1 - 2)
    med = stats.beta.ppf(0.5, a1, b1)

    ax.plot(p, post, color=BLUE, linewidth=2.6, zorder=5)
    ax.fill_between(p, post, color=BLUE, alpha=0.14, zorder=3)

    marks = [(mean, "사후평균", "제곱오차 손실", RED, 3.05),
             (med, "사후중앙값", "절대오차 손실", PURPLE, 2.35),
             (mode, "MAP (최빈값)", "0-1 손실", GREEN, 1.65)]
    for v, name, loss, color, ytext in marks:
        ax.plot([v, v], [0, stats.beta.pdf(v, a1, b1)], color=color,
                linewidth=1.6, linestyle="--", zorder=6)
        ax.annotate(f"{name}  ${v:.3f}$\n{loss}", xy=(v, ytext * 0.62),
                    xytext=(0.985, ytext), fontsize=10.5, color=color,
                    ha="right", va="center", linespacing=1.4, zorder=7,
                    arrowprops=dict(arrowstyle="->", color=color,
                                    linewidth=1.1,
                                    connectionstyle="arc3,rad=0.15"))

    ax.set_xlim(0.28, 1.0)
    ax.set_ylim(0, 3.9)
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.set_title("한 사후분포에서 나오는 세 가지 요약", fontsize=13, pad=10)

    fig.suptitle("자료를 보기 전의 믿음이 자료를 만나 갱신된다", fontsize=13.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, OUT + "prior_likelihood_posterior.png")


# ==================================================================
# 2. 켤레성 — 모수만 갈아 끼운다
# ==================================================================
def conjugate_updating():
    a0, b0 = 1, 1
    k, n = 7, 10
    a1, b1 = a0 + k, b0 + n - k

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) 갱신 규칙을 상자로
    ax = axes[0]

    def box(xy, text, edge, face, w=3.3, h=1.25, fs=12.5):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.14",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.8, zorder=4))
        ax.text(x, y, text, fontsize=fs, color=edge, ha="center",
                va="center", zorder=5, linespacing=1.45)

    box((2.3, 3.0), "사전분포\n$\\mathrm{Beta}(\\alpha, \\beta)$", ORANGE,
        ORANGE_F)
    box((8.4, 3.0), "사후분포\n$\\mathrm{Beta}(\\alpha + k,\\; \\beta + n - k)$",
        BLUE, BLUE_F, w=4.4)
    ax.add_patch(FancyArrowPatch((4.1, 3.0), (6.1, 3.0), arrowstyle="-|>",
                                 mutation_scale=18, color=INK, linewidth=1.8,
                                 zorder=3))
    ax.text(5.1, 3.5, "성공 $k$, 실패 $n-k$", fontsize=11.5, color=INK,
            ha="center", va="bottom")
    ax.text(5.1, 1.95, "분포족은 그대로,\n모수만 더해진다", fontsize=11.5,
            color=RED, ha="center", va="top", linespacing=1.45)
    ax.text(5.35, 0.75,
            "$\\alpha$ 와 $\\beta$ 를 미리 본 성공·실패 횟수로 읽을 수 있다",
            fontsize=11, color=MUTED, ha="center", va="center")

    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.set_title("켤레성 — 갱신에 대해 닫혀 있다", fontsize=13, pad=10)

    # (b) 쪽의 보기 그대로
    ax = axes[1]
    p = np.linspace(0, 1, 800)
    ax.plot(p, stats.beta.pdf(p, a0, b0), color=ORANGE, linewidth=2.2,
            label=f"사전분포  Beta({a0}, {b0})")
    ax.fill_between(p, stats.beta.pdf(p, a0, b0), color=ORANGE, alpha=0.12)
    ax.plot(p, stats.beta.pdf(p, a1, b1), color=BLUE, linewidth=2.6,
            label=f"사후분포  Beta({a1}, {b1})")
    ax.fill_between(p, stats.beta.pdf(p, a1, b1), color=BLUE, alpha=0.14)

    # 세 값이 가까우므로 축 아래가 아니라 위쪽에 서로 어긋나게 적는다.
    ax.plot([0.5, 0.5], [0, 3.2], color=ORANGE, linewidth=1.6,
            linestyle="--", zorder=6)
    ax.text(0.485, 3.3, "사전평균 $0.5$", fontsize=10.5, color=ORANGE,
            ha="right", va="bottom")
    ax.plot([k / n, k / n], [0, 3.2], color=MUTED, linewidth=1.6,
            linestyle=":", zorder=6)
    ax.text(0.715, 3.3, "표본비율 $0.7$", fontsize=10.5, color=MUTED,
            ha="left", va="bottom")
    ax.plot([a1 / (a1 + b1), a1 / (a1 + b1)], [0, 2.75], color=BLUE,
            linewidth=1.8, zorder=6)
    ax.annotate("사후평균 $0.667$", xy=(a1 / (a1 + b1), 1.35),
                xytext=(0.33, 1.9), fontsize=10.5, color=BLUE, ha="center",
                va="center", zorder=7,
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.2))
    ax.text(0.33, 1.45, "둘 사이에 놓인다", fontsize=10, color=MUTED,
            ha="center", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3.95)
    ax.set_xlabel("모수 $p$", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title(f"$n = {n}$ 에서 $k = {k}$ 를 보았을 때", fontsize=13, pad=10)

    fig.tight_layout()
    save(fig, OUT + "conjugate_updating.png")


# ==================================================================
# 3. MAP = MLE + 벌점
# ==================================================================
def map_vs_mle_penalty():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) 로그가능도 + 로그사전분포 = 로그사후분포
    ax = axes[0]
    xbar, n, sigma, s0 = 2.0, 1, 1.0, 1.0
    th = np.linspace(-1.6, 3.6, 700)
    ll = -n * (th - xbar) ** 2 / (2 * sigma ** 2)
    lp = -th ** 2 / (2 * s0 ** 2)
    lpost = ll + lp
    map_hat = n * s0 ** 2 / (n * s0 ** 2 + sigma ** 2) * xbar

    ax.plot(th, ll, color=GREEN, linewidth=2.2, label="로그가능도")
    ax.plot(th, lp, color=ORANGE, linewidth=2.2, label="로그 사전분포")
    ax.plot(th, lpost, color=BLUE, linewidth=2.8, label="둘의 합 = 로그 사후분포")

    for v, color, name in [(xbar, GREEN, "MLE"), (0.0, ORANGE, "사전평균"),
                           (map_hat, BLUE, "MAP")]:
        ax.plot([v, v], [-7.4, np.interp(v, th, lpost if color == BLUE
                                         else (ll if color == GREEN else lp))],
                color=color, linewidth=1.4, linestyle="--", zorder=3)
        ax.text(v, -7.7, f"{name}\n${v:g}$", fontsize=10.5, color=color,
                ha="center", va="top", linespacing=1.4)

    ax.add_patch(FancyArrowPatch((xbar, -5.6), (map_hat, -5.6),
                                 arrowstyle="-|>", mutation_scale=14,
                                 color=RED, linewidth=1.5, zorder=6))
    ax.text((xbar + map_hat) / 2, -5.35, "사전분포가 끌어당긴 만큼",
            fontsize=10.5, color=RED, ha="center", va="bottom")

    ax.set_xlim(-1.6, 3.6)
    ax.set_ylim(-7.4, 0.9)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title(f"$n = {n}$, $\\bar x = {xbar:g}$, 사전분포 $N(0, 1)$",
                 fontsize=13, pad=10)

    # (b) 로그 사전분포가 곧 벌점이다
    ax = axes[1]
    t = np.linspace(-3, 3, 700)
    ax.plot(t, t ** 2 / 2, color=BLUE, linewidth=2.4,
            label="정규 사전분포  $\\theta^2 / 2\\sigma_0^2$  →  능형(L2)")
    ax.plot(t, np.abs(t), color=ORANGE, linewidth=2.4,
            label="라플라스 사전분포  $|\\theta| / b$  →  Lasso(L1)")
    ax.plot([0], [0], "o", color=RED, markersize=8, zorder=6)
    ax.annotate("$0$ 에서 꺾인다 —\n계수를 정확히 $0$ 으로 만든다",
                xy=(0, 0), xytext=(1.95, 0.62), fontsize=10.5, color=ORANGE,
                ha="center", va="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.2))
    ax.text(-2.85, 4.1, "벌점 $= -\\log \\pi(\\theta)$", fontsize=12,
            color=INK, ha="left", va="top")

    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("모수 $\\theta$", fontsize=11.5, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="upper center")
    ax.set_title("사전분포를 고르는 일이 벌점을 고르는 일이다", fontsize=13,
                 pad=10)

    fig.suptitle("MAP 는 로그가능도에 로그 사전분포를 더해 최대화한다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "map_vs_mle_penalty.png")


# ==================================================================
# 4. 사후평균 = 정밀도로 가중한 절충
# ==================================================================
def posterior_weighted_average():
    mu0, s0, sigma, xbar = 0.0, 1.0, 2.0, 3.0
    ns = [1, 5, 50]
    colors = ["#90CAF9", "#1E88E5", "#0D47A1"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) n 이 커질수록 사후분포가 x̄ 쪽으로
    ax = axes[0]
    x = np.linspace(-2.6, 5.2, 800)
    ax.plot(x, stats.norm.pdf(x, mu0, s0), color=ORANGE, linewidth=2.2,
            label="사전분포  $N(0, 1)$")
    ax.fill_between(x, stats.norm.pdf(x, mu0, s0), color=ORANGE, alpha=0.12)

    for n, c in zip(ns, colors):
        mu_n = (sigma ** 2 * mu0 + n * s0 ** 2 * xbar) / (sigma ** 2
                                                          + n * s0 ** 2)
        tau = np.sqrt(sigma ** 2 * s0 ** 2 / (sigma ** 2 + n * s0 ** 2))
        ax.plot(x, stats.norm.pdf(x, mu_n, tau), color=c, linewidth=2.4,
                label=f"사후분포  $n = {n}$   ($\\mu_n = {mu_n:.2f}$)")

    ax.plot([xbar, xbar], [0, 1.65], color=RED, linewidth=1.8, zorder=6)
    ax.text(xbar, 1.7, "$\\bar x = 3$", fontsize=11.5, color=RED,
            ha="center", va="bottom")
    ax.plot([mu0, mu0], [0, 0.44], color=ORANGE, linewidth=1.8, zorder=6)
    # 세로선 왼쪽에 적는다. 가운데에 두면 선이 글자를 가른다.
    ax.text(mu0 - 0.12, 0.47, "$\\mu_0 = 0$", fontsize=11.5, color=ORANGE,
            ha="right", va="bottom")

    ax.set_xlim(-2.6, 5.2)
    ax.set_ylim(-0.02, 2.0)
    ax.set_xlabel("모수 $\\theta$", fontsize=11.5, color=INK, labelpad=18)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("자료가 쌓이면 사후분포가 자료 쪽으로 옮겨 가며 좁아진다",
                 fontsize=13, pad=10)

    # (b) 가중치가 옮겨 가는 모습
    ax = axes[1]
    nn = np.arange(0, 51)
    w_data = nn * s0 ** 2 / (sigma ** 2 + nn * s0 ** 2)
    ax.plot(nn, w_data, color=BLUE, linewidth=2.6, label="자료의 몫")
    ax.plot(nn, 1 - w_data, color=ORANGE, linewidth=2.6,
            label="사전분포의 몫")
    ax.fill_between(nn, w_data, 1, color=ORANGE, alpha=0.10)
    ax.fill_between(nn, 0, w_data, color=BLUE, alpha=0.10)

    for n, c in zip(ns, colors):
        w = n * s0 ** 2 / (sigma ** 2 + n * s0 ** 2)
        ax.plot([n], [w], "o", color=c, markersize=8, zorder=6)
        ax.text(n + 1.2, w - 0.045, f"$n = {n}$ : {w:.0%}", fontsize=10.5,
                color=c, ha="left", va="center")

    ax.set_xlim(0, 57)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("표본크기 $n$", fontsize=11.5, color=INK)
    ax.set_ylabel("사후평균에서 차지하는 몫", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="center right")
    ax.set_title("몫은 정밀도의 비로 정해진다", fontsize=13, pad=10)

    fig.suptitle("사후평균 $= $ 사전평균과 표본평균의 정밀도 가중평균",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "posterior_weighted_average.png")


if __name__ == "__main__":
    prior_likelihood_posterior()
    conjugate_updating()
    map_vs_mle_penalty()
    posterior_weighted_average()
