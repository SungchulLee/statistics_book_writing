r"""6.3 추정량의 품질 다섯 쪽의 그림을 생성한다.

다섯 쪽 모두 그림이 없던 곳이다. 쪽마다 그 쪽이 정의하는 개념 하나를 그렸다.

  ch06/estimator_quality/img/mse_decomposition.png      MSE = 분산 + 편향²
  ch06/estimator_quality/img/bias_variance_dartboard.png 다트판 네 칸
  ch06/estimator_quality/img/shrinkage_mse.png          축소추정량의 최적 λ
  ch06/estimator_quality/img/consistency_asymptotics.png 일치성과 점근정규성
  ch06/estimator_quality/img/crlb_information.png       휘어짐·정보량·분산 하한
  ch06/estimator_quality/img/sufficiency_ladder.png     압축 사다리와 Rao–Blackwell

실행:  python3 scripts/make_ch06_quality_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# === 공통 설정 — 3장·5장 그림과 같은 팔레트 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch06/estimator_quality/img/"


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
# 1. MSE = 분산 + 편향²
# ==================================================================
def mse_decomposition():
    theta, mu, sd = 4.0, 4.8, 1.2          # 참값, 추정량의 평균, 표준오차
    bias = mu - theta

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.linspace(theta - 4.2, theta + 5.4, 700)
    y = normal_pdf(x, mu, sd)
    top = y.max()

    ax.plot(x, y, color=BLUE, linewidth=2.0, zorder=5)
    ax.fill_between(x, y, color=BLUE, alpha=0.16, zorder=3)

    # 참값과 추정량의 평균
    ax.plot([theta, theta], [0, top * 1.12], color=RED, linewidth=1.8,
            zorder=6)
    ax.text(theta, top * 1.15, "참값 $\\theta$", fontsize=12, color=RED,
            ha="center", va="bottom")
    ax.plot([mu, mu], [0, top * 1.02], color=BLUE, linewidth=1.8,
            linestyle="--", zorder=6)
    ax.text(mu + 0.1, top * 1.05, "추정량의 평균 $E[\\hat\\theta]$", fontsize=12,
            color=BLUE, ha="left", va="bottom")

    # 편향 = 두 세로선 사이의 거리
    ax.add_patch(FancyArrowPatch((theta, top * 0.62), (mu, top * 0.62),
                                 arrowstyle="<|-|>", mutation_scale=13,
                                 color=INK, linewidth=1.5, zorder=7))
    ax.text((theta + mu) / 2, top * 0.66, "편향", fontsize=11.5, color=INK,
            ha="center", va="bottom")

    # 분산 = 분포의 퍼짐
    ax.add_patch(FancyArrowPatch((mu - sd, top * 0.30), (mu + sd, top * 0.30),
                                 arrowstyle="<|-|>", mutation_scale=13,
                                 color=GREEN, linewidth=1.5, zorder=7))
    # 라벨을 화살표 오른쪽에 둔다. 가운데에 두면 세로 파선이 글자를 가른다.
    ax.text(mu + sd + 0.18, top * 0.30, "퍼짐 — 분산", fontsize=11.5,
            color=GREEN, ha="left", va="center")

    ax.text(theta - 3.9, top * 0.95,
            "$\\mathrm{MSE} = \\mathrm{Var} + \\mathrm{Bias}^2$\n"
            f"$= {sd ** 2:.2f} + {bias:.1f}^2 = {sd ** 2 + bias ** 2:.2f}$",
            fontsize=12.5, color=INK, ha="left", va="top", linespacing=1.5)
    ax.text(theta - 3.9, top * 0.52,
            "편향은 중심이 **어디에** 있는지,\n분산은 그 중심 둘레로 얼마나\n"
            "흩어지는지를 잰다.".replace("**", ""),
            fontsize=11, color=MUTED, ha="left", va="top", linespacing=1.5)

    ax.set_xlim(theta - 4.2, theta + 5.4)
    ax.set_ylim(0, top * 1.35)
    ax.set_xticks([theta, mu])
    ax.set_xticklabels(["$\\theta$", "$E[\\hat\\theta]$"], fontsize=12)
    bare_axis(ax)
    ax.set_title("추정량의 표본분포 위에서 본 평균제곱오차", fontsize=13.5,
                 pad=12)
    fig.tight_layout()
    save(fig, OUT + "mse_decomposition.png")


# ==================================================================
# 2. 다트판 네 칸
# ==================================================================
def bias_variance_dartboard():
    rng = np.random.default_rng(7)
    cases = [
        ("낮은 편향 · 낮은 분산", (0.0, 0.0), 0.16, GREEN, "이상적"),
        ("낮은 편향 · 높은 분산", (0.0, 0.0), 0.52, BLUE, None),
        ("높은 편향 · 낮은 분산", (0.62, 0.42), 0.16, ORANGE, None),
        ("높은 편향 · 높은 분산", (0.62, 0.42), 0.52, RED, "최악"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.8))
    for ax, (name, center, spread, color, tag) in zip(axes.ravel(), cases):
        for r, shade in [(1.0, "#ECEFF1"), (0.72, "#CFD8DC"),
                         (0.44, "#B0BEC5"), (0.18, "#FFCDD2")]:
            ax.add_patch(Circle((0, 0), r, facecolor=shade, edgecolor="white",
                                linewidth=1.2, zorder=1))
        # 편향이 0 인 칸에서는 +와 ×가 같은 자리에 온다. +를 맨 위에 그려
        # 둘이 겹쳐 있다는 것이 보이게 한다.
        ax.plot(0, 0, "+", color=RED, markersize=15, markeredgewidth=2.2,
                zorder=8)

        # 표본잡음 때문에 점들의 평균이 의도한 자리에서 밀리면 "낮은 편향"이
        # 낮아 보이지 않는다. 도식이므로 평균을 의도한 자리에 정확히 맞춘다.
        pts = rng.normal(0.0, spread, size=(16, 2))
        pts = pts - pts.mean(axis=0) + np.asarray(center)
        ax.plot(pts[:, 0], pts[:, 1], "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

        # 다트의 평균 — 편향이 보이도록 참값과 이어 준다
        m = pts.mean(axis=0)
        ax.plot(*m, "x", color=INK, markersize=11, markeredgewidth=2.2,
                zorder=7)
        if abs(center[0]) > 0.01:
            ax.add_patch(FancyArrowPatch((0, 0), tuple(m), arrowstyle="-|>",
                                         mutation_scale=12, color=INK,
                                         linewidth=1.4, zorder=6))

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")
        title = name if tag is None else f"{name}   ({tag})"
        ax.set_title(title, fontsize=12, color=color, pad=8)

    fig.text(0.5, 0.055,
             "붉은 +가 참값 $\\theta$, 검은 ×가 추정량의 평균 $E[\\hat\\theta]$, "
             "점 하나가 표본 하나에서 얻은 추정값이다.\n"
             "두 점 사이의 거리가 편향이고, 점들이 흩어진 정도가 분산이다.",
             fontsize=11.5, color=INK, ha="center", va="center",
             linespacing=1.6)
    fig.suptitle("편향과 분산은 서로 다른 것을 잰다", fontsize=14, y=0.965)
    fig.tight_layout(rect=[0, 0.10, 1, 0.945])
    save(fig, OUT + "bias_variance_dartboard.png")


# ==================================================================
# 3. 축소추정량 — 편향을 조금 들이면 MSE 가 준다
# ==================================================================
def shrinkage_mse():
    mu, sigma, n = 1.0, 1.0, 4
    v = sigma ** 2 / n                       # Var(X̄)
    lam = np.linspace(0, 1.25, 500)
    var_term = lam ** 2 * v
    bias_term = (1 - lam) ** 2 * mu ** 2
    mse = var_term + bias_term
    lam_star = mu ** 2 / (mu ** 2 + v)
    mse_star = lam_star ** 2 * v + (1 - lam_star) ** 2 * mu ** 2

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(lam, var_term, color=GREEN, linewidth=1.9,
            label="분산  $\\lambda^2 \\sigma^2 / n$")
    ax.plot(lam, bias_term, color=ORANGE, linewidth=1.9,
            label="편향제곱  $(1-\\lambda)^2 \\mu^2$")
    ax.plot(lam, mse, color=BLUE, linewidth=2.6, label="MSE  (둘의 합)")

    ax.plot([lam_star, lam_star], [0, mse_star], color=RED, linewidth=1.5,
            linestyle="--")
    ax.plot([lam_star], [mse_star], "o", color=RED, markersize=8, zorder=6)
    # 축 눈금 0.8 과 겹치지 않도록 파선 왼쪽 안쪽에 적는다.
    ax.text(lam_star - 0.025, 0.025, f"$\\lambda^* = {lam_star:.1f}$",
            fontsize=12, color=RED, ha="right", va="bottom")

    ax.plot([1.0], [v], "o", color=INK, markersize=7, zorder=6)
    ax.annotate(f"표본평균 $\\bar X$  ($\\lambda = 1$)\n"
                f"MSE $= {v:.2f}$",
                xy=(1.0, v), xytext=(1.12, 0.42), fontsize=11, color=INK,
                ha="center", va="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2))
    ax.annotate(f"축소추정량  MSE $= {mse_star:.2f}$",
                xy=(lam_star, mse_star), xytext=(0.33, 0.115), fontsize=11,
                color=RED, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2))

    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("축소계수 $\\lambda$", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper center", ncol=3)
    ax.set_title("편향을 조금 들이면 분산이 더 많이 준다 — "
                 f"$\\mu = {mu:g}$, $\\sigma = {sigma:g}$, $n = {n}$",
                 fontsize=13, pad=12)
    fig.tight_layout()
    save(fig, OUT + "shrinkage_mse.png")


# ==================================================================
# 4. 일치성과 점근정규성
# ==================================================================
def consistency_asymptotics():
    theta, sigma = 5.0, 2.0
    ns = [10, 40, 160, 640]
    colors = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]
    eps = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) 일치성 — 분포가 참값으로 오그라든다
    ax = axes[0]
    x = np.linspace(theta - 2.6, theta + 2.6, 800)
    for n, c in zip(ns, colors):
        sd = sigma / np.sqrt(n)
        ax.plot(x, normal_pdf(x, theta, sd), color=c, linewidth=2.0,
                label=f"$n = {n}$")
    # 세로선은 그림 영역 안쪽까지만. 축 아래 설명글을 가로지르면 빼기 기호가
    # 더하기처럼 보인다.
    for sgn in (-1, 1):
        ax.plot([theta + sgn * eps, theta + sgn * eps], [0, 8.0], color=RED,
                linewidth=1.2, linestyle="--")
    ax.text(theta + eps, -0.55, "$\\theta + \\varepsilon$", fontsize=11,
            color=RED, ha="center", va="top")
    ax.text(theta - eps, -0.55, "$\\theta - \\varepsilon$", fontsize=11,
            color=RED, ha="center", va="top")
    ax.text(theta, -1.5,
            "$n$ 이 커질수록 막대 밖으로 나가는 확률이 $0$ 으로 간다",
            fontsize=11, color=RED, ha="center", va="top")
    ax.plot([theta, theta], [0, 8.4], color=INK, linewidth=1.4)
    ax.text(theta, 8.6, "참값 $\\theta$", fontsize=11.5, color=INK,
            ha="center", va="bottom")
    ax.set_ylim(-2.4, 9.6)
    ax.set_xlim(theta - 2.6, theta + 2.6)
    ax.set_xticks([theta - 2, theta, theta + 2])
    ax.set_xticklabels(["$\\theta-2$", "$\\theta$", "$\\theta+2$"],
                       fontsize=11)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("일치성 — $\\hat\\theta_n$ 의 분포", fontsize=13, pad=10)

    # (b) 점근정규성 — √n 을 곱해 놓으면 같은 곡선
    ax = axes[1]
    z = np.linspace(-3.4 * sigma, 3.4 * sigma, 800)
    base = normal_pdf(z, 0, sigma)
    # 네 곡선이 정확히 같으므로 파선의 위상만 어긋나게 준다. 이런 곡선은
    # 범례로 구분되지 않으므로 아래에 글로 적는다.
    for j, (n, c) in enumerate(zip(ns, colors)):
        ax.plot(z, base, color=c, linewidth=2.8, alpha=0.95,
                linestyle=(j * 6, (6, 18)), solid_capstyle="butt")
    ax.plot([0, 0], [0, normal_pdf(0, 0, sigma) * 1.05], color=INK,
            linewidth=1.4)
    ax.text(0, normal_pdf(0, 0, sigma) * 1.08, "$0$", fontsize=11.5,
            color=INK, ha="center", va="bottom")
    ax.text(0, -0.035,
            "네 곡선이 완전히 포개져 있다 — 오그라드는 속도가 정확히 $1/\\sqrt{n}$ 이다",
            fontsize=11, color=INK, ha="center", va="top")
    ax.set_ylim(-0.075, 0.235)
    ax.set_xlim(-3.6 * sigma, 3.6 * sigma)
    ax.set_xticks([-2 * sigma, 0, 2 * sigma])
    ax.set_xticklabels(["$-2\\sigma$", "$0$", "$2\\sigma$"], fontsize=11)
    bare_axis(ax)
    ax.text(-3.45 * sigma, 0.206, "$n = 10,\\; 40,\\; 160,\\; 640$",
            fontsize=11, color=INK, ha="left", va="center")
    ax.set_title("점근정규성 — $\\sqrt{n}\\,(\\hat\\theta_n - \\theta)$ 의 분포",
                 fontsize=13, pad=10)

    fig.suptitle("오그라드는 것과, 오그라드는 속도", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "consistency_asymptotics.png")


# ==================================================================
# 5. 휘어짐 → 정보량 → 분산의 바닥
# ==================================================================
def crlb_information():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (a) 로그가능도의 휘어짐
    ax = axes[0]
    t = np.linspace(-3, 3, 600)
    for info, color, name in [(2.0, BLUE, "정보량이 크다"),
                              (0.3, ORANGE, "정보량이 작다")]:
        ax.plot(t, -info * t ** 2 / 2, color=color, linewidth=2.2,
                label=f"{name}  ($I = {info:g}$)")
    ax.plot([0], [0], "o", color=INK, markersize=8, zorder=6)
    ax.text(0, 0.28, "$\\hat\\theta$", fontsize=12.5, color=INK, ha="center")
    ax.text(0, -4.6, "급하게 휜 곡선일수록 꼭대기의 자리가 분명하다",
            fontsize=11, color=INK, ha="center", va="top")
    ax.text(2.55, -1.1, "$I(\\theta) = -E[\\ell''(\\theta)]$", fontsize=12,
            color=INK, ha="right", va="center")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-5.4, 1.2)
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.set_xticklabels(["$\\theta$"], fontsize=12)
    ax.tick_params(axis="x", labelsize=11, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("로그가능도 $\\ell(\\theta)$ 의 휘어짐", fontsize=13, pad=10)

    # (b) 그 결과 — CRLB 와 두 추정량
    ax = axes[1]
    sigma, n = 1.0, 25
    sd_mean = sigma / np.sqrt(n)                       # CRLB 를 달성한다
    sd_med = np.sqrt(np.pi / 2) * sigma / np.sqrt(n)   # 점근분산 πσ²/2n
    x = np.linspace(-1.05, 1.05, 700)
    ax.plot(x, normal_pdf(x, 0, sd_mean), color=BLUE, linewidth=2.3,
            label="표본평균  효율 $1$")
    ax.fill_between(x, normal_pdf(x, 0, sd_mean), color=BLUE, alpha=0.14)
    ax.plot(x, normal_pdf(x, 0, sd_med), color=ORANGE, linewidth=2.3,
            label="표본중앙값  효율 $2/\\pi \\approx 0.64$")
    ax.fill_between(x, normal_pdf(x, 0, sd_med), color=ORANGE, alpha=0.14)

    peak = normal_pdf(0, 0, sd_mean)
    ax.add_patch(FancyArrowPatch((-sd_mean, peak * 0.52),
                                 (sd_mean, peak * 0.52),
                                 arrowstyle="<|-|>", mutation_scale=12,
                                 color=BLUE, linewidth=1.4, zorder=7))
    # 라벨을 곡선 바깥 왼쪽에 둔다. 가운데에 두면 글자가 곡선을 가로지른다.
    ax.text(-1.0, peak * 0.60, "파란 곡선의 폭이\nCRLB 가 허용하는\n가장 좁은 폭",
            fontsize=10.5, color=BLUE, ha="left", va="center",
            linespacing=1.45)
    ax.text(0.62, peak * 0.30,
            "어떤 불편추정량도\n파란 곡선보다\n좁아질 수 없다", fontsize=10.5,
            color=INK, ha="center", va="center", linespacing=1.45)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, peak * 1.22)
    ax.set_xticks([0])
    ax.set_xticklabels(["$\\theta$"], fontsize=12)
    bare_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title(f"두 불편추정량의 표본분포  ($n = {n}$)", fontsize=13, pad=10)

    fig.suptitle("휘어짐이 정보량이고, 정보량이 분산의 바닥을 정한다",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "crlb_information.png")


# ==================================================================
# 6. 압축 사다리와 Rao–Blackwell
# ==================================================================
def sufficiency_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) 자료에서 최소충분통계량으로
    ax = axes[0]

    def box(xy, title, sub, edge, face, w=4.6, h=1.15):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.7, zorder=4))
        ax.text(x, y + 0.2, title, fontsize=11.5, color=edge, ha="center",
                va="center", zorder=5)
        ax.text(x, y - 0.26, sub, fontsize=10, color=edge, ha="center",
                va="center", zorder=5)

    box((3.0, 5.3), "자료 전체  $\\mathbf{X} = (X_1, \\ldots, X_n)$",
        "충분하다 — 그러나 전혀 압축하지 못한다", INK, "#ECEFF1")
    box((3.0, 3.1), "$T'(\\mathbf{X}) = (X_1, \\;\\; X_2 + \\cdots + X_n)$",
        "충분하다 — 아직 덜 압축되었다", BLUE, BLUE_F)
    box((3.0, 0.9), "$T(\\mathbf{X}) = X_1 + X_2 + \\cdots + X_n$",
        "최소충분 — 더 줄일 수 없다", GREEN, GREEN_F)

    for y0, y1 in [(4.72, 3.68), (2.52, 1.48)]:
        ax.add_patch(FancyArrowPatch((3.0, y0), (3.0, y1), arrowstyle="-|>",
                                     mutation_scale=15, color=MUTED,
                                     linewidth=1.6, zorder=3))
        ax.text(3.25, (y0 + y1) / 2, "더 압축", fontsize=10, color=MUTED,
                ha="left", va="center")

    ax.text(5.6, 4.2, "$\\theta$ 에 대한 정보는\n한 단계도 줄지 않는다",
            fontsize=11, color=RED, ha="center", va="center", linespacing=1.5)
    ax.plot([5.6, 5.6], [1.55, 3.55], color=RED, linewidth=1.3,
            linestyle=":", zorder=3)
    ax.text(3.0, -0.35, "포아송 표본의 예 — 인수분해가 관측값의 합을 집어낸다",
            fontsize=11, color=INK, ha="center", va="center")

    ax.set_xlim(0.2, 7.0)
    ax.set_ylim(-0.9, 6.3)
    ax.axis("off")
    ax.set_title("압축의 사다리", fontsize=13, pad=10)

    # (b) Rao–Blackwell
    ax = axes[1]
    x = np.linspace(-3.6, 3.6, 700)
    ax.plot(x, normal_pdf(x, 0, 1.25), color=ORANGE, linewidth=2.3,
            label="$\\hat\\theta$  (아무 불편추정량)")
    ax.fill_between(x, normal_pdf(x, 0, 1.25), color=ORANGE, alpha=0.14)
    ax.plot(x, normal_pdf(x, 0, 0.72), color=GREEN, linewidth=2.3,
            label="$E[\\hat\\theta \\mid T]$  (조건을 건 뒤)")
    ax.fill_between(x, normal_pdf(x, 0, 0.72), color=GREEN, alpha=0.14)

    peak = normal_pdf(0, 0, 0.72)
    ax.plot([0, 0], [0, peak * 1.06], color=INK, linewidth=1.4)
    ax.text(0, peak * 1.09, "참값 $\\theta$", fontsize=11.5, color=INK,
            ha="center", va="bottom")
    ax.text(0, -0.055, "중심은 그대로, 폭만 줄어든다", fontsize=11,
            color=GREEN, ha="center", va="top")

    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-0.115, peak * 1.28)
    ax.set_xticks([0])
    ax.set_xticklabels(["$\\theta$"], fontsize=12)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("Rao–Blackwell — 충분통계량으로 조건을 걸면", fontsize=13,
                 pad=10)

    fig.tight_layout()
    save(fig, OUT + "sufficiency_ladder.png")


if __name__ == "__main__":
    mse_decomposition()
    bias_variance_dartboard()
    shrinkage_mse()
    consistency_asymptotics()
    crlb_information()
    sufficiency_ladder()
