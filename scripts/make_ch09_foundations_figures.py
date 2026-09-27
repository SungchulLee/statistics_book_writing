r"""9장 가설검정의 기초 여섯 쪽의 그림을 생성한다.

같은 장의 type12_error_visualization 과 rejection_region_demo 에는 이미
그림이 있다. 그 둘은 임계값 하나를 고정하고 두 오류의 넓이를 보이므로,
여기서는 겹치지 않는 것들을 그린다.

  ch09/foundations/img/hypothesis_types.png        모수축 위의 H0 와 H1
  ch09/foundations/img/test_procedure_flow.png     검정의 다섯 걸음
  ch09/foundations/img/pvalue_tail_and_uniform.png 꼬리넓이와 p-값의 분포
  ch09/foundations/img/alpha_longrun.png           alpha 는 장기 비율이다
  ch09/errors_and_power/img/alpha_beta_tradeoff.png  문턱을 옮기면
  ch09/errors_and_power/img/ci_test_duality.png      구간과 검정의 쌍대성

실행:  python3 scripts/make_ch09_foundations_figures.py   (저장소 최상위에서)
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

FOUND = "docs/ch09/foundations/img/"
ERR = "docs/ch09/errors_and_power/img/"


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
# 1. 모수축을 어떻게 가르는가
# ==================================================================
def hypothesis_types():
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.6))

    cases = [
        ("양측", "$H_0: \\mu = \\mu_0$", "$H_1: \\mu \\neq \\mu_0$",
         "both"),
        ("우측 (단측)", "$H_0: \\mu \\leq \\mu_0$", "$H_1: \\mu > \\mu_0$",
         "right"),
        ("좌측 (단측)", "$H_0: \\mu \\geq \\mu_0$", "$H_1: \\mu < \\mu_0$",
         "left"),
    ]

    for ax, (name, h0, h1, kind) in zip(axes, cases):
        ax.plot([-1, 1], [0, 0], color=MUTED, linewidth=1.4, zorder=2)
        if kind == "both":
            ax.plot([-1, -0.02], [0, 0], color=ORANGE, linewidth=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([0.02, 1], [0, 0], color=ORANGE, linewidth=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([0], [0], "o", color=BLUE, markersize=13, zorder=5)
        elif kind == "right":
            ax.plot([-1, 0], [0, 0], color=BLUE, linewidth=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([0.02, 1], [0, 0], color=ORANGE, linewidth=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([0], [0], "o", color=BLUE, markersize=9, zorder=5)
        else:
            ax.plot([0, 1], [0, 0], color=BLUE, linewidth=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([-1, -0.02], [0, 0], color=ORANGE, linewidth=6,
                    solid_capstyle="butt", zorder=3)
            ax.plot([0], [0], "o", color=BLUE, markersize=9, zorder=5)

        ax.text(0, -0.55, "$\\mu_0$", fontsize=12.5, color=INK,
                ha="center", va="top")
        ax.text(-1.12, 0, name, fontsize=12, color=INK, ha="right",
                va="center")
        ax.text(1.1, 0.42, h0, fontsize=11.5, color=BLUE, ha="left",
                va="center")
        ax.text(1.1, -0.42, h1, fontsize=11.5, color=ORANGE, ha="left",
                va="center")
        ax.set_xlim(-2.0, 2.15)
        ax.set_ylim(-1.0, 1.0)
        ax.axis("off")

    axes[0].text(0, 0.72, "한 점 — 단순가설", fontsize=10.5, color=BLUE,
                 ha="center", va="bottom")
    axes[1].text(-0.5, 0.42, "구간 — 복합가설", fontsize=10.5, color=BLUE,
                 ha="center", va="bottom")
    fig.suptitle("가설을 세운다는 것은 모수축을 둘로 가르는 일이다",
                 fontsize=13.5, y=1.0)
    fig.text(0.5, -0.02,
             "파랑이 $H_0$ 가 차지하는 부분, 주황이 $H_1$ 이 차지하는 부분이다. "
             "둘은 겹치지 않고, 합하면 모수축 전체가 된다.",
             fontsize=11, color=INK, ha="center", va="center")
    fig.tight_layout()
    save(fig, FOUND + "hypothesis_types.png")


# ==================================================================
# 2. 검정의 다섯 걸음
# ==================================================================
def test_procedure_flow():
    fig, ax = plt.subplots(figsize=(13.5, 3.9))

    steps = [
        ("1. 가설", "$H_0$ 와 $H_1$ 을\n자료를 보기 전에", BLUE, BLUE_F),
        ("2. 유의수준", "$\\alpha$ 를 정한다\n(흔히 $0.05$)", BLUE, BLUE_F),
        ("3. 검정통계량", "자료를 한 수로\n요약한다", GREEN, GREEN_F),
        ("4. 귀무분포", "$H_0$ 가 참일 때\n그 수의 분포", GREEN, GREEN_F),
        ("5. 판정", "$p < \\alpha$ 이면\n$H_0$ 를 기각", ORANGE, ORANGE_F),
    ]
    w, gap = 2.45, 0.62
    for i, (title, sub, edge, face) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, 1.7,
                                    boxstyle="round,pad=0,rounding_size=0.14",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=1.8, zorder=4))
        ax.text(x + w / 2, 2.28, title, fontsize=12, color=edge,
                ha="center", va="center", zorder=5)
        ax.text(x + w / 2, 1.62, sub, fontsize=10.5, color=edge,
                ha="center", va="center", zorder=5, linespacing=1.45)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.08, 1.85),
                                         (x + w + gap - 0.08, 1.85),
                                         arrowstyle="-|>", mutation_scale=15,
                                         color=MUTED, linewidth=1.6, zorder=3))

    ax.annotate("", xy=(1.2, 0.72), xytext=(2 * (w + gap) + 1.2, 0.72),
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.5,
                                linestyle="--"))
    ax.text((2 * (w + gap) + 2.4) / 2, 0.42,
            "1·2 는 자료를 보기 전에 정한다 — 보고 나서 고치면 $\\alpha$ 가 "
            "지켜지지 않는다",
            fontsize=11, color=RED, ha="center", va="top")

    ax.set_xlim(-0.3, 5 * w + 4 * gap + 0.3)
    ax.set_ylim(-0.25, 3.0)
    ax.axis("off")
    ax.set_title("가설검정의 다섯 걸음", fontsize=13.5, pad=8)
    fig.tight_layout()
    save(fig, FOUND + "test_procedure_flow.png")


# ==================================================================
# 3. p-값 — 꼬리넓이, 그리고 그 자체의 분포
# ==================================================================
def pvalue_tail_and_uniform():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # (a) 관측값과 꼬리넓이
    ax = axes[0]
    z_obs = 1.85
    z = np.linspace(-4, 4, 700)
    ax.plot(z, stats.norm.pdf(z), color=INK, linewidth=2.2, zorder=5)
    for side in (1, -1):
        m = (z * side) >= z_obs
        ax.fill_between(z[m], stats.norm.pdf(z[m]), color=RED, alpha=0.35,
                        zorder=4)
    ax.plot([z_obs, z_obs], [0, stats.norm.pdf(z_obs)], color=RED,
            linewidth=1.8, zorder=6)
    p_two = 2 * (1 - stats.norm.cdf(z_obs))
    ax.annotate(f"관측된 통계량\n$z = {z_obs}$", xy=(z_obs, 0.035),
                xytext=(3.05, 0.22), fontsize=11, color=RED, ha="center",
                va="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2))
    ax.text(0, 0.16, f"$p = {p_two:.3f}$\n(양쪽 꼬리넓이의 합)", fontsize=12.5,
            color=INK, ha="center", va="center", linespacing=1.5)
    ax.text(0, -0.085,
            "$p$ 는 $H_0$ 가 참일 때 이만큼 극단적인 값이 나올 확률이다",
            fontsize=10.5, color=MUTED, ha="center", va="top")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.03, 0.44)
    ax.set_xlabel("검정통계량", fontsize=11.5, color=INK, labelpad=30)
    bare_axis(ax)
    ax.set_title("$p$-값은 귀무분포의 꼬리넓이다", fontsize=13, pad=10)

    # (b) p-값 자체의 분포
    ax = axes[1]
    rng = np.random.default_rng(0)
    B, n = 40000, 25
    z0 = rng.normal(0, 1, B)                      # H0 가 참
    z1 = rng.normal(0.6 * np.sqrt(n) / 2, 1, B)   # H1 이 참 (효과 있음)
    p0 = 2 * (1 - stats.norm.cdf(np.abs(z0)))
    p1 = 2 * (1 - stats.norm.cdf(np.abs(z1)))
    bins = np.linspace(0, 1, 21)
    ax.hist(p0, bins=bins, density=True, color=BLUE_F, edgecolor=BLUE,
            linewidth=1.0, label="$H_0$ 가 참일 때")
    ax.hist(p1, bins=bins, density=True, color=ORANGE, alpha=0.45,
            edgecolor=ORANGE, linewidth=1.0, label="$H_1$ 이 참일 때")
    ax.axhline(1.0, color=INK, linewidth=1.6, linestyle="--", zorder=6)
    ax.text(0.97, 1.12, "균등분포의 높이 $1$", fontsize=10.5, color=INK,
            ha="right", va="bottom")
    ax.axvline(0.05, color=RED, linewidth=1.5, zorder=6)
    ax.text(0.075, 4.6, "$\\alpha = 0.05$", fontsize=11, color=RED,
            ha="left", va="center")
    ax.text(0.5, 3.4,
            f"$H_0$ 아래에서 $p < 0.05$ 인 비율  ${np.mean(p0 < 0.05):.3f}$\n"
            f"$H_1$ 아래에서는  ${np.mean(p1 < 0.05):.3f}$  (= 검정력)",
            fontsize=11, color=INK, ha="center", va="center", linespacing=1.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6.2)
    ax.set_xlabel("$p$-값", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper center")
    ax.set_title("$H_0$ 가 참이면 $p$-값은 균등분포를 따른다", fontsize=13,
                 pad=10)

    fig.tight_layout()
    save(fig, FOUND + "pvalue_tail_and_uniform.png")


# ==================================================================
# 4. alpha 는 한 번의 확신이 아니라 장기 비율이다
# ==================================================================
def alpha_longrun():
    rng = np.random.default_rng(11)
    m, n, alpha = 200, 25, 0.05
    zc = stats.norm.ppf(1 - alpha / 2)
    xbar = rng.normal(0, 1 / np.sqrt(n), m)        # H0 가 참
    z = xbar * np.sqrt(n)
    rej = np.abs(z) > zc

    fig, ax = plt.subplots(figsize=(13, 4.6))
    idx = np.arange(1, m + 1)
    ax.axhspan(-zc, zc, color=BLUE_F, alpha=0.6, zorder=1)
    ax.axhline(0, color=MUTED, linewidth=1.2, zorder=2)
    for s in (-zc, zc):
        ax.axhline(s, color=RED, linewidth=1.4, linestyle="--", zorder=3)
    ax.plot(idx[~rej], z[~rej], "o", color=MUTED, markersize=5, zorder=4,
            label=f"기각하지 못함  {int((~rej).sum())}번")
    ax.plot(idx[rej], z[rej], "o", color=RED, markersize=7, zorder=5,
            label=f"기각  {int(rej.sum())}번  (모두 거짓 양성)")
    for i in idx[rej]:
        ax.plot([i, i], [0, z[i - 1]], color=RED, linewidth=1.0, alpha=0.7,
                zorder=4)

    ax.text(m * 0.5, -3.55,
            f"$H_0$ 가 참인 실험을 {m}번 했다.  "
            f"기대되는 기각 횟수 ${alpha * m:.0f}$번,  실제 ${int(rej.sum())}$번.",
            fontsize=11.5, color=INK, ha="center", va="center")
    ax.text(m * 0.995, zc + 0.12, "$\\pm 1.96$", fontsize=11, color=RED,
            ha="right", va="bottom")

    ax.set_xlim(0, m + 1)
    ax.set_ylim(-4.1, 3.6)
    ax.set_xlabel("실험 번호", fontsize=11.5, color=INK)
    ax.set_ylabel("검정통계량 $z$", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left", ncol=2)
    ax.set_title("$\\alpha$ 는 한 번의 판정이 아니라 되풀이했을 때의 비율이다",
                 fontsize=13.5, pad=10)
    fig.tight_layout()
    save(fig, FOUND + "alpha_longrun.png")


# ==================================================================
# 5. 문턱을 옮기면 두 오류가 맞바뀐다
# ==================================================================
def alpha_beta_tradeoff():
    d = 2.5                      # 효과 크기
    x = np.linspace(-4, 7, 800)
    f0, f1 = stats.norm.pdf(x), stats.norm.pdf(x - d)
    cuts = [1.0, 1.645, 2.5]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) 세 문턱을 겹쳐 그린다
    ax = axes[0]
    ax.plot(x, f0, color=BLUE, linewidth=2.2, label="$H_0$ 가 참")
    ax.plot(x, f1, color=GREEN, linewidth=2.2, label="$H_1$ 이 참")
    c = cuts[1]
    ax.fill_between(x, f0, where=x >= c, color=RED, alpha=0.35)
    ax.fill_between(x, f1, where=x <= c, color=ORANGE, alpha=0.35)
    for cc, style in zip(cuts, ["-", "-", "-"]):
        ax.plot([cc, cc], [0, 0.42], color=INK if cc == c else MUTED,
                linewidth=2.0 if cc == c else 1.2, linestyle=style,
                alpha=1.0 if cc == c else 0.8)
        ax.text(cc, 0.435, f"${cc}$", fontsize=10.5,
                color=INK if cc == c else MUTED, ha="center", va="bottom")
    ax.text(2.55, 0.055, "$\\alpha$", fontsize=14, color=RED, ha="center")
    ax.text(0.95, 0.055, "$\\beta$", fontsize=14, color="#BF360C",
            ha="center")
    ax.text(3.85, 0.345, "문턱을 오른쪽으로 옮기면\n"
                         "$\\alpha$ 는 줄고 $\\beta$ 는 는다",
            fontsize=11, color=INK, ha="left", va="center", linespacing=1.5)
    ax.set_xlim(-4, 7)
    ax.set_ylim(0, 0.50)
    ax.set_xlabel("검정통계량", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title(f"문턱 하나가 두 넓이를 함께 정한다  (효과 크기 ${d}$)",
                 fontsize=12.5, pad=10)

    # (b) 문턱을 훑으면 두 오류가 그리는 곡선
    ax = axes[1]
    cs = np.linspace(-1.5, 6.0, 400)
    a = 1 - stats.norm.cdf(cs)
    b = stats.norm.cdf(cs - d)
    ax.plot(a, b, color=PURPLE, linewidth=2.6, zorder=4)
    for cc, color in zip(cuts, [MUTED, INK, MUTED]):
        aa, bb = 1 - stats.norm.cdf(cc), stats.norm.cdf(cc - d)
        ax.plot([aa], [bb], "o", color=color, markersize=9, zorder=6)
        ax.text(aa + 0.02, bb + 0.03,
                f"문턱 ${cc}$\n$\\alpha={aa:.3f}$, $\\beta={bb:.3f}$",
                fontsize=9.5, color=color, ha="left", va="bottom",
                linespacing=1.4)
    ax.set_xlim(0, 0.42)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("$\\alpha$  (제1종 오류)", fontsize=11.5, color=INK)
    ax.set_ylabel("$\\beta$  (제2종 오류)", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.text(0.22, 0.45, "곡선 위에서만 움직일 수 있다.\n"
                        "둘을 함께 줄이려면 $n$ 을 키워야 한다.",
            fontsize=10.5, color=INK, ha="center", va="center",
            linespacing=1.5)
    ax.set_title("한쪽을 줄이면 다른 쪽이 는다", fontsize=12.5, pad=10)

    fig.tight_layout()
    save(fig, ERR + "alpha_beta_tradeoff.png")


# ==================================================================
# 6. 구간과 검정은 같은 것을 말한다
# ==================================================================
def ci_test_duality():
    rng = np.random.default_rng(5)
    n, sigma, mu0 = 25, 1.0, 0.0
    m = 20
    se = sigma / np.sqrt(n)
    half = 1.96 * se
    xbar = rng.normal(0.25, se, m)              # 참 평균은 0.25
    reject = np.abs(xbar - mu0) > half

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # (a) 스무 번의 실험 — 구간이 mu0 를 품는가
    ax = axes[0]
    for i, (xb, rj) in enumerate(zip(xbar, reject), start=1):
        c = RED if rj else MUTED
        ax.plot([xb - half, xb + half], [i, i], color=c, linewidth=2.4,
                solid_capstyle="butt", zorder=4)
        ax.plot([xb], [i], "o", color=c, markersize=5, zorder=5)
    ax.axvline(mu0, color=BLUE, linewidth=2.0, zorder=6)
    ax.text(mu0, m + 1.4, "$\\mu_0 = 0$", fontsize=12, color=BLUE,
            ha="center", va="bottom")
    # 구간 위쪽 빈 공간에 적는다. 안쪽에 두면 구간선과 겹친다.
    ax.text(xbar.min() - half, m + 3.2,
            f"$\\mu_0$ 를 놓친 구간 {int(reject.sum())}개 "
            f"= 기각한 검정 {int(reject.sum())}개",
            fontsize=11, color=RED, ha="left", va="center")
    ax.set_ylim(0, m + 5)
    ax.set_xlabel("$\\mu$", fontsize=12, color=INK)
    ax.set_ylabel("실험 번호", fontsize=11.5, color=INK)
    ax.set_yticks([1, 5, 10, 15, 20])
    clean_axis(ax)
    ax.set_title("95% 신뢰구간 스무 개", fontsize=12.5, pad=10)

    # (b) 반대 방향 — 기각되지 않는 mu0 를 모으면 구간이 된다
    ax = axes[1]
    xb = xbar[0]
    grid = np.linspace(xb - 3 * half, xb + 3 * half, 400)
    zstat = np.abs(xb - grid) / se
    ok = zstat <= 1.96
    ax.plot(grid, zstat, color=INK, linewidth=2.2, zorder=5)
    ax.axhline(1.96, color=RED, linewidth=1.5, linestyle="--", zorder=6)
    ax.text(grid[-1], 2.06, "$1.96$", fontsize=11, color=RED, ha="right",
            va="bottom")
    ax.fill_between(grid, 0, 1.96, where=ok, color=GREEN_F, alpha=0.75,
                    zorder=2)
    lo, hi = grid[ok][0], grid[ok][-1]
    ax.plot([lo, hi], [-0.42, -0.42], color=GREEN, linewidth=4.0,
            solid_capstyle="butt", zorder=6)
    ax.text((lo + hi) / 2, -0.62,
            f"기각되지 않는 $\\mu_0$ 의 집합 = 신뢰구간\n"
            f"$[{lo:.3f},\\; {hi:.3f}]$",
            fontsize=10.5, color=GREEN, ha="center", va="top",
            linespacing=1.5)
    ax.plot([xb], [0], "o", color=INK, markersize=7, zorder=7)
    ax.text(xb, 0.12, "$\\bar x$", fontsize=12, color=INK, ha="center",
            va="bottom")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-1.35, 3.2)
    ax.set_xlabel("시험해 본 $\\mu_0$", fontsize=11.5, color=INK)
    ax.set_ylabel("$|z|$", fontsize=11.5, color=INK)
    ax.set_yticks([0, 1, 2, 3])
    clean_axis(ax)
    ax.set_title("첫 번째 실험 하나를 놓고 $\\mu_0$ 를 훑으면", fontsize=12.5,
                 pad=10)

    fig.suptitle("같은 계산을 두 방향에서 읽은 것이다", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, ERR + "ci_test_duality.png")


if __name__ == "__main__":
    hypothesis_types()
    test_procedure_flow()
    pvalue_tail_and_uniform()
    alpha_longrun()
    alpha_beta_tradeoff()
    ci_test_duality()
