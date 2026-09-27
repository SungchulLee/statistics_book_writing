r"""9.4 이표본 검정 여섯 쪽의 그림을 생성한다.

  ch09/two_sample_tests/img/difference_variance_adds.png  차의 퍼짐
  ch09/two_sample_tests/img/pooled_vs_welch_alpha.png     합동 t 가 깨지는 자리
  ch09/two_sample_tests/img/t_vs_mannwhitney_power.png    모양이 다르면
  ch09/two_sample_tests/img/allocation_power.png          표본을 어떻게 나눌까
  ch09/two_sample_tests/img/pooled_se_null.png            왜 합동하는가
  ch09/two_sample_tests/img/z_squared_chi2.png            z 제곱이 카이제곱이다

실행:  python3 scripts/make_ch09_twosample_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch09/two_sample_tests/img/"
ALPHA = 0.05


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
# 1. 차를 구하는데 분산은 더해진다
# ==================================================================
def difference_variance_adds():
    s1, s2, n1, n2 = 3.0, 2.0, 25, 25
    se1, se2 = s1 / np.sqrt(n1), s2 / np.sqrt(n2)
    se_d = np.sqrt(se1 ** 2 + se2 ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    x = np.linspace(-3, 3, 700)
    ax.plot(x, stats.norm.pdf(x, 0, se1), color=BLUE, linewidth=2.4,
            label=f"$\\bar X$ 의 분포   표준오차 {se1:.2f}")
    ax.plot(x, stats.norm.pdf(x, 0, se2), color=GREEN, linewidth=2.4,
            label=f"$\\bar Y$ 의 분포   표준오차 {se2:.2f}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("평균으로부터의 거리", fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title(f"두 표본평균  ($n_1 = n_2 = {n1}$)", fontsize=13, pad=10)

    ax = axes[1]
    ax.plot(x, stats.norm.pdf(x, 0, se1), color=BLUE, linewidth=1.4,
            linestyle=":", alpha=0.8)
    ax.plot(x, stats.norm.pdf(x, 0, se2), color=GREEN, linewidth=1.4,
            linestyle=":", alpha=0.8)
    ax.plot(x, stats.norm.pdf(x, 0, se_d), color=PURPLE, linewidth=2.8,
            zorder=5)
    ax.fill_between(x, stats.norm.pdf(x, 0, se_d), color=PURPLE, alpha=0.13,
                    zorder=3)
    top = stats.norm.pdf(0, 0, se_d)
    for s, c, name in [(se1, BLUE, ""), (se2, GREEN, ""), (se_d, PURPLE, "")]:
        ax.plot([-s, s], [stats.norm.pdf(s, 0, s)] * 2, color=c,
                linewidth=2.0, alpha=0.9, zorder=6)
    ax.text(0, top * 1.06,
            f"차의 표준오차  $\\sqrt{{{se1:.2f}^2 + {se2:.2f}^2}} = "
            f"{se_d:.2f}$", fontsize=12, color=PURPLE, ha="center",
            va="bottom")
    ax.text(0, top * 0.42,
            "어느 쪽이 흔들려도\n차는 흔들린다", fontsize=11, color=INK,
            ha="center", va="center", linespacing=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("$\\bar X - \\bar Y$", fontsize=12, color=INK)
    bare_axis(ax)
    ax.set_title("차의 분포는 둘 중 어느 것보다도 넓다", fontsize=13, pad=10)

    fig.suptitle("빼는데 더한다 — 두 표본이 독립이면 분산이 합쳐진다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "difference_variance_adds.png")


# ==================================================================
# 2. 합동 t 가 깨지는 자리
# ==================================================================
def pooled_vs_welch_alpha():
    rng = np.random.default_rng(2)
    B = 20000
    n1, n2 = 10, 30
    ratios = np.array([0.25, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0])
    pooled, welch = [], []

    for r in ratios:
        a = rng.normal(0, r, (B, n1))         # 작은 표본의 표준편차 = r
        b = rng.normal(0, 1.0, (B, n2))
        p_pool = stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue
        p_welch = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
        pooled.append(np.mean(p_pool < ALPHA))
        welch.append(np.mean(p_welch < ALPHA))

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(ratios, pooled, "o-", color=RED, linewidth=2.6, markersize=7,
            label="합동 $t$ (등분산 가정)")
    ax.plot(ratios, welch, "o-", color=GREEN, linewidth=2.6, markersize=7,
            label="Welch $t$")
    ax.axhline(ALPHA, color=INK, linewidth=1.6, linestyle="--")
    ax.text(0.235, ALPHA + 0.004, "명목 $\\alpha = 0.05$", fontsize=11,
            color=INK, ha="left", va="bottom")
    ax.axvline(1.0, color=MUTED, linewidth=1.2, linestyle=":")

    ax.text(0.27, 0.105,
            "작은 표본의 분산이 작으면\n합동 $t$ 는 너무 드물게 기각한다",
            fontsize=11, color=RED, ha="left", va="center", linespacing=1.5)
    ax.text(1.35, 0.043,
            "작은 표본의 분산이 크면\n너무 자주 기각한다",
            fontsize=11, color=RED, ha="left", va="top", linespacing=1.5)
    for i in (0, len(ratios) - 1):
        ax.text(ratios[i], pooled[i] + 0.006, f"{pooled[i]:.3f}",
                fontsize=10.5, color=RED, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_xlim(0.22, 4.6)
    ax.set_ylim(0, 0.27)
    ax.set_xlabel(f"$\\sigma_1 / \\sigma_2$   "
                  f"(표본크기는 $n_1 = {n1}$, $n_2 = {n2}$ 로 고정)",
                  fontsize=11.5, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper center")
    ax.set_title("표본크기가 다르고 분산도 다르면 합동 $t$ 는 수준을 지키지 못한다",
                 fontsize=13, pad=10)
    fig.tight_layout()
    save(fig, OUT + "pooled_vs_welch_alpha.png")


# ==================================================================
# 3. t 와 Mann-Whitney
# ==================================================================
def t_vs_mannwhitney_power():
    rng = np.random.default_rng(5)
    B, n = 4000, 20
    shifts = np.linspace(0, 1.6, 9)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    cases = [("정규모집단", lambda s: rng.normal(0, 1, s), 1.0),
             ("로그정규모집단", lambda s: rng.lognormal(0, 1, s),
              float(np.sqrt((np.e - 1) * np.e)))]

    for ax, (name, gen, sd) in zip(axes, cases):
        pt, pu = [], []
        for d in shifts:
            a = gen((B, n))
            b = gen((B, n)) + d * sd            # 효과는 표준편차 단위로
            pt.append(np.mean(stats.ttest_ind(a, b, axis=1).pvalue < ALPHA))
            u = [stats.mannwhitneyu(a[i], b[i]).pvalue for i in range(B)]
            pu.append(np.mean(np.array(u) < ALPHA))
        ax.plot(shifts, pt, "o-", color=BLUE, linewidth=2.4, markersize=6,
                label="이표본 $t$ 검정")
        ax.plot(shifts, pu, "o-", color=ORANGE, linewidth=2.4, markersize=6,
                label="Mann–Whitney $U$")
        ax.axhline(ALPHA, color=MUTED, linewidth=1.3, linestyle="--")
        ax.text(1.6, ALPHA + 0.015, "$\\alpha = 0.05$", fontsize=10.5,
                color=MUTED, ha="right", va="bottom")
        ax.set_xlim(0, 1.62)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("이동 (표준편차 단위)", fontsize=11.5, color=INK)
        clean_axis(ax)
        ax.set_title(name, fontsize=13, pad=10)

    axes[0].set_ylabel("검정력", fontsize=11.5, color=INK)
    axes[0].legend(fontsize=11, frameon=False, loc="lower right")
    axes[0].text(0.85, 0.30, "정규에서는 $t$ 가 조금 낫다", fontsize=10.5,
                 color=INK, ha="center", va="center")
    axes[1].text(0.85, 0.30, "꼬리가 두꺼우면\n순위검정이 크게 앞선다",
                 fontsize=10.5, color=INK, ha="center", va="center",
                 linespacing=1.5)

    fig.suptitle(f"$n_1 = n_2 = {n}$ 에서의 검정력 비교", fontsize=13.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, OUT + "t_vs_mannwhitney_power.png")


# ==================================================================
# 4. 표본을 어떻게 나눌 것인가
# ==================================================================
def allocation_power():
    rng = np.random.default_rng(3)
    B, N = 8000, 60
    n1s = np.arange(6, N - 5, 3)
    delta = 0.9

    fig, ax = plt.subplots(figsize=(11, 5.0))
    for (s1, s2), color, name in [((1.0, 1.0), BLUE, "$\\sigma_1 = \\sigma_2$"),
                                  ((3.0, 1.0), ORANGE,
                                   "$\\sigma_1 = 3\\sigma_2$")]:
        power = []
        for n1 in n1s:
            n2 = N - n1
            a = rng.normal(0, s1, (B, n1))
            b = rng.normal(delta * (s1 + s2) / 2, s2, (B, n2))
            p = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
            power.append(np.mean(p < ALPHA))
        power = np.array(power)
        ax.plot(n1s / N, power, "o-", color=color, linewidth=2.4,
                markersize=6, label=name)
        best = n1s[int(np.argmax(power))] / N
        ax.plot([best], [power.max()], "*", color=color, markersize=18,
                zorder=6)
        ax.text(best, power.max() + 0.022,
                f"최적 배분 {best:.2f} : {1-best:.2f}", fontsize=10.5,
                color=color, ha="center", va="bottom")

    ax.axvline(0.5, color=MUTED, linewidth=1.3, linestyle=":")
    ax.text(0.505, 0.985, "균등 배분", fontsize=10.5, color=MUTED,
            ha="left", va="top")
    ax.text(0.22, 0.42,
            "분산이 같으면 반씩 나누는 것이 최선이고,\n"
            "한쪽이 더 흔들리면 그쪽에 더 많이 배정해야 한다\n"
            "(최적 비는 $\\sigma_1 : \\sigma_2$ 에 비례한다)",
            fontsize=10.5, color=INK, ha="left", va="center",
            linespacing=1.6)

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.25, 1.0)
    ax.set_xlabel(f"첫 집단에 배정한 비율  (전체 $N = {N}$ 고정)",
                  fontsize=11.5, color=INK)
    ax.set_ylabel("검정력", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.set_title("같은 예산이면 어떻게 나누는 것이 가장 나은가", fontsize=13.5,
                 pad=10)
    fig.tight_layout()
    save(fig, OUT + "allocation_power.png")


# ==================================================================
# 5. 왜 귀무가설 아래에서는 합동하는가
# ==================================================================
def pooled_se_null():
    rng = np.random.default_rng(8)
    B = 40000
    n1, n2 = 40, 60
    ps = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
    a_pool, a_unpool = [], []

    for p in ps:
        x1 = rng.binomial(n1, p, B)
        x2 = rng.binomial(n2, p, B)
        p1, p2 = x1 / n1, x2 / n2
        # 합동
        pp = (x1 + x2) / (n1 + n2)
        se_p = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        # 비합동
        se_u = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_p = np.where(se_p > 0, (p1 - p2) / se_p, 0.0)
            z_u = np.where(se_u > 0, (p1 - p2) / se_u, 0.0)
        a_pool.append(np.mean(np.abs(z_p) > 1.96))
        a_unpool.append(np.mean(np.abs(z_u) > 1.96))

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    xs = np.arange(len(ps))
    ax.bar(xs - 0.19, a_pool, width=0.36, color=GREEN_F, edgecolor=GREEN,
           linewidth=1.6, label="합동 표준오차 (교재의 방식)")
    ax.bar(xs + 0.19, a_unpool, width=0.36, color=ORANGE_F,
           edgecolor=ORANGE, linewidth=1.6, label="비합동 표준오차")
    ax.axhline(ALPHA, color=RED, linewidth=1.6, linestyle="--")
    ax.text(len(ps) - 0.45, ALPHA + 0.004, "명목 $\\alpha = 0.05$",
            fontsize=11, color=RED, ha="right", va="bottom")
    for j in range(len(ps)):
        ax.text(j - 0.19, a_pool[j] + 0.002, f"{a_pool[j]:.3f}", fontsize=9.5,
                color=GREEN, ha="center", va="bottom")
        ax.text(j + 0.19, a_unpool[j] + 0.002, f"{a_unpool[j]:.3f}",
                fontsize=9.5, color=ORANGE, ha="center", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"$p = {p}$" for p in ps], fontsize=11)
    ax.set_ylim(0, 0.105)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title(f"$H_0: p_1 = p_2$ 가 참일 때  "
                 f"($n_1 = {n1}$, $n_2 = {n2}$, 4만 번)", fontsize=13,
                 pad=10)
    fig.text(0.5, -0.02,
             "귀무가설이 참이면 두 비율이 같으므로, 자료를 합쳐 추정한 "
             "표준오차가 더 정확하다.", fontsize=11, color=INK, ha="center")
    fig.tight_layout()
    save(fig, OUT + "pooled_se_null.png")


# ==================================================================
# 6. z 를 제곱하면 카이제곱이다
# ==================================================================
def z_squared_chi2():
    rng = np.random.default_rng(1)
    # 오른쪽 칸에서 z^2 을 chi2_1 과 견주므로 H0 가 참인 표를 만든다.
    # (왼쪽 칸의 항등식은 어떤 표에서나 성립한다.)
    m = 3000
    n1, n2 = 60, 80
    x1 = rng.binomial(n1, 0.40, m)
    x2 = rng.binomial(n2, 0.40, m)
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    ok = se > 0
    z = np.zeros(m)
    z[ok] = (p1[ok] - p2[ok]) / se[ok]

    chi = np.empty(m)
    for i in range(m):
        tab = np.array([[x1[i], n1 - x1[i]], [x2[i], n2 - x2[i]]])
        chi[i] = stats.chi2_contingency(tab, correction=False)[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    ax = axes[0]
    lim = max(z ** 2) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.6, linestyle="--",
            zorder=3)
    ax.plot(z ** 2, chi, "o", color=PURPLE, markersize=4, alpha=0.35,
            zorder=5)
    ax.text(lim * 0.52, lim * 0.30,
            f"두 값이 모든 표에서 같다\n(최대 차이 {np.max(np.abs(z**2-chi)):.2e})",
            fontsize=11, color=INK, ha="center", va="center",
            linespacing=1.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("$z^2$  (이표본 비율 검정)", fontsize=11.5, color=INK)
    ax.set_ylabel("$\\chi^2$  ($2\\times2$ 독립성 검정)", fontsize=11.5,
                  color=INK)
    clean_axis(ax)
    ax.set_title(f"같은 표 {m//1000}천 개를 두 방법으로", fontsize=13,
                 pad=10)

    ax = axes[1]
    x = np.linspace(0, 10, 600)
    ax.plot(x, stats.chi2.pdf(x, 1), color=PURPLE, linewidth=2.6,
            label="$\\chi^2_1$", zorder=5)
    ax.hist(z ** 2, bins=np.linspace(0, 10, 45), density=True,
            color=BLUE_F, edgecolor=BLUE, linewidth=0.8, label="$z^2$ 의 분포")
    ax.axvline(3.841, color=RED, linewidth=1.6, linestyle="--")
    ax.text(3.95, 0.55, "$\\chi^2_{0.95,1} = 3.841 = 1.96^2$", fontsize=11,
            color=RED, ha="left", va="center")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("$z^2$", fontsize=12, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.set_title("임계값까지 정확히 대응한다", fontsize=13, pad=10)

    fig.suptitle("$2\\times2$ 표에서 비율 검정과 카이제곱 검정은 같은 검정이다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "z_squared_chi2.png")


if __name__ == "__main__":
    difference_variance_adds()
    pooled_vs_welch_alpha()
    t_vs_mannwhitney_power()
    allocation_power()
    pooled_se_null()
    z_squared_chi2()
