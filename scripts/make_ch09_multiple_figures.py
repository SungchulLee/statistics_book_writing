r"""9.5 다중검정 일곱 쪽의 그림을 생성한다.

  ch09/multiple_testing/img/fwer_growth.png          검정을 늘리면 벌어지는 일
  ch09/multiple_testing/img/fwer_bound_dependence.png 합집합 한계와 상관의 영향
  ch09/multiple_testing/img/bonferroni_vs_holm.png   같은 자료, 두 문턱
  ch09/multiple_testing/img/bh_procedure.png         BH 의 계단
  ch09/multiple_testing/img/correction_comparison.png 네 방법을 같은 자료에
  ch09/multiple_testing/img/optional_stopping.png    엿보면 벌어지는 일
  ch09/multiple_testing/img/cutoff_search_inflation.png 절단점 탐색의 팽창

cutoff_search_inflation 은 타이태닉 자료가 있어야 다시 돌릴 수 있으므로
(자료는 gitignore 되어 있다) 그 쪽 예제 3 이 보고한 수치를 그대로 옮겨
그린다. 나머지는 이 스크립트가 직접 모의실험한다.

실행:  python3 scripts/make_ch09_multiple_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch09/multiple_testing/img/"
ALPHA = 0.05


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 검정을 늘리면 거짓 양성이 늘어난다
# ==================================================================
def fwer_growth():
    m = np.arange(1, 101)
    fwer = 1 - (1 - ALPHA) ** m
    fwer_bonf = 1 - (1 - ALPHA / m) ** m

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(m, fwer, color=RED, linewidth=2.6, zorder=5)
    ax.axhline(ALPHA, color=BLUE, linewidth=1.6, linestyle="--", zorder=4)
    ax.text(100, ALPHA + 0.02, "$\\alpha = 0.05$", fontsize=11, color=BLUE,
            ha="right", va="bottom")
    for k in (1, 5, 10, 20, 50):
        v = 1 - (1 - ALPHA) ** k
        ax.plot([k], [v], "o", color=RED, markersize=7, zorder=6)
        ax.text(k + 2, v, f"$m={k}$ : {v:.0%}", fontsize=10.5, color=INK,
                ha="left", va="center")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("검정 횟수 $m$", fontsize=11.5, color=INK)
    ax.set_ylabel("적어도 하나가 거짓 양성일 확률", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.set_title("보정하지 않으면  $1 - (1-\\alpha)^m$", fontsize=13, pad=10)

    ax = axes[1]
    ax.plot(m, fwer, color=RED, linewidth=2.0, linestyle=":",
            label="보정 없음")
    ax.plot(m, fwer_bonf, color=GREEN, linewidth=2.6, label="Bonferroni")
    ax.axhline(ALPHA, color=BLUE, linewidth=1.6, linestyle="--")
    ax.text(100, ALPHA + 0.012, "$\\alpha = 0.05$", fontsize=11, color=BLUE,
            ha="right", va="bottom")
    ax.text(52, 0.030,
            "문턱을 $\\alpha/m$ 으로 낮추면\n$m$ 이 아무리 커져도 $\\alpha$ "
            "아래에 머문다", fontsize=10.5, color=GREEN, ha="center",
            va="center", linespacing=1.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.10)
    ax.set_xlabel("검정 횟수 $m$", fontsize=11.5, color=INK)
    ax.set_ylabel("FWER", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("세로 눈금을 $0.1$ 까지만 늘려 본 같은 그림", fontsize=13,
                 pad=10)

    fig.suptitle("검정을 스무 번 하면 거짓 양성이 하나쯤 나오는 것이 정상이다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "fwer_growth.png")


# ==================================================================
# 2. 합집합 한계는 얼마나 느슨한가
# ==================================================================
def fwer_bound_dependence():
    m = np.arange(1, 41)
    exact = 1 - (1 - ALPHA) ** m
    bound = np.minimum(ALPHA * m, 1.0)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.plot(m, bound, color=RED, linewidth=2.4, linestyle="--",
            label="합집합 한계  $m\\alpha$")
    ax.plot(m, exact, color=BLUE, linewidth=2.8,
            label="독립일 때  $1 - (1-\\alpha)^m$")
    ax.axhline(ALPHA, color=GREEN, linewidth=2.4,
               label="완전히 종속일 때  $\\alpha$")
    ax.fill_between(m, exact, ALPHA, color=BLUE_F, alpha=0.5, zorder=1)

    ax.text(24, 0.34, "상관이 높을수록 아래쪽에 가깝다", fontsize=11,
            color=INK, ha="center", va="center")
    ax.annotate("", xy=(30, ALPHA + 0.01), xytext=(30, exact[29] - 0.01),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, linewidth=1.4))
    ax.text(30.6, (ALPHA + exact[29]) / 2, "실제 FWER 은\n이 사이에 있다",
            fontsize=10.5, color=INK, ha="left", va="center",
            linespacing=1.5)
    ax.text(12, 0.72, "$m$ 이 커지면 한계가 $1$ 을 넘어\n아무 말도 해 주지 "
                      "못한다", fontsize=10.5, color=RED, ha="center",
            va="center", linespacing=1.5)

    ax.set_xlim(1, 40)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("검정 횟수 $m$", fontsize=11.5, color=INK)
    ax.set_ylabel("FWER", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("같은 $\\alpha$, 같은 $m$ 이라도 FWER 은 하나로 정해지지 않는다",
                 fontsize=13.5, pad=10)
    fig.tight_layout()
    save(fig, OUT + "fwer_bound_dependence.png")


# ==================================================================
# 3. Bonferroni 와 Holm — 쪽의 예제 그대로
# ==================================================================
def bonferroni_vs_holm():
    p = np.array([0.003, 0.013, 0.042, 0.130])
    names = ["$H_A$", "$H_B$", "$H_C$", "$H_D$"]
    m = len(p)
    i = np.arange(1, m + 1)
    bonf = np.full(m, ALPHA / m)
    holm = ALPHA / (m - i + 1)

    # Holm 은 처음 실패에서 멈춘다
    n_holm = 0
    for k in range(m):
        if p[k] <= holm[k]:
            n_holm += 1
        else:
            break
    n_bonf = int((p <= ALPHA / m).sum())

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.step(np.append(i - 0.5, m + 0.5), np.append(holm, holm[-1]),
            where="post", color=GREEN, linewidth=2.4,
            label="Holm 의 문턱  $\\alpha/(m-i+1)$")
    ax.plot([0.5, m + 0.5], [ALPHA / m, ALPHA / m], color=RED,
            linewidth=2.4, linestyle="--",
            label="Bonferroni 의 문턱  $\\alpha/m$")

    for k in range(m):
        col = GREEN if k < n_holm else MUTED
        ax.plot([i[k], i[k]], [0, p[k]], color=col, linewidth=1.4, alpha=0.6,
                zorder=3)
        ax.plot([i[k]], [p[k]], "o", color=col, markersize=11, zorder=6)
        ax.text(i[k], p[k] + 0.006, f"${p[k]:.3f}$", fontsize=10.5,
                color=col, ha="center", va="bottom")

    ax.text(2.5, 0.105,
            f"Bonferroni 는 {n_bonf}개, Holm 은 {n_holm}개를 기각한다.\n"
            "Holm 의 문턱이 뒤로 갈수록 느슨해지기 때문이다.",
            fontsize=11.5, color=INK, ha="center", va="center",
            linespacing=1.6)

    ax.set_xticks(i)
    ax.set_xticklabels([f"{k+1}\n{names[k]}" for k in range(m)], fontsize=11)
    ax.set_xlim(0.5, m + 0.5)
    ax.set_ylim(0, 0.145)
    ax.set_xlabel("정렬한 순서 $i$", fontsize=11.5, color=INK)
    ax.set_ylabel("$p$-값", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("같은 네 개의 $p$-값, 두 가지 문턱  ($\\alpha = 0.05$)",
                 fontsize=13.5, pad=10)
    fig.tight_layout()
    save(fig, OUT + "bonferroni_vs_holm.png")


# ==================================================================
# 4. BH 절차의 계단
# ==================================================================
def bh_procedure():
    rng = np.random.default_rng(3)
    m, m1 = 50, 12                      # 가설 50개 중 12개에 실제 효과
    z = np.concatenate([rng.normal(0, 1, m - m1),
                        rng.normal(3.1, 1, m1)])
    truth = np.array([False] * (m - m1) + [True] * m1)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    order = np.argsort(p)
    p_s, truth_s = p[order], truth[order]
    i = np.arange(1, m + 1)
    bh_line = i * ALPHA / m

    below = np.where(p_s <= bh_line)[0]
    k = below.max() + 1 if len(below) else 0
    n_bonf = int((p_s <= ALPHA / m).sum())

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.plot(i, bh_line, color=GREEN, linewidth=2.4,
            label="BH 의 직선  $i\\alpha/m$")
    ax.axhline(ALPHA / m, color=RED, linewidth=2.0, linestyle="--",
               label="Bonferroni  $\\alpha/m$")
    ax.axhline(ALPHA, color=MUTED, linewidth=1.4, linestyle=":",
               label="보정 없음  $\\alpha$")

    for j in range(m):
        rejected = j < k
        col = (GREEN if (rejected and truth_s[j]) else
               RED if rejected else MUTED)
        ax.plot([i[j]], [p_s[j]], "o", color=col,
                markersize=7 if rejected else 5,
                alpha=1.0 if rejected else 0.65, zorder=5)

    ax.plot([k, k], [0, p_s[k - 1]], color=INK, linewidth=1.6,
            linestyle="--", zorder=4)
    ax.text(k + 0.6, 0.004, f"$k = {k}$", fontsize=12, color=INK,
            ha="left", va="bottom")
    n_false = int((~truth_s[:k]).sum())
    ax.text(33, 0.0425,
            f"BH: 기각 {k}개 (그중 거짓 {n_false}개)\n"
            f"Bonferroni: 기각 {n_bonf}개",
            fontsize=11.5, color=INK, ha="center", va="center",
            linespacing=1.6)

    ax.set_xlim(0, m + 1)
    ax.set_ylim(0, 0.062)
    ax.set_xlabel("정렬한 순서 $i$", fontsize=11.5, color=INK)
    ax.set_ylabel("$p_{(i)}$", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title(f"가설 {m}개 (실제 효과 {m1}개) 를 정렬해 직선과 견준다",
                 fontsize=13.5, pad=10)
    fig.text(0.5, -0.02,
             "초록은 맞게 기각한 것, 붉은색은 잘못 기각한 것, 회색은 기각하지 "
             "않은 것이다.", fontsize=11, color=INK, ha="center")
    fig.tight_layout()
    save(fig, OUT + "bh_procedure.png")


# ==================================================================
# 5. 네 방법을 같은 자료에 적용하면
# ==================================================================
def correction_comparison():
    rng = np.random.default_rng(1)
    m, m1, B = 100, 20, 3000
    alpha = ALPHA
    res = {k: {"disc": [], "false": [], "any_false": [], "fdp": []}
           for k in ["보정 없음", "Bonferroni", "Holm", "BH"]}

    for _ in range(B):
        z = np.concatenate([rng.normal(0, 1, m - m1),
                            rng.normal(3.0, 1, m1)])
        truth = np.array([False] * (m - m1) + [True] * m1)
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        order = np.argsort(p)
        ps, ts = p[order], truth[order]
        i = np.arange(1, m + 1)

        rej = {}
        rej["보정 없음"] = ps <= alpha
        rej["Bonferroni"] = ps <= alpha / m
        hol = alpha / (m - i + 1)
        first_fail = np.where(ps > hol)[0]
        cut = first_fail[0] if len(first_fail) else m
        rej["Holm"] = i <= cut
        below = np.where(ps <= i * alpha / m)[0]
        kk = below.max() + 1 if len(below) else 0
        rej["BH"] = i <= kk

        for name, r in rej.items():
            d = int(r.sum())
            f = int((r & ~ts).sum())
            res[name]["disc"].append(d)
            res[name]["false"].append(f)
            res[name]["any_false"].append(f > 0)
            res[name]["fdp"].append(f / d if d else 0.0)

    names = list(res)
    true_disc = [np.mean(np.array(res[n]["disc"]) - np.array(res[n]["false"]))
                 for n in names]
    false_disc = [np.mean(res[n]["false"]) for n in names]
    fwer = [np.mean(res[n]["any_false"]) for n in names]
    fdr = [np.mean(res[n]["fdp"]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, true_disc, width=0.55, color=GREEN_F, edgecolor=GREEN,
           linewidth=1.6, label="맞게 기각 (참 발견)")
    ax.bar(x, false_disc, bottom=true_disc, width=0.55, color="#FFCDD2",
           edgecolor=RED, linewidth=1.6, label="잘못 기각 (거짓 발견)")
    for j, n in enumerate(names):
        ax.text(j, true_disc[j] + false_disc[j] + 0.4,
                f"{true_disc[j]:.1f} + {false_disc[j]:.1f}", fontsize=10.5,
                color=INK, ha="center", va="bottom")
    ax.axhline(20, color=MUTED, linewidth=1.2, linestyle=":")
    ax.text(3.45, 20.4, "실제 효과 20개", fontsize=10.5, color=MUTED,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 27)
    ax.set_ylabel("평균 기각 수", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_title("얼마나 찾아내는가", fontsize=13, pad=10)

    ax = axes[1]
    ax.bar(x - 0.19, fwer, width=0.36, color=BLUE_F, edgecolor=BLUE,
           linewidth=1.6, label="FWER (거짓 발견이 하나라도 있을 확률)")
    ax.bar(x + 0.19, fdr, width=0.36, color=ORANGE_F, edgecolor=ORANGE,
           linewidth=1.6, label="FDR (기각 중 거짓의 비율)")
    ax.axhline(alpha, color=RED, linewidth=1.6, linestyle="--")
    # BH 막대와 겹치지 않도록 왼쪽 끝에 적는다.
    ax.text(-0.42, alpha + 0.025, "$\\alpha = 0.05$", fontsize=11, color=RED,
            ha="left", va="bottom")
    for j in range(len(names)):
        ax.text(j - 0.19, fwer[j] + 0.02, f"{fwer[j]:.2f}", fontsize=10,
                color=BLUE, ha="center", va="bottom")
        ax.text(j + 0.19, fdr[j] + 0.02, f"{fdr[j]:.2f}", fontsize=10,
                color=ORANGE, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    clean_axis(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper center")
    ax.set_title("무엇을 얼마로 지키는가", fontsize=13, pad=10)

    fig.suptitle(f"가설 {m}개 중 {m1}개에 실제 효과가 있는 자료를 "
                 f"{B//1000}천 번 되풀이", fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "correction_comparison.png")


# ==================================================================
# 6. 엿보면서 멈추면
# ==================================================================
def optional_stopping():
    rng = np.random.default_rng(7)
    step, n_max = 10, 200
    peeks = np.arange(step, n_max + 1, step)

    def run(rg):
        a = rg.normal(0, 1, n_max)
        b = rg.normal(0, 1, n_max)
        return np.array([stats.ttest_ind(a[:k], b[:k]).pvalue for k in peeks])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # (a) 몇 개의 경로
    ax = axes[0]
    shown, crossed = 0, 0
    rg = np.random.default_rng(12)
    while shown < 6:
        path = run(rg)
        hit = np.where(path < ALPHA)[0]
        col = RED if len(hit) else MUTED
        if len(hit):
            crossed += 1
            ax.plot(peeks[:hit[0] + 1], path[:hit[0] + 1], color=col,
                    linewidth=1.8, alpha=0.9, zorder=5)
            ax.plot([peeks[hit[0]]], [path[hit[0]]], "o", color=RED,
                    markersize=8, zorder=6)
            # 멈춘 뒤의 경로는 그리지 않는다. 실제로 그 자리에서 멈추므로.
        else:
            ax.plot(peeks, path, color=col, linewidth=1.4, alpha=0.55,
                    zorder=4)
        shown += 1
    ax.axhline(ALPHA, color=RED, linewidth=1.6, linestyle="--", zorder=6)
    ax.text(n_max, ALPHA + 0.015, "$\\alpha = 0.05$", fontsize=11, color=RED,
            ha="right", va="bottom")
    ax.text(n_max * 0.98, 0.955,
            f"$H_0$ 가 참인 실험 6개 가운데\n{crossed}개가 한 번이라도 선 "
            "아래로 내려간다.\n그때 멈추면 '유의한 결과'가 된다.",
            fontsize=10.5, color=INK, ha="right", va="top", linespacing=1.6,
            bbox=dict(facecolor="white", edgecolor="none", pad=3.0))
    ax.set_xlim(0, n_max + 2)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("표본크기 $n$", fontsize=11.5, color=INK)
    ax.set_ylabel("$p$-값", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.set_title("자료를 모으며 $p$-값을 들여다보면", fontsize=13, pad=10)

    # (b) 엿본 횟수에 따른 거짓 양성률
    ax = axes[1]
    B = 4000
    rg = np.random.default_rng(99)
    a = rg.normal(0, 1, (B, n_max))
    b = rg.normal(0, 1, (B, n_max))
    P = np.empty((B, len(peeks)))
    for j, k in enumerate(peeks):
        P[:, j] = stats.ttest_ind(a[:, :k], b[:, :k], axis=1).pvalue
    rate = [(P[:, :j + 1] < ALPHA).any(axis=1).mean()
            for j in range(len(peeks))]

    ax.plot(np.arange(1, len(peeks) + 1), rate, "o-", color=RED,
            linewidth=2.4, markersize=6)
    ax.axhline(ALPHA, color=BLUE, linewidth=1.6, linestyle="--")
    ax.text(len(peeks), ALPHA + 0.008, "$\\alpha = 0.05$", fontsize=11,
            color=BLUE, ha="right", va="bottom")
    for j in (0, 4, 9, 19):
        ax.text(j + 1, rate[j] + 0.012, f"{rate[j]:.2f}", fontsize=10.5,
                color=INK, ha="center", va="bottom")
    ax.set_xlim(0.5, len(peeks) + 0.5)
    ax.set_ylim(0, 0.32)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("엿본 횟수", fontsize=11.5, color=INK)
    ax.set_ylabel("한 번이라도 $p < 0.05$ 가 된 비율", fontsize=11.5,
                  color=INK)
    clean_axis(ax)
    ax.set_title("엿볼수록 거짓 양성률이 오른다", fontsize=13, pad=10)

    fig.suptitle("검정 하나하나는 정당한데 절차 전체가 정당하지 않다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "optional_stopping.png")


# ==================================================================
# 7. 절단점을 고르면 — 그 쪽 예제 3 의 수치를 옮겨 그린다
# ==================================================================
def cutoff_search_inflation():
    ks = np.array([1, 2, 3, 5, 10, 20, 64])
    observed = np.array([0.0521, 0.1111, 0.1476, 0.2129, 0.3399, 0.4324,
                         0.5211])
    grid = np.linspace(1, 64, 300)
    indep = 1 - (1 - ALPHA) ** grid

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(grid, indep, color=MUTED, linewidth=2.0, linestyle="--",
            label="검정들이 독립이라면  $1 - (1-\\alpha)^k$")
    ax.plot(ks, observed, "o-", color=RED, linewidth=2.6, markersize=9,
            label="타이태닉 절단점 탐색 (순열검정)")
    ax.axhline(ALPHA, color=BLUE, linewidth=1.6, linestyle=":")
    ax.text(64, ALPHA + 0.02, "명목 $\\alpha = 0.05$", fontsize=11,
            color=BLUE, ha="right", va="bottom")

    # k = 2, 3 은 점이 붙어 있어 라벨을 생략한다.
    for k, v in zip(ks, observed):
        if k in (2, 3):
            continue
        ax.text(k, v - 0.035, f"{v:.2f}", fontsize=10.5, color=RED,
                ha="center", va="top")

    ax.annotate("", xy=(2, observed[1]), xytext=(1, observed[0]),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6))
    ax.text(4.0, 0.075, "후보를 둘만 써도 두 배가 된다", fontsize=11,
            color=INK, ha="left", va="center")
    ax.text(34, 0.30,
            "인접한 절단점의 검정은 서로 강하게 상관되어 있다.\n"
            "그래서 독립일 때보다는 천천히 오르지만,\n"
            "오르는 것 자체는 막지 못한다.",
            fontsize=11, color=INK, ha="center", va="center",
            linespacing=1.6)

    ax.set_xlim(0, 67)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("시도한 절단점 후보 수 $k$", fontsize=11.5, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("가장 좋은 절단점을 고르면 명목 수준이 지켜지지 않는다",
                 fontsize=13.5, pad=10)
    fig.tight_layout()
    save(fig, OUT + "cutoff_search_inflation.png")


if __name__ == "__main__":
    fwer_growth()
    fwer_bound_dependence()
    bonferroni_vs_holm()
    bh_procedure()
    correction_comparison()
    optional_stopping()
    cutoff_search_inflation()
