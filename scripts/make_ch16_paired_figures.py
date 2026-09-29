r"""16.4 대응표본 비모수 검정 네 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch16/paired_sample_nonparametric/img/permutation_floor.png    소표본 순열검정의 바닥
  ch16/paired_sample_nonparametric/img/sign_power_floor.png     부호검정에 필요한 쌍의 수
  ch16/paired_sample_nonparametric/img/midranks_and_ties.png    동점 절대차이와 분산 보정
  ch16/paired_sample_nonparametric/img/wplus_vs_min.png         min 을 쓰면 방향이 사라진다

실행:  python3 scripts/make_ch16_paired_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import itertools
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/paired_sample_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9.5, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 소표본 순열검정의 바닥  (paired_permutation.md)
# ==================================================================
def permutation_floor():
    D = np.array([6, -3, 7, 3, 4], float)
    absD = np.abs(D)
    Tobs = D.sum()
    vals = np.array([np.dot(s, absD)
                     for s in itertools.product([-1, 1], repeat=len(D))])
    vals = np.sort(vals)
    p = (np.abs(vals) >= abs(Tobs)).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))

    # (가) 32가지 배정의 T 값
    ax = axes[0]
    extreme = np.abs(vals) >= abs(Tobs)
    x = np.arange(len(vals))
    ax.bar(x[~extreme], vals[~extreme], width=0.78, color=BLUE_F,
           edgecolor=BLUE, lw=1.0)
    ax.bar(x[extreme], vals[extreme], width=0.78, color=ORANGE_F,
           edgecolor=ORANGE, lw=1.4)
    ax.axhline(Tobs, color=RED, lw=1.3, ls=(0, (5, 3)))
    ax.axhline(-Tobs, color=RED, lw=1.3, ls=(0, (5, 3)))
    ax.axhline(0, color=INK, lw=1.0)
    ax.text(-0.4, Tobs + 1.2, f"관측값 $T$ = {Tobs:.0f}", fontsize=10.5,
            color=RED, ha="left", va="bottom")
    ax.text(len(vals) - 0.4, -Tobs - 1.2, f"$-T$ = {-Tobs:.0f}",
            fontsize=10.5, color=RED, ha="right", va="top")
    ax.set_xlim(-1.2, len(vals) + 0.2)
    ax.set_ylim(-29, 30)
    clean(ax)
    ax.set_xticks([])
    ax.set_xlabel("$2^5 = 32$ 가지 부호 배정 (값 순으로 정렬)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("$T(\\mathbf{s}) = \\sum s_i |D_i|$", fontsize=10.5,
                  color=INK)
    ax.text(0.5, 26.5,
            f"$|T| \\geq {abs(Tobs):.0f}$ 인 배정 {int(extreme.sum())} 가지"
            f"     $p = {int(extreme.sum())}/32$ = {p:.4f}",
            fontsize=11, color=INK, ha="left", va="top")
    ax.set_title("(가) 부호를 모두 뒤집어 보며 만든 귀무분포",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) n 별로 도달 가능한 최소 양측 p값
    ax = axes[1]
    ns = np.arange(3, 13)
    floor = 2.0 / 2.0 ** ns
    ok = floor <= 0.05
    ax.bar(ns[~ok], floor[~ok], width=0.68, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.3)
    ax.bar(ns[ok], floor[ok], width=0.68, color=GREEN_F, edgecolor=GREEN,
           lw=1.3)
    ax.axhline(0.05, color=RED, lw=1.4, ls=(0, (5, 3)))
    ax.text(12.4, 0.056, "$\\alpha = 0.05$", fontsize=10.5, color=RED,
            ha="right", va="bottom")
    for n, f in zip(ns[:5], floor[:5]):
        ax.text(n, f + 0.006, f"{f:.3f}", fontsize=9.5, color=INK,
                ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_yticks([0.001, 0.01, 0.05, 0.1, 0.25])
    ax.set_yticklabels(["0.001", "0.01", "0.05", "0.1", "0.25"])
    ax.minorticks_off()
    ax.set_ylim(0.0004, 0.6)
    ax.set_xticks(ns)
    clean(ax)
    ax.set_xlabel("쌍의 개수 $n$", fontsize=10.5, color=INK)
    ax.set_ylabel("도달 가능한 최소 양측 $p$값  $2/2^n$", fontsize=10.5,
                  color=INK)
    ax.annotate("$n \\leq 5$ 에서는 어떤 자료를 얻어도\n$0.05$ 아래로 내려갈 수 없다",
                xy=(5, floor[2]), xytext=(6.4, 0.30), fontsize=10.5,
                color=ORANGE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.set_title("(나) 순열검정이 넘지 못하는 바닥",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    fig.suptitle("순열검정의 $p$값은 $2^n$ 분의 몇이라서 $n$ 이 작으면 바닥이 생긴다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "permutation_floor.png")
    print("T=%.0f  extreme=%d  p=%.4f" % (Tobs, extreme.sum(), p))
    print("floors", dict(zip(ns.tolist(), np.round(floor, 5).tolist())))


# ==================================================================
# 2. 부호검정에 필요한 쌍의 수  (paired_sign.md)
# ==================================================================
def sign_power_floor():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1.1]))

    # (가) n = 9 의 정확 귀무분포
    ax = axes[0]
    n, nplus = 9, 7
    ks = np.arange(n + 1)
    pmf = stats.binom.pmf(ks, n, 0.5)
    crit = int(min(k for k in ks if stats.binom.sf(k - 1, n, 0.5) <= 0.05))
    tail = ks >= nplus
    ax.bar(ks[~tail], pmf[~tail], width=0.84, color=BLUE_F, edgecolor=BLUE,
           lw=1.1)
    ax.bar(ks[tail], pmf[tail], width=0.84, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.4)
    p_obs = stats.binom.sf(nplus - 1, n, 0.5)
    p_crit = stats.binom.sf(crit - 1, n, 0.5)
    ax.axvline(nplus - 0.5, color=ORANGE, lw=1.3, ls=(0, (4, 3)))
    ax.axvline(crit - 0.5, color=GREEN, lw=1.6, ls=(0, (5, 3)))
    ax.set_xticks(ks)
    ax.set_ylim(0, 0.345)
    clean(ax)
    ax.set_xlabel("양의 차이 개수 $n_+$", fontsize=10.5, color=INK)
    ax.set_ylabel("확률", fontsize=10.5, color=INK)
    ax.text(-0.4, 0.335,
            f"관측 $n_+$ = {nplus}:  $P(S \\geq {nplus})$ = {p_obs:.4f}\n"
            f"기각에 필요한 값 $n_+ \\geq {crit}$:  $P$ = {p_crit:.4f}",
            fontsize=10.5, color=INK, ha="left", va="top")
    ax.text(crit - 0.4, 0.14, "기각역의 경계", fontsize=10, color=GREEN,
            ha="left", va="center", rotation=90)
    ax.set_title("(가) $n = 9$ 에서는 9쌍 중 8쌍이 같은 방향이어야 한다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 검정력 곡선
    ax = axes[1]
    ns = np.arange(5, 61)
    for ptrue, color, lab in [(0.95, GREEN, "$p = 0.95$"),
                              (0.85, BLUE, "$p = 0.85$"),
                              (0.75, ORANGE, "$p = 0.75$"),
                              (0.65, PURPLE, "$p = 0.65$")]:
        pw = []
        for nn in ns:
            c = min(k for k in range(nn + 1)
                    if stats.binom.sf(k - 1, nn, 0.5) <= 0.05)
            pw.append(stats.binom.sf(c - 1, nn, ptrue))
        ax.plot(ns, pw, lw=2.0, color=color, label=lab)
        if ptrue == 0.75:
            need = ns[np.array(pw) >= 0.80][0]
            print("p=0.75 needs n =", need)
    ax.axhline(0.80, color=MUTED, lw=1.2, ls=(0, (5, 3)))
    ax.text(60, 0.815, "검정력 0.80", fontsize=10, color=MUTED, ha="right",
            va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(4, 61)
    clean(ax)
    ax.set_xlabel("0 이 아닌 쌍의 개수 $n$", fontsize=10.5, color=INK)
    ax.set_ylabel("단측 부호검정의 검정력", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="lower right", frameon=False,
              title="참인 $P(D > 0)$", title_fontsize=10)
    ax.set_title("(나) 톱니가 보이는 것은 $n_+$ 가 정수이기 때문이다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    fig.suptitle("9쌍 중 7쌍이 개선되어도 유의하지 않은 것은 자료가 아니라 설계의 문제다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "sign_power_floor.png")
    print("n=9 crit=%d p_obs=%.4f p_crit=%.4f" % (crit, p_obs, p_crit))


# ==================================================================
# 3. 동점 절대차이와 분산 보정  (paired_wilcoxon.md)
# ==================================================================
def midranks_and_ties():
    D = np.array([7, 3, -2, 7, 5, 5, 9, -8, 11, 11, 12, 13], float)
    n = len(D)
    absD = np.abs(D)
    order = np.argsort(absD, kind="stable")
    r = stats.rankdata(absD)
    wplus = r[D > 0].sum()
    wminus = r[D < 0].sum()

    var0 = n * (n + 1) * (2 * n + 1) / 24
    _, counts = np.unique(absD, return_counts=True)
    corr = sum(t ** 3 - t for t in counts) / 48
    var1 = var0 - corr
    mu = n * (n + 1) / 4
    z0 = (wplus - mu) / np.sqrt(var0)
    z1 = (wplus - mu) / np.sqrt(var1)
    p0 = stats.norm.sf(z0)
    p1 = stats.norm.sf(z1)
    p_ex = stats.wilcoxon(D, alternative="greater", method="exact").pvalue

    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    for pos, idx in enumerate(order):
        c = BLUE if D[idx] > 0 else RED
        cf = BLUE_F if D[idx] > 0 else "#FBDCDC"
        ax.add_patch(plt.Rectangle((pos, 1.05), 0.9, 0.85, facecolor=cf,
                                   edgecolor=c, lw=1.4))
        ax.text(pos + 0.45, 1.47, f"{absD[idx]:.0f}", fontsize=12, color=c,
                ha="center", va="center")
        ax.text(pos + 0.45, 0.72, f"{pos + 1}", fontsize=10, color=MUTED,
                ha="center", va="center")
        ax.text(pos + 0.45, 0.25, f"{r[idx]:g}", fontsize=11.5, color=INK,
                ha="center", va="center")
        ax.text(pos + 0.45, -0.22, f"{D[idx]:+.0f}", fontsize=10, color=c,
                ha="center", va="center")
    ax.text(-0.3, 1.47, "절대차이 $|D|$", fontsize=10.5, color=INK,
            ha="right", va="center")
    ax.text(-0.3, 0.72, "자리", fontsize=10.5, color=INK, ha="right",
            va="center")
    ax.text(-0.3, 0.25, "중간순위", fontsize=10.5, color=INK, ha="right",
            va="center")
    ax.text(-0.3, -0.22, "부호 붙은 $D$", fontsize=10.5, color=INK,
            ha="right", va="center")

    # 동점 집단을 묶는 괄호
    sortedAbs = absD[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sortedAbs[j + 1] == sortedAbs[i]:
            j += 1
        if j > i:
            ax.plot([i + 0.08, i + 0.08, j + 0.82, j + 0.82],
                    [2.04, 2.22, 2.22, 2.04], color=PURPLE, lw=1.6)
            ax.text((i + j) / 2 + 0.45, 2.34, f"동점 {j - i + 1}개",
                    fontsize=10, color=PURPLE, ha="center", va="bottom")
        i = j + 1

    ax.set_xlim(-4.6, n + 0.4)
    ax.set_ylim(-0.95, 3.35)
    ax.axis("off")
    ax.text(-4.6, 3.05,
            f"$W^+$ = {wplus:g},   $W^-$ = {wminus:g},   합 = {wplus + wminus:g}"
            f" $= n(n+1)/2$",
            fontsize=11.5, color=INK, ha="left", va="center")
    ax.text(-4.6, -0.72,
            f"보정 전 분산 {var0:.3f} → 보정항 {corr:.3f} → 보정 후 {var1:.3f}"
            f"        단측 $p$:  보정 전 {p0:.5f},  보정 후 {p1:.5f},"
            f"  정확값 {p_ex:.5f}",
            fontsize=11, color=INK, ha="left", va="center")
    ax.set_title("부호순위검정은 값이 아니라 자리를 쓴다 — 동점은 자리를 나눠 갖는다",
                 fontsize=13.5, color=INK, loc="left", pad=14)
    fig.tight_layout()
    save(fig, "midranks_and_ties.png")
    print("W+ %g W- %g var0 %.4f corr %.4f var1 %.4f z0 %.4f z1 %.4f "
          "p0 %.5f p1 %.5f exact %.5f"
          % (wplus, wminus, var0, corr, var1, z0, z1, p0, p1, p_ex))


# ==================================================================
# 4. min 을 쓰면 방향이 사라진다  (paired.md)
# ==================================================================
def wplus_vs_min():
    rng = np.random.default_rng(11)
    n, B = 15, 60_000
    signs = rng.integers(0, 2, (B, n))
    ranks = np.arange(1, n + 1)
    wplus = (signs * ranks).sum(1).astype(float)
    wminus = n * (n + 1) / 2 - wplus
    wmin = np.minimum(wplus, wminus)
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z_plus = (wplus - mu) / sd
    z_min = (wmin - mu) / sd

    rej_r_plus = (z_plus > 1.645).mean()
    rej_r_min = (z_min > 1.645).mean()
    rej_l_min = (z_min < -1.645).mean()
    rej_two_min = (np.abs(z_min) > 1.96).mean()
    rej_two_plus = (np.abs(z_plus) > 1.96).mean()

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    bins = (np.arange(-0.5, n * (n + 1) / 2 + 1.5, 1.0) - mu) / sd
    ax.hist(z_plus, bins=bins, density=True, color=BLUE_F, edgecolor=BLUE,
            lw=0.7, label="$Z$ 를 $W^+$ 로 계산")
    ax.hist(z_min, bins=bins, density=True, histtype="step", color=ORANGE,
            lw=2.2, label="$Z$ 를 $\\min(W^+, W^-)$ 로 계산")
    ax.axvline(1.645, color=RED, lw=1.4, ls=(0, (5, 3)))
    ax.axvline(0, color=MUTED, lw=1.0)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(0, 1.04)
    clean(ax)
    ax.set_xlabel("$Z$ 통계량", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax.annotate("단측 기각역 $Z > 1.645$", xy=(1.72, 0.10), xytext=(2.30, 0.33),
                fontsize=10.5, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.text(4.15, 0.70,
            f"오른쪽 단측 기각률\n   $W^+$ 기준  {rej_r_plus:.4f}\n"
            f"   $\\min$ 기준  {rej_r_min:.4f}",
            fontsize=10.5, color=INK, ha="right", va="top")
    ax.text(0.15, 0.97, "$\\min$ 은 $\\mu$ 를 넘지 못한다",
            fontsize=10.5, color=ORANGE, ha="left", va="top")
    ax.set_title("$\\min(W^+, W^-)$ 로 표준화하면 분포가 반으로 접혀 방향이 사라진다",
                 fontsize=13, color=INK, loc="left", pad=12)
    fig.tight_layout()
    save(fig, "wplus_vs_min.png")
    print("right-tail: W+ %.4f  min %.4f | left-tail min %.4f"
          % (rej_r_plus, rej_r_min, rej_l_min))
    print("two-sided: W+ %.4f  min %.4f" % (rej_two_plus, rej_two_min))


if __name__ == "__main__":
    permutation_floor()
    sign_power_floor()
    midranks_and_ties()
    wplus_vs_min()
