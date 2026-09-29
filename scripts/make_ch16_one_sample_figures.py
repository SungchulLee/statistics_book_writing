r"""16.2 일표본 비모수 검정 여덟 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch16/one_sample_nonparametric/img/magnitude_discarded.png    부호검정은 크기를 못 본다
  ch16/one_sample_nonparametric/img/exact_size_sawtooth.png    정확검정의 실제 크기는 톱니다
  ch16/one_sample_nonparametric/img/symmetry_broken.png        대칭성이 깨지면 크기가 발산한다
  ch16/one_sample_nonparametric/img/runs_lag1_blindspot.png    런 검정은 지연 1 만 본다
  ch16/one_sample_nonparametric/img/information_ladder.png     세 검정이 보는 정보의 층
  ch16/one_sample_nonparametric/img/signed_rank_enumeration.png  귀무분포를 열거로 만든다
  ch16/one_sample_nonparametric/img/normal_vs_exact_tail.png   정규근사가 깎아 먹는 꼬리
  ch16/one_sample_nonparametric/img/runs_null_two_tails.png    런은 적어도 많아도 탈이다

실행:  python3 scripts/make_ch16_one_sample_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch16/one_sample_nonparametric/img/"
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


def bare(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# ==================================================================
# 1. 부호검정은 크기를 보지 못한다  (sign_test.md)
# ==================================================================
def magnitude_discarded():
    dA = np.array([10, 5, -5, 15, 2, 5, 10, -3, 15, 10], float)
    dB = np.array([10, 5, -25, 15, 2, 5, 10, -30, 15, 10], float)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.0), sharex="col")
    for col, (d, name) in enumerate([(dA, "자료 A"), (dB, "자료 B")]):
        r = stats.rankdata(np.abs(d))
        sr = np.sign(d) * r
        wplus = r[d > 0].sum()
        p_sign = stats.binomtest(int((d > 0).sum()), len(d), 0.5,
                                 alternative="greater").pvalue
        p_w = stats.wilcoxon(d, alternative="greater").pvalue
        x = np.arange(1, len(d) + 1)

        ax = axes[0, col]
        cols = [BLUE if v > 0 else RED for v in d]
        ax.vlines(x, 0, d, color=cols, lw=3.0)
        ax.plot(x, d, "o", ms=7, color="white", mec="none")
        for xi, v, c in zip(x, d, cols):
            ax.plot([xi], [v], "o", ms=7, color=c, mec="white", mew=1.0)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_ylim(-34, 20)
        clean(ax)
        ax.set_ylabel("차이 $d_i$", fontsize=10.5, color=INK)
        ax.set_title(f"{name} — 원래의 차이", fontsize=12, color=INK,
                     loc="left", pad=7)
        for xi, v in zip(x, d):
            ax.text(xi, v + (1.6 if v > 0 else -1.6), f"{v:g}", fontsize=8.5,
                    color=INK, ha="center",
                    va="bottom" if v > 0 else "top")

        ax = axes[1, col]
        ax.vlines(x, 0, sr, color=cols, lw=3.0)
        for xi, v, c in zip(x, sr, cols):
            ax.plot([xi], [v], "o", ms=7, color=c, mec="white", mew=1.0)
        ax.axhline(0, color=INK, lw=1.0)
        ax.set_ylim(-12.5, 15.5)
        clean(ax)
        ax.set_xticks(x)
        ax.set_xlabel("환자 번호", fontsize=10.5, color=INK)
        ax.set_ylabel("부호순위", fontsize=10.5, color=INK)
        ax.set_title(f"{name} — 절대차이의 부호순위", fontsize=12, color=INK,
                     loc="left", pad=7)
        ax.text(1.0, -11.9,
                f"$W^+$ = {wplus:g}      부호검정 $p$ = {p_sign:.4f}"
                f"      Wilcoxon $p$ = {p_w:.4f}",
                fontsize=10.5, color=INK, ha="left", va="bottom")
        print(name, "W+ =", wplus, "sign p = %.4f" % p_sign,
              "wilcoxon p = %.4f" % p_w)

    fig.suptitle("두 자료는 부호가 똑같아 부호검정이 구별하지 못한다",
                 fontsize=13.5, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "magnitude_discarded.png")


# ==================================================================
# 2. 정확검정의 실제 크기는 톱니다  (binomial_test.md)
# ==================================================================
def exact_size_sawtooth():
    ns = np.arange(5, 101)
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    for p0, color, lab in [(0.5, BLUE, "$p_0 = 0.5$ (부호검정)"),
                           (0.3, ORANGE, "$p_0 = 0.3$")]:
        sizes = []
        for n in ns:
            pv = np.array([stats.binomtest(k, n, p0).pvalue
                           for k in range(n + 1)])
            pmf = stats.binom.pmf(np.arange(n + 1), n, p0)
            sizes.append(pmf[pv <= 0.05].sum())
        sizes = np.array(sizes)
        ax.plot(ns, sizes, lw=1.6, color=color, label=lab)
        for n0 in (20, 50, 100):
            i = int(n0 - 5)
            ax.plot([n0], [sizes[i]], "o", ms=6, color=color, mec="white",
                    mew=1.0, zorder=4)
            print(f"p0={p0} n={n0} size={sizes[i]:.4f}")
        if p0 == 0.5:
            ax.annotate(f"{sizes[15]:.4f}", xy=(20, sizes[15]),
                        xytext=(24, 0.0175), fontsize=10, color=color,
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
        else:
            ax.annotate(f"{sizes[15]:.4f}", xy=(20, sizes[15]),
                        xytext=(24, 0.0095), fontsize=10, color=color,
                        ha="left", va="center",
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

    ax.axhline(0.05, color=RED, lw=1.3, ls=(0, (5, 3)))
    ax.text(100, 0.0513, "명목 $\\alpha = 0.05$", fontsize=10.5, color=RED,
            ha="right", va="bottom")
    ax.set_ylim(0, 0.058)
    ax.set_xlim(4, 101)
    clean(ax)
    ax.set_xlabel("표본크기 $n$", fontsize=10.5, color=INK)
    ax.set_ylabel("정확 양측검정의 실제 크기", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="lower right", frameon=False,
              bbox_to_anchor=(1.0, 0.03))
    ax.set_title("'정확'은 귀무분포가 정확하다는 뜻이지 크기가 $\\alpha$ 라는 뜻이 아니다",
                 fontsize=13, color=INK, loc="left", pad=10)
    fig.tight_layout()
    save(fig, "exact_size_sawtooth.png")


# ==================================================================
# 3. 대칭성이 깨지면 크기가 발산한다  (wilcoxon_signed_rank.md)
# ==================================================================
def symmetry_broken():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3),
                             gridspec_kw=dict(width_ratios=[1.1, 1]))

    # (가) 원자료의 분포와 Walsh 평균의 분포
    ax = axes[0]
    grid = np.linspace(-1.2, 3.2, 600)
    shift = np.log(2)
    dens_x = np.where(grid + shift > 0, np.exp(-(grid + shift)), 0.0)
    ax.plot(grid, dens_x, color=BLUE, lw=2.0,
            label="관측값 $X$ 의 분포")
    ax.fill_between(grid, 0, dens_x, color=BLUE_F, alpha=0.75)

    y = rng.exponential(1, 400_000) - shift
    w = (y[:200_000] + y[200_000:]) / 2
    ax.hist(w, bins=np.linspace(-1.2, 3.2, 160), density=True,
            histtype="step", color=ORANGE, lw=2.0,
            label="Walsh 평균 $(X_i+X_j)/2$ 의 분포")
    med_w = np.median(w)
    ax.axvline(0, color=BLUE, lw=1.3, ls=(0, (4, 3)))
    ax.axvline(med_w, color=ORANGE, lw=1.3, ls=(0, (4, 3)))
    ax.set_xlim(-1.2, 3.2)
    ax.set_ylim(0, 1.62)
    bare(ax)
    ax.set_xlabel("값", fontsize=10.5, color=INK)
    ax.annotate("관측값의 중앙값 0", xy=(0.02, 0.66), xytext=(0.36, 1.10),
                fontsize=10.5, color=BLUE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.annotate(f"Walsh 평균의 중앙값 {med_w:.3f}", xy=(med_w + 0.02, 0.38),
                xytext=(0.95, 0.80), fontsize=10.5, color=ORANGE,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0))
    ax.legend(fontsize=9.5, loc="upper right", frameon=False)
    ax.set_title("(가) 부호순위검정이 보는 것은 오른쪽 분포다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 표본크기별 실제 크기
    ax = axes[1]
    ns = [10, 20, 50, 100, 200]
    B = 3000
    rw, rs = [], []
    for n in ns:
        x = rng.exponential(1, (B, n)) - shift
        pw = np.array([stats.wilcoxon(x[i]).pvalue for i in range(B)])
        npos = (x > 0).sum(1)
        ps = 2 * np.minimum(stats.binom.cdf(np.minimum(npos, n - npos), n, 0.5),
                            0.5)
        rw.append((pw < 0.05).mean())
        rs.append((ps < 0.05).mean())
    ax.plot(ns, rw, "-o", color=ORANGE, lw=2.0, ms=6,
            label="Wilcoxon 부호순위")
    ax.plot(ns, rs, "-s", color=BLUE, lw=2.0, ms=6, label="부호검정")
    ax.axhline(0.05, color=RED, lw=1.2, ls=(0, (5, 3)))
    ax.text(203, 0.065, "명목 $\\alpha = 0.05$", fontsize=10, color=RED,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])   # 로그축 tofu 방지
    ax.minorticks_off()
    ax.set_ylim(0, 0.72)
    clean(ax)
    ax.set_xlabel("표본크기 $n$", fontsize=10.5, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.set_title("(나) 중앙값이 참으로 0 인데도 기각률이 올라간다",
                 fontsize=11.5, color=INK, loc="left", pad=8)
    for n, v in zip(ns, rw):
        ax.annotate(f"{v:.3f}", xy=(n, v), xytext=(0, 9),
                    textcoords="offset points", fontsize=9, color=ORANGE,
                    ha="center")
    print("walsh median %.4f" % med_w, "wilcoxon sizes", np.round(rw, 3),
          "sign sizes", np.round(rs, 3))

    fig.suptitle("대칭성이 깨지면 Wilcoxon 부호순위검정은 중앙값을 검정하지 않는다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "symmetry_broken.png")


# ==================================================================
# 런 개수의 정확 귀무분포 (열거)
# ==================================================================
def runs_pmf(n1, n2):
    """+가 n1개, -가 n2개인 모든 배열에서 런 개수의 정확분포."""
    N = n1 + n2
    counts = {}
    for pos in itertools.combinations(range(N), n1):
        s = np.zeros(N, dtype=int)
        s[list(pos)] = 1
        R = 1 + int(np.sum(s[1:] != s[:-1]))
        counts[R] = counts.get(R, 0) + 1
    total = sum(counts.values())
    ks = np.array(sorted(counts))
    pr = np.array([counts[k] / total for k in ks])
    return ks, pr


# ==================================================================
# 4. 런 검정은 지연 1 만 본다  (runs_test.md)
# ==================================================================
def runs_stats(s):
    s = np.asarray(s)
    N = len(s)
    R = 1 + int(np.sum(s[1:] != s[:-1]))
    n1 = int(s.sum())
    n2 = N - n1
    mu = 2 * n1 * n2 / N + 1
    sd = np.sqrt((mu - 1) * (mu - 2) / (N - 1))
    z = (R - mu) / sd
    return R, mu, z, 2 * stats.norm.sf(abs(z))


def runs_lag1_blindspot():
    seqs = [
        ("무작위 수열", np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
                              1, 0, 0, 1, 1])),
        ("완전 교대 $S_1$", np.array([1, 0] * 10)),
        ("주기 4 인 $S_2$", np.array([1, 1, 0, 0] * 5)),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 5.4))
    for ax, (name, s) in zip(axes, seqs):
        R, mu, z, p = runs_stats(s)
        N = len(s)
        for i, v in enumerate(s):
            ax.add_patch(plt.Rectangle((i, 0), 0.92, 1.0,
                                       facecolor=BLUE_F if v else ORANGE_F,
                                       edgecolor=BLUE if v else ORANGE,
                                       lw=1.3))
            ax.text(i + 0.46, 0.5, "$+$" if v else "$-$", fontsize=13,
                    color=BLUE if v else ORANGE, ha="center", va="center")
        # 런 경계 표시
        ax.set_xlim(-0.4, N + 6.6)
        ax.set_ylim(-0.45, 1.5)
        ax.axis("off")
        ax.text(-0.3, 1.34, name, fontsize=11.5, color=INK, ha="left",
                va="center")
        ax.text(N + 0.5, 0.92,
                f"런 $R$ = {R}   (기댓값 {mu:.2f})", fontsize=10.5, color=INK,
                ha="left", va="center")
        verdict = "무작위 아님" if p < 0.05 else "무작위성 기각 못 함"
        vcol = RED if p < 0.05 else GREEN
        ax.text(N + 0.5, 0.36, f"$Z$ = {z:+.2f},  $p$ = {p:.4f}",
                fontsize=10.5, color=INK, ha="left", va="center")
        ax.text(N + 0.5, -0.18, verdict, fontsize=10.5, color=vcol,
                ha="left", va="center")
        print(name, R, round(mu, 3), round(z, 3), round(p, 5))

    fig.suptitle("주기가 4 인 완벽하게 결정론적인 수열이 런 검정을 그대로 통과한다",
                 fontsize=13.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "runs_lag1_blindspot.png")


# ==================================================================
# 5. 세 검정이 보는 정보의 층  (one_sample.md)
# ==================================================================
def information_ladder():
    d = np.array([10, 5, -5, 15, 2, 5, 10, -3, 15, 10], float)
    n = len(d)
    x = np.arange(1, n + 1)
    r = stats.rankdata(np.abs(d))
    sr = np.sign(d) * r
    p_t = stats.ttest_1samp(d, 0, alternative="greater").pvalue
    p_w = stats.wilcoxon(d, alternative="greater").pvalue
    p_s = stats.binomtest(int((d > 0).sum()), n, 0.5,
                          alternative="greater").pvalue
    cols = [BLUE if v > 0 else RED for v in d]

    fig, axes = plt.subplots(3, 1, figsize=(10.6, 6.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 0.34]))

    ax = axes[0]
    ax.vlines(x, 0, d, color=cols, lw=3.2)
    for xi, v, c in zip(x, d, cols):
        ax.plot([xi], [v], "o", ms=7, color=c, mec="white", mew=1.0)
        ax.text(xi, v + (1.0 if v > 0 else -1.0), f"{v:g}", fontsize=9,
                color=INK, ha="center", va="bottom" if v > 0 else "top")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_ylim(-9, 20)
    clean(ax)
    ax.set_ylabel("차이 $d_i$", fontsize=10.5, color=INK)
    ax.set_title(f"원값을 그대로 쓴다 — 대응 $t$ 검정,  $p$ = {p_t:.4f}",
                 fontsize=12, color=INK, loc="left", pad=7)

    ax = axes[1]
    ax.vlines(x, 0, sr, color=cols, lw=3.2)
    for xi, v, c in zip(x, sr, cols):
        ax.plot([xi], [v], "o", ms=7, color=c, mec="white", mew=1.0)
        ax.text(xi, v + (0.6 if v > 0 else -0.6), f"{v:g}", fontsize=9,
                color=INK, ha="center", va="bottom" if v > 0 else "top")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_ylim(-6, 13)
    clean(ax)
    ax.set_ylabel("부호순위", fontsize=10.5, color=INK)
    ax.set_title(f"크기를 순위로 뭉갠다 — Wilcoxon 부호순위검정,  $p$ = {p_w:.4f}",
                 fontsize=12, color=INK, loc="left", pad=7)

    ax = axes[2]
    for xi, v, c in zip(x, d, cols):
        ax.text(xi, 0, "$+$" if v > 0 else "$-$", fontsize=22, color=c,
                ha="center", va="center")
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xticks(x)
    ax.tick_params(labelsize=9.5, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("환자 번호", fontsize=10.5, color=INK)
    ax.set_title(f"부호만 남긴다 — 부호검정,  $p$ = {p_s:.4f}",
                 fontsize=12, color=INK, loc="left", pad=7)
    ax.text(n + 0.7, 0, f"$n_+$ = {(d > 0).sum()},  $n_-$ = {(d < 0).sum()}",
            fontsize=10.5, color=INK, ha="left", va="center")
    ax.set_xlim(0.3, n + 3.0)

    fig.suptitle("같은 자료를 얼마나 들여다보느냐가 곧 검정의 선택이다",
                 fontsize=13.5, color=INK, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "information_ladder.png")
    print("ladder p: t %.4f  w %.4f  sign %.4f" % (p_t, p_w, p_s))


# ==================================================================
# 6. 귀무분포를 열거로 만든다  (wilcoxon_tests.md)
# ==================================================================
def signed_rank_enumeration():
    D = np.array([4, -1, 7, 3, -2, 5], float)
    n = len(D)
    r = stats.rankdata(np.abs(D))
    wobs = r[D > 0].sum()

    # 2^6 = 64 가지 부호 배정을 모두 열거한다
    vals = []
    for signs in itertools.product([0, 1], repeat=n):
        vals.append(sum(i + 1 for i, s in enumerate(signs) if s))
    vals = np.array(vals)
    ks = np.arange(0, n * (n + 1) // 2 + 1)
    pr = np.array([(vals == k).mean() for k in ks])
    p_one = (vals >= wobs).mean()
    p_two = min(1.0, 2 * min((vals >= wobs).mean(), (vals <= wobs).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5),
                             gridspec_kw=dict(width_ratios=[1, 1.35]))

    # (가) 관측자료
    ax = axes[0]
    x = np.arange(1, n + 1)
    cols = [BLUE if v > 0 else RED for v in D]
    sr = np.sign(D) * r
    ax.vlines(x, 0, sr, color=cols, lw=3.4)
    for xi, v, c, dv, rv in zip(x, sr, cols, D, r):
        ax.plot([xi], [v], "o", ms=8, color=c, mec="white", mew=1.0)
        ax.text(xi, v + (0.35 if v > 0 else -0.35), f"{v:+.0f}", fontsize=10,
                color=INK, ha="center", va="bottom" if v > 0 else "top")
        ax.text(xi, -6.0, f"{dv:+.0f}", fontsize=9.5, color=MUTED,
                ha="center", va="center")
    ax.axhline(0, color=INK, lw=1.0)
    ax.text(0.25, -6.0, "$D_i$", fontsize=10, color=MUTED, ha="left",
            va="center")
    ax.set_ylim(-7.2, 8.4)
    ax.set_xlim(0.2, n + 0.8)
    ax.set_xticks(x)
    clean(ax)
    ax.set_ylabel("부호순위", fontsize=10.5, color=INK)
    ax.set_title(f"(가) 관측자료의 부호순위:  $W^+$ = {wobs:g}",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (나) 열거로 만든 귀무분포
    ax = axes[1]
    tail = ks >= wobs
    ax.bar(ks[~tail], pr[~tail], width=0.82, color=BLUE_F, edgecolor=BLUE,
           lw=1.0)
    ax.bar(ks[tail], pr[tail], width=0.82, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.0)
    ax.axvline(wobs, color=ORANGE, lw=1.4, ls=(0, (4, 3)))
    clean(ax)
    ax.set_xlabel("$W^+$", fontsize=10.5, color=INK)
    ax.set_ylabel("확률", fontsize=10.5, color=INK)
    ax.set_ylim(0, pr.max() * 1.45)
    ax.set_xticks(np.arange(0, 22, 3))
    ax.annotate(f"관측값 $W^+$ = {wobs:g}\n$P(W^+ \\geq {wobs:g})$ = {p_one:.4f}",
                xy=(wobs + 0.4, pr[int(wobs)] + 0.004),
                xytext=(11.5, pr.max() * 1.33), fontsize=10.5, color=ORANGE,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
    ax.text(0.2, pr.max() * 1.33,
            f"$2^{{{n}}}$ = {2 ** n} 가지 부호 배정을\n모두 세어 만든 분포",
            fontsize=10.5, color=INK, ha="left", va="top")
    ax.set_title("(나) 정확 귀무분포 — 근사도 표도 필요 없다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    fig.suptitle("부호순위검정의 귀무분포는 모집단이 아니라 동전던지기에서 나온다",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "signed_rank_enumeration.png")
    print("W+ =", wobs, "one-sided %.4f" % p_one, "two-sided %.4f" % p_two,
          "scipy %.4f" % stats.wilcoxon(D, alternative="greater").pvalue)


# ==================================================================
# 7. 정규근사가 깎아 먹는 꼬리  (sign_test_code.md)
# ==================================================================
def normal_vs_exact_tail():
    n, p0 = 12, 0.5
    ks = np.arange(n + 1)
    pmf = stats.binom.pmf(ks, n, p0)
    mu, sd = n * p0, np.sqrt(n * p0 * (1 - p0))
    sobs = 10

    p_exact = stats.binom.sf(sobs - 1, n, p0)
    z_plain = (sobs / n - p0) / np.sqrt(p0 * (1 - p0) / n)
    p_plain = stats.norm.sf(z_plain)
    z_cc = (sobs - 0.5 - mu) / sd
    p_cc = stats.norm.sf(z_cc)

    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    tail = ks >= sobs
    ax.bar(ks[~tail], pmf[~tail], width=0.86, color=BLUE_F, edgecolor=BLUE,
           lw=1.1, label="정확 귀무분포 $\\mathrm{Bin}(12,\\,0.5)$")
    ax.bar(ks[tail], pmf[tail], width=0.86, color=ORANGE_F, edgecolor=ORANGE,
           lw=1.4)
    grid = np.linspace(-0.5, 12.5, 600)
    ax.plot(grid, stats.norm.pdf(grid, mu, sd), color=INK, lw=2.0,
            label="정규근사 $\\mathcal{N}(6,\\,3)$")
    gr = grid[grid >= sobs]
    ax.fill_between(gr, 0, stats.norm.pdf(gr, mu, sd), color=INK, alpha=0.22,
                    zorder=0)
    ax.axvline(sobs - 0.5, color=GREEN, lw=1.5, ls=(0, (5, 3)))

    ax.set_xlim(-0.6, 13.6)
    ax.set_ylim(0, 0.275)
    ax.set_xticks(ks)
    clean(ax)
    ax.set_xlabel("양의 부호 개수 $S$", fontsize=10.5, color=INK)
    ax.set_ylabel("확률 / 밀도", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    ax.text(9.63, 0.032, "연속성 보정선 $9.5$", rotation=90, fontsize=10.5,
            color=GREEN, ha="left", va="bottom")
    ax.text(10.5, 0.175,
            f"정확        $P(S \\geq 10)$ = {p_exact:.4f}\n"
            f"보정 없음  = {p_plain:.4f}\n"
            f"보정 있음  = {p_cc:.4f}",
            fontsize=10.5, color=INK, ha="left", va="top")
    ax.set_title("계단을 곡선으로 덮으면 꼬리의 절반이 사라진다",
                 fontsize=13, color=INK, loc="left", pad=10)
    fig.tight_layout()
    save(fig, "normal_vs_exact_tail.png")
    print("exact %.4f plain %.4f cc %.4f z %.4f" %
          (p_exact, p_plain, p_cc, z_plain))


# ==================================================================
# 8. 런은 적어도 많아도 탈이다  (runs_test_code.md)
# ==================================================================
def runs_null_two_tails():
    cases = [
        ("뭉친 수열", np.array([1] * 6 + [0] * 11)),
        ("지나치게 교대하는 수열",
         np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for ax, (name, s) in zip(axes, cases):
        N = len(s)
        n1 = int(s.sum())
        n2 = N - n1
        Robs, mu, z, p = runs_stats(s)
        ks, pr = runs_pmf(n1, n2)
        lo = pr[ks <= Robs].sum()
        hi = pr[ks >= Robs].sum()
        p_exact = min(1.0, 2 * min(lo, hi))

        tail = ks <= Robs if Robs < mu else ks >= Robs
        ax.bar(ks[~tail], pr[~tail], width=0.82, color=BLUE_F,
               edgecolor=BLUE, lw=1.0)
        ax.bar(ks[tail], pr[tail], width=0.82, color=ORANGE_F,
               edgecolor=ORANGE, lw=1.3)
        ax.axvline(mu, color=INK, lw=1.4, ls=(0, (4, 3)))
        ax.set_xlim(0.2, ks.max() + 1.6)
        ax.set_xticks(np.arange(2, ks.max() + 1, 2))
        ax.set_ylim(0, pr.max() * 1.55)
        clean(ax)
        ax.set_xlabel("런의 개수 $R$", fontsize=10.5, color=INK)
        ax.set_ylabel("확률", fontsize=10.5, color=INK)
        ax.set_title(f"{name}  ($n_+$ = {n1}, $n_-$ = {n2})",
                     fontsize=11.5, color=INK, loc="left", pad=8)
        ax.annotate(f"관측 $R$ = {Robs}", xy=(Robs, pr[ks == Robs][0] + 0.004),
                    xytext=(Robs + (2.6 if Robs < mu else -2.6),
                            pr.max() * 1.18),
                    fontsize=10.5, color=ORANGE,
                    ha="left" if Robs < mu else "right", va="center",
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
        ax.text(0.5, pr.max() * 1.48,
                f"$\\mu_R$ = {mu:.2f},  $Z$ = {z:+.2f}\n"
                f"정규근사 $p$ = {p:.4f}\n정확 $p$ = {p_exact:.4f}",
                fontsize=10.5, color=INK, ha="left", va="top")
        # 평균 표시
        ax.text(mu + 0.15, pr.max() * 0.62, "$\\mu_R$", fontsize=10.5,
                color=INK, ha="left", va="center")
        print(name, "R", Robs, "mu %.3f" % mu, "z %.3f" % z,
              "approx p %.4f" % p, "exact p %.4f" % p_exact)

    fig.suptitle("무작위성은 양쪽에서 깨진다 — 런이 너무 적어도, 너무 많아도",
                 fontsize=13.5, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "runs_null_two_tails.png")


if __name__ == "__main__":
    magnitude_discarded()
    exact_size_sawtooth()
    symmetry_broken()
    runs_lag1_blindspot()
    information_ladder()
    signed_rank_enumeration()
    normal_vs_exact_tail()
    runs_null_two_tails()
