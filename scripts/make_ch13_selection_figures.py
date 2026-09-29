r"""13장 '모형선택' 여덟 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/model_selection/img/criterion_tradeoff.png   적합항과 벌점항의 저울질
  ch13/model_selection/img/aic_decomposition.png    AIC 의 두 조각과 가중치
  ch13/model_selection/img/bic_consistency.png      BIC 의 일치성
  ch13/model_selection/img/cv_k_bias_variance.png   K 를 고르는 절충
  ch13/model_selection/img/cv_method_comparison.png 검증집합 대 LOOCV 대 10겹
  ch13/model_selection/img/forward_selection_path.png  전진선택의 경로
  ch13/model_selection/img/subset_cost.png          전수탐색의 비용과 탐욕의 대가
  ch13/model_selection/img/three_methods_agree.png  세 선택법의 일치와 불일치

실행:  python3 scripts/make_ch13_selection_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os
from itertools import combinations

import numpy as np

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

OUT = "docs/ch13/model_selection/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def ols_rss(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return ((y - X @ b) ** 2).sum()


def fit_term(n, rss):
    """가우스 오차 아래에서 -2 ln L 의 모형 의존 부분."""
    return n * np.log(rss / n)


# === 1. 적합항과 벌점항의 저울질 ===
def fig_tradeoff():
    rng = np.random.default_rng(2)
    n, P = 100, 14
    X = rng.normal(0, 1, (n, P))
    beta = np.zeros(P)
    beta[:3] = [2.5, -1.8, 1.2]
    beta[3] = 0.45                      # 네 번째는 아슬아슬하게 약하다
    y = X @ beta + rng.normal(0, 2.0, n)

    ks, fits, pen_a, pen_b = [], [], [], []
    for p in range(0, P + 1):
        A = np.column_stack([np.ones(n), X[:, :p]])
        rss = ols_rss(A, y)
        k = p + 2                       # 계수 p+1 개와 sigma^2
        ks.append(p)
        fits.append(fit_term(n, rss))
        pen_a.append(2 * k)
        pen_b.append(k * np.log(n))
    fits = np.array(fits)
    pen_a, pen_b = np.array(pen_a), np.array(pen_b)
    aic, bic = fits + pen_a, fits + pen_b

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.plot(ks, fits, color=BLUE, lw=2.6, marker="o", ms=5,
            label=r"적합항  $n\ln(\mathrm{RSS}/n)$")
    ax.plot(ks, pen_a, color=ORANGE, lw=2.6, marker="s", ms=5,
            label=r"AIC 벌점  $2k$")
    ax.plot(ks, pen_b, color=GREEN, lw=2.6, marker="^", ms=5,
            label=r"BIC 벌점  $k\ln n$")
    ax.axvline(3, color=MUTED, lw=1.4, ls=":")
    ax.text(3.25, 146, "뚜렷한 변수 3개", fontsize=9.6, color=INK)
    ax.set_xlabel("모형에 넣은 설명변수 개수 p", fontsize=10.5, color=INK)
    ax.set_ylabel("값", fontsize=10.5, color=INK)
    ax.set_title(r"$n = 100$ — 하나는 내려가고 둘은 올라간다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="center right", frameon=False)
    clean(ax)

    ax = axes[1]
    ax.plot(ks, aic, color=ORANGE, lw=2.8, marker="s", ms=6, label="AIC")
    ax.plot(ks, bic, color=GREEN, lw=2.8, marker="^", ms=6, label="BIC")
    ia, ib = int(np.argmin(aic)), int(np.argmin(bic))
    ax.scatter([ks[ia]], [aic[ia]], s=130, facecolor="none", edgecolor=ORANGE,
               lw=2.4, zorder=6)
    ax.scatter([ks[ib]], [bic[ib]], s=130, facecolor="none", edgecolor=GREEN,
               lw=2.4, zorder=6)
    ax.axvline(3, color=MUTED, lw=1.4, ls=":")
    ax.annotate(f"AIC 최소 p = {ks[ia]}", xy=(ks[ia], aic[ia]),
                xytext=(7.4, 148), fontsize=9.8, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.annotate(f"BIC 최소 p = {ks[ib]}", xy=(ks[ib], bic[ib]),
                xytext=(7.4, 232), fontsize=9.8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.set_ylim(138, 332)
    ax.text(0.04, 0.80,
            r"모수 하나당 벌점:  AIC $2$,   BIC $\ln 100 = 4.61$",
            transform=ax.transAxes, fontsize=9.8, color=INK)
    ax.set_xlabel("모형에 넣은 설명변수 개수 p", fontsize=10.5, color=INK)
    ax.set_ylabel("기준값", fontsize=10.5, color=INK)
    ax.set_title("두 U자의 바닥이 다른 곳에 있다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=10, loc="upper center", frameon=False, ncol=2)
    clean(ax)

    fig.tight_layout()
    save(fig, "criterion_tradeoff.png")
    print(f"  AIC 최소 p={ks[ia]} ({aic[ia]:.2f}), BIC 최소 p={ks[ib]} "
          f"({bic[ib]:.2f}), ln(100)={np.log(100):.3f}")
    for p in (2, 3, 4, 6, 10, 14):
        print(f"   p={p:2d}  적합항={fits[p]:8.2f}  AIC={aic[p]:8.2f}  "
              f"BIC={bic[p]:8.2f}")


# === 2. AIC 의 두 조각과 가중치 ===
def fig_aic_decomp():
    n = 50
    names = ["A", "B", "C"]
    p_ = [1, 3, 6]
    k_ = [3, 5, 8]
    sse = [120.0, 90.0, 85.0]
    fit = [n * np.log(s / n) for s in sse]
    pen = [2 * k for k in k_]
    aic = [f + q for f, q in zip(fit, pen)]
    d = [a - min(aic) for a in aic]
    w = np.exp(-np.array(d) / 2)
    w = w / w.sum()
    cols = [MUTED, GREEN, PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    idx = np.arange(3)
    ax.bar(idx, fit, 0.5, color=BLUE, label=r"적합항  $n\ln(\mathrm{SSE}/n)$")
    ax.bar(idx, pen, 0.5, bottom=fit, color=ORANGE, label=r"벌점  $2k$")
    for i in range(3):
        ax.text(i, aic[i] + 1.6, f"AIC = {aic[i]:.1f}", ha="center",
                fontsize=11, color=INK)
        ax.text(i, fit[i] / 2, f"{fit[i]:.1f}", ha="center", va="center",
                fontsize=10, color="white")
        ax.text(i, fit[i] + pen[i] / 2, f"+{pen[i]}", ha="center",
                va="center", fontsize=10, color="white")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"모형 {m}\n$p$={p}, $k$={k}, SSE={s:.0f}"
                        for m, p, k, s in zip(names, p_, k_, sse)],
                       fontsize=10)
    ax.set_ylim(0, 62)
    ax.set_ylabel("AIC", fontsize=10.5, color=INK)
    ax.set_title(r"$n = 50$ — 적합을 사는 값과 그 값어치", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper right", frameon=False)
    clean(ax)

    ax = axes[1]
    ax.bar(idx, w, 0.5, color=cols)
    for i in range(3):
        ax.text(i, w[i] + 0.022, f"{w[i]:.3f}", ha="center", fontsize=12,
                color=INK)
        ax.text(i, w[i] + 0.075, r"$\Delta$ = " + f"{d[i]:.1f}", ha="center",
                fontsize=10, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"모형 {m}" for m in names], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Akaike 가중치", fontsize=10.5, color=INK)
    ax.set_title("상대적 근거를 확률처럼 읽는다", fontsize=11.5, color=INK,
                 pad=6)
    ax.text(0.03, 0.86,
            r"$w_j = \dfrac{\exp(-\Delta_j/2)}{\sum_m \exp(-\Delta_m/2)}$",
            transform=ax.transAxes, fontsize=12.5, color=INK, ha="left",
            va="top")
    clean(ax)

    fig.tight_layout()
    save(fig, "aic_decomposition.png")
    for m, f, q, a, dd, ww in zip(names, fit, pen, aic, d, w):
        print(f"  모형 {m}: 적합항={f:.2f} 벌점={q} AIC={a:.2f} "
              f"delta={dd:.2f} 가중치={ww:.4f}")


# === 3. BIC 의 일치성 ===
def fig_bic_consistency():
    rng = np.random.default_rng(77)
    P, reps = 8, 400
    beta = np.zeros(P)
    beta[:3] = [1.2, -0.9, 0.7]
    ns = [50, 100, 200, 500, 1000, 3000, 10000]
    res = {"AIC": [], "BIC": []}
    size = {"AIC": [], "BIC": []}
    for n in ns:
        hit = {"AIC": 0, "BIC": 0}
        sz = {"AIC": [], "BIC": []}
        for _ in range(reps):
            X = rng.normal(0, 1, (n, P))
            y = X @ beta + rng.normal(0, 1.0, n)
            best = {"AIC": (np.inf, None), "BIC": (np.inf, None)}
            # 참 변수 3개는 늘 넣고, 잡음 변수의 부분집합을 모두 훑는다
            noise = list(range(3, P))
            for r in range(len(noise) + 1):
                for combo in combinations(noise, r):
                    cols = [0, 1, 2] + list(combo)
                    A = np.column_stack([np.ones(n), X[:, cols]])
                    rss = ols_rss(A, y)
                    k = len(cols) + 2
                    for nm, pen in (("AIC", 2 * k), ("BIC", k * np.log(n))):
                        v = fit_term(n, rss) + pen
                        if v < best[nm][0]:
                            best[nm] = (v, cols)
            for nm in ("AIC", "BIC"):
                cols = best[nm][1]
                sz[nm].append(len(cols))
                if len(cols) == 3:
                    hit[nm] += 1
        for nm in ("AIC", "BIC"):
            res[nm].append(hit[nm] / reps)
            size[nm].append(np.mean(sz[nm]))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ticks = ns
    tlabs = ["50", "100", "200", "500", "1000", "3000", "10000"]

    ax = axes[0]
    ax.plot(ns, res["BIC"], color=GREEN, lw=2.6, marker="^", ms=7,
            label="BIC")
    ax.plot(ns, res["AIC"], color=ORANGE, lw=2.6, marker="s", ms=6,
            label="AIC")
    ax.axhline(1.0, color=MUTED, lw=1.4, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel("참 모형을 정확히 고른 비율", fontsize=10.5, color=INK)
    ax.set_title("BIC 는 1 로 가고 AIC 는 가지 않는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=10, loc="center right", frameon=False)
    ax.annotate(f"n=10000 에서 BIC {res['BIC'][-1]:.2f}",
                xy=(10000, res["BIC"][-1]), xytext=(330, 0.80), fontsize=9.6,
                color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.annotate(f"AIC 는 {res['AIC'][-1]:.2f} 에서 멈춘다",
                xy=(10000, res["AIC"][-1]), xytext=(330, 0.30), fontsize=9.6,
                color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    clean(ax)

    ax = axes[1]
    ax.plot(ns, size["BIC"], color=GREEN, lw=2.6, marker="^", ms=7,
            label="BIC")
    ax.plot(ns, size["AIC"], color=ORANGE, lw=2.6, marker="s", ms=6,
            label="AIC")
    ax.axhline(3, color=MUTED, lw=1.6, ls="--")
    ax.text(10000, 3.06, "참 변수 개수 3", fontsize=9.6, color=INK,
            ha="right")
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(2.8, 4.3)
    ax.set_xlabel("표본크기 n", fontsize=10.5, color=INK)
    ax.set_ylabel("고른 변수 개수의 평균", fontsize=10.5, color=INK)
    ax.set_title("AIC 의 과대선택은 사라지지 않는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=10, loc="center right", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "bic_consistency.png")
    for i, n in enumerate(ns):
        print(f"  n={n:6d}  AIC 적중={res['AIC'][i]:.3f} 평균크기="
              f"{size['AIC'][i]:.3f}   BIC 적중={res['BIC'][i]:.3f} 평균크기="
              f"{size['BIC'][i]:.3f}")


# === 4. K 를 고르는 절충 ===
def fig_cv_k():
    rng = np.random.default_rng(101)
    n, P, reps, m_test = 80, 5, 600, 20000
    beta = np.array([1.5, -1.0, 0.8, 0.0, 0.0])
    Xt = rng.normal(0, 1, (m_test, P))
    At = np.column_stack([np.ones(m_test), Xt])
    Ks = [2, 5, 10, 20, 40, n]
    err = {K: [] for K in Ks}
    for _ in range(reps):
        X = rng.normal(0, 1, (n, P))
        y = X @ beta + rng.normal(0, 1.0, n)
        A = np.column_stack([np.ones(n), X])
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        yt = Xt @ beta + rng.normal(0, 1.0, m_test)
        target = ((yt - At @ b) ** 2).mean()
        for K in Ks:
            folds = np.array_split(rng.permutation(n), K)
            sc = []
            for f in folds:
                mk = np.ones(n, bool)
                mk[f] = False
                bf = np.linalg.lstsq(A[mk], y[mk], rcond=None)[0]
                sc.append(((y[f] - A[f] @ bf) ** 2).mean())
            err[K].append(np.mean(sc) - target)
    bias = [np.mean(err[K]) for K in Ks]
    sd = [np.std(err[K]) for K in Ks]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    xs = np.arange(len(Ks))
    labs = [str(K) for K in Ks[:-1]] + [f"{n}\n(LOOCV)"]

    ax = axes[0]
    ax.plot(xs, bias, color=BLUE, lw=2.6, marker="o", ms=7)
    ax.axhline(0, color=MUTED, lw=1.4, ls="--")
    for x, v in zip(xs, bias):
        ax.text(x, v + 0.012, f"{v:+.3f}", ha="center", fontsize=9.8,
                color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylim(-0.04, 0.27)
    ax.set_xlabel("겹의 개수 K  (관측 80개)", fontsize=10.5, color=INK)
    ax.set_ylabel("평균 오차 (추정 MSE 빼기 실제 MSE)", fontsize=10,
                  color=INK)
    ax.set_title("K 가 작으면 훈련자료가 작아 비관적이 된다", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    ax = axes[1]
    ax.plot(xs, sd, color=RED, lw=2.6, marker="s", ms=7)
    for x, v in zip(xs, sd):
        ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=9.8,
                color=INK)
    imin = int(np.argmin(sd))
    ax.scatter([xs[imin]], [sd[imin]], s=140, facecolor="none",
               edgecolor=GREEN, lw=2.4, zorder=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylim(min(sd) * 0.9, max(sd) * 1.12)
    ax.set_xlabel("겹의 개수 K  (관측 80개)", fontsize=10.5, color=INK)
    ax.set_ylabel("오차의 표준편차", fontsize=10.5, color=INK)
    ax.set_title("겹이 서로 닮을수록 평균의 이득이 줄어든다", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    fig.tight_layout()
    save(fig, "cv_k_bias_variance.png")
    for K, b, s in zip(Ks, bias, sd):
        print(f"  K={K:3d}  평균오차={b:+.4f}  표준편차={s:.4f}")


# === 5. 검증집합 대 LOOCV 대 10겹 ===
def fig_cv_methods():
    np.random.seed(42)
    n = 200
    x = np.random.uniform(1, 10, n)
    y = 5 + 2 * x - 0.3 * x ** 2 + np.random.normal(0, 2, n)
    degs = np.arange(1, 11)

    def design(v, d):
        return np.vander(v, d + 1, increasing=True)

    rng = np.random.default_rng(0)
    val = np.zeros((20, len(degs)))
    for r in range(20):
        idx = rng.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        for j, d in enumerate(degs):
            A = design(x[tr], d)
            b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            val[r, j] = ((y[te] - design(x[te], d) @ b) ** 2).mean()

    loo, kf = [], []
    for d in degs:
        A = design(x, d)
        H = A @ np.linalg.pinv(A.T @ A) @ A.T
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        e = y - A @ b
        h = np.diag(H)
        loo.append(((e / (1 - h)) ** 2).mean())
        folds = np.array_split(rng.permutation(n), 10)
        sc = []
        for f in folds:
            mk = np.ones(n, bool)
            mk[f] = False
            bf = np.linalg.lstsq(A[mk], y[mk], rcond=None)[0]
            sc.append(((y[f] - A[f] @ bf) ** 2).mean())
        kf.append(np.mean(sc))
    loo, kf = np.array(loo), np.array(kf)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    for r in range(20):
        ax.plot(degs, val[r], color=MUTED, lw=1.2, alpha=0.75)
    picks = degs[val.argmin(1)]
    ax.plot(degs, val.mean(0), color=RED, lw=2.8, label="20번의 평균")
    ax.set_ylim(3.0, 8.0)
    ax.set_xticks(degs)
    ax.set_xlabel("다항식의 차수", fontsize=10.5, color=INK)
    ax.set_ylabel("검증집합 MSE", fontsize=10.5, color=INK)
    ax.set_title("검증집합을 20번 다르게 나누면", fontsize=11.5, color=INK,
                 pad=6)
    ax.legend(fontsize=9.8, loc="lower left", frameon=False)
    vals, cnts = np.unique(picks, return_counts=True)
    txt = ",  ".join(f"차수 {v}: {c}번" for v, c in zip(vals, cnts))
    ax.text(5.5, 7.3, "고른 차수가 제각각이다\n" + txt, fontsize=9.6,
            color=RED, ha="center", linespacing=1.6)
    clean(ax)

    ax = axes[1]
    ax.plot(degs, loo, color=BLUE, lw=2.8, marker="o", ms=6, label="LOOCV")
    ax.plot(degs, kf, color=GREEN, lw=2.8, marker="^", ms=6, label="10겹")
    ax.axvline(2, color=MUTED, lw=1.4, ls=":")
    ax.text(2.15, 6.4, "참 차수 2", fontsize=9.8, color=INK)
    ax.annotate(f"차수 1 → 2 에서 {loo[0]:.2f} → {loo[1]:.2f}",
                xy=(2, loo[1]), xytext=(3.2, 5.6), fontsize=9.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(6.0, 4.3, "그 뒤로는 평평하다가 서서히 오른다", fontsize=9.8,
            color=INK, ha="center")
    ax.set_ylim(3.0, 8.0)
    ax.set_xticks(degs)
    ax.set_xlabel("다항식의 차수", fontsize=10.5, color=INK)
    ax.set_ylabel("교차검증 MSE", fontsize=10.5, color=INK)
    ax.set_title("같은 자료, 흔들리지 않는 곡선", fontsize=11.5, color=INK,
                 pad=6)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "cv_method_comparison.png")
    print(f"  검증집합이 고른 차수: {dict(zip(vals.tolist(), cnts.tolist()))}")
    print(f"  검증집합 MSE 범위(차수 2): {val[:,1].min():.3f} ~ "
          f"{val[:,1].max():.3f}")
    for i, d in enumerate(degs):
        if d <= 4 or d == 10:
            print(f"   차수 {d:2d}  LOOCV={loo[i]:.4f}  10겹={kf[i]:.4f}")


# === 6. 전진선택의 경로 ===
def fig_forward():
    np.random.seed(42)
    n, P = 200, 8
    X = np.random.randn(n, P)
    beta = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
    y = X @ beta + np.random.randn(n) * 2

    rem, sel = list(range(P)), []
    aic_h, bic_h, cv_h = [], [], []
    rng = np.random.default_rng(5)
    for _ in range(P):
        best, bj = np.inf, None
        for j in rem:
            A = np.column_stack([np.ones(n), X[:, sel + [j]]])
            v = fit_term(n, ols_rss(A, y)) + 2 * (len(sel) + 2)
            if v < best:
                best, bj = v, j
        sel.append(bj)
        rem.remove(bj)
        A = np.column_stack([np.ones(n), X[:, sel]])
        rss = ols_rss(A, y)
        k = len(sel) + 1
        aic_h.append(fit_term(n, rss) + 2 * k)
        bic_h.append(fit_term(n, rss) + k * np.log(n))
        folds = np.array_split(np.arange(n), 5)      # 섞지 않은 5겹
        sc = []
        for f in folds:
            mk = np.ones(n, bool)
            mk[f] = False
            bf = np.linalg.lstsq(A[mk], y[mk], rcond=None)[0]
            sc.append(((y[f] - A[f] @ bf) ** 2).mean())
        cv_h.append(np.mean(sc))

    sizes = np.arange(1, P + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.plot(sizes, aic_h, color=ORANGE, lw=2.6, marker="s", ms=6, label="AIC")
    ax.plot(sizes, bic_h, color=GREEN, lw=2.6, marker="^", ms=6, label="BIC")
    ia, ib = int(np.argmin(aic_h)), int(np.argmin(bic_h))
    ax.scatter([sizes[ia]], [aic_h[ia]], s=140, facecolor="none",
               edgecolor=ORANGE, lw=2.4, zorder=6)
    ax.scatter([sizes[ib]], [bic_h[ib]], s=140, facecolor="none",
               edgecolor=GREEN, lw=2.4, zorder=6)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}\n$x_{{{j+1}}}$" for s, j in zip(sizes, sel)],
                       fontsize=9.6)
    ax.set_xlabel("모형 크기와 그 단계에서 들어온 변수", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("기준값", fontsize=10.5, color=INK)
    ax.set_title("전진선택의 경로 — 참 변수 셋이 먼저 들어온다",
                 fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    ax.text(4.3, 395,
            f"3 → 4 로 갈 때"
            f"\nAIC 는 {aic_h[3]-aic_h[2]:+.2f},  BIC 는 {bic_h[3]-bic_h[2]:+.2f}",
            fontsize=9.8, color=INK, linespacing=1.6)
    clean(ax)

    ax = axes[1]
    ax.plot(sizes, cv_h, color=BLUE, lw=2.8, marker="o", ms=7)
    ic = int(np.argmin(cv_h))
    ax.scatter([sizes[ic]], [cv_h[ic]], s=150, facecolor="none",
               edgecolor=RED, lw=2.4, zorder=6)
    for s, v in zip(sizes, cv_h):
        ax.text(s, v + 0.13, f"{v:.2f}", ha="center", fontsize=9.4,
                color=INK)
    ax.axhline(4.0, color=MUTED, lw=1.4, ls="--")
    ax.annotate(r"잡음 $\sigma^2 = 4$", xy=(6, 4.0), xytext=(4.6, 5.3),
                fontsize=9.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xticks(sizes)
    ax.set_xlabel("모형 크기", fontsize=10.5, color=INK)
    ax.set_ylabel("5겹 교차검증 MSE", fontsize=10.5, color=INK)
    ax.set_title("교차검증도 같은 곳에서 멈춘다", fontsize=11.5, color=INK,
                 pad=6)
    ax.set_ylim(3.2, 11.5)
    clean(ax)

    fig.tight_layout()
    save(fig, "forward_selection_path.png")
    print(f"  선택 순서(0 기준): {sel}")
    for s in range(P):
        print(f"   크기 {s+1}: AIC={aic_h[s]:.2f} BIC={bic_h[s]:.2f} "
              f"CV={cv_h[s]:.4f}")


# === 7. 전수탐색의 비용과 탐욕의 대가 ===
def fig_subset_cost():
    ps = np.arange(2, 41)
    full = 2.0 ** ps
    fwd = ps * (ps + 1) / 2

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    ax.plot(ps, full, color=RED, lw=2.8, label=r"최량 부분집합  $2^p$")
    ax.plot(ps, fwd, color=BLUE, lw=2.8, label=r"전진 단계선택  $p(p+1)/2$")
    ax.set_yscale("log")
    ax.set_yticks([1e1, 1e3, 1e5, 1e7, 1e9, 1e11])
    ax.set_yticklabels(["10", "1000", "10만", "1천만", "10억", "1000억"],
                       fontsize=9)
    ax.minorticks_off()
    for p, col in [(10, MUTED), (20, MUTED), (40, MUTED)]:
        ax.axvline(p, color=col, lw=1.0, ls=":")
    ax.annotate(f"p=20:  {2**20:,} 대 {20*21//2}", xy=(20, 2.0 ** 20),
                xytext=(5.0, 1e9), fontsize=9.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate(f"p=40:  1조가 넘는다", xy=(40, 2.0 ** 40),
                xytext=(21, 4e11), fontsize=9.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_ylim(5, 5e12)
    ax.set_xlabel("설명변수 개수 p", fontsize=10.5, color=INK)
    ax.set_ylabel("적합해야 하는 모형 개수 (로그 눈금)", fontsize=10,
                  color=INK)
    ax.set_title("전수탐색은 곧 불가능해진다", fontsize=11.5, color=INK,
                 pad=6)
    ax.legend(fontsize=10, loc="center left", frameon=False)
    clean(ax)

    # 전진선택이 최량 두 변수 모형을 놓치는 자료 (억제 효과)
    rng = np.random.default_rng(0)
    n = 400
    z1, z2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x2 = z1
    x3 = 0.9 * z1 + np.sqrt(1 - 0.81) * z2   # x2 와 상관 0.9
    core = x2 - x3                            # 둘의 차이만이 y 를 만든다
    y = core + rng.normal(0, 0.10, n)
    x1 = 0.60 * core / core.std() + rng.normal(0, 0.80, n)
    X = np.column_stack([x1, x2, x3])
    names = [r"$x_1$", r"$x_2$", r"$x_3$"]
    tss = ((y - y.mean()) ** 2).sum()

    def r2_of(cols):
        A = np.column_stack([np.ones(n), X[:, list(cols)]])
        return 1 - ols_rss(A, y) / tss

    singles = [(i, r2_of([i])) for i in range(3)]
    pairs = [(c, r2_of(c)) for c in combinations(range(3), 2)]
    first = max(singles, key=lambda t: t[1])[0]
    fwd_pair = max([c for c in pairs if first in c[0]], key=lambda t: t[1])
    best_pair = max(pairs, key=lambda t: t[1])

    ax = axes[1]
    lab1 = [names[i] for i, _ in singles]
    v1 = [v for _, v in singles]
    lab2 = ["+".join(names[j] for j in c) for c, _ in pairs]
    v2 = [v for _, v in pairs]
    xs = np.arange(6)
    cols = [GREEN if i == first else MUTED for i in range(3)]
    cols += [RED if c == best_pair[0] else
             (ORANGE if c == fwd_pair[0] else MUTED) for c, _ in pairs]
    ax.bar(xs, v1 + v2, 0.6, color=cols)
    for x, v in zip(xs, v1 + v2):
        ax.text(x, v + 0.018, f"{v:.3f}", ha="center", fontsize=9.6,
                color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab1 + lab2, fontsize=10.5)
    ax.set_ylim(0, 1.34)
    ax.set_ylabel(r"$R^2$", fontsize=11, color=INK)
    ax.set_title("전진선택이 최선의 두 변수 모형을 놓친다", fontsize=11.5,
                 color=INK, pad=6)
    ax.annotate("1단계에서 고르는 변수", xy=(first, v1[first] + 0.06),
                xytext=(0.45, 0.72), fontsize=9.6, color=GREEN, ha="left",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    ax.annotate("그래서 도달하는 두 변수 모형",
                xy=(3 + list(pairs).index(fwd_pair), fwd_pair[1] + 0.06),
                xytext=(0.45, 0.55), fontsize=9.6, color=ORANGE, ha="left",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate("실제 최선의 두 변수 모형",
                xy=(3 + list(pairs).index(best_pair), best_pair[1] + 0.06),
                xytext=(2.4, 1.20), fontsize=9.6, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    clean(ax)

    fig.tight_layout()
    save(fig, "subset_cost.png")
    print(f"  p=10: {2**10} 대 {10*11//2},  p=20: {2**20:,} 대 {20*21//2},  "
          f"p=40: {2**40:,} 대 {40*41//2}")
    for i, v in singles:
        print(f"  단일 {names[i]}: R2={v:.4f}")
    for c, v in pairs:
        print(f"  쌍 {c}: R2={v:.4f}")
    print(f"  전진선택 경로 -> {fwd_pair[0]} (R2 {fwd_pair[1]:.4f}), "
          f"최선 -> {best_pair[0]} (R2 {best_pair[1]:.4f})")


# === 8. 세 선택법의 일치와 불일치 ===
def fig_three_methods():
    P, reps = 8, 400
    beta = np.array([3.0, 1.5, -2.0, 0.6, 0, 0, 0, 0])
    rng = np.random.default_rng(42)
    n = 60
    freq = {m: np.zeros(P) for m in ("최량 부분집합", "전진", "후진")}
    exact = {m: 0 for m in freq}
    size_target = 4
    agree = 0

    for _ in range(reps):
        Z = rng.normal(0, 1, (n, P))
        X = Z.copy()
        X[:, 4] = 0.75 * Z[:, 3] + np.sqrt(1 - 0.5625) * Z[:, 4]
        y = X @ beta + rng.normal(0, 2, n)

        def rss_of(cols):
            A = np.column_stack([np.ones(n), X[:, list(cols)]])
            return ols_rss(A, y)

        best = min(combinations(range(P), size_target), key=rss_of)
        sel = []
        rem = list(range(P))
        for _ in range(size_target):
            j = min(rem, key=lambda t: rss_of(sel + [t]))
            sel.append(j)
            rem.remove(j)
        cur = list(range(P))
        while len(cur) > size_target:
            j = min(cur, key=lambda t: rss_of([c for c in cur if c != t]))
            cur = [c for c in cur if c != j]
        for m, s in (("최량 부분집합", best), ("전진", sel), ("후진", cur)):
            for j in s:
                freq[m][j] += 1
            if set(s) == {0, 1, 2, 3}:
                exact[m] += 1
        if set(best) == set(sel) == set(cur):
            agree += 1

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))

    ax = axes[0]
    ms = list(freq)
    w = 0.26
    xs = np.arange(P)
    for i, (m, c) in enumerate(zip(ms, [RED, BLUE, GREEN])):
        ax.bar(xs + (i - 1) * w, freq[m] / reps, w, color=c, label=m)
    ax.axvspan(-0.5, 3.5, color=GREEN_F, alpha=0.45, zorder=0)
    ax.text(1.5, 1.08, "참 계수가 0 이 아닌 변수", fontsize=9.8, color=GREEN,
            ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$x_{{{j+1}}}$" for j in range(P)], fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("크기 4 모형에 뽑힌 비율", fontsize=10.5, color=INK)
    ax.set_title(f"$n = 60$ 에서 같은 실험 {reps}번 — 변수별 선택 빈도", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="center right", frameon=False)
    clean(ax)

    ax = axes[1]
    vals = [exact[m] / reps for m in ms]
    ax.bar(np.arange(3), vals, 0.5, color=[RED, BLUE, GREEN])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=13, color=INK)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(ms, fontsize=11.5)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel(r"$\{x_1,x_2,x_3,x_4\}$ 를 정확히 고른 비율", fontsize=10.5,
                  color=INK)
    ax.set_title("세 방법이 사실상 같은 답을 낸다", fontsize=11.5, color=INK,
                 pad=6)
    ax.text(1.0, 1.05, r"참 계수 $\beta_4 = 0.6$ 이 작아"
            "\n네 번째 변수를 놓치는 일이 생긴다"
            f"\n세 방법이 완전히 같은 답을 낸 비율 {agree/reps:.1%}",
            fontsize=9.8, color=INK, ha="center", va="top", linespacing=1.7)
    clean(ax)

    fig.tight_layout()
    save(fig, "three_methods_agree.png")
    for m in ms:
        print(f"  {m:10s} 정확 적중={exact[m]/reps:.3f}  "
              f"빈도={np.round(freq[m]/reps, 3)}")
    print(f"  세 방법 완전 일치 비율 = {agree/reps:.4f}")


if __name__ == "__main__":
    fig_tradeoff()
    fig_aic_decomp()
    fig_bic_consistency()
    fig_cv_k()
    fig_cv_methods()
    fig_forward()
    fig_subset_cost()
    fig_three_methods()
