"""12장 들머리 그림 여섯 장 (진단 넷 · 다항회귀 · 장 개요).

만드는 파일:
  ch12/diagnostics/img/eye_sees_funnels.png          없는 깔때기를 보는 눈
  ch12/diagnostics/img/precision_not_coefficient.png 무너진 것은 정밀도다
  ch12/diagnostics/img/pairwise_misses_it.png        쌍만 보면 못 본다
  ch12/diagnostics/img/ceiling_draws_a_line.png      천장이 그리는 직선
  ch12/interaction_polynomial/img/floor_is_the_noise.png 바닥은 잡음이 정한다
  ch12/img/six_points_hide_fifty_three.png           여섯 점이 쉰셋을 가린다

본문 그림이 이미 들머리의 수를 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

DIAG = "docs/ch12/diagnostics/img/"
POLY = "docs/ch12/interaction_polynomial/img/"
IDX = "docs/ch12/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_eye_sees_funnels():
    """11.4 잔차 — 1.94 배는 등분산에서 흔히 나온다.

    쪽의 residuals_48·_99 는 잔차 그림 자체를, residual_vs_observed 상자는
    가로축을 바꾸면 무늬가 생기는 것을 다룬다. 여기서는 그 셋이 건드리지
    않는 것 --- **들쭉날쭉함의 귀무분포** --- 만 본다. 등분산으로 일부러
    만든 자료에서 사분위 표준편차의 최대/최소 비가 얼마나 흔히 1.94 를
    넘는지 세고, 칸을 키우면 그 비가 어떻게 가라앉는지 본다.
    """
    N_PAGE, RATIO_PAGE = 50, 1.94
    REPS = 20_000
    rng = np.random.default_rng(1140)

    def ratio_sample(n, gen):
        x = gen.normal(0, 1, n)
        y = 1 + 2 * x + gen.normal(0, 3, n)       # 등분산으로 만든다
        res = stats.linregress(x, y)
        e = y - (res.intercept + res.slope * x)
        fit = res.intercept + res.slope * x
        q = np.quantile(fit, [0.25, 0.5, 0.75])
        g = np.digitize(fit, q)
        sds = np.array([e[g == k].std(ddof=1) for k in range(4)])
        return sds.max() / sds.min()

    rat = np.array([ratio_sample(N_PAGE, rng) for _ in range(REPS)])
    share = (rat >= RATIO_PAGE).mean()

    NS = [50, 100, 200, 400, 800, 1600]
    p95 = [np.quantile(rat, 0.95)]          # n = 50 은 위 실험을 그대로 쓴다
    for n in NS[1:]:
        v = np.array([ratio_sample(n, rng) for _ in range(4000)])
        p95.append(np.quantile(v, 0.95))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.hist(rat, bins=np.linspace(1, 6, 90), color="0.62", alpha=0.85)
    ax1.axvline(RATIO_PAGE, color="tab:red", lw=2.6)
    ax1.text(RATIO_PAGE + 0.08, ax1.get_ylim()[1] * 0.92,
             f"이 쪽이 본 {RATIO_PAGE} 배", fontsize=11.5, color="tab:red",
             fontweight="bold", va="top")
    ax1.set_xlim(1, 6)
    ax1.set_xlabel("사분위 네 칸의 잔차 표준편차 — 최대 ÷ 최소", fontsize=11)
    ax1.set_ylabel("횟수", fontsize=11)
    ax1.set_title(f"등분산으로 만든 자료 $n = {N_PAGE}$ 를 {REPS:,} 번",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.97, 0.55, f"그만큼 벌어지는 일이 {100 * share:.1f}%\n"
             f"중앙값은 {np.median(rat):.2f} 배다",
             transform=ax1.transAxes, ha="right", va="top", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(NS, p95, "o-", ms=9, lw=2.6, color="tab:purple")
    ax2.axhline(RATIO_PAGE, color="tab:red", ls="--", lw=1.8)
    ax2.text(1580, RATIO_PAGE + 0.06, f"{RATIO_PAGE} 배", ha="right",
             va="bottom", fontsize=10.5, color="tab:red", fontweight="bold")
    for n, v in zip(NS, p95):
        ax2.text(n, v + 0.07, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=10.5, color="tab:purple", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xticks(NS, [str(n) for n in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(1, 3.4)
    ax2.set_xlabel("표본 크기 n (로그 눈금)", fontsize=11)
    ax2.set_ylabel("그 비의 $95$ 번째 백분위", fontsize=11)
    ax2.set_title("칸이 커져야 비가 가라앉는다", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("눈이 보는 깔때기가 자료에 있는지 칸이 작아서인지는 "
                 "세어 보면 안다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "eye_sees_funnels.png")

    print(f"\n[잔차] 등분산 n = {N_PAGE} · {REPS:,} 번")
    print(f"  비의 중앙값 {np.median(rat):.3f} · {RATIO_PAGE} 배 이상이 "
          f"{100 * share:.2f}% · 95 번째 백분위 {np.quantile(rat, .95):.3f}")
    for n, v in zip(NS, p95):
        print(f"  n = {n:>5d}  95 번째 백분위 {v:.3f}")


# ════════════════════════════════════════════════════════════════════
def fig_precision_not_coefficient():
    """11.4 영향점 — 무너진 것은 계수가 아니라 정밀도다.

    쪽의 influence_54 는 자료와 적합을, masking·delete_then_test 상자는
    가림과 걷어낸 뒤의 p 값을 다룬다. 여기서는 들머리의 첫 문단 ---
    **계수는 거의 안 움직이는데 그 곁의 구간이 벌어진다** --- 와, 쿡
    거리가 넷을 정확히 집어내는 모습만 본다. 자료는 쪽의 코드와 같다
    (np.random.seed(0), make_regression(100, 1, noise=10), 95~98 번을
    +80 · -80 · +60 · -60 으로 흔든다).
    """
    from sklearn.datasets import make_regression

    np.random.seed(0)
    X, y0 = make_regression(n_samples=100, n_features=1, noise=10)
    x = X.ravel()
    n = len(x)
    y = y0.copy()
    PLANT = [95, 96, 97, 98]
    for i, d in zip(PLANT, [80, -80, 60, -60]):
        y[i] += d

    def fit(yy):
        r = stats.linregress(x, yy)
        res = yy - (r.intercept + r.slope * x)
        s = np.sqrt((res ** 2).sum() / (n - 2))
        sxx = ((x - x.mean()) ** 2).sum()
        return r.slope, s, r.rvalue ** 2, stats.t.ppf(0.975, n - 2) * s / np.sqrt(sxx)

    b0, s0, r0, h0 = fit(y0)
    b1, s1, r1, h1 = fit(y)

    # 쿡 거리 --- 다시 적합하지 않고 잰다
    r = stats.linregress(x, y)
    res = y - (r.intercept + r.slope * x)
    sxx = ((x - x.mean()) ** 2).sum()
    lev = 1 / n + (x - x.mean()) ** 2 / sxx
    s2 = (res ** 2).sum() / (n - 2)
    cook = res ** 2 * lev / (2 * s2 * (1 - lev) ** 2)
    thr = 4 / n
    order = np.argsort(cook)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    NAMES = ["기울기", "잔차의 표준편차", "$R^2$", "기울기 구간의 반폭"]
    before = [b0, s0, r0, h0]
    after = [b1, s1, r1, h1]
    rel = [a / b for a, b in zip(after, before)]
    xs = np.arange(4)
    cols = ["tab:blue", "tab:red", "tab:red", "tab:red"]
    ax1.bar(xs, [100 * (v - 1) for v in rel], width=0.5, color=cols,
            alpha=0.88)
    ax1.axhline(0, color="0.15", lw=1.8)
    for x_, v, b, a in zip(xs, rel, before, after):
        ax1.text(x_, 100 * (v - 1) + np.sign(v - 1) * 2.2,
                 f"{100 * (v - 1):+.1f}%", ha="center",
                 va="bottom" if v > 1 else "top", fontsize=12,
                 fontweight="bold", color="0.15")
        ax1.text(x_, -62, f"{b:.4f}\n↓\n{a:.4f}", ha="center", va="top",
                 fontsize=10, color="0.3")
    ax1.set_xticks(xs, NAMES, fontsize=10.5)
    ax1.set_ylim(-115, 75)
    ax1.set_yticks([-50, -25, 0, 25, 50])
    ax1.set_ylabel("이상치 넷을 심은 뒤의 변화 (%)", fontsize=11)
    ax1.set_title("계수는 거의 그대로인데 그 곁이 벌어진다", fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)

    ax2.vlines(np.arange(n), 0, cook, color="0.72", lw=1.6)
    ax2.plot(np.arange(n), cook, "o", ms=4, color="0.6")
    ax2.vlines(PLANT, 0, cook[PLANT], color="tab:red", lw=2.6)
    ax2.plot(PLANT, cook[PLANT], "o", ms=9, color="tab:red", zorder=5)
    ax2.axhline(thr, color="0.2", ls="--", lw=1.8)
    ax2.text(1, thr * 1.12, f"흔히 쓰는 문턱 $4/n = {thr:.2f}$", fontsize=10.5,
             color="0.2", fontweight="bold")
    fifth = cook[order[4]]
    ax2.annotate(f"다섯째로 큰 값 {fifth:.4f}\n— 문턱 아래다",
                 xy=(order[4], fifth), xytext=(26, 0.17), fontsize=10.5,
                 color="0.25", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.6))
    ax2.set_xlim(-2, n + 1)
    ax2.set_ylim(0, max(cook) * 1.18)
    ax2.set_xlabel("관측 번호", fontsize=11)
    ax2.set_ylabel("쿡 거리", fontsize=11)
    ax2.set_title("다시 적합하지 않고 넷을 정확히 집어낸다", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("계수를 읽는 사람은 아무 이상을 못 느끼는데 구간이 "
                 "벌어져 있다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "precision_not_coefficient.png")

    print(f"\n[영향점] 심기 전 → 후")
    for nm, b, a in zip(["기울기", "sigma", "R^2", "구간 반폭"],
                        before, after):
        print(f"  {nm:<10s} {b:9.4f} → {a:9.4f}  ({a / b:.3f} 배)")
    print(f"  쿡 거리 문턱 {thr:.2f} 위: {np.sort(np.where(cook > thr)[0])} "
          f"· 다섯째 {fifth:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_pairwise_misses_it():
    """11.4 다중공선성 — 상관행렬이 깨끗해도 설계행렬은 특이할 수 있다.

    쪽의 multicollinearity_22 는 주택 자료의 VIF 를, longley 상자는 옛
    계산기 시절을 다룬다. 여기서는 들머리의 둘째·셋째 문단 --- **독립인
    m 개에 그 합을 얹으면 쌍별 상관이 모두 1/sqrt(m) 로 작아지는데 VIF 는
    십만에 이른다** --- 만 본다.
    """
    M_PAGE, N = 50, 5000
    TAU = 0.01            # 합에 섞는 아주 작은 잡음 (없으면 VIF 가 무한이다)
    ms = np.arange(2, 101)
    rng = np.random.default_rng(1141)

    Z = rng.normal(0, 1, (N, M_PAGE))
    X = np.column_stack([Z, Z.sum(axis=1) + rng.normal(0, TAU, N)])
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(M_PAGE + 1, 1)
    pair = np.abs(C[iu])

    vifs = []
    for j in range(M_PAGE + 1):
        others = np.delete(np.arange(M_PAGE + 1), j)
        A = np.column_stack([np.ones(N), X[:, others]])
        beta = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        res = X[:, j] - A @ beta
        r2 = 1 - (res ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        vifs.append(1 / (1 - r2))
    vifs = np.array(vifs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(ms, 1 / np.sqrt(ms), "-", lw=2.8, color="tab:blue",
             label="합과 각 변수의 상관  $1/\\sqrt{m}$")
    for lv, nm, col in ((0.8, "흔히 쓰는 문턱 0.8", "tab:red"),
                        (0.9, "0.9", "tab:orange")):
        ax1.axhline(lv, color=col, ls="--", lw=1.8)
        ax1.text(99, lv + 0.015, nm, ha="right", va="bottom", fontsize=10.5,
                 color=col, fontweight="bold")
    ax1.plot([M_PAGE], [1 / np.sqrt(M_PAGE)], "o", ms=13, color="tab:blue",
             zorder=5)
    ax1.annotate(f"$m = {M_PAGE}$ 이면 {1 / np.sqrt(M_PAGE):.3f}\n"
                 f"— 어떤 문턱도 넘지 않는다",
                 xy=(M_PAGE, 1 / np.sqrt(M_PAGE)), xytext=(56, 0.46),
                 fontsize=11.5, color="tab:blue", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
    ax1.set_xlim(2, 100)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("독립인 설명변수의 수 m", fontsize=11)
    ax1.set_ylabel("쌍별 상관의 절댓값", fontsize=11)
    ax1.set_title("독립인 m 개에 그 합을 하나 얹으면", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.hist(vifs, bins=np.logspace(np.log10(vifs.min() * 0.8),
                                    np.log10(vifs.max() * 1.2), 28),
             color="tab:red", alpha=0.8)
    ax2.set_xscale("log")
    ax2.set_xticks([1e4, 1e5, 1e6], ["10,000", "100,000", "1,000,000"])
    ax2.set_xticks([], minor=True)
    ax2.axvline(10, color="0.2", ls="--", lw=1.8)
    ax2.set_xlabel("VIF (로그 눈금)", fontsize=11)
    ax2.set_ylabel("변수 수", fontsize=11)
    ax2.set_title(f"같은 자료의 VIF — 최댓값 {vifs.max():,.0f}",
                  fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.5, 0.95, f"중앙값 {np.median(vifs):,.0f}\n"
             f"쌍별 상관의 최댓값은 {pair.max():.3f} 뿐인데\n"
             f"VIF 는 모두 십만 언저리다",
             transform=ax2.transAxes, ha="center", va="top", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("상관행렬이 깨끗하다고 공선성이 없는 것이 아니다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "pairwise_misses_it.png")

    print(f"\n[공선성] m = {M_PAGE}, n = {N}, 합에 섞은 잡음 {TAU}")
    print(f"  쌍별 상관 최댓값 {pair.max():.4f} (이론값 "
          f"{1 / np.sqrt(M_PAGE):.4f})")
    print(f"  VIF 최댓값 {vifs.max():,.1f} · 중앙값 {np.median(vifs):,.1f}")


# ════════════════════════════════════════════════════════════════════
def fig_ceiling_draws_a_line():
    """11.4 다중회귀 진단 — 천장이 그리는 직선은 항등식이다.

    쪽의 multi_reg_diag_93 은 잔차-적합값 그림을, six_households 상자는
    같은 자료의 다른 이상한 칸을 다룬다. 여기서는 그 둘이 수로 말하지
    않는 것 --- **그 줄이 근사가 아니라 항등식이라는 것**과, 모형을
    고쳐도 줄이 사라지지 않는다는 것 --- 만 본다.

    자료는 쪽과 같은 California Housing 이다.
    """
    from sklearn.datasets import fetch_california_housing

    d = fetch_california_housing(download_if_missing=False)
    X, y = d.data, d.target
    cap = y.max()
    at_cap = y >= cap - 1e-9

    def fit_resid(Xm):
        A = np.column_stack([np.ones(len(Xm)), Xm])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        yhat = A @ beta
        return yhat, y - yhat

    yhat, res = fit_resid(X)
    sl = np.polyfit(yhat[at_cap], res[at_cap], 1)[0]

    # 모형을 키워도 줄은 그대로다 --- 로그 변환과 제곱항을 더해 본다
    Xb = np.column_stack([X, np.log1p(X[:, 0]), X[:, 0] ** 2,
                          X[:, 5] ** 2, X[:, 0] * X[:, 5]])
    yhat2, res2 = fit_resid(Xb)
    sl2 = np.polyfit(yhat2[at_cap], res2[at_cap], 1)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9), sharey=True)

    for ax, yh, rs, s, name in ((ax1, yhat, res, sl, "원래 설명변수 여덟"),
                                (ax2, yhat2, res2, sl2,
                                 "로그 · 제곱 · 교호작용을 더한 모형")):
        ax.scatter(yh[~at_cap], rs[~at_cap], s=4, color="0.72", alpha=0.35,
                   edgecolors="none")
        g = np.array([yh[at_cap].min(), yh[at_cap].max()])
        ax.plot(g, cap - g, "--", lw=3.2, color="0.15", zorder=4)
        ax.scatter(yh[at_cap], rs[at_cap], s=12, color="tab:red", alpha=0.9,
                   edgecolors="none", zorder=5)
        ax.axhline(0, color="0.3", lw=1.4)
        ax.set_xlim(-0.6, 6.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_xlabel("적합값", fontsize=11)
        ax.set_title(f"{name}\n천장 위 점에 맞춘 기울기 {s:.6f}",
                     fontsize=12)
        ax.grid(alpha=0.25)
    ax1.set_ylabel("잔차", fontsize=11)
    ax1.text(0.03, 0.04, f"반응변수가 {cap} 에서 잘려 있다\n"
             f"그 점이 {at_cap.sum():,} 건 · 전체의 "
             f"{100 * at_cap.mean():.2f}%", transform=ax1.transAxes,
             ha="left", va="bottom", fontsize=11, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))
    ax2.text(0.97, 0.04, "항을 더해도 줄은 그대로다\n"
             "— 모형이 아니라 기록 방식에서 왔기 때문이다",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=11,
             fontweight="bold", color="tab:red",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0))

    fig.suptitle(f"잔차 = {cap} 빼기 적합값 — 근사가 아니라 항등식이다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, DIAG + "ceiling_draws_a_line.png")

    print(f"\n[진단] California Housing n = {len(y):,}")
    print(f"  천장값 {cap} · 그 위 {at_cap.sum():,} 건 "
          f"({100 * at_cap.mean():.2f}%)")
    print(f"  천장 점에 맞춘 기울기: 기본 모형 {sl:.6f} · "
          f"항을 더한 모형 {sl2:.6f}")


# ════════════════════════════════════════════════════════════════════
def fig_floor_is_the_noise():
    """11.6 다항회귀 — 검정 RMSE 의 바닥은 잡음이 정한다.

    쪽의 polynomial_degree 는 차수를 올린 적합들을, runge_edges 상자는
    끝에서 벌어지는 일을 그린다. 여기서는 들머리의 마지막 문단 ---
    **검정 RMSE 의 바닥이 잡음의 표준편차이고 그 아래로는 어떤 모형도
    내려갈 수 없다** --- 만 본다. 잡음을 셋으로 바꾸어 가며 바닥이
    따라 옮겨 가는 것을 보인다.
    """
    N, DEG = 40, np.arange(1, 13)
    SIGMAS = [1.0, 2.0, 4.0]
    REPS = 300
    rng = np.random.default_rng(1142)

    def curve(x):
        return 5 + 3 * x - 0.8 * x ** 2

    out = {}
    for sg in SIGMAS:
        tr = np.zeros(len(DEG))
        te = np.zeros(len(DEG))
        for _ in range(REPS):
            x = rng.uniform(-3, 3, N)
            y = curve(x) + rng.normal(0, sg, N)
            xt = rng.uniform(-3, 3, 400)
            yt = curve(xt) + rng.normal(0, sg, 400)
            for j, dg in enumerate(DEG):
                c = np.polyfit(x, y, dg)
                tr[j] += np.sqrt(np.mean((y - np.polyval(c, x)) ** 2))
                te[j] += np.sqrt(np.mean((yt - np.polyval(c, xt)) ** 2))
        out[sg] = (tr / REPS, te / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    tr, te = out[2.0]
    ax1.plot(DEG, tr, "o-", ms=8, lw=2.6, color="tab:blue",
             label="훈련 RMSE")
    ax1.plot(DEG, te, "s-", ms=8, lw=2.6, color="tab:red",
             label="검정 RMSE")
    ax1.axhline(2.0, color="0.25", ls="--", lw=1.8)
    ax1.text(12, 2.12, "잡음의 표준편차 $2.0$", ha="right", va="bottom",
             fontsize=10.5, color="0.25", fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylim(0.6, 60)
    ax1.set_yticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax1.set_yticks([], minor=True)
    ax1.set_xticks(DEG)
    ax1.set_xlabel("다항식의 차수", fontsize=11)
    ax1.set_ylabel("RMSE (로그 눈금)", fontsize=11)
    ax1.set_title(f"$n = {N}$ · {REPS} 번 평균", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(0.97, 0.06, "훈련 RMSE 는 한 번도 올라가지 않는다\n"
             "— 낮은 차수가 높은 차수 안에 들어 있기 때문이다",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=10.5,
             fontweight="bold", color="tab:blue",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:blue", lw=1.0))

    COLS = ["tab:green", "tab:red", "tab:purple"]
    for sg, col in zip(SIGMAS, COLS):
        _, te = out[sg]
        ax2.plot(DEG, te, "o-", ms=7, lw=2.4, color=col,
                 label=f"잡음 σ = {sg:g}")
        ax2.axhline(sg, color=col, ls=":", lw=1.8)
        j = int(np.argmin(te))
        ax2.plot([DEG[j]], [te[j]], "o", ms=13, mfc="none", mec=col, mew=2.4)
    ax2.set_yscale("log")
    ax2.set_ylim(0.7, 200)
    ax2.set_yticks([1, 2, 5, 10, 50, 100],
                   ["1", "2", "5", "10", "50", "100"])
    ax2.set_yticks([], minor=True)
    ax2.set_xticks(DEG)
    ax2.set_xlabel("다항식의 차수", fontsize=11)
    ax2.set_ylabel("검정 RMSE (로그 눈금)", fontsize=11)
    ax2.set_title("바닥은 잡음이 정한다 — 점선이 각자의 σ 다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("바닥에 닿았다는 것이 더 손댈 것이 없다는 신호다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, POLY + "floor_is_the_noise.png")

    print(f"\n[다항회귀] n = {N} · {REPS} 번 평균")
    for sg in SIGMAS:
        tr, te = out[sg]
        j = int(np.argmin(te))
        print(f"  σ = {sg:g}  가장 좋은 차수 {DEG[j]}  검정 RMSE "
              f"{te[j]:.3f}  (σ 와의 비 {te[j] / sg:.3f})")


# ════════════════════════════════════════════════════════════════════
def fig_six_points():
    """12장 개요 — 여섯 점이 쉰셋을 가린다.

    들머리가 세우는 두 가지를 그린다. 점을 묶어 평균을 내면 (1) 어떤
    곡선을 그어도 덩어리 평균이 그 가까이 오므로 **잘 맞는다는 증거가
    되지 못하고**, (2) 묶은 구간이 넓고 곡선이 가파르면 평균점이 곡선
    **위에 있지도 않다.**

    자료는 이 책이 만든 것이다. 12.5절의 phillips_six_crosses 는 원래
    그림을 그대로 보이므로, 여기서는 묶는 일 자체가 하는 일만 본다.
    """
    N = 53
    rng = np.random.default_rng(1143)
    x = rng.uniform(1.0, 11.0, N)
    true = lambda v: 10.0 / v - 0.9
    y = true(x) + rng.normal(0, 1.1, N)

    edges = np.quantile(x, np.linspace(0, 1, 7))
    edges[0] -= 1e-9
    g = np.digitize(x, edges[1:-1])
    gx = np.array([x[g == k].mean() for k in range(6)])
    gy = np.array([y[g == k].mean() for k in range(6)])

    def r2(xx, yy):
        p = np.polyfit(1 / xx, yy, 1)
        pred = np.polyval(p, 1 / xx)
        return 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()

    r2_all, r2_six = r2(x, y), r2(gx, gy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    gg = np.linspace(1.0, 11.0, 300)
    ax1.scatter(x, y, s=34, color="0.68", alpha=0.75, edgecolors="none",
                label=f"해마다 한 점 — 모두 {N} 개")
    ax1.plot(gg, true(gg), "-", lw=2.4, color="tab:blue", label="참 곡선")
    for e in edges[1:-1]:
        ax1.axvline(e, color="0.85", lw=1.0)
    ax1.plot(gx, gy, "X", ms=16, color="tab:red", zorder=5,
             label="여섯 덩어리의 평균")
    ax1.set_xlabel("실업률 꼴의 가로축", fontsize=11)
    ax1.set_ylabel("임금 상승률 꼴의 세로축", fontsize=11)
    ax1.set_title(f"묶으면 $R^2$ 이 {r2_all:.3f} 에서 {r2_six:.3f} 로 "
                  f"뛴다", fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    widths = np.linspace(0.5, 6.0, 200)
    lo = 1.2
    bias = [np.mean(true(np.linspace(lo, lo + w, 400))) - true(lo + w / 2)
            for w in widths]
    ax2.plot(widths, bias, "-", lw=2.8, color="tab:red")
    ax2.axhline(0, color="0.2", lw=1.6)
    w_page = edges[1] - edges[0]
    b_page = np.mean(true(np.linspace(lo, lo + w_page, 400))) \
        - true(lo + w_page / 2)
    ax2.plot([w_page], [b_page], "o", ms=12, color="tab:red", zorder=5)
    ax2.annotate(f"가장 왼쪽 덩어리의 폭 {w_page:.2f}\n"
                 f"평균점이 곡선보다 {b_page:+.3f} 위에 선다",
                 xy=(w_page, b_page), xytext=(2.85, 0.19), ha="left",
                 fontsize=11, color="tab:red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
    ax2.set_xlim(0.5, 6.0)
    ax2.set_xlabel("한 덩어리로 묶은 구간의 폭", fontsize=11)
    ax2.set_ylabel("덩어리 평균점이 곡선에서 벗어난 거리", fontsize=11)
    ax2.set_title("곡선이 가파른 자리를 넓게 묶으면", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("점을 묶어 평균을 내면 무언가가 사라지고 무언가는 비뚤어진다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, IDX + "six_points_hide_fifty_three.png")

    print(f"\n[장 개요] 이 책이 만든 자료 n = {N}")
    print(f"  R^2: 점 {N} 개에서 {r2_all:.4f} · 덩어리 여섯에서 "
          f"{r2_six:.4f}")
    print(f"  가장 왼쪽 덩어리 폭 {w_page:.3f} · 평균점의 치우침 "
          f"{b_page:+.4f}")


fig_eye_sees_funnels()
fig_precision_not_coefficient()
fig_pairwise_misses_it()
fig_ceiling_draws_a_line()
fig_floor_is_the_noise()
fig_six_points()
