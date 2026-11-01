"""11장 들머리 그림 일곱 장 (꾸러미 셋 · 계수 검정 넷).

만드는 파일:
  ch11/package_usage/img/when_the_trap_bites.png      절편을 빠뜨리면 언제 아픈가
  ch11/package_usage/img/test_above_train.png         시험이 훈련보다 높을 확률
  ch11/package_usage/img/one_number_five_numbers.png  한 수와 다섯 수
  ch11/testing_coefficients/img/buying_significance.png 유의성에 드는 값
  ch11/testing_coefficients/img/center_fixed_width_grows.png 중심은 그대로 폭만
  ch11/testing_coefficients/img/same_p_two_intervals.png 같은 p, 다른 구간
  ch11/testing_coefficients/img/which_column_breaks.png  어느 열이 먼저 무너지나

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

PKG = "docs/ch11/package_usage/img/"
TC = "docs/ch11/testing_coefficients/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_when_the_trap_bites():
    """11.10 statsmodels — 절편을 빠뜨리면 언제 아픈가.

    쪽의 no_intercept_trap 은 그 덫에 걸린 한 자리를 그리고,
    cond_no_warning 상자는 조건수 각주를 다룬다. 여기서는 그 둘이
    말하지 않는 것 --- **언제 아프고 언제 아프지 않은가** --- 만 본다.
    절편 없는 기울기는 Σxy/Σx² 이므로 x 를 중심에서 옮길수록 참 기울기에서
    멀어지고, x 의 평균이 0 이면 아예 어긋나지 않는다.
    """
    n, B0, B1, SIG = 200, 19.5, 1.4, 2.0
    SPREAD = 1.0
    xbars = np.linspace(0, 12, 200)
    rng = np.random.default_rng(1200)

    # 닫힌 꼴: E[Σxy/Σx²] = B1 + B0*xbar/(xbar² + s²)
    closed = B1 + B0 * xbars / (xbars ** 2 + SPREAD ** 2)

    meas = []
    for xb in (0.0, 2.0, 5.0, 10.0):
        v = []
        for _ in range(300):
            x = rng.normal(xb, SPREAD, n)
            y = B0 + B1 * x + rng.normal(0, SIG, n)
            v.append((x @ y) / (x @ x))
        meas.append((xb, np.mean(v)))

    PAGE_XB = 5.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(xbars, closed, "-", lw=2.8, color="tab:red",
             label="절편을 빠뜨린 기울기")
    ax1.axhline(B1, color="tab:blue", ls="--", lw=2.2,
                label=f"참 기울기 {B1}")
    for xb, v in meas:
        ax1.plot([xb], [v], "o", ms=11, color="0.25", zorder=5)
        ax1.text(xb, v + 0.55, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=10.5, color="0.2", fontweight="bold")
    ax1.plot([0], [B1], "o", ms=15, mfc="none", mec="tab:blue", mew=2.6,
             zorder=6)
    ax1.annotate("x 의 평균이 $0$ 이면\n아예 어긋나지 않는다", xy=(0, B1),
                 xytext=(2.6, 9.6), fontsize=11, color="tab:blue",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
    ax1.set_xlim(-0.4, 12.2)
    ax1.set_ylim(0, 12)
    ax1.set_xlabel("설명변수의 평균 위치", fontsize=11)
    ax1.set_ylabel("기울기", fontsize=11)
    ax1.set_title(f"참 절편 {B0} · 참 기울기 {B1} · 퍼짐 {SPREAD}",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    sp = np.linspace(0.3, 8, 200)
    err = B0 * PAGE_XB / (PAGE_XB ** 2 + sp ** 2)
    ax2.plot(sp, 100 * err / B1, "-", lw=2.8, color="tab:purple")
    ax2.axhline(0, color="0.2", lw=1.6)
    ax2.plot([SPREAD], [100 * B0 * PAGE_XB /
                        (PAGE_XB ** 2 + SPREAD ** 2) / B1], "o", ms=12,
             color="tab:purple", zorder=5)
    ax2.annotate("이 쪽의 자리", xy=(SPREAD, 100 * B0 * PAGE_XB /
                                  (PAGE_XB ** 2 + SPREAD ** 2) / B1),
                 xytext=(2.4, 120), fontsize=11, color="tab:purple",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:purple",
                                 lw=1.8))
    ax2.set_xlim(0.3, 8)
    ax2.set_ylim(0, 160)
    ax2.set_xlabel("설명변수의 퍼짐 (표준편차)", fontsize=11)
    ax2.set_ylabel("참 기울기에서 벗어난 비율 (%)", fontsize=11)
    ax2.set_title(f"평균을 {PAGE_XB:.0f} 로 못 박고 퍼짐만 바꾸면",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.05, "퍼짐이 평균에 견주어 커질수록 덫이 무뎌진다\n"
             "— 그래도 사라지지는 않는다", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=10.5, fontweight="bold",
             color="0.2",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("절편을 빠뜨리면 꾸러미는 오류를 내지 않고 다른 모형을 "
                 "적합한다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PKG + "when_the_trap_bites.png")

    print(f"\n[statsmodels] 참 절편 {B0} · 참 기울기 {B1}")
    for xb, v in meas:
        print(f"  x 의 평균 {xb:4.1f}  절편 없는 기울기 {v:7.4f}  "
              f"(닫힌 꼴 {B1 + B0 * xb / (xb ** 2 + SPREAD ** 2):.4f})")


# ════════════════════════════════════════════════════════════════════
def fig_test_above_train():
    """11.10 sklearn — 시험이 훈련보다 높은 일은 얼마나 흔한가.

    쪽의 train_vs_cv 는 분할 200 번의 분포를, negative_r2 상자는 음수가
    나오는 자리를 다룬다. 여기서는 들머리의 마지막 문단 --- **「시험이
    훈련보다 높다」를 「과적합이 없다」로 읽으면 안 된다** --- 를 확률로
    재어 본다. 모형이 맞고 과적합이 전혀 없어도 흔히 일어나는 일이다.
    """
    N_PAGE, REPS = 100, 4000
    SHARES = [0.1, 0.2, 0.3, 0.4, 0.5]
    rng = np.random.default_rng(1201)

    def once(n, share, gen):
        x = gen.normal(0, 1, n)
        y = 2 + 1.5 * x + gen.normal(0, 1, n)
        m = int(n * share)
        idx = gen.permutation(n)
        te, tr = idx[:m], idx[m:]
        r = stats.linregress(x[tr], y[tr])

        def r2(ii):
            pred = r.intercept + r.slope * x[ii]
            return 1 - ((y[ii] - pred) ** 2).sum() / \
                ((y[ii] - y[ii].mean()) ** 2).sum()
        return r2(tr), r2(te)

    share_prob = []
    for s in SHARES:
        v = np.array([once(N_PAGE, s, rng) for _ in range(REPS)])
        share_prob.append((v[:, 1] > v[:, 0]).mean())
    v20 = np.array([once(N_PAGE, 0.2, rng) for _ in range(REPS)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.hist(v20[:, 1], bins=70, color="tab:blue", alpha=0.55,
             label="시험 $R^2$")
    ax1.hist(v20[:, 0], bins=70, color="0.5", alpha=0.6,
             label="훈련 $R^2$")
    ax1.axvline(0.9855, color="tab:red", lw=2.6)
    ax1.text(0.9855 - 0.004, ax1.get_ylim()[1] * 0.95,
             "이 쪽이 본 $0.9855$", ha="right", va="top", fontsize=11,
             color="tab:red", fontweight="bold")
    ax1.set_xlabel("$R^2$", fontsize=11)
    ax1.set_ylabel("횟수", fontsize=11)
    ax1.set_title(f"$n = {N_PAGE}$ 을 $80/20$ 으로 가르기를 {REPS:,} 번",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot([100 * s for s in SHARES], [100 * p for p in share_prob],
             "o-", ms=10, lw=2.8, color="tab:purple")
    for s, p in zip(SHARES, share_prob):
        ax2.text(100 * s, 100 * p + 1.2, f"{100 * p:.1f}%", ha="center",
                 va="bottom", fontsize=11, color="tab:purple",
                 fontweight="bold")
    ax2.axhline(50, color="0.3", ls="--", lw=1.8)
    ax2.text(49, 51, "절반", ha="right", va="bottom", fontsize=10.5,
             color="0.3", fontweight="bold")
    ax2.set_xlim(5, 55)
    ax2.set_ylim(0, 60)
    ax2.set_xlabel("떼어 낸 시험자료의 비율 (%)", fontsize=11)
    ax2.set_ylabel("시험 $R^2$ 가 훈련 $R^2$ 보다 높을 확률 (%)",
                   fontsize=11)
    ax2.set_title("모형이 맞고 과적합이 전혀 없는 자료에서", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("「시험이 훈련보다 높다」는 과적합이 없다는 뜻이 아니다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PKG + "test_above_train.png")

    print(f"\n[sklearn] n = {N_PAGE} · 각 {REPS:,} 번")
    print(f"  80/20 분할: 시험 R^2 평균 {v20[:,1].mean():.4f} · 표준편차 "
          f"{v20[:,1].std(ddof=1):.4f} · 최소 {v20[:,1].min():.4f} · "
          f"최대 {v20[:,1].max():.4f}")
    for s, p in zip(SHARES, share_prob):
        print(f"  시험 비율 {100*s:4.0f}%  시험이 더 높을 확률 "
              f"{100*p:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_one_number_five_numbers():
    """11.10 두 꾸러미 비교 — 한 수와 다섯 수.

    쪽의 two_libraries 는 두 결과가 1e-14 까지 같다는 것을,
    boston_b_variable 상자는 그 자료의 내력을 다룬다. 여기서는 들머리의
    셋째 문단 --- **훈련 R^2 하나와 겹마다의 다섯 수는 쓸모가 전혀
    다르다** --- 만 본다. 겹을 늘리면 가장 나쁜 겹이 어디까지 내려가는지
    센다.
    """
    n = 200
    rng = np.random.default_rng(1202)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    y = 1 + 2 * x1 - 1.2 * x2 + 0.4 * x3 + rng.normal(0, 2.2, n)
    X = np.column_stack([np.ones(n), x1, x2, x3])

    def fit_score(tr, te):
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred = X[te] @ b
        return 1 - ((y[te] - pred) ** 2).sum() / \
            ((y[te] - y[te].mean()) ** 2).sum()

    b_all = np.linalg.lstsq(X, y, rcond=None)[0]
    r2_train = 1 - ((y - X @ b_all) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    KS = [2, 5, 10, 20, 50]   # 하나빼기는 겹 안의 분산이 0 이라 뺀다
    worst, best = [], []
    folds5 = None
    for K in KS:
        idx = np.random.default_rng(3).permutation(n)
        parts = np.array_split(idx, K)
        sc = [fit_score(np.setdiff1d(idx, p), p) for p in parts]
        worst.append(min(sc))
        best.append(max(sc))
        if K == 5:
            folds5 = sc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(len(folds5))
    cols = ["tab:red" if v < 0 else "tab:blue" for v in folds5]
    ax1.bar(xs, folds5, width=0.56, color=cols, alpha=0.88)
    ax1.axhline(r2_train, color="0.2", ls="--", lw=2.2)
    ax1.text(4.42, r2_train + 0.012, f"훈련 $R^2$ 하나 = {r2_train:.4f}",
             ha="right", va="bottom", fontsize=11, color="0.2",
             fontweight="bold")
    ax1.axhline(0, color="0.15", lw=1.6)
    for x_, v in zip(xs, folds5):
        ax1.text(x_, v + np.sign(v) * 0.012, f"{v:+.4f}", ha="center",
                 va="bottom" if v > 0 else "top", fontsize=11,
                 fontweight="bold", color="0.15")
    ax1.set_xticks(xs, [f"{i + 1} 겹" for i in xs], fontsize=11.5)
    ax1.set_ylim(min(min(folds5), 0) - 0.12, max(r2_train, max(folds5)) + 0.13)
    ax1.set_ylabel("$R^2$", fontsize=11)
    ax1.set_title("$5$ 겹 교차검증 — 겹마다의 다섯 수", fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(KS, worst, "o-", ms=9, lw=2.6, color="tab:red",
             label="가장 나쁜 겹")
    ax2.plot(KS, best, "s-", ms=8, lw=2.6, color="tab:blue",
             label="가장 좋은 겹")
    ax2.axhline(r2_train, color="0.2", ls="--", lw=2.0)
    ax2.text(50, r2_train + 0.35, "훈련 $R^2$", ha="right", va="bottom",
             fontsize=10.5, color="0.2", fontweight="bold")
    ax2.axhline(0, color="0.15", lw=1.6)
    ax2.set_xscale("log")
    ax2.set_xticks(KS, [str(k) for k in KS])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(min(worst) * 1.12, 1.15)
    ax2.set_xlabel("겹의 수 K (로그 눈금)", fontsize=11)
    ax2.set_ylabel("$R^2$", fontsize=11)
    ax2.set_title("겹을 잘게 쪼갤수록 양 끝이 벌어진다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="center left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("한 수만 보고 있었다면 그런 겹이 있다는 사실조차 몰랐을 것이다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PKG + "one_number_five_numbers.png")

    print(f"\n[두 꾸러미] n = {n} · 훈련 R^2 {r2_train:.4f}")
    print(f"  5 겹: " + " · ".join(f"{v:+.4f}" for v in folds5))
    for K, w, b in zip(KS, worst, best):
        print(f"  K = {K:>3d}  가장 나쁜 겹 {w:+.4f}  가장 좋은 겹 {b:+.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_buying_significance():
    """11.11 유의성 — 문턱을 넘기는 데 드는 값.

    쪽의 stat_vs_practical 은 유의성과 크기의 네 조합을,
    ablation_is_ftest 상자는 다른 이야기를 다룬다. 여기서는 들머리의
    둘째 문단 --- **t 가 sqrt(n) 에 비례하므로 표본만 늘리면 거의 모든
    계수가 유의해진다** --- 를 식으로 그린다. 문턱을 넘는 데 드는 표본이
    (1.96 sigma / beta)^2 이다.
    """
    SIG = 1.0
    betas = np.logspace(np.log10(0.002), np.log10(1.0), 300)
    need = (1.96 * SIG / betas) ** 2
    B_PAGE = 0.02

    ns = np.logspace(2, 7, 300)
    t_page = B_PAGE * np.sqrt(ns) / SIG
    half = 1.96 * SIG / np.sqrt(ns)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(betas, need, "-", lw=2.8, color="tab:red")
    ax1.plot([B_PAGE], [(1.96 * SIG / B_PAGE) ** 2], "o", ms=13,
             color="tab:red", zorder=5)
    ax1.annotate(f"참 효과 ${B_PAGE}$ 이면\n"
                 f"{(1.96 * SIG / B_PAGE) ** 2:,.0f} 개면 된다",
                 xy=(B_PAGE, (1.96 * SIG / B_PAGE) ** 2),
                 xytext=(0.06, 2e5), fontsize=11.5, color="tab:red",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xticks([0.002, 0.01, 0.05, 0.2, 1.0],
                   ["0.002", "0.01", "0.05", "0.2", "1"])
    ax1.set_xticks([], minor=True)
    ax1.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6],
                   ["100", "1,000", "10,000", "100,000", "1,000,000"])
    ax1.set_yticks([], minor=True)
    ax1.set_xlabel("참 효과의 크기 (로그 눈금)", fontsize=11)
    ax1.set_ylabel("$5\\%$ 에서 유의해지는 데 드는 표본 (로그 눈금)",
                   fontsize=11)
    ax1.set_title(f"$(1.96\\sigma/\\beta)^2$ — 효과가 절반이면 표본은 "
                  f"네 배", fontsize=12.5)
    ax1.grid(alpha=0.25, which="major")

    ax2.plot(ns, t_page, "-", lw=2.8, color="tab:blue", label="$t$ 값")
    ax2.axhline(1.96, color="0.25", ls="--", lw=1.8)
    ax2.text(1e7, 2.2, "문턱 $1.96$", ha="right", va="bottom", fontsize=10.5,
             color="0.25", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e2, 1e7)
    ax2.set_ylim(0.1, 80)
    ax2.set_yticks([0.1, 0.5, 1.96, 10, 50],
                   ["0.1", "0.5", "1.96", "10", "50"])
    ax2.set_yticks([], minor=True)
    ax2.set_xlabel("표본 크기 n (로그 눈금)", fontsize=11)
    ax2.set_ylabel("$t$ 값 (로그 눈금)", fontsize=11, color="tab:blue")
    ax2.tick_params(axis="y", colors="tab:blue")
    ax2.set_title(f"참 효과를 ${B_PAGE}$ 로 못 박아 두고 표본만 키우면",
                  fontsize=12.5)
    ax2.grid(alpha=0.25, which="major")
    ax3 = ax2.twinx()
    ax3.plot(ns, 100 * half / B_PAGE, "-", lw=2.8, color="tab:red")
    ax3.set_yscale("log")
    ax3.set_ylim(1, 1e5)
    ax3.set_yticks([10, 100, 1000, 10000],
                   ["10", "100", "1,000", "10,000"])
    ax3.set_yticks([], minor=True)
    ax3.set_ylabel("구간의 반폭이 효과의 몇 % 인가", fontsize=11,
                   color="tab:red")
    ax3.tick_params(axis="y", colors="tab:red")
    n_sig = (1.96 * SIG / B_PAGE) ** 2
    ax2.axvline(n_sig, color="0.4", ls=":", lw=1.8)
    ax2.text(n_sig * 1.3, 0.14, f"{n_sig:,.0f} 개", ha="left", va="bottom",
             fontsize=10.5, color="0.3", fontweight="bold")

    fig.suptitle("세상은 조금도 변하지 않았고 변한 것은 자료의 양뿐이다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, TC + "buying_significance.png")

    print(f"\n[유의성] sigma = {SIG}")
    for b in (0.005, 0.01, 0.02, 0.05, 0.2):
        print(f"  참 효과 {b:<6g} → 표본 {(1.96 * SIG / b) ** 2:12,.0f} 개")
    for nn in (1000, 9604, 100_000, 1_000_000):
        print(f"  n = {nn:>9,}  t = {B_PAGE * np.sqrt(nn) / SIG:7.2f}  "
              f"구간 반폭 {1.96 * SIG / np.sqrt(nn):.6f}")


# ════════════════════════════════════════════════════════════════════
def fig_center_fixed_width_grows():
    """11.11 가설검정 — 중심은 그대로이고 폭만 벌어진다.

    쪽의 t_stat_and_vif 는 VIF 와 t 의 관계를, table2_fallacy 상자는
    표 읽기를 다룬다. 여기서는 들머리의 셋째 문단 --- **추정값은
    처음부터 끝까지 2.574 였고 달라진 것은 폭뿐이다** --- 를 구간의
    사다리로 그린다.
    """
    BHAT, SE0, N, K = 2.574, 0.332, 100, 2
    tcrit = stats.t.ppf(0.975, N - K - 1)
    VIFS = [1, 2, 4, 9, 16, 25]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    for i, v in enumerate(VIFS):
        se = SE0 * np.sqrt(v)
        lo, hi = BHAT - tcrit * se, BHAT + tcrit * se
        sig = lo > 0
        col = "tab:blue" if sig else "tab:red"
        y = len(VIFS) - 1 - i
        ax1.plot([lo, hi], [y, y], "-", lw=7.0, color=col, alpha=0.85,
                 solid_capstyle="butt")
        ax1.plot([BHAT], [y], "o", ms=11, mfc="white", mec="0.15", mew=2.4,
                 zorder=5)
        ax1.text(hi + 0.08, y, f"$t = {BHAT / se:.2f}$ · "
                 f"{'유의하다' if sig else '유의하지 않다'}", ha="left",
                 va="center", fontsize=11, color=col, fontweight="bold")
    ax1.axvline(0, color="0.15", lw=2.2)
    ax1.set_yticks(range(len(VIFS))[::-1],
                   [f"VIF = {v}\n표준오차 {SE0 * np.sqrt(v):.3f}"
                    for v in VIFS], fontsize=10.5)
    ax1.set_xlim(-2.6, 8.6)
    ax1.set_ylim(-0.7, len(VIFS) - 0.3)
    ax1.set_xlabel("계수", fontsize=11)
    ax1.set_title(f"추정값을 ${BHAT}$ 로 못 박고 VIF 만 올렸다 "
                  f"($n = {N}$)", fontsize=12.5)
    ax1.grid(axis="x", alpha=0.25)
    ax1.text(0.03, 0.04, "가운데 동그라미가 추정값이다\n"
             "— 한 번도 움직이지 않았다", transform=ax1.transAxes,
             ha="left", va="bottom", fontsize=11, fontweight="bold",
             color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    vv = np.linspace(1, 30, 300)
    ax2.plot(vv, BHAT / (SE0 * np.sqrt(vv)), "-", lw=2.8, color="tab:purple")
    ax2.axhline(tcrit, color="0.25", ls="--", lw=1.8)
    ax2.text(29.5, tcrit + 0.25, f"임계값 {tcrit:.3f}", ha="right",
             va="bottom", fontsize=10.5, color="0.25", fontweight="bold")
    v_cross = (BHAT / (SE0 * tcrit)) ** 2
    ax2.axvline(v_cross, color="tab:red", ls=":", lw=1.8)
    ax2.text(v_cross + 0.6, 6.8, f"VIF = {v_cross:.1f} 에서\n"
             "판정이 뒤집힌다", ha="left", va="top", fontsize=11,
             color="tab:red", fontweight="bold")
    ax2.set_xlim(1, 30)
    ax2.set_ylim(0, 8.5)
    ax2.set_xlabel("VIF", fontsize=11)
    ax2.set_ylabel("$t$ 값", fontsize=11)
    ax2.set_title("효과가 작아진 것이 아니라 분간할 정보가 없어진 것이다",
                  fontsize=12)
    ax2.grid(alpha=0.25)

    fig.suptitle("「유의하지 않다」는 효과가 없다는 말이 아니라 묻지 "
                 "않기로 했다는 말이다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, TC + "center_fixed_width_grows.png")

    print(f"\n[가설검정] 추정값 {BHAT} 고정 · n = {N}")
    for v in VIFS:
        se = SE0 * np.sqrt(v)
        print(f"  VIF {v:>3d}  SE {se:.4f}  t {BHAT / se:5.2f}  "
              f"구간 ({BHAT - tcrit * se:+.4f}, {BHAT + tcrit * se:+.4f})")
    print(f"  판정이 뒤집히는 VIF {v_cross:.2f}")


# ════════════════════════════════════════════════════════════════════
def fig_same_p_two_intervals():
    """11.11 p 값과 신뢰구간 — 같은 p 값에 전혀 다른 구간.

    쪽의 pvalue_ci_duality 는 p 값이 구간의 끝이 0 에 닿는 유의수준이라는
    쌍대성을, banned_both 상자는 학술지 이야기를 다룬다. 여기서는 들머리의
    마지막 문단 --- **p = 0.008 짜리 계수가 (1.2, 7.2) 일 수도 (4.1, 4.3)
    일 수도 있다** --- 만 본다.
    """
    DF, P = 40, 0.00783
    t_fixed = stats.t.isf(P / 2, DF)
    tcrit = stats.t.ppf(0.975, DF)
    SES = [1.5, 0.8, 0.35, 0.12, 0.04]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    for i, se in enumerate(SES):
        b = t_fixed * se
        lo, hi = b - tcrit * se, b + tcrit * se
        y = len(SES) - 1 - i
        ax1.plot([lo, hi], [y, y], "-", lw=7.0, color="tab:blue", alpha=0.85,
                 solid_capstyle="butt")
        ax1.plot([b], [y], "o", ms=11, mfc="white", mec="0.15", mew=2.4,
                 zorder=5)
        ax1.text(hi + 0.12, y, f"$({lo:.3f},\\ {hi:.3f})$", ha="left",
                 va="center", fontsize=11, color="0.15", fontweight="bold")
    ax1.axvline(0, color="0.15", lw=2.2)
    ax1.set_yticks(range(len(SES))[::-1],
                   [f"추정값 {t_fixed * se:.3f}\n표준오차 {se}"
                    for se in SES], fontsize=10.5)
    ax1.set_xlim(-0.6, 11.5)
    ax1.set_ylim(-0.7, len(SES) - 0.3)
    ax1.set_xlabel("계수", fontsize=11)
    ax1.set_title(f"다섯 줄 모두 $p = {P}$ 다 (자유도 {DF})", fontsize=12.5)
    ax1.grid(axis="x", alpha=0.25)
    ax1.text(0.97, 0.05, "$p$ 값은 「$0$ 에서 얼마나 멀리 있나」만 말한다\n"
             "구간은 「얼마나 큰 효과까지 허락하는가」를 더한다",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=11,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ses = np.logspace(np.log10(0.02), np.log10(2.0), 300)
    width = 2 * tcrit * ses
    ax2.plot(ses, width, "-", lw=2.8, color="tab:purple")
    for se in SES:
        ax2.plot([se], [2 * tcrit * se], "o", ms=10, color="tab:purple",
                 zorder=5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks([0.02, 0.1, 0.5, 2.0], ["0.02", "0.1", "0.5", "2"])
    ax2.set_xticks([], minor=True)
    ax2.set_yticks([0.1, 0.5, 2, 8], ["0.1", "0.5", "2", "8"])
    ax2.set_yticks([], minor=True)
    ax2.set_xlabel("표준오차 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("$95\\%$ 구간의 폭 (로그 눈금)", fontsize=11)
    ax2.set_title("$p$ 값이 같아도 폭은 어디든 될 수 있다", fontsize=12.5)
    ax2.grid(alpha=0.25, which="major")

    fig.suptitle("$p$ 값 하나로는 결정을 내릴 수 없다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, TC + "same_p_two_intervals.png")

    print(f"\n[p 값과 구간] p = {P} 고정 · 자유도 {DF} · t = {t_fixed:.4f}")
    for se in SES:
        b = t_fixed * se
        print(f"  SE {se:<5g} 추정값 {b:7.4f}  구간 "
              f"({b - tcrit * se:+.4f}, {b + tcrit * se:+.4f})  폭 "
              f"{2 * tcrit * se:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_which_column_breaks():
    """11.11 회귀표 — 자료가 나빠질 때 어느 열이 먼저 흔들리나.

    쪽의 regression_table_t 는 손으로 만든 표와 꾸러미의 표를 맞대고,
    forbes_residuals 상자는 옛 자료로 같은 일을 한다. 여기서는 들머리의
    마지막 문단 --- **어느 열이 먼저 흔들리는지** --- 만 본다. 설계행렬이
    특이에 가까워질수록 두 풀이(정규방정식과 QR)의 답이 벌어지는데,
    계수보다 표준오차가 먼저 벌어진다.
    """
    n = 120
    EPS = [1.0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    rng = np.random.default_rng(1203)

    rows = []
    for e in EPS:
        a = rng.normal(0, 1, n)
        b = a + e * rng.normal(0, 1, n)   # e 가 작을수록 두 열이 같아진다
        X = np.column_stack([np.ones(n), a, b])
        y = 1 + 0.8 * a - 0.5 * b + rng.normal(0, 1, n)
        cond = np.linalg.cond(X)
        # 두 가지 풀이
        beta_ne = np.linalg.solve(X.T @ X, X.T @ y)
        beta_qr = np.linalg.lstsq(X, y, rcond=None)[0]
        res_ne = y - X @ beta_ne
        res_qr = y - X @ beta_qr
        s2_ne = (res_ne ** 2).sum() / (n - 3)
        s2_qr = (res_qr ** 2).sum() / (n - 3)
        se_ne = np.sqrt(np.diag(s2_ne * np.linalg.inv(X.T @ X)))
        q, r = np.linalg.qr(X)
        ri = np.linalg.inv(r)
        se_qr = np.sqrt(np.diag(s2_qr * (ri @ ri.T)))
        d_b = np.max(np.abs(beta_ne - beta_qr) / np.abs(beta_qr))
        d_se = np.max(np.abs(se_ne - se_qr) / np.abs(se_qr))
        d_t = np.max(np.abs(beta_ne / se_ne - beta_qr / se_qr)
                     / np.abs(beta_qr / se_qr))
        rows.append((cond, d_b, d_se, d_t))

    rows = np.array(rows)
    FLOOR = 1e-17
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    for j, (nm, col) in enumerate((("계수", "tab:blue"),
                                   ("표준오차", "tab:red"),
                                   ("$t$ 값", "tab:purple"))):
        ax1.plot(rows[:, 0], np.maximum(rows[:, j + 1], FLOOR), "o-", ms=9,
                 lw=2.6, color=col, label=nm)
    ref = np.logspace(0, np.log10(rows[-1, 0]), 50)
    ax1.plot(ref, 1e-16 * ref ** 2, ":", lw=2.4, color="0.35",
             label="1e-16 곱하기 조건수의 제곱")
    ax1.axhline(1e-16, color="0.55", ls="--", lw=1.6)
    ax1.text(rows[-1, 0], 1.6e-16, "배정밀도의 반올림 한 단위", ha="right",
             va="bottom", fontsize=10.5, color="0.4", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(FLOOR, 1e-1)
    # 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에 없는 U+2212 를 부른다
    ax1.set_yticks([1e-16, 1e-13, 1e-10, 1e-7, 1e-4, 1e-1],
                   ["1e-16", "1e-13", "1e-10", "1e-7", "1e-4", "1e-1"])
    ax1.set_yticks([], minor=True)
    ax1.set_xticks([1, 1e2, 1e4, 1e6],
                   ["1", "100", "10,000", "1,000,000"])
    ax1.set_xticks([], minor=True)
    ax1.set_xlabel("설계행렬의 조건수 (로그 눈금)", fontsize=11)
    ax1.set_ylabel("두 풀이가 어긋난 상대 크기 (로그 눈금)", fontsize=11)
    ax1.set_title(f"$n = {n}$ · 정규방정식과 QR 로 각각 풀었다",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25, which="major")

    conds = np.logspace(0, 9, 300)
    ax2.plot(conds, 1e-16 * conds, "-", lw=2.8, color="tab:blue",
             label="계수 — 조건수에 비례")
    ax2.plot(conds, 1e-16 * conds ** 2, "-", lw=2.8, color="tab:red",
             label="정규방정식 — 조건수의 제곱에 비례")
    ax2.axhline(1e-3, color="0.25", ls="--", lw=1.8)
    ax2.text(1e9, 1.4e-3, "소수 셋째 자리", ha="right", va="bottom",
             fontsize=10.5, color="0.25", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1e-1)
    ax2.set_yticks([1e-16, 1e-13, 1e-10, 1e-7, 1e-4, 1e-1],
                   ["1e-16", "1e-13", "1e-10", "1e-7", "1e-4", "1e-1"])
    ax2.set_yticks([], minor=True)
    ax2.set_xticks([1, 1e3, 1e6, 1e9], ["1", "1,000", "1,000,000",
                                        "1,000,000,000"])
    ax2.set_xticks([], minor=True)
    ax2.set_xlabel("조건수 (로그 눈금)", fontsize=11)
    ax2.set_ylabel("들어올 수 있는 상대오차 (로그 눈금)", fontsize=11)
    ax2.set_title("두 가지 법칙을 견주면", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25, which="major")

    fig.suptitle("$\\mathbf{X}^\\top\\mathbf{X}$ 를 만드는 순간 "
                 "조건수가 제곱되어 들어온다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, TC + "which_column_breaks.png")

    print(f"\n[회귀표] n = {n}")
    print("   조건수        계수         표준오차       t 값")
    for c, a, b, t in rows:
        print(f"  {c:11.4g}  {a:11.4g}  {b:11.4g}  {t:11.4g}")


fig_when_the_trap_bites()
fig_test_above_train()
fig_one_number_five_numbers()
fig_buying_significance()
fig_center_fixed_width_grows()
fig_same_p_two_intervals()
fig_which_column_breaks()
