"""12장 들머리 그림 다섯 장 (가정 점검 다섯 쪽).

만드는 파일:
  ch12/assumptions/img/n_does_not_help.png        표본을 키워도 사라지지 않는다
  ch12/assumptions/img/pretest_vs_always.png      묻고 선택하기와 처음부터 로버스트
  ch12/assumptions/img/effective_sample_size.png  예순 개가 다섯 개 값어치
  ch12/assumptions/img/misses_on_one_side.png     빗나간 구간이 한쪽에 몰린다
  ch12/assumptions/img/one_voice_counted_thirty.png 한 사람이 쥔 몫과 눈금의 덫

쪽마다 본문 그림이 이미 들머리의 수를 그리고 있으므로, 여기서는 그
그림들이 건드리지 않는 각만 골랐다. 고른 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch12/assumptions/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_n_does_not_help():
    """11.3 선형성 — 표본을 키워도 사라지지 않는 유일한 실패.

    쪽의 linearity_curvature 는 휜 관계에 직선을 그은 모습과 잔차의 모양을
    그린다. 여기서는 그 그림이 말하지 않는 것 --- **표본을 키우면 어떻게
    되는가** --- 만 본다. 다른 가정 위반은 n 이 커지면 적어도 증상이
    또렷해지는데, 이것은 반대다.

    x 가 [0, 10] 균등이고 참 관계가 5 + 3x - 0.30x^2 이면 최적 직선의
    모집단 기울기가 정확히 0 이다 (3 - 0.3 * Cov(x^2,x)/Var(x) = 3 - 3).
    그래서 n 을 아무리 키워도 기울기는 0 에 모인다 --- 자료 속 관계가
    아무리 강해도 출력표에는 아무것도 남지 않는다. 기각률도 n 과 무관하게
    10% 언저리에 머문다(5% 가 아닌 까닭은 빠뜨린 2 차항이 오차에 섞여
    들어가 t 검정의 귀무분포까지 어긋나기 때문이다).
    """
    A, B, C, SIG = 5.0, 3.0, -0.30, 1.4
    NS = [50, 200, 1000, 5000, 20000]
    REPS = 400
    rng = np.random.default_rng(1120)
    BINS = [(0, 2, "x < 2"), (4, 6, "4 < x < 6"), (8, 10, "x > 8")]

    slope_m, slope_s, rej = [], [], []
    bin_means = {}
    for n in NS:
        sl = np.empty(REPS)
        hit = np.empty(REPS, bool)
        acc = np.zeros((REPS, len(BINS)))
        for i in range(REPS):
            x = rng.uniform(0, 10, n)
            y = A + B * x + C * x ** 2 + rng.normal(0, SIG, n)
            res = stats.linregress(x, y)
            sl[i] = res.slope
            hit[i] = res.pvalue < 0.05
            r = y - (res.intercept + res.slope * x)
            for j, (lo, hi, _) in enumerate(BINS):
                acc[i, j] = r[(x >= lo) & (x < hi)].mean()
        slope_m.append(sl.mean())
        slope_s.append(sl.std(ddof=1))
        rej.append(hit.mean())
        bin_means[n] = acc.mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    width = 0.24
    SHOW = [50, 1000, 20000]
    cols = ["tab:blue", "tab:purple", "tab:red"]
    xs = np.arange(len(BINS))
    for k, (n, col) in enumerate(zip(SHOW, cols)):
        ax1.bar(xs + (k - 1) * width, bin_means[n], width=width * 0.92,
                color=col, alpha=0.88, label=f"n = {n:,}")
    ax1.axhline(0, color="0.15", lw=1.6)
    ax1.axhspan(-SIG, SIG, color="0.6", alpha=0.18)
    ax1.text(2.42, SIG + 0.08, f"잡음의 표준편차 ±{SIG}", ha="right",
             va="bottom", fontsize=10.5, color="0.3", fontweight="bold")
    for k, (n, col) in enumerate(zip(SHOW, cols)):
        for j, v in enumerate(bin_means[n]):
            ax1.text(j + (k - 1) * width, v + np.sign(v) * 0.12,
                     f"{v:+.2f}", ha="center",
                     va="bottom" if v > 0 else "top", fontsize=9.5,
                     color=col, fontweight="bold")
    ax1.set_xticks(xs, [b[2] for b in BINS], fontsize=12)
    ax1.set_ylim(-3.6, 4.0)
    ax1.set_ylabel("구간별 잔차의 평균", fontsize=11)
    ax1.set_title(f"잔차에 남은 모양은 표본과 무관하다 (각 {REPS} 번 평균)",
                  fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    ax2.errorbar(NS, slope_m, yerr=slope_s, fmt="o-", ms=9, lw=2.4,
                 capsize=6, color="tab:red", label="직선의 기울기")
    ax2.axhline(0, color="0.15", lw=1.8, ls="--")
    ax2.set_xscale("log")
    ax2.set_xticks(NS, [f"{n:,}" for n in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_xlabel("표본 크기 n (로그 척도)", fontsize=11)
    ax2.set_ylabel("기울기 추정값", fontsize=11, color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")
    ax2.set_ylim(-0.75, 0.95)
    ax2.grid(alpha=0.25)
    ax3 = ax2.twinx()
    ax3.plot(NS, [100 * r for r in rej], "s--", ms=8, lw=2.2,
             color="tab:blue")
    ax3.axhline(5, color="0.45", ls=":", lw=1.8)
    ax3.set_ylim(0, 30)
    ax3.set_ylabel("$5\\%$ 수준의 기각률 (%)", fontsize=11, color="tab:blue")
    ax3.tick_params(axis="y", colors="tab:blue")
    ax3.text(20000, 23.5, f"기각률도 n 과 무관하게\n"
             f"{100 * rej[-1]:.0f}% 언저리에 머문다", ha="right", va="bottom",
             fontsize=10.5, color="tab:blue", fontweight="bold")
    ax2.set_title("표본을 사백 배로 키워도 기울기는 $0$ 이다", fontsize=12.5)

    fig.suptitle("휜 관계에 직선을 그으면 표본을 키울수록 「아무것도 없다」가 "
                 "더 또렷해진다", fontsize=13.5, y=1.0)
    plt.tight_layout()
    save(fig, "n_does_not_help.png")

    print("\n[선형성] n 을 키우면")
    for n, m, s, r in zip(NS, slope_m, slope_s, rej):
        print(f"  n = {n:>6,}  기울기 {m:+.4f} (sd {s:.4f})  "
              f"5% 기각률 {100 * r:5.1f}%  구간별 잔차 평균 "
              f"{np.array2string(bin_means[n], precision=2, sign='+')}")


# ════════════════════════════════════════════════════════════════════
def fig_pretest_vs_always():
    """11.3 등분산 — 묻고 고르는 길과 처음부터 로버스트인 길.

    쪽의 homoscedasticity_se 는 이분산 한 자리에서 표준오차가 얼마나
    작게 찍히는지를, sandwich_coverage 상자는 로버스트 공식의 포함률을
    그린다. 여기서는 그 둘이 건드리지 않는 것 --- 실제로 흔히 쓰는
    **「검정하고 통과하면 고전 공식」**이라는 세 번째 길이 어떻게 되는지 ---
    만 본다. 이분산의 세기를 0 에서 키워 가며 세 길을 나란히 잰다.

    재어 보면 세 번째 길이 **로버스트 쪽에 포개진다.** n = 60 에서 검정의
    힘이 충분해 거의 언제나 로버스트를 선택하기 때문이다. 그러면 묻는 단계는
    버는 것이 없고, 등분산일 때 로버스트가 치르는 값도 5.0 대 5.2 로
    거의 없다. 쪽이 적어 둔 「묻지 말고 처음부터 로버스트」가 그 결론이다.
    """
    n, REPS = 60, 4000
    GAMMA = np.linspace(0, 1.2, 9)
    rng = np.random.default_rng(1121)
    x = np.linspace(0.2, 4.0, n)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()

    def one(g, gen):
        sd = np.exp(g * (x - x.mean()) / x.std())
        y = 1.0 + 0.0 * x + gen.normal(0, sd, n)       # 참 기울기는 0
        b = (xc @ (y - y.mean())) / sxx
        res = y - (y.mean() + b * xc)
        s2 = (res ** 2).sum() / (n - 2)
        se_cl = np.sqrt(s2 / sxx)
        h = 1 / n + xc ** 2 / sxx
        se_hc3 = np.sqrt(((xc ** 2) * (res / (1 - h)) ** 2).sum()) / sxx
        # 화이트 검정 대용 --- 잔차제곱을 x 에 회귀한 보조회귀의 유의성
        r2 = stats.linregress(x, res ** 2).rvalue ** 2
        bp_p = stats.chi2.sf(n * r2, 1)
        return b, se_cl, se_hc3, bp_p

    tcrit = stats.t.ppf(0.975, n - 2)
    out = {k: [] for k in ("고전", "사전검정", "로버스트")}
    for g in GAMMA:
        rows = np.array([one(g, rng) for _ in range(REPS)])
        b, se_cl, se_hc3, p = rows.T
        out["고전"].append((np.abs(b) > tcrit * se_cl).mean())
        out["로버스트"].append((np.abs(b) > tcrit * se_hc3).mean())
        use_cl = p >= 0.05
        se_pick = np.where(use_cl, se_cl, se_hc3)
        out["사전검정"].append((np.abs(b) > tcrit * se_pick).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.2, 1]})

    STYLE = [("고전", "tab:red", "o-", "고전 공식"),
             ("사전검정", "tab:orange", "s-", "검정하고 통과하면 고전 공식"),
             ("로버스트", "tab:blue", "D-", "묻지 말고 처음부터 로버스트")]
    for key, col, mk, lab in STYLE:
        ax1.plot(GAMMA, [100 * v for v in out[key]], mk, ms=8, lw=2.4,
                 color=col, label=lab)
    ax1.axhline(5, color="0.2", ls="--", lw=1.8)
    ax1.text(0.62, 3.0, "약속한 $5\\%$", ha="center", va="top",
             fontsize=10.5, color="0.2", fontweight="bold")
    ax1.annotate("주황과 파랑이 포개진다\n— 묻는 단계가 버는 것이 없다",
                 xy=(0.75, 100 * out["로버스트"][5]), xytext=(0.30, 17.5),
                 fontsize=11, color="tab:blue", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.6))
    ax1.set_xlim(-0.03, 1.23)
    ax1.set_ylim(0, 24)
    ax1.set_xlabel("이분산의 세기 (0 이면 등분산)", fontsize=11)
    ax1.set_ylabel("참인 귀무가설의 기각률 (%)", fontsize=11)
    ax1.set_title(f"$n = {n}$ · 각 {REPS:,} 번", fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    i0, i1 = 0, len(GAMMA) - 1
    xs = np.arange(3)
    v0 = [100 * out[k][i0] for k, *_ in STYLE]
    v1 = [100 * out[k][i1] for k, *_ in STYLE]
    ax2.bar(xs - 0.19, v0, width=0.36, color="0.68", alpha=0.9,
            label="등분산일 때 (치르는 값)")
    ax2.bar(xs + 0.19, v1, width=0.36,
            color=[c for _, c, *_ in STYLE], alpha=0.9,
            label=f"세기 {GAMMA[i1]:.1f} 일 때")
    for x_, v in zip(xs - 0.19, v0):
        ax2.text(x_, v + 0.4, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=10.5, color="0.3", fontweight="bold")
    for x_, v in zip(xs + 0.19, v1):
        ax2.text(x_, v + 0.4, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=10.5, color="0.15", fontweight="bold")
    ax2.axhline(5, color="0.2", ls="--", lw=1.8)
    ax2.set_xticks(xs, ["고전", "사전검정", "로버스트"], fontsize=12)
    ax2.set_ylim(0, 26)
    ax2.set_ylabel("기각률 (%)", fontsize=11)
    ax2.set_title("등분산에서 치르는 값과 이분산에서 버는 값",
                  fontsize=12.5)
    ax2.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("묻고 고르는 길은 처음부터 로버스트인 길과 같은 자리에 "
                 "닿는다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "pretest_vs_always.png")

    print(f"\n[등분산] n = {n} · 각 {REPS:,} 번 · 참 기울기 0")
    print("  세기    고전    사전검정   로버스트")
    for i, g in enumerate(GAMMA):
        print(f"  {g:.2f}  {100*out['고전'][i]:6.2f}  "
              f"{100*out['사전검정'][i]:6.2f}  {100*out['로버스트'][i]:6.2f}")


# ════════════════════════════════════════════════════════════════════
def fig_effective_sample_size():
    """11.3 독립성 — 예순 개가 다섯 개 값어치.

    쪽의 independence_se 는 자기상관 오차에서 표집분포와 보고된 표준오차가
    어긋나는 모습을, clustered_se 는 군집 표준오차를 다룬다. 여기서는
    그 둘이 건드리지 않는 것 --- **유효표본크기**가 상관계수에 따라 어떻게
    줄어드는지, 그리고 **표본을 키워도 기각률이 내려오지 않는다**는 것 ---
    만 본다. AR(1) 에서 n_eff = n(1-rho)/(1+rho) 이고, 쪽이 쓰는
    rho = 0.85, n = 60 이면 4.86 이다.
    """
    N_PAGE, RHO_PAGE, REPS = 60, 0.85, 3000
    rng = np.random.default_rng(1122)

    def ar1(n, rho, gen):
        u = gen.normal(0, 1, n)
        e = np.empty(n)
        e[0] = u[0] / np.sqrt(1 - rho ** 2) if rho else u[0]
        for t in range(1, n):
            e[t] = rho * e[t - 1] + u[t]
        return e

    def reject_rate(n, rho, reps):
        x = np.linspace(0, 1, n)
        xc = x - x.mean()
        sxx = (xc ** 2).sum()
        tcrit = stats.t.ppf(0.975, n - 2)
        hit = 0
        for _ in range(reps):
            y = 1.0 + ar1(n, rho, rng)           # 참 기울기는 0
            b = (xc @ (y - y.mean())) / sxx
            res = y - (y.mean() + b * xc)
            se = np.sqrt((res ** 2).sum() / (n - 2) / sxx)
            hit += abs(b) > tcrit * se
        return hit / reps

    rr = np.linspace(0, 0.95, 200)
    neff = N_PAGE * (1 - rr) / (1 + rr)
    NS = [30, 60, 120, 240, 480]
    rate_ar = [reject_rate(n, RHO_PAGE, REPS) for n in NS]
    rate_ind = [reject_rate(n, 0.0, REPS) for n in NS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(rr, neff, "-", lw=2.8, color="tab:purple")
    ax1.axhline(N_PAGE, color="0.4", ls=":", lw=1.8)
    ax1.text(0.02, N_PAGE - 2.5, f"실제로 모은 관측 {N_PAGE} 개",
             fontsize=10.5, color="0.35", fontweight="bold", va="top")
    v = N_PAGE * (1 - RHO_PAGE) / (1 + RHO_PAGE)
    ax1.plot([RHO_PAGE], [v], "o", ms=13, color="tab:purple", zorder=5)
    ax1.annotate(f"이 쪽의 자료 ρ = {RHO_PAGE}\n유효표본크기 {v:.2f} 개",
                 xy=(RHO_PAGE, v), xytext=(0.48, 24), fontsize=11.5,
                 color="tab:purple", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.8))
    ax1.set_xlim(0, 0.97)
    ax1.set_ylim(0, 66)
    ax1.set_xlabel("오차의 자기상관 ρ", fontsize=11)
    ax1.set_ylabel("유효표본크기  $n(1-\\rho)/(1+\\rho)$", fontsize=11)
    ax1.set_title(f"관측 {N_PAGE} 개가 몇 개 값어치인가", fontsize=12.5)
    ax1.grid(alpha=0.25)

    ax2.plot(NS, [100 * v for v in rate_ar], "o-", ms=9, lw=2.6,
             color="tab:red", label=f"자기상관 오차 (ρ = {RHO_PAGE})")
    ax2.plot(NS, [100 * v for v in rate_ind], "D-", ms=8, lw=2.6,
             color="tab:blue", label="독립 오차")
    ax2.axhline(5, color="0.2", ls="--", lw=1.8)
    ax2.text(470, 6.5, "약속한 $5\\%$", ha="right", va="bottom",
             fontsize=10.5, color="0.2", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xticks(NS, [str(n) for n in NS])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(0, 75)
    ax2.set_xlabel("표본 크기 n (로그 척도)", fontsize=11)
    ax2.set_ylabel("참인 귀무가설의 기각률 (%)", fontsize=11)
    ax2.set_title(f"표본을 열여섯 배로 키워도 (각 {REPS:,} 번)",
                  fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="center right", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("표준오차 공식이 틀린 것이지 표본이 작은 것이 아니다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "effective_sample_size.png")

    print(f"\n[독립성] rho = {RHO_PAGE} 에서 유효표본크기")
    for n in NS:
        print(f"  n = {n:>4d}  유효 {n * (1 - RHO_PAGE) / (1 + RHO_PAGE):6.2f} 개")
    print("  기각률")
    for n, a, b in zip(NS, rate_ar, rate_ind):
        print(f"  n = {n:>4d}  자기상관 {100*a:5.1f}%  독립 {100*b:5.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_misses_on_one_side():
    """11.3 정규성 — 빗나간 예측구간이 전부 한쪽에 몰린다.

    쪽의 normality_clt_vs_pi 는 계수와 예측구간의 두 운명을, 상자의
    normality_two_jobs 는 분포마다의 수를 다룬다. 여기서는 그 둘이
    건드리지 않는 것 --- **빗나가는 방향**이 표본을 키워도 한쪽에 몰려
    있다는 것과, 그 까닭이 오차분포의 비대칭 그 자체라는 것 --- 만 본다.
    """
    SIGMA_LOG = 1.0                 # 왜도 6.184 --- 쪽이 적은 6.18
    NS = [10, 30, 100, 300, 1000]
    REPS = 4000
    rng = np.random.default_rng(1123)

    def err(m, gen):
        e = gen.lognormal(0, SIGMA_LOG, m)
        return e - np.exp(SIGMA_LOG ** 2 / 2)

    lo_share, hi_share, cover = [], [], []
    for n in NS:
        x = np.linspace(0, 10, n)
        xc = x - x.mean()
        sxx = (xc ** 2).sum()
        below = above = 0
        for _ in range(REPS):
            y = 1.0 + 0.5 * x + err(n, rng)
            b = (xc @ (y - y.mean())) / sxx
            a = y.mean() - b * x.mean()
            res = y - (a + b * x)
            s = np.sqrt((res ** 2).sum() / (n - 2))
            x0 = x.mean()
            se = s * np.sqrt(1 + 1 / n + (x0 - x.mean()) ** 2 / sxx)
            lo = a + b * x0 - stats.t.ppf(0.975, n - 2) * se
            hi = a + b * x0 + stats.t.ppf(0.975, n - 2) * se
            y0 = 1.0 + 0.5 * x0 + err(1, rng)[0]
            if y0 < lo:
                below += 1
            elif y0 > hi:
                above += 1
        lo_share.append(below / REPS)
        hi_share.append(above / REPS)
        cover.append(1 - (below + above) / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(len(NS))
    ax1.bar(xs, [100 * v for v in lo_share], width=0.56, color="tab:blue",
            alpha=0.88, label="아래로 빗나감")
    ax1.bar(xs, [100 * v for v in hi_share], width=0.56,
            bottom=[100 * v for v in lo_share], color="tab:red", alpha=0.88,
            label="위로 빗나감")
    ax1.axhline(5, color="0.2", ls="--", lw=1.8)
    ax1.text(len(NS) - 0.55, 5.25, "약속한 $5\\%$", ha="right", va="bottom",
             fontsize=10.5, color="0.2", fontweight="bold")
    for x_, (l, h) in zip(xs, zip(lo_share, hi_share)):
        ax1.text(x_, 100 * (l + h) + 0.18,
                 f"{100 * (l + h):.1f}%", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="0.15")
        ax1.text(x_, 100 * l / 2 if l > 0.004 else 0.2, f"{100 * l:.2f}",
                 ha="center", va="center", fontsize=9.5, color="white",
                 fontweight="bold")
    ax1.set_xticks(xs, [f"n = {n}" for n in NS], fontsize=11)
    ax1.set_ylim(0, 9.5)
    ax1.set_ylabel("예측구간이 빗나간 비율 (%)", fontsize=11)
    ax1.set_title(f"빗나감은 거의 전부 위쪽이다 (각 {REPS:,} 번)",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    t = np.linspace(-5.0, 7.5, 700)
    shift = np.exp(SIGMA_LOG ** 2 / 2)
    dens = np.where(t + shift > 0,
                    stats.lognorm.pdf(np.maximum(t + shift, 1e-9),
                                      s=SIGMA_LOG), 0.0)
    sd = np.sqrt((np.exp(SIGMA_LOG ** 2) - 1) * np.exp(SIGMA_LOG ** 2))
    lo, hi = -1.96 * sd, 1.96 * sd
    ax2.plot(t, dens, "-", lw=2.6, color="0.25")
    ax2.fill_between(t[(t >= lo) & (t <= hi)], dens[(t >= lo) & (t <= hi)],
                     color="0.7", alpha=0.45)
    ax2.fill_between(t[t > hi], dens[t > hi], color="tab:red", alpha=0.55)
    ax2.fill_between(t[t < lo], dens[t < lo], color="tab:blue", alpha=0.55)
    p_hi = stats.lognorm.sf(hi + shift, s=SIGMA_LOG)
    p_lo = stats.lognorm.cdf(max(lo + shift, 1e-9), s=SIGMA_LOG)
    for v, col in ((lo, "tab:blue"), (hi, "tab:red")):
        ax2.axvline(v, color=col, lw=2.0, ls="--")
    ax2.set_xlim(-5.0, 7.5)
    ax2.set_xlabel("오차 (평균을 0 으로 맞춘 대수정규)", fontsize=11)
    ax2.set_ylabel("밀도", fontsize=11)
    ax2.set_title("$n$ 이 커지면 구간은 $\\pm 1.96\\sigma$ 로 굳는다",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.97, 0.96, f"왼쪽 꼬리는 $\\pm1.96\\sigma$ 안에 통째로 들어\n"
             f"아래로 빗나갈 확률이 {100 * p_lo:.2f}% 뿐이고\n"
             f"오른쪽으로 빗나갈 확률이 {100 * p_hi:.2f}% 다\n"
             f"— 합은 {100 * (p_lo + p_hi):.2f}% 로 $5\\%$ 에 가까운데\n"
             f"방향이 한쪽이다", transform=ax2.transAxes, ha="right",
             va="top", fontsize=10.5, fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("포함률은 돌아와도 빗나가는 방향은 돌아오지 않는다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "misses_on_one_side.png")

    print(f"\n[정규성] 대수정규 오차 (sigma_log = {SIGMA_LOG})")
    for n, l, h, c in zip(NS, lo_share, hi_share, cover):
        print(f"  n = {n:>5d}  포함률 {100*c:5.2f}%  아래 {100*l:5.2f}%  "
              f"위 {100*h:5.2f}%")
    print(f"  n 이 커질 때의 극한: 아래 {100*p_lo:.2f}% · 위 {100*p_hi:.2f}%")


# ════════════════════════════════════════════════════════════════════
def fig_one_voice():
    """11.3 가중회귀 — 한 사람이 쥔 몫과, 눈금의 덫.

    쪽의 weights_leverage 는 가중치와 지렛대를, wls_efficiency 는 최소제곱과
    가중최소제곱의 폭을 견준다. 여기서는 그 둘이 건드리지 않는 것 ---
    **한 사람의 가중치가 커질 때 그가 쥔 몫**과, **가중치를 통째로 키우면
    추정값은 그대로인데 표준오차만 줄어드는 덫** --- 만 본다.
    """
    N_PANEL = 3000
    mult = np.linspace(1, 60, 400)
    share = mult / (N_PANEL - 1 + mult)
    M_PAGE = 30.0
    share_page = M_PAGE / (N_PANEL - 1 + M_PAGE)

    n, REPS = 40, 2000
    rng = np.random.default_rng(1124)
    x = np.linspace(0.5, 4.0, n)
    w = 1 / x ** 2
    xc = x - np.average(x, weights=w)
    scales = np.array([1.0, 10.0, 100.0, 1000.0])

    est, se_rep = [], []
    for c in scales:
        ww = c * w
        b = np.empty(REPS)
        s = np.empty(REPS)
        for i in range(REPS):
            y = 1.0 + 2.0 * x + rng.normal(0, x, n)
            X = np.column_stack([np.ones(n), x])
            W = np.diag(ww)
            A = X.T @ W @ X
            beta = np.linalg.solve(A, X.T @ W @ y)
            # 가중치를 「이미 아는 정밀도」로 보는 공식. sigma^2 을 따로
            # 추정하지 않으므로 가중치를 통째로 키우면 척도만 줄어든다.
            b[i] = beta[1]
            s[i] = np.sqrt(np.linalg.inv(A)[1, 1])
        est.append(b.mean())
        se_rep.append(s.mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(mult, 100 * share, "-", lw=2.8, color="tab:red")
    ax1.plot([M_PAGE], [100 * share_page], "o", ms=13, color="tab:red",
             zorder=5)
    ax1.annotate(f"가중치가 평균의 {M_PAGE:.0f} 배이면\n"
                 f"그 한 사람이 {100 * share_page:.2f}% 를 쥔다",
                 xy=(M_PAGE, 100 * share_page), xytext=(6.5, 1.55),
                 fontsize=11.5, color="tab:red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
    ax1.set_xlim(0, 61)
    ax1.set_ylim(0, 2.1)
    ax1.set_xlabel("한 사람의 가중치가 평균의 몇 배인가", fontsize=11)
    ax1.set_ylabel("그 한 사람이 쥔 몫 (%)", fontsize=11)
    ax1.set_title(f"패널이 {N_PANEL:,} 명일 때", fontsize=12.5)
    ax1.grid(alpha=0.25)
    ax1.text(0.5, 0.045, "지지 후보를 바꾸는 것만으로\n가중평균이 그만큼 움직인다",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=11,
             fontweight="bold", color="0.2",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    xs = np.arange(len(scales))
    ax2.bar(xs, se_rep, width=0.5, color="tab:orange", alpha=0.88)
    for x_, v, e in zip(xs, se_rep, est):
        ax2.text(x_, v * 1.25, f"{v:.6f}", ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color="tab:orange")
        ax2.text(x_, 3.5e-4, f"기울기\n{e:.4f}", ha="center", va="bottom",
                 fontsize=10.5, color="0.2", fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_ylim(2.5e-4, 2.0)
    ax2.set_yticks([1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", "1"])
    ax2.set_yticks([], minor=True)
    ax2.set_xticks(xs, [f"가중치 ×{int(c):,}" for c in scales], fontsize=11)
    ax2.set_ylabel("공식이 주는 표준오차 (로그 척도)", fontsize=11)
    ax2.set_title(f"추정값은 꿈쩍도 않는데 척도만 줄어든다 ({REPS:,} 번 평균)",
                  fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("가중치에서 뜻을 갖는 것은 절대 크기가 아니라 비율이다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "one_voice_counted_thirty.png")

    print(f"\n[가중회귀] 패널 {N_PANEL:,} 명 · 한 사람이 평균의 "
          f"{M_PAGE:.0f} 배이면 {100 * share_page:.2f}%")
    for c, e, s in zip(scales, est, se_rep):
        print(f"  가중치 x{int(c):>5,}  기울기 {e:.6f}  표준오차 {s:.6f}")


fig_n_does_not_help()
fig_pretest_vs_always()
fig_effective_sample_size()
fig_misses_on_one_side()
fig_one_voice()
