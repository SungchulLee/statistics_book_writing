"""12장 들머리 그림 여섯 장 (회귀의 기초와 추정·추론).

만드는 파일:
  ch12/linear_regression/img/standard_units_regression.png   표준단위로 보면 셈이 한 줄이다
  ch12/linear_regression/img/coding_order_matters.png        코드 번호를 어떻게 붙이느냐
  ch12/estimation_inference/img/mean_vs_individual_se.png    평균반응과 개별반응
  ch12/estimation_inference/img/what_sets_the_width.png      폭을 정하는 것과 고르는 순서
  ch12/estimation_inference/img/sum_is_known_split_is_not.png 합은 알고 나눔은 모른다
  ch12/estimation_inference/img/valley_progression.png       그릇이 골짜기가 되기까지

쪽마다 들머리가 세우는 것만 그린다. 본문 그림과 겹치지 않게 고른 각은
각 함수의 설명글에 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

LR = "docs/ch12/linear_regression/img/"
EI = "docs/ch12/estimation_inference/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_standard_units():
    """11.1 단순회귀 — 표준단위로 보면 3 표준편차가 3r 로 줄어든다.

    쪽의 simple_318 은 원래 단위에서 회귀직선과 SD 직선을 겹쳐 그리고,
    beta_is_an_estimate 상자는 베타의 표준오차를 다룬다. 여기서는 그 둘이
    건드리지 않는 것 --- 표준단위에서의 한 줄 셈과, 「되돌아오는 것은
    예측값이지 사람이 아니다」 --- 만 본다.

    자료는 쪽이 적어 둔 다섯 요약값(평균 둘, 표준편차 둘, 상관)에 표본
    적률이 정확히 맞도록 다시 지은 것이다. 쪽이 드는 수 109.7 · 95.0 ·
    14.7 · 0.714 · 0.931 은 모두 그 다섯에서만 따라 나온다.
    """
    N = 247
    MX, SX = 177.75, 7.18          # 키 (cm)
    MY, SY = 78.14, 10.51          # 몸무게 (kg)
    R = 0.5347
    rng = np.random.default_rng(1101)

    def unit(v):
        v = v - v.mean()
        return v / v.std(ddof=1)

    a = unit(rng.normal(size=N))
    b = rng.normal(size=N)
    b = unit(b - (b @ a) / (a @ a) * a)
    zy = R * a + np.sqrt(1 - R ** 2) * b
    zy = unit(zy) * 1.0            # 표본 상관이 정확히 R 이 된다
    zx = a

    sd_pred = 3.0                  # SD 직선: 표준단위에서 기울기 1
    reg_pred = 3.0 * R             # 회귀직선: 기울기 r
    kg_sd = MY + SY * sd_pred
    kg_reg = MY + SY * reg_pred
    rss_reg = 1 - R ** 2
    rss_sd = 2 * (1 - R)
    sd_cond = np.sqrt(1 - R ** 2)
    share_above = stats.norm.sf(0, loc=reg_pred, scale=sd_cond)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    ax1.scatter(zx, zy, s=26, color="0.62", alpha=0.65, edgecolors="none")
    g = np.linspace(-3.4, 3.6, 2)
    ax1.plot(g, g, "--", lw=2.6, color="tab:orange",
             label="SD 직선 — 기울기 $1$")
    ax1.plot(g, R * g, "-", lw=2.8, color="tab:blue",
             label=f"회귀직선 — 기울기 $r = {R}$")
    ax1.axhline(0, color="0.8", lw=1.0)
    ax1.axvline(0, color="0.8", lw=1.0)
    ax1.plot([3, 3], [reg_pred, sd_pred], "-", lw=2.4, color="0.3")
    ax1.plot([3], [sd_pred], "o", ms=12, color="tab:orange", zorder=5)
    ax1.plot([3], [reg_pred], "o", ms=12, color="tab:blue", zorder=5)
    ax1.text(2.85, sd_pred, f"{sd_pred:.2f} 표준편차\n= {kg_sd:.1f} kg",
             ha="right", va="center", fontsize=10.5, color="tab:orange",
             fontweight="bold")
    ax1.text(2.85, reg_pred - 0.1, f"{reg_pred:.2f} 표준편차\n= {kg_reg:.1f} kg",
             ha="right", va="top", fontsize=10.5, color="tab:blue",
             fontweight="bold")
    ax1.set_xlim(-3.5, 3.7)
    ax1.set_ylim(-3.5, 3.9)
    ax1.set_xlabel("키 (표준단위)", fontsize=11)
    ax1.set_ylabel("몸무게 (표준단위)", fontsize=11)
    ax1.set_title(f"표준단위에서는 셈이 한 줄이다 — $3 \\times {R} = "
                  f"{reg_pred:.2f}$", fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(0.98, 0.03, f"잔차제곱합을 $S_{{yy}}$ 로 나누면\n"
             f"회귀직선 {rss_reg:.3f} · SD 직선 {rss_sd:.3f}\n"
             f"— SD 직선이 {rss_sd / rss_reg:.2f} 배 나쁘다",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=10.5,
             fontweight="bold", color="0.2",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    t = np.linspace(-3.2, 4.4, 500)
    dens = stats.norm.pdf(t, loc=reg_pred, scale=sd_cond)
    ax2.plot(t, dens, "-", lw=2.6, color="tab:blue")
    ax2.fill_between(t[t > 0], dens[t > 0], color="tab:blue", alpha=0.25)
    ax2.axvline(0, color="0.15", lw=2.0)
    ax2.axvline(reg_pred, color="tab:blue", lw=2.2, ls="--")
    ax2.text(0.06, dens.max() * 0.93, "평균 몸무게", fontsize=10.5,
             color="0.15", rotation=90, va="top")
    ax2.text(reg_pred + 0.08, dens.max() * 1.02,
             f"예측값 {reg_pred:.2f}", fontsize=11, color="tab:blue",
             fontweight="bold", va="top")
    ax2.set_xlim(-3.2, 4.4)
    ax2.set_ylim(0, dens.max() * 1.22)
    ax2.set_xlabel("몸무게 (표준단위)", fontsize=11)
    ax2.set_ylabel("밀도", fontsize=11)
    ax2.set_title("키가 $3$ 표준편차인 사람들의 실제 몸무게", fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.5, 0.42, f"그 사람들의 {100 * share_above:.1f}% 는\n"
             f"여전히 평균보다 무겁다\n— 되돌아오는 것은 **예측값**이다"
             .replace("**", ""), transform=ax2.transAxes, ha="center",
             va="top", fontsize=11.5, fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("회귀직선은 SD 직선에 상관을 한 번 더 곱한 직선이다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, LR + "standard_units_regression.png")

    print(f"\n[단순회귀] n = {N} · 표본상관 {np.corrcoef(zx, zy)[0,1]:.4f}")
    print(f"  3 표준편차인 사람: SD 직선 {kg_sd:.1f} kg · 회귀직선 "
          f"{kg_reg:.1f} kg · 차이 {kg_sd - kg_reg:.1f} kg")
    print(f"  잔차제곱합/Syy: 회귀 {rss_reg:.4f} · SD {rss_sd:.4f} "
          f"({rss_sd / rss_reg:.2f} 배)")
    print(f"  키 3 표준편차인 사람 가운데 평균보다 무거운 비율 "
          f"{100 * share_above:.2f}%")


# ════════════════════════════════════════════════════════════════════
def fig_coding_order():
    """11.1 범주형 설명변수 — 코드 번호를 어떻게 붙이느냐가 결과를 바꾼다.

    쪽의 categorical_coding 은 정수 코딩과 더미 부호화를 명목형·순서형
    두 판에서 견준다. 여기서는 그 그림이 건드리지 않는 것 --- 같은 자료에
    **번호만 다시 붙이면** R^2 가 얼마나 달라지는지, 그리고 정수 코딩이
    건지는 몫이 집단평균의 배치로 **정확히 정해진다**는 것 --- 만 본다.

    자료는 쪽의 코드와 같다 (default_rng(0), n = 900, 참 평균 10·16·12).
    """
    rng = np.random.default_rng(0)
    n = 900
    city = rng.integers(0, 3, n)
    CITY = ["서울", "부산", "대구"]
    true = np.array([10.0, 16.0, 12.0])
    y = true[city] + rng.normal(0, 2, n)
    means = np.array([y[city == k].mean() for k in range(3)])

    def r2_codes(code_of_city):
        c = np.asarray(code_of_city, float)[city]
        s = np.polyfit(c, y, 1)
        pred = np.polyval(s, c)
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    r2_dummy = 1 - sum(((y[city == k] - means[k]) ** 2).sum()
                       for k in range(3)) / ((y - y.mean()) ** 2).sum()

    # 서로 다른 번호 붙이기. 뒤집으면 R^2 가 같으므로 셋만 남는다.
    ORDERS = [([0, 1, 2], "서울 0 · 부산 1 · 대구 2"),
              ([0, 2, 1], "서울 0 · 대구 1 · 부산 2"),
              ([1, 0, 2], "부산 0 · 서울 1 · 대구 2")]
    vals = [(lab, r2_codes(o)) for o, lab in ORDERS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    xs = np.arange(len(vals) + 1)
    hs = [v for _, v in vals] + [r2_dummy]
    labs = [lab for lab, _ in vals] + ["더미 부호화\n(번호를 안 쓴다)"]
    cols = ["tab:red", "tab:orange", "tab:purple", "tab:blue"]
    ax1.bar(xs, hs, width=0.56, color=cols, alpha=0.88)
    for x, h in zip(xs, hs):
        ax1.text(x, h + 0.012, f"{h:.4f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="0.15")
    ax1.set_xticks(xs, labs, fontsize=10.5)
    ax1.set_ylim(0, r2_dummy * 1.22)
    ax1.set_ylabel("$R^2$", fontsize=12)
    ax1.set_title(f"같은 자료 · 같은 방법 — 번호만 다시 붙였다 (n = {n})",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.5, 0.955, f"집단평균 셋: " +
             " · ".join(f"{c} {m:.2f}" for c, m in zip(CITY, means)),
             transform=ax1.transAxes, ha="center", va="top", fontsize=10.5,
             color="0.25", fontweight="bold")

    # 가운데 집단의 평균을 움직이면 정수 코딩이 건지는 몫이 어떻게 변하나
    ms = np.linspace(7.0, 19.0, 400)
    share = []
    for m in ms:
        mu = np.array([10.0, m, 12.0])
        between = mu.var()
        slope = ((np.array([0, 1, 2]) - 1) * (mu - mu.mean())).sum() / 2
        share.append(slope ** 2 * (2 / 3) / between)
    share = np.array(share)
    ax2.plot(ms, 100 * share, "-", lw=2.8, color="tab:red")
    ax2.axhline(100, color="0.35", ls="--", lw=1.6)
    ax2.text(18.8, 97, "더미 부호화가 건지는 몫", ha="right", va="top",
             fontsize=10.5, color="0.3", fontweight="bold")
    i_page = int(np.argmin(np.abs(ms - 16.0)))
    ax2.plot([16.0], [100 * share[i_page]], "o", ms=12, color="tab:red",
             zorder=5)
    ax2.annotate(f"이 쪽의 자료 (부산 $16$)\n{100 * share[i_page]:.1f}%",
                 xy=(16.0, 100 * share[i_page]), xytext=(13.6, 38),
                 fontsize=11, color="tab:red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
    ax2.plot([11.0], [100], "o", ms=12, mfc="none", mec="0.25", mew=2.4,
             zorder=5)
    ax2.annotate("칸 사이가 정확히 고른 자리\n— 여기서만 $100\\%$ 다",
                 xy=(11.0, 100), xytext=(7.4, 62), fontsize=11, color="0.2",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6))
    ax2.set_xlim(7, 19)
    ax2.set_ylim(0, 118)
    ax2.set_xlabel("가운데 번호를 받은 집단의 평균", fontsize=11)
    ax2.set_ylabel("정수 코딩이 건지는 몫 (%)", fontsize=11)
    ax2.set_title("서울 $10$ · 대구 $12$ 를 못 박고 가운데만 움직이면",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("이름을 수로 바꾸는 한 번의 선택이 모형의 내용을 바꾼다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, LR + "coding_order_matters.png")

    print(f"\n[범주형] 집단평균 " +
          " · ".join(f"{c} {m:.4f}" for c, m in zip(CITY, means)))
    for lab, v in vals:
        print(f"  {lab:<24s} R^2 = {v:.4f}")
    print(f"  {'더미 부호화':<24s} R^2 = {r2_dummy:.4f}")
    print(f"  가운데가 11 일 때 정수 코딩이 건지는 몫 "
          f"{100 * share[int(np.argmin(np.abs(ms - 11.0)))]:.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_mean_vs_individual():
    """11.2 표집분포 — 평균반응과 개별반응의 표준오차.

    쪽의 three_variance_pieces 는 예측오차의 분산을 세 조각으로 가르고,
    ci_vs_pi_bands 는 n 을 키울 때 두 대가 어떻게 달라지는지를 그린다.
    여기서는 그 둘이 건드리지 않는 것 --- 두 표준오차의 **비가 자리에 따라
    어떻게 줄어드는지** --- 만 본다. 중심에서 5.10 배이던 것이 자료 범위
    밖에서는 2.21 배다.
    """
    n, sigma = 25, 1.0
    x = np.linspace(0, 10, n)
    xbar = x.mean()
    sxx = ((x - xbar) ** 2).sum()

    def se_mean(v):
        return sigma * np.sqrt(1 / n + (v - xbar) ** 2 / sxx)

    def se_indiv(v):
        return sigma * np.sqrt(1 + 1 / n + (v - xbar) ** 2 / sxx)

    # 쪽이 드는 「자료 범위를 벗어난 자리」 --- 평균반응 표준오차가 0.507
    d_out = np.sqrt((0.507 ** 2 - 1 / n) * sxx)
    x_out = xbar + d_out

    g = np.linspace(xbar - d_out * 1.12, xbar + d_out * 1.12, 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.axvspan(x.min(), x.max(), color="0.75", alpha=0.25)
    ax1.text((x.min() + x.max()) / 2, 1.33, "자료가 있는 구간", ha="center",
             fontsize=10.5, color="0.3", fontweight="bold")
    ax1.plot(g, se_indiv(g), "-", lw=2.8, color="tab:red",
             label="개별반응 — 다음 한 건이 어디에")
    ax1.plot(g, se_mean(g), "-", lw=2.8, color="tab:blue",
             label="평균반응 — 평균적으로 얼마")
    for v, col in ((xbar, "0.25"), (x_out, "0.25")):
        ax1.plot([v, v], [se_mean(v), se_indiv(v)], "-", lw=2.0, color=col)
        ax1.plot([v], [se_mean(v)], "o", ms=10, color="tab:blue", zorder=5)
        ax1.plot([v], [se_indiv(v)], "o", ms=10, color="tab:red", zorder=5)
        ax1.text(v + 0.18, (se_mean(v) + se_indiv(v)) / 2,
                 f"{se_indiv(v):.3f}\n{se_mean(v):.3f}", ha="left",
                 va="center", fontsize=10.5, color="0.2", fontweight="bold")
    ax1.set_ylim(0, 1.45)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("표준오차", fontsize=11)
    ax1.set_title(f"$n = {n}$, $\\sigma = {sigma:.0f}$ 에서의 두 표준오차",
                  fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.plot(g, se_indiv(g) / se_mean(g), "-", lw=2.8, color="tab:purple")
    ax2.axvspan(x.min(), x.max(), color="0.75", alpha=0.25)
    for v, nm in ((xbar, "자료의 중심"), (x_out, "자료 범위 밖")):
        r = se_indiv(v) / se_mean(v)
        ax2.plot([v], [r], "o", ms=12, color="tab:purple", zorder=5)
        ax2.text(v, r + 0.18, f"{nm}\n{r:.2f} 배", ha="center", va="bottom",
                 fontsize=11, color="tab:purple", fontweight="bold")
    ax2.axhline(1, color="0.3", ls="--", lw=1.4)
    ax2.set_ylim(0, 6.6)
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel("개별반응 표준오차 ÷ 평균반응 표준오차", fontsize=11)
    ax2.set_title("두 대의 폭 비는 자리에 따라 달라진다", fontsize=12.5)
    ax2.grid(alpha=0.25)
    ax2.text(0.5, 0.04, "중심에서 멀어질 때 커지는 항은 두 식에 똑같이 한 번씩만\n"
             "들어 있고, 개별반응 쪽에는 1 이 이미 깔려 있다",
             transform=ax2.transAxes, ha="center", va="bottom", fontsize=10.5,
             fontweight="bold", color="0.2",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("「평균적으로 얼마인가」와 「다음 한 건이 어디인가」는 "
                 "다른 물음이다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, EI + "mean_vs_individual_se.png")

    print(f"\n[표집분포] n = {n}, sigma = {sigma}")
    for v, nm in ((xbar, "중심"), (x_out, "범위 밖")):
        print(f"  {nm:<8s} x = {v:6.3f}  평균반응 {se_mean(v):.3f} · "
              f"개별반응 {se_indiv(v):.3f} · 비 "
              f"{se_indiv(v) / se_mean(v):.2f}")


# ════════════════════════════════════════════════════════════════════
def fig_what_sets_the_width():
    """11.2 계수의 신뢰구간 — 폭을 정하는 것과, 수준을 고르는 순서.

    쪽의 famous_coefficient 는 카드·크루거의 계수와 구간을, slope_ci_levels
    는 같은 추정값에 세 신뢰수준을 붙인 모습을 그린다. 여기서는 그 둘이
    건드리지 않는 것 --- 폭을 절반으로 줄이는 세 가지 길과, **구간을 보고
    수준을 고르면** 약속이 깨진다는 것 --- 만 본다.
    """
    n0, sigma0, spread0 = 30, 1.0, 1.0
    REPS = 40_000
    LEVELS = [0.90, 0.95, 0.99]

    def half_width(n, sigma, spread):
        x = np.linspace(0, 10 * spread, n)
        sxx = ((x - x.mean()) ** 2).sum()
        return stats.t.ppf(0.975, n - 2) * sigma / np.sqrt(sxx)

    base = half_width(n0, sigma0, spread0)
    CASES = [("기준", n0, sigma0, spread0, "0.55"),
             ("표본을 네 배로", 4 * n0, sigma0, spread0, "tab:blue"),
             ("x 의 퍼짐을 두 배로", n0, sigma0, 2 * spread0, "tab:green"),
             ("잡음을 절반으로", n0, sigma0 / 2, spread0, "tab:orange")]

    rng = np.random.default_rng(1102)
    z = rng.normal(size=REPS)                 # 참값이 0 인 자리의 표준화 추정값
    crit = {L: stats.norm.ppf(0.5 + L / 2) for L in LEVELS}
    # 자료를 보고 「0 을 배제하는 가장 좁은 수준」을 고르는 규칙
    picked = np.full(REPS, LEVELS[-1], float)
    for L in LEVELS:                          # 좁은 것부터
        hit = (np.abs(z) > crit[L]) & (picked == LEVELS[-1])
        picked[np.abs(z) > crit[L]] = np.minimum(picked[np.abs(z) > crit[L]],
                                                 L)
    covered = np.abs(z) <= np.array([crit[L] for L in picked])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9))

    xs = np.arange(len(CASES))
    ws = [half_width(n, s, sp) for _, n, s, sp, _ in CASES]
    ax1.bar(xs, ws, width=0.56, color=[c for *_, c in CASES], alpha=0.88)
    ax1.axhline(base / 2, color="0.25", ls="--", lw=1.8)
    ax1.text(-0.42, base / 2 + base * 0.015, "기준의 절반", ha="left",
             va="bottom", fontsize=10.5, color="0.25", fontweight="bold")
    for x, w in zip(xs, ws):
        ax1.text(x, w + base * 0.02, f"{w:.4f}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color="0.15")
    ax1.set_xticks(xs, [c[0] for c in CASES], fontsize=11)
    ax1.set_ylim(0, base * 1.18)
    ax1.set_ylabel("$95\\%$ 구간의 반폭", fontsize=11)
    ax1.set_title(f"폭을 절반으로 줄이는 세 가지 길 (기준은 $n = {n0}$)",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)

    xs2 = np.arange(len(LEVELS) + 1)
    # 네 막대를 모두 같은 난수에서 **재어** 놓는다
    nominal = [100 * (np.abs(z) <= crit[L]).mean() for L in LEVELS] \
        + [100 * covered.mean()]
    labs2 = [f"{int(100 * L)}%" for L in LEVELS] + ["구간을 보고\n고른 수준"]
    cols2 = ["0.62", "0.62", "0.62", "tab:red"]
    ax2.bar(xs2, nominal, width=0.56, color=cols2, alpha=0.88)
    ax2.axhline(100 * LEVELS[0], color="tab:red", ls="--", lw=1.8)
    for x, v in zip(xs2, nominal):
        ax2.text(x, v + 0.4, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color="0.15")
    ax2.set_xticks(xs2, labs2, fontsize=11)
    ax2.set_ylim(85, 102)
    ax2.set_ylabel("실제 포함률", fontsize=11)
    ax2.set_title(f"「$0$ 을 배제하는 가장 좁은 수준」을 고르면 "
                  f"({REPS:,} 번)", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.5, 0.06, "가장 느슨한 수준의 포함률로 떨어진다\n"
             "— $99\\%$ 라 적어도 실제는 $90\\%$ 다",
             transform=ax2.transAxes, ha="center", va="bottom", fontsize=11,
             fontweight="bold", color="tab:red",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0))

    fig.suptitle("폭을 정하는 것은 셋이고, 수준은 자료를 보기 전에 고른다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, EI + "what_sets_the_width.png")

    print(f"\n[신뢰구간] 95% 반폭")
    for (nm, n, s, sp, _), w in zip(CASES, ws):
        print(f"  {nm:<16s} {w:.5f}   (기준의 {w / base:.3f} 배)")
    print(f"  구간을 보고 수준을 고르면 실제 포함률 "
          f"{100 * covered.mean():.2f}% ({REPS:,} 번)")


# ════════════════════════════════════════════════════════════════════
def fig_sum_vs_split():
    """11.2 계수의 검정 — 합은 아주 잘 알고 나눔은 거의 모른다.

    쪽의 joint_vs_marginal_ci 는 결합 신뢰영역과 개별 구간이 어긋나는
    모습을 그린다. 여기서는 그 그림이 수로 말하지 않는 것 --- 설명변수의
    상관이 커질 때 **개별 계수의 표준오차는 터지는데 합의 표준오차는
    오히려 줄어든다**는 것 --- 만 본다. 표준화한 두 설명변수에서

        SE(b1) = sigma / sqrt(n(1-rho^2))
        SE(b1+b2) = sigma sqrt(2/(n(1+rho)))
        SE(b1-b2) = sigma sqrt(2/(n(1-rho)))

    이고, 쪽이 드는 n = 60, rho = 0.95, 참 계수 0.6 을 그대로 쓴다.
    """
    n, sigma, beta = 60, 1.0, 0.6
    rho = np.linspace(0, 0.985, 400)
    se1 = sigma / np.sqrt(n * (1 - rho ** 2))
    se_sum = sigma * np.sqrt(2 / (n * (1 + rho)))
    se_dif = sigma * np.sqrt(2 / (n * (1 - rho)))
    RP = 0.95

    def at(r):
        return (sigma / np.sqrt(n * (1 - r ** 2)),
                sigma * np.sqrt(2 / (n * (1 + r))),
                sigma * np.sqrt(2 / (n * (1 - r))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    ax1.plot(rho, se1, "-", lw=2.8, color="tab:red",
             label="계수 하나  $\\hat\\beta_1$")
    ax1.plot(rho, se_dif, "-", lw=2.8, color="tab:purple",
             label="차  $\\hat\\beta_1 - \\hat\\beta_2$")
    ax1.plot(rho, se_sum, "-", lw=2.8, color="tab:blue",
             label="합  $\\hat\\beta_1 + \\hat\\beta_2$")
    s1, ss, sd = at(RP)
    for v, col in ((s1, "tab:red"), (ss, "tab:blue"), (sd, "tab:purple")):
        ax1.plot([RP], [v], "o", ms=10, color=col, zorder=5)
    ax1.axvline(RP, color="0.4", ls="--", lw=1.6)
    ax1.text(RP - 0.015, 2.35, f"이 쪽의 자료\n$\\rho = {RP}$", ha="right",
             va="top", fontsize=11, color="0.25", fontweight="bold")
    ax1.set_xlim(0, 0.99)
    ax1.set_ylim(0, 2.5)
    ax1.set_xlabel("두 설명변수의 상관 ρ", fontsize=11)
    ax1.set_ylabel("표준오차", fontsize=11)
    ax1.set_title(f"$n = {n}$, $\\sigma = {sigma:.0f}$ — 닫힌 꼴 그대로",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ITEMS = [("$\\hat\\beta_1$", beta, s1, "tab:red"),
             ("$\\hat\\beta_2$", beta, s1, "tab:red"),
             ("$\\hat\\beta_1 + \\hat\\beta_2$", 2 * beta, ss, "tab:blue")]
    xs = np.arange(len(ITEMS))
    ts = [v / s for _, v, s, _ in ITEMS]
    ax2.bar(xs, ts, width=0.52, color=[c for *_, c in ITEMS], alpha=0.88)
    ax2.axhline(stats.t.ppf(0.975, n - 3), color="0.25", ls="--", lw=1.8)
    ax2.text(-0.44, stats.t.ppf(0.975, n - 3) + 0.18, "$5\\%$ 에서의 임계값",
             ha="left", va="bottom", fontsize=10.5, color="0.25",
             fontweight="bold")
    for x, (nm, v, s, col), t in zip(xs, ITEMS, ts):
        ax2.text(x, t / 2, f"t = {t:.2f}", ha="center", va="center",
                 fontsize=12.5, fontweight="bold", color="white")
        ax2.text(x, -0.75, f"참값 {v:.1f}\nSE {s:.3f}", ha="center",
                 va="top", fontsize=10.5, color="0.3")
    ax2.set_xticks(xs, [nm for nm, *_ in ITEMS], fontsize=13)
    ax2.set_ylim(-2.6, 11.5)
    ax2.set_ylabel("참값을 표준오차로 나눈 값", fontsize=11)
    ax2.set_title(f"$\\rho = {RP}$ 에서 — 개별은 묻히고 합은 또렷하다",
                  fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("자료는 합이 얼마인지는 아주 잘 알고, 그 합을 어떻게 "
                 "나눌지는 거의 모른다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, EI + "sum_is_known_split_is_not.png")

    print(f"\n[계수의 검정] n = {n}, sigma = {sigma}, 참 계수 {beta}")
    for r in (0.0, 0.5, 0.9, 0.95, 0.98):
        a, b, c = at(r)
        print(f"  rho = {r:.2f}  SE(b1) = {a:.4f} · SE(합) = {b:.4f} · "
              f"SE(차) = {c:.4f}")
    print(f"  rho = {RP}: t(개별) = {beta / s1:.2f} · "
          f"t(합) = {2 * beta / ss:.2f}")


# ════════════════════════════════════════════════════════════════════
def fig_valley_progression():
    """11.2 RSS 곡면 — 그릇이 골짜기가 되기까지.

    쪽의 rss_surface_visualization_71 은 잘 선 곡면 하나를, flat_valley 는
    열이 완전히 겹쳤을 때의 골짜기를 그린다. 여기서는 그 사이 ---
    **상관이 커지면서 그릇이 어떻게 늘어나는가** --- 를 본다. 등고선
    타원의 축비가 sqrt((1+rho)/(1-rho)) 이므로 쪽이 말하는 두 끝이
    한 곡선 위의 두 점이 된다.
    """
    n, sigma = 100, 1.0
    TRUE = np.array([1.0, 1.0])
    SHOW = [(0.0, "서로 무관하다"), (0.9, "닮았다"), (0.995, "거의 같은 열")]
    rng = np.random.default_rng(1103)

    fig = plt.figure(figsize=(14.2, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.34)

    for i, (r, name) in enumerate(SHOW):
        ax = fig.add_subplot(gs[0, i])
        a = rng.normal(size=n)
        b = r * a + np.sqrt(1 - r ** 2) * rng.normal(size=n)
        X = np.column_stack([a, b])
        y = X @ TRUE + rng.normal(0, sigma, n)
        hat = np.linalg.lstsq(X, y, rcond=None)[0]
        g1 = np.linspace(hat[0] - 3, hat[0] + 3, 160)
        g2 = np.linspace(hat[1] - 3, hat[1] + 3, 160)
        G1, G2 = np.meshgrid(g1, g2)
        R = np.empty_like(G1)
        for p in range(G1.shape[0]):
            res = y[None, :] - (G1[p][:, None] * X[:, 0][None, :]
                                + G2[p][:, None] * X[:, 1][None, :])
            R[p] = (res ** 2).sum(axis=1)
        lo = R.min()
        ax.contour(G1, G2, R, levels=lo + np.array([2, 8, 20, 45, 90, 160]),
                   colors="tab:blue", linewidths=1.5)
        ax.plot([hat[0]], [hat[1]], "o", ms=10, color="tab:red", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("$\\beta_1$", fontsize=11)
        if i == 0:
            ax.set_ylabel("$\\beta_2$", fontsize=11)
        ax.set_title(f"ρ = {r:g}\n{name}", fontsize=12)
        ax.grid(alpha=0.2)

    ax4 = fig.add_subplot(gs[0, 3])
    rr = np.linspace(0, 0.995, 400)
    ax4.plot(rr, np.sqrt((1 + rr) / (1 - rr)), "-", lw=2.8,
             color="tab:purple")
    for r, _ in SHOW:
        v = np.sqrt((1 + r) / (1 - r))
        ax4.plot([r], [v], "o", ms=10, color="tab:purple", zorder=5)
        ax4.text(r - 0.02, v + 1.2, f"{v:.1f}", ha="right", va="bottom",
                 fontsize=11, color="tab:purple", fontweight="bold")
    ax4.set_xlim(0, 1.0)
    ax4.set_ylim(0, 24)
    ax4.set_xlabel("두 열의 상관 ρ", fontsize=11)
    ax4.set_ylabel("등고선 타원의 긴 축 ÷ 짧은 축", fontsize=11)
    ax4.set_title("$\\sqrt{(1+\\rho)/(1-\\rho)}$ — $1$ 에서 터진다",
                  fontsize=12)
    ax4.grid(alpha=0.25)

    fig.suptitle("바닥이 점에서 골짜기로 바뀌는 일은 갑자기 오지 않는다",
                 fontsize=14, y=1.02)
    save(fig, EI + "valley_progression.png")

    print(f"\n[RSS 곡면] 등고선 타원의 축비")
    for r in (0.0, 0.5, 0.9, 0.95, 0.99, 0.995):
        print(f"  rho = {r:.3f}  →  {np.sqrt((1 + r) / (1 - r)):8.2f}")


fig_standard_units()
fig_coding_order()
fig_mean_vs_individual()
fig_what_sets_the_width()
fig_sum_vs_split()
fig_valley_progression()
