"""14장 들어가는 이야기 그림 — 첫 묶음 여섯.

쪽마다 들머리가 세우는 것만 그린다. ch14 는 쪽이 이미 그림을 서넛씩
들고 있어서 들머리의 beat 가 **이미 그려져 있는 일이 잦다.** 그래서 각
쪽의 그림 목록을 먼저 뽑아 두고, 그 가운데 어느 것도 건드리지 않는
beat 만 골랐다. 고른 까닭을 그림마다 적는다.

  1. index                 같은 자료를 네 눈으로 보면 — 눈이 놓치고 자가 잡는다
  2. normality             평균의 구간은 올라가고 분산의 구간은 내려간다
  3. role_in_inference     왜도를 올려 보면 — n 이 고치는 것과 못 고치는 것
  4. financial_context     한 세대에 한 번이 한 임기에 한 번으로
  5. histogram_density     이 그림이 보는 것과 못 보는 것 (Q-Q 와 서로 반대다)
  6. boxplots              울타리가 잘라내는 비율과, 한쪽만 비는 것

만드는 파일:
  docs/ch14/img/four_views_of_one_sample.png
  docs/ch14/introduction/img/one_rises_one_falls.png
  docs/ch14/introduction/img/skew_axis_not_n_axis.png
  docs/ch14/introduction/img/generation_to_term.png
  docs/ch14/graphical_methods/img/what_the_histogram_cannot_see.png
  docs/ch14/graphical_methods/img/fence_cuts_and_one_sided.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

IDX = "docs/ch14/img/"
INTRO = "docs/ch14/introduction/img/"
GRAPH = "docs/ch14/graphical_methods/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 1. 장 개요 — 같은 자료를 네 눈으로 본다
#
# 쪽에 그림이 하나도 없다. 들머리는 「25 표준편차짜리 날」인데 그 이야기의
# 그림은 14.7 의 sigma_ruler 가 이미 그린다(기다리는 햇수와 t 분포에서
# 250 일로 잰 시그마). 그래서 그것을 되풀이하지 않고 **들머리의 마지막
# 문단**을 그린다 — 「종 모양에서 벗어났다는 것을 무엇으로 알아보며」.
#
# 장 개요는 독자가 첨도도 Q-Q 도 아직 배우지 않은 자리다. 그러니 한 표본을
# 네 가지 눈으로 보여 주고 **어느 눈이 놓치고 어느 눈이 잡는지**만 보인다.
# 이 장의 목차가 그대로 그림이 된다.
# ════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(14_001)
NFV = 1000
x = rng.standard_t(5, NFV)
x = (x - x.mean()) / x.std()
p_sw = stats.shapiro(x).pvalue
print("[장 개요] 꼬리가 두꺼운 표본 하나를 네 눈으로 본다")
print(f"  n = {NFV}, 초과첨도 {stats.kurtosis(x):.3f}, "
      f"샤피로–윌크 p = {p_sw:.3g}")

fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3))

ax = axes[0]
ax.hist(x, bins=45, density=True, color="0.72", edgecolor="none")
g = np.linspace(-5, 5, 400)
ax.plot(g, stats.norm.pdf(g), color="tab:red", lw=2.4)
ax.set_xlim(-5, 5)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_title("① 히스토그램", fontsize=13)
ax.set_xlabel("봉우리가 조금 높을 뿐이다", fontsize=11.5,
              color="tab:green", fontweight="bold")
ax.set_yticks([])
ax.grid(alpha=0.2)

ax = axes[1]
bp = ax.boxplot([x], vert=False, widths=0.5, patch_artist=True)
bp["boxes"][0].set_facecolor("0.82")
nout = int(((x < np.percentile(x, 25) - 1.5 * stats.iqr(x)) |
            (x > np.percentile(x, 75) + 1.5 * stats.iqr(x))).sum())
ax.set_yticks([])
ax.set_xlim(-6.5, 6.5)
ax.set_title("② 상자그림", fontsize=13)
ax.set_xlabel(f"바깥 점 {nout} 개 — 많은 건가?", fontsize=11.5,
              color="tab:orange", fontweight="bold")
ax.grid(axis="x", alpha=0.2)

ax = axes[2]
(osm, osr), _ = stats.probplot(x, dist="norm")
ax.plot(osm, osr, "o", ms=3.2, color="0.35")
lim = [-3.6, 3.6]
ax.plot(lim, lim, color="tab:red", lw=2.2)
ax.set_xlim(-3.6, 3.6)
ax.set_ylim(-7.5, 7.5)
ax.set_title("③ Q-Q 그림", fontsize=13)
ax.set_xlabel("양 끝이 휜다", fontsize=11.5, color="tab:red",
              fontweight="bold")
ax.grid(alpha=0.2)

ax = axes[3]
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("④ 검정", fontsize=13)
ax.text(0.5, 0.63, "샤피로–윌크", ha="center", va="center", fontsize=13,
        color="0.3", fontweight="bold")
ax.text(0.5, 0.47, f"p = {p_sw:.2g}", ha="center", va="center",
        fontsize=21, color="tab:red", fontweight="bold")
ax.text(0.5, 0.30, "정규가 아니라고\n단호히 말한다", ha="center",
        va="center", fontsize=12.5, color="tab:red", fontweight="bold")

fig.suptitle("같은 표본 하나 — 어느 눈은 놓치고 어느 눈은 잡는다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, IDX + "four_views_of_one_sample.png")


# ════════════════════════════════════════════════════════════════════
# 2. introduction/normality — 하나는 올라가고 하나는 내려간다
#
# 쪽의 clt_mean_vs_variance 는 **표집분포**가 이론 분포와 맞는지를 그린다.
# 들머리가 드는 수는 그것이 아니라 **포함률**이다(0.928 → 0.944 와
# 0.688 → 0.618 → 0.571). 표본을 늘리는 것이 한쪽에는 치료이고 다른
# 쪽에는 아무것도 아니라는 말이 그 두 곡선이므로 그것만 그린다.
# ════════════════════════════════════════════════════════════════════
SIG = 0.6                      # 대수정규의 로그 표준편차
NS2 = [10, 20, 30, 50, 100, 200, 400, 800, 1600]
REPS2 = 20_000
mu_true = np.exp(SIG ** 2 / 2)
var_true = (np.exp(SIG ** 2) - 1) * np.exp(SIG ** 2)
skew_true = (np.exp(SIG ** 2) + 2) * np.sqrt(np.exp(SIG ** 2) - 1)

cov_mu, cov_s2 = [], []
r2 = np.random.default_rng(14_002)
for n in NS2:
    s = np.exp(r2.normal(0, SIG, size=(REPS2, n)))
    m, sd = s.mean(1), s.std(1, ddof=1)
    tcrit = stats.t.ppf(0.975, n - 1)
    half = tcrit * sd / np.sqrt(n)
    cov_mu.append(((m - half <= mu_true) & (mu_true <= m + half)).mean())
    lo = (n - 1) * sd ** 2 / stats.chi2.ppf(0.975, n - 1)
    hi = (n - 1) * sd ** 2 / stats.chi2.ppf(0.025, n - 1)
    cov_s2.append(((lo <= var_true) & (var_true <= hi)).mean())

print(f"\n[normality] 대수정규(왜도 {skew_true:.2f})에서 두 구간의 포함률")
for n, a, b in zip(NS2, cov_mu, cov_s2):
    print(f"  n = {n:4d}:  평균의 구간 {a:.4f} · 분산의 구간 {b:.4f}")

fig, ax = plt.subplots(figsize=(10.6, 5.6))
ax.axhline(0.95, color="0.3", lw=2.2, ls="--")
ax.text(1700, 0.9545, "약속한 0.95", ha="right", va="bottom", fontsize=12,
        color="0.3", fontweight="bold")
ax.plot(NS2, cov_mu, "o-", color="tab:blue", lw=3.0, ms=11,
        label="평균의 95% 신뢰구간")
ax.plot(NS2, cov_s2, "s-", color="tab:red", lw=3.0, ms=10,
        label="분산의 95% 신뢰구간")
ax.set_xscale("log")
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에서 두부가 된다
ax.set_xticks(NS2, [str(n) for n in NS2])
ax.set_xlabel("표본크기 n (로그 눈금)", fontsize=12)
ax.set_ylabel("실제 포함률", fontsize=12)
ax.set_ylim(0.52, 0.985)
ax.legend(fontsize=11.5, loc="center", bbox_to_anchor=(0.70, 0.56))
ax.grid(alpha=0.25)
ax.annotate(f"표본을 늘리면 약속에 닿는다\n({cov_mu[2]:.3f} → "
            f"{cov_mu[-1]:.3f})", xy=(NS2[-2], cov_mu[-2]),
            xytext=(120, 0.845), fontsize=11.5, color="tab:blue",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax.annotate(f"늘릴수록 **내려간다** — 약속에서 멀어진다\n({cov_s2[2]:.3f} → "
            f"{cov_s2[-1]:.3f})".replace("**", ""),
            xy=(NS2[-2], cov_s2[-2]), xytext=(90, 0.665), fontsize=11.5,
            color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.set_title(f"같은 모집단, 같은 표본 — 왜도 {skew_true:.2f} 인 대수정규",
             fontsize=13)
fig.suptitle("평균의 구간은 버티고 분산의 구간은 무너진다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, INTRO + "one_rises_one_falls.png")


# ════════════════════════════════════════════════════════════════════
# 3. introduction/role_in_inference — 축을 바꿔 본다
#
# 쪽의 type1_vs_n 이 「t 검정과 분산 검정의 1 종 오류율을 **표본크기에**
# 따라」 그린다. 들머리의 두 수(0.097 → 0.051 과 0.323)가 그 그림이다.
# 그래서 같은 것을 되풀이하지 않고 **축을 바꾼다** — 가로를 모집단의
# 왜도로 놓고 n 을 둘 고정한다. 그러면 들머리가 묻고 끝내는 것
# (「표본을 늘려 고쳐지는 왜곡과 늘려도 굳어 있는 왜곡은 무엇으로
# 갈리는가」)이 **한 그림에서 두 번** 보인다 — 작은 n 에서는 둘 다
# 올라가는데, 큰 n 에서는 한쪽만 제자리로 돌아온다.
# ════════════════════════════════════════════════════════════════════
SIGS = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
skews = [(np.exp(s ** 2) + 2) * np.sqrt(np.exp(s ** 2) - 1) for s in SIGS]
NPAIR = [20, 400]
REPS3 = 20_000
res = {n: {"t": [], "chi": []} for n in NPAIR}
r3 = np.random.default_rng(14_003)
for n in NPAIR:
    for s in SIGS:
        d = np.exp(r3.normal(0, s, size=(REPS3, n)))
        m0 = np.exp(s ** 2 / 2)
        v0 = (np.exp(s ** 2) - 1) * np.exp(s ** 2)
        xb, sd = d.mean(1), d.std(1, ddof=1)
        tstat = (xb - m0) / (sd / np.sqrt(n))
        res[n]["t"].append((abs(tstat) > stats.t.ppf(0.975, n - 1)).mean())
        q = (n - 1) * sd ** 2 / v0
        res[n]["chi"].append(((q < stats.chi2.ppf(0.025, n - 1)) |
                              (q > stats.chi2.ppf(0.975, n - 1))).mean())

print("\n[role_in_inference] 가로축을 n 이 아니라 모집단 왜도로 놓는다")
for n in NPAIR:
    print(f"  n = {n}")
    for sk, a, b in zip(skews, res[n]["t"], res[n]["chi"]):
        print(f"    왜도 {sk:6.2f}:  평균의 t 검정 {a:.4f} · "
              f"분산의 카이제곱 검정 {b:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(13.4, 5.0), sharey=True)
for ax, n in zip(axs, NPAIR):
    ax.axhline(0.05, color="0.3", lw=2.2, ls="--")
    ax.plot(skews, res[n]["t"], "o-", color="tab:blue", lw=3.0, ms=11,
            label="평균에 대한 t 검정")
    ax.plot(skews, res[n]["chi"], "s-", color="tab:red", lw=3.0, ms=10,
            label="분산에 대한 카이제곱 검정")
    ax.set_xlabel("모집단의 왜도", fontsize=12)
    ax.set_title(f"n = {n}", fontsize=13.5)
    ax.set_ylim(0, 0.78)
    ax.grid(alpha=0.25)
axs[0].set_ylabel("실제 1 종 오류율 (약속은 0.05)", fontsize=12)
axs[0].legend(fontsize=11, loc="upper left")
axs[0].text(skews[-1], 0.016, "가로 점선이 약속한 0.05", ha="right",
            va="bottom", fontsize=11, color="0.3", fontweight="bold")
axs[1].annotate(f"n 을 스무 배로 키우면 거의 돌아온다\n"
                f"(같은 왜도에서 {res[20]['t'][-1]:.3f} → "
                f"{res[400]['t'][-1]:.3f})",
                xy=(skews[-2] + 0.3, res[400]["t"][-2] + 0.012),
                xytext=(0.25, 0.265), fontsize=11.3, color="tab:blue",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
axs[1].annotate(f"n 을 키우면 오히려 **나빠진다**\n"
                f"(같은 왜도에서 {res[20]['chi'][-1]:.3f} → "
                f"{res[400]['chi'][-1]:.3f})".replace("**", ""),
                xy=(skews[-1], res[400]["chi"][-1] - 0.02),
                xytext=(0.25, 0.60), fontsize=11.3, color="tab:red",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
fig.suptitle("사라지는 오차와 굳는 편향 — 같은 자료, 두 절차", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, INTRO + "skew_axis_not_n_axis.png")


# ════════════════════════════════════════════════════════════════════
# 4. introduction/financial_context — 한 세대에 한 번이 한 임기에 한 번으로
#
# 쪽의 bachelier_1900 은 「값을 정규로 두면 0 아래에 확률이 남는다」를
# 그린다. 들머리의 머리글은 그것이 아니라 **재현주기**다(62.6 년과
# 6.7 년). 14.7 의 sigma_ruler 가 정규에서 기다리는 햇수를 그리지만
# **두 모형을 맞대지는 않는다** — 맞대는 것이 들머리의 요점이고,
# 「한 세대에 한 번 볼 일이 한 임기에 한 번 볼 일로」가 그 맞댐이다.
# ════════════════════════════════════════════════════════════════════
TRADING = 252
KS4 = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
DF4 = 10          # 초과첨도가 정확히 1 — 쪽의 연습문제 9 가 고른 자유도
scale_t = np.sqrt((DF4 - 2) / DF4)       # 분산을 1 로 맞춘다


def years_normal(k):
    return 1.0 / (2 * stats.norm.sf(k) * TRADING)


def years_t(k, df=DF4):
    return 1.0 / (2 * stats.t.sf(k / scale_t, df) * TRADING)


yn = np.array([years_normal(k) for k in KS4])
yt = np.array([years_t(k) for k in KS4])
print("\n[financial_context] 같은 하루를 두 모형이 몇 해에 한 번이라 부르는가")
for k, a, b in zip(KS4, yn, yt):
    print(f"  {k:3.1f} 표준편차:  정규 {a:12.2f} 해 · "
          f"조정 t({DF4}) {b:8.2f} 해   (비 {a / b:7.1f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                               gridspec_kw={"width_ratios": [1.25, 1]})
ax1.plot(KS4, yn, "o-", color="tab:red", lw=3.0, ms=11,
         label="정규라고 믿을 때")
ax1.plot(KS4, yt, "s-", color="tab:blue", lw=3.0, ms=10,
         label="꼬리를 두껍게 잡을 때 (초과첨도 1)")
ax1.set_yscale("log")
# 로그 눈금 이름표를 맨 문자열로 적는다 (기본 이름표는 mathtext 라 두부가 된다)
ax1.set_yticks([0.1, 1, 10, 100, 1000, 10_000, 100_000],
               ["0.1", "1", "10", "100", "1,000", "1만", "10만"])
ax1.set_ylim(0.08, 3e5)
ax1.axhline(40, color="0.45", lw=1.8, ls=":")
ax1.text(2.05, 46, "한 세대 (40 해)", ha="left", va="bottom", fontsize=10.5,
         color="0.35", fontweight="bold")
ax1.axhline(5, color="0.45", lw=1.8, ls=":")
ax1.text(2.05, 5.6, "한 임기 (5 해)", ha="left", va="bottom", fontsize=10.5,
         color="0.35", fontweight="bold")
i4 = int(np.where(KS4 == 4.0)[0][0])
ax1.plot([4.0], [yn[i4]], "o", ms=20, mfc="none", mec="tab:red", mew=2.6)
ax1.plot([4.0], [yt[i4]], "o", ms=20, mfc="none", mec="tab:blue", mew=2.6)
ax1.annotate(f"4 표준편차짜리 하루\n정규는 {yn[i4]:.1f} 해에 한 번",
             xy=(4.0, yn[i4]), xytext=(2.08, 1.9e4), fontsize=11.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.annotate(f"같은 하루가 {yt[i4]:.1f} 해에 한 번", xy=(4.0, yt[i4]),
             xytext=(3.95, 0.20), fontsize=11.5, color="tab:blue",
             fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax1.set_xlabel("평균에서 몇 표준편차 떨어진 하루인가", fontsize=12)
ax1.set_ylabel("그런 하루를 기다리는 햇수 (로그 눈금)", fontsize=12)
ax1.set_title("같은 하루에 두 모형이 다른 이름을 붙인다", fontsize=13)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(alpha=0.25)

ratio = yn / yt
ax2.bar(np.arange(len(KS4)), ratio, color="tab:purple", width=0.6)
for i, r in enumerate(ratio):
    ax2.text(i, r * 1.12, f"{r:,.0f} 배" if r >= 10 else f"{r:.1f} 배",
             ha="center", va="bottom", fontsize=10.8, fontweight="bold",
             color="tab:purple")
ax2.set_yscale("log")
ax2.set_yticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
ax2.set_ylim(0.8, 6000)
ax2.set_xticks(np.arange(len(KS4)), [f"{k:.1f}" for k in KS4], fontsize=11)
ax2.set_xlabel("평균에서 몇 표준편차 떨어진 하루인가", fontsize=12)
ax2.set_ylabel("정규가 부른 햇수 ÷ 두꺼운 꼬리가 부른 햇수", fontsize=11.5)
ax2.set_title("꼬리로 갈수록 두 이름이 빠르게 벌어진다", fontsize=13)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("한 세대에 한 번 볼 일이 한 임기 안에 두어 번 볼 일로",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, INTRO + "generation_to_term.png")


# ════════════════════════════════════════════════════════════════════
# 5. graphical_methods/histogram_density — 보는 것과 못 보는 것
#
# 쪽의 histogram_bins_roughness 가 「구간 수를 바꿔 그린 같은 자료」를
# 이미 그린다. 곧 들머리의 첫 문단(봉우리가 1 에서 53 으로)은 그 그림의
# 것이다. 여기서는 셋째 문단만 그린다 — 「꼬리가 두꺼워도 이 그림에서는
# 거의 티가 나지 않는다. 그래도 봉우리가 둘인 자료는 여기서 가장 잘
# 보이고 Q-Q 그림은 오히려 그것을 놓친다.」
#
# 그래서 두 자료 × 두 그림의 2×2 를 그린다. 대각선으로 엇갈리는 것이
# 요점이다 — 한 그림이 못 보는 것을 다른 그림이 보고, 그 반대도 맞다.
# ════════════════════════════════════════════════════════════════════
r5 = np.random.default_rng(14_005)
N5 = 1000
heavy = r5.standard_t(3, N5)
heavy = (heavy - heavy.mean()) / heavy.std()
lab = r5.random(N5) < 0.5
bimod = np.where(lab, r5.normal(-1.6, 0.75, N5), r5.normal(1.6, 0.75, N5))
bimod = (bimod - bimod.mean()) / bimod.std()

print("\n[histogram_density] 두 자료를 두 그림으로 본다")
for nm, d in [("꼬리가 두꺼운 자료", heavy), ("봉우리가 둘인 자료", bimod)]:
    print(f"  {nm}: 초과첨도 {stats.kurtosis(d):+.3f} · "
          f"샤피로–윌크 p = {stats.shapiro(d).pvalue:.3g}")

fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.2))
g = np.linspace(-4.2, 4.2, 400)
for row, (nm, d, verdict) in enumerate([
        ("꼬리가 두꺼운 자료", heavy, ("봉우리만 높고 꼬리는 안 보인다",)),
        ("봉우리가 둘인 자료", bimod, ("한눈에 보인다",))]):
    ax = axs[row, 0]
    ax.hist(d, bins=42, density=True, color="0.74", edgecolor="none")
    ax.plot(g, stats.norm.pdf(g), color="tab:red", lw=2.4)
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylabel(nm, fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    good = row == 1
    ax.text(0.97, 0.93, verdict[0], transform=ax.transAxes, ha="right",
            va="top", fontsize=12.5, fontweight="bold",
            color="tab:green" if good else "tab:red",
            bbox=dict(boxstyle="round,pad=0.38", fc="white",
                      ec="tab:green" if good else "tab:red"))

    ax = axs[row, 1]
    (osm, osr), _ = stats.probplot(d, dist="norm")
    ax.plot(osm, osr, "o", ms=3.0, color="0.35")
    ax.plot([-3.5, 3.5], [-3.5, 3.5], color="tab:red", lw=2.2)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-6.2, 6.2)
    ax.grid(alpha=0.2)
    hit = row == 0
    ax.text(0.03, 0.93, "양 끝이 크게 휜다" if hit
            else "휘긴 하지만 봉우리가 둘이라는 것은 읽히지 않는다",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12.5 if hit else 11.2,
            fontweight="bold", color="tab:green" if hit else "tab:red",
            bbox=dict(boxstyle="round,pad=0.38", fc="white",
                      ec="tab:green" if hit else "tab:red"))
axs[0, 0].set_title("히스토그램", fontsize=14)
axs[0, 1].set_title("Q-Q 그림", fontsize=14)
axs[1, 0].set_xlabel("표준화한 값", fontsize=11)
axs[1, 1].set_xlabel("정규 분위수", fontsize=11)
fig.suptitle("한 그림이 못 보는 것을 다른 그림이 본다 — 그 반대도 맞다",
             fontsize=15, y=0.985)
plt.tight_layout()
save(fig, GRAPH + "what_the_histogram_cannot_see.png")


# ════════════════════════════════════════════════════════════════════
# 6. graphical_methods/boxplots — 울타리가 잘라내는 비율
#
# 쪽은 네 분포의 상자그림을 각각 그려 **모양**을 보인다. 들머리가 드는
# 것은 모양이 아니라 **비율**이다(정규 0.70% 대 지수 4.81%, 그리고
# 대수정규에서 이상점 21 개가 전부 위쪽에만). 어느 그림도 그 비율을
# 맞대어 놓지 않으므로 그것만 그린다. 들머리의 판정 규칙이 「이상점이
# 있다」가 아니라 「정규의 기준선보다 많은가」이므로 기준선을 가로선으로
# 깔고, 한쪽만 비는 것이 가장 강한 신호라는 말도 위·아래로 쪼개 보인다.
# ════════════════════════════════════════════════════════════════════
r6 = np.random.default_rng(14_006)
N6, REPS6 = 1000, 4000


def fence_rates(sampler):
    up = lo = 0.0
    for _ in range(REPS6):
        d = sampler(N6)
        q1, q3 = np.percentile(d, [25, 75])
        iq = q3 - q1
        lo += (d < q1 - 1.5 * iq).mean()
        up += (d > q3 + 1.5 * iq).mean()
    return lo / REPS6, up / REPS6


DISTS = [("정규", lambda n: r6.standard_normal(n)),
         ("균등", lambda n: r6.uniform(-1, 1, n)),
         ("t(3)", lambda n: r6.standard_t(3, n)),
         ("지수", lambda n: r6.exponential(1, n)),
         ("대수정규", lambda n: np.exp(r6.normal(0, 1, n)))]
los, ups = [], []
print("\n[boxplots] 1.5 사분위범위 울타리가 잘라내는 비율")
for nm, f in DISTS:
    a, b = fence_rates(f)
    los.append(a)
    ups.append(b)
    print(f"  {nm:7s} 아래 {100 * a:6.3f}% · 위 {100 * b:6.3f}% · "
          f"합 {100 * (a + b):6.3f}%   (n = {N6} 이면 {N6 * (a + b):5.1f} 개)")

base = los[0] + ups[0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
bx = np.arange(len(DISTS))
w = 0.38
ax1.bar(bx - w / 2, [100 * v for v in los], width=w, color="tab:blue",
        label="아래쪽 울타리 바깥")
ax1.bar(bx + w / 2, [100 * v for v in ups], width=w, color="tab:red",
        label="위쪽 울타리 바깥")
for i, (a, b) in enumerate(zip(los, ups)):
    ax1.text(i - w / 2, 100 * a + 0.08, f"{100 * a:.2f}", ha="center",
             va="bottom", fontsize=10.5, fontweight="bold", color="tab:blue")
    ax1.text(i + w / 2, 100 * b + 0.08, f"{100 * b:.2f}", ha="center",
             va="bottom", fontsize=10.5, fontweight="bold", color="tab:red")
ax1.axhline(100 * base / 2, color="0.3", lw=2.0, ls="--")
ax1.text(1.0, 100 * base / 2 + 0.22,
         f"가로 점선이 정규의 한쪽 기준선 {100 * base / 2:.2f}%",
         ha="center", va="bottom", fontsize=10.8, color="0.3",
         fontweight="bold")
ax1.set_xticks(bx, [d[0] for d in DISTS], fontsize=12)
ax1.set_ylabel("울타리 바깥에 찍히는 점의 비율 (퍼센트)", fontsize=11.5)
ax1.set_ylim(0, 9.4)
ax1.set_title("「이상점이 있다」가 아니라 「기준선보다 많은가」",
              fontsize=12.5)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(axis="y", alpha=0.25)

# 균등분포는 울타리 바깥이 아예 없으므로 몫이 정의되지 않는다
share = [(b / (a + b) if a + b > 0 else None) for a, b in zip(los, ups)]
cols = ["0.5" if (s is None or abs(s - 0.5) < 0.25) else "tab:red"
        for s in share]
ax2.barh(bx, [0 if s is None else 100 * s for s in share],
         color=cols, height=0.55)
ax2.axvline(50, color="0.3", lw=2.2, ls="--")
for i, s in enumerate(share):
    if s is None:
        ax2.text(2.0, i, "울타리 바깥이 하나도 없다", va="center",
                 ha="left", fontsize=11.5, fontweight="bold", color="0.4")
    else:
        ax2.text(100 * s + 1.6, i, f"{100 * s:.0f}%", va="center",
                 ha="left", fontsize=12.5, fontweight="bold")
ax2.set_yticks(bx, [d[0] for d in DISTS], fontsize=12)
ax2.set_xlim(0, 128)
ax2.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
ax2.set_xlabel("울타리 바깥 점 가운데 위쪽에 찍힌 몫 (퍼센트)",
               fontsize=11.5)
ax2.set_title("한쪽이 거의 비는 것이 가장 강한 신호다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)
ax2.set_ylim(-0.95, len(DISTS) - 0.3)
ax2.text(50, -0.72, "대칭이면 50%", ha="center", va="center", fontsize=10.8,
         color="0.3", fontweight="bold")

fig.suptitle("정규 1000 개에도 이상점이 일곱 개 — 세어야 할 것은 개수가 "
             "아니다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, GRAPH + "fence_cuts_and_one_sided.png")
