"""14장 들어가는 이야기 그림 — 넷째 묶음 여섯 (마지막).

  1. sensitivity_practical  양측이라 적었는데 한쪽이 전부였다
  2. transformations        왜도를 0 으로 맞추면 첨도가 반대쪽으로 지나간다
  3. nonparametric          U 가 보는 모수는 평균이 아니다
  4. bootstrapping          좁은 것이 나은 것이 아니다
  5. t_tests_anova          세 번 따로 걸기와 잔차를 모으기
  6. financial_returns      적어 둔 간격과 실제 간격

만드는 파일:
  docs/ch14/limitations/img/one_tail_is_all_of_it.png
  docs/ch14/non_normal_data/img/fixing_skew_overshoots_kurtosis.png
  docs/ch14/non_normal_data/img/u_sees_a_different_parameter.png
  docs/ch14/non_normal_data/img/narrower_is_not_better.png
  docs/ch14/applications/img/three_tests_or_pooled_residuals.png
  docs/ch14/applications/img/written_gap_actual_gap.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

LIM = "docs/ch14/limitations/img/"
NND = "docs/ch14/non_normal_data/img/"
APP = "docs/ch14/applications/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def lognorm_skew(s):
    return (np.exp(s ** 2) + 2) * np.sqrt(np.exp(s ** 2) - 1)


# ════════════════════════════════════════════════════════════════════
# 1. sensitivity_practical — 양측이라 적었는데 한쪽이 전부였다
#
# 쪽의 practical_vs_statistical 은 「가벼운 이탈과 심한 이탈에서 검정의
# 기각률과 t 구간 포함률」을 그린다. 곧 **양측** 수치다. 들머리의 머리글은
# 그것을 쪼갠 데 있다 — 「양측 기각률이 0.2533 인데 아래쪽 꼬리가 0.2532,
# 위쪽이 0.0001 이다. 명목 0.05 짜리 양측검정의 거의 전부가 한쪽 꼬리에서
# 나온다.」 두 꼬리를 따로 센 그림이 이 쪽에 없으므로 그것만 그린다.
# ════════════════════════════════════════════════════════════════════
SIGS = [0.1, 0.4, 0.8, 1.2, 1.6, 2.0]
skews1 = [lognorm_skew(s) for s in SIGS]
NN1 = [15, 100]
REPS1 = 40_000
r1 = np.random.default_rng(14_301)
res = {n: {"lo": [], "hi": []} for n in NN1}
for n in NN1:
    for s in SIGS:
        d = np.exp(r1.normal(0, s, size=(REPS1, n)))
        mu0 = np.exp(s ** 2 / 2)
        tstat = (d.mean(1) - mu0) / (d.std(1, ddof=1) / np.sqrt(n))
        c = stats.t.ppf(0.975, n - 1)
        res[n]["lo"].append((tstat < -c).mean())
        res[n]["hi"].append((tstat > c).mean())

print("[sensitivity_practical] 일표본 t 검정의 두 꼬리를 따로 센다")
for n in NN1:
    print(f"  n = {n}")
    for sk, a, b in zip(skews1, res[n]["lo"], res[n]["hi"]):
        print(f"    왜도 {sk:8.2f}:  아래쪽 {a:.4f} · 위쪽 {b:.4f} · "
              f"합 {a + b:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
for ax, n in zip(axs, NN1):
    lo = np.array(res[n]["lo"])
    hi = np.array(res[n]["hi"])
    ax.plot(skews1, lo + hi, "o-", color="0.3", lw=3.0, ms=11,
            label="양측 기각률 (적어 두는 수)")
    ax.plot(skews1, lo, "s-", color="tab:red", lw=2.6, ms=9,
            label="아래쪽 꼬리")
    ax.plot(skews1, hi, "^-", color="tab:blue", lw=2.6, ms=9,
            label="위쪽 꼬리")
    ax.axhline(0.05, color="0.4", lw=2.0, ls="--")
    ax.axhline(0.025, color="0.6", lw=1.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(skews1, [f"{s:.1f}" if s < 10 else f"{s:.0f}"
                           for s in skews1])
    ax.set_xlabel("모집단의 왜도 (로그 눈금)", fontsize=11.5)
    ax.set_title(f"n = {n}", fontsize=13.5)
    ax.set_ylim(0, 0.45)
    ax.grid(alpha=0.25)
axs[0].set_ylabel("기각률", fontsize=11.5)
axs[0].legend(fontsize=10.6, loc="upper left")
axs[0].text(skews1[0], 0.056, "명목 0.05", ha="left", va="bottom",
            fontsize=10.5, color="0.35", fontweight="bold")
axs[0].text(skews1[0], 0.0075, "한쪽에 있어야 할 0.025", ha="left",
            va="bottom", fontsize=10.5, color="0.5", fontweight="bold")
i_last = len(SIGS) - 1
axs[0].annotate(f"왜도 {skews1[i_last]:.0f} 에서 양측 "
                f"{res[15]['lo'][i_last] + res[15]['hi'][i_last]:.4f} 가\n"
                f"거의 전부 아래쪽이다 "
                f"({res[15]['hi'][i_last]:.4f} 만 위쪽)",
                xy=(skews1[i_last], res[15]["lo"][i_last]),
                xytext=(skews1[1], 0.33), fontsize=11, color="tab:red",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
axs[1].annotate("n 을 일곱 배로 늘려도\n한쪽 쏠림은 그대로다",
                xy=(skews1[i_last], res[100]["lo"][i_last]),
                xytext=(skews1[0], 0.33), fontsize=11, color="tab:red",
                fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

fig.suptitle("양측이라 적었는데 한쪽이 전부였다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, LIM + "one_tail_is_all_of_it.png")


# ════════════════════════════════════════════════════════════════════
# 2. transformations — 왜도를 잡으면 첨도가 반대쪽으로 지나간다
#
# 쪽의 그림 넷은 변환 **전후의 히스토그램**이다. 들머리의 머리글은
# 그것으로 보이지 않는 것이다 — 「왜도가 2.05 에서 0.075 로 내려가는데
# 샤피로–윌크는 p = 3.33e-11 로 기각한다. 초과첨도가 +6.48 에서 -0.92 로
# 부호를 건너 반대쪽으로 지나쳐 버렸다.」
#
# 그래서 거듭제곱 지수를 훑으며 두 적률을 함께 그린다. 왜도를 0 으로
# 만드는 지수와 첨도를 0 으로 만드는 지수가 **같은 자리가 아니라는 것**이
# 한눈에 보이고, 「고쳐 주는 것과 고쳐 주지 못하는 것이 어디서 갈리는가」에
# 그림으로 답한다.
# ════════════════════════════════════════════════════════════════════
# 쪽의 보기 1 과 **똑같이** 만든다 — seed 0, 척도 2, n = 1000, 그리고
# 결정적으로 boxcox(x + 1) 이다. 처음에 이동상수 없이 boxcox(x) 로 훑었더니
# 람다가 +0.28 이 나오고 샤피로–윌크가 p = 0.32 로 **통과**해 버려서
# 들머리와 어긋났다. 쪽이 거는 것은 x + 1 이고, 그 이동 하나가 람다를
# -0.41 로 끌고 가 결과를 뒤집는다. 들머리의 셋째 문단이 적는
# 「손잡이가 조용히 숨는 문제」가 바로 이것이다.
np.random.seed(0)
x2 = np.random.exponential(scale=2, size=1000) + 1.0
LAM = np.linspace(-1.2, 0.8, 201)
sk2, ku2, pw2 = [], [], []
for lam in LAM:
    y = np.log(x2) if abs(lam) < 1e-9 else (x2 ** lam - 1) / lam
    sk2.append(stats.skew(y))
    ku2.append(stats.kurtosis(y))
    pw2.append(stats.shapiro(y).pvalue)
sk2, ku2, pw2 = map(np.array, (sk2, ku2, pw2))
lam_sk = LAM[int(np.argmin(np.abs(sk2)))]
lam_ku = LAM[int(np.argmin(np.abs(ku2)))]
lam_bc = stats.boxcox_normmax(x2)
j_bc = int(np.argmin(np.abs(LAM - lam_bc)))
print("\n[transformations] 쪽의 보기 1 과 같은 자료로 지수를 훑는다")
print(f"  원자료(+1 하기 전): 왜도 {stats.skew(x2 - 1):+.4f} · "
      f"초과첨도 {stats.kurtosis(x2 - 1):+.4f}  (쪽은 +2.0526 · +6.4761)")
print(f"  Box-Cox 가 고르는 지수 {lam_bc:+.4f}  (쪽은 -0.4127)")
print(f"    그 자리: 왜도 {sk2[j_bc]:+.4f} · 초과첨도 {ku2[j_bc]:+.4f} · "
      f"p = {pw2[j_bc]:.3g}  (쪽은 +0.0747 · -0.9237 · 3.33e-11)")
print(f"  왜도를 0 으로 만드는 지수 {lam_sk:+.3f} → "
      f"초과첨도 {ku2[int(np.argmin(np.abs(sk2)))]:+.4f}")
print(f"  초과첨도를 0 으로 만드는 지수 {lam_ku:+.3f} → "
      f"왜도 {sk2[int(np.argmin(np.abs(ku2)))]:+.4f}")
print(f"  p 의 최댓값 {pw2.max():.4g} (지수 {LAM[int(np.argmax(pw2))]:+.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.2, 1]})
ax1.axhline(0, color="0.3", lw=2.0, ls="--")
ax1.plot(LAM, sk2, "-", color="tab:blue", lw=3.2, label="왜도")
ax1.plot(LAM, ku2, "-", color="tab:red", lw=3.2, label="초과첨도")
ax1.axvline(lam_sk, color="tab:blue", lw=2.0, ls=":")
ax1.axvline(lam_ku, color="tab:red", lw=2.0, ls=":")
ax1.plot([lam_sk], [0], "o", ms=14, mfc="white", mec="tab:blue", mew=3)
ax1.plot([lam_ku], [0], "o", ms=14, mfc="white", mec="tab:red", mew=3)
ax1.axvline(lam_bc, color="0.25", lw=2.4)
ax1.text(lam_bc - 0.025, -3.1, f"Box-Cox 가 고르는 지수 {lam_bc:+.3f}",
         ha="right", va="bottom", fontsize=11.3, color="0.2",
         fontweight="bold", rotation=90)
ax1.set_xlim(LAM[0], LAM[-1])
ax1.set_ylim(-3.2, 7.2)
ax1.set_xlabel("Box-Cox 의 거듭제곱 지수", fontsize=11.5)
ax1.set_ylabel("변환한 자료의 적률", fontsize=11.5)
ax1.set_title("두 적률이 같은 자리에서 0 이 되지 않는다", fontsize=13)
ax1.legend(fontsize=11.5, loc="lower right")
ax1.grid(alpha=0.25)
j_sk = int(np.argmin(np.abs(sk2)))
ax1.annotate(f"왜도를 0 으로 맞추면({lam_sk:+.2f})\n"
             f"초과첨도가 {ku2[j_sk]:+.2f} 로 지나쳐 있다",
             xy=(lam_sk, ku2[j_sk]), xytext=(lam_sk - 0.06, -2.6),
             fontsize=11.3, color="tab:red", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
j_ku = int(np.argmin(np.abs(ku2)))
ax1.annotate(f"첨도를 0 으로 맞추면({lam_ku:+.2f})\n"
             f"왜도가 {sk2[j_ku]:+.2f} 로 남는다",
             xy=(lam_ku, sk2[j_ku]), xytext=(lam_ku + 0.06, 6.6),
             fontsize=11.3, color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))

ax2.plot(LAM, np.maximum(pw2, 1e-20), "-", color="0.25", lw=3.2)
ax2.axhline(0.05, color="tab:red", lw=2.2, ls="--")
ax2.axvline(lam_sk, color="tab:blue", lw=2.0, ls=":")
ax2.axvline(lam_bc, color="0.25", lw=2.4)
ax2.plot([lam_bc], [pw2[j_bc]], "o", ms=14, mfc="white", mec="0.2", mew=3)
ax2.annotate(f"Box-Cox 가 고른 자리\np = {pw2[j_bc]:.2g}",
             xy=(lam_bc, pw2[j_bc]), xytext=(-1.15, 1e-5),
             fontsize=11.3, color="0.2", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="0.2", lw=1.9))
ax2.text(lam_sk - 0.03, 2e-19, "왜도를 0 으로 맞춘 자리", ha="right",
         va="bottom", fontsize=10.8, color="tab:blue", fontweight="bold",
         rotation=90)
ax2.set_yscale("log")
ax2.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 0.05, 1],
               ["1e-16", "1e-12", "1e-8", "1e-4", "0.05", "1"])
ax2.set_ylim(1e-20, 4)
ax2.set_xlim(LAM[0], LAM[-1])
ax2.set_xlabel("Box-Cox 의 거듭제곱 지수", fontsize=11.5)
ax2.set_ylabel("샤피로–윌크 p 값 (로그 눈금)", fontsize=11.5)
ax2.set_title("어느 지수에서도 정규성을 통과하지 못한다",
              fontsize=12.5)
ax2.grid(alpha=0.25, which="both")
ax2.annotate(f"가장 높은 자리는 지수 {LAM[int(np.argmax(pw2))]:+.2f} 의 "
             f"{pw2.max():.2g} 다", xy=(LAM[int(np.argmax(pw2))], pw2.max()),
             xytext=(0.78, 2e-6), fontsize=11.3,
             color="tab:red", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

fig.suptitle("왜도를 잡았는데 검정은 여전히 기각한다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, NND + "fixing_skew_overshoots_kurtosis.png")


# ════════════════════════════════════════════════════════════════════
# 3. nonparametric — U 가 보는 모수는 평균이 아니다
#
# 쪽의 rank_test_power 가 「세 모집단에서 두 검정의 검정력」을 그린다.
# 곧 들머리의 둘째·셋째 문단은 그 그림의 것이다. 남은 것은 **첫 문단**이고
# 그것이 이 쪽의 머리글이다 — 「검정이 실제로 보는 모수는 세타이고, 이
# 자료에서 그 값은 0.5 가 아닌 0.4 다.」
#
# 가장 센 꼴로 보인다. **평균이 정확히 같은데 세타가 0.5 가 아닌 짝**을
# 만들면 t 검정은 「차이 없다」고, 순위검정은 「차이 있다」고 말한다.
# 둘 다 자기 모수에 대해서는 맞는 말이다.
# ════════════════════════════════════════════════════════════════════
N3, REPS3 = 200, 4000
r3 = np.random.default_rng(14_303)
# 처음 만든 짝은 세타가 0.503 으로 나와 두 모수가 **둘 다** 귀무였다.
# 평균을 묶어 두고 세타만 떼어 놓으려면 B 를 두 덩어리로 갈라야 한다.
#   A ~ 1 근처에 좁게 몰린 값
#   B ~ 열에 여섯은 0 가까이, 넷은 2.35 가까이  (평균은 1 로 같다)
# 그러면 A 가 더 클 확률이 정확히 0.6 이다 — 평균은 같은데 세타가 아니다.
PLOW = 0.6
HIGH = (1.0 - PLOW * 0.1) / (1.0 - PLOW)


def draw_a(m, n):
    return r3.normal(1.0, 0.10, (m, n))


def draw_b(m, n):
    lowgrp = r3.random((m, n)) < PLOW
    return np.where(lowgrp, r3.uniform(0.0, 0.2, (m, n)),
                    r3.uniform(HIGH - 0.1, HIGH + 0.1, (m, n)))


A = draw_a(REPS3, N3)
B = draw_b(REPS3, N3)
a_big = draw_a(1, 400_000)[0]
b_big = draw_b(1, 400_000)[0]
theta_true = (a_big > b_big).mean()
mean_a, mean_b = a_big.mean(), b_big.mean()
t_rej = np.array([stats.ttest_ind(A[i], B[i], equal_var=False).pvalue
                  for i in range(REPS3)]) < 0.05
u_rej = np.array([stats.mannwhitneyu(A[i], B[i]).pvalue
                  for i in range(REPS3)]) < 0.05
print("\n[nonparametric] 평균이 같은데 세타가 0.5 가 아닌 짝")
print(f"  모집단 평균 {mean_a:.4f} 대 {mean_b:.4f} (차이 "
      f"{mean_a - mean_b:+.4f})")
print(f"  세타 = P(A > B) = {theta_true:.4f}")
print(f"  n = {N3} 에서 t 검정 기각률 {t_rej.mean():.4f} · "
      f"순위검정 기각률 {u_rej.mean():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.25, 1]})
g = np.linspace(0, 5, 400)
bb = np.linspace(-0.2, 3.0, 130)
ax1.hist(a_big, bins=bb, density=True, alpha=0.55, color="tab:blue",
         label="집단 A")
ax1.hist(b_big, bins=bb, density=True, alpha=0.55, color="tab:red",
         label="집단 B")
ax1.axvline(mean_a, color="0.15", lw=3.0)
ax1.text(mean_a + 0.06, ax1.get_ylim()[1] * 0.9, "두 평균이 여기서 겹친다",
         ha="left", va="top", fontsize=11.5, color="0.15",
         fontweight="bold")
ax1.set_xlim(-0.2, 3.0)
ax1.set_xlabel("값", fontsize=11.5)
ax1.set_ylabel("밀도", fontsize=11.5)
ax1.set_title(f"평균은 {mean_a:.3f} 과 {mean_b:.3f} 로 같다",
              fontsize=13)
ax1.legend(fontsize=11.5, loc="lower right")
ax1.grid(alpha=0.25)
ax1.text(0.97, 0.62, f"그런데 아무 둘을 집어\n"
         f"A 가 더 클 확률은 {theta_true:.3f} 다", transform=ax1.transAxes,
         ha="right", va="top", fontsize=12, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.65"))

bars = [t_rej.mean(), u_rej.mean()]
ax2.bar([0, 1], bars, color=["tab:blue", "tab:red"], width=0.5)
for i, v in enumerate(bars):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
             fontsize=15, fontweight="bold")
ax2.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax2.text(-0.44, 0.08, "그 모수에 차이가 없으면 0.05", ha="left",
         va="bottom",
         fontsize=11, color="0.3", fontweight="bold")
ax2.set_xticks([0, 1], ["평균을 보는 t 검정", "순위를 보는 검정"],
               fontsize=12.5)
ax2.set_ylabel(f"n = {N3} 에서 기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, 1.12)
ax2.set_title("둘 다 자기 모수에 대해서는 맞는 말이다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("U 가 본 것은 평균이 아니었다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, NND + "u_sees_a_different_parameter.png")


# ════════════════════════════════════════════════════════════════════
# 4. bootstrapping — 좁은 것이 나은 것이 아니다
#
# 쪽의 bootstrap_shape 는 「붓스트랩 분포가 표집분포의 비대칭을 따라가는
# 모습과 두 구간의 포함률」을, bootstrap_tail 은 꼬리 분위수에서의 붕괴를
# 그린다. 들머리의 **첫 문단**은 그 둘이 건드리지 않는다 — 「붓스트랩
# 구간이 [1.48, 11.33] 이고 t 구간이 [0.52, 12.21] 로 더 넓다. 좁아진
# 이쪽이 더 믿을 만한가. 아니다.」
#
# 그래서 폭과 포함률을 **한 평면에** 놓는다. 붓스트랩이 왼쪽 아래 —
# 좁은데 더 자주 빗나가는 자리 — 에 있는 것이 보인다.
# ════════════════════════════════════════════════════════════════════
NS4 = [10, 15, 20, 30, 50, 80]
REPS4, B4 = 4000, 999
r4 = np.random.default_rng(14_304)
SIG4 = 0.9
mu4 = np.exp(SIG4 ** 2 / 2)
cov_t, cov_b, wid_t, wid_b = [], [], [], []
for n in NS4:
    d = np.exp(r4.normal(0, SIG4, size=(REPS4, n)))
    m, sd = d.mean(1), d.std(1, ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    cov_t.append(((m - half <= mu4) & (mu4 <= m + half)).mean())
    wid_t.append((2 * half).mean())
    idx = r4.integers(0, n, size=(REPS4, B4, n))
    bs = np.take_along_axis(d[:, None, :], idx, axis=2).mean(2)
    lo = np.percentile(bs, 2.5, axis=1)
    hi = np.percentile(bs, 97.5, axis=1)
    cov_b.append(((lo <= mu4) & (mu4 <= hi)).mean())
    wid_b.append((hi - lo).mean())

print("\n[bootstrapping] 폭과 포함률을 함께 본다")
for n, ct, cb, wt, wb in zip(NS4, cov_t, cov_b, wid_t, wid_b):
    print(f"  n = {n:3d}:  t 구간 포함률 {ct:.4f} 폭 {wt:.4f} · "
          f"붓스트랩 포함률 {cb:.4f} 폭 {wb:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.axhline(0.95, color="0.3", lw=2.2, ls="--")
ax1.text(NS4[-1], 0.953, "약속한 0.95", ha="right", va="bottom",
         fontsize=11, color="0.3", fontweight="bold")
ax1.plot(NS4, cov_t, "o-", color="tab:blue", lw=3.0, ms=11,
         label="t 구간")
ax1.plot(NS4, cov_b, "s-", color="tab:red", lw=3.0, ms=10,
         label="백분위수 붓스트랩 구간")
ax1.set_xlabel("표본크기 n", fontsize=11.5)
ax1.set_ylabel("실제 포함률", fontsize=11.5)
ax1.set_ylim(0.72, 0.97)
ax1.set_title("붓스트랩 쪽이 작은 표본에서 더 자주 빗나간다", fontsize=12.5)
ax1.legend(fontsize=11.5, loc="lower right")
ax1.grid(alpha=0.25)

ax2.plot(wid_t, cov_t, "o-", color="tab:blue", lw=2.6, ms=12,
         label="t 구간")
ax2.plot(wid_b, cov_b, "s-", color="tab:red", lw=2.6, ms=11,
         label="백분위수 붓스트랩 구간")
for w_, c_, n in zip(wid_t, cov_t, NS4):
    ax2.annotate(f"n = {n}", xy=(w_, c_), xytext=(6, 5),
                 textcoords="offset points", fontsize=9.5,
                 color="tab:blue", fontweight="bold")
for w_, c_, n in zip(wid_b, cov_b, NS4):
    ax2.annotate(f"n = {n}", xy=(w_, c_), xytext=(-7, -12),
                 textcoords="offset points", fontsize=9.5,
                 color="tab:red", fontweight="bold", ha="right")
ax2.axhline(0.95, color="0.3", lw=2.2, ls="--")
ax2.set_xlabel("구간의 평균 폭 (좁을수록 왼쪽)", fontsize=11.5)
ax2.set_ylabel("실제 포함률", fontsize=11.5)
ax2.set_ylim(0.72, 0.97)
ax2.set_title("같은 n 에서 붓스트랩이 더 좁고 더 자주 빗나간다",
              fontsize=12.5)
ax2.legend(fontsize=11.5, loc="lower right")
ax2.grid(alpha=0.25)
ax2.text(0.03, 0.96, "왼쪽 아래가 「좁은데 못 미더운」 자리다",
         transform=ax2.transAxes, ha="left", va="top", fontsize=11.3,
         color="0.25", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="#f6f6f6", ec="0.72"))

fig.suptitle("붓스트랩이 늘 나은 것은 아니다 — 좁은 것이 나은 것이 아니다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, NND + "narrower_is_not_better.png")


# ════════════════════════════════════════════════════════════════════
# 5. t_tests_anova — 세 번 따로 걸기와 잔차를 모으기
#
# 쪽의 anova_pooled_vs_residuals 가 「합치면 다봉이지만 잔차는 정규」를
# 그려 들머리의 첫 문단을 담고, balance_beats_skew 가 균형 설계의 beat 를
# 담는다. 남은 것은 둘째 문단이다 — 「집단별로 따로 세 번 검정하는 것도
# 좋은 길이 아니다. 집단당 30 개씩 쪼개 쓰면서 수준마저 0.0167 로 낮춰야
# 하니 남는 검정력이 거의 없다. 잔차를 한데 모으면 n = 90 을 다 쓰고
# 수준도 0.05 를 그대로 쓴다.」 그 값을 수로 잰다.
# ════════════════════════════════════════════════════════════════════
NG, K5 = 30, 3
REPS5 = 20_000
DFS = [50, 20, 10, 7, 5, 4, 3]
r5 = np.random.default_rng(14_305)
pool_pow, split_pow = [], []
for df in DFS:
    sc = np.sqrt((df - 2) / df)
    d = r5.standard_t(df, size=(REPS5, K5, NG)) * sc
    d = d + np.array([0.0, 5.0, 10.0])[None, :, None]
    resid = (d - d.mean(2, keepdims=True)).reshape(REPS5, K5 * NG)
    pool_pow.append(np.array([stats.shapiro(row).pvalue
                              for row in resid]) < 0.05)
    hit = np.zeros(REPS5, bool)
    for g in range(K5):
        hit |= np.array([stats.shapiro(row).pvalue
                         for row in d[:, g, :]]) < 0.05 / K5
    split_pow.append(hit)
pool_pow = np.array([p.mean() for p in pool_pow])
split_pow = np.array([p.mean() for p in split_pow])
exk = [6 / (df - 4) if df > 4 else np.nan for df in DFS]
print("\n[t_tests_anova] 잔차를 모으기와 집단별로 세 번 걸기")
for df, e, a, b in zip(DFS, exk, pool_pow, split_pow):
    print(f"  자유도 {df:3d} (초과첨도 {e if e == e else float('inf'):6.2f}):"
          f"  잔차를 모으면 {a:.4f} · 세 번 따로 걸면 {b:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
xx = np.arange(len(DFS))
ax1.plot(xx, pool_pow, "o-", color="tab:blue", lw=3.0, ms=11,
         label=f"잔차를 한데 모아 한 번 (n = {K5 * NG}, 수준 0.05)")
ax1.plot(xx, split_pow, "s-", color="tab:red", lw=3.0, ms=10,
         label=f"집단별로 세 번 (n = {NG} 씩, 수준 {0.05 / K5:.4f})")
ax1.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax1.set_xticks(xx, [f"t({df})" for df in DFS], fontsize=11)
ax1.set_xlabel("집단 안의 분포 (오른쪽으로 갈수록 꼬리가 두껍다)",
               fontsize=11.5)
ax1.set_ylabel("정규성 이탈을 잡아내는 비율", fontsize=11.5)
ax1.set_ylim(0, 1.05)
ax1.set_title("같은 자료인데 거는 자리가 검정력을 가른다", fontsize=12.5)
ax1.legend(fontsize=10.6, loc="upper left")
ax1.grid(alpha=0.25)

gain = pool_pow - split_pow
ax2.bar(xx, gain, color="tab:blue", width=0.6)
for i, v in enumerate(gain):
    ax2.text(i, v + 0.006, f"{v:+.3f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax2.axhline(0, color="0.3", lw=1.8)
ax2.set_xticks(xx, [f"t({df})" for df in DFS], fontsize=11)
ax2.set_xlabel("집단 안의 분포", fontsize=11.5)
ax2.set_ylabel("잔차를 모아서 더 잡는 양", fontsize=11.5)
ax2.set_ylim(min(0, gain.min() * 1.3), max(gain) * 1.35)
ax2.set_title("쪼개 쓰고 수준까지 낮추면 남는 것이 적다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("확인은 어디에 거는가 — 집단 안이고, 더 정확히는 잔차다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, APP + "three_tests_or_pooled_residuals.png")


# ════════════════════════════════════════════════════════════════════
# 6. financial_returns — 적어 둔 간격과 실제 간격
#
# 쪽의 aggregation_kurtosis 는 집계를, var_tail_ratio 는 분위수의 비를,
# sigma_ruler 는 정규에서 기다리는 햇수를 그린다. 들머리의 **첫 문단**이
# 남는다 — 「3 표준편차를 넘는 날의 비율이 0.0118 로 모형의 4.4 배다.
# 한 해에 한 번쯤이라 적어 둔 일이 석 달에 한 번씩 오는 셈이다.」
#
# 햇수가 아니라 **간격**으로 그린다. 「적어 둔 간격」과 「실제 간격」을
# 달력 눈금에 나란히 놓으면 통계 어휘 없이도 읽힌다.
# ════════════════════════════════════════════════════════════════════
TRADING = 252
KS6 = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
DF6 = 5
sc6 = np.sqrt((DF6 - 2) / DF6)
p_norm = 2 * stats.norm.sf(KS6)
p_real = 2 * stats.t.sf(KS6 / sc6, DF6)
gap_norm = 1 / (p_norm * TRADING)        # 해 단위
gap_real = 1 / (p_real * TRADING)
print("\n[financial_returns] 적어 둔 간격과 실제 간격")
for k, a, b in zip(KS6, gap_norm, gap_real):
    print(f"  {k:3.1f} 표준편차:  정규가 적는 간격 {a:8.2f} 해 · "
          f"꼬리를 두껍게 잡으면 {b:7.2f} 해   (비 {a / b:6.1f})")


def human(years):
    days = years * 365.25
    if days < 31:
        return f"{days:.0f} 일마다"
    if years < 1:
        return f"{days / 30.44:.1f} 달마다"
    return f"{years:.1f} 해마다"


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.15, 1]})
yy = np.arange(len(KS6))
h = 0.36
ax1.barh(yy + h / 2, gap_norm, height=h, color="tab:blue",
         label="정규라고 믿을 때 적어 두는 간격")
ax1.barh(yy - h / 2, gap_real, height=h, color="tab:red",
         label="꼬리를 두껍게 잡았을 때의 간격")
for i in range(len(KS6)):
    ax1.text(gap_norm[i] * 1.15, yy[i] + h / 2, human(gap_norm[i]),
             va="center", ha="left", fontsize=11, color="tab:blue",
             fontweight="bold")
    ax1.text(gap_real[i] * 1.15, yy[i] - h / 2, human(gap_real[i]),
             va="center", ha="left", fontsize=11, color="tab:red",
             fontweight="bold")
ax1.set_xscale("log")
ax1.set_xticks([0.01, 0.1, 1, 10, 100],
               ["0.01", "0.1", "1", "10", "100"])
ax1.set_xlim(0.006, 900)
ax1.set_yticks(yy, [f"{k:.1f} 표준편차" for k in KS6], fontsize=12)
ax1.set_xlabel("그런 하루가 오는 간격 (해, 로그 눈금)", fontsize=11.5)
ax1.set_title("적어 둔 간격과 실제 간격", fontsize=13)
ax1.legend(fontsize=10.6, loc="lower right")
ax1.grid(axis="x", alpha=0.25)

ratio6 = p_real / p_norm
ax2.bar(yy, ratio6, color="tab:purple", width=0.6)
for i, v in enumerate(ratio6):
    ax2.text(i, v * 1.1, f"{v:.1f} 배", ha="center", va="bottom",
             fontsize=11.5, fontweight="bold", color="tab:purple")
ax2.axhline(1.0, color="0.3", lw=2.0, ls="--")
ax2.set_yscale("log")
ax2.set_yticks([1, 10, 100], ["1", "10", "100"])
ax2.set_ylim(0.6, 400)
ax2.set_xticks(yy, [f"{k:.1f}" for k in KS6], fontsize=12)
ax2.set_xlabel("평균에서 몇 표준편차 떨어진 하루인가", fontsize=11.5)
ax2.set_ylabel("실제 빈도 ÷ 정규가 적는 빈도 (로그 눈금)", fontsize=11.5)
ax2.set_title("어긋남의 방향이 한쪽으로 정해져 있다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)
ax2.text(0.03, 0.95, "모두 1 보다 위다 —\n정규 모형은 늘 위험을 작게 적는다",
         transform=ax2.transAxes, ha="left", va="top", fontsize=11.3,
         color="tab:purple", fontweight="bold")

fig.suptitle("한 해에 한 번쯤이라 적어 둔 일이 석 달에 한 번씩 온다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, APP + "written_gap_actual_gap.png")
