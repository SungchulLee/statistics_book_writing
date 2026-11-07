"""15장 들어가는 이야기 그림 — 첫 그룹 일곱.

  1. index                    평균에는 뒷배가 있고 산포에는 없다
  2. motivation               시그마가 뮤보다 훨씬 정밀하게 추정된다
  3. common_assumptions       초과첨도가 정하고 n 은 약분된다
  4. chi_square_distribution  정규로 가는 속도가 느리다
  5. chi_square_variance      방향이 첨도의 부호 하나로 갈린다
  6. ci_variance              평균을 재던 감각으로는 표본이 한 자릿수 모자란다
  7. f_distribution_details   비가 1 보다 크다는 것만으로는 증거가 아니다

만드는 파일:
  docs/ch15/img/no_backstop_for_spread.png
  docs/ch15/introduction/img/sigma_is_the_precise_one.png
  docs/ch15/introduction/img/kurtosis_decides_n_cancels.png
  docs/ch15/chi_square_test/img/slow_walk_to_normal.png
  docs/ch15/chi_square_test/img/kurtosis_splits_the_direction.png
  docs/ch15/chi_square_test/img/one_digit_short.png
  docs/ch15/f_test/img/bigger_than_one_is_not_evidence.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

IDX = "docs/ch15/img/"
INTRO = "docs/ch15/introduction/img/"
CHI = "docs/ch15/chi_square_test/img/"
FT = "docs/ch15/f_test/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 1. 장 개요 — 평균에는 뒷배가 있고 산포에는 없다
#
# 쪽에 그림이 하나도 없다. 들머리는 변동성 뭉침으로 열지만 그 그림은
# 15.7 의 volatility_clustering 이 이미 그린다. 들머리가 **이 장의 일**로
# 적는 것은 그 다음 문단이다 — 「평균을 견줄 때는 자료가 종 모양이 아니어도
# 웬만하면 버텨 주는 든든한 뒷배가 있는데, 퍼진 정도를 견줄 때는 그런
# 뒷배가 없다.」
#
# 장 개요는 독자가 첨도도 카이제곱도 아직 모르는 자리다. 그러니 어휘를
# 요구하지 않는 꼴로 그린다 — 같은 비정규 자료에 두 물음을 걸고, 표본을
# 늘릴 때 한쪽만 약속으로 돌아오는 것을 보인다.
# ════════════════════════════════════════════════════════════════════
NS1 = [10, 20, 40, 80, 160, 320, 640, 1280]
REPS1 = 40_000
r1 = np.random.default_rng(15_001)
mean_rate, var_rate = [], []
for n in NS1:
    d = r1.exponential(1.0, size=(REPS1, n))
    xb, sd = d.mean(1), d.std(1, ddof=1)
    tstat = (xb - 1.0) / (sd / np.sqrt(n))
    c = stats.t.ppf(0.975, n - 1)
    mean_rate.append((np.abs(tstat) > c).mean())
    q = (n - 1) * sd ** 2 / 1.0
    var_rate.append(((q < stats.chi2.ppf(0.025, n - 1)) |
                     (q > stats.chi2.ppf(0.975, n - 1))).mean())

print("[장 개요] 같은 자료에 두 물음을 건다 (지수 모집단, 둘 다 참)")
for n, a, b in zip(NS1, mean_rate, var_rate):
    print(f"  n = {n:5d}:  「평균이 그만한가」 {a:.4f} · "
          f"「산포가 그만한가」 {b:.4f}")

fig, ax = plt.subplots(figsize=(10.8, 5.8))
ax.axhline(0.05, color="0.3", lw=2.4, ls="--")
ax.text(1350, 0.062, "약속한 5%", ha="right", va="bottom", fontsize=12,
        color="0.3", fontweight="bold")
ax.plot(NS1, mean_rate, "o-", color="tab:blue", lw=3.2, ms=12,
        label="「두 평균이 같은가」를 묻는 검정")
ax.plot(NS1, var_rate, "s-", color="tab:red", lw=3.2, ms=11,
        label="「두 산포가 같은가」를 묻는 검정")
ax.set_xscale("log")
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에서 두부가 된다
ax.set_xticks(NS1, [f"{n:,}" for n in NS1])
ax.set_xlabel("표본크기 (로그 척도)", fontsize=12)
ax.set_ylabel("아무 차이가 없는데도 「다르다」고 말하는 비율", fontsize=12)
ax.set_ylim(0, 0.46)
ax.legend(fontsize=11.8, loc="center left", bbox_to_anchor=(0.02, 0.38))
ax.grid(alpha=0.25)
ax.annotate(f"자료를 모을수록 약속으로 돌아온다\n"
            f"({mean_rate[0]:.3f} → {mean_rate[-1]:.3f})",
            xy=(NS1[-3], mean_rate[-3] + 0.012), xytext=(330, 0.175),
            fontsize=12, color="tab:blue", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax.annotate(f"약속으로 가는 것이 아니라\n올라가서 자기 자리에 앉는다\n"
            f"({var_rate[0]:.3f} → {var_rate[-1]:.3f})",
            xy=(NS1[-3], var_rate[-3] + 0.006), xytext=(11, 0.425),
            fontsize=12,
            color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
ax.set_title("같은 자료에 두 물음을 걸어 보면", fontsize=13)
fig.suptitle("평균을 견줄 때 받던 뒷배가 산포에는 없다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, IDX + "no_backstop_for_spread.png")


# ════════════════════════════════════════════════════════════════════
# 2. motivation — 시그마가 뮤보다 훨씬 정밀하게 추정된다
#
# 쪽의 mean_vs_variance_quality 가 「평균이 같고 분산만 다른 두 충전기」를
# 그려 들머리의 첫 문단을 담는다. size_by_distribution 은 넷째 문단이다.
# 남은 것은 **둘째 문단**이고 그것이 가장 덜 알려진 사실이다 —
# 「시그마가 뮤보다 훨씬 정밀하게 추정된다. 10 년치 수익률로 재면
# 뮤햇의 상대오차가 79% 인데 시그마햇은 1.4% 다.」
#
# 까닭은 식에 있다. 연 수익률의 상대오차는 기간의 제곱근에 반비례해
# 천천히 줄고, 변동성의 상대오차는 **관측 횟수**의 제곱근에 반비례하므로
# 같은 기간이라도 날마다 재면 훨씬 빨리 준다. 그 둘을 함께 그린다.
# ════════════════════════════════════════════════════════════════════
MU, SIGMA = 0.08, 0.20            # 연 기대수익률과 연 변동성
YEARS = np.array([1, 2, 5, 10, 20, 50, 100], float)
PER_YEAR = 252
se_mu_rel = (SIGMA / np.sqrt(YEARS)) / MU
se_sig_rel = 1.0 / np.sqrt(2 * YEARS * PER_YEAR)
print("\n[motivation] 같은 기간에서 두 추정값의 상대오차")
for y, a, b in zip(YEARS, se_mu_rel, se_sig_rel):
    print(f"  {y:5.0f} 해:  기대수익률 {100 * a:7.1f}% · "
          f"변동성 {100 * b:5.2f}%   (비 {a / b:6.1f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(YEARS, 100 * se_mu_rel, "o-", color="tab:red", lw=3.0, ms=11,
         label="기대수익률 (평균)")
ax1.plot(YEARS, 100 * se_sig_rel, "s-", color="tab:blue", lw=3.0, ms=10,
         label="변동성 (표준편차)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(YEARS, [f"{y:.0f}" for y in YEARS])
ax1.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax1.set_xlabel("재는 기간 (해, 로그 척도)", fontsize=11.5)
ax1.set_ylabel("추정값의 상대오차 (퍼센트, 로그 척도)", fontsize=11.5)
ax1.set_title("같은 기간인데 두 추정값의 정밀도가 자릿수로 갈린다",
              fontsize=12.5)
ax1.legend(fontsize=11.5, loc="lower left")
ax1.grid(alpha=0.25, which="both")
i10 = int(np.where(YEARS == 10)[0][0])
ax1.plot([10], [100 * se_mu_rel[i10]], "o", ms=20, mfc="none",
         mec="tab:red", mew=2.6)
ax1.plot([10], [100 * se_sig_rel[i10]], "o", ms=20, mfc="none",
         mec="tab:blue", mew=2.6)
ax1.annotate(f"10 년치로 재면 {100 * se_mu_rel[i10]:.0f}%",
             xy=(9, 100 * se_mu_rel[i10]), xytext=(1.05, 24),
             fontsize=11.5, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.annotate(f"같은 10 년치로 {100 * se_sig_rel[i10]:.1f}%",
             xy=(11, 100 * se_sig_rel[i10]), xytext=(17, 3.4),
             fontsize=11.5, color="tab:blue", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))

ax2.plot(YEARS, se_mu_rel / se_sig_rel, "o-", color="tab:purple", lw=3.0,
         ms=11)
ax2.set_xscale("log")
ax2.set_xticks(YEARS, [f"{y:.0f}" for y in YEARS])
ax2.set_xlabel("재는 기간 (해, 로그 척도)", fontsize=11.5)
ax2.set_ylabel("평균의 상대오차 ÷ 변동성의 상대오차", fontsize=11.5)
ax2.set_title("기간을 늘려도 이 비는 줄지 않는다", fontsize=12.5)
ax2.set_ylim(0, max(se_mu_rel / se_sig_rel) * 1.2)
ax2.grid(alpha=0.25)
ax2.text(0.5, 0.42, "둘 다 기간의 제곱근으로 줄어드니\n"
         "비는 처음부터 끝까지 그대로다 —\n"
         "평균 쪽이 더 오래 기다려야 한다는 뜻이 아니라\n"
         "**아무리 기다려도** 따라잡지 못한다는 뜻이다".replace("**", ""),
         transform=ax2.transAxes, ha="center", va="center", fontsize=11.3,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="0.7"))

fig.suptitle("남은 손잡이가 시그마 하나뿐인데, 다행히 그쪽이 더 정밀하다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, INTRO + "sigma_is_the_precise_one.png")


# ════════════════════════════════════════════════════════════════════
# 3. common_assumptions — 약속한 자리가 아니라 첨도가 정한 자리로 간다
#
# 쪽의 chisq_reference_wrong 은 「비정규 자료에서 카이제곱 표집분포가
# 실제 분포와 어긋난다」를 그린다 — 곧 **한 자리에서의 불일치**이다.
# 들머리의 머리글은 그것이 아니라 n 이 하는 일이다 — 「표본을 늘려도
# 낫지 않는다. 지수모집단에서 극한의 실제 크기는 0.327 로 굳는다.」
#
# 처음에는 「가로를 첨도로 놓으면 n 마다의 곡선이 포개진다」를 그리려
# 했는데 재어 보니 포개지지 않는다. 유한한 n 에서는 아직 올라가는
# 중이고(지수에서 n = 25 의 0.146 이 n = 1600 에 0.198), 초과첨도가 같은
# 지수와 t(5) 도 서로 다른 자리에 있다. **포개지는 것은 극한에서다.**
# 그래서 가로를 n 으로 놓고 각자의 점근값을 가로선으로 깔았다. 곡선이
# 0.05 가 아니라 **자기 선**으로 올라가는 것이 들머리의 그림이다.
#
# 점근값은 식으로 나온다. S^2 의 표집분포 폭이 (g2+2)/n 에 비례하므로
# 양측 5% 의 극한 크기가 2*Phi_bar(1.95996*sqrt(2/(g2+2))) 이고,
# 초과첨도 6 에서 그 값이 0.3271 — 쪽이 적는 0.327 이다.
# ════════════════════════════════════════════════════════════════════
FAM = [("정규", 0.0, lambda r, m, n: r.standard_normal((m, n))),
       ("t(8)", 1.5, lambda r, m, n: r.standard_t(8, (m, n))
        * np.sqrt(6 / 8)),
       ("t(6)", 3.0, lambda r, m, n: r.standard_t(6, (m, n))
        * np.sqrt(4 / 6)),
       ("지수", 6.0, lambda r, m, n: r.exponential(1.0, (m, n)))]
NS3 = [25, 50, 100, 200, 400, 800, 1600, 3200]
REPS3 = 30_000
r3 = np.random.default_rng(15_003)


def asym_size(g2):
    """양측 5% 카이제곱 분산검정의 극한 크기."""
    return 2 * stats.norm.sf(stats.norm.ppf(0.975) * np.sqrt(2 / (g2 + 2)))


tracks = {nm: [] for nm, _, _ in FAM}
for n in NS3:
    lo, hi = stats.chi2.ppf([0.025, 0.975], n - 1)
    for nm, g2, f in FAM:
        d = f(r3, REPS3, n)
        q = (n - 1) * d.var(1, ddof=1) / 1.0     # 모분산을 1 로 맞췄다
        tracks[nm].append(((q < lo) | (q > hi)).mean())
print("\n[common_assumptions] 각자의 점근값으로 올라간다 (양측 5%)")
for nm, g2, _ in FAM:
    row = " ".join(f"{v:.4f}" for v in tracks[nm])
    print(f"  {nm:5s} 초과첨도 {g2:4.1f} → 극한 {asym_size(g2):.4f} :  {row}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
COLS = ["0.35", "tab:green", "tab:orange", "tab:red"]
for (nm, g2, _), c in zip(FAM, COLS):
    ax.plot(NS3, tracks[nm], "o-", color=c, lw=2.8, ms=10,
            label=f"{nm} (초과첨도 {g2:.1f})")
    ax.axhline(asym_size(g2), color=c, lw=1.8, ls=":")
    ax.text(3400, asym_size(g2), f"{asym_size(g2):.4f}", ha="left",
            va="center", fontsize=11, color=c, fontweight="bold")
ax.set_xscale("log")
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에서 두부가 된다
ax.set_xticks(NS3, [f"{n:,}" for n in NS3])
ax.set_xlim(22, 3300)
ax.set_ylim(0, 0.39)
ax.set_xlabel("표본크기 n (로그 척도)", fontsize=12)
ax.set_ylabel("명목 5% 양측 검정의 실제 기각률", fontsize=12)
ax.legend(fontsize=11.3, loc="upper left")
ax.grid(alpha=0.25)
ax.annotate("점선은 식으로 구한 극한값이다 —\n"
            "n 을 늘리면 0.05 가 아니라 **이쪽**으로 간다".replace("**", ""),
            xy=(1500, asym_size(3.0)), xytext=(60, 0.252),
            fontsize=12, color="0.2", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.2", lw=2.0))
ax.set_title("곡선마다 자기 극한이 있고 그 극한은 첨도가 정한다",
             fontsize=13)
fig.suptitle("표본을 늘려도 낫지 않는다 — 불일치를 정하는 것은 첨도다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, INTRO + "kurtosis_decides_n_cancels.png")


# ════════════════════════════════════════════════════════════════════
# 4. chi_square_distribution — 정규로 가는 속도가 느리다
#
# 쪽의 그림 넷은 밀도와 누적분포, 두 구성법의 일치, 표본분산 확인이다.
# 들머리가 **마지막 문단**에서 던지는 것은 속도다 — 「자유도를 키우면
# 정규에 가까워지기는 하지만 왜도가 루트 8/d 로만 줄어들어 느리다.」
# 어느 그림도 그 속도를 재지 않으므로 그것만 그린다.
# ════════════════════════════════════════════════════════════════════
DS = np.array([2, 5, 10, 20, 49, 100, 200, 500, 1000, 2000], float)
skew_th = np.sqrt(8 / DS)
r4 = np.random.default_rng(15_004)
ks = []
for d in DS:
    x = stats.chi2.rvs(d, size=60_000, random_state=r4)
    z = (x - d) / np.sqrt(2 * d)
    ks.append(stats.kstest(z, "norm").statistic)
ks = np.array(ks)
need = 8 / 0.1 ** 2          # 왜도가 0.1 아래로 내려가는 자유도
print("\n[chi_square_distribution] 정규로 가는 속도")
for d, s, k in zip(DS, skew_th, ks):
    print(f"  자유도 {d:7.0f}:  왜도 {s:.4f} · 표준화 뒤 KS 거리 {k:.4f}")
print(f"  왜도가 0.1 아래가 되려면 자유도가 {need:.0f} 는 되어야 한다")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(DS, skew_th, "-", color="tab:red", lw=3.2, label="왜도")
ax1.plot(DS, ks, "o", color="tab:blue", ms=10,
         label="표준화한 뒤 정규와의 거리 (재어 본 값)")
ax1.axhline(0.1, color="0.35", lw=2.0, ls=":")
ax1.axvline(need, color="0.35", lw=2.0, ls=":")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks([2, 10, 49, 100, 800, 2000],
               ["2", "10", "49", "100", "800", "2,000"])
ax1.set_yticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
ax1.set_xlabel("자유도 (로그 척도)", fontsize=11.5)
ax1.set_ylabel("정규에서 벗어난 정도 (로그 척도)", fontsize=11.5)
ax1.set_title("자유도의 제곱근으로만 줄어든다", fontsize=13)
ax1.legend(fontsize=10.8, loc="lower left")
ax1.grid(alpha=0.25, which="both")
ax1.annotate(f"왜도가 0.1 아래로 가려면\n자유도가 {need:.0f} 는 되어야 한다",
             xy=(need, 0.1), xytext=(3.0, 0.023), fontsize=11.3,
             color="0.25", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.25", lw=1.9))

for d, col in [(5, "tab:red"), (49, "tab:orange"), (500, "tab:blue")]:
    g = np.linspace(-4, 5, 500)
    ax2.plot(g, stats.chi2.pdf(d + g * np.sqrt(2 * d), d) * np.sqrt(2 * d),
             color=col, lw=2.8, label=f"자유도 {d}")
g = np.linspace(-4, 5, 500)
ax2.plot(g, stats.norm.pdf(g), color="0.25", lw=2.6, ls="--",
         label="정규")
ax2.set_xlim(-4, 5)
ax2.set_ylim(0, 0.52)
ax2.set_xlabel("평균에서 몇 표준편차 떨어졌는가", fontsize=11.5)
ax2.set_ylabel("밀도", fontsize=11.5)
ax2.set_title("자유도 49 에서도 오른쪽이 아직 길다", fontsize=12.5)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.25)

fig.suptitle("평균 49 에 분산 98 — 그래도 아직 정규가 아니다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, CHI + "slow_walk_to_normal.png")


# ════════════════════════════════════════════════════════════════════
# 5. chi_square_variance — 방향이 첨도의 부호 하나로 갈린다
#
# 쪽의 two_sided_critical_values 가 「꼬리마다 알파를 배분하는 두 방식」을
# 그려 들머리의 둘째 문단을 담는다. 들머리의 **셋째 문단**은 그리지 않는다 —
# 「꼬리가 무거우면 오류율이 올라가 위험해지고, 가벼우면 기각을 거의 못 해
# 검정력을 통째로 버린다.」
#
# 처음에 대립을 분산 2 배로 잡았더니 **꼬리가 얇은 쪽의 검정력이 가장
# 높게** 나와(0.890 대 정규 0.785) 들머리와 반대였다. 까닭은 2 배가 너무
# 센 대립이라 크기가 0.003 이어도 거의 다 잡히기 때문이다. 약한 대립에서는
# 순서가 뒤집힌다(1.3 배에서 0.110 대 0.198). 그래서 **둘 다** 그린다 —
# 뒤집히는 자리를 숨기지 않는 쪽이 맞다.
# ════════════════════════════════════════════════════════════════════
SHAPES = [("균등", -1.2, lambda r, m, n: r.uniform(-np.sqrt(3), np.sqrt(3),
                                                   (m, n))),
          ("삼각", -0.6, lambda r, m, n: (r.triangular(-1, 0, 1, (m, n))
                                          * np.sqrt(6))),
          ("정규", 0.0, lambda r, m, n: r.standard_normal((m, n))),
          ("t(10)", 1.0, lambda r, m, n: r.standard_t(10, (m, n))
           * np.sqrt(8 / 10)),
          ("t(7)", 2.0, lambda r, m, n: r.standard_t(7, (m, n))
           * np.sqrt(5 / 7)),
          ("t(5)", 6.0, lambda r, m, n: r.standard_t(5, (m, n))
           * np.sqrt(3 / 5))]
N5, REPS5 = 30, 40_000
ALTS = [1.3, 2.0]
r5 = np.random.default_rng(15_005)
lo5, hi5 = stats.chi2.ppf([0.025, 0.975], N5 - 1)
size5, g25 = [], []
pow5 = {a: [] for a in ALTS}
for nm, g2, f in SHAPES:
    d = f(r5, REPS5, N5)
    q = (N5 - 1) * d.var(1, ddof=1)
    size5.append(((q < lo5) | (q > hi5)).mean())
    for a_ in ALTS:
        pow5[a_].append(((q * a_ < lo5) | (q * a_ > hi5)).mean())
    g25.append(g2)
print(f"\n[chi_square_variance] n = {N5}, 양측 5%")
for (nm, g2, _), s_ in zip(SHAPES, size5):
    print(f"  {nm:6s} 초과첨도 {g2:+5.1f}:  크기 {s_:.4f}", end="")
    for a_ in ALTS:
        print(f" · {a_} 배 검정력 {pow5[a_][g25.index(g2)]:.4f}", end="")
    print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.axvline(0, color="0.3", lw=2.0, ls="--")
ax1.axhline(0.05, color="0.3", lw=2.0, ls=":")
ax1.plot(g25, size5, "o-", color="tab:red", lw=3.0, ms=12)
ax1.set_xlabel("모집단의 초과첨도 (정규가 0)", fontsize=11.5)
ax1.set_ylabel("아무 차이가 없을 때 기각하는 비율", fontsize=11.5)
ax1.set_ylim(0, 0.21)
ax1.grid(alpha=0.25)
ax1.text(-1.15, 0.198, "꼬리가 얇은 쪽", ha="left", va="top", fontsize=12,
         color="0.3", fontweight="bold")
ax1.text(5.9, 0.198, "꼬리가 두꺼운 쪽", ha="right", va="top", fontsize=12,
         color="0.3", fontweight="bold")
ax1.annotate(f"{size5[0]:.4f} — 거의 기각하지 않는다",
             xy=(g25[0], size5[0]), xytext=(-0.9, 0.055), fontsize=11.3,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.annotate(f"{size5[-1]:.4f} — 약속의 세 배가 넘는다",
             xy=(g25[-1], size5[-1]), xytext=(4.9, 0.115), fontsize=11.3,
             color="tab:red", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.set_title("약속한 0.05 에서 양쪽으로 벌어진다", fontsize=12.5)

ax2.axvline(0, color="0.3", lw=2.0, ls="--")
ax2.plot(g25, pow5[1.3], "s-", color="tab:blue", lw=3.0, ms=11,
         label="분산이 1.3 배일 때")
ax2.plot(g25, pow5[2.0], "^--", color="tab:green", lw=2.6, ms=10,
         label="분산이 2.0 배일 때")
ax2.set_xlabel("모집단의 초과첨도 (정규가 0)", fontsize=11.5)
ax2.set_ylabel("참으로 다를 때 잡아내는 비율", fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11, loc="center right")
ax2.grid(alpha=0.25)
ax2.annotate(f"약한 차이에서는 꼬리가 얇은 쪽이\n"
             f"가장 못 잡는다 ({pow5[1.3][0]:.3f} 대 정규 "
             f"{pow5[1.3][2]:.3f})", xy=(g25[0], pow5[1.3][0]),
             xytext=(-0.9, 0.44), fontsize=11.3, color="tab:blue",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax2.annotate(f"센 차이에서는 순서가 뒤집힌다\n"
             f"({pow5[2.0][0]:.3f} 대 {pow5[2.0][2]:.3f})",
             xy=(g25[0], pow5[2.0][0]), xytext=(1.1, 0.985), fontsize=11.3,
             color="tab:green", fontweight="bold", ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.9))
ax2.set_title("크기가 다르니 검정력은 대립의 세기에 따라 갈린다",
              fontsize=12)

fig.suptitle("정규성을 가정하는 것이 아니라 첨도가 3 임을 가정한다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, CHI + "kurtosis_splits_the_direction.png")


# ════════════════════════════════════════════════════════════════════
# 6. ci_variance — 같은 정밀도에 두 배쯤 더 든다
#
# 쪽의 ci_width_by_n 이 「표본크기에 따른 폭과 비대칭」을 그려 들머리의
# 첫·둘째 문단을 담는다. 셋째 문단의 마지막 문장은 그리지 않는다 —
# 「평균을 재던 감각으로 분산을 재려 들면 표본이 한 자릿수 모자란다.」
#
# 재어 보니 **한 자릿수는 아니다.** 상한/하한을 같은 배수 안으로 줄이는
# 데 드는 n 이 평균 38 대 분산 66(2 배 목표), 99 대 189(1.5 배),
# 468 대 927(1.2 배) 로 어느 목표에서나 **두 배 남짓**이다. 그래서 제목을
# 재어 본 수에 맞추고, 들머리의 「한 자릿수」도 함께 고쳤다.
# 거북한 것은 배수가 아니라 작은 n 에서의 **폭**이다 — n = 5 에서
# 상한이 하한의 23 배이고, 그 수가 들머리의 것이다.
# ════════════════════════════════════════════════════════════════════
NS6 = np.arange(4, 2001)
ratio_var = (stats.chi2.ppf(0.975, NS6 - 1) /
             stats.chi2.ppf(0.025, NS6 - 1))
# 평균의 구간은 척도가 있어야 비가 정의되므로 변동계수 1 인 모집단으로 둔다
CV = 1.0
half = stats.t.ppf(0.975, NS6 - 1) * CV / np.sqrt(NS6)
ratio_mu = (1 + half) / (1 - half)
ratio_mu[half >= 0.999] = np.nan


def first_n(ratio, target):
    ok = np.where(ratio <= target)[0]
    return int(NS6[ok[0]]) if len(ok) else None


print("\n[ci_variance] 상한이 하한의 몇 배인가")
for n in (5, 25, 100, 500):
    j = int(np.where(NS6 == n)[0][0])
    print(f"  n = {n:4d}:  분산의 구간 {ratio_var[j]:6.2f} 배 · "
          f"평균의 구간 {ratio_mu[j]:5.2f} 배")
for tgt in (2.0, 1.5, 1.2):
    a, b = first_n(ratio_mu, tgt), first_n(ratio_var, tgt)
    print(f"  상한/하한을 {tgt} 배 안으로 줄이려면 — "
          f"평균은 n = {a}, 분산은 n = {b}  ({b / a:.1f} 배)")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.plot(NS6, ratio_mu, "-", color="tab:blue", lw=3.2,
        label="평균의 95% 구간 (변동계수 1 인 모집단)")
ax.plot(NS6, ratio_var, "-", color="tab:red", lw=3.2,
        label="분산의 95% 구간")
for tgt, col in [(2.0, "0.45"), (1.5, "0.3")]:
    a, b = first_n(ratio_mu, tgt), first_n(ratio_var, tgt)
    ax.axhline(tgt, color=col, lw=1.8, ls=":")
    ax.annotate("", xy=(b, tgt), xytext=(a, tgt),
                arrowprops=dict(arrowstyle="<->", color=col, lw=2.2))
    ax.text(np.sqrt(a * b), tgt * 1.04,
            f"{tgt} 배로 줄이는 데 {a} 대 {b} — {b / a:.0f} 배", ha="center",
            va="bottom", fontsize=11.3, color="0.2", fontweight="bold")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([4, 10, 25, 100, 500, 2000],
              ["4", "10", "25", "100", "500", "2,000"])
ax.set_yticks([1.1, 1.5, 2, 3, 5, 10, 25], ["1.1", "1.5", "2", "3", "5",
                                            "10", "25"])
ax.set_ylim(1.05, 30)
ax.set_xlim(4, 2000)
ax.set_xlabel("표본크기 n (로그 척도)", fontsize=12)
ax.set_ylabel("구간의 상한 ÷ 하한 (로그 척도)", fontsize=12)
ax.legend(fontsize=11.5, loc="upper right")
ax.grid(alpha=0.25, which="both")
ax.set_title("같은 목표에 드는 표본이 어느 자리에서나 두 배 남짓이다",
             fontsize=13)
fig.suptitle("점추정값이 구간 한가운데 있지 않다 — 그리고 폭이 거북하다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, CHI + "one_digit_short.png")


# ════════════════════════════════════════════════════════════════════
# 7. f_distribution_details — 1 보다 크다는 것만으로는 증거가 아니다
#
# 쪽의 f_order_matters 가 「자유도 순서에 따른 차이와 평균·최빈값의 위치」를
# 그려 들머리의 둘째 문단을 담는다. 남은 것은 첫째와 셋째다 — 「비가 45/28
# 인데 양측 p 가 0.3184 다」와 「역수 성질 덕분에 표의 한쪽 꼬리만으로
# 양측검정을 할 수 있었다」.
#
# 왼쪽이 **관측비의 자리**, 오른쪽이 **역수 성질**이다. 둘 다 이 쪽의
# 그림이 건드리지 않는다.
# ════════════════════════════════════════════════════════════════════
D1, D2 = 15, 20
FOBS = 45 / 28
p_two = 2 * min(stats.f.cdf(FOBS, D1, D2), stats.f.sf(FOBS, D1, D2))
hi = stats.f.ppf(0.975, D1, D2)
lo = stats.f.ppf(0.025, D1, D2)
print(f"\n[f_distribution_details] 자유도 ({D1}, {D2})")
print(f"  관측비 {FOBS:.4f} · 양측 p = {p_two:.4f}  (쪽은 0.3184)")
print(f"  양측 기각 임계값 {lo:.4f} 와 {hi:.4f}")
print(f"  평균 {D2 / (D2 - 2):.4f} · 최빈값 "
      f"{(D1 - 2) * D2 / (D1 * (D2 + 2)):.4f}")
rec = 1 / stats.f.ppf(0.975, D2, D1)
print(f"  역수 성질 확인: 1/F(0.975; {D2},{D1}) = {rec:.6f} · "
      f"F(0.025; {D1},{D2}) = {lo:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
g = np.linspace(0.001, 4.2, 800)
pdf = stats.f.pdf(g, D1, D2)
ax1.plot(g, pdf, color="0.25", lw=3.0)
ax1.fill_between(g[g >= hi], 0, pdf[g >= hi], color="tab:red", alpha=0.55)
ax1.fill_between(g[g <= lo], 0, pdf[g <= lo], color="tab:red", alpha=0.55)
ax1.axvline(1.0, color="0.45", lw=2.0, ls=":")
ax1.axvline(FOBS, color="tab:blue", lw=3.0)
ax1.text(1.0, 0.98, "1", ha="center", va="bottom", fontsize=12,
         color="0.35", fontweight="bold")
ax1.annotate(f"관측비 {FOBS:.3f}\n양측 p = {p_two:.4f}", xy=(FOBS, 0.42),
             xytext=(2.5, 0.78), fontsize=12, color="tab:blue",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax1.annotate(f"기각하려면 {hi:.2f} 를 넘어야 한다", xy=(hi, 0.06),
             xytext=(2.3, 0.30), fontsize=11.5, color="tab:red",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.set_xlim(0, 4.2)
ax1.set_ylim(0, 1.05)
ax1.set_xlabel(f"두 표본분산의 비 (자유도 {D1}, {D2})", fontsize=11.5)
ax1.set_ylabel("밀도", fontsize=11.5)
ax1.set_title("1 보다 크다는 것만으로는 아직 증거가 아니다", fontsize=12.5)
ax1.grid(alpha=0.25)

# 자유도 (15, 20) 과 (20, 15) 은 밀도가 거의 포개져 보여 「순서를 바꾸면
# 다른 분포」가 눈에 들어오지 않는다. 역수 성질 자체는 어느 자유도에서나
# 같으므로 모양이 확연히 갈리는 (5, 30) 과 (30, 5) 로 보인다.
E1, E2 = 5, 30
lo_e = stats.f.ppf(0.025, E1, E2)
hi_rev = stats.f.ppf(0.975, E2, E1)
g2 = np.linspace(0.001, 4.2, 800)
ax2.plot(g2, stats.f.pdf(g2, E1, E2), color="tab:blue", lw=3.0,
         label=f"자유도 ({E1}, {E2})")
ax2.plot(g2, stats.f.pdf(g2, E2, E1), color="tab:red", lw=3.0, ls="--",
         label=f"자유도 ({E2}, {E1}) — 순서를 바꾼 것")
ax2.axvline(lo_e, color="tab:blue", lw=2.4)
ax2.axvline(hi_rev, color="tab:red", lw=2.4, ls="--")
ax2.annotate(f"아래쪽 임계값 {lo_e:.4f}", xy=(lo_e, 0.75),
             xytext=(0.62, 1.03), fontsize=11.3, color="tab:blue",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax2.annotate(f"순서를 바꾼 쪽의 위 임계값 {hi_rev:.4f}", xy=(hi_rev, 0.12),
             xytext=(4.1, 0.52), fontsize=11.3, color="tab:red",
             fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax2.set_xlim(0, 4.2)
ax2.set_ylim(0, 1.15)
ax2.set_xlabel("F", fontsize=12)
ax2.set_ylabel("밀도", fontsize=11.5)
ax2.set_title("두 임계값이 서로의 역수다", fontsize=12.5)
ax2.legend(fontsize=10.8, loc="upper right")
ax2.grid(alpha=0.25)
ax2.text(0.5, 0.055, f"{lo_e:.4f} 와 {hi_rev:.4f} 를 곱하면 "
         f"{lo_e * hi_rev:.4f} 다", transform=ax2.transAxes, ha="center",
         va="bottom", fontsize=11.5, color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("평균이 1 이 아니다 — 그래서 비가 1 보다 커도 아직 아무 "
             "말이 아니다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, FT + "bigger_than_one_is_not_evidence.png")
