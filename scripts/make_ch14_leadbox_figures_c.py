"""14장 들어가는 이야기 그림 — 셋째 묶음 일곱 (정식 검정 넷과 한계 셋).

  1. shapiro_wilk      W 의 귀무 띠가 n 과 함께 1 로 붙는다
  2. ks_test           추정한 모수를 넣으면 백 배 보수적이 된다
  3. lilliefors_test   임계값을 잘못 고르면 「놓치는 쪽」으로 틀린다
  4. anderson_darling  표본이 커지는데 임계값도 커진다
  5. sample_size_power 눈으로도 수로도 흠이 없는데 p 가 10 의 -21 승
  6. limitations       같은 p 값이 허용하는 부적합이 n 과 함께 좁아진다
  7. choosing_test     균등분포에서 JB 가 왜 못 넘는가

만드는 파일:
  docs/ch14/formal_tests/img/w_band_closes_on_one.png
  docs/ch14/formal_tests/img/plugging_in_costs_everything.png
  docs/ch14/formal_tests/img/wrong_constant_misses_it.png
  docs/ch14/formal_tests/img/critical_value_rises_with_n.png
  docs/ch14/limitations/img/clean_sample_impossible_p.png
  docs/ch14/limitations/img/same_p_different_tolerance.png
  docs/ch14/limitations/img/why_jb_cannot_reach.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

FT = "docs/ch14/formal_tests/img/"
LIM = "docs/ch14/limitations/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 1. shapiro_wilk — W 의 귀무 띠가 n 과 함께 1 로 붙는다
#
# 쪽의 shapiro_null_scale 은 n = 30 과 300 **둘**의 귀무밀도를 견주고
# W = 0.975 의 자리를 찍는다. shapiro_w_as_qq_fit 은 모양과 W 를 잇는다.
# 들머리가 그 둘 너머로 말하는 것은 규칙이다 — 「W 를 적을 때 n 을 반드시
# 함께 적어야 하고, 다른 표본크기에서 얻은 W 끼리 크기를 비교해서는 안
# 된다.」 그러려면 두 점이 아니라 **n 전체에 걸친 띠**가 있어야 한다.
# W = 0.975 가 그 띠를 가로질러 평범에서 불가능으로 건너가는 것이 보인다.
# ════════════════════════════════════════════════════════════════════
NS1 = np.array([10, 15, 20, 30, 50, 100, 200, 300, 500, 1000])
REPS1 = 20_000
r1 = np.random.default_rng(14_201)
q05, q50, q95, crit = [], [], [], []
for n in NS1:
    w = np.array([stats.shapiro(r1.standard_normal(n)).statistic
                  for _ in range(min(REPS1, 4000 if n > 200 else REPS1))])
    q05.append(np.percentile(w, 5))
    q50.append(np.percentile(w, 50))
    q95.append(np.percentile(w, 95))
    crit.append(np.percentile(w, 5))
q05, q50, q95 = map(np.array, (q05, q50, q95))

WMARK = 0.975
print("[shapiro_wilk] 정규 자료에서 W 의 귀무 띠")
for n, a, b, c in zip(NS1, q05, q50, q95):
    side = "평범" if a <= WMARK <= c else ("불가능" if WMARK < a else "흔함")
    print(f"  n = {n:5d}:  5% {a:.4f} · 가운데 {b:.4f} · 95% {c:.4f}"
          f"   → W = {WMARK} 는 {side}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.2, 1]})
ax1.fill_between(NS1, q05, q95, color="tab:blue", alpha=0.22,
                 label="정규 자료에서 W 가 놓이는 폭 (가운데 90%)")
ax1.plot(NS1, q50, "-", color="tab:blue", lw=2.6, label="가운데값")
ax1.plot(NS1, q05, "--", color="tab:blue", lw=1.8)
ax1.axhline(WMARK, color="tab:red", lw=3.0)
ax1.text(1050, WMARK - 0.004, f"W = {WMARK}", ha="right", va="top",
         fontsize=13, color="tab:red", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xticks(NS1, [f"{n:,}" for n in NS1])
ax1.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax1.set_ylabel("W", fontsize=12)
ax1.set_ylim(0.80, 1.005)
ax1.set_title("띠가 n 과 함께 1 로 붙는다 — 같은 W 가 자리를 바꾼다",
              fontsize=12.5)
ax1.legend(fontsize=10.5, loc="lower right")
ax1.grid(alpha=0.25)
i30 = int(np.where(NS1 == 30)[0][0])
i300 = int(np.where(NS1 == 300)[0][0])
ax1.annotate("n = 30 에서는 띠 한가운데", xy=(30, WMARK),
             xytext=(10.5, 0.868), fontsize=11, color="0.25",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.25", lw=1.8))
ax1.annotate("n = 300 에서는 띠 아래로 한참", xy=(300, WMARK),
             xytext=(320, 0.895), fontsize=11, color="tab:red",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

# 같은 W 가 n 에 따라 받는 p 값
pw = []
for n in NS1:
    d = r1.standard_normal((4000, n))
    w = np.array([stats.shapiro(row).statistic for row in d])
    pw.append((w <= WMARK).mean())
pw = np.array(pw)
print(f"  같은 W = {WMARK} 가 받는 p 값")
for n, v in zip(NS1, pw):
    print(f"    n = {n:5d}:  {v:.5f}")
ax2.plot(NS1, np.maximum(pw, 1e-4), "o-", color="tab:red", lw=3.0, ms=11)
ax2.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax2.text(1050, 0.058, "0.05", ha="right", va="bottom", fontsize=11,
         color="0.3", fontweight="bold")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xticks(NS1, [f"{n:,}" for n in NS1])
ax2.set_yticks([1e-4, 1e-3, 0.01, 0.05, 0.2, 1.0],
               ["0.0001 아래", "0.001", "0.01", "0.05", "0.2", "1"])
ax2.set_ylim(7e-5, 1.4)
ax2.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax2.set_ylabel(f"W = {WMARK} 가 받는 p 값 (로그 눈금)", fontsize=11.5)
ax2.set_title("W 하나만 적으면 판정이 적히지 않는다", fontsize=12.5)
ax2.grid(alpha=0.25, which="both")
ax2.annotate(f"n = 30 이면 {pw[i30]:.2f} — 아주 평범하다",
             xy=(30, pw[i30]), xytext=(45, 0.55), fontsize=11,
             color="0.25", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.25", lw=1.8))
ax2.annotate("n = 300 이면 눈금 아래로 내려간다", xy=(300, max(pw[i300], 1e-4)),
             xytext=(240, 0.004), fontsize=11, color="tab:red",
             fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

fig.suptitle("같은 W 가 n 에 따라 평범해지고 불가능해진다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, FT + "w_band_closes_on_one.png")


# ════════════════════════════════════════════════════════════════════
# 2. ks_test — 추정한 모수를 넣으면 백 배 보수적이 된다
#
# 쪽의 ks_ecdf_sup 은 한 표본에서 가장 긴 세로 거리를 보이고,
# ks_estimated_params_shrink 는 못박은 분포함수와 적합한 분포함수를
# **한 표본에서** 견준다. 곧 통계량이 줄어드는 것은 그려져 있다.
# 들머리의 머리글은 그 다음이다 — 「명목 5% 로 쓴 검정의 실제 기각률이
# 0.00050, 백 배 보수적이다.」 기각률은 그려져 있지 않고, 더구나
# **n 을 늘려도 낫지 않는다**는 것이 요점이라 n 축이 있어야 한다.
# ════════════════════════════════════════════════════════════════════
NS2 = [20, 40, 80, 160, 320, 640]
REPS2 = 20_000
r2 = np.random.default_rng(14_202)
wrong, right, power_w, power_r = [], [], [], []
for n in NS2:
    d = r2.standard_normal((REPS2, n))
    ds = np.sort(d, axis=1)
    i = np.arange(1, n + 1)
    # ① 흔한 오용 — 자료의 평균·표준편차를 그대로 kstest 에 넣는다
    zz = (ds - d.mean(1, keepdims=True)) / d.std(1, ddof=1, keepdims=True)
    F = stats.norm.cdf(zz)
    Dw = np.maximum(i[None, :] / n - F, F - (i[None, :] - 1) / n).max(1)
    wrong.append((stats.kstwo.sf(Dw, n) < 0.05).mean())
    # ② 모수를 정말로 못박은 경우 (귀무가설이 참)
    F0 = stats.norm.cdf(ds)
    D0 = np.maximum(i[None, :] / n - F0, F0 - (i[None, :] - 1) / n).max(1)
    right.append((stats.kstwo.sf(D0, n) < 0.05).mean())
    # 검정력 — 같은 두 길을 지수분포 자료에 건다 (평균·분산을 맞췄다)
    e = r2.exponential(1.0, size=(REPS2, n))
    e = (e - 1.0)
    es = np.sort(e, axis=1)
    zz = (es - e.mean(1, keepdims=True)) / e.std(1, ddof=1, keepdims=True)
    Fe = stats.norm.cdf(zz)
    De = np.maximum(i[None, :] / n - Fe, Fe - (i[None, :] - 1) / n).max(1)
    power_w.append((stats.kstwo.sf(De, n) < 0.05).mean())
    Fe0 = stats.norm.cdf(es)
    De0 = np.maximum(i[None, :] / n - Fe0, Fe0 - (i[None, :] - 1) / n).max(1)
    power_r.append((stats.kstwo.sf(De0, n) < 0.05).mean())

print("\n[ks_test] 자료에서 추정한 모수를 그대로 넣으면")
for n, a, b, c, d_ in zip(NS2, wrong, right, power_w, power_r):
    print(f"  n = {n:4d}:  1 종 오류율 {a:.5f} (못박으면 {b:.4f}) · "
          f"지수 자료에서 기각률 {c:.4f} (못박으면 {d_:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS2, right, "o-", color="tab:blue", lw=2.8, ms=11,
         label="모수를 정말로 못박았을 때")
ax1.plot(NS2, wrong, "s-", color="tab:red", lw=2.8, ms=10,
         label="자료에서 추정해 넣었을 때 (흔한 오용)")
ax1.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax1.text(650, 0.054, "명목 0.05", ha="right", va="bottom", fontsize=11,
         color="0.3", fontweight="bold")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(NS2, [str(n) for n in NS2])
ax1.set_yticks([1e-4, 1e-3, 0.01, 0.05],
               ["0.0001", "0.001", "0.01", "0.05"])
ax1.set_ylim(5e-5, 0.12)
ax1.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax1.set_ylabel("실제 1 종 오류율 (로그 눈금)", fontsize=11.5)
ax1.set_title("백 배 보수적이고, n 을 늘려도 낫지 않는다", fontsize=12.5)
ax1.legend(fontsize=10.6, loc="lower left")
ax1.grid(alpha=0.25, which="both")

ax2.plot(NS2, power_r, "o-", color="tab:blue", lw=2.8, ms=11,
         label="모수를 정말로 못박았을 때")
ax2.plot(NS2, power_w, "s-", color="tab:red", lw=2.8, ms=10,
         label="자료에서 추정해 넣었을 때")
ax2.set_xscale("log")
ax2.set_xticks(NS2, [str(n) for n in NS2])
ax2.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax2.set_ylabel("지수분포 자료를 기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.set_title("틀리는 방향이 「놓치는 쪽」이다", fontsize=12.5)
ax2.legend(fontsize=10.6, loc="upper left")
ax2.grid(alpha=0.25)
ax2.annotate("비정규인 자료를 그대로 통과시키면서\n"
             "「검정을 했고 통과했다」는 말은 남는다",
             xy=(NS2[1], power_w[1] + 0.03), xytext=(640, 0.30),
             fontsize=11, color="tab:red", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

fig.suptitle("모수를 추정하면 백 배 보수적이 된다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, FT + "plugging_in_costs_everything.png")


# ════════════════════════════════════════════════════════════════════
# 3. lilliefors_test — 잘못 고른 상수가 「놓치는 쪽」으로 틀린다
#
# 쪽의 lilliefors_null_shift 는 두 **귀무분포**를 견주고,
# lilliefors_1000_runs 는 되풀이 횟수가 표의 정밀도를 어떻게 정하는지
# 보인다. 들머리에서 그 둘이 건드리지 않는 것은 값이다 — 「틀리는 방향이
# 「놓치는 쪽」이라는 점이 특히 고약하다. 아무 경고도 나오지 않고 p 값은
# 크게 찍히며, 보고서에는 「정규성 검정을 통과했다」가 남는다.」
# 그래서 **검정력**을 그린다. 두 상수를 같은 비정규 자료에 걸어 본다.
# ════════════════════════════════════════════════════════════════════
NS3 = [20, 30, 50, 80, 120, 200]
REPS3 = 20_000
r3 = np.random.default_rng(14_203)


def lillie_D(sample):
    n = sample.shape[1]
    s = np.sort(sample, axis=1)
    z = (s - sample.mean(1, keepdims=True)) / sample.std(1, ddof=1,
                                                         keepdims=True)
    F = stats.norm.cdf(z)
    i = np.arange(1, n + 1)
    return np.maximum(i[None, :] / n - F, F - (i[None, :] - 1) / n).max(1)


pow_wrong, pow_right, size_wrong, size_right = [], [], [], []
for n in NS3:
    d0 = r3.standard_normal((REPS3, n))
    D0 = lillie_D(d0)
    c_right = np.percentile(D0, 95)          # 올바른 임계값 (모의실험)
    c_wrong = 1.358 / np.sqrt(n)             # 못박은 쪽의 임계값
    size_right.append((D0 > c_right).mean())
    size_wrong.append((D0 > c_wrong).mean())
    alt = r3.exponential(1.0, size=(REPS3, n))
    Da = lillie_D(alt)
    pow_right.append((Da > c_right).mean())
    pow_wrong.append((Da > c_wrong).mean())

print("\n[lilliefors] 두 임계값을 같은 자료에 걸어 본다")
for n, a, b, c, d_ in zip(NS3, size_wrong, size_right, pow_wrong, pow_right):
    print(f"  n = {n:4d}:  1 종 오류율 {a:.4f} 대 {b:.4f} · "
          f"지수 자료 기각률 {c:.4f} 대 {d_:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS3, size_right, "o-", color="tab:blue", lw=2.8, ms=11,
         label="모수를 추정했을 때의 올바른 임계값")
ax1.plot(NS3, size_wrong, "s-", color="tab:red", lw=2.8, ms=10,
         label="못박은 쪽의 임계값을 그대로 썼을 때")
ax1.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax1.text(205, 0.0535, "명목 0.05", ha="right", va="bottom", fontsize=11,
         color="0.3", fontweight="bold")
ax1.set_xlabel("표본크기 n", fontsize=11.5)
ax1.set_ylabel("실제 1 종 오류율", fontsize=11.5)
ax1.set_ylim(0, 0.068)
ax1.set_title("잘못 고른 상수는 검정을 거의 꺼 버린다", fontsize=12.5)
ax1.legend(fontsize=10.4, loc="center right")
ax1.grid(alpha=0.25)

ax2.plot(NS3, pow_right, "o-", color="tab:blue", lw=2.8, ms=11,
         label="올바른 임계값")
ax2.plot(NS3, pow_wrong, "s-", color="tab:red", lw=2.8, ms=10,
         label="못박은 쪽의 임계값")
for i, n in enumerate(NS3):
    if n in (30, 80):
        ax2.annotate("", xy=(n, pow_wrong[i]), xytext=(n, pow_right[i]),
                     arrowprops=dict(arrowstyle="<->", color="0.3", lw=2.0))
        ax2.text(n + 3, (pow_right[i] + pow_wrong[i]) / 2,
                 f"{pow_right[i] - pow_wrong[i]:.2f} 만큼\n놓친다",
                 ha="left", va="center", fontsize=10.8, color="0.25",
                 fontweight="bold")
ax2.set_xlabel("표본크기 n", fontsize=11.5)
ax2.set_ylabel("지수분포 자료를 기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.set_title("틀리는 방향이 「놓치는 쪽」이다", fontsize=12.5)
ax2.legend(fontsize=10.6, loc="lower right")
ax2.grid(alpha=0.25)

fig.suptitle("같은 자료에 임계값이 둘 — 하나를 고르면 검정이 바뀐다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, FT + "wrong_constant_misses_it.png")


# ════════════════════════════════════════════════════════════════════
# 4. anderson_darling — 표본이 커지는데 임계값도 커진다
#
# 쪽의 ad_weight_function 은 가중함수를, ad_vs_ks_tail_power 는 꼬리
# 오염에서의 검정력을 그린다. 들머리의 머리글은 그 둘과 무관하다 —
# 「n = 20 의 5% 임계값이 0.692 로 n = 1000 의 0.784 보다 12% 낮다.
# 표본이 작은데 기준이 느슨해지는 셈이다.」
#
# 임계값은 내가 모의실험으로 다시 재지 않고 **꾸러미가 돌려주는 것**을
# 쓴다. 쪽이 적는 다섯 수가 그것이고(n = 1000 에서 0.574 · 0.653 ·
# 0.784 · 0.914 · 1.088), 독자가 자기 자리에서 보는 것도 그것이다.
# 오른쪽은 그 표를 쓰면 검정이 실제로 어떤 크기로 굴러가는지를 잰다.
# ════════════════════════════════════════════════════════════════════
NS4 = [10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000]
REPS4 = 20_000
r4 = np.random.default_rng(14_204)


def ad_stat(sample):
    n = sample.shape[1]
    s = np.sort(sample, axis=1)
    z = (s - sample.mean(1, keepdims=True)) / sample.std(1, ddof=1,
                                                         keepdims=True)
    F = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - (((2 * i - 1) * (np.log(F) + np.log(1 - F[:, ::-1]))).sum(1)
                 / n)


c5_pkg, size_pkg, size_asym = [], [], []
for n in NS4:
    cv = stats.anderson(r4.standard_normal(n), "norm").critical_values[2]
    c5_pkg.append(cv)
    a_ = ad_stat(r4.standard_normal((REPS4, n)))
    size_pkg.append((a_ > cv).mean())
    size_asym.append((a_ > 0.787).mean())
print("\n[anderson_darling] 꾸러미가 돌려주는 5% 임계값과 그 실제 크기")
for n, c, s_, sa in zip(NS4, c5_pkg, size_pkg, size_asym):
    print(f"  n = {n:5d}:  5% 임계값 {c:.4f} → 실제 크기 {s_:.4f}"
          f"   (점근값 0.787 을 쓰면 {sa:.4f})")
i20 = NS4.index(20)
i1000 = NS4.index(1000)
print(f"  n = 20 의 {c5_pkg[i20]:.3f} 이 n = 1000 의 {c5_pkg[i1000]:.3f} 보다 "
      f"{100 * (1 - c5_pkg[i20] / c5_pkg[i1000]):.0f}% 낮다")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS4, c5_pkg, "o-", color="tab:blue", lw=3.0, ms=11,
         label="꾸러미가 돌려주는 5% 임계값")
ax1.axhline(0.787, color="tab:red", lw=2.2, ls="--")
ax1.text(2050, 0.792, "점근값 0.787", ha="right", va="bottom",
         fontsize=11.5, color="tab:red", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xticks(NS4, [f"{n:,}" for n in NS4])
ax1.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax1.set_ylabel("5% 임계값", fontsize=11.5)
ax1.set_ylim(0.60, 0.83)
ax1.set_title("표본이 작을수록 기준이 **느슨하다**".replace("**", ""),
              fontsize=13)
ax1.legend(fontsize=11, loc="lower right")
ax1.grid(alpha=0.25)
ax1.annotate(f"n = 20 에서 {c5_pkg[i20]:.3f}", xy=(20, c5_pkg[i20]),
             xytext=(11, 0.755), fontsize=11.5, color="0.25",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.25", lw=1.8))
ax1.annotate(f"n = 1000 에서 {c5_pkg[i1000]:.3f} —\n"
             f"{100 * (1 - c5_pkg[i20] / c5_pkg[i1000]):.0f}% 더 높다",
             xy=(1000, c5_pkg[i1000]), xytext=(900, 0.655), fontsize=11.5,
             color="0.25", fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="0.25", lw=1.8))

ax2.plot(NS4, size_pkg, "o-", color="tab:blue", lw=3.0, ms=11,
         label="꾸러미의 표를 쓸 때")
ax2.plot(NS4, size_asym, "s--", color="tab:red", lw=2.6, ms=9,
         label="보정을 생략하고 점근값 0.787 을 쓸 때")
ax2.axhline(0.05, color="0.3", lw=2.0, ls="--")
ax2.text(2050, 0.0515, "명목 0.05", ha="right", va="bottom", fontsize=11,
         color="0.3", fontweight="bold")
ax2.set_xscale("log")
ax2.set_xticks(NS4, [f"{n:,}" for n in NS4])
ax2.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax2.set_ylabel("실제 1 종 오류율", fontsize=11.5)
ax2.set_ylim(0, 0.078)
ax2.set_title("기준이 n 과 함께 움직이는 까닭이 여기 있다", fontsize=12.5)
ax2.legend(fontsize=10.6, loc="lower right")
ax2.grid(alpha=0.25)
ax2.annotate(f"작은 표본에 점근값을 그대로 쓰면\n"
             f"n = 20 에서 {size_asym[i20]:.4f} — 명목보다 작다",
             xy=(20, size_asym[i20]), xytext=(33, 0.0115), fontsize=11,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

fig.suptitle("표본이 커지는데 임계값도 커진다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, FT + "critical_value_rises_with_n.png")


# ════════════════════════════════════════════════════════════════════
# 5. sample_size_power — 기각을 만든 것이 누구인가
#
# 처음에는 「흠이 없는 그 표본」을 그리려 했는데, 쪽의 berkson_200000 의
# **오른쪽 칸이 바로 그것**이다(Q-Q 는 직선인데 p 가 10 의 -21 승).
# power_paradox 는 검정력 곡선을 그린다. 그래서 들머리가 세우는 것 가운데
# 어느 그림도 건드리지 않는 자리로 간다 — 「기각은 전적으로 그 188 개가
# 만든 것」이라는 본문 §4 의 문장이다. 같은 자료에서 그 188 개를 빼면
# JB 가 92.5 에서 0.48 로 내려앉는다.
#
# 곧 「이탈이 실무적으로 큰가」와 「n 이 그 이탈을 볼 만큼 큰가」가 어떻게
# 갈리는지를 **관측값 쪽에서** 보인다. 20 만 개 가운데 몇 백 개가 통계량을
# 거의 혼자 만든다.
# ════════════════════════════════════════════════════════════════════
N5, K5 = 200_000, 188
r5 = np.random.default_rng(14_205)
x5 = r5.standard_normal(N5)
idx = r5.choice(N5, K5, replace=False)
x5[idx] *= 3.0


def jb(v):
    n = len(v)
    s, k = stats.skew(v), stats.kurtosis(v)
    return n / 6 * (s ** 2 + k ** 2 / 4)


jb_full = jb(x5)
keep = np.ones(N5, bool)
keep[idx] = False
jb_clean = jb(x5[keep])
print("\n[sample_size_power] 기각을 만든 것이 누구인가")
print(f"  왜도 {stats.skew(x5):+.4f} · 초과첨도 {stats.kurtosis(x5):+.4f}")
print(f"  JB = {jb_full:.1f} (p = {stats.chi2.sf(jb_full, 2):.1e})")
print(f"  오염된 {K5} 개를 빼면 JB = {jb_clean:.2f} "
      f"(p = {stats.chi2.sf(jb_clean, 2):.3f})")

# 처음에는 「가장 바깥 m 개를 빼 가며」를 그리려 했는데 재어 보니
# 곡선이 단조가 아니다 — 많이 깎으면 꼬리가 **너무 얇아져** JB 가 도로
# 올라간다(188 개를 깎으면 44, 1000 개를 깎으면 428). 오염된 188 개와
# 「가장 바깥 188 개」가 같은 집합이 아니기 때문이기도 하다.
# 그래서 깎는 대신 **누가 네제곱 합을 내는지**를 쌓아 보인다. 이쪽은
# 정의상 단조이고, 「기각을 만든 것이 누구인가」에 바로 답한다.
z5 = (x5 - x5.mean()) / x5.std()
contrib = np.sort(z5 ** 4)[::-1]
cum = np.cumsum(contrib) / contrib.sum()
# 흠 없는 정규 표본도 같은 꼴로 쌓아 두 곡선을 맞댄다 — 갈라지는 자리가 답이다
zc = r5.standard_normal(N5)
zc = (zc - zc.mean()) / zc.std()
cum_clean = np.cumsum(np.sort(zc ** 4)[::-1]) / (zc ** 4).sum()
MARKS = [10, 50, 188, 500, 2000, 10_000]
print("  네제곱 합에서 가장 바깥 몇 개가 내는 몫")
for m in MARKS:
    print(f"    가장 바깥 {m:6d} 개 ({100 * m / N5:6.3f}%) 가 "
          f"{100 * cum[m - 1]:5.1f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.15]})
bars = [jb_full, jb_clean]
ax1.bar([0, 1], bars, color=["tab:red", "0.55"], width=0.5)
for i, v in enumerate(bars):
    ax1.text(i, v + 8, f"{v:.1f}", ha="center", va="bottom", fontsize=15,
             fontweight="bold")
ax1.axhline(stats.chi2.ppf(0.95, 2), color="0.3", lw=2.2, ls="--")
ax1.text(-0.42, stats.chi2.ppf(0.95, 2) + 8, "가로 점선이 5% 문턱 5.99",
         ha="left", va="bottom", fontsize=11.5, color="0.3",
         fontweight="bold")
ax1.set_xticks([0, 1], ["20 만 개 그대로", f"오염된 {K5} 개만 빼고"],
               fontsize=12)
ax1.set_ylabel("자르크–베라 통계량", fontsize=11.5)
ax1.set_ylim(0, max(bars) * 1.25)
ax1.set_title(f"{K5} 개가 통계량을 거의 혼자 만든다", fontsize=13)
ax1.grid(axis="y", alpha=0.25)
ax1.text(0.5, max(bars) * 0.58, f"{100 * K5 / N5:.3f}% 를 빼면\n"
         f"{jb_full:.0f} 이 {jb_clean:.2f} 로", ha="center", va="center",
         fontsize=12.5, color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.65"))

xs = np.arange(1, N5 + 1)
ax2.plot(xs, 100 * cum_clean, "-", color="0.6", lw=2.8,
         label="흠 없는 정규 20 만 개")
ax2.plot(xs, 100 * cum, "-", color="tab:red", lw=3.0,
         label=f"오염된 {K5} 개가 섞인 20 만 개")
ax2.legend(fontsize=10.8, loc="lower right")
ax2.set_xscale("log")
ax2.set_xticks([1, 10, 100, 1000, 10_000, 100_000],
               ["1", "10", "100", "1,000", "1만", "10만"])
ax2.set_xlim(1, N5)
ax2.set_ylim(0, 104)
ax2.set_xlabel("가장 바깥에서 센 관측값의 개수 (로그 눈금)", fontsize=11.5)
ax2.set_ylabel("네제곱 합에서 그들이 내는 몫 (퍼센트)", fontsize=11.5)
ax2.set_title("두 곡선이 갈라지는 자리가 범인이다", fontsize=12.5)
ax2.grid(alpha=0.25, which="both")
ax2.plot([K5], [100 * cum[K5 - 1]], "o", ms=13, color="tab:red",
         mec="white", mew=2)
ax2.plot([K5], [100 * cum_clean[K5 - 1]], "o", ms=13, color="0.5",
         mec="white", mew=2)
gap = 100 * (cum[K5 - 1] - cum_clean[K5 - 1])
ax2.annotate(f"가장 바깥 {K5} 개 ({100 * K5 / N5:.3f}%) 에서\n"
             f"두 곡선이 {gap:.1f} 퍼센트 점 벌어진다 —\n"
             f"그 틈이 곧 기각이다",
             xy=(K5, 100 * (cum[K5 - 1] + cum_clean[K5 - 1]) / 2),
             xytext=(1.6, 84), fontsize=11.5, color="tab:red",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax2.annotate("1만 개를 넘어서면 두 곡선이 다시 붙는다 —\n"
             "가운데는 멀쩡하다는 뜻이다", xy=(30_000, 100 * cum[29_999]),
             xytext=(1100, 52), fontsize=11, color="0.3",
             fontweight="bold", ha="right",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.8))

fig.suptitle("왜도 0.002 에 p 가 10 의 마이너스 스물몇 승 — 누가 만든 "
             "수인가", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, LIM + "clean_sample_impossible_p.png")


# ════════════════════════════════════════════════════════════════════
# 6. limitations — 같은 p 가 허용하는 부적합이 n 과 함께 좁아진다
#
# 쪽의 pvalue_vs_effectsize 는 「n 이 커질 때 p 값과 표본 초과첨도가
# 각각 어떻게 달라지는지」를 그린다. 들머리의 머리글은 그것이 아니라
# **허용 폭**이다 — 「5% 임계값으로 재면 허용되는 부적합 1 - W 가
# 0.0965 에서 0.000344 로 280 배 좁아진다. p 값은 적합도의 척도가 아니다.」
#
# 14.4 의 shapiro_wilk 들머리 그림이 **W 한 점이 n 에 따라 받는 p** 를
# 그리는 것과 짝이 되도록, 여기서는 **p 한 점이 n 마다 허용하는 부적합**을
# 그린다. 같은 곡선의 반대 읽기이고 두 쪽이 서로를 가리킨다.
# ════════════════════════════════════════════════════════════════════
NS6 = np.array([10, 20, 50, 100, 500, 1000, 5000, 10_000])
REPS6 = 4000
r6 = np.random.default_rng(14_206)
tol5, tol_typ = [], []
for n in NS6:
    w = np.array([stats.shapiro(r6.standard_normal(n)).statistic
                  for _ in range(REPS6 if n <= 1000 else 800)])
    tol5.append(1 - np.percentile(w, 5))        # 5% 에서 허용되는 1-W
    tol_typ.append(1 - np.percentile(w, 70))    # p = 0.7 쯤의 자리
print("\n[limitations] 5% 임계값에서 허용되는 부적합 1 빼기 W")
for n, a, b in zip(NS6, tol5, tol_typ):
    print(f"  n = {n:6d}:  허용 {a:.6f} · p = 0.7 쯤의 자리 {b:.6f}")
print(f"  n = 20 과 10,000 의 비 {tol5[1] / tol5[-1]:.0f} 배")

fig, ax = plt.subplots(figsize=(10.8, 5.8))
ax.plot(NS6, tol5, "o-", color="tab:red", lw=3.0, ms=12,
        label="5% 임계값이 허용하는 부적합")
ax.plot(NS6, tol_typ, "s--", color="0.5", lw=2.4, ms=9,
        label="p 가 0.7 쯤 되는 자리의 부적합")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(NS6, [f"{n:,}" for n in NS6])
ax.set_yticks([1e-4, 1e-3, 0.01, 0.1],
              ["0.0001", "0.001", "0.01", "0.1"])
ax.set_xlabel("표본크기 n (로그 눈금)", fontsize=12)
ax.set_ylabel("허용되는 부적합 1 빼기 W (로그 눈금)", fontsize=12)
ax.set_title("같은 p 값이 n 마다 전혀 다른 적합도를 뜻한다", fontsize=13)
ax.legend(fontsize=11.5, loc="lower left")
ax.grid(alpha=0.25, which="both")
j20 = int(np.where(NS6 == 20)[0][0])
j1e4 = int(np.where(NS6 == 10_000)[0][0])
ax.annotate(f"n = 20 이면 {tol5[j20]:.4f} 까지 어긋나도 통과한다",
            xy=(22, tol5[j20]), xytext=(38, 0.082), fontsize=11.5,
            color="tab:red", fontweight="bold", ha="left", va="top",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.annotate(f"n = 10,000 이면 {tol5[j1e4]:.6f} 가 한계다 —\n"
            f"{tol5[j20] / tol5[j1e4]:.0f} 배 좁아졌다",
            xy=(10_000, tol5[j1e4] * 1.3), xytext=(9500, 0.022),
            fontsize=11.5, color="tab:red", fontweight="bold", ha="right",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

fig.suptitle("같은 p 값인데 허용 폭이 몇 백 배 다르다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, LIM + "same_p_different_tolerance.png")


# ════════════════════════════════════════════════════════════════════
# 7. choosing_test — 균등분포에서 JB 가 왜 못 넘는가
#
# 쪽의 test_power_by_alternative 가 「다섯 검정 × 네 대립분포」의 기각률을
# 표로 그린다. 곧 들머리의 수들(0.000 과 0.802 등)은 그 표의 것이다.
# 남은 것은 **까닭**이다 — 「n = 50 에서 균등분포의 표본 초과첨도는
# -1.2 근처에 좁게 몰리는데, 이것을 JB 공식에 넣으면 3.0 언저리가 되어
# 카이제곱의 95% 값 5.99 에 닿지 못한다.」
# 그래서 JB 통계량의 **분포 자체**를 세 자료에 대해 그리고 5.99 선을 긋는다.
# 기각률 표가 말하지 않는 「왜 0.000 인가」가 거기서 보인다.
# ════════════════════════════════════════════════════════════════════
N7, REPS7 = 50, 60_000
r7 = np.random.default_rng(14_207)
CASES = [("정규", lambda m, n: r7.standard_normal((m, n)), "0.5"),
         ("균등 (꼬리가 얇다)", lambda m, n: r7.uniform(-1, 1, (m, n)),
          "tab:purple"),
         ("t(3) (꼬리가 두껍다)", lambda m, n: r7.standard_t(3, (m, n)),
          "tab:red")]
jb_all, k2_all, kur_all = {}, {}, {}
for nm, f, _ in CASES:
    d = f(REPS7, N7)
    s, kk = stats.skew(d, axis=1), stats.kurtosis(d, axis=1)
    jb_all[nm] = N7 / 6 * (s ** 2 + kk ** 2 / 4)
    k2_all[nm] = stats.normaltest(d, axis=1).statistic
    kur_all[nm] = kk
CRIT = stats.chi2.ppf(0.95, 2)
print(f"\n[choosing_test] n = {N7} 에서 두 통계량의 분포 (문턱 {CRIT:.2f})")
for nm, _, _ in CASES:
    print(f"  {nm:18s} 초과첨도 가운데값 {np.median(kur_all[nm]):+.3f} · "
          f"JB 기각률 {(jb_all[nm] > CRIT).mean():.4f} · "
          f"K 제곱 기각률 "
          f"{(k2_all[nm] > CRIT).mean():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
bins = np.linspace(0, 20, 150)
for nm, _, col in CASES:
    ax1.hist(np.clip(jb_all[nm], 0, 19.9), bins=bins, density=True,
             histtype="step", lw=2.8, color=col, label=nm)
ax1.axvline(CRIT, color="0.2", lw=2.6, ls="--")
ax1.text(CRIT + 0.4, 0.60, f"카이제곱의 95% 값 {CRIT:.2f}", ha="left",
         va="top", fontsize=11.5, color="0.2", fontweight="bold")
ax1.set_xlim(0, 19.4)
ax1.set_ylim(0, 0.68)
ax1.set_xlabel(f"자르크–베라 통계량 (n = {N7})", fontsize=11.5)
ax1.set_ylabel("밀도", fontsize=11.5)
ax1.set_title("균등분포의 통계량이 문턱에 닿지 못한다", fontsize=12.5)
ax1.legend(fontsize=10.8, loc="upper right")
ax1.grid(alpha=0.25)
uni = "균등 (꼬리가 얇다)"
ax1.annotate(f"가운데값 {np.median(jb_all[uni]):.2f} —\n"
             f"기각률이 {(jb_all[uni] > CRIT).mean():.3f} 인 까닭",
             xy=(np.median(jb_all[uni]), 0.14), xytext=(6.0, 0.46),
             fontsize=11.3, color="tab:purple", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.9))

bins2 = np.linspace(0, 40, 150)
for nm, _, col in CASES:
    ax2.hist(np.clip(k2_all[nm], 0, 39.9), bins=bins2, density=True,
             histtype="step", lw=2.8, color=col, label=nm)
ax2.axvline(CRIT, color="0.2", lw=2.6, ls="--")
ax2.text(CRIT + 0.7, 0.30, f"같은 문턱 {CRIT:.2f}", ha="left", va="top",
         fontsize=11.5, color="0.2", fontweight="bold")
ax2.set_xlim(0, 38.8)
ax2.set_ylim(0, 0.34)
ax2.set_xlabel(f"다고스티노 K 제곱 (n = {N7})", fontsize=11.5)
ax2.set_ylabel("밀도", fontsize=11.5)
ax2.set_title("같은 자료가 이쪽에서는 문턱을 넘어간다", fontsize=12.5)
ax2.legend(fontsize=10.8, loc="upper right")
ax2.grid(alpha=0.25)
ax2.annotate(f"기각률 {(k2_all[uni] > CRIT).mean():.3f}",
             xy=(np.median(k2_all[uni]), 0.045), xytext=(20.0, 0.17),
             fontsize=11.3, color="tab:purple", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.9))

fig.suptitle("분포를 바꾸면 검정의 순서가 뒤집힌다 — 그 까닭", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, LIM + "why_jb_cannot_reach.png")
