"""16장 들어가는 이야기 그림 — 둘째 그룹 일곱.

  8.  binomial_test       양측 p 값의 정의가 둘로 갈린다
  9.  runs_test           사람이 흉내 낸 무작위는 덩어리가 많다
  10. paired_sign         p 값이 버린 개수를 기억하지 못한다
  11. paired_wilcoxon     내려간 하나를 키워 가면 셋이 다르게 반응한다
  12. paired_permutation  세 검정이 같은 바닥을 쓴다
  13. rank_sum            계단 하나의 절반을 깎아 주면
  14. mann_whitney        마흔두 번은 독립으로 잰 마흔둘이 아니다

만드는 파일:
  docs/ch16/one_sample_nonparametric/img/two_definitions_of_two_sided.png
  docs/ch16/one_sample_nonparametric/img/people_avoid_streaks.png
  docs/ch16/paired_sample_nonparametric/img/p_forgets_what_was_dropped.png
  docs/ch16/paired_sample_nonparametric/img/ranks_soften_the_one_drop.png
  docs/ch16/paired_sample_nonparametric/img/three_tests_one_floor.png
  docs/ch16/two_sample_nonparametric/img/half_a_step_back.png
  docs/ch16/two_sample_nonparametric/img/forty_two_is_not_forty_two.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from itertools import combinations, product
from scipy import stats

ONE = "docs/ch16/one_sample_nonparametric/img/"
PAIR = "docs/ch16/paired_sample_nonparametric/img/"
TWO = "docs/ch16/two_sample_nonparametric/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 8. binomial_test — 양측 p 값의 정의가 둘로 갈린다
#
# 쪽의 exact_size_sawtooth 가 실제 크기의 톱니를 그려 들머리의 셋째
# 문단을 담는다. **둘째 문단**은 그려져 있지 않다 — 「가설값이 0.5 가
# 아니면 「작은 쪽 꼬리를 두 배 한다」와 「관측값만큼 가능성 낮은 결과를
# 다 더한다」가 같은 값이 아니다. 같은 자료에서 0.0959 와 0.0834 다.」
# 두 정의를 관측값 전체에 대해 그려 어디서 갈리는지 본다.
# ════════════════════════════════════════════════════════════════════
N8, P8 = 20, 0.3
ks = np.arange(0, N8 + 1)
pmf = stats.binom.pmf(ks, N8, P8)
p_double = np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(ks, N8, P8),
                                          stats.binom.sf(ks - 1, N8, P8)))
p_likely = np.array([pmf[pmf <= pmf[k] + 1e-12].sum() for k in ks])
print(f"[binomial_test] n = {N8}, 가설값 {P8}")
for k in range(2, 13):
    print(f"  성공 {k:2d}:  작은 꼬리의 두 배 {p_double[k]:.4f} · "
          f"가능성 낮은 쪽을 다 더함 {p_likely[k]:.4f}  "
          f"(비 {p_double[k] / p_likely[k]:.2f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.bar(ks, pmf, color="0.72", width=0.75)
ax1.axvline(N8 * P8, color="0.3", lw=2.0, ls=":")
ax1.text(N8 * P8 + 0.25, pmf.max() * 0.95, f"기댓값 {N8 * P8:.0f}",
         ha="left", va="top", fontsize=11.5, color="0.3",
         fontweight="bold")
K8 = 10
ax1.bar([K8], [pmf[K8]], color="tab:red", width=0.75)
ax1.annotate(f"관측값 {K8}", xy=(K8, pmf[K8]), xytext=(K8 + 2.0, 0.14),
             fontsize=12, color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
lowmass = pmf <= pmf[K8] + 1e-12
ax1.bar(ks[lowmass & (ks < N8 * P8)], pmf[lowmass & (ks < N8 * P8)],
        color="tab:blue", width=0.75)
ax1.set_xticks(np.arange(0, N8 + 1, 2))
ax1.set_xlabel("성공 횟수", fontsize=11.5)
ax1.set_ylabel("확률", fontsize=11.5)
ax1.set_title("가설값이 0.5 가 아니면 분포가 대칭이 아니다", fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)
ax1.text(0.97, 0.62, "파란 칸이 「관측값만큼\n가능성 낮은 결과」다",
         transform=ax1.transAxes, ha="right", va="top", fontsize=11,
         color="tab:blue", fontweight="bold")

ax2.plot(ks, p_double, "s-", color="tab:red", lw=2.8, ms=9,
         label="작은 쪽 꼬리를 두 배")
ax2.plot(ks, p_likely, "o-", color="tab:blue", lw=2.8, ms=9,
         label="가능성 낮은 결과를 다 더함")
ax2.axhline(0.05, color="0.2", lw=2.0, ls="--")
ax2.plot([K8], [p_double[K8]], "o", ms=18, mfc="none", mec="tab:red",
         mew=2.4)
ax2.plot([K8], [p_likely[K8]], "o", ms=18, mfc="none", mec="tab:blue",
         mew=2.4)
ax2.annotate(f"관측값 {K8} 에서\n{p_double[K8]:.4f} 대 {p_likely[K8]:.4f}",
             xy=(K8, (p_double[K8] + p_likely[K8]) / 2),
             xytext=(13.0, 0.52), fontsize=11.5, color="0.2",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.2", lw=1.9))
ax2.set_xticks(np.arange(0, N8 + 1, 2))
ax2.set_xlabel("성공 횟수", fontsize=11.5)
ax2.set_ylabel("양측 p 값", fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.25)
ax2.set_title("두 정의가 같은 값을 주지 않는다", fontsize=12.5)

fig.suptitle("「정확」이 약속하지 않는 것 — 정의가 하나가 아니다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, ONE + "two_definitions_of_two_sided.png")


# ════════════════════════════════════════════════════════════════════
# 9. runs_test — 사람이 흉내 낸 무작위는 덩어리가 많다
#
# 쪽의 runs_null_two_tails 가 두 꼬리를, runs_lag1_blindspot 이 눈먼
# 자리를 그린다. 들머리의 **둘째 문단**은 사람 이야기인데 그려져 있지
# 않다 — 「사람이 무작위를 흉내 낼 때 같은 값을 연달아 적기를 꺼려
# 생기는 자리다.」 그 꺼림을 되풀이 확률로 흉내 내어 진짜 무작위와
# 덩어리 개수 분포를 맞댄다.
# ════════════════════════════════════════════════════════════════════
N9, REPS9 = 25, 60_000
r9 = np.random.default_rng(16_109)


def nruns(s):
    return 1 + int((s[1:] != s[:-1]).sum())


real = np.array([nruns(r9.integers(0, 2, N9)) for _ in range(REPS9)])


def human(rng, n, q=0.35):
    """앞 값을 되풀이할 확률이 q 인 사람 흉내 (무작위라면 0.5).

    처음에는 「같은 값이 셋 이어지면 뒤집는다」로 흉내 냈는데 기각률이
    0.052 밖에 되지 않아 신호가 거의 없었다. 사람의 버릇은 「셋을 피한다」가
    아니라 **되풀이 자체를 덜 적는 것**이라 이 꼴이 실제에 가깝다.
    """
    s = np.empty(n, int)
    s[0] = rng.integers(0, 2)
    for i in range(1, n):
        s[i] = s[i - 1] if rng.random() < q else 1 - s[i - 1]
    return s


faked = np.array([nruns(human(r9, N9)) for _ in range(REPS9)])
mu9 = (N9 + 1) / 2          # +와 - 가 반반일 때의 대략적 기댓값
print(f"\n[runs_test] 길이 {N9} 수열의 덩어리 개수")
print(f"  진짜 무작위: 평균 {real.mean():.3f} · 가운데값 "
      f"{np.median(real):.1f}")
print(f"  사람 흉내  : 평균 {faked.mean():.3f} · 가운데값 "
      f"{np.median(faked):.1f}")

# 두 수열에 런 검정을 걸어 기각률을 센다 (큰 표본 근사, 양측 5%)
def runs_p(s):
    n1 = int(s.sum())
    n2 = len(s) - n1
    if n1 == 0 or n2 == 0:
        return 1.0
    R = nruns(s)
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)
           / ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    if var <= 0:
        return 1.0
    return 2 * stats.norm.sf(abs(R - mu) / np.sqrt(var))


rej_real = np.mean([runs_p(r9.integers(0, 2, N9)) < 0.05
                    for _ in range(20_000)])
rej_fake = np.mean([runs_p(human(r9, N9)) < 0.05 for _ in range(20_000)])
print(f"  기각률 — 진짜 {rej_real:.4f} · 사람 흉내 {rej_fake:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.2, 1]})
bins9 = np.arange(1.5, N9 + 1.5)
ax1.hist(real, bins=bins9, density=True, alpha=0.6, color="0.5",
         label="진짜 무작위")
ax1.hist(faked, bins=bins9, density=True, alpha=0.6, color="tab:red",
         label="사람 흉내 (되풀이를 덜 적는다)")
ax1.set_xlabel(f"같은 값이 이어지는 덩어리의 개수 (길이 {N9})",
               fontsize=11.5)
ax1.set_ylabel("비율", fontsize=11.5)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("흉내 낸 쪽은 덩어리가 많은 쪽으로 밀려 있다", fontsize=12.5)
ax1.annotate(f"평균 {real.mean():.1f} 대 {faked.mean():.1f}",
             xy=(faked.mean(), 0.02), xytext=(6.0, 0.22), fontsize=11.8,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

ax2.bar([0, 1], [rej_real, rej_fake], color=["0.5", "tab:red"], width=0.55)
for i, v in enumerate([rej_real, rej_fake]):
    ax2.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
             fontsize=14, fontweight="bold")
ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.set_xticks([0, 1], ["진짜 무작위", "사람 흉내"], fontsize=12.5)
ax2.set_ylabel("런 검정이 기각하는 비율", fontsize=11.5)
ax2.set_ylim(0, max(rej_fake, 0.1) * 1.3)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("「잘 섞인 듯」한 수열이 기각된다", fontsize=12.5)

fig.suptitle("잘 섞여 보이는 것과 무작위인 것", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, ONE + "people_avoid_streaks.png")


# ════════════════════════════════════════════════════════════════════
# 10. paired_sign — p 값이 버린 개수를 기억하지 못한다
#
# 쪽의 sign_zeros_hypothesis 가 「0 을 버리면 기각확률이 1 로 간다」를,
# sign_power_floor 가 귀무분포와 검정력을 그린다. 들머리의 **둘째 문단**은
# 그려져 있지 않다 — 「양이 일곱, 음이 둘이면 그대로인 쌍이 하나든
# 아흔하나든 p 값은 0.0898 로 똑같다.」
# ════════════════════════════════════════════════════════════════════
POS10, NEG10 = 7, 2
ZEROS = np.array([0, 1, 5, 11, 21, 51, 91])
# 쪽이 적는 0.0898 은 **단측**이다 — 본문의 코드가 alternative='greater' 다.
# 독자가 방금 읽은 수와 맞추려고 같은 꼴로 센다.
p10 = stats.binom.sf(POS10 - 1, POS10 + NEG10, 0.5)
totals = POS10 + NEG10 + ZEROS
frac = POS10 / totals
print(f"\n[paired_sign] 양 {POS10} · 음 {NEG10} 을 고정하고 0 만 늘린다")
for z, tot, fr in zip(ZEROS, totals, frac):
    print(f"  그대로인 쌍 {z:3d}:  전체 {tot:3d} 쌍 · "
          f"좋아진 비율 {fr:.3f} · p = {min(p10, 1.0):.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
xx = np.arange(len(ZEROS))
ax1.bar(xx, np.full(len(ZEROS), min(p10, 1.0)), color="tab:red",
        width=0.6)
for i in xx:
    ax1.text(i, min(p10, 1.0) + 0.004, f"{min(p10, 1.0):.4f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax1.set_xticks(xx, [f"{z}" for z in ZEROS], fontsize=12)
ax1.set_xlabel("그대로인 쌍의 개수 (양 7 · 음 2 는 고정)", fontsize=11.5)
ax1.set_ylabel("부호검정의 단측 p 값", fontsize=11.5)
ax1.set_ylim(0, 0.135)
ax1.grid(axis="y", alpha=0.25)
ax1.set_title("하나든 아흔하나든 한 자리도 움직이지 않는다", fontsize=12.5)

ax2.plot(ZEROS, frac, "o-", color="tab:blue", lw=3.0, ms=12)
ax2.set_xlabel("그대로인 쌍의 개수", fontsize=11.5)
ax2.set_ylabel("전체 쌍 가운데 좋아진 쪽의 비율", fontsize=11.5)
ax2.set_ylim(0, 0.85)
ax2.grid(alpha=0.25)
for z, fr, tot in zip(ZEROS, frac, totals):
    if z in (0, 11, 91):
        ax2.annotate(f"{tot} 쌍 가운데 {POS10} 명\n= {fr:.2f}",
                     xy=(z, fr), xytext=(0, 14),
                     textcoords="offset points", ha="center",
                     fontsize=10.8, color="tab:blue", fontweight="bold")
ax2.set_title("그런데 실제 이야기는 이만큼 달라진다", fontsize=12.5)
ax2.text(0.97, 0.93, "「백 명 가운데 일곱이 좋아졌다」와\n"
         "「열 명 가운데 일곱이 좋아졌다」를\n검정은 가르지 못한다",
         transform=ax2.transAxes, ha="right", va="top", fontsize=11.5,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))

fig.suptitle("그대로인 사람을 버리는 것은 옳다 — 버린 개수를 잊는 것이 "
             "문제다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, PAIR + "p_forgets_what_was_dropped.png")


# ════════════════════════════════════════════════════════════════════
# 11. paired_wilcoxon — 내려간 하나를 키워 가면
#
# 쪽의 midranks_and_ties 와 wplus_vs_min 은 중간순위와 두 표준화를
# 그린다. 들머리의 **첫 문단**이 던지는 물음 — 「내린 쪽 하나가 올라간
# 쪽들보다 클 때 둘을 어떻게 저울질할 것인가」 — 은 그려져 있지 않다.
# 그 하나의 크기를 손잡이로 돌리면 세 검정이 저마다 다르게 반응한다.
# ════════════════════════════════════════════════════════════════════
BASE11 = np.array([13., 9., 7., 6., 5., 4., 3., 2., 1., 1.5, -2.])
DROPS = np.array([2, 4, 8, 15, 30, 60], float)
p_sign, p_sr, p_t = [], [], []
for d in DROPS:
    x = np.r_[BASE11, -d]
    pos = int((x > 0).sum())
    n = len(x)
    p_sign.append(2 * min(stats.binom.cdf(pos, n, 0.5),
                          stats.binom.sf(pos - 1, n, 0.5)))
    p_sr.append(stats.wilcoxon(x).pvalue)
    p_t.append(stats.ttest_1samp(x, 0).pvalue)
print("\n[paired_wilcoxon] 내려간 한 사람의 크기를 키워 간다")
for d, a, b, c in zip(DROPS, p_sign, p_sr, p_t):
    print(f"  내려간 값 -{d:4.0f}:  부호 {min(a,1):.4f} · "
          f"부호순위 {b:.4f} · 대응 t {c:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1, 1.1]})
show = np.r_[BASE11, -30.0]
rk = stats.rankdata(np.abs(show))
ax1.barh(np.arange(len(show)), show,
         color=["tab:blue" if v > 0 else "tab:red" for v in show],
         height=0.65)
for i, v in enumerate(show):
    ax1.text(v + (1.2 if v > 0 else -1.2), i, f"순위 {rk[i]:.0f}",
             ha="left" if v > 0 else "right", va="center", fontsize=10.5,
             color="0.25", fontweight="bold")
ax1.axvline(0, color="0.2", lw=2.0)
ax1.set_yticks([])
ax1.set_xlim(-42, 26)
ax1.set_xlabel("처치 전후의 차이", fontsize=11.5)
ax1.set_title("크기는 -30 인데 순위로는 12 다", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)
ax1.text(0.03, 0.05, "가장 큰 차이가 받는 것도\n제 크기가 아니라 순위다",
         transform=ax1.transAxes, ha="left", va="bottom", fontsize=11,
         color="0.25", fontweight="bold")

ax2.plot(DROPS, np.minimum(p_sign, 1.0), "^-", color="0.5", lw=2.8, ms=11,
         label="부호만 세는 검정")
ax2.plot(DROPS, p_sr, "o-", color="tab:blue", lw=3.0, ms=12,
         label="크기의 순위를 쓰는 검정")
ax2.plot(DROPS, p_t, "s-", color="tab:red", lw=2.8, ms=11,
         label="원값을 다 쓰는 대응 t 검정")
ax2.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax2.set_xscale("log")
ax2.set_xticks(DROPS, [f"{d:.0f}" for d in DROPS])
ax2.set_xlabel("내려간 한 사람의 크기 (나머지 열하나는 그대로)",
               fontsize=11.5)
ax2.set_ylabel("p 값", fontsize=11.5)
ax2.set_ylim(0, max(p_t) * 1.2)
ax2.legend(fontsize=11, loc="upper left")
ax2.grid(alpha=0.25)
ax2.text(0.97, 0.06, "부호는 꿈쩍 않고, t 는 끌려가고,\n"
         "순위는 그 사이에서 완곡하게 받는다", transform=ax2.transAxes,
         ha="right", va="bottom", fontsize=11.3, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("내려간 한 사람이 올라간 셋보다 클 때", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, PAIR + "ranks_soften_the_one_drop.png")


# ════════════════════════════════════════════════════════════════════
# 12. paired_permutation — 세 검정이 같은 바닥을 쓴다
#
# 쪽의 permutation_floor 가 「32 가지 통계량과 표본크기별 최소 p 값」을
# 그린다. 들머리의 **셋째 문단 끝**이 남는다 — 「바닥의 높이는 부호검정과
# 부호순위검정에서도 똑같다. 세 검정이 같은 2 의 n 제곱 가지를 귀무분포로
# 쓰기 때문이다.」 세 검정의 최소 가능 p 를 n 마다 각각 세어 맞댄다.
# ════════════════════════════════════════════════════════════════════
NS12 = range(3, 13)
floor_perm, floor_sign, floor_sr = [], [], []
for n in NS12:
    signs = np.array(list(product([-1, 1], repeat=n)))
    # ① 부호 뒤집기 순열 — 차이의 합을 통계량으로
    d = np.arange(1, n + 1, dtype=float)      # 아무 자료나 (모두 다른 값)
    tot = signs @ d
    obs = d.sum()
    floor_perm.append((np.abs(tot) >= abs(obs) - 1e-9).mean())
    # ② 부호검정 — 양의 개수
    pos = (signs > 0).sum(1)
    k = n
    floor_sign.append(2 * min(stats.binom.cdf(k, n, 0.5),
                              stats.binom.sf(k - 1, n, 0.5)))
    # ③ 부호순위검정 — 양의 순위합
    rk = np.arange(1, n + 1, dtype=float)
    wplus = ((signs > 0) * rk).sum(1)
    wobs = rk.sum()
    floor_sr.append(min(1.0, 2 * (wplus >= wobs - 1e-9).mean()))
floor_sign = [min(v, 1.0) for v in floor_sign]
print("\n[paired_permutation] 세 검정이 낼 수 있는 가장 작은 양측 p")
for n, a, b, c in zip(NS12, floor_perm, floor_sign, floor_sr):
    print(f"  n = {n:2d}:  순열 {a:.5f} · 부호 {b:.5f} · "
          f"부호순위 {c:.5f}   (2/2^n = {2 / 2 ** n:.5f})")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ns = list(NS12)
ax.plot(ns, floor_perm, "o-", color="tab:blue", lw=3.4, ms=14,
        label="부호 뒤집기 순열검정")
ax.plot(ns, floor_sign, "s--", color="tab:red", lw=2.4, ms=10,
        label="부호검정")
ax.plot(ns, floor_sr, "^:", color="tab:green", lw=2.4, ms=10,
        label="부호순위검정")
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.axhline(0.01, color="0.5", lw=1.8, ls=":")
ax.set_yscale("log")
ax.set_yticks([0.001, 0.01, 0.05, 0.25, 1],
              ["0.001", "0.01", "0.05", "0.25", "1"])
ax.set_xticks(ns)
ax.set_xlabel("쌍의 개수 n", fontsize=12)
ax.set_ylabel("어떤 자료가 와도 그보다 작아질 수 없는 양측 p 값",
              fontsize=12)
ax.legend(fontsize=11.5, loc="upper right")
ax.grid(alpha=0.25, which="both")
ax.text(5.1, 0.056, "0.05", ha="left", va="bottom", fontsize=11,
        color="0.2", fontweight="bold")
ax.text(5.1, 0.0112, "0.01", ha="left", va="bottom", fontsize=11,
        color="0.5", fontweight="bold")
ax.annotate("세 선이 포개져 있다 —\n같은 2 의 n 제곱 가지를 쓰기 때문이다",
            xy=(8, floor_perm[5]), xytext=(8.6, 0.38), fontsize=12,
            color="0.2", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.2", lw=2.0))
ax.annotate("다섯 쌍으로는 어떤 자료가 와도\n0.05 에서 기각이 일어나지 않는다",
            xy=(5, floor_perm[2]), xytext=(5.4, 0.78), fontsize=12,
            color="tab:blue", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.0))
ax.set_title("바닥의 높이는 자료가 아니라 쌍의 개수가 정한다", fontsize=13)
fig.suptitle("서른두 가지뿐이면 세면 된다 — 그런데 서른둘로는 모자란다",
             fontsize=15, y=0.975)
plt.tight_layout()
save(fig, PAIR + "three_tests_one_floor.png")


# ════════════════════════════════════════════════════════════════════
# 13. rank_sum — 계단 하나의 절반을 깎아 주면
#
# 쪽의 ranksum_null_enumeration 이 924 가지 열거와 정확 귀무분포를
# 그린다. 들머리의 **셋째 문단**은 그려져 있지 않다 — 「양쪽이 다섯과
# 여섯인 자료에서 정확값이 0.536797 인데 근사는 0.465209 다. 계단 하나의
# 절반을 깎아 주면 0.522817 로 되돌아온다.」 작은 표본에서 정확값과
# 두 근사를 모두 그려 보정이 하는 일을 보인다.
# ════════════════════════════════════════════════════════════════════
N1, N2 = 5, 6
NTOT = N1 + N2
allr = np.arange(1, NTOT + 1)
sums = np.array([sum(c) for c in combinations(allr, N1)])
mu13 = N1 * (NTOT + 1) / 2
sd13 = np.sqrt(N1 * N2 * (NTOT + 1) / 12)
grid13 = np.arange(sums.min(), sums.max() + 1)
exact = np.array([2 * min((sums <= w).mean(), (sums >= w).mean())
                  for w in grid13])
exact = np.minimum(exact, 1.0)
naive = np.minimum(1.0, 2 * stats.norm.sf(np.abs(grid13 - mu13) / sd13))
corr = np.minimum(1.0, 2 * stats.norm.sf(
    np.maximum(np.abs(grid13 - mu13) - 0.5, 0) / sd13))
print(f"\n[rank_sum] n1 = {N1}, n2 = {N2} — 배정 {len(sums)} 가지")
for w in grid13:
    i = int(w - grid13[0])
    if abs(w - mu13) <= 3:
        print(f"  순위합 {w:2d}:  정확 {exact[i]:.6f} · 보정 없음 "
              f"{naive[i]:.6f} · 반 계단 보정 {corr[i]:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.step(grid13, exact, where="mid", color="0.2", lw=3.0, label="정확값")
ax1.plot(grid13, naive, "s-", color="tab:red", lw=2.2, ms=7,
         label="보정 없는 정규근사")
ax1.plot(grid13, corr, "o-", color="tab:blue", lw=2.2, ms=7,
         label="반 계단을 깎은 근사")
ax1.axhline(0.05, color="0.45", lw=1.8, ls=":")
ax1.set_xlabel(f"작은 집단의 순위합 (n1 = {N1}, n2 = {N2})", fontsize=11.5)
ax1.set_ylabel("양측 p 값", fontsize=11.5)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=11, loc="lower center")
ax1.grid(alpha=0.25)
ax1.set_title("근사가 계단을 가운데로 가로지른다", fontsize=12.5)

err_n = naive - exact
err_c = corr - exact
ax2.plot(grid13, err_n, "s-", color="tab:red", lw=2.6, ms=8,
         label="보정 없는 근사")
ax2.plot(grid13, err_c, "o-", color="tab:blue", lw=2.6, ms=8,
         label="반 계단을 깎은 근사")
ax2.axhline(0, color="0.2", lw=2.2, ls="--")
ax2.set_xlabel("작은 집단의 순위합", fontsize=11.5)
ax2.set_ylabel("근사 빼기 정확값", fontsize=11.5)
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(alpha=0.25)
ax2.set_title("보정 없는 쪽은 가운데에서 크게 아래로 틀어진다",
              fontsize=12.5)
j = int(np.argmin(err_n))
ax2.annotate(f"가장 크게 어긋난 자리 {err_n[j]:+.4f}\n"
             f"(보정하면 {err_c[j]:+.4f})", xy=(grid13[j], err_n[j]),
             xytext=(grid13[j] + 1.2, err_n[j] * 0.45), fontsize=11.3,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

fig.suptitle("계단꼴을 매끄러운 곡선으로 덮으면 가운데가 깎인다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TWO + "half_a_step_back.png")


# ════════════════════════════════════════════════════════════════════
# 14. mann_whitney — 마흔두 번은 독립으로 잰 마흔둘이 아니다
#
# 쪽의 pairwise_wins_grid 가 42 번 겨루기를, mw_tests_not_mean 이 셋째
# 문단을 그린다. 들머리의 **둘째 문단**은 그려져 있지 않다 — 「42 라는
# 수가 독립으로 재어 온 42 개가 아니라 관측값 열셋을 짝지어 만든
# 것이기 때문이다.」 겨룸이 독립이었다면 가졌을 흔들림과 실제 흔들림을
# 맞대면 76% 가 왜 유의하지 않은지가 보인다.
# ════════════════════════════════════════════════════════════════════
NA, NB = 7, 6
M14 = NA * NB
WINS = 32
mu14 = M14 / 2
sd_true = np.sqrt(NA * NB * (NA + NB + 1) / 12)
sd_naive = np.sqrt(M14 * 0.25)
print(f"\n[mann_whitney] {NA} 대 {NB} — 겨룸 {M14} 번")
print(f"  독립인 {M14} 번이었다면 표준편차 {sd_naive:.3f}")
print(f"  실제 표준편차 {sd_true:.3f}  ({sd_true / sd_naive:.2f} 배)")
print(f"  관측 {WINS} 승은 기댓값 {mu14:.0f} 에서 "
      f"{(WINS - mu14) / sd_true:.3f} 표준편차 "
      f"(독립이라면 {(WINS - mu14) / sd_naive:.3f})")

r14 = np.random.default_rng(16_114)
REPS14 = 200_000
a = r14.standard_normal((REPS14, NA))
b = r14.standard_normal((REPS14, NB))
wins = (a[:, :, None] > b[:, None, :]).sum((1, 2))
print(f"  되풀이해 재어 본 표준편차 {wins.std():.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
bb = np.arange(-0.5, M14 + 1.5)
ax1.hist(wins, bins=bb, density=True, color="0.72",
         label=f"겨룸 {M14} 번의 실제 귀무분포")
g = np.linspace(0, M14, 400)
ax1.plot(g, stats.norm.pdf(g, mu14, sd_naive), color="tab:red", lw=3.0,
         label=f"독립인 {M14} 번이었다면")
ax1.plot(g, stats.norm.pdf(g, mu14, sd_true), color="tab:blue", lw=3.0,
         ls="--", label="실제 폭")
ax1.axvline(WINS, color="0.15", lw=3.0)
ax1.annotate(f"관측 {WINS} 승", xy=(WINS, 0.045), xytext=(36, 0.082),
             fontsize=12, color="0.15", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.15", lw=1.9))
ax1.set_xlim(0, M14)
ax1.set_xlabel(f"A 가 이긴 횟수 ({M14} 번 가운데)", fontsize=11.5)
ax1.set_ylabel("비율", fontsize=11.5)
ax1.legend(fontsize=10.6, loc="upper left")
ax1.grid(alpha=0.25)
ax1.set_title("겨룸은 서로 독립이 아니다 — 폭이 두 배 넓다", fontsize=12.5)

ax2.bar([0, 1], [sd_naive, sd_true], color=["tab:red", "tab:blue"],
        width=0.5)
for i, v in enumerate([sd_naive, sd_true]):
    ax2.text(i, v + 0.12, f"{v:.3f}", ha="center", va="bottom",
             fontsize=16, fontweight="bold")
ax2.set_xticks([0, 1], [f"독립인 {M14} 번이었다면",
                        "관측값 열셋을 짝지어 만든 실제"], fontsize=11.5)
ax2.set_ylabel("이긴 횟수의 표준편차", fontsize=11.5)
ax2.set_ylim(0, sd_true * 1.35)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title(f"{sd_true / sd_naive:.2f} 배 — 그래서 {WINS} 승이 "
              f"{(WINS - mu14) / sd_true:.2f} 표준편차에 그친다",
              fontsize=12)
ax2.text(0.5, 0.06, f"76% 라는 큰 비에도 정확 양측 p 가 0.1375 인 까닭이\n"
         f"이 {sd_true / sd_naive:.2f} 배다", transform=ax2.transAxes,
         ha="center", va="bottom", fontsize=11.5, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))

fig.suptitle("하루씩 맞대어 마흔두 번 겨루면 — 그 마흔둘은 독립이 아니다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, TWO + "forty_two_is_not_forty_two.png")
