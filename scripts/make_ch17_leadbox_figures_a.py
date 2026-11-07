"""17장 들어가는 이야기 그림 — 첫 묶음 일곱.

  1. index              평균에는 공식이 있고 가운데 값에는 없다
  2. principle          횟수는 촘촘함을 정하고 닮음은 표본크기가 정한다
  3. resampling_method  표본을 바꿀 때와 횟수를 바꿀 때
  4. nonparametric      빠지는 36.8% 는 손실이 아니다
  5. parametric         틀린 모형이 가장 좁은 구간을 낸다
  6. percentile         포함확률 오차가 제곱근으로만 줄어든다
  7. basic              반사한 끝점은 있을 수 없는 값이 될 수 있다

만드는 파일:
  docs/ch17/img/no_formula_for_the_middle.png
  docs/ch17/bootstrap/img/b_versus_n.png
  docs/ch17/bootstrap/img/two_kinds_of_wobble.png
  docs/ch17/bootstrap/img/the_left_out_third.png
  docs/ch17/bootstrap/img/wrong_model_narrowest.png
  docs/ch17/bootstrap_ci/img/error_shrinks_only_by_root_n.png
  docs/ch17/bootstrap_ci/img/mirror_can_leave_the_range.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

IDX = "docs/ch17/img/"
BOOT = "docs/ch17/bootstrap/img/"
CI = "docs/ch17/bootstrap_ci/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def boot_se(x, stat, B, rng):
    idx = rng.integers(0, len(x), (B, len(x)))
    return stat(x[idx], axis=1).std(ddof=1)


# ════════════════════════════════════════════════════════════════════
# 1. 장 개요 — 평균에는 공식이 있고 가운데 값에는 없다
#
# 쪽에 그림이 하나도 없다. 들머리의 알맹이는 한 문단이다 — 「평균이라면
# 답이 쉽다. 「이만큼 흔들린다」를 적어 주는 공식이 있다. 그런데 한가운데
# 값에는 그런 공식이 없다. 구하기는 평균보다 쉬운데 얼마나 흔들리는지를
# 재는 일은 훨씬 어렵다.」
#
# 장 개요는 독자가 붓스트랩도 표준오차도 아직 모르는 자리다. 그러니
# 수식 없이 **세 칸을 맞대어** 보인다 — 실제로 되풀이 조사해 본 흔들림,
# 교과서 공식이 적어 주는 흔들림, 그리고 표본 하나에서 다시 뽑아 낸
# 흔들림. 평균에서는 셋이 맞고, 가운데 값에서는 가운데 칸이 비어 있다.
# ════════════════════════════════════════════════════════════════════
r1 = np.random.default_rng(17_001)
N1, REPS1, B1 = 2000, 4000, 2000
SIG1 = 0.9                       # 소득처럼 오른쪽으로 길게 늘어진 모집단


def draw_income(m, n):
    return np.exp(r1.normal(np.log(649), SIG1, (m, n)))


pop = draw_income(1, 4_000_000)[0]
true_mean, true_med = pop.mean(), np.median(pop)
samples = draw_income(REPS1, N1)
real_mean_sd = samples.mean(1).std(ddof=1)
real_med_sd = np.median(samples, axis=1).std(ddof=1)
one = samples[0]
formula_mean = one.std(ddof=1) / np.sqrt(N1)
boot_mean = boot_se(one, np.mean, B1, r1)
boot_med = boot_se(one, np.median, B1, r1)
print("[장 개요] 소득꼴 모집단 — 흔들림을 세 가지로 재어 본다")
print(f"  평균:    되풀이 조사 {real_mean_sd:7.2f} · 공식 "
      f"{formula_mean:7.2f} · 다시 뽑기 {boot_mean:7.2f}")
print(f"  가운데:  되풀이 조사 {real_med_sd:7.2f} · 공식 (없다) · "
      f"다시 뽑기 {boot_med:7.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                               gridspec_kw={"width_ratios": [1, 1.15]})
ax1.hist(pop[pop < 3000], bins=120, density=True, color="0.78")
ax1.axvline(true_med, color="tab:blue", lw=3.0)
ax1.axvline(true_mean, color="tab:red", lw=3.0, ls="--")
ax1.text(true_med - 25, ax1.get_ylim()[1] * 0.92, "한가운데 집", ha="right",
         va="top", fontsize=12.5, color="tab:blue", fontweight="bold",
         rotation=90)
ax1.text(true_mean + 35, ax1.get_ylim()[1] * 0.92, "평균", ha="left",
         va="top", fontsize=12.5, color="tab:red", fontweight="bold",
         rotation=90)
ax1.set_xlim(0, 3000)
ax1.set_yticks([])
ax1.set_xlabel("월 소득 (만 원)", fontsize=11.5)
ax1.set_title("소득은 오른쪽으로 길게 늘어진다", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)

xx = np.arange(2)
w = 0.26
vals_real = [real_mean_sd, real_med_sd]
vals_form = [formula_mean, 0.0]
vals_boot = [boot_mean, boot_med]
ax2.bar(xx - w, vals_real, width=w, color="0.35",
        label="2 만 집을 되풀이 조사해 본 실제 흔들림")
ax2.bar(xx, vals_form, width=w, color="tab:red",
        label="교과서 공식이 적어 주는 값")
ax2.bar(xx + w, vals_boot, width=w, color="tab:blue",
        label="표본 하나에서 다시 뽑아 낸 값")
for i in xx:
    ax2.text(i - w, vals_real[i] + 0.25, f"{vals_real[i]:.1f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.text(i + w, vals_boot[i] + 0.25, f"{vals_boot[i]:.1f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             color="tab:blue")
ax2.text(1, 0.6, "공식이 없다", ha="center", va="bottom", fontsize=13,
         color="tab:red", fontweight="bold")
ax2.text(0, vals_form[0] + 0.25, f"{vals_form[0]:.1f}", ha="center",
         va="bottom", fontsize=11, fontweight="bold", color="tab:red")
ax2.set_xticks(xx, ["평균", "한가운데 값"], fontsize=13)
ax2.set_ylabel("그 수가 조사마다 흔들리는 폭 (만 원)", fontsize=11.5)
ax2.set_ylim(0, max(vals_real) * 1.45)
ax2.legend(fontsize=10.6, loc="upper left")
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("구하기는 더 쉬운데 흔들림을 재는 일은 훨씬 어렵다",
              fontsize=12.5)

fig.suptitle("한가운데 값에는 「이만큼 흔들린다」를 적어 주는 공식이 없다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, IDX + "no_formula_for_the_middle.png")


# ════════════════════════════════════════════════════════════════════
# 2. principle — 횟수는 촘촘함을, 닮음은 표본크기가 정한다
#
# 쪽의 bootstrap_world_analogy 가 두 세계를, recursive_resampling 이
# 되풀이 재표집을, jackknife_median 이 잭나이프의 실패를 그린다.
# 들머리의 **넷째 문단**은 그려져 있지 않다 — 「두 분포가 가장 벌어진
# 자리의 간격은 다시 뽑는 횟수를 아무리 키워도 줄지 않는다. 그 횟수는
# 흉내 낸 세계를 얼마나 촘촘히 훑었는지만 정하고, 그 세계가 현실과
# 얼마나 닮았는지는 **표본 크기**가 정하기 때문이다.」
#
# 그 말은 두 손잡이를 따로 돌려야 보인다 — 횟수 B 와 표본크기 n.
# ════════════════════════════════════════════════════════════════════
r2 = np.random.default_rng(17_002)
BS = [50, 200, 1000, 5000, 20000]
NS = [10, 20, 40, 80, 160]
TRUE_REPS = 60_000


def true_dist(n, m):
    x = r2.standard_normal((m, n))
    return np.median(x, axis=1)


def boot_dist(x, B):
    idx = r2.integers(0, len(x), (B, len(x)))
    return np.median(x[idx], axis=1)


# 표본 하나로 재면 그 표본이 어떤 것이냐에 따라 간격이 크게 흔들려
# n 의 효과가 묻힌다. 표본 여러 벌에 걸쳐 평균을 낸다.
SAMPLES = 25
gap_by_B, gap_by_n = [], []
xs_fixed = [r2.standard_normal(30) for _ in range(SAMPLES)]
td30 = true_dist(30, TRUE_REPS)
for B in BS:
    gs = [stats.ks_2samp(td30, boot_dist(x, B) - np.median(x)).statistic
          for x in xs_fixed]
    gap_by_B.append(float(np.mean(gs)))
for n in NS:
    td = true_dist(n, TRUE_REPS)
    gs = []
    for _ in range(SAMPLES):
        xs = r2.standard_normal(n)
        gs.append(stats.ks_2samp(
            td, boot_dist(xs, 20000) - np.median(xs)).statistic)
    gap_by_n.append(float(np.mean(gs)))
print("\n[principle] 두 손잡이를 따로 돌려 본다")
print("  횟수 B 를 키울 때 (표본은 30 개로 고정)")
for B, g in zip(BS, gap_by_B):
    print(f"    B = {B:6d}:  두 분포의 간격 {g:.4f}")
print("  표본크기 n 을 키울 때 (횟수는 20,000 으로 고정)")
for n, g in zip(NS, gap_by_n):
    print(f"    n = {n:4d}:  두 분포의 간격 {g:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0), sharey=True)
ax1.plot(BS, gap_by_B, "o-", color="tab:red", lw=3.2, ms=13)
ax1.set_xscale("log")
ax1.set_xticks(BS, [f"{b:,}" for b in BS])
ax1.set_xlabel("다시 뽑는 횟수 B (표본은 30 개로 고정)", fontsize=11.5)
ax1.set_ylabel("흉내 낸 세계와 참 세계의 간격", fontsize=11.5)
ax1.set_ylim(0, max(max(gap_by_B), max(gap_by_n)) * 1.25)
ax1.grid(alpha=0.25)
ax1.set_title("횟수를 사백 배로 키워도 줄지 않는다", fontsize=12.5)
ax1.annotate("횟수는 흉내 낸 세계를\n얼마나 촘촘히 훑었는지만 정한다",
             xy=(BS[-2], gap_by_B[-2]), xytext=(70, 0.06), fontsize=11.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))

ax2.plot(NS, gap_by_n, "o-", color="tab:blue", lw=3.2, ms=13)
ax2.set_xscale("log")
ax2.set_xticks(NS, [str(n) for n in NS])
ax2.set_xlabel("표본크기 n (횟수는 20,000 으로 고정)", fontsize=11.5)
ax2.grid(alpha=0.25)
ax2.set_title("표본을 키우면 줄어든다", fontsize=12.5)
ax2.annotate("닮음을 정하는 것은 이쪽이다",
             xy=(NS[-2], gap_by_n[-2]), xytext=(12, 0.06), fontsize=11.5,
             color="tab:blue", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))

fig.suptitle("손잡이가 둘이고 서로 다른 것을 움직인다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, BOOT + "b_versus_n.png")


# ════════════════════════════════════════════════════════════════════
# 3. resampling_method — 표본을 바꿀 때와 횟수를 바꿀 때
#
# 쪽의 resampling_method_77 이 중앙값의 붓스트랩 분포 하나를 그린다.
# 들머리의 **셋째 문단**은 그려져 있지 않다 — 「중앙값의 붓스트랩
# 표준오차가 평균 684.9, 표준편차 60.4 로 흔들리고 평균의 것은 703.0,
# 표준편차 18.9 로 흔들린다. 평균 쪽의 흔들림은 거의 전부 되풀이
# 횟수에서 오므로 횟수를 키우면 줄지만, 중앙값 쪽은 키워도 남는다.」
# ════════════════════════════════════════════════════════════════════
r3 = np.random.default_rng(17_003)
N3, REPS3 = 5000, 300
SCALE3 = 54971 / np.log(2)
asym = SCALE3 / np.sqrt(N3)              # 지수꼴에서는 둘이 같다
BS3 = [100, 400, 1600, 6400]
sd_by_B = {"평균": [], "중앙값": []}
for B in BS3:
    se_m, se_md = [], []
    for _ in range(REPS3):
        x = r3.exponential(SCALE3, N3)
        se_m.append(boot_se(x, np.mean, B, r3))
        se_md.append(boot_se(x, np.median, B, r3))
    sd_by_B["평균"].append(np.std(se_m, ddof=1))
    sd_by_B["중앙값"].append(np.std(se_md, ddof=1))
print(f"\n[resampling_method] 이론 표준오차 {asym:.3f} (둘이 같다)")
for B, a, b in zip(BS3, sd_by_B["평균"], sd_by_B["중앙값"]):
    print(f"  B = {B:5d}:  표준오차 추정값의 흔들림 — "
          f"평균 {a:6.2f} · 중앙값 {b:6.2f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.plot(BS3, sd_by_B["평균"], "o-", color="tab:blue", lw=3.2, ms=13,
        label="평균의 표준오차")
ax.plot(BS3, sd_by_B["중앙값"], "s-", color="tab:red", lw=3.2, ms=12,
        label="중앙값의 표준오차")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(BS3, [f"{b:,}" for b in BS3])
ax.set_yticks([3, 10, 30, 60], ["3", "10", "30", "60"])
ax.set_xlabel("되풀이 횟수 B (로그 눈금)", fontsize=12)
ax.set_ylabel("표준오차 추정값 자체가 흔들리는 폭 (로그 눈금)",
              fontsize=12)
ax.legend(fontsize=11.8, loc="upper right")
ax.grid(alpha=0.25, which="both")
ax.annotate("평균 쪽은 횟수를 키우면 줄어든다 —\n"
            "흔들림이 거의 전부 되풀이에서 오기 때문이다",
            xy=(BS3[-2], sd_by_B["평균"][-2]), xytext=(110, 4.2),
            fontsize=11.8, color="tab:blue", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax.annotate("중앙값 쪽은 키워도 바닥이 남는다 —\n"
            "자료가 가진 값이 성기기 때문이다",
            xy=(BS3[-1], sd_by_B["중앙값"][-1]), xytext=(130, 48),
            fontsize=11.8, color="tab:red", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax.set_title(f"소득꼴 자료 {N3:,} 개 · 표본을 바꿔 가며 {REPS3} 번",
             fontsize=13)
fig.suptitle("같은 수를 겨누는데 한쪽만 횟수로 고쳐진다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, BOOT + "two_kinds_of_wobble.png")


# ════════════════════════════════════════════════════════════════════
# 4. nonparametric — 빠지는 36.8% 는 손실이 아니다
#
# 쪽의 resample_pairs_vs_separate 가 들머리의 둘째·셋째 문단을 담는다.
# **넷째 문단**이 묻고 끝난다 — 「복원추출이라 한 번 뽑을 때 원래
# 관측값은 평균 63.2% 만 들어온다. 그러면 나머지는 손실인가, 아니면
# 쓸 데가 있는가.」 그 답(빠진 것으로 정직한 오차를 잰다)이 그려져 있지
# 않다.
# ════════════════════════════════════════════════════════════════════
r4 = np.random.default_rng(17_004)
NS4 = [5, 10, 20, 50, 100, 500, 2000]
inbag = []
for n in NS4:
    idx = r4.integers(0, n, (4000, n))
    inbag.append(np.mean([len(np.unique(row)) / n for row in idx]))
limit = 1 - 1 / np.e
print(f"\n[nonparametric] 한 번 뽑을 때 들어오는 비율 (극한 {limit:.4f})")
for n, v in zip(NS4, inbag):
    print(f"  n = {n:5d}: {v:.4f}")

# 빠진 것으로 재는 오차가 정직한지 — 들어온 것으로 재면 낙관적이다
N4B, REPS4 = 60, 600
in_err, oob_err, true_err = [], [], []
for _ in range(REPS4):
    x = r4.standard_normal(N4B)
    y = 1.5 * x + r4.standard_normal(N4B) * 1.0
    idx = r4.integers(0, N4B, N4B)
    xi, yi = x[idx], y[idx]
    b = np.polyfit(xi, yi, 3)
    oob = np.setdiff1d(np.arange(N4B), idx)
    in_err.append(np.mean((yi - np.polyval(b, xi)) ** 2))
    if len(oob):
        oob_err.append(np.mean((y[oob] - np.polyval(b, x[oob])) ** 2))
    xn = r4.standard_normal(4000)
    yn = 1.5 * xn + r4.standard_normal(4000)
    true_err.append(np.mean((yn - np.polyval(b, xn)) ** 2))
print(f"  3 차식을 맞췄을 때의 제곱오차 — 들어온 것 "
      f"{np.mean(in_err):.3f} · 빠진 것 {np.mean(oob_err):.3f} · "
      f"새 자료 {np.mean(true_err):.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS4, inbag, "o-", color="tab:blue", lw=3.2, ms=13)
ax1.axhline(limit, color="tab:red", lw=2.4, ls="--")
ax1.text(NS4[-1], limit + 0.006, f"극한 {limit:.4f}", ha="right",
         va="bottom", fontsize=12, color="tab:red", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xticks(NS4, [f"{n:,}" for n in NS4])
ax1.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax1.set_ylabel("한 번 뽑을 때 들어오는 원래 관측값의 비율", fontsize=11.5)
ax1.set_ylim(0.6, 0.86)
ax1.grid(alpha=0.25)
ax1.set_title("n 이 커지면 금세 한 값으로 굳는다", fontsize=12.5)

bars = [np.mean(in_err), np.mean(oob_err), np.mean(true_err)]
ax2.bar(np.arange(3), bars, color=["tab:red", "tab:blue", "0.35"],
        width=0.55)
for i, v in enumerate(bars):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
             fontsize=14, fontweight="bold")
ax2.set_xticks(np.arange(3),
               ["재표본에 들어온 것으로\n재면", "빠진 것으로\n재면",
                "새 자료로\n재면 (참값)"], fontsize=11.3)
ax2.set_ylabel("제곱오차", fontsize=11.5)
ax2.set_ylim(0, max(bars) * 1.3)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("빠진 것이 참값을 짚는다", fontsize=12.5)
ax2.text(0.5, 0.80, "들어온 것으로 재면 낙관적이다 —\n"
         "그 자료로 맞춘 것을 그 자료로 채점하기 때문이다",
         transform=ax2.transAxes, ha="center", va="top", fontsize=11.3,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("빠지는 36.8% 는 손실이 아니라 채점지다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, BOOT + "the_left_out_third.png")


# ════════════════════════════════════════════════════════════════════
# 5. parametric — 틀린 모형이 가장 좁은 구간을 낸다
#
# 쪽의 parametric_vs_nonparametric 이 세 방식의 재표집 결과를 그린다.
# 들머리의 **셋째 문단**에는 그 그림이 담지 않는 한 수가 있다 —
# 「틀린 그 모형이 내놓는 구간이 셋 가운데 가장 **좁다**. 폭이 1.10 으로
# 옳은 모형의 1.93 보다 좁다.」 그리고 그 모형이 **대기시간이 음수인
# 가짜 자료**를 만든다는 것도. 좁음과 틀림이 함께 오는 모양을 그린다.
# ════════════════════════════════════════════════════════════════════
r5 = np.random.default_rng(17_005)
N5, REPS5, B5 = 30, 1500, 1500
true_q90 = stats.expon.ppf(0.90)
res5 = {}
for name in ("자료에서 그대로", "옳은 모형 (지수)", "틀린 모형 (정규)"):
    cov, wid = 0, []
    for _ in range(REPS5):
        x = r5.exponential(1.0, N5)
        if name == "자료에서 그대로":
            bx = x[r5.integers(0, N5, (B5, N5))]
        elif name.startswith("옳은"):
            bx = r5.exponential(x.mean(), (B5, N5))
        else:
            bx = r5.normal(x.mean(), x.std(ddof=1), (B5, N5))
        q = np.percentile(bx, 90, axis=1)
        lo, hi = np.percentile(q, [2.5, 97.5])
        cov += lo <= true_q90 <= hi
        wid.append(hi - lo)
    res5[name] = (cov / REPS5, float(np.mean(wid)))
xq = r5.exponential(1.0, N5)
neg_mass = stats.norm.cdf(0, xq.mean(), xq.std(ddof=1))
print(f"\n[parametric] 90 백분위수의 참값 {true_q90:.4f}")
for k, (c, w_) in res5.items():
    print(f"  {k:16s} 포함확률 {c:.3f} · 폭 {w_:.3f}")
print(f"  정규모형이 음수 쪽에 두는 확률질량 {neg_mass:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.1, 1]})
g5 = np.linspace(-1.6, 5, 500)
ax1.plot(g5, stats.expon.pdf(g5), color="tab:blue", lw=3.0,
         label="옳은 모형 (지수)")
nd = stats.norm.pdf(g5, xq.mean(), xq.std(ddof=1))
ax1.plot(g5, nd, color="tab:red", lw=3.0, ls="--",
         label="틀린 모형 (정규)")
neg = g5 < 0
ax1.fill_between(g5[neg], 0, nd[neg], color="tab:red", alpha=0.45)
ax1.axvline(0, color="0.2", lw=2.0)
ax1.annotate(f"대기시간이 음수인 가짜 자료를\n"
             f"{100 * neg_mass:.0f}% 만들어 낸다", xy=(-0.4, 0.12),
             xytext=(0.6, 0.72), fontsize=11.8, color="tab:red",
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.9))
ax1.set_xlim(-1.6, 5)
ax1.set_ylim(0, 1.05)
ax1.set_xlabel("대기시간", fontsize=11.5)
ax1.set_ylabel("밀도", fontsize=11.5)
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(alpha=0.25)
ax1.set_title("틀린 모형은 있을 수 없는 값을 만든다", fontsize=12.5)

names5 = list(res5)
covs = [res5[k][0] for k in names5]
wids = [res5[k][1] for k in names5]
cols5 = ["0.5", "tab:blue", "tab:red"]
for nm, c_, w_, col in zip(names5, covs, wids, cols5):
    ax2.plot([w_], [c_], "o", ms=20, color=col)
    ax2.annotate(nm, xy=(w_, c_), xytext=(0, 18),
                 textcoords="offset points", ha="center", fontsize=11.5,
                 color=col, fontweight="bold")
ax2.axhline(0.95, color="0.2", lw=2.2, ls="--")
ax2.set_xlabel("구간의 평균 폭 (좁을수록 왼쪽)", fontsize=11.5)
ax2.set_ylabel("참값을 담는 비율", fontsize=11.5)
ax2.set_ylim(min(covs) - 0.06, 1.02)
ax2.set_xlim(min(wids) * 0.82, max(wids) * 1.14)
ax2.grid(alpha=0.25)
ax2.set_title("가장 좁은 것이 가장 못 미덥다", fontsize=12.5)
ax2.text(0.03, 0.06, "왼쪽 아래가 「좁은데 틀린」 자리다",
         transform=ax2.transAxes, ha="left", va="bottom", fontsize=11.5,
         color="tab:red", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="tab:red"))

fig.suptitle("좁은 구간과 틀린 답이 함께 온다", fontsize=15, y=1.0)
plt.tight_layout()
save(fig, BOOT + "wrong_model_narrowest.png")


# ════════════════════════════════════════════════════════════════════
# 6. percentile — 포함확률 오차가 제곱근으로만 줄어든다
#
# 쪽의 percentile_invariance 가 두 척도를, league_table_ranks 가 순위표를
# 그린다. 들머리의 **마지막 문단**은 그려져 있지 않다 — 「이 방법의
# 포함확률 오차는 표본크기의 제곱근에 반비례해서만 줄고, 치우친
# 통계량에서는 명목 95% 가 실제로 85–90% 에 그친다.」
# ════════════════════════════════════════════════════════════════════
r6 = np.random.default_rng(17_006)
NS6 = [10, 20, 40, 80, 160, 320]
REPS6, B6 = 1200, 1200
cov6 = []
for n in NS6:
    hit = 0
    for _ in range(REPS6):
        x = r6.exponential(1.0, n)
        bx = x[r6.integers(0, n, (B6, n))]
        lo, hi = np.percentile(bx.mean(1), [2.5, 97.5])
        hit += lo <= 1.0 <= hi
    cov6.append(hit / REPS6)
err6 = 0.95 - np.array(cov6)
print("\n[percentile] 명목 95% 의 실제 포함확률 (지수 자료의 평균)")
for n, c, e in zip(NS6, cov6, err6):
    print(f"  n = {n:4d}: {c:.4f}  (모자란 양 {e:.4f} · "
          f"루트 n 배 {e * np.sqrt(n):.3f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(NS6, cov6, "o-", color="tab:blue", lw=3.2, ms=13)
ax1.axhline(0.95, color="0.2", lw=2.4, ls="--")
ax1.set_xscale("log")
ax1.set_xticks(NS6, [str(n) for n in NS6])
ax1.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax1.set_ylabel("명목 95% 구간이 참값을 담는 비율", fontsize=11.5)
ax1.set_ylim(min(cov6) - 0.03, 0.98)
ax1.grid(alpha=0.25)
ax1.text(NS6[0], 0.955, "약속한 0.95", ha="left", va="bottom",
         fontsize=11.5, color="0.2", fontweight="bold")
ax1.set_title("치우친 통계량에서 약속에 못 미친다", fontsize=12.5)

ax2.plot(NS6, err6, "o-", color="tab:red", lw=3.2, ms=13,
         label="모자란 양")
ref = err6[0] * np.sqrt(NS6[0]) / np.sqrt(NS6)
ax2.plot(NS6, ref, "--", color="0.45", lw=2.4,
         label="제곱근에 반비례하는 기준선")
slope = np.polyfit(np.log(NS6), np.log(np.maximum(err6, 1e-6)), 1)[0]
print(f"  재어 본 기울기 {slope:.3f} (제곱근이면 -0.5)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xticks(NS6, [str(n) for n in NS6])
ax2.set_yticks([0.005, 0.01, 0.03, 0.09], ["0.005", "0.01", "0.03",
                                           "0.09"])
ax2.set_xlabel("표본크기 n (로그 눈금)", fontsize=11.5)
ax2.set_ylabel("0.95 에서 모자란 양 (로그 눈금)", fontsize=11.5)
ax2.legend(fontsize=11.5, loc="upper right")
ax2.grid(alpha=0.25, which="both")
ax2.set_title("기준선과 나란하거나 조금 가파르다 — 어느 쪽이든 느리다",
              fontsize=12)
ax2.text(0.03, 0.06, "표본을 네 배로 늘려야\n모자란 양이 겨우 절반쯤 된다",
         transform=ax2.transAxes, ha="left", va="bottom", fontsize=11.5,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("척도를 고르지 않아도 되는 대신 치르는 값", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, CI + "error_shrinks_only_by_root_n.png")


# ════════════════════════════════════════════════════════════════════
# 7. basic — 반사한 끝점은 있을 수 없는 값이 될 수 있다
#
# 쪽의 basic_is_a_mirror 가 두 구간을 그려 들머리의 첫·둘째 문단을
# 담는다. **셋째 문단**은 그려져 있지 않다 — 「뒤집어 얻은 끝점은 실제
# 재표본에서 계산된 값이 아니므로 통계량이 가질 수 없는 값이 될 수
# 있다. 분산의 구간을 만들어 보면 다섯 번에 한 번꼴로 하한이 음수가
# 된다.」 쪽이 변동계수 0.510 이라는 임계값으로 못 박으므로 그 축으로
# 그린다.
# ════════════════════════════════════════════════════════════════════
r7 = np.random.default_rng(17_007)
CVS = np.array([0.25, 0.4, 0.51, 0.65, 0.8, 1.0, 1.4])
NS7 = [10, 20, 40]
REPS7, B7 = 1500, 800
neg_rate = {n: [] for n in NS7}
for n in NS7:
    for cv in CVS:
        sig = np.sqrt(np.log(1 + cv ** 2))
        hit = 0
        for _ in range(REPS7):
            x = np.exp(r7.normal(0, sig, n))
            v = x.var(ddof=1)
            bv = x[r7.integers(0, n, (B7, n))].var(ddof=1, axis=1)
            lo_basic = 2 * v - np.percentile(bv, 97.5)
            hit += lo_basic < 0
        neg_rate[n].append(hit / REPS7)
print("\n[basic] 반사한 하한이 음수가 되는 비율")
for n in NS7:
    print(f"  n = {n:3d}: " + " ".join(f"{v:.3f}" for v in neg_rate[n]))

fig, ax = plt.subplots(figsize=(11.0, 5.8))
for n, c in zip(NS7, ["tab:red", "tab:orange", "tab:blue"]):
    ax.plot(CVS, neg_rate[n], "o-", color=c, lw=3.0, ms=12,
            label=f"n = {n}")
ax.axvline(0.510, color="0.2", lw=2.4, ls="--")
ax.text(0.520, 0.92, "쪽이 못 박는 임계값 0.510", ha="left", va="top",
        fontsize=12, color="0.2", fontweight="bold", rotation=90)
ax.set_xlabel("자료의 변동계수", fontsize=12)
ax.set_ylabel("분산의 하한이 **음수**가 되는 비율".replace("**", ""),
              fontsize=12)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=11.8, loc="upper left")
ax.grid(alpha=0.25)
ax.annotate("분산은 음수가 될 수 없는 양인데\n"
            "반사한 끝점은 실제 재표본의 값이 아니다",
            xy=(0.85, neg_rate[20][4]), xytext=(0.30, 0.68),
            fontsize=12, color="0.2", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.2", lw=2.0))
ax.set_title("분위수를 그대로 읽는 쪽은 한 번도 음수로 내려가지 않는다",
             fontsize=13)
fig.suptitle("반사는 공짜가 아니다 — 있을 수 없는 값이 나온다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, CI + "mirror_can_leave_the_range.png")
