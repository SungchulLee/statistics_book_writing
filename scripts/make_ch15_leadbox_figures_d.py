"""15장 들어가는 이야기 그림 — 넷째 그룹 여섯 (마지막).

  21. anova_pretest         두 단계의 크기는 두 전략을 파이로 섞은 값이다
  22. financial_volatility  꼬리에 듣던 로버스트성이 종속에는 듣지 않는다
  23. regression_variance   R 제곱이 작다고 이분산이 약한 것이 아니다
  24. bootstrap_variance    붓스트랩도 분산 앞에서는 반쯤만 듣는다
  25. likelihood_ratio      둘이 같은 뿌리에서 자랐다 — 수로 확인한다
  26. bayesian_variance     베이즈라고 분포 가정이 없는 것이 아니다

만드는 파일:
  docs/ch15/applications/img/mixed_by_pi.png
  docs/ch15/applications/img/robust_to_tails_not_to_clustering.png
  docs/ch15/applications/img/r2_does_not_measure_it.png
  docs/ch15/advanced_methods/img/bootstrap_half_fixes_it.png
  docs/ch15/advanced_methods/img/same_root_two_names.png
  docs/ch15/advanced_methods/img/bayes_inherits_the_assumption.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

APP = "docs/ch15/applications/img/"
ADV = "docs/ch15/advanced_methods/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
# 21. anova_pretest — 두 단계의 크기는 두 전략을 파이로 섞은 값이다
#
# 쪽의 pretest_strategies 가 세 전략의 크기를, pretest_conditional 이
# 통과분의 분포를 그린다. 들머리의 **둘째 문단**은 식으로만 적혀 있다 —
# 「두 단계의 크기는 두 전략의 크기를 사전검정의 기각률 파이로 섞은
# 값이고, 그 파이가 (5,5,5,5) 에서 0.0276, (5,5,5,10) 에서 0.6108,
# (2,4,6,10) 에서 0.9798 이다.」 섞인다는 말은 **파이를 가로축에 놓아야**
# 보인다. 두 단계 곡선이 두 전략 사이를 직선으로 잇는 것이 그림이다.
# ════════════════════════════════════════════════════════════════════
PATTERNS = [("(5,5,5,5)", np.array([5., 5., 5., 5.])),
            ("(5,5,5,7)", np.array([5., 5., 5., 7.])),
            ("(5,5,5,10)", np.array([5., 5., 5., 10.])),
            ("(3,5,7,10)", np.array([3., 5., 7., 10.])),
            ("(2,4,6,10)", np.array([2., 4., 6., 10.]))]
K21, N21, REPS21 = 4, 15, 8000
r21 = np.random.default_rng(15_321)
pis, anova_s, welch_s, two_s = [], [], [], []
for nm, sds in PATTERNS:
    hit_pi = hit_a = hit_w = hit_2 = 0
    for _ in range(REPS21):
        g = [r21.standard_normal(N21) * s for s in sds]
        pre = stats.levene(*g, center="median").pvalue < 0.05
        hit_pi += pre
        pa = stats.f_oneway(*g).pvalue < 0.05
        pw = stats.alexandergovern(*g).pvalue < 0.05
        hit_a += pa
        hit_w += pw
        hit_2 += pw if pre else pa
    pis.append(hit_pi / REPS21)
    anova_s.append(hit_a / REPS21)
    welch_s.append(hit_w / REPS21)
    two_s.append(hit_2 / REPS21)
print("[anova_pretest] 사전검정의 기각률 파이와 세 전략의 크기")
for (nm, _), p_, a, w, t2 in zip(PATTERNS, pis, anova_s, welch_s, two_s):
    print(f"  {nm:12s} 파이 {p_:.4f}:  분산분석 {a:.4f} · "
          f"웰치 {w:.4f} · 두 단계 {t2:.4f}  "
          f"(섞은 값 {(1 - p_) * a + p_ * w:.4f})")

fig, ax = plt.subplots(figsize=(11.0, 5.9))
order = np.argsort(pis)
pi_s = np.array(pis)[order]
ax.plot(pi_s, np.array(anova_s)[order], "s--", color="tab:red", lw=2.6,
        ms=11, label="늘 표준 분산분석만")
ax.plot(pi_s, np.array(welch_s)[order], "^--", color="tab:green", lw=2.6,
        ms=11, label="늘 웰치만")
ax.plot(pi_s, np.array(two_s)[order], "o-", color="tab:blue", lw=3.4,
        ms=13, label="등분산을 먼저 검정해 갈라 쓰기")
mix = [(1 - p_) * a + p_ * w for p_, a, w in zip(pis, anova_s, welch_s)]
ax.plot(pi_s, np.array(mix)[order], color="0.25", lw=1.8, ls=":",
        label="두 전략을 파이로 섞은 값")
ax.axhline(0.05, color="0.2", lw=2.0, ls="--")
for i in order:
    ax.annotate(PATTERNS[i][0], xy=(pis[i], two_s[i]), xytext=(0, 14),
                textcoords="offset points", ha="center", fontsize=10.5,
                color="tab:blue", fontweight="bold")
ax.set_xlabel("사전검정이 기각하는 비율 파이", fontsize=12)
ax.set_ylabel("네 평균이 모두 같은데 기각하는 비율", fontsize=12)
ax.set_ylim(0.02, 0.085)
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=11, loc="lower left")
ax.grid(alpha=0.25)
ax.annotate("파이가 가운데일 때 두 단계가 가장 나쁘다 —\n"
            "사전검정이 자주 놓치는 자리다",
            xy=(pis[2], two_s[2]), xytext=(0.12, 0.078), fontsize=11.5,
            color="tab:blue", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax.set_title("두 단계는 새 전략이 아니라 두 전략을 섞은 것이다",
             fontsize=13)
fig.suptitle("미리 검정하고 들어가면 — 섞이는 비율이 파이다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, APP + "mixed_by_pi.png")


# ════════════════════════════════════════════════════════════════════
# 22. financial_volatility — 꼬리에 듣던 것이 종속에는 안 듣는다
#
# 쪽의 volatility_estimation_noise 가 「일정한데 흔들린다」를,
# volatility_clustering 이 뭉침 자체를 그린다. 들머리의 **셋째 문단**은
# 그 둘이 건드리지 않는다 — 「뭉침까지 넣으면 세 검정이 함께 무너진다.
# 꼬리에는 듣던 로버스트성이 종속에는 듣지 않는 것이다.」
#
# 세 검정 × 세 세계(독립 정규 · 독립 두꺼운 꼬리 · 뭉침)를 나란히 놓으면
# 가운데 칸에서 갈라지던 것이 오른쪽 칸에서 다시 모인다.
# ════════════════════════════════════════════════════════════════════
N22, REPS22 = 250, 4000
r22 = np.random.default_rng(15_322)


def garch(n, rng, omega=0.05, a=0.10, b=0.85, burn=300):
    m = n + burn
    e = np.empty(m)
    s2 = omega / (1 - a - b)
    for i in range(m):
        e[i] = rng.standard_normal() * np.sqrt(s2)
        s2 = omega + a * e[i] ** 2 + b * s2
    return e[burn:]


WORLDS = [("독립 정규", lambda n: r22.standard_normal(n)),
          ("독립 두꺼운 꼬리", lambda n: r22.standard_t(5, n)
           * np.sqrt(3 / 5)),
          ("뭉쳐 오는 변동성", lambda n: garch(n, r22))]
res22 = {nm: [] for nm in ("F 검정", "브라운–포사이드", "플리그너–킬린")}
for wnm, gen in WORLDS:
    cnt = {k: 0 for k in res22}
    for _ in range(REPS22):
        x = gen(2 * N22)
        a_, b_ = x[:N22], x[N22:]
        d = N22 - 1
        fr = a_.var(ddof=1) / b_.var(ddof=1)
        pf = 2 * min(stats.f.cdf(fr, d, d), stats.f.sf(fr, d, d))
        cnt["F 검정"] += pf < 0.05
        cnt["브라운–포사이드"] += stats.levene(a_, b_,
                                        center="median").pvalue < 0.05
        cnt["플리그너–킬린"] += stats.fligner(a_, b_).pvalue < 0.05
    for k in res22:
        res22[k].append(cnt[k] / REPS22)
print("\n[financial_volatility] 세 세계에서 세 검정 "
      f"(두 기간 각 {N22} 일, 변동성은 어느 쪽도 변하지 않는다)")
for k, v in res22.items():
    print(f"  {k:12s}: " + " · ".join(f"{x:.3f}" for x in v))

fig, ax = plt.subplots(figsize=(11.0, 5.8))
xx = np.arange(len(WORLDS))
w = 0.26
COL22 = [("F 검정", "tab:red"), ("브라운–포사이드", "tab:blue"),
         ("플리그너–킬린", "tab:green")]
for i, (nm, c) in enumerate(COL22):
    ax.bar(xx + (i - 1) * w, res22[nm], width=w, color=c, label=nm)
    for j in xx:
        ax.text(xx[j] + (i - 1) * w, res22[nm][j] + 0.008,
                f"{res22[nm][j]:.3f}", ha="center", va="bottom",
                fontsize=10.3, fontweight="bold", color=c)
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.set_xticks(xx, [nm for nm, _ in WORLDS], fontsize=12.5)
ax.set_xlabel("자료가 오는 방식 (어느 쪽도 변동성이 변하지 않았다)",
              fontsize=12)
ax.set_ylabel("「변동성이 달라졌다」고 답하는 비율", fontsize=12)
ax.set_ylim(0, max(max(v) for v in res22.values()) * 1.32)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.annotate("꼬리에는 두 쪽이 버티는데", xy=(1 + 0.0, res22["브라운–포사이드"][1]),
            xytext=(0.62, 0.30), fontsize=11.8, color="tab:blue",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.9))
ax.annotate("뭉침 앞에서는 셋이 함께 무너진다", xy=(2 - 0.1, 0.22),
            xytext=(1.15, 0.41), fontsize=11.8, color="0.2",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.2", lw=1.9))
ax.set_title("로버스트성은 꼬리에 대한 것이지 종속에 대한 것이 아니다",
             fontsize=13)
fig.suptitle("변동성이 그대로인데 달라졌다고 답한다", fontsize=15, y=0.975)
plt.tight_layout()
save(fig, APP + "robust_to_tails_not_to_clustering.png")


# ════════════════════════════════════════════════════════════════════
# 23. regression_variance — R 제곱은 이분산의 세기를 재지 않는다
#
# 쪽의 breusch_pagan_mechanism 이 잔차 그림과 보조회귀를, learned_variance
# 가 배운 분산을 그린다. 들머리의 **셋째 문단**이 경고 하나를 남긴다 —
# 「뚜렷한 이분산에서도 보조회귀의 R 제곱이 0.114 에 머문다. R 제곱이
# 작다고 이분산이 약한 것이 아니다. 판정은 nR 제곱으로 해야 한다.」
#
# 이분산의 세기를 손잡이로 돌려 R 제곱과 nR 제곱을 함께 그리면, 한쪽은
# 낮게 머무는데 다른 쪽은 임계값을 훌쩍 넘는다.
# ════════════════════════════════════════════════════════════════════
N23, REPS23 = 120, 3000
SLOPES = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
r23 = np.random.default_rng(15_323)
r2_med, nr2_med, rej23 = [], [], []
for sl in SLOPES:
    r2s, nr2s, rj = [], [], 0
    for _ in range(REPS23):
        x = r23.uniform(0, 10, N23)
        sd = 0.4 + sl * x
        y = 2 + x + r23.standard_normal(N23) * sd
        X = np.column_stack([np.ones(N23), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        e2 = (y - X @ beta) ** 2
        g = np.linalg.lstsq(X, e2, rcond=None)[0]
        fit = X @ g
        ss = 1 - ((e2 - fit) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
        r2s.append(ss)
        nr2s.append(N23 * ss)
        rj += stats.chi2.sf(N23 * ss, 1) < 0.05
    r2_med.append(np.median(r2s))
    nr2_med.append(np.median(nr2s))
    rej23.append(rj / REPS23)
print("\n[regression_variance] 이분산의 세기를 키워 간다")
for sl, a, b, c in zip(SLOPES, r2_med, nr2_med, rej23):
    print(f"  기울기 {sl:4.2f}:  보조회귀 R2 {a:.4f} · "
          f"nR2 {b:7.2f} · 기각률 {c:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(SLOPES, r2_med, "o-", color="tab:purple", lw=3.2, ms=12)
ax1.set_xlabel("오차 표준편차가 x 를 따라 커지는 기울기", fontsize=11.5)
ax1.set_ylabel("보조회귀의 R 제곱 (가운데값)", fontsize=11.5)
ax1.set_ylim(0, 0.2)
ax1.grid(alpha=0.25)
ax1.set_title("설명력은 끝까지 낮게 머문다", fontsize=12.5)
ax1.annotate(f"가장 센 자리에서도 {r2_med[-1]:.3f} 다 —\n"
             f"제곱잔차 자체의 변동이 워낙 커서다",
             xy=(SLOPES[-1], r2_med[-1]), xytext=(0.06, 0.165),
             fontsize=11.3, color="tab:purple", fontweight="bold",
             ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.9))

ax2.plot(SLOPES, nr2_med, "o-", color="tab:red", lw=3.2, ms=12,
         label="nR 제곱 (가운데값)")
ax2.axhline(stats.chi2.ppf(0.95, 1), color="0.2", lw=2.2, ls="--")
ax2.text(SLOPES[-1], stats.chi2.ppf(0.95, 1) + 0.8,
         f"자유도 1 의 5% 임계값 {stats.chi2.ppf(0.95, 1):.2f}", ha="right",
         va="bottom", fontsize=11, color="0.2", fontweight="bold")
ax2.set_xlabel("오차 표준편차가 x 를 따라 커지는 기울기", fontsize=11.5)
ax2.set_ylabel("nR 제곱 (가운데값)", fontsize=11.5)
ax2.set_ylim(0, max(nr2_med) * 1.2)
ax2.grid(alpha=0.25)
ax2.set_title("같은 R 제곱에 n 을 곱한 수는 임계값을 훌쩍 넘는다",
              fontsize=12.5)
ax2.legend(fontsize=11, loc="upper left")
for i, sl in enumerate(SLOPES):
    if sl in (0.5, 1.0):
        ax2.text(sl, nr2_med[i] + 0.8, f"기각률 {rej23[i]:.2f}",
                 ha="center", va="bottom", fontsize=11,
                 fontweight="bold", color="tab:red")

fig.suptitle("R 제곱이 작다고 이분산이 약한 것이 아니다", fontsize=15,
             y=1.0)
plt.tight_layout()
save(fig, APP + "r2_does_not_measure_it.png")


# ════════════════════════════════════════════════════════════════════
# 24. bootstrap_variance — 붓스트랩도 반쯤만 듣는다
#
# 쪽의 bootstrap_vs_f_size 는 **두** 검정(붓스트랩과 F)의 크기를 그린다.
# 들머리의 셋째 문단은 거기에 **셋째**를 들여온다 — 「그 0.089 도 명목값의
# 두 배이고, 같은 조건에서 브라운–포사이드는 0.048 이다. 붓스트랩도
# 분산 앞에서는 반쯤만 듣는다.」 셋을 나란히 놓아야 「반쯤」이 보인다.
# ════════════════════════════════════════════════════════════════════
N24, REPS24, B24 = 20, 1500, 399
r24 = np.random.default_rng(15_324)
POP24 = [("정규", lambda n: r24.standard_normal(n)),
         ("t(5)", lambda n: r24.standard_t(5, n)),
         ("지수", lambda n: r24.exponential(1.0, n)),
         ("대수정규", lambda n: np.exp(r24.normal(0, 1, n)))]
res24 = {nm: [] for nm in ("F 검정", "붓스트랩", "브라운–포사이드")}
for pnm, gen in POP24:
    cnt = {k: 0 for k in res24}
    for _ in range(REPS24):
        a_, b_ = gen(N24), gen(N24)
        d = N24 - 1
        fr = a_.var(ddof=1) / b_.var(ddof=1)
        pf = 2 * min(stats.f.cdf(fr, d, d), stats.f.sf(fr, d, d))
        cnt["F 검정"] += pf < 0.05
        cnt["브라운–포사이드"] += stats.levene(a_, b_,
                                        center="median").pvalue < 0.05
        # 귀무를 올바르게 부과한다 — 각자의 표준편차로 나눠 척도를 맞춘다
        za = (a_ - a_.mean()) / a_.std(ddof=1)
        zb = (b_ - b_.mean()) / b_.std(ddof=1)
        pool = np.r_[za, zb]
        obs = abs(np.log(fr))
        ia = r24.integers(0, len(pool), (B24, N24))
        ib = r24.integers(0, len(pool), (B24, N24))
        boot = np.abs(np.log(pool[ia].var(1, ddof=1)
                             / pool[ib].var(1, ddof=1)))
        cnt["붓스트랩"] += (boot >= obs).mean() < 0.05
    for k in res24:
        res24[k].append(cnt[k] / REPS24)
print(f"\n[bootstrap_variance] 집단당 {N24}, 등분산이 참")
for k, v in res24.items():
    print(f"  {k:12s}: " + " · ".join(f"{x:.3f}" for x in v))

fig, ax = plt.subplots(figsize=(11.0, 5.8))
xx = np.arange(len(POP24))
w = 0.26
for i, (nm, c) in enumerate([("F 검정", "tab:red"),
                             ("붓스트랩", "tab:orange"),
                             ("브라운–포사이드", "tab:blue")]):
    ax.bar(xx + (i - 1) * w, res24[nm], width=w, color=c, label=nm)
    for j in xx:
        ax.text(xx[j] + (i - 1) * w, res24[nm][j] + 0.008,
                f"{res24[nm][j]:.3f}", ha="center", va="bottom",
                fontsize=10.3, fontweight="bold", color=c)
ax.axhline(0.05, color="0.2", lw=2.2, ls="--")
ax.axhline(0.10, color="0.5", lw=1.8, ls=":")
ax.set_xticks(xx, [nm for nm, _ in POP24], fontsize=12.5)
ax.set_xlabel("모집단 (분산은 어느 자리에서나 두 집단이 같다)",
              fontsize=12)
ax.set_ylabel("「분산이 다르다」고 답하는 비율", fontsize=12)
ax.set_ylim(0, max(max(v) for v in res24.values()) * 1.3)
ax.legend(fontsize=11.5, loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.text(xx[0] - 0.42, 0.112, "두 배 선 (0.10)", ha="left", va="bottom",
        fontsize=10.8, color="0.45", fontweight="bold")
ax.annotate("재표집이 F 의 왜곡을 크게 덜어 주지만\n"
            "명목선까지는 못 데려간다 — **반쯤**이다".replace("**", ""),
            xy=(2 + 0.0, res24["붓스트랩"][2]), xytext=(0.25, 0.42),
            fontsize=11.8, color="tab:orange", fontweight="bold",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.9))
ax.set_title("재표집이 고치는 것과 고치지 못하는 것", fontsize=13)
fig.suptitle("붓스트랩도 분산 앞에서는 반쯤만 듣는다", fontsize=15, y=0.975)
plt.tight_layout()
save(fig, ADV + "bootstrap_half_fixes_it.png")


# ════════════════════════════════════════════════════════════════════
# 25. likelihood_ratio — 둘이 같은 뿌리에서 자랐다
#
# 쪽의 lrt_g_function 은 벌점함수의 모양과 기각역을 그린다. 들머리의
# **셋째 문단**이 이 쪽의 주장이다 — 「이 쪽은 새 검정을 하나 더 내놓는
# 자리가 아니라 이미 세운 검정들이 하나의 원리에서 나온다는 것을 보이는
# 자리다. 한쪽이 다른 쪽의 특별한 경우인 것이 아니라 둘이 같은 뿌리에서
# 자란 것이다.」
#
# 주장을 그림으로 확인한다. 같은 자료에서 두 통계량을 각각 계산해
# 맞대면 점들이 한 줄에 정확히 올라앉는다 — 그 줄이 보정인자 C 다.
# ════════════════════════════════════════════════════════════════════
r25 = np.random.default_rng(15_325)
NS25 = [(15, 18), (8, 8), (5, 30), (10, 10, 10), (6, 9, 12, 20)]
pts_lrt, pts_bart, cs = [], [], []
for sizes in NS25:
    for _ in range(300):
        g = [r25.standard_normal(n) * 1.0 for n in sizes]
        k = len(g)
        nu = np.array([len(x) - 1 for x in g], float)
        s2 = np.array([x.var(ddof=1) for x in g])
        sp2 = (nu * s2).sum() / nu.sum()
        lam = (nu * np.log(sp2 / s2)).sum()          # 보정 전 가능도비
        C = 1 + (np.sum(1 / nu) - 1 / nu.sum()) / (3 * (k - 1))
        pts_lrt.append(lam)
        pts_bart.append(stats.bartlett(*g).statistic)
        cs.append(C)
pts_lrt = np.array(pts_lrt)
pts_bart = np.array(pts_bart)
cs = np.array(cs)
err = np.abs(pts_bart - pts_lrt / cs).max()
print("\n[likelihood_ratio] 같은 자료에서 두 통계량을 각각 계산해 맞댄다")
print(f"  표본 {len(pts_lrt)} 벌 · 가장 크게 어긋난 자리 {err:.3e}")
print(f"  보정인자 C 의 범위 {cs.min():.4f} ~ {cs.max():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))
ax1.plot(pts_lrt / cs, pts_bart, "o", ms=4, color="tab:blue", alpha=0.45)
lim = [0, max(pts_bart.max(), (pts_lrt / cs).max()) * 1.05]
ax1.plot(lim, lim, color="tab:red", lw=2.4, ls="--",
         label="두 수가 같다면 이 선 위")
ax1.set_xlim(lim)
ax1.set_ylim(lim)
ax1.set_xlabel("가능도비 통계량을 C 로 나눈 값", fontsize=11.5)
ax1.set_ylabel("바틀렛 통계량", fontsize=11.5)
ax1.set_title(f"점 {len(pts_lrt)} 개가 모두 선 위에 있다", fontsize=12.5)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(alpha=0.25)
ax1.text(0.97, 0.06, f"가장 크게 어긋난 자리가 {err:.1e} 다 —\n"
         f"수치오차이고 공식이 같다는 뜻이다", transform=ax1.transAxes,
         ha="right", va="bottom", fontsize=11, color="0.2",
         fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

rr = np.linspace(0.25, 4.0, 400)
pen = rr - 1 - np.log(rr)
ax2.plot(rr, pen, color="tab:purple", lw=3.2)
ax2.axvline(1.0, color="0.35", lw=2.0, ls=":")
for r_, c in [(0.5, "tab:blue"), (2.0, "tab:red")]:
    v = r_ - 1 - np.log(r_)
    ax2.plot([r_], [v], "o", ms=13, color=c)
    ax2.annotate(f"비가 {r_} 이면 벌점 {v:.3f}", xy=(r_, v),
                 xytext=(r_ + (0.35 if r_ > 1 else 0.1), v + 0.42),
                 fontsize=11.3, color=c, fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.9))
ax2.set_xlabel("두 분산의 비", fontsize=11.5)
ax2.set_ylabel("가능도가 매기는 벌점", fontsize=11.5)
ax2.set_ylim(0, 1.6)
ax2.set_xlim(0.25, 4.0)
ax2.grid(alpha=0.25)
ax2.set_title("벌점이 1 을 중심으로 대칭이 아니다", fontsize=12.5)
ax2.text(0.97, 0.93, "절반이 된 것보다 두 배가 된 것에\n"
         f"{(2 - 1 - np.log(2)) / (0.5 - 1 - np.log(0.5)):.1f} 배 강한 "
         f"증거를 준다", transform=ax2.transAxes, ha="right", va="top",
         fontsize=11, color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="0.7"))

fig.suptitle("한쪽이 다른 쪽의 특별한 경우가 아니라 둘이 같은 뿌리다",
             fontsize=15, y=1.0)
plt.tight_layout()
save(fig, ADV + "same_root_two_names.png")


# ════════════════════════════════════════════════════════════════════
# 26. bayesian_variance — 베이즈라고 분포 가정이 없는 것이 아니다
#
# 쪽의 그림 다섯이 사전 강도·두 구간·사후 요약을 모두 담는다. 들머리의
# **넷째 문단**만 그려져 있지 않다 — 「베이즈라고 분포 가정이 없는 것이
# 아니다. 여전히 정규 가능도를 쓰므로 비정규성 앞에서는 같은 처지다.」
#
# 「같은 처지」를 재려면 **포함률**을 보아야 한다. 확산 사전분포를 쓴
# 신용구간과 빈도주의 신뢰구간을 같은 비정규 자료에 걸면 두 곡선이
# 포개진 채로 함께 내려간다.
# ════════════════════════════════════════════════════════════════════
N26, REPS26 = 30, 20_000
r26 = np.random.default_rng(15_326)
POP26 = [("균등", -1.2, lambda m, n: r26.uniform(-np.sqrt(3), np.sqrt(3),
                                                 (m, n))),
         ("정규", 0.0, lambda m, n: r26.standard_normal((m, n))),
         ("t(10)", 1.0, lambda m, n: r26.standard_t(10, (m, n))
          * np.sqrt(8 / 10)),
         ("t(6)", 3.0, lambda m, n: r26.standard_t(6, (m, n))
          * np.sqrt(4 / 6)),
         ("지수", 6.0, lambda m, n: r26.exponential(1.0, (m, n)))]
A0, B0 = 0.001, 0.001          # 확산 사전분포
cov_freq, cov_bayes, g26 = [], [], []
for nm, g2, gen in POP26:
    d = gen(REPS26, N26)
    s2 = d.var(1, ddof=1)
    lo = (N26 - 1) * s2 / stats.chi2.ppf(0.975, N26 - 1)
    hi = (N26 - 1) * s2 / stats.chi2.ppf(0.025, N26 - 1)
    cov_freq.append(((lo <= 1.0) & (1.0 <= hi)).mean())
    an = A0 + (N26 - 1) / 2
    bn = B0 + (N26 - 1) * s2 / 2
    blo = stats.invgamma.ppf(0.025, an, scale=bn)
    bhi = stats.invgamma.ppf(0.975, an, scale=bn)
    cov_bayes.append(((blo <= 1.0) & (1.0 <= bhi)).mean())
    g26.append(g2)
print(f"\n[bayesian_variance] 확산 사전분포 (n = {N26}, 참 분산 1)")
for (nm, g2, _), a, b in zip(POP26, cov_freq, cov_bayes):
    print(f"  {nm:6s} 초과첨도 {g2:+5.1f}:  신뢰구간 {a:.4f} · "
          f"신용구간 {b:.4f}")

fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.axhline(0.95, color="0.2", lw=2.4, ls="--")
ax.plot(g26, cov_freq, "o-", color="tab:blue", lw=3.4, ms=14,
        label="빈도주의 95% 신뢰구간")
ax.plot(g26, cov_bayes, "s--", color="tab:red", lw=2.6, ms=10,
        label="확산 사전분포의 95% 신용구간")
for i in range(len(POP26)):
    ax.text(g26[i], cov_bayes[i] - 0.018, f"{cov_bayes[i]:.3f}",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="tab:red")
ax.set_xlabel("모집단의 초과첨도 (정규가 0)", fontsize=12)
ax.set_ylabel("참 분산을 담는 비율", fontsize=12)
ax.set_ylim(0.55, 1.02)
ax.legend(fontsize=11.8, loc="lower left")
ax.grid(alpha=0.25)
ax.text(-1.15, 0.957, "약속한 0.95", ha="left", va="bottom", fontsize=11.5,
        color="0.2", fontweight="bold")
ax.annotate("두 곡선이 포개진 채 함께 내려간다 —\n"
            "사전분포를 퍼뜨려도 정규 가능도는 그대로다",
            xy=(4.2, (cov_freq[-1] + cov_bayes[-1]) / 2 + 0.02),
            xytext=(-1.0, 0.70), fontsize=12, color="0.2",
            fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.2", lw=2.0))
ax.set_title(f"같은 자료, 같은 표본크기 (n = {N26}) — 같은 처지다",
             fontsize=13)
fig.suptitle("베이즈라고 분포 가정이 없는 것이 아니다", fontsize=15,
             y=0.975)
plt.tight_layout()
save(fig, ADV + "bayes_inherits_the_assumption.png")
