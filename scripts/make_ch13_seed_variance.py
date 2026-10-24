r"""방법마다 시드 분산이 다를 때, 시드를 몇 개씩 돌렸느냐가 결론을 뒤집는다.

세 방법의 **참 성능이 모두 같고** 시드 표준편차만 $\sigma = (1, 2, 4)$ 로 다르다.
여기에 시드 수를 배치하는 방식만 바꾸어 본다.

① 흔들리는 쪽에 시드를 **적게** 주면 고전 $F$ 의 기각률이 치솟고, 반대로 주면
   지나치게 보수적이 된다. Welch 는 어느 쪽에서도 $0.05$ 를 지킨다.
② 시드를 다 합쳐 $18$ 개여도 Welch 의 유효 자유도는 흔들리는 쪽이 정한다.
③ 「가장 좋은 시드를 적는」 관행은 시드를 더 돌릴수록 더 나빠진다.

만드는 파일: docs/ch13/anova_welch/img/seed_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

SIG = np.array([1.0, 2.0, 4.0])      # 방법마다 다른 시드 표준편차
K = len(SIG)
TOTAL = 18                            # 쓸 수 있는 시드 수의 총합
LAYOUTS = {
    "흔들리는 쪽에 적게\n$n = (10, 5, 3)$": np.array([10, 5, 3]),
    "똑같이\n$n = (6, 6, 6)$": np.array([6, 6, 6]),
    "흔들리는 쪽에 많이\n$n = (3, 5, 10)$": np.array([3, 5, 10]),
}
REPS = 20000
ALPHA = 0.05


def welch_anova(groups):
    """Welch 의 일원분산분석 — 통계량, 자유도, p 값."""
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) for g in groups])
    w = n / v
    W = w.sum()
    mt = (w * m).sum() / W
    k = len(groups)
    A = (w * (m - mt) ** 2).sum() / (k - 1)
    lam = 3.0 / (k * k - 1) * (((1 - w / W) ** 2) / (n - 1)).sum()
    f = A / (1 + 2 * (k - 2) * lam)
    df2 = 1.0 / lam
    return f, k - 1, df2, stats.f.sf(f, k - 1, df2)


# --- k = 2 이면 Welch t 검정과 같아야 한다 (쪽의 연습문제가 말하는 것) ---
a = RNG.normal(0, 1, 9)
b = RNG.normal(0, 3, 6)
fw, _, dfw, pw = welch_anova([a, b])
tw = stats.ttest_ind(a, b, equal_var=False)
print(f"검산 (k=2): F_W = {fw:.6f} 대 t_W^2 = {tw.statistic ** 2:.6f}, "
      f"자유도 {dfw:.4f} 대 {tw.df:.4f}, p {pw:.6f} 대 {tw.pvalue:.6f}")

# === ① 배치에 따른 기각률 ===========================================
print(f"\n① 참 성능이 모두 같을 때의 기각률 (명목 {ALPHA}, {REPS} 회)")
print(f"{'배치':>16}{'고전 F':>10}{'Welch':>10}")
res = {}
for name, ns in LAYOUTS.items():
    cf = wf = 0
    for _ in range(REPS):
        g = [RNG.normal(0.0, s, int(n)) for s, n in zip(SIG, ns)]
        if stats.f_oneway(*g).pvalue < ALPHA:
            cf += 1
        if welch_anova(g)[3] < ALPHA:
            wf += 1
    res[name] = (cf / REPS, wf / REPS)
    print(f"{name.replace(chr(10), ' '):>16}{res[name][0]:>10.3f}"
          f"{res[name][1]:>10.3f}")

# === ② 유효 자유도 ==================================================
RATIOS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
NS2 = LAYOUTS["흔들리는 쪽에 적게\n$n = (10, 5, 3)$"]
print(f"\n② 시드를 다 합쳐 {TOTAL} 개일 때의 유효 자유도 (배치 {tuple(NS2)})")
print(f"{'σ3/σ1':>8}{'고전 df2':>11}{'Welch df2':>12}")
df_cl = TOTAL - K
dfw_m = []
for r in RATIOS:
    s = np.array([1.0, np.sqrt(r), r])
    d = [welch_anova([RNG.normal(0, si, int(ni))
                      for si, ni in zip(s, NS2)])[2] for _ in range(4000)]
    dfw_m.append(float(np.median(d)))
    print(f"{r:>8.1f}{df_cl:>11}{dfw_m[-1]:>12.2f}")
dfw_m = np.array(dfw_m)

# === ③ 가장 좋은 시드를 적으면 ======================================
SEEDS = np.arange(1, 21)
print(f"\n③ 참 성능이 같을 때, 흔들리는 쪽이 「최고 시드」로 이길 확률")
print(f"{'시드 수':>8}" + "".join(f"{f'σ비 {r:.0f}':>12}" for r in (2, 4)))
win = {}
for r in (2, 4):
    win[r] = []
    for nseed in SEEDS:
        x = RNG.normal(0, 1.0, (REPS, nseed)).max(axis=1)
        y = RNG.normal(0, r, (REPS, nseed)).max(axis=1)
        win[r].append(float((y > x).mean()))
for i, nseed in enumerate(SEEDS):
    if nseed in (1, 3, 5, 10, 20):
        print(f"{nseed:>8}" + "".join(f"{win[r][i]:>12.3f}" for r in (2, 4)))

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

names = list(LAYOUTS)
xpos = np.arange(len(names))
cls = [res[n][0] for n in names]
wel = [res[n][1] for n in names]
ax1.bar(xpos - 0.2, cls, 0.38, color="#d62728", label="고전 $F$")
ax1.bar(xpos + 0.2, wel, 0.38, color="#1f77b4", label="Welch")
for x, v in zip(xpos - 0.2, cls):
    ax1.text(x, v + 0.004, f"${v:.3f}$", ha="center", fontsize=10.5,
             color="#d62728")
for x, v in zip(xpos + 0.2, wel):
    ax1.text(x, v + 0.004, f"${v:.3f}$", ha="center", fontsize=10.5,
             color="#1f77b4")
ax1.axhline(ALPHA, color="black", lw=1.5, ls=":")
ax1.text(1.55, ALPHA + 0.006, "명목 $\\alpha = 0.05$", fontsize=10)
ax1.set_xticks(xpos)
ax1.set_xticklabels(names, fontsize=9.5)
ax1.set_ylabel("참 성능이 같을 때의 기각률")
ax1.set_ylim(0, max(cls) * 1.25)
ax1.set_title("① 시드를 어디에 쓰느냐가 정한다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")

ax2.plot(RATIOS, dfw_m, "o-", color="#1f77b4", lw=2.6, ms=8,
         label="Welch 의 유효 자유도")
ax2.axhline(df_cl, color="#d62728", lw=2.2, ls="--",
            label=f"고전 $F$ 가 쓰는 자유도 (${df_cl}$)")
ax2.set_xlabel("가장 흔들리는 방법과 가장 안정된 방법의 표준편차 비")
ax2.set_ylabel("오차 자유도")
ax2.set_ylim(0, df_cl * 1.2)
ax2.set_title("② 시드 $18$ 개가 $18$ 개가 아니다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25)

for r, c in zip((2, 4), ("#ff7f0e", "#d62728")):
    ax3.plot(SEEDS, win[r], "-", color=c, lw=2.6,
             label=f"표준편차가 ${r}$ 배 흔들리는 방법")
ax3.axhline(0.5, color="black", lw=1.5, ls=":")
ax3.text(10.5, 0.515, "참 성능이 같으니 $0.5$ 여야 한다", fontsize=10)
ax3.set_xlabel("방법마다 돌린 시드 수")
ax3.set_ylabel("「최고 시드」로 이길 확률")
ax3.set_xticks([1, 5, 10, 15, 20])
ax3.set_xticklabels(["$1$", "$5$", "$10$", "$15$", "$20$"])
ax3.set_ylim(0.45, 1.0)
ax3.set_title("③ 더 돌릴수록 더 벌어진다", fontsize=12)
ax3.legend(loc="center right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/anova_welch/img/seed_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
