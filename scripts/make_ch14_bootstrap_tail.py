r"""붓스트랩이 꼬리로 갈수록 말을 못 하는 것을 잰다.

쪽의 $\S 5$ 는 "표본최댓값처럼 매끄럽지 않은 통계량에서는 표준 비모수 붓스트랩이
실패할 수 있다"고 한 줄로 단언한다. 그 한 줄을 수로 만든다. 그리고 **매끄러움은
있고 없고가 아니라 정도**라는 것을 본다 — 평균에서 최댓값으로 가는 길목에 꼬리
분위수들이 놓여 있고, 포함률은 그 길을 따라 연속적으로 무너진다.

① 로그정규 자료에서 백분위 붓스트랩 구간의 포함률을 $n$ 에 대해 잰다.
   평균과 $p_{50}$ 은 금방 $0.95$ 에 닿고 $p_{99.9}$ 는 멀찍이 떨어져 있다.
② 까닭 — 꼬리 분위수의 붓스트랩 분포는 관측값 몇 개 위에만 올라앉은 이산분포다.
③ 최댓값에서는 $n$ 을 $500$ 배 키워도 포함률이 한 값에 굳는다. 정확구간과 견준다.

만드는 파일: docs/ch14/non_normal_data/img/bootstrap_tail.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261022)

ALPHA = 0.05
QS = [0.50, 0.90, 0.99, 0.999]
NS_Q = [100, 200, 500, 1000, 2000, 5000]
NS_M = [20, 50, 100, 500, 2000, 10000]
REPS_Q, B_Q = 1000, 400
REPS_M, B_M = 2000, 800
N_SHOW, B_SHOW = 200, 4000

TRUE_MEAN = float(np.exp(0.5))                      # 로그정규(0,1) 의 평균
TRUE_Q = {q: float(stats.lognorm.ppf(q, 1.0)) for q in QS}
LAB = {0.50: "$p_{50}$", 0.90: "$p_{90}$",
       0.99: "$p_{99}$", 0.999: "$p_{99.9}$"}

print(f"로그정규$(0,1)$ — 참값 : 평균 {TRUE_MEAN:.4f}, " +
      ", ".join(f"p{100 * q:g} {v:.4f}" for q, v in TRUE_Q.items()))

# === ① 포함률 ========================================================
print(f"\n① 백분위 붓스트랩 구간의 포함률 (명목 {1 - ALPHA}, "
      f"되풀이 {REPS_Q}, 재표본 {B_Q})")
print("   n        평균      p50      p90      p99    p99.9")
cov = {k: [] for k in ["평균"] + QS}
for n in NS_Q:
    hit = {k: 0 for k in cov}
    for _ in range(REPS_Q):
        x = RNG.lognormal(0.0, 1.0, n)
        r = x[RNG.integers(0, n, (B_Q, n))]
        lo, hi = np.quantile(r.mean(axis=1), [ALPHA / 2, 1 - ALPHA / 2])
        hit["평균"] += int(lo <= TRUE_MEAN <= hi)
        for q in QS:
            lo, hi = np.quantile(np.quantile(r, q, axis=1),
                                 [ALPHA / 2, 1 - ALPHA / 2])
            hit[q] += int(lo <= TRUE_Q[q] <= hi)
    for k in cov:
        cov[k].append(hit[k] / REPS_Q)
    print(f"{n:>5}" + "".join(f"{cov[k][-1]:>9.3f}" for k in cov))

# === ② 붓스트랩 분포의 이산성 ========================================
print(f"\n② 붓스트랩 분포가 올라앉은 서로 다른 값 (n = {N_SHOW}, "
      f"재표본 {B_SHOW})")
x0 = RNG.lognormal(0.0, 1.0, N_SHOW)
r0 = x0[RNG.integers(0, N_SHOW, (B_SHOW, N_SHOW))]
boot = {}
for q in QS:
    s = np.quantile(r0, q, axis=1)
    u, c = np.unique(s, return_counts=True)
    boot[q] = (s, len(u), c.max() / B_SHOW)
    print(f"   {f'p{100 * q:g}':>7} : {len(u):>4} 개, "
          f"가장 흔한 값 하나가 {100 * c.max() / B_SHOW:>5.1f}%")
rank_lo = int(np.searchsorted(np.sort(x0), np.unique(boot[0.999][0])[0]))
print(f"   p99.9 의 붓스트랩 분포는 정렬한 {N_SHOW} 개 가운데 "
      f"상위 {N_SHOW - rank_lo} 개만 쓴다")

# === ③ 최댓값 ========================================================
print(f"\n③ 최댓값 — 균등$(0,1)$, $\\theta = 1$ (되풀이 {REPS_M})")
print(f"{'n':>7}{'붓스트랩':>11}{'정확구간':>10}{'P(max*=max)':>14}"
      f"{'1-(1-1/n)^n':>14}")
cov_b, cov_e, atoms = [], [], []
for n in NS_M:
    cb = ce = 0
    at = np.zeros(REPS_M)
    for i in range(REPS_M):
        x = RNG.random(n)
        th = x.max()
        s = x[RNG.integers(0, n, (B_M, n))].max(axis=1)
        at[i] = (s == th).mean()
        lo, hi = 2 * th - np.quantile(s, [1 - ALPHA / 2, ALPHA / 2])
        cb += int(lo <= 1.0 <= hi)
        ce += int(th <= 1.0 <= th / ALPHA ** (1 / n))   # 정확구간
    cov_b.append(cb / REPS_M)
    cov_e.append(ce / REPS_M)
    atoms.append(at.mean())
    print(f"{n:>7}{cov_b[-1]:>11.3f}{cov_e[-1]:>10.3f}{atoms[-1]:>14.3f}"
          f"{1 - (1 - 1 / n) ** n:>14.3f}")
flat = float(np.mean(cov_b[2:]))
print(f"   붓스트랩 쪽은 n 을 {NS_M[-1] // NS_M[0]} 배 키우는 동안 "
      f"{flat:.3f} 에 머문다 (MC SE "
      f"{np.sqrt(flat * (1 - flat) / (REPS_M * 4)):.4f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 4.9))
COL = {"평균": "#555555", 0.50: "#2ca02c", 0.90: "#1f77b4",
       0.99: "#ff7f0e", 0.999: "#d62728"}

for k in ["평균"] + QS:
    ax1.plot(NS_Q, cov[k], "o-", color=COL[k], lw=2.2, ms=6,
             label="평균" if k == "평균" else LAB[k])
ax1.axhline(1 - ALPHA, color="black", lw=1.4, ls="--")
ax1.text(NS_Q[0] * 1.05, 0.957, f"명목 ${1 - ALPHA}$", fontsize=10)
ax1.set_xscale("log")
ax1.set_xticks(NS_Q)
ax1.set_xticklabels([f"${v}$" for v in NS_Q], fontsize=9)
ax1.minorticks_off()
ax1.set_xlabel("표본 수 $n$")
ax1.set_ylabel("붓스트랩 구간의 포함률")
ax1.set_ylim(0.1, 1.02)
ax1.set_title("① 꼬리로 갈수록 무너진다", fontsize=12)
ax1.legend(loc="lower right", fontsize=9.5, ncol=2)
ax1.grid(alpha=0.25)

for q in QS:
    s, nu, top = boot[q]
    z = np.sort((s - s.mean()) / s.std())
    ax2.step(z, np.arange(1, len(z) + 1) / len(z), where="post",
             color=COL[q], lw=2.0,
             label=f"{LAB[q]} — 서로 다른 값 ${nu}$ 개")
ax2.text(0.035, 0.975,
         f"$n = {N_SHOW}$, 재표본 $"
         + f"{B_SHOW:,}".replace(",", "{,}") + "$ 번\n"
         f"{LAB[0.999]} 는 가장 흔한 값 하나가 "
         f"${100 * boot[0.999][2]:.1f}\\%$ 를 차지한다\n"
         f"(그림의 가장 큰 계단). 그 분포가 쓰는\n"
         f"관측값은 정렬한 ${N_SHOW}$ 개 가운데 상위 ${N_SHOW - rank_lo}$ 개뿐이다",
         transform=ax2.transAxes, va="top", fontsize=9.5,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax2.set_xlabel("붓스트랩 분포 (각자의 표준편차 단위)")
ax2.set_ylabel("누적비율")
ax2.set_xlim(-3.4, 3.4)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_ylim(0, 1.22)
ax2.set_title("② 꼬리 분위수는 이산분포가 된다", fontsize=12)
ax2.legend(loc="lower right", fontsize=9.5)
ax2.grid(alpha=0.25)

ax3.plot(NS_M, cov_b, "o-", color="#d62728", lw=2.6, ms=7,
         label="붓스트랩 기본구간")
ax3.plot(NS_M, cov_e, "s-", color="#2ca02c", lw=2.6, ms=7,
         label="정확구간 $[X_{(n)},\\ X_{(n)}/\\alpha^{1/n}]$")
ax3.plot(NS_M, atoms, "^:", color="#8c564b", lw=2.0, ms=7,
         label="$P(\\max^* = \\max)$")
ax3.axhline(1 - ALPHA, color="black", lw=1.4, ls="--")
ax3.axhline(1 - np.exp(-1), color="#8c564b", lw=1.0, ls=":")
ax3.text(NS_M[-1] * 0.72, 1 - np.exp(-1) + 0.022, "$1 - e^{-1}$",
         fontsize=10, color="#8c564b", ha="right")
ax3.annotate(f"${flat:.3f}$ 에 굳는다", xy=(NS_M[-3], cov_b[-3]),
             xytext=(26, 0.805), fontsize=10.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax3.set_xscale("log")
ax3.set_xticks(NS_M)
ax3.set_xticklabels([f"${v}$" for v in NS_M], fontsize=9)
ax3.minorticks_off()
ax3.set_xlabel("표본 수 $n$")
ax3.set_ylabel("포함률 · 확률")
ax3.set_ylim(0.55, 1.02)
ax3.set_title("③ 최댓값에서는 $n$ 이 듣지 않는다", fontsize=12)
ax3.legend(loc="lower center", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch14/non_normal_data/img/bootstrap_tail.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
