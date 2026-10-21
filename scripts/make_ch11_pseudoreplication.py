r"""분산분석의 $n$ 은 관측값의 수가 아니라 **실험단위**의 수다.

처리마다 수조를 $m$ 개 두고 수조마다 물고기를 $n$ 마리 잰다. 모형은

$$y_{ijl} = \mu + \tau_i + b_{ij} + e_{ijl},\qquad
  b \sim N(0, \sigma_b^2),\ e \sim N(0, \sigma_w^2)$$

이고 $H_0: \tau_i = 0$ 이다. 물고기 $kmn$ 마리를 그냥 일원분산분석에 넣는 것이
허버트(1984)가 말한 유사반복이다.

① 물고기를 더 재면 기각률이 **올라간다.** 수조 평균으로 하면 $0.05$ 에 붙어 있다.
② 왜인가 — $F$ 의 기댓값이 설계효과 $1 + (n-1)\varrho$ 만큼 부풀기 때문이다.
③ 그리고 정밀도에는 수조 수가 정하는 바닥 $\sigma_b/\sqrt{m}$ 이 있다.

만드는 파일: docs/ch11/assumptions/img/pseudoreplication.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

K = 3                       # 처리 수
SB2, SW2 = 0.25, 1.0        # 수조 사이 분산, 수조 안 분산
RHO = SB2 / (SB2 + SW2)     # 급내상관
NS = np.array([1, 2, 5, 10, 20, 50])     # 수조마다 잰 물고기 수
MS = (1, 2, 4, 8)                        # 처리마다 둔 수조 수
REPS = 4000
ALPHA = 0.05

print(f"수조 사이 분산 {SB2}, 수조 안 분산 {SW2}  ->  급내상관 ρ = {RHO:.3f}")


def draw(m, n):
    """H0 아래에서 (K, m, n) 짜리 자료를 한 벌 뽑는다."""
    b = RNG.normal(0.0, np.sqrt(SB2), size=(K, m, 1))
    e = RNG.normal(0.0, np.sqrt(SW2), size=(K, m, n))
    return b + e


def f_oneway(groups):
    """행이 처리, 열이 관측인 (K, q) 배열에 일원분산분석을 돌린다.

    q = 1 이면 오차 자유도가 0 이어서 검정이 성립하지 않는다.
    """
    k, q = groups.shape
    if q < 2:
        return np.nan, np.nan, np.nan
    gm = groups.mean()
    msb = q * ((groups.mean(axis=1) - gm) ** 2).sum() / (k - 1)
    msw = ((groups - groups.mean(axis=1, keepdims=True)) ** 2).sum() / (k * (q - 1))
    f = msb / msw
    return msb, msw, stats.f.sf(f, k - 1, k * (q - 1))


def ms_ratio(m, n):
    """E[MSB]/E[MSW] — 유사반복한 F 가 향하는 값."""
    msb = n * SB2 + SW2
    msw = SW2 + SB2 * n * (m - 1) / (m * n - 1) if m * n > 1 else np.nan
    return msb / msw


# === ① 기각률 ========================================================
print(f"\n① H0 아래 기각률 (명목 {ALPHA}, {REPS} 회)")
print(f"{'수조 m':>7}" + "".join(f"{f'물고기 {n}':>12}" for n in NS))
naive = {m: [] for m in MS}
corr = {m: [] for m in MS}
rbar = {m: [] for m in MS}
for m in MS:
    rown, rowc = "", ""
    for n in NS:
        y = np.array([draw(m, n) for _ in range(REPS)])        # (REPS, K, m, n)
        # 유사반복 — 물고기를 그대로 다 넣는다
        st = np.array([f_oneway(s.reshape(K, m * n)) for s in y])
        naive[m].append(float((st[:, 2] < ALPHA).mean()) if m * n > 1 else np.nan)
        rbar[m].append(float(st[:, 0].mean() / st[:, 1].mean())
                       if m * n > 1 else np.nan)
        # 올바른 검정 — 수조 평균이 관측이다 (m = 1 이면 오차 자유도가 0)
        if m > 1:
            qv = np.array([f_oneway(s.mean(axis=2))[2] for s in y])
            corr[m].append(float((qv < ALPHA).mean()))
        rown += f"{naive[m][-1]:>12.3f}" if m * n > 1 else f"{'—':>12}"
        rowc += f"{corr[m][-1]:>12.3f}" if m > 1 else f"{'—':>12}"
    print(f"{m:>7} 유사반복{rown}")
    print(f"{'':>7} 수조평균{rowc}")

# === ② 분모가 틀렸다 =================================================
print(f"\n② E[MSB]/E[MSW] 와 실제 평균비, 그리고 설계효과 1+(n-1)ρ")
print(f"{'물고기 n':>9}{'이론 비':>11}{'실제 비':>11}{'1+(n-1)ρ':>12}")
deff = 1 + (NS - 1) * RHO
theo = np.array([ms_ratio(MS[-1], n) for n in NS])
for n, t, r, d in zip(NS, theo, rbar[MS[-1]], deff):
    print(f"{n:>9}{t:>11.2f}{r:>11.2f}{d:>12.2f}")

# === ③ 정밀도의 바닥 =================================================
NG = np.unique(np.concatenate([NS, np.logspace(0, 3, 40).astype(int)]))
print(f"\n③ 처리평균의 표준오차 (바닥은 σ_b/√m)")
se = {}
for m in MS:
    se[m] = np.sqrt(SB2 / m + SW2 / (m * NG))
    floor = np.sqrt(SB2 / m)
    i10 = int(np.argmin(np.abs(NG - 10)))
    print(f"  m = {m}: n=1 에서 {se[m][0]:.3f}, n=10 에서 {se[m][i10]:.3f}, "
          f"n=1000 에서 {se[m][-1]:.3f}, 바닥 {floor:.3f}")
se_floor_m1 = np.sqrt(SB2)
se_m2_n10 = np.sqrt(SB2 / 2 + SW2 / 20)
print(f"  수조 1 개에 물고기 1000 마리 ({se[1][-1]:.3f}) 가 "
      f"수조 2 개에 10 마리 ({se_m2_n10:.3f}) 보다 나쁘다")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = {1: "#8c564b", 2: "#d62728", 4: "#ff7f0e", 8: "#1f77b4"}

for m in MS:
    ax1.plot(NS, naive[m], "o-", color=COL[m], lw=2.5, ms=7,
             label=f"유사반복, 수조 ${m}$ 개")
for m in MS[1:]:
    ax1.plot(NS, corr[m], "s--", color=COL[m], lw=1.4, ms=5, alpha=0.75)
ax1.axhline(ALPHA, color="black", lw=1.6, ls=":")
ax1.text(1.05, 0.013, "명목 $\\alpha = 0.05$", fontsize=10)
ax1.text(26, 0.12, "수조 평균으로\n한 검정 (점선)", fontsize=9.5, ha="center",
         color="#555555")
ax1.set_xscale("log")
ax1.set_xticks(NS)
ax1.set_xticklabels([f"${v}$" for v in NS], fontsize=9.5)
ax1.minorticks_off()
ax1.set_xlabel("수조마다 잰 물고기 수 $n$")
ax1.set_ylabel("$H_0$ 아래 기각률")
ax1.set_ylim(0, 0.88)
ax1.set_title("① 더 재면 더 틀린다", fontsize=12)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.25)

ax2.plot(NS, theo, "-", color="#333333", lw=2.4,
         label="이론값 $E[\\mathrm{MSB}]/E[\\mathrm{MSW}]$")
ax2.plot(NS, deff, color="#9467bd", lw=1.6, ls="--",
         label="설계효과 $1+(n-1)\\varrho$")
ax2.plot(NS, rbar[MS[-1]], "o", color="#1f77b4", ms=9,
         label=f"실제 평균비 (수조 ${MS[-1]}$ 개)")
ax2.axhline(1.0, color="black", lw=1.2, ls=":")
ax2.text(1.15, 1.9, "$H_0$ 가 참이면 $1$ 이어야 한다", fontsize=10)
ax2.set_xscale("log")
ax2.set_xticks(NS)
ax2.set_xticklabels([f"${v}$" for v in NS], fontsize=9.5)
ax2.minorticks_off()
ax2.set_xlabel("수조마다 잰 물고기 수 $n$")
ax2.set_ylabel("$F$ 의 크기")
ax2.set_title("② 분모가 재는 것이 틀렸다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

for m in MS:
    ax3.loglog(NG, se[m], "-", color=COL[m], lw=2.5,
               label=f"수조 ${m}$ 개")
    ax3.axhline(np.sqrt(SB2 / m), color=COL[m], lw=1.1, ls=":")
ax3.annotate(f"수조 $1$ 개의 바닥\n${se_floor_m1:.2f}$",
             xy=(260, se_floor_m1), xytext=(14, 0.62), fontsize=10,
             color="#8c564b",
             arrowprops=dict(arrowstyle="->", color="#8c564b", lw=1.0))
ax3.set_xticks([1, 10, 100, 1000])
ax3.set_xticklabels(["$1$", "$10$", "$100$", "$1{,}000$"], fontsize=9.5)
ax3.set_yticks([0.1, 0.3, 1.0])
ax3.set_yticklabels(["$0.1$", "$0.3$", "$1$"])
ax3.minorticks_off()
ax3.set_xlabel("수조마다 잰 물고기 수 $n$")
ax3.set_ylabel("처리평균의 표준오차")
ax3.set_title("③ 바닥은 수조 수가 정한다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch11/assumptions/img/pseudoreplication.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
