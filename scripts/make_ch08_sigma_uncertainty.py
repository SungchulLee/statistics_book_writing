r"""다섯 번 돌려 적은 $\pm s$ 가 말하는 것 — $\sigma$ 자체의 구간.

이 쪽은 $\sigma^2$ 의 신뢰구간을 세운다. 그 구간이 **얼마나 넓은지**를 배수로 적어
보면, 반복 몇 번으로 적은 $\pm s$ 가 무엇을 보장하는지가 드러난다.

① $n$ 에 따른 $\sigma$ 의 $95\%$ 구간 — $n = 5$ 면 $[0.60\sigma,\ 2.87\sigma]$ 다.
② 그 구간은 비대칭이라 $\pm$ 라는 모양 자체가 맞지 않는다.
③ 독자가 읽는 것은 「다음 번도 이 안에」인데, $\bar x \pm 1.96s$ 는 그것이 아니다.

만드는 파일: docs/ch08/one_sample_intervals/img/sigma_uncertainty.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
ALPHA = 0.05

# === ① sigma 의 구간 ================================================
NS = [3, 5, 10, 20, 30, 50, 100, 1000]
print("① s 를 1 로 두었을 때 sigma 의 95% 구간 (배수)")
print(f"{'n':>7}{'아래':>9}{'위':>9}{'위/아래':>10}{'위 - 1':>9}{'1 - 아래':>10}")
LO, HI = [], []
for n in NS:
    d = n - 1
    lo = np.sqrt(d / stats.chi2.ppf(1 - ALPHA / 2, d))
    hi = np.sqrt(d / stats.chi2.ppf(ALPHA / 2, d))
    LO.append(lo)
    HI.append(hi)
    print(f"{n:>7}{lo:>9.3f}{hi:>9.3f}{hi / lo:>10.2f}"
          f"{hi - 1:>9.3f}{1 - lo:>10.3f}")
print("   n = 5 면 sigma 가 s 의 0.60 배일 수도 2.87 배일 수도 있다 — "
      "**거의 다섯 배 폭**이다.")
print("   위쪽이 아래쪽보다 훨씬 멀다. ± 로 적을 수 있는 모양이 아니다.")

# === ② 모의로 확인 ===================================================
REP = 200_000
print(f"\n② 모의로 포함률 확인 (반복 {REP:,}, 참 sigma = 1)")
print(f"{'n':>7}{'포함률':>10}")
for n in (3, 5, 10, 30):
    s = RNG.standard_normal((REP, n)).std(axis=1, ddof=1)
    d = n - 1
    lo = s * np.sqrt(d / stats.chi2.ppf(1 - ALPHA / 2, d))
    hi = s * np.sqrt(d / stats.chi2.ppf(ALPHA / 2, d))
    print(f"{n:>7}{float(np.mean((lo <= 1) & (1 <= hi))):>10.4f}")

# === ③ 「다음 번도 이 안에」 =========================================
NS3 = [3, 5, 10, 20, 50, 100]
REP3 = 400_000
print(f"\n③ 다음 한 번이 구간 안에 들어오는 비율 (반복 {REP3:,})")
print(f"{'n':>7}{'xbar +- 1.96 s':>18}{'xbar +- t s':>14}"
      f"{'xbar +- t s sqrt(1+1/n)':>26}")
C196, CT, CPI = [], [], []
for n in NS3:
    x = RNG.standard_normal((REP3, n))
    m = x.mean(1)
    s = x.std(1, ddof=1)
    nxt = RNG.standard_normal(REP3)
    t = stats.t.ppf(1 - ALPHA / 2, n - 1)
    C196.append(float(np.mean(np.abs(nxt - m) <= 1.96 * s)))
    CT.append(float(np.mean(np.abs(nxt - m) <= t * s)))
    CPI.append(float(np.mean(np.abs(nxt - m) <= t * s * np.sqrt(1 + 1 / n))))
    print(f"{n:>7}{C196[-1]:>18.4f}{CT[-1]:>14.4f}{CPI[-1]:>26.4f}")
print("   맨 오른쪽이 '다음 한 번'을 위한 올바른 구간이다. "
      "n 이 작을수록 1.96 을 쓰는 쪽과 크게 갈린다.")
print(f"   n = 5 에서 ±1.96s 는 {100 * C196[1]:.1f}% 만 담는다 — "
      f"독자가 95% 로 읽는 그 막대다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_L, C_H, C_T = "#1f77b4", "#d62728", "#2ca02c"

ys = np.arange(len(NS))
for i, n in enumerate(NS):
    ax1.plot([LO[i], HI[i]], [ys[i], ys[i]], color=C_H, lw=4, alpha=0.75)
    ax1.plot([1], [ys[i]], "|", color="#333333", ms=14, mew=2)
    ax1.text(HI[i] * 1.06, ys[i], f"${LO[i]:.2f}$–${HI[i]:.2f}$",
             va="center", fontsize=10, color=C_H)
ax1.axvline(1.0, color="#333333", lw=1.4, ls="--")
ax1.set_xscale("log")
ax1.set_yticks(ys)
ax1.set_yticklabels([f"$n = {n}$" for n in NS], fontsize=10.5)
ax1.set_xlabel("$\\sigma \\,/\\, s$")
ax1.set_xticks([0.5, 1, 2, 4])
ax1.set_xticklabels(["$0.5$", "$1$", "$2$", "$4$"])
ax1.minorticks_off()
ax1.set_xlim(0.4, 7.5)
ax1.invert_yaxis()
ax1.set_title("① $s$ 하나로 아는 $\\sigma$", fontsize=12)
ax1.grid(alpha=0.25, axis="x")

ax2.semilogx(NS, [h - 1 for h in HI], "o-", color=C_H, lw=2.6, ms=9,
             label="위쪽으로 벌어진 몫 (위 $-\\,1$)")
ax2.semilogx(NS, [1 - l for l in LO], "s-", color=C_L, lw=2.6, ms=9,
             label="아래쪽으로 벌어진 몫 ($1-$ 아래)")
ax2.set_xlabel("반복 횟수 $n$")
ax2.set_ylabel("$s$ 에서 벌어진 몫")
ax2.set_xticks([10, 100, 1000])
ax2.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$"])
ax2.minorticks_off()
ax2.set_title("② 위아래가 다르다 — $\\pm$ 가 아니다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.semilogx(NS3, C196, "o-", color=C_H, lw=2.6, ms=9,
             label="$\\bar x \\pm 1.96 s$")
ax3.semilogx(NS3, CT, "s-", color="#ff7f0e", lw=2.4, ms=8,
             label="$\\bar x \\pm t\\, s$")
ax3.semilogx(NS3, CPI, "^-", color=C_T, lw=2.6, ms=9,
             label="$\\bar x \\pm t\\, s\\sqrt{1+1/n}$")
ax3.axhline(0.95, color="#333333", lw=1.8, ls="--")
ax3.text(3.1, 0.956, "말한 $95\\%$", fontsize=10.5, color="#333333")
ax3.set_xlabel("반복 횟수 $n$")
ax3.set_ylabel("다음 한 번이 들어오는 비율")
ax3.set_xticks([3, 10, 100])
ax3.set_xticklabels(["$3$", "$10$", "$10^{2}$"])
ax3.minorticks_off()
ax3.set_ylim(0.6, 1.0)
ax3.set_title("③ 「다음 번도 이 안에」를 바란다면", fontsize=12)
ax3.legend(loc="lower right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/one_sample_intervals/img/sigma_uncertainty.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
