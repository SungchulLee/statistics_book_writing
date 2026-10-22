r"""「오차한계 $\pm 3.5\%$」가 맞는지, 선거 결과를 기다리지 않고 재는 법.

같은 선거를 잰 여론조사 $k$ 개가 있다. 각 조사가 말하는 표준오차는 단순무작위추출이
주는 $\sigma_0 = \sqrt{p(1-p)/n}$ 이다. 그러면 **조사들끼리의 흩어짐**이 $\sigma_0$
와 맞아야 한다 — 이 쪽의 카이제곱 검정이 그것을 본다.

$$\chi^2 = \frac{(k-1)s^2}{\sigma_0^2} \sim \chi^2(k-1)$$

① 참 흩어짐이 $\sigma_0$ 의 두 배면 조사 $10$ 개로도 거의 언제나 잡힌다.
② 그런데 **모든 조사가 같은 방향으로 틀린 몫**은 이 검정에 전혀 걸리지 않는다.
③ 그래서 「$95\%$ 오차한계」가 실제로 덮는 비율이 $68\%$ 로 내려간다.

만드는 파일: docs/ch15/chi_square_test/img/poll_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

NPOLL = 800                                   # 조사 하나의 응답자 수
SIG0 = 100 * np.sqrt(0.25 / NPOLL)            # SRS 표준오차 (퍼센트포인트)
MOE = stats.norm.isf(0.025) * SIG0            # 보고되는 95% 오차한계
KS = np.array([5, 8, 12, 20, 30, 50])
CS = np.array([1.0, 1.25, 1.5, 2.0, 2.5])     # 참 흩어짐이 SIG0 의 몇 배인가
BIASES = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 모든 조사가 공유하는 치우침
REPS = 20000
ALPHA = 0.05

print(f"응답자 {NPOLL} 명이면 SRS 표준오차 {SIG0:.2f} pp, "
      f"보고되는 오차한계 ±{MOE:.2f} pp")


def reject(k, c, bias=0.0, reps=REPS):
    """조사 k 개의 흩어짐이 SIG0 보다 큰지 — 상위꼬리 카이제곱 검정."""
    y = bias + RNG.normal(0.0, c * SIG0, (reps, k))
    s2 = y.var(axis=1, ddof=1)
    chi = (k - 1) * s2 / SIG0 ** 2
    return float((stats.chi2.sf(chi, k - 1) < ALPHA).mean())


print(f"\n① 참 흩어짐이 SRS 의 c 배일 때 잡아낼 확률 ({REPS} 회)")
print(f"{'조사 수 k':>10}" + "".join(f"{f'c = {c}':>10}" for c in CS))
grid = np.zeros((len(CS), len(KS)))
for j, k in enumerate(KS):
    row = f"{k:>10}"
    for i, c in enumerate(CS):
        grid[i, j] = reject(int(k), c)
        row += f"{grid[i, j]:>10.3f}"
    print(row)

KFIX = 20
print(f"\n② 공유된 치우침을 넣어도 달라지지 않는다 (k = {KFIX}, c = 1)")
print(f"{'치우침 b (pp)':>14}{'기각률':>10}")
rb = [reject(KFIX, 1.0, b) for b in BIASES]
for b, r in zip(BIASES, rb):
    print(f"{b:>14.1f}{r:>10.4f}")

# === ③ 보고된 오차한계가 실제로 덮는 비율 =============================
RMSE = 3.5          # 논문이 보고한 실제 평균오차 (pp)
ABS_BIAS = 2.0      # 논문이 보고한 선거 수준 절대편향의 평균 (pp)
tau = ABS_BIAS / np.sqrt(2 / np.pi)             # bias ~ N(0, tau) 로 보면
sig_v = np.sqrt(max(RMSE ** 2 - tau ** 2, 0))   # 남는 것이 조사 사이 흩어짐
print(f"\n③ 실제 오차를 평균 0, 표준편차 {RMSE} pp 로 보면")
print(f"   선거 수준 치우침 tau = {tau:.2f} pp, 조사 사이 흩어짐 {sig_v:.2f} pp")
cover_claim = 0.95
cover_true = float(2 * stats.norm.cdf(MOE / RMSE) - 1)
print(f"   보고된 ±{MOE:.2f} pp 가 실제로 덮는 비율 {cover_true:.3f}")
LEVELS = np.linspace(1.0, 4.0, 60)              # 실제 오차 표준편차를 바꿔 가며
cover = 2 * stats.norm.cdf(MOE / LEVELS) - 1

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = plt.cm.viridis(np.linspace(0.08, 0.86, len(CS)))

for i, (c, col) in enumerate(zip(CS, COL)):
    ax1.plot(KS, grid[i], "o-", color=col, lw=2.6, ms=8,
             label=f"참 흩어짐 $= {c}\\,\\sigma_0$")
ax1.axhline(ALPHA, color="black", lw=1.3, ls=":")
ax1.text(5.4, 0.085, "명목 $\\alpha = 0.05$", fontsize=10)
ax1.set_xticks(KS)
ax1.set_xticklabels([f"${v}$" for v in KS])
ax1.set_xlabel("같은 선거를 잰 조사의 수 $k$")
ax1.set_ylabel("「흩어짐이 너무 크다」고 할 확률")
ax1.set_ylim(0, 1.05)
ax1.set_title("① 선거 결과 없이도 알 수 있다", fontsize=12)
ax1.legend(loc="lower right", fontsize=9.5)
ax1.grid(alpha=0.25)

ax2.bar(np.arange(len(BIASES)), rb, 0.6, color="#1f77b4")
for i, r in enumerate(rb):
    ax2.text(i, r + 0.004, f"${r:.3f}$", ha="center", fontsize=11)
ax2.axhline(ALPHA, color="black", lw=1.4, ls=":")
ax2.text(4.45, 0.088, "점선은 명목 $0.05$", fontsize=10.5, ha="right")
ax2.set_xticks(np.arange(len(BIASES)))
ax2.set_xticklabels([f"${b:.0f}$" for b in BIASES])
ax2.set_xlabel("모든 조사가 공유하는 치우침 (퍼센트포인트)")
ax2.set_ylabel(f"기각률 ($k = {KFIX}$, 흩어짐은 제대로)")
ax2.set_ylim(0, 0.1)
ax2.set_title("② 그런데 이 검정은 치우침을 못 본다", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

ax3.plot(LEVELS, cover, color="#1f77b4", lw=2.8)
ax3.axhline(cover_claim, color="#2ca02c", lw=2.0, ls="--",
            label="보고되는 $95\\%$")
ax3.axvline(SIG0, color="#999999", lw=1.4, ls=":")
ax3.text(SIG0 + 0.05, 0.42, "SRS 가 말하는\n표준오차", fontsize=10,
         color="#555555")
ax3.axvline(RMSE, color="#d62728", lw=2.0, ls="-.")
ax3.plot(RMSE, cover_true, "o", color="#d62728", ms=11)
ax3.annotate(f"실제 평균오차 ${RMSE}$ pp\n덮는 비율 ${cover_true:.3f}$",
             xy=(RMSE, cover_true), xytext=(1.45, 0.52), fontsize=11,
             color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax3.set_xlabel("실제 오차의 표준편차 (퍼센트포인트)")
ax3.set_ylabel(f"$\\pm{MOE:.1f}$ pp 가 실제로 덮는 비율")
ax3.set_ylim(0.3, 1.02)
ax3.set_title("③ 「$95\\%$ 오차한계」가 덮는 것", fontsize=12)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch15/chi_square_test/img/poll_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
