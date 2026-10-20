r"""한 자료, 두 가지 산출물.

피셔의 유의성 검정은 귀무가설 **하나**를 놓고 자료가 그것과 얼마나 안 맞는지를
$p$ 값 하나로 적는다. 네이만–피어슨의 가설검정은 가설 **둘**을 놓고 **규칙** 하나를
만들며, 그 규칙에 붙는 성질이 $(\alpha,\ 1-\beta)$ 짝이다.

① 피셔가 내놓는 것 — $p$ 값의 분포. 귀무가설 아래에서 균등하다.
② 네이만–피어슨이 내놓는 것 — 규칙 하나와 그 규칙의 검정력 곡선.
③ 「가장 강력한 검정」이 늘 하나로 정해지지는 않는다 — 양측에는 UMP 가 없다.

만드는 파일: docs/ch09/foundations/img/fisher_vs_np.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N, SIG = 25, 1.0
SE = SIG / np.sqrt(N)
ALPHA = 0.05
MU1 = 0.3                      # ① 에서 쓸 대립가설
M = 400_000


def pvals(mu):
    """z 검정의 양측 p 값을 M 번."""
    z = (RNG.normal(mu, SE, M)) / SE
    return 2 * stats.norm.sf(np.abs(z))


p0, p1 = pvals(0.0), pvals(MU1)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

# ---- ① 피셔 -------------------------------------------------------
bins = np.linspace(0, 1, 51)
ax1.hist(p0, bins=bins, density=True, color="#9e9e9e", alpha=0.85,
         edgecolor="none", label="$H_0$ 가 참일 때")
ax1.hist(p1, bins=bins, density=True, color="#d62728", alpha=0.55,
         edgecolor="none", label=f"$\\mu = {MU1}$ 일 때")
ax1.axhline(1.0, color="black", lw=1.5, ls="--")
ax1.axvline(ALPHA, color="#1f77b4", lw=1.8)
ax1.text(ALPHA + 0.02, 4.5, f"$p = {ALPHA}$", fontsize=10, color="#1f77b4")
a = np.mean((p0 >= .04) & (p0 < .05)) * 100
b = np.mean((p0 >= .05) & (p0 < .06)) * 100
ax1.annotate(f"$H_0$ 아래에서는 평평하다 —\n$0.04$–$0.05$ 에 들 확률 ${a:.2f}\\%$,\n"
             f"$0.05$–$0.06$ 에 들 확률 ${b:.2f}\\%$.\n"
             "$0.05$ 에 자연의 경계는 없다",
             xy=(0.60, 1.0), xytext=(0.30, 2.1), fontsize=10,
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax1.set_xlabel("$p$ 값")
ax1.set_ylabel("밀도")
ax1.set_title("① 피셔 — 자료 하나, $p$ 하나", fontsize=12)
ax1.set_ylim(0, 5.2)
ax1.legend(loc="upper right", fontsize=9.5)
ax1.grid(alpha=0.25)

# ---- ② 네이만–피어슨 -----------------------------------------------
mus = np.linspace(-0.8, 0.8, 400)
crit = stats.norm.isf(ALPHA / 2)
pow2 = (stats.norm.sf(crit - mus / SE) + stats.norm.cdf(-crit - mus / SE))
ax2.plot(mus, pow2 * 100, color="#1f77b4", lw=2.8, label="규칙: $|Z| > 1.96$")
ax2.axhline(ALPHA * 100, color="black", lw=1.5, ls="--")
ax2.text(-0.78, ALPHA * 100 + 2.5, f"$\\alpha = {ALPHA}$", fontsize=10)
ax2.axvline(0, color="#999999", lw=1.0, ls=":")
for m, c in ((MU1, "#d62728"),):
    p_at = stats.norm.sf(crit - m / SE) + stats.norm.cdf(-crit - m / SE)
    ax2.plot([m], [p_at * 100], "o", color=c, ms=9, zorder=5)
    ax2.annotate(f"$\\mu = {m}$ 에서\n검정력 ${p_at * 100:.0f}\\%$",
                 xy=(m, p_at * 100), xytext=(0.33, 52), fontsize=10, color=c,
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.0))
ax2.set_xlabel("참 평균 $\\mu$")
ax2.set_ylabel("기각할 확률 ($\\%$)")
ax2.set_title(f"② 네이만–피어슨 — 규칙 하나, 짝 하나 ($n = {N}$)", fontsize=12)
ax2.set_ylim(0, 103)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

# ---- ③ UMP 가 없다 -------------------------------------------------
c1 = stats.norm.isf(ALPHA)
pow_up = stats.norm.sf(c1 - mus / SE)                      # mu>0 에 최적
pow_dn = stats.norm.cdf(-c1 - mus / SE)                    # mu<0 에 최적
ax3.plot(mus, pow_up * 100, color="#d62728", lw=2.6,
         label="오른쪽에 최적 ($Z > 1.645$)")
ax3.plot(mus, pow_dn * 100, color="#2ca02c", lw=2.6,
         label="왼쪽에 최적 ($Z < -1.645$)")
ax3.plot(mus, pow2 * 100, color="#1f77b4", lw=2.6, ls="--",
         label="양측 ($|Z| > 1.96$)")
ax3.axhline(ALPHA * 100, color="black", lw=1.3, ls=":")
ax3.set_xlabel("참 평균 $\\mu$")
ax3.set_ylabel("기각할 확률 ($\\%$)")
ax3.set_title("③ 양측 문제에는 가장 강력한 검정이 없다", fontsize=12)
ax3.set_ylim(0, 103)
ax3.legend(loc="upper center", fontsize=9.5)
ax3.grid(alpha=0.25)
ax3.annotate("어느 곡선도 다른 둘을\n모든 $\\mu$ 에서 이기지 못한다",
             xy=(-0.45, stats.norm.cdf(-c1 + 0.45 / SE) * 100),
             xytext=(-0.76, 42), fontsize=9.5, ha="left",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

fig.tight_layout()
out = "docs/ch09/foundations/img/fisher_vs_np.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① $H_0$ 아래 p 값 — 평균 {p0.mean():.4f} (균등이면 0.5), "
      f"0.05 미만 비율 {np.mean(p0 < 0.05) * 100:.2f}%")
print(f"   0.04–0.05 에 드는 비율 {np.mean((p0 >= .04) & (p0 < .05)) * 100:.2f}%, "
      f"0.05–0.06 은 {np.mean((p0 >= .05) & (p0 < .06)) * 100:.2f}% — 같다")
print(f"   mu = {MU1} 일 때 0.05 미만 비율 {np.mean(p1 < 0.05) * 100:.1f}% "
      f"(= 검정력)")

print(f"\n② 규칙 |Z| > {crit:.2f} 의 검정력")
print(f"{'mu':>8}{'검정력':>10}")
for m in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6):
    pw = stats.norm.sf(crit - m / SE) + stats.norm.cdf(-crit - m / SE)
    print(f"{m:>8.1f}{pw * 100:>9.1f}%")

print(f"\n③ 세 규칙의 검정력 (%)")
print(f"{'mu':>8}{'오른쪽 최적':>12}{'왼쪽 최적':>12}{'양측':>10}")
for m in (-0.4, -0.2, 0.0, 0.2, 0.4):
    a = stats.norm.sf(c1 - m / SE) * 100
    b = stats.norm.cdf(-c1 - m / SE) * 100
    c = (stats.norm.sf(crit - m / SE) + stats.norm.cdf(-crit - m / SE)) * 100
    print(f"{m:>8.1f}{a:>11.1f}%{b:>11.1f}%{c:>9.1f}%")
print("   어느 하나도 모든 mu 에서 가장 크지 않다 — 그것이 UMP 가 없다는 뜻이다")
