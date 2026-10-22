r"""9.5절 「그림으로 보는 통계학」 — 짝이 지워 주는 것.

대응설계의 이득은 짝 안의 상관 $\rho$ 가 정한다. 같은 측정 횟수를 쓰면

    대응:  SD(D̄) = σ√(2(1-ρ)/n),  자유도 n-1
    독립:  SD(차) = σ√(2/n),       자유도 2n-2

이므로 $\rho$ 가 클수록 대응이 이기고, $\rho$ 가 $0$ 에 가까우면 **자유도 손실만 남아**
독립이 이긴다. 떨어져 자란 일란성 쌍둥이의 지능 상관($0.69$–$0.78$)을 그 자리에 놓고 본다.

① 같은 검정력에 필요한 표본 — ρ 에 따라.
② 검정력 곡선 — 쌍 20 에서 ρ 가 하는 일.
③ 짝이 지는 구간 — n 이 작을수록 넓다.

만드는 파일: docs/ch09/paired_sample_tests/img/twins_pairing.png
실행:  python3 scripts/make_ch09_twins_pairing.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats

OUT = "docs/ch09/paired_sample_tests/img/twins_pairing.png"
ALPHA, D = 0.05, 0.5        # 유의수준과 효과크기 (σ 단위)
RHOS = (0.69, 0.78)


def power_paired(n, rho, d=D, alpha=ALPHA):
    """쌍 n 개, 측정 2n 번."""
    if n < 2:
        return 0.0
    ncp = d * np.sqrt(n / (2 * (1 - rho)))
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return (stats.nct.sf(crit, n - 1, ncp) + stats.nct.cdf(-crit, n - 1, ncp))


def power_indep(n, d=D, alpha=ALPHA):
    """군당 n 명, 측정 2n 번."""
    if n < 2:
        return 0.0
    ncp = d * np.sqrt(n / 2)
    crit = stats.t.ppf(1 - alpha / 2, 2 * n - 2)
    return (stats.nct.sf(crit, 2 * n - 2, ncp) + stats.nct.cdf(-crit, 2 * n - 2, ncp))


def need(fn, target=0.80, hi=4000, **kw):
    for n in range(2, hi):
        if fn(n, **kw) >= target:
            return n
    return hi


n_ind = need(power_indep)
print(f"효과크기 {D}, α = {ALPHA}, 검정력 0.80 기준")
print(f"독립설계: 군당 {n_ind} 명 (측정 {2*n_ind} 번)")
grid_rho = np.linspace(-0.3, 0.95, 70)
n_pair = np.array([need(power_paired, rho=r) for r in grid_rho])
for r in (0.0, 0.3, 0.5, *RHOS, 0.9):
    print(f"  ρ = {r:+.2f}:  대응설계 {need(power_paired, rho=r):>4} 쌍 "
          f"(측정 {2*need(power_paired, rho=r):>4} 번)")

# --- ③ 짝이 지는 경계 ---------------------------------------------------
print("\n짝이 독립을 이기기 시작하는 ρ (같은 측정 횟수)")
cross = {}
for n in (5, 10, 20, 50, 100):
    lo, hi = -0.5, 0.95
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_paired(n, mid) < power_indep(n):
            lo = mid
        else:
            hi = mid
    cross[n] = (lo + hi) / 2
    print(f"   n = {n:>3}:  ρ* = {cross[n]:+.3f}  "
          f"(검정력 대응 {power_paired(n, cross[n]):.3f} = 독립 {power_indep(n):.3f})")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
C1, C2 = "#1f6f8b", "#c0532b"

ax = axes[0]
ax.plot(grid_rho, n_pair, color=C1, lw=2.2, label="대응설계 (쌍 수)")
ax.axhline(n_ind, color=C2, lw=1.8, ls="--", label=f"독립설계 (군당 {n_ind} 명)")
for r, ty in zip(RHOS, (52, 30)):
    nn = need(power_paired, rho=r)
    ax.plot([r], [nn], "o", color=C1, ms=8)
    ax.annotate(f"ρ = {r:.2f} → {nn} 쌍", xy=(r, nn), xytext=(-0.05, ty),
                color=C1, fontsize=9, arrowprops=dict(arrowstyle="->", color=C1))
ax.set_xlabel("짝 안의 상관 ρ")
ax.set_ylabel("검정력 0.80 에 필요한 수")
ax.set_title(f"① 같은 검정력을 사는 값 (d = {D})", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 140)

ax = axes[1]
ns = np.arange(3, 81)
for r, ls in ((0.0, ":"), (0.5, "--"), (0.69, "-")):
    ax.plot(ns, [power_paired(int(n), r) for n in ns], ls, color=C1, lw=2,
            label=f"대응  ρ = {r:.2f}")
ax.plot(ns, [power_indep(int(n)) for n in ns], color=C2, lw=2.2, label="독립")
ax.axhline(0.80, color="0.4", ls=":", lw=1.2)
ax.set_xlabel("쌍 수 (대응) 또는 군당 인원 (독립) — 측정은 둘 다 그 두 배")
ax.set_ylabel("검정력")
ax.set_title("② 같은 측정 횟수로 얻는 검정력", fontsize=11)
ax.legend(fontsize=9, loc="lower right")

ax = axes[2]
ks = sorted(cross)
ax.plot(ks, [cross[k] for k in ks], "o-", color=C2, lw=2.2)
ax.axhline(0, color="0.4", lw=1)
for k in ks:
    ax.annotate(f"{cross[k]:+.3f}", xy=(k, cross[k]), xytext=(k, cross[k] + 0.012),
                fontsize=9, color=C2, ha="center")
ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("쌍 수 n")
ax.set_ylabel("짝이 이기기 시작하는 ρ")
ax.set_title("③ 자유도 손실이 요구하는 최소 상관", fontsize=11)
ax.set_ylim(-0.02, 0.27)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
