"""10.3절 두 상관의 비교 — p = 0.335 가 허용하는 세상들.

들머리의 요점을 그린다. r1 = 0.72 (n1 = 100), r2 = 0.65 (n2 = 120) 에서
p = 0.3351 이 나왔다. 그 수를 「같다」로 읽으면 안 된다는 것을, 그 자료와
양립하는 세상을 모두 늘어놓아 보인다. rho2 를 0.65 로 고정하면 rho1 은
0.5639 에서 0.8264 까지 열려 있고, 그 구간은 0.65 를 **안쪽에** 담으므로
어느 쪽이 더 큰지조차 정해지지 않는다.

오른쪽은 그 열린 구간의 양 끝과 가운데를 참값으로 못 박고 표본을 다시
뽑은 것이다. 세 세상이 모두 관측된 차이 0.07 을 편하게 낸다.

쪽의 fisher_z_stabilize 는 r 의 표준편차가 rho 에 따라 무너지는 것을 다루고,
comparing_power 는 참값 0.65 와 0.40 을 고정한 검정력을 다룬다. 여기서는
그 둘이 건드리지 않는 것 --- 기각하지 못한 뒤에 남는 구간 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/correlation_test/img/"

R1, N1 = 0.72, 100
R2, N2 = 0.65, 120
Z95 = 1.96

z1, z2 = np.arctanh(R1), np.arctanh(R2)
se = np.sqrt(1 / (N1 - 3) + 1 / (N2 - 3))
zstat = (z1 - z2) / se
pval = 2 * stats.norm.sf(abs(zstat))
dlo, dhi = (z1 - z2) - Z95 * se, (z1 - z2) + Z95 * se
RHO_LO, RHO_HI = np.tanh(z2 + dlo), np.tanh(z2 + dhi)

OBS = R1 - R2                      # 관측된 차이 0.07
REPS = 200_000
rng = np.random.default_rng(1003)


def sample_r(rho, n, reps):
    """참 상관 rho 인 이변량정규에서 표본상관을 reps 개 뽑는다."""
    out = np.empty(reps)
    step = 20_000
    for s in range(0, reps, step):
        m = min(step, reps - s)
        a = rng.standard_normal((m, n))
        b = rng.standard_normal((m, n))
        y = rho * a + np.sqrt(1 - rho ** 2) * b
        a = a - a.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        out[s:s + m] = (a * y).sum(1) / np.sqrt((a * a).sum(1) * (y * y).sum(1))
    return out


WORLDS = [(RHO_LO, "ρ₁ 이 더 작은 세상", "tab:red"),
          (R2, "둘이 똑같은 세상", "0.35"),
          (RHO_HI, "ρ₁ 이 더 큰 세상", "tab:blue")]

diffs = [sample_r(rho, N1, REPS) - sample_r(R2, N2, REPS)
         for rho, _, _ in WORLDS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8),
                               gridspec_kw={"width_ratios": [1, 1.15]})

# ── (좌) 기각하지 못한 뒤에 남는 구간 ──────────────────────────────
ax1.axvspan(RHO_LO, R2, color="tab:red", alpha=0.12)
ax1.axvspan(R2, RHO_HI, color="tab:blue", alpha=0.12)
ax1.plot([RHO_LO, RHO_HI], [0, 0], "-", lw=9.0, color="tab:purple",
         solid_capstyle="butt")
for x in (RHO_LO, RHO_HI):
    ax1.plot([x, x], [-0.12, 0.12], "-", lw=4.0, color="tab:purple")
    ax1.text(x, -0.20, f"{x:.4f}", ha="center", va="top", fontsize=11.5,
             color="tab:purple", fontweight="bold")
ax1.axvline(R2, color="0.15", lw=2.2)
ax1.text(R2, 0.52, f"작은 쪽을 {R2} 로\n못 박았다", ha="center", va="bottom",
         fontsize=11, color="0.15", fontweight="bold")
ax1.plot([R1], [0], "o", ms=12, color="white", mec="0.15", mew=2.4, zorder=5)
ax1.text(R1, 0.17, f"관측된 {R1}", ha="center", va="bottom", fontsize=10.5,
         color="0.35")
ax1.text((RHO_LO + R2) / 2, 0.30, "더 작다", ha="center", fontsize=11.5,
         color="tab:red", fontweight="bold")
ax1.text((R2 + RHO_HI) / 2, 0.30, "더 크다", ha="center", fontsize=11.5,
         color="tab:blue", fontweight="bold")
ax1.set_xlim(0.46, 0.93)
ax1.set_ylim(-0.95, 0.80)
ax1.set_yticks([])
ax1.set_xlabel("큰 쪽의 참 상관 ρ₁", fontsize=11)
ax1.set_title(f"$p = {pval:.4f}$ 이 지우지 못한 ρ₁ 의 범위", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)
ax1.text(0.5, 0.022, "구간이 0.65 를 안쪽에 담는다\n— 어느 쪽이 큰지조차 "
         "정해지지 않는다", transform=ax1.transAxes, ha="center", va="bottom",
         fontsize=11.5, fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

# ── (우) 그 세 세상이 모두 같은 자료를 낸다 ─────────────────────────
bins = np.linspace(-0.35, 0.45, 160)
for d, (rho, name, col) in zip(diffs, WORLDS):
    ax2.hist(d, bins=bins, density=True, histtype="step", lw=2.2, color=col,
             label=f"{name} (ρ₁ = {rho:.4f})")
ax2.axvline(OBS, color="tab:green", lw=2.6)
ax2.set_xlim(-0.36, 0.42)
ax2.annotate(f"관측된 차이 {OBS:.2f}", xy=(OBS, ax2.get_ylim()[1] * 0.62),
             xytext=(0.47, 0.97), textcoords="axes fraction", ha="center",
             va="top", fontsize=11, color="tab:green", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       alpha=0.94, edgecolor="tab:green", lw=1.0),
             arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.5))
ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax2.set_xlabel("표본상관의 차이 r₁ 빼기 r₂", fontsize=11)
ax2.set_ylabel("밀도", fontsize=11)
ax2.set_title(f"세 세상에서 표본을 {REPS:,} 번 다시 뽑으면", fontsize=12.5)
ax2.grid(alpha=0.25)
lines = []
for d, (rho, name, col) in zip(diffs, WORLDS):
    side = "이상이" if rho <= R2 else "이하가"
    pr = (d >= OBS).mean() if rho <= R2 else (d <= OBS).mean()
    lines.append(f"{name} — 차가 {OBS:.2f} {side} {100 * pr:.1f}%")
ax2.text(0.985, 0.035, "\n".join(lines), transform=ax2.transAxes, ha="right",
         va="bottom", fontsize=10.5, fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

fig.suptitle("「유의하지 않다」는 「같다」가 아니다 — 세 세상이 남아 있다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "not_significant_not_same.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nr1 = {R1} (n = {N1}), r2 = {R2} (n = {N2})")
print(f"  z1 = {z1:.6f}, z2 = {z2:.6f}, 차이 = {z1 - z2:.6f}, SE = {se:.6f}")
print(f"  Z = {zstat:.4f}, p = {pval:.4f}")
print(f"  zeta 차의 95% 구간 ({dlo:+.4f}, {dhi:+.4f}) · 폭 {dhi - dlo:.4f}")
print(f"  rho2 = {R2} 로 고정했을 때 rho1 의 범위 "
      f"({RHO_LO:.4f}, {RHO_HI:.4f})")
print(f"\n관측된 차이 {OBS:.4f} 를 세 세상이 내는 확률 (각 {REPS:,} 번)")
for d, (rho, name, _) in zip(diffs, WORLDS):
    print(f"  {name:<16s} rho1 = {rho:.4f}  "
          f"P(차 >= {OBS:.2f}) = {(d >= OBS).mean():.4f}  "
          f"P(차 <= {OBS:.2f}) = {(d <= OBS).mean():.4f}")
