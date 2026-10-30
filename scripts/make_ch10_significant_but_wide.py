"""10.4절 피어슨 상관의 검정 — 「유의하다」와 「세다」는 다른 말이다.

학생 25 명에서 r = 0.45 를 얻으면 t = 2.417 이 임계값 2.069 를 넘어 유의한데,
같은 r 로 만든 rho 의 95% 구간은 (0.07, 0.72) 로 「거의 0」부터 「꽤 강함」까지
걸쳐 있다. 두 결과는 어긋나지 않는다 — 다른 것을 묻기 때문이다.

오른쪽은 그 구간의 폭이 r 이 아니라 n 에서 온다는 것을 보인다.
(이 쪽에 이미 있는 significance_vs_effect.png 는 「유의해지는 최소 상관」이라
다른 것을 그린다 — 겹치지 않는다.)
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/correlation_test/img/"
R, N = 0.45, 25
CL = 0.95


def fisher_ci(r, n, cl=CL):
    """피셔 z 변환으로 rho 의 신뢰구간."""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    half = stats.norm.ppf(0.5 + cl / 2) * se
    return np.tanh(z - half), np.tanh(z + half)


lo, hi = fisher_ci(R, N)
t_stat = R * np.sqrt(N - 2) / np.sqrt(1 - R**2)
t_crit = stats.t.ppf(0.975, N - 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.0),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ── (좌) 같은 r 에서 나온 두 결과 ────────────────────────────────────
BANDS = [(0.0, 0.3, "약함", "#eef4fb"), (0.3, 0.5, "중간", "#dbe9f7"),
         (0.5, 1.0, "강함", "#bfd8f0")]
for a, b, lab, c in BANDS:
    ax1.axvspan(a, b, color=c, zorder=0)
    ax1.text((a + b) / 2, 0.47, lab, ha="center", va="center", fontsize=10.5,
             color="0.45")

ax1.hlines(0, lo, hi, color="tab:blue", lw=9, alpha=0.75, zorder=3)
ax1.scatter([R], [0], s=170, c="tab:blue", zorder=5, edgecolor="white", lw=2)
ax1.axvline(0, color="tab:red", ls="--", lw=1.8, zorder=4)

# 축 왼쪽 끝이라 가운데 맞춤이면 잘린다
ax1.text(0.02, -0.30, "귀무가설 ρ = 0 — 구간 밖에 있다 → 유의하다", ha="left",
         va="top", fontsize=10.5, color="tab:red")
ax1.annotate(f"r = {R}", (R, 0), textcoords="offset points", xytext=(0, 20),
             ha="center", fontsize=12, fontweight="bold", color="tab:blue")
ax1.text(lo, 0.17, f"{lo:.2f}", ha="center", va="bottom", fontsize=11,
         color="tab:blue")
ax1.text(hi, 0.17, f"{hi:.2f}", ha="center", va="bottom", fontsize=11,
         color="tab:blue")
ax1.annotate("", xy=(lo, -0.14), xytext=(hi, -0.14),
             arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
ax1.text((lo + hi) / 2, -0.19, "구간은 「약함」부터 「강함」까지 걸쳐 있다",
         ha="center", va="top", fontsize=10.5, color="0.25")

ax1.set_xlim(-0.12, 1.0)
ax1.set_ylim(-0.55, 0.58)
ax1.set_yticks([])
ax1.set_xlabel("모상관 ρ")
ax1.set_title(f"학생 {N} 명, r = {R}   "
              f"(t = {t_stat:.3f} > 임계값 {t_crit:.3f})", fontsize=11.5)

# ── (우) 폭을 정하는 것은 r 이 아니라 n 이다 ─────────────────────────
ns = [25, 50, 100, 400]
ys = np.arange(len(ns))[::-1]
for yy, n in zip(ys, ns):
    a, b = fisher_ci(R, n)
    c = "tab:blue" if n == N else "0.55"
    ax2.hlines(yy, a, b, color=c, lw=7, alpha=0.75)
    ax2.scatter([R], [yy], s=70, c=c, zorder=4, edgecolor="white", lw=1.2)
    ax2.text(b + 0.02, yy, f"({a:.2f}, {b:.2f})   폭 {b - a:.2f}", va="center",
             ha="left", fontsize=10, color=c)

ax2.axvline(0, color="tab:red", ls="--", lw=1.5)
ax2.set_yticks(ys, [f"n = {n}" for n in ns], fontsize=11)
ax2.set_xlim(-0.08, 1.12)
ax2.set_ylim(-0.6, len(ns) - 0.4)
ax2.set_xlabel("모상관 ρ")
ax2.set_title(f"r 을 {R} 로 고정하고 n 만 바꾸면", fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "significant_but_wide.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"r = {R}, n = {N}:  t = {t_stat:.4f}, 임계값 = {t_crit:.4f}  "
      f"→ {'유의' if t_stat > t_crit else '유의하지 않음'}")
print(f"  95% 구간 = ({lo:.4f}, {hi:.4f})   (쪽: 약 (0.07, 0.72))")
for n in ns:
    a, b = fisher_ci(R, n)
    print(f"  n = {n:>4}: ({a:.3f}, {b:.3f})  폭 {b - a:.3f}")
