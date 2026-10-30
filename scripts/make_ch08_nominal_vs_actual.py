"""8.1절 포함확률 — 「명목 95%」를 단 절차들의 실제 포함률.

왼쪽: 비율 하나를 가두는 네 방법 (n = 40, p = 0.1). 이항은 유한하므로
      모의 없이 정확히 센다 — 쪽의 수와 자리까지 같아야 한다.
오른쪽: 평균의 t 구간과 분산의 카이제곱 구간을 네 분포에서 나란히.
      쪽 §6 의 모의 결과를 그대로 쓴다(아래 주석).
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/foundations/img/"
Z = 1.96
NOMINAL = 0.95


def exact_coverage(n, p, kind):
    """구간이 p^ 의 함수이므로 덮는 k 의 점질량을 더하면 포함률이 나온다."""
    k = np.arange(n + 1)
    ph = k / n
    pmf = stats.binom(n, p).pmf(k)

    if kind == "wald":
        half = Z * np.sqrt(ph * (1 - ph) / n)
        lo, hi = ph - half, ph + half
    elif kind == "wilson":
        c = 1 + Z**2 / n
        ctr = (ph + Z**2 / (2 * n)) / c
        half = Z * np.sqrt(ph * (1 - ph) / n + Z**2 / (4 * n**2)) / c
        lo, hi = ctr - half, ctr + half
    elif kind == "ac":
        nt = n + 4
        pt = (k + 2) / nt
        half = Z * np.sqrt(pt * (1 - pt) / nt)
        lo, hi = pt - half, pt + half
    elif kind == "cp":
        lo = np.where(k == 0, 0.0, stats.beta.ppf(0.025, k, n - k + 1))
        hi = np.where(k == n, 1.0, stats.beta.ppf(0.975, k + 1, n - k))
    return pmf[(lo <= p) & (p <= hi)].sum()


# ── 왼쪽 패널 ────────────────────────────────────────────────────────
PROPS = [("왈드 z", "wald"), ("윌슨", "wilson"),
         ("아그레스티–코울", "ac"), ("클로퍼–피어슨", "cp")]
prop_cov = [exact_coverage(40, 0.10, kind) for _, kind in PROPS]

# ── 오른쪽 패널 — 쪽 §6 의 모의 결과(되풀이 2 만 회) ─────────────────
# 분포 · 초과첨도 · t 구간 포함률 · 카이제곱 구간 포함률
DISTS = [("정규", 0.01, 0.951, 0.951),
         ("균등", -1.20, 0.949, 0.997),
         ("t(5)", 4.73, 0.953, 0.834),
         ("지수", 5.95, 0.923, 0.720)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.3),
                               gridspec_kw={"width_ratios": [1, 1.12]})

# ── (좌) 비율 네 방법 ────────────────────────────────────────────────
y = np.arange(len(PROPS))[::-1]
colors = ["tab:red" if c < NOMINAL else "tab:blue" for c in prop_cov]
ax1.hlines(y, NOMINAL, prop_cov, color=colors, lw=2.4, alpha=0.55)
ax1.scatter(prop_cov, y, s=110, c=colors, zorder=3)
ax1.axvline(NOMINAL, color="black", ls="--", lw=1.2)
ax1.text(NOMINAL + 0.0015, len(PROPS) - 0.42, "명목 0.95", ha="left",
         va="bottom", fontsize=9.5,
         bbox=dict(fc="white", ec="none", pad=1.5))

for yy, c in zip(y, prop_cov):
    off = 0.004 if c >= NOMINAL else -0.004
    ha = "left" if c >= NOMINAL else "right"
    ax1.text(c + off, yy, f"{c:.4f}", va="center", ha=ha, fontsize=10)

ax1.set_yticks(y, [n for n, _ in PROPS], fontsize=10.5)
ax1.set_xlim(0.900, 0.985)
ax1.set_ylim(-0.6, len(PROPS) - 0.1)
ax1.set_xlabel("실제 포함률")
ax1.set_title("비율 하나를 가두는 네 방법   (n = 40, p = 0.1, 정확 계산)",
              fontsize=11)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 평균의 t 구간 대 분산의 카이제곱 구간 ───────────────────────
yd = np.arange(len(DISTS))[::-1]
t_cov = [d[2] for d in DISTS]
c_cov = [d[3] for d in DISTS]

ax2.axvline(NOMINAL, color="black", ls="--", lw=1.2)
ax2.text(NOMINAL - 0.006, len(DISTS) - 0.42, "명목 0.95", ha="right",
         va="bottom", fontsize=9.5,
         bbox=dict(fc="white", ec="none", pad=1.5))
ax2.hlines(yd, t_cov, c_cov, color="0.75", lw=1.6, zorder=1)
ax2.scatter(t_cov, yd, s=105, c="tab:blue", zorder=3,
            label="평균의 t 구간")
ax2.scatter(c_cov, yd, s=105, c="tab:orange", marker="D", zorder=3,
            label="분산의 카이제곱 구간")

for yy, (name, kurt, tc, cc) in zip(yd, DISTS):
    if abs(cc - tc) < 0.005:        # 정규에서는 둘이 겹친다 — 꼬리표 하나만
        ax2.text(max(tc, cc) + 0.012, yy, f"둘 다 {tc:.3f}", va="center",
                 ha="left", fontsize=10, color="0.25")
        continue
    ax2.text(cc + (0.012 if cc > tc else -0.012), yy, f"{cc:.3f}",
             va="center", ha="left" if cc > tc else "right", fontsize=10,
             color="tab:orange")
    ax2.text(tc + (0.012 if tc > cc else -0.012), yy, f"{tc:.3f}",
             va="center", ha="left" if tc > cc else "right", fontsize=10,
             color="tab:blue")

ax2.set_yticks(yd, [f"{n}\n초과첨도 {k:.2f}" for n, k, _, _ in DISTS],
               fontsize=10)
ax2.set_xlim(0.655, 1.055)
ax2.set_ylim(-0.6, len(DISTS) - 0.1)
ax2.set_xlabel("실제 포함률")
ax2.set_title("같은 자료에 평균의 구간과 분산의 구간을 함께   (n = 20)",
              fontsize=11)
ax2.grid(axis="x", alpha=0.25)
ax2.legend(fontsize=9.5, loc="upper left", framealpha=0.95)

plt.tight_layout()
path = OUT + "nominal_vs_actual.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print("\n왼쪽 패널 — 정확 포함률 (쪽의 수와 맞아야 한다)")
for (name, _), c in zip(PROPS, prop_cov):
    print(f"  {name:16s} {c:.4f}")
print("\n오른쪽 패널 — 쪽 §6 에서 그대로 옮긴 모의 결과")
for name, k, tc, cc in DISTS:
    print(f"  {name:6s} 초과첨도 {k:>6.2f}   t {tc:.3f}   카이제곱 {cc:.3f}")
