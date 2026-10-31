"""8.2절 σ²의 신뢰구간 — 길이는 자료가 정하지 않는다.

들머리의 문장 하나를 그린다. 분산의 카이제곱 구간은 상한이 하한의 몇
배인지가 **자료를 보기 전에** 정해져 있다 --- 그 비가 자유도만으로
결정되기 때문이다. n = 12 에서 21.920/3.816 = 5.744 배이고, n = 100 에서도
1.751 배로밖에 줄지 않는다.

왼쪽은 그것을 로그 축에서 보인다. 표본마다 s 가 달라 구간의 **자리**는
옮겨 가지만 **로그 자에서의 길이**는 모두 같다. 오른쪽은 그 비가 n 에
따라 줄어드는 꼴이다.

쪽의 2 절 그림 ci_sigma2_103 은 선형 축에 구간 100 개를 쌓아 포함률을
보이는 것이고, 상자의 sigma_uncertainty 는 sigma/s 구간과 비대칭과
다음 한 번을 다룬다. 여기서는 그 둘이 건드리지 않는 것만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/one_sample_intervals/img/"

N0 = 12
SIGMA2 = 4.0           # 참 분산
NS = np.arange(3, 301)


def ratio(n):
    """상한 / 하한. 자유도만으로 정해진다."""
    df = n - 1
    return stats.chi2.ppf(0.975, df) / stats.chi2.ppf(0.025, df)


def interval(s2, n=N0):
    df = n - 1
    return (df * s2 / stats.chi2.ppf(0.975, df),
            df * s2 / stats.chi2.ppf(0.025, df))


rng = np.random.default_rng(812)
s2s = np.array([(rng.normal(0, np.sqrt(SIGMA2), N0)).var(ddof=1)
                for _ in range(400)])
pick = [int(np.argmin(np.abs(s2s - q)))
        for q in np.quantile(s2s, [0.08, 0.5, 0.92])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) 자리는 옮겨 가도 로그 길이는 같다 ──────────────────────────
for j, idx in enumerate(pick):
    s2 = s2s[idx]
    lo, hi = interval(s2)
    y = 2 - j
    ax1.plot([lo, hi], [y, y], "-", lw=6.0, color="tab:blue",
             solid_capstyle="butt")
    for x in (lo, hi):
        ax1.plot([x, x], [y - 0.13, y + 0.13], "-", lw=3.0, color="tab:blue")
    ax1.plot([s2], [y], "o", ms=8, color="white", mec="tab:blue", mew=2.2)
    ax1.text(lo * 0.92, y, f"$s^2$ = {s2:.3f}", ha="right", va="center",
             fontsize=10.5, color="0.3")
    ax1.text(hi * 1.08, y, f"상한 ÷ 하한 = {hi / lo:.3f}", ha="left",
             va="center", fontsize=10.5, color="tab:blue", fontweight="bold")
ax1.axvline(SIGMA2, color="tab:red", lw=2.0)
ax1.text(SIGMA2 * 1.04, 2.62, f"참 분산 {SIGMA2:.0f}", fontsize=11,
         color="tab:red", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xlim(0.28, 120)
ax1.set_ylim(-0.75, 2.95)
ax1.set_yticks([])
ax1.set_xlabel("분산 (로그 눈금)", fontsize=11)
ax1.set_title(f"같은 n = {N0}, 다른 표본 셋 — 로그 자에서 길이가 같다",
              fontsize=12.5)
ax1.grid(axis="x", alpha=0.25, which="both")
ax1.text(0.5, 0.035, "자료가 정하는 것은 **자리**뿐이다".replace("**", ""),
         transform=ax1.transAxes, ha="center", va="bottom", fontsize=11,
         fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92,
                   edgecolor="0.5", lw=1.0))

# ── (우) 그 비는 n 만으로 정해진다 ──────────────────────────────────
r = ratio(NS)
ax2.plot(NS, r, color="tab:purple", lw=2.4)
ax2.axhline(2.0, color="0.4", ls="--", lw=1.5)
need2 = int(NS[r < 2.0][0])
ax2.text(300, 2.06, "두 배", ha="right", fontsize=10.5, color="0.4")
for n, c in ((12, "tab:blue"), (100, "tab:orange")):
    ax2.plot([n], [ratio(n)], "o", ms=9, color=c)
    ax2.annotate(f"n = {n}\n{ratio(n):.3f} 배", xy=(n, ratio(n)),
                 xytext=(n + 18, ratio(n) + (1.6 if n == 12 else 1.0)),
                 fontsize=10.5, color=c, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
ax2.plot([need2], [ratio(need2)], "o", ms=8, color="0.3")
ax2.annotate(f"두 배 아래로 내려가는 데\n{need2} 개가 든다",
             xy=(need2, 2.0), xytext=(need2 + 30, 3.6), fontsize=10.5,
             color="0.2", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
ax2.set_xlim(0, 305)
ax2.set_ylim(1.0, 9.0)
ax2.set_xlabel("표본 크기 n", fontsize=11)
ax2.set_ylabel("상한 ÷ 하한", fontsize=11)
ax2.set_title("그 비는 자유도만으로 정해진다", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("분산의 구간은 자료를 보기 전에 폭이 정해져 있다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "sigma_ratio_fixed.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N0}: chi2 임계값 {stats.chi2.ppf(0.025, N0-1):.3f} 과 "
      f"{stats.chi2.ppf(0.975, N0-1):.3f}  →  비 {ratio(N0):.4f}")
print(f"n = 100: 비 {ratio(100):.4f}")
for idx in pick:
    s2 = s2s[idx]; lo, hi = interval(s2)
    print(f"  s^2 = {s2:.4f}  구간 ({lo:.4f}, {hi:.4f})  비 {hi/lo:.4f}")
print(f"비가 두 배 아래로 내려가는 첫 n = {need2} (비 {ratio(need2):.4f})")
