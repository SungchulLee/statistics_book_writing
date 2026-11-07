"""8.2절 σ²의 신뢰구간 — 길이는 자료가 정하지 않는다.

들머리의 주장을 그린다. 분산의 카이제곱 구간은 상한이 하한의 몇 배인지가
**자료를 보기 전에** 정해져 있다 --- 그 비가 자유도만으로 결정되기
때문이다. 들머리의 이야기인 **씨앗 다섯**에서 23.003 배이고, 쪽의 보기인
n = 12 에서 5.745 배, n = 100 에서도 1.751 배로밖에 줄지 않는다.

왼쪽은 그것을 로그 축에서 보인다. 두 그룹을 함께 놓는다 --- 처음에 n = 12
하나만 그렸는데, 들머리의 이야기가 **다섯 번 돌린 ±** 여서 주장이 그
이야기의 n 에 내려앉지 않았다. 그룹 **안에서는** 표본이 달라도 로그 자
위 길이가 같고, 그룹 **사이에서는** n 이 길이를 정한다. 오른쪽은 그 비가
n 에 따라 줄어드는 꼴이다.

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
GROUPS = [(5, "씨앗 다섯 — 들머리의 이야기", "tab:orange", 2),
          (12, "n = 12 — 이 쪽의 보기", "tab:blue", 3)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ── (좌) 그룹 안에서는 길이가 같고, 그룹 사이에서는 n 이 정한다 ─────
y = 0.0
yticks, ylabs = [], []
for n, lab, col, m in GROUPS:
    s2s = np.array([(rng.normal(0, np.sqrt(SIGMA2), n)).var(ddof=1)
                    for _ in range(400)])
    qs = [0.12, 0.5, 0.88][:m] if m == 3 else [0.15, 0.85]
    pick = [int(np.argmin(np.abs(s2s - q))) for q in np.quantile(s2s, qs)]
    y0 = y
    for idx in pick:
        s2 = s2s[idx]
        lo, hi = interval(s2, n)
        ax1.plot([lo, hi], [y, y], "-", lw=6.0, color=col,
                 solid_capstyle="butt")
        for x in (lo, hi):
            ax1.plot([x, x], [y - 0.13, y + 0.13], "-", lw=3.0, color=col)
        ax1.plot([s2], [y], "o", ms=8, color="white", mec=col, mew=2.2)
        ax1.text(lo * 0.90, y, f"$s^2$ = {s2:.3f}", ha="right", va="center",
                 fontsize=10, color="0.35")
        ax1.text(hi * 1.10, y, f"÷ = {hi / lo:.3f}", ha="left", va="center",
                 fontsize=10.5, color=col, fontweight="bold")
        y += 1.0
    yticks.append((y0 + y - 1) / 2)
    ylabs.append(lab)
    y += 0.7
ax1.axvline(SIGMA2, color="tab:red", lw=2.0)
ax1.text(SIGMA2 * 1.06, y - 0.45, f"참 분산 {SIGMA2:.0f}", fontsize=11,
         color="tab:red", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xlim(0.26, 260)
ax1.set_xticks([0.3, 1, 3, 10, 30, 100],
               ["0.3", "1", "3", "10", "30", "100"])
ax1.set_ylim(-1.15, y - 0.25)
ax1.set_yticks(yticks, ylabs, fontsize=11)
ax1.set_xlabel("분산 (로그 척도)", fontsize=11)
ax1.set_title("그룹 안에서는 길이가 같고, 그룹 사이에서는 n 이 정한다",
              fontsize=12.5)
ax1.grid(axis="x", alpha=0.25, which="both")
ax1.text(0.5, 0.018, "자료가 정하는 것은 자리뿐이다",
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
for n, c, dx, dy in ((5, "tab:orange", 26, -2.0), (12, "tab:blue", 30, 2.0),
                     (100, "tab:green", 24, 1.4)):
    ax2.plot([n], [ratio(n)], "o", ms=9, color=c)
    ax2.annotate(f"n = {n}\n{ratio(n):.3f} 배", xy=(n, ratio(n)),
                 xytext=(n + dx, ratio(n) + dy),
                 fontsize=10.5, color=c, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
ax2.plot([need2], [ratio(need2)], "o", ms=8, color="0.3")
ax2.annotate(f"두 배 아래로 내려가는 데\n{need2} 개가 든다",
             xy=(need2, 2.0), xytext=(need2 + 36, 6.2), fontsize=10.5,
             color="0.2", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4))
ax2.set_xlim(0, 305)
ax2.set_ylim(1.0, 25.5)
ax2.set_xlabel("표본 크기 n", fontsize=11)
ax2.set_ylabel("상한 ÷ 하한", fontsize=11)
ax2.set_title("그 비는 자유도만으로 정해진다", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("분산의 구간은 자료를 보기 전에 폭이 정해져 있다 — "
             "씨앗 다섯이면 상한이 하한의 23 배다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "sigma_ratio_fixed.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print("\n상한 ÷ 하한 — 자유도만으로 정해진다")
for n in (3, 5, 12, 25, need2, 100):
    print(f"  n = {n:3d} (자유도 {n-1:3d}): {ratio(n):8.3f} 배"
          f"   (σ 로는 {np.sqrt(ratio(n)):.3f} 배)")
print(f"\nn = {N0}: chi2 임계값 {stats.chi2.ppf(0.025, N0-1):.3f} 과 "
      f"{stats.chi2.ppf(0.975, N0-1):.3f}")
print(f"비가 두 배 아래로 내려가는 첫 n = {need2} (비 {ratio(need2):.4f})")
