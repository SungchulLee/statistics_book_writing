r"""다섯 수가 똑같은 네 자료 — 상자그림판 앤스컴.

상자그림은 자료에서 다섯 수만 가져간다. 최솟값·$Q_1$·중앙값·$Q_3$·최댓값이다.
표본 수 $n$ 을 $101$ 로 두면 그 다섯이 정확히 $x_{(1)}, x_{(26)}, x_{(51)},
x_{(76)}, x_{(101)}$ 이므로, **그 다섯 자리만 못 박고 나머지는 마음대로 채워도**
상자그림이 한 점 다르지 않다.

① 네 자료의 상자그림 — 구별되지 않는다.
② 같은 네 자료의 점구름 — 전혀 다르다.
③ 상자그림에는 $n$ 이 없다. $n = 21$ 과 $n = 2001$ 이 같은 그림을 낸다.

만드는 파일: docs/ch02/visualization/img/boxplot_identical.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(3)
N = 101
POS = [0, 25, 50, 75, 100]
VAL = [2.0, 8.0, 12.0, 16.0, 22.0]
B_BOOT = 20000


def build(fill, n=N, pos=POS, val=VAL):
    """다섯 자리를 못 박고 사이를 fill 로 채운다."""
    x = np.empty(n)
    for i in range(4):
        a, b = pos[i], pos[i + 1]
        lo, hi = val[i], val[i + 1]
        x[a] = lo
        if b - a > 1:
            x[a + 1:b] = np.sort(lo + (hi - lo) * fill(b - a - 1, i))
    x[pos[-1]] = val[-1]
    return np.sort(x)


def mound(k, i):
    """가운데(값 12)로 모은다 — 아래 두 칸은 오른쪽, 위 두 칸은 왼쪽."""
    t = RNG.random(k)
    return t ** 0.35 if i < 2 else t ** 2.8


def hollow(k, i):
    """가운데를 비우고 양끝 쪽으로 밀어낸다."""
    t = RNG.random(k)
    return t ** 2.8 if i < 2 else t ** 0.35


SETS = {
    "한 덩어리": build(mound),
    "가운데가 빈": build(hollow),
    "눈금 위": build(lambda k, i: np.round(RNG.random(k) * 2) / 2),
    "고른 분포": build(lambda k, i: RNG.random(k)),
}
NAMES = list(SETS)

print(f"다섯 자리를 못 박은 네 자료 (n = {N})")
print(f"{'자료':<10}{'min':>6}{'Q1':>7}{'중앙값':>8}{'Q3':>7}{'max':>7}"
      f"{'평균':>8}{'표준편차':>9}{'왜도':>8}{'서로 다른 값':>12}")
for nm, x in SETS.items():
    q = np.percentile(x, [0, 25, 50, 75, 100])
    print(f"{nm:<10}" + "".join(f"{v:>7.2f}" for v in q)
          + f"{x.mean():>8.3f}{x.std(ddof=1):>9.3f}{stats.skew(x):>8.3f}"
          + f"{len(np.unique(np.round(x, 6))):>12d}")
Q = np.percentile(SETS[NAMES[0]], [25, 75])
IQR = Q[1] - Q[0]
print(f"   사분위범위 {IQR:g}, 울타리 [{Q[0] - 1.5 * IQR:g}, "
      f"{Q[1] + 1.5 * IQR:g}] — 최솟값 {VAL[0]:g} 과 최댓값 {VAL[-1]:g} 이")
print(f"   둘 다 울타리 안이므로 수염이 끝까지 가고 이상점 표시도 없다. "
      f"네 상자그림이 한 점도 다르지 않다.")
sd = [x.std(ddof=1) for x in SETS.values()]
print(f"   그런데 표준편차는 {min(sd):.3f} 에서 {max(sd):.3f} 까지, "
      f"{max(sd) / min(sd):.2f} 배 벌어진다.")

# === ③ n 이 보이지 않는다 ===========================================
# 두 자료의 모양을 같게 두어야 n 의 효과만 남는다 — 칸 안을 고르게 채운다
FLAT = lambda k, i: RNG.random(k)     # noqa: E731
SMALL = build(FLAT, n=21, pos=[0, 5, 10, 15, 20], val=VAL)
BIG = build(FLAT, n=2001, pos=[0, 500, 1000, 1500, 2000], val=VAL)
print(f"\n③ 같은 다섯 수, 다른 표본 수")
CI = {}
for nm, v in (("n = 21", SMALL), ("n = 2001", BIG)):
    med = np.median(v[RNG.integers(0, len(v), (B_BOOT, len(v)))], axis=1)
    CI[nm] = np.percentile(med, [2.5, 97.5])
    q = np.percentile(v, [0, 25, 50, 75, 100])
    print(f"   {nm:<9} 다섯 수 " + " ".join(f"{t:g}" for t in q)
          + f"   중앙값의 95% 붓스트랩 구간 "
            f"[{CI[nm][0]:.2f}, {CI[nm][1]:.2f}]  폭 "
            f"{CI[nm][1] - CI[nm][0]:.2f}")
print(f"   폭의 비 {((CI['n = 21'][1] - CI['n = 21'][0]) / (CI['n = 2001'][1] - CI['n = 2001'][0])):.1f} 배 "
      f"— 상자그림은 둘을 구별해 주지 않는다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

bp = ax1.boxplot([SETS[nm] for nm in NAMES], widths=0.55, patch_artist=True,
                 medianprops=dict(color="black", lw=2))
for p in bp["boxes"]:
    p.set_facecolor("#cccccc")
ax1.set_xticklabels(NAMES, fontsize=10.5)
ax1.set_ylabel("값")
ax1.set_ylim(0, 24)
ax1.set_title("① 네 자료의 상자그림", fontsize=12)
ax1.grid(alpha=0.25, axis="y")
ax1.text(0.5, 0.03, "네 그림이 한 점도 다르지 않다",
         transform=ax1.transAxes, ha="center", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))

for k, nm in enumerate(NAMES):
    x = SETS[nm]
    ax2.plot(k + 1 + RNG.uniform(-0.22, 0.22, N), x, ".", color=COL[k], ms=5,
             alpha=0.8)
    ax2.text(k + 1, 23.2, f"표준편차 ${x.std(ddof=1):.2f}$", ha="center",
             fontsize=9.5)
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels(NAMES, fontsize=10.5)
ax2.set_xlim(0.4, 4.6)
ax2.set_ylim(0, 24)
ax2.set_ylabel("값")
ax2.set_title("② 같은 네 자료의 점", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

bp3 = ax3.boxplot([SMALL, BIG], widths=0.45, positions=[1, 2],
                  patch_artist=True, medianprops=dict(color="black", lw=2))
for p in bp3["boxes"]:
    p.set_facecolor("#cccccc")
for k, nm in enumerate(("n = 21", "n = 2001")):
    lo, hi = CI[nm]
    ax3.plot([k + 1.42, k + 1.42], [lo, hi], color="#d62728", lw=5,
             solid_capstyle="butt")
    ax3.text(k + 1.52, (lo + hi) / 2, f"폭 ${hi - lo:.2f}$", fontsize=10,
             color="#d62728", va="center")
ax3.plot([], [], color="#d62728", lw=5, label="중앙값의 $95\\%$ 붓스트랩 구간")
ax3.set_xticks([1, 2])
ax3.set_xticklabels(["$n = 21$", "$n = 2001$"], fontsize=10.5)
ax3.set_xlim(0.4, 3.0)
ax3.set_ylim(0, 24)
ax3.set_ylabel("값")
ax3.set_title("③ 상자그림에는 $n$ 이 없다", fontsize=12)
ax3.legend(loc="lower center", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch02/visualization/img/boxplot_identical.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
