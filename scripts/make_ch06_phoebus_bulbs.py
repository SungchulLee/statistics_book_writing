r"""6.3절 「그림으로 보는 통계학」 — 천 시간을 지키는지 확인하려면.

$1924$ 년 포이보스 카르텔은 백열전구의 수명을 $1{,}000$ 시간으로 묶고, 회원 공장이
보낸 표본을 스위스의 시험소에서 태워 보았다. 규정보다 **길어도 짧아도** 벌금이었다.
수명이 지수분포에 가깝다면 평균의 최대가능도추정량은 표본평균이고, 그 상대표준오차는
$1/\sqrt n$ 이다. 그러니 「천 시간을 지켰는가」를 묻는 일은 **전구를 몇 개나 끝까지
태울 것인가**를 묻는 일이 된다.

① 표본평균의 분포 — 전구 $n$ 개를 태워 평균을 낼 때.
② 규칙의 특성곡선 — 참 평균이 θ 일 때 「천 시간 초과」로 판정될 확률.
③ 한 번의 시험으로 $10\%$ 초과를 잡으려면 $n$ 이 얼마나 필요한가.

만드는 파일: docs/ch06/mle/img/phoebus_bulbs.png
실행:  python3 scripts/make_ch06_phoebus_bulbs.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats

OUT = "docs/ch06/mle/img/phoebus_bulbs.png"
STD = 1000.0             # 카르텔이 정한 수명
NS = (10, 50, 200)

# --- ① 표본평균의 분포 — x̄ ~ Gamma(n, θ/n) ----------------------------
print("① 참 평균이 1,000 시간일 때 표본평균의 흩어짐")
for n in NS:
    g = stats.gamma(a=n, scale=STD / n)
    lo, hi = g.ppf([0.025, 0.975])
    print(f"   n = {n:>4}:  상대표준오차 {100/np.sqrt(n):5.1f}%   "
          f"가운데 95% = [{lo:7.1f}, {hi:7.1f}] 시간")

# --- ② 규칙의 특성곡선 --------------------------------------------------
TH = np.linspace(700, 1500, 300)
print("\n② 「표본평균이 1,000 시간을 넘으면 벌금」 규칙에 걸릴 확률")
curves = {}
for n in NS:
    pr = 1 - stats.gamma(a=n, scale=TH / n).cdf(STD)
    curves[n] = pr
    for th in (1000, 1100, 1200):
        p = 1 - stats.gamma(a=n, scale=th / n).cdf(STD)
        print(f"   n = {n:>4}, 참 평균 {th:>5} 시간 → 걸릴 확률 {p:.3f}")

# --- ③ 10% 초과를 90% 확률로 잡으려면 ----------------------------------
print("\n③ 참 평균이 규정보다 얼마나 넘을 때, n 이 얼마면 90% 로 잡는가")
need = {}
for over in (0.05, 0.10, 0.20):
    th = STD * (1 + over)
    n = 1
    while 1 - stats.gamma(a=n, scale=th / n).cdf(STD) < 0.90:
        n += 1
        if n > 5000:
            break
    need[over] = n
    print(f"   {100*over:4.0f}% 초과 → 전구 {n:>4} 개를 끝까지 태워야 한다")

OVER = np.linspace(0.02, 0.5, 120)
nreq = []
for o in OVER:
    th = STD * (1 + o)
    n = 1
    while 1 - stats.gamma(a=n, scale=th / n).cdf(STD) < 0.90 and n < 20000:
        n = int(n * 1.05) + 1
    nreq.append(n)

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
COLS = ("#c0532b", "#6a8a3a", "#1f6f8b")

ax = axes[0]
grid = np.linspace(300, 2200, 600)
for n, c in zip(NS, COLS):
    ax.plot(grid, stats.gamma(a=n, scale=STD / n).pdf(grid), color=c, lw=2,
            label=f"n = {n}  (상대 SE {100/np.sqrt(n):.0f}%)")
ax.axvline(STD, color="0.35", ls="--", lw=1.3)
ax.text(STD + 20, ax.get_ylim()[1] * 0.92, "규정 1,000 시간", color="0.35", fontsize=9)
ax.set_xlabel("표본평균 수명 (시간)")
ax.set_ylabel("밀도")
ax.set_title("① 참 평균이 정확히 1,000 시간이어도", fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
for n, c in zip(NS, COLS):
    ax.plot(TH, curves[n], color=c, lw=2, label=f"n = {n}")
ax.axvline(STD, color="0.35", ls="--", lw=1.3)
ax.axhline(0.5, color="0.6", ls=":", lw=1.1)
ax.set_xlabel("참 평균 수명 (시간)")
ax.set_ylabel("「천 시간 초과」로 판정될 확률")
ax.set_title("② 규칙이 실제로 가르는 곳", fontsize=11)
ax.legend(fontsize=9, loc="lower right")

ax = axes[2]
ax.plot(100 * OVER, nreq, color="#1f6f8b", lw=2.2)
ax.set_yscale("log")
for o, c in zip((0.05, 0.10, 0.20), COLS):
    ax.plot([100 * o], [need[o]], "o", color=c, ms=8)
    ax.annotate(f"{100*o:.0f}% 초과 → {need[o]:,} 개", xy=(100 * o, need[o]),
                xytext=(100 * o + 3, need[o] * 1.6), color=c, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=c))
ax.set_xlabel("참 평균이 규정을 넘은 정도 (%)")
ax.set_ylabel("끝까지 태워야 하는 전구 수")
ax.set_title("③ 어기는 것을 90% 로 잡으려면", fontsize=11)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
